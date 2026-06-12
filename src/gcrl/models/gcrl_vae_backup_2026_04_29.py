# src/gcrl/models/gcrl_vae.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Dict, Tuple, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import anndata as ad  # type: ignore
except Exception:  # keep optional
    ad = None  # pragma: no cover

intervention_mapping = Literal["hard", "soft"]


@dataclass
class GCRLVAEConfig:
    input_dim: int  # Number of TFs (not all genes)
    z_dim: int
    c_dim: int
    output_dim: int  # Number of all genes (TF + TG)
    ct_dim: int = 0
    intervention_mapping: intervention_mapping = "hard"
    hidden_enc: Tuple[int, ...] = (128,)  # Simplified: single hidden layer
    use_polynomial_decoder: bool = True  # Linear + quadratic terms
    leak_slope: float = 0.1
    eps_var: float = 1e-6
    inv_eps: float = 1e-6


class GCRLVAE(nn.Module):
    """
    gCRL-VAE: Causal VAE with GRN-informed latent structure and DAG-based interventions.

    Architecture Flow:
        1. Encoder: x_TF (TFs only) → [MLP] → (μ, σ²)
        2. Sampler: z ~ N(μ, σ²)          [observational latent]
        3. DAG Transform: z → [Intervention + Causal Graph G] → u [intervened latent]
        4. Decoder: u → [Polynomial] → x_rec (all genes: TF + TG)

    Notation:
        - x_TF: TF expression only (input_dim = n_TFs)
        - z: Observational latent (before intervention)
        - u: Intervened latent (after DAG transform, called y_hat in forward())
        - x_rec: Reconstructed gene expression for all genes (output_dim = n_genes)

    Key Features:
        - TF-only input, all-gene output (TF → latent → all genes)
        - Community-structured latent space (p+1 dimensions)
        - Polynomial decoder (linear + quadratic terms) for interpretability
        - Hard or soft routing of interventions to latent dimensions
        - Upper-triangular causal matrix G for DAG structure
        - Optional cell-type conditioning (conditional VAE)

    Training Requirements:
        - self.tf_index (Dict[str,int]): Maps TF names to intervention indices
        - self.tf_to_latent (LongTensor[c_dim]): Maps TFs to latent indices (hard routing)
        - self.ct_index (Dict[str,int]): Maps cell types to indices (if ct_dim > 0)

    Note on variance:
        - encode() returns variance σ², NOT log-variance
        - For KL loss, convert via: logvar = torch.log(var)
    """

    def __init__(self, cfg: GCRLVAEConfig, device: Optional[torch.device] = None):
        super().__init__()
        self.cfg = cfg
        self.device_ = device
        self.tf_index: Optional[Dict[str, int]] = None
        self.tf_to_latent: Optional[torch.LongTensor] = None
        self.ct_index: Optional[Dict[str, int]] = None

        # Encoder: TF expression (input_dim) → hidden → latent parameters
        in_enc = cfg.input_dim + (cfg.ct_dim if cfg.ct_dim > 0 else 0)

        # Build encoder with variable number of hidden layers
        encoder_layers = []
        prev_dim = in_enc
        for hidden_dim in cfg.hidden_enc:
            encoder_layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.LeakyReLU(cfg.leak_slope, inplace=True),
                ]
            )
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(prev_dim, cfg.z_dim)
        self.fc_var = nn.Linear(prev_dim, cfg.z_dim)

        # Causal graph G (upper triangular)
        self.G = nn.Parameter(torch.zeros(cfg.z_dim, cfg.z_dim))

        # Intervention routing head (for soft routing)
        if cfg.intervention_mapping == "soft":
            self.c_head = nn.Sequential(
                nn.Linear(cfg.c_dim, max(64, cfg.c_dim)),
                nn.LeakyReLU(cfg.leak_slope, inplace=True),
                nn.Linear(max(64, cfg.c_dim), cfg.z_dim),
            )
        else:
            self.c_head = None

        # Learnable intervention strength parameters (one per TF)
        self.c_shift = nn.Parameter(torch.zeros(cfg.c_dim))

        # Decoder: latent → all gene expression (polynomial)
        in_dec = cfg.z_dim + (cfg.ct_dim if cfg.ct_dim > 0 else 0)

        if cfg.use_polynomial_decoder:
            # Polynomial decoder: linear + quadratic terms
            self.decoder_linear = nn.Linear(in_dec, cfg.output_dim)
            # For quadratic: compute outer product dimension
            self.decoder_quadratic = nn.Linear(in_dec * in_dec, cfg.output_dim)
            self.decoder = None  # Will use polynomial decode
        else:
            # Fallback: simple linear decoder
            self.decoder = nn.Linear(in_dec, cfg.output_dim)
            self.decoder_linear = None
            self.decoder_quadratic = None

        self.reset_parameters()

    # ----------------------- setters -----------------------
    def set_tf_index(self, tf_names: Sequence[str] | Dict[str, int]):
        if isinstance(tf_names, dict):
            self.tf_index = dict(tf_names)
        else:
            assert len(tf_names) == self.cfg.c_dim, "len(tf_names) must equal c_dim"
            self.tf_index = {n: i for i, n in enumerate(tf_names)}

    def set_tf_to_latent(self, tf_to_latent: Union[Sequence[int], torch.LongTensor]):
        if not isinstance(tf_to_latent, torch.Tensor):
            tf_to_latent = torch.as_tensor(tf_to_latent, dtype=torch.long)
        assert (
            tf_to_latent.numel() == self.cfg.c_dim
        ), "tf_to_latent length must equal c_dim"
        self.tf_to_latent = tf_to_latent.to(self._device())

    def set_ct_index(self, ct_names: Sequence[str] | Dict[str, int]):
        if isinstance(ct_names, dict):
            self.ct_index = dict(ct_names)
        else:
            assert len(ct_names) == self.cfg.ct_dim, "len(ct_names) must equal ct_dim"
            self.ct_index = {n: i for i, n in enumerate(ct_names)}

    # ----------------------- init utils -----------------------
    @staticmethod
    def truncated_normal_(tensor: torch.Tensor, mean=0.0, std=0.02):
        with torch.no_grad():
            tensor.normal_(mean=mean, std=std)
            tensor.clamp_(mean - 2 * std, mean + 2 * std)
        return tensor

    def weights_init(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            self.truncated_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def reset_parameters(self):
        self.apply(self.weights_init)
        with torch.no_grad():
            self.G.zero_()
            self.c_shift.zero_()

    # ----------------------- helpers -----------------------
    def _device(self):
        if self.device_ is not None:
            return self.device_
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _to_tensor(self, x):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return x.to(self._device())
        return torch.as_tensor(x, dtype=torch.float32, device=self._device())

    def _ct_vector(self, cell_type: Optional[str], B: int):
        if self.cfg.ct_dim <= 0 or cell_type is None or self.ct_index is None:
            return None
        idx = self.ct_index.get(cell_type, None)
        if idx is None:
            raise KeyError(f"cell_type '{cell_type}' missing from ct_index")
        ct = torch.zeros(B, self.cfg.ct_dim, device=self._device())
        ct[:, idx] = 1.0
        return ct

    def _intervention_to_c(self, intervention: str, B: int):
        if self.tf_index is None:
            raise RuntimeError("tf_index not set. Call set_tf_index(...)")
        c = torch.zeros(B, self.cfg.c_dim, device=self._device())
        if intervention is None or intervention == "unperturbed":
            return c
        parts = sorted(intervention.split("+"))
        for name in parts:
            if name not in self.tf_index:
                raise KeyError(f"Intervention TF '{name}' missing from tf_index")
            c[:, self.tf_index[name]] = 1.0
        return c

    # ----------------------- encoder/decoder -----------------------
    def encode(self, x: torch.Tensor, ct: Optional[torch.Tensor] = None):
        """
        Encode input to latent distribution parameters.

        Args:
            x: Input expression (B, input_dim)
            ct: Optional cell-type one-hot (B, ct_dim)

        Returns:
            mu: Latent mean (B, z_dim)
            var: Latent variance (B, z_dim) - NOT log-variance
                 Note: This returns variance σ², not log(σ²).
                 For KL loss, convert via: logvar = torch.log(var)
        """
        if self.cfg.ct_dim > 0 and ct is not None:
            x = torch.cat([x, ct], dim=-1)
        h = self.encoder(x)
        mu = self.fc_mu(h)
        var = F.softplus(self.fc_var(h)) + self.cfg.eps_var
        return mu, var

    @staticmethod
    def reparameterize(mu: torch.Tensor, var: torch.Tensor):
        """
        Reparameterization trick for VAE: z = μ + ε * σ

        Args:
            mu: Latent mean (B, z_dim)
            var: Latent variance σ² (B, z_dim) - NOT log-variance

        Returns:
            z: Sampled latent (B, z_dim)
        """
        eps = torch.randn_like(mu)
        return mu + eps * torch.sqrt(var)

    def decode(self, u: torch.Tensor, ct: Optional[torch.Tensor] = None):
        """
        Decode latent to gene expression using polynomial decoder.

        Args:
            u: Latent representation (B, z_dim)
            ct: Optional cell-type one-hot (B, ct_dim)

        Returns:
            x_rec: Reconstructed gene expression (B, output_dim) - all genes
        """
        if self.cfg.ct_dim > 0 and ct is not None:
            u = torch.cat([u, ct], dim=-1)

        if self.cfg.use_polynomial_decoder:
            # Polynomial decoder: linear + quadratic terms
            # Linear term
            out = self.decoder_linear(u)

            # Quadratic term: outer product of u with itself
            # u: (B, d) -> outer: (B, d, d) -> flatten: (B, d*d)
            u_outer = u.unsqueeze(-1) * u.unsqueeze(-2)  # (B, d, d)
            u_squared = u_outer.reshape(u.shape[0], -1)  # (B, d*d)
            out = out + self.decoder_quadratic(u_squared)

            return out
        else:
            # Fallback: simple linear decoder
            return self.decoder(u)

    # ----------------------- routing -----------------------
    def _soft_routing(self, c: torch.Tensor, temp: float):
        assert self.c_head is not None, "soft routing head missing"
        logits = self.c_head(c)
        bc = F.softmax(logits / max(1e-6, temp), dim=-1)
        return bc

    def _hard_routing(
        self, c: torch.Tensor, tf_to_latent: torch.Tensor
    ) -> torch.Tensor:
        idx = torch.argmax(c, dim=-1)
        lat_idx = tf_to_latent[idx]
        B = c.shape[0]
        # IMPORTANT: route into the full latent width (p + 1), not just max(tf_to_latent)+1
        z_dim = int(self.cfg.z_dim)
        bc = torch.zeros(B, z_dim, device=c.device, dtype=c.dtype)
        bc.scatter_(1, lat_idx.view(-1, 1), 1.0)
        return bc

    # ----------------------- DAG transform -----------------------
    def _dag_transform(self, z, bc, csz, bc2=None, csz2=None):
        B, K = z.shape
        G_up = torch.triu(self.G, diagonal=1)
        I = torch.eye(K, device=z.device, dtype=z.dtype)
        A = I - G_up
        A = A + self.cfg.inv_eps * I
        Ainv = torch.linalg.inv(A)

        y = z
        if bc is not None and csz is not None:
            y = y + bc * csz
        if bc2 is not None and csz2 is not None:
            y = y + bc2 * csz2
        u = y @ Ainv.T
        return u

    # ----------------------- forward -----------------------
    def forward(
        self, x, c=None, c2=None, *, num_interv=1, temp=1.0, tf_to_latent=None, ct=None
    ):
        mu, var = self.encode(x, ct)
        z = self.reparameterize(mu, var)

        bc = bc2 = None
        csz = csz2 = None

        if num_interv >= 1 and c is not None:
            csz = torch.matmul(c, self.c_shift).view(-1, 1)
            if self.cfg.intervention_mapping == "soft":
                bc = self._soft_routing(c, temp)
            else:
                if tf_to_latent is None:
                    raise ValueError("Hard mapping requires tf_to_latent")
                bc = self._hard_routing(c, tf_to_latent)
        if num_interv >= 2 and c2 is not None:
            csz2 = torch.matmul(c2, self.c_shift).view(-1, 1)
            if self.cfg.intervention_mapping == "soft":
                bc2 = self._soft_routing(c2, temp)
            else:
                if tf_to_latent is None:
                    raise ValueError("Hard mapping requires tf_to_latent for c2")
                bc2 = self._hard_routing(c2, tf_to_latent)

        y_hat = self._dag_transform(z, bc, csz, bc2, csz2)
        x_rec = self.decode(y_hat, ct)
        return y_hat, x_rec, mu, var, self.G, {"routing": self.cfg.intervention_mapping}

    # ================== Convenience API (new) ==================
    @torch.no_grad()
    def encode_batch(self, X, ct=None):
        self.eval()
        x_t = self._to_tensor(X)
        ct_t = self._to_tensor(ct) if ct is not None else None
        mu, var = self.encode(x_t, ct_t)
        return {"mu": mu.detach().cpu().numpy(), "var": var.detach().cpu().numpy()}

    @torch.no_grad()
    def apply_intervention(
        self, mu, intervention: str, cell_type: Optional[str] = None
    ):
        self.eval()
        mu_t = self._to_tensor(mu)
        B = mu_t.shape[0]
        c = self._intervention_to_c(intervention, B)
        if self.cfg.intervention_mapping == "soft":
            bc = self._soft_routing(c, temp=1.0)
        else:
            if self.tf_to_latent is None:
                raise RuntimeError(
                    "Hard routing requires tf_to_latent; call set_tf_to_latent(...)"
                )
            bc = self._hard_routing(c, self.tf_to_latent.to(self._device()))
        csz = torch.matmul(c, self.c_shift).view(-1, 1)
        u = self._dag_transform(mu_t, bc, csz, None, None)
        return u.detach().cpu().numpy()

    @torch.no_grad()
    def decode_batch(self, z, ct=None):
        self.eval()
        z_t = self._to_tensor(z)
        ct_t = self._to_tensor(ct) if ct is not None else None
        x_pred = self.decode(z_t, ct_t)
        return x_pred.detach().cpu().numpy()

    @torch.no_grad()
    def predict(
        self,
        adata: ad.AnnData,
        *,
        set_key: str = "set",
        intervention_key: str = "intervention",
        cell_type_key: str = "cell_type",
        control_labels: tuple = (
            "control",
            "unperturbed",
            "na",
            "NA",
            "None",
            "none",
            "",
        ),
        seed: Optional[int] = None,
    ) -> ad.AnnData:
        """
        Generate predictions for all test interventions by simulating perturbations on control cells.

        This method automatically:
        1. Identifies all unique (cell_type, intervention) pairs in the test set
        2. For each pair, samples control cells from training set (matching cell type)
        3. Simulates the intervention using the learned causal model
        4. Returns predictions as a new AnnData object with obs['set'] = 'prediction'

        Args:
            adata: AnnData with both training and test cells
                   - Must contain obs columns: set_key, intervention_key, cell_type_key
                   - Training controls used as baseline for simulation
                   - Test set defines which (cell_type, intervention) pairs to predict
            set_key: Column name for train/test split (default: "set")
            intervention_key: Column name for intervention labels (default: "intervention")
            cell_type_key: Column name for cell type labels (default: "cell_type")
            control_labels: Tuple of strings considered as control/unperturbed (default: standard set)
            seed: Random seed for reproducibility (default: None)

        Returns:
            AnnData with predictions:
                - X: Predicted expression (n_predictions, n_genes)
                - obs: Metadata with 'set' = 'prediction', plus cell_type and intervention
                - var: Same as input adata.var
                - obs['source_cell_idx']: Index of control cell used for simulation

        Raises:
            KeyError: If required obs columns are missing
            ValueError: If no test cells found or no training controls available
            RuntimeError: If model not properly wired (tf_index, tf_to_latent missing)

        Example:
            >>> # After training
            >>> predictions = model.predict(adata)
            >>> # predictions.obs will have 'set' == 'prediction'
            >>> # Can compare with real test cells: adata[adata.obs['set'] == 'test']
        """
        if ad is None:
            raise RuntimeError("anndata not installed; install anndata to use predict")

        # Set random seed
        rng = np.random.default_rng(seed)

        # Validate required columns
        for col in [set_key, intervention_key, cell_type_key]:
            if col not in adata.obs.columns:
                raise KeyError(f"adata.obs must contain '{col}' column")

        # Split data
        is_train = adata.obs[set_key].astype(str).values == "training"
        is_test = adata.obs[set_key].astype(str).values == "test"
        is_control = adata.obs[intervention_key].astype(str).isin(control_labels).values

        train_controls = adata[is_train & is_control].copy()
        test_set = adata[is_test].copy()

        if test_set.n_obs == 0:
            raise ValueError("No test cells found (adata.obs[set_key] == 'test')")
        if train_controls.n_obs == 0:
            raise ValueError("No training control cells found")

        # Extract TF expression indices if using TF-only input
        if "kind" in adata.var.columns:
            tf_mask = adata.var["kind"].astype(str).values == "TF"
            tf_idx = np.where(tf_mask)[0]
            use_tf_only = True
        else:
            tf_idx = np.arange(adata.n_vars)
            use_tf_only = False

        # Group test cells by (cell_type, intervention)
        test_groups = test_set.obs.groupby(
            [cell_type_key, intervention_key], observed=True
        )

        predictions_list = []

        for (cell_type, intervention), group_obs in test_groups:
            # Skip control cells in test set (nothing to predict)
            if str(intervention) in control_labels:
                continue

            n_pred = len(group_obs)

            # Get training controls for this cell type
            ct_controls = train_controls[
                train_controls.obs[cell_type_key].astype(str) == str(cell_type)
            ]

            if ct_controls.n_obs == 0:
                print(
                    f"Warning: No training controls for cell_type='{cell_type}', skipping intervention '{intervention}'"
                )
                continue

            # Sample control cells (with replacement if n_pred > n_controls)
            sample_idx = rng.integers(0, ct_controls.n_obs, size=n_pred)
            X_ctrl = np.asarray(ct_controls.X)[sample_idx]

            # Extract TF-only expression if needed
            if use_tf_only:
                X_ctrl_input = X_ctrl[:, tf_idx]
            else:
                X_ctrl_input = X_ctrl

            # Build cell-type conditioning vector
            ct_vec = None
            if self.cfg.ct_dim > 0 and self.ct_index is not None:
                ct_t = self._ct_vector(str(cell_type), n_pred)
                if ct_t is not None:
                    ct_vec = ct_t.cpu().numpy()

            # Parse intervention (handles single and double perturbations)
            intervention_str = str(intervention)
            parts = sorted(intervention_str.split("+"))

            # Build intervention vectors
            c = self._intervention_to_c(
                parts[0] if len(parts) >= 1 else "unperturbed", n_pred
            )
            c2 = (
                self._intervention_to_c(
                    parts[1] if len(parts) >= 2 else "unperturbed", n_pred
                )
                if len(parts) >= 2
                else None
            )
            num_interv = min(len(parts), 2)

            # Encode controls
            enc = self.encode_batch(X_ctrl_input, ct=ct_vec)
            mu = torch.from_numpy(enc["mu"]).to(self._device())

            # Apply intervention(s) via DAG transform
            csz = torch.matmul(c, self.c_shift).view(-1, 1)
            if self.cfg.intervention_mapping == "soft":
                bc = self._soft_routing(c, temp=1.0)
            else:
                if self.tf_to_latent is None:
                    raise RuntimeError(
                        "Hard routing requires tf_to_latent; call set_tf_to_latent(...)"
                    )
                bc = self._hard_routing(c, self.tf_to_latent.to(self._device()))

            # Handle double perturbations
            bc2 = None
            csz2 = None
            if num_interv >= 2 and c2 is not None:
                csz2 = torch.matmul(c2, self.c_shift).view(-1, 1)
                if self.cfg.intervention_mapping == "soft":
                    bc2 = self._soft_routing(c2, temp=1.0)
                else:
                    bc2 = self._hard_routing(c2, self.tf_to_latent.to(self._device()))

            # DAG transform
            u = self._dag_transform(mu, bc, csz, bc2, csz2)

            # Decode to expression space
            u_cpu = u.cpu()
            ct_t_decode = torch.from_numpy(ct_vec) if ct_vec is not None else None
            X_pred = self.decode_batch(u_cpu.numpy(), ct=ct_vec)

            # Create AnnData for this group
            pred_group = ad.AnnData(X_pred, var=adata.var.copy())
            pred_group.obs[set_key] = "prediction"
            pred_group.obs[cell_type_key] = str(cell_type)
            pred_group.obs[intervention_key] = intervention_str
            pred_group.obs["source_cell_idx"] = ct_controls.obs.index[sample_idx].values

            predictions_list.append(pred_group)

        if not predictions_list:
            raise ValueError("No predictions generated (no valid test groups found)")

        # Concatenate all predictions
        predictions = ad.concat(predictions_list, axis=0, join="outer")
        predictions.obs_names_make_unique()

        # add var from original adata
        predictions.var = adata.var.copy()

        return predictions
