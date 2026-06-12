# src/gcrl/models/gcrl_vae.py
# -*- coding: utf-8 -*-
"""
gCRL-VAE model.

Two operating modes controlled by GCRLVAEConfig:

    use_tf_only=True,  use_GRN_priors=True  -> gCRL-VAE
        TF-only encoder input, hard community routing, MCC alignment loss term.

    use_tf_only=False, use_GRN_priors=False -> discrepancy-VAE (CMVAE)
        All-gene encoder input, soft MLP routing, no MCC term.

Hybrid combinations (one flag True, the other False) are also supported.

Architecture (always):
    Encoder : fc1(input_dim -> hidden_enc) -> LeakyReLU(0.2) -> fc_mean / fc_var
    Decoder : d1(z_dim -> 128) -> LeakyReLU(0.2) -> d2(128 -> output_dim)
    DAG     : u = z_interv @ inv(I - triu(G))

Routing (use_GRN_priors=True):
    A (z_dim x z_dim) score matrix S is learned (initialized to identity).
    The hard permutation pi = argmax(S) row-by-row maps each community to a
    latent dimension and simultaneously defines the causal ordering for the DAG.
    Hard routing: bc = one_hot(pi[community_of_TF]).

Routing (use_GRN_priors=False):
    Soft MLP + softmax routing (CMVAE style).

Alignment (use_GRN_priors=True only):
    MCC loss term: partial_mcc_loss_torch(E[:, pi], mu)
    Eigengene columns are reordered by pi so column k of E aligns with
    latent dimension pi[k], with no additional learnable alignment matrix.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import anndata as ad
except Exception:
    ad = None


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class GCRLVAEConfig:
    input_dim: int          # encoder input: n_TFs (use_tf_only=True) or n_genes
    z_dim: int              # latent dimension (= p+1)
    c_dim: int              # number of intervened TFs
    output_dim: int         # decoder output: always n_genes (TF + TG)
    hidden_enc: int = 128   # encoder hidden layer size
    intervention_mapping: str = "hard"   # "hard" or "soft"
    use_tf_only: bool = True             # informational; used by trainer
    use_GRN_priors: bool = True          # hard routing + MCC if True


# ---------------------------------------------------------------------------
# Weight initialisation (from discrepancy-VAE)
# ---------------------------------------------------------------------------

def _truncated_normal_(tensor: torch.Tensor, mean: float = 0.0, std: float = 0.02):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor


def _weights_init(m: nn.Module):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class GCRLVAE(nn.Module):
    """
    gCRL-VAE / discrepancy-VAE unified model.

    After construction call:
        model.set_tf_index(tf_names)              always
        model.set_tf_comm(tf_comm_ids, z_dim)     when use_GRN_priors=True
            tf_comm_ids[k] = community index (0..p-1) for the k-th intervened TF
            S is initialised to identity so pi[k] = k at the start.
    """

    def __init__(self, cfg: GCRLVAEConfig):
        super().__init__()
        self.cfg = cfg

        # -- Encoder (variable hidden size) --
        self.fc1     = nn.Linear(cfg.input_dim, cfg.hidden_enc)
        self.fc_mean = nn.Linear(cfg.hidden_enc, cfg.z_dim)
        self.fc_var  = nn.Linear(cfg.hidden_enc, cfg.z_dim)

        # -- DAG matrix G (full square; only triu used) --
        self.G = nn.Parameter(torch.zeros(cfg.z_dim, cfg.z_dim))

        # -- Intervention strength (one scalar per TF) --
        self.c_shift = nn.Parameter(torch.ones(cfg.c_dim))

        # -- Permutation score matrix S (use_GRN_priors=True only) --
        # S is (z_dim x z_dim); pi = argmax(S, dim=1) row-by-row.
        # Initialized to identity so pi starts as the trivial permutation.
        if cfg.use_GRN_priors:
            self.S = nn.Parameter(torch.eye(cfg.z_dim))
        else:
            self.S = None

        # -- Soft routing head (use_GRN_priors=False only) --
        if not cfg.use_GRN_priors:
            self.c_head = nn.Sequential(
                nn.Linear(cfg.c_dim, cfg.hidden_enc),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(cfg.hidden_enc, cfg.z_dim),
            )
        else:
            self.c_head = None

        # -- Decoder (fixed 128 hidden, same as CMVAE) --
        self.d1 = nn.Linear(cfg.z_dim, 128)
        self.d2 = nn.Linear(128, cfg.output_dim)

        # -- Runtime indices (set by trainer) --
        self.tf_index: Optional[Dict[str, int]] = None
        # tf_comm_ids[k] = community index for k-th intervened TF (0..p-1)
        self.tf_comm_ids: Optional[torch.LongTensor] = None

        self.apply(_weights_init)
        with torch.no_grad():
            self.G.zero_()
            self.c_shift.fill_(1.0)
            if self.S is not None:
                self.S.copy_(torch.eye(cfg.z_dim))

    # ------------------------------------------------------------------
    # Setters
    # ------------------------------------------------------------------

    def set_tf_index(self, tf_names: Union[Sequence[str], Dict[str, int]]):
        if isinstance(tf_names, dict):
            self.tf_index = dict(tf_names)
        else:
            assert len(tf_names) == self.cfg.c_dim
            self.tf_index = {n: i for i, n in enumerate(tf_names)}

    def set_tf_comm(
        self,
        tf_comm_ids: Union[Sequence[int], np.ndarray],
    ):
        """
        Store community indices for the intervened TFs.

        tf_comm_ids[k] is the community index (0..p-1) for the k-th intervened TF.
        This replaces the old set_tf_to_latent(); the actual latent dimension is
        determined at runtime via pi = _hard_perm().
        """
        assert self.cfg.use_GRN_priors, "set_tf_comm only needed when use_GRN_priors=True"
        ids = torch.as_tensor(tf_comm_ids, dtype=torch.long)
        assert ids.numel() == self.cfg.c_dim
        self.tf_comm_ids = ids.to(self._device())

    def set_ct_index(self, ct_names):
        # reserved for future cell-type conditioning; no-op for now
        pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _to_tensor(self, x) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.to(self._device())
        return torch.as_tensor(x, dtype=torch.float32, device=self._device())

    def _hard_perm(self) -> torch.LongTensor:
        """
        Derive the hard permutation pi from S.
        pi[k] = argmax(S[k, :])  — the latent dimension assigned to position k.
        Shape: (z_dim,)
        """
        return torch.argmax(self.S, dim=1)  # (z_dim,)

    def _intervention_to_c(self, intervention: str, B: int) -> torch.Tensor:
        if self.tf_index is None:
            raise RuntimeError("tf_index not set; call set_tf_index() first")
        c = torch.zeros(B, self.cfg.c_dim, device=self._device())
        if intervention in ("", "unperturbed", "control"):
            return c
        for name in re.split(r"[+;,]", intervention):
            name = name.strip()
            if name and name in self.tf_index:
                c[:, self.tf_index[name]] = 1.0
        return c

    # ------------------------------------------------------------------
    # Encoder / decoder (discrepancy-VAE style)
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (mu, var); var is softplus (positive), not log-var."""
        h = F.leaky_relu(self.fc1(x), 0.2)
        return self.fc_mean(h), F.softplus(self.fc_var(h))

    @staticmethod
    def reparameterize(mu: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        return mu + torch.randn_like(mu) * torch.sqrt(var)

    def decode(self, u: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(self.d2(F.leaky_relu(self.d1(u), 0.2)), 0.2)

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def _hard_routing(self, c: torch.Tensor) -> torch.Tensor:
        """
        One-hot routing: maps each intervened TF to its latent dimension via pi.

        Community of TF k  ->  latent dimension pi[community_k].
        pi is derived from S at each forward pass.
        """
        if self.tf_comm_ids is None:
            raise RuntimeError("tf_comm_ids not set; call set_tf_comm() first")
        pi = self._hard_perm()                        # (z_dim,)
        tf_idx = torch.argmax(c, dim=-1)              # (B,) index into c_dim
        comm   = self.tf_comm_ids[tf_idx]             # (B,) community index
        lat    = pi[comm]                             # (B,) latent dimension
        bc = torch.zeros(c.shape[0], self.cfg.z_dim, device=c.device, dtype=c.dtype)
        bc.scatter_(1, lat.view(-1, 1), 1.0)
        return bc

    def _soft_routing(self, c: torch.Tensor, temp: float = 1.0) -> torch.Tensor:
        """MLP + softmax routing (CMVAE c_encode)."""
        assert self.c_head is not None
        logits = self.c_head(c)
        return F.softmax(logits / max(temp, 1e-6), dim=-1)

    def _route(self, c: torch.Tensor, temp: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (bc, csz) for one intervention vector."""
        csz = (c @ self.c_shift).view(-1, 1)
        if self.cfg.use_GRN_priors:
            bc = self._hard_routing(c)
        else:
            bc = self._soft_routing(c, temp)
        return bc, csz

    # ------------------------------------------------------------------
    # DAG transform (discrepancy-VAE dag())
    # The causal ordering is implicitly defined by pi: lower pi-index =
    # more upstream, so triu(G) is meaningful relative to the learned pi.
    # ------------------------------------------------------------------

    def _dag_transform(
        self,
        z: torch.Tensor,
        bc: Optional[torch.Tensor],
        csz: Optional[torch.Tensor],
        bc2: Optional[torch.Tensor] = None,
        csz2: Optional[torch.Tensor] = None,
        num_interv: int = 1,
    ) -> torch.Tensor:
        K = z.shape[1]
        G_triu = torch.triu(self.G, diagonal=1)
        I = torch.eye(K, device=z.device, dtype=z.dtype)
        Ainv = torch.linalg.inv(I - G_triu)

        if num_interv == 0:
            z_interv = z
        elif num_interv == 1:
            z_interv = z + bc * csz
        else:
            z_interv = z + bc * csz + bc2 * csz2

        return z_interv @ Ainv

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        c: Optional[torch.Tensor] = None,
        c2: Optional[torch.Tensor] = None,
        *,
        num_interv: int = 1,
        temp: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args
            x          : (B, input_dim) control cell expression
            c          : (B, c_dim)  first intervention binary vector
            c2         : (B, c_dim)  second intervention binary vector (double perturb)
            num_interv : 0, 1, or 2
            temp       : softmax temperature (soft routing only)

        Returns
            y_hat   : (B, output_dim)  predicted intervened expression
            x_recon : (B, output_dim)  reconstructed control expression
            mu      : (B, z_dim)
            var     : (B, z_dim)
            G       : (z_dim, z_dim)
        """
        mu, var = self.encode(x)
        z = self.reparameterize(mu, var)

        bc = bc2 = csz = csz2 = None
        if num_interv >= 1 and c is not None:
            bc, csz = self._route(c, temp)
        if num_interv >= 2 and c2 is not None:
            bc2, csz2 = self._route(c2, temp)

        y_hat   = self.decode(self._dag_transform(z, bc, csz, bc2, csz2, num_interv))
        x_recon = self.decode(self._dag_transform(z, None, None, None, None, num_interv=0))

        return y_hat, x_recon, mu, var, self.G

    # ------------------------------------------------------------------
    # Convenience inference API (signatures unchanged)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_batch(self, X, **_) -> Dict[str, np.ndarray]:
        self.eval()
        mu, var = self.encode(self._to_tensor(X))
        return {"mu": mu.cpu().numpy(), "var": var.cpu().numpy()}

    @torch.no_grad()
    def decode_batch(self, z, **_) -> np.ndarray:
        self.eval()
        return self.decode(self._to_tensor(z)).cpu().numpy()

    @torch.no_grad()
    def apply_intervention(self, mu, intervention: str, **_) -> np.ndarray:
        self.eval()
        mu_t = self._to_tensor(mu)
        B = mu_t.shape[0]
        c = self._intervention_to_c(intervention, B)
        bc, csz = self._route(c, temp=1.0)
        u = self._dag_transform(mu_t, bc, csz, num_interv=1)
        return u.cpu().numpy()

    @torch.no_grad()
    def predict(
        self,
        adata: "ad.AnnData",
        *,
        set_key: str = "set",
        intervention_key: str = "intervention",
        cell_type_key: str = "cell_type",
        control_labels: tuple = ("control", "unperturbed", "na", "NA", "None", "none", ""),
        seed: Optional[int] = None,
    ) -> "ad.AnnData":
        """
        Generate predictions for all test (cell_type, intervention) pairs by
        simulating perturbations on matched training control cells.

        Returns AnnData with obs['set'] = 'prediction'.
        """
        if ad is None:
            raise RuntimeError("anndata not installed")

        rng = np.random.default_rng(seed)

        for col in [set_key, intervention_key, cell_type_key]:
            if col not in adata.obs.columns:
                raise KeyError(f"adata.obs must contain '{col}'")

        is_train   = adata.obs[set_key].astype(str) == "training"
        is_test    = adata.obs[set_key].astype(str) == "test"
        is_control = adata.obs[intervention_key].astype(str).isin(control_labels)

        train_controls = adata[is_train & is_control].copy()
        test_set       = adata[is_test].copy()

        if test_set.n_obs == 0:
            raise ValueError("No test cells found")
        if train_controls.n_obs == 0:
            raise ValueError("No training control cells found")

        # TF-only input if configured
        if self.cfg.use_tf_only and "kind" in adata.var.columns:
            tf_mask = adata.var["kind"].astype(str) == "TF"
            tf_idx  = np.where(tf_mask)[0]
        else:
            tf_idx = np.arange(adata.n_vars)

        def _to_dense(X):
            import scipy.sparse as sp
            return X.toarray() if sp.issparse(X) else np.asarray(X)

        predictions_list = []

        for (cell_type, intervention), group_obs in test_set.obs.groupby(
            [cell_type_key, intervention_key], observed=True
        ):
            if str(intervention) in control_labels:
                continue

            n_pred = len(group_obs)

            ct_controls = train_controls[
                train_controls.obs[cell_type_key].astype(str) == str(cell_type)
            ]
            if ct_controls.n_obs == 0:
                print(f"Warning: no controls for cell_type='{cell_type}', skipping '{intervention}'")
                continue

            sample_idx  = rng.integers(0, ct_controls.n_obs, size=n_pred)
            X_ctrl_full = _to_dense(ct_controls.X)[sample_idx]
            X_ctrl_in   = X_ctrl_full[:, tf_idx]

            enc  = self.encode_batch(X_ctrl_in)
            mu_t = self._to_tensor(enc["mu"])

            intervention_str = str(intervention)
            parts = sorted(re.split(r"[+;,]", intervention_str))
            parts = [p.strip() for p in parts if p.strip()]

            c  = self._intervention_to_c(parts[0] if parts else "unperturbed", n_pred)
            c2 = self._intervention_to_c(parts[1], n_pred) if len(parts) >= 2 else None
            num_interv = min(len(parts), 2)

            bc,   csz   = self._route(c, temp=1.0)
            bc2_, csz2_ = self._route(c2, temp=1.0) if c2 is not None else (None, None)

            u      = self._dag_transform(mu_t, bc, csz, bc2_, csz2_, num_interv)
            X_pred = self.decode_batch(u.cpu().numpy())

            pred_group = ad.AnnData(X_pred, var=adata.var.copy())
            pred_group.obs[set_key]            = "prediction"
            pred_group.obs[cell_type_key]      = str(cell_type)
            pred_group.obs[intervention_key]   = intervention_str
            pred_group.obs["test_cell_idx"]    = group_obs.index.values
            pred_group.obs["control_cell_idx"] = ct_controls.obs.index[sample_idx].values

            predictions_list.append(pred_group)

        if not predictions_list:
            raise ValueError("No predictions generated")

        out = ad.concat(predictions_list, axis=0, join="outer")
        out.obs_names_make_unique()
        out.var = adata.var.copy()
        return out
