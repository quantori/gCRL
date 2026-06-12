#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gCRL-VAE trainer.

Closely follows the discrepancy-VAE (CMVAE) training loop with two additions
controlled by GCRLVAEConfig flags:

    use_tf_only=True   : encoder receives TF-only expression
    use_GRN_priors=True: hard community routing via learned permutation S,
                         plus MCC alignment loss term

When both flags are False the trainer reduces to the original discrepancy-VAE.

Permutation learning (use_GRN_priors=True):
    A score matrix S in R^(z_dim x z_dim) is an nn.Parameter initialized to
    the identity.  The hard permutation pi = argmax(S, dim=1) is derived at
    each forward pass.  pi simultaneously defines:
      - which latent dimension each community is routed to (routing)
      - the causal ordering for the DAG upper triangle
    The MCC loss is partial_mcc_loss_torch(E[:, pi], mu), reordering eigengene
    columns by pi so column k aligns with latent dimension pi[k].
    Gradients flow through the MCC values into mu and into S; pi itself is
    non-differentiable (argmax) but S is updated because the loss value changes
    as the column assignments change.

Loss (per batch):
    L = alpha(t) * MMD(y_hat, y)          [default: use_centroid_loss=False]
      or
      alpha(t) * centroid_dist(y_hat, y)  [use_centroid_loss=True]
      + MSE(x_recon, x)
      + beta(t)  * KLD
      + lambda_sparse * L1(triu(G))
      [+ lambda_mcc * MCC(E[:, pi], mu)]   # only when use_GRN_priors=True

# --- CENTROID LOSS SWITCH (removable) -------------------------------------------
# VAEConfig.use_centroid_loss: when True, replaces MMD with centroid distance
# (MSE between batch means of y_hat and y).  Logged as "dist" instead of "mmd".
# To revert: remove use_centroid_loss from VAEConfig, loss_function, and the
# training loop (search "CENTROID LOSS SWITCH").
# ---------------------------------------------------------------------------------

Schedules (same as discrepancy-VAE):
    beta  : 0 for epochs <10, then linear ramp to beta_kld_max
    alpha : 0 for epochs <5,  then linear ramp to alpha_mmd_max at mid-training,
            plateau at alpha_mmd_max thereafter
    temp  : linear ramp from 1.0 to temp_max (soft routing only)

Preprocessing:
    Raw adata.X (normalized + log1p, float32) used as-is — no z-scoring —
    matching discrepancy-VAE behaviour.

Batch sampler:
    Groups cells by intervention label (same as SCDATA_sampler in discrepancy-VAE).
    Each batch contains cells from a single intervention condition.

Signatures kept:
    train_gcrl_vae(adata, cfg, *, eigengenes_key="X_comm_eig")
"""
from __future__ import annotations

import json
import os
import random
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(it, **kw):
        return it

from gcrl.alignment.partial_mcc import partial_mcc_loss_torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_dense(X) -> np.ndarray:
    try:
        import scipy.sparse as sp
        if sp.issparse(X):
            return X.toarray()
    except Exception:
        pass
    return np.asarray(X)


def _parse_intervention(s: str) -> List[str]:
    if not s or str(s).strip().lower() in ("", "control", "unperturbed", "na", "none"):
        return []
    return [p.strip() for p in re.split(r"[+;,]", str(s)) if p.strip()]


# ---------------------------------------------------------------------------
# MMD loss (verbatim from discrepancy-VAE utils.py)
# ---------------------------------------------------------------------------

class MMD_loss(nn.Module):
    def __init__(self, kernel_mul: float = 2.0, kernel_num: int = 5, fix_sigma=None):
        super().__init__()
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
        self.fix_sigma  = fix_sigma

    def gaussian_kernel(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_samples = source.size(0) + target.size(0)
        total  = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(total.size(0), total.size(0), total.size(1))
        total1 = total.unsqueeze(1).expand(total.size(0), total.size(0), total.size(1))
        L2     = ((total0 - total1) ** 2).sum(2)
        if self.fix_sigma:
            bandwidth = self.fix_sigma
        else:
            bandwidth = torch.sum(L2.data) / (n_samples ** 2 - n_samples)
        bandwidth /= self.kernel_mul ** (self.kernel_num // 2)
        bws = [bandwidth * (self.kernel_mul ** i) for i in range(self.kernel_num)]
        return sum(torch.exp(-L2 / bw) for bw in bws)

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bs = source.size(0)
        K  = self.gaussian_kernel(source, target)
        return torch.mean(K[:bs, :bs] + K[bs:, bs:] - K[:bs, bs:] - K[bs:, :bs])


# ---------------------------------------------------------------------------
# Loss function (discrepancy-VAE + optional MCC term)
# ---------------------------------------------------------------------------

def loss_function(
    y_hat:         torch.Tensor,
    y:             torch.Tensor,
    x_recon:       torch.Tensor,
    x:             torch.Tensor,
    mu:            torch.Tensor,
    var:           torch.Tensor,
    G:             torch.Tensor,
    mmd_criterion: MMD_loss,
    *,
    E:             Optional[torch.Tensor] = None,
    pi:            Optional[torch.Tensor] = None,
    lambda_mcc:    float = 0.0,
    use_centroid_loss: bool = False,  # CENTROID LOSS SWITCH
):
    """
    Returns (dist, mse, kld, l1, mcc) — all non-negative scalar tensors.
    'dist' is either MMD (default) or centroid distance (use_centroid_loss=True).

    MCC term: 1 + partial_mcc_loss_torch(E[:, pi], mu)
    partial_mcc_loss_torch returns values in [−1, 0] (negative mean partial
    correlation), so adding 1 shifts the range to [0, 1]: 0 = perfect alignment,
    1 = no alignment.  This makes the MCC term non-negative and directly
    comparable in scale to MSE and MMD.
    Active only when E, pi are provided and lambda_mcc > 0.
    """
    # CENTROID LOSS SWITCH — replace the next two lines to revert
    if use_centroid_loss:
        dist = F.mse_loss(y_hat.mean(0), y.mean(0), reduction="mean")
    else:
        dist = mmd_criterion(y_hat, y)
    # END CENTROID LOSS SWITCH
    mse = F.mse_loss(x_recon, x, reduction="mean")

    logvar = torch.log(var)
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    l1 = torch.norm(torch.triu(G, diagonal=1), 1)

    # MCC loss: 1 + partial_mcc_loss maps [perfect=−1, random=0] → [0, 1],
    # making it non-negative and directly comparable to MSE/MMD in scale.
    mcc = torch.tensor(0.0, device=mu.device)
    if lambda_mcc > 0 and E is not None and pi is not None:
        mcc = 1.0 + partial_mcc_loss_torch(E[:, pi], mu)

    return dist, mse, kld, l1, mcc


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class VAEConfig:
    outdir: str = "runs/gcrl_vae"

    # Optimisation
    batch_size: int   = 32
    epochs:     int   = 100
    lr:         float = 1e-3

    # Loss weights
    lambda_sparse: float = 1e-3   # L1 on triu(G)
    lambda_mcc:    float = 1.0    # MCC alignment (use_GRN_priors=True only)

    # Scheduled weights
    beta_kld_max:  float = 2.0    # max KLD weight  (ramps from epoch 10)
    alpha_mmd_max: float = 10.0   # max MMD weight  (ramps from epoch 5)
    temp_max:      float = 5.0    # max softmax temperature

    # MMD kernel
    mmd_kernel_mul: float          = 2.0
    mmd_kernel_num: int            = 5
    mmd_fix_sigma:  Optional[float] = None

    # Model mode flags (passed through to GCRLVAEConfig)
    use_tf_only:    bool = True   # TF-only encoder input (False = all genes, CMVAE style)
    use_GRN_priors: bool = True   # hard routing + MCC loss (False = soft routing, no MCC)

    # Latent dimension override (use_GRN_priors=False only)
    # When None, z_dim defaults to c_dim (one latent per intervened TF, CMVAE default).
    # Set to a smaller integer to get a compressed latent space (e.g. z_dim=6 to match
    # the number of communities used by gCRL-VAE).  Must satisfy 2 <= z_dim <= c_dim.
    # Ignored when use_GRN_priors=True (z_dim is always derived from communities).
    z_dim: Optional[int] = None

    # CENTROID LOSS SWITCH — remove this field to revert
    use_centroid_loss: bool = False  # True: replace MMD with centroid distance loss

    # Misc
    num_workers: int = 0
    seed:        int = 0


# ---------------------------------------------------------------------------
# Batch sampler (discrepancy-VAE SCDATA_sampler style)
# Groups cells by intervention label; each batch is one condition.
# ---------------------------------------------------------------------------

class InterventionBatchSampler(Sampler):
    """
    Yields batches where every cell shares the same intervention label.
    Analogous to discrepancy-VAE's SCDATA_sampler.
    """

    def __init__(self, intervention_labels: np.ndarray, batch_size: int):
        self.batch_size = batch_size
        self.groups: List[np.ndarray] = []
        for label in np.unique(intervention_labels):
            idx = np.where(intervention_labels == label)[0]
            self.groups.append(idx)

    def __iter__(self):
        batches = []
        for idx in self.groups:
            idx = idx.copy()
            np.random.shuffle(idx)
            chunks = [
                idx[i : i + self.batch_size]
                for i in range(0, len(idx), self.batch_size)
            ]
            # drop last incomplete batch (matches discrepancy-VAE chunk() behaviour)
            if chunks and len(chunks[-1]) < self.batch_size:
                chunks = chunks[:-1]
            batches.extend([c.tolist() for c in chunks])
        random.shuffle(batches)
        return iter(batches)

    def __len__(self) -> int:
        return sum(len(g) // self.batch_size for g in self.groups)


# ---------------------------------------------------------------------------
# Dataset  (mirrors discrepancy-VAE SCDataset.__getitem__)
# Each item: (x_ctrl_enc, y_interv, c_binary, x_ctrl_all, eig)
# ---------------------------------------------------------------------------

class _PairedDataset(Dataset):
    """
    Pairs each intervened cell with a randomly sampled control cell.

    At construction time a random control index is drawn for every intervened
    cell (matching discrepancy-VAE's rand_ctrl_samples strategy).  When
    eigengenes are provided (use_GRN_priors=True), the eigengene rows for
    those same sampled controls are stored directly so that the per-batch
    eigengene slice always corresponds to the control cell whose mu is computed
    in that batch step.

    Returns per item: (x_ctrl_enc, y_interv, c_binary, x_ctrl_all, eig)
        x_ctrl_enc : (encoder_input_dim,)  encoder input for the sampled control
        y_interv   : (n_genes,)            ground-truth perturbed expression
        c_binary   : (c_dim,)              binary intervention vector
        x_ctrl_all : (n_genes,)            full expression of the sampled control
        eig        : (n_eig_cols,) or (0,) full eigengene row for the sampled control
    """

    def __init__(
        self,
        X_ctrl:       np.ndarray,           # (n_ctrl, n_genes)
        X_interv:     np.ndarray,           # (n_interv, n_genes)
        X_interv_enc: np.ndarray,           # (n_interv, encoder_input_dim)
        X_ctrl_enc:   np.ndarray,           # (n_ctrl,   encoder_input_dim)
        C:            np.ndarray,           # (n_interv, c_dim)
        A_ctrl:       Optional[np.ndarray], # (n_ctrl, n_eig_cols) or None
        seed:         int = 0,
    ):
        self.X_interv     = torch.from_numpy(X_interv.astype(np.float32))
        self.X_interv_enc = torch.from_numpy(X_interv_enc.astype(np.float32))
        self.C            = torch.from_numpy(C.astype(np.float32))

        # Pre-sample a matched control for each intervened cell (as in SCDataset)
        rng = np.random.default_rng(seed)
        ctrl_idx = rng.integers(0, X_ctrl.shape[0], size=X_interv.shape[0])

        self.X_ctrl     = torch.from_numpy(X_ctrl[ctrl_idx].astype(np.float32))
        self.X_ctrl_enc = torch.from_numpy(X_ctrl_enc[ctrl_idx].astype(np.float32))

        # Eigengenes for the sampled controls (None when use_GRN_priors=False)
        if A_ctrl is not None:
            self.E = torch.from_numpy(A_ctrl[ctrl_idx].astype(np.float32))
        else:
            self.E = None

    def __len__(self) -> int:
        return self.X_interv.shape[0]

    def __getitem__(self, i):
        eig = self.E[i] if self.E is not None else torch.empty(0)
        return self.X_ctrl_enc[i], self.X_interv[i], self.C[i], self.X_ctrl[i], eig


# ---------------------------------------------------------------------------
# Core training function
# ---------------------------------------------------------------------------

def train_gcrl_vae(adata, cfg: VAEConfig, *, eigengenes_key: str = "X_comm_eig"):
    """
    Train a GCRLVAE on adata.

    Parameters
    ----------
    adata : AnnData
        Must contain:
          obs columns : 'set' ('training'/'test'), intervention column, 'cell_type'
          var columns : 'kind' ('TF'/'TG') when use_tf_only=True
                        'community' when use_GRN_priors=True
          obsm[eigengenes_key] : eigengene matrix (n_cells, >= p+1 columns)
                                 when use_GRN_priors=True
    cfg   : VAEConfig
    eigengenes_key : key in adata.obsm for eigengene matrix
                     (used only when use_GRN_priors=True)

    Returns
    -------
    model   : GCRLVAE
    history : list[dict]  per-epoch averaged losses
    """
    from gcrl.models.gcrl_vae import GCRLVAE, GCRLVAEConfig

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(cfg.seed)
    os.makedirs(cfg.outdir, exist_ok=True)

    print("=" * 60)
    print("gCRL-VAE Training")
    print(f"  Device : {device}")
    if device == "cuda":
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Identify keys
    # ------------------------------------------------------------------
    _ctrl_values = {"control", "unperturbed", "na", "NA", "None", "none", ""}

    _ct_key = next(
        (k for k in ["cell_type", "celltype"] if k in adata.obs.columns), None
    )
    if _ct_key is None:
        raise KeyError("adata.obs must contain 'cell_type' or 'celltype'")

    _cond_key = next(
        (k for k in ["intervention", "perturbation", "perturb", "treatment"]
         if k in adata.obs.columns),
        None,
    )
    if _cond_key is None:
        raise KeyError("adata.obs must contain an intervention/perturbation column")

    if "set" not in adata.obs.columns:
        raise KeyError("adata.obs must contain a 'set' column")

    # ------------------------------------------------------------------
    # 2. Split training data
    # ------------------------------------------------------------------
    _is_train = adata.obs["set"].astype(str).values == "training"

    adata_train  = adata[_is_train].copy()
    _is_ctrl_tr  = adata_train.obs[_cond_key].astype(str).isin(_ctrl_values).values
    adata_ctrl   = adata_train[_is_ctrl_tr]
    adata_interv = adata_train[~_is_ctrl_tr]

    if adata_interv.n_obs == 0:
        raise ValueError("No intervened training cells found")

    # Safety check: double perturbations in the training set are not supported.
    # For hard routing (use_GRN_priors=True) the behaviour is silently wrong
    # (argmax picks only one TF); for soft routing the shift magnitude is also
    # incorrect. Move doubles to the test split before calling train_gcrl_vae().
    _train_doubles = adata_interv.obs[_cond_key].astype(str).str.contains(
        r"[+;,]", regex=True
    )
    if _train_doubles.any():
        double_labels = adata_interv.obs.loc[_train_doubles, _cond_key].unique().tolist()
        raise ValueError(
            f"Double perturbations found in the training set: {double_labels}. "
            "Move them to the test split before training "
            "(set adata.obs.loc[mask, 'set'] = 'test')."
        )

    var_names = np.array(adata.var_names.astype(str))
    name2idx  = {g: i for i, g in enumerate(var_names)}

    # ------------------------------------------------------------------
    # 3. Gene indices
    # ------------------------------------------------------------------
    if cfg.use_tf_only:
        if "kind" not in adata.var.columns:
            raise KeyError("adata.var must contain 'kind' when use_tf_only=True")
        tf_mask    = adata.var["kind"].astype(str).values == "TF"
        tf_idx_all = np.where(tf_mask)[0]
    else:
        tf_idx_all = np.arange(adata.n_vars)

    # ------------------------------------------------------------------
    # 4. Intervened TF list and community mapping
    # ------------------------------------------------------------------
    if cfg.use_GRN_priors and "community" not in adata.var.columns:
        raise KeyError("adata.var must contain 'community' when use_GRN_priors=True")

    import pandas as pd

    intervened_tf_names: List[str] = sorted({
        t
        for v in adata.obs[_cond_key].values
        for t in _parse_intervention(str(v))
        if t.lower() not in _ctrl_values
    })

    bad, intervened_tf_idx = [], []
    for g in intervened_tf_names:
        j = name2idx.get(g)
        if j is None:
            bad.append(g)
        elif cfg.use_GRN_priors and str(adata.var.loc[g, "kind"]) != "TF":
            bad.append(g)
        else:
            intervened_tf_idx.append(j)
    if bad:
        raise ValueError(f"Interventions on unknown or non-TF genes: {bad}")

    intervened_tf_idx   = np.array(intervened_tf_idx, dtype=np.int64)
    intervened_tf_names = np.array(intervened_tf_names, dtype=object)

    col_map = {g: k for k, g in enumerate(intervened_tf_names)}

    # Community setup (use_GRN_priors=True only)
    if cfg.use_GRN_priors:
        var_comm = pd.Categorical(adata.var["community"])
        intervened_comm_ids = np.unique(var_comm.codes[intervened_tf_idx])
        intervened_comm_ids = intervened_comm_ids[intervened_comm_ids >= 0]
        comm_order = np.sort(intervened_comm_ids).tolist()
        p = len(comm_order)
        comm2k = {int(c): k for k, c in enumerate(comm_order)}

        # tf_comm_ids[k] = community index (0..p-1) for k-th intervened TF
        tf_comm_ids_np = np.zeros(len(intervened_tf_names), dtype=np.int64)
        for k, g in enumerate(intervened_tf_names):
            c = var_comm.codes[name2idx[g]]
            if int(c) not in comm2k:
                raise ValueError(f"TF {g} community {c} not in intervened communities")
            tf_comm_ids_np[k] = comm2k[int(c)]
    else:
        # soft routing: default z_dim = c_dim (CMVAE default), overridable via cfg.z_dim
        c_dim_auto = len(intervened_tf_names)
        if cfg.z_dim is not None:
            if not (2 <= cfg.z_dim <= c_dim_auto):
                raise ValueError(
                    f"cfg.z_dim={cfg.z_dim} out of range; must be 2 <= z_dim <= c_dim={c_dim_auto}"
                )
            p = cfg.z_dim - 1
        else:
            p = c_dim_auto - 1
        tf_comm_ids_np = None

    # ------------------------------------------------------------------
    # 5. Expression matrices (raw, no z-score — same as discrepancy-VAE)
    # ------------------------------------------------------------------
    X_ctrl_all   = _to_dense(adata_ctrl.X).astype(np.float32)
    X_interv_all = _to_dense(adata_interv.X).astype(np.float32)

    X_ctrl_enc   = X_ctrl_all[:, tf_idx_all]
    X_interv_enc = X_interv_all[:, tf_idx_all]

    # ------------------------------------------------------------------
    # 6. Binary intervention vectors  C  (n_interv, c_dim)
    # ------------------------------------------------------------------
    c_dim = len(intervened_tf_names)
    C = np.zeros((adata_interv.n_obs, c_dim), dtype=np.float32)
    for i, v in enumerate(adata_interv.obs[_cond_key].values):
        for tf in _parse_intervention(str(v)):
            if tf in col_map:
                C[i, col_map[tf]] = 1.0

    intervention_labels = adata_interv.obs[_cond_key].astype(str).values

    # ------------------------------------------------------------------
    # 7. Eigengenes (all columns stored; pi selects p+1 of them at loss time)
    # ------------------------------------------------------------------
    A_ctrl_np: Optional[np.ndarray] = None
    if cfg.use_GRN_priors:
        if eigengenes_key not in adata_ctrl.obsm:
            raise KeyError(
                f"adata_ctrl.obsm['{eigengenes_key}'] not found. "
                "Call compute_eigengenes() before training, or set use_GRN_priors=False."
            )
        A_raw = adata_ctrl.obsm[eigengenes_key]
        A_raw = A_raw.toarray() if hasattr(A_raw, "toarray") else np.asarray(A_raw)
        if A_raw.shape[1] < p + 1:
            raise ValueError(
                f"Eigengene matrix has {A_raw.shape[1]} columns but need at least {p+1}."
            )
        # Store all p+1 columns; column ordering is handled by pi at loss time
        A_ctrl_np = A_raw[:, : p + 1].astype(np.float32)

    # ------------------------------------------------------------------
    # 8. Dataset and DataLoader
    # ------------------------------------------------------------------
    dataset = _PairedDataset(
        X_ctrl_all, X_interv_all, X_interv_enc, X_ctrl_enc, C,
        A_ctrl=A_ctrl_np,
        seed=cfg.seed,
    )
    sampler = InterventionBatchSampler(intervention_labels, cfg.batch_size)
    loader  = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=(device == "cuda"),
    )

    # ------------------------------------------------------------------
    # 9. Build model
    # ------------------------------------------------------------------
    n_genes = X_ctrl_all.shape[1]
    n_tfs   = X_ctrl_enc.shape[1]
    z_dim   = max(p + 1, 2)

    cfg_m = GCRLVAEConfig(
        input_dim=n_tfs,
        z_dim=z_dim,
        c_dim=c_dim,
        output_dim=n_genes,
        hidden_enc=128,
        intervention_mapping="hard" if cfg.use_GRN_priors else "soft",
        use_tf_only=cfg.use_tf_only,
        use_GRN_priors=cfg.use_GRN_priors,
    )
    model = GCRLVAE(cfg_m).to(device)
    model.set_tf_index(list(map(str, intervened_tf_names)))
    if cfg.use_GRN_priors:
        # tf_comm_ids[k] = community index for k-th TF; S (identity init) maps
        # community k -> latent dimension k at the start
        model.set_tf_comm(tf_comm_ids_np)

    # Save eval wiring (pi will be updated at end of training)
    wiring = {
        "tf_names":       list(map(str, intervened_tf_names)),
        "tf_comm_ids":    tf_comm_ids_np.tolist() if tf_comm_ids_np is not None else None,
        "z_dim":          int(z_dim),
        "c_dim":          int(c_dim),
        "routing":        "hard" if cfg.use_GRN_priors else "soft",
        "use_tf_only":    cfg.use_tf_only,
        "use_GRN_priors": cfg.use_GRN_priors,
        "perm":           None,  # filled after training
    }

    # ------------------------------------------------------------------
    # 10. Optimizer and schedules (discrepancy-VAE style)
    # S is already in model.parameters() — no special registration needed
    # ------------------------------------------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    beta_schedule  = np.zeros(cfg.epochs)
    beta_schedule[10:] = np.linspace(0, cfg.beta_kld_max, max(1, cfg.epochs - 10))

    alpha_schedule = np.full(cfg.epochs, cfg.alpha_mmd_max)
    alpha_schedule[:5] = 0.0
    mid = cfg.epochs // 2
    alpha_schedule[5:mid] = np.linspace(0, cfg.alpha_mmd_max, max(1, mid - 5))

    temp_schedule = np.ones(cfg.epochs)
    temp_schedule[5:] = np.linspace(1.0, cfg.temp_max, max(1, cfg.epochs - 5))

    mmd_criterion = MMD_loss(
        kernel_mul=cfg.mmd_kernel_mul,
        kernel_num=cfg.mmd_kernel_num,
        fix_sigma=cfg.mmd_fix_sigma,
    ).to(device)

    # ------------------------------------------------------------------
    # 11. Training loop
    # ------------------------------------------------------------------
    history    = []
    min_loss   = np.inf
    best_model = deepcopy(model)

    epoch_pbar = tqdm(range(cfg.epochs), desc="gCRL-VAE", unit="epoch")

    for epoch in epoch_pbar:
        model.train()
        beta  = float(beta_schedule[epoch])
        alpha = float(alpha_schedule[epoch])
        temp  = float(temp_schedule[epoch])

        # CENTROID LOSS SWITCH
        _dist_key = "dist" if cfg.use_centroid_loss else "mmd"
        agg = {"loss": 0., _dist_key: 0., "mse": 0., "kld": 0., "l1": 0., "mcc": 0.}
        n_batches = 0

        for x_ctrl_enc, y_interv, c_bin, x_ctrl_all, eig in loader:
            x_ctrl_enc = x_ctrl_enc.to(device)
            y_interv   = y_interv.to(device)
            c_bin      = c_bin.to(device)
            x_ctrl_all = x_ctrl_all.to(device)

            optimizer.zero_grad(set_to_none=True)

            y_hat, x_recon, mu, var, G = model(
                x_ctrl_enc, c_bin, c_bin, num_interv=1, temp=temp
            )

            # Derive pi from S for MCC loss (hard, non-differentiable)
            if cfg.use_GRN_priors:
                E  = eig.to(device)
                pi = model._hard_perm()   # (z_dim,) LongTensor
            else:
                E  = None
                pi = None

            # CENTROID LOSS SWITCH — remove use_centroid_loss kwarg to revert
            dist, mse, kld, l1, mcc = loss_function(
                y_hat, y_interv, x_recon, x_ctrl_all, mu, var, G,
                mmd_criterion,
                E=E,
                pi=pi,
                lambda_mcc=cfg.lambda_mcc if cfg.use_GRN_priors else 0.0,
                use_centroid_loss=cfg.use_centroid_loss,
            )
            # END CENTROID LOSS SWITCH

            loss = (
                alpha * dist
                + mse
                + beta * kld
                + cfg.lambda_sparse * l1
                + cfg.lambda_mcc * mcc
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            for key, val in zip(
                ["loss", _dist_key, "mse", "kld", "l1", "mcc"],
                [loss,   dist,      mse,   kld,   l1,   mcc],
            ):
                agg[key] += float(val.detach().cpu())
            n_batches += 1

        if n_batches == 0:
            continue

        avg = {k: v / n_batches for k, v in agg.items()}
        avg["epoch"] = epoch
        avg["beta"]  = beta
        avg["alpha"] = alpha
        history.append(avg)

        epoch_pbar.set_postfix({
            "loss":    f"{avg['loss']:.4f}",
            _dist_key: f"{avg[_dist_key]:.4f}",
            "mse":     f"{avg['mse']:.4f}",
            "mcc":     f"{avg['mcc']:.4f}",
        })

        # Save best model by training loss to disk (for reference); last epoch is returned
        train_loss = avg[_dist_key] + avg["mse"] + avg["kld"] + avg["l1"]
        if train_loss < min_loss:
            min_loss   = train_loss
            best_model = deepcopy(model)
            torch.save(best_model, os.path.join(cfg.outdir, "best_model.pt"))

    torch.save(model, os.path.join(cfg.outdir, "last_model.pt"))

    # Save final permutation to wiring file
    if cfg.use_GRN_priors:
        wiring["perm"] = model._hard_perm().cpu().tolist()
    with open(os.path.join(cfg.outdir, "eval_wiring.json"), "w") as f:
        json.dump(wiring, f, indent=2)

    with open(os.path.join(cfg.outdir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"Training complete. Last model returned; both models saved to {cfg.outdir}/")
    return model, history
