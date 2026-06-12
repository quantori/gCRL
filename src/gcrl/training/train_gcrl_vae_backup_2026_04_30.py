#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GCRL-VAE trainer (p+1 hard-intervention mapping) with:
- TF-ONLY INPUT: Encoder takes only TF expression (adata.var['kind'] == 'TF')
- ALL-GENE OUTPUT: Decoder reconstructs all genes (TF + TG)
- POLYNOMIAL DECODER: Linear + quadratic terms for interpretability
- Interventions parsed from adata.obs['intervention']
- tf_to_latent built ONLY for intervened TFs (columns of C); error if intervention TF not found or not kind=='TF'
- Single-CT batching using adata.obs['cell_type']
- Prefer per-cell eigengenes from adata.obsm["X_comm_eig"] (+ adata.uns metadata); fallback to on-the-fly PC1
- Alignment via partial_mcc_loss_torch(E, MU) on (B x (p+1)) matrices
- Training uses adata.X (normalized, log1p)
- Complete loss: MMD + KL + Reconstruction + Alignment + L1 Sparsity + Centroid (optional)
- Controls-only reconstruction with scheduled MMD/KL losses
- MMD computed on ORIGINAL intervened cell expression (simulated vs. real)
- Centroid loss uses same schedule as MMD for consistency

LOSS FUNCTION (7 components):
    L_total = 1.0 × L_rec                          # Reconstruction (constant)
            + β(t) × L_KL                          # β: 0 → 0.01 (epochs 10+)
            + 1.0 × L_mcc                          # Alignment (constant)
            + 0.001 × L_sparse                     # Sparsity (tiny, epochs 10+)
            + 1.0 × L_dag                          # DAG acyclicity (NOTEARS, epochs 10+)
            + α(t) × L_MMD                         # α: 0 → 1.0 (epochs 5+)
            + α(t) × 0.1 × L_centroid              # Scheduled weight × constant weight

    Key insights:
    - Centroid has BOTH schedule (α) AND weight (0.1) = [0 → 0.1] (10× weaker than MMD)
    - Sparse & DAG penalties start at epoch 10 to allow early learning freedom

DATA NAMING CONVENTIONS:
    - *_tf: TF expression only (encoder input) - shape: (n_samples, n_TFs)
    - *_all or no suffix: ALL genes (decoder output) - shape: (n_samples, n_genes)
    - "_all" emphasizes: contains TF + TG genes, not just TFs
"""
from __future__ import annotations
import os, re, json
from dataclasses import dataclass
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

try:
    from tqdm.auto import tqdm
except ImportError:
    # Fallback if tqdm not available
    def tqdm(iterable, **kwargs):
        return iterable


# Import your canonical partial MCC
from gcrl.alignment.partial_mcc import partial_mcc_loss_torch


# ----------------- helpers -----------------
def set_seed(seed: int = 0):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_dense(x):
    try:
        import scipy.sparse as sp

        if sp.issparse(x):
            return x.toarray()
    except Exception:
        pass
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def _zscore(t: torch.Tensor, dim: int = 0, eps: float = 1e-8) -> torch.Tensor:
    mu = t.mean(dim=dim, keepdim=True)
    std = t.std(dim=dim, unbiased=False, keepdim=True).clamp_min(eps)
    return (t - mu) / std


def _first_pc_scores(X: torch.Tensor) -> torch.Tensor:
    Xc = X - X.mean(dim=0, keepdim=True)
    n, _ = Xc.shape
    v = torch.randn(n, device=X.device, dtype=X.dtype)
    v = v / (v.norm() + 1e-8)
    for _ in range(12):
        v = Xc @ (Xc.t() @ v)
        v = v / (v.norm() + 1e-8)
    return _zscore(v, dim=0)


def _build_eigengene_matrix_for_batch(
    X_batch_tf: torch.Tensor, tf_comm_ids_full: torch.Tensor, comm_order: List[int]
) -> torch.Tensor:
    E_cols = []
    for c in comm_order:
        mask = tf_comm_ids_full == c
        if mask.sum() < 2:
            E_cols.append(
                torch.zeros(
                    X_batch_tf.shape[0],
                    device=X_batch_tf.device,
                    dtype=X_batch_tf.dtype,
                )
            )
        else:
            E_cols.append(_first_pc_scores(X_batch_tf[:, mask]))
    if X_batch_tf.shape[1] >= 2:
        E_cols.append(_first_pc_scores(X_batch_tf))
    else:
        E_cols.append(
            torch.zeros(
                X_batch_tf.shape[0], device=X_batch_tf.device, dtype=X_batch_tf.dtype
            )
        )
    return torch.stack(E_cols, dim=1)  # (B, p+1)


# ----------------- MMD Loss (from discrepancy-VAE) -----------------
class MMD_loss(nn.Module):
    """
    Maximum Mean Discrepancy loss with multi-scale Gaussian kernels.
    From: https://github.com/uhlerlab/discrepancy_vae

    Compares distributions in low-dimensional latent space using adaptive bandwidth.
    """

    def __init__(
        self,
        kernel_mul: float = 2.0,
        kernel_num: int = 5,
        fix_sigma: Optional[float] = None,
    ):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma

    def gaussian_kernel(
        self, source: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute multi-scale Gaussian kernel matrix.

        Args:
            source: (n_source, dim)
            target: (n_target, dim)

        Returns:
            kernel_val: ((n_source + n_target), (n_source + n_target))
        """
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        # Compute pairwise L2 distances
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1))
        )
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1))
        )
        L2_distance = ((total0 - total1) ** 2).sum(2)

        # Adaptive bandwidth selection
        if self.fix_sigma:
            bandwidth = self.fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2 - n_samples)
            bandwidth = bandwidth.clamp_min(1e-8)

        # Multi-scale kernels
        bandwidth /= self.kernel_mul ** (self.kernel_num // 2)
        bandwidth_list = [
            bandwidth * (self.kernel_mul**i) for i in range(self.kernel_num)
        ]

        # Compute kernel values across scales
        kernel_val = [torch.exp(-L2_distance / (bw + 1e-8)) for bw in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute MMD loss between source and target distributions.

        Args:
            source: (batch_size, dim) - simulated intervention latents
            target: (batch_size, dim) - real intervention latents

        Returns:
            mmd_loss: scalar
        """
        batch_size = int(source.size()[0])
        kernels = self.gaussian_kernel(source, target)

        # Split kernel matrix into blocks
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]

        # MMD^2 = E[k(x,x')] + E[k(y,y')] - 2E[k(x,y)]
        loss = torch.mean(XX + YY - XY - YX)
        return loss


def notears_acyclicity(G: torch.Tensor, use_poly_approx: bool = True) -> torch.Tensor:
    """
    NOTEARS-style smooth acyclicity constraint on the full adjacency matrix G.

    h(G) = tr(exp(G ⊙ G)) - d, where ⊙ is elementwise product.
    We remove self-loops from consideration by zeroing the diagonal.

    Args:
        G: Adjacency matrix (d × d)
        use_poly_approx: If True, use memory-efficient polynomial approximation.
                        If False, use exact matrix exponential (memory-intensive!).

    Memory Optimization:
        The exact matrix exponential torch.matrix_exp() is extremely memory-intensive
        during backpropagation, especially for large batch sizes. The polynomial
        approximation (default) provides a good balance of accuracy and memory efficiency.
    """
    # Remove self-loops
    G_offdiag = G - torch.diag_embed(torch.diag(G))
    # Elementwise square to ensure non-negativity as in original NOTEARS
    A = G_offdiag * G_offdiag
    d = G.shape[0]

    if use_poly_approx:
        # Memory-efficient polynomial approximation of tr(exp(A))
        # exp(A) ≈ I + A + A²/2 + A³/6 + A⁴/24
        # tr(exp(A)) ≈ d + tr(A) + tr(A²)/2 + tr(A³)/6 + tr(A⁴)/24
        # For small latent dims (d < 20), this approximation is very accurate
        A2 = A @ A
        A3 = A2 @ A
        A4 = A3 @ A

        h = (
            torch.trace(A)
            + torch.trace(A2) / 2.0
            + torch.trace(A3) / 6.0
            + torch.trace(A4) / 24.0
        )
    else:
        # Exact but memory-intensive (use only for small models or debugging)
        exp_A = torch.matrix_exp(A)
        h = torch.trace(exp_A) - d

    return h


class SingleCTBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        ct: np.ndarray,
        batch_size: int,
        drop_last: bool = False,
        shuffle: bool = True,
    ):
        self.ct = ct.astype(int)
        self.batch_size = int(batch_size)
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.ct_to_idx: Dict[int, List[int]] = {}
        for i, c in enumerate(self.ct):
            self.ct_to_idx.setdefault(int(c), []).append(i)
        self.cts = list(self.ct_to_idx.keys())

    def __iter__(self):
        from random import shuffle as rshuffle, choice

        pools = {c: idxs.copy() for c, idxs in self.ct_to_idx.items()}
        if self.shuffle:
            for v in pools.values():
                rshuffle(v)
        active = {c: 0 for c in self.cts}
        while True:
            cands = [c for c in self.cts if active[c] < len(pools[c])]
            if not cands:
                break
            c = choice(cands)
            start = active[c]
            end = start + self.batch_size
            if end <= len(pools[c]):
                batch = pools[c][start:end]
                active[c] = end
                yield batch
            else:
                if not self.drop_last and start < len(pools[c]):
                    batch = pools[c][start : len(pools[c])]
                    active[c] = len(pools[c])
                    yield batch
                else:
                    active[c] = len(pools[c])

    def __len__(self):
        import math

        total = 0
        for idxs in self.ct_to_idx.values():
            total += (
                (len(idxs) // self.batch_size)
                if self.drop_last
                else int(np.ceil(len(idxs) / self.batch_size))
            )
        return total


# ----------------- config -----------------
@dataclass
class VAEConfig:
    outdir: str = "runs/pplus1"
    intervention_mapping: str = "hard"
    batch_size: int = 256
    epochs: int = 50
    lr: float = 2e-3

    # Loss weights (constant)
    alpha_rec: float = 1.0  # Reconstruction loss weight
    lambda_mcc: float = 1.0  # Eigengene alignment (partial-MCC) weight
    lambda_sparse: float = (
        1e-3  # L1 sparsity on causal matrix G (tiny; acts as tie-breaker)
    )
    lambda_centroid: float = 0.1  # Centroid distance loss WEIGHT (set to 0 to disable)
    lambda_dag: float = 1.0  # NOTEARS DAG acyclicity penalty weight
    dag_start_epoch: int = 10  # Epoch at which to start applying DAG penalty
    sparse_start_epoch: int = 10  # Epoch at which to start L1 sparsity on G
    # NOTE: Effective weight = alpha_mmd(t) × lambda_centroid
    # - alpha_mmd(t): SCHEDULE (0 → 1.0 from epoch 5+)
    # - lambda_centroid: WEIGHT (constant 0.1)
    # - Result: centroid ramps from 0 → 0.1 (10× weaker than MMD)

    # Loss weights (scheduled - ramp from 0)
    beta_kld_max: float = 1e-2  # KL divergence max weight (ramps from epoch 10)
    # Increased from 1e-3 to 1e-2 for stronger regularization
    alpha_mmd_max: float = 1.0  # MMD loss max weight (ramps from epoch 5)
    # This schedule also applies to centroid loss

    # MMD kernel parameters
    mmd_kernel_mul: float = 2.0
    mmd_kernel_num: int = 5
    mmd_fix_sigma: Optional[float] = None  # None = adaptive bandwidth

    # Temperature scheduling for soft routing (if used)
    temp_start: float = 1.0
    temp_max: float = 10.0

    # Preprocessing parameters
    preproc_epsilon: float = 0.1  # Z-score shrinkage epsilon
    preproc_clip_value: float = 6.0  # Z-score clipping threshold

    # Training parameters
    num_workers: int = 0
    seed: int = 0


# ----------------- Auto-configuration -----------------
def analyze_dataset_and_suggest_config(
    adata, outdir: str = "runs/pplus1", verbose: bool = True
) -> VAEConfig:
    """
    Analyze dataset characteristics and suggest optimal VAEConfig parameters.

    This function examines:
    - Number of control cells per cell type (affects batch size)
    - Number of interventions (affects training strategy)
    - Number of latent factors/communities (affects regularization)
    - Dataset size (affects epochs and learning rate)

    Args:
        adata: AnnData object with preprocessed data
        outdir: Output directory for results
        verbose: If True, print detailed analysis

    Returns:
        VAEConfig with suggested parameters
    """
    # Identify keys
    _ctrl_values = {"control", "unperturbed", "na", "NA", "None", "none", ""}
    _cond_key = None
    for _k in ["intervention", "perturbation", "perturb", "treatment"]:
        if _k in adata.obs.columns:
            _cond_key = _k
            break
    if _cond_key is None:
        raise KeyError("Could not find intervention/perturbation column in adata.obs")

    _ct_key = "cell_type" if "cell_type" in adata.obs.columns else "celltype"
    if _ct_key not in adata.obs.columns:
        raise KeyError("adata.obs must contain a cell-type column")

    if "set" not in adata.obs.columns:
        raise KeyError("adata.obs must contain a 'set' column")

    # Extract dataset characteristics
    _is_train = adata.obs["set"].astype(str).values == "training"
    _ctrl_mask = adata.obs[_cond_key].astype(str).isin(_ctrl_values).values
    _is_ctrl_train = _is_train & _ctrl_mask

    n_total_cells = adata.n_obs
    n_train_cells = _is_train.sum()
    n_ctrl_train = _is_ctrl_train.sum()
    n_interv_train = (_is_train & ~_ctrl_mask).sum()

    # Cell types
    cell_types = adata.obs.loc[_is_train, _ct_key].unique()
    n_cell_types = len(cell_types)

    # Controls per cell type
    ctrl_per_ct = adata.obs.loc[_is_ctrl_train, _ct_key].value_counts()
    min_ctrl_per_ct = ctrl_per_ct.min() if len(ctrl_per_ct) > 0 else 0
    mean_ctrl_per_ct = ctrl_per_ct.mean() if len(ctrl_per_ct) > 0 else 0

    # Interventions
    interventions = adata.obs.loc[_is_train & ~_ctrl_mask, _cond_key].unique()
    n_interventions = len(interventions)

    # TFs and communities
    if "kind" not in adata.var.columns:
        raise KeyError("adata.var must contain 'kind' column")
    if "community" not in adata.var.columns:
        raise KeyError("adata.var must contain 'community' column")

    n_tfs = (adata.var["kind"] == "TF").sum()
    n_genes = adata.n_vars
    communities = (
        adata.var.loc[adata.var["kind"] == "TF", "community"].dropna().unique()
    )
    n_communities = len(communities)
    n_latent = n_communities + 1  # p + 1

    if verbose:
        print("\n" + "=" * 70)
        print("Dataset Analysis for gCRL-VAE Configuration")
        print("=" * 70)
        print(f"\n📊 Dataset Size:")
        print(f"  - Total cells: {n_total_cells:,}")
        print(
            f"  - Training cells: {n_train_cells:,} ({100*n_train_cells/n_total_cells:.1f}%)"
        )
        print(f"  - Control cells (training): {n_ctrl_train:,}")
        print(f"  - Intervened cells (training): {n_interv_train:,}")
        print(f"\n🧬 Gene Information:")
        print(f"  - Total genes: {n_genes:,}")
        print(f"  - Transcription factors (TFs): {n_tfs}")
        print(f"  - Target genes (TGs): {n_genes - n_tfs}")
        print(f"\n🔬 Cell Types:")
        print(f"  - Number of cell types: {n_cell_types}")
        print(
            f"  - Controls per cell type: min={min_ctrl_per_ct}, mean={mean_ctrl_per_ct:.0f}"
        )
        print(f"\n🧪 Interventions:")
        print(f"  - Number of unique interventions: {n_interventions}")
        print(f"\n🧠 Latent Structure:")
        print(f"  - TF communities: {n_communities}")
        print(f"  - Latent dimensions (p+1): {n_latent}")

    # ========== INTELLIGENT PARAMETER SELECTION ==========

    # 1. BATCH SIZE - based on controls per cell type and GPU memory
    # UPDATED (2025-11-23): Improved memory estimation with GPU query
    # - Accounts for all memory components (encoder, decoder, MMD, gradients)
    # - Queries actual GPU memory for adaptive sizing
    # - More conservative target to prevent OOM

    def estimate_batch_memory_gb(bs, n_genes, z_dim, n_tfs):
        """
        Comprehensive GPU memory estimation for one training batch.

        Accounts for:
        - Model parameters (encoder + decoder)
        - Optimizer states (Adam: 2 states per parameter)
        - Forward pass activations (including polynomial decoder intermediate tensors)
        - Backward pass gradients
        - MMD loss kernel matrices (quadratic in batch size!)
        - Eigengene computation (covariance matrices)
        - PyTorch memory allocator overhead and fragmentation

        Args:
            bs: Batch size
            n_genes: Total number of genes (TF + TG)
            z_dim: Latent dimension (p + 1)
            n_tfs: Number of transcription factors

        Returns:
            Estimated peak GPU memory in GB
        """
        # Model parameters (float32 = 4 bytes)
        encoder_params = n_tfs * 128 + 128 * z_dim * 2  # input -> hidden -> (mu, var)
        decoder_params = (z_dim + z_dim**2) * n_genes   # linear + quadratic terms
        total_params = encoder_params + decoder_params

        # Parameter memory: params + gradients + 2 Adam states (m, v)
        param_mem = total_params * 4 * 4 / 1e9  # 4 bytes × 4 (param + grad + 2 states)

        # Forward pass activations (stored for backward pass)
        # CRITICAL: Polynomial decoder creates LARGE intermediate tensors!
        forward_mem = (
            bs * n_tfs +              # encoder input (x_tf)
            bs * n_genes +            # full gene expression (x_all, target)
            bs * 128 +                # encoder hidden layer
            bs * z_dim * 2 +          # mu, var (encoder output)
            bs * z_dim +              # z samples (reparameterization)
            bs * z_dim +              # u (DAG transform)
            bs * z_dim * z_dim +      # Quadratic input (z ⊗ z for polynomial decoder)
            bs * n_genes +            # Linear decoder output
            bs * n_genes +            # Quadratic decoder output
            bs * n_genes              # Final reconstruction
        ) * 4 / 1e9

        # MMD loss memory (LARGE for big batch sizes!)
        # CRITICAL: MMD is computed on HALF the batch (bs//2 simulated vs bs//2 real)
        # So effective batch for MMD is bs//2, kernel is (bs) × (bs)
        mmd_batch = bs // 2  # Actual samples per group
        mmd_kernel_mem = (2 * mmd_batch) ** 2 * 4 / 1e9 * 5  # 5 Gaussian kernels
        mmd_data_mem = (
            mmd_batch * n_genes +     # x_sim (simulated)
            mmd_batch * n_genes +     # X_real_all (ground truth)
            mmd_batch * n_tfs +       # X_ctrl_sample_tf
            (2 * mmd_batch) * n_genes # Distance matrix for kernel computation
        ) * 4 / 1e9
        mmd_mem = mmd_kernel_mem + mmd_data_mem

        # Eigengene alignment memory (partial MCC computation)
        # Includes eigengene matrix E and covariance computations
        eigengene_mem = (
            bs * (z_dim + 1) +        # Eigengene matrix E
            bs * n_tfs * z_dim +      # TF expression subsets for communities
            (z_dim + 1) ** 2 * 2      # Covariance matrices for partial correlation
        ) * 4 / 1e9

        # Backward pass memory
        # CRITICAL: Backward through polynomial decoder is memory-intensive
        # Gradients for quadratic terms create large intermediate tensors
        backward_mem = forward_mem * 3  # 3x forward (more conservative than 2x)

        # DAG transform memory (matrix inversion and propagation)
        dag_mem = (
            z_dim * z_dim * 4 +       # G matrix operations
            bs * z_dim * 3            # Multiple latent transformations
        ) * 4 / 1e9

        # PyTorch memory allocator overhead + fragmentation
        # CRITICAL: Real-world testing shows the polynomial decoder + autograd
        # consumes 10-15x more memory than the naive calculation suggests!
        #
        # Empirical calibration (Norman2019: 5000 genes, batch_size=384 works, 512 OOMs):
        # - Naive calculation: 0.20 GB
        # - Real usage: ~10-12 GB
        # - Overhead factor needed: ~50-60x
        #
        # This massive overhead comes from:
        # 1. PyTorch's autograd graph (stores all intermediate tensors)
        # 2. Polynomial decoder quadratic terms (huge intermediate matrices)
        # 3. Memory fragmentation (PyTorch allocator can't reuse fragmented blocks)
        # 4. Gradient checkpointing NOT enabled (would reduce memory but slow training)

        # Base overhead: very conservative
        base_overhead = 10.0  # 10x minimum (empirically validated)

        # Additional scaling for large gene counts (polynomial decoder scales badly!)
        # The quadratic decoder creates (bs × z²) @ (z² × n_genes) which requires
        # materializing the full (bs × n_genes) matrix multiple times during backprop
        if n_genes > 4000:
            gene_factor = 5.0  # 5x additional for >4000 genes (total 50x!)
        elif n_genes > 2500:
            gene_factor = 3.5  # 3.5x additional for >2500 genes (total 35x)
        elif n_genes > 1500:
            gene_factor = 2.0  # 2x additional for >1500 genes (total 20x)
        else:
            gene_factor = 1.0  # Standard overhead for small datasets (total 10x)

        overhead_factor = base_overhead * gene_factor

        total = (param_mem + forward_mem + backward_mem + mmd_mem +
                eigengene_mem + dag_mem) * overhead_factor
        return total

    def auto_select_batch_size(n_genes, z_dim, n_tfs, target_gpu_util=0.35):
        """
        Automatically select optimal batch size based on available GPU memory.

        Uses binary search to find the largest batch size that fits within
        target GPU utilization (default 35% - conservative but not wasteful).

        Args:
            n_genes: Total number of genes
            z_dim: Latent dimension
            n_tfs: Number of TFs
            target_gpu_util: Fraction of GPU memory to use (default 0.35 = 35%)

        Returns:
            Optimal batch size (power of 2)
        """
        # Query GPU memory if available
        if torch.cuda.is_available():
            gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            target_mem_gb = gpu_mem_gb * target_gpu_util
        else:
            # CPU fallback: assume moderate memory, use conservative batch size
            return 128

        # Binary search for optimal batch size
        min_bs, max_bs = 16, 1024
        best_bs = min_bs

        while min_bs <= max_bs:
            mid_bs = (min_bs + max_bs) // 2
            estimated = estimate_batch_memory_gb(mid_bs, n_genes, z_dim, n_tfs)

            if estimated <= target_mem_gb:
                best_bs = mid_bs
                min_bs = mid_bs + 1
            else:
                max_bs = mid_bs - 1

        # Round down to nearest power of 2 for efficiency
        import math
        best_bs = 2 ** int(math.log2(max(best_bs, 1)))

        return max(16, best_bs)  # Minimum batch size of 16

    # Determine ideal batch size based on dataset size
    if min_ctrl_per_ct < 100:
        ideal_batch_size = 64
    elif min_ctrl_per_ct < 500:
        ideal_batch_size = 128
    elif min_ctrl_per_ct < 2000:
        ideal_batch_size = 256
    else:
        ideal_batch_size = 512

    # Auto-select batch size based on GPU memory (more accurate!)
    memory_aware_batch_size = auto_select_batch_size(n_genes, n_latent, n_tfs)

    # Use the smaller of ideal and memory-aware batch sizes
    batch_size = min(ideal_batch_size, memory_aware_batch_size)

    # Estimate memory for selected batch size
    estimated_mem = estimate_batch_memory_gb(batch_size, n_genes, n_latent, n_tfs)

    # Warn if still high (should rarely happen with new approach)
    memory_warning = estimated_mem > 8.0

    # 2. EPOCHS - based on dataset size and complexity
    if n_ctrl_train < 1000:
        epochs = 100  # More epochs for small datasets
    elif n_ctrl_train < 5000:
        epochs = 75
    elif n_ctrl_train < 20000:
        epochs = 50
    else:
        epochs = 40  # Fewer epochs for large datasets

    # 3. LEARNING RATE - based on batch size and dataset size
    if batch_size <= 128:
        lr = 1e-3  # Lower lr for small batches
    else:
        lr = 2e-3  # Standard lr

    # 4. ALPHA_REC - reconstruction weight
    # UPDATED: Based on v1 vs v2 analysis, dramatic increases backfire
    # v1's alpha_rec=1.0 outperformed v2's alpha_rec=5.0 by 73%!
    # Keep it simple: use 1.0 for most cases, slight boost for extreme expansions only
    expansion_ratio = n_genes / n_tfs
    if expansion_ratio > 50:
        alpha_rec = 2.0  # Very high expansion, gentle boost
    elif expansion_ratio > 30:
        alpha_rec = 1.5  # High expansion, minimal boost
    else:
        alpha_rec = 1.0  # Standard (PROVEN to work on Norman2019 with 19.4× expansion)

    # 5. BETA_KLD_MAX - KL divergence regularization
    # UPDATED: Based on v1 vs v2 analysis, over-regularization backfires
    # v2's beta_kld_max=0.05 resulted in WORSE KL (1.05) than v1's 0.01 (0.64)!
    # High beta values constrain latent space too much, causing KL to increase
    # Keep regularization gentle for most cases
    if n_ctrl_train < 1000 or n_latent > 15:
        beta_kld_max = 0.02  # Gentle regularization for small/complex cases
    else:
        beta_kld_max = 0.01  # Standard gentle regularization (PROVEN on Norman2019)

    # 6. LAMBDA_MCC - alignment weight (CRITICAL for gCRL-VAE!)
    # UPDATED: Based on v1 vs v2 analysis, alignment is FOUNDATIONAL
    # v2's lambda_mcc=0.5 undermined the entire model structure
    # v1's lambda_mcc=0.75 achieved excellent alignment AND better reconstruction
    # gCRL-VAE requires strong eigengene alignment - never weaken this!
    if n_communities <= 3:
        lambda_mcc = 0.85  # Fewer communities, strong alignment needed
    elif n_communities <= 8:
        lambda_mcc = 0.75  # Standard (PROVEN on Norman2019 with 6 communities)
    else:
        lambda_mcc = 0.65  # Many communities, can slightly reduce (but stay strong!)

    # 7. LAMBDA_SPARSE - based on latent dimension
    # Higher dimensional latent spaces need more sparsity
    if n_latent <= 5:
        lambda_sparse = 1e-3
    elif n_latent <= 10:
        lambda_sparse = 2e-3
    else:
        lambda_sparse = 5e-3

    # 8. ALPHA_MMD_MAX - based on intervention diversity
    # More interventions → can reduce MMD weight
    if n_interventions < 10:
        alpha_mmd_max = 1.5  # Strong intervention matching
    elif n_interventions < 30:
        alpha_mmd_max = 1.0  # Standard
    else:
        alpha_mmd_max = 0.75  # Many interventions, reduce weight

    # 9. LAMBDA_CENTROID - based on dataset characteristics
    # Enable by default, but reduce for many interventions
    if n_interventions < 20:
        lambda_centroid = 0.1
    else:
        lambda_centroid = 0.05

    # 10. PREPROCESSING - based on data variance
    preproc_epsilon = 0.1
    preproc_clip_value = 6.0

    cfg = VAEConfig(
        outdir=outdir,
        intervention_mapping="hard",
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        alpha_rec=alpha_rec,  # Now adaptive based on decoder complexity
        lambda_mcc=lambda_mcc,
        lambda_sparse=lambda_sparse,
        lambda_centroid=lambda_centroid,
        beta_kld_max=beta_kld_max,
        alpha_mmd_max=alpha_mmd_max,
        mmd_kernel_mul=2.0,
        mmd_kernel_num=5,
        mmd_fix_sigma=None,
        temp_start=1.0,
        temp_max=10.0,
        preproc_epsilon=preproc_epsilon,
        preproc_clip_value=preproc_clip_value,
        num_workers=0,
        seed=0,
    )

    if verbose:
        print(f"\n⚙️  Suggested Configuration:")

        # Show GPU info and memory-adjusted batch size
        if torch.cuda.is_available():
            gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  - GPU Memory: {gpu_mem_gb:.1f} GB available")
            print(f"    → Target usage: 35% ({gpu_mem_gb * 0.35:.1f} GB) for safe training")

        # Show batch size selection logic
        if batch_size < ideal_batch_size:
            print(f"  - Batch size: {batch_size} (GPU memory limited)")
            print(f"    → Ideal from dataset: {ideal_batch_size}")
            print(f"    → GPU memory allows: {memory_aware_batch_size}")
            print(f"    → Selected: {batch_size} (conservative)")
            print(f"    → Estimated memory: {estimated_mem:.1f} GB")
        else:
            print(f"  - Batch size: {batch_size} (dataset optimal)")
            print(f"    → Based on {min_ctrl_per_ct} min controls/cell-type")
            print(f"    → GPU memory allows: {memory_aware_batch_size}")
            print(f"    → Estimated memory: {estimated_mem:.1f} GB")

        print(f"  - Epochs: {epochs} (based on {n_ctrl_train:,} control cells)")
        print(f"  - Learning rate: {lr} (based on batch size)")
        print(
            f"  - Alpha Rec: {alpha_rec} (based on {expansion_ratio:.1f}× TF→gene expansion)"
        )
        print(
            f"  - Beta KLD max: {beta_kld_max} (based on {n_ctrl_train:,} cells, {n_latent} latent dims)"
        )
        print(f"  - Lambda MCC: {lambda_mcc} (based on {n_communities} communities)")
        print(f"  - Lambda Sparse: {lambda_sparse} (based on {n_latent} latent dims)")
        print(
            f"  - Alpha MMD max: {alpha_mmd_max} (based on {n_interventions} interventions)"
        )
        print(
            f"  - Lambda Centroid: {lambda_centroid} (based on {n_interventions} interventions)"
        )
        print("\n💡 Configuration Philosophy:")
        print("  ✅ Balanced weights (not aggressive individual weights)")
        print(
            "  ✅ Strong alignment (lambda_mcc ≥ 0.65) - critical for gCRL-VAE structure"
        )
        print("  ✅ Gentle regularization (beta_kld ≤ 0.02) - avoid over-constraint")
        print("  ✅ Moderate reconstruction weight - dramatic increases backfire")
        print("  → Expected: Good reconstruction, excellent alignment, controlled KL")

        # Memory optimization warnings and recommendations
        if memory_warning or n_genes > 3500:
            print("\n⚠️  High Memory Usage Detected:")
            print(f"  - Dataset: {n_genes:,} genes, batch_size={batch_size}")
            print(f"  - Estimated peak GPU memory: {estimated_mem:.1f} GB")

            if memory_warning:
                print(f"  - ⚠️  WARNING: Estimated memory > 8 GB may cause OOM errors!")

            print("\n  💡 Recommendations to reduce memory:")
            print(f"     1. Reduce genes (recommended): Keep top 2000-3000 HVGs")
            print(f"        → Current: {n_genes:,} genes")
            print(f"        → Target:  2500 genes would use ~{estimate_batch_memory_gb(batch_size, 2500, n_latent, n_tfs):.1f} GB")
            print(f"     2. Further reduce batch size:")
            print(f"        → cfg.batch_size = {batch_size // 2} would use ~{estimate_batch_memory_gb(batch_size // 2, n_genes, n_latent, n_tfs):.1f} GB")
            print(f"     3. Use gene selection focused on GRN:")
            print("        → TFs + TF targets + highly variable genes")

            print("\n  📊 Memory Scaling:")
            print(f"     - Polynomial Decoder: batch_size × z_dim² × n_genes")
            print(f"       Current: {batch_size} × {n_latent}² × {n_genes:,} = {batch_size * n_latent**2 * n_genes:,} elements")
            print(f"     - MMD Kernel (largest): (2×batch_size)² = {(2*batch_size)**2:,} elements")
            print(f"       ⚠️  MMD scales quadratically with batch size!")

        elif batch_size < ideal_batch_size:
            print("\n✅ Automatic Memory Optimization Applied:")
            print(f"  - Batch size: {ideal_batch_size} → {batch_size} (GPU memory aware)")
            print(f"  - Estimated memory: {estimated_mem:.1f} GB")
            if torch.cuda.is_available():
                gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"  - GPU headroom: {gpu_mem_gb - estimated_mem:.1f} GB remaining")

        print("=" * 70 + "\n")

    return cfg


# ----------------- core training -----------------
def train_gcrl_vae(adata, cfg: VAEConfig, *, eigengenes_key: str = "X_comm_eig"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Display GPU information
    print("=" * 70)
    print("gCRL-VAE Training Configuration")
    print("=" * 70)
    if device == "cuda":
        print(f"✓ GPU ENABLED: {torch.cuda.get_device_name(0)}")
        print(
            f"  - GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )
        print(f"  - CUDA Version: {torch.version.cuda}")
    else:
        print("⚠ WARNING: Training on CPU (no GPU detected)")
        print("  Training will be significantly slower without GPU acceleration")
    print(f"Device: {device}")
    print("=" * 70)

    set_seed(cfg.seed)
    os.makedirs(cfg.outdir, exist_ok=True)

    # Expressions (normalized, log1p) from adata.X
    ### BEGIN gCRL PREPROCESSING (controls-only per-CT z-scoring) ###
    # - adata.X assumed normalized+log1p with final gene set.
    # - We z-score per cell type using TRAINING CONTROLS ONLY; stats are frozen and applied to all cells.
    # - Only cells with adata.obs['set']=="training" are kept for training inside this function.
    import numpy as _np
    from scipy import sparse as _sps

    def _to_dense(X):
        return X.toarray() if _sps.issparse(X) else _np.asarray(X)

    def _fit_global_stats_from_controls(_adata_ctrl, _genes):
        Xc = _to_dense(_adata_ctrl[:, _genes].X)
        mu_g = Xc.mean(axis=0)
        sd_g = Xc.std(axis=0, ddof=1)
        return mu_g, sd_g

    def _fit_zscaler_per_ct(_adata_ctrl, _genes, _ct_key, epsilon=0.1, clip_value=6.0):
        mu_ct, sd_ct = {}, {}
        cts = _np.unique(_adata_ctrl.obs[_ct_key].astype(str).values)
        for ct in cts:
            rows = _adata_ctrl.obs[_ct_key].astype(str).values == ct
            Xc = _to_dense(_adata_ctrl[rows, _genes].X)
            mu = Xc.mean(axis=0)
            sd = Xc.std(axis=0, ddof=1)
            sd = _np.sqrt(sd**2 + epsilon).astype(_np.float32)
            mu_ct[ct] = mu.astype(_np.float32)
            sd_ct[ct] = sd
        g_mu, g_sd = _fit_global_stats_from_controls(_adata_ctrl, _genes)
        g_sd = _np.sqrt(g_sd**2 + epsilon).astype(_np.float32)
        return {
            "mode": "per_ct",
            "genes": _np.array(_genes).astype(str).tolist(),
            "mu_ct": {k: v.tolist() for k, v in mu_ct.items()},
            "sd_ct": {k: v.tolist() for k, v in sd_ct.items()},
            "global_mu": g_mu.astype(_np.float32).tolist(),
            "global_sd": g_sd.tolist(),
            "epsilon": float(epsilon),
            "clip": float(clip_value),
            "ct_list": sorted(cts.tolist()),
        }

    def _apply_z_to_adata(_adata_any, _zscaler, _ct_key, out_layer=None):
        _genes = _np.array(_zscaler["genes"], dtype=str)
        if not _np.array_equal(_adata_any.var_names.astype(str).values, _genes):
            raise RuntimeError(
                "Gene order mismatch. Ensure var_names match training gene order."
            )
        X = _to_dense(_adata_any[:, _genes].X).astype(_np.float32)
        clipv = float(_zscaler["clip"])
        mu_ct = {
            k: _np.asarray(v, dtype=_np.float32) for k, v in _zscaler["mu_ct"].items()
        }
        sd_ct = {
            k: _np.asarray(v, dtype=_np.float32) for k, v in _zscaler["sd_ct"].items()
        }
        g_mu = _np.asarray(_zscaler["global_mu"], dtype=_np.float32)
        g_sd = _np.asarray(_zscaler["global_sd"], dtype=_np.float32)
        ct_vals = _adata_any.obs[_ct_key].astype(str).values
        Z = _np.empty_like(X, dtype=_np.float32)
        for i, ct in enumerate(ct_vals):
            mu = mu_ct.get(ct, g_mu)
            sd = sd_ct.get(ct, g_sd)
            Zi = (X[i] - mu) / sd
            Z[i] = _np.clip(Zi, -clipv, clipv)
        if out_layer is None:
            _adata_any.X = Z
        else:
            _adata_any.layers[out_layer] = Z
        return Z

    # Keys and checks
    _ct_key = (
        "cell_type"
        if "cell_type" in adata.obs.columns
        else ("celltype" if "celltype" in adata.obs.columns else "cell_type")
    )
    if _ct_key not in adata.obs.columns:
        raise KeyError("adata.obs must contain a cell-type column (e.g., 'cell_type').")
    if "set" not in adata.obs.columns:
        raise KeyError(
            "adata.obs must contain a 'set' column with values 'training' or 'test'."
        )

    # intervention key (intervention)
    _cond_key = None
    for _k in ["intervention", "perturbation", "intervention", "perturb", "treatment"]:
        if _k in adata.obs.columns:
            _cond_key = _k
            break
    if _cond_key is None:
        raise KeyError(
            "Could not find a intervention/perturbation column in adata.obs."
        )

    # Identify training controls
    _ctrl_values = {"control", "unperturbed", "na", "NA", "None", "none", ""}
    _is_train = adata.obs["set"].astype(str).values == "training"
    _ctrl_mask = adata.obs[_cond_key].astype(str).isin(_ctrl_values).values
    _fit_mask = _is_train & _ctrl_mask

    # Persist original matrix
    if adata.layers is None:
        adata.layers = {}
    adata.layers.setdefault("X_log1p", _to_dense(adata.X).astype(_np.float32))

    # Fit scaler on training controls only (use config parameters)
    _genes = adata.var_names.astype(str).values
    _zscaler = _fit_zscaler_per_ct(
        adata[_fit_mask].copy(),
        _genes,
        _ct_key,
        epsilon=cfg.preproc_epsilon,
        clip_value=cfg.preproc_clip_value,
    )

    # Apply to all cells (frozen stats)
    _apply_z_to_adata(adata, _zscaler, _ct_key, out_layer=None)

    # Store metadata
    adata.uns["gcrl_preproc"] = {
        "version": 1,
        "ct_key": _ct_key,
        "cond_key": _cond_key,
        "control_values": sorted(list(_ctrl_values)),
        "set_key": "set",
        "genes": _zscaler["genes"],
        "zscaler": _zscaler,
        "notes": "adata.X overwritten with per-CT controls-only z-scores; original saved in layer 'X_log1p'.",
    }

    # Keep only training rows for training
    adata_train = adata[adata.obs["set"].astype(str).values == "training"].copy()
    ### END gCRL PREPROCESSING (controls-only per-CT z-scoring) ###

    # Separate controls and interventions
    # Controls: used for VAE reconstruction + eigengene alignment
    # Interventions: used as ground truth for MMD loss
    _is_ctrl_train = adata_train.obs[_cond_key].astype(str).isin(_ctrl_values).values
    adata_ctrl = adata_train[_is_ctrl_train].copy()
    adata_interv = adata_train[~_is_ctrl_train].copy()

    X_all = to_dense(adata_ctrl.X).astype(
        np.float32
    )  # Use only controls for main training
    var_names = np.array(adata.var_names.values, dtype=object)
    name2idx = {g: i for i, g in enumerate(var_names)}

    # --- Derive cell types for controls ---
    if "cell_type" not in adata_ctrl.obs.columns:
        raise KeyError("adata.obs must contain 'cell_type'")
    ct_codes_ctrl, ct_uniques = pd.factorize(
        adata_ctrl.obs["cell_type"].astype(str).values
    )

    # Store all cell types for model conditioning
    ct_codes_train, _ = pd.factorize(
        adata_train.obs["cell_type"].astype(str).values, sort=True
    )
    ct_uniques_all = pd.unique(adata_train.obs["cell_type"].astype(str))

    # --- Derive intervened TFs from obs['intervention'] and validate against var['kind']=="TF" ---
    if "intervention" not in adata.obs.columns:
        raise KeyError(
            "adata.obs must contain 'intervention' with the intervention TF names"
        )
    if "kind" not in adata.var.columns:
        raise KeyError("adata.var must contain 'kind' with values 'TF' or 'TG'")

    def _parse_condition_to_tfset(s):
        if s is None:
            return []
        s = str(s)
        if s.strip().lower() in ("", "control", "unperturbed", "na"):
            return []
        parts = re.split(r"[+;,]", s)
        return [p.strip() for p in parts if p.strip()]

    intervened_tf_names = set()
    for v in adata.obs["intervention"].values:
        intervened_tf_names.update(_parse_condition_to_tfset(v))
    intervened_tf_names = sorted(
        {
            t
            for t in intervened_tf_names
            if t and t.lower() not in ("control", "unperturbed")
        }
    )
    # Validate
    bad = []
    intervened_tf_idx = []
    for g in intervened_tf_names:
        j = name2idx.get(g, None)
        if j is None or str(adata.var.loc[g, "kind"]) != "TF":
            bad.append(g)
        else:
            intervened_tf_idx.append(j)
    if bad:
        raise ValueError(f"Intervention on unknown or non-TF genes: {bad}")
    intervened_tf_idx = np.array(intervened_tf_idx, dtype=np.int64)
    intervened_tf_names = np.array(intervened_tf_names, dtype=object)

    # Build col_map for intervention TF indices
    col_map = {g: k for k, g in enumerate(intervened_tf_names)}

    # Group intervened cells by intervention type for MMD loss
    # Each group contains cells with the same intervention
    interv_groups = {}
    if adata_interv.n_obs > 0:
        X_interv_all = to_dense(adata_interv.X).astype(np.float32)
        ct_codes_interv, _ = pd.factorize(
            adata_interv.obs["cell_type"].astype(str).values
        )

        for interv_name in adata_interv.obs[_cond_key].unique():
            mask = adata_interv.obs[_cond_key].values == interv_name
            interv_groups[str(interv_name)] = {
                "X": X_interv_all[mask],
                "ct": ct_codes_interv[mask],
                "n_cells": int(mask.sum()),
            }

    # --- Community set from *intervened* TFs only ---
    if "community" not in adata.var.columns:
        raise KeyError("adata.var must contain 'community' for p+1 mapping")
    var_comm = pd.Categorical(adata.var["community"])
    intervened_comm_ids = np.unique(var_comm.codes[intervened_tf_idx])
    intervened_comm_ids = intervened_comm_ids[intervened_comm_ids >= 0]
    comm_order = np.sort(intervened_comm_ids).tolist()
    p = len(comm_order)
    comm2k = {int(c): k for k, c in enumerate(comm_order)}

    # Use control expression for VAE training
    X_ctrl = X_all
    # All TFs present (for global eigengene and community grouping)
    tf_mask = adata.var["kind"].astype(str).values == "TF"
    tf_idx_all = np.where(tf_mask)[0]
    tf_comm_ids_all = var_comm.codes[tf_idx_all].astype(int)

    # CRITICAL: Extract TF-only expression for encoder input
    # Naming convention:
    # - X_ctrl_tf: Control cells, TF expression ONLY (n_cells, n_TFs)
    # - X_ctrl: Control cells, ALL genes (n_cells, n_genes = n_TFs + n_TGs)
    # - "_all" suffix: Emphasizes "all genes" (not just TFs)
    X_ctrl_tf = X_ctrl[:, tf_idx_all]  # (n_cells, n_TFs)
    print(
        f"[gCRL-VAE] Using TF-only input: {X_ctrl_tf.shape[1]} TFs (from {X_ctrl.shape[1]} total genes)"
    )

    # Build DataLoader with single-CT batches of controls
    sampler = SingleCTBatchSampler(
        ct_codes_ctrl, batch_size=cfg.batch_size, drop_last=False, shuffle=True
    )
    # Tensor dataset for controls (TF-only for input, all genes for reconstruction target)
    X_ctrl_tf_t = torch.from_numpy(X_ctrl_tf)  # TF-only input (encoder)
    X_ctrl_t = torch.from_numpy(X_ctrl)  # All genes (decoder target)
    ct_ctrl_t = torch.from_numpy(ct_codes_ctrl)

    class _CtrlDS(torch.utils.data.Dataset):
        def __len__(self):
            return X_ctrl_tf_t.shape[0]

        def __getitem__(self, i):
            return {
                "x_tf": X_ctrl_tf_t[i],
                "x_all": X_ctrl_t[i],
                "ct": ct_ctrl_t[i],
                "idx": i,
            }

    loader = DataLoader(
        _CtrlDS(),
        batch_sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=(device == "cuda"),
    )

    # Optional: Preload per-cell eigengenes for controls
    A_ctrl = None
    eig_meta_comm = None
    eig_meta_global = None
    if eigengenes_key in adata_ctrl.obsm:
        A_all = adata_ctrl.obsm[eigengenes_key]
        A_all = A_all.toarray() if hasattr(A_all, "toarray") else np.asarray(A_all)
        A_ctrl = torch.from_numpy(A_all.astype(np.float32)).to(device)
        if isinstance(adata_ctrl.uns, dict):
            eig_meta_comm = adata_ctrl.uns.get(f"{eigengenes_key}_comm_ids", None)
            eig_meta_global = adata_ctrl.uns.get(f"{eigengenes_key}_global_index", None)

    # --- Hard mapping: ONLY for intervened TFs (columns of C) ---
    tf_to_latent = None
    if cfg.intervention_mapping == "hard":
        tf_to_latent_np = np.zeros(len(intervened_tf_names), dtype=np.int64)
        for k, g in enumerate(intervened_tf_names):
            j = name2idx[g]  # validated above
            c = var_comm.codes[j]
            if c < 0 or (int(c) not in comm2k):
                raise ValueError(
                    f"Intervened TF {g} has unknown/non-intervened community id {c}"
                )
            tf_to_latent_np[k] = comm2k[int(c)]
        tf_to_latent = torch.from_numpy(tf_to_latent_np).long().to(device)

    # --- Build/Load model (using your project's modules) ---
    # Expect your project to define these; we keep names to match your repo.
    from gcrl.models.gcrl_vae import GCRLVAE, GCRLVAEConfig

    # Determine cell-type conditioning dimension
    ct_dim = len(ct_uniques_all) if len(ct_uniques_all) > 1 else 0

    cfg_m = GCRLVAEConfig(
        input_dim=X_ctrl_tf.shape[1],  # Number of TFs (input)
        z_dim=max(p + 1, 2),
        c_dim=len(intervened_tf_names),
        output_dim=X_ctrl.shape[1],  # Number of all genes (output)
        ct_dim=ct_dim,  # Enable cell-type conditioning
        intervention_mapping=cfg.intervention_mapping,
        hidden_enc=(128,),  # Simplified encoder: single hidden layer of 128
        use_polynomial_decoder=True,  # Polynomial decoder (linear + quadratic)
    )
    model = GCRLVAE(cfg_m).to(device)

    # Set cell-type index if conditioning is enabled
    if ct_dim > 0:
        model.set_ct_index(list(ct_uniques_all))

    # ---------------- NEW: wire for evaluation ----------------
    # These two lines make the evaluator work out-of-the-box:
    model.set_tf_index(list(map(str, intervened_tf_names)))
    if tf_to_latent is not None:  # hard routing
        # Accepts list/ndarray/LongTensor; we pass a CPU list for portability
        model.set_tf_to_latent(tf_to_latent.detach().cpu().tolist())

    # (optional) save a tiny manifest so you can verify wiring later
    with open(os.path.join(cfg.outdir, "eval_wiring.json"), "w") as f:
        json.dump(
            {
                "tf_names": list(map(str, intervened_tf_names)),
                "tf_to_latent": (
                    tf_to_latent.detach().cpu().tolist()
                    if tf_to_latent is not None
                    else None
                ),
                "z_dim": int(cfg_m.z_dim),
                "c_dim": int(cfg_m.c_dim),
                "routing": cfg.intervention_mapping,
            },
            f,
            indent=2,
        )
    # ----------------------------------------------------------

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # --- Loss scheduling (from discrepancy-VAE) ---
    # Beta (KL weight): 0 for first 10 epochs, then linear ramp
    beta_schedule = np.concatenate(
        [np.zeros(10), np.linspace(0, cfg.beta_kld_max, max(1, cfg.epochs - 10))]
    )
    if len(beta_schedule) < cfg.epochs:
        beta_schedule = np.pad(
            beta_schedule,
            (0, cfg.epochs - len(beta_schedule)),
            constant_values=cfg.beta_kld_max,
        )

    # Alpha (MMD weight): 0 for first 5 epochs, then linear ramp
    alpha_schedule = np.concatenate(
        [np.zeros(5), np.linspace(0, cfg.alpha_mmd_max, max(1, cfg.epochs - 5))]
    )
    if len(alpha_schedule) < cfg.epochs:
        alpha_schedule = np.pad(
            alpha_schedule,
            (0, cfg.epochs - len(alpha_schedule)),
            constant_values=cfg.alpha_mmd_max,
        )

    # Temperature schedule: linear increase for soft routing
    temp_schedule = np.linspace(cfg.temp_start, cfg.temp_max, cfg.epochs)

    # Initialize MMD loss criterion
    mmd_criterion = MMD_loss(
        kernel_mul=cfg.mmd_kernel_mul,
        kernel_num=cfg.mmd_kernel_num,
        fix_sigma=cfg.mmd_fix_sigma,
    ).to(device)

    # ============================================================================
    # TRAINING LOOP - Data naming conventions:
    # ============================================================================
    # TF-only data (encoder input):
    #   - x_tf, X_ctrl_tf, X_ctrl_sample_tf: (n_samples, n_TFs)
    # All-gene data (decoder output/target):
    #   - x_all, X_ctrl, X_real_all: (n_samples, n_genes) where n_genes = n_TFs + n_TGs
    #   - "_all" suffix: Emphasizes "all genes" not just TFs
    # ============================================================================
    history = []

    # OOM detection tracking
    oom_count = 0
    max_oom_retries = 5
    oom_batch_sizes = []  # Track batch sizes that caused OOM

    # Progress bar for epochs
    epoch_pbar = tqdm(range(1, cfg.epochs + 1), desc="Training gCRL-VAE", unit="epoch")

    for epoch in epoch_pbar:
        epoch_logs = []

        # Get scheduled parameters for this epoch
        beta_kl = beta_schedule[epoch - 1]
        alpha_mmd = alpha_schedule[epoch - 1]
        temp = temp_schedule[epoch - 1]

        # Progress bar for batches within epoch
        batch_pbar = tqdm(
            loader, desc=f"Epoch {epoch}/{cfg.epochs}", leave=False, unit="batch"
        )

        for batch_idx, batch in enumerate(batch_pbar):

            try:
                # batch data for training (controls only)
                x_tf = batch["x_tf"].to(device)  # TF-only input
                x_all = batch["x_all"].to(device)  # All genes (target)
                ct_batch = batch["ct"].to(device)

                # Create cell-type one-hot vector if conditioning is enabled
                ct_vec = None
                if ct_dim > 0:
                    ct_vec = F.one_hot(ct_batch.long(), num_classes=ct_dim).float()

                # === 1. Standard VAE forward pass on CONTROLS ===
                mu, var = model.encode(x_tf, ct=ct_vec)  # Encode TF-only input
                logvar = torch.log(var)  # Convert to log-variance for KL formula
                z = model.reparameterize(mu, var)
                recon = model.decode(z, ct=ct_vec)  # Decode to all genes

                # Reconstruction loss (controls only, all genes)
                loss_rec = F.mse_loss(recon, x_all, reduction="mean")

                # KL divergence loss (scheduled)
                loss_kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

                # Start with base VAE loss
                loss = cfg.alpha_rec * loss_rec + beta_kl * loss_kld

                # === 2. Eigengene alignment via partial MCC ===
                loss_mcc = torch.tensor(0.0, device=device)
                if cfg.lambda_mcc > 0:
                    idx_rows = batch["idx"].cpu().numpy()
                    # Build E from precomputed eigengenes if available; else compute on-the-fly
                    if A_ctrl is not None:
                        if (
                            eig_meta_comm is not None
                            and isinstance(eig_meta_comm, (list, tuple))
                            and len(eig_meta_comm) >= p
                        ):
                            comm2col = {int(c): i for i, c in enumerate(eig_meta_comm)}
                            col_idx = [
                                comm2col[int(c)] for c in comm_order if int(c) in comm2col
                            ]
                            if eig_meta_global is not None:
                                col_idx.append(int(eig_meta_global))
                            else:
                                col_idx.append(A_ctrl.shape[1] - 1)
                        else:
                            col_idx = list(range(p)) + [A_ctrl.shape[1] - 1]
                        E = A_ctrl[idx_rows][:, col_idx]
                    else:
                        # Fallback compute from batch TF expression
                        X_batch_tf = torch.from_numpy(X_ctrl[idx_rows][:, tf_idx_all]).to(
                            device
                        )
                        tf_comm_ids_full = torch.from_numpy(tf_comm_ids_all).to(device)
                        E = _build_eigengene_matrix_for_batch(
                            X_batch_tf, tf_comm_ids_full, comm_order
                        )
                    MU = mu[:, : (p + 1)]
                    E = E.to(MU.dtype).to(MU.device)
                    if MU.shape[1] != E.shape[1]:
                        raise RuntimeError(f"MU/E mismatch: {MU.shape} vs {E.shape}")
                    loss_mcc = partial_mcc_loss_torch(E, MU)  # negative mean partial corr
                    loss = loss + cfg.lambda_mcc * loss_mcc

                # === 3. L1 Sparsity on causal matrix G (full, tiny, late onset) ===
                loss_sparse = torch.tensor(0.0, device=device)
                if cfg.lambda_sparse > 0 and epoch >= getattr(cfg, "sparse_start_epoch", 0):
                    loss_sparse = torch.norm(model.G, p=1)
                    loss = loss + cfg.lambda_sparse * loss_sparse

                # === 4. DAG acyclicity penalty (NOTEARS on full G) ===
                loss_dag = torch.tensor(0.0, device=device)
                if getattr(cfg, "lambda_dag", 0.0) > 0 and epoch >= getattr(
                    cfg, "dag_start_epoch", 0
                ):
                    loss_dag = notears_acyclicity(model.G)
                    loss = loss + cfg.lambda_dag * loss_dag

                # === 5. MMD loss: simulate interventions and compare to real ===
                # FIXED: MMD computed on reconstructed expression, not DAG latents
                loss_mmd = torch.tensor(0.0, device=device)
                loss_centroid = torch.tensor(0.0, device=device)
                if alpha_mmd > 0 and len(interv_groups) > 0:
                    # Sample one random intervention type
                    interv_name = np.random.choice(list(interv_groups.keys()))
                    interv_data = interv_groups[interv_name]

                    # Sample real intervened cells
                    n_samples = min(cfg.batch_size // 2, interv_data["n_cells"])
                    if n_samples > 0:
                        sample_idx = np.random.choice(
                            interv_data["n_cells"], n_samples, replace=False
                        )

                        # X_real_all: Real intervened cells, ALL genes (ground truth)
                        # "_all" = contains all genes (TF + TG), not just TFs
                        # This is the ORIGINAL expression we want to match
                        X_real_all = torch.from_numpy(interv_data["X"][sample_idx]).to(
                            device
                        )  # (n_samples, n_genes)
                        ct_real_codes = interv_data["ct"][sample_idx]

                        # Create cell-type vector for real intervened cells
                        ct_real_vec = None
                        if ct_dim > 0:
                            ct_real_vec = F.one_hot(
                                torch.from_numpy(ct_real_codes).long().to(device),
                                num_classes=ct_dim,
                            ).float()

                        # Sample control cells (match cell types if possible)
                        ctrl_idx = np.random.choice(len(X_ctrl), n_samples, replace=True)
                        X_ctrl_sample_tf = torch.from_numpy(X_ctrl_tf[ctrl_idx]).to(
                            device
                        )  # TF-only for encoder

                        # Build intervention vector c
                        c_interv = torch.zeros(
                            n_samples, len(intervened_tf_names), device=device
                        )
                        for tf in _parse_condition_to_tfset(interv_name):
                            if tf in col_map:
                                c_interv[:, col_map[tf]] = 1.0

                        # === Simulate intervention on control cells ===
                        # Step 1: Encode controls (TF-only input)
                        mu_ctrl, var_ctrl = model.encode(X_ctrl_sample_tf, ct=ct_real_vec)
                        z_ctrl = model.reparameterize(mu_ctrl, var_ctrl)

                        # Step 2: Apply intervention via DAG transform
                        csz = torch.matmul(c_interv, model.c_shift).view(-1, 1)
                        if cfg.intervention_mapping == "soft":
                            bc = model._soft_routing(c_interv, temp)
                        else:
                            bc = model._hard_routing(c_interv, tf_to_latent)
                        u_sim = model._dag_transform(z_ctrl, bc, csz, None, None)

                        # Step 3: Decode to expression space (all genes)
                        # x_sim: Model's PREDICTION of what intervention does (n_samples, n_genes)
                        x_sim = model.decode(u_sim, ct=ct_real_vec)

                        # === Real intervened cells: use ORIGINAL expression directly ===
                        # X_real_all: What intervention ACTUALLY did (ground truth)
                        # No need to encode-decode; we already have the true intervened expression!
                        # Adding encoder-decoder would only add noise to the ground truth.

                        # === MMD on expression: simulated vs. real (all genes) ===
                        # Compare: Model prediction (x_sim) vs. Ground truth (X_real_all)
                        # Both are in expression space (same units, same scale)
                        loss_mmd = mmd_criterion(x_sim, X_real_all)
                        loss = loss + alpha_mmd * loss_mmd  # Apply schedule

                        # === 5. Centroid loss (optional, uses same schedule as MMD) ===
                        if cfg.lambda_centroid > 0:
                            # Compare gene-wise means (centroid = mean shift)
                            centroid_sim = x_sim.mean(dim=0)  # Simulated mean (n_genes,)
                            centroid_real = X_real_all.mean(dim=0)  # Real mean (n_genes,)
                            loss_centroid = F.mse_loss(centroid_sim, centroid_real)

                            # Apply schedule AND weight: alpha_mmd(t) × lambda_centroid
                            # - alpha_mmd: 0 → 1.0 (epochs 5+)
                            # - lambda_centroid: 0.1 (constant)
                            # - Effective: 0 → 0.1 (10× weaker than MMD)
                            loss = loss + alpha_mmd * cfg.lambda_centroid * loss_centroid

                # === Optimization step ===
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 0.5
                )  # Updated from 5.0 to 0.5
                opt.step()

                # === Logging all 7 loss components ===
                logs = dict(
                    loss=float(loss.detach().cpu()),
                    loss_rec=float(loss_rec.detach().cpu()),
                    loss_kld=float(loss_kld.detach().cpu()),
                    loss_mcc=float(loss_mcc.detach().cpu()),
                    loss_sparse=float(loss_sparse.detach().cpu()),
                    loss_dag=float(loss_dag.detach().cpu()),
                    loss_mmd=float(loss_mmd.detach().cpu()),
                    loss_centroid=float(loss_centroid.detach().cpu()),
                    beta_kl=float(beta_kl),
                    alpha_mmd=float(alpha_mmd),
                    temp=float(temp),
                )
                epoch_logs.append(logs)

                # Update batch progress bar with current loss
                batch_pbar.set_postfix(
                    {
                        "loss": f"{logs['loss']:.4f}",
                        "rec": f"{logs['loss_rec']:.4f}",
                        "mmd": f"{logs['loss_mmd']:.4f}",
                    }
                )

            except RuntimeError as e:
                # Check if this is an out-of-memory error
                if "out of memory" in str(e).lower() or "cuda out of memory" in str(e).lower():
                    oom_count += 1
                    oom_batch_sizes.append(len(batch["x_tf"]))

                    # Clear GPU cache
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    # Clear gradients
                    opt.zero_grad(set_to_none=True)

                    # Log detailed OOM information
                    print(f"\n{'='*70}")
                    print(f"⚠️  GPU OUT OF MEMORY ERROR")
                    print(f"{'='*70}")
                    print(f"  Epoch: {epoch}/{cfg.epochs}")
                    print(f"  Batch: {batch_idx}")
                    print(f"  Batch size: {len(batch['x_tf'])} cells")
                    print(f"  Dataset: {X_ctrl.shape[1]:,} genes ({X_ctrl_tf.shape[1]} TFs)")
                    if torch.cuda.is_available():
                        print(f"  GPU: {torch.cuda.get_device_name(0)}")
                        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
                        print(f"  Total GPU memory: {total_mem:.1f} GB")
                    print(f"  OOM events so far: {oom_count}/{max_oom_retries}")
                    print(f"{'='*70}")

                    if oom_count >= max_oom_retries:
                        print(f"\n❌ Too many OOM errors ({oom_count}). Stopping training.")
                        print(f"\n💡 Recommendations:")
                        print(f"   1. Reduce batch size: cfg.batch_size = {cfg.batch_size // 2}")
                        print(f"   2. Reduce number of genes to 2000-3000 highly variable genes")
                        print(f"   3. Use a GPU with more memory")
                        if oom_batch_sizes:
                            avg_oom_bs = sum(oom_batch_sizes) / len(oom_batch_sizes)
                            print(f"   4. Average batch size at OOM: {avg_oom_bs:.0f} cells")
                            print(f"      → Try: cfg.batch_size = {int(avg_oom_bs * 0.7)}")
                        print(f"{'='*70}\n")
                        raise
                    else:
                        # Skip this batch and continue
                        print(f"  → Skipping batch and continuing training...")
                        print(f"  → If this happens frequently, reduce cfg.batch_size")
                        print(f"{'='*70}\n")
                        continue
                else:
                    # Re-raise non-OOM errors
                    raise

        # Aggregate
        if epoch_logs:
            avg = {
                k: float(np.mean([d[k] for d in epoch_logs]))
                for k in epoch_logs[0].keys()
            }
        else:
            avg = {}
        avg["epoch"] = epoch
        history.append(avg)

        # Update epoch progress bar with summary
        epoch_pbar.set_postfix(
            {
                "loss": f"{avg['loss']:.4f}",
                "rec": f"{avg['loss_rec']:.4f}",
                "mmd": f"{avg['loss_mmd']:.4f}",
                "mcc": f"{avg['loss_mcc']:.4f}",
            }
        )

    # Save history
    with open(os.path.join(cfg.outdir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    return model, history
