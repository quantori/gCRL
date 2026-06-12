
# src/gcrl/alignment/partial_mcc_perm_experiments.py
# -*- coding: utf-8 -*-
"""
Monte Carlo permutation experiments for partial-MCC alignment significance testing.

This module provides a statistical framework to assess whether the alignment between
community eigengenes (A) and latent embeddings (B) is significant. It uses permutation
testing to build a null distribution by shuffling TF-to-community assignments.

Key steps:
1) Compute eigengenes A from the real (unpermuted) GRN structure
2) Optimize partial-MCC alignment between A and B for multiple random seeds (real scores)
3) Build null distribution by:
   - Permuting TF community labels
   - Recomputing eigengenes under permuted structure
   - Running partial-MCC optimization
4) Compare real vs. null scores to assess statistical significance

Key functions:
--------------
run_partial_mcc_perm_experiments : Main function for permutation-based significance testing

Notes
-----
- AnnData is mutated in-place by `compute_eigengenes` (stores A in `.obsm["X_comm_eig"]`)
- Original community labels are restored after permutations
- Device parameter supports CPU/GPU acceleration for optimization
"""

from __future__ import annotations

from typing import Dict, Tuple, Optional, Sequence
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import anndata as ad
from tqdm.auto import tqdm

from gcrl.grn.eigengenes import compute_eigengenes
from gcrl.alignment.partial_mcc import optimize_partial_mcc


def _assert_row_alignment(A: np.ndarray, B: np.ndarray) -> None:
    """
    Validate that matrices A and B are row-aligned (same number of cells).

    Parameters
    ----------
    A : np.ndarray
        Eigengene matrix (n_cells, n_communities+1)
    B : np.ndarray
        Embedding matrix (n_cells, latent_dim)

    Raises
    ------
    ValueError
        If A and B have different numbers of rows

    Notes
    -----
    Row alignment is critical for partial-MCC optimization, as it assumes
    each row index corresponds to the same cell in both matrices.
    """
    if A.shape[0] != B.shape[0]:
        raise ValueError(
            f"Row mismatch between eigengenes and embeddings: "
            f"A has {A.shape[0]} cells, B has {B.shape[0]} cells. "
            "Ensure adata and embeddings are aligned by cell (row) index."
        )


def _permute_tf_communities_inplace(adata: ad.AnnData, community_col: str, seed: int) -> None:
    """
    Permute TF community assignments in-place within adata.var (TF genes only).

    This function shuffles community labels ONLY among TF genes, preserving:
    - The set of TF genes (identity)
    - The multiset of community labels (frequencies)
    - TG gene labels (unchanged)

    Parameters
    ----------
    adata : ad.AnnData
        AnnData object with var['kind'] and var[community_col]
    community_col : str
        Column name in adata.var containing community assignments
    seed : int
        Random seed for reproducible shuffling

    Raises
    ------
    ValueError
        If community_col is missing from adata.var

    Notes
    -----
    This permutation breaks the TF-to-community structure while maintaining
    marginal distributions, creating a null model for significance testing.
    """
    rng = np.random.default_rng(seed)
    var = adata.var

    # Identify TF genes
    kinds = var["kind"].astype(str).str.upper()
    tf_mask = (kinds == "TF").to_numpy()

    # Validate community column exists
    if community_col not in var.columns:
        raise ValueError(
            f"adata.var must contain column '{community_col}'. "
            f"Available columns: {list(var.columns)}"
        )

    # Extract current TF community labels
    comm_vals = var.loc[tf_mask, community_col].to_numpy()

    # Randomly shuffle community labels among TFs
    perm = rng.permutation(len(comm_vals))

    # Assign shuffled labels back to TF genes in-place
    var.loc[tf_mask, community_col] = comm_vals[perm]


def run_partial_mcc_perm_experiments(
    adata: ad.AnnData,
    embeddings: np.ndarray,
    # eigengene computation params
    community_col: str = "community",
    reference_query: str = 'intervention == "unperturbed"',
    mode: str = "all_cells",
    method: str = "PC",
    cell_type_col: str = "cell_type",
    # experiment sizes
    n_real_seeds: int = 100,
    n_permutations: int = 100,
    n_perm_seeds: int = 10,
    # optimization params
    lr: float = 1e-2,
    steps: int = 500,
    device: Optional[str] = None,    # "cpu" | "cuda" | None (auto)
    # reproducibility
    master_seed: int = 42,
    # eigengene selection
    include_pooled_tf: bool = True,
    # optional plot
    save_density_path: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """
    Run real and permuted partial-MCC experiments.

    Parameters
    ----------
    adata : AnnData
        Contains normalized/log1p X, var["kind"] in {"TF","TG"}, var[community_col] cats.
        Will be mutated by compute_eigengenes; this function restores original communities after permutations.
    embeddings : np.ndarray (n_cells, pB)
        AE latent embeddings B aligned row-wise to `adata.obs_names`.
    community_col, reference_query, mode, method, cell_type_col
        Parameters forwarded to `compute_eigengenes`.
    n_real_seeds : int
        Number of optimizer seeds for the real (unpermuted) A.
    n_permutations : int
        Number of TF-community permutations to build the null.
    n_perm_seeds : int
        Number of optimizer seeds per permutation.
    lr, steps, device
        Optimizer parameters forwarded to `optimize_partial_mcc`.
    master_seed : int
        Controls RNG streams for reproducibility (seeds for permutations and runs).
    include_pooled_tf : bool, default True
        If False, the last column of the eigengene matrix (pooled-TF eigengene) is dropped
        before every partial-MCC optimization, for both real and permuted runs.
    save_density_path : Optional[str]
        If provided, saves a Matplotlib PNG comparing kernel densities of real vs permuted scores.

    Returns
    -------
    Dict[str, np.ndarray] with keys:
        "scores_real" : shape (n_real_seeds,)
        "scores_perm" : shape (n_permutations, n_perm_seeds)
    """
    rng = np.random.default_rng(master_seed)

    # --- Compute A for the real setting ---
    compute_eigengenes(
        adata,
        community_col=community_col,
        reference_query=reference_query,
        mode=mode,
        method=method,
        cell_type_col=cell_type_col,
        seed=master_seed,
    )
    def _maybe_drop_pooled(mat: np.ndarray) -> np.ndarray:
        """Drop the last column (pooled-TF eigengene) when include_pooled_tf is False."""
        if not include_pooled_tf and mat.shape[1] > 1:
            return mat[:, :-1]
        return mat

    A_real = _maybe_drop_pooled(adata.obsm["X_comm_eig"].astype(np.float32))
    B = embeddings.astype(np.float32, copy=False)
    _assert_row_alignment(A_real, B)

    # --- Real seeds ---
    real_seeds = rng.integers(0, 2**31 - 1, size=n_real_seeds, endpoint=False)
    scores_real = np.zeros(n_real_seeds, dtype=np.float32)

    # Run real (unpermuted) experiments with progress bar
    for i, s in enumerate(tqdm(real_seeds, desc="Real runs", unit="run")):
        score, _, _ = optimize_partial_mcc(A_real, B, lr=lr, steps=steps, seed=int(s), device=device)
        scores_real[i] = score

    # --- Prepare for permutations: keep original community labels to restore later ---
    original_comm = adata.var[community_col].copy()

    # --- Permutations ---
    perm_scores = np.zeros((n_permutations, n_perm_seeds), dtype=np.float32)

    # Run permutation experiments with progress bar
    for p in tqdm(range(n_permutations), desc="Permutations", unit="perm"):
        # permute TFs across communities
        _permute_tf_communities_inplace(adata, community_col=community_col, seed=int(rng.integers(0, 2**31 - 1)))
        # recompute eigengenes under permuted communities
        compute_eigengenes(
            adata,
            community_col=community_col,
            reference_query=reference_query,
            mode=mode,
            method=method,
            cell_type_col=cell_type_col,
            seed=int(rng.integers(0, 2**31 - 1)),
        )
        A_perm = _maybe_drop_pooled(adata.obsm["X_comm_eig"].astype(np.float32))
        _assert_row_alignment(A_perm, B)

        perm_run_seeds = rng.integers(0, 2**31 - 1, size=n_perm_seeds, endpoint=False)
        for j, s in enumerate(perm_run_seeds):
            score, _, _ = optimize_partial_mcc(A_perm, B, lr=lr, steps=steps, seed=int(s), device=device)
            perm_scores[p, j] = score

    # --- Restore original communities ---
    adata.var[community_col] = original_comm

    # --- Optional density plot ---
    if save_density_path is not None:
        flat_perm = perm_scores.ravel()
        plt.figure(figsize=(7, 5))
        plt.hist(scores_real, bins=30, density=True, alpha=0.5, label=f"Real (n={len(scores_real)})")
        plt.hist(flat_perm, bins=30, density=True, alpha=0.5, label=f"Permuted (n={len(flat_perm)})")
        plt.xlabel("Partial MCC")
        plt.ylabel("Density")
        plt.title("Partial-MCC: Real vs Permuted")
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_density_path, dpi=200)
        plt.close()

    return {"scores_real": scores_real, "scores_perm": perm_scores}
