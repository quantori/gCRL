# src/gcrl/grn/eigengenes.py
# -*- coding: utf-8 -*-
"""
Eigengene computation utilities for gCRL.

This module computes per-community transcription factor (TF) eigengenes as PC1 scores
from an AnnData object. PC1s are fit on a reference subset of cells (typically unperturbed),
and all cells are projected onto these reference-fitted directions.

Key functions:
--------------
compute_eigengenes : Computes eigengenes and stores them in-place in adata.obsm["X_comm_eig"]
"""

from __future__ import annotations
from typing import Tuple, Dict, List, Any
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import anndata as ad


def _to_dense(X) -> np.ndarray:
    """
    Return a dense float32 NumPy array for scanpy/anndata matrices.

    Handles both sparse matrices (with .toarray() method) and regular arrays.
    Always returns float32 to ensure consistent precision across operations.
    """
    if hasattr(X, "toarray"):
        X = X.toarray()
    return np.asarray(X, dtype=np.float32)


def _fit_pc1_on_reference(
    X_all: np.ndarray,
    gene_idx: np.ndarray,
    ref_idx: np.ndarray,
    random_state: int,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Fit PC1 on reference rows (ref_idx) using only columns gene_idx; project all rows.

    Parameters
    ----------
    X_all : np.ndarray
        Full expression matrix (n_cells, n_genes)
    gene_idx : np.ndarray
        Indices of genes to use for PCA fitting
    ref_idx : np.ndarray
        Indices of reference cells (e.g., unperturbed) to fit PCA on
    random_state : int
        Random seed for PCA reproducibility

    Returns
    -------
    eig : np.ndarray
        PC1 scores for ALL cells (n_cells,)
    stats : dict
        Dictionary containing 'mean', 'std', 'components', and 'explained_variance_ratio'
        used for standardization and projection. 'explained_variance_ratio' is the fraction
        of variance in the reference-standardized data explained by PC1.

    Notes
    -----
    - Standardization (z-score) is computed using ONLY reference cells
    - All cells are then projected using the reference-derived mean, std, and PC1 loadings
    - Zero-variance genes are set to std=1.0 to avoid division by zero
    """
    # Extract reference subset for the selected genes
    X_ref = X_all[np.asarray(ref_idx)[:, None], gene_idx]

    # Compute standardization parameters on reference cells only
    mean = X_ref.mean(axis=0)
    std = X_ref.std(axis=0, ddof=0)
    std[std == 0] = 1.0  # Guard against zero-variance genes (prevents NaN)

    # Standardize reference data
    X_ref_std = (X_ref - mean) / std

    # Fit PCA on standardized reference data
    pca = PCA(n_components=1, random_state=random_state)
    pca.fit(X_ref_std)
    comp = pca.components_[0]  # PC1 loadings (n_genes,)

    # Project ALL cells onto the reference-fitted PC1
    X_all_std = (X_all[:, gene_idx] - mean) / std
    eig = (X_all_std @ comp.reshape(-1, 1)).ravel().astype(np.float32)

    # Store standardization and PCA parameters for potential downstream use
    # explained_variance_ratio: fraction of variance explained by PC1 among the
    # reference-standardized TFs in this community (from sklearn PCA on reference cells).
    stats = {
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "components": comp.astype(np.float32),
        "explained_variance_ratio": float(pca.explained_variance_ratio_[0]),
    }
    return eig, stats


def _compute_single_gene_eigengene(
    X_view: np.ndarray,
    gene_idx: int,
    ref_idx: np.ndarray,
    gene_name: str,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], List[str]]:
    """
    Compute eigengene for a single-gene community.

    For communities with only one TF, the eigengene is simply the z-scored
    expression of that gene (standardized using reference cells).

    Parameters
    ----------
    X_view : np.ndarray
        Expression matrix for current view (n_cells, n_genes)
    gene_idx : int
        Index of the single gene
    ref_idx : np.ndarray
        Indices of reference cells for standardization
    gene_name : str
        Name of the gene

    Returns
    -------
    eig : np.ndarray
        Z-scored expression for all cells (n_cells,)
    stats : dict
        Standardization statistics (mean, std, components=[1.0])
    genes : list
        List containing the single gene name
    """
    X_ref = X_view[ref_idx, gene_idx]
    mean = X_ref.mean()
    std = X_ref.std()

    # Prevent division by zero for constant genes
    if std == 0:
        std = 1.0

    eig = ((X_view[:, gene_idx] - mean) / std).astype(np.float32)

    stats = {
        "mean": np.array([mean], dtype=np.float32),
        "std": np.array([std], dtype=np.float32),
        "components": np.array([1.0], dtype=np.float32),  # No rotation for single gene
        "explained_variance_ratio": 1.0,  # Single gene: eigengene captures 100% of variance
    }

    return eig, stats, [gene_name]


def _compute_average_eigengene(
    X_all: np.ndarray,
    gene_idx: np.ndarray,
    ref_idx: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Compute the average z-scored expression across a set of genes.

    Standardization parameters (mean, std) are estimated on reference cells only
    and applied to all cells, matching the convention used in _fit_pc1_on_reference.

    Parameters
    ----------
    X_all : np.ndarray
        Full expression matrix (n_cells, n_genes)
    gene_idx : np.ndarray
        Indices of genes to include in the average
    ref_idx : np.ndarray
        Indices of reference cells for standardization

    Returns
    -------
    eig : np.ndarray
        Mean z-score across genes for all cells (n_cells,)
    stats : dict
        Standardization statistics (mean, std, components=uniform weights)
    """
    X_ref = X_all[np.asarray(ref_idx)[:, None], gene_idx]

    mean = X_ref.mean(axis=0)
    std = X_ref.std(axis=0, ddof=0)
    std[std == 0] = 1.0

    X_all_std = (X_all[:, gene_idx] - mean) / std
    eig = X_all_std.mean(axis=1).astype(np.float32)

    n = gene_idx.size
    stats = {
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "components": np.full(n, 1.0 / n, dtype=np.float32),  # uniform weights
        "explained_variance_ratio": float("nan"),  # not applicable for averaging
    }
    return eig, stats


def _compute_eigs_given_view_matrix(
    X_view: np.ndarray,
    var: pd.DataFrame,
    comm_col: str,
    tf_mask: np.ndarray,
    ref_idx: np.ndarray,
    seed: int,
    use_average: bool = False,
) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
    """
    Compute per-community eigengenes + pooled TF eigengene for a matrix view.

    This is the core computation function that handles:
    - Empty communities (0 TFs): returns zero eigengene
    - Single-gene communities (1 TF): returns z-scored expression
    - Multi-gene communities (2+ TFs): returns PC1 scores, or mean z-score if use_average=True

    Parameters
    ----------
    X_view : np.ndarray
        Expression matrix for current view (n_cells, n_genes)
    var : pd.DataFrame
        Gene metadata (must contain comm_col and gene names in index)
    comm_col : str
        Column name for community assignments
    tf_mask : np.ndarray
        Boolean mask indicating which genes are TFs (n_genes,)
    ref_idx : np.ndarray
        Indices of reference cells for PCA fitting / standardization
    seed : int
        Random seed for PCA reproducibility (ignored when use_average=True)
    use_average : bool, default=False
        If True, compute eigengenes as the mean of z-scored TF expression
        rather than PC1 scores.

    Returns
    -------
    eig_mat : np.ndarray
        Eigengene matrix (n_cells, n_communities + 1)
        Last column is the pooled TF eigengene
    col_names : list
        Column names for eigengene matrix
    meta : dict
        Metadata including per-community genes and statistics
    """
    comm_series = var[comm_col].astype("category")
    comm_ids = list(comm_series.cat.categories)
    n_cells = X_view.shape[0]

    # Initialize output matrix: n_communities + 1 (for pooled TF eigengene)
    eig_mat = np.zeros((n_cells, len(comm_ids) + 1), dtype=np.float32)

    per_comm_genes: Dict[str, List[str]] = {}
    per_comm_stats: Dict[str, Dict[str, np.ndarray]] = {}
    gene_names = np.asarray(var.index, dtype=str)

    # Compute eigengene for each community
    for j, comm in enumerate(comm_ids):
        # Find TF genes in this community
        comm_mask = (comm_series.to_numpy() == comm) & tf_mask
        gene_idx = np.where(comm_mask)[0]

        # Case 1: Empty community (no TFs) → zero eigengene
        if gene_idx.size == 0:
            eig_mat[:, j] = 0.0
            per_comm_genes[str(comm)] = []
            per_comm_stats[str(comm)] = {
                "mean": np.array([], dtype=np.float32),
                "std": np.array([], dtype=np.float32),
                "components": np.array([], dtype=np.float32),
                "explained_variance_ratio": 0.0,  # No TFs: undefined, stored as 0
            }
            continue

        # Case 2: Single TF → z-scored expression (no PCA needed)
        elif gene_idx.size == 1:
            g = gene_idx[0]
            eig, stats, genes = _compute_single_gene_eigengene(
                X_view, g, ref_idx, gene_names[g]
            )
            eig_mat[:, j] = eig
            per_comm_genes[str(comm)] = genes
            per_comm_stats[str(comm)] = stats
            continue

        # Case 3: Multiple TFs → compute PC1 or mean z-score
        if use_average:
            eig, stats = _compute_average_eigengene(X_view, gene_idx, ref_idx)
        else:
            eig, stats = _fit_pc1_on_reference(X_view, gene_idx, ref_idx, seed)
        eig_mat[:, j] = eig
        per_comm_genes[str(comm)] = list(gene_names[gene_idx])
        per_comm_stats[str(comm)] = stats

    # --- Compute pooled TF eigengene (using ALL TFs regardless of community) ---
    all_tf_idx = np.where(tf_mask)[0]

    # Case 1: No TFs at all → zero eigengene
    if all_tf_idx.size == 0:
        pooled = np.zeros(n_cells, dtype=np.float32)
        pooled_stats = {
            "mean": np.array([], dtype=np.float32),
            "std": np.array([], dtype=np.float32),
            "components": np.array([], dtype=np.float32),
            "explained_variance_ratio": 0.0,
        }

    # Case 2: Single TF → z-scored expression
    elif all_tf_idx.size == 1:
        g = all_tf_idx[0]
        pooled, pooled_stats, _ = _compute_single_gene_eigengene(
            X_view, g, ref_idx, gene_names[g]
        )

    # Case 3: Multiple TFs → compute PC1 or mean z-score across all TFs
    else:
        if use_average:
            pooled, pooled_stats = _compute_average_eigengene(X_view, all_tf_idx, ref_idx)
        else:
            pooled, pooled_stats = _fit_pc1_on_reference(X_view, all_tf_idx, ref_idx, seed)

    # Place pooled eigengene in the last column
    eig_mat[:, -1] = pooled

    # Create column names: one per community + one for pooled TF
    col_names = [f"eig_comm_{c}" for c in comm_ids] + ["eig_all_TF"]

    # Package metadata for downstream use
    meta = {
        "communities": comm_ids,
        "columns": col_names,
        "per_comm_genes": per_comm_genes,
        "per_comm_stats": per_comm_stats,
        "pooled_tf_stats": pooled_stats,
    }
    return eig_mat, col_names, meta


def _ref_indices_from_query(obs: pd.DataFrame, reference_query: str) -> np.ndarray:
    """
    Extract reference cell indices from a query string applied to obs DataFrame.

    Parameters
    ----------
    obs : pd.DataFrame
        Cell metadata (typically adata.obs or a subset)
    reference_query : str
        Pandas query string to select reference cells (e.g., 'intervention == "unperturbed"')

    Returns
    -------
    ref_idx : np.ndarray
        Integer indices of reference cells. If query is empty or returns no cells,
        returns all indices (i.e., use entire dataset as reference).

    Notes
    -----
    This ensures we always have valid reference cells for standardization and PCA fitting.
    If the query fails or is empty, we fall back to using all cells.
    """
    if reference_query and reference_query.strip():
        try:
            # Apply query and get integer positions
            idx = obs.index.get_indexer(obs.query(reference_query).index)
            idx = idx[idx >= 0]  # Remove invalid indices (-1)

            # Validate that we found at least some reference cells
            if idx.size == 0:
                return np.arange(obs.shape[0])
            return idx
        except Exception:
            # If query fails, use all cells as fallback
            return np.arange(obs.shape[0])
    return np.arange(obs.shape[0])


def compute_eigengenes(
    adata: ad.AnnData,
    community_col: str = "community",
    reference_query: str = 'intervention == "unperturbed"',
    mode: str = "all_cells",
    method: str = "PC",
    cell_type_col: str = "cell_type",
    seed: int = 42,
) -> None:
    """
    Compute TF-community eigengenes from an AnnData and **mutate** the object in-place.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData object containing:
        - X: normalized, log1p-transformed expression matrix
        - var['kind']: gene type labels ('TF' or 'TG')
        - var[community_col]: community assignments for genes
        - obs[cell_type_col]: cell type labels (required for "by_cell_type" and "by_cell_type_reference" modes)
    community_col : str, default="community"
        Column name in adata.var for community assignments
    reference_query : str, default='intervention == "unperturbed"'
        Pandas query string to select reference cells for standardization / PCA fitting.
        Used only when mode is "by_reference" or "by_cell_type_reference".
    mode : str, default="all_cells"
        Determines which cells are used to compute the standardization (z-score) and fit:
        - "all_cells": use all cells globally
        - "by_reference": use reference cells (selected by reference_query) globally,
          project all cells
        - "by_cell_type": use all cells of each cell type independently
        - "by_cell_type_reference": use reference cells within each cell type,
          project all cells of that type
    method : str, default="PC"
        How to summarize the standardized TF expression into a single eigengene per community:
        - "PC": first principal component (PC1) of the z-scored TF matrix
        - "average": mean of the z-scored TF expression
    cell_type_col : str, default="cell_type"
        Column name in adata.obs for cell type labels.
        Required when mode is "by_cell_type" or "by_cell_type_reference".
    seed : int, default=42
        Random seed for PCA reproducibility (ignored when method="average").

    Side effects
    ------------
    - Writes eigengene matrix to `adata.obsm["X_comm_eig"]`
      Shape: (n_cells, n_communities + 1), where last column is pooled TF eigengene
    - Writes community IDs to `adata.uns["X_comm_eig_comm_ids"]`
    - Writes global eigengene index to `adata.uns["X_comm_eig_global_index"]`
    - Writes metadata to `adata.uns["comm_eig_meta"]`

    Raises
    ------
    ValueError
        If required columns are missing or contain invalid values, if mode/method are
        unrecognised, or if no reference cells are found for "by_reference" modes.
    """
    _VALID_MODES = {"all_cells", "by_reference", "by_cell_type", "by_cell_type_reference"}
    _VALID_METHODS = {"PC", "average"}

    if mode not in _VALID_MODES:
        raise ValueError(f"Invalid mode '{mode}'. Must be one of: {sorted(_VALID_MODES)}.")
    if method not in _VALID_METHODS:
        raise ValueError(f"Invalid method '{method}'. Must be one of: {sorted(_VALID_METHODS)}.")

    use_average = method == "average"

    np.random.seed(seed)

    X_all = _to_dense(adata.X)
    var = adata.var.copy()

    # --- Validate 'kind' column ---
    if "kind" not in var.columns:
        raise ValueError(
            "adata.var must contain column 'kind' with values 'TF' or 'TG'. "
            "This column is required to identify transcription factors."
        )

    kinds_clean = var["kind"].astype(str).str.strip().str.upper()
    unique_kinds = set(kinds_clean.unique())

    if not unique_kinds.issubset({"TF", "TG"}):
        raise ValueError(
            f"adata.var['kind'] must only contain 'TF' or 'TG' values. "
            f"Found invalid values: {unique_kinds - {'TF', 'TG'}}"
        )

    tf_mask = (kinds_clean == "TF").to_numpy()

    if not np.any(tf_mask):
        raise ValueError(
            "No TF genes found in adata.var['kind']. "
            "At least one gene must be labeled as 'TF' to compute eigengenes."
        )

    # --- Validate community column ---
    if community_col not in var.columns:
        raise ValueError(
            f"adata.var must contain column '{community_col}' for community labels. "
            f"Available columns: {list(var.columns)}"
        )

    var[community_col] = var[community_col].astype("category")
    comm_ids = list(var[community_col].cat.categories)

    if len(comm_ids) == 0:
        raise ValueError(
            f"No communities found in adata.var['{community_col}']. "
            "At least one community must be defined."
        )

    col_names = [f"eig_comm_{c}" for c in comm_ids] + ["eig_all_TF"]

    # --- Helper: store results ---
    def _store(eig_mat: np.ndarray, meta: dict) -> None:
        adata.obsm["X_comm_eig"] = eig_mat
        adata.uns["X_comm_eig_comm_ids"] = comm_ids
        adata.uns["X_comm_eig_global_index"] = int(eig_mat.shape[1] - 1)
        adata.uns["comm_eig_meta"] = meta

    # --- Helper: run per-cell-type loop ---
    def _by_cell_type_loop(get_ref_idx):
        if cell_type_col not in adata.obs.columns:
            raise ValueError(
                f"adata.obs must contain column '{cell_type_col}' for mode='{mode}'. "
                f"Available columns: {list(adata.obs.columns)}"
            )
        cell_types = list(pd.Categorical(adata.obs[cell_type_col]).categories)
        if len(cell_types) == 0:
            raise ValueError(
                f"No cell types found in adata.obs['{cell_type_col}']."
            )
        eig_all = np.zeros((adata.n_obs, len(col_names)), dtype=np.float32)
        meta_per_ct: Dict[str, Any] = {}
        for ct in cell_types:
            mask_ct = adata.obs[cell_type_col].values == ct
            if not np.any(mask_ct):
                continue
            X_ct = X_all[mask_ct, :]
            ref_idx_local = get_ref_idx(ct, X_ct, adata.obs.loc[mask_ct])
            eig_ct, cols_ct, meta_ct = _compute_eigs_given_view_matrix(
                X_ct, var, community_col, tf_mask, ref_idx_local, seed, use_average
            )
            if cols_ct != col_names:
                raise ValueError(
                    f"Column mismatch for cell type '{ct}': "
                    f"expected {col_names}, got {cols_ct}."
                )
            eig_all[mask_ct, :] = eig_ct
            meta_per_ct[str(ct)] = meta_ct
        return eig_all, cell_types, meta_per_ct

    # ------------------------------------------------------------------ #
    # Mode: all_cells
    # ------------------------------------------------------------------ #
    if mode == "all_cells":
        ref_idx = np.arange(adata.n_obs)
        eig_mat, cols, inner_meta = _compute_eigs_given_view_matrix(
            X_all, var, community_col, tf_mask, ref_idx, seed, use_average
        )
        _store(eig_mat, {"mode": mode, "method": method, **inner_meta})
        return

    # ------------------------------------------------------------------ #
    # Mode: by_reference
    # ------------------------------------------------------------------ #
    if mode == "by_reference":
        ref_idx = _ref_indices_from_query(adata.obs, reference_query)
        if len(ref_idx) == 0:
            raise ValueError(
                f"Reference query '{reference_query}' returned no cells. "
                "Cannot fit without reference data."
            )
        eig_mat, cols, inner_meta = _compute_eigs_given_view_matrix(
            X_all, var, community_col, tf_mask, ref_idx, seed, use_average
        )
        _store(eig_mat, {"mode": mode, "method": method, **inner_meta})
        return

    # ------------------------------------------------------------------ #
    # Mode: by_cell_type
    # ------------------------------------------------------------------ #
    if mode == "by_cell_type":
        def _get_ref_all(ct, X_ct, obs_ct):
            return np.arange(X_ct.shape[0])

        eig_all, cell_types, meta_per_ct = _by_cell_type_loop(_get_ref_all)
        _store(eig_all, {
            "mode": mode, "method": method,
            "cell_type_col": cell_type_col,
            "cell_types": [str(ct) for ct in cell_types],
            "per_cell_type": meta_per_ct,
            "community_col": community_col,
            "seed": seed, "columns": col_names,
        })
        return

    # ------------------------------------------------------------------ #
    # Mode: by_cell_type_reference
    # ------------------------------------------------------------------ #
    if mode == "by_cell_type_reference":
        def _get_ref_from_query(ct, X_ct, obs_ct):
            idx = _ref_indices_from_query(obs_ct, reference_query)
            if len(idx) == 0:
                raise ValueError(
                    f"Reference query '{reference_query}' returned no cells "
                    f"for cell type '{ct}'."
                )
            return idx

        eig_all, cell_types, meta_per_ct = _by_cell_type_loop(_get_ref_from_query)
        _store(eig_all, {
            "mode": mode, "method": method,
            "cell_type_col": cell_type_col,
            "cell_types": [str(ct) for ct in cell_types],
            "per_cell_type": meta_per_ct,
            "community_col": community_col,
            "reference_query": reference_query,
            "seed": seed, "columns": col_names,
        })
        return
