# src/gcrl/training/train_gcrl_ae.py
# -*- coding: utf-8 -*-
"""
Training utilities for gCRL-AE (Gene Community Representation Learning Autoencoder).

This module trains an autoencoder with:
- MLP encoder: maps input genes → latent factors
- Polynomial decoder: quadratic readout Φ(z) = [1, z, z²] → gene reconstruction

Key features:
- Optional cell type filtering for subset training
- Reference-based z-scoring using unperturbed cells
- TF-only or all-gene encoder inputs
- Full or partial gene reconstruction
- Artifact persistence for downstream analysis

Assumes adata.X is normalized and log1p-transformed.

Key functions:
--------------
train_gcrl_ae : Main training function that returns a TrainResult dataclass
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, Sequence

import os
import time
import json
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F

import anndata as ad

from gcrl.models.gcrl_ae import GCRLAE


@dataclass
class TrainResult:
    model: GCRLAE
    embeddings: np.ndarray  # (n_cells, latent_dim) for the selected subset
    decoder_weights: np.ndarray  # (output_dim, poly_feats)
    history: Dict[str, list]  # training loss per epoch
    scalers: Dict[str, Any]  # z-score (mean/std) or min-max (min/max)
    config: Dict[str, Any]  # training config for reproducibility


def _to_dense(X) -> np.ndarray:
    """
    Convert sparse or dense matrix to dense float32 NumPy array.

    Handles both sparse matrices (with .toarray() method) and regular arrays.
    Always returns float32 to ensure consistent precision across operations.
    """
    if hasattr(X, "toarray"):
        X = X.toarray()
    return np.asarray(X, dtype=np.float32)


def _zscore_using_reference(
    X: np.ndarray, ref_idx: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Z-score all columns using stats computed on reference rows only.

    Parameters
    ----------
    X : np.ndarray
        Full expression matrix (n_cells, n_genes)
    ref_idx : np.ndarray
        Indices of reference cells for computing mean/std

    Returns
    -------
    Xz : np.ndarray
        Z-scored matrix for ALL cells (n_cells, n_genes)
    mean : np.ndarray
        Mean per gene computed from reference cells (n_genes,)
    std : np.ndarray
        Std per gene computed from reference cells (n_genes,)

    Notes
    -----
    - Zero-variance genes (std=0) are set to std=1.0 to prevent NaN
    - All cells are standardized using reference-derived parameters
    """
    # Extract reference subset
    X_ref = X[ref_idx, :]

    # Compute statistics on reference cells only
    mean = X_ref.mean(axis=0)
    std = X_ref.std(axis=0, ddof=0)

    # Guard against zero-variance genes (prevents division by zero → NaN)
    std[std == 0] = 1.0

    # Apply standardization to ALL cells using reference stats
    Xz = (X - mean) / std

    return Xz.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def _minmax_scale_0_1(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Min-max scale all columns to [0, 1] range.

    Parameters
    ----------
    X : np.ndarray
        Expression matrix (n_cells, n_genes)

    Returns
    -------
    Xs : np.ndarray
        Scaled matrix in [0, 1] range (n_cells, n_genes)
    xmin : np.ndarray
        Minimum per gene (1, n_genes)
    xmax : np.ndarray
        Maximum per gene (1, n_genes)

    Notes
    -----
    - Constant genes (xmax == xmin) are handled by setting span=1.0
    - This prevents division by zero and keeps constant genes at 0
    """
    xmin = X.min(axis=0, keepdims=True)
    xmax = X.max(axis=0, keepdims=True)

    # Guard against constant genes (span=0)
    span = np.where((xmax - xmin) == 0.0, 1.0, (xmax - xmin))

    # Scale to [0, 1]
    Xs = (X - xmin) / span

    return Xs.astype(np.float32), xmin.astype(np.float32), xmax.astype(np.float32)


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
    This ensures we always have valid reference cells for standardization.
    If the query fails or is empty, we fall back to using all cells.
    """
    if reference_query and reference_query.strip():
        try:
            # Apply query and get integer positions
            idx = obs.index.get_indexer(obs.query(reference_query).index)
            idx = idx[idx >= 0]  # Remove invalid indices (-1)

            # Fallback to all cells if query returns nothing
            if idx.size == 0:
                return np.arange(obs.shape[0])
            return idx
        except Exception:
            # If query fails, use all cells as fallback
            return np.arange(obs.shape[0])
    return np.arange(obs.shape[0])


def train_gcrl_ae(
    adata: ad.AnnData,
    # columns
    community_col: str = "community",
    kind_col: str = "kind",
    intervention_col: str = "intervention",
    cell_type_col: str = "cell_type",
    cell_type: Optional[object] = None,  # value to match in obs[cell_type_col]
    # encoder inputs / decoder outputs
    input_mode: str = "TF",  # {"TF","ALL"}
    reconstruct_all: bool = True,
    # latent dim
    latent_dim: Optional[int] = None,  # if None -> (#communities + 1)
    # encoder MLP
    hidden_dims: Sequence[int] = (256,),
    activation: nn.Module = nn.ReLU(),
    # training
    batch_size: int = 1024,
    num_epochs: int = 100,
    lr: float = 1e-3,
    lr_step: int = 50,
    lr_gamma: float = 0.5,
    weight_decay: float = 0.0,
    device: Optional[str] = None,  # "cpu" | "cuda" | None (auto)
    seed: int = 42,
    # validation
    val_frac: float = 0.1,  # fraction of cells held out for validation (0 to disable)
    # preprocessing
    reference_query: str = 'intervention == "unperturbed"',
    standardize: str = "zscore_ref",  # {"zscore_ref","minmax_0_1","none"}
    # persistence
    outdir: Optional[str] = None,  # if provided, save artifacts for downstream analyses
) -> TrainResult:
    """
    Train gCRL-AE (Gene Causal Representation Learning Autoencoder) with polynomial decoder.

    The gCRL-AE model consists of:
    - Encoder (MLP): maps input genes → latent factors z
    - Decoder (Polynomial): quadratic readout Φ(z) = [1, z, z²] → gene reconstruction

    The model can be trained on all cells or a specific cell type subset.
    Reference-based scaling ensures unperturbed cells define the standardization parameters.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData object with:
        - X: normalized, log1p-transformed expression matrix
        - var[kind_col]: gene type labels ('TF' or 'TG')
        - var[community_col]: community assignments (used if latent_dim=None)
        - obs[cell_type_col]: cell type labels (used if cell_type is provided)
    community_col : str, default="community"
        Column in adata.var for community assignments (for inferring latent_dim)
    kind_col : str, default="kind"
        Column in adata.var for gene type labels ('TF' or 'TG')
    intervention_col : str, default="intervention"
        Column in adata.obs for intervention labels (used in default reference_query)
    cell_type_col : str, default="cell_type"
        Column in adata.obs for cell type labels
    cell_type : Optional[object], default=None
        Value to match in adata.obs[cell_type_col] for subsetting rows.
        If None, uses all cells. If provided, training is restricted to this cell type.
    input_mode : str, default="TF"
        Encoder input selection:
        - "TF": encoder sees only TF genes
        - "ALL": encoder sees all genes
    reconstruct_all : bool, default=True
        If True, decoder reconstructs all genes.
        If False, decoder reconstructs only encoder inputs.
    latent_dim : Optional[int], default=None
        Latent space dimensionality. If None, inferred as (#communities + 1).
    hidden_dims : Sequence[int], default=(256,)
        Hidden layer sizes for encoder MLP
    activation : nn.Module, default=nn.ReLU()
        Activation function for encoder hidden layers
    batch_size : int, default=1024
        Training batch size
    num_epochs : int, default=100
        Number of training epochs
    lr : float, default=1e-3
        Initial learning rate for Adam optimizer
    lr_step : int, default=50
        Step size for learning rate scheduler (StepLR)
    lr_gamma : float, default=0.5
        Multiplicative factor for learning rate decay
    weight_decay : float, default=0.0
        L2 regularization weight for Adam optimizer
    device : Optional[str], default=None
        Device for training ("cpu", "cuda", or None for auto-detection)
    seed : int, default=42
        Random seed for reproducibility (NumPy and PyTorch)
    val_frac : float, default=0.1
        Fraction of cells held out as a random validation set for monitoring
        reconstruction loss during training. Sampled uniformly at random after
        standardization (z-score stats are computed on the full reference set
        before the split). Set to 0.0 to disable validation.
        Note: validation cells are excluded from the training DataLoader but
        embeddings are still extracted for ALL cells after training.
    reference_query : str, default='intervention == "unperturbed"'
        Pandas query string to select reference cells for standardization
    standardize : str, default="zscore_ref"
        Standardization strategy:
        - "zscore_ref": z-score using reference cells only
        - "zscore": z-score using all cells
        - "minmax_0_1": min-max scaling to [0,1]
        - "none": no standardization
    outdir : Optional[str], default=None
        Directory to save artifacts (model, embeddings, weights, scalers, config).
        If None, nothing is saved to disk.

    Returns
    -------
    TrainResult
        Dataclass containing:
        - model: trained GCRLAE instance
        - embeddings: latent representations (n_cells, latent_dim)
        - decoder_weights: polynomial decoder weights (output_dim, poly_feats)
        - history: training loss per epoch; also contains 'val_loss' if val_frac > 0
        - scalers: standardization parameters
        - config: hyperparameters and metadata

    Raises
    ------
    ValueError
        If cell_type is specified but no cells match
        If kind_col is missing or contains invalid values
        If latent_dim=None but community_col is missing
        If input_mode is not 'TF' or 'ALL'
        If standardize is not in {'zscore_ref', 'minmax_0_1', 'none'}

    Notes
    -----
    - The cell_type argument does NOT create a conditional autoencoder; it only
      filters which rows are used for training and scaling.
    - Reference cells define standardization parameters (mean/std for z-score).
      If reference_query returns no cells, all cells are used as reference.
    - The decoder uses polynomial features: Φ(z) = [1, z_1, ..., z_D, z_1², z_1*z_2, ..., z_D²]
      where D = latent_dim. Total features = 1 + D + D*(D+1)/2.
    - Zero-variance genes are handled by setting std=1.0 to prevent NaN.

    Examples
    --------
    Train on all cells with TF-to-all-gene reconstruction:

    >>> res = train_gcrl_ae(
    ...     adata,
    ...     input_mode="TF",
    ...     reconstruct_all=True,
    ...     latent_dim=None,  # infer from communities
    ...     num_epochs=100,
    ...     outdir="results/ae"
    ... )
    >>> Z = res.embeddings  # latent representations

    Train on specific cell type:

    >>> res = train_gcrl_ae(
    ...     adata,
    ...     cell_type="B cell",
    ...     cell_type_col="cell_type",
    ...     input_mode="TF",
    ...     reference_query='intervention == "unperturbed"'
    ... )
    """
    # --- Step 1: Set random seeds for reproducibility ---
    torch.manual_seed(seed)
    np.random.seed(seed)

    # --- Step 2: Optional cell type subsetting ---
    # If cell_type is specified, filter to only those cells
    # Note: This does NOT make the model conditional; it just restricts the training data
    if cell_type is not None:
        # Create boolean mask for cells matching the requested cell type
        mask = adata.obs[cell_type_col].values == cell_type

        # Validate that at least some cells match
        if not np.any(mask):
            raise ValueError(
                f"No cells match {cell_type_col} == {cell_type!r}. "
                f"Available cell types: {adata.obs[cell_type_col].unique().tolist()}"
            )

        # Create subset AnnData with filtered rows
        adata_sub = ad.AnnData(
            X=adata.X[mask, :],  # Subset expression matrix rows
            obs=adata.obs.loc[mask].copy(),  # Subset observation metadata
            var=adata.var.copy(),  # Keep all gene metadata (columns unchanged)
            layers=(
                # Subset any layers if they exist and are row-aligned
                {
                    k: (
                        v[mask, :]
                        if hasattr(v, "shape") and v.shape[0] == adata.n_obs
                        else v
                    )
                    for k, v in adata.layers.items()
                }
                if hasattr(adata, "layers")
                else None
            ),
        )
    else:
        # Use all cells (no subsetting)
        adata_sub = adata

    # --- Step 3: Validate gene type annotations ---
    if kind_col not in adata_sub.var.columns:
        raise ValueError(
            f"adata.var must contain column '{kind_col}' with 'TF'/'TG' labels. "
            f"Available columns: {list(adata_sub.var.columns)}"
        )

    # Clean and standardize gene type labels (handle whitespace and case)
    kinds = adata_sub.var[kind_col].astype(str).str.strip().str.upper().values

    # Ensure all genes are labeled as either TF or TG
    if not set(np.unique(kinds)).issubset({"TF", "TG"}):
        invalid = set(np.unique(kinds)) - {"TF", "TG"}
        raise ValueError(
            f"adata.var['{kind_col}'] must only contain 'TF' or 'TG' labels. "
            f"Found invalid values: {invalid}"
        )

    # --- Step 4: Extract and prepare expression data ---
    X = _to_dense(adata_sub.X)  # Convert to dense float32 (assumes normalized/log1p)
    n_cells, n_genes = X.shape
    gene_names = adata_sub.var_names.astype(str).tolist()
    obs_names = adata_sub.obs_names.astype(str).tolist()

    # --- Step 5: Determine latent dimensionality ---
    if latent_dim is None:
        # Infer from community structure: latent_dim = #communities + 1
        if community_col not in adata_sub.var.columns:
            raise ValueError(
                f"latent_dim is None and adata.var lacks '{community_col}' to infer it. "
                "Either provide latent_dim explicitly or ensure adata.var contains community labels."
            )
        n_comm = pd.Categorical(adata_sub.var[community_col]).categories.size
        latent_dim = int(n_comm) + 1

    # --- Step 6: Identify reference cells for standardization ---
    # Reference cells (e.g., unperturbed) define mean/std for z-score normalization
    ref_idx = _ref_indices_from_query(adata_sub.obs, reference_query)

    # Validate that we have reference cells
    if len(ref_idx) == 0:
        raise ValueError(
            f"Reference query '{reference_query}' returned no cells. "
            "Cannot compute standardization parameters without reference data."
        )

    # --- Step 7: Standardize expression matrix ---
    # Note: Standardization is applied to ALL genes, not just encoder inputs
    scalers: Dict[str, Any] = {"strategy": standardize}

    if standardize == "zscore_ref":
        # Z-score using reference cells only (unperturbed as baseline)
        X_proc, mean, std = _zscore_using_reference(X, ref_idx)
        scalers.update({"mean": mean, "std": std, "reference_query": reference_query})

    elif standardize == "zscore":
        # Z-score using all cells
        X_proc, mean, std = _zscore_using_reference(X, np.arange(n_cells))
        scalers.update({"mean": mean, "std": std})

    elif standardize == "minmax_0_1":
        # Min-max scale to [0, 1] range
        X_proc, xmin, xmax = _minmax_scale_0_1(X)
        scalers.update({"min": xmin, "max": xmax})

    elif standardize == "none":
        # No standardization (just ensure float32)
        X_proc = X.astype(np.float32, copy=False)

    else:
        raise ValueError(
            f"Invalid standardization strategy '{standardize}'. "
            "Must be one of {{'zscore_ref', 'zscore', 'minmax_0_1', 'none'}}"
        )

    # --- Step 8: Select encoder inputs ---
    # Encoder can use either TF genes only or all genes
    if input_mode.upper() == "TF":
        # TF-only mode: encoder sees only transcription factors
        tf_idx = np.where(kinds == "TF")[0]

        if tf_idx.size == 0:
            raise ValueError(
                "No TF genes found in adata.var[kind]. "
                "Cannot use input_mode='TF' without TF genes."
            )

        X_in = X_proc[:, tf_idx]  # Select only TF columns
        input_dim = X_in.shape[1]

    elif input_mode.upper() == "ALL":
        # All-gene mode: encoder sees all genes
        tf_idx = None
        X_in = X_proc  # Use full standardized matrix
        input_dim = n_genes

    else:
        raise ValueError(
            f"Invalid input_mode '{input_mode}'. Must be 'TF' or 'ALL'."
        )

    # --- Step 9: Determine decoder output dimensions ---
    # Decoder can reconstruct all genes or just the encoder inputs
    output_dim = n_genes if reconstruct_all else input_dim

    # --- Step 9b: Random train/val split ---
    # Split is performed on cell indices after standardization so that z-score
    # statistics (computed on the full reference set above) are not affected.
    # Embeddings are still extracted for ALL cells after training.
    all_idx = np.arange(n_cells)
    if val_frac > 0.0:
        n_val = max(1, int(n_cells * val_frac))
        rng = np.random.default_rng(seed)
        val_idx = rng.choice(all_idx, size=n_val, replace=False)
        train_idx = np.setdiff1d(all_idx, val_idx)
    else:
        train_idx = all_idx
        val_idx = np.array([], dtype=int)

    # --- Step 10: Set up compute device ---
    if device is None:
        # Auto-detect: use GPU if available, otherwise CPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    # --- Step 11: Create PyTorch DataLoaders ---
    # Inputs: encoder inputs (TFs or all genes)
    # Targets: full standardized matrix (for reconstruction loss)
    tens_in  = torch.tensor(X_in,   dtype=torch.float32)
    tens_tgt = torch.tensor(X_proc, dtype=torch.float32)

    # Training loader: shuffled, train cells only
    ds_train = TensorDataset(tens_in[train_idx], tens_tgt[train_idx])
    loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=False)

    # Validation loader: unshuffled, val cells only (empty if val_frac == 0)
    if len(val_idx) > 0:
        ds_val = TensorDataset(tens_in[val_idx], tens_tgt[val_idx])
        val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False, drop_last=False)
    else:
        val_loader = None

    # --- Step 12: Initialize model, optimizer, and scheduler ---
    # Create gCRL-AE model with MLP encoder and polynomial decoder
    model = GCRLAE(
        input_dim=input_dim,  # Number of input genes (TFs or all)
        latent_dim=latent_dim,  # Latent space dimensionality
        hidden_dims=hidden_dims,  # MLP hidden layer sizes
        activation=activation,  # Activation function for hidden layers
        output_dim=output_dim,  # Number of output genes to reconstruct
    ).to(dev)

    # Adam optimizer with optional L2 regularization
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Learning rate scheduler: multiply lr by gamma every lr_step epochs
    sched = optim.lr_scheduler.StepLR(opt, step_size=lr_step, gamma=lr_gamma)

    # --- Step 13: Training loop ---
    history = {"loss": [], "val_loss": [] if val_loader is not None else None}
    model.train()  # Set model to training mode

    for epoch in range(1, num_epochs + 1):
        # --- Training pass ---
        model.train()
        epoch_loss_sum = 0.0  # Accumulate loss for this epoch
        epoch_samples = 0  # Count samples processed

        for xb, yb in loader:
            xb = xb.to(dev)
            yb = yb.to(dev)

            opt.zero_grad()
            yhat, _ = model(xb)
            loss = F.mse_loss(yhat, yb[:, :output_dim])
            loss.backward()
            opt.step()

            epoch_loss_sum += loss.item() * xb.size(0)
            epoch_samples += xb.size(0)

        sched.step()
        history["loss"].append(epoch_loss_sum / max(1, epoch_samples))

        # --- Validation pass (no gradients) ---
        if val_loader is not None:
            model.eval()
            val_loss_sum = 0.0
            val_samples = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(dev)
                    yb = yb.to(dev)
                    yhat, _ = model(xb)
                    val_loss_sum += F.mse_loss(yhat, yb[:, :output_dim]).item() * xb.size(0)
                    val_samples += xb.size(0)
            history["val_loss"].append(val_loss_sum / max(1, val_samples))

    # --- Step 14: Extract latent embeddings for all cells ---
    model.eval()  # Set model to evaluation mode

    # Determine embedding batch size based on device and sample size
    # Strategy: Use larger batches on GPU, smaller on CPU; scale with dataset size
    if dev.type == "cuda":
        # GPU: larger batches are more efficient (memory permitting)
        # Scale with dataset size: min 1024, max 8192
        EMBEDDING_BATCH_SIZE = min(8192, max(1024, n_cells // 10))
    else:
        # CPU: moderate batches to avoid memory issues
        # Scale with dataset size: min 512, max 4096
        EMBEDDING_BATCH_SIZE = min(4096, max(512, n_cells // 20))

    with torch.no_grad():
        Z = []
        # Process in batches to handle large datasets efficiently
        for i in range(0, X_in.shape[0], EMBEDDING_BATCH_SIZE):
            xb = torch.tensor(
                X_in[i : i + EMBEDDING_BATCH_SIZE],
                dtype=torch.float32,
                device=dev
            )
            z = model.encoder(xb)  # Encode batch to latent space
            Z.append(z.detach().cpu().numpy())  # Move to CPU and convert to NumPy
        Z = np.concatenate(Z, axis=0)  # Shape: (n_cells, latent_dim)

    # --- Step 15: Extract decoder weights ---
    # Weights map polynomial features Φ(z) → gene reconstructions
    # Shape: (output_dim, poly_feats) where poly_feats = 1 + D + D*(D+1)/2
    W = model.decoder.fc.weight.detach().cpu().numpy()

    # --- Step 16: Create configuration snapshot for reproducibility ---
    config = dict(
        # Column names
        community_col=community_col,
        kind_col=kind_col,
        intervention_col=intervention_col,
        cell_type_col=cell_type_col,
        cell_type=None if cell_type is None else str(cell_type),
        # Model architecture
        input_mode=input_mode,
        reconstruct_all=reconstruct_all,
        latent_dim=latent_dim,
        hidden_dims=list(hidden_dims),
        # Training hyperparameters
        batch_size=batch_size,
        num_epochs=num_epochs,
        lr=lr,
        lr_step=lr_step,
        lr_gamma=lr_gamma,
        weight_decay=weight_decay,
        device=device,
        seed=seed,
        # Validation split
        val_frac=val_frac,
        n_train=int(len(train_idx)),
        n_val=int(len(val_idx)),
        # Preprocessing
        reference_query=reference_query,
        standardize=standardize,
        # Data dimensions
        n_cells=int(n_cells),
        n_genes=int(n_genes),
        # Metadata
        obs_names_preview=obs_names[:10],
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )

    # --- Step 17: Optional persistence to disk ---
    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)

        # Save trained model with metadata
        torch.save(
            {
                "state_dict": model.state_dict(),  # Model weights
                "input_dim": input_dim,
                "latent_dim": latent_dim,
                "output_dim": output_dim,
                "hidden_dims": list(hidden_dims),
            },
            os.path.join(outdir, "gcrl_ae_model.pth"),
        )

        # Save latent embeddings (both NumPy and CSV formats)
        np.save(os.path.join(outdir, "embeddings.npy"), Z)
        pd.DataFrame(
            Z,
            index=obs_names,
            columns=[f"Latent_{i+1}" for i in range(Z.shape[1])]
        ).to_csv(os.path.join(outdir, "embeddings.csv"))

        # Save decoder weights (both NumPy and CSV formats)
        np.save(os.path.join(outdir, "decoder_weights.npy"), W)
        pd.DataFrame(
            W,
            index=adata_sub.var_names[:output_dim],
            columns=[f"phi_{i}" for i in range(W.shape[1])],
        ).to_csv(os.path.join(outdir, "decoder_weights.csv"))

        # Save training history (loss per epoch)
        with open(os.path.join(outdir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

        # Save configuration for reproducibility
        with open(os.path.join(outdir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        # Save standardization parameters for downstream inference
        if standardize in ("zscore_ref", "zscore"):
            np.save(os.path.join(outdir, "zscore_mean.npy"), scalers["mean"])
            np.save(os.path.join(outdir, "zscore_std.npy"), scalers["std"])
        elif standardize == "minmax_0_1":
            np.save(os.path.join(outdir, "min.npy"), scalers["min"])
            np.save(os.path.join(outdir, "max.npy"), scalers["max"])

        # Save TF indices if using TF-only input mode (helps with downstream analysis)
        if input_mode.upper() == "TF":
            np.save(os.path.join(outdir, "tf_indices.npy"), np.where(kinds == "TF")[0])

        # Save cell names used in training (important for cell type subsets)
        pd.Series(obs_names).to_csv(
            os.path.join(outdir, "obs_names_used.csv"),
            index=False,
            header=False
        )

    # --- Step 18: Return results ---
    return TrainResult(
        model=model,  # Trained gCRL-AE model
        embeddings=Z,  # Latent representations (n_cells, latent_dim)
        decoder_weights=W,  # Polynomial decoder weights (output_dim, poly_feats)
        history=history,  # Training loss per epoch
        scalers=scalers,  # Standardization parameters
        config=config,  # Full configuration for reproducibility
    )
