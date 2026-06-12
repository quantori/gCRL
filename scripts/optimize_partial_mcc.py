#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Optimize X in A = B X where:
- A is the (#communities + 1) eigengenes matrix from an AnnData object (e.g., adata.obsm["X_comm_eig"])
- B is the embedding matrix from the polynomial AE (cells x latent_dim)

We maximize the (average) partial correlation between columns of A and columns of B X
via a differentiable "negative partial MCC" loss.

This version supports:
- Reading A from .h5ad (obsm key, default 'X_comm_eig')
- Reading B from .npy (assumes same cell order) or .csv (aligned by cell IDs index)
- Optional cell_type filtering
- Saving .npy outputs for A, B, X, and B@X
- Optional CSV outputs with --save_csv
- Optional correlation heatmap with --plot
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import scanpy as sc
import pandas as pd


# ----------------------- Argparse ----------------------- #

def build_parser():
    p = argparse.ArgumentParser(description="Optimize X in A = B X by maximizing partial MCC.")
    # Inputs
    p.add_argument("--h5ad", required=True, help="Path to AnnData .h5ad file containing eigengenes in .obsm.")
    p.add_argument("--obsm_key", default="X_comm_eig", help="adata.obsm key for eigengene matrix A (default: X_comm_eig).")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--embeddings_npy", help="Path to AE embeddings .npy (assumes same cell order as adata).")
    group.add_argument("--embeddings_csv", help="Path to AE embeddings .csv (rows indexed by cell IDs to align).")
    p.add_argument("--cell_type", default=None, help="If provided, subset adata to this cell_type before reading A.")
    # Optimization
    p.add_argument("--lr", type=float, default=1e-2, help="Learning rate for Adam (default: 1e-2).")
    p.add_argument("--steps", type=int, default=500, help="Optimization steps (default: 500).")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    # Output and plotting
    p.add_argument("--outdir", default="results_partial_mcc", help="Output directory.")
    p.add_argument("--plot", action="store_true", help="If set, save correlation heatmap PNG.")
    p.add_argument("--save_csv", action="store_true", help="If set, also save A, B, X, B@X as CSV files.")
    return p


# ----------------------- Utilities ----------------------- #

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def standardize_columns_torch(M: torch.Tensor) -> torch.Tensor:
    """Center and scale the columns of a torch tensor."""
    mean = M.mean(dim=0, keepdim=True)
    std = M.std(dim=0, keepdim=True)
    return (M - mean) / (std + 1e-8)


def residuals(Y: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """Project Y orthogonally to the space orthogonal to X: Y - X (X^+ Y)."""
    if X.ndim == 1 or X.shape[1] == 0:
        return Y
    X_pinv = torch.linalg.pinv(X)
    beta = X_pinv @ Y
    Y_hat = X @ beta
    return Y - Y_hat


def partial_mcc_loss_torch(A: torch.Tensor, BX: torch.Tensor) -> torch.Tensor:
    """
    Differentiable loss: negative partial MCC (maximize by minimizing).
    A and BX should be (n, p) with the same p (number of signals).
    """
    A = standardize_columns_torch(A)
    BX = standardize_columns_torch(BX)

    n, p = A.shape
    partial_corrs = []

    for j in range(p):
        idx = [i for i in range(p) if i != j]
        A_j = A[:, j]
        A_rest = A[:, idx]
        BX_j = BX[:, j]

        r_A = residuals(A_j.view(-1, 1), A_rest)
        r_BX = residuals(BX_j.view(-1, 1), A_rest)

        r_A_std = standardize_columns_torch(r_A)
        r_BX_std = standardize_columns_torch(r_BX)

        cos_sim = torch.sum(r_A_std * r_BX_std) / (torch.norm(r_A_std) * torch.norm(r_BX_std) + 1e-8)
        partial_corrs.append(cos_sim)

    partial_corrs = torch.stack(partial_corrs)
    return -torch.mean(partial_corrs)  # negative for minimization


def optimize_with_partial_mcc(A_np: np.ndarray, B_np: np.ndarray, lr=1e-2, steps=500, seed=42, outdir=".", save_csv=False, a_cols=None, b_cols=None, cell_ids=None):
    """Optimize X (pB x pA) to maximize partial MCC between A and B @ X."""
    set_seed(seed)
    A = torch.tensor(A_np, dtype=torch.float32)
    B = torch.tensor(B_np, dtype=torch.float32)

    nA, pA = A.shape
    nB, pB = B.shape
    if nA != nB:
        raise ValueError(f"A and B row counts must match (cells). Got A:{nA} vs B:{nB}")

    # Rectangular mapping pB -> pA
    X = torch.randn(pB, pA, requires_grad=True)

    optimizer = torch.optim.Adam([X], lr=lr)
    for _ in range(steps):
        optimizer.zero_grad()
        BX = B @ X
        loss = partial_mcc_loss_torch(A, BX)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        BX_final = (B @ X).detach()
        final_score = -partial_mcc_loss_torch(A, BX_final).item()

    # Save NPY
    np.save(os.path.join(outdir, "Partial_MCC_A.npy"), A.detach().cpu().numpy())
    np.save(os.path.join(outdir, "Partial_MCC_B.npy"), B.detach().cpu().numpy())
    np.save(os.path.join(outdir, "Partial_MCC_X.npy"), X.detach().cpu().numpy())
    np.save(os.path.join(outdir, "Partial_MCC_BX.npy"), BX_final.detach().cpu().numpy())

    # Optional CSVs
    if save_csv:
        # Column names fallbacks
        if a_cols is None:
            a_cols = [f"A_{i+1}" for i in range(pA)]
        if b_cols is None:
            b_cols = [f"Latent_{i+1}" for i in range(pB)]
        aligned_cols = [f"Aligned_{i+1}" for i in range(pA)]
        x_rows = b_cols  # rows correspond to latent dims
        x_cols = a_cols  # cols correspond to eigengene dims

        # Cell IDs fallback
        if cell_ids is None:
            cell_ids = [f"cell_{i+1}" for i in range(nA)]

        pd.DataFrame(A.detach().cpu().numpy(), index=cell_ids, columns=a_cols).to_csv(os.path.join(outdir, "Partial_MCC_A.csv"))
        pd.DataFrame(B.detach().cpu().numpy(), index=cell_ids, columns=b_cols).to_csv(os.path.join(outdir, "Partial_MCC_B.csv"))
        pd.DataFrame(BX_final.detach().cpu().numpy(), index=cell_ids, columns=aligned_cols).to_csv(os.path.join(outdir, "Partial_MCC_BX.csv"))
        pd.DataFrame(X.detach().cpu().numpy(), index=x_rows, columns=x_cols).to_csv(os.path.join(outdir, "Partial_MCC_X.csv"))

    return final_score, X.detach(), BX_final.detach()


def plot_correlation_matrix_with_matches(A: np.ndarray, BX: np.ndarray, out_png: str, title="Correlation: A vs B@X"):
    """Save correlation matrix and highlight the best matching columns."""
    A_np = A if isinstance(A, np.ndarray) else A.detach().cpu().numpy()
    BX_np = BX if isinstance(BX, np.ndarray) else BX.detach().cpu().numpy()

    n, p = A_np.shape
    corr_matrix = np.zeros((p, p))
    for i in range(p):
        a_col = A_np[:, i]
        a_col = (a_col - a_col.mean()) / (a_col.std() + 1e-12)
        for j in range(p):
            bx_col = BX_np[:, j]
            bx_col = (bx_col - bx_col.mean()) / (bx_col.std() + 1e-12)
            corr_matrix[i, j] = np.corrcoef(a_col, bx_col)[0, 1]

    plt.figure(figsize=(8, 6))
    im = plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(im, label="Correlation Coefficient")
    plt.title(title)
    plt.xlabel("Columns of B @ X")
    plt.ylabel("Columns of A")
    plt.xticks(range(p))
    plt.yticks(range(p))
    # Mark best match per A col
    for i in range(p):
        best_j = np.argmax(corr_matrix[i, :])
        plt.plot(best_j, i, 'ko', markersize=8, markerfacecolor='none', markeredgewidth=2)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    return corr_matrix


# ----------------------- Main ----------------------- #

def main():
    parser = build_parser()
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Load A from AnnData
    adata = sc.read_h5ad(args.h5ad)
    if args.cell_type is not None:
        mask = (adata.obs["cell_type"].astype(str) == args.cell_type)
        if mask.sum() == 0:
            raise ValueError(f"No cells found for cell_type == '{args.cell_type}'")
        adata = adata[mask].copy()

    if args.obsm_key not in adata.obsm:
        raise KeyError(f"'{args.obsm_key}' not found in adata.obsm")
    A = np.asarray(adata.obsm[args.obsm_key], dtype=np.float32)  # (cells, p)
    cell_ids = adata.obs_names.astype(str).tolist()

    # Column names for A if available
    a_cols = None
    if "comm_eig_meta" in adata.uns and "columns" in adata.uns["comm_eig_meta"]:
        a_cols = list(adata.uns["comm_eig_meta"]["columns"])
        if len(a_cols) != A.shape[1]:
            a_cols = None  # fallback later

    # Load B (embeddings)
    if args.embeddings_npy:
        B = np.load(args.embeddings_npy)
        if B.shape[0] != A.shape[0]:
            raise ValueError(f"Row mismatch: A has {A.shape[0]} cells but embeddings_npy has {B.shape[0]}")
        b_cols = [f"Latent_{i+1}" for i in range(B.shape[1])]
    else:
        df = pd.read_csv(args.embeddings_csv, index_col=0)
        # Align rows by cell IDs
        if not adata.obs_names.isin(df.index).all():
            missing = adata.obs_names[~adata.obs_names.isin(df.index)]
            raise ValueError(f"Embeddings CSV is missing {len(missing)} cells from AnnData (e.g., {missing[:5].tolist()} ...)")
        df = df.loc[adata.obs_names]  # reorder to match AnnData
        B = df.values.astype(np.float32)
        b_cols = list(df.columns.astype(str))

    # Optimize
    score, X, BX = optimize_with_partial_mcc(
        A, B, lr=args.lr, steps=args.steps, seed=args.seed, outdir=args.outdir,
        save_csv=args.save_csv, a_cols=a_cols, b_cols=b_cols, cell_ids=cell_ids
    )
    print(f"✅ Final partial MCC score: {score:.4f}")
    with open(os.path.join(args.outdir, "partial_mcc_score.txt"), "w") as f:
        f.write(f"{score}\n")

    # Optional plot
    if args.plot:
        out_png = os.path.join(args.outdir, "corr_A_vs_BX.png")
        plot_correlation_matrix_with_matches(A, BX, out_png=out_png)
        print(f"Saved correlation heatmap → {out_png}")


if __name__ == "__main__":
    main()
