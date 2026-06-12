
# src/gcrl/alignment/partial_mcc.py
# -*- coding: utf-8 -*-
"""
Partial-MCC alignment between eigengenes (A) and AE embeddings (B).

Core idea: find X such that A ≈ B @ X by maximizing average **partial** correlation
between matched columns, intervention mapping each pair on the remaining columns.

Adapted from T-MEX: https://arxiv.org/pdf/2505.17708

This module contains **pure functions** with no file I/O side effects.
"""

from __future__ import annotations

from typing import Tuple, Optional
import numpy as np
import torch
import matplotlib.pyplot as plt


def set_seed(seed: int) -> None:
    """Set numpy/torch RNG seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def standardize_columns_torch(M: torch.Tensor) -> torch.Tensor:
    """Column-wise standardization (zero mean, unit variance)."""
    mean = M.mean(dim=0, keepdim=True)
    std = M.std(dim=0, keepdim=True)
    return (M - mean) / (std + 1e-8)


def residuals(Y: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """
    Orthogonal projection residuals of Y on the column space of X.
    If X has zero columns, returns Y unchanged.
    """
    if X.ndim != 2 or X.shape[1] == 0:
        return Y
    X_pinv = torch.linalg.pinv(X)
    beta = X_pinv @ Y
    Y_hat = X @ beta
    return Y - Y_hat


def partial_mcc_loss_torch(A: torch.Tensor, BX: torch.Tensor) -> torch.Tensor:
    """
    Compute the **negative** mean partial correlation between corresponding columns of A and BX.
    We minimize this loss to maximize average partial correlation (MCC).

    Args
    ----
    A : (n, p) tensor
    BX: (n, p) tensor

    Returns
    -------
    torch.Tensor (scalar): negative mean partial correlation
    """
    A = standardize_columns_torch(A)
    BX = standardize_columns_torch(BX)
    n, p = A.shape
    partial_corrs = []

    for j in range(p):
        idx = [i for i in range(p) if i != j]
        A_j = A[:, j]
        A_rest = A[:, idx] if idx else A[:, :0]
        BX_j = BX[:, j]

        r_A = residuals(A_j.view(-1, 1), A_rest)
        r_BX = residuals(BX_j.view(-1, 1), A_rest)

        r_A_std = standardize_columns_torch(r_A)
        r_BX_std = standardize_columns_torch(r_BX)

        num = torch.sum(r_A_std * r_BX_std)
        den = torch.norm(r_A_std) * torch.norm(r_BX_std) + 1e-8
        cos_sim = num / den
        partial_corrs.append(cos_sim)

    partial_corrs = torch.stack(partial_corrs)
    return -torch.mean(partial_corrs)


def optimize_partial_mcc(
    A_np: np.ndarray,
    B_np: np.ndarray,
    lr: float = 1e-2,
    steps: int = 500,
    seed: int = 42,
    device: Optional[str] = None,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Optimize X to maximize partial MCC such that A ≈ B @ X.

    Args
    ----
    A_np : (n, pA) NumPy array
        Target columns (e.g., eigengenes).
    B_np : (n, pB) NumPy array
        Source columns (e.g., AE embeddings).
    lr : float
        Adam learning rate.
    steps : int
        Number of optimization steps.
    seed : int
        Random seed for reproducibility.
    device : {"cpu","cuda",None}
        If None, auto-select "cuda" if available else "cpu".

    Returns
    -------
    score : float
        Final (positive) mean partial MCC after optimization.
    X_opt : (pB, pA) NumPy array
        Learned linear mapping.
    BX_opt : (n, pA) NumPy array
        Product B @ X_opt at optimum.
    """
    set_seed(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    A = torch.tensor(A_np, dtype=torch.float32, device=dev)
    B = torch.tensor(B_np, dtype=torch.float32, device=dev)

    if A.shape[0] != B.shape[0]:
        raise ValueError(f"Row mismatch: A has {A.shape[0]} rows, B has {B.shape[0]} rows.")

    X = torch.randn(B.shape[1], A.shape[1], device=dev, requires_grad=True)

    opt = torch.optim.Adam([X], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = partial_mcc_loss_torch(A, B @ X)
        loss.backward()
        opt.step()

    with torch.no_grad():
        BX = B @ X
        score = -partial_mcc_loss_torch(A, BX).item()
        X_np = X.detach().cpu().numpy()
        BX_np = BX.detach().cpu().numpy()
    return score, X_np, BX_np


def correlation_matrix(A: np.ndarray, BX: np.ndarray) -> np.ndarray:
    """
    Compute the column-wise Pearson correlation matrix between A and BX.

    Returns
    -------
    corr : (pA, pA) np.ndarray
        corr[i, j] = corr(A[:, i], BX[:, j])
    """
    A0 = (A - A.mean(axis=0, keepdims=True)) / (A.std(axis=0, keepdims=True) + 1e-8)
    B0 = (BX - BX.mean(axis=0, keepdims=True)) / (BX.std(axis=0, keepdims=True) + 1e-8)
    pA = A.shape[1]
    corr = np.zeros((pA, pA), dtype=np.float64)
    for i in range(pA):
        for j in range(pA):
            corr[i, j] = np.dot(A0[:, i], B0[:, j]) / (len(A0) - 1)
    return corr


def plot_correlation_matrix_with_matches(
    A: np.ndarray,
    BX: np.ndarray,
    title: str = "Correlation Matrix between A and B*X",
    show: bool = True,
    save_path: Optional[str] = None,
) -> np.ndarray:
    """
    Plot correlation matrix between A and BX and highlight the best matching columns.

    Parameters
    ----------
    A : np.ndarray
        Ground-truth matrix (n x p).
    BX : np.ndarray
        Reconstructed matrix (n x p).
    title : str
        Title of the plot.
    show : bool
        Whether to display the plot with matplotlib.
    save_path : Optional[str]
        If provided, save the figure to this path.

    Returns
    -------
    np.ndarray
        Correlation matrix of shape (p, p).
    """
    C = correlation_matrix(A, BX)
    p = C.shape[0]

    plt.figure(figsize=(8, 6))
    im = plt.imshow(C, cmap="coolwarm", vmin=-1, vmax=1, origin="upper")
    plt.colorbar(im, label="Correlation Coefficient")
    plt.title(title)
    plt.xlabel("Columns of B @ X")
    plt.ylabel("Columns of A")
    plt.xticks(range(p))
    plt.yticks(range(p))

    # Mark best match for each column of A
    for i in range(p):
        best_j = int(np.argmax(C[i, :]))
        plt.plot(best_j, i, "ko", markersize=8, markerfacecolor="none", markeredgewidth=2)

    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
    if show:
        plt.show()
    else:
        plt.close()

    return C
