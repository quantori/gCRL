
# tests/test_partial_mcc.py
# -*- coding: utf-8 -*-
import numpy as np
import torch

from gcrl.alignment.partial_mcc import (
    partial_mcc_loss_torch,
    optimize_partial_mcc,
    correlation_matrix,
    set_seed,
)

def test_optimize_partial_mcc_recovers_mapping():
    set_seed(123)
    n, pA, pB = 64, 5, 7

    # Create synthetic data: A = B @ X_true + small noise
    rng = np.random.default_rng(0)
    B = rng.normal(size=(n, pB)).astype(np.float32)
    X_true = rng.normal(size=(pB, pA)).astype(np.float32)
    A_clean = B @ X_true
    noise = 0.05 * rng.normal(size=A_clean.shape).astype(np.float32)
    A = A_clean + noise

    # Optimize
    score, X_opt, BX_opt = optimize_partial_mcc(A, B, lr=5e-2, steps=300, seed=123, device="cpu")

    # Basic checks
    assert X_opt.shape == (pB, pA)
    assert BX_opt.shape == (n, pA)
    assert score > 0.5  # partial MCC should be reasonably high for this synthetic setup

    # Correlation matrix peaks along diagonal for matched columns
    C = correlation_matrix(A, BX_opt)
    best_matches = C.argmax(axis=1)
    # Not necessarily perfect ordering, but diagonal should be competitive
    assert (C[np.arange(pA), best_matches] > 0.3).all()

def test_partial_mcc_loss_shapes():
    torch.manual_seed(0)
    A = torch.randn(32, 4)
    B = torch.randn(32, 6)
    X = torch.randn(6, 4, requires_grad=True)
    loss = partial_mcc_loss_torch(A, B @ X)
    assert loss.ndim == 0


def test_plot_correlation_matrix_with_matches_runs_and_bounds(tmp_path):
    # small synthetic example
    n, pA, pB = 32, 4, 6
    rng = np.random.default_rng(1)
    B = rng.normal(size=(n, pB)).astype(np.float32)
    X_true = rng.normal(size=(pB, pA)).astype(np.float32)
    A = (B @ X_true + 0.02 * rng.normal(size=(n, pA))).astype(np.float32)

    # Optimize a bit to get BX close to A
    score, X_opt, BX_opt = optimize_partial_mcc(A, B, lr=5e-2, steps=150, seed=99, device="cpu")

    # Plot (no display) and save to a temp file
    out_png = tmp_path / "corr.png"
    from gcrl.alignment.partial_mcc import plot_correlation_matrix_with_matches
    C = plot_correlation_matrix_with_matches(A, BX_opt, show=False, save_path=str(out_png))

    assert C.shape == (pA, pA)
    assert (C <= 1.0 + 1e-6).all() and (C >= -1.0 - 1e-6).all()
    assert out_png.exists()
