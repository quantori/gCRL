#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GPU-enabled partial-MCC alignment experiments.

Adds a --use_gpu flag:
- If set, optimization (partial-MCC) runs on CUDA (if available).
- When --use_gpu is set, the script enforces --n_jobs 1 to avoid GPU contention.
- Eigengene (PCA) computations remain CPU-based (scikit-learn).

Two stages:
1) Baseline: run N seeds (parallel on CPU if --use_gpu is False; else 1-by-1 on GPU).
2) Permutation test: P permutations; for each, recompute eigengenes once, then run K seeds
   (parallel on CPU if --use_gpu is False; else 1-by-1 on GPU).

Outputs:
- baseline_scores.npy / .csv
- perm_scores.npy / .csv (with perm_id and seed)
- plot_density.png
- run_config.json
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.decomposition import PCA
import torch
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
from gcrl.grn.eigengenes import compute_eigengenes

# -------------------- Utility: set seed -------------------- #
def set_seed(seed:int):
    np.random.seed(seed)
    torch.manual_seed(seed)

# -------------------- Utilities: eigengenes -------------------- #
def _to_dense(X) -> np.ndarray:
    if hasattr(X, "toarray"):
        X = X.toarray()
    return np.asarray(X, dtype=np.float32)

def _fit_pc1_on_reference(X_all, gene_idx, ref_idx, random_state):
    X_ref = X_all[np.asarray(ref_idx)[:, None], gene_idx]
    mean = X_ref.mean(axis=0)
    std = X_ref.std(axis=0, ddof=0)
    std[std == 0] = 1.0
    X_ref_std = (X_ref - mean) / std
    pca = PCA(n_components=1, random_state=random_state)
    pca.fit(X_ref_std)
    comp = pca.components_[0]
    X_all_std = (X_all[:, gene_idx] - mean) / std
    eig = (X_all_std @ comp.reshape(-1, 1)).ravel().astype(np.float32)
    return eig, {"mean": mean.astype(np.float32), "std": std.astype(np.float32), "components": comp.astype(np.float32)}

def _compute_eigs_given_view_matrix(X_view, var, comm_col, tf_mask, ref_idx, seed):
    comm_series = var[comm_col].astype("category")
    comm_ids = list(comm_series.cat.categories)
    n_cells = X_view.shape[0]
    eig_mat = np.zeros((n_cells, len(comm_ids) + 1), dtype=np.float32)

    for j, comm in enumerate(comm_ids):
        comm_mask = (comm_series.to_numpy() == comm) & tf_mask
        gene_idx = np.where(comm_mask)[0]
        if gene_idx.size == 0:
            eig_mat[:, j] = 0.0
            continue
        elif gene_idx.size == 1:
            g = gene_idx[0]
            X_ref = X_view[ref_idx, g]
            mean = X_ref.mean(); std = X_ref.std() or 1.0
            eig = ((X_view[:, g] - mean) / std).astype(np.float32)
            eig_mat[:, j] = eig
            continue
        eig, _ = _fit_pc1_on_reference(X_view, gene_idx, ref_idx, seed)
        eig_mat[:, j] = eig

    # pooled TF
    all_tf_idx = np.where(tf_mask)[0]
    if all_tf_idx.size == 0:
        pooled = np.zeros(n_cells, dtype=np.float32)
    elif all_tf_idx.size == 1:
        g = all_tf_idx[0]
        X_ref = X_view[ref_idx, g]
        mean = X_ref.mean(); std = X_ref.std() or 1.0
        pooled = ((X_view[:, g] - mean) / std).astype(np.float32)
    else:
        pooled, _ = _fit_pc1_on_reference(X_view, all_tf_idx, ref_idx, seed)
    eig_mat[:, -1] = pooled

    comm_ids = list(var[comm_col].astype("category").cat.categories)
    col_names = [f"eig_comm_{c}" for c in comm_ids] + ["eig_all_TF"]
    return eig_mat, col_names

def compute_eigengenes_matrix(adata, mode:str, community_col:str, reference_query:str, seed:int, method:str="PC", cell_type_col:str="cell_type"):
    compute_eigengenes(
        adata,
        community_col=community_col,
        reference_query=reference_query,
        mode=mode,
        method=method,
        cell_type_col=cell_type_col,
        seed=seed,
    )
    cols = adata.uns["comm_eig_meta"]["columns"]
    return adata.obsm["X_comm_eig"], cols

# -------------------- Partial-MCC optimization (GPU-capable) -------------------- #
def standardize_columns_torch(M: torch.Tensor) -> torch.Tensor:
    mean = M.mean(dim=0, keepdim=True)
    std = M.std(dim=0, keepdim=True)
    return (M - mean) / (std + 1e-8)

def residuals(Y: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    if X.ndim == 1 or X.shape[1] == 0:
        return Y
    X_pinv = torch.linalg.pinv(X)
    beta = X_pinv @ Y
    Y_hat = X @ beta
    return Y - Y_hat

def partial_mcc_loss_torch(A: torch.Tensor, BX: torch.Tensor) -> torch.Tensor:
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
    return -torch.mean(partial_corrs)

def optimize_partial_mcc(A_np: np.ndarray, B_np: np.ndarray, lr=1e-2, steps=500, seed=42, device="cpu"):
    set_seed(seed)
    dev = torch.device(device)
    A = torch.tensor(A_np, dtype=torch.float32, device=dev)
    B = torch.tensor(B_np, dtype=torch.float32, device=dev)
    nA, pA = A.shape
    nB, pB = B.shape
    if nA != nB:
        raise ValueError(f"A and B must have same #rows (cells). A:{nA}, B:{nB}")
    X = torch.randn(pB, pA, device=dev, requires_grad=True)
    opt = torch.optim.Adam([X], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = partial_mcc_loss_torch(A, B @ X)
        loss.backward()
        opt.step()
    with torch.no_grad():
        score = -partial_mcc_loss_torch(A, B @ X).item()
    return score

# -------------- Worker wrapper (CPU) -------------- #
def _worker_optimize_cpu(seed, A, B, lr, steps):
    return optimize_partial_mcc(A, B, lr=lr, steps=steps, seed=seed, device="cpu")

# -------------- Main -------------- #
def main():
    ap = argparse.ArgumentParser(description="GPU-enabled baseline + permutation partial-MCC experiments.")
    ap.add_argument("--h5ad", required=True, help="AnnData file (.h5ad) with X, obs/var metadata.")
    ap.add_argument("--community_col", default="community", help="var column with community labels.")
    ap.add_argument("--mode", choices=["all_cells","by_reference","by_cell_type","by_cell_type_reference"],
                    default="all_cells", help="Which cells to use for standardization/fitting.")
    ap.add_argument("--method", choices=["PC","average"], default="PC",
                    help="How to summarize TF expression per community: PC1 or mean z-score.")
    ap.add_argument("--reference_query", default='intervention == \"unperturbed\"', help="Reference subset query.")
    ap.add_argument("--cell_type_col", default="cell_type", help="obs column for cell type (by_cell_type mode).")
    ap.add_argument("--embeddings_npy", help="Path to AE embeddings .npy (rows aligned to adata.obs).")
    ap.add_argument("--embeddings_csv", help="Path to AE embeddings .csv (index must match adata.obs_names).")
    ap.add_argument("--baseline_runs", type=int, default=100, help="#seeds for baseline.")
    ap.add_argument("--perm_outer", type=int, default=100, help="#TF-permutation replicates.")
    ap.add_argument("--perm_inner", type=int, default=10, help="#seeds per permutation replicate.")
    ap.add_argument("--steps", type=int, default=500, help="Optimization steps per run.")
    ap.add_argument("--lr", type=float, default=1e-2, help="Learning rate for optimizer.")
    ap.add_argument("--seed0", type=int, default=123, help="Master seed; per-run seeds derived from this.")
    ap.add_argument("--n_jobs", type=int, default=1, help="Parallel CPU workers when --use_gpu is False.")
    ap.add_argument("--use_gpu", action="store_true", help="Run the optimizer on CUDA if available (forces --n_jobs 1).")
    ap.add_argument("--outdir", default="results_mcc_gpu", help="Output directory.")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "run_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # Determine device
    device = "cuda" if (args.use_gpu and torch.cuda.is_available()) else "cpu"
    if args.use_gpu and device != "cuda":
        print("⚠️  --use_gpu was set but CUDA is not available. Falling back to CPU.")
    if args.use_gpu and args.n_jobs != 1:
        print("ℹ️  --use_gpu forces --n_jobs=1 to avoid GPU contention.")
        args.n_jobs = 1

    # Load data
    adata = sc.read_h5ad(args.h5ad)
    if args.embeddings_npy:
        B = np.load(args.embeddings_npy)
        if B.shape[0] != adata.n_obs:
            raise ValueError(f"Embeddings rows {B.shape[0]} != adata cells {adata.n_obs}")
    elif args.embeddings_csv:
        dfB = pd.read_csv(args.embeddings_csv, index_col=0)
        if not adata.obs_names.isin(dfB.index).all():
            missing = adata.obs_names[~adata.obs_names.isin(dfB.index)]
            raise ValueError(f"Embeddings CSV is missing {len(missing)} cells (e.g. {missing[:5].tolist()})")
        dfB = dfB.loc[adata.obs_names]
        B = dfB.values.astype(np.float32)
    else:
        raise ValueError("Provide --embeddings_npy or --embeddings_csv")

    # ---------- Part 1: Baseline ---------- #
    A, cols = compute_eigengenes_matrix(
        adata, mode=args.mode, community_col=args.community_col,
        reference_query=args.reference_query, seed=args.seed0,
        method=args.method, cell_type_col=args.cell_type_col
    )
    baseline_seeds = [args.seed0 + i for i in range(args.baseline_runs)]
    if device == "cpu" and args.n_jobs > 1:
        with ProcessPoolExecutor(max_workers=args.n_jobs) as ex:
            futs = {ex.submit(_worker_optimize_cpu, s, A, B, args.lr, args.steps): s for s in baseline_seeds}
            seed_to_score = {}
            for fut, s in futs.items():
                seed_to_score[s] = fut.result()
            baseline_scores = np.array([seed_to_score[s] for s in baseline_seeds], dtype=np.float32)
    else:
        baseline_scores = np.array(
            [optimize_partial_mcc(A, B, lr=args.lr, steps=args.steps, seed=s, device=device)
             for s in baseline_seeds],
            dtype=np.float32
        )

    np.save(os.path.join(args.outdir, "baseline_scores.npy"), baseline_scores)
    pd.DataFrame({"seed": baseline_seeds, "score": baseline_scores}).to_csv(
        os.path.join(args.outdir, "baseline_scores.csv"), index=False
    )

    # ---------- Part 2: Permutations ---------- #
    perm_scores_all = []
    perm_ids_all = []
    perm_seeds_all = []
    for p in range(args.perm_outer):
        perm_seed = args.seed0 + 10_000 + p
        ad_perm = adata.copy()
        # Permute TF communities
        kinds_clean = ad_perm.var["kind"].astype(str).str.strip().str.upper()
        tf_idx = np.where(kinds_clean == "TF")[0]
        comm_vals = ad_perm.var[args.community_col].values.copy()
        shuffled = comm_vals[tf_idx].copy()
        rng = np.random.default_rng(perm_seed)
        rng.shuffle(shuffled)
        comm_vals[tf_idx] = shuffled
        ad_perm.var[args.community_col] = pd.Categorical(comm_vals)

        # Recompute eigengenes for permuted communities
        A_perm, _ = compute_eigengenes_matrix(
            ad_perm, mode=args.mode, community_col=args.community_col,
            reference_query=args.reference_query, seed=perm_seed,
            method=args.method, cell_type_col=args.cell_type_col
        )

        inner_seeds = [args.seed0 + 100_000 + (p * args.perm_inner + j) for j in range(args.perm_inner)]
        if device == "cpu" and args.n_jobs > 1:
            with ProcessPoolExecutor(max_workers=args.n_jobs) as ex:
                futs = {ex.submit(_worker_optimize_cpu, s, A_perm, B, args.lr, args.steps): s for s in inner_seeds}
                seed_to_score = {}
                for fut, s in futs.items():
                    seed_to_score[s] = fut.result()
                scores = [seed_to_score[s] for s in inner_seeds]
        else:
            scores = [
                optimize_partial_mcc(A_perm, B, lr=args.lr, steps=args.steps, seed=s, device=device)
                for s in inner_seeds
            ]
        perm_scores_all.extend(scores)
        perm_ids_all.extend([p]*len(inner_seeds))
        perm_seeds_all.extend(inner_seeds)

    perm_scores = np.asarray(perm_scores_all, dtype=np.float32)
    df_perm = pd.DataFrame({"perm_id": perm_ids_all, "seed": perm_seeds_all, "score": perm_scores})
    np.save(os.path.join(args.outdir, "perm_scores.npy"), perm_scores)
    df_perm.to_csv(os.path.join(args.outdir, "perm_scores.csv"), index=False)

    # ---------- Plot ---------- #
    plt.figure(figsize=(7,5))
    try:
        from scipy.stats import gaussian_kde
        x1 = baseline_scores
        x2 = perm_scores
        xs = np.linspace(min(x1.min(), x2.min()), max(x1.max(), x2.max()), 400)
        kde1 = gaussian_kde(x1)
        kde2 = gaussian_kde(x2)
        plt.plot(xs, kde1(xs), label=f"Baseline (n={len(x1)})")
        plt.plot(xs, kde2(xs), label=f"Permuted TF↔community (n={len(x2)})")
    except Exception:
        plt.hist(baseline_scores, bins=30, density=True, alpha=0.6, label=f"Baseline (n={len(baseline_scores)})")
        plt.hist(perm_scores, bins=30, density=True, alpha=0.6, label=f"Permuted (n={len(perm_scores)})")
    plt.xlabel("Partial MCC")
    plt.ylabel("Density")
    plt.title("Partial MCC: baseline vs TF-community permutation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "plot_density.png"), dpi=200)
    plt.close()

    print("✅ Done (GPU-enabled)." )
    print(f"Device used for optimizer: {device}")
    print(f"Baseline mean±sd: {baseline_scores.mean():.4f} ± {baseline_scores.std():.4f}")
    print(f"Permuted mean±sd: {perm_scores.mean():.4f} ± {perm_scores.std():.4f}")
    print(f"Outputs in: {args.outdir}")

if __name__ == "__main__":
    main()
