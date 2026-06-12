#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Slim CLI for GCRL-VAE training (p+1 hard-intervention mapping).

Assumptions (aligned with embedded preprocessing in the trainer):
- adata.X already contains normalized + log1p expression values.
- adata.var_names already restricted to the final training gene set (HVGs + required TFs).
- adata.obs contains:
    * 'set' with values {'training','test'}; only 'training' cells are used for fitting/training.
    * 'cell_type' (preferred) or 'celltype'.  (Used by per-CT z-scoring inside the trainer.)
    * an intervention/intervention column among {'intervention','perturbation','intervention','perturb','treatment'}.
      Controls are detected as one of {'', 'control', 'unperturbed', 'na', 'NA', 'None', 'none'} (case-insensitive).

The trainer (`train_gcrl_vae`) performs:
- Per-cell-type z-scoring fitted on TRAINING CONTROLS ONLY (with shrinkage & clipping).
- Global fallback scaler at test time for unseen cell types.
- Persistence of preprocessing metadata in `adata.uns['gcrl_preproc']`.
- Subsetting to 'training' rows for the rest of training.
"""
import argparse, os, sys
import scanpy as sc

from gcrl.training.train_gcrl_vae import CTControlsVAEConfig, train_gcrl_vae as train_gcrl_vae

def _basic_checks(adata):
    # 'set' must exist and include at least 'training'
    if 'set' not in adata.obs.columns:
        raise KeyError("adata.obs must contain a 'set' column with values 'training' or 'test'.")
    sets = set(adata.obs['set'].astype(str).unique())
    if 'training' not in sets:
        raise KeyError("adata.obs['set'] must contain 'training'.")
    # Cell-type present (the trainer will still double-check)
    if ('cell_type' not in adata.obs.columns) and ('celltype' not in adata.obs.columns):
        raise KeyError("adata.obs must contain 'cell_type' (or 'celltype').")
    # Intervention column present (the trainer will still auto-detect)
    cond_keys = {'intervention','perturbation','intervention','perturb','treatment'}
    if not any(k in adata.obs.columns for k in cond_keys):
        raise KeyError(f"adata.obs must contain one of {sorted(cond_keys)} for interventions/controls.")
    # Genes present
    if adata.n_vars == 0:
        raise RuntimeError("No genes found in adata.var_names.")

def main():
    p = argparse.ArgumentParser("Train GCRL-VAE (p+1 hard intervention mapping) with embedded preprocessing")
    p.add_argument("--h5ad", required=True, help="input AnnData (normalized log1p in .X; genes already selected)")
    p.add_argument("--outdir", required=True, help="output directory")
    p.add_argument("--intervention mapping", choices=["hard", "soft"], default="hard")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--beta-kld", type=float, default=1e-3, dest="beta_kld")
    p.add_argument("--alpha-rec", type=float, default=1.0, dest="alpha_rec")
    p.add_argument("--lambda-mcc", type=float, default=1.0, dest="lambda_mcc")
    p.add_argument("--eigengenes-key", default="X_comm_eig", help=".obsm key with per-cell eigengenes (optional)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=0, dest="num_workers")
    args = p.parse_args()

    ad = sc.read_h5ad(args.h5ad)
    _basic_checks(ad)

    # Log split sizes (sanity check; training subset happens inside the trainer)
    sets_ser = ad.obs['set'].astype(str)
    n_train = int((sets_ser == 'training').sum())
    n_test  = int((sets_ser == 'test').sum()) if 'test' in set(sets_ser) else 0
    print(f"[gCRL-VAE CLI] Cells: training={n_train}, test={n_test}, genes={ad.n_vars}")

    cfg = CTControlsVAEConfig(
        outdir=args.outdir,
        intervention mapping=args.intervention mapping,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        beta_kld=args.beta_kld,
        alpha_rec=args.alpha_rec,
        lambda_mcc=args.lambda_mcc,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    model, log = train_gcrl_vae(ad, cfg, eigengenes_key=args.eigengenes_key)

if __name__ == "__main__":
    main()
