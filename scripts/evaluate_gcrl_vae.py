#!/usr/bin/env python3
from __future__ import annotations
import argparse, anndata as ad
from gcrl.evaluation import evaluate_gcrl_vae, EvalConfig

def main():
    p = argparse.ArgumentParser(description="Evaluate gCRL-VAE predictions (confounding-aware).")
    p.add_argument("--adata", required=True, help="Path to .h5ad with obs columns set/train/test, cell_type, intervention.")
    p.add_argument("--out_dir", default="./experiments/generalization", help="Output directory for CSVs/PNGs.")
    args = p.parse_args()

    adata = ad.read_h5ad(args.adata)
    cfg = EvalConfig(out_dir=args.out_dir)
    # Here we run with LinearDelta baseline unless you import and pass your actual model predictor.
    res = evaluate_gcrl_vae(adata, model=None, predictor=None, cfg=cfg)
    print(res["summary_by_ct_int"].head())

if __name__ == "__main__":
    main()
