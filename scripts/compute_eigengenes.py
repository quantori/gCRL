
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Thin CLI wrapper around gcrl.grn.eigengenes.compute_eigengenes
(AnnData is mutated in place by the library function.)

Example:
    python scripts/compute_eigengenes.py \
        --in_h5ad data/my.h5ad \
        --out_h5ad data/my_with_eigs.h5ad \
        --mode by_cell_type \
        --reference_query 'intervention == "unperturbed"' \
        --community_col community \
        --cell_type_col cell_type \
        --seed 42 \
        --export_csv results/eigs.csv \
        --export_npy results/eigs.npy
"""
import argparse
import numpy as np
import pandas as pd
import scanpy as sc

from gcrl.grn.eigengenes import compute_eigengenes

def parse_args():
    p = argparse.ArgumentParser(description="Compute community eigengenes (PC1) with optional per-cell-type stacking.")
    p.add_argument("--in_h5ad", required=True, help="Input .h5ad with required obs/var columns present.")
    p.add_argument("--out_h5ad", required=True, help="Output .h5ad with eigengenes saved to .obsm/.uns")
    p.add_argument("--reference_query", default='intervention == \"unperturbed\"',
                   help="Pandas query on adata.obs for reference rows (fit PC1s).")
    p.add_argument("--community_col", default="community",
                   help="Column in adata.var with community assignment.")
    p.add_argument("--mode", choices=["all_cells", "by_reference", "by_cell_type", "by_cell_type_reference"],
                   default="all_cells", help="Which cells to use for standardization/fitting.")
    p.add_argument("--method", choices=["PC", "average"], default="PC",
                   help="How to summarize TF expression per community: PC1 or mean z-score.")
    p.add_argument("--cell_type_col", default="cell_type",
                   help="obs column with cell type labels (for by_cell_type modes).")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    p.add_argument("--export_csv", default=None, help="Optional path to save eigengene matrix as CSV.")
    p.add_argument("--export_npy", default=None, help="Optional path to save eigengene matrix as NPY (float32).")
    return p.parse_args()

def main():
    args = parse_args()
    adata = sc.read_h5ad(args.in_h5ad)

    compute_eigengenes(
        adata=adata,
        community_col=args.community_col,
        reference_query=args.reference_query,
        mode=args.mode,
        method=args.method,
        cell_type_col=args.cell_type_col,
        seed=args.seed,
    )

    # AnnData has been mutated already; just write to disk
    sc.write(args.out_h5ad, adata, compression="gzip")

    # Optional exports
    if args.export_csv:
        pd.DataFrame(adata.obsm["X_comm_eig"], index=adata.obs_names, columns=res.columns).to_csv(args.export_csv)
    if args.export_npy:
        np.save(args.export_npy, adata.obsm["X_comm_eig"].astype(np.float32))

    print(f"✅ Wrote eigengenes to {args.out_h5ad} with shape {res.matrix.shape}")

if __name__ == "__main__":
    main()
