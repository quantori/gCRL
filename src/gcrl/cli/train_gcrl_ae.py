
# src/gcrl/cli/train_gcrl_ae.py
# -*- coding: utf-8 -*-
"""
Package CLI entry for training gCRL-AE.

This mirrors scripts/train_gcrl_ae.py but is importable as a package module:
    from gcrl.cli.train_gcrl_ae import main
"""

from __future__ import annotations

import argparse
import anndata as ad

from gcrl.training.train_gcrl_ae import train_gcrl_ae


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train gCRL-AE on an AnnData file")
    p.add_argument("--in-h5ad", required=True, help="Input AnnData .h5ad (normalized, log1p-transformed)")
    p.add_argument("--outdir", required=False, default=None, help="Directory to save artifacts (model, embeddings, etc.)")

    # Column names
    p.add_argument("--community-col", default="community")
    p.add_argument("--kind-col", default="kind")
    p.add_argument("--intervention-col", default="intervention")
    p.add_argument("--cell-type-col", default="cell_type")
    p.add_argument("--cell-type", default=None, help="Value in obs[cell_type_col] to select (string or int)")

    # Model/training
    p.add_argument("--input-mode", choices=["TF","ALL"], default="TF")
    p.add_argument("--reconstruct-all", action="store_true", help="Decoder reconstructs all genes (default)")
    p.add_argument("--no-reconstruct-all", dest="reconstruct_all", action="store_false", help="Decoder reconstructs inputs only")
    p.set_defaults(reconstruct_all=True)

    p.add_argument("--latent-dim", type=int, default=None, help="If omitted, inferred as (#communities + 1)")
    p.add_argument("--hidden-dims", type=str, default="256", help="Comma-separated hidden sizes, e.g. '256,128'")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-step", type=int, default=50)
    p.add_argument("--lr-gamma", type=float, default=0.5)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--device", type=str, default=None, help="'cpu', 'cuda', or None (auto)")
    p.add_argument("--seed", type=int, default=42)

    # Preprocessing
    p.add_argument("--reference-query", type=str, default='intervention == "unperturbed"')
    p.add_argument("--standardize", choices=["zscore_ref","minmax_0_1","none"], default="zscore_ref")

    return p


def _parse_cell_type(value: str | None):
    if value is None or value == "None":
        return None
    try:
        return int(value)
    except Exception:
        return value


def main(argv=None):
    p = build_parser()
    args = p.parse_args(argv)

    # Parse hidden dims
    hidden_dims = tuple(int(x) for x in args.hidden_dims.split(",") if x.strip())

    adata = ad.read_h5ad(args.in_h5ad)

    _ = train_gcrl_ae(
        adata=adata,
        community_col=args.community_col,
        kind_col=args.kind_col,
        intervention_col=args.intervention_col,
        cell_type_col=args.cell_type_col,
        cell_type=_parse_cell_type(args.cell_type),
        input_mode=args.input_mode,
        reconstruct_all=args.reconstruct_all,
        latent_dim=args.latent_dim,
        hidden_dims=hidden_dims,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        lr_step=args.lr_step,
        lr_gamma=args.lr_gamma,
        weight_decay=args.weight_decay,
        device=args.device,
        seed=args.seed,
        reference_query=args.reference_query,
        standardize=args.standardize,
        outdir=args.outdir,
    )
