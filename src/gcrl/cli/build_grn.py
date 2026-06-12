
# src/gcrl/cli/build_grn.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse
from gcrl.simulation.grn import build_grn, Params

def build_parser():
    p = argparse.ArgumentParser(description="Build a synthetic GRN for SERGIO")
    p.add_argument("--outdir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-communities", type=int, default=5)
    p.add_argument("--n-targets", type=int, default=1000)
    p.add_argument("--n-bins", type=int, default=3)
    return p

def main(argv=None):
    args = build_parser().parse_args(argv)
    build_grn(
        outdir=args.outdir,
        seed=args.seed,
        n_communities=args.n_communities,
        n_targets=args.n_targets,
        n_bins=args.n_bins,
    )

if __name__ == "__main__":
    main()
