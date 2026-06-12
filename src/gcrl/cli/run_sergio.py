
# src/gcrl/cli/run_sergio.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse
from gcrl.simulation.sergio_sim import run_sergio_sim

def build_parser():
    p = argparse.ArgumentParser(description="Run SERGIO simulations from a GRN folder")
    p.add_argument("--grn-dir", required=True)
    p.add_argument("--out-h5ad", required=True)
    p.add_argument("--n-unperturbed", type=int, default=2000)
    p.add_argument("--n-perturbed-per-comm", type=int, default=1000)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--sergio-dir", default=None, help="Optional explicit path to SERGIO repo (else use config)")
    p.add_argument("--jobs", type=int, default=None, help="Parallel workers")
    return p

def main(argv=None):
    args = build_parser().parse_args(argv)
    run_sergio_sim(
        grn_dir=args.grn_dir,
        out_h5ad=args.out_h5ad,
        n_unperturbed=args.n_unperturbed,
        n_perturbed_per_comm=args.n_perturbed_per_comm,
        seed=args.seed,
        sergio_dir=args.sergio_dir,
        jobs=args.jobs,
    )

if __name__ == "__main__":
    main()
