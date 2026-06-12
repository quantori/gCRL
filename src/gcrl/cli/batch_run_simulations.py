
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch runner for SERGIO simulations (gCRL).

Generates multiple GRNs and datasets using:
- gcrl.simulation.grn.build_grn
- gcrl.simulation.sergio_sim.run_sergio_sim

Outputs are saved under data/simulated/ with subfolders reflecting grid settings:
C{n_communities}_TG{n_targets}_B{n_bins}_U{n_unperturbed}_P{n_perturbed}_S{seed}

Example:
    python scripts/batch_run_simulations.py \
        --seeds 0 1 \
        --n-communities 3 5 \
        --n-targets 400 800 \
        --n-bins 3 \
        --n-unperturbed 1000 \
        --n-perturbed-per-comm 400 \
        --noise-params 1.0 --decays 0.8 --sampling-state 15 --noise-type dpd
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Dict

import pandas as pd

from gcrl.simulation.grn import build_grn
from gcrl.simulation.sergio_sim import run_sergio_sim
from gcrl.config import resolve_sergio_dir


def combo_tag(cfg: Dict, n_unperturbed: int, n_perturbed_per_comm: int) -> str:
    return f"C{cfg['n_communities']}_TG{cfg['n_targets']}_B{cfg['n_bins']}_U{n_unperturbed}_P{n_perturbed_per_comm}_S{cfg['seed']}"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Batch-run SERGIO simulations with gCRL tools")
    p.add_argument("--out-root", default="data/simulated", help="Root folder for outputs (default: data/simulated)")

    # grids
    p.add_argument("--seeds", type=int, nargs="+", default=[0], help="Seeds for GRN/SERGIO")
    p.add_argument("--n-communities", type=int, nargs="+", default=[3], help="List of community counts")
    p.add_argument("--n-targets", type=int, nargs="+", default=[400], help="List of targets per run")
    p.add_argument("--n-bins", type=int, default=3, help="Number of bins for SERGIO MRs")

    # sizes
    p.add_argument("--n-unperturbed", type=int, default=1000)
    p.add_argument("--n-perturbed-per-comm", type=int, default=400)

    # SERGIO params
    p.add_argument("--noise-params", type=float, default=1.0)
    p.add_argument("--decays", type=float, default=0.8)
    p.add_argument("--sampling-state", type=int, default=15)
    p.add_argument("--noise-type", type=str, default="dpd")

    # Optional: explicit SERGIO repo dir (else config/environment used)
    p.add_argument("--sergio-dir", default=None, help="Path to SERGIO repo (overrides config)")

    # Parallel workers within each dataset run (affects internal SERGIO jobs only)
    p.add_argument("--jobs", type=int, default=None, help="Workers for per-dataset SERGIO (default auto)")

    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # show SERGIO location
    resolved_sergio = args.sergio_dir or resolve_sergio_dir(None)
    print(f"[i] SERGIO resolved at: {resolved_sergio}")

    # grid
    grid: List[Dict] = []
    for seed in args.seeds:
        for nc in args.n_communities:
            for nt in args.n_targets:
                grid.append(dict(seed=seed, n_communities=nc, n_targets=nt, n_bins=args.n_bins))

    print(f"[i] Total configurations: {len(grid)}")

    summary_rows = []

    for i, cfg in enumerate(grid, 1):
        seed = cfg['seed']
        nc = cfg['n_communities']
        nt = cfg['n_targets']
        nb = cfg['n_bins']

        tag = combo_tag(cfg, args.n_unperturbed, args.n_perturbed_per_comm)
        combo_dir = out_root / tag
        grn_dir = combo_dir / "grn"
        out_h5ad = combo_dir / f"sergio_{tag}.h5ad"
        combo_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{i}/{len(grid)}] {tag}")
        print(f"  • Build GRN → {grn_dir}")
        _ = build_grn(
            outdir=str(grn_dir),
            seed=seed,
            n_communities=nc,
            n_targets=nt,
            n_bins=nb,
            inter_density='five_per_pair',
            p_repression=0.30,
        )

        print(f"  • Run SERGIO → {out_h5ad}")
        adata = run_sergio_sim(
            grn_dir=str(grn_dir),
            out_h5ad=str(out_h5ad),
            n_unperturbed=args.n_unperturbed,
            n_perturbed_per_comm=args.n_perturbed_per_comm,
            seed=seed + 123,  # different seed if desired
            sergio_dir=args.sergio_dir,
            noise_params=args.noise_params,
            decays=args.decays,
            sampling_state=args.sampling_state,
            noise_type=args.noise_type,
            jobs=args.jobs,
        )

        # collect metadata
        n_cells = adata.n_obs
        n_genes = adata.n_vars
        n_tf = int((adata.var['kind'] == 'TF').sum())
        n_tg = int((adata.var['kind'] == 'TG').sum())
        n_comms = int(pd.Categorical(adata.var['community']).categories.size)

        summary_rows.append(dict(
            tag=tag,
            seed_grn=seed,
            seed_sergio=seed+123,
            n_cells=n_cells,
            n_genes=n_genes,
            n_tf=n_tf,
            n_tg=n_tg,
            n_communities=n_comms,
            n_unperturbed=args.n_unperturbed,
            n_perturbed_per_comm=args.n_perturbed_per_comm,
            grn_dir=str(grn_dir),
            out_h5ad=str(out_h5ad),
        ))

    summary = pd.DataFrame(summary_rows)
    summary_path = out_root / 'summary.csv'
    summary.to_csv(summary_path, index=False)
    print(f"[✓] Wrote summary → {summary_path}")


if __name__ == "__main__":
    main()
