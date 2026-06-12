
# tests/test_sergio_cli.py
# -*- coding: utf-8 -*-
import os, pytest
from pathlib import Path

from gcrl.cli.build_grn import main as build_grn_cli
from gcrl.cli.run_sergio import main as run_sergio_cli

def _sergio_available():
    try:
        from gcrl.simulation.sergio_sim import _import_sergio
        _import_sergio(None)
        return True
    except Exception:
        return False

@pytest.mark.skipif(not _sergio_available(), reason="SERGIO not available/configured")
def test_end_to_end_sergio(tmp_path):
    grn_dir = tmp_path / "grn"
    out_h5ad = tmp_path / "sim.h5ad"
    # build GRN
    build_grn_cli(["--outdir", str(grn_dir), "--seed", "0", "--n-communities", "3", "--n-targets", "50", "--n-bins", "2"])
    # run SERGIO (small sizes)
    run_sergio_cli([
        "--grn-dir", str(grn_dir),
        "--out-h5ad", str(out_h5ad),
        "--n-unperturbed", "50",
        "--n-perturbed-per-comm", "20",
        "--seed", "1",
    ])
    assert out_h5ad.exists()
