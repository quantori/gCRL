
# tests/test_grn_sim.py
# -*- coding: utf-8 -*-
import os
from pathlib import Path
from gcrl.simulation.grn import build_grn

def test_build_grn_writes_files(tmp_path):
    outdir = tmp_path / "grn"
    build_grn(outdir=str(outdir), seed=0, n_communities=3, n_targets=50, n_bins=2)
    # Expect core files
    for fname in ["nodes.csv", "edges_tf_tf.csv", "edges_tf_target.csv", "input_targets.txt", "input_regs.txt", "README.csv"]:
        assert (outdir / fname).exists(), f"missing {fname}"
