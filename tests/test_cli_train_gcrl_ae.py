
# tests/test_cli_train_gcrl_ae.py
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import anndata as ad

from gcrl.cli.train_gcrl_ae import main as cli_main


def _toy_adata(n_cells=24, n_genes=10, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_cells, n_genes)).astype(np.float32)

    ct = np.array([0]*(n_cells//3) + [1]*(n_cells//3) + [2]*(n_cells - 2*(n_cells//3)))
    cond = np.array(["unperturbed"]*(n_cells//2) + ["perturbed"]*(n_cells - n_cells//2))
    rng.shuffle(cond)

    obs = pd.DataFrame({
        "condition": cond,
        "cell_type": ct,
    }, index=[f"cell{i}" for i in range(n_cells)])

    var = pd.DataFrame(index=[f"g{i}" for i in range(n_genes)])
    var["kind"] = ["TF","TF","TF"] + ["TG"]*(n_genes-3)
    var["community"] = ["A","A","B"] + [None]*(n_genes-3)
    var["community"] = pd.Categorical(var["community"])

    return ad.AnnData(X=X, obs=obs, var=var)


def test_cli_runs_with_celltype_and_saves(tmp_path):
    adata = _toy_adata()
    in_h5ad = tmp_path / "toy.h5ad"
    adata.write_h5ad(in_h5ad)

    outdir = tmp_path / "run_ct1"
    argv = [
        "--in-h5ad", str(in_h5ad),
        "--outdir", str(outdir),
        "--input-mode", "TF",
        "--epochs", "3",
        "--lr", "1e-3",
        "--standardize", "zscore_ref",
        "--cell-type-col", "cell_type",
        "--cell-type", "1",
    ]
    cli_main(argv)

    # Check key artifacts exist
    expect = [
        "gcrl_ae_model.pth",
        "embeddings.npy",
        "decoder_weights.npy",
        "history.json",
        "config.json",
        "zscore_mean.npy",
        "zscore_std.npy",
        "tf_indices.npy",
        "obs_names_used.csv",
    ]
    for f in expect:
        assert (outdir / f).exists(), f"missing artifact: {f}"

    # Check embeddings row count matches subset size
    import numpy as np
    Z = np.load(outdir / "embeddings.npy")
    # Subset size is number of cells with cell_type==1
    subset_n = int((adata.obs["cell_type"].values == 1).sum())
    assert Z.shape[0] == subset_n
