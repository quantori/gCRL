
# tests/test_train_gcrl_ae.py
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import anndata as ad
import pytest

from gcrl.training.train_gcrl_ae import train_gcrl_ae


def _toy_adata(n_cells=48, n_genes=10, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_cells, n_genes)).astype(np.float32)

    ct = np.array([0] * (n_cells // 3) + [1] * (n_cells // 3) + [2] * (n_cells - 2 * (n_cells // 3)))
    intervention = np.array(["unperturbed"] * (n_cells // 2) + ["perturbed"] * (n_cells - n_cells // 2))
    rng.shuffle(intervention)

    obs = pd.DataFrame(
        {"intervention": intervention, "cell_type": ct},
        index=[f"cell{i}" for i in range(n_cells)],
    )

    var = pd.DataFrame(index=[f"g{i}" for i in range(n_genes)])
    var["kind"] = ["TF", "TF", "TF"] + ["TG"] * (n_genes - 3)
    var["community"] = ["A", "A", "B"] + [None] * (n_genes - 3)
    var["community"] = pd.Categorical(var["community"])

    return ad.AnnData(X=X, obs=obs, var=var)


def test_train_gcrl_ae_returns_result_and_saves(tmp_path):
    adata = _toy_adata()
    outdir = str(tmp_path / "ae_run")

    res = train_gcrl_ae(
        adata,
        input_mode="TF",
        reconstruct_all=True,
        hidden_dims=(32,),
        num_epochs=3,
        lr=1e-3,
        standardize="zscore_ref",
        val_frac=0.1,
        outdir=outdir,
        seed=0,
    )

    # TrainResult has embeddings and model
    assert res.embeddings is not None
    assert res.embeddings.shape[0] == adata.n_obs

    # Key artifacts saved to disk
    import os
    for fname in ["gcrl_ae_model.pth", "embeddings.npy", "history.json", "config.json"]:
        assert os.path.exists(os.path.join(outdir, fname)), f"missing artifact: {fname}"


def test_train_gcrl_ae_cell_type_subset(tmp_path):
    adata = _toy_adata(n_cells=48)

    res = train_gcrl_ae(
        adata,
        cell_type=1,
        input_mode="TF",
        hidden_dims=(16,),
        num_epochs=2,
        standardize="zscore",
        val_frac=0.0,
        outdir=str(tmp_path / "ae_ct1"),
        seed=1,
    )

    expected_n = int((adata.obs["cell_type"].values == 1).sum())
    assert res.embeddings.shape[0] == expected_n


@pytest.mark.parametrize("standardize", ["zscore", "zscore_ref", "minmax_0_1", "none"])
def test_train_gcrl_ae_standardize_modes(standardize):
    adata = _toy_adata(n_cells=32)
    res = train_gcrl_ae(
        adata,
        input_mode="ALL",
        hidden_dims=(16,),
        num_epochs=2,
        standardize=standardize,
        val_frac=0.0,
        seed=0,
    )
    assert res.embeddings.shape[0] == adata.n_obs
