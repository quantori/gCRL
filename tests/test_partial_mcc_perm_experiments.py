
# tests/test_mcc_experiments.py
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import anndata as ad

from gcrl.alignment.partial_mcc_perm_experiments import run_partial_mcc_perm_experiments

def _toy_adata(n_cells=40, n_genes=12, seed=0):
    rng = np.random.default_rng(seed)
    # obs
    cell_type = np.array([0]*(n_cells//2) + [1]*(n_cells - n_cells//2))
    cond = np.array(["unperturbed"]*(2*n_cells//3) + ["perturbed"]*(n_cells - 2*n_cells//3))
    rng.shuffle(cond)
    obs = pd.DataFrame({"cell_type": cell_type, "condition": cond},
                       index=[f"cell{i}" for i in range(n_cells)])

    # var
    var = pd.DataFrame(index=[f"g{i}" for i in range(n_genes)])
    var["kind"] = ["TF","TF","TF","TF"] + ["TG"]*(n_genes-4)
    var["community"] = ["A","A","B","B"] + [None]*(n_genes-4)
    var["community"] = pd.Categorical(var["community"])

    # data
    X = rng.normal(size=(n_cells, n_genes)).astype(np.float32)
    return ad.AnnData(X=X, obs=obs, var=var)

def test_run_partial_mcc_perm_experiments_shapes(tmp_path):
    adata = _toy_adata()
    rng = np.random.default_rng(1)
    # Fake AE embeddings with 6 dims
    B = rng.normal(size=(adata.n_obs, 6)).astype(np.float32)

    out = run_partial_mcc_perm_experiments(
        adata=adata,
        embeddings=B,
        community_col="community",
        reference_query='condition == "unperturbed"',
        mode="by_cell_type_reference",
        method="PC",
        cell_type_col="cell_type",
        n_real_seeds=5,
        n_permutations=3,
        n_perm_seeds=2,
        lr=5e-2,
        steps=100,
        device="cpu",
        master_seed=123,
        save_density_path=str(tmp_path / "dens.png"),
    )

    assert "scores_real" in out and "scores_perm" in out
    assert out["scores_real"].shape == (5,)
    assert out["scores_perm"].shape == (3, 2)
