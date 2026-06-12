
# tests/test_eigengenes.py
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import anndata as ad

from gcrl.grn.eigengenes import compute_eigengenes

def _toy_anndata(n_cells=2000,seed=0):
    
    rng = np.random.default_rng(seed)
    
    # Make two cell types (0,1), half-half; condition: 2/3 unperturbed, 1/3 perturbed
    # 3 TFs, 5 TGs; two communities among TFs: {0,1} -> A, {2} -> B
    n_genes=8 
    
    cell_type = np.array([0]*(n_cells//2) + [1]*(n_cells - n_cells//2))
    cond = np.array(["unperturbed"]*(2*n_cells//3) + ["perturbed"]*(n_cells - 2*n_cells//3))
    rng.shuffle(cond)

    X = rng.normal(0, 1, size=(n_cells, n_genes)).astype(np.float32)

    obs = pd.DataFrame({
        "cell_type": cell_type,
        "condition": cond,
    }, index=[f"cell{i}" for i in range(n_cells)])

    var = pd.DataFrame(index=[f"g{i}" for i in range(n_genes)])
    var["kind"] = ["TF","TF","TF"] + ["TG"]*(n_genes-3)
    var["community"] = [ "A", "A", "B" ] + [None]*(n_genes-3)
    # Ensure categorical
    var["community"] = pd.Categorical(var["community"])

    adata = ad.AnnData(X=X, obs=obs, var=var)
    return adata

def test_global_mode_shapes_and_columns():
    adata = _toy_anndata()
    compute_eigengenes(adata, mode="by_reference", community_col="community",
                       reference_query='condition == "unperturbed"', seed=42)

    assert "X_comm_eig" in adata.obsm
    M, N = adata.obsm["X_comm_eig"].shape
    assert M == adata.n_obs

    meta = adata.uns["comm_eig_meta"]
    assert meta["mode"] == "by_reference"
    cols = meta["columns"]
    # Communities A,B + pooled
    assert cols == ["eig_comm_A", "eig_comm_B", "eig_all_TF"]
    assert N == len(cols)

def test_by_cell_type_mode_shapes_and_columns():
    adata = _toy_anndata()
    compute_eigengenes(adata, mode="by_cell_type_reference", community_col="community",
                       reference_query='condition == "unperturbed"', cell_type_col="cell_type", seed=7)

    assert "X_comm_eig" in adata.obsm
    M, N = adata.obsm["X_comm_eig"].shape
    assert M == adata.n_obs

    meta = adata.uns["comm_eig_meta"]
    assert meta["mode"] == "by_cell_type_reference"
    cols = meta["columns"]
    assert cols == ["eig_comm_A", "eig_comm_B", "eig_all_TF"]
    assert N == len(cols)

    # Ensure per-cell-type stats exist
    assert "per_cell_type" in meta
    assert set(meta["cell_types"]) == {"0","1"}
