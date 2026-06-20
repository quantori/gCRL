# tests/test_train_gcrl_vae.py
# -*- coding: utf-8 -*-
import json
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
import pytest

from gcrl.training.train_gcrl_vae import train_gcrl_vae, VAEConfig
from gcrl.grn.eigengenes import compute_eigengenes


def _toy_adata(n_cells=200, n_genes=40, n_tfs=8, n_ct=2, seed=0):
    rng = np.random.default_rng(seed)

    X = rng.normal(loc=0.0, scale=1.0, size=(n_cells, n_genes)).astype(np.float32)

    var = pd.DataFrame(index=[f"G{i}" for i in range(n_genes)])
    var["kind"] = pd.Categorical(["TF"] * n_tfs + ["TG"] * (n_genes - n_tfs))
    p = 3  # communities: 0, 1, 2
    comm = np.full(n_genes, -1, dtype=int)
    # Assign each TF to one of p communities, ensuring all p are represented
    for k in range(n_tfs):
        comm[k] = k % p
    comm[n_tfs:] = rng.integers(-1, p, size=n_genes - n_tfs)
    var["community"] = pd.Categorical(comm)

    obs = pd.DataFrame(index=[f"cell{i}" for i in range(n_cells)])
    obs["cell_type"] = pd.Categorical(rng.integers(0, n_ct, size=n_cells))

    # 70% training, 30% test
    sets = rng.choice(["training", "test"], size=n_cells, p=[0.7, 0.3])
    obs["set"] = pd.Categorical(sets)

    # Interventions: single TF names only (no doubles — VAE rejects them in training)
    tf_names = var.index[:n_tfs].tolist()
    interventions = []
    for i in range(n_cells):
        if rng.random() < 0.3:
            interventions.append("unperturbed")
        else:
            interventions.append(rng.choice(tf_names))
    obs["intervention"] = pd.Categorical(interventions)

    # Ensure every training-set intervention label has at least batch_size=4 cells
    # by overriding a subset of test cells as training for each TF
    for tf in tf_names:
        mask = np.array(obs["intervention"] == tf)
        train_mask = mask & (np.array(obs["set"]) == "training")
        need = max(0, 4 - int(train_mask.sum()))
        if need > 0:
            candidates = np.where(mask & (np.array(obs["set"]) == "test"))[0]
            pick = rng.choice(candidates, size=min(need, len(candidates)), replace=False)
            obs.loc[obs.index[pick], "set"] = "training"

    return ad.AnnData(X=X, obs=obs, var=var)


@pytest.mark.parametrize("use_grn_priors", [True, False])
def test_train_gcrl_vae_returns_model_and_history(tmp_path, use_grn_priors):
    adata = _toy_adata(seed=42)

    if use_grn_priors:
        compute_eigengenes(
            adata,
            mode="by_reference",
            community_col="community",
            reference_query='intervention == "unperturbed"',
            seed=0,
        )

    outdir = tmp_path / f"vae_grn{use_grn_priors}"
    cfg = VAEConfig(
        outdir=str(outdir),
        batch_size=4,
        epochs=2,
        lr=2e-3,
        use_GRN_priors=use_grn_priors,
        seed=0,
    )

    model, history = train_gcrl_vae(adata, cfg)

    assert model is not None
    assert isinstance(history, list) and len(history) >= 1

    hist_file = outdir / "training_history.json"
    assert hist_file.exists(), "training_history.json not written"
    saved = json.loads(hist_file.read_text())
    assert saved, "Saved history is empty"
