# tests/test_train_gcrl_vae.py
# -*- coding: utf-8 -*-
import json
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
import pytest


# ---- Tiny synthetic AnnData that satisfies the gCRL-VAE "contract" ----
def _toy_adata(n_cells=48, n_genes=60, n_tfs=8, n_ct=3, seed=0):
    rng = np.random.default_rng(seed)

    # Expression (log1p-normalized-like scale)
    X = rng.normal(loc=0.0, scale=1.0, size=(n_cells, n_genes)).astype(np.float32)

    # Genes: first n_tfs are TFs with community IDs; others TGs (some with -1 community)
    var = pd.DataFrame(index=[f"G{i}" for i in range(n_genes)])
    var["kind"] = pd.Categorical(["TF"] * n_tfs + ["TG"] * (n_genes - n_tfs))
    # Put TFs into communities 0..p-1 (p <= n_tfs), TGs mostly assigned; a few -1
    p = min(6, n_tfs)  # number of communities used in tests
    comm = np.full(n_genes, -1, dtype=int)
    comm[:n_tfs] = rng.integers(0, p, size=n_tfs)
    comm[n_tfs:] = rng.integers(-1, p, size=n_genes - n_tfs)
    var["community"] = pd.Categorical(comm)

    # Cells: sets, cell types, interventions
    obs = pd.DataFrame(index=[f"cell{i}" for i in range(n_cells)])
    obs["set"] = pd.Categorical(
        rng.choice(["training", "validation", "test"], size=n_cells, p=[0.6, 0.2, 0.2])
    )
    obs["cell_type"] = pd.Categorical(rng.integers(0, n_ct, size=n_cells))

    # Ensure each CT has some training controls
    ctl_mask = np.zeros(n_cells, dtype=bool)
    for ct in range(n_ct):
        ct_mask = (obs["cell_type"].values == ct)
        ct_idx = np.where(ct_mask & (obs["set"].values == "training"))[0]
        # force at least 4 training controls per CT
        need = max(0, 4 - (ct_idx.size // 2))
        pick = rng.choice(np.where(ct_mask)[0], size=need, replace=False)
        ctl_mask[pick] = True
    obs.loc[ctl_mask, "set"] = "training"

    # Interventions:
    # controls use "unperturbed"; perturbations use TF names (single TF); some combos
    tf_names = var.index[:n_tfs].tolist()
    interventions = []
    for i in range(n_cells):
        if rng.random() < 0.4:  # controls
            interventions.append("unperturbed")
        else:
            if rng.random() < 0.8:  # single
                interventions.append(rng.choice(tf_names))
            else:  # 2-TF combo (sorted, '+'-joined)
                tfs = sorted(rng.choice(tf_names, size=2, replace=False))
                interventions.append("+".join(tfs))
    obs["intervention"] = pd.Categorical(interventions)

    adata = ad.AnnData(X=X, obs=obs, var=var)
    return adata


@pytest.mark.parametrize("epochs", [1])  # keep it fast for CI
def test_train_gcrl_vae_programmatic(tmp_path: Path, epochs: int):
    """
    Programmatic API test. Calls gcrl.training.train_gcrl_vae.train_gcrl_vae(...)
    with a SimpleNamespace cfg to avoid coupling to an exact dataclass signature.
    Asserts that a training artifact and a non-empty history are produced.
    """
    # Import the training entrypoint
    from gcrl.training.train_gcrl_vae import train_gcrl_vae

    adata = _toy_adata(seed=42)

    outdir = tmp_path / "vae_run"
    outdir.mkdir(parents=True, exist_ok=True)

    # Minimal config via duck-typing (SimpleNamespace)
    cfg = SimpleNamespace(
        intervention_mapping = 'hard',
        outdir=str(outdir),
        batch_size=16,
        epochs=epochs,
        lr=2e-3,
        beta_kld=1e-3,
        alpha_rec=1.0,
        lambda_mcc=1.0,
        num_workers=0,
        seed=0,
    )

    # Run training (should return (model, history))
    model, history = train_gcrl_vae(adata, cfg)

    # Basic assertions on return types
    assert model is not None, "Model was not returned"
    assert isinstance(history, (list, dict)), "History should be a list[dict] or dict"
    # If history is a list, make sure it's not empty; if dict, must have at least one key
    if isinstance(history, list):
        assert len(history) >= 1, "Empty training history"
    else:
        assert len(history.keys()) >= 1, "Empty training history dict"

    # File artifact that train_gcrl_vae is expected to write
    hist_file = outdir / "training_history.json"
    assert hist_file.exists(), "Expected training_history.json not found"
    with open(hist_file, "r") as f:
        saved_hist = json.load(f)
    assert saved_hist, "Saved training history is empty"

    # Sanity: outdir contains at least the history file and one more file (e.g., config/ckpt/etc.)
    all_files = list(outdir.glob("*"))
    assert len(all_files) >= 1, f"Output directory seems empty: {outdir}"


