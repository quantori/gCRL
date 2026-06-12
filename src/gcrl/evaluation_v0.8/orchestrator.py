from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
import anndata as ad
from typing import List, Dict, Tuple
from .config import EvalConfig
from .baselines import BaselineProvider
from .metrics import DEFAULT_METRICS_SUITE, MetricFn
from .umap_viz import fit_umap_on_training, transform_to_umap, plot_pred_test_overlay

def _ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)

def _groupby_ct_int(adata: ad.AnnData, cfg: EvalConfig):
    g = adata.obs.groupby([cfg.cell_type_key, cfg.intervention_key], observed=False)  # or True if you prefer the future default
    return {k: np.asarray(v.index) for k, v in g}

def _compute_deg_weights_vs_rest(adata_train: ad.AnnData, ct: str, itv: str, cfg: EvalConfig):
    obs = adata_train.obs
    ctl = cfg.control_label; ck = cfg.cell_type_key; ik = cfg.intervention_key
    m_pos = (obs[ck] == ct) & (obs[ik] == itv)
    m_rest = (obs[ck] == ct) & (obs[ik] != itv) & (obs[ik] != ctl)
    if m_pos.sum() < cfg.deg_min_cells or m_rest.sum() < cfg.deg_min_cells:
        return np.ones(adata_train.n_vars), np.array([], dtype=int)
    Xp = np.asarray(adata_train[m_pos].X); Xr = np.asarray(adata_train[m_rest].X)
    mp, mr = Xp.mean(axis=0), Xr.mean(axis=0)
    vp, vr = Xp.var(axis=0, ddof=1) + 1e-8, Xr.var(axis=0, ddof=1) + 1e-8
    t = (mp - mr) / np.sqrt(vp / Xp.shape[0] + vr / Xr.shape[0])
    a = np.abs(t); a -= a.min(); a = a / a.max() if a.max() > 0 else a
    w = (a ** 2); w = w / (w.mean() + 1e-12); w = np.maximum(w, cfg.deg_weight_floor)
    k = min(cfg.topk, len(a)); topk_idx = np.argsort(-a)[:k]
    return w, topk_idx

def evaluate_dataset(adata: ad.AnnData, predictor, metrics: List[MetricFn], cfg: EvalConfig):
    rng = np.random.default_rng(cfg.random_state)
    _ensure_outdir(cfg.out_dir)

    is_train = (adata.obs[cfg.set_key] == "training")
    is_test  = (adata.obs[cfg.set_key] == "test")
    adata_train = adata[is_train].copy()
    adata_test  = adata[is_test].copy()

    baselines = BaselineProvider(adata, cfg)

    reducer = idx_ref = emb_ref = None
    if cfg.make_umap_plots:
        reducer, idx_ref, emb_ref = fit_umap_on_training(adata_train, cfg, rng)

    groups_test = _groupby_ct_int(adata_test, cfg)

    # Precompute observed centroids for Systema-style diagnostics (within CT)
    obs_centroids_by_ct: Dict[str, Tuple[np.ndarray, list[str]]] = {}
    for (ct, itv), idx in groups_test.items():
        if ct not in obs_centroids_by_ct:
            gi = [(c,i) for (c,i) in groups_test.keys() if c == ct]
            C, labels = [], []
            for (c2,i2) in gi:
                Xg = np.asarray(adata_test[groups_test[(c2,i2)]].X)
                C.append(Xg.mean(axis=0)); labels.append(i2)
            obs_centroids_by_ct[ct] = (np.vstack(C), labels)

    rows = []
    umap_plots = []

    for (ct, itv), idx in groups_test.items():
        test_grp = adata_test[idx]
        # training controls of same CT
        m_ctl = (adata_train.obs[cfg.cell_type_key] == ct) & (adata_train.obs[cfg.intervention_key] == cfg.control_label)
        ctl_ct = adata_train[m_ctl]
        if ctl_ct.n_obs == 0:
            rows.append({cfg.cell_type_key: ct, cfg.intervention_key: itv, "n_test": test_grp.n_obs, "note": "no_train_controls_for_ct"})
            continue

        # predictions
        n_pred = test_grp.n_obs
        pred_grp = predictor.predict_group(ctl_ct, itv, ct, n_pred, rng)

        # Metric context
        ctx = {}
        ctx["baseline"] = baselines.mean_by_ct(ct)
        ctx["mu_all"] = baselines.mu_all(ct if cfg.mu_all_per_ct else None)

        w, topk_idx = _compute_deg_weights_vs_rest(adata_train, ct, itv, cfg)
        ctx["deg_weights"] = w; ctx["deg_idx_topk"] = topk_idx

        allC, lbls = obs_centroids_by_ct[ct]
        gi = lbls.index(itv)
        ctx["all_obs_centroids"] = allC; ctx["group_index"] = gi

        obsX = np.asarray(test_grp.X); predX = np.asarray(pred_grp.X)

        mvals = {}
        for fn in metrics:
            try: mvals.update(fn(obsX, predX, ctx))
            except Exception: mvals[fn.__name__] = np.nan

        row = {cfg.cell_type_key: ct, cfg.intervention_key: itv, "n_test": test_grp.n_obs, "n_pred": pred_grp.n_obs}
        row.update(mvals); rows.append(row)

        if cfg.make_umap_plots:
            title = f"{ct} | {itv}"
            emb_pred = transform_to_umap(reducer, predX)
            emb_test = transform_to_umap(reducer, obsX)
            emb_tr = emb_ref if emb_ref is not None else None
            out_png = os.path.join(cfg.out_dir, f"umap_pred_vs_test__{ct}__{itv}.png")
            plot_pred_test_overlay(emb_tr, emb_pred, emb_test, title, out_png, k_overlap=cfg.knn_k_overlap)
            umap_plots.append(out_png)

    summary_df = pd.DataFrame(rows).sort_values([cfg.cell_type_key, cfg.intervention_key])

    # Optional: CT-level centroid accuracy (pred centroid closest to its own test centroid)
    ca_rows = []
    for ct, (allC, lbls) in obs_centroids_by_ct.items():
        pred_centroids, present = [], []
        for itv in lbls:
            mask = (summary_df[cfg.cell_type_key]==ct) & (summary_df[cfg.intervention_key]==itv)
            if not mask.any():
                pred_centroids.append(None); present.append(False); continue
            test_idx = (adata_test.obs[cfg.cell_type_key]==ct) & (adata_test.obs[cfg.intervention_key]==itv)
            ctl_idx  = (adata_train.obs[cfg.cell_type_key]==ct) & (adata_train.obs[cfg.intervention_key]==cfg.control_label)
            if ctl_idx.sum()==0 or test_idx.sum()==0:
                pred_centroids.append(None); present.append(False); continue
            pred_grp = predictor.predict_group(adata_train[ctl_idx], itv, ct, int(test_idx.sum()), np.random.default_rng(0))
            pred_centroids.append(np.asarray(pred_grp.X).mean(axis=0)); present.append(True)
        idx_keep = [i for i,ok in enumerate(present) if ok]
        if len(idx_keep)==0: continue
        A = np.vstack([pred_centroids[i] for i in idx_keep]); B = allC[idx_keep]
        from scipy.spatial.distance import cdist
        D = cdist(A, B, metric="euclidean"); nn = D.argmin(axis=1)
        acc = float(np.mean(nn == np.arange(len(idx_keep))))
        ca_rows.append({cfg.cell_type_key: ct, "centroid_accuracy": acc, "n_groups": len(idx_keep)})
    centroid_acc_df = pd.DataFrame(ca_rows) if ca_rows else pd.DataFrame(columns=[cfg.cell_type_key, "centroid_accuracy","n_groups"])

    # Save artifacts
    _ensure_outdir(cfg.out_dir)
    summary_csv = os.path.join(cfg.out_dir, "summary_by_ct_int.csv")
    summary_df.to_csv(summary_csv, index=False)
    centroid_acc_csv = os.path.join(cfg.out_dir, "centroid_accuracy_by_ct.csv")
    centroid_acc_df.to_csv(centroid_acc_csv, index=False)
    meta = {"config": vars(cfg), "n_train": int(adata_train.n_obs), "n_test": int(adata_test.n_obs), "notes": "Confounding-aware evaluation; phenotype AUCs omitted."}
    with open(os.path.join(cfg.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return {"summary_by_ct_int": summary_df, "centroid_accuracy_by_ct": centroid_acc_df}
