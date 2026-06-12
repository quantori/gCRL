from __future__ import annotations
import numpy as np
from typing import Callable
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import cdist

MetricFn = Callable[[np.ndarray, np.ndarray, dict], dict]

def _safe_mean(X):
    X = np.asarray(X)
    return X.mean(axis=0) if X.ndim == 2 else X

def rmse(obs, pred, ctx):
    e = _safe_mean((pred - obs) ** 2)
    return {'rmse': float(np.sqrt(e.mean()))}

def rmse_topk_degs(obs, pred, ctx):
    idx = ctx.get('deg_idx_topk')
    if idx is None or len(idx) == 0:
        return {'rmse_topk_degs': np.nan}
    e = _safe_mean((pred[:, idx] - obs[:, idx]) ** 2)
    return {'rmse_topk_degs': float(np.sqrt(e.mean()))}

def deg_restricted_corr(obs, pred, ctx):
    idx = ctx.get('deg_idx_topk')
    if idx is None or len(idx) == 0:
        return {'pearson_deg': np.nan, 'spearman_deg': np.nan}
    o = _safe_mean(obs); p = _safe_mean(pred)
    return {
        'pearson_deg': float(pearsonr(o[idx], p[idx])[0]),
        'spearman_deg': float(spearmanr(o[idx], p[idx]).correlation),
    }

def cosine_centroid(obs, pred, ctx):
    o = _safe_mean(obs); p = _safe_mean(pred)
    return {'cosine_centroid': float((o @ p) / (np.linalg.norm(o) * np.linalg.norm(p) + 1e-12))}

def diversity_weighted_mse(obs, pred, ctx):
    w = ctx.get('deg_weights')
    if w is None: return {'wmse': np.nan}
    e = ((pred - obs) ** 2).mean(axis=0) * w
    return {'wmse': float(e.mean())}

def diversity_weighted_r2_delta(obs, pred, ctx):
    mu = ctx.get('mu_all'); w = ctx.get('deg_weights')
    if mu is None or w is None: return {'r2w_delta': np.nan}
    o = _safe_mean(obs) - mu; p = _safe_mean(pred) - mu; w = w / (w.mean() + 1e-12)
    sse = float(((o - p) ** 2 * w).sum()); sst = float(((o - 0.0) ** 2 * w).sum())
    return {'r2w_delta': (1.0 - sse / sst) if sst > 1e-12 else np.nan}

def centroid_distance_ratio(obs, pred, ctx):
    allC = ctx.get('all_obs_centroids'); gi = ctx.get('group_index')
    if allC is None or gi is None: return {'centroid_distance_ratio': np.nan}
    pC = _safe_mean(pred)[None, :]
    d = cdist(pC, allC, metric='euclidean')[0]
    own = d[gi]; other = np.delete(d, gi)
    return {'centroid_distance_ratio': float(own / (other.min() + 1e-12)) if other.size else np.nan}

DEFAULT_METRICS_SUITE = [
    rmse,
    rmse_topk_degs,
    deg_restricted_corr,
    cosine_centroid,
    diversity_weighted_mse,
    diversity_weighted_r2_delta,
    centroid_distance_ratio,
]
