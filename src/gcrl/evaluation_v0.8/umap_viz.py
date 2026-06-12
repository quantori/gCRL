from __future__ import annotations
import numpy as np
import anndata as ad
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import umap
from matplotlib.patches import Ellipse
from .config import EvalConfig

def fit_umap_on_training(adata_train: ad.AnnData, cfg: EvalConfig, rng: np.random.Generator):
    n = adata_train.n_obs
    idx = np.arange(n)
    if n > cfg.umap_sample_train:
        idx = rng.choice(idx, size=cfg.umap_sample_train, replace=False)
    X = np.asarray(adata_train.X)[idx]
    reducer = umap.UMAP(n_neighbors=cfg.umap_neighbors, min_dist=cfg.umap_min_dist, metric=cfg.umap_metric, random_state=cfg.random_state)
    emb = reducer.fit_transform(X)
    return reducer, idx, emb

def transform_to_umap(reducer, X):
    return reducer.transform(np.asarray(X))

def _centroid_and_cov2d(P):
    mu = P.mean(axis=0); C = np.cov(P.T); return mu, C

def _ellipse_params(C):
    vals, vecs = np.linalg.eigh(C); order = np.argsort(vals)[::-1]
    vals = vals[order]; vecs = vecs[:, order]
    width, height = 2 * np.sqrt(vals + 1e-9); angle = np.degrees(np.arctan2(vecs[1,0], vecs[0,0])); return width, height, angle

def _plot_group(ax, P, label, alpha=0.2):
    ax.scatter(P[:,0], P[:,1], s=5, alpha=alpha, label=label)
    mu, C = _centroid_and_cov2d(P); w, h, a = _ellipse_params(C)
    ax.scatter([mu[0]],[mu[1]], s=60, marker='X'); ax.add_patch(Ellipse(mu, w, h, angle=a, fill=False))
    return mu

def knn_overlap(A, B, k):
    allP = np.vstack([A, B]); labels = np.array([0]*len(A) + [1]*len(B))
    nn = NearestNeighbors(n_neighbors=min(k+1, len(allP))).fit(allP)
    idxs = nn.kneighbors(A, return_distance=False)
    return float((labels[idxs[:,1:]] == 1).any(axis=1).mean())

def plot_pred_test_overlay(emb_train, emb_pred, emb_test, title, out_path, k_overlap=10):
    fig, ax = plt.subplots(figsize=(6,5))
    if emb_train is not None and len(emb_train):
        _plot_group(ax, emb_train, 'train (ref)', alpha=0.05)
    mu_p = _plot_group(ax, emb_pred, 'pred', alpha=0.25)
    mu_t = _plot_group(ax, emb_test, 'test', alpha=0.25)
    ax.plot([mu_p[0], mu_t[0]], [mu_p[1], mu_t[1]], 'k--', lw=1)
    ko = knn_overlap(emb_pred, emb_test, k=k_overlap)
    ax.set_title(f"{title}\ncentroid_dist={np.linalg.norm(mu_p-mu_t):.3f}  kNN_overlap@{k_overlap}={ko:.2f}")
    ax.legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig(out_path, dpi=200); plt.close(fig)
