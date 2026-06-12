"""
Evaluation metrics for gCRL-VAE predictions vs. ground truth.
"""

from __future__ import annotations
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import anndata as ad


# ---------------------------------------------------------------------------
# Metric functions
# Each accepts two AnnData objects and an options dict, returns a float.
# Convention: adata_a is the "predicted/proxy" side, adata_b is the ground truth.
# ---------------------------------------------------------------------------

def _to_dense(X) -> np.ndarray:
    import scipy.sparse as sp
    return X.toarray() if sp.issparse(X) else np.asarray(X)


def centroid_distance(adata_a: ad.AnnData, adata_b: ad.AnnData, options: dict) -> float:
    """Euclidean distance between the centroids of two AnnData objects in gene space."""
    centroid_a = _to_dense(adata_a.X).mean(axis=0)
    centroid_b = _to_dense(adata_b.X).mean(axis=0)
    return float(np.linalg.norm(centroid_a - centroid_b))


def rmse(adata_a: ad.AnnData, adata_b: ad.AnnData, options: dict) -> float:
    """RMSE between the centroids of two AnnData objects in gene space."""
    centroid_a = _to_dense(adata_a.X).mean(axis=0)
    centroid_b = _to_dense(adata_b.X).mean(axis=0)
    return float(np.sqrt(np.mean((centroid_a - centroid_b) ** 2)))


AVAILABLE_METRICS: dict[str, Callable] = {
    "centroid_distance": centroid_distance,
    "rmse": rmse,
}

DEFAULT_METRICS = list(AVAILABLE_METRICS.keys())


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def evaluate_predictions(
    adata: ad.AnnData,
    preds: ad.AnnData,
    output_dir: str | Path,
    metrics: list[str] | None = None,
    metric_options: dict | None = None,
    set_key: str = "set",
    intervention_key: str = "intervention",
    cell_type_key: str = "cell_type",
    control_labels: tuple[str, ...] = ("control", "unperturbed"),
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Compute quantitative metrics comparing predicted and actual intervention effects.

    Parameters
    ----------
    adata : ad.AnnData
        Original AnnData with obs columns: set_key, intervention_key, cell_type_key.
    preds : ad.AnnData
        Predictions from model.predict(). Should have obs['set'] == 'prediction'.
    output_dir : str | Path
        Directory where metrics CSV will be saved.
    metrics : list[str] | None
        Names of metrics to compute. Defaults to all available metrics.
        Available: 'centroid_distance', 'rmse'.
    metric_options : dict | None
        Options dict passed to each metric function.
    set_key : str
        Column in adata.obs indicating train/test split.
    intervention_key : str
        Column in adata.obs and preds.obs indicating intervention type.
    cell_type_key : str
        Column in adata.obs and preds.obs indicating cell type.
    control_labels : tuple[str, ...]
        Labels identifying control/unperturbed cells.
    random_seed : int
        Random seed for the perfect baseline random split.

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with columns:
        cell_type, intervention, method, metric_name, metric_value,
        n_test_cells, n_pred_cells.
        method is one of: 'actual', 'worst_case', 'perfect_baseline'.
    """
    if metrics is None:
        metrics = DEFAULT_METRICS
    if metric_options is None:
        metric_options = {}

    unknown = set(metrics) - set(AVAILABLE_METRICS)
    if unknown:
        raise ValueError(f"Unknown metrics: {unknown}. Available: {set(AVAILABLE_METRICS)}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _validate_inputs(adata, preds, set_key, intervention_key, cell_type_key)

    test_data = adata[adata.obs[set_key] == "test"].copy()
    train_controls = adata[
        (adata.obs[set_key] == "training") &
        (adata.obs[intervention_key].isin(control_labels))
    ].copy()

    rng = np.random.RandomState(random_seed)
    rows = []

    for (cell_type, intervention), group_df in test_data.obs.groupby([cell_type_key, intervention_key]):
        test_mask = test_data.obs.index.isin(group_df.index)
        test_adata = test_data[test_mask]

        pred_mask = (
            (preds.obs[cell_type_key] == cell_type) &
            (preds.obs[intervention_key] == intervention)
        )
        if pred_mask.sum() == 0:
            continue

        pred_adata = preds[pred_mask]
        n_test = test_adata.n_obs
        n_pred = pred_adata.n_obs

        # Build the three comparison pairs
        control_adata = train_controls[train_controls.obs[cell_type_key] == cell_type]

        # Perfect baseline: split test cells in half
        shuffled = rng.permutation(n_test)
        split = n_test // 2
        half1 = test_adata[shuffled[:split]]
        half2 = test_adata[shuffled[split:]]

        method_pairs = [
            ("actual",           pred_adata,    test_adata),
            ("worst_case",       control_adata, test_adata),
            ("perfect_baseline", half1,         half2),
        ]

        for method_name, adata_a, adata_b in method_pairs:
            if adata_a.n_obs == 0:
                continue
            for metric_name in metrics:
                metric_fn = AVAILABLE_METRICS[metric_name]
                value = metric_fn(adata_a, adata_b, metric_options)
                rows.append({
                    "cell_type":    cell_type,
                    "intervention": intervention,
                    "method":       method_name,
                    "metric_name":  metric_name,
                    "metric_value": value,
                    "n_test_cells": n_test,
                    "n_pred_cells": n_pred,
                })

    results_df = pd.DataFrame(rows, columns=[
        "cell_type", "intervention", "method", "metric_name", "metric_value",
        "n_test_cells", "n_pred_cells",
    ])

    results_df.to_csv(output_dir / "prediction_metrics.csv", index=False)
    return results_df


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

def _validate_inputs(
    adata: ad.AnnData,
    preds: ad.AnnData,
    set_key: str,
    intervention_key: str,
    cell_type_key: str,
) -> None:
    for key in [set_key, intervention_key, cell_type_key]:
        if key not in adata.obs.columns:
            raise ValueError(f"Column '{key}' not found in adata.obs")
    for key in [intervention_key, cell_type_key]:
        if key not in preds.obs.columns:
            raise ValueError(f"Column '{key}' not found in preds.obs")
    if adata.n_vars != preds.n_vars:
        raise ValueError(
            f"Number of variables mismatch: adata has {adata.n_vars}, preds has {preds.n_vars}"
        )
