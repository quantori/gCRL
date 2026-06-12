"""
Evaluation metrics for gCRL-VAE predictions vs. ground truth.

This module provides functions to compute quantitative metrics comparing predicted
intervention effects against actual test data.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import anndata as ad


def evaluate_predictions(
    adata: ad.AnnData,
    preds: ad.AnnData,
    output_dir: str | Path,
    set_key: str = "set",
    intervention_key: str = "intervention",
    cell_type_key: str = "cell_type",
    control_labels: tuple[str, ...] = ("control", "unperturbed"),
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Compute quantitative metrics comparing predicted and actual intervention effects.

    Computes centroid distances in **gene expression space** (not UMAP space) to match
    the training objective where centroid loss is MSE between mean gene expressions.

    Parameters
    ----------
    adata : ad.AnnData
        Original AnnData object containing training controls and test data.
        Must have obs columns: set_key, intervention_key, cell_type_key
    preds : ad.AnnData
        Predictions from model.predict(). Should have obs['set'] == 'prediction'
    output_dir : str | Path
        Directory where metrics will be saved
    set_key : str, default='set'
        Column in adata.obs indicating train/test split
    intervention_key : str, default='intervention'
        Column in adata.obs and preds.obs indicating intervention type
    cell_type_key : str, default='cell_type'
        Column in adata.obs and preds.obs indicating cell type
    control_labels : tuple[str, ...], default=('control', 'unperturbed')
        Labels identifying control/unperturbed cells
    random_seed : int, default=42
        Random seed for splitting test cells in half (for perfect prediction baseline)

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - cell_type: Cell type
        - intervention: Intervention name
        - centroid_distance: Euclidean distance between actual and predicted centroids
        - baseline_worst_case: Distance between actual perturbed and control centroids (no prediction)
        - baseline_perfect: Distance between two random halves of actual test cells (perfect prediction)
        - n_test_cells: Number of test cells
        - n_pred_cells: Number of predicted cells

    Examples
    --------
    >>> # After training and generating predictions
    >>> model, history = train_gcrl_vae(adata, cfg)
    >>> preds = model.predict(adata, seed=42)
    >>>
    >>> # Compute evaluation metrics
    >>> metrics_df = evaluate_predictions(
    ...     adata=adata,
    ...     preds=preds,
    ...     output_dir="results/metrics"
    ... )
    >>> print(metrics_df.sort_values('centroid_distance').head(10))
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("gCRL-VAE Prediction Evaluation")
    print("=" * 70)

    # Validate inputs
    _validate_inputs(adata, preds, set_key, intervention_key, cell_type_key)

    # Extract test data
    test_data = adata[adata.obs[set_key] == "test"].copy()

    # Extract training controls for worst-case baseline
    train_controls = adata[
        (adata.obs[set_key] == "training") &
        (adata.obs[intervention_key].isin(control_labels))
    ].copy()

    print(f"\n📊 Data Summary:")
    print(f"  Test data: {test_data.n_obs:,} cells")
    print(f"  Predictions: {preds.n_obs:,} cells")
    print(f"  Training controls: {train_controls.n_obs:,} cells")
    print(f"  Cell types: {sorted(test_data.obs[cell_type_key].unique())}")
    print(f"  Test interventions: {test_data.obs[intervention_key].nunique()}")

    # Compute metrics for each (cell_type, intervention) pair
    print("\n📈 Computing metrics...")
    print("  - centroid_distance: Model prediction vs. actual")
    print("  - baseline_worst_case: Control centroid vs. actual (no prediction)")
    print("  - baseline_perfect: Random split of actual cells (perfect prediction)")
    metrics_list = []

    # Set random seed for reproducibility
    rng = np.random.RandomState(random_seed)

    # Group test data by (cell_type, intervention)
    test_groups = test_data.obs.groupby([cell_type_key, intervention_key])

    for (cell_type, intervention), group_df in test_groups:
        # Get indices for this group
        test_idx = group_df.index
        test_mask = test_data.obs.index.isin(test_idx)

        # Get corresponding predictions
        pred_mask = (preds.obs[cell_type_key] == cell_type) & (
            preds.obs[intervention_key] == intervention
        )

        if pred_mask.sum() == 0:
            print(f"   ⚠️  No predictions found for ({cell_type}, {intervention})")
            continue

        # Extract gene expression for centroid distance (in gene space)
        test_coords_expr = np.asarray(test_data[test_mask].X)
        pred_coords_expr = np.asarray(preds[pred_mask].X)

        # 1. Calculate centroid distance in GENE EXPRESSION SPACE (model prediction)
        test_centroid_expr = test_coords_expr.mean(axis=0)
        pred_centroid_expr = pred_coords_expr.mean(axis=0)
        centroid_dist = np.linalg.norm(test_centroid_expr - pred_centroid_expr)

        # 2. BASELINE WORST CASE: Distance between test and control centroids
        # This simulates predicting no change (control = perturbed)
        # Get control cells for this cell type
        control_mask = train_controls.obs[cell_type_key] == cell_type
        if control_mask.sum() == 0:
            print(f"   ⚠️  No control cells found for cell type {cell_type}")
            baseline_worst = np.nan
        else:
            control_coords_expr = np.asarray(train_controls[control_mask].X)
            control_centroid_expr = control_coords_expr.mean(axis=0)
            baseline_worst = np.linalg.norm(test_centroid_expr - control_centroid_expr)

        # 3. BASELINE PERFECT: Distance between two random halves of test cells
        # This simulates perfect prediction (comparing two samples from same distribution)
        n_test = len(test_coords_expr)
        if n_test < 2:
            # Not enough cells to split
            baseline_perfect = np.nan
        else:
            # Randomly shuffle and split in half
            shuffled_indices = rng.permutation(n_test)
            split_point = n_test // 2
            half1_indices = shuffled_indices[:split_point]
            half2_indices = shuffled_indices[split_point:]

            # Compute centroids of each half
            half1_centroid = test_coords_expr[half1_indices].mean(axis=0)
            half2_centroid = test_coords_expr[half2_indices].mean(axis=0)
            baseline_perfect = np.linalg.norm(half1_centroid - half2_centroid)

        # Store metrics
        metrics_list.append({
            "cell_type": cell_type,
            "intervention": intervention,
            "centroid_distance": centroid_dist,
            "baseline_worst_case": baseline_worst,
            "baseline_perfect": baseline_perfect,
            "n_test_cells": len(test_coords_expr),
            "n_pred_cells": len(pred_coords_expr),
        })

    # Create DataFrame
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df = metrics_df.sort_values("centroid_distance")

    # Save overall metrics
    overall_path = output_dir / "prediction_metrics.csv"
    metrics_df.to_csv(overall_path, index=False)
    print(f"\n💾 Overall metrics saved to: {overall_path}")

    # Save per-cell-type metrics
    print("\n💾 Per-cell-type metrics:")
    for cell_type in sorted(metrics_df["cell_type"].unique()):
        ct_metrics = metrics_df[metrics_df["cell_type"] == cell_type]
        safe_filename = cell_type.replace("+", "_").replace("/", "_")
        ct_path = output_dir / f"{safe_filename}_metrics.csv"
        ct_metrics.to_csv(ct_path, index=False)
        print(f"   {cell_type}: {ct_path.name}")

    # Print summary statistics
    print("\n📊 Summary Statistics:")
    print(f"  Total interventions evaluated: {len(metrics_df)}")
    print(f"\n  Model Prediction (centroid_distance):")
    print(f"    Mean: {metrics_df['centroid_distance'].mean():.4f}")
    print(f"    Median: {metrics_df['centroid_distance'].median():.4f}")
    print(f"    Min: {metrics_df['centroid_distance'].min():.4f}")
    print(f"    Max: {metrics_df['centroid_distance'].max():.4f}")

    print(f"\n  Worst Case Baseline (control as prediction):")
    print(f"    Mean: {metrics_df['baseline_worst_case'].mean():.4f}")
    print(f"    Median: {metrics_df['baseline_worst_case'].median():.4f}")
    print(f"    Min: {metrics_df['baseline_worst_case'].min():.4f}")
    print(f"    Max: {metrics_df['baseline_worst_case'].max():.4f}")

    print(f"\n  Perfect Baseline (random split of actual):")
    print(f"    Mean: {metrics_df['baseline_perfect'].mean():.4f}")
    print(f"    Median: {metrics_df['baseline_perfect'].median():.4f}")
    print(f"    Min: {metrics_df['baseline_perfect'].min():.4f}")
    print(f"    Max: {metrics_df['baseline_perfect'].max():.4f}")

    # Compute relative performance
    print(f"\n  Relative Performance:")
    # How much better than worst case (lower is better)
    improvement_over_worst = (
        (metrics_df['baseline_worst_case'] - metrics_df['centroid_distance']) /
        metrics_df['baseline_worst_case'] * 100
    )
    print(f"    Improvement over worst case: {improvement_over_worst.mean():.1f}% (mean)")

    # How much worse than perfect (should be close to 0)
    gap_to_perfect = (
        (metrics_df['centroid_distance'] - metrics_df['baseline_perfect']) /
        metrics_df['baseline_perfect'] * 100
    )
    print(f"    Gap to perfect baseline: {gap_to_perfect.mean():.1f}% (mean)")

    print("\n✅ Evaluation complete!")
    print(f"   Metrics saved to: {output_dir.absolute()}")
    print("=" * 70)

    return metrics_df


def _validate_inputs(
    adata: ad.AnnData,
    preds: ad.AnnData,
    set_key: str,
    intervention_key: str,
    cell_type_key: str,
) -> None:
    """Validate that required columns exist in data."""

    # Check adata
    for key in [set_key, intervention_key, cell_type_key]:
        if key not in adata.obs.columns:
            raise ValueError(f"Column '{key}' not found in adata.obs")

    # Check preds
    for key in [intervention_key, cell_type_key]:
        if key not in preds.obs.columns:
            raise ValueError(f"Column '{key}' not found in preds.obs")

    # Check dimensions match
    if adata.n_vars != preds.n_vars:
        raise ValueError(
            f"Number of variables mismatch: adata has {adata.n_vars}, "
            f"preds has {preds.n_vars}"
        )
