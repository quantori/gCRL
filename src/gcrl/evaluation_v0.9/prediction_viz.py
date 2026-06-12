"""
UMAP visualization of gCRL-VAE predictions vs. ground truth.

This module provides functions to visualize predicted vs. actual intervention effects
using UMAP embeddings, with support for both individual intervention plots and
per-cell-type aggregated views.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Optional, Literal

import numpy as np
import anndata as ad
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
import scanpy as sc
from sklearn.decomposition import PCA
import umap


def visualize_predictions(
    adata: ad.AnnData,
    preds: ad.AnnData,
    output_dir: str | Path,
    set_key: str = "set",
    intervention_key: str = "intervention",
    cell_type_key: str = "cell_type",
    control_labels: tuple[str, ...] = ("control", "unperturbed"),
    umap_fit_on: Literal["controls", "all"] = "controls",
    n_pcs: int = 30,
    n_pcs_max=100,
    target_variance: float = 0.80,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.3,
    umap_metric: str = "euclidean",
    random_state: int = 42,
    figsize_single: tuple[float, float] = (8, 7),
    figsize_celltype: tuple[float, float] = (10, 8),
    dpi: int = 150,
    show_legend: bool = True,
) -> None:
    """
    Create UMAP visualizations comparing predicted and actual intervention effects.

    Generates two types of plots:
    1. Per-intervention plots: One UMAP per (cell_type, intervention) pair
    2. Per-cell-type plots: One UMAP per cell_type showing all interventions

    **Note**: This function focuses on visualization only. For quantitative metrics
    (e.g., centroid distances), use the `evaluate_predictions()` function from
    `gcrl.evaluation.prediction_eval`.

    Parameters
    ----------
    adata : ad.AnnData
        Original AnnData object containing training controls and test data.
        Must have obs columns: set_key, intervention_key, cell_type_key
    preds : ad.AnnData
        Predictions from model.predict(). Should have obs['set'] == 'prediction'
    output_dir : str | Path
        Directory where plots will be saved
    set_key : str, default='set'
        Column in adata.obs indicating train/test split
    intervention_key : str, default='intervention'
        Column in adata.obs and preds.obs indicating intervention type
    cell_type_key : str, default='cell_type'
        Column in adata.obs and preds.obs indicating cell type
    control_labels : tuple[str, ...], default=('control', 'unperturbed')
        Labels identifying control/unperturbed cells
    umap_fit_on : {'controls', 'all'}, default='controls'
        Strategy for fitting UMAP:
        - 'controls': Fit on training controls only, then project test/preds
          (consistent baseline, avoids data leakage)
        - 'all': Fit on all cells (controls + test + predictions)
          (better global structure, may show relationships between conditions)
    n_pcs : int, default=30
        Number of principal components for PCA preprocessing
    umap_n_neighbors : int, default=15
        UMAP n_neighbors parameter
    umap_min_dist : float, default=0.3
        UMAP min_dist parameter
    umap_metric : str, default='euclidean'
        Distance metric for UMAP
    random_state : int, default=42
        Random seed for reproducibility
    figsize_single : tuple[float, float], default=(8, 7)
        Figure size for individual intervention plots
    figsize_celltype : tuple[float, float], default=(10, 8)
        Figure size for cell-type aggregate plots
    dpi : int, default=150
        Resolution for saved figures
    show_legend : bool, default=True
        Whether to show legend on plots

    Returns
    -------
    None
        Saves plots to output_dir

    Examples
    --------
    >>> # After training and generating predictions
    >>> model, history = train_gcrl_vae(adata, cfg)
    >>> preds = model.predict(adata, seed=42)
    >>>
    >>> # Create visualizations
    >>> visualize_predictions(
    ...     adata=adata,
    ...     preds=preds,
    ...     output_dir="results/predictions_viz"
    ... )
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("gCRL-VAE Prediction Visualization")
    print("=" * 70)

    # Validate inputs
    _validate_inputs(adata, preds, set_key, intervention_key, cell_type_key)

    # Extract training controls and test data
    train_controls = adata[
        (adata.obs[set_key] == "training")
        & (adata.obs[intervention_key].isin(control_labels))
    ].copy()

    test_data = adata[adata.obs[set_key] == "test"].copy()

    print(f"\n📊 Data Summary:")
    print(f"  Training controls: {train_controls.n_obs:,} cells")
    print(f"  Test data: {test_data.n_obs:,} cells")
    print(f"  Predictions: {preds.n_obs:,} cells")
    print(f"  Cell types: {sorted(test_data.obs[cell_type_key].unique())}")
    print(f"  Test interventions: {test_data.obs[intervention_key].nunique()}")

    # Fit UMAP based on strategy
    if umap_fit_on == "controls":
        print("\n🔧 Fitting UMAP on training controls only...")
        umap_model, controls_umap = _fit_umap_on_data(
            train_controls,
            n_pcs=n_pcs,
            target_variance=target_variance,
            n_pcs_max=n_pcs_max,
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
            metric=umap_metric,
            random_state=random_state,
        )

        # Transform test and predictions
        print("🔄 Projecting test and predictions to UMAP space...")
        test_umap = umap_model.transform(np.asarray(test_data.X))
        preds_umap = umap_model.transform(np.asarray(preds.X))

    elif umap_fit_on == "all":
        print("\n🔧 Fitting UMAP on all cells (controls + test + predictions)...")
        # Concatenate all data for fitting
        all_data = ad.concat([train_controls, test_data, preds], axis=0)

        umap_model, all_umap = _fit_umap_on_data(
            all_data,
            n_pcs=n_pcs,
            target_variance=target_variance,
            n_pcs_max=n_pcs_max,
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
            metric=umap_metric,
            random_state=random_state,
        )

        # Split back the UMAP coordinates
        n_controls = train_controls.n_obs
        n_test = test_data.n_obs
        controls_umap = all_umap[:n_controls]
        test_umap = all_umap[n_controls:n_controls + n_test]
        preds_umap = all_umap[n_controls + n_test:]

    else:
        raise ValueError(f"umap_fit_on must be 'controls' or 'all', got '{umap_fit_on}'")

    # Create per-intervention plots
    print("\n📈 Creating per-intervention plots...")
    _create_per_intervention_plots(
        test_data=test_data,
        preds=preds,
        controls_umap=controls_umap,
        test_umap=test_umap,
        preds_umap=preds_umap,
        output_dir=output_dir / "per_intervention",
        intervention_key=intervention_key,
        cell_type_key=cell_type_key,
        figsize=figsize_single,
        dpi=dpi,
        show_legend=show_legend,
    )

    # Create per-cell-type plots
    print("\n📈 Creating per-cell-type plots...")
    _create_per_celltype_plots(
        test_data=test_data,
        preds=preds,
        controls_umap=controls_umap,
        test_umap=test_umap,
        preds_umap=preds_umap,
        output_dir=output_dir / "per_celltype",
        intervention_key=intervention_key,
        cell_type_key=cell_type_key,
        figsize=figsize_celltype,
        dpi=dpi,
        show_legend=show_legend,
    )

    print("\n✅ Visualization complete!")
    print(f"   Plots saved to: {output_dir.absolute()}")
    print("=" * 70)


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


def _fit_umap_on_data(
    adata: ad.AnnData,
    n_pcs: int,
    target_variance: float,
    n_pcs_max: float,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    random_state: int,
) -> tuple[umap.UMAP, np.ndarray]:
    """Fit UMAP model on provided data and return model + embeddings.

    Returns
    -------
    tuple[umap.UMAP, np.ndarray]
        (umap_model, umap_embeddings)
    """

    X = np.asarray(adata.X)

    # PCA preprocessing (optional but recommended for high-dimensional data)
    if n_pcs > 0 and n_pcs < X.shape[1]:
        # First fit with specified n_pcs to check variance explained
        pca_initial = PCA(n_components=n_pcs, random_state=random_state)
        pca_initial.fit(X)

        # Check if we need more PCs to reach required_variance% variance
        cumsum_var = np.cumsum(pca_initial.explained_variance_ratio_)

        if cumsum_var[-1] < target_variance:
            # Need more PCs - find how many are needed for 80%
            pca_full = PCA(random_state=random_state)
            pca_full.fit(X)
            cumsum_var_full = np.cumsum(pca_full.explained_variance_ratio_)
            n_pcs_needed = np.searchsorted(cumsum_var_full, target_variance) + 1
            n_pcs_needed = min(n_pcs_needed, X.shape[1])

            if n_pcs_needed > n_pcs_max:
                n_pcs_needed = n_pcs_max

            pca = PCA(n_components=n_pcs_needed, random_state=random_state)
            X_pca = pca.fit_transform(X)

        else:

            # Requested n_pcs is sufficient
            X_pca = pca_initial.transform(X)

        print(
            f"   PCA: {X.shape[1]} genes → {X_pca.shape[1]} PCs "
            f"(explained variance: {pca_initial.explained_variance_ratio_.sum():.2%})"
        )

    else:
        X_pca = X

    # Fit UMAP and get embeddings
    umap_model = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    umap_embeddings = umap_model.fit_transform(X_pca)

    return umap_model, umap_embeddings


def _create_per_intervention_plots(
    test_data: ad.AnnData,
    preds: ad.AnnData,
    controls_umap: np.ndarray,
    test_umap: np.ndarray,
    preds_umap: np.ndarray,
    output_dir: Path,
    intervention_key: str,
    cell_type_key: str,
    figsize: tuple[float, float],
    dpi: int,
    show_legend: bool,
) -> None:
    """Create one UMAP plot per (cell_type, intervention) pair."""

    output_dir.mkdir(parents=True, exist_ok=True)

    # Group test data by (cell_type, intervention)
    test_groups = test_data.obs.groupby([cell_type_key, intervention_key])

    n_plots = len(test_groups)
    print(f"   Generating {n_plots} plots...")

    for i, ((cell_type, intervention), group_df) in enumerate(test_groups, 1):
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

        # Extract UMAP coordinates
        test_coords_umap = test_umap[test_mask]
        pred_coords_umap = preds_umap[pred_mask]

        # Create plot
        fig, ax = plt.subplots(figsize=figsize)

        # Plot unperturbed cells (background)
        ax.scatter(
            controls_umap[:, 0],
            controls_umap[:, 1],
            c="lightgray",
            s=5,
            alpha=0.3,
            label="Unperturbed (training)",
            rasterized=True,
        )

        # Plot actual test cells
        ax.scatter(
            test_coords_umap[:, 0],
            test_coords_umap[:, 1],
            c="#2E86AB",  # Blue
            s=30,
            alpha=0.6,
            label=f"Actual ({len(test_coords_umap)} cells)",
            edgecolors="white",
            linewidths=0.5,
        )

        # Plot predicted cells
        ax.scatter(
            pred_coords_umap[:, 0],
            pred_coords_umap[:, 1],
            c="#A23B72",  # Purple
            s=30,
            alpha=0.6,
            label=f"Predicted ({len(pred_coords_umap)} cells)",
            edgecolors="white",
            linewidths=0.5,
        )

        # Add centroids (UMAP space for visualization)
        test_centroid_umap = test_coords_umap.mean(axis=0)
        pred_centroid_umap = pred_coords_umap.mean(axis=0)

        ax.scatter(
            test_centroid_umap[0],
            test_centroid_umap[1],
            c="#2E86AB",
            s=100,
            marker="X",
            edgecolors="black",
            linewidths=1.5,
            label="Actual centroid",
            zorder=10,
        )

        ax.scatter(
            pred_centroid_umap[0],
            pred_centroid_umap[1],
            c="#A23B72",
            s=100,
            marker="X",
            edgecolors="black",
            linewidths=1.5,
            label="Predicted centroid",
            zorder=10,
        )

        # Draw line between centroids
        ax.plot(
            [test_centroid_umap[0], pred_centroid_umap[0]],
            [test_centroid_umap[1], pred_centroid_umap[1]],
            "k--",
            linewidth=1.5,
            alpha=0.7,
            zorder=5,
        )

        # Add ellipses for distribution spread
        _add_ellipse(ax, test_coords_umap, color="#2E86AB", alpha=0.2)
        _add_ellipse(ax, pred_coords_umap, color="#A23B72", alpha=0.2)

        # Title and labels
        title = f"{cell_type} | {intervention}"
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("UMAP 1", fontsize=10)
        ax.set_ylabel("UMAP 2", fontsize=10)

        if show_legend:
            ax.legend(loc="best", frameon=True, fontsize=8, framealpha=0.9)

        ax.set_aspect("equal", "box")
        ax.grid(False)

        # Save
        safe_filename = f"{cell_type}_{intervention}".replace("+", "_").replace(
            "/", "_"
        )
        fig.tight_layout()
        fig.savefig(output_dir / f"{safe_filename}.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        if i % 10 == 0:
            print(f"   Progress: {i}/{n_plots} plots generated")

    print(f"   ✓ {n_plots} plots saved to {output_dir.name}/")


def _create_per_celltype_plots(
    test_data: ad.AnnData,
    preds: ad.AnnData,
    controls_umap: np.ndarray,
    test_umap: np.ndarray,
    preds_umap: np.ndarray,
    output_dir: Path,
    intervention_key: str,
    cell_type_key: str,
    figsize: tuple[float, float],
    dpi: int,
    show_legend: bool,
) -> None:
    """Create one UMAP plot per cell_type showing all interventions."""

    output_dir.mkdir(parents=True, exist_ok=True)

    cell_types = sorted(test_data.obs[cell_type_key].unique())

    print(f"   Generating {len(cell_types)} plots...")

    for cell_type in cell_types:
        # Get all interventions for this cell type
        ct_test_mask = test_data.obs[cell_type_key] == cell_type
        ct_pred_mask = preds.obs[cell_type_key] == cell_type

        if ct_test_mask.sum() == 0:
            continue

        # Get unique interventions for this cell type
        interventions = sorted(test_data.obs[ct_test_mask][intervention_key].unique())

        # Create plot
        fig, ax = plt.subplots(figsize=figsize)

        # Plot unperturbed cells (background)
        ax.scatter(
            controls_umap[:, 0],
            controls_umap[:, 1],
            c="lightgray",
            s=5,
            alpha=0.3,
            label="Unperturbed",
            rasterized=True,
        )

        # Use a color palette for interventions
        colors = plt.cm.tab20(np.linspace(0, 1, len(interventions)))

        # Plot each intervention
        for intervention, color in zip(interventions, colors):
            # Test data
            test_mask = ct_test_mask & (test_data.obs[intervention_key] == intervention)
            test_coords_umap = test_umap[test_mask]

            # Predictions
            pred_mask = ct_pred_mask & (preds.obs[intervention_key] == intervention)
            pred_coords_umap = preds_umap[pred_mask]

            if len(test_coords_umap) == 0 or len(pred_coords_umap) == 0:
                continue

            # Plot cells
            ax.scatter(
                test_coords_umap[:, 0],
                test_coords_umap[:, 1],
                c=[color],
                s=20,
                alpha=0.4,
                edgecolors="white",
                linewidths=0.3,
            )

            ax.scatter(
                pred_coords_umap[:, 0],
                pred_coords_umap[:, 1],
                c=[color],
                s=20,
                alpha=0.4,
                marker="s",
                edgecolors="white",
                linewidths=0.3,
            )

            # Centroids (UMAP space for visualization)
            test_centroid_umap = test_coords_umap.mean(axis=0)
            pred_centroid_umap = pred_coords_umap.mean(axis=0)

            ax.scatter(
                test_centroid_umap[0],
                test_centroid_umap[1],
                c=[color],
                s=80,
                marker="o",
                edgecolors="black",
                linewidths=1.5,
                zorder=10,
            )

            ax.scatter(
                pred_centroid_umap[0],
                pred_centroid_umap[1],
                c=[color],
                s=80,
                marker="s",
                edgecolors="black",
                linewidths=1.5,
                zorder=10,
            )

            # Line between centroids
            ax.plot(
                [test_centroid_umap[0], pred_centroid_umap[0]],
                [test_centroid_umap[1], pred_centroid_umap[1]],
                color=color,
                linestyle="--",
                linewidth=1.0,
                alpha=0.5,
                zorder=5,
            )

        # Title
        title = f"Cell Type: {cell_type}"
        subtitle = f"{len(interventions)} interventions"
        ax.set_title(f"{title}\n{subtitle}", fontsize=14, fontweight="bold")
        ax.set_xlabel("UMAP 1", fontsize=10)
        ax.set_ylabel("UMAP 2", fontsize=10)

        # Custom legend
        if show_legend:
            # Create legend elements
            legend_elements = [
                mpatches.Patch(color="lightgray", label="Unperturbed (background)"),
                mpatches.Patch(color="gray", label="○ = Actual centroid"),
                mpatches.Patch(color="gray", label="□ = Predicted centroid"),
            ]
            ax.legend(
                handles=legend_elements,
                loc="upper right",
                frameon=True,
                fontsize=8,
                framealpha=0.9,
            )

        ax.set_aspect("equal", "box")
        ax.grid(False)

        # Save
        fig.tight_layout()
        safe_filename = cell_type.replace("+", "_").replace("/", "_")
        fig.savefig(
            output_dir / f"{safe_filename}_all_interventions.png",
            dpi=dpi,
            bbox_inches="tight",
        )
        plt.close(fig)

    print(f"   ✓ {len(cell_types)} plots saved to {output_dir.name}/")


def _add_ellipse(
    ax: plt.Axes,
    coords: np.ndarray,
    color: str,
    alpha: float = 0.2,
    n_std: float = 2.0,
) -> None:
    """Add covariance ellipse to plot."""

    if len(coords) < 2:
        return

    mean = coords.mean(axis=0)
    cov = np.cov(coords.T)

    # Eigendecomposition
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    # Ellipse parameters
    theta = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2 * n_std * np.sqrt(vals)

    ellipse = Ellipse(
        xy=mean,
        width=width,
        height=height,
        angle=theta,
        facecolor=color,
        alpha=alpha,
        edgecolor=color,
        linewidth=1.5,
    )

    ax.add_patch(ellipse)
