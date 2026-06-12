# Prediction Visualization Guide

## Overview

The `visualize_predictions()` function creates comprehensive UMAP visualizations comparing predicted vs. actual intervention effects from the gCRL-VAE model.

**Note**: For quantitative metrics (centroid distances), see [`evaluate_predictions()`](prediction_eval_guide.md) instead. This module focuses purely on visualization.

## Features

- **Per-intervention plots**: One UMAP per (cell_type, intervention) pair
- **Per-cell-type plots**: One UMAP per cell_type showing all interventions
- **Centroid markers**: Highlights centroids of predicted vs. actual distributions with connecting lines
- **Distribution ellipses**: Shows spread of predicted and actual cells
- **Background context**: Shows unperturbed cells in light gray for reference

## Installation

Make sure `umap-learn` is installed:

```bash
pip install umap-learn
```

Or install the full package with updated dependencies:

```bash
pip install -e .
```

## Basic Usage

```python
from gcrl.training.train_gcrl_vae import train_gcrl_vae
from gcrl.evaluation import visualize_predictions

# 1. Train model
model, history = train_gcrl_vae(adata, cfg)

# 2. Generate predictions
preds = model.predict(adata, seed=42)

# 3. Create visualizations
visualize_predictions(
    adata=adata,
    preds=preds,
    output_dir="results/predictions_viz"
)
```

## Output Structure

The function creates two subdirectories:

```
results/visualizations/
├── per_intervention/
│   ├── erythroid_KLF1.png
│   ├── erythroid_GATA1.png
│   ├── erythroid_GATA1+CEBPA.png
│   └── ...
└── per_celltype/
    ├── erythroid_all_interventions.png
    └── ...
```

### Per-Intervention Plots

Each plot shows:
- **Light gray dots**: Unperturbed training cells (background)
- **Blue circles**: Actual test cells for this intervention
- **Purple circles**: Predicted cells for this intervention
- **Large X markers**: Centroids (blue = actual, purple = predicted)
- **Dashed line**: Connection between centroids
- **Ellipses**: 2-std covariance ellipses showing distribution spread

**Title format**: `{cell_type} | {intervention}`

### Per-Cell-Type Plots

Each plot shows:
- **Light gray dots**: Unperturbed training cells (background)
- **Colored circles**: Actual test cells (one color per intervention)
- **Colored squares**: Predicted cells (one color per intervention)
- **Large circle markers**: Actual centroids
- **Large square markers**: Predicted centroids
- **Dashed lines**: Connect corresponding centroids

**Title format**: `Cell Type: {cell_type}`
**Subtitle**: Number of interventions shown

## Advanced Parameters

```python
visualize_predictions(
    adata=adata,
    preds=preds,
    output_dir="results/predictions_viz",

    # Column names (if different from defaults)
    set_key="set",                    # Column for train/test split
    intervention_key="intervention",   # Column for intervention labels
    cell_type_key="cell_type",        # Column for cell types
    control_labels=("control", "unperturbed"),  # Labels for control cells

    # UMAP fitting strategy
    umap_fit_on="controls",           # 'controls' or 'all'

    # UMAP parameters
    n_pcs=30,                         # PCA components before UMAP
    umap_n_neighbors=15,              # UMAP n_neighbors
    umap_min_dist=0.3,                # UMAP min_dist
    umap_metric="euclidean",          # Distance metric
    random_state=42,                  # Random seed

    # Plot appearance
    figsize_single=(8, 7),            # Size for individual plots
    figsize_celltype=(10, 8),         # Size for cell-type plots
    dpi=150,                          # Resolution
    show_legend=True,                 # Show legend
)
```

### UMAP Fitting Strategies

**`umap_fit_on="controls"` (default)**:
- Fits UMAP only on training control (unperturbed) cells
- Projects test and predicted cells onto this space
- **Pros**: Consistent baseline, avoids data leakage, interpretable reference
- **Cons**: May not capture global structure of all conditions
- **Use when**: You want to see how perturbations move away from unperturbed baseline

**`umap_fit_on="all"`**:
- Fits UMAP on all cells (controls + test + predictions)
- Shows global structure across all conditions
- **Pros**: Better global structure, shows relationships between all conditions
- **Cons**: Less interpretable as baseline, predictions influence the embedding
- **Use when**: You want to see overall relationships between all conditions

## Interpretation Guide

### Good Predictions
- **Small centroid distance**: Centroids are close together
- **Overlapping ellipses**: Distribution shapes are similar
- **Similar density patterns**: Cells cluster in similar regions

### Poor Predictions
- **Large centroid distance**: Centroids are far apart
- **Non-overlapping distributions**: Different regions of UMAP space
- **Different spread**: One distribution much wider/tighter than the other

### Using the Visualizations

1. **Quick assessment**: Look at per-cell-type plots to see overall performance
2. **Detailed analysis**: Check per-intervention plots for specific interventions
3. **Quantitative metrics**: Use `evaluate_predictions()` to compute centroid distances
4. **Patterns**: Look for systematic biases (e.g., all double perturbations perform worse)

## Integration with Notebook

Add this cell to your notebook after training:

```python
# Cell: Generate predictions and create visualizations
from gcrl.evaluation import evaluate_predictions, visualize_predictions

print("Generating predictions...")
preds = model.predict(adata, seed=42)

print(f"\nGenerated {preds.n_obs:,} predictions")
print(f"Interventions: {preds.obs['intervention'].nunique()}")
print(f"Cell types: {preds.obs['cell_type'].nunique()}")

# Compute metrics
print("\nComputing metrics...")
metrics_df = evaluate_predictions(
    adata=adata,
    preds=preds,
    output_dir=cfg.outdir + "/metrics"
)

# Create visualizations
print("\nCreating visualizations...")
visualize_predictions(
    adata=adata,
    preds=preds,
    output_dir=cfg.outdir + "/visualizations"
)

print("\n✅ Complete! Check the metrics/ and visualizations/ folders.")
```

## Computational Notes

- **Memory**: UMAP fitting happens once on training controls, then transforms all data
- **Speed**: Uses PCA preprocessing to reduce dimensionality before UMAP
- **Batch processing**: All plots are generated in one function call
- **GPU**: Not required (UMAP runs on CPU)

### Typical Runtime

| Dataset Size | Training Controls | Time |
|--------------|-------------------|------|
| Small        | 1,000 cells       | ~30s |
| Medium       | 10,000 cells      | ~2min |
| Large        | 50,000 cells      | ~10min |

## Troubleshooting

### ImportError: No module named 'umap'

```bash
pip install umap-learn
```

### ValueError: Column 'X' not found in adata.obs

Check that your AnnData object has the required columns:
- `set` (or custom `set_key`)
- `intervention` (or custom `intervention_key`)
- `cell_type` (or custom `cell_type_key`)

### No predictions found for (cell_type, intervention)

This can happen if:
- The intervention wasn't in the test set
- The cell type has no test data
- The prediction generation failed for this group

Check `preds.obs` to verify which predictions were generated.

### UMAP looks very different from expected

Try adjusting UMAP parameters:
- Increase `n_pcs` if you have many genes
- Decrease `umap_n_neighbors` for more local structure
- Increase `umap_min_dist` for more spread-out embeddings

## Color Scheme

The default color scheme is colorblind-friendly:

- **Background**: Light gray (#D3D3D3, 15% alpha)
- **Actual cells**: Blue (#2E86AB)
- **Predicted cells**: Purple (#A23B72)
- **Per-cell-type**: Tab20 colormap for multiple interventions

## Examples

### Example 1: Norman2019 Dataset

```python
# After training on Norman2019
model, history = train_gcrl_vae(adata, cfg)
preds = model.predict(adata, seed=42)

visualize_predictions(
    adata=adata,
    preds=preds,
    output_dir="../../results/real/Norman2019/VAE/predictions_viz"
)

# Output:
# - per_intervention/KLF1.png
# - per_intervention/GATA1.png
# - per_intervention/GATA1+CEBPA.png (double perturbation)
# - per_celltype/all_all_interventions.png
# - per_celltype/all_centroid_distances.csv
```

### Example 2: Lee Dataset (Multiple Cell Types)

```python
# Lee dataset has multiple cell types
visualize_predictions(
    adata=adata,
    preds=preds,
    output_dir="../../results/real/Lee/VAE/predictions_viz"
)

# Output (one cell type example):
# - per_intervention/erythroid_KLF1.png
# - per_intervention/erythroid_GATA1.png
# - per_intervention/megakaryocyte_KLF1.png
# - per_intervention/megakaryocyte_GATA1.png
# - per_celltype/erythroid_all_interventions.png
# - per_celltype/megakaryocyte_all_interventions.png
```

### Example 3: Custom Column Names

```python
# If your data uses different column names
visualize_predictions(
    adata=adata,
    preds=preds,
    output_dir="results/viz",
    set_key="split",              # Instead of 'set'
    intervention_key="perturbation",  # Instead of 'intervention'
    cell_type_key="celltype",     # Instead of 'cell_type'
    control_labels=("ctrl", "neg")  # Custom control labels
)
```

## Citation

If you use this visualization in your work, please cite:

```bibtex
@software{gcrl_vae,
  title = {gCRL-VAE: Causal Representation Learning with GRN Priors},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/gCRL}
}
```
