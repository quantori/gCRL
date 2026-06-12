# gCRL-VAE Prediction Evaluation & Visualization

Comprehensive tools for evaluating and visualizing predicted vs. actual intervention effects.

## Quick Start

```python
from gcrl.training.train_gcrl_vae import train_gcrl_vae
from gcrl.evaluation import evaluate_predictions, visualize_predictions

# 1. Train model
model, history = train_gcrl_vae(adata, cfg)

# 2. Generate predictions
preds = model.predict(adata, seed=42)

# 3. Compute metrics (fast)
metrics_df = evaluate_predictions(
    adata=adata,
    preds=preds,
    output_dir="results/metrics"
)

# 4. Create visualizations (slower)
visualize_predictions(
    adata=adata,
    preds=preds,
    output_dir="results/visualizations"
)
```

## Installation

Make sure `umap-learn` is installed:

```bash
pip install umap-learn
```

Or reinstall the package with updated dependencies:

```bash
cd /path/to/gCRL
pip install -e .
```

## What It Creates

### Metrics (`evaluate_predictions`)

**1. Overall Metrics CSV**
- **Output**: `prediction_metrics.csv`
- Contains metrics for all (cell_type, intervention) pairs
- Columns: `cell_type`, `intervention`, `centroid_distance`, `n_test_cells`, `n_pred_cells`
- Sorted by centroid distance (best predictions first)

**2. Per-Cell-Type Metrics CSV**
- **Output**: `{celltype}_metrics.csv`
- Metrics for each cell type separately

**Centroid distances** are computed in **gene expression space** (not UMAP space) to match the training objective.

---

### Visualizations (`visualize_predictions`)

**1. Per-Intervention Plots**
Individual UMAP for each (cell_type, intervention) pair:
- Light gray background: unperturbed cells
- Blue circles: actual test cells
- Purple circles: predicted cells
- Large X markers: centroids (with connecting line)
- Ellipses: 2-std covariance

**Output**: `per_intervention/{celltype}_{intervention}.png`

**2. Per-Cell-Type Plots**
Aggregate UMAP showing all interventions for each cell type:
- Light gray background: unperturbed cells
- Colored circles: actual test cells (one color per intervention)
- Colored squares: predicted cells (matching colors)
- Large markers: centroids (circles=actual, squares=predicted)

**Output**: `per_celltype/{celltype}_all_interventions.png`

## Parameters

### `evaluate_predictions` Parameters
```python
evaluate_predictions(
    adata,                    # Original AnnData (required)
    preds,                    # Predictions AnnData (required)
    output_dir,               # Output directory (required)

    # Column names
    set_key="set",
    intervention_key="intervention",
    cell_type_key="cell_type",
    control_labels=("control", "unperturbed"),
)
```

### `visualize_predictions` Parameters
```python
visualize_predictions(
    adata,                    # Original AnnData (required)
    preds,                    # Predictions AnnData (required)
    output_dir,               # Output directory (required)

    # Column names
    set_key="set",
    intervention_key="intervention",
    cell_type_key="cell_type",
    control_labels=("control", "unperturbed"),

    # UMAP fitting strategy
    umap_fit_on="controls",   # 'controls' (default) or 'all'

    # UMAP parameters
    n_pcs=30,
    umap_n_neighbors=15,
    umap_min_dist=0.3,
    umap_metric="euclidean",
    random_state=42,

    # Plot appearance
    figsize_single=(8, 7),
    figsize_celltype=(10, 8),
    dpi=150,
    show_legend=True,
)
```

**UMAP Fitting Strategies**:
- `umap_fit_on="controls"` (default): Fit on controls only, project test/preds
- `umap_fit_on="all"`: Fit on all cells for global structure

## Interpretation

### Good Predictions
- ✅ Small centroid distance (< 1.0)
- ✅ Overlapping blue/purple points
- ✅ Similar ellipse shapes and sizes
- ✅ Both distributions in same region

### Poor Predictions
- ❌ Large centroid distance (> 3.0)
- ❌ Non-overlapping distributions
- ❌ Very different shapes/spreads
- ❌ Predictions in wrong region

## Example Workflow

### In Jupyter Notebook

```python
# After training (see 2_Norman_VAE.ipynb)

# Generate predictions
preds = model.predict(adata, seed=42)

# Compute metrics
from gcrl.evaluation import evaluate_predictions, visualize_predictions

metrics_df = evaluate_predictions(
    adata=adata,
    preds=preds,
    output_dir=cfg.outdir + "/metrics"
)

# Create visualizations
visualize_predictions(
    adata=adata,
    preds=preds,
    output_dir=cfg.outdir + "/visualizations"
)

# Analyze results
print("\nTop 10 best predictions:")
print(metrics_df.sort_values('centroid_distance').head(10))

print("\nTop 10 worst predictions:")
print(metrics_df.sort_values('centroid_distance', ascending=False).head(10))
```

## Files and Documentation

- **Metrics module**: `src/gcrl/evaluation/prediction_eval.py`
- **Visualization module**: `src/gcrl/evaluation/prediction_viz.py`
- **Metrics guide**: `docs/claude_memos/04_evaluation/prediction_eval_guide.md`
- **Visualization guide**: `docs/claude_memos/04_evaluation/prediction_viz_guide.md`
- **Implementation summary**: `docs/claude_memos/04_evaluation/implementation_summary.md`
- **Notebook example**: `notebooks/20_modeling_gcrl_vae/2_Norman_VAE.ipynb` (cells at end)

## Features

### Metrics (`evaluate_predictions`)
✅ Centroid distance in gene expression space (matches training objective)
✅ Per-intervention and per-cell-type metrics
✅ CSV export for programmatic analysis
✅ Fast computation (no UMAP required)
✅ Summary statistics

### Visualization (`visualize_predictions`)
✅ Automatic batch processing of all test interventions
✅ UMAP fitted on training controls for consistent embedding
✅ Support for single and double perturbations
✅ Centroid markers with connecting lines
✅ Distribution visualization with covariance ellipses
✅ Colorblind-friendly color scheme
✅ High-resolution output (150 DPI default)
✅ Comprehensive error handling and validation

## Customization Examples

### Fit UMAP on All Cells
```python
visualize_predictions(
    adata, preds, output_dir,
    umap_fit_on="all",     # Show global structure
)
```

### More Local Detail
```python
visualize_predictions(
    adata, preds, output_dir,
    umap_n_neighbors=5,    # Smaller neighborhoods
    umap_min_dist=0.1,     # Tighter clusters
)
```

### Higher Resolution
```python
visualize_predictions(
    adata, preds, output_dir,
    figsize_single=(12, 10),
    dpi=300,
)
```

### Custom Column Names
```python
visualize_predictions(
    adata, preds, output_dir,
    set_key="split",
    intervention_key="perturbation",
    cell_type_key="celltype",
)
```

## Troubleshooting

### ImportError: No module named 'umap'
```bash
pip install umap-learn
```

### Plots look strange
Try adjusting UMAP parameters:
- Increase `n_pcs` (more PCA components)
- Change `umap_n_neighbors` (5-30 range)
- Adjust `umap_min_dist` (0.1-0.5 range)

### Out of memory
- Reduce `n_pcs` (e.g., 20 instead of 30)
- Process cell types separately
- Subsample controls before fitting

### Colors hard to distinguish
- Edit colormap in source: `plt.cm.tab20` → `plt.cm.Set3`
- Reduce interventions per plot
- Create separate plots for subsets

## Technical Details

### UMAP Strategy
1. Fit UMAP on training control cells only
2. Transform test and predicted cells to this space
3. Ensures consistent embedding without "cheating"

### Computational Efficiency
- PCA preprocessing reduces dimensionality
- Single UMAP fit (not per-plot)
- Vectorized NumPy operations
- Rasterized scatter plots for smaller files

### Memory Usage
- 10k cells: ~500MB
- 50k cells: ~2GB

### Runtime
- Small dataset (1k controls): ~30s
- Medium dataset (10k controls): ~2min
- Large dataset (50k controls): ~10min

## Color Scheme

**Per-intervention plots:**
- Background: Light gray (#D3D3D3, 15% alpha)
- Actual: Blue (#2E86AB, 60% alpha)
- Predicted: Purple (#A23B72, 60% alpha)

**Per-cell-type plots:**
- Background: Light gray (#D3D3D3, 15% alpha)
- Interventions: tab20 colormap (automatic)

All colors are colorblind-friendly.

## Citation

If you use this visualization tool, please cite:

```bibtex
@software{gcrl_vae_viz,
  title = {gCRL-VAE Prediction Visualization},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/gCRL}
}
```

## Support

For questions or issues:
1. Check the comprehensive guide: `docs/prediction_visualization_guide.md`
2. See example output: `docs/visualization_example_output.txt`
3. Review the notebook example: `notebooks/20_modeling_gcrl_vae/2_Norman_VAE.ipynb`
4. Open an issue on GitHub

## Future Enhancements

Potential additions:
- Interactive Plotly visualizations
- Gene-level marker highlighting
- 3D UMAP embeddings
- Statistical distribution tests
- Density contour plots
- Batch effect visualization

---

**Created**: 2025-11-12
**Version**: 1.0
**Module**: `gcrl.evaluation.visualize_predictions`
