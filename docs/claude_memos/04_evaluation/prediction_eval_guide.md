# Prediction Evaluation Guide

## Overview

The `evaluate_predictions()` function computes quantitative metrics comparing predicted intervention effects against actual test data from the gCRL-VAE model.

**Note**: This function focuses purely on metrics computation. For visualizations, see [`visualize_predictions()`](prediction_viz_guide.md).

## Features

- **Centroid distance**: Euclidean distance in **gene expression space** (matches training objective)
- **Per-intervention metrics**: Individual metrics for each (cell_type, intervention) pair
- **Per-cell-type metrics**: Aggregated metrics by cell type
- **CSV export**: Machine-readable format for downstream analysis
- **Summary statistics**: Quick overview of overall performance
- **Fast computation**: No UMAP fitting required

## Installation

No additional dependencies required beyond the base package.

## Basic Usage

```python
from gcrl.training.train_gcrl_vae import train_gcrl_vae
from gcrl.evaluation import evaluate_predictions

# 1. Train model
model, history = train_gcrl_vae(adata, cfg)

# 2. Generate predictions
preds = model.predict(adata, seed=42)

# 3. Compute metrics
metrics_df = evaluate_predictions(
    adata=adata,
    preds=preds,
    output_dir="results/metrics"
)

# 4. Analyze results
print(metrics_df.sort_values('centroid_distance').head(10))
```

## Output Structure

The function creates CSV files:

```
results/metrics/
├── prediction_metrics.csv          # All interventions
├── K562_metrics.csv               # Per-cell-type (if applicable)
└── ...
```

## Output Format

### Overall Metrics CSV (`prediction_metrics.csv`)

Contains one row per (cell_type, intervention) pair:

| Column | Type | Description |
|--------|------|-------------|
| `cell_type` | str | Cell type name |
| `intervention` | str | Intervention name (e.g., "GATA1", "GATA1+CEBPA") |
| `centroid_distance` | float | Euclidean distance in gene expression space |
| `n_test_cells` | int | Number of actual test cells |
| `n_pred_cells` | int | Number of predicted cells |

**Sorted by**: `centroid_distance` (ascending) - best predictions first

### Per-Cell-Type Metrics CSV (`{celltype}_metrics.csv`)

Same format as above, but filtered to a single cell type.

## Parameters

```python
evaluate_predictions(
    adata,                    # Original AnnData (required)
    preds,                    # Predictions AnnData (required)
    output_dir,               # Output directory (required)

    # Column names (optional)
    set_key="set",
    intervention_key="intervention",
    cell_type_key="cell_type",
    control_labels=("control", "unperturbed"),
)
```

### Parameter Details

- **adata**: Original AnnData with training controls and test data
  - Must have `obs[set_key]` column with "training" and "test" values
  - Must have `obs[intervention_key]` and `obs[cell_type_key]` columns

- **preds**: Predictions from `model.predict()`
  - Should have `obs['set'] == 'prediction'` for all cells
  - Must have matching `intervention_key` and `cell_type_key` columns

- **output_dir**: Directory for saving CSV files

- **set_key**: Column name for train/test split (default: "set")

- **intervention_key**: Column name for interventions (default: "intervention")

- **cell_type_key**: Column name for cell types (default: "cell_type")

- **control_labels**: Labels identifying control cells (default: ("control", "unperturbed"))

## Return Value

Returns a pandas DataFrame with the same structure as `prediction_metrics.csv`:

```python
metrics_df = evaluate_predictions(adata, preds, output_dir)

# DataFrame columns:
# - cell_type
# - intervention
# - centroid_distance
# - n_test_cells
# - n_pred_cells

# Analyze best predictions
best = metrics_df.sort_values('centroid_distance').head(10)
print("Best predictions:")
print(best)

# Analyze worst predictions
worst = metrics_df.sort_values('centroid_distance', ascending=False).head(10)
print("\nWorst predictions:")
print(worst)

# Filter by cell type
k562_metrics = metrics_df[metrics_df['cell_type'] == 'K562']
```

## Interpretation Guide

### Centroid Distance

**What it measures**: Euclidean distance between the mean gene expression of actual and predicted cells.

**Why gene expression space?**: This matches the training objective where centroid loss is computed as MSE between mean expressions, not UMAP coordinates.

### Good vs Poor Predictions

**Good predictions** (low distance):
- ✅ Distance < 1.0: Excellent
- ✅ Distance < 2.0: Good
- ⚠️ Distance 2.0-3.0: Acceptable

**Poor predictions** (high distance):
- ❌ Distance > 3.0: Poor
- ❌ Distance > 5.0: Very poor

**Note**: Absolute thresholds depend on your data scale and normalization.

### Analysis Patterns

Look for:
1. **Intervention complexity**: Do double perturbations have higher distances?
2. **Cell type effects**: Are certain cell types harder to predict?
3. **Gene targets**: Do certain genes lead to poor predictions?
4. **Systematic biases**: Are predictions consistently over/under-shooting?

## Integration with Notebook

Add this cell to your notebook after training:

```python
# Cell: Compute prediction metrics
from gcrl.evaluation import evaluate_predictions
import pandas as pd

# Generate predictions
preds = model.predict(adata, seed=42)

# Compute metrics
metrics_df = evaluate_predictions(
    adata=adata,
    preds=preds,
    output_dir=cfg.outdir + "/metrics"
)

# Analyze results
print("\n" + "="*70)
print("Top 10 Best Predictions")
print("="*70)
print(metrics_df.sort_values('centroid_distance').head(10).to_string(index=False))

print("\n" + "="*70)
print("Top 10 Worst Predictions")
print("="*70)
print(metrics_df.sort_values('centroid_distance', ascending=False).head(10).to_string(index=False))

# Per-cell-type summary
print("\n" + "="*70)
print("Per-Cell-Type Summary")
print("="*70)
summary = metrics_df.groupby('cell_type')['centroid_distance'].agg(['mean', 'median', 'min', 'max', 'count'])
print(summary)
```

## Advanced Usage

### Custom Column Names

```python
metrics_df = evaluate_predictions(
    adata=adata,
    preds=preds,
    output_dir="results/metrics",
    set_key="split",              # Custom split column
    intervention_key="perturbation",  # Custom intervention column
    cell_type_key="celltype",     # Custom cell type column
    control_labels=("ctrl", "neg")  # Custom control labels
)
```

### Filtering and Analysis

```python
# Filter by intervention type
single_pert = metrics_df[~metrics_df['intervention'].str.contains('+')]
double_pert = metrics_df[metrics_df['intervention'].str.contains('+')]

print(f"Single perturbations: mean distance = {single_pert['centroid_distance'].mean():.3f}")
print(f"Double perturbations: mean distance = {double_pert['centroid_distance'].mean():.3f}")

# Find interventions with large sample size
large_sample = metrics_df[metrics_df['n_test_cells'] >= 100]
print(f"\nInterventions with ≥100 test cells: {len(large_sample)}")

# Correlation between sample size and accuracy
import numpy as np
correlation = np.corrcoef(metrics_df['n_test_cells'], metrics_df['centroid_distance'])[0, 1]
print(f"Correlation (sample size vs distance): {correlation:.3f}")
```

### Visualization of Metrics

```python
import matplotlib.pyplot as plt

# Distribution of centroid distances
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(metrics_df['centroid_distance'], bins=30, edgecolor='black')
axes[0].set_xlabel('Centroid Distance')
axes[0].set_ylabel('Count')
axes[0].set_title('Distribution of Centroid Distances')

# Per-cell-type comparison
metrics_df.boxplot(column='centroid_distance', by='cell_type', ax=axes[1])
axes[1].set_xlabel('Cell Type')
axes[1].set_ylabel('Centroid Distance')
axes[1].set_title('Prediction Accuracy by Cell Type')
plt.suptitle('')  # Remove default title

plt.tight_layout()
plt.savefig('metrics_analysis.png', dpi=150)
```

## Computational Notes

- **Speed**: Fast - no UMAP fitting required
- **Memory**: Minimal - only computes means and distances
- **Scalability**: Linear in number of interventions
- **Typical runtime**: <1 second for 100 interventions

## Comparison with Visualization

| Feature | `evaluate_predictions` | `visualize_predictions` |
|---------|----------------------|------------------------|
| **Output** | CSV files | PNG plots |
| **Speed** | Fast (<1s) | Slow (~2min) |
| **Purpose** | Quantitative analysis | Visual inspection |
| **Centroid distance** | Gene expression space | N/A (visual only) |
| **UMAP** | Not required | Required |
| **Memory** | Low | Moderate-High |

**Recommendation**: Run `evaluate_predictions` first for quick metrics, then `visualize_predictions` for detailed visual analysis.

## Troubleshooting

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

### Different number of test and predicted cells

This is normal - the model may generate a different number of predictions than test cells. Metrics are still valid as they compare centroids (means).

## Examples

### Example 1: Norman2019 Dataset

```python
# After training on Norman2019
metrics_df = evaluate_predictions(
    adata=adata,
    preds=preds,
    output_dir="../../results/real/Norman2019/VAE/metrics"
)

# Output:
# prediction_metrics.csv with all interventions
# K562_metrics.csv (if K562 is the cell type)
```

### Example 2: Multiple Cell Types

```python
# Lee dataset with multiple cell types
metrics_df = evaluate_predictions(
    adata=adata,
    preds=preds,
    output_dir="../../results/real/Lee/VAE/metrics"
)

# Compare cell types
for ct in metrics_df['cell_type'].unique():
    ct_df = metrics_df[metrics_df['cell_type'] == ct]
    print(f"{ct}: mean distance = {ct_df['centroid_distance'].mean():.3f}")
```

### Example 3: Statistical Testing

```python
from scipy import stats

# Compare single vs double perturbations
single = metrics_df[~metrics_df['intervention'].str.contains('+')]['centroid_distance']
double = metrics_df[metrics_df['intervention'].str.contains('+')]['centroid_distance']

t_stat, p_value = stats.ttest_ind(single, double)
print(f"Single vs Double perturbations:")
print(f"  Single: {single.mean():.3f} ± {single.std():.3f}")
print(f"  Double: {double.mean():.3f} ± {double.std():.3f}")
print(f"  t-statistic: {t_stat:.3f}, p-value: {p_value:.4f}")
```

## Citation

If you use this evaluation tool in your work, please cite:

```bibtex
@software{gcrl_vae,
  title = {gCRL-VAE: Causal Representation Learning with GRN Priors},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/gCRL}
}
```

## See Also

- [Prediction Visualization Guide](prediction_viz_guide.md) - Visual analysis of predictions
- [Implementation Summary](implementation_summary.md) - Technical details
- [README](README.md) - Overview of evaluation tools
