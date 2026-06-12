# Centroid Distance: Gene Expression Space vs. UMAP Space

## Summary

**Centroid distances are computed in gene expression space (not UMAP space)** to match the training objective.

## Why This Matters

During training, the gCRL-VAE model optimizes a centroid loss that compares mean gene expressions:

```python
# From train_gcrl_vae.py lines 1148-1151
centroid_sim = x_sim.mean(dim=0)      # Mean across all genes (n_genes,)
centroid_real = X_real_all.mean(dim=0)  # Mean across all genes (n_genes,)
loss_centroid = F.mse_loss(centroid_sim, centroid_real)  # MSE in gene space
```

This is an **n_genes dimensional** comparison (e.g., 2,701 genes for Norman2019).

## Visualization vs. Evaluation

### UMAP Space (for visualization only)
- **Purpose**: 2D projection for human-interpretable plots
- **Dimensions**: 2 (UMAP 1, UMAP 2)
- **Use**: Shows spatial relationships between cell populations
- **What we plot**: Cell positions, centroid markers, connecting lines

### Gene Expression Space (for metrics)
- **Purpose**: Quantitative evaluation matching training objective
- **Dimensions**: n_genes (e.g., 2,701)
- **Use**: Measures actual prediction accuracy
- **What we compute**: Centroid distance metric

## Implementation

The visualization function:
1. **Fits UMAP** on training control cells
2. **Transforms** test and predicted cells to UMAP space
3. **Plots** everything in UMAP space (2D visualization)
4. **Computes** centroid distance in gene expression space (n_genes dimensions)

```python
# UMAP coordinates (for plotting)
test_coords_umap = test_umap[test_mask]       # (n_cells, 2)
pred_coords_umap = preds_umap[pred_mask]      # (n_cells, 2)

# Gene expression (for metrics)
test_coords_expr = np.asarray(test_data[test_mask].X)  # (n_cells, n_genes)
pred_coords_expr = np.asarray(preds[pred_mask].X)      # (n_cells, n_genes)

# Visualize in UMAP space
test_centroid_umap = test_coords_umap.mean(axis=0)  # (2,)
pred_centroid_umap = pred_coords_umap.mean(axis=0)  # (2,)
ax.scatter(test_centroid_umap[0], test_centroid_umap[1], ...)

# Measure in gene expression space
test_centroid_expr = test_coords_expr.mean(axis=0)  # (n_genes,)
pred_centroid_expr = pred_coords_expr.mean(axis=0)  # (n_genes,)
centroid_dist = np.linalg.norm(test_centroid_expr - pred_centroid_expr)
```

## Why Not Use UMAP Distance?

UMAP distance would be misleading because:

1. **Dimensionality reduction loses information**: 2,701 genes → 2 dimensions
2. **Non-linear transformation**: UMAP distorts distances to preserve topology
3. **Inconsistent with training**: Model was trained on gene expression, not UMAP
4. **Not comparable**: UMAP distances depend on hyperparameters (n_neighbors, min_dist)

## Example Comparison

Consider an intervention with:
- **Gene expression space**: 1,000 genes change by 0.1 → distance ≈ 3.16
- **UMAP space**: Points appear close but distance ≈ 0.5

The gene expression distance (3.16) is more meaningful because:
- It reflects the actual prediction error
- It matches what the model was trained to minimize
- It's comparable across different UMAP embeddings

## Consistency with Existing Code

This approach matches:

### Training Loss (train_gcrl_vae.py:1148-1151)
```python
centroid_sim = x_sim.mean(dim=0)      # n_genes
centroid_real = X_real_all.mean(dim=0)  # n_genes
loss_centroid = F.mse_loss(centroid_sim, centroid_real)
```

### Evaluation Metrics (evaluation_v0.9/metrics.py:51-57)
```python
pC = _safe_mean(pred)  # Mean across genes
d = cdist(pC, allC, metric='euclidean')  # Euclidean in gene space
```

### Old UMAP Viz (evaluation_v0.9/umap_viz.py:51)
```python
# Even the old code computed centroid distance in UMAP space,
# but this was inconsistent with training!
centroid_dist = np.linalg.norm(mu_p - mu_t)  # Should have been in gene space
```

## Interpretation

When you see:
- **`Centroid distance: 0.523`** on a plot
  - This is Euclidean distance in 2,701-dimensional gene expression space
  - Smaller = better prediction
  - Comparable across different interventions and cell types
  - Matches what the model was trained to minimize

Not:
- ~~Distance in 2D UMAP space~~ ❌
- ~~Arbitrary units~~ ❌
- ~~Different for each UMAP fit~~ ❌

## Technical Note

The centroid distance metric units are in the same space as the input data:
- If `adata.X` is log-normalized counts: distance is in log-space
- If `adata.X` is z-scored: distance is in standard deviations
- For Norman2019: normalized log-transformed counts

## Summary

✅ **Centroid distance = Euclidean distance in gene expression space**
- Matches training objective
- Quantitatively meaningful
- Comparable across experiments

✅ **UMAP = Visual representation only**
- Helps understand spatial relationships
- Shows cluster separation
- Not used for distance metrics

This design ensures that the visualization tool reports metrics that are consistent with how the model was trained and evaluated.
