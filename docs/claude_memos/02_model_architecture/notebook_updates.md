# Notebook Updates: 2_Norman_VAE.ipynb

## Summary of Changes

Updated the Norman VAE notebook to work with the new gCRL-VAE training implementation.

## Key Updates

### 1. VAEConfig Parameters (Cell: cfg configuration)
**Changed**: Updated all configuration parameters to match new schema

**New parameters added:**
- `alpha_rec = 1.0` - Reconstruction loss weight (constant)
- `lambda_mcc = 1.0` - Eigengene alignment weight (constant)
- `lambda_sparse = 0.01` - L1 sparsity on causal matrix G (constant)
- `lambda_centroid = 0.1` - Centroid loss weight (constant, uses MMD schedule)
- `beta_kld_max = 1e-2` - KL divergence max weight (scheduled: ramps from epoch 10)
- `alpha_mmd_max = 1.0` - MMD loss max weight (scheduled: ramps from epoch 5)
- `mmd_kernel_mul = 2.0` - MMD kernel bandwidth multiplier
- `mmd_kernel_num = 5` - Number of MMD kernel scales
- `mmd_fix_sigma = None` - Adaptive bandwidth (None) or fixed
- `preproc_epsilon = 0.1` - Z-score shrinkage epsilon
- `preproc_clip_value = 6.0` - Z-score clipping threshold

**Removed parameters:**
- Old `beta_kld` parameter replaced with scheduled `beta_kld_max`

**Changed:**
- `epochs = 100` (reduced from 1000 for reasonable training time)

### 2. Model Architecture Cell
**Updated**: Added detailed inspection of model architecture

**Now displays:**
- Latent dimension (z_dim = p+1)
- Intervention dimension (c_dim)
- Input dimension (TFs only, not all genes)
- Output dimension (all genes: TF + TG)
- Cell-type dimension (if available)
- DAG causal matrix G shape

**Key architecture notes:**
- Encoder: [n_TFs → 128 → 2(p+1)] - simplified single hidden layer
- Decoder: Polynomial (linear + quadratic terms, no hidden layers)
- Input: TF expression ONLY
- Output: All gene expression (TF + TG)

### 3. Training Cell
**Updated**: Added comments explaining the training process

**Key training features:**
- Per-cell-type z-scoring using training controls only
- Training on controls only (reconstruction + eigengene alignment)
- Validation against real interventions using MMD loss
- Alternative loading instructions for pre-trained models

### 4. Saving Cell
**Changed**: Fixed filenames from "lee_*" to "norman_*"

**Now saves:**
- `norman_post_training.h5ad` - AnnData with preprocessed data
- `norman_model_state.pt` - PyTorch state dict
- `norman_model.pt` - Full PyTorch model
- `training_history.json` - Already saved by train_gcrl_vae

**Changed from pickle to PyTorch save** for better compatibility

### 5. Training History Visualization (NEW CELL)
**Added**: Complete visualization of all 6 loss components

**Plots created:**
1. 6-panel plot showing all loss components:
   - Reconstruction loss (constant weight)
   - KL divergence loss (β scheduled from epoch 10)
   - Eigengene alignment loss (partial-MCC, constant weight)
   - Sparsity loss (L1 on G, constant weight)
   - MMD loss (α scheduled from epoch 5)
   - Centroid loss (α × λ scheduled from epoch 5)
2. Total loss over epochs

**Outputs:**
- `training_loss_components.png`
- `training_total_loss.png`
- Final loss values printed to console

### 6. Causal Matrix Inspection (NEW CELL)
**Added**: Visualization and statistics for learned causal matrix G

**Visualizations:**
- Heatmap of causal matrix G (upper-triangular DAG)
- Annotated with values

**Statistics computed:**
- Matrix shape
- Total possible edges (upper triangle)
- Non-zero edges (threshold > 0.01)
- Sparsity percentage
- L1 norm of upper triangle

**Outputs:**
- `causal_matrix_G.png`

### 7. Evaluation Cells
**Changed**: Commented out evaluation cells (module not ready yet)

**Cells affected:**
- EvalConfig cell (cell_id: 7676f324-5c3c-4553-afbf-c4d7f0f61467)
- evaluate_gcrl_vae call (cell_id: 5ea10549-a393-4dd1-90b7-5cff60fca773)

**Note**: Uncomment when evaluation module is implemented

## Loss Function Changes

The new implementation uses 6 loss components (previously had fewer):

```
L_total = 1.0 × L_rec                    # Reconstruction (constant)
        + β(t) × L_KL                    # β: 0 → 0.01 (epochs 10+)
        + 1.0 × L_mcc                    # Alignment (constant)
        + 0.01 × L_sparse                # Sparsity (constant)
        + α(t) × L_MMD                   # α: 0 → 1.0 (epochs 5+)
        + α(t) × 0.1 × L_centroid        # Centroid (scheduled + weighted)
```

### Loss Scheduling:
- **Epochs 1-5**: Reconstruction + Alignment only (establish baseline)
- **Epochs 6-10**: MMD + Centroid ramp in (α: 0 → 1.0)
- **Epochs 11+**: KL ramps in (β: 0 → 0.01)

## Key Implementation Details

### Preprocessing
- Per-cell-type z-scoring using **training controls only**
- Shrinkage: σ̂ = √(σ² + 0.1)
- Clip to [-6, 6]
- Frozen stats applied to all cells (prevents leakage)
- Original values saved in `adata.layers["X_log1p"]`

### Training Strategy
- **Controls-only reconstruction**: Only unperturbed cells used for VAE training
- **Single cell-type batching**: Each batch contains cells from only one cell type
- **MMD validation**: Interventions validated against real perturbed cells
- **MMD computed on original expression**: Real cells NOT passed through encoder/decoder

### Model Architecture
- **TF-only input**: Encoder takes only TF expression (n_TFs)
- **All-gene output**: Decoder reconstructs all genes (n_TFs + n_TGs)
- **Polynomial decoder**: Linear + quadratic terms (no hidden layers, no non-linearities)
- **Hard routing**: Each TF maps deterministically to its community's latent dimension
- **Cell-type conditioning**: One-hot cell-type vectors for multi-cell-type datasets

## Files Generated by Training

Training will create the following files in `cfg.outdir`:

1. `training_history.json` - All loss components per epoch (auto-saved)
2. `eval_wiring.json` - TF names, tf_to_latent mapping (auto-saved)
3. `norman_post_training.h5ad` - AnnData with preprocessing metadata
4. `norman_model_state.pt` - PyTorch state dict
5. `norman_model.pt` - Full PyTorch model
6. `training_loss_components.png` - 6-panel loss visualization
7. `training_total_loss.png` - Total loss plot
8. `causal_matrix_G.png` - Learned causal matrix heatmap

## Next Steps

1. Run the updated notebook to train the model
2. Inspect training history to verify:
   - Reconstruction loss decreasing
   - KL loss stabilizes after epoch 10
   - MMD loss decreases after epoch 5
   - Alignment loss (MCC) is negative and increasing toward 0
   - Sparsity loss decreases (G becomes sparse)
   - Centroid loss decreases after epoch 5
3. Inspect causal matrix G for biological interpretability
4. Once evaluation module is ready, uncomment evaluation cells

## Compatibility Notes

- Requires `train_gcrl_vae` function from `gcrl.training.train_gcrl_vae`
- Requires `VAEConfig` dataclass with new parameters
- Requires `GCRLVAE` model with polynomial decoder
- Requires `compute_eigengenes` function from `gcrl.grn.eigengenes`
- Works with AnnData objects containing:
  - `adata.obs`: set, cell_type, intervention
  - `adata.var`: kind (TF/TG), community
  - `adata.X`: normalized, log1p-transformed expression
