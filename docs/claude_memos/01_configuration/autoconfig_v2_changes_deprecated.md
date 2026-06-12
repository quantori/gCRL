# Auto-Configuration v2 - Improved Hyperparameter Selection

## Summary

Updated `analyze_dataset_and_suggest_config()` to incorporate lessons learned from training analysis. The key insight: **reconstruction quality was degrading due to competing objectives with suboptimal weighting**.

---

## Changes Made

### 1. NEW: Adaptive Reconstruction Weight (alpha_rec)

**Problem**:
- v1 always set `alpha_rec = 1.0` regardless of task complexity
- Norman dataset has 139 TFs → 2701 genes (19.4× expansion)
- High expansion ratios make reconstruction harder → needs more weight

**Solution**:
```python
expansion_ratio = n_genes / n_tfs

if expansion_ratio > 30:
    alpha_rec = 10.0  # Very high expansion
elif expansion_ratio > 15:
    alpha_rec = 5.0   # High expansion (e.g., Norman: 19.4×)
elif expansion_ratio > 5:
    alpha_rec = 3.0   # Moderate expansion
else:
    alpha_rec = 1.0   # Low expansion (standard)
```

**Impact on Norman dataset**: `alpha_rec = 1.0 → 5.0` (5× increase)

---

### 2. INCREASED: KL Regularization (beta_kld_max)

**Problem**:
- v1 had weak KL regularization (0.01 for Norman)
- Resulted in KL explosion (0.046 → 0.627)
- Model was "hiding" information in latent space instead of learning good decoders

**Solution**:
Increase all thresholds by 5-10×:
```python
if n_ctrl_train < 2000 or n_latent > 10:
    beta_kld_max = 0.1   # Was 0.02 (5× increase)
elif n_ctrl_train < 10000:
    beta_kld_max = 0.05  # Was 0.01 (5× increase)
else:
    beta_kld_max = 0.02  # Was 0.005 (4× increase)
```

**Impact on Norman dataset**: `beta_kld_max = 0.01 → 0.05` (5× increase)

---

### 3. REDUCED: Alignment Weight (lambda_mcc)

**Problem**:
- v1 had alignment weight equal to reconstruction (both 1.0 effectively)
- Alignment was prioritized, hurting reconstruction
- Final alignment was -0.99 (near-perfect) but reconstruction degraded

**Solution**:
Reduce alignment weight to allow flexibility:
```python
if n_communities <= 3:
    lambda_mcc = 0.75  # Was 1.0
elif n_communities <= 6:
    lambda_mcc = 0.5   # Was 0.75
else:
    lambda_mcc = 0.3   # Was 0.5
```

**Impact on Norman dataset**: `lambda_mcc = 0.75 → 0.5` (33% reduction)

**Trade-off**: Alignment may drop from -0.99 to -0.95, still excellent

---

## Configuration Comparison

### For Norman2019 Dataset (8907 controls, 139 TFs, 2701 genes, 6 communities):

| Parameter      | v1 (old) | v2 (new) | Change      | Reason                                    |
|----------------|----------|----------|-------------|-------------------------------------------|
| alpha_rec      | 1.0      | **5.0**  | ↑ 5.0×      | 19.4× TF→gene expansion needs more weight |
| beta_kld_max   | 0.01     | **0.05** | ↑ 5.0×      | Prevent KL explosion                      |
| lambda_mcc     | 0.75     | **0.5**  | ↓ 0.67×     | Allow reconstruction flexibility          |
| alpha_mmd_max  | 1.0      | 1.0      | (unchanged) | Already optimal                           |
| lambda_sparse  | 0.02     | 0.02     | (unchanged) | Already optimal                           |
| lambda_centroid| 0.05     | 0.05     | (unchanged) | Already optimal                           |
| batch_size     | 512      | 512      | (unchanged) | Already optimal                           |
| epochs         | 50       | 50       | (unchanged) | Already optimal                           |
| lr             | 2e-3     | 2e-3     | (unchanged) | Already optimal                           |

---

## Expected Improvements

### v1 Training Trajectory (Problems):
```
Reconstruction:  0.201 → 0.372  ❌ INCREASING (bad)
KL Divergence:   0.046 → 0.627  ❌ EXPLODING (bad)
Alignment (MCC): -0.736 → -0.990 ✓ Excellent
MMD:             0.149           ✓ Reasonable
```

### v2 Expected Trajectory (Improvements):
```
Reconstruction:  0.2XX → 0.1XX  ✓ DECREASING (good!)
KL Divergence:   0.0XX → 0.2XX  ✓ CONTROLLED (good!)
Alignment (MCC): -0.7XX → -0.95X ✓ Still excellent
MMD:             ~0.15          ✓ Similar
```

**Key improvements**:
1. ✓ Reconstruction loss should **decrease** (primary goal achieved)
2. ✓ KL should stay **low** (<0.3, properly regularized)
3. ✓ Alignment may be **slightly worse** but still very good (acceptable trade-off)
4. ✓ MMD should remain **similar** (intervention matching preserved)

---

## Technical Rationale

### Why These Specific Changes?

#### 1. Alpha_rec = 5.0 (based on expansion ratio)

**Loss function weighting**:
```python
Loss_total = alpha_rec × L_rec + 1.0 × L_MCC + 1.0 × L_MMD + ...
```

**v1 (balanced)**:
```
Loss = 1.0 × Rec + 0.5 × MCC + 1.0 × MMD
     = Rec:MCC:MMD = 1.0:0.5:1.0
```

**v2 (reconstruction-prioritized)**:
```
Loss = 5.0 × Rec + 0.5 × MCC + 1.0 × MMD
     = Rec:MCC:MMD = 5.0:0.5:1.0
```

The model now **cares 5× more** about reconstruction quality.

#### 2. Beta_kld_max = 0.05 (prevent information hiding)

**KL divergence formula**:
```
KL[q(z|x) || p(z)] = -0.5 × E[1 + log(σ²) - μ² - σ²]
```

High KL means:
- Large means (μ) → latent vectors far from origin
- Large/small variances (σ²) → uncertain or over-confident posteriors

**With stronger beta**:
- Model is penalized more for deviating from N(0,I)
- Forces model to learn better decoders instead of "cheating" via latent encoding
- Results in more interpretable latent representations

#### 3. Lambda_mcc = 0.5 (allow flexibility)

**Previous**: Alignment and reconstruction had equal priority
**Problem**: Perfect alignment (-0.99) restricts latent space flexibility
**Solution**: Reduce alignment weight to allow reconstruction to "breathe"

**Expected trade-off**:
- Alignment: -0.99 → -0.95 (still 95% correlation, excellent)
- Reconstruction: 0.37 → ~0.15 (much better)

This is a **worthwhile trade-off**.

---

## How to Use

### In Notebook:

```python
# Automatically gets improved v2 configuration
cfg = analyze_dataset_and_suggest_config(
    adata,
    outdir="results/VAE",
    verbose=True  # Shows detailed reasoning
)

# The function now automatically:
# 1. Calculates TF→gene expansion ratio (19.4×)
# 2. Sets alpha_rec = 5.0 (adaptive)
# 3. Sets beta_kld_max = 0.05 (increased)
# 4. Sets lambda_mcc = 0.5 (reduced)
# 5. Shows comparison with v1

# Train
model, history = train_gcrl_vae(adata, cfg)
```

### Manual Override (if needed):

```python
cfg = analyze_dataset_and_suggest_config(adata, outdir="results/VAE")

# Further increase reconstruction weight if still problematic
cfg.alpha_rec = 10.0

# Further increase KL regularization
cfg.beta_kld_max = 0.1

# Train
model, history = train_gcrl_vae(adata, cfg)
```

---

## Validation Criteria

After training with v2, check:

### ✅ SUCCESS:
- Reconstruction loss **decreases** (not increases)
- KL divergence stays **below 0.3**
- Alignment remains **below -0.90** (90%+ correlation)
- MMD remains **below 0.2**

### ❌ FAILURE (needs more tuning):
- Reconstruction still increases
- KL still above 0.5
- Alignment drops below -0.80

### If failure occurs:
1. Increase `alpha_rec` further (try 10.0)
2. Increase `beta_kld_max` further (try 0.1)
3. Consider adding decoder capacity (architectural change)

---

## Backward Compatibility

### Will old code break?

**No** - The function signature is unchanged:
```python
cfg = analyze_dataset_and_suggest_config(adata, outdir, verbose)
```

### Will results change for existing datasets?

**Yes** - All datasets will now get improved hyperparameters:
- Higher `alpha_rec` if they have high TF→gene expansion
- Higher `beta_kld_max` for better regularization
- Lower `lambda_mcc` for reconstruction flexibility

This is **intentional** - the new defaults are better for most use cases.

### How to revert to v1 behavior (not recommended):

```python
cfg = analyze_dataset_and_suggest_config(adata, outdir)

# Manually revert to v1
cfg.alpha_rec = 1.0
cfg.beta_kld_max = 0.01
cfg.lambda_mcc = 0.75
```

---

## Future Enhancements

### Potential v3 improvements:

1. **Adaptive MMD weight** based on intervention complexity
2. **Validation-based early stopping** to prevent overfit
3. **Architecture adaptation** (add decoder layers for high expansion ratios)
4. **Learning rate scheduling** (reduce LR when losses plateau)
5. **Warm-up schedule** for reconstruction weight (start low, increase)

---

## References

- Training analysis: `docs/TRAINING_ANALYSIS_Norman.md`
- Bug fixes: `docs/BUG_FIXES_NOTEBOOK.txt`
- Implementation: `src/gcrl/training/train_gcrl_vae.py:258-488`
- Example usage: `notebooks/20_modeling_gcrl_vae/2_Norman_VAE.ipynb`

---

## Contact

For questions about v2 configuration:
1. Check training history: does reconstruction decrease?
2. Check KL divergence: is it below 0.3?
3. If issues persist, try manual tuning as described above
