# Recommendations: Next Steps After v1 vs v2 Analysis

## Bottom Line

**Use v1 configuration for all future work.** v1 outperforms v2 on 4 out of 5 metrics with dramatically better results.

---

## Quick Comparison

| Configuration | Reconstruction | Alignment (MCC) | KL Divergence | Recommendation |
|---------------|----------------|-----------------|---------------|----------------|
| **v1** ✅ | **0.1971** | **-0.9919** | **0.6402** | **USE THIS** |
| v2 ❌ | 0.3411 (73% worse) | -0.9851 (worse) | 1.0492 (64% worse) | Discard |

---

## Immediate Actions

### 1. Use v1 for Production ✅

```python
# Recommended configuration
cfg = VAEConfig(
    alpha_rec=1.0,         # ✅ Balanced reconstruction weight
    beta_kld_max=0.01,     # ✅ Gentle KL regularization
    lambda_mcc=0.75,       # ✅ Strong alignment (critical!)
    alpha_mmd_max=1.0,     # ✅ Standard MMD weight
    batch_size=512,
    epochs=50,
    lr=2e-3,
)
```

### 2. Use v1 Model for Downstream Tasks

```python
# Load the v1 model (better performance)
model = torch.load('results/real/Norman2019/VAE/norman_model.pt')

# NOT v2:
# model = torch.load('results/real/Norman2019/VAE_v2/norman_model.pt')  # ❌
```

---

## If You Want to Tune Further (Optional)

Only attempt if v1 isn't meeting your needs. Try **ONE change at a time** with small adjustments:

### Option A: Slightly Better Reconstruction (if needed)

```python
cfg = VAEConfig(
    alpha_rec=1.5,         # +50% (not 5×!)
    beta_kld_max=0.01,     # unchanged
    lambda_mcc=0.75,       # unchanged (don't touch!)
    ...
)
```

**Expected**: Marginally better reconstruction, similar alignment/KL.

### Option B: Even Stronger Alignment

```python
cfg = VAEConfig(
    alpha_rec=1.0,         # unchanged
    beta_kld_max=0.01,     # unchanged
    lambda_mcc=0.85,       # +13% (alignment is key!)
    ...
)
```

**Expected**: Potentially -0.995 alignment (near perfect), slightly worse reconstruction.

### Option C: Tighter KL Control

```python
cfg = VAEConfig(
    alpha_rec=1.0,         # unchanged
    beta_kld_max=0.015,    # +50% (not 5×!)
    lambda_mcc=0.75,       # unchanged (don't touch!)
    ...
)
```

**Expected**: KL might go from 0.64 to 0.50, but watch reconstruction closely.

---

## What NOT to Do ⚠️

### ❌ Don't Use v2's Hyperparameters

```python
# This configuration is PROVEN to be worse:
alpha_rec=5.0          # ❌ Way too strong
beta_kld_max=0.05      # ❌ Over-regularizes
lambda_mcc=0.5         # ❌ Too weak (breaks alignment)
```

### ❌ Don't Make Multiple Large Changes

```python
# Bad approach:
cfg = VAEConfig(
    alpha_rec=3.0,      # Changed!
    beta_kld_max=0.03,  # Changed!
    lambda_mcc=0.6,     # Changed!
    ...
)
```

**Why bad**: Can't isolate which change helped/hurt.

### ❌ Don't Lower lambda_mcc Below 0.7

```python
lambda_mcc=0.5  # ❌ v2 tried this, failed
lambda_mcc=0.6  # ⚠️ Risky
lambda_mcc=0.7  # ✅ Acceptable minimum
lambda_mcc=0.75 # ✅ v1's proven value
```

**Alignment is critical for gCRL-VAE's causal structure.**

---

## Evaluation Checklist

If you train a new variant, compare against v1 on these metrics:

### Must-Have Improvements
- [ ] Reconstruction loss < 0.197 (v1's value)
- [ ] Alignment MCC < -0.99 (more negative = better)
- [ ] KL divergence < 0.64 (v1's value)

### Nice-to-Have
- [ ] MMD < 0.155 (v1's value)
- [ ] Total loss < 0.69 (v1's value)

**If new config doesn't beat v1 on ≥3 of the "must-have" metrics, stick with v1.**

---

## Why v2 Failed: Summary

1. **Over-regularization**: `beta_kld_max=0.05` constrained latent space too aggressively
2. **Weak alignment**: `lambda_mcc=0.5` undermined the causal structure
3. **Broken balance**: The ratio of weights matters more than absolute values
4. **Compounding effects**: Multiple large changes interacted negatively

---

## Key Principles Going Forward

### 1. Alignment First 🎯
- `lambda_mcc` should be ≥0.7 (v1's 0.75 is proven)
- gCRL-VAE depends on eigengene alignment for its causal structure
- **Don't sacrifice alignment for other metrics**

### 2. Gentle Regularization 🧘
- Strong regularization (high beta_kld) can backfire
- v1's `beta_kld_max=0.01` is sufficient
- If KL is too high, look for other causes before increasing beta_kld

### 3. Balance Over Strength ⚖️
- Moderate, balanced weights > aggressive individual weights
- v1's ratio (rec:mcc ≈ 1.0:0.75) is effective
- Maintain relative proportions when tuning

### 4. One Change at a Time 🔬
- Change one hyperparameter per experiment
- Small adjustments (≤50% change)
- Compare each variant to v1 baseline

---

## For Different Datasets

If applying to a new dataset:

### Start with v1's proven ratios:
```python
# Base configuration (v1 ratios)
alpha_rec = 1.0
lambda_mcc = 0.75
beta_kld_max = 0.01

# Scale alpha_rec based on TF→gene expansion ratio (if desired)
# But keep the RATIO of alpha_rec:lambda_mcc ≈ 1.33:1
expansion_ratio = n_genes / n_tfs
if expansion_ratio > 20:
    alpha_rec = min(expansion_ratio / 20, 2.0)  # Cap at 2.0, not 5.0!
    lambda_mcc = alpha_rec * 0.75  # Maintain ratio
```

**Key insight**: If you scale one weight, scale related weights proportionally.

---

## Success Metrics

Ultimately, judge model quality by:

1. **Reconstruction loss** (~0.20 is good for Norman2019)
2. **Alignment MCC** (< -0.99 is excellent)
3. **KL divergence** (< 0.7 is reasonable for this architecture)
4. **Intervention prediction accuracy** (evaluate on held-out interventions)
5. **Latent space structure** (visualize with UMAP/t-SNE)

**v1 already achieves excellent scores on 1-3. Verify it's also good on 4-5.**

---

## Questions for Further Investigation

1. **Why did v1's first run have reconstruction degradation?**
   - Was it random seed?
   - Different data split?
   - Early stopping vs full 50 epochs?

2. **How well does v1 predict held-out interventions?**
   - This is the ultimate test
   - Can create test set from Norman2019 data

3. **What's the theoretical optimal balance?**
   - Could run small grid search around v1
   - But only if v1 isn't meeting requirements

4. **How does v1 perform on other datasets?**
   - Lee2022, Replogle2022, etc.
   - May need minor adjustments per dataset

---

## Final Recommendation

> **Stick with v1. It works.**
>
> Only deviate if you have:
> 1. A specific, measured deficiency in v1
> 2. A hypothesis for why a change would help
> 3. A plan to test ONE change at a time
> 4. Metrics to evaluate success
>
> v2 taught us that "improvements" can make things worse. Trust the data.

---

## Quick Reference Commands

### Train with v1 config (recommended):
```bash
conda run -n deep_learning python -c "
from gcrl.training.train_gcrl_vae import VAEConfig, train_gcrl_vae
import scanpy as sc

adata = sc.read_h5ad('data/processed/norman2019_preprocessed.h5ad')

cfg = VAEConfig(
    outdir='results/real/Norman2019/VAE_production',
    alpha_rec=1.0,
    beta_kld_max=0.01,
    lambda_mcc=0.75,
    batch_size=512,
    epochs=50,
    lr=2e-3,
)

model, history = train_gcrl_vae(adata, cfg)
"
```

### Load v1 model:
```python
import torch
model = torch.load('results/real/Norman2019/VAE/norman_model.pt')
```

### Compare any new run to v1:
```bash
conda run -n deep_learning python analyze_v1_v2_comparison.py
# Edit script to point to new history file
```

---

**Remember**: v1 achieves 0.197 reconstruction, -0.992 alignment, 0.640 KL. That's the bar to beat.
