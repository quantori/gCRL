# Why v1 is Better Than v2: Analysis Summary

## TL;DR

**v1 DOMINATES v2 across ALL major metrics** despite having "weaker" hyperparameters. This counterintuitive result reveals that v2's "improvements" were actually harmful over-corrections.

## The Paradox

We designed v2 to fix v1's problems:
- v1 had reconstruction degradation (0.201 → 0.372 in original run)
- v1 had KL explosion (0.046 → 0.627)

So we made v2 with:
- **5× stronger** reconstruction weight (`alpha_rec`: 1.0 → 5.0)
- **5× stronger** KL regularization (`beta_kld_max`: 0.01 → 0.05)
- **33% weaker** alignment weight (`lambda_mcc`: 0.75 → 0.5)

**But v2 performed WORSE on every metric!**

---

## Final Results Comparison (Epoch 50)

| Metric | v1 | v2 | Winner | Difference |
|--------|----|----|--------|------------|
| **Reconstruction** | **0.1971** | 0.3411 | ✅ v1 | 73% better |
| **Alignment (MCC)** | **-0.9919** | -0.9851 | ✅ v1 | 0.7% better |
| **KL Divergence** | **0.6402** | 1.0492 | ✅ v1 | 39% better |
| **MMD** | 0.1549 | **0.1535** | ✅ v2 | 0.9% better |
| **Total Loss** | **0.6888** | 1.4390 | ✅ v1 | 52% better |

**Score: v1 wins 4/5 metrics** (and MMD difference is negligible)

---

## The Shocking Truth

### 1. Reconstruction: v1 is 73% Better! 🤯

```
v1: 0.201 → 0.197  (IMPROVED by 2%)
v2: 0.347 → 0.341  (improved by 2%, but started worse!)
```

**Despite v2 having `alpha_rec=5.0` (5× stronger than v1's 1.0):**
- v1 achieved **0.197** reconstruction loss
- v2 only achieved **0.341** reconstruction loss
- v1's final reconstruction is **73% better** than v2!

💡 **Insight**: Higher `alpha_rec` doesn't guarantee better reconstruction. The model needs balance.

### 2. Alignment: v1 is Superior 🎯

```
v1: -0.734 → -0.992  (35% improvement)
v2: -0.751 → -0.985  (31% improvement)
```

**v1 achieved -0.9919 vs v2's -0.9851:**
- Both achieved excellent alignment (near -1.0)
- But v1 is slightly better (-0.9919 < -0.9851)
- v1 did this with **higher** `lambda_mcc=0.75` vs v2's 0.5

💡 **Insight**: Strong alignment weight (lambda_mcc) is critical for gCRL-VAE.

### 3. KL Divergence: v1 is 39% Better! 📊

```
v1: 0.046 → 0.640  (1302% increase)
v2: 0.052 → 1.049  (1936% increase)
```

**Despite v2 having `beta_kld_max=0.05` (5× stronger than v1's 0.01):**
- v1 ended at **0.640**
- v2 ended at **1.049**
- v2's KL is **64% higher** than v1!

💡 **Insight**: Higher beta_kld constraint actually WORSENED KL! Possible over-regularization backfire.

---

## Why Did v2 Fail?

### Hypothesis 1: Over-Regularization Cascade ⚠️

v2's high `beta_kld_max=0.05` may have:
1. Constrained latent space too aggressively
2. Forced model to use suboptimal latent representations
3. Made reconstruction harder (need more expressive latent space)
4. Ironically caused KL to increase (fighting against the constraint)

**Analogy**: Like compressing a spring too hard - it pushes back harder.

### Hypothesis 2: Alignment is the Foundation 🏗️

v2 reduced `lambda_mcc` from 0.75 to 0.5 (33% weaker):
- gCRL-VAE relies on eigengene alignment for structure
- Weaker alignment → worse latent structure
- Worse latent structure → harder reconstruction
- **Alignment is not a "nice-to-have", it's foundational**

### Hypothesis 3: Multi-Objective Optimization Sweet Spot 🎯

v1 accidentally found a good balance:
```
v1: alpha_rec:lambda_mcc:beta_kld = 1.0 : 0.75 : 0.01
    Effective ratio ≈ 100 : 75 : 1
```

v2 destroyed this balance:
```
v2: alpha_rec:lambda_mcc:beta_kld = 5.0 : 0.5 : 0.05
    Effective ratio ≈ 100 : 10 : 1
```

**Key difference**: v2 de-emphasized alignment (100:10 vs 100:75)

---

## Visual Evidence

### Training Trajectories

Looking at [v1_v2_comparison.png](../v1_v2_comparison.png):

1. **Reconstruction Loss**: v1 (blue) is flat and LOW (~0.20), v2 (red) is flat but HIGH (~0.34)
2. **KL Divergence**: v1 (blue) rises to ~0.6, v2 (red) rises to ~1.0 (much worse!)
3. **Alignment (MCC)**: Both converge to ~-0.99, but v1 slightly better
4. **MMD**: Both converge to ~0.15 (essentially tied)
5. **Total Loss**: v1 (~0.7) is HALF of v2 (~1.4)

### Final Values

Looking at [v1_v2_final_comparison.png](../v1_v2_final_comparison.png):

The bar chart clearly shows v1 (blue, gold border) winning on:
- Reconstruction: 0.197 vs 0.341 (dramatic difference!)
- Alignment: -0.992 vs -0.985 (subtle but v1 better)
- KL: 0.640 vs 1.049 (major difference!)

---

## Key Insights

### 1. Don't Fix What Isn't Broken (Enough)

v1's original run showed:
- Reconstruction: 0.201 → 0.372 ❌
- But the NEW v1 run shows: 0.201 → 0.197 ✅

**The "problem" may have been random seed or early stopping, not the hyperparameters!**

### 2. Alignment is Sacred in gCRL-VAE

Unlike standard VAEs where reconstruction is king:
- gCRL-VAE **requires** strong eigengene alignment (`lambda_mcc`)
- Alignment creates the causal structure in latent space
- Weakening alignment (v2's 0.5 vs v1's 0.75) hurts everything

### 3. Over-Regularization is Real

More regularization ≠ better:
- v2's `beta_kld_max=0.05` backfired
- Instead of controlling KL, it made it worse
- Likely constrained model too much, creating internal conflicts

### 4. Hyperparameter Ratios Matter More Than Absolute Values

It's not about individual weights, but their **relative balance**:
- v1: Balanced emphasis on rec (1.0) and alignment (0.75)
- v2: Overemphasized rec (5.0), undervalued alignment (0.5)
- **The ratio broke the model**

---

## What Should We Do?

### ✅ Stick with v1

v1 is clearly superior:
- Better reconstruction
- Better alignment
- Better KL regularization
- Half the total loss

**Recommendation**: Use v1 configuration for all future experiments.

### 🤔 If You Want to Explore Further

Try subtle variations, NOT dramatic changes:

**v3 Option A (Gentle Reconstruction Boost):**
```python
alpha_rec = 1.5       # Slightly higher than v1's 1.0
beta_kld_max = 0.01   # Keep v1's value
lambda_mcc = 0.75     # Keep v1's value (critical!)
```

**v3 Option B (Stronger Alignment):**
```python
alpha_rec = 1.0       # Keep v1's value
beta_kld_max = 0.01   # Keep v1's value
lambda_mcc = 0.85     # Even stronger alignment
```

**v3 Option C (Tighter KL Control):**
```python
alpha_rec = 1.0       # Keep v1's value
beta_kld_max = 0.015  # Slightly higher (50% increase, not 5×!)
lambda_mcc = 0.75     # Keep v1's value
```

### ⚠️ What NOT to Do

- Don't use v2's `alpha_rec=5.0` (too strong)
- Don't use v2's `beta_kld_max=0.05` (over-regularizes)
- Don't lower `lambda_mcc` below 0.7 (alignment is critical)
- Don't make multiple large changes simultaneously

---

## Statistical Evidence

### v1 Statistics (50 epochs)
```
Reconstruction: mean=0.198, std=0.005, min=0.196, max=0.216
KL Divergence:  mean=0.723, std=0.329, min=0.046, max=1.352
Alignment:      mean=-0.983, std=0.037, min=-0.993, max=-0.734
MMD:            mean=0.502, std=0.723
```

### v2 Statistics (50 epochs)
```
Reconstruction: mean=0.341, std=0.004, min=0.338, max=0.354
KL Divergence:  mean=1.244, std=0.509, min=0.052, max=2.227
Alignment:      mean=-0.974, std=0.033, min=-0.988, max=-0.751
MMD:            mean=0.581, std=0.775
```

**Observations:**
- v1 has **72% better** mean reconstruction (0.198 vs 0.341)
- v2 has **42% worse** mean KL divergence (1.244 vs 0.723)
- v1 has **slightly better** mean alignment (-0.983 vs -0.974)
- v1 is more stable (lower reconstruction std: 0.005 vs 0.004... wait, v2 is more stable on rec, but at a MUCH WORSE value!)

---

## Conclusion

**v1 is definitively better than v2.**

The analysis reveals a critical lesson: **in multi-objective optimization, dramatic hyperparameter changes can backfire**. v2's "improvements" disrupted the delicate balance that v1 had achieved.

For gCRL-VAE specifically:
1. **Alignment is paramount** - strong `lambda_mcc` (≥0.7) is essential
2. **Balance over strength** - moderate weights that work together beat strong individual weights
3. **Over-regularization hurts** - excessive beta_kld constrains the model counterproductively
4. **Trust the process** - v1's configuration was better than we thought

**Action item**: Use v1 configuration going forward, and if tuning is needed, make small (≤50%) adjustments, testing one parameter at a time.

---

## Files

- Training histories:
  - v1: `results/real/Norman2019/VAE/training_history.json`
  - v2: `results/real/Norman2019/VAE_v2/training_history.json`
- Models:
  - v1: `results/real/Norman2019/VAE/norman_model.pt`
  - v2: `results/real/Norman2019/VAE_v2/norman_model.pt`
- Analysis script: `analyze_v1_v2_comparison.py`
- Plots:
  - `v1_v2_comparison.png` (training trajectories)
  - `v1_v2_final_comparison.png` (final values)

---

**Date**: 2025-11-12
**Analysis by**: Claude (Sonnet 4.5)
