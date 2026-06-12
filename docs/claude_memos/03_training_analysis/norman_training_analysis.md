# Training Analysis: gCRL-VAE on Norman2019 Dataset

## Executive Summary

🚨 **CRITICAL ISSUES DETECTED** - The model shows signs of problematic training dynamics:

1. **Reconstruction loss INCREASED** (85% worse): 0.201 → 0.372
2. **KL divergence EXPLODED** (1268% increase): 0.046 → 0.627
3. **Alignment actually IMPROVED** ✓ (more negative is better): -0.736 → -0.990
4. **MMD loss reasonable** ✓: 0.149 (intervention matching is working)

**Verdict**: The model is **overfitting to alignment and intervention objectives at the expense of reconstruction quality**. This is a **trade-off problem**, not a bug.

---

## Detailed Loss Component Analysis

### 1. Reconstruction Loss (CRITICAL ISSUE) ❌

**Trend**: 0.2010 → 0.3717 (↑85% increase)

**What this means**:
- The model is getting **progressively worse** at reconstructing control cell gene expression
- This is the **most concerning** finding - basic VAE functionality is degrading

**Why this is happening**:
```
Loss_total = 1.0 × L_rec + 0.01 × L_KL + 1.0 × L_MCC + 0.01 × L_sparse + 1.0 × L_MMD + 0.1 × L_centroid
```

The model is optimizing:
- **Alignment (MCC)**: Weight = 1.0 (equal to reconstruction!)
- **MMD**: Weight = 1.0 (equal to reconstruction!)
- **Reconstruction**: Weight = 1.0

**Problem**: Three equally-weighted objectives are competing, and reconstruction is losing.

**Evidence**:
```
Epoch  1: Rec=0.201, MCC=-0.736, MMD=0.000 → Total=-0.351
Epoch 50: Rec=0.372, MCC=-0.990, MMD=0.149 → Total=-0.215
```
The model **sacrificed** reconstruction (+0.171) to **improve** alignment (+0.254) and achieve MMD matching.

---

### 2. KL Divergence Loss (CRITICAL ISSUE) ❌

**Trend**: 0.0458 → 0.6267 (↑1268% increase)

**What this means**:
- The latent distribution is **diverging far** from the standard normal N(0,I)
- The posterior q(z|x) is becoming very different from the prior p(z)

**Why this is happening**:

The KL loss formula:
```
KL = -0.5 × mean(1 + log(σ²) - μ² - σ²)
```

High KL (0.627) indicates either:
1. **Large means (μ)**: Latent vectors are far from origin
2. **Large variances (σ²)**: Uncertain/spread-out latent representations
3. **Small variances**: Over-confident/collapsed posterior

**Why this matters**:
- High KL suggests the model is **encoding a LOT of information** in the latent space
- This is **compensating for poor reconstruction** - the model is "hiding" information in the latent structure rather than learning good decoders
- Beta schedule (0 → 0.01) is **too weak** to constrain this

**Evidence**:
```
Epoch 10: KL=0.139 (beta=0.000) - unconstrained
Epoch 20: KL=0.172 (beta=0.0025) - slight constraint
Epoch 50: KL=0.627 (beta=0.010) - still growing despite 0.01 weight!
```

---

### 3. Alignment Loss / Partial-MCC (ACTUALLY GOOD!) ✓

**Trend**: -0.7357 → -0.9904 (↑34% improvement, more negative = better)

**What this means**:
- The latent dimensions are **increasingly well-aligned** with TF community eigengenes
- Negative values indicate **positive correlation** (the loss is negative mean partial correlation)
- Getting closer to -1.0 is GOOD

**Why this looks "wrong"**:
- Partial-MCC is **correctly formulated** as negative mean partial correlation
- **More negative = better alignment**
- The "increase" in absolute value is actually **improvement**

**Progression**:
```
Epoch  1: MCC=-0.736  (73.6% alignment)
Epoch 10: MCC=-0.994  (99.4% alignment) ← Peak performance
Epoch 50: MCC=-0.990  (99.0% alignment) ← Slight degradation
```

**Verdict**: Alignment is working **very well**. The model successfully learned the community structure.

---

### 4. MMD Loss (GOOD) ✓

**Trend**: 0.000 → 0.1488 (increases as schedule ramps up)

**What this means**:
- MMD measures distribution difference between **simulated** and **real** interventions
- Final value of 0.149 is **reasonable** (not perfect, but acceptable)

**Why this pattern**:
- Alpha schedule: 0 (epochs 1-5) → 1.0 (epoch 50)
- Model only starts caring about MMD after epoch 5

**Evidence**:
```
Epoch  1: MMD=0.000 (alpha=0.0) - not optimized yet
Epoch 10: MMD=0.891 (alpha=0.111) - initial spike as schedule turns on
Epoch 20: MMD=0.317 (alpha=0.333) - learning to match
Epoch 50: MMD=0.149 (alpha=1.0) - stabilized at reasonable level
```

**Interpretation**:
- The spike at epoch 10 (MMD=0.891) shows the model initially **struggled** when MMD was activated
- It then **learned** to reduce MMD while maintaining alignment
- Final MMD=0.149 means simulated interventions are **reasonably close** to real ones

---

### 5. Sparsity Loss (EXPECTED BEHAVIOR) ✓

**Trend**: 0.000 → 0.0167 (↑ as model learns structure)

**What this means**:
- Sparsity loss = L1 norm of upper-triangular G (the causal matrix)
- Increasing value means the model is **filling in** causal connections

**Why this is OK**:
- Starting at 0: G is initialized to zeros
- Ending at 0.0167: Model learned ~1.67 total weight in causal edges
- With lambda_sparse=0.02, the model is **penalized** for non-zero edges
- The fact that it still learned edges means they're **important** for the loss

---

## Root Cause Analysis

### The Core Problem: **Competing Objectives**

The total loss function has **three major competing objectives** with equal weights:

```python
Loss_total = 1.0 × L_rec              # Reconstruct control cells
           + 1.0 × L_MCC              # Align with eigengenes
           + 1.0 × L_MMD              # Match interventions
           + 0.01 × L_KL              # Regularize latent (very weak)
           + 0.01 × L_sparse          # Sparsify causal graph
           + 0.1 × L_centroid         # Match intervention centroids
```

**The trade-off**:
1. **Alignment (MCC)** wants latent space to match eigengene structure → Restricts latent flexibility
2. **MMD** wants to match intervention effects → Requires expressive latent space
3. **Reconstruction** wants accurate gene expression → Needs decoder capacity
4. **KL** is too weak (0.01) to constrain the latent space effectively

**What happened**:
- Model prioritized alignment (-0.736 → -0.990, huge improvement)
- Model achieved decent MMD matching (0.149)
- Model sacrificed reconstruction quality (0.201 → 0.372) to achieve above
- High KL (0.627) indicates model is "cheating" by encoding information densely in latent space

---

## Why Alignment and MMD Look "Random" to You

**Your observation**: "All components except MCC and MMD seem to take higher values or evolve randomly"

**The truth**:
- **MCC and MMD are actually the most "well-behaved"** - they show clear learning curves
- **Reconstruction and KL look "broken"** because they're increasing (bad direction)

**Why this perception**:
1. **You expected all losses to decrease** (standard ML intuition)
2. **But gCRL-VAE is a multi-objective optimization** - not all losses should decrease
3. **MCC and MMD are the primary goals** of gCRL-VAE (they SHOULD improve)
4. **Reconstruction is being sacrificed** (this is the problem)

---

## Is This a Bug or Design Issue?

### Not a Bug ❌
- All losses are computed correctly
- Schedules are working as designed
- Optimization is functioning properly

### Design Issue: Hyperparameters ✓
- **Alpha_rec (reconstruction weight)** should be **higher** (e.g., 5.0 instead of 1.0)
- **Beta_kld_max (KL weight)** should be **higher** (e.g., 0.05 instead of 0.01)
- **Lambda_mcc (alignment weight)** could be **lower** (e.g., 0.5 instead of 0.75)

---

## Recommendations

### IMMEDIATE ACTION (Critical)

#### 1. Increase Reconstruction Weight
```python
cfg.alpha_rec = 5.0  # Up from 1.0
```
**Rationale**: Make reconstruction 5× more important than alignment/MMD

#### 2. Increase KL Regularization
```python
cfg.beta_kld_max = 0.05  # Up from 0.01
```
**Rationale**: Constrain latent space more strongly to prevent information "hiding"

#### 3. Reduce Alignment Weight (Optional)
```python
cfg.lambda_mcc = 0.5  # Down from 0.75
```
**Rationale**: Allow more flexibility for reconstruction

### SUGGESTED NEW CONFIGURATION

```python
cfg = VAEConfig(
    outdir="results/Norman2019/VAE_v2",

    # ADJUSTED WEIGHTS
    alpha_rec=5.0,           # ← INCREASED (was 1.0)
    lambda_mcc=0.5,          # ← DECREASED (was 0.75)
    lambda_sparse=0.02,      # (unchanged)
    lambda_centroid=0.05,    # (unchanged)

    # SCHEDULED WEIGHTS
    beta_kld_max=0.05,       # ← INCREASED (was 0.01)
    alpha_mmd_max=1.0,       # (unchanged)

    # OTHER PARAMETERS
    batch_size=512,
    epochs=50,
    lr=2e-3,
    # ... rest unchanged
)
```

**Expected result**:
- Reconstruction loss should **decrease** (not increase)
- KL should stay **lower** (< 0.3)
- Alignment might be **slightly worse** (e.g., -0.95 instead of -0.99)
- MMD should remain **similar** (< 0.2)
- **Overall**: Better balance between objectives

---

## Understanding the Loss Landscape

### Epoch 10: The "MMD Shock"

```
Epoch  9: rec=0.245, kld=0.123, mcc=-0.994, mmd=0.000 (alpha_mmd=0.0)
Epoch 10: rec=0.248, kld=0.139, mcc=-0.994, mmd=0.891 (alpha_mmd=0.111)
```

**What happened**: MMD schedule activated, model suddenly realized interventions don't match

**Why MMD spiked**:
- For 9 epochs, model optimized reconstruction + alignment without caring about interventions
- At epoch 10, alpha_mmd became non-zero
- Model's predictions were WAY OFF from real interventions → huge MMD
- Model then spent 40 epochs learning to reduce MMD

**This is normal** for scheduled losses!

---

## Comparison to Expected Behavior

### Ideal Training Trajectory

```
Component    | Expected Trend       | Actual Trend        | Status
-------------|---------------------|---------------------|--------
Reconstruction| DECREASE (↓)       | INCREASE (↑)        | ❌ BAD
KL Divergence | STABLE (→)         | INCREASE (↑)        | ❌ BAD
Alignment     | IMPROVE (more neg) | IMPROVED            | ✓ GOOD
Sparsity      | INCREASE (↑)       | INCREASED           | ✓ GOOD
MMD           | DECREASE (↓)       | SPIKE then DECREASE | ✓ EXPECTED
Centroid      | DECREASE (↓)       | LOW                 | ✓ GOOD
```

**Key insight**: 2/6 components are behaving badly, but they're the **most important** ones (reconstruction and regularization).

---

## Auto-Configuration Limitations

The `analyze_dataset_and_suggest_config()` function made reasonable choices:
- Batch size: 512 ✓ (good for 8907 control cells)
- Epochs: 50 ✓ (reasonable)
- Beta_kld_max: 0.01 ✓ (standard for large datasets)
- Lambda_mcc: 0.75 ✓ (6 communities)

**But it didn't account for**:
- The Norman dataset has **very complex** gene-gene relationships
- 139 TFs → 2701 genes is a **53× expansion** (decoder must learn complex mappings)
- Polynomial decoder (linear + quadratic) may have **insufficient capacity**

**Suggested enhancement** to auto-config:
```python
# In analyze_dataset_and_suggest_config()
n_tfs = (adata.var['kind'] == 'TF').sum()
n_genes = adata.n_vars
expansion_ratio = n_genes / n_tfs

# If decoder must predict many more outputs than inputs, prioritize reconstruction
if expansion_ratio > 10:
    alpha_rec = 5.0  # Increase reconstruction weight
    beta_kld_max = 0.05  # Increase regularization
```

---

## Next Steps

### 1. Re-train with Adjusted Weights (RECOMMENDED)
- Use configuration above
- Monitor that reconstruction **decreases**
- Accept that alignment might be slightly worse (-0.95 vs -0.99)

### 2. Consider Architectural Changes (ADVANCED)
- Add decoder hidden layers: `decoder_hidden=(256, 128)`
- This increases capacity for complex TF→gene mappings
- But requires modifying `gcrl_vae.py`

### 3. Early Stopping Consideration
Looking at epoch 10-20, reconstruction was better (~0.25 vs 0.37 at epoch 50).
Consider stopping earlier or using a validation set to pick best checkpoint.

---

## Theoretical Context

### Why VAEs Can Have Increasing Reconstruction Loss

In standard VAE training, reconstruction should always decrease. But gCRL-VAE is **not a standard VAE**:

1. **Multi-objective optimization**: Trading off multiple competing goals
2. **Structured latent space**: Alignment constraint limits latent flexibility
3. **Intervention matching**: Model must work well on both seen and unseen (intervention) data
4. **Scheduled losses**: MMD/KL turn on mid-training, disrupting learned representations

This is similar to:
- **Conditional VAEs**: Where conditioning can hurt reconstruction
- **Disentangled VAEs**: Where disentanglement constraints hurt reconstruction
- **Adversarial training**: Where discriminator objective conflicts with generator

**The solution**: Explicitly prioritize reconstruction via weighting.

---

## Conclusion

**Summary**:
- ✓ Alignment is working beautifully (-0.99 correlation)
- ✓ MMD intervention matching is reasonable (0.15)
- ❌ Reconstruction is degrading (0.20 → 0.37, BAD)
- ❌ KL is exploding (0.05 → 0.63, BAD)

**Root cause**: Competing objectives with suboptimal weighting

**Solution**: Increase `alpha_rec` to 5.0 and `beta_kld_max` to 0.05

**Expected outcome**: Reconstruction improves at slight cost to alignment (~-0.95 instead of -0.99), which is an acceptable trade-off for a more robust model.
