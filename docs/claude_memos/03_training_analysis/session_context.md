# Context Summary: gCRL-VAE Training Analysis & Hyperparameter Tuning

## Quick Summary

We implemented 3 improvements to gCRL-VAE training, then discovered the v1 hyperparameters were actually better than the "improved" v2. This document provides all context for continuing the analysis.

---

## What We Implemented (Session 1)

### 1. GPU Detection & Confirmation ✓
- **File**: `src/gcrl/training/train_gcrl_vae.py:487-499`
- Displays GPU info at training start
- Shows: GPU name, memory, CUDA version
- Warns when training on CPU

### 2. Training Progress Bars ✓
- **File**: `src/gcrl/training/train_gcrl_vae.py:47-52, 592, 603, 761, 772-777`
- Dual-level progress tracking with tqdm
- Epoch-level and batch-level progress bars
- Real-time loss display

### 3. Intelligent Auto-Configuration ✓
- **File**: `src/gcrl/training/train_gcrl_vae.py:258-481`
- Function: `analyze_dataset_and_suggest_config()`
- Analyzes dataset characteristics and suggests hyperparameters
- Key insight: Added adaptive `alpha_rec` based on TF→gene expansion ratio

### 4. Bug Fix in Notebook ✓
- **File**: `notebooks/20_modeling_gcrl_vae/2_Norman_VAE.ipynb`
- Fixed: `model.z_dim` → `model.cfg.z_dim`
- Model config stored in `model.cfg`, not as direct attributes

---

## Training Analysis Results

### Dataset: Norman2019
- 8,907 control cells (training)
- 139 TFs → 2,701 genes (19.4× expansion ratio)
- 6 TF communities → 7 latent dimensions (p+1)
- 20 unique interventions (training)

### v1 Configuration (Original Auto-Config)
```python
alpha_rec = 1.0
beta_kld_max = 0.01
lambda_mcc = 0.75
alpha_mmd_max = 1.0
batch_size = 512
epochs = 50
lr = 2e-3
```

### v1 Training Results
```
Reconstruction:  0.201 → 0.372  (↑85%, WORSE)
KL Divergence:   0.046 → 0.627  (↑1268%, EXPLODED)
Alignment (MCC): -0.736 → -0.990 (↑34%, EXCELLENT - more negative = better)
MMD:             0.149           (REASONABLE)
Sparsity:        0.000 → 0.017   (EXPECTED)
```

**Issues identified**:
1. ❌ Reconstruction loss INCREASED (should decrease)
2. ❌ KL divergence EXPLODED (should stay <0.3)
3. ✓ Alignment improved to near-perfect (-0.99)
4. ✓ MMD reasonable (0.149)

**Diagnosis**: Multi-objective optimization trade-off problem. Model prioritized alignment and MMD at expense of reconstruction.

### v2 Configuration ("Improved" - But Apparently Wrong!)
```python
alpha_rec = 5.0     # ← INCREASED from 1.0 (5× stronger)
beta_kld_max = 0.05 # ← INCREASED from 0.01 (5× stronger)
lambda_mcc = 0.5    # ← REDUCED from 0.75 (allow flexibility)
alpha_mmd_max = 1.0 # (unchanged)
batch_size = 512    # (unchanged)
epochs = 50         # (unchanged)
lr = 2e-3           # (unchanged)
```

**Rationale for v2 changes**:
- Higher `alpha_rec`: Prioritize reconstruction (make it 5× more important)
- Higher `beta_kld_max`: Prevent KL explosion (constrain latent space)
- Lower `lambda_mcc`: Allow flexibility for reconstruction

**Expected v2 improvements**:
- Reconstruction should DECREASE
- KL should stay LOW (<0.3)
- Alignment may drop slightly (~-0.95 instead of -0.99, still excellent)

### v2 Training Results (NEED TO SEE THESE!)

**User says**: "v1 seems much better than v2!"

This suggests v2 may have:
- Worse reconstruction than expected?
- Worse alignment than acceptable?
- Some other unexpected behavior?

**CRITICAL**: We need to see v2's actual training history to understand what went wrong.

---

## Key Files & Locations

### Source Code
- **Training function**: `src/gcrl/training/train_gcrl_vae.py:484-783`
- **Auto-config function**: `src/gcrl/training/train_gcrl_vae.py:258-481`
- **Model architecture**: `src/gcrl/models/gcrl_vae.py`
- **Alignment loss**: `src/gcrl/alignment/partial_mcc.py`

### Notebooks
- **Main notebook**: `notebooks/20_modeling_gcrl_vae/2_Norman_VAE.ipynb`
- Has cells for both v1 and v2 training
- Has comparison visualization cell

### Training Results
- **v1 history**: `results/real/Norman2019/VAE/training_history.json`
- **v2 history**: `results/real/Norman2019/VAE_v2/training_history.json`
- **v1 model**: `results/real/Norman2019/VAE/norman_model.pt`
- **v2 model**: `results/real/Norman2019/VAE_v2/norman_model.pt`

### Documentation
- **Training analysis (v1)**: `docs/TRAINING_ANALYSIS_Norman.md`
- **Auto-config v2 changes**: `docs/AUTO_CONFIG_v2_CHANGES.md`
- **Improvements v3**: `docs/IMPROVEMENTS_v3.txt`
- **Bug fixes**: `docs/BUG_FIXES_NOTEBOOK.txt`

---

## Loss Function Explained

### Complete Loss (6 Components)
```python
Loss_total = alpha_rec × L_rec              # Reconstruction (controls only)
           + beta_kld × L_KL                # KL divergence (scheduled)
           + lambda_mcc × L_MCC             # Eigengene alignment (partial-MCC)
           + lambda_sparse × L_sparse       # Sparsity on causal matrix G
           + alpha_mmd × L_MMD              # Intervention matching (scheduled)
           + alpha_mmd × lambda_centroid × L_centroid  # Centroid loss (scheduled)
```

### Schedules
- **Beta (KL)**: 0 for epochs 1-10, then ramp 0 → beta_kld_max
- **Alpha (MMD)**: 0 for epochs 1-5, then ramp 0 → alpha_mmd_max
- **Temperature**: Linear 1.0 → 10.0 (for soft routing, not used in hard mode)

### Loss Interpretations
1. **L_rec** (MSE): Lower is better, should DECREASE over training
2. **L_KL**: Should be low (<0.3), measures distance from N(0,I)
3. **L_MCC**: Negative mean partial correlation, MORE NEGATIVE = BETTER (range: -1 to 0)
4. **L_sparse**: L1 norm of G, measures causal graph complexity
5. **L_MMD**: Distribution difference, lower = better intervention matching
6. **L_centroid**: Mean shift difference, lower = better

---

## The Multi-Objective Trade-Off Problem

### Why v1 Had Issues

With equal weights (alpha_rec=1.0, lambda_mcc=0.75, alpha_mmd=1.0):
```
Effective weights: Rec:MCC:MMD ≈ 1.0:0.75:1.0
```

The model optimized:
- Alignment: -0.736 → -0.990 (huge improvement)
- MMD: 0.000 → 0.149 (achieved matching)
- **But sacrificed**: Reconstruction 0.201 → 0.372 (got worse!)

### Why We Thought v2 Would Help

With new weights (alpha_rec=5.0, lambda_mcc=0.5, alpha_mmd=1.0):
```
Effective weights: Rec:MCC:MMD ≈ 5.0:0.5:1.0
```

We expected:
- Reconstruction: Should improve (5× more important)
- Alignment: May drop to -0.95 (still excellent)
- MMD: Should remain similar

### But User Says v1 is Better!

**Possible reasons v2 failed**:

1. **Over-regularization**: Beta_kld=0.05 too high, constraining latent space too much
2. **Under-alignment**: Lambda_mcc=0.5 too low, lost the structure needed for good representations
3. **Trade-off sweet spot**: v1 accidentally found the right balance
4. **Reconstruction not the right metric**: Maybe near-perfect alignment (-0.99) is more valuable than lower reconstruction loss
5. **Overfitting**: v2 might have overfit to reconstruction, hurting generalization

---

## Questions for Next Session

### 1. What Does "v1 is Better" Mean?
- Better reconstruction loss?
- Better alignment?
- Better MMD?
- Better overall performance on some evaluation metric?
- Better qualitatively (visualizations, predictions)?

### 2. What are v2's Actual Numbers?
Need to see:
```
v2 Reconstruction: X → Y
v2 KL: X → Y
v2 Alignment: X → Y
v2 MMD: X
```

### 3. Is There an Evaluation Metric?
- Do we have held-out test data?
- Intervention prediction accuracy?
- Downstream task performance?

### 4. What Should We Actually Optimize For?
- Is reconstruction loss the right objective?
- Should we prioritize alignment even if reconstruction suffers?
- Is there a "validation" metric to judge model quality?

---

## Hypotheses to Test

### Hypothesis 1: v1's High Alignment is Actually Good
- Near-perfect alignment (-0.99) might be essential for gCRL-VAE
- Poor reconstruction might be acceptable trade-off
- Test: Check if v1 makes better intervention predictions despite higher reconstruction loss

### Hypothesis 2: v2 Over-Corrected
- alpha_rec=5.0 might be too much
- Try alpha_rec=2.0 or 3.0 instead
- Try beta_kld_max=0.02 instead of 0.05

### Hypothesis 3: There's a Sweet Spot
- Neither v1 nor v2 is optimal
- Need to find balance between reconstruction and alignment
- Maybe alpha_rec=2.0, lambda_mcc=0.6, beta_kld_max=0.02

### Hypothesis 4: The Evaluation is Wrong
- We've been focusing on training losses
- The real metric is intervention prediction accuracy
- v1 might be better at this despite worse reconstruction

---

## Recommended Next Steps

### 1. Analyze v2 Results (URGENT)
```python
# Load and compare both histories
with open('results/real/Norman2019/VAE/training_history.json') as f:
    history_v1 = json.load(f)
with open('results/real/Norman2019/VAE_v2/training_history.json') as f:
    history_v2 = json.load(f)

# Compare final values
print("v1 final:", history_v1[-1])
print("v2 final:", history_v2[-1])
```

### 2. Visualize Comparison
The notebook has a comparison cell that plots v1 vs v2 side-by-side.

### 3. Evaluate on Interventions
```python
# Check how well each model predicts interventions
# Use model.predict_group() to simulate interventions
# Compare predictions to real held-out data
```

### 4. Try v3 (Middle Ground)
```python
cfg_v3 = analyze_dataset_and_suggest_config(adata, outdir="results/VAE_v3")
# Manual override with middle ground values
cfg_v3.alpha_rec = 2.0      # Less aggressive than v2's 5.0
cfg_v3.beta_kld_max = 0.02  # Less aggressive than v2's 0.05
cfg_v3.lambda_mcc = 0.6     # Between v1's 0.75 and v2's 0.5
```

### 5. Check Literature
- What do other causal VAE papers optimize for?
- Is high KL actually acceptable if alignment is good?
- What's the typical reconstruction loss range for VAEs?

---

## Data for Next Conversation

### Ask User For:
1. **v2's training history** (the actual numbers)
2. **Why they think v1 is better** (specific criteria)
3. **Evaluation metrics** (how are models assessed?)
4. **Downstream task** (what will the model be used for?)

### Have Ready:
- v1 training history (already analyzed)
- Model architecture details
- Loss function formulas
- Auto-config logic

---

## Key Insights So Far

1. **Multi-objective optimization is hard**: No free lunch - improving one objective often hurts others
2. **Default weights matter**: v1 accidentally worked reasonably well
3. **Alignment is critical**: gCRL-VAE specifically designed for eigengene alignment
4. **Reconstruction might not be primary goal**: Unlike standard VAEs
5. **Need evaluation metrics**: Training losses alone don't tell the full story

---

## Open Questions

1. Why is v1 better? (Need specifics)
2. What's the right balance between objectives?
3. Should we even try to reduce reconstruction loss if it hurts alignment?
4. Is high KL (0.627) actually a problem if the model works well?
5. How should we evaluate gCRL-VAE models properly?

---

## Code Snippets for Next Session

### Load Both Histories
```python
import json
import pandas as pd

with open('results/real/Norman2019/VAE/training_history.json') as f:
    history_v1 = json.load(f)
with open('results/real/Norman2019/VAE_v2/training_history.json') as f:
    history_v2 = json.load(f)

df_v1 = pd.DataFrame(history_v1)
df_v2 = pd.DataFrame(history_v2)
```

### Compare Final Metrics
```python
print("=" * 60)
print("COMPARISON: v1 vs v2 (final epoch)")
print("=" * 60)
metrics = ['loss_rec', 'loss_kld', 'loss_mcc', 'loss_mmd', 'loss']
for metric in metrics:
    v1_val = df_v1[metric].iloc[-1]
    v2_val = df_v2[metric].iloc[-1]
    better = 'v1' if v1_val < v2_val else 'v2'
    if metric == 'loss_mcc':  # More negative is better
        better = 'v1' if v1_val < v2_val else 'v2'
    print(f"{metric:12} | v1: {v1_val:7.4f} | v2: {v2_val:7.4f} | Better: {better}")
```

### Try Middle Ground Config
```python
from gcrl.training.train_gcrl_vae import VAEConfig, train_gcrl_vae

cfg_v3 = VAEConfig(
    outdir="results/real/Norman2019/VAE_v3",
    alpha_rec=2.0,       # Middle ground
    beta_kld_max=0.02,   # Middle ground
    lambda_mcc=0.6,      # Middle ground
    batch_size=512,
    epochs=50,
    lr=2e-3,
)

model_v3, history_v3 = train_gcrl_vae(adata, cfg_v3)
```

---

## Final Note

The key question is: **What makes a gCRL-VAE model "good"?**

We initially thought:
- Low reconstruction loss
- Low KL divergence
- Good alignment
- Low MMD

But maybe the hierarchy is:
1. **Alignment** (most important - defines the latent structure)
2. **MMD** (important - validates intervention effects)
3. **KL** (moderate - prevents collapse but high values might be OK)
4. **Reconstruction** (less important - as long as model captures key patterns)

This would explain why v1 (great alignment, poor reconstruction) might be better than v2!

---

## Contact Points

- **User environment**: `/home/laganiv/Desktop/projects/CausalEmbed/grn_crl/gCAL/gCRL/`
- **Conda env**: `deep_learning`
- **GPU**: NVIDIA RTX A4500, 21GB
- **Python**: 3.10
- **PyTorch**: CUDA 11.8

---

## Session End Status

✅ Implemented 3 improvements (GPU info, progress bars, auto-config)
✅ Fixed notebook bug (model.cfg attribute access)
✅ Analyzed v1 training (found reconstruction degradation)
✅ Designed v2 with improved hyperparameters
✅ Created comprehensive documentation
❓ v2 training completed but results apparently worse than v1
❓ Need to understand why v1 is actually better
❓ Need proper evaluation metrics beyond training losses

**Next session should start with**: Analyzing v2 results and understanding what "better" means.
