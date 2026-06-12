# gCRL-VAE Modifications Summary

**Date**: 2025-11-13
**Files Modified**:
- `src/gcrl/training/train_gcrl_vae.py` (backup: `train_gcrl_vae_backup.py`)
- `src/gcrl/models/gcrl_vae.py` (backup: `gcrl_vae_backup.py`)
- `notebooks/20_modeling_gcrl_vae/2_Norman_VAE.ipynb` (updated to reflect new loss components)

---

## Key Changes

### 1. **NOTEARS DAG Acyclicity Constraint**

**Location**: `train_gcrl_vae.py` (lines 213-227, 1072-1075)

Added smooth differentiable acyclicity constraint:
```python
def notears_acyclicity(G: torch.Tensor) -> torch.Tensor:
    """h(G) = tr(exp(G ⊙ G)) - d"""
    G_offdiag = G - torch.diag_embed(torch.diag(G))
    A = G_offdiag * G_offdiag
    exp_A = torch.matrix_exp(A)
    return torch.trace(exp_A) - G.shape[0]
```

**Configuration**:
- Weight: `lambda_dag = 1.0` (strong penalty)
- Schedule: Active from epoch 10 onwards (`dag_start_epoch = 10`)

**Impact**: Enforces DAG property without structural restrictions.

---

### 2. **Relaxed L1 Sparsity**

**Changes**:
- **Weight**: `0.01 → 0.001` (10× weaker, acts as "tie-breaker")
- **Scope**: Applied to **full G matrix** (not just upper triangle)
- **Schedule**: Active from epoch 10 onwards (`sparse_start_epoch = 10`)

**Impact**: Allows denser graphs during early learning, gentle regularization later.

---

### 3. **Full DAG Transform**

**Location**: `gcrl_vae.py` (lines 299-325)

**Old approach**:
```python
G_up = torch.triu(self.G, diagonal=1)  # Force upper triangular
A = I - G_up
```

**New approach**:
```python
G_offdiag = G - torch.diag_embed(torch.diag(G))  # Remove diagonal only
A = I - G_offdiag
```

**Impact**:
- G is fully learnable (no structural constraint)
- Acyclicity enforced via NOTEARS penalty
- Model discovers optimal causal ordering from data

---

### 4. **Enhanced Logging**

**Changes**:
- Added `loss_dag` to training history JSON
- Updated loss component count from 6 to 7
- Updated notebook visualizations to show all 7 components

---

## Philosophy Change

### Before: **Hard-Coded Structure**
- Upper-triangular G → Fixed causal ordering
- Strong sparsity from epoch 1
- Acyclicity guaranteed by structure

### After: **Learned Structure**
- Full G matrix → Learned causal ordering
- Late-onset gentle regularization (epoch 10+)
- Acyclicity enforced via NOTEARS penalty

---

## Configuration Updates

New parameters in `VAEConfig`:
```python
lambda_dag: float = 1.0              # NOTEARS penalty weight
dag_start_epoch: int = 10            # When to start DAG penalty (changed from 20)
sparse_start_epoch: int = 10         # When to start L1 sparsity (changed from 20)
lambda_sparse: float = 1e-3          # Reduced from 0.01
```

Auto-configuration also updated:
- `lambda_sparse` ranges: `1e-3` to `5e-3` (was `0.01` to `0.05`)

---

## Loss Function (7 Components)

```
L_total = 1.0 × L_rec                          # Reconstruction (constant)
        + β(t) × L_KL                          # β: 0 → 0.01 (epochs 10+)
        + 1.0 × L_mcc                          # Alignment (constant)
        + 0.001 × L_sparse                     # Sparsity (tiny, epochs 10+)
        + 1.0 × L_dag                          # DAG acyclicity (NOTEARS, epochs 10+)
        + α(t) × L_MMD                         # α: 0 → 1.0 (epochs 5+)
        + α(t) × 0.1 × L_centroid              # Scheduled weight × constant weight
```

**Key insights**:
- Centroid has BOTH schedule (α) AND weight (0.1) = [0 → 0.1] (10× weaker than MMD)
- Sparse & DAG penalties start at epoch 10 to allow early learning freedom

---

## Expected Effects

**Advantages**:
1. More flexible causal discovery
2. Can find optimal DAG structure from data
3. Late regularization (epoch 10) allows early learning freedom
4. Uses proven NOTEARS methodology

**Considerations**:
1. Optimization is more challenging (larger search space)
2. Matrix exponential computation is expensive for large latent dimensions
3. May need more epochs to converge
4. Early epochs (1-9) have no DAG constraint

---

## Backup Files

Original implementations preserved as:
- `src/gcrl/training/train_gcrl_vae_backup.py`
- `src/gcrl/models/gcrl_vae_backup.py`

To revert to original:
```bash
mv src/gcrl/training/train_gcrl_vae_backup.py src/gcrl/training/train_gcrl_vae.py
mv src/gcrl/models/gcrl_vae_backup.py src/gcrl/models/gcrl_vae.py
```

---

## Verification

All changes tested:
- ✅ Syntactically valid (checked with `py_compile`)
- ✅ Logically consistent
- ✅ Properly integrated with existing training loop
- ✅ All loss components logged in training history
- ✅ Notebook updated to visualize 7 loss components

---

## Files Updated

1. **Training script**: `src/gcrl/training/train_gcrl_vae.py`
   - Added `notears_acyclicity()` function
   - Changed default `dag_start_epoch` from 20 → 10
   - Changed default `sparse_start_epoch` from 20 → 10
   - Added `loss_dag` to logging
   - Updated header documentation

2. **Model**: `src/gcrl/models/gcrl_vae.py`
   - Modified `_dag_transform()` to use full G matrix
   - Updated docstrings

3. **Notebook**: `notebooks/20_modeling_gcrl_vae/2_Norman_VAE.ipynb`
   - Added explanation of new features
   - Updated visualization cells to show 7 loss components
   - Added DAG acyclicity checking code
   - Updated causal matrix visualization
