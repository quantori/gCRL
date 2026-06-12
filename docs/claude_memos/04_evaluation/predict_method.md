# Updated `predict()` Method in gCRL-VAE

## Summary

The `predict_group()` method has been replaced with a more powerful `predict()` method that automatically generates predictions for all test interventions.

---

## Key Changes

### **Old API** (`predict_group`)
```python
predictions = model.predict_group(
    controls_train_adata=controls,
    intervention="GATA1",
    cell_type="erythroid",
    n_pred=100,
    rng=rng
)
```

**Problems:**
- Required manual iteration over each (cell_type, intervention) pair
- User had to manage control selection
- Single intervention only
- Verbose and error-prone

---

### **New API** (`predict`)
```python
predictions = model.predict(
    adata,
    set_key="set",
    intervention_key="intervention",
    cell_type_key="cell_type",
    seed=42
)
```

**Improvements:**
- ✅ Automatic processing of all test groups
- ✅ Handles both single and double perturbations
- ✅ Built-in control cell selection
- ✅ Returns unified AnnData with obs['set'] = 'prediction'
- ✅ TF-only input handling (if adata.var['kind'] == 'TF')

---

## How It Works

### **Input Requirements**

The input `adata` must have:

1. **Training controls**: cells with `obs['set'] == 'training'` and `obs['intervention']` in control labels
2. **Test set**: cells with `obs['set'] == 'test'` defining which interventions to predict
3. **Metadata columns**: `set`, `intervention`, `cell_type`

### **Processing Flow**

For each `(cell_type, intervention)` pair in the test set:

1. **Sample control cells** from training set (same cell type)
   - n_samples = number of test cells in this group
   - Sampling with replacement if needed

2. **Extract TF expression** (if `adata.var['kind'] == 'TF'` exists)
   - Automatically detects TF-only input requirement
   - Falls back to all genes if 'kind' column missing

3. **Build cell-type conditioning** (if `ct_dim > 0`)
   - One-hot encode cell type for conditional VAE

4. **Parse intervention string**
   - Single: `"GATA1"` → sets `c` vector
   - Double: `"GATA1+CEBPA"` → sets `c` and `c2` vectors
   - Handles up to 2 simultaneous perturbations

5. **Simulate intervention**
   ```
   Encode: X_ctrl_TF → μ
   Intervene: μ → [DAG transform with c, c2] → u
   Decode: u → X_pred
   ```

6. **Create prediction AnnData**
   - `obs['set']` = `"prediction"`
   - `obs['cell_type']` = actual cell type
   - `obs['intervention']` = intervention string
   - `obs['source_cell_idx']` = index of sampled control cell

7. **Concatenate all predictions** into single AnnData

---

## Usage Examples

### **Basic Usage**
```python
# After training
model, history = train_gcrl_vae(adata, cfg)

# Generate predictions for all test interventions
predictions = model.predict(adata, seed=42)

# Compare with real test data
test_real = adata[adata.obs['set'] == 'test']
test_pred = predictions

print(f"Generated {predictions.n_obs} predictions")
print(f"Interventions: {predictions.obs['intervention'].unique()}")
print(f"Cell types: {predictions.obs['cell_type'].unique()}")
```

### **Custom Column Names**
```python
predictions = model.predict(
    adata,
    set_key="split",
    intervention_key="perturbation",
    cell_type_key="celltype",
    seed=123
)
```

### **Access Predictions by Group**
```python
# Get predictions for specific intervention
gata1_pred = predictions[predictions.obs['intervention'] == 'GATA1']

# Get predictions for specific cell type
ery_pred = predictions[predictions.obs['cell_type'] == 'erythroid']

# Specific (cell_type, intervention) pair
mask = (predictions.obs['cell_type'] == 'erythroid') & \
       (predictions.obs['intervention'] == 'GATA1')
gata1_ery_pred = predictions[mask]
```

---

## Implementation Details

### **Double Perturbation Handling**

The method automatically detects double perturbations:

```python
# Intervention string: "GATA1+CEBPA"
parts = sorted(intervention_str.split("+"))  # ["CEBPA", "GATA1"]

# Build two intervention vectors
c = self._intervention_to_c(parts[0], n_pred)   # CEBPA
c2 = self._intervention_to_c(parts[1], n_pred)  # GATA1

# Apply both via DAG transform
u = self._dag_transform(mu, bc, csz, bc2, csz2)
```

### **Control Cell Sampling**

```python
# Match cell type
ct_controls = train_controls[
    train_controls.obs[cell_type_key] == cell_type
]

# Sample same number as test cells
n_pred = len(test_group)
sample_idx = rng.integers(0, ct_controls.n_obs, size=n_pred)
```

### **TF-Only Input Detection**

```python
if "kind" in adata.var.columns:
    tf_mask = adata.var["kind"] == "TF"
    tf_idx = np.where(tf_mask)[0]
    X_ctrl_input = X_ctrl[:, tf_idx]  # Extract TF expression
```

---

## Output Structure

The returned AnnData has:

```python
predictions.X              # (n_predictions, n_genes) - predicted expression
predictions.obs            # Metadata DataFrame:
  ├─ set                  # "prediction" for all cells
  ├─ cell_type            # Cell type of prediction
  ├─ intervention         # Intervention applied (e.g., "GATA1", "GATA1+CEBPA")
  └─ source_cell_idx      # Index of control cell used as baseline
predictions.var            # Same as input adata.var
```

---

## Comparison with Training MMD Logic

The prediction logic **matches the MMD simulation** used during training:

### **Training (MMD loss)**
```python
# Sample control cells
X_ctrl_sample_tf = sample_controls(n_samples)

# Encode
mu_ctrl, var_ctrl = model.encode(X_ctrl_sample_tf, ct=ct_vec)
z_ctrl = model.reparameterize(mu_ctrl, var_ctrl)

# Apply intervention
bc = hard_routing(c, tf_to_latent)
csz = c @ c_shift
u_sim = dag_transform(z_ctrl, bc, csz, bc2, csz2)

# Decode
x_sim = model.decode(u_sim, ct=ct_vec)

# Compare with real: MMD(x_sim, X_real_all)
```

### **Prediction (inference)**
```python
# Sample control cells
X_ctrl_sample = sample_controls(n_pred)

# Encode (use μ directly, no sampling)
enc = model.encode_batch(X_ctrl_input, ct=ct_vec)
mu = enc["mu"]

# Apply intervention (same DAG transform)
bc = hard_routing(c, tf_to_latent)
csz = c @ c_shift
u = dag_transform(mu, bc, csz, bc2, csz2)

# Decode (same decoder)
x_pred = model.decode_batch(u, ct=ct_vec)
```

**Key difference**: Prediction uses `μ` directly (deterministic), training samples `z ~ N(μ, σ²)` (stochastic).

---

## Error Handling

The method validates:

- ✓ Required columns exist (`set`, `intervention`, `cell_type`)
- ✓ Test set is not empty
- ✓ Training controls are available
- ✓ Model is properly wired (`tf_index`, `tf_to_latent` set)
- ✓ Each cell type has training controls

Warnings:
- If a cell type in test has no training controls → skips that group

---

## Migration Guide

### **Old Code (predict_group)**
```python
# Manual iteration
test_groups = test_adata.obs.groupby(['cell_type', 'intervention'])
predictions_list = []

for (ct, interv), group in test_groups:
    controls_ct = train_controls[train_controls.obs['cell_type'] == ct]
    pred = model.predict_group(
        controls_ct,
        intervention=interv,
        cell_type=ct,
        n_pred=len(group),
        rng=rng
    )
    predictions_list.append(pred)

predictions = ad.concat(predictions_list)
```

### **New Code (predict)**
```python
# Single call
predictions = model.predict(adata, seed=42)
```

---

## Next Steps

Now that predictions are generated, you can:

1. **Evaluate predictions** vs. ground truth test data
2. **Visualize** predicted vs. observed expression
3. **Compute metrics** (RMSE, correlation, etc.)
4. **Analyze** per-gene or per-intervention accuracy

This is where the new evaluation module comes in!
