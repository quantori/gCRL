# gCRL-VAE Development Documentation

This directory contains comprehensive documentation of code changes, analyses, and implementation decisions for the gCRL-VAE project. All files were created during development sessions to document important changes and findings.

## Directory Structure

### 📁 [01_configuration/](01_configuration/)
**Configuration and Hyperparameter Optimization**

Documentation about the v1 vs v2 configuration comparison and optimal hyperparameter selection.

- **[v1_vs_v2_detailed_analysis.md](01_configuration/v1_vs_v2_detailed_analysis.md)** - Complete analysis showing why v1 dominates v2 across all metrics
- **[v1_v2_quick_reference.md](01_configuration/v1_v2_quick_reference.md)** - Quick reference table for v1 vs v2 comparison
- **[v1_v2_timeline.md](01_configuration/v1_v2_timeline.md)** - Timeline explaining the configuration evolution
- **[optimal_config_guide.md](01_configuration/optimal_config_guide.md)** - Guide to using the optimal (v1) configuration
- **[recommendations.md](01_configuration/recommendations.md)** - Best practices and recommendations going forward
- **[autoconfig_update_v1_strategy.md](01_configuration/autoconfig_update_v1_strategy.md)** - How auto-config was updated to use v1 strategy
- **[autoconfig_v2_changes_deprecated.md](01_configuration/autoconfig_v2_changes_deprecated.md)** - ⚠️ DEPRECATED: v2 changes that didn't work
- **[test_autoconfig_update.py](01_configuration/test_autoconfig_update.py)** - Test script to verify auto-config generates v1 values

**Key Finding:** v1 configuration (alpha_rec=1.0, beta_kld_max=0.01, lambda_mcc=0.75) is superior to v2 across all major metrics. Balance matters more than aggressive individual weights.

---

### 📁 [02_model_architecture/](02_model_architecture/)
**Model Architecture Changes**

Documentation of structural changes to the gCRL-VAE model.

- **[notears_dag_implementation.md](02_model_architecture/notears_dag_implementation.md)** - Implementation of NOTEARS DAG acyclicity constraint
- **[notebook_updates.md](02_model_architecture/notebook_updates.md)** - Updates to training notebooks reflecting new architecture

**Key Changes:**
- Added NOTEARS-based differentiable DAG constraint (lambda_dag=1.0)
- Relaxed L1 sparsity from 0.01 to 0.001 (acts as tie-breaker)
- Full DAG transform (removed upper-triangular constraint)
- Late-onset regularization (epoch 10+ for sparsity and DAG)

---

### 📁 [03_training_analysis/](03_training_analysis/)
**Training Analysis and Results**

Analysis of training dynamics, loss components, and performance verification.

- **[norman_training_analysis.md](03_training_analysis/norman_training_analysis.md)** - Detailed analysis of Norman2019 training showing multi-objective trade-offs
- **[v1_verification_results.md](03_training_analysis/v1_verification_results.md)** - Verification of v1 configuration performance
- **[session_context.md](03_training_analysis/session_context.md)** - Context summary for continuing development (historical reference)

**Key Insights:**
- Multi-objective optimization creates trade-offs between reconstruction, alignment, and intervention matching
- Alignment (MCC) is critical for gCRL-VAE - should maintain lambda_mcc ≥ 0.75
- Training shows expected patterns: MMD spike when schedule activates, then convergence

---

### 📁 [04_evaluation/](04_evaluation/)
**Model Evaluation and Visualization**

Documentation for model evaluation including the UMAP-based prediction visualization tool.

- **[README.md](04_evaluation/README.md)** - Quick start guide for evaluation and visualization
- **[prediction_viz_guide.md](04_evaluation/prediction_viz_guide.md)** - Comprehensive user guide
- **[predict_method.md](04_evaluation/predict_method.md)** - Documentation of the updated predict() method
- **[implementation_summary.md](04_evaluation/implementation_summary.md)** - Technical implementation details
- **[example_output.txt](04_evaluation/example_output.txt)** - Example of visualization output

**Features:**
- Per-intervention UMAP plots with centroid tracking
- Per-cell-type aggregate plots
- Quantitative metrics (centroid distances) saved to CSV
- Automatic handling of single and double perturbations

---

### 📁 [05_technical_notes/](05_technical_notes/)
**Technical Explanations**

In-depth technical explanations of specific implementation decisions.

- **[centroid_distance_explained.md](05_technical_notes/centroid_distance_explained.md)** - Why centroid distances are computed in gene expression space, not UMAP space

**Key Concept:** Centroid distances are measured in n-dimensional gene expression space to match the training objective, while UMAP is only used for visualization.

---

### 📁 [00_archive/](00_archive/)
**Historical Files**

Older documentation files kept for historical reference.

- BUG_FIXES_NOTEBOOK.txt
- CHANGES_v2.txt
- FINAL_FIXES.txt
- IMPROVEMENTS_v3.txt

---

## Quick Navigation by Topic

### Want to understand the optimal configuration?
1. Start with [v1_v2_quick_reference.md](01_configuration/v1_v2_quick_reference.md)
2. Read [optimal_config_guide.md](01_configuration/optimal_config_guide.md)
3. See detailed analysis in [v1_vs_v2_detailed_analysis.md](01_configuration/v1_vs_v2_detailed_analysis.md)

### Want to understand training dynamics?
1. Read [norman_training_analysis.md](03_training_analysis/norman_training_analysis.md)
2. See verification in [v1_verification_results.md](03_training_analysis/v1_verification_results.md)

### Want to evaluate and visualize predictions?
1. Start with [04_evaluation/README.md](04_evaluation/README.md)
2. Read comprehensive guide: [prediction_viz_guide.md](04_evaluation/prediction_viz_guide.md)

### Want to understand model architecture?
1. Read [notears_dag_implementation.md](02_model_architecture/notears_dag_implementation.md)

---

## Key Takeaways

### 1. Configuration (MOST IMPORTANT)
**USE v1 configuration:**
```python
alpha_rec = 1.0         # Balanced reconstruction
beta_kld_max = 0.01     # Gentle KL regularization
lambda_mcc = 0.75       # Strong alignment (CRITICAL!)
alpha_mmd_max = 1.0     # Standard MMD weight
batch_size = 512
epochs = 50
lr = 2e-3
```

**Results:**
- Reconstruction: 0.197 (excellent)
- Alignment: -0.992 (near-perfect)
- KL: 0.640 (controlled)
- MMD: 0.155 (good intervention matching)

### 2. Model Architecture
- 7 loss components: reconstruction, KL, alignment (MCC), sparsity, DAG (NOTEARS), MMD, centroid
- NOTEARS constraint enforces DAG without structural restrictions
- Late-onset regularization (epoch 10+) allows early learning freedom

### 3. Training Philosophy
- **Alignment is sacred** - lambda_mcc ≥ 0.75 is critical for causal structure
- **Balance over strength** - Moderate balanced weights > aggressive individual weights
- **Gentle regularization** - Over-regularization backfires (v2 lesson)
- **Multi-objective trade-offs** - Cannot optimize all objectives perfectly

### 4. Evaluation
- Use prediction visualization to assess model quality
- Centroid distances measure accuracy in gene expression space
- Alignment (MCC) is more important than raw reconstruction loss

---

## Development History

### Phase 1: Initial Implementation
- Basic gCRL-VAE with auto-configuration
- Training showed reconstruction degradation

### Phase 2: v2 Optimization Attempt (FAILED)
- Increased alpha_rec to 5.0, beta_kld_max to 0.05
- Reduced lambda_mcc to 0.5
- Result: Performed worse on 4/5 metrics

### Phase 3: v1 Validation (SUCCESS)
- Discovered v1 was actually optimal
- Updated auto-config to use v1 strategy
- Documented why v2 failed

### Phase 4: Architecture Enhancements
- Added NOTEARS DAG constraint
- Relaxed sparsity penalty
- Full DAG transform without structural constraints

### Phase 5: Visualization System
- Implemented comprehensive UMAP-based visualization
- Prediction comparison with quantitative metrics
- Per-intervention and per-cell-type plots

---

## Contributing to Documentation

When adding new documentation:

1. **Choose the right directory:**
   - Configuration/hyperparameters → 01_configuration/
   - Model architecture changes → 02_model_architecture/
   - Training analysis → 03_training_analysis/
   - Evaluation/visualization → 04_evaluation/
   - Technical explanations → 05_technical_notes/

2. **Use descriptive filenames:**
   - Use lowercase with underscores
   - Include topic and key concept
   - Example: `notears_dag_implementation.md`

3. **Include context:**
   - Date of changes
   - Why the change was made
   - What was learned
   - Links to related files

4. **Update this README:**
   - Add entry to appropriate section
   - Update key takeaways if applicable

---

## Related Documentation

- **Main project README:** [../../README.md](../../README.md)
- **Source code:** [../../src/gcrl/](../../src/gcrl/)
- **Notebooks:** [../../notebooks/20_modeling_gcrl_vae/](../../notebooks/20_modeling_gcrl_vae/)
- **Results:** [../../results/real/Norman2019/VAE/](../../results/real/Norman2019/VAE/)

---

## Questions?

If you're unsure where to find information:
1. Check the [Quick Navigation](#quick-navigation-by-topic) section above
2. Look at the [Key Takeaways](#key-takeaways) for high-level insights
3. Browse the relevant directory based on your topic

---

**Last Updated:** November 14, 2025
**Maintained by:** Development team and Claude Code documentation
