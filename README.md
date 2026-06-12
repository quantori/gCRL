
# gCRL

**gCRL-AE** and **gCRL-VAE**: Causal Representation Learning with GRN priors, eigengene alignment (partial-MCC), and generalization to **zero-shot single-perturbation** and **double-perturbation**.

## Quick start

```bash
# 1) Clone the repo, cd into it, then scaffold folders
make init

# 2) Activate the deep_learning conda environment and install in editable mode
conda activate deep_learning
pip install -e .

# 3) For CellOracle-based GRN calculations (uses Docker)
./run_celloracle.sh  # Interactive shell
# OR
./run_celloracle_jupyter.sh  # Jupyter Lab
```

See [CELLORACLE_SETUP.md](CELLORACLE_SETUP.md) for CellOracle Docker usage details.

## Repository layout

```
src/gcrl/                  # Python package (import gcrl)
  data/                    # IO & preprocessing
  grn/                     # communities, eigengenes
  models/                  # gCRL-AE / gCRL-VAE (nn.Modules) + polynomial decoder
  training/                # training loops, schedulers, callbacks
  alignment/               # A = B X alignment & partial-MCC
  evaluation/              # metrics & plotting
  utils/                   # seed, device, logging, config

scripts/                   # CLI entrypoints (train, eigengenes, MCC, etc.)
configs/                   # YAML configs for experiments

notebooks/
  00_data_preprocessing/   # real data prep, QC, GRN analysis
  10_modeling_gcrl_ae/
  20_modeling_gcrl_vae/
  30_alignment/
  40_generalization/       # zero-shot & double-perturbation analyses
  90_figures_for_paper/

simulation/
  code/SERGIO/             # SERGIO and simulation scripts
  notebooks/
  generated_data/

data/
  example/                 # tiny subsets for tests
  real/                    # (large data via LFS/DVC or external)
  simulated/               # (large data via LFS/DVC or external)

results/                   # unified results directory (replaces 'experiments/')
  generalization/
    zero_shot_single/
    double_perturb/
  mcc_alignment/
  ablations/
  figures/
    main/
    supplementary/
  tables/

tests/                     # unit/integration tests (with tiny fixtures)
docs/                      # optional docs site (mkdocs/sphinx)
```

## Installation notes

- Distribution name is **`gCRL`**, import as **`gcrl`**:

```python
import gcrl
from gcrl.models import gcrl_ae, gcrl_vae
```

- Recommended Python ≥ 3.10
- For large `.h5ad`, `.pt`, `.npy` files use **Git LFS** or DVC. See `.gitattributes`.

## Reproducibility tips

- Use configs in `configs/` to standardize experiments.
- Keep heavy artifacts (models, big matrices) in LFS or an external store.
- Keep figure notebooks thin: load precomputed results from `results/` and render plots.

## Development Environments

This project uses two separate environments:

### 1. Main Environment: `deep_learning` (Conda)
- **Purpose**: gCRL package development, model training, evaluation
- **Python**: 3.10.18
- **PyTorch**: 2.7.1+cu118 (CUDA 11.8)
- **GPU**: Automatically detected and used when available
- **Activation**: `conda activate deep_learning`

### 2. CellOracle Environment: Docker Container
- **Purpose**: GRN calculations and preprocessing notebooks
- **Python**: 3.10.11
- **CellOracle**: 0.18.0
- **Usage**: `./run_celloracle.sh` or `./run_celloracle_jupyter.sh`
- **Documentation**: See [CELLORACLE_SETUP.md](CELLORACLE_SETUP.md)

The environments are isolated to avoid dependency conflicts between PyTorch/gCRL and CellOracle.

## Citation

Add a `CITATION.cff` when ready so GitHub can render citation info.
