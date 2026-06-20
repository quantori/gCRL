
# gCRL

**gCRL** (graph-guided Causal Representation Learning) is a framework for learning structured latent representations of single-cell CRISPR perturbation data. It uses Gene Regulatory Networks (GRNs) as structural priors to align the latent space with biologically meaningful axes (eigengenes), enabling zero-shot generalisation to unseen combinations of perturbations.

Two model variants are provided: **gCRL-AE** (autoencoder) and **gCRL-VAE** (variational autoencoder).

---

## Installation

The project requires **two separate environments** that must be kept isolated because CellOracle has dependency conflicts with PyTorch.

| Environment | Purpose | Notebooks |
|---|---|---|
| `gcrl` conda environment | gCRL package, model training, evaluation | All notebooks **except** GRN calculation |
| CellOracle (Docker) | GRN inference | `1_GRN_calculation.ipynb` only |

### Environment 1 — gCRL (conda + pip)

**Prerequisites:** [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html), and a CUDA-capable GPU (recommended; CPU-only execution is possible but slow for model training).

```bash
# 1. Clone the repository
git clone <repository-url>
cd gCRL

# 2. Create and activate the conda environment
conda env create -f environment.yml
conda activate gcrl

# 3. Install the gCRL package in editable mode
pip install -e .
```

After installation, the package is importable as:

```python
import gcrl
from gcrl.models import gcrl_ae, gcrl_vae
```

> **GPU note:** The `environment.yml` installs a CPU-only PyTorch build by default. GPU support is strongly recommended for the modeling notebooks. To enable it, visit [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/), select your operating system and CUDA version, and run the generated `pip install torch ...` command after activating the `gcrl` environment. This will replace the CPU build with the correct CUDA-enabled wheel for your system.

### Environment 2 — CellOracle (Docker)

CellOracle is used exclusively for GRN inference (`1_GRN_calculation.ipynb` in each preprocessing pipeline) and must be run inside the provided Docker container.

**Prerequisites:** [Docker](https://docs.docker.com/get-docker/) installed and running.

```bash
# Pull the image (one-time, ~11 GB)
docker pull kenjikamimoto126/celloracle_ubuntu:0.18.0
```

Once the image is available, launch Jupyter Lab inside the container and open the GRN notebook in your browser. The entire repository should be mounted as a volume so that all notebooks, data, and outputs are directly accessible.

For setup and usage details refer to the [official CellOracle documentation](https://morris-lab.github.io/CellOracle.documentation/).

---

## Data

Raw data files are not included in the repository. Download instructions for each dataset are provided in the corresponding data folders:

- [`data/real/Joung2023/README.md`](data/real/Joung2023/README.md) — Joung et al. 2023 TF Atlas (GEO: GSE217460, GSE217066)
- `data/real/Norman2019/README.md` — Norman et al. 2019 (GEO: GSE133344)

---

## Repository layout

```
src/gcrl/                  # Python package (import gcrl)
  data/                    # IO & preprocessing utilities
  grn/                     # GRN communities and eigengene computation
  models/                  # gCRL-AE / gCRL-VAE (nn.Modules)
  training/                # training loops, schedulers, callbacks
  alignment/               # latent-to-eigengene alignment & partial-MCC
  evaluation/              # metrics and plotting
  utils/                   # seed, device, logging, config helpers

configs/                   # YAML configurations for experiments

notebooks/
  00_data_preprocessing/
    Joung2023_preprocessing/   # notebooks 0–3 for the Joung 2023 dataset
    Norman2019_preprocessing/  # notebooks 1–3 for the Norman 2019 dataset
  10_modeling_gcrl_ae/         # AE training and evaluation
  20_modeling_gcrl_vae/        # VAE training and evaluation

data/
  real/
    Joung2023/             # downloaded externally (see README therein)
    Norman2019/            # downloaded externally (see README therein)

results/                   # outputs produced by the modeling notebooks

tests/                     # unit tests
```

---

## Running the notebooks

Notebooks within each preprocessing pipeline should be executed in order (0 → 1 → 2 → 3). The notebook that requires the CellOracle Docker is indicated below.

### Joung 2023 (`notebooks/00_data_preprocessing/Joung2023_preprocessing/`)

| Notebook | Environment |
|---|---|
| `0_data_preprocessing.ipynb` | `gcrl` conda env |
| `1_GRN_calculation.ipynb` | **CellOracle Docker** |
| `2_TF_module_identification_50_1.0.ipynb` | `gcrl` conda env |
| `3_data_preparation.ipynb` | `gcrl` conda env |

### Norman 2019 (`notebooks/00_data_preprocessing/Norman2019_preprocessing/`)

| Notebook | Environment |
|---|---|
| `1_GRN_calculation.ipynb` | **CellOracle Docker** |
| `2_TF_module_identification_40_1.1.ipynb` | `gcrl` conda env |
| `3_data_preparation.ipynb` | `gcrl` conda env |

### Modeling (`notebooks/10_modeling_gcrl_ae/`, `notebooks/20_modeling_gcrl_vae/`)

All modeling notebooks use the `gcrl` conda environment.

---

## Citation

*Citation information will be added upon publication.*
