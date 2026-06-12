# CellOracle Docker Setup

This document explains how to use the CellOracle Docker container for GRN (Gene Regulatory Network) calculations in the gCRL project.

## Overview

CellOracle is installed via Docker to avoid dependency conflicts with the main `deep_learning` environment. The Docker image (`kenjikamimoto126/celloracle_ubuntu:0.18.0`) includes:
- Ubuntu 20.04
- Python 3.10.11
- CellOracle 0.18.0 with all dependencies
- Jupyter Lab for notebook execution

## Quick Start

### 1. Run Python Commands

Execute CellOracle Python commands:
```bash
./run_celloracle.sh "python script.py"
```

### 2. Interactive Shell

Start an interactive bash session:
```bash
./run_celloracle.sh
```
Inside the container, you can run any Python commands or scripts. The `celloracle_env` conda environment is automatically activated.

### 3. Jupyter Lab (Recommended for Notebooks)

Start Jupyter Lab server:
```bash
./run_celloracle_jupyter.sh [port]
```
- Default port: 8888
- Access via: `http://localhost:8888` (copy the token from the terminal output)
- The gCRL directory is mounted at `/workspace`

To stop Jupyter Lab, press `Ctrl+C` twice.

## Usage Examples

### Run a Python Script
```bash
./run_celloracle.sh "python notebooks/00_data_preprocessing/Norman_preprocessing/GRN_calculation.py"
```

### Import CellOracle in Python
```bash
./run_celloracle.sh "python -c 'import celloracle; print(celloracle.__version__)'"
```

### Execute Notebook
```bash
# Start Jupyter Lab and open the notebook in your browser
./run_celloracle_jupyter.sh
```

## Technical Details

### Docker Image Info
- **Image**: `kenjikamimoto126/celloracle_ubuntu:0.18.0`
- **Size**: 11.2 GB
- **Base**: Ubuntu 20.04
- **Python**: 3.10.11
- **CellOracle**: 0.18.0

### Volume Mounting
The entire gCRL directory is mounted at `/workspace` in the container, so all your files (notebooks, data, scripts) are accessible.

### File Permissions
Files created in the container will have your user ID and group ID, so there are no permission issues.

## Troubleshooting

### Container Already Running
If you see an error about the container name already being in use:
```bash
docker stop celloracle_workspace
# or
docker stop celloracle_jupyter
```

### Check Docker Status
```bash
docker info
```

### Verify Image is Available
```bash
docker images | grep celloracle
```

### Re-pull Image (if needed)
```bash
docker pull kenjikamimoto126/celloracle_ubuntu:0.18.0
```

## Integration with VSCode

You can also use the Docker container from VSCode:
1. Install the "Remote - Containers" extension
2. Attach to the running container
3. Edit and run notebooks directly

## Notes

- The CellOracle environment is isolated from your main `deep_learning` environment
- No need to install CellOracle dependencies in your main environment
- All CellOracle-related work should be done via these Docker scripts
- The container is ephemeral (removed after exit) but your files persist in the mounted directory

## Reference

Official CellOracle documentation: https://morris-lab.github.io/CellOracle.documentation/
