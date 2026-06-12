#!/bin/bash
# Helper script to run CellOracle Docker container with Jupyter Lab
# Usage: ./run_celloracle_jupyter.sh [port]
#   Default port: 8888
#
# The current gCRL directory is mounted at /workspace in the container

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
IMAGE_NAME="kenjikamimoto126/celloracle_ubuntu:0.18.0"
PORT="${1:-8888}"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}CellOracle Docker - Jupyter Lab${NC}"
echo "================================"
echo "Image: $IMAGE_NAME"
echo "Workspace: $SCRIPT_DIR"
echo "Port: $PORT"
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker daemon is not running"
    exit 1
fi

# Check if image exists
if ! docker images | grep -q "celloracle_ubuntu"; then
    echo "Error: CellOracle image not found. Please run:"
    echo "  docker pull $IMAGE_NAME"
    exit 1
fi

echo -e "${GREEN}Starting Jupyter Lab...${NC}"
echo -e "${YELLOW}Note: Copy the URL with token from the output below${NC}"
echo ""

# Run container with Jupyter Lab
docker run -it --rm \
    --name celloracle_jupyter \
    -p $PORT:8888 \
    -v "$SCRIPT_DIR:/workspace" \
    -w /workspace \
    $IMAGE_NAME \
    jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
