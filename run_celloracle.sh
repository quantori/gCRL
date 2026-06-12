#!/bin/bash
# Helper script to run CellOracle Docker container
# Usage: ./run_celloracle.sh [command]
#   - No arguments: Start interactive bash session
#   - With arguments: Run specified command
#
# The current gCRL directory is mounted at /workspace in the container
# so you can access all files from the notebooks and data folders.

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
IMAGE_NAME="kenjikamimoto126/celloracle_ubuntu:0.18.0"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}CellOracle Docker Helper${NC}"
echo "=========================="
echo "Image: $IMAGE_NAME"
echo "Workspace: $SCRIPT_DIR (mounted at /workspace)"
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

# Run container with appropriate settings
if [ $# -eq 0 ]; then
    # Interactive mode
    echo -e "${GREEN}Starting interactive bash session...${NC}"
    echo "Tip: The celloracle_env conda environment is auto-activated"
    echo ""
    docker run -it --rm \
        --name celloracle_workspace \
        -v "$SCRIPT_DIR:/workspace" \
        -w /workspace \
        $IMAGE_NAME \
        /bin/bash
else
    # Command mode
    echo -e "${GREEN}Running command: $@${NC}"
    echo ""
    docker run --rm \
        --name celloracle_workspace \
        -v "$SCRIPT_DIR:/workspace" \
        -w /workspace \
        $IMAGE_NAME \
        /bin/bash -c "$@"
fi
