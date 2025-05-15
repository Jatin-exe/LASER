#!/bin/bash
set -e

# Constants
CONDA_DIR="$HOME/miniconda3"
ENV_NAME="hf_tools"
CONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"

# Step 1: Install Miniconda if not installed
if [ ! -d "$CONDA_DIR" ]; then
    echo "Miniconda not found. Installing..."
    wget "$CONDA_URL" -O miniconda.sh
    bash miniconda.sh -b -p "$CONDA_DIR"
    rm miniconda.sh
    "$CONDA_DIR/bin/conda" init bash
    echo "Miniconda installed at $CONDA_DIR"
else
    echo "Miniconda already installed."
fi

# Load conda into shell
source "$CONDA_DIR/etc/profile.d/conda.sh"

# Step 2: Create the 'hf_tools' environment if it doesn't exist
if ! conda info --envs | grep -q "^$ENV_NAME"; then
    echo "Creating conda environment '$ENV_NAME'..."
    conda create -y -n "$ENV_NAME" python=3.12
    echo "Environment '$ENV_NAME' created."
else
    echo "Environment '$ENV_NAME' already exists."
fi

# Step 3: Activate the environment
echo "Activating environment '$ENV_NAME'..."
conda activate "$ENV_NAME"
echo "Environment '$ENV_NAME' is now active."

# Step 4: Install huggingface_hub if not already installed
if ! python -c "import huggingface_hub" &> /dev/null; then
    echo "Installing huggingface_hub in '$ENV_NAME'..."
    pip install huggingface_hub[hf_xet]
    echo "huggingface_hub installed."
else
    echo "huggingface_hub already installed."
fi

# Step 5: Download from HF
echo "Running download_models.py..."
python scripts/download_models.py
echo "Dataset download complete."

# Step 6: Remove conda environment after download
echo "Cleaning up by removing '$ENV_NAME' environment..."
conda deactivate
conda env remove -n "$ENV_NAME" -y
echo "Environment '$ENV_NAME' removed."