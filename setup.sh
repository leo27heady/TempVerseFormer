#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status

REPO_URL="https://github.com/leo27heady/simple-shape-dataset-toolbox.git"
DESTINATION="./tempverse/shape_data_stream/simple_shape_rotation"

REPO_NAME=$(basename "$REPO_URL" .git)
VENV_DIR="venv"

# Clone the repository
git clone "$REPO_URL"

# Define source path
SOURCE_PATH="$REPO_NAME/modules"

# Check if 'modules' directory exists
if [ ! -d "$SOURCE_PATH" ]; then
    echo "Error: 'modules' folder not found in repository."
    rm -rf "$REPO_NAME"
    exit 1
fi

# Ensure destination directory exists
mkdir -p "$DESTINATION"

# Copy contents of 'modules' folder
cp -r "$SOURCE_PATH"/* "$DESTINATION"/

echo "Repository cloned, 'modules' copied."

# Create virtual environment if it does not exist
if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists. Please remove it if you want a fresh one."
    exit 1
fi


# Windows
python -m venv "$VENV_DIR"
source "$VENV_DIR/Scripts/activate"

# Linux
# python3 -m venv "$VENV_DIR"
# source "$VENV_DIR/bin/activate"

echo "Virtual environment created and activated."

# Install dependencies
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install einops==0.8.0 wandb==0.19.1 lpips==0.1.4 python-dotenv==1.0.1

# Install from requirements.txt if exists
REQ_FILE="$REPO_NAME/requirements.txt"
if [ -f "$REQ_FILE" ]; then
    pip install -r "$REQ_FILE"
fi

echo "All dependencies installed."

# Remove the cloned repository
rm -rf "$REPO_NAME"

echo "Repository removed."
