#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status

REPO_URL="https://github.com/leo27heady/simple-shape-dataset-toolbox.git"

REPO_NAME=$(basename "$REPO_URL" .git)

# Clone the repository
git clone "$REPO_URL"

# Move to the repository
cd "$REPO_NAME"

# Install the shapekit library
pip install -e .

# Move back to the root
cd ..

# Remove the cloned repository
rm -rf "$REPO_NAME"

echo "shapekit library has successfully installed!"

# Install dependencies
pip install -r requirements.txt

echo "All dependencies installed."
