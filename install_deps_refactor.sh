#!/bin/bash
# Install missing dependencies for refactoring tests

cd /home/xai/DEV/mdm-refactor-2025
source .venv/bin/activate

echo "Installing missing dependencies..."
pip install pytest-timeout deepdiff jsondiff

echo "Dependencies installed."