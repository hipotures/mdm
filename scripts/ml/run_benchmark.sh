#!/bin/bash
# Run MDM benchmark script with proper initialization

# Ensure we're in the MDM directory
cd "$(dirname "$0")/../.."

# Install ML dependencies if not already installed
echo "Checking ML dependencies..."
if ! python -c "import ydf" 2>/dev/null; then
    echo "Installing ML dependencies..."
    uv pip install -e ".[ml]"
fi

# Run the benchmark script using Python directly
# This ensures MDM is properly initialized
echo "Running benchmark..."
python -m scripts.ml.benchmark_generic_features "$@"