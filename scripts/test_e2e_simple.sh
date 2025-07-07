#!/bin/bash
#
# Simple End-to-End Test Script for MDM
# Usage: ./scripts/test_e2e_simple.sh <dataset_name> <path_to_data>
#
# Minimal script for basic testing with no colors or clear commands.

set -e  # Exit on error

if [ $# -lt 2 ]; then
    echo "Usage: $0 <dataset_name> <path_to_data>"
    exit 1
fi

DATASET_NAME=$1
DATA_PATH=$2

echo "MDM Simple E2E Test"
echo ""

# Register
echo ">>> mdm dataset register '$DATASET_NAME' '$DATA_PATH'"
mdm dataset register "$DATASET_NAME" "$DATA_PATH"

# Info
echo ""
echo ">>> mdm dataset info '$DATASET_NAME'"
mdm dataset info "$DATASET_NAME"

# Stats
echo ""
echo ">>> mdm dataset stats '$DATASET_NAME'"
mdm dataset stats "$DATASET_NAME"

# Export
echo ""
echo ">>> mdm dataset export '$DATASET_NAME' --output-dir './test_export' --format csv"
mdm dataset export "$DATASET_NAME" --output-dir "./test_export" --format csv

# Remove
echo ""
echo ">>> mdm dataset remove '$DATASET_NAME' --force"
mdm dataset remove "$DATASET_NAME" --force

# Cleanup
rm -rf ./test_export

echo ""
echo "Test completed."