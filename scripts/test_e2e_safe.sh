#!/bin/bash
#
# Safe End-to-End Test Script for MDM
# Usage: ./scripts/test_e2e_safe.sh <dataset_name> <path_to_data>
#
# This script tests MDM functionality without removing datasets,
# making it safe to run on production data.

set -e  # Exit on error

# Check arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <dataset_name> <path_to_data>"
    echo "Example: $0 test_dataset ./data/sample"
    exit 1
fi

DATASET_NAME=$1
DATA_PATH=$2
DELAY=2
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPORT_DIR="./test_exports_safe_${TIMESTAMP}"

# Helper functions
print_section() {
    echo ""
    echo "==== $1 ===="
}

run_command() {
    echo ">>> $1"
    eval "$1"
    sleep $DELAY
}

# Start testing
echo "MDM Safe E2E Test (Non-destructive)"
echo "==================================="
echo "Dataset: $DATASET_NAME"
echo "Data Path: $DATA_PATH"
echo ""

# Check if dataset already exists
if mdm dataset info "$DATASET_NAME" >/dev/null 2>&1; then
    echo "Note: Dataset '$DATASET_NAME' already exists. Will skip registration."
    SKIP_REGISTER=1
else
    SKIP_REGISTER=0
fi

# 1. System Information
print_section "System Information"
run_command "mdm version"
run_command "mdm info"

# 2. Dataset Registration (if needed)
if [ $SKIP_REGISTER -eq 0 ]; then
    print_section "Dataset Registration"
    run_command "mdm dataset register '$DATASET_NAME' '$DATA_PATH' --description 'Safe E2E test dataset'"
else
    print_section "Dataset Already Registered"
    echo "Skipping registration..."
fi

# 3. Dataset Information (non-destructive)
print_section "Dataset Information"
run_command "mdm dataset info '$DATASET_NAME'"
run_command "mdm dataset stats '$DATASET_NAME'"

# 4. Search Operations (non-destructive)
print_section "Search Operations"
run_command "mdm dataset search '${DATASET_NAME:0:4}'"
run_command "mdm dataset list"

# 5. Export Operations (non-destructive)
print_section "Export Operations"
mkdir -p "$EXPORT_DIR"
run_command "mdm dataset export '$DATASET_NAME' --output-dir '$EXPORT_DIR' --format csv"

echo ""
echo "Exported files:"
find "$EXPORT_DIR" -type f -name "*.*" | sort

# 6. Batch Operations (non-destructive)
print_section "Batch Operations"
run_command "mdm batch stats --pattern '${DATASET_NAME:0:4}*'"

# Summary
print_section "Test Summary"
echo "Safe test completed successfully!"
echo "Dataset '$DATASET_NAME' was tested without modification or removal."
echo ""
echo "Test artifacts saved to: $EXPORT_DIR"
echo ""
echo "Note: To remove test exports, run:"
echo "  rm -rf $EXPORT_DIR"