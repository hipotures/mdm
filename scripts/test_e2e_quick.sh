#!/bin/bash
#
# Quick End-to-End Test Script for MDM
# Usage: ./scripts/test_e2e_quick.sh <dataset_name> <path_to_data>
#
# This script runs a faster version of the e2e tests with reduced delays.

set -e  # Exit on error

# Check arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <dataset_name> <path_to_data>"
    echo "Example: $0 test_dataset ./data/sample"
    exit 1
fi

DATASET_NAME=$1
DATA_PATH=$2
DELAY=1  # Reduced delay for quick testing
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPORT_DIR="./test_exports_quick_${TIMESTAMP}"

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

cleanup() {
    if [ -d "$EXPORT_DIR" ]; then
        rm -rf "$EXPORT_DIR"
    fi
}

trap cleanup EXIT

# Quick test sequence
echo "MDM Quick E2E Test - $DATASET_NAME"
echo ""

print_section "Registration"
run_command "mdm dataset register '$DATASET_NAME' '$DATA_PATH'"

print_section "Info & Stats"
run_command "mdm dataset info '$DATASET_NAME'"
run_command "mdm dataset stats '$DATASET_NAME'"

print_section "Export"
mkdir -p "$EXPORT_DIR"
run_command "mdm dataset export '$DATASET_NAME' --output-dir '$EXPORT_DIR' --format csv"

print_section "Cleanup"
run_command "mdm dataset remove '$DATASET_NAME' --force"

echo ""
echo "Quick test completed successfully!"