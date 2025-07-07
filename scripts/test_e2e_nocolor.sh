#!/bin/bash
#
# End-to-End Test Script for MDM (No Colors)
# Usage: ./scripts/test_e2e_nocolor.sh <dataset_name> <path_to_data>
#
# This script tests the complete lifecycle of dataset management in MDM
# without using terminal colors (suitable for CI/CD pipelines).

set -e  # Exit on error

# Check arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <dataset_name> <path_to_data>"
    echo "Example: $0 test_dataset ./data/sample"
    exit 1
fi

DATASET_NAME=$1
DATA_PATH=$2
DELAY=3  # Seconds between commands
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPORT_DIR="./test_exports_${TIMESTAMP}"

# Helper functions
print_section() {
    echo ""
    echo "================================================================"
    echo "==== $1"
    echo "================================================================"
}

run_command() {
    echo ""
    echo ">>> Running: $1"
    echo "----------------------------------------------------------------"
    eval "$1"
    local status=$?
    if [ $status -eq 0 ]; then
        echo ">>> Success"
    else
        echo ">>> Failed with status $status"
        exit $status
    fi
    sleep $DELAY
}

cleanup() {
    echo ""
    print_section "Cleanup"
    if [ -d "$EXPORT_DIR" ]; then
        echo "Removing export directory: $EXPORT_DIR"
        rm -rf "$EXPORT_DIR"
    fi
}

# Set trap for cleanup
trap cleanup EXIT

# Start testing
echo "MDM End-to-End Test"
echo "==================="
echo "Dataset: $DATASET_NAME"
echo "Data Path: $DATA_PATH"
echo "Timestamp: $TIMESTAMP"
echo ""

# 1. System Information
print_section "System Information"
run_command "mdm version"
run_command "mdm info"

# 2. Pre-registration Check
print_section "Pre-registration Check"
run_command "mdm dataset list"

# 3. Dataset Registration
print_section "Dataset Registration"
run_command "mdm dataset register '$DATASET_NAME' '$DATA_PATH' --description 'E2E test dataset' --tags 'test,e2e'"

# 4. Dataset Information
print_section "Dataset Information"
run_command "mdm dataset info '$DATASET_NAME'"
run_command "mdm dataset info '$DATASET_NAME' --details"

# 5. Dataset Statistics
print_section "Dataset Statistics"
run_command "mdm dataset stats '$DATASET_NAME'"

# 6. Search Operations
print_section "Search Operations"
run_command "mdm dataset search '${DATASET_NAME:0:4}'"
run_command "mdm dataset search --tag 'test'"

# 7. Export Operations
print_section "Export Operations"
mkdir -p "$EXPORT_DIR"

# Export to different formats
run_command "mdm dataset export '$DATASET_NAME' --output-dir '$EXPORT_DIR/json' --format json"
run_command "mdm dataset export '$DATASET_NAME' --output-dir '$EXPORT_DIR/parquet' --format parquet"
run_command "mdm dataset export '$DATASET_NAME' --output-dir '$EXPORT_DIR/csv' --format csv"

# List exported files
echo ""
echo ">>> Exported files:"
find "$EXPORT_DIR" -type f -name "*.*" | sort

# 8. Metadata Updates
print_section "Metadata Updates"
run_command "mdm dataset update '$DATASET_NAME' --description 'Updated description for E2E test'"
run_command "mdm dataset update '$DATASET_NAME' --display-name 'E2E Test Dataset'"
run_command "mdm dataset update '$DATASET_NAME' --tags 'test,e2e,updated'"

# Verify updates
run_command "mdm dataset info '$DATASET_NAME'"

# 9. Batch Operations
print_section "Batch Operations"
run_command "mdm batch export --pattern '${DATASET_NAME:0:4}*' --output-dir '$EXPORT_DIR/batch' --format csv"
run_command "mdm batch stats --pattern '*test*'"

# 10. Dataset Removal
print_section "Dataset Removal"
run_command "mdm dataset remove '$DATASET_NAME' --force"

# 11. Post-removal Check
print_section "Post-removal Check"
run_command "mdm dataset list"

# Summary
print_section "Test Summary"
echo "All tests completed successfully!"
echo "Dataset lifecycle tested: registration -> info -> search -> export -> update -> remove"
echo ""
echo "Test artifacts were saved to: $EXPORT_DIR"
echo "Log file available at: ~/.mdm/logs/mdm.log"