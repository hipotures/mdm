#!/bin/bash
#
# Demo End-to-End Test Script for MDM (With Colors)
# Usage: ./scripts/test_e2e_demo.sh <dataset_name>
#
# This script provides an interactive demonstration of MDM features
# with colored output and visual feedback.

set -e  # Exit on error

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <dataset_name>"
    echo "Example: $0 demo_dataset"
    echo ""
    echo "Note: This script uses sample data from ./data/sample"
    exit 1
fi

DATASET_NAME=$1
DATA_PATH="./data/sample"  # Default sample data
DELAY=4  # Longer delay for demo visibility

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Helper functions
print_section() {
    echo ""
    echo -e "${CYAN}================================================================${NC}"
    echo -e "${CYAN}==== ${YELLOW}$1${NC}"
    echo -e "${CYAN}================================================================${NC}"
}

print_command() {
    echo ""
    echo -e "${BLUE}>>> Running:${NC} ${GREEN}$1${NC}"
    echo -e "${BLUE}----------------------------------------------------------------${NC}"
}

run_command() {
    print_command "$1"
    eval "$1"
    local status=$?
    if [ $status -eq 0 ]; then
        echo -e "${GREEN}✓ Success${NC}"
    else
        echo -e "${RED}✗ Failed with status $status${NC}"
        exit $status
    fi
    sleep $DELAY
}

clear_screen() {
    sleep 2
    clear
}

# Start demo
clear
echo -e "${MAGENTA}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║              MDM - ML Data Manager Demo                      ║"
echo "║                                                              ║"
echo "║  This demo will showcase the complete dataset lifecycle      ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"
echo ""
echo -e "${YELLOW}Dataset:${NC} $DATASET_NAME"
echo -e "${YELLOW}Data Path:${NC} $DATA_PATH"
echo ""
echo -e "${GREEN}Press Enter to start the demo...${NC}"
read -r

# 1. System Information
clear_screen
print_section "System Information"
echo -e "${YELLOW}Let's start by checking the MDM version and configuration...${NC}"
sleep 2
run_command "mdm version"
run_command "mdm info"

echo ""
echo -e "${GREEN}Press Enter to continue...${NC}"
read -r

# 2. Dataset Registration
clear_screen
print_section "Dataset Registration"
echo -e "${YELLOW}Now we'll register a new dataset with MDM...${NC}"
sleep 2
run_command "mdm dataset register '$DATASET_NAME' '$DATA_PATH' --description 'Demo dataset for MDM showcase' --tags 'demo,showcase'"

echo ""
echo -e "${GREEN}Dataset registered successfully! Press Enter to continue...${NC}"
read -r

# 3. Dataset Information
clear_screen
print_section "Dataset Information"
echo -e "${YELLOW}Let's explore the dataset information...${NC}"
sleep 2
run_command "mdm dataset info '$DATASET_NAME'"
echo ""
echo -e "${YELLOW}And now with more details...${NC}"
sleep 2
run_command "mdm dataset info '$DATASET_NAME' --details"

echo ""
echo -e "${GREEN}Press Enter to continue...${NC}"
read -r

# 4. Dataset Statistics
clear_screen
print_section "Dataset Statistics"
echo -e "${YELLOW}MDM can compute statistics for your datasets...${NC}"
sleep 2
run_command "mdm dataset stats '$DATASET_NAME'"

echo ""
echo -e "${GREEN}Press Enter to continue...${NC}"
read -r

# 5. Search and Discovery
clear_screen
print_section "Search and Discovery"
echo -e "${YELLOW}You can search for datasets by name or tags...${NC}"
sleep 2
run_command "mdm dataset search '${DATASET_NAME:0:4}'"
run_command "mdm dataset search --tag 'demo'"

echo ""
echo -e "${GREEN}Press Enter to continue...${NC}"
read -r

# 6. Export Operations
clear_screen
print_section "Export Operations"
echo -e "${YELLOW}MDM supports exporting datasets to various formats...${NC}"
sleep 2

EXPORT_DIR="./demo_exports_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$EXPORT_DIR"

run_command "mdm dataset export '$DATASET_NAME' --output-dir '$EXPORT_DIR' --format csv"
run_command "mdm dataset export '$DATASET_NAME' --output-dir '$EXPORT_DIR' --format json"

echo ""
echo -e "${YELLOW}Exported files:${NC}"
find "$EXPORT_DIR" -type f -name "*.*" | while read -r file; do
    echo -e "  ${GREEN}✓${NC} $file"
done

echo ""
echo -e "${GREEN}Press Enter to continue...${NC}"
read -r

# 7. Update Operations
clear_screen
print_section "Update Operations"
echo -e "${YELLOW}You can update dataset metadata at any time...${NC}"
sleep 2
run_command "mdm dataset update '$DATASET_NAME' --description 'Updated demo dataset with new features'"
run_command "mdm dataset update '$DATASET_NAME' --tags 'demo,showcase,updated'"

echo ""
echo -e "${GREEN}Press Enter to continue...${NC}"
read -r

# 8. Dataset Removal
clear_screen
print_section "Dataset Removal"
echo -e "${YELLOW}Finally, let's remove the demo dataset...${NC}"
sleep 2
run_command "mdm dataset remove '$DATASET_NAME' --force"

# Cleanup
if [ -d "$EXPORT_DIR" ]; then
    rm -rf "$EXPORT_DIR"
fi

# Complete
clear_screen
echo -e "${MAGENTA}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    Demo Complete!                            ║"
echo "║                                                              ║"
echo "║  You've seen the complete MDM dataset lifecycle:             ║"
echo "║  • Registration with auto-detection                          ║"
echo "║  • Information and statistics display                        ║"
echo "║  • Search and discovery features                             ║"
echo "║  • Export to multiple formats                                ║"
echo "║  • Metadata updates                                          ║"
echo "║  • Dataset removal                                           ║"
echo "║                                                              ║"
echo "║  Ready to manage your ML datasets with MDM!                  ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"
echo ""