#!/bin/bash
#
# Test Runner for MDM
# This script runs various test suites for MDM

set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
print_header() {
    echo -e "\n${YELLOW}=== $1 ===${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Parse arguments
RUN_UNIT=true
RUN_INTEGRATION=true
RUN_E2E=true
RUN_COVERAGE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --unit-only)
            RUN_INTEGRATION=false
            RUN_E2E=false
            shift
            ;;
        --integration-only)
            RUN_UNIT=false
            RUN_E2E=false
            shift
            ;;
        --e2e-only)
            RUN_UNIT=false
            RUN_INTEGRATION=false
            shift
            ;;
        --coverage)
            RUN_COVERAGE=true
            shift
            ;;
        --help)
            echo "MDM Test Runner"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --unit-only        Run only unit tests"
            echo "  --integration-only Run only integration tests"
            echo "  --e2e-only        Run only end-to-end tests"
            echo "  --coverage        Generate coverage report"
            echo "  --help           Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Start testing
echo -e "${YELLOW}"
echo "╔══════════════════════════════════════════════╗"
echo "║           MDM Test Suite                     ║"
echo "╚══════════════════════════════════════════════╝"
echo -e "${NC}"

# Run unit tests
if [ "$RUN_UNIT" = true ]; then
    print_header "Running Unit Tests"
    if pytest tests/unit/ -v; then
        print_success "Unit tests passed"
    else
        print_error "Unit tests failed"
        exit 1
    fi
fi

# Run integration tests
if [ "$RUN_INTEGRATION" = true ]; then
    print_header "Running Integration Tests"
    if pytest tests/integration/ -v; then
        print_success "Integration tests passed"
    else
        print_error "Integration tests failed"
        exit 1
    fi
fi

# Run E2E tests
if [ "$RUN_E2E" = true ]; then
    print_header "Running End-to-End Tests"
    
    # Create temporary test name
    E2E_TEST_NAME="test_e2e_$(date +%s)"
    
    if ./scripts/test_e2e_quick.sh "$E2E_TEST_NAME" ./data/sample > /tmp/e2e_test.log 2>&1; then
        print_success "E2E tests passed"
    else
        print_error "E2E tests failed"
        echo "Check log at: /tmp/e2e_test.log"
        exit 1
    fi
fi

# Generate coverage report
if [ "$RUN_COVERAGE" = true ]; then
    print_header "Generating Coverage Report"
    pytest --cov=mdm --cov-report=html --cov-report=term
    print_success "Coverage report generated in htmlcov/"
fi

# Summary
echo -e "\n${GREEN}"
echo "╔══════════════════════════════════════════════╗"
echo "║         All Tests Passed! ✓                  ║"
echo "╚══════════════════════════════════════════════╝"
echo -e "${NC}\n"