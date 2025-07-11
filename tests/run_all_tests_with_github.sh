#!/bin/bash
# Run all MDM tests and create GitHub issues for failures

echo "MDM Test Runner with GitHub Integration"
echo "======================================="
echo ""

# Check if --no-dry-run is passed
DRY_RUN="--dry-run"
if [[ "$*" == *"--no-dry-run"* ]]; then
    DRY_RUN="--no-dry-run"
    echo "WARNING: Running in LIVE mode - GitHub issues WILL be created!"
else
    echo "Running in DRY RUN mode - no issues will be created"
fi
echo ""

# Run the unified test analyzer
# This runs ALL tests (unit, integration, e2e) and creates GitHub issues
./tests/analyze_test_failures.py \
    --scope all \
    --github \
    $DRY_RUN \
    --github-limit 50 \
    --output test-results-$(date +%Y%m%d-%H%M%S).json

# Check exit code
if [ $? -eq 0 ]; then
    echo ""
    echo "✓ All tests passed!"
else
    echo ""
    echo "✗ Some tests failed. Check the report and GitHub issues."
fi