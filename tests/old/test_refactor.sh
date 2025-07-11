#!/bin/bash
# Test runner for refactoring code

cd /home/xai/DEV/mdm-refactor-2025
source .venv/bin/activate

echo "Running feature flags tests..."
python -m pytest tests/unit/core/test_feature_flags.py -v

echo -e "\nRunning metrics tests..."
python -m pytest tests/unit/core/test_metrics.py -v

echo -e "\nRunning A/B testing tests..."
python -m pytest tests/unit/core/test_ab_testing.py -v

echo -e "\nRunning comparison framework tests..."
python -m pytest tests/unit/testing/test_comparison.py -v

echo -e "\nRunning integration tests..."
python -m pytest tests/integration/test_parallel_development.py -v