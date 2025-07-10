#!/bin/bash
# Monitor parallel development progress

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Directories
RESULTS_DIR="$HOME/.mdm/comparison_results"
METRICS_DIR="$HOME/.mdm/metrics"
FLAGS_FILE="$HOME/.mdm/feature_flags.yaml"

clear
echo "=== MDM Parallel Development Monitor ==="
echo "Time: $(date)"
echo

# Feature flags status
echo "Feature Flags:"
if [ -f "$FLAGS_FILE" ]; then
    echo "  $(grep 'use_new_backend:' $FLAGS_FILE || echo 'use_new_backend: false')"
    echo "  $(grep 'use_new_registrar:' $FLAGS_FILE || echo 'use_new_registrar: false')"
    echo "  $(grep 'use_new_features:' $FLAGS_FILE || echo 'use_new_features: false')"
    echo "  Rollout percentages:"
    grep -A3 'rollout_percentage:' $FLAGS_FILE | tail -3 | sed 's/^/    /'
else
    echo "  No feature flags file found"
fi
echo

# Recent comparison results
echo "Recent Comparison Tests:"
if [ -d "$RESULTS_DIR" ]; then
    for f in $(ls -t "$RESULTS_DIR"/*.json 2>/dev/null | head -5); do
        if [ -f "$f" ]; then
            TEST_NAME=$(jq -r '.test_name' "$f" 2>/dev/null || echo "unknown")
            PASSED=$(jq -r '.passed' "$f" 2>/dev/null || echo "unknown")
            PERF_DELTA=$(jq -r '.performance_delta' "$f" 2>/dev/null || echo "0")
            
            if [ "$PASSED" = "true" ]; then
                STATUS="${GREEN}✅${NC}"
            else
                STATUS="${RED}❌${NC}"
            fi
            
            printf "  $STATUS %-30s Perf: %+6.1f%%\n" "$TEST_NAME" "$PERF_DELTA"
        fi
    done
else
    echo "  No comparison results found"
fi
echo

# Recent metrics
echo "Recent Metrics:"
if [ -d "$METRICS_DIR" ]; then
    LATEST_METRICS=$(ls -t "$METRICS_DIR"/*.json 2>/dev/null | head -1)
    if [ -f "$LATEST_METRICS" ]; then
        echo "  From: $(basename $LATEST_METRICS)"
        
        # Extract some key metrics
        TOTAL=$(jq -r '.summary.total_metrics' "$LATEST_METRICS" 2>/dev/null || echo "0")
        echo "  Total metrics collected: $TOTAL"
        
        # Show counters
        echo "  Key counters:"
        jq -r '.summary.counters | to_entries[] | "    \(.key): \(.value)"' "$LATEST_METRICS" 2>/dev/null | head -5
    else
        echo "  No metrics files found"
    fi
else
    echo "  No metrics directory found"
fi

echo
echo "Press Ctrl+C to exit"