# MDM Generic Features Benchmark

This directory contains scripts for benchmarking MDM's generic feature engineering capabilities using YDF (Yggdrasil Decision Forests) across multiple Kaggle competitions.

## Overview

The benchmark demonstrates the value of MDM's automatic feature engineering by comparing model performance with and without MDM's generic features across 12 Kaggle competitions covering various ML tasks:

- Binary Classification (Titanic, Loan Default, Depression Detection, Personality)
- Multi-class Classification (Obesity Level, Student Outcomes, Fertilizer)
- Multi-label Classification (Steel Defects)
- Regression (Abalone Age, Flood Probability, Insurance Premium, Sales)

## Installation

First, install MDM with ML dependencies:

```bash
# From MDM root directory
uv pip install -e ".[ml]"

# Or using pip
pip install -e ".[ml]"
```

This installs:
- `ydf` - Yggdrasil Decision Forests library
- `matplotlib` - For visualization

## Usage

### Basic Usage

Run benchmark on all competitions:

```bash
python scripts/ml/benchmark_generic_features.py
```

### Specific Competitions

Benchmark specific competitions only:

```bash
python scripts/ml/benchmark_generic_features.py --competitions titanic playground-s4e11
```

### Options

- `--competitions` / `-c`: List of competitions to benchmark
- `--output-dir` / `-o`: Output directory for results (default: `benchmark_results`)
- `--no-cache`: Do not use cached MDM datasets

### Example

```bash
# Run only classification tasks with custom output directory
python scripts/ml/benchmark_generic_features.py \
    -c titanic playground-s4e10 playground-s4e11 \
    -o my_results
```

## Results

The benchmark generates:

1. **JSON Results File**: `benchmark_results/benchmark_results_YYYYMMDD_HHMMSS.json`
   - Complete results for all competitions
   - Individual model scores with/without features
   - Improvement percentages
   - Summary statistics

2. **Console Output**: Rich-formatted progress and summary table

3. **Visualizations** (optional):
   ```bash
   python scripts/ml/utils/visualization.py benchmark_results/benchmark_results_*.json
   ```
   
   Generates:
   - `improvement_summary_*.png` - Bar chart of improvements
   - `feature_counts_*.png` - Feature count comparison
   - `metric_comparison_*.png` - Actual metric values
   - `summary_table_*.csv` - CSV summary

## Competition Details

| Competition | Problem Type | Target | Metric | Features |
|------------|--------------|--------|--------|----------|
| titanic | Binary Classification | Survived | Accuracy | 11 → 125+ |
| playground-s4e2 | Multi-class | NObeyesdad | Accuracy | 17 → 200+ |
| playground-s4e3 | Multi-label | 7 defects | ROC-AUC | 28 → 300+ |
| playground-s4e4 | Regression | Rings | RMSE | 8 → 100+ |
| playground-s4e5 | Regression | FloodProbability | MAE | 20 → 250+ |
| playground-s4e6 | Multi-class | Target | Accuracy | 36 → 400+ |
| playground-s4e10 | Binary | loan_status | ROC-AUC | 11 → 150+ |
| playground-s4e11 | Binary | Depression | Accuracy | 19 → 250+ |
| playground-s4e12 | Regression | Premium Amount | RMSE | 19 → 250+ |
| playground-s5e1 | Regression | num_sold | RMSE | Time series |
| playground-s5e6 | Multi-class | Fertilizer Name | Accuracy | 9 → 120+ |
| playground-s5e7 | Binary | Personality | Accuracy | 7 → 90+ |

## How It Works

1. **Dataset Registration**: Each competition is registered twice in MDM:
   - With features: `{competition}_features`
   - Without features: `{competition}_raw`

2. **Feature Engineering**: MDM automatically generates:
   - Statistical features (z-score, log, outliers, percentiles)
   - Temporal features (date components, cyclical encoding)
   - Categorical features (one-hot, frequency, target encoding)
   - Text features (length, word count, patterns)
   - Missing data features
   - And more...

3. **Model Training**: Uses YDF with 5-fold cross-validation:
   - Gradient Boosted Trees (GBT)
   - Random Forest (RF)
   - Default hyperparameters optimized for each task

4. **Evaluation**: Appropriate metrics for each problem type:
   - Classification: Accuracy, ROC-AUC
   - Regression: RMSE, MAE
   - Multi-label: Macro-averaged ROC-AUC

## Expected Results

MDM's generic features typically improve model performance by:
- **3-8%** for most competitions
- **10%+** for datasets with temporal, text, or high-cardinality features
- Minimal improvement for already well-engineered datasets

## Module Structure

```
scripts/ml/
├── benchmark_generic_features.py  # Main benchmark script
├── utils/
│   ├── competition_configs.py     # Competition metadata
│   ├── metrics.py                 # Metric implementations
│   ├── ydf_helpers.py            # YDF utilities
│   └── visualization.py          # Result visualization
├── competitions/                  # Symlink to competition data
├── benchmark_results/            # Output directory
├── BENCHMARK_PLAN.md             # Detailed implementation plan
└── README.md                     # This file
```

## Troubleshooting

### Memory Issues
Some competitions have large datasets. If you encounter memory issues:
- Process competitions individually
- Reduce CV folds: modify `n_splits=3` in the script
- Use a subset of features

### YDF Installation
If YDF installation fails:
```bash
# Install from specific version
pip install ydf==0.4.3

# Or install TensorFlow Decision Forests (alternative)
pip install tensorflow_decision_forests
```

### MDM Registration Errors
- Ensure train.csv exists in competition directory
- Check target column name matches configuration
- Use `--force` flag in MDM registration (already enabled)

## Contributing

To add new competitions:

1. Add competition folder to `/mnt/ml/competitions/`
2. Update `utils/competition_configs.py` with metadata
3. Ensure train.csv has proper format
4. Run benchmark on new competition

## Citation

If you use this benchmark, please cite:

```
MDM Generic Features Benchmark
https://github.com/hipotures/mdm
```