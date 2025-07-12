# Universal ML Benchmarking Script for MDM Generic Features

## Overview
Create a comprehensive benchmarking script that demonstrates the value of MDM's generic feature engineering by comparing YDF (Yggdrasil Decision Forests) model performance with and without MDM's automatic features across 12 Kaggle competitions.

## Competitions to Benchmark
1. Titanic - Binary Classification (Survived)
2. playground-series-s4e2 - Multi-class Classification (NObeyesdad)
3. playground-series-s4e3 - TBD
4. playground-series-s4e4 - TBD
5. playground-series-s4e5 - TBD
6. playground-series-s4e6 - TBD
7. playground-series-s4e10 - Binary Classification (based on roc_auc metric)
8. playground-series-s4e11 - Binary Classification (Depression)
9. playground-series-s4e12 - Regression (Premium Amount)
10. playground-series-s5e1 - TBD
11. playground-series-s5e6 - TBD
12. playground-series-s5e7 - TBD

## Script Components

### 1. Competition Configuration (`utils/competition_configs.py`)
```python
COMPETITIONS = {
    'titanic': {
        'path': '/mnt/ml/competitions/Titanic',
        'target': 'Survived',
        'problem_type': 'binary_classification',
        'metric': 'accuracy',  # Also ROC-AUC
        'id_column': 'PassengerId'
    },
    # ... all 12 competitions
}
```

### 2. Main Workflow (`benchmark_generic_features.py`)
For each competition:
1. **Register dataset in MDM twice:**
   - With features: `mdm dataset register {name}_features path/train.csv --target {target}`
   - Without features: `mdm dataset register {name}_raw path/train.csv --target {target} --no-features`

2. **Load data using MDM API:**
   ```python
   # With generic features
   df_features = mdm.load_dataset(f"{comp}_features", split="train")
   
   # Without features (raw)
   df_raw = mdm.load_dataset(f"{comp}_raw", split="train")
   ```

3. **Train YDF models with cross-validation:**
   - Use `ydf.GradientBoostedTreesLearner` and `ydf.RandomForestLearner`
   - Implement 5-fold cross-validation
   - Apply hyperparameter tuning using YDF's tuning capabilities
   - Use feature selection (top K features based on importance)

4. **Support custom metrics:**
   - accuracy, roc_auc, rmse, mae, log_loss
   - Custom Kaggle competition metrics as needed

5. **Compare results:**
   - Store CV scores for both approaches
   - Calculate improvement percentage
   - Generate comprehensive report

### 3. YDF Implementation Details

**Cross-validation:**
```python
def cross_validate_ydf(df, target, n_splits=5, model_type='gbt'):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in kf.split(df):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        
        if model_type == 'gbt':
            learner = ydf.GradientBoostedTreesLearner(
                label=target,
                num_trees=100,
                max_depth=6,
                shrinkage=0.1
            )
        else:  # Random Forest
            learner = ydf.RandomForestLearner(
                label=target,
                num_trees=100
            )
        
        model = learner.train(train_df)
        predictions = model.predict(val_df)
        score = calculate_metric(val_df[target], predictions, metric)
        scores.append(score)
    
    return np.mean(scores), np.std(scores)
```

**Feature selection:**
```python
def select_top_features(model, df, k=50):
    importance = model.variable_importances()
    top_features = importance.head(k)['feature'].tolist()
    return df[top_features + [target]]
```

**Hyperparameter tuning:**
```python
tuner = ydf.RandomSearchTuner(
    num_trials=50,
    search_space={
        "num_trees": [50, 100, 200, 300],
        "max_depth": [4, 6, 8, 10, 12],
        "shrinkage": [0.05, 0.1, 0.15, 0.2],
        "subsample": [0.6, 0.8, 1.0]
    }
)
```

### 4. Output Structure

**Results JSON:**
```json
{
    "benchmark_date": "2025-01-12",
    "mdm_version": "1.0.1",
    "results": {
        "titanic": {
            "with_features": {
                "gbt": {"mean_score": 0.842, "std": 0.015, "n_features": 125},
                "rf": {"mean_score": 0.836, "std": 0.018, "n_features": 125}
            },
            "without_features": {
                "gbt": {"mean_score": 0.815, "std": 0.020, "n_features": 11},
                "rf": {"mean_score": 0.808, "std": 0.022, "n_features": 11}
            },
            "improvement": {
                "gbt": "+3.31%",
                "rf": "+3.47%"
            }
        },
        // ... other competitions
    },
    "summary": {
        "average_improvement": "+4.2%",
        "best_improvement": "playground-series-s4e12: +7.8%",
        "competitions_improved": 11,
        "competitions_no_change": 1
    }
}
```

**Console Output:**
```
MDM Generic Features Benchmark
==============================

Processing: Titanic
  - Registering datasets...
  - Training models with features (125 features)...
    ✓ GBT: 0.842 ± 0.015
    ✓ RF: 0.836 ± 0.018
  - Training models without features (11 features)...
    ✓ GBT: 0.815 ± 0.020
    ✓ RF: 0.808 ± 0.022
  - Improvement: GBT +3.31%, RF +3.47%
```

### 5. Additional Features

1. **Parallel processing** for multiple competitions (optional)
2. **Progress bars** using Rich
3. **Error handling** for failed datasets
4. **Caching** of registered datasets to avoid re-registration
5. **Detailed logs** for debugging
6. **Visualization** of results (matplotlib charts)

## Files to Create

1. `scripts/ml/benchmark_generic_features.py` - Main script
2. `scripts/ml/utils/competition_configs.py` - Competition metadata
3. `scripts/ml/utils/metrics.py` - Custom metric implementations
4. `scripts/ml/utils/ydf_helpers.py` - YDF utility functions
5. `scripts/ml/README.md` - Documentation

## Dependencies to Add
- `ydf` (Yggdrasil Decision Forests)
- `scikit-learn` (for metrics)
- `matplotlib` (for visualization)
- Already have: pandas, numpy, rich

## Implementation Steps
1. ✓ Save this plan to scripts/ml/BENCHMARK_PLAN.md
2. Analyze all competitions to determine targets and problem types
3. Create competition configuration module
4. Implement metrics module
5. Create YDF helper functions
6. Implement main benchmark script
7. Test with Titanic dataset
8. Run full benchmark
9. Generate report and visualizations