# Benchmark Version Comparison

This directory contains 10 different versions of the benchmark code, each implementing a different approach while maintaining the same core algorithm:

## Version 1: Original YDF BackwardSelectionFeatureSelector (default)
- Uses YDF's built-in `BackwardSelectionFeatureSelector`
- Feature selection happens **inside each CV fold**
- Each fold may select different features
- This is the current default behavior

**Run:** `python version_1.py -c titanic`

## Version 2: Custom backward selection with train/validation split
- Custom implementation of backward feature selection
- Uses a single train/validation split (80/20) for feature selection
- Removes features based on importance scores
- Includes early stopping with patience
- After selection, runs 5-fold CV on selected features

**Run:** `python version_2.py -c titanic`

## Version 3: Custom backward selection with CV inside (3-fold)
- Custom implementation with more robust feature evaluation
- Uses 3-fold CV to evaluate each feature set during selection
- More computationally expensive but more stable selection
- After selection, runs 5-fold CV on selected features

**Run:** `python version_3.py -c titanic`

## Version 4: Feature selection first, then CV evaluation
- Uses YDF's `BackwardSelectionFeatureSelector` on a validation set
- This is the "proper CV" approach from the original benchmark
- Selects features once, then evaluates with 5-fold CV
- Avoids overfitting from selecting features inside CV folds

**Run:** `python version_4.py -c titanic`

## Version 5: Functional Programming Style
- Same algorithm as Version 1 but using functional programming principles
- Pure functions, function composition, currying
- Immutable data patterns
- Same results but different implementation style

**Run:** `python version_5.py -c titanic`

## Version 6: Async/Await Implementation
- Same algorithm as Version 1 but using async/await patterns
- Asynchronous processing where possible
- Concurrent execution of independent operations
- Same results but with async programming style

**Run:** `python version_6.py -c titanic`

## Version 7: Pipeline-Based Implementation
- Same algorithm as Version 1 but organized as a processing pipeline
- Pipeline pattern with modular steps
- Clean separation of concerns
- Same results but with pipeline architecture

**Run:** `python version_7.py -c titanic`

## Version 8: State Machine Implementation
- Same algorithm as Version 1 but using state machine patterns
- Finite state machine for benchmark flow
- Event-driven processing
- Same results but with state machine design

**Run:** `python version_8.py -c titanic`

## Version 9: Decorator-Based Implementation
- Same algorithm as Version 1 but using decorator patterns
- Method decorators for cross-cutting concerns
- Enhanced functionality through decoration
- Same results but with decorator architecture

**Run:** `python version_9.py -c titanic`

## Version 10: Iterator-Based Implementation
- Same algorithm as Version 1 but using iterator patterns
- Lazy evaluation where possible
- Generator functions for data flow
- Iterator-based table updates
- Same results but with iterator design

**Run:** `python version_10.py -c titanic`

## Key Differences Summary

All versions use the same core algorithm:
- Feature selection with CV inside (3-fold) + tuning inside each fold
- Returns CV score from feature selection as final result (NO additional CV)
- Same table format (5 columns, live update)
- Same defaults: CV=3, removal_ratio=0.1, tuning=True

| Version | Implementation Style | Key Features | Purpose |
|---------|---------------------|--------------|----------|
| V1 | Object-Oriented | YDF BackwardSelectionFeatureSelector | Original reference implementation |
| V2 | Custom Selection | Custom backward selection with train/val split | Alternative selection method |
| V3 | Custom with CV | Custom selection with CV evaluation | Robust custom selection |
| V4 | Proper CV | Feature selection before CV | Clean separation approach |
| V5 | Functional Programming | Pure functions, immutable data | Functional programming demo |
| V6 | Async/Await | Asynchronous processing | Concurrency patterns demo |
| V7 | Pipeline | Modular pipeline steps | Pipeline architecture demo |
| V8 | State Machine | Event-driven state transitions | State machine patterns demo |
| V9 | Decorator | Method decoration patterns | Decorator architecture demo |
| V10 | Iterator | Lazy evaluation, generators | Iterator patterns demo |

## Running All Versions

To run all versions on a single competition:
```bash
for i in {1..10}; do
    echo "Running Version $i..."
    python version_$i.py -c titanic
done
```

## Output

Each version saves results to JSON files with the pattern:
- `v1_results_TIMESTAMP.json`
- `v2_results_TIMESTAMP.json`
- `v3_results_TIMESTAMP.json`
- `v4_results_TIMESTAMP.json`
- `v5_results_TIMESTAMP.json`
- `v6_results_TIMESTAMP.json`
- `v7_results_TIMESTAMP.json`
- `v8_results_TIMESTAMP.json`
- `v9_results_TIMESTAMP.json`
- `v10_results_TIMESTAMP.json`

Results include:
- Mean scores and standard deviations
- Number of features selected (except V5)
- Best features identified
- Improvement percentages
- Method description