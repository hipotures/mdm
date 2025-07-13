#!/usr/bin/env python3
"""
Cross-validation evaluator module for YDF models.
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Tuple, List, Any, Optional, Dict
from sklearn.model_selection import StratifiedKFold, KFold
import ydf
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.metrics import calculate_metric, needs_probabilities

from spinner_utils import IterationSpinner


def create_ydf_model(model_type: str, problem_type: str, label: str = 'target', **kwargs):
    """Create a YDF model based on type and problem."""
    if model_type == 'gbt':
        if problem_type in ['binary_classification', 'multiclass_classification']:
            return ydf.GradientBoostedTreesLearner(
                label=label,
                num_trees=kwargs.get('num_trees', 100),
                max_depth=kwargs.get('max_depth', 6),
                min_examples=kwargs.get('min_examples', 5),
                subsample=kwargs.get('subsample', 1.0),
                **{k: v for k, v in kwargs.items() if k not in ['num_trees', 'max_depth', 'min_examples', 'subsample']}
            )
        else:  # regression
            return ydf.GradientBoostedTreesLearner(
                label=label,
                task=ydf.Task.REGRESSION,
                num_trees=kwargs.get('num_trees', 100),
                max_depth=kwargs.get('max_depth', 6),
                min_examples=kwargs.get('min_examples', 5),
                subsample=kwargs.get('subsample', 1.0),
                **{k: v for k, v in kwargs.items() if k not in ['num_trees', 'max_depth', 'min_examples', 'subsample']}
            )
    elif model_type == 'rf':
        if problem_type in ['binary_classification', 'multiclass_classification']:
            return ydf.RandomForestLearner(
                label=label,
                num_trees=kwargs.get('num_trees', 100),
                max_depth=kwargs.get('max_depth', 16),
                min_examples=kwargs.get('min_examples', 5),
                **{k: v for k, v in kwargs.items() if k not in ['num_trees', 'max_depth', 'min_examples']}
            )
        else:  # regression
            return ydf.RandomForestLearner(
                label=label,
                task=ydf.Task.REGRESSION,
                num_trees=kwargs.get('num_trees', 100),
                max_depth=kwargs.get('max_depth', 16),
                min_examples=kwargs.get('min_examples', 5),
                **{k: v for k, v in kwargs.items() if k not in ['num_trees', 'max_depth', 'min_examples']}
            )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def evaluate_fold(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_type: str,
    problem_type: str,
    metric_name: str,
    hyperparams: Optional[Dict[str, Any]] = None,
    show_progress: bool = True
) -> Tuple[float, Any]:
    """
    Evaluate a single fold.
    
    Returns:
        Tuple of (score, trained_model)
    """
    spinner = IterationSpinner() if show_progress else None
    
    try:
        if show_progress and spinner:
            spinner.start(f"Training {model_type.upper()} model")
        
        # Create model with hyperparameters
        model_params = hyperparams or {}
        model = create_ydf_model(model_type, problem_type, label='target', **model_params)
        
        # Prepare training data
        train_df = X_train.copy()
        train_df['target'] = y_train
        
        # Train model
        trained_model = model.train(train_df)
        
        if show_progress and spinner:
            spinner.update("Making predictions")
        
        # Make predictions
        predictions = trained_model.predict(X_val)
        
        # Handle different YDF prediction formats
        if hasattr(predictions, 'iloc'):
            # DataFrame predictions (probabilities)
            if needs_probabilities(metric_name):
                if len(predictions.columns) == 2:  # Binary classification
                    y_pred = predictions.iloc[:, 1].values
                else:  # Multiclass or other
                    y_pred = predictions.values
            else:
                # Need class predictions
                if len(predictions.columns) > 1:
                    y_pred = predictions.idxmax(axis=1).values
                else:
                    y_pred = predictions.iloc[:, 0].values
        else:
            # Array-like predictions
            y_pred = predictions if isinstance(predictions, np.ndarray) else np.array(predictions)
        
        # Calculate metric using utility function
        score = calculate_metric(y_val.values, y_pred, metric_name, problem_type)
        
        if show_progress and spinner:
            spinner.stop(f"Score: {score:.4f}")
        
        return score, trained_model
        
    except Exception as e:
        if show_progress and spinner:
            spinner.stop(f"Error: {str(e)}")
        raise e


def cross_validate_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str,
    problem_type: str,
    metric_name: str,
    n_splits: int = 3,
    hyperparams: Optional[Dict[str, Any]] = None,
    show_progress: bool = True
) -> Tuple[float, float, List[float], List[Any]]:
    """
    Perform cross-validation on a model.
    
    Args:
        X: Feature matrix
        y: Target vector
        model_type: 'gbt' or 'rf'
        problem_type: Problem type
        metric_name: Metric to evaluate
        n_splits: Number of CV folds
        hyperparams: Model hyperparameters
        show_progress: Whether to show progress
    
    Returns:
        Tuple of (mean_score, std_score, fold_scores, trained_models)
    """
    # Create cross-validation splitter
    if problem_type in ['binary_classification', 'multiclass_classification']:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_splits = list(cv.split(X, y))
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_splits = list(cv.split(X))
    
    fold_scores = []
    trained_models = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Evaluate fold
        score, model = evaluate_fold(
            X_train, y_train, X_val, y_val,
            model_type, problem_type, metric_name,
            hyperparams, show_progress
        )
        
        fold_scores.append(score)
        trained_models.append(model)
    
    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)
    
    return mean_score, std_score, fold_scores, trained_models


def hyperparameter_tune(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str,
    problem_type: str,
    metric_name: str,
    n_splits: int = 3,
    show_progress: bool = True
) -> Dict[str, Any]:
    """
    Simple hyperparameter tuning using grid search.
    
    Returns:
        Best hyperparameters found
    """
    spinner = IterationSpinner() if show_progress else None
    
    # Define parameter grids
    if model_type == 'gbt':
        param_grid = {
            'num_trees': [50, 100, 200],
            'max_depth': [4, 6, 8],
            'min_examples': [5, 10, 20],
            'subsample': [0.8, 1.0]
        }
    else:  # rf
        param_grid = {
            'num_trees': [50, 100, 200],
            'max_depth': [8, 16, 32],
            'min_examples': [5, 10, 20]
        }
    
    best_score = float('-inf') if metric_name not in ['rmse', 'mae'] else float('inf')
    best_params = {}
    
    # Simple grid search (could be improved with random search)
    param_combinations = [
        dict(zip(param_grid.keys(), values))
        for values in [
            [param_grid[key][i % len(param_grid[key])] for key in param_grid.keys()]
            for i in range(max(len(v) for v in param_grid.values()))
        ]
    ]
    
    # Test a few key combinations to keep it fast
    test_combinations = param_combinations[:6]  # Test 6 combinations max
    
    for i, params in enumerate(test_combinations):
        if show_progress and spinner:
            spinner.start(f"Tuning {i+1}/{len(test_combinations)}: {params}")
        
        try:
            mean_score, _, _, _ = cross_validate_model(
                X, y, model_type, problem_type, metric_name,
                n_splits=n_splits, hyperparams=params, show_progress=False
            )
            
            # Check if this is the best score
            is_better = (
                (metric_name in ['rmse', 'mae'] and mean_score < best_score) or
                (metric_name not in ['rmse', 'mae'] and mean_score > best_score)
            )
            
            if is_better:
                best_score = mean_score
                best_params = params.copy()
            
            if show_progress and spinner:
                spinner.update(f"Score: {mean_score:.4f}, Best: {best_score:.4f}")
            
        except Exception as e:
            if show_progress and spinner:
                spinner.update(f"Failed: {str(e)}")
            continue
    
    if show_progress and spinner:
        spinner.stop(f"Best params: {best_params}")
    
    return best_params if best_params else {}