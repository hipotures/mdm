"""YDF (Yggdrasil Decision Forests) helper functions."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.model_selection import KFold, StratifiedKFold
from .metrics import calculate_metric, needs_probabilities

try:
    import ydf
    HAS_YDF = True
except ImportError:
    HAS_YDF = False
    # Create placeholder classes to avoid import errors
    class YDFPlaceholder:
        Task = type('Task', (), {
            'CLASSIFICATION': 'CLASSIFICATION',
            'REGRESSION': 'REGRESSION'
        })()
        
        def GradientBoostedTreesLearner(self, **kwargs):
            raise ImportError("YDF not installed. Run: pip install ydf")
        
        def RandomForestLearner(self, **kwargs):
            raise ImportError("YDF not installed. Run: pip install ydf")
    
    ydf = YDFPlaceholder()


def create_learner(
    model_type: str,
    label: str,
    task: Optional[Any] = None,
    **kwargs
):
    """
    Create a YDF learner with default parameters.
    
    Args:
        model_type: 'gbt' for Gradient Boosted Trees or 'rf' for Random Forest
        label: Target column name
        task: YDF task (CLASSIFICATION, REGRESSION, etc.)
        **kwargs: Additional learner parameters
    
    Returns:
        YDF learner instance
    """
    default_params = {
        'label': label,
        'task': task
    }
    
    if model_type == 'gbt':
        # Default GBT parameters
        gbt_defaults = {
            'num_trees': 100,
            'max_depth': 6,
            'shrinkage': 0.1,
            'subsample': 0.8,
            'min_examples': 5,
            'use_hessian_gain': True,
            'growing_strategy': 'BEST_FIRST_GLOBAL'
        }
        gbt_defaults.update(kwargs)
        default_params.update(gbt_defaults)
        return ydf.GradientBoostedTreesLearner(**default_params)
    
    elif model_type == 'rf':
        # Default Random Forest parameters
        rf_defaults = {
            'num_trees': 100,
            'max_depth': 16,
            'min_examples': 5,
            'num_candidate_attributes_ratio': -1.0,  # sqrt(num_features)
            'bootstrap_training_dataset': True
        }
        rf_defaults.update(kwargs)
        default_params.update(rf_defaults)
        return ydf.RandomForestLearner(**default_params)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'gbt' or 'rf'")


def determine_task(problem_type: str):
    """Determine YDF task from problem type."""
    if 'classification' in problem_type:
        return ydf.Task.CLASSIFICATION
    elif problem_type == 'regression':
        return ydf.Task.REGRESSION
    else:
        raise ValueError(f"Unknown problem type: {problem_type}")


def cross_validate_ydf(
    df: pd.DataFrame,
    target: str,
    model_type: str,
    problem_type: str,
    metric_name: str,
    n_splits: int = 5,
    random_state: int = 42,
    learner_params: Optional[Dict[str, Any]] = None
) -> Tuple[float, float, List[float]]:
    """
    Perform cross-validation with YDF model.
    
    Args:
        df: Input DataFrame
        target: Target column name(s)
        model_type: 'gbt' or 'rf'
        problem_type: Problem type
        metric_name: Metric to calculate
        n_splits: Number of CV folds
        random_state: Random seed
        learner_params: Additional learner parameters
    
    Returns:
        Tuple of (mean_score, std_score, individual_scores)
    """
    # Handle multi-label case
    if problem_type == 'multilabel_classification':
        return cross_validate_multilabel(
            df, target, model_type, metric_name, n_splits, random_state, learner_params
        )
    
    # Prepare data
    X = df.drop(columns=[target])
    y = df[target]
    
    # Choose appropriate cross-validation strategy
    if 'classification' in problem_type:
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = kf.split(X, y)
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = kf.split(X)
    
    scores = []
    task = determine_task(problem_type)
    
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        # Split data
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        
        # Create and train model
        learner_params = learner_params or {}
        learner = create_learner(model_type, target, task, **learner_params)
        
        model = learner.train(train_df)
        
        # Make predictions
        if needs_probabilities(metric_name):
            # Get probabilities
            predictions = model.predict(val_df)
            if problem_type == 'binary_classification':
                # For binary classification, get probability of positive class
                if hasattr(predictions, 'probability'):
                    # If predictions have probability attribute
                    pred_proba = predictions.probability(1)  # Probability of class 1
                else:
                    # If predictions are already probabilities
                    pred_proba = predictions
                y_pred = pred_proba
            else:
                # Multi-class - get all probabilities
                y_pred = predictions
        else:
            # Get class predictions
            predictions = model.predict(val_df)
            # For classification, convert predictions to class labels
            if 'classification' in problem_type:
                # YDF returns predictions as array, we need integer classes
                y_pred = predictions.astype(int)
            else:
                y_pred = predictions
        
        # Calculate metric
        y_true = val_df[target].values
        score = calculate_metric(y_true, y_pred, metric_name, problem_type)
        scores.append(score)
    
    return np.mean(scores), np.std(scores), scores


def cross_validate_multilabel(
    df: pd.DataFrame,
    targets: List[str],
    model_type: str,
    metric_name: str,
    n_splits: int = 5,
    random_state: int = 42,
    learner_params: Optional[Dict[str, Any]] = None
) -> Tuple[float, float, List[float]]:
    """
    Cross-validation for multi-label classification.
    
    Trains separate models for each label.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_scores = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(df)):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        
        # Predictions for all labels
        all_predictions = {}
        y_true_all = []
        
        # Train model for each label
        for label_idx, label in enumerate(targets):
            learner_params = learner_params or {}
            learner = create_learner(
                model_type, label, ydf.Task.CLASSIFICATION, **learner_params
            )
            
            model = learner.train(train_df)
            
            # Get predictions
            if needs_probabilities(metric_name):
                predictions = model.predict(val_df)
                if hasattr(predictions, 'probability'):
                    pred_proba = predictions.probability(1)
                else:
                    pred_proba = predictions
                all_predictions[label] = pred_proba
            else:
                all_predictions[label] = model.predict(val_df)
            
            y_true_all.append(val_df[label].values)
        
        # Convert to arrays
        y_true = np.column_stack(y_true_all)
        y_pred = np.column_stack([all_predictions[label] for label in targets])
        
        # Calculate metric
        score = calculate_metric(y_true, y_pred, metric_name, 'multilabel_classification')
        fold_scores.append(score)
    
    return np.mean(fold_scores), np.std(fold_scores), fold_scores


def tune_hyperparameters(
    df: pd.DataFrame,
    target: str,
    model_type: str,
    problem_type: str,
    metric_name: str,
    n_trials: int = 50,
    n_splits: int = 3,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Tune hyperparameters using random search.
    
    Returns:
        Best hyperparameters found
    """
    # Define search space
    if model_type == 'gbt':
        search_space = {
            'num_trees': [50, 100, 200, 300],
            'max_depth': [4, 6, 8, 10, 12],
            'shrinkage': [0.05, 0.1, 0.15, 0.2],
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'min_examples': [5, 10, 20]
        }
    else:  # Random Forest
        search_space = {
            'num_trees': [50, 100, 200, 300, 500],
            'max_depth': [8, 12, 16, 20, -1],  # -1 means no limit
            'min_examples': [5, 10, 20],
            'num_candidate_attributes_ratio': [0.5, 0.7, 1.0, -1.0]  # -1 means sqrt
        }
    
    best_score = -np.inf if metric_name != 'rmse' else np.inf
    best_params = {}
    
    # Random search
    for trial in range(n_trials):
        # Sample hyperparameters
        params = {}
        for param, values in search_space.items():
            params[param] = np.random.choice(values)
        
        # Evaluate with cross-validation
        try:
            mean_score, _, _ = cross_validate_ydf(
                df, target, model_type, problem_type, metric_name,
                n_splits=n_splits, random_state=random_state,
                learner_params=params
            )
            
            # Check if better
            is_better = (
                (metric_name in ['rmse', 'mae'] and mean_score < best_score) or
                (metric_name not in ['rmse', 'mae'] and mean_score > best_score)
            )
            
            if is_better:
                best_score = mean_score
                best_params = params.copy()
        
        except Exception as e:
            # Skip failed trials
            continue
    
    return best_params


def get_feature_importance(
    model: Any,
    top_k: Optional[int] = None
) -> pd.DataFrame:
    """
    Get feature importance from trained model.
    
    Args:
        model: Trained YDF model
        top_k: Return only top K features
    
    Returns:
        DataFrame with feature names and importance scores
    """
    importance = model.variable_importances()
    
    if top_k is not None:
        importance = importance.head(top_k)
    
    return importance


def select_top_features(
    df: pd.DataFrame,
    model: Any,
    target: str,
    top_k: int = 50
) -> pd.DataFrame:
    """
    Select top K features based on importance.
    
    Args:
        df: Input DataFrame
        model: Trained YDF model
        target: Target column name
        top_k: Number of top features to select
    
    Returns:
        DataFrame with selected features and target
    """
    importance = get_feature_importance(model, top_k)
    top_features = importance['feature'].tolist()
    
    # Ensure target is included
    if target not in top_features:
        top_features.append(target)
    
    # Handle case where we have fewer features than requested
    available_features = [f for f in top_features if f in df.columns]
    
    return df[available_features]