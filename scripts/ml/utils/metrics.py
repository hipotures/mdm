"""Metrics module for evaluating model performance."""

import numpy as np
from typing import Union, List, Callable, Dict, Any
from sklearn import metrics


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate accuracy score."""
    return metrics.accuracy_score(y_true, y_pred)


def roc_auc(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """Calculate ROC-AUC score for binary classification."""
    if len(np.unique(y_true)) > 2:
        # Multi-class case
        return metrics.roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro')
    else:
        # Binary case - y_pred_proba should be probabilities for positive class
        if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] == 2:
            y_pred_proba = y_pred_proba[:, 1]
        return metrics.roc_auc_score(y_true, y_pred_proba)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Root Mean Squared Error."""
    return np.sqrt(metrics.mean_squared_error(y_true, y_pred))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Error."""
    return metrics.mean_absolute_error(y_true, y_pred)


def log_loss(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """Calculate log loss."""
    return metrics.log_loss(y_true, y_pred_proba)


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Root Mean Squared Logarithmic Error."""
    # Ensure non-negative predictions
    y_pred = np.maximum(0, y_pred)
    y_true = np.maximum(0, y_true)
    return np.sqrt(metrics.mean_squared_log_error(y_true, y_pred))


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate balanced accuracy score."""
    return metrics.balanced_accuracy_score(y_true, y_pred)


def multilabel_roc_auc(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """Calculate ROC-AUC for multi-label classification."""
    # Calculate AUC for each label and average
    n_labels = y_true.shape[1]
    auc_scores = []
    
    for i in range(n_labels):
        if len(np.unique(y_true[:, i])) > 1:  # Skip if only one class present
            auc = metrics.roc_auc_score(y_true[:, i], y_pred_proba[:, i])
            auc_scores.append(auc)
    
    return np.mean(auc_scores) if auc_scores else 0.0


def get_metric_function(metric_name: str) -> Callable:
    """Get metric function by name."""
    metric_map = {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'rmse': rmse,
        'mae': mae,
        'log_loss': log_loss,
        'rmsle': rmsle,
        'balanced_accuracy': balanced_accuracy,
        'multilabel_roc_auc': multilabel_roc_auc
    }
    
    if metric_name not in metric_map:
        raise ValueError(f"Unknown metric: {metric_name}. Available: {list(metric_map.keys())}")
    
    return metric_map[metric_name]


def calculate_metric(
    y_true: np.ndarray, 
    y_pred: Union[np.ndarray, Dict[str, np.ndarray]], 
    metric_name: str,
    problem_type: str
) -> float:
    """
    Calculate metric based on problem type and predictions.
    
    Args:
        y_true: True labels
        y_pred: Predictions - can be class labels, probabilities, or dict for multi-label
        metric_name: Name of the metric
        problem_type: Type of problem (binary_classification, multiclass_classification, etc.)
    
    Returns:
        Metric score
    """
    metric_fn = get_metric_function(metric_name)
    
    # Handle multi-label case
    if problem_type == 'multilabel_classification':
        if isinstance(y_pred, dict):
            # Convert dict predictions to array
            n_samples = len(y_true)
            n_labels = len(y_pred)
            y_pred_array = np.zeros((n_samples, n_labels))
            for i, (label, preds) in enumerate(y_pred.items()):
                y_pred_array[:, i] = preds
            y_pred = y_pred_array
        
        if metric_name == 'roc_auc':
            return multilabel_roc_auc(y_true, y_pred)
    
    # Handle probability-based metrics
    if metric_name in ['roc_auc', 'log_loss']:
        # These metrics need probabilities, not class predictions
        return metric_fn(y_true, y_pred)
    else:
        # These metrics need class predictions
        if problem_type in ['binary_classification', 'multiclass_classification']:
            # If we have probabilities, convert to class predictions
            if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
                y_pred = np.argmax(y_pred, axis=1)
            elif problem_type == 'binary_classification' and y_pred.dtype == float:
                # Binary probabilities - convert to 0/1
                y_pred = (y_pred > 0.5).astype(int)
        
        return metric_fn(y_true, y_pred)


def needs_probabilities(metric_name: str) -> bool:
    """Check if metric requires probability predictions."""
    return metric_name in ['roc_auc', 'log_loss', 'multilabel_roc_auc']