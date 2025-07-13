#!/usr/bin/env python3
"""
Custom feature selection module with backward selection and CV inside.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Set, Optional, Dict, Any
from sklearn.model_selection import StratifiedKFold, KFold
import warnings
warnings.filterwarnings('ignore')

from cv_evaluator import cross_validate_model
from spinner_utils import CVSpinner, IterationSpinner, create_cv_spinner


class BackwardFeatureSelector:
    """
    Custom backward feature selection with cross-validation inside.
    
    This implementation performs feature selection with CV inside and returns
    the CV score as the final result (no additional CV).
    """
    
    def __init__(
        self,
        cv_folds: int = 3,
        removal_ratio: float = 0.1,
        use_tuning: bool = True,
        min_features: int = 5,
        patience: int = 3,
        random_state: int = 42
    ):
        """
        Initialize backward feature selector.
        
        Args:
            cv_folds: Number of CV folds for evaluation
            removal_ratio: Fraction of features to remove each iteration
            use_tuning: Whether to use hyperparameter tuning
            min_features: Minimum number of features to keep
            patience: Number of iterations without improvement before stopping
            random_state: Random seed
        """
        self.cv_folds = cv_folds
        self.removal_ratio = removal_ratio
        self.use_tuning = use_tuning
        self.min_features = min_features
        self.patience = patience
        self.random_state = random_state
        
        # Results
        self.best_features_: Optional[List[str]] = None
        self.best_score_: Optional[float] = None
        self.best_hyperparams_: Optional[Dict[str, Any]] = None
        self.selection_history_: List[Dict] = []
        self.cv_scores_: List[float] = []
        
        # Progress tracking
        self.cv_spinner: Optional[CVSpinner] = None
        self.iteration_spinner: Optional[IterationSpinner] = None
    
    def _create_cv_splits(self, X: pd.DataFrame, y: pd.Series, problem_type: str):
        """Create cross-validation splits."""
        if problem_type in ['binary_classification', 'multiclass_classification']:
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            return list(cv.split(X, y))
        else:
            cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            return list(cv.split(X))
    
    def _evaluate_features_with_cv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        features: List[str],
        model_type: str,
        problem_type: str,
        metric_name: str,
        hyperparams: Optional[Dict[str, Any]] = None
    ) -> Tuple[float, float, List[float]]:
        """
        Evaluate a feature set using cross-validation.
        
        Returns:
            Tuple of (mean_score, std_score, fold_scores)
        """
        X_subset = X[features]
        
        # Perform cross-validation
        mean_score, std_score, fold_scores, _ = cross_validate_model(
            X_subset, y, model_type, problem_type, metric_name,
            n_splits=self.cv_folds, hyperparams=hyperparams, show_progress=False
        )
        
        return mean_score, std_score, fold_scores
    
    def _select_features_to_remove(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        current_features: List[str],
        model_type: str,
        problem_type: str,
        metric_name: str,
        hyperparams: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Select features to remove based on importance or random selection.
        
        For now, we'll use random selection to keep it simple and fast.
        In a production version, you could use feature importance.
        """
        n_to_remove = max(1, int(len(current_features) * self.removal_ratio))
        n_to_remove = min(n_to_remove, len(current_features) - self.min_features)
        
        if n_to_remove <= 0:
            return []
        
        # Random selection for simplicity
        np.random.seed(self.random_state)
        features_to_remove = np.random.choice(
            current_features, size=n_to_remove, replace=False
        ).tolist()
        
        return features_to_remove
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_type: str,
        problem_type: str,
        metric_name: str
    ) -> 'BackwardFeatureSelector':
        """
        Fit the feature selector.
        
        Args:
            X: Feature matrix
            y: Target vector
            model_type: 'gbt' or 'rf'
            problem_type: Problem type
            metric_name: Metric to optimize
        
        Returns:
            Self for chaining
        """
        # Initialize progress tracking
        self.cv_spinner = create_cv_spinner(folds=self.cv_folds, style='heavy')
        self.iteration_spinner = IterationSpinner()
        
        # Initialize
        current_features = list(X.columns)
        best_score = float('-inf') if metric_name not in ['rmse', 'mae'] else float('inf')
        best_features = current_features.copy()
        best_hyperparams = {}
        iterations_without_improvement = 0
        
        self.selection_history_ = []
        self.cv_scores_ = []
        
        iteration = 0
        
        # Start feature selection process
        self.iteration_spinner.start(f"Starting backward feature selection with {len(current_features)} features")
        
        while len(current_features) > self.min_features and iterations_without_improvement < self.patience:
            iteration += 1
            
            self.iteration_spinner.update(
                f"Iteration {iteration}: {len(current_features)} features, {iterations_without_improvement}/{self.patience} patience"
            )
            
            # Hyperparameter tuning (if enabled)
            if self.use_tuning:
                from cv_evaluator import hyperparameter_tune
                current_hyperparams = hyperparameter_tune(
                    X[current_features], y, model_type, problem_type, metric_name,
                    n_splits=self.cv_folds, show_progress=False
                )
            else:
                current_hyperparams = {}
            
            # Start CV evaluation
            self.cv_spinner.start(f"CV evaluation with {len(current_features)} features")
            
            # Evaluate current feature set with CV
            cv_splits = self._create_cv_splits(X[current_features], y, problem_type)
            fold_scores = []
            
            for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
                self.cv_spinner.update_fold(fold_idx, f"Fold {fold_idx + 1}")
                
                X_train, X_val = X[current_features].iloc[train_idx], X[current_features].iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Evaluate this fold
                from cv_evaluator import evaluate_fold
                score, _ = evaluate_fold(
                    X_train, y_train, X_val, y_val,
                    model_type, problem_type, metric_name,
                    current_hyperparams, show_progress=False
                )
                fold_scores.append(score)
            
            # Calculate CV score
            mean_score = np.mean(fold_scores)
            std_score = np.std(fold_scores)
            
            self.cv_spinner.stop(f"CV Score: {mean_score:.4f} ± {std_score:.4f}")
            
            # Record results
            self.selection_history_.append({
                'iteration': iteration,
                'n_features': len(current_features),
                'features': current_features.copy(),
                'mean_score': mean_score,
                'std_score': std_score,
                'fold_scores': fold_scores.copy(),
                'hyperparams': current_hyperparams.copy()
            })
            self.cv_scores_.append(mean_score)
            
            # Check if this is the best score
            is_better = (
                (metric_name in ['rmse', 'mae'] and mean_score < best_score) or
                (metric_name not in ['rmse', 'mae'] and mean_score > best_score)
            )
            
            if is_better:
                best_score = mean_score
                best_features = current_features.copy()
                best_hyperparams = current_hyperparams.copy()
                iterations_without_improvement = 0
                
                self.iteration_spinner.update(
                    f"Iteration {iteration}: NEW BEST! Score: {mean_score:.4f}, Features: {len(current_features)}"
                )
            else:
                iterations_without_improvement += 1
                self.iteration_spinner.update(
                    f"Iteration {iteration}: Score: {mean_score:.4f}, No improvement ({iterations_without_improvement}/{self.patience})"
                )
            
            # Early stopping check
            if iterations_without_improvement >= self.patience:
                self.iteration_spinner.update(f"Early stopping: no improvement for {self.patience} iterations")
                break
            
            # Remove features for next iteration
            features_to_remove = self._select_features_to_remove(
                X, y, current_features, model_type, problem_type, metric_name, current_hyperparams
            )
            
            if not features_to_remove:
                self.iteration_spinner.update("Cannot remove more features (minimum reached)")
                break
            
            current_features = [f for f in current_features if f not in features_to_remove]
            
            # Safety check
            if len(current_features) < self.min_features:
                current_features = best_features.copy()
                break
        
        # Store final results
        self.best_features_ = best_features
        self.best_score_ = best_score
        self.best_hyperparams_ = best_hyperparams
        
        self.iteration_spinner.stop(
            f"Feature selection completed: {len(best_features)} features, score: {best_score:.4f}"
        )
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform X to selected features."""
        if self.best_features_ is None:
            raise ValueError("Selector not fitted. Call fit() first.")
        return X[self.best_features_]
    
    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_type: str,
        problem_type: str,
        metric_name: str
    ) -> pd.DataFrame:
        """Fit selector and transform X."""
        return self.fit(X, y, model_type, problem_type, metric_name).transform(X)
    
    def get_cv_score(self) -> Tuple[float, float]:
        """
        Get the cross-validation score of the best feature set.
        
        Returns:
            Tuple of (mean_score, std_score)
        """
        if not self.selection_history_:
            raise ValueError("Selector not fitted. Call fit() first.")
        
        # Find the best iteration
        best_iteration = None
        best_score = float('-inf')
        
        for hist in self.selection_history_:
            if hist['mean_score'] > best_score:  # Assuming higher is better
                best_score = hist['mean_score']
                best_iteration = hist
        
        if best_iteration:
            return best_iteration['mean_score'], best_iteration['std_score']
        else:
            return self.selection_history_[-1]['mean_score'], self.selection_history_[-1]['std_score']
    
    def get_feature_importance_history(self) -> pd.DataFrame:
        """Get feature selection history as DataFrame."""
        if not self.selection_history_:
            return pd.DataFrame()
        
        records = []
        for hist in self.selection_history_:
            records.append({
                'iteration': hist['iteration'],
                'n_features': hist['n_features'],
                'mean_score': hist['mean_score'],
                'std_score': hist['std_score']
            })
        
        return pd.DataFrame(records)


def select_features_with_cv(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str,
    problem_type: str,
    metric_name: str,
    cv_folds: int = 3,
    removal_ratio: float = 0.1,
    use_tuning: bool = True
) -> Tuple[float, float, List[str], Dict[str, Any]]:
    """
    Convenience function for feature selection with CV.
    
    Args:
        X: Feature matrix
        y: Target vector
        model_type: 'gbt' or 'rf'
        problem_type: Problem type
        metric_name: Metric to optimize
        cv_folds: Number of CV folds
        removal_ratio: Fraction of features to remove each iteration
        use_tuning: Whether to use hyperparameter tuning
    
    Returns:
        Tuple of (mean_score, std_score, selected_features, best_hyperparams)
    """
    selector = BackwardFeatureSelector(
        cv_folds=cv_folds,
        removal_ratio=removal_ratio,
        use_tuning=use_tuning
    )
    
    selector.fit(X, y, model_type, problem_type, metric_name)
    
    mean_score, std_score = selector.get_cv_score()
    
    return (
        mean_score,
        std_score,
        selector.best_features_,
        selector.best_hyperparams_
    )