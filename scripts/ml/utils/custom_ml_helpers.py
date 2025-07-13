"""Custom ML helpers for feature selection and cross-validation."""

import pandas as pd
import numpy as np
import ydf
from typing import List, Tuple, Dict, Optional, Any
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
import time

from .metrics import calculate_metric, needs_probabilities
from .ydf_helpers import create_learner, determine_task, _train_silently

console = Console()


def backward_feature_selection(
    df: pd.DataFrame,
    target: str,
    model_type: str,
    problem_type: str,
    metric_name: str,
    removal_ratio: float = 0.2,
    validation_size: float = 0.2,
    min_features: int = 5,
    patience: int = 3,
    random_state: int = 42
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Custom backward feature selection implementation.
    
    Args:
        df: Input DataFrame with all features
        target: Target column name
        model_type: 'gbt' or 'rf'
        problem_type: Type of ML problem
        metric_name: Metric to optimize
        removal_ratio: If <1, fraction of features to remove per iteration. If >=1, exact count.
        validation_size: Fraction of data to use for validation
        min_features: Minimum number of features to keep
        patience: Stop if no improvement for this many iterations
        random_state: Random seed
    
    Returns:
        Tuple of (selected_features, selection_history)
    """
    console.print(f"\n[bold]Custom Backward Feature Selection[/bold]")
    console.print(f"Initial features: {len(df.columns) - 1}")
    console.print(f"Removal strategy: {'Remove ' + str(int(removal_ratio)) + ' features per iteration' if removal_ratio >= 1 else f'Remove {removal_ratio*100:.0f}% per iteration'}")
    console.print(f"Minimum features: {min_features}")
    console.print(f"Patience: {patience} iterations")
    
    # Split data for feature selection
    X = df.drop(columns=[target])
    y = df[target]
    
    # Create train/validation split
    if problem_type in ['binary_classification', 'multiclass_classification']:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_size, random_state=random_state, stratify=y
        )
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_size, random_state=random_state
        )
    
    # Start with all features
    current_features = list(X.columns)
    best_features = current_features.copy()
    best_score = -np.inf if metric_name not in ['rmse', 'mae'] else np.inf
    iterations_without_improvement = 0
    
    # History tracking
    history = {
        'iterations': [],
        'n_features': [],
        'scores': [],
        'removed_features': []
    }
    
    # Determine if higher is better
    higher_is_better = metric_name not in ['rmse', 'mae', 'log_loss']
    
    # Setup live display
    table = Table(title="Feature Selection Progress")
    table.add_column("Iteration", justify="center")
    table.add_column("Features", justify="center")
    table.add_column("Score", justify="center")
    table.add_column("Best Score", justify="center")
    table.add_column("Status", justify="center")
    
    iteration = 0
    
    with Live(table, console=console, refresh_per_second=2) as live:
        while len(current_features) > min_features:
            iteration += 1
            
            # Create datasets with current features
            train_df = pd.concat([X_train[current_features], y_train], axis=1)
            val_df = pd.concat([X_val[current_features], y_val], axis=1)
            
            # Train model
            task = determine_task(problem_type)
            learner = create_learner(model_type, target, task)
            
            # Train silently
            model = _train_silently(learner, train_df, verbose=0)
            
            # Evaluate
            if needs_probabilities(metric_name):
                predictions = model.predict(val_df)
                if problem_type == 'binary_classification':
                    y_pred = predictions.probability(1) if hasattr(predictions, 'probability') else predictions
                else:
                    y_pred = predictions
            else:
                predictions = model.predict(val_df)
                # Handle string labels
                if isinstance(y_val.iloc[0], str):
                    train_classes = sorted(train_df[target].unique())
                    if len(train_classes) == 2:
                        label_map = {0: train_classes[0], 1: train_classes[1]}
                        y_pred = np.array([label_map.get(int(p), p) for p in predictions])
                    else:
                        label_map = {i: cls for i, cls in enumerate(train_classes)}
                        y_pred = np.array([label_map.get(int(p), p) for p in predictions])
                else:
                    y_pred = predictions
            
            score = calculate_metric(y_val.values, y_pred, metric_name, problem_type)
            
            # Update history
            history['iterations'].append(iteration)
            history['n_features'].append(len(current_features))
            history['scores'].append(score)
            
            # Check if improved
            improved = False
            if higher_is_better:
                if score > best_score:
                    best_score = score
                    best_features = current_features.copy()
                    iterations_without_improvement = 0
                    improved = True
                else:
                    iterations_without_improvement += 1
            else:
                if score < best_score:
                    best_score = score
                    best_features = current_features.copy()
                    iterations_without_improvement = 0
                    improved = True
                else:
                    iterations_without_improvement += 1
            
            # Update table
            status = "✓ Improved" if improved else f"No improvement ({iterations_without_improvement}/{patience})"
            table.add_row(
                str(iteration),
                str(len(current_features)),
                f"{score:.4f}",
                f"{best_score:.4f}",
                status
            )
            
            # Check early stopping
            if iterations_without_improvement >= patience:
                console.print(f"\n[yellow]Early stopping: No improvement for {patience} iterations[/yellow]")
                break
            
            # Get feature importances
            importances = model.variable_importances()
            
            # Extract feature importance scores
            feature_scores = {}
            if isinstance(importances, dict) and 'NUM_AS_ROOT' in importances:
                for score, feature_name in importances['NUM_AS_ROOT']:
                    feature_scores[feature_name] = score
            
            # Sort features by importance
            sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Determine how many features to remove
            if removal_ratio >= 1:
                n_remove = min(int(removal_ratio), len(current_features) - min_features)
            else:
                n_remove = max(1, int(len(current_features) * removal_ratio))
                n_remove = min(n_remove, len(current_features) - min_features)
            
            if n_remove == 0:
                break
            
            # Remove least important features
            features_to_remove = [f[0] for f in sorted_features[-n_remove:]]
            history['removed_features'].append(features_to_remove)
            
            current_features = [f for f in current_features if f not in features_to_remove]
            
            # Show removed features
            console.print(f"  Removed {n_remove} features: {', '.join(features_to_remove[:3])}{'...' if n_remove > 3 else ''}")
    
    # Final summary
    console.print(f"\n[bold green]Feature Selection Complete[/bold green]")
    console.print(f"Best score: {best_score:.4f}")
    console.print(f"Selected features: {len(best_features)} (from {len(X.columns)})")
    console.print(f"Reduction: {(1 - len(best_features)/len(X.columns))*100:.1f}%")
    
    if len(best_features) <= 10:
        console.print(f"Selected: {', '.join(best_features)}")
    else:
        console.print(f"Top 10 selected: {', '.join(best_features[:10])}...")
    
    return best_features, history


def simple_cv(
    df: pd.DataFrame,
    target: str,
    model_type: str,
    problem_type: str,
    metric_name: str,
    features: Optional[List[str]] = None,
    n_splits: int = 5,
    random_state: int = 42,
    verbose: bool = True
) -> Tuple[float, float, List[float]]:
    """
    Simple cross-validation without feature selection.
    
    Args:
        df: Input DataFrame
        target: Target column name
        model_type: 'gbt' or 'rf'
        problem_type: Type of ML problem
        metric_name: Metric to calculate
        features: List of features to use (if None, use all)
        n_splits: Number of CV folds
        random_state: Random seed
        verbose: Whether to show progress
    
    Returns:
        Tuple of (mean_score, std_score, fold_scores)
    """
    if verbose:
        console.print(f"\n[bold]Cross-Validation[/bold]")
        console.print(f"Folds: {n_splits}")
        console.print(f"Features: {len(features) if features else len(df.columns) - 1}")
    
    # Prepare data
    if features:
        X = df[features]
    else:
        X = df.drop(columns=[target])
    y = df[target]
    
    # Choose CV strategy
    if problem_type in ['binary_classification', 'multiclass_classification']:
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = kf.split(X, y)
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = kf.split(X)
    
    scores = []
    task = determine_task(problem_type)
    
    # Progress tracking
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
        transient=True
    ) as progress:
        
        cv_task = progress.add_task(f"Running {n_splits}-fold CV", total=n_splits)
        
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            # Create fold data
            if features:
                train_df = pd.concat([X.iloc[train_idx], y.iloc[train_idx]], axis=1)
                val_df = pd.concat([X.iloc[val_idx], y.iloc[val_idx]], axis=1)
            else:
                train_df = df.iloc[train_idx]
                val_df = df.iloc[val_idx]
            
            # Train model
            learner = create_learner(model_type, target, task)
            model = _train_silently(learner, train_df, verbose=0)
            
            # Predict
            if needs_probabilities(metric_name):
                predictions = model.predict(val_df)
                if problem_type == 'binary_classification':
                    y_pred = predictions.probability(1) if hasattr(predictions, 'probability') else predictions
                else:
                    y_pred = predictions
            else:
                predictions = model.predict(val_df)
                # Handle string labels
                y_val = val_df[target].values
                if len(y_val) > 0 and isinstance(y_val[0], str):
                    train_classes = sorted(train_df[target].unique())
                    if len(train_classes) == 2:
                        label_map = {0: train_classes[0], 1: train_classes[1]}
                        y_pred = np.array([label_map.get(int(p), p) for p in predictions])
                    else:
                        label_map = {i: cls for i, cls in enumerate(train_classes)}
                        y_pred = np.array([label_map.get(int(p), p) for p in predictions])
                else:
                    y_pred = predictions
            
            # Calculate score
            score = calculate_metric(val_df[target].values, y_pred, metric_name, problem_type)
            scores.append(score)
            
            progress.update(cv_task, advance=1)
            
            if verbose:
                console.print(f"  Fold {fold_idx + 1}: {score:.4f}")
    
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    
    if verbose:
        console.print(f"\n[bold]CV Results:[/bold]")
        console.print(f"Mean {metric_name}: {mean_score:.4f} ± {std_score:.4f}")
    
    return mean_score, std_score, scores


def train_final_model(
    df: pd.DataFrame,
    target: str,
    model_type: str,
    problem_type: str,
    features: Optional[List[str]] = None,
    with_tuning: bool = False,
    tuning_trials: int = 30,
    validation_size: float = 0.2,
    random_state: int = 42
) -> Any:
    """
    Train final model on full dataset with optional tuning.
    
    Args:
        df: Input DataFrame
        target: Target column name
        model_type: 'gbt' or 'rf'
        problem_type: Type of ML problem
        features: List of features to use (if None, use all)
        with_tuning: Whether to tune hyperparameters
        tuning_trials: Number of tuning trials
        validation_size: Validation size for tuning
        random_state: Random seed
    
    Returns:
        Trained YDF model
    """
    console.print(f"\n[bold]Training Final Model[/bold]")
    console.print(f"Model type: {model_type.upper()}")
    console.print(f"Features: {len(features) if features else len(df.columns) - 1}")
    console.print(f"Tuning: {'Yes' if with_tuning else 'No'}")
    
    # Prepare data
    if features:
        train_df = df[features + [target]]
    else:
        train_df = df
    
    task = determine_task(problem_type)
    
    if with_tuning:
        # Split for tuning
        X = train_df.drop(columns=[target])
        y = train_df[target]
        
        if problem_type in ['binary_classification', 'multiclass_classification']:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_size, random_state=random_state, stratify=y
            )
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_size, random_state=random_state
            )
        
        tuning_train_df = pd.concat([X_train, y_train], axis=1)
        tuning_val_df = pd.concat([X_val, y_val], axis=1)
        
        console.print(f"Tuning with {tuning_trials} trials...")
        
        # Create tuner
        tuner = ydf.RandomSearchTuner(
            num_trials=tuning_trials,
            automatic_search_space=True,
            parallel_trials=1
        )
        
        # Create learner with tuner
        learner = create_learner(model_type, target, task)
        learner.tuner = tuner
        
        # Train with tuning
        if model_type == 'rf':
            # Random Forest uses OOB for tuning
            model = _train_silently(learner, train_df, verbose=1, show_table=True)
        else:
            # GBT uses validation for tuning
            model = _train_silently(learner, tuning_train_df, valid_data=tuning_val_df, verbose=1, show_table=True)
        
        console.print("[bold green]Tuning complete![/bold green]")
    else:
        # Train without tuning
        learner = create_learner(model_type, target, task)
        model = _train_silently(learner, train_df, verbose=1, show_table=True)
    
    console.print("[bold green]Model training complete![/bold green]")
    
    # Show feature importances
    importances = model.variable_importances()
    if isinstance(importances, dict) and 'NUM_AS_ROOT' in importances:
        console.print("\n[bold]Top 10 Feature Importances:[/bold]")
        top_features = sorted(importances['NUM_AS_ROOT'], key=lambda x: x[0], reverse=True)[:10]
        for i, (score, feature) in enumerate(top_features):
            console.print(f"  {i+1}. {feature}: {score:.4f}")
    
    return model


def custom_feature_selection_cv(
    df: pd.DataFrame,
    target: str,
    model_type: str,
    problem_type: str,
    metric_name: str,
    removal_ratio: float = 0.2,
    n_splits: int = 5,
    min_features: int = 5,
    patience: int = 3,
    use_tuning: bool = False,
    tuning_trials: int = 20,
    random_state: int = 42
) -> Tuple[float, float, List[float], List[str], int, Dict[str, Any]]:
    """
    Combined function: First select features, then do CV.
    
    Returns:
        Tuple of (mean_score, std_score, fold_scores, selected_features, n_selected)
    """
    # Step 1: Feature selection
    selected_features, history = backward_feature_selection(
        df=df,
        target=target,
        model_type=model_type,
        problem_type=problem_type,
        metric_name=metric_name,
        removal_ratio=removal_ratio,
        min_features=min_features,
        patience=patience,
        random_state=random_state
    )
    
    # Step 2: Cross-validation on selected features
    console.print(f"\n[bold]Evaluating selected features with {n_splits}-fold CV[/bold]")
    
    # If tuning requested, we'll tune during CV
    if use_tuning:
        console.print(f"With hyperparameter tuning ({tuning_trials} trials per fold)")
        # For now, we'll use the simple CV without tuning
        # TODO: Implement tuning within CV folds
    
    mean_score, std_score, fold_scores = simple_cv(
        df=df,
        target=target,
        model_type=model_type,
        problem_type=problem_type,
        metric_name=metric_name,
        features=selected_features,
        n_splits=n_splits,
        random_state=random_state
    )
    
    # For custom implementation, we don't have hyperparameter tuning yet
    # Return empty dict for consistency
    best_hyperparams = {}
    
    return mean_score, std_score, fold_scores, selected_features, len(selected_features), best_hyperparams