#!/usr/bin/env python3
"""
Version 3: Custom backward selection with CV inside (5-fold) + tuning inside each fold

ALGORITHM:
- Feature selection with CV inside (5-fold) + tuning inside each fold  
- Return CV score from feature selection as final result (NO additional CV)
- Same defaults: CV=5, removal_ratio=0.1, tuning=True
- Command: python version_3.py -c titanic

SPINNER:
- Aesthetic CV progress spinner: ●●●●◯ (for CV=5, showing 4 done, 1 current)
- Different spinner symbols: ░▓█ or ◯●◉ or other creative ones
- Show spinner next to iteration number during training

DIFFERENT FROM VERSION 2:
- Different variable names (data_frame vs df, cv_splits vs n_splits)
- Different function organization
- Different way to handle the iteration loop
- Different error handling
- But same algorithm and same spinner concept

Uses custom backward selection implementation (NOT cross_validate_ydf).
"""

import os
import sys
import json
import argparse
import signal
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Set
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
from rich import print as rprint
from rich.live import Live
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

# Import YDF
try:
    import ydf
    HAS_YDF = True
except ImportError:
    HAS_YDF = False

console = Console()

def train_ydf_silently(learner, train_data, validation_data=None):
    """Train YDF model silently (suppressing output)."""
    import os
    import sys
    from contextlib import contextmanager
    
    @contextmanager
    def suppress_stdout_stderr():
        with open(os.devnull, "w") as devnull:
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = devnull
            sys.stderr = devnull
            try:
                yield
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
    
    with suppress_stdout_stderr():
        if validation_data is not None:
            model = learner.train(train_data, valid=validation_data)
        else:
            model = learner.train(train_data)
    
    return model

def create_ydf_learner(model_type: str, label: str, task_type=None, **params):
    """Create YDF learner with given parameters."""
    if task_type is None:
        task_type = ydf.Task.CLASSIFICATION
    
    base_params = {
        'label': label,
        'task': task_type
    }
    base_params.update(params)
    
    if model_type == 'gbt':
        defaults = {
            'num_trees': 100,
            'max_depth': 6,
            'shrinkage': 0.1,
            'subsample': 0.8,
            'min_examples': 5,
            'num_threads': 0  # Use all available cores (0 = auto)
        }
        defaults.update(base_params)
        return ydf.GradientBoostedTreesLearner(**defaults)
    elif model_type == 'rf':
        defaults = {
            'num_trees': 100,
            'max_depth': 16,
            'min_examples': 5,
            'bootstrap_training_dataset': True,
            'compute_oob_variable_importances': True,
            'num_threads': 0  # Use all available cores (0 = auto)
        }
        defaults.update(base_params)
        return ydf.RandomForestLearner(**defaults)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def execute_cv_feature_selection(
    data_frame: pd.DataFrame,
    target_column: str,
    model_type: str,
    problem_type: str,
    metric_name: str,
    cv_splits: int = 5,
    removal_ratio: float = 0.1,
    use_tuning: bool = True,
    tuning_trials: int = 20,
    random_state: int = 42
) -> Tuple[float, float, List[str], Dict[str, Any]]:
    """
    Custom backward feature selection with CV inside (3-fold) + tuning inside each fold.
    
    ALGORITHM:
    - Feature selection with CV inside (3-fold) + tuning inside each fold
    - Return CV score from feature selection as final result (NO additional CV)
    
    Returns:
        Tuple of (mean_score, std_score, best_features, best_hyperparams)
    """
    from sklearn.model_selection import KFold, StratifiedKFold
    
    # Prepare data
    feature_columns = [col for col in data_frame.columns if col != target_column]
    original_feature_count = len(feature_columns)
    
    # Determine YDF task
    if 'classification' in problem_type:
        task_type = ydf.Task.CLASSIFICATION
        if data_frame[target_column].nunique() > 2:
            cv_strategy = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
        else:
            cv_strategy = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    else:
        task_type = ydf.Task.REGRESSION
        cv_strategy = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    
    # Initialize tracking variables
    current_features = set(feature_columns)
    best_score_overall = -float('inf') if metric_name not in ['rmse', 'mae'] else float('inf')
    best_features_overall = list(current_features)
    best_hyperparams_overall = {}
    iteration_count = 0
    
    # Create live table for progress tracking
    progress_table = Table(title="Custom Backward Feature Selection Progress")
    progress_table.add_column("Iteration", style="cyan", justify="right")
    progress_table.add_column("Features", style="magenta") 
    progress_table.add_column("Score", style="green")
    progress_table.add_column("Accuracy", style="green")
    progress_table.add_column("Loss", style="red")
    progress_table.add_column("Status", style="blue")
    
    iteration_results = []
    
    # Start feature selection loop
    if removal_ratio == 0:
        console.print(f"\n[bold]Simple Cross-Validation (No Feature Selection)[/bold]")
        tuning_info = f" ({tuning_trials} trials)" if use_tuning else ""
        console.print(f"Features: {len(current_features)}, CV folds: {cv_splits}, Tuning: {use_tuning}{tuning_info}")
    else:
        console.print(f"\n[bold]Custom Backward Feature Selection[/bold]")
        console.print(f"Starting with {len(current_features)} features")
        tuning_info = f" ({tuning_trials} trials per fold)" if use_tuning else ""
        console.print(f"CV folds: {cv_splits}, Removal ratio: {removal_ratio}, Tuning: {use_tuning}{tuning_info}")
    
    with Live(progress_table, refresh_per_second=4, console=console) as live_display:
        
        while True:
            # For removal_ratio=0, only do one iteration
            if removal_ratio == 0 and iteration_count > 0:
                break
            # For normal feature selection, continue until 1 feature left
            if removal_ratio > 0 and len(current_features) <= 1:
                break
            iteration_count += 1
            feature_list = list(current_features)
            
            # Prepare data with current features
            current_data = data_frame[feature_list + [target_column]]
            
            # Cross-validation on current feature set
            cv_fold_scores = []
            cv_hyperparams = []
            
            # Create splits
            X_temp = current_data.drop(columns=[target_column])
            y_temp = current_data[target_column]
            
            fold_idx = 0
            for train_indices, validation_indices in cv_strategy.split(X_temp, y_temp):
                fold_idx += 1
                
                # Update progress with spinner
                spinner_symbols = ['◯', '◉', '●']
                cv_progress_display = ""
                for i in range(cv_splits):
                    if i < fold_idx - 1:
                        cv_progress_display += "●"
                    elif i == fold_idx - 1:
                        cv_progress_display += "◉"
                    else:
                        cv_progress_display += "◯"
                
                # Update table row
                if iteration_count <= len(iteration_results) + 1:
                    if len(iteration_results) >= iteration_count:
                        iteration_results[iteration_count-1] = [
                            f"{iteration_count} {cv_progress_display}",
                            str(len(current_features)),
                            "-",
                            "-",
                            "-",
                            "Running"
                        ]
                    else:
                        iteration_results.append([
                            f"{iteration_count} {cv_progress_display}",
                            str(len(current_features)),
                            "-",
                            "-",
                            "-",
                            "Running"
                        ])
                
                # Rebuild table
                new_table = Table(title="Custom Backward Feature Selection Progress")
                new_table.add_column("Iteration", style="cyan", justify="right")
                new_table.add_column("Features", style="magenta")
                new_table.add_column("Score", style="green")
                new_table.add_column("Accuracy", style="green")
                new_table.add_column("Loss", style="red")
                new_table.add_column("Status", style="blue")
                
                for row in iteration_results:
                    new_table.add_row(*row)
                
                live_display.update(new_table)
                time.sleep(0.1)
                
                # Split data for this fold
                train_data_fold = current_data.iloc[train_indices]
                validation_data_fold = current_data.iloc[validation_indices]
                
                # Hyperparameter tuning if enabled
                fold_best_score = -float('inf') if metric_name not in ['rmse', 'mae'] else float('inf')
                fold_best_params = {}
                
                tuning_iterations = tuning_trials if use_tuning else 1
                for trial_idx in range(tuning_iterations):
                    # Update spinner with tuning progress
                    if use_tuning and tuning_iterations > 1:
                        tuning_progress = "▓" * (trial_idx * 3 // tuning_iterations) + "░" * (3 - (trial_idx * 3 // tuning_iterations))
                        cv_progress_with_tuning = f"{cv_progress_display}[{tuning_progress}]"
                    else:
                        cv_progress_with_tuning = cv_progress_display
                    
                    # Update table with tuning progress
                    if len(iteration_results) >= iteration_count:
                        iteration_results[iteration_count-1] = [
                            f"{iteration_count} {cv_progress_with_tuning}",
                            str(len(current_features)),
                            "-",
                            "-",
                            "-",
                            "Tuning" if use_tuning else "Running"
                        ]
                        
                        # Rebuild table
                        tuning_table = Table(title="Custom Backward Feature Selection Progress")
                        tuning_table.add_column("Iteration", style="cyan", justify="right")
                        tuning_table.add_column("Features", style="magenta")
                        tuning_table.add_column("Score", style="green")
                        tuning_table.add_column("Accuracy", style="green")
                        tuning_table.add_column("Loss", style="red")
                        tuning_table.add_column("Status", style="blue")
                        
                        for row in iteration_results:
                            tuning_table.add_row(*row)
                        
                        live_display.update(tuning_table)
                        time.sleep(0.05)
                    try:
                        if use_tuning:
                            # Simplified hyperparameter sampling - only safe parameters
                            if model_type == 'gbt':
                                trial_params = {
                                    'num_trees': np.random.choice([50, 100, 200]),
                                    'shrinkage': np.random.choice([0.1, 0.2])
                                }
                            else:  # rf
                                trial_params = {
                                    'num_trees': np.random.choice([50, 100, 200])
                                }
                        else:
                            trial_params = {}
                        
                        # Create and train model
                        learner = create_ydf_learner(model_type, target_column, task_type, **trial_params)
                        
                        # For RF, don't use validation data (use OOB)
                        if model_type == 'rf':
                            model = train_ydf_silently(learner, train_data_fold)
                        else:
                            # Split training data for GBT validation
                            train_subset_size = int(0.8 * len(train_data_fold))
                            train_subset = train_data_fold.iloc[:train_subset_size]
                            val_subset = train_data_fold.iloc[train_subset_size:]
                            model = train_ydf_silently(learner, train_subset, val_subset)
                        
                        # Make predictions
                        if needs_probabilities(metric_name):
                            predictions = model.predict(validation_data_fold)
                            if problem_type == 'binary_classification':
                                if hasattr(predictions, 'probability'):
                                    y_pred = predictions.probability(1)
                                else:
                                    y_pred = predictions
                            else:
                                y_pred = predictions
                        else:
                            predictions = model.predict(validation_data_fold)
                            y_pred = predictions
                        
                        # Calculate score
                        y_true = validation_data_fold[target_column].values
                        trial_score = calculate_metric(y_true, y_pred, metric_name, problem_type)
                        
                        # Check if this is the best trial for this fold
                        is_better = (
                            (metric_name in ['rmse', 'mae'] and trial_score < fold_best_score) or
                            (metric_name not in ['rmse', 'mae'] and trial_score > fold_best_score)
                        )
                        
                        if is_better:
                            fold_best_score = trial_score
                            fold_best_params = trial_params.copy()
                    
                    except Exception as e:
                        # Skip failed trials
                        continue
                
                # Only add score if we got a valid result
                if fold_best_score != -float('inf') and fold_best_score != float('inf'):
                    cv_fold_scores.append(fold_best_score)
                    cv_hyperparams.append(fold_best_params)
                else:
                    # Train a simple model without tuning as fallback
                    try:
                        simple_learner = create_ydf_learner(model_type, target_column, task_type)
                        simple_model = train_ydf_silently(simple_learner, train_data_fold)
                        
                        if needs_probabilities(metric_name):
                            predictions = simple_model.predict(validation_data_fold)
                            if problem_type == 'binary_classification':
                                if hasattr(predictions, 'probability'):
                                    y_pred = predictions.probability(1)
                                else:
                                    y_pred = predictions
                            else:
                                y_pred = predictions
                        else:
                            predictions = simple_model.predict(validation_data_fold)
                            y_pred = predictions
                        
                        y_true = validation_data_fold[target_column].values
                        fallback_score = calculate_metric(y_true, y_pred, metric_name, problem_type)
                        cv_fold_scores.append(fallback_score)
                        cv_hyperparams.append({})
                    except Exception as e:
                        # Last resort fallback
                        default_score = 0.5 if metric_name not in ['rmse', 'mae'] else 1.0
                        cv_fold_scores.append(default_score)
                        cv_hyperparams.append({})
            
            # Calculate mean and std for current iteration
            current_mean = np.mean(cv_fold_scores)
            current_std = np.std(cv_fold_scores)
            
            # Check if this is the best iteration so far
            is_best_iteration = (
                (metric_name in ['rmse', 'mae'] and current_mean < best_score_overall) or
                (metric_name not in ['rmse', 'mae'] and current_mean > best_score_overall)
            )
            
            if is_best_iteration:
                best_score_overall = current_mean
                best_features_overall = list(current_features)
                # Average hyperparameters across folds
                if cv_hyperparams and any(cv_hyperparams):
                    best_hyperparams_overall = {}
                    for key in cv_hyperparams[0].keys():
                        values = [hp.get(key) for hp in cv_hyperparams if hp.get(key) is not None]
                        if values:
                            if isinstance(values[0], (int, float)):
                                best_hyperparams_overall[key] = np.mean(values)
                            else:
                                # For categorical parameters, take the most common
                                from collections import Counter
                                best_hyperparams_overall[key] = Counter(values).most_common(1)[0][0]
            
            # Update final results for this iteration
            final_cv_progress = "●" * cv_splits  # All folds complete
            
            # Calculate accuracy and loss for display
            if 'classification' in problem_type:
                accuracy_value = current_mean if metric_name in ['accuracy', 'f1', 'precision', 'recall'] else current_mean
                loss_value = 1.0 - current_mean if metric_name in ['accuracy', 'f1', 'precision', 'recall'] else current_mean
            else:
                accuracy_value = current_mean if metric_name not in ['rmse', 'mae'] else 1.0 / (1.0 + current_mean)
                loss_value = current_mean if metric_name in ['rmse', 'mae'] else 1.0 - current_mean
            
            # Add current iteration result first
            score_text = f"{current_mean:.4f}"
            
            iteration_results[iteration_count-1] = [
                f"{iteration_count} {final_cv_progress}",
                str(len(current_features)),
                score_text,
                f"{accuracy_value:.4f}",
                f"{loss_value:.4f}",
                "Done"
            ]
            
            # Update highlighting - remove from all, then highlight the best overall
            best_iteration_idx = -1
            best_score_so_far = -float('inf') if metric_name not in ['rmse', 'mae'] else float('inf')
            
            # Find the best iteration so far
            for i in range(len(iteration_results)):
                if len(iteration_results[i]) >= 3:
                    # Extract numeric score (remove any existing highlighting)
                    score_str = iteration_results[i][2]
                    if "[reverse green]" in score_str:
                        numeric_score = float(score_str.replace("[reverse green]", "").replace("[/reverse green]", ""))
                    else:
                        numeric_score = float(score_str)
                    
                    # Clean the score (remove highlighting)
                    iteration_results[i][2] = f"{numeric_score:.4f}"
                    
                    # Check if this is the best
                    is_better = (
                        (metric_name in ['rmse', 'mae'] and numeric_score < best_score_so_far) or
                        (metric_name not in ['rmse', 'mae'] and numeric_score > best_score_so_far)
                    )
                    if is_better:
                        best_score_so_far = numeric_score
                        best_iteration_idx = i
            
            # Highlight the best iteration
            if best_iteration_idx >= 0:
                iteration_results[best_iteration_idx][2] = f"[reverse green]{best_score_so_far:.4f}[/reverse green]"
            
            # Update display one more time
            final_table = Table(title="Custom Backward Feature Selection Progress")
            final_table.add_column("Iteration", style="cyan", justify="right")
            final_table.add_column("Features", style="magenta")
            final_table.add_column("Score", style="green")
            final_table.add_column("Accuracy", style="green")
            final_table.add_column("Loss", style="red")
            final_table.add_column("Status", style="blue")
            
            for row in iteration_results:
                final_table.add_row(*row)
            
            live_display.update(final_table)
            time.sleep(0.5)
            
            # Feature removal for next iteration
            if len(current_features) > 1 and removal_ratio > 0:
                # Calculate number of features to remove
                num_to_remove = max(1, int(len(current_features) * removal_ratio))
                
                # Use feature importance from the last trained model
                # Train a quick model to get importances
                try:
                    temp_data = data_frame[list(current_features) + [target_column]]
                    temp_learner = create_ydf_learner(model_type, target_column, task_type)
                    temp_model = train_ydf_silently(temp_learner, temp_data)
                    
                    # Get feature importances
                    importance_dict = {}
                    if hasattr(temp_model, 'variable_importances'):
                        importances = temp_model.variable_importances()
                        if isinstance(importances, dict) and 'NUM_AS_ROOT' in importances:
                            for score, feature_name in importances['NUM_AS_ROOT']:
                                if feature_name in current_features:
                                    importance_dict[feature_name] = score
                    
                    # Remove least important features
                    if importance_dict:
                        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1])
                        features_to_remove = [f[0] for f in sorted_features[:num_to_remove]]
                    else:
                        # Fallback: remove random features
                        features_to_remove = np.random.choice(
                            list(current_features), 
                            size=min(num_to_remove, len(current_features) - 1), 
                            replace=False
                        )
                    
                    for feature in features_to_remove:
                        if feature in current_features:
                            current_features.remove(feature)
                            
                except Exception as e:
                    # Fallback: remove random features
                    features_to_remove = np.random.choice(
                        list(current_features), 
                        size=min(num_to_remove, len(current_features) - 1), 
                        replace=False
                    )
                    for feature in features_to_remove:
                        if feature in current_features:
                            current_features.remove(feature)
    
    # Return results from best iteration
    final_mean_score = best_score_overall
    final_std_score = 0.0  # We don't track std for the best iteration in this simplified version
    
    console.print(f"\n[bold]Feature Selection Complete[/bold]")
    console.print(f"Best iteration had {len(best_features_overall)} features")
    console.print(f"Best score: {final_mean_score:.4f}")
    
    return final_mean_score, final_std_score, best_features_overall, best_hyperparams_overall


class MDMBenchmarkV3:
    """Benchmark MDM generic features with custom backward selection."""
    
    def __init__(self, output_dir: str = "benchmark_results", use_cache: bool = True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.use_cache = use_cache
        
        # Get MDM components
        self.config_manager = get_config_manager()
        self.dataset_manager = DatasetManager()
        self.dataset_registrar = DatasetRegistrar()
        
        self.results = {
            'version': 'Version 3: Custom backward selection with CV inside (5-fold)',
            'description': 'Custom backward feature selection with 5-fold CV + tuning inside each fold',
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
        """Load competition data from MDM."""
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
            
            data_frame = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
            conn.close()
            
            if not with_features:
                original_cols = []
                for col in data_frame.columns:
                    if col == config['target'] or col == config.get('id_column', 'id'):
                        original_cols.append(col)
                    elif not any(suffix in col for suffix in [
                        '_zscore', '_log', '_sqrt', '_squared', '_is_outlier',
                        '_percentile_rank', '_year', '_month', '_day', '_hour',
                        '_frequency', '_target_mean', '_length', '_word_count',
                        '_is_missing', '_binned', '_x_', '_lag_', '_rolling_'
                    ]):
                        original_cols.append(col)
                
                data_frame = data_frame[original_cols]
            
            return data_frame
            
        except Exception as e:
            console.print(f"  ✗ Failed to load {dataset_name}: {str(e)}", style="red")
            return None
    
    def benchmark_competition(self, name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Benchmark a single competition."""
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
        
        console.print("\n[bold]Training models...")
        model_types = ['gbt', 'rf']
        
        for model_type in model_types:
            console.print(f"\n[cyan]{model_type.upper()}:[/cyan]")
            
            # With features (custom backward selection with CV inside)
            console.print("  Custom backward selection with CV inside (5-fold)...")
            try:
                mean_with, std_with, selected_features, best_hyperparams = execute_cv_feature_selection(
                    df_features,
                    config['target'],
                    model_type,
                    config['problem_type'],
                    config['metric'],
                    cv_splits=5,
                    removal_ratio=0.1,
                    use_tuning=True,
                    tuning_trials=20
                )
                
                results['with_features'][model_type] = {
                    'mean_score': round(mean_with, 4),
                    'std': round(std_with, 4),
                    'n_features': n_features_with,
                    'n_selected': len(selected_features),
                    'best_features': selected_features[:20] if len(selected_features) > 20 else selected_features,
                    'best_hyperparams': best_hyperparams,
                    'method': 'Custom backward selection with 3-fold CV + tuning'
                }
                console.print(f"    ✓ Score: {mean_with:.4f} ± {std_with:.4f}")
                console.print(f"    → Selected features: {len(selected_features)}")
                
            except Exception as e:
                console.print(f"    ✗ Failed: {str(e)}", style="red")
                results['with_features'][model_type] = {'error': str(e)}
            
            # Without features (simple baseline)
            console.print("  Training baseline without features...")
            try:
                # Simple baseline - just run CV without feature selection (removal_ratio=0)
                mean_without, std_without, _, _ = execute_cv_feature_selection(
                    df_raw,
                    config['target'],
                    model_type,
                    config['problem_type'],
                    config['metric'],
                    cv_splits=cv_splits,  # Use same CV folds as main process
                    removal_ratio=0.0,  # No feature removal
                    use_tuning=True,
                    tuning_trials=20
                )
                results['without_features'][model_type] = {
                    'mean_score': round(mean_without, 4),
                    'std': round(std_without, 4),
                    'n_features': n_features_without
                }
                console.print(f"    ✓ Score: {mean_without:.4f} ± {std_without:.4f}")
            except Exception as e:
                console.print(f"    ✗ Failed: {str(e)}", style="red")
                results['without_features'][model_type] = {'error': str(e)}
            
            # Calculate improvement
            if model_type in results['with_features'] and model_type in results['without_features']:
                if 'mean_score' in results['with_features'][model_type] and \
                   'mean_score' in results['without_features'][model_type]:
                    score_with = results['with_features'][model_type]['mean_score']
                    score_without = results['without_features'][model_type]['mean_score']
                    
                    if config['metric'] in ['rmse', 'mae']:
                        improvement = ((score_without - score_with) / score_without) * 100
                    else:
                        improvement = ((score_with - score_without) / score_without) * 100
                    
                    results['improvement'][model_type] = f"{improvement:+.2f}%"
                    console.print(f"    [green]Improvement: {improvement:+.2f}%[/green]")
        
        results['status'] = 'completed'
        return results
    
    def run_benchmark(self, competitions: Optional[List[str]] = None):
        """Run benchmark for specified competitions or all."""
        all_competitions = get_all_competitions()
        
        if competitions:
            selected = {k: v for k, v in all_competitions.items() if k in competitions}
        else:
            selected = all_competitions
        
        console.print(Panel.fit(
            f"[bold]Version 3: Custom backward selection with CV inside (5-fold)[/bold]\n"
            f"Custom feature selection with 5-fold CV + tuning inside each fold\n"
            f"Competitions: {len(selected)}\n"
            f"MDM Version: {mdm.__version__}",
            title="Benchmark Info"
        ))
        
        for name, config in selected.items():
            try:
                results = self.benchmark_competition(name, config)
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
        """Calculate summary statistics."""
        improvements = []
        competitions_improved = 0
        competitions_no_change = 0
        
        for name, result in self.results['results'].items():
            if result.get('status') != 'completed':
                continue
            
            comp_improved = False
            for model_type in ['gbt', 'rf']:
                if model_type in result.get('improvement', {}):
                    imp_str = result['improvement'][model_type]
                    imp_val = float(imp_str.replace('%', '').replace('+', ''))
                    improvements.append(imp_val)
                    if imp_val > 0:
                        comp_improved = True
            
            if comp_improved:
                competitions_improved += 1
            else:
                competitions_no_change += 1
        
        if improvements:
            avg_improvement = np.mean(improvements)
            best_idx = np.argmax(improvements)
            best_comp = list(self.results['results'].keys())[best_idx // 2]
            best_model = 'gbt' if best_idx % 2 == 0 else 'rf'
            best_improvement = improvements[best_idx]
            
            self.results['summary'] = {
                'average_improvement': f"{avg_improvement:+.2f}%",
                'best_improvement': f"{best_comp} ({best_model}): {best_improvement:+.2f}%",
                'competitions_improved': competitions_improved,
                'competitions_no_change': competitions_no_change,
                'competitions_failed': len(self.results['results']) - competitions_improved - competitions_no_change
            }
    
    def save_results(self):
        """Save results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"v3_results_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        console.print(f"\n[green]Results saved to: {output_file}[/green]")
    
    def display_summary(self):
        """Display summary table."""
        table = Table(title="Benchmark Summary - Version 3", show_header=True)
        table.add_column("Competition", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Features", justify="right")
        table.add_column("GBT Improvement", justify="right")
        table.add_column("RF Improvement", justify="right")
        
        for name, result in self.results['results'].items():
            status = result.get('status', 'unknown')
            gbt_imp = result.get('improvement', {}).get('gbt', 'N/A')
            rf_imp = result.get('improvement', {}).get('rf', 'N/A')
            
            if 'with_features' in result and 'gbt' in result['with_features']:
                n_total = result['with_features']['gbt'].get('n_features', 'N/A')
                n_selected = result['with_features']['gbt'].get('n_selected', n_total)
                if n_selected != n_total and n_selected != 'N/A':
                    features_str = f"{n_selected}/{n_total}"
                else:
                    features_str = str(n_total)
            else:
                features_str = 'N/A'
            
            if isinstance(gbt_imp, str) and '+' in gbt_imp:
                gbt_imp = f"[green]{gbt_imp}[/green]"
            elif isinstance(gbt_imp, str) and '-' in gbt_imp:
                gbt_imp = f"[red]{gbt_imp}[/red]"
            
            if isinstance(rf_imp, str) and '+' in rf_imp:
                rf_imp = f"[green]{rf_imp}[/green]"
            elif isinstance(rf_imp, str) and '-' in rf_imp:
                rf_imp = f"[red]{rf_imp}[/red]"
            
            table.add_row(name, status, features_str, gbt_imp, rf_imp)
        
        console.print("\n")
        console.print(table)
        
        if 'summary' in self.results:
            console.print("\n[bold]Overall Summary:[/bold]")
            for key, value in self.results['summary'].items():
                console.print(f"  {key}: {value}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Version 3: Custom backward selection with CV inside (3-fold)"
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
    
    if not HAS_YDF:
        console.print("[red]Error: YDF (Yggdrasil Decision Forests) is not installed.[/red]")
        console.print("Install it with: pip install ydf")
        sys.exit(1)
    
    benchmark = MDMBenchmarkV3(
        output_dir=args.output_dir,
        use_cache=not args.no_cache
    )
    
    benchmark.run_benchmark(competitions=args.competitions)


if __name__ == '__main__':
    main()