#!/usr/bin/env python3
"""
Version 10: Iterator-Based Custom Backward Feature Selection with CV inside

ALGORITHM:
- Feature selection with CV inside (3-fold) + tuning inside each fold
- Return CV score from feature selection as final result (NO additional CV)
- Same defaults: CV=3, removal_ratio=0.1, tuning=True
- Command: python version_10.py -c titanic

SPINNER:
- Aesthetic CV progress spinner: ▰▰▱ (for CV=3, showing 2 done, 1 current)
- Use different symbols: 🌙⭐ or 🎈🎁 or 🎨🖼️ or other emoji pairs
- Show spinner next to iteration number during training

ITERATOR APPROACH:
- Iterator-based processing (FeatureIterator, CVIterator)
- Lazy evaluation where possible with generators
- yield statements for data flow and progress
- Iterator-based table updates with next()
- But same core algorithm with custom backward selection

Makes use of custom backward selection implementation (NOT cross_validate_ydf).
"""

import os
import sys
import json
import argparse
import signal
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Iterator, Generator
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

# Import our modular components
from cv_evaluator import cross_validate_model, evaluate_fold, hyperparameter_tune, create_ydf_model

console = Console()


class EmojiSpinner:
    """Aesthetic spinner with emojis for CV progress display."""
    
    def __init__(self, total_folds: int = 3, emoji_set: str = 'moon_star'):
        """
        Initialize emoji spinner.
        
        Args:
            total_folds: Total number of CV folds
            emoji_set: Set of emojis to use
        """
        self.total_folds = total_folds
        self.current_fold = 0
        self.emoji_sets = {
            'moon_star': ('🌙', '⭐'),
            'balloon_gift': ('🎈', '🎁'),
            'art_frame': ('🎨', '🖼️'),
            'gem_star': ('💎', '⭐'),
            'fire_snow': ('🔥', '❄️'),
            'sun_moon': ('☀️', '🌙'),
            'heart_spark': ('💝', '✨'),
            'rocket_star': ('🚀', '⭐')
        }
        
        self.filled_emoji, self.empty_emoji = self.emoji_sets.get(emoji_set, self.emoji_sets['moon_star'])
        self.current_message = ""
        
    def _create_progress_bar(self, current: int, total: int) -> str:
        """Create a visual progress bar with emojis."""
        filled = self.filled_emoji * current
        empty = self.empty_emoji * (total - current)
        return f"{filled}{empty}"
    
    def update_fold(self, fold: int, message: str = ""):
        """Update current fold progress."""
        self.current_fold = fold
        if message:
            self.current_message = message
    
    def display(self, message: str = ""):
        """Display current progress."""
        if message:
            self.current_message = message
        progress_bar = self._create_progress_bar(self.current_fold, self.total_folds)
        console.print(f"  CV Progress: {progress_bar} ({self.current_fold}/{self.total_folds}) - {self.current_message}")
    
    def complete(self, final_message: str = "Completed"):
        """Show completion."""
        progress_bar = self._create_progress_bar(self.total_folds, self.total_folds)
        console.print(f"  ✓ CV Progress: {progress_bar} ({self.total_folds}/{self.total_folds}) - {final_message}")


class FeatureIterator:
    """Iterator for feature selection process with lazy evaluation."""
    
    def __init__(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        model_type: str,
        problem_type: str,
        metric_name: str,
        removal_ratio: float = 0.1,
        min_features: int = 5,
        patience: int = 3,
        cv_folds: int = 3,
        use_tuning: bool = True
    ):
        self.X = X
        self.y = y
        self.model_type = model_type
        self.problem_type = problem_type
        self.metric_name = metric_name
        self.removal_ratio = removal_ratio
        self.min_features = min_features
        self.patience = patience
        self.cv_folds = cv_folds
        self.use_tuning = use_tuning
        
        # State
        self.current_features = list(X.columns)
        self.best_features = self.current_features.copy()
        self.best_score = float('-inf') if metric_name not in ['rmse', 'mae'] else float('inf')
        self.best_hyperparams = {}
        self.iterations_without_improvement = 0
        self.iteration = 0
        self.selection_history = []
        
        # Progress tracking
        self.emoji_spinner = EmojiSpinner(total_folds=cv_folds, emoji_set='moon_star')
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Return iterator over feature selection steps."""
        return self
    
    def __next__(self) -> Dict[str, Any]:
        """Get next iteration of feature selection."""
        # Check termination conditions
        if (len(self.current_features) <= self.min_features or 
            self.iterations_without_improvement >= self.patience):
            raise StopIteration
        
        self.iteration += 1
        
        console.print(f"\n[bold cyan]Iteration {self.iteration}:[/bold cyan] {len(self.current_features)} features")
        
        # Hyperparameter tuning (if enabled)
        if self.use_tuning:
            console.print("  🔧 Hyperparameter tuning...")
            current_hyperparams = hyperparameter_tune(
                self.X[self.current_features], self.y, self.model_type, 
                self.problem_type, self.metric_name,
                n_splits=self.cv_folds, show_progress=False
            )
        else:
            current_hyperparams = {}
        
        # Perform CV evaluation with iterator
        console.print("  📊 Cross-validation evaluation:")
        cv_iterator = CVIterator(
            self.X[self.current_features], 
            self.y,
            self.model_type,
            self.problem_type,
            self.metric_name,
            n_splits=self.cv_folds,
            hyperparams=current_hyperparams,
            emoji_spinner=self.emoji_spinner
        )
        
        # Collect CV results using iterator
        fold_scores = []
        for fold_result in cv_iterator:
            fold_scores.append(fold_result['score'])
        
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        
        self.emoji_spinner.complete(f"CV Score: {mean_score:.4f} ± {std_score:.4f}")
        
        # Record results
        iteration_result = {
            'iteration': self.iteration,
            'n_features': len(self.current_features),
            'features': self.current_features.copy(),
            'mean_score': mean_score,
            'std_score': std_score,
            'fold_scores': fold_scores.copy(),
            'hyperparams': current_hyperparams.copy()
        }
        self.selection_history.append(iteration_result)
        
        # Check if this is the best score
        is_better = (
            (self.metric_name in ['rmse', 'mae'] and mean_score < self.best_score) or
            (self.metric_name not in ['rmse', 'mae'] and mean_score > self.best_score)
        )
        
        if is_better:
            self.best_score = mean_score
            self.best_features = self.current_features.copy()
            self.best_hyperparams = current_hyperparams.copy()
            self.iterations_without_improvement = 0
            
            console.print(f"  ⭐ NEW BEST! Score: {mean_score:.4f}, Features: {len(self.current_features)}")
        else:
            self.iterations_without_improvement += 1
            console.print(f"  📉 Score: {mean_score:.4f}, No improvement ({self.iterations_without_improvement}/{self.patience})")
        
        # Prepare for next iteration by removing features
        self._remove_features_for_next_iteration()
        
        return iteration_result
    
    def _remove_features_for_next_iteration(self):
        """Remove features for next iteration using lazy evaluation."""
        if len(self.current_features) <= self.min_features:
            return
        
        n_to_remove = max(1, int(len(self.current_features) * self.removal_ratio))
        n_to_remove = min(n_to_remove, len(self.current_features) - self.min_features)
        
        if n_to_remove <= 0:
            return
        
        # Random selection for simplicity (could be improved with feature importance)
        np.random.seed(42)
        features_to_remove = np.random.choice(
            self.current_features, size=n_to_remove, replace=False
        ).tolist()
        
        self.current_features = [f for f in self.current_features if f not in features_to_remove]
        console.print(f"  🗑️ Removed {len(features_to_remove)} features: {features_to_remove[:3]}{'...' if len(features_to_remove) > 3 else ''}")
    
    def _update_table_display(self):
        """Update the live table display."""
        # Clear and rebuild table with dynamic title
        title = f"Iterator-Based Feature Selection Progress (Best: {self.best_score:.4f})"
        self.table = Table(title=title)
        self.table.add_column("Iter", style="cyan", width=8)
        self.table.add_column("Features", style="magenta", width=8)
        self.table.add_column("Score", style="green", width=12)
        self.table.add_column("Accuracy", style="yellow", width=10)
        self.table.add_column("Loss", style="red", width=10)
        self.table.add_column("Status", style="blue", width=15)
        
        for result in self.table_results:
            # Format score with highlighting for best
            score_str = f"{result['score']:.6f}" if result['score'] > 0 else "-"
            if result['score'] == self.best_score and result['score'] > 0:
                score_str = f"[reverse]{score_str}[/reverse]"
            
            # Format accuracy 
            acc_str = f"{result['accuracy']:.3f}" if result['accuracy'] > 0 else "-"
            
            # Format loss
            loss_str = f"{result['loss']:.4f}" if result['loss'] > 0 and result['loss'] < float('inf') else "-"
            
            self.table.add_row(
                str(result['iteration']),
                str(result['features']),
                score_str,
                acc_str,
                loss_str,
                result['status']
            )
    
    def get_best_results(self) -> Tuple[float, float, List[str], Dict[str, Any]]:
        """Get the best results from feature selection."""
        return self.best_score, 0.0, self.best_features, self.best_hyperparams


class CVIterator:
    """Iterator for cross-validation folds with progress tracking."""
    
    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_type: str,
        problem_type: str,
        metric_name: str,
        n_splits: int = 3,
        hyperparams: Optional[Dict[str, Any]] = None,
        emoji_spinner: Optional[EmojiSpinner] = None,
        table_row: Optional[Dict[str, Any]] = None,
        table_updater: Optional[callable] = None
    ):
        self.X = X
        self.y = y
        self.model_type = model_type
        self.problem_type = problem_type
        self.metric_name = metric_name
        self.n_splits = n_splits
        self.hyperparams = hyperparams or {}
        self.emoji_spinner = emoji_spinner
        self.table_row = table_row
        self.table_updater = table_updater
        
        # Create CV splits
        from sklearn.model_selection import StratifiedKFold, KFold
        if problem_type in ['binary_classification', 'multiclass_classification']:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            self.cv_splits = list(cv.split(X, y))
        else:
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            self.cv_splits = list(cv.split(X))
        
        self.current_fold = 0
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Return iterator over CV folds."""
        return self
    
    def __next__(self) -> Dict[str, Any]:
        """Get next CV fold result."""
        if self.current_fold >= len(self.cv_splits):
            raise StopIteration
        
        train_idx, val_idx = self.cv_splits[self.current_fold]
        fold_num = self.current_fold + 1
        
        # Update progress spinner and table
        if self.emoji_spinner:
            spinner_emoji = ''.join(["🌙" if i < fold_num else "⭐" for i in range(self.n_splits)])
            if self.table_row:
                self.table_row['status'] = f"{self.table_row['iteration']} {spinner_emoji}"
            if self.table_updater:
                self.table_updater()
        
        # Prepare fold data
        X_train, X_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
        y_train, y_val = self.y.iloc[train_idx], self.y.iloc[val_idx]
        
        # Evaluate fold
        score, model = evaluate_fold(
            X_train, y_train, X_val, y_val,
            self.model_type, self.problem_type, self.metric_name,
            self.hyperparams, show_progress=False
        )
        
        # Calculate additional metrics for display
        accuracy = 0.0
        loss = 0.0
        
        try:
            # Make predictions for accuracy/loss calculation
            if needs_probabilities(self.metric_name):
                predictions = model.predict(X_val)
                if self.problem_type == 'binary_classification':
                    if hasattr(predictions, 'probability'):
                        y_pred = predictions.probability(1)
                    else:
                        y_pred = predictions
                else:
                    y_pred = predictions
            else:
                predictions = model.predict(X_val)
                
                # Handle string labels
                y_true_sample = y_val.values
                if len(y_true_sample) > 0 and isinstance(y_true_sample[0], str):
                    train_classes = sorted(y_train.unique())
                    if len(train_classes) == 2:
                        label_map = {0: train_classes[0], 1: train_classes[1]}
                        y_pred = np.array([label_map.get(int(p), p) for p in predictions])
                    else:
                        label_map = {i: cls for i, cls in enumerate(train_classes)}
                        y_pred = np.array([label_map.get(int(p), p) for p in predictions])
                else:
                    y_pred = predictions
            
            # Calculate accuracy
            if self.problem_type in ['binary_classification', 'multiclass_classification']:
                if not needs_probabilities(self.metric_name):
                    accuracy = np.mean(y_val.values == y_pred)
                else:
                    # For probability metrics, calculate accuracy from class predictions
                    if self.problem_type == 'binary_classification':
                        pred_classes = (y_pred > 0.5).astype(int)
                        if isinstance(y_val.values[0], str):
                            train_classes = sorted(y_train.unique())
                            label_map = {train_classes[0]: 0, train_classes[1]: 1}
                            y_true_numeric = np.array([label_map[val] for val in y_val.values])
                            accuracy = np.mean(y_true_numeric == pred_classes)
                        else:
                            accuracy = np.mean(y_val.values == pred_classes)
                    else:
                        pred_classes = np.argmax(y_pred, axis=1)
                        accuracy = np.mean(y_val.values == pred_classes)
                
                # Calculate log loss as a common loss metric
                try:
                    from sklearn.metrics import log_loss
                    if self.problem_type == 'binary_classification':
                        if isinstance(y_val.values[0], str):
                            train_classes = sorted(y_train.unique())
                            label_map = {train_classes[0]: 0, train_classes[1]: 1}
                            y_true_numeric = np.array([label_map[val] for val in y_val.values])
                            loss = log_loss(y_true_numeric, y_pred)
                        else:
                            loss = log_loss(y_val.values, y_pred)
                    else:
                        # For multiclass, y_pred should be probabilities
                        if len(y_pred.shape) == 1:
                            # Single value predictions, convert to probabilities
                            n_classes = len(np.unique(y_val.values))
                            y_pred_proba = np.zeros((len(y_pred), n_classes))
                            for i, pred in enumerate(y_pred):
                                y_pred_proba[i, int(pred)] = 1.0
                            loss = log_loss(y_val.values, y_pred_proba)
                        else:
                            loss = log_loss(y_val.values, y_pred)
                except:
                    loss = 0.0
            else:
                # Regression - use MSE as loss
                accuracy = 0.0  # No accuracy for regression
                loss = np.mean((y_val.values - y_pred) ** 2)
                
        except Exception as e:
            # If any error occurs, use defaults
            accuracy = 0.0
            loss = 0.0
        
        fold_result = {
            'fold': fold_num,
            'score': score,
            'accuracy': accuracy,
            'loss': loss,
            'model': model,
            'train_size': len(train_idx),
            'val_size': len(val_idx)
        }
        
        self.current_fold += 1
        return fold_result


class MDMBenchmarkV10:
    """Iterator-based benchmark with emoji spinners and lazy evaluation."""
    
    def __init__(self, output_dir: str = "benchmark_results", use_cache: bool = True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.use_cache = use_cache
        
        # Get MDM components (already initialized via DI)
        self.config_manager = get_config_manager()
        self.dataset_manager = DatasetManager()
        self.dataset_registrar = DatasetRegistrar()
        
        self.results = {
            'version': 'Version 10: Iterator-Based Custom Backward Feature Selection with CV inside',
            'description': 'Iterator approach with emoji spinners, lazy evaluation, and generators',
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
            
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
            conn.close()
            
            if not with_features:
                original_cols = []
                for col in df.columns:
                    if col == config['target'] or col == config.get('id_column', 'id'):
                        original_cols.append(col)
                    elif not any(suffix in col for suffix in [
                        '_zscore', '_log', '_sqrt', '_squared', '_is_outlier',
                        '_percentile_rank', '_year', '_month', '_day', '_hour',
                        '_frequency', '_target_mean', '_length', '_word_count',
                        '_is_missing', '_binned', '_x_', '_lag_', '_rolling_'
                    ]):
                        original_cols.append(col)
                
                df = df[original_cols]
            
            return df
            
        except Exception as e:
            console.print(f"  ✗ Failed to load {dataset_name}: {str(e)}", style="red")
            return None
    
    def select_features_with_iterator(
        self,
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
        Feature selection using iterator approach.
        
        Returns:
            Tuple of (mean_score, std_score, selected_features, best_hyperparams)
        """
        console.print(f"  🚀 Starting iterator-based feature selection with {len(X.columns)} features")
        
        # Create feature iterator
        feature_iterator = FeatureIterator(
            X, y, model_type, problem_type, metric_name,
            removal_ratio=removal_ratio,
            min_features=5,
            patience=3,
            cv_folds=cv_folds,
            use_tuning=use_tuning
        )
        
        # Process iterations using iterator with live table display
        iteration_count = 0
        with Live(feature_iterator.table, refresh_per_second=2, console=console) as live:
            for iteration_result in feature_iterator:
                iteration_count += 1
                # Update live display
                live.update(feature_iterator.table)
                time.sleep(0.1)  # Small delay to show updates
        
        # Get final results
        best_score, best_std, best_features, best_hyperparams = feature_iterator.get_best_results()
        
        console.print(f"  🎯 Feature selection completed: {len(best_features)} features selected")
        console.print(f"     Best score: {best_score:.4f}")
        console.print(f"     Iterations: {iteration_count}")
        
        return best_score, best_std, best_features, best_hyperparams
    
    def benchmark_competition(self, name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Benchmark a single competition using iterator-based approach."""
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
            
            # With features - iterator-based feature selection with CV inside
            console.print("  Iterator-based feature selection with CV inside...")
            try:
                # Remove target from features for feature selection
                X_features = df_features.drop(columns=[config['target']])
                y_features = df_features[config['target']]
                
                # Use iterator-based feature selector
                mean_with, std_with, selected_features, best_hyperparams = self.select_features_with_iterator(
                    X_features,
                    y_features,
                    model_type,
                    config['problem_type'],
                    config['metric'],
                    cv_folds=3,  # Default CV=3
                    removal_ratio=0.1,  # Default removal_ratio=0.1
                    use_tuning=True  # Default tuning=True
                )
                
                results['with_features'][model_type] = {
                    'mean_score': round(mean_with, 4),
                    'std': round(std_with, 4),
                    'n_features': n_features_with,
                    'n_selected': len(selected_features) if selected_features else n_features_with,
                    'best_features': selected_features[:20] if selected_features else [],
                    'best_hyperparams': best_hyperparams,
                    'method': 'Iterator-based backward selection with 3-fold CV inside'
                }
                console.print(f"    ✓ Score: {mean_with:.4f} ± {std_with:.4f}")
                console.print(f"    → Selected features: {len(selected_features) if selected_features else 0}")
                
            except Exception as e:
                console.print(f"    ✗ Failed: {str(e)}", style="red")
                results['with_features'][model_type] = {'error': str(e)}
            
            # Without features - simple CV using iterator
            console.print("  Training without features (iterator CV evaluator)...")
            try:
                X_raw = df_raw.drop(columns=[config['target']])
                y_raw = df_raw[config['target']]
                
                # Use iterator-based CV evaluator for baseline
                cv_iterator = CVIterator(
                    X_raw, y_raw, model_type, config['problem_type'], 
                    config['metric'], n_splits=3,
                    emoji_spinner=EmojiSpinner(total_folds=3, emoji_set='balloon_gift')
                )
                
                fold_scores = []
                for fold_result in cv_iterator:
                    fold_scores.append(fold_result['score'])
                
                mean_without = np.mean(fold_scores)
                std_without = np.std(fold_scores)
                
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
            f"[bold]Version 10: Iterator-Based Custom Backward Feature Selection with CV inside[/bold]\n"
            f"Iterator approach with emoji spinners, lazy evaluation, and generators\n"
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
        output_file = self.output_dir / f"v10_results_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        console.print(f"\n[green]Results saved to: {output_file}[/green]")
    
    def display_summary(self):
        """Display summary table."""
        table = Table(title="Benchmark Summary - Version 10", show_header=True)
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
        description="Version 10: Iterator-Based Custom Backward Feature Selection with CV inside"
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
    
    benchmark = MDMBenchmarkV10(
        output_dir=args.output_dir,
        use_cache=not args.no_cache
    )
    
    benchmark.run_benchmark(competitions=args.competitions)


if __name__ == '__main__':
    main()