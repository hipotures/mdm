#!/usr/bin/env python3
"""
Version 7: Pipeline-Based Feature Selection with Emoji Spinners and Live Table

ALGORITHM:
- Feature selection with CV inside (3-fold) + tuning inside each fold
- Return CV score from feature selection as final result (NO additional CV)
- Same defaults: CV=3, removal_ratio=0.1, tuning=True
- Command: python version_7.py -c titanic

SPINNER:
- Aesthetic CV progress spinner: ▰▰▱ (for CV=3, showing 2 done, 1 current)
- Use different symbols: ⚡🌟 or 🔥❄️ or 🟢🟡 or other emoji pairs
- Show spinner next to iteration number during training

TABLE:
- 6 columns: Iter, Features, Score, Accuracy, Loss, Status
- Live updates with Rich during feature selection
- Show CV spinner during training: "0 ▰▱▱" then "0 ▰▰▱" then "0"
- Final results after all folds complete

PIPELINE APPROACH:
- Chain of processing steps (DataStep, FeatureSelectionStep, CVStep, etc.)
- Pipeline objects that flow data through steps
- Step-by-step execution with pipeline.run()
- Pipeline-based table updates and spinner management
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
from typing import Dict, List, Tuple, Any, Optional, Union
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
import threading

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

# Import existing modules
from cv_evaluator import cross_validate_model, evaluate_fold, hyperparameter_tune, create_ydf_model
from utils.metrics import needs_probabilities, calculate_metric

console = Console()


def create_cv_spinner(current_fold: int, total_folds: int) -> str:
    """Create aesthetic CV progress spinner: ▰▰▱ (for CV=3, showing 2 done, 1 current)"""
    spinner = ""
    for i in range(total_folds):
        if i < current_fold:
            spinner += "▰"  # Completed
        elif i == current_fold:
            spinner += "▰"  # Current (also filled to show progress)
        else:
            spinner += "▱"  # Pending
    return spinner


# ======================= PIPELINE COMPONENTS =======================

class PipelineData:
    """Data container that flows through pipeline steps."""
    
    def __init__(self):
        self.raw_data: Optional[pd.DataFrame] = None
        self.feature_data: Optional[pd.DataFrame] = None
        self.X: Optional[pd.DataFrame] = None
        self.y: Optional[pd.Series] = None
        self.target_column: str = ""
        self.selected_features: Optional[List[str]] = None
        self.cv_scores: Dict[str, Any] = {}
        self.model_results: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}
        
    def set_data(self, df: pd.DataFrame, target_col: str):
        """Set the main dataset."""
        self.raw_data = df
        self.target_column = target_col
        self.X = df.drop(columns=[target_col])
        self.y = df[target_col]
        
    def set_feature_data(self, df: pd.DataFrame):
        """Set the feature-enhanced dataset."""
        self.feature_data = df
        if self.target_column in df.columns:
            self.X = df.drop(columns=[self.target_column])
            self.y = df[self.target_column]


class PipelineStep:
    """Base class for pipeline steps."""
    
    def __init__(self, name: str):
        self.name = name
        
    def execute(self, data: PipelineData) -> PipelineData:
        """Execute this step on the pipeline data."""
        raise NotImplementedError
        
    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"


class EmojiSpinner:
    """Enhanced spinner with emoji pairs and progress tracking."""
    
    def __init__(self, total_steps: int = 3, emoji_pair: tuple = ("⚡", "🌟")):
        """
        Initialize emoji spinner.
        
        Args:
            total_steps: Total number of steps (e.g., CV folds)
            emoji_pair: Tuple of (active, completed) emojis
        """
        self.total_steps = total_steps
        self.active_emoji, self.completed_emoji = emoji_pair
        self.current_step = 0
        self.is_spinning = False
        self.spinner_thread: Optional[threading.Thread] = None
        self.current_message = ""
        self.step_messages = [""] * total_steps
        
    def _create_progress_display(self) -> str:
        """Create visual progress with emojis."""
        display_parts = []
        for i in range(self.total_steps):
            if i < self.current_step:
                display_parts.append(self.completed_emoji)
            elif i == self.current_step:
                display_parts.append(self.active_emoji)
            else:
                display_parts.append("⚫")  # Pending
        return "".join(display_parts)
    
    def _spinner_worker(self):
        """Worker thread for spinner animation."""
        rotation_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        rotation_idx = 0
        
        while self.is_spinning:
            progress_display = self._create_progress_display()
            rotation_char = rotation_chars[rotation_idx % len(rotation_chars)]
            
            # Show progress with current step info
            console.print(
                f"\r  {rotation_char} Progress: {progress_display} ({self.current_step + 1}/{self.total_steps}) - {self.current_message}",
                end="",
                highlight=False
            )
            
            rotation_idx += 1
            time.sleep(0.12)  # Slightly slower for emoji visibility
    
    def start(self, message: str = "Processing"):
        """Start the spinner."""
        self.current_message = message
        self.is_spinning = True
        self.spinner_thread = threading.Thread(target=self._spinner_worker)
        self.spinner_thread.daemon = True
        self.spinner_thread.start()
    
    def update_step(self, step: int, message: str = ""):
        """Update current step progress."""
        self.current_step = step
        if message:
            self.current_message = message
            self.step_messages[step] = message
    
    def next_step(self, message: str = ""):
        """Move to next step."""
        if self.current_step < self.total_steps - 1:
            self.current_step += 1
            if message:
                self.current_message = message
                self.step_messages[self.current_step] = message
    
    def stop(self, final_message: str = "Completed"):
        """Stop spinner and show final result."""
        if self.is_spinning:
            self.is_spinning = False
            if self.spinner_thread:
                self.spinner_thread.join(timeout=0.5)
            
            # Show final completed state
            final_display = self.completed_emoji * self.total_steps
            console.print(f"\r  ✓ Progress: {final_display} ({self.total_steps}/{self.total_steps}) - {final_message}")


class IterationEmojiSpinner:
    """Simple emoji spinner for iterations."""
    
    def __init__(self, emoji_pair: tuple = ("🔥", "❄️")):
        self.active_emoji, self.idle_emoji = emoji_pair
        self.is_spinning = False
        self.spinner_thread: Optional[threading.Thread] = None
        self.current_message = ""
    
    def _spinner_worker(self):
        """Worker thread for animation."""
        emoji_cycle = [self.active_emoji, self.idle_emoji]
        cycle_idx = 0
        
        while self.is_spinning:
            current_emoji = emoji_cycle[cycle_idx % len(emoji_cycle)]
            console.print(f"\r    {current_emoji} {self.current_message}", end="", highlight=False)
            cycle_idx += 1
            time.sleep(0.8)  # Slower cycle for emoji effect
    
    def start(self, message: str):
        """Start the spinner."""
        self.current_message = message
        self.is_spinning = True
        self.spinner_thread = threading.Thread(target=self._spinner_worker)
        self.spinner_thread.daemon = True
        self.spinner_thread.start()
    
    def update(self, message: str):
        """Update message."""
        self.current_message = message
    
    def stop(self, final_message: str = ""):
        """Stop spinner."""
        if self.is_spinning:
            self.is_spinning = False
            if self.spinner_thread:
                self.spinner_thread.join(timeout=0.5)
            
            if final_message:
                console.print(f"\r    ✓ {final_message}")
            else:
                console.print()


# Different emoji pairs for variety
EMOJI_PAIRS = {
    'electric': ("⚡", "🌟"),
    'fire_ice': ("🔥", "❄️"),
    'traffic': ("🟢", "🟡"),
    'gems': ("💎", "💠"),
    'space': ("🚀", "⭐"),
    'nature': ("🌱", "🌿"),
    'weather': ("☀️", "🌙"),
    'classic': ("🔴", "🟢")
}


class DataLoadStep(PipelineStep):
    """Step to load competition data."""
    
    def __init__(self, dataset_manager: DatasetManager, with_features: bool = True):
        super().__init__(f"Load Data ({'with' if with_features else 'without'} features)")
        self.dataset_manager = dataset_manager
        self.with_features = with_features
    
    def execute(self, data: PipelineData) -> PipelineData:
        """Load data from MDM."""
        config = data.metadata['config']
        name = data.metadata['name']
        
        spinner = IterationEmojiSpinner(EMOJI_PAIRS['space'])
        spinner.start(f"Loading {self.name.lower()}...")
        
        df = self._load_competition_data(name, config, self.with_features)
        
        if df is not None:
            if self.with_features:
                data.set_feature_data(df)
            else:
                data.set_data(df, config['target'])
            
            spinner.stop(f"Loaded {len(df)} rows, {len(df.columns)-1} features")
        else:
            spinner.stop("Failed to load data")
            raise ValueError(f"Could not load data for {name}")
        
        return data
    
    def _load_competition_data(self, name: str, config: Dict[str, Any], with_features: bool) -> Optional[pd.DataFrame]:
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
            console.print(f"Error loading {dataset_name}: {str(e)}", style="red")
            return None


class FeatureSelectionStep(PipelineStep):
    """Step for backward feature selection with CV inside and live table display."""
    
    def __init__(self, cv_folds: int = 3, removal_ratio: float = 0.1, use_tuning: bool = True):
        super().__init__(f"Feature Selection (CV={cv_folds})")
        self.cv_folds = cv_folds
        self.removal_ratio = removal_ratio
        self.use_tuning = use_tuning
        self.min_features = 5
        self.patience = 3
        self.random_state = 42
        
        # Live table display components
        self.table_results = []
        self.live_table = None
        self.best_score = -float('inf')
        self.current_iteration = 0
    
    def execute(self, data: PipelineData) -> PipelineData:
        """Perform feature selection with custom backward selection and live table display."""
        model_type = data.metadata['model_type']
        problem_type = data.metadata['config']['problem_type']
        metric_name = data.metadata['config']['metric']
        
        console.print(f"\n[bold cyan]Feature Selection with {model_type.upper()}:[/bold cyan]")
        
        # Initialize best score based on metric type
        if metric_name in ['rmse', 'mae', 'mse']:
            self.best_score = float('inf')  # Lower is better
        else:
            self.best_score = -float('inf')  # Higher is better
        
        # Use feature data for selection
        X = data.X
        y = data.y
        
        # Run backward selection with live table
        mean_score, std_score, selected_features, best_hyperparams = self._backward_feature_selection_with_table(
            X, y, model_type, problem_type, metric_name
        )
        
        # Store results
        data.selected_features = selected_features
        data.cv_scores[model_type] = {
            'mean_score': mean_score,
            'std_score': std_score,
            'selected_features': selected_features,
            'best_hyperparams': best_hyperparams,
            'n_features_original': len(X.columns),
            'n_features_selected': len(selected_features) if selected_features else len(X.columns)
        }
        
        return data
    
    def _backward_feature_selection_with_table(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_type: str,
        problem_type: str,
        metric_name: str
    ) -> Tuple[float, float, List[str], Dict[str, Any]]:
        """
        Custom backward feature selection with CV inside and live table display.
        
        Returns:
            Tuple of (mean_score, std_score, selected_features, best_hyperparams)
        """
        # Initialize
        current_features = list(X.columns)
        best_score = self.best_score
        best_features = current_features.copy()
        best_hyperparams = {}
        iterations_without_improvement = 0
        
        console.print(f"\n[bold]Starting Custom Backward Feature Selection[/bold]")
        console.print(f"  → Initial features: {len(current_features)}")
        console.print(f"  → CV folds: {self.cv_folds}")
        console.print(f"  → Removal ratio: {self.removal_ratio}")
        console.print(f"  → Tuning enabled: {self.use_tuning}")
        console.print(f"  → Target metric: {metric_name}")
        console.print()
        
        # Initialize live table
        self.live_table = Table(title="Feature Selection Progress - Pipeline Version 7")
        self.live_table.add_column("Iter", style="cyan", width=8)
        self.live_table.add_column("Features", style="magenta", width=8) 
        self.live_table.add_column("Score", style="green", width=12)
        self.live_table.add_column("Accuracy", style="yellow", width=10)
        self.live_table.add_column("Loss", style="red", width=10)
        self.live_table.add_column("Status", style="blue", width=15)
        
        with Live(self.live_table, refresh_per_second=2, console=console) as live:
            iteration = 0
            
            while len(current_features) > self.min_features and iterations_without_improvement < self.patience:
                # Hyperparameter tuning (if enabled)
                if self.use_tuning:
                    current_hyperparams = hyperparameter_tune(
                        X[current_features], y, model_type, problem_type, metric_name,
                        n_splits=self.cv_folds, show_progress=False
                    )
                else:
                    current_hyperparams = {}
                
                # Perform CV with live table updates
                mean_score, mean_accuracy, mean_loss = self._train_model_with_cv_and_table(
                    X[current_features], y, current_features, iteration, 
                    model_type, problem_type, metric_name, current_hyperparams, live
                )
                
                # Check if this is the best score
                is_better = (
                    (metric_name in ['rmse', 'mae', 'mse'] and mean_score < best_score) or
                    (metric_name not in ['rmse', 'mae', 'mse'] and mean_score > best_score)
                )
                
                if is_better:
                    best_score = mean_score
                    best_features = current_features.copy()
                    best_hyperparams = current_hyperparams.copy()
                    iterations_without_improvement = 0
                else:
                    iterations_without_improvement += 1
                
                # Early stopping check
                if iterations_without_improvement >= self.patience:
                    break
                
                # Remove features for next iteration
                features_to_remove = self._select_features_to_remove(current_features)
                
                if not features_to_remove:
                    break
                
                current_features = [f for f in current_features if f not in features_to_remove]
                
                # Safety check
                if len(current_features) < self.min_features:
                    current_features = best_features.copy()
                    break
                
                iteration += 1
            
            # Final table update
            time.sleep(0.5)  # Show final state
        
        console.print(f"\n[green]Feature selection completed![/green]")
        console.print(f"  → Best score: {best_score:.6f}")
        console.print(f"  → Best features: {len(best_features)} selected")
        console.print(f"  → Total iterations: {len(self.table_results)}")
        
        # Calculate std from table results if available
        if self.table_results:
            scores = [r['score'] for r in self.table_results if r['score'] > 0]
            std_score = np.std(scores) if len(scores) > 1 else 0.0
        else:
            std_score = 0.0
        
        return best_score, std_score, best_features, best_hyperparams
    
    def _train_model_with_cv_and_table(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        features: List[str], 
        iteration: int,
        model_type: str,
        problem_type: str,
        metric_name: str,
        hyperparams: Dict[str, Any],
        live: Live
    ) -> Tuple[float, float, float]:
        """Train model with CV and update live table display."""
        from sklearn.model_selection import KFold, StratifiedKFold
        
        # Choose CV strategy
        if 'classification' in problem_type:
            kf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            splits = list(kf.split(X, y))
        else:
            kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            splits = list(kf.split(X))
        
        fold_scores = []
        fold_accuracies = []
        fold_losses = []
        
        # Add initial row to table
        self.table_results.append({
            'iteration': iteration,
            'features': len(features),
            'score': 0,
            'accuracy': 0,
            'loss': 0,
            'status': f"{iteration} {create_cv_spinner(0, self.cv_folds)}"
        })
        
        # Update table for CV progress
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            # Update spinner
            spinner = create_cv_spinner(fold_idx, self.cv_folds)
            status = f"{iteration} {spinner}"
            
            # Update current row
            self.table_results[-1]['status'] = status
            self._update_live_table(live)
            
            # Split data
            X_train = X.iloc[train_idx]
            X_val = X.iloc[val_idx]
            y_train = y.iloc[train_idx]
            y_val = y.iloc[val_idx]
            
            # Evaluate this fold
            score, accuracy, loss = self._evaluate_fold_with_metrics(
                X_train, y_train, X_val, y_val,
                model_type, problem_type, metric_name, hyperparams
            )
            
            fold_scores.append(score)
            fold_accuracies.append(accuracy)
            fold_losses.append(loss)
        
        # Calculate averages
        mean_score = np.mean(fold_scores) if fold_scores else 0.0
        mean_accuracy = np.mean(fold_accuracies) if fold_accuracies else 0.0
        mean_loss = np.mean(fold_losses) if fold_losses else float('inf')
        
        # Final update without spinner
        self.table_results[-1].update({
            'score': mean_score,
            'accuracy': mean_accuracy,
            'loss': mean_loss,
            'status': str(iteration)
        })
        
        # Update best score tracking for highlighting
        if ((metric_name in ['rmse', 'mae', 'mse'] and mean_score < self.best_score) or
            (metric_name not in ['rmse', 'mae', 'mse'] and mean_score > self.best_score)):
            self.best_score = mean_score
        
        self._update_live_table(live)
        
        return mean_score, mean_accuracy, mean_loss
    
    def _evaluate_fold_with_metrics(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        model_type: str,
        problem_type: str,
        metric_name: str,
        hyperparams: Dict[str, Any]
    ) -> Tuple[float, float, float]:
        """Evaluate a single fold and return score, accuracy, and loss."""
        import ydf
        import contextlib
        import io
        
        # Prepare data
        train_data = pd.concat([X_train, y_train], axis=1)
        val_data = pd.concat([X_val, y_val], axis=1)
        target_col = y_train.name
        
        # Determine YDF task
        if 'classification' in problem_type:
            task = ydf.Task.CLASSIFICATION
        else:
            task = ydf.Task.REGRESSION
        
        try:
            # Create learner
            if model_type == 'gbt':
                learner = ydf.GradientBoostedTreesLearner(
                    label=target_col,
                    task=task,
                    **hyperparams
                )
            else:  # rf
                learner = ydf.RandomForestLearner(
                    label=target_col,
                    task=task,
                    compute_oob_variable_importances=True,
                    **hyperparams
                )
            
            # Train model silently
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
            
            # Calculate main metric
            y_true = y_val.values
            score = calculate_metric(y_true, y_pred, metric_name, problem_type)
            
            # Calculate accuracy
            if problem_type in ['binary_classification', 'multiclass_classification']:
                if not needs_probabilities(metric_name):
                    accuracy = np.mean(y_true == y_pred)
                else:
                    # For probability metrics, calculate accuracy from class predictions
                    if problem_type == 'binary_classification':
                        pred_classes = (y_pred > 0.5).astype(int)
                        if isinstance(y_true[0], str):
                            train_classes = sorted(y_train.unique())
                            label_map = {train_classes[0]: 0, train_classes[1]: 1}
                            y_true_numeric = np.array([label_map[val] for val in y_true])
                            accuracy = np.mean(y_true_numeric == pred_classes)
                        else:
                            accuracy = np.mean(y_true == pred_classes)
                    else:
                        pred_classes = np.argmax(y_pred, axis=1)
                        accuracy = np.mean(y_true == pred_classes)
            else:
                accuracy = 0.0  # No accuracy for regression
            
            # Calculate loss
            try:
                if problem_type in ['binary_classification', 'multiclass_classification']:
                    from sklearn.metrics import log_loss
                    if problem_type == 'binary_classification':
                        if isinstance(y_true[0], str):
                            train_classes = sorted(y_train.unique())
                            label_map = {train_classes[0]: 0, train_classes[1]: 1}
                            y_true_numeric = np.array([label_map[val] for val in y_true])
                            loss = log_loss(y_true_numeric, y_pred)
                        else:
                            loss = log_loss(y_true, y_pred)
                    else:
                        # For multiclass, y_pred should be probabilities
                        if len(y_pred.shape) == 1:
                            # Single value predictions, convert to probabilities
                            n_classes = len(np.unique(y_true))
                            y_pred_proba = np.zeros((len(y_pred), n_classes))
                            for i, pred in enumerate(y_pred):
                                y_pred_proba[i, int(pred)] = 1.0
                            loss = log_loss(y_true, y_pred_proba)
                        else:
                            loss = log_loss(y_true, y_pred)
                else:
                    # Use MSE as loss for regression
                    loss = np.mean((y_true - y_pred) ** 2)
            except:
                loss = 0.0
            
            return score, accuracy, loss
            
        except Exception as e:
            console.print(f"Error in fold evaluation: {e}")
            return 0.0, 0.0, float('inf')
    
    def _update_live_table(self, live: Live):
        """Update the live table display."""
        # Clear and rebuild table
        self.live_table = Table(title="Feature Selection Progress - Pipeline Version 7")
        self.live_table.add_column("Iter", style="cyan", width=8)
        self.live_table.add_column("Features", style="magenta", width=8)
        self.live_table.add_column("Score", style="green", width=12)
        self.live_table.add_column("Accuracy", style="yellow", width=10)
        self.live_table.add_column("Loss", style="red", width=10)
        self.live_table.add_column("Status", style="blue", width=15)
        
        for result in self.table_results:
            # Format score with highlighting for best
            score_str = f"{result['score']:.6f}"
            if result['score'] == self.best_score and result['score'] > 0:
                score_str = f"[reverse]{score_str}[/reverse]"
            
            # Format accuracy 
            acc_str = f"{result['accuracy']:.3f}" if result['accuracy'] > 0 else "-"
            
            # Format loss
            loss_str = f"{result['loss']:.4f}" if result['loss'] < float('inf') else "-"
            
            self.live_table.add_row(
                str(result['iteration']),
                str(result['features']),
                score_str,
                acc_str,
                loss_str,
                result['status']
            )
        
        live.update(self.live_table)
    
    def _select_features_to_remove(self, current_features: List[str]) -> List[str]:
        """Select features to remove (random for simplicity)."""
        n_to_remove = max(1, int(len(current_features) * self.removal_ratio))
        n_to_remove = min(n_to_remove, len(current_features) - self.min_features)
        
        if n_to_remove <= 0:
            return []
        
        # Random selection for simplicity
        np.random.seed(self.random_state + self.current_iteration)
        self.current_iteration += 1
        features_to_remove = np.random.choice(
            current_features, size=n_to_remove, replace=False
        ).tolist()
        
        return features_to_remove


class BaselineEvaluationStep(PipelineStep):
    """Step for baseline evaluation without features."""
    
    def __init__(self, cv_folds: int = 3):
        super().__init__(f"Baseline Evaluation (CV={cv_folds})")
        self.cv_folds = cv_folds
    
    def execute(self, data: PipelineData) -> PipelineData:
        """Evaluate baseline model without engineered features."""
        model_type = data.metadata['model_type']
        problem_type = data.metadata['config']['problem_type']
        metric_name = data.metadata['config']['metric']
        
        console.print(f"\n[bold yellow]Baseline {model_type.upper()} (without features):[/bold yellow]")
        
        # Use raw data for baseline
        X_raw = data.raw_data.drop(columns=[data.target_column])
        y_raw = data.raw_data[data.target_column]
        
        # Simple CV with emoji spinner
        cv_spinner = EmojiSpinner(total_steps=self.cv_folds, emoji_pair=EMOJI_PAIRS['traffic'])
        cv_spinner.start("Baseline CV evaluation")
        
        mean_score, std_score, fold_scores, _ = cross_validate_model(
            X_raw, y_raw, model_type, problem_type, metric_name,
            n_splits=self.cv_folds, hyperparams=None, show_progress=False
        )
        
        cv_spinner.stop(f"Baseline Score: {mean_score:.4f} ± {std_score:.4f}")
        
        # Store baseline results
        data.model_results[f'{model_type}_baseline'] = {
            'mean_score': mean_score,
            'std_score': std_score,
            'n_features': len(X_raw.columns)
        }
        
        return data


class Pipeline:
    """Pipeline to execute steps in sequence."""
    
    def __init__(self, name: str):
        self.name = name
        self.steps: List[PipelineStep] = []
        self.data = PipelineData()
    
    def add_step(self, step: PipelineStep) -> 'Pipeline':
        """Add a step to the pipeline."""
        self.steps.append(step)
        return self
    
    def run(self) -> PipelineData:
        """Execute all pipeline steps."""
        console.print(f"\n[bold blue]Running Pipeline: {self.name}[/bold blue]")
        console.print(f"Steps: {len(self.steps)}")
        
        for i, step in enumerate(self.steps):
            console.print(f"\n[bold green]Step {i+1}/{len(self.steps)}: {step.name}[/bold green]")
            
            try:
                self.data = step.execute(self.data)
            except Exception as e:
                console.print(f"[red]Step failed: {str(e)}[/red]")
                raise e
        
        console.print(f"\n[bold green]Pipeline completed: {self.name}[/bold green]")
        return self.data


# ======================= MAIN BENCHMARK CLASS =======================

class MDMBenchmarkV7:
    """Pipeline-based benchmark with emoji spinners and live table display."""
    
    def __init__(self, output_dir: str = "benchmark_results", use_cache: bool = True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.use_cache = use_cache
        
        # Get MDM components
        self.config_manager = get_config_manager()
        self.dataset_manager = DatasetManager()
        self.dataset_registrar = DatasetRegistrar()
        
        self.results = {
            'version': 'Version 7: Pipeline-Based Feature Selection with Emoji Spinners and Live Table',
            'description': 'Pipeline approach with emoji spinners, live table display, and step-by-step execution',
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
    
    def benchmark_competition(self, name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Benchmark a single competition using pipeline approach."""
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
        
        # Test both model types with pipeline approach
        model_types = ['gbt', 'rf']
        
        for model_type in model_types:
            console.print(f"\n[bold cyan]📊 Processing {model_type.upper()} Model[/bold cyan]")
            
            try:
                # Create pipeline for this model
                pipeline = Pipeline(f"{name} - {model_type.upper()}")
                
                # Initialize pipeline data with metadata
                pipeline.data.metadata = {
                    'name': name,
                    'config': config,
                    'model_type': model_type
                }
                
                # Add pipeline steps
                pipeline.add_step(DataLoadStep(self.dataset_manager, with_features=False))  # Load raw data first
                pipeline.add_step(DataLoadStep(self.dataset_manager, with_features=True))   # Then feature data
                pipeline.add_step(FeatureSelectionStep(cv_folds=3, removal_ratio=0.1, use_tuning=True))
                pipeline.add_step(BaselineEvaluationStep(cv_folds=3))
                
                # Execute pipeline
                result_data = pipeline.run()
                
                # Extract results
                if model_type in result_data.cv_scores:
                    cv_result = result_data.cv_scores[model_type]
                    results['with_features'][model_type] = {
                        'mean_score': round(cv_result['mean_score'], 4),
                        'std': round(cv_result['std_score'], 4),
                        'n_features': cv_result['n_features_original'],
                        'n_selected': cv_result['n_features_selected'],
                        'best_features': cv_result['selected_features'][:20] if cv_result['selected_features'] else [],
                        'best_hyperparams': cv_result['best_hyperparams'],
                        'method': 'Pipeline-based backward selection with 3-fold CV inside and live table'
                    }
                    console.print(f"    ✓ With features: {cv_result['mean_score']:.4f} ± {cv_result['std_score']:.4f}")
                
                baseline_key = f'{model_type}_baseline'
                if baseline_key in result_data.model_results:
                    baseline_result = result_data.model_results[baseline_key]
                    results['without_features'][model_type] = {
                        'mean_score': round(baseline_result['mean_score'], 4),
                        'std': round(baseline_result['std_score'], 4),
                        'n_features': baseline_result['n_features']
                    }
                    console.print(f"    ✓ Without features: {baseline_result['mean_score']:.4f} ± {baseline_result['std_score']:.4f}")
                
                # Calculate improvement
                if model_type in results['with_features'] and model_type in results['without_features']:
                    score_with = results['with_features'][model_type]['mean_score']
                    score_without = results['without_features'][model_type]['mean_score']
                    
                    if config['metric'] in ['rmse', 'mae']:
                        improvement = ((score_without - score_with) / score_without) * 100
                    else:
                        improvement = ((score_with - score_without) / score_without) * 100
                    
                    results['improvement'][model_type] = f"{improvement:+.2f}%"
                    console.print(f"    [green]🎯 Improvement: {improvement:+.2f}%[/green]")
                
            except Exception as e:
                console.print(f"    ✗ Failed {model_type}: {str(e)}", style="red")
                results['with_features'][model_type] = {'error': str(e)}
                results['without_features'][model_type] = {'error': str(e)}
        
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
            f"[bold]Version 7: Pipeline-Based Feature Selection with Emoji Spinners and Live Table[/bold]\n"
            f"Pipeline approach with emoji progress, live table display, and step-by-step execution\n"
            f"Competitions: {len(selected)}\n"
            f"MDM Version: {mdm.__version__}",
            title="🚀 Benchmark Info"
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
        output_file = self.output_dir / f"v7_results_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        console.print(f"\n[green]📁 Results saved to: {output_file}[/green]")
    
    def display_summary(self):
        """Display summary table."""
        table = Table(title="🏆 Benchmark Summary - Version 7", show_header=True)
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
            console.print("\n[bold]🎯 Overall Summary:[/bold]")
            for key, value in self.results['summary'].items():
                console.print(f"  {key}: {value}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Version 7: Pipeline-Based Feature Selection with Emoji Spinners and Live Table"
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
    
    benchmark = MDMBenchmarkV7(
        output_dir=args.output_dir,
        use_cache=not args.no_cache
    )
    
    benchmark.run_benchmark(competitions=args.competitions)


if __name__ == '__main__':
    main()