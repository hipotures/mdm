#!/usr/bin/env python3
"""
Version 8: Event-Driven Feature Selection with CV inside

ALGORITHM:
- Feature selection with CV inside (3-fold) + tuning inside each fold
- Return CV score from feature selection as final result (NO additional CV)
- Same defaults: CV=3, removal_ratio=0.1, tuning=True
- Command: python version_8.py -c titanic

SPINNER:
- Aesthetic CV progress spinner: ▰▰▱ (for CV=3, showing 2 done, 1 current)
- Use different symbols: ⭐💫 or 🎯🎪 or 🌸🌺 or other emoji pairs
- Show spinner next to iteration number during training

EVENT-DRIVEN APPROACH:
- Event system with EventBus, EventHandler classes
- Observer pattern for table updates and spinners
- Events: fold_started, fold_completed, iteration_started, etc.
- Event-based progress tracking and table management
- But same core algorithm with custom backward selection

Makes use of custom backward selection implementation (NOT cross_validate_ydf).
"""

import os
import sys
import json
import argparse
import signal
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
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
from rich.live import Live
from rich.text import Text
from rich import print as rprint

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

# Import modular components
from cv_evaluator import cross_validate_model, evaluate_fold, hyperparameter_tune, create_ydf_model

console = Console()


# ============================================================================
# EVENT SYSTEM
# ============================================================================

class EventType(Enum):
    """Types of events in the system."""
    FOLD_STARTED = "fold_started"
    FOLD_COMPLETED = "fold_completed"
    ITERATION_STARTED = "iteration_started"
    ITERATION_COMPLETED = "iteration_completed"
    FEATURE_SELECTION_STARTED = "feature_selection_started"
    FEATURE_SELECTION_COMPLETED = "feature_selection_completed"
    CV_STARTED = "cv_started"
    CV_COMPLETED = "cv_completed"
    HYPERPARAMETER_TUNING_STARTED = "hyperparameter_tuning_started"
    HYPERPARAMETER_TUNING_COMPLETED = "hyperparameter_tuning_completed"


@dataclass
class Event:
    """Event data structure."""
    type: EventType
    data: Dict[str, Any]
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class EventHandler:
    """Base class for event handlers."""
    
    def handle(self, event: Event) -> None:
        """Handle an event."""
        pass


class EventBus:
    """Event bus for managing event handlers and dispatching events."""
    
    def __init__(self):
        self.handlers: Dict[EventType, List[EventHandler]] = {}
        self.global_handlers: List[EventHandler] = []
    
    def subscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """Subscribe a handler to specific event type."""
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler)
    
    def subscribe_all(self, handler: EventHandler) -> None:
        """Subscribe a handler to all events."""
        self.global_handlers.append(handler)
    
    def emit(self, event_type: EventType, data: Dict[str, Any] = None) -> None:
        """Emit an event."""
        if data is None:
            data = {}
        
        event = Event(event_type, data)
        
        # Dispatch to specific handlers
        if event_type in self.handlers:
            for handler in self.handlers[event_type]:
                try:
                    handler.handle(event)
                except Exception as e:
                    console.print(f"Error in event handler: {e}", style="red")
        
        # Dispatch to global handlers
        for handler in self.global_handlers:
            try:
                handler.handle(event)
            except Exception as e:
                console.print(f"Error in global event handler: {e}", style="red")


# ============================================================================
# EVENT-DRIVEN PROGRESS TRACKING
# ============================================================================

class EmojiCVSpinner(EventHandler):
    """Event-driven CV spinner with emoji aesthetics."""
    
    EMOJI_SETS = {
        'stars': ("⭐", "💫"),
        'targets': ("🎯", "🎪"), 
        'flowers': ("🌸", "🌺"),
        'gems': ("💎", "💍"),
        'fire': ("🔥", "✨"),
        'nature': ("🌿", "🍀"),
        'space': ("🚀", "🌙"),
        'magic': ("🪄", "⚡")
    }
    
    def __init__(self, total_folds: int = 3, style: str = 'stars'):
        """Initialize emoji CV spinner."""
        self.total_folds = total_folds
        self.filled_emoji, self.empty_emoji = self.EMOJI_SETS.get(style, self.EMOJI_SETS['stars'])
        self.current_fold = 0
        self.current_message = ""
        self.is_active = False
        self.spinner_thread: Optional[threading.Thread] = None
        self.stop_spinner = threading.Event()
    
    def handle(self, event: Event) -> None:
        """Handle CV-related events."""
        if event.type == EventType.CV_STARTED:
            self.start(event.data.get('message', 'CV Progress'))
        elif event.type == EventType.FOLD_STARTED:
            fold_idx = event.data.get('fold_idx', 0)
            message = event.data.get('message', f'Fold {fold_idx + 1}')
            self.update_fold(fold_idx, message)
        elif event.type == EventType.CV_COMPLETED:
            final_message = event.data.get('message', 'CV Completed')
            self.stop(final_message)
    
    def _create_progress_display(self) -> str:
        """Create emoji-based progress display."""
        filled = self.filled_emoji * self.current_fold
        empty = self.empty_emoji * (self.total_folds - self.current_fold)
        return f"{filled}{empty}"
    
    def _spinner_worker(self):
        """Worker thread for spinner animation."""
        animation_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        idx = 0
        
        while not self.stop_spinner.wait(0.1):
            if self.is_active:
                progress = self._create_progress_display()
                spinner_char = animation_chars[idx % len(animation_chars)]
                
                console.print(
                    f"\r  {spinner_char} CV: {progress} ({self.current_fold}/{self.total_folds}) - {self.current_message}",
                    end="", highlight=False
                )
                idx += 1
    
    def start(self, message: str):
        """Start the CV spinner."""
        self.current_message = message
        self.is_active = True
        self.stop_spinner.clear()
        self.spinner_thread = threading.Thread(target=self._spinner_worker)
        self.spinner_thread.daemon = True
        self.spinner_thread.start()
    
    def update_fold(self, fold_idx: int, message: str = ""):
        """Update current fold progress."""
        self.current_fold = fold_idx + 1  # +1 for display (1-based)
        if message:
            self.current_message = message
    
    def stop(self, final_message: str):
        """Stop the spinner."""
        if self.is_active:
            self.is_active = False
            self.stop_spinner.set()
            if self.spinner_thread:
                self.spinner_thread.join(timeout=0.5)
            
            progress = self._create_progress_display()
            console.print(f"\r  ✓ CV: {progress} ({self.total_folds}/{self.total_folds}) - {final_message}")


class IterationProgressTracker(EventHandler):
    """Event-driven iteration progress tracker."""
    
    def __init__(self):
        self.current_iteration = 0
        self.current_message = ""
        self.is_active = False
        self.spinner_thread: Optional[threading.Thread] = None
        self.stop_spinner = threading.Event()
    
    def handle(self, event: Event) -> None:
        """Handle iteration-related events."""
        if event.type == EventType.ITERATION_STARTED:
            iteration = event.data.get('iteration', 0)
            message = event.data.get('message', f'Iteration {iteration}')
            self.start_iteration(iteration, message)
        elif event.type == EventType.ITERATION_COMPLETED:
            message = event.data.get('message', 'Iteration completed')
            self.complete_iteration(message)
        elif event.type == EventType.FEATURE_SELECTION_COMPLETED:
            message = event.data.get('message', 'Feature selection completed')
            self.stop(message)
    
    def _spinner_worker(self):
        """Worker thread for iteration spinner."""
        spinner_chars = ["🔄", "🔃", "↻", "↺"]
        idx = 0
        
        while not self.stop_spinner.wait(0.2):
            if self.is_active:
                spinner_char = spinner_chars[idx % len(spinner_chars)]
                console.print(
                    f"\r    {spinner_char} {self.current_message}",
                    end="", highlight=False
                )
                idx += 1
    
    def start_iteration(self, iteration: int, message: str):
        """Start a new iteration."""
        self.current_iteration = iteration
        self.current_message = message
        self.is_active = True
        self.stop_spinner.clear()
        self.spinner_thread = threading.Thread(target=self._spinner_worker)
        self.spinner_thread.daemon = True
        self.spinner_thread.start()
    
    def update_message(self, message: str):
        """Update the current message."""
        self.current_message = message
    
    def complete_iteration(self, message: str):
        """Complete current iteration."""
        if self.is_active:
            self.is_active = False
            console.print(f"\r    ✓ {message}")
    
    def stop(self, final_message: str):
        """Stop the tracker."""
        if self.is_active:
            self.is_active = False
            self.stop_spinner.set()
            if self.spinner_thread:
                self.spinner_thread.join(timeout=0.5)
            console.print(f"\r    🏁 {final_message}")


class ProgressTableHandler(EventHandler):
    """Event handler for updating progress tables."""
    
    def __init__(self):
        self.progress_data = {}
        self.table = None
        self.live_display = None
    
    def handle(self, event: Event) -> None:
        """Handle events for table updates."""
        # For now, we'll just collect data
        # In a full implementation, this could update a live table
        pass


# ============================================================================
# EVENT-DRIVEN FEATURE SELECTOR
# ============================================================================

class EventDrivenBackwardFeatureSelector:
    """
    Event-driven backward feature selector with CV inside.
    """
    
    def __init__(
        self,
        event_bus: EventBus,
        cv_folds: int = 3,
        removal_ratio: float = 0.1,
        use_tuning: bool = True,
        min_features: int = 5,
        patience: int = 3,
        random_state: int = 42
    ):
        """Initialize event-driven feature selector."""
        self.event_bus = event_bus
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
    
    def _create_cv_splits(self, X: pd.DataFrame, y: pd.Series, problem_type: str):
        """Create cross-validation splits."""
        from sklearn.model_selection import StratifiedKFold, KFold
        
        if problem_type in ['binary_classification', 'multiclass_classification']:
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            return list(cv.split(X, y))
        else:
            cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            return list(cv.split(X))
    
    def _select_features_to_remove(self, current_features: List[str]) -> List[str]:
        """Select features to remove (random selection for simplicity)."""
        n_to_remove = max(1, int(len(current_features) * self.removal_ratio))
        n_to_remove = min(n_to_remove, len(current_features) - self.min_features)
        
        if n_to_remove <= 0:
            return []
        
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
    ) -> 'EventDrivenBackwardFeatureSelector':
        """Fit the feature selector with event-driven progress."""
        
        # Emit feature selection started event
        self.event_bus.emit(EventType.FEATURE_SELECTION_STARTED, {
            'n_features': len(X.columns),
            'model_type': model_type,
            'message': f"Starting backward selection with {len(X.columns)} features"
        })
        
        # Initialize
        current_features = list(X.columns)
        best_score = float('-inf') if metric_name not in ['rmse', 'mae'] else float('inf')
        best_features = current_features.copy()
        best_hyperparams = {}
        iterations_without_improvement = 0
        
        self.selection_history_ = []
        iteration = 0
        
        while len(current_features) > self.min_features and iterations_without_improvement < self.patience:
            iteration += 1
            
            # Emit iteration started event
            self.event_bus.emit(EventType.ITERATION_STARTED, {
                'iteration': iteration,
                'n_features': len(current_features),
                'patience': f"{iterations_without_improvement}/{self.patience}",
                'message': f"Iteration {iteration}: {len(current_features)} features, {iterations_without_improvement}/{self.patience} patience"
            })
            
            # Hyperparameter tuning (if enabled)
            if self.use_tuning:
                self.event_bus.emit(EventType.HYPERPARAMETER_TUNING_STARTED, {
                    'iteration': iteration,
                    'message': 'Tuning hyperparameters'
                })
                
                current_hyperparams = hyperparameter_tune(
                    X[current_features], y, model_type, problem_type, metric_name,
                    n_splits=self.cv_folds, show_progress=False
                )
                
                self.event_bus.emit(EventType.HYPERPARAMETER_TUNING_COMPLETED, {
                    'iteration': iteration,
                    'hyperparams': current_hyperparams,
                    'message': f'Best params: {current_hyperparams}'
                })
            else:
                current_hyperparams = {}
            
            # Start CV evaluation
            self.event_bus.emit(EventType.CV_STARTED, {
                'iteration': iteration,
                'n_features': len(current_features),
                'message': f"CV evaluation with {len(current_features)} features"
            })
            
            # Evaluate current feature set with CV
            cv_splits = self._create_cv_splits(X[current_features], y, problem_type)
            fold_scores = []
            
            for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
                # Emit fold started event
                self.event_bus.emit(EventType.FOLD_STARTED, {
                    'iteration': iteration,
                    'fold_idx': fold_idx,
                    'message': f"Fold {fold_idx + 1}/{self.cv_folds}"
                })
                
                X_train, X_val = X[current_features].iloc[train_idx], X[current_features].iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Evaluate this fold
                score, _ = evaluate_fold(
                    X_train, y_train, X_val, y_val,
                    model_type, problem_type, metric_name,
                    current_hyperparams, show_progress=False
                )
                fold_scores.append(score)
                
                # Emit fold completed event
                self.event_bus.emit(EventType.FOLD_COMPLETED, {
                    'iteration': iteration,
                    'fold_idx': fold_idx,
                    'score': score,
                    'message': f"Fold {fold_idx + 1}: {score:.4f}"
                })
            
            # Calculate CV score
            mean_score = np.mean(fold_scores)
            std_score = np.std(fold_scores)
            
            # Emit CV completed event
            self.event_bus.emit(EventType.CV_COMPLETED, {
                'iteration': iteration,
                'mean_score': mean_score,
                'std_score': std_score,
                'fold_scores': fold_scores,
                'message': f"CV Score: {mean_score:.4f} ± {std_score:.4f}"
            })
            
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
                
                message = f"NEW BEST! Score: {mean_score:.4f}, Features: {len(current_features)}"
            else:
                iterations_without_improvement += 1
                message = f"Score: {mean_score:.4f}, No improvement ({iterations_without_improvement}/{self.patience})"
            
            # Emit iteration completed event
            self.event_bus.emit(EventType.ITERATION_COMPLETED, {
                'iteration': iteration,
                'is_best': is_better,
                'mean_score': mean_score,
                'n_features': len(current_features),
                'patience': f"{iterations_without_improvement}/{self.patience}",
                'message': message
            })
            
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
        
        # Store final results
        self.best_features_ = best_features
        self.best_score_ = best_score
        self.best_hyperparams_ = best_hyperparams
        
        # Emit feature selection completed event
        self.event_bus.emit(EventType.FEATURE_SELECTION_COMPLETED, {
            'best_score': best_score,
            'n_best_features': len(best_features),
            'total_iterations': iteration,
            'message': f"Feature selection completed: {len(best_features)} features, score: {best_score:.4f}"
        })
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform X to selected features."""
        if self.best_features_ is None:
            raise ValueError("Selector not fitted. Call fit() first.")
        return X[self.best_features_]
    
    def get_cv_score(self) -> Tuple[float, float]:
        """Get the CV score of the best feature set."""
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


# ============================================================================
# MAIN BENCHMARK CLASS
# ============================================================================

class MDMBenchmarkV8:
    """Event-driven benchmark with aesthetic spinners and observability."""
    
    def __init__(self, output_dir: str = "benchmark_results", use_cache: bool = True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.use_cache = use_cache
        
        # Initialize event system
        self.event_bus = EventBus()
        self._setup_event_handlers()
        
        # Get MDM components (already initialized via DI)
        self.config_manager = get_config_manager()
        self.dataset_manager = DatasetManager()
        self.dataset_registrar = DatasetRegistrar()
        
        self.results = {
            'version': 'Version 8: Event-Driven Feature Selection with CV inside',
            'description': 'Event-driven architecture with observer pattern and aesthetic emoji spinners',
            'benchmark_date': datetime.now().isoformat(),
            'mdm_version': mdm.__version__,
            'results': {},
            'summary': {}
        }
    
    def _setup_event_handlers(self):
        """Setup event handlers for progress tracking."""
        # Create and register spinner handlers
        self.cv_spinner = EmojiCVSpinner(total_folds=3, style='stars')
        self.iteration_tracker = IterationProgressTracker()
        self.progress_table = ProgressTableHandler()
        
        # Subscribe handlers to events
        self.event_bus.subscribe(EventType.CV_STARTED, self.cv_spinner)
        self.event_bus.subscribe(EventType.FOLD_STARTED, self.cv_spinner)
        self.event_bus.subscribe(EventType.CV_COMPLETED, self.cv_spinner)
        
        self.event_bus.subscribe(EventType.ITERATION_STARTED, self.iteration_tracker)
        self.event_bus.subscribe(EventType.ITERATION_COMPLETED, self.iteration_tracker)
        self.event_bus.subscribe(EventType.FEATURE_SELECTION_COMPLETED, self.iteration_tracker)
        
        self.event_bus.subscribe_all(self.progress_table)
    
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
    
    def benchmark_competition(self, name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Benchmark a single competition using event-driven approach."""
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
            
            # With features - event-driven feature selection with CV inside
            console.print("  Event-driven feature selection with CV inside...")
            try:
                # Remove target from features for feature selection
                X_features = df_features.drop(columns=[config['target']])
                y_features = df_features[config['target']]
                
                # Use event-driven feature selector
                selector = EventDrivenBackwardFeatureSelector(
                    event_bus=self.event_bus,
                    cv_folds=3,
                    removal_ratio=0.1,
                    use_tuning=True
                )
                
                selector.fit(X_features, y_features, model_type, config['problem_type'], config['metric'])
                mean_with, std_with = selector.get_cv_score()
                
                results['with_features'][model_type] = {
                    'mean_score': round(mean_with, 4),
                    'std': round(std_with, 4),
                    'n_features': n_features_with,
                    'n_selected': len(selector.best_features_) if selector.best_features_ else n_features_with,
                    'best_features': selector.best_features_[:20] if selector.best_features_ else [],
                    'best_hyperparams': selector.best_hyperparams_,
                    'method': 'Event-driven backward selection with 3-fold CV inside'
                }
                console.print(f"    ✓ Score: {mean_with:.4f} ± {std_with:.4f}")
                console.print(f"    → Selected features: {len(selector.best_features_) if selector.best_features_ else 0}")
                
            except Exception as e:
                console.print(f"    ✗ Failed: {str(e)}", style="red")
                results['with_features'][model_type] = {'error': str(e)}
            
            # Without features - simple CV
            console.print("  Training without features (baseline CV)...")
            try:
                X_raw = df_raw.drop(columns=[config['target']])
                y_raw = df_raw[config['target']]
                
                # Use regular CV for baseline
                mean_without, std_without, fold_scores, trained_models = cross_validate_model(
                    X_raw,
                    y_raw,
                    model_type,
                    config['problem_type'],
                    config['metric'],
                    n_splits=3,
                    hyperparams=None,
                    show_progress=True
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
            f"[bold]Version 8: Event-Driven Feature Selection with CV inside[/bold]\n"
            f"Event-driven architecture with observer pattern and emoji spinners ⭐💫\n"
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
        output_file = self.output_dir / f"v8_results_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        console.print(f"\n[green]Results saved to: {output_file}[/green]")
    
    def display_summary(self):
        """Display summary table."""
        table = Table(title="Benchmark Summary - Version 8 (Event-Driven)", show_header=True)
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
        description="Version 8: Event-Driven Feature Selection with CV inside"
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
    
    benchmark = MDMBenchmarkV8(
        output_dir=args.output_dir,
        use_cache=not args.no_cache
    )
    
    benchmark.run_benchmark(competitions=args.competitions)


if __name__ == '__main__':
    main()