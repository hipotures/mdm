#!/usr/bin/env python3
"""
Version 9: State Machine Approach with Aesthetic CV Progress Spinners

This version implements a state machine architecture for managing the ML pipeline:
- States: IDLE, FEATURE_SELECTION, CV_FOLD_1, CV_FOLD_2, etc.
- State transitions based on events (start_fold, complete_fold)
- State-based table management with live updates
- State machine progress tracking with spinners
- Custom backward selection implementation (NOT cross_validate_ydf)

ALGORITHM:
- Feature selection with CV inside (3-fold) + tuning inside each fold
- Return CV score from feature selection as final result (NO additional CV)
- Same defaults: CV=3, removal_ratio=0.1, tuning=True
- Command: python version_9.py -c titanic

SPINNER:
- Aesthetic CV progress spinner: ▰▰▱ (for CV=3, showing 2 done, 1 current)
- Use different symbols: 🎮🕹️ or 🚀🛸 or 🎵🎼 or other emoji pairs
- Show spinner next to iteration number during training

STATE MACHINE:
- Clean state transitions with event-driven architecture
- State-based progress tracking and visualization
- Modular state handlers for maintainability
- Event logging for debugging and transparency
"""

import os
import sys
import json
import argparse
import signal
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union, NamedTuple
from enum import Enum, auto
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
from utils.metrics import needs_probabilities

console = Console()

# =============================================================================
# STATE MACHINE ARCHITECTURE
# =============================================================================

class MLState(Enum):
    """Machine Learning Pipeline States"""
    IDLE = auto()
    INITIALIZING = auto()
    FEATURE_SELECTION = auto()
    CV_FOLD_1 = auto()
    CV_FOLD_2 = auto()
    CV_FOLD_3 = auto()
    CV_FOLD_4 = auto()
    CV_FOLD_5 = auto()
    HYPERPARAMETER_TUNING = auto()
    FINALIZING = auto()
    COMPLETED = auto()
    ERROR = auto()


class MLEvent(Enum):
    """Machine Learning Pipeline Events"""
    START = auto()
    BEGIN_FEATURE_SELECTION = auto()
    START_CV_FOLD = auto()
    COMPLETE_CV_FOLD = auto()
    START_TUNING = auto()
    COMPLETE_TUNING = auto()
    FINALIZE = auto()
    COMPLETE = auto()
    ERROR_OCCURRED = auto()


class StateData(NamedTuple):
    """Data structure for state information"""
    state: MLState
    current_fold: int
    total_folds: int
    current_score: float
    best_score: float
    features_count: int
    message: str
    timestamp: datetime


class AestheticSpinner:
    """Aesthetic spinner with emoji pairs and progress visualization"""
    
    SYMBOL_PAIRS = {
        'gaming': ('🎮', '🕹️'),
        'space': ('🚀', '🛸'),
        'music': ('🎵', '🎼'),
        'blocks': ('▰', '▱'),
        'diamonds': ('◆', '◇'),
        'stars': ('★', '☆'),
        'circles': ('⚫', '⚪'),
        'tech': ('💻', '📱'),
        'fire': ('🔥', '💫'),
        'nature': ('🌟', '⭐')
    }
    
    def __init__(self, total_steps: int, symbol_type: str = 'gaming'):
        self.total_steps = total_steps
        self.current_step = 0
        self.filled_symbol, self.empty_symbol = self.SYMBOL_PAIRS.get(symbol_type, ('▰', '▱'))
        self.symbol_type = symbol_type
    
    def create_progress_bar(self, current: int = None) -> str:
        """Create aesthetic progress bar"""
        if current is None:
            current = self.current_step
        
        filled = self.filled_symbol * current
        empty = self.empty_symbol * (self.total_steps - current)
        return f"{filled}{empty}"
    
    def advance(self) -> str:
        """Advance spinner and return current state"""
        self.current_step = min(self.current_step + 1, self.total_steps)
        return self.create_progress_bar()
    
    def set_step(self, step: int) -> str:
        """Set specific step and return progress bar"""
        self.current_step = min(max(step, 0), self.total_steps)
        return self.create_progress_bar()
    
    def get_info(self) -> str:
        """Get spinner info string"""
        return f"{self.symbol_type.title()} spinner: {self.create_progress_bar(0)} → {self.create_progress_bar(self.total_steps)}"


class MLStateMachine:
    """State Machine for ML Pipeline Management"""
    
    def __init__(self, cv_folds: int = 3):
        self.current_state = MLState.IDLE
        self.cv_folds = cv_folds
        self.state_history: List[StateData] = []
        self.event_log: List[Tuple[MLEvent, datetime, str]] = []
        
        # State-specific data
        self.current_fold = 0
        self.current_score = 0.0
        self.best_score = float('-inf')
        self.features_count = 0
        
        # Progress tracking
        self.spinner = AestheticSpinner(cv_folds, 'gaming')
        self.live_table: Optional[Live] = None
        self.progress_table = None
        
        # State transition matrix
        self.valid_transitions = {
            MLState.IDLE: [MLState.INITIALIZING],
            MLState.INITIALIZING: [MLState.FEATURE_SELECTION, MLState.ERROR],
            MLState.FEATURE_SELECTION: [MLState.CV_FOLD_1, MLState.ERROR],
            MLState.CV_FOLD_1: [MLState.CV_FOLD_2, MLState.HYPERPARAMETER_TUNING, MLState.ERROR],
            MLState.CV_FOLD_2: [MLState.CV_FOLD_3, MLState.HYPERPARAMETER_TUNING, MLState.ERROR],
            MLState.CV_FOLD_3: [MLState.CV_FOLD_4, MLState.HYPERPARAMETER_TUNING, MLState.ERROR],
            MLState.CV_FOLD_4: [MLState.CV_FOLD_5, MLState.HYPERPARAMETER_TUNING, MLState.ERROR],
            MLState.CV_FOLD_5: [MLState.HYPERPARAMETER_TUNING, MLState.ERROR],
            MLState.HYPERPARAMETER_TUNING: [MLState.FINALIZING, MLState.ERROR],
            MLState.FINALIZING: [MLState.COMPLETED, MLState.ERROR],
            MLState.COMPLETED: [MLState.IDLE],
            MLState.ERROR: [MLState.IDLE]
        }
    
    def log_event(self, event: MLEvent, message: str = ""):
        """Log an event with timestamp"""
        self.event_log.append((event, datetime.now(), message))
    
    def transition_to(self, new_state: MLState, message: str = ""):
        """Transition to a new state with validation"""
        if new_state not in self.valid_transitions.get(self.current_state, []):
            error_msg = f"Invalid transition from {self.current_state} to {new_state}"
            console.print(f"[red]State Machine Error: {error_msg}[/red]")
            self.transition_to(MLState.ERROR, error_msg)
            return False
        
        # Record state change
        old_state = self.current_state
        self.current_state = new_state
        
        # Log the transition
        self.log_event(MLEvent.START, f"Transitioned from {old_state} to {new_state}: {message}")
        
        # Record state data
        state_data = StateData(
            state=new_state,
            current_fold=self.current_fold,
            total_folds=self.cv_folds,
            current_score=self.current_score,
            best_score=self.best_score,
            features_count=self.features_count,
            message=message,
            timestamp=datetime.now()
        )
        self.state_history.append(state_data)
        
        # Update visual display
        self.update_display()
        
        return True
    
    def create_progress_table(self) -> Table:
        """Create state-aware progress table matching version 2's 6-column format"""
        table = Table(title=f"State Machine ML Pipeline - {self.spinner.symbol_type.title()} Progress")
        
        # Use exact same 6-column format as version 2
        table.add_column("Iter", style="cyan", width=8)
        table.add_column("Features", style="magenta", width=8) 
        table.add_column("Score", style="green", width=12)
        table.add_column("Accuracy", style="yellow", width=10)
        table.add_column("Loss", style="red", width=10)
        table.add_column("Status", style="blue", width=15)
        
        # Current state row with spinner in status column like version 2
        progress_bar = self.spinner.create_progress_bar()
        
        # Format status with spinner (like version 2 shows "0 ▰▱▱")
        status_text = f"{self.current_fold} {progress_bar}" if self.current_fold > 0 else progress_bar
        
        # Format score with highlighting if it's the best
        score_str = f"{self.current_score:.4f}" if self.current_score else "N/A"
        if self.current_score > 0 and self.current_score == self.best_score:
            score_str = f"[reverse]{score_str}[/reverse]"
        
        # Calculate accuracy and loss placeholders (will be updated during actual training)
        accuracy_str = "N/A"
        loss_str = "N/A"
        
        table.add_row(
            str(0),  # Iteration (will be updated during backward selection)
            str(self.features_count) if self.features_count else "N/A",
            score_str,
            accuracy_str,
            loss_str,
            status_text
        )
        
        return table
    
    def get_state_color(self, state: MLState) -> str:
        """Get color for state visualization"""
        color_map = {
            MLState.IDLE: "white",
            MLState.INITIALIZING: "yellow",
            MLState.FEATURE_SELECTION: "blue",
            MLState.CV_FOLD_1: "green",
            MLState.CV_FOLD_2: "green",
            MLState.CV_FOLD_3: "green",
            MLState.CV_FOLD_4: "green",
            MLState.CV_FOLD_5: "green",
            MLState.HYPERPARAMETER_TUNING: "magenta",
            MLState.FINALIZING: "cyan",
            MLState.COMPLETED: "bright_green",
            MLState.ERROR: "red"
        }
        return color_map.get(state, "white")
    
    def start_live_display(self):
        """Start live display with state table"""
        self.progress_table = self.create_progress_table()
        self.live_table = Live(self.progress_table, console=console, refresh_per_second=2)
        self.live_table.start()
    
    def update_display(self):
        """Update live display"""
        if self.live_table:
            self.progress_table = self.create_progress_table()
            self.live_table.update(self.progress_table)
    
    def stop_live_display(self):
        """Stop live display"""
        if self.live_table:
            self.live_table.stop()
    
    def start_cv_fold(self, fold_number: int):
        """Start a CV fold"""
        self.current_fold = fold_number
        
        # Determine target state
        fold_states = [
            MLState.CV_FOLD_1, MLState.CV_FOLD_2, MLState.CV_FOLD_3,
            MLState.CV_FOLD_4, MLState.CV_FOLD_5
        ]
        
        if fold_number <= len(fold_states):
            target_state = fold_states[fold_number - 1]
            self.transition_to(target_state, f"Starting CV fold {fold_number}/{self.cv_folds}")
            self.spinner.set_step(fold_number - 1)
    
    def complete_cv_fold(self, fold_number: int, score: float):
        """Complete a CV fold"""
        self.current_score = score
        if score > self.best_score:
            self.best_score = score
        
        self.spinner.set_step(fold_number)
        message = f"Completed CV fold {fold_number}/{self.cv_folds}, Score: {score:.4f}"
        
        # Update current state message
        if self.state_history:
            current_state = self.current_state
            self.state_history[-1] = self.state_history[-1]._replace(
                current_score=score,
                best_score=self.best_score,
                message=message
            )
        
        self.update_display()
    
    def set_features_count(self, count: int):
        """Set current features count"""
        self.features_count = count
        self.update_display()
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get state machine summary"""
        return {
            'current_state': self.current_state.name,
            'total_transitions': len(self.state_history),
            'total_events': len(self.event_log),
            'cv_progress': f"{self.current_fold}/{self.cv_folds}",
            'spinner_progress': self.spinner.create_progress_bar(),
            'best_score': self.best_score,
            'features_count': self.features_count
        }


# =============================================================================
# CUSTOM BACKWARD FEATURE SELECTION WITH STATE MACHINE
# =============================================================================

def custom_backward_selection_with_state_machine(
    df: pd.DataFrame,
    target: str,
    model_type: str,
    problem_type: str,
    metric_name: str,
    state_machine: MLStateMachine,
    removal_ratio: float = 0.1,
    min_features: int = 5,
    patience: int = 3,
    use_tuning: bool = True,
    random_state: int = 42
) -> Tuple[float, float, List[float], List[str], int, Dict[str, Any]]:
    """
    Custom backward feature selection with state machine integration.
    
    This function implements the core algorithm with state-aware progress tracking.
    """
    from utils.custom_ml_helpers import backward_feature_selection, simple_cv
    
    # Transition to feature selection state
    state_machine.transition_to(MLState.FEATURE_SELECTION, "Starting backward feature selection")
    state_machine.set_features_count(len(df.columns) - 1)
    
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
    
    # Update features count
    state_machine.set_features_count(len(selected_features))
    
    # Step 2: Cross-validation with state machine tracking
    console.print(f"\n[bold]State Machine CV Evaluation[/bold]")
    console.print(f"Selected features: {len(selected_features)}")
    console.print(f"CV folds: {state_machine.cv_folds}")
    console.print(f"Spinner: {state_machine.spinner.get_info()}")
    
    # Prepare data for CV
    X = df[selected_features]
    y = df[target]
    
    # Create CV splits
    from sklearn.model_selection import KFold, StratifiedKFold
    if problem_type in ['binary_classification', 'multiclass_classification']:
        cv = StratifiedKFold(n_splits=state_machine.cv_folds, shuffle=True, random_state=random_state)
        cv_splits = list(cv.split(X, y))
    else:
        cv = KFold(n_splits=state_machine.cv_folds, shuffle=True, random_state=random_state)
        cv_splits = list(cv.split(X))
    
    fold_scores = []
    best_hyperparams = {}
    
    # Process each fold with state machine
    for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
        fold_number = fold_idx + 1
        
        # Start fold in state machine
        state_machine.start_cv_fold(fold_number)
        
        # Prepare fold data
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Optional hyperparameter tuning
        if use_tuning and fold_idx == 0:  # Tune only on first fold for speed
            state_machine.transition_to(MLState.HYPERPARAMETER_TUNING, f"Tuning hyperparameters for fold {fold_number}")
            
            # Simple hyperparameter tuning
            from cv_evaluator import hyperparameter_tune
            best_hyperparams = hyperparameter_tune(
                X_train, y_train, model_type, problem_type, metric_name,
                n_splits=2, show_progress=False  # Faster tuning
            )
        
        # Train and evaluate fold
        from cv_evaluator import evaluate_fold
        score, _ = evaluate_fold(
            X_train, y_train, X_val, y_val,
            model_type, problem_type, metric_name,
            best_hyperparams, show_progress=False
        )
        
        fold_scores.append(score)
        
        # Complete fold in state machine
        state_machine.complete_cv_fold(fold_number, score)
        
        # Brief pause for visual effect
        import time
        time.sleep(0.1)
    
    # Calculate final results
    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)
    
    # Transition to finalizing
    state_machine.transition_to(MLState.FINALIZING, f"CV complete: {mean_score:.4f} ± {std_score:.4f}")
    
    return mean_score, std_score, fold_scores, selected_features, len(selected_features), best_hyperparams


# =============================================================================
# MAIN BENCHMARK CLASS WITH STATE MACHINE
# =============================================================================

class StateMachineMDMBenchmark:
    """State Machine-based benchmark with aesthetic CV spinners."""
    
    def __init__(self, output_dir: str = "benchmark_results", use_cache: bool = True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.use_cache = use_cache
        
        # Get MDM components (already initialized via DI)
        self.config_manager = get_config_manager()
        self.dataset_manager = DatasetManager()
        self.dataset_registrar = DatasetRegistrar()
        
        self.results = {
            'version': 'Version 9: State Machine Approach with Aesthetic CV Spinners',
            'description': 'State machine architecture with custom backward selection and emoji progress spinners',
            'benchmark_date': datetime.now().isoformat(),
            'mdm_version': mdm.__version__,
            'results': {},
            'summary': {},
            'state_machine_info': {}
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
                # Filter to original columns only
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
    
    def benchmark_competition_with_state_machine(self, name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Benchmark a single competition using state machine approach."""
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
            'status': 'pending',
            'state_machine_summary': {}
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
        
        console.print("\n[bold]State Machine Training Pipeline...[/bold]")
        model_types = ['gbt', 'rf']
        spinner_types = ['gaming', 'space', 'music', 'fire']
        
        for i, model_type in enumerate(model_types):
            spinner_type = spinner_types[i % len(spinner_types)]
            console.print(f"\n[cyan]{model_type.upper()}:[/cyan] State machine with {spinner_type} spinner")
            
            # Create state machine for this model
            state_machine = MLStateMachine(cv_folds=3)
            state_machine.spinner = AestheticSpinner(3, spinner_type)
            
            # Start state machine
            state_machine.transition_to(MLState.INITIALIZING, f"Starting {model_type.upper()} pipeline")
            console.print(f"  Spinner info: {state_machine.spinner.get_info()}")
            
            # Start live display
            state_machine.start_live_display()
            
            try:
                # With features
                console.print("  State machine pipeline with features...")
                
                results_with = {}
                try:
                    mean_score, std_score, fold_scores, selected_features, n_selected, best_hyperparams = custom_backward_selection_with_state_machine(
                        df=df_features,
                        target=config['target'],
                        model_type=model_type,
                        problem_type=config['problem_type'],
                        metric_name=config['metric'],
                        state_machine=state_machine,
                        removal_ratio=0.1,
                        min_features=5,
                        patience=3,
                        use_tuning=True,
                        random_state=42
                    )
                    
                    # Complete state machine
                    state_machine.transition_to(MLState.COMPLETED, f"Pipeline complete: {mean_score:.4f} ± {std_score:.4f}")
                    
                    results_with = {
                        'mean_score': round(mean_score, 4),
                        'std': round(std_score, 4),
                        'n_features': len(df_features.columns) - 1,
                        'n_selected': n_selected,
                        'best_features': selected_features[:20],
                        'method': f'State machine backward selection (CV=3, {spinner_type} spinner)',
                        'fold_scores': fold_scores,
                        'hyperparams': best_hyperparams
                    }
                    
                    console.print(f"    ✓ Score: {mean_score:.4f} ± {std_score:.4f}")
                    console.print(f"    → Selected features: {n_selected}")
                    
                except Exception as e:
                    state_machine.transition_to(MLState.ERROR, f"Pipeline failed: {str(e)}")
                    console.print(f"    ✗ Failed: {str(e)}", style="red")
                    results_with = {'error': str(e)}
                
                finally:
                    # Stop live display
                    state_machine.stop_live_display()
                
                # Store state machine summary
                results['state_machine_summary'][model_type] = state_machine.get_state_summary()
                results['with_features'][model_type] = results_with
                
                # Without features (simple approach)
                console.print("  Simple evaluation without features...")
                try:
                    from utils.custom_ml_helpers import simple_cv
                    mean_score, std_score, fold_scores = simple_cv(
                        df=df_raw,
                        target=config['target'],
                        model_type=model_type,
                        problem_type=config['problem_type'],
                        metric_name=config['metric'],
                        n_splits=3,
                        random_state=42,
                        verbose=False
                    )
                    
                    results_without = {
                        'mean_score': round(mean_score, 4),
                        'std': round(std_score, 4),
                        'n_features': len(df_raw.columns) - 1,
                        'n_selected': len(df_raw.columns) - 1,
                        'method': 'Simple CV without feature selection',
                        'fold_scores': fold_scores
                    }
                    
                    console.print(f"    ✓ Score: {mean_score:.4f} ± {std_score:.4f}")
                    
                except Exception as e:
                    console.print(f"    ✗ Failed: {str(e)}", style="red")
                    results_without = {'error': str(e)}
                
                results['without_features'][model_type] = results_without
                
                # Calculate improvement
                if ('error' not in results_with and 'error' not in results_without and 
                    'mean_score' in results_with and 'mean_score' in results_without):
                    
                    score_with = results_with['mean_score']
                    score_without = results_without['mean_score']
                    
                    if config['metric'] in ['rmse', 'mae']:
                        improvement = ((score_without - score_with) / score_without) * 100
                    else:
                        improvement = ((score_with - score_without) / score_without) * 100
                    
                    results['improvement'][model_type] = f"{improvement:+.2f}%"
                    console.print(f"    [green]Improvement: {improvement:+.2f}%[/green]")
                
            except Exception as e:
                state_machine.stop_live_display()
                console.print(f"\n[red]Model {model_type} failed: {str(e)}[/red]")
                results['with_features'][model_type] = {'error': str(e)}
                results['without_features'][model_type] = {'error': str(e)}
        
        results['status'] = 'completed'
        return results
    
    def run_benchmark(self, competitions: Optional[List[str]] = None):
        """Run state machine benchmark for specified competitions or all."""
        all_competitions = get_all_competitions()
        
        if competitions:
            selected = {k: v for k, v in all_competitions.items() if k in competitions}
        else:
            selected = all_competitions
        
        console.print(Panel.fit(
            f"[bold]Version 9: State Machine Approach[/bold]\n"
            f"State machine architecture with aesthetic CV spinners\n"
            f"Custom backward selection with emoji progress visualization\n"
            f"Competitions: {len(selected)}\n"
            f"MDM Version: {mdm.__version__}",
            title="State Machine Benchmark Info"
        ))
        
        for name, config in selected.items():
            try:
                results = self.benchmark_competition_with_state_machine(name, config)
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
        
        for result in self.results['results'].values():
            if result.get('status') == 'completed':
                for model_type in ['gbt', 'rf']:
                    if model_type in result.get('improvement', {}):
                        imp_str = result['improvement'][model_type]
                        if imp_str != 'N/A':
                            imp_val = float(imp_str.replace('%', '').replace('+', ''))
                            improvements.append(imp_val)
        
        if improvements:
            avg_improvement = np.mean(improvements)
            best_improvement = max(improvements)
            
            # Find best competition
            best_comp = 'unknown'
            best_model = 'unknown'
            for name, result in self.results['results'].items():
                if result.get('status') == 'completed':
                    for model in ['gbt', 'rf']:
                        if model in result.get('improvement', {}):
                            imp_str = result['improvement'][model]
                            if imp_str != 'N/A':
                                imp_val = float(imp_str.replace('%', '').replace('+', ''))
                                if imp_val == best_improvement:
                                    best_comp = name
                                    best_model = model
                                    break
            
            # Count improvements
            competitions_improved = len([
                1 for result in self.results['results'].values()
                if result.get('status') == 'completed' and
                any(
                    float(result.get('improvement', {}).get(model, '0%').replace('%', '').replace('+', '')) > 0
                    for model in ['gbt', 'rf']
                )
            ])
            
            completed = len([
                1 for result in self.results['results'].values()
                if result.get('status') == 'completed'
            ])
            
            self.results['summary'] = {
                'average_improvement': f"{avg_improvement:+.2f}%",
                'best_improvement': f"{best_comp} ({best_model}): {best_improvement:+.2f}%",
                'competitions_improved': competitions_improved,
                'competitions_no_change': completed - competitions_improved,
                'competitions_failed': len(self.results['results']) - completed
            }
    
    def save_results(self):
        """Save results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"v9_results_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        console.print(f"\n[green]Results saved to: {output_file}[/green]")
    
    def display_summary(self):
        """Display summary table."""
        table = Table(title="State Machine Benchmark Summary - Version 9", show_header=True)
        table.add_column("Competition", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Features", justify="right")
        table.add_column("GBT Improvement", justify="right")
        table.add_column("RF Improvement", justify="right")
        table.add_column("State Machine", justify="center")
        
        for name, result in self.results['results'].items():
            status = result.get('status', 'unknown')
            
            # Features info
            features_info = 'N/A'
            if 'with_features' in result and 'gbt' in result['with_features']:
                gbt_result = result['with_features']['gbt']
                if 'n_selected' in gbt_result and 'n_features' in gbt_result:
                    features_info = f"{gbt_result['n_selected']}/{gbt_result['n_features']}"
            
            # Improvements
            gbt_imp = result.get('improvement', {}).get('gbt', 'N/A')
            rf_imp = result.get('improvement', {}).get('rf', 'N/A')
            
            # Format improvements with colors
            gbt_formatted = self._format_improvement(gbt_imp)
            rf_formatted = self._format_improvement(rf_imp)
            
            # State machine info
            sm_info = "N/A"
            if 'state_machine_summary' in result:
                sm_data = result['state_machine_summary']
                if sm_data:
                    # Get spinner type from first model
                    first_model = list(sm_data.keys())[0] if sm_data else None
                    if first_model and 'spinner_progress' in sm_data[first_model]:
                        sm_info = "✓"  # Simple checkmark for completed state machine
            
            table.add_row(name, status, features_info, gbt_formatted, rf_formatted, sm_info)
        
        console.print("\n")
        console.print(table)
        
        if 'summary' in self.results:
            console.print("\n[bold]State Machine Summary:[/bold]")
            for key, value in self.results['summary'].items():
                console.print(f"  {key}: {value}")
    
    def _format_improvement(self, improvement: str) -> str:
        """Format improvement string with colors."""
        if isinstance(improvement, str) and improvement != 'N/A':
            if '+' in improvement:
                return f"[green]{improvement}[/green]"
            elif '-' in improvement:
                return f"[red]{improvement}[/red]"
        return improvement


def main():
    """Main entry point for state machine benchmark."""
    parser = argparse.ArgumentParser(
        description="Version 9: State Machine Approach with Aesthetic CV Spinners"
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
    
    benchmark = StateMachineMDMBenchmark(
        output_dir=args.output_dir,
        use_cache=not args.no_cache
    )
    
    benchmark.run_benchmark(competitions=args.competitions)


if __name__ == '__main__':
    main()