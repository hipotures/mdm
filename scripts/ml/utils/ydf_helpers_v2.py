"""YDF helper functions with live feature selection monitoring."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import ydf
import subprocess
import sys
import os
import tempfile
import json
import re
import time
from datetime import datetime
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.live import Live
from sklearn.model_selection import KFold, StratifiedKFold
from .metrics import calculate_metric, needs_probabilities
from .ydf_helpers import cross_validate_ydf, tune_hyperparameters, SAVE_YDF_LOGS, YDF_LOG_DIR

console = Console()


def select_features_then_cv(
    X: pd.DataFrame,
    y: Union[pd.Series, pd.DataFrame],
    learner_type: str = 'RANDOM_FOREST',
    learner_kwargs: Optional[Dict[str, Any]] = None,
    metric: str = 'accuracy',
    cv_folds: int = 5,
    random_state: int = 42,
    feature_selection_kwargs: Optional[Dict[str, Any]] = None,
    show_table: bool = True,
    verbose: int = 1
) -> Tuple[List[str], Dict[str, Any], List[int]]:
    """
    Select features using validation set, then evaluate with CV.
    
    Args:
        X: Feature DataFrame
        y: Target variable(s)
        learner_type: Type of YDF learner
        learner_kwargs: Additional learner parameters
        metric: Evaluation metric
        cv_folds: Number of CV folds
        random_state: Random state
        feature_selection_kwargs: Feature selection parameters
        show_table: Whether to show live monitoring table
        verbose: Verbosity level
        
    Returns:
        Tuple of (selected_features, cv_results, selected_indices)
    """
    if learner_kwargs is None:
        learner_kwargs = {}
    if feature_selection_kwargs is None:
        feature_selection_kwargs = {}
    
    # Set up stratified split for validation
    if len(y.shape) == 1 or y.shape[1] == 1:
        # Single target
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=random_state, stratify=y
        )
    else:
        # Multi-label: stratify by first label
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=random_state, stratify=y.iloc[:, 0]
        )
    
    # Combine for YDF
    train_df = pd.concat([X_train, y_train], axis=1)
    val_df = pd.concat([X_val, y_val], axis=1)
    
    # Set up YDF datasets
    if isinstance(y, pd.DataFrame):
        label = list(y.columns)
    else:
        label = y.name if hasattr(y, 'name') and y.name else 'target'
    
    # Configure learner based on type
    learner_class = getattr(ydf, f'{learner_type.title().replace("_", "")}Learner')
    
    # Configure feature selection
    fs_params = feature_selection_kwargs.copy()
    removal_ratio = fs_params.pop('removal_ratio', 0.2)
    maximize_objective = fs_params.pop('maximize_objective', 
                                      metric not in ['rmse', 'mae', 'mse', 'log_loss'])
    
    if show_table:
        # Use subprocess approach for live monitoring
        return _select_features_with_monitoring(
            train_df=train_df,
            val_df=val_df,
            label=label,
            learner_type=learner_type,
            learner_kwargs=learner_kwargs,
            removal_ratio=removal_ratio,
            maximize_objective=maximize_objective,
            fs_params=fs_params,
            X=X,
            y=y,
            metric=metric,
            cv_folds=cv_folds,
            random_state=random_state,
            verbose=verbose
        )
    else:
        # Standard approach without monitoring
        fs = ydf.learner.BackwardSelectionFeatureSelector(
            removal_ratio=removal_ratio,
            objective_metric=f"@{metric}",
            maximize_objective=maximize_objective,
            **fs_params
        )
        
        # Create learner with feature selection
        learner = learner_class(
            label=label,
            feature_selector=fs,
            **learner_kwargs
        )
        
        # Train with feature selection
        model = learner.train(train_df, valid_data=val_df, verbose=verbose)
        
        # Get selected features
        selected_features = list(model.features())
        selected_indices = [X.columns.get_loc(f) for f in selected_features]
        
        # Run CV on selected features
        X_selected = X[selected_features]
        cv_results = cross_validate_ydf(
            X_selected, y, 
            learner_type=learner_type,
            learner_kwargs=learner_kwargs,
            metric=metric,
            cv_folds=cv_folds,
            random_state=random_state,
            verbose=0
        )
        
        return selected_features, cv_results, selected_indices


def _select_features_with_monitoring(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    label: Union[str, List[str]],
    learner_type: str,
    learner_kwargs: Dict[str, Any],
    removal_ratio: float,
    maximize_objective: bool,
    fs_params: Dict[str, Any],
    X: pd.DataFrame,
    y: Union[pd.Series, pd.DataFrame],
    metric: str,
    cv_folds: int,
    random_state: int,
    verbose: int
) -> Tuple[List[str], Dict[str, Any], List[int]]:
    """Run feature selection with live monitoring using subprocess."""
    
    # Set up log file
    if SAVE_YDF_LOGS:
        os.makedirs(YDF_LOG_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(YDF_LOG_DIR, f"ydf_{timestamp}.log")
    else:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as tmp:
            log_path = tmp.name
    
    # Save data and config
    train_path = log_path.replace('.log', '_train.pkl')
    val_path = log_path.replace('.log', '_val.pkl')
    config_path = log_path.replace('.log', '_config.json')
    
    train_df.to_pickle(train_path)
    val_df.to_pickle(val_path)
    
    config = {
        'label': label,
        'learner_type': learner_type,
        'learner_kwargs': learner_kwargs,
        'removal_ratio': removal_ratio,
        'maximize_objective': maximize_objective,
        'fs_params': fs_params,
        'metric': metric,
        'log_path': log_path
    }
    with open(config_path, 'w') as f:
        json.dump(config, f)
    
    # Create training script
    script_path = log_path.replace('.log', '_train.py')
    script_content = f'''
import pandas as pd
import json
import ydf
import os
import sys

# Load config
with open('{config_path}', 'r') as f:
    config = json.load(f)

# Load data
train_df = pd.read_pickle('{train_path}')
val_df = pd.read_pickle('{val_path}')

# Redirect output to log file
log_fd = os.open('{log_path}', os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
os.dup2(log_fd, sys.stdout.fileno())
os.dup2(log_fd, sys.stderr.fileno())

# Configure feature selector
fs = ydf.learner.BackwardSelectionFeatureSelector(
    removal_ratio=config['removal_ratio'],
    objective_metric=f"@{{config['metric']}}",
    maximize_objective=config['maximize_objective'],
    **config['fs_params']
)

# Create learner
learner_class = getattr(ydf, f"{{config['learner_type'].title().replace('_', '')}}Learner")
learner = learner_class(
    label=config['label'],
    feature_selector=fs,
    **config['learner_kwargs']
)

# Train with feature selection
model = learner.train(train_df, valid_data=val_df, verbose=2)

# Save selected features
selected_features = list(model.features())
result_path = '{log_path}'.replace('.log', '_result.json')
with open(result_path, 'w') as f:
    json.dump({{'selected_features': selected_features}}, f)

# Close log file
os.close(log_fd)
'''
    
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Start training process
    process = subprocess.Popen(
        [sys.executable, script_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    
    # Monitor log file with Rich Live
    class LogMonitor:
        def __init__(self):
            self.iterations = {}
            self.best_score = -float('inf') if maximize_objective else float('inf')
            self.best_accuracy = -float('inf')
            self.best_loss = float('inf')
            self.pattern = re.compile(
                r'\[Iteration (\d+)\].*?Score:([\d.-]+).*?Metrics:{([^}]+)}'
            )
        
        def parse_metrics(self, metrics_str: str) -> Dict[str, float]:
            """Parse metrics string."""
            metrics = {}
            # Handle format: 'accuracy': 0.854, 'loss': 0.405
            for match in re.finditer(r"'(\w+)':\s*([\d.-]+)", metrics_str):
                key, value = match.groups()
                metrics[key] = float(value)
            return metrics
        
        def update_from_line(self, line: str):
            """Update iterations from log line."""
            match = self.pattern.search(line)
            if match:
                iteration = int(match.group(1))
                score = float(match.group(2))
                metrics = self.parse_metrics(match.group(3))
                
                # Update best values
                if maximize_objective:
                    if score > self.best_score:
                        self.best_score = score
                else:
                    if score < self.best_score:
                        self.best_score = score
                        
                if 'accuracy' in metrics and metrics['accuracy'] > self.best_accuracy:
                    self.best_accuracy = metrics['accuracy']
                if 'loss' in metrics and metrics['loss'] < self.best_loss:
                    self.best_loss = metrics['loss']
                
                self.iterations[iteration] = {
                    'score': score,
                    'metrics': metrics,
                    'status': 'Complete'
                }
        
        def create_table(self) -> Table:
            """Create Rich table with current state."""
            table = Table(title="Feature Selection Progress")
            table.add_column("Iteration", style="cyan")
            table.add_column("Features", style="green")
            table.add_column("Score", style="yellow")
            table.add_column("Accuracy", style="magenta")
            table.add_column("Loss", style="red")
            table.add_column("Status", style="blue")
            
            # Calculate features count for each iteration
            total_features = len(X.columns)
            
            for iteration in sorted(self.iterations.keys()):
                data = self.iterations[iteration]
                
                # Estimate features count
                features_removed = int(total_features * removal_ratio * iteration)
                features_count = max(1, total_features - features_removed)
                
                # Format values with highlighting
                score_str = f"{data['score']:.4f}"
                if maximize_objective and data['score'] == self.best_score:
                    score_str = f"[reverse]{score_str}[/reverse]"
                elif not maximize_objective and data['score'] == self.best_score:
                    score_str = f"[reverse]{score_str}[/reverse]"
                
                accuracy_str = ""
                if 'accuracy' in data['metrics']:
                    accuracy_str = f"{data['metrics']['accuracy']:.4f}"
                    if data['metrics']['accuracy'] == self.best_accuracy:
                        accuracy_str = f"[reverse]{accuracy_str}[/reverse]"
                
                loss_str = ""
                if 'loss' in data['metrics']:
                    loss_str = f"{data['metrics']['loss']:.4f}"
                    if data['metrics']['loss'] == self.best_loss:
                        loss_str = f"[reverse]{loss_str}[/reverse]"
                
                table.add_row(
                    str(iteration),
                    str(features_count),
                    score_str,
                    accuracy_str,
                    loss_str,
                    data['status']
                )
            
            # Add current iteration if process is running
            if process.poll() is None:
                current_iter = len(self.iterations)
                features_removed = int(total_features * removal_ratio * current_iter)
                features_count = max(1, total_features - features_removed)
                
                table.add_row(
                    str(current_iter),
                    str(features_count),
                    "...",
                    "...",
                    "...",
                    "Running..."
                )
            
            return table
    
    monitor = LogMonitor()
    
    # Monitor with Rich Live
    with Live(monitor.create_table(), refresh_per_second=2, console=console) as live:
        last_size = 0
        
        while process.poll() is None:
            # Read new log content
            if os.path.exists(log_path):
                with open(log_path, 'r') as f:
                    f.seek(last_size)
                    new_content = f.read()
                    if new_content:
                        for line in new_content.splitlines():
                            monitor.update_from_line(line)
                        last_size = f.tell()
                        live.update(monitor.create_table())
            
            time.sleep(0.1)
        
        # Final read after process completes
        time.sleep(0.5)
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                f.seek(last_size)
                new_content = f.read()
                if new_content:
                    for line in new_content.splitlines():
                        monitor.update_from_line(line)
                    
                    # Update last iteration status
                    if monitor.iterations:
                        last_iter = max(monitor.iterations.keys())
                        monitor.iterations[last_iter]['status'] = 'Done'
                    
                    live.update(monitor.create_table())
    
    # Check if process succeeded
    if process.returncode != 0:
        raise RuntimeError("Feature selection training failed")
    
    # Load results
    result_path = log_path.replace('.log', '_result.json')
    with open(result_path, 'r') as f:
        result = json.load(f)
    
    selected_features = result['selected_features']
    selected_indices = [X.columns.get_loc(f) for f in selected_features]
    
    # Clean up temporary files if not saving logs
    if not SAVE_YDF_LOGS:
        for path in [log_path, train_path, val_path, config_path, script_path, result_path]:
            if os.path.exists(path):
                os.remove(path)
    
    # Run CV on selected features
    if verbose > 0:
        console.print(f"\n[green]Selected {len(selected_features)} features[/green]")
        console.print("[cyan]Running cross-validation on selected features...[/cyan]")
    
    X_selected = X[selected_features]
    cv_results = cross_validate_ydf(
        X_selected, y, 
        learner_type=learner_type,
        learner_kwargs=learner_kwargs,
        metric=metric,
        cv_folds=cv_folds,
        random_state=random_state,
        verbose=0
    )
    
    return selected_features, cv_results, selected_indices