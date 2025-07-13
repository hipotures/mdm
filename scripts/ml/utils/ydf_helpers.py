"""YDF (Yggdrasil Decision Forests) helper functions."""

import ydf
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.model_selection import KFold, StratifiedKFold
from rich.console import Console
from .metrics import calculate_metric, needs_probabilities

console = Console()

# Global flag to control log saving
SAVE_YDF_LOGS = True
YDF_LOG_DIR = "ydf_logs"

# Monitor type: 'status' (writes to file), 'web' (opens browser), or 'console' (Rich table)
YDF_MONITOR_TYPE = 'live'  # Live Rich table in same terminal

# Helper function for silent training
def _train_silently(learner, train_data, valid_data=None, verbose=0, show_table=False):
    """Train YDF model with output redirected to temporary file."""
    
    if show_table:
        # Use subprocess for training with live monitoring
        import subprocess
        import tempfile
        import os
        import sys
        from datetime import datetime
        import json
        import re
        from rich.table import Table
        from rich.live import Live
        import time
        
        if SAVE_YDF_LOGS:
            os.makedirs(YDF_LOG_DIR, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = os.path.join(YDF_LOG_DIR, f"ydf_{timestamp}.log")
        else:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as tmp:
                log_path = tmp.name
        
        # Save training data and learner config
        train_path = log_path.replace('.log', '_train.pkl')
        valid_path = log_path.replace('.log', '_valid.pkl') if valid_data is not None else None
        learner_path = log_path.replace('.log', '_learner.pkl')
        
        train_data.to_pickle(train_path)
        if valid_data is not None:
            valid_data.to_pickle(valid_path)
        
        # Get learner parameters - YDF stores them in _kwargs
        learner_type = type(learner).__name__
        learner_params = {}
        
        # Try to get parameters from learner's internal storage
        if hasattr(learner, '_label'):
            learner_params['label'] = learner._label
        if hasattr(learner, '_task'):
            learner_params['task'] = str(learner._task) if learner._task else None
        if hasattr(learner, '_num_trees'):
            learner_params['num_trees'] = learner._num_trees
            
        # If that doesn't work, get from kwargs if available
        if hasattr(learner, '_kwargs'):
            for key, value in learner._kwargs.items():
                if key == 'task' and value is not None:
                    learner_params[key] = str(value)
                elif key == 'feature_selector':
                    # Skip - handle separately
                    pass
                elif isinstance(value, (str, int, float, bool, type(None))):
                    learner_params[key] = value
        
        # Ensure we have label (it's required)
        if 'label' not in learner_params:
            # Try to guess from data
            learner_params['label'] = 'target'  # Default fallback
        
        # Handle feature selector separately
        fs = None
        if hasattr(learner, '_feature_selector'):
            fs = learner._feature_selector
        elif hasattr(learner, '_kwargs') and 'feature_selector' in learner._kwargs:
            fs = learner._kwargs['feature_selector']
            
        if fs is not None:
            # Get objective metric
            obj_metric = None
            if hasattr(fs, '_objective_metric'):
                obj_metric = fs._objective_metric
            elif hasattr(fs, 'objective_metric'):
                obj_metric = fs.objective_metric
                
            learner_params['feature_selector_config'] = {
                'removal_ratio': getattr(fs, '_removal_ratio', getattr(fs, 'removal_ratio', 0.2)),
                'objective_metric': obj_metric,
                'maximize_objective': getattr(fs, '_maximize_objective', getattr(fs, 'maximize_objective', True))
            }
        
        # Save as JSON-serializable config
        import json
        with open(learner_path, 'w') as f:
            json.dump({
                'learner_type': learner_type,
                'learner_params': learner_params,
                'verbose': verbose
            }, f)
        
        # Create training script
        script_content = f"""
import sys
import pandas as pd
import ydf

# Redirect output to log
with open('{log_path}', 'w', buffering=1) as log_file:
    sys.stdout = log_file
    sys.stderr = log_file
    
    try:
        # Load data
        train_data = pd.read_pickle('{train_path}')
        valid_data = {'pd.read_pickle("' + valid_path + '")' if valid_path else 'None'}
        
        # Recreate learner
        import json
        with open('{learner_path}', 'r') as f:
            config = json.load(f)
        
        # Recreate feature selector if needed
        feature_selector = None
        if 'feature_selector_config' in config['learner_params']:
            fs_config = config['learner_params']['feature_selector_config']
            feature_selector = ydf.BackwardSelectionFeatureSelector(
                removal_ratio=fs_config['removal_ratio'],
                objective_metric=fs_config['objective_metric'],
                maximize_objective=fs_config['maximize_objective']
            )
        
        # Create learner with all parameters
        params = config['learner_params'].copy()
        params.pop('feature_selector_config', None)  # Remove this, we handle it separately
        
        # Convert task string back to enum
        if 'task' in params and params['task']:
            if 'CLASSIFICATION' in params['task']:
                params['task'] = ydf.Task.CLASSIFICATION
            elif 'REGRESSION' in params['task']:
                params['task'] = ydf.Task.REGRESSION
        
        # Add feature selector
        if feature_selector:
            params['feature_selector'] = feature_selector
        
        # Create learner
        if config['learner_type'] == 'GradientBoostedTreesLearner':
            learner = ydf.GradientBoostedTreesLearner(**params)
        else:
            learner = ydf.RandomForestLearner(**params)
        
        # Train
        if valid_data is not None:
            model = learner.train(train_data, valid=valid_data)
        else:
            model = learner.train(train_data)
        
        # Save model
        model.save('{log_path.replace('.log', '_model')}')
        
        print("Training completed successfully")
        
    except Exception as e:
        print(f"Error in training: {{e}}")
        import traceback
        traceback.print_exc()
"""
        
        script_path = log_path.replace('.log', '_train.py')
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Start training subprocess
        process = subprocess.Popen(
            [sys.executable, script_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        # Monitor log file with Rich Live
        class LogMonitor:
            def __init__(self):
                self.iterations = {}
                self.last_pos = 0
                # Track best values
                self.best_score = -float('inf')
                self.best_accuracy = -float('inf')
                self.best_loss = float('inf')  # Lower is better for loss
                
            def parse_lines(self, lines):
                for line in lines:
                    if match := re.search(r'Run backward feature selection on (\d+) features', line):
                        self.iterations[0] = {'features': int(match.group(1)), 'status': 'Starting'}
                    elif match := re.search(r'\[Iteration (\d+)\] Train model on (\d+) features', line):
                        self.iterations[int(match.group(1))] = {'features': int(match.group(2)), 'status': 'Training'}
                    elif match := re.search(r'\[Iteration (\d+)\] Score:([0-9.]+)', line):
                        iter_num = int(match.group(1))
                        if iter_num not in self.iterations:
                            self.iterations[iter_num] = {}
                        self.iterations[iter_num]['score'] = float(match.group(2))
                        self.iterations[iter_num]['status'] = 'Done'
                        if acc_match := re.search(r"'accuracy': ([0-9.]+)", line):
                            self.iterations[iter_num]['accuracy'] = float(acc_match.group(1))
                        if loss_match := re.search(r"'loss': ([0-9.]+)", line):
                            self.iterations[iter_num]['loss'] = float(loss_match.group(1))
                        
                        # Update best values
                        if self.iterations[iter_num]['score'] > self.best_score:
                            self.best_score = self.iterations[iter_num]['score']
                        if 'accuracy' in self.iterations[iter_num] and self.iterations[iter_num]['accuracy'] > self.best_accuracy:
                            self.best_accuracy = self.iterations[iter_num]['accuracy']
                        if 'loss' in self.iterations[iter_num] and self.iterations[iter_num]['loss'] < self.best_loss:
                            self.best_loss = self.iterations[iter_num]['loss']
            
            def create_table(self):
                table = Table(title="YDF Feature Selection Progress")
                table.add_column("Iter", style="cyan")
                table.add_column("Features", style="magenta")
                table.add_column("Score", style="green")
                table.add_column("Accuracy", style="yellow")
                table.add_column("Loss", style="red")
                table.add_column("Status", style="blue")
                
                for i in sorted(self.iterations.keys()):
                    data = self.iterations[i]
                    
                    # Format values and check if they are best
                    score_str = f"{data.get('score', 0):.6f}" if 'score' in data else '-'
                    acc_str = f"{data.get('accuracy', 0):.3f}" if 'accuracy' in data else '-'
                    loss_str = f"{data.get('loss', 0):.4f}" if 'loss' in data else '-'
                    
                    # Highlight best values
                    if 'score' in data and data['score'] == self.best_score:
                        score_str = f"[reverse]{score_str}[/reverse]"
                    if 'accuracy' in data and data['accuracy'] == self.best_accuracy:
                        acc_str = f"[reverse]{acc_str}[/reverse]"
                    if 'loss' in data and data['loss'] == self.best_loss:
                        loss_str = f"[reverse]{loss_str}[/reverse]"
                    
                    table.add_row(
                        str(i),
                        str(data.get('features', '-')),
                        score_str,
                        acc_str,
                        loss_str,
                        data.get('status', '-')
                    )
                return table
        
        monitor = LogMonitor()
        
        # Wait for log file
        while not os.path.exists(log_path):
            time.sleep(0.01)
        
        with Live(monitor.create_table(), refresh_per_second=2, console=console) as live:
            while process.poll() is None:
                try:
                    with open(log_path, 'r') as f:
                        f.seek(monitor.last_pos)
                        new_lines = f.readlines()
                        if new_lines:
                            monitor.last_pos = f.tell()
                            monitor.parse_lines(new_lines)
                            live.update(monitor.create_table())
                except:
                    pass
                time.sleep(0.5)
            
            # Process finished - do one final read to catch any remaining output
            time.sleep(0.5)  # Give process time to flush buffers
            try:
                with open(log_path, 'r') as f:
                    f.seek(monitor.last_pos)
                    new_lines = f.readlines()
                    if new_lines:
                        monitor.parse_lines(new_lines)
                        live.update(monitor.create_table())
                        time.sleep(0.5)  # Show final update
            except:
                pass
        
        # Load model
        model = ydf.load_model(log_path.replace('.log', '_model'))
        
        # Don't show log path here - it's too verbose
        
        # Cleanup
        try:
            os.unlink(train_path)
            if valid_path:
                os.unlink(valid_path)
            os.unlink(learner_path)
            os.unlink(script_path)
            if not SAVE_YDF_LOGS:
                os.unlink(log_path)
        except:
            pass
        
        return model
    
    # Regular silent training
    from .ydf_file_logger import train_with_file_logging
    import tempfile
    import os
    
    # For regular training, always use temp file (no need to save these logs)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as tmp:
        log_path = tmp.name
    
    try:
        model, _ = train_with_file_logging(
            learner, train_data, valid_data=valid_data, 
            log_path=log_path, silent=True, verbose=verbose
        )
        # Don't show log path for non-feature-selection training
        return model
    finally:
        # Always delete temp logs for regular training
        if os.path.exists(log_path):
            os.unlink(log_path)

try:
    import ydf
    HAS_YDF = True
except ImportError:
    HAS_YDF = False


def create_learner(
    model_type: str,
    label: str,
    task: Optional[Any] = None,
    feature_selector: Optional[Any] = None,
    **kwargs
):
    """
    Create a YDF learner with default parameters.
    
    Args:
        model_type: 'gbt' for Gradient Boosted Trees or 'rf' for Random Forest
        label: Target column name
        task: YDF task (CLASSIFICATION, REGRESSION, etc.)
        feature_selector: Optional feature selector object
        **kwargs: Additional learner parameters
    
    Returns:
        YDF learner instance
    """
    default_params = {
        'label': label,
        'task': task
    }
    
    # Add feature selector if provided
    if feature_selector is not None:
        default_params['feature_selector'] = feature_selector
    
    if model_type == 'gbt':
        # Default GBT parameters
        gbt_defaults = {
            'num_trees': 100,
            'max_depth': 6,
            'shrinkage': 0.1,
            'subsample': 0.8,
            'min_examples': 5,
            'use_hessian_gain': True,
            'growing_strategy': 'BEST_FIRST_GLOBAL'
        }
        gbt_defaults.update(kwargs)
        default_params.update(gbt_defaults)
        return ydf.GradientBoostedTreesLearner(**default_params)
    
    elif model_type == 'rf':
        # Default Random Forest parameters
        rf_defaults = {
            'num_trees': 100,
            'max_depth': 16,
            'min_examples': 5,
            'bootstrap_training_dataset': True,
            'compute_oob_variable_importances': True  # Required for feature selection
        }
        # Only set num_candidate_attributes_ratio if num_candidate_attributes not specified
        if 'num_candidate_attributes' not in kwargs:
            rf_defaults['num_candidate_attributes_ratio'] = -1.0  # sqrt(num_features)
        rf_defaults.update(kwargs)
        default_params.update(rf_defaults)
        return ydf.RandomForestLearner(**default_params)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'gbt' or 'rf'")


def determine_task(problem_type: str):
    """Determine YDF task from problem type."""
    if 'classification' in problem_type:
        return ydf.Task.CLASSIFICATION
    elif problem_type == 'regression':
        return ydf.Task.REGRESSION
    else:
        raise ValueError(f"Unknown problem type: {problem_type}")


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
    
    # Train with feature selection - use our special function for monitoring
    model = _train_silently(learner, train_df, valid_data=val_df, verbose=verbose, show_table=show_table)
    
    # Get selected features
    selected_features = list(model.features())
    selected_indices = [X.columns.get_loc(f) for f in selected_features]
    
    # Run CV on selected features
    X_selected = X[selected_features]
    cv_results = cross_validate_ydf_simple(
        X_selected, y, 
        learner_type=learner_type,
        learner_kwargs=learner_kwargs,
        metric=metric,
        cv_folds=cv_folds,
        random_state=random_state,
        verbose=0
    )
    
    return selected_features, cv_results, selected_indices


def cross_validate_ydf(
    df: pd.DataFrame,
    target: str,
    model_type: str,
    problem_type: str,
    metric_name: str,
    n_splits: int = 5,
    random_state: int = 42,
    learner_params: Optional[Dict[str, Any]] = None,
    use_feature_selection: bool = False,
    feature_removal_ratio: float = 0.1,
    use_tuning: bool = False,
    tuning_trials: int = 20
) -> Tuple[float, float, List[float], Optional[List[str]], Optional[int]]:
    """
    Perform cross-validation with YDF model.
    
    Args:
        df: Input DataFrame
        target: Target column name(s)
        model_type: 'gbt' or 'rf'
        problem_type: Problem type
        metric_name: Metric to calculate
        n_splits: Number of CV folds
        random_state: Random seed
        learner_params: Additional learner parameters
        use_feature_selection: Whether to use backward feature selection
        feature_removal_ratio: Ratio of features to remove at each iteration (0.1 = 10%)
        use_tuning: Whether to tune hyperparameters after feature selection
        tuning_trials: Number of tuning trials for hyperparameter search
    
    Returns:
        Tuple of (mean_score, std_score, individual_scores, selected_features, avg_n_selected)
    """
    # Handle multi-label case
    if problem_type == 'multilabel_classification':
        return cross_validate_multilabel(
            df, target, model_type, metric_name, n_splits, random_state, learner_params
        )
    
    # Prepare data
    X = df.drop(columns=[target])
    y = df[target]
    
    # Choose appropriate cross-validation strategy
    if 'classification' in problem_type:
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = kf.split(X, y)
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = kf.split(X)
    
    scores = []
    task = determine_task(problem_type)
    selected_features_list = []
    n_selected_list = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        # Split data
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        
        # Create and train model
        learner_params = learner_params or {}
        
        # Add feature selection if requested
        if use_feature_selection:
            # For feature selection, create a validation split from the training data
            # Use 20% of training data for feature selection validation
            fs_train_size = int(0.8 * len(train_df))
            fs_indices = np.random.permutation(len(train_df))
            fs_train_df = train_df.iloc[fs_indices[:fs_train_size]]
            fs_valid_df = train_df.iloc[fs_indices[fs_train_size:]]
            
            # Configure feature selector with more verbose settings
            # Map our metrics to YDF metrics
            ydf_metric = None
            maximize = None
            if metric_name == 'accuracy':
                ydf_metric = 'accuracy'
                maximize = True
            elif metric_name == 'roc_auc':
                ydf_metric = 'characteristic_0:roc_auc'  # YDF's name for binary AUC
                maximize = True
            elif metric_name == 'rmse':
                ydf_metric = 'rmse'
                maximize = False
            elif metric_name == 'mae':
                ydf_metric = 'mae' 
                maximize = False
            
            feature_selector = ydf.BackwardSelectionFeatureSelector(
                removal_ratio=feature_removal_ratio,
                objective_metric=ydf_metric,
                maximize_objective=maximize,
                # Don't set allow_structural_variable_importance when using validation
            )
            
            # Force verbose output for feature selection
            console.print(f"    → BackwardSelectionFeatureSelector configured:")
            console.print(f"      removal_ratio={feature_removal_ratio}")
            console.print(f"      objective_metric={ydf_metric}")
            console.print(f"      maximize_objective={maximize}")
            
            # Create learner WITH feature selector as parameter (method that works)
            learner = create_learner(
                model_type, target, task, 
                feature_selector=feature_selector,
                **learner_params
            )
            
            # Feature selection will start - no need for verbose details
        
        # Train with verbose output if feature selection is enabled
        if use_feature_selection:
            # Use file logging for completely silent operation
            console.print(f"    → Running feature selection...")
            
            if model_type == 'rf':
                # For Random Forest, merge train and validation for feature selection
                console.print(f"    → Note: Random Forest uses internal OOB for feature selection")
                model = _train_silently(learner, train_df, verbose=1, show_table=True)
            else:
                model = _train_silently(learner, fs_train_df, valid_data=fs_valid_df, verbose=1, show_table=True)
            
            # Show what model reports after training
            console.print(f"    → Model training completed")
            console.print(f"      Number of input features: {len(model.input_features())}")
            
            # Check if model has feature selection info
            if hasattr(model, 'feature_selection_logs'):
                try:
                    logs = model.feature_selection_logs()
                    console.print(f"      Feature selection logs available")
                except:
                    console.print(f"      No feature selection logs")
            
            # If tuning is requested, get selected features and retrain with tuning
            if use_tuning:
                # Get the selected features
                selected_features = [f.name for f in model.input_features()]
                console.print(f"    → Tuning hyperparameters on {len(selected_features)} selected features...")
                
                # Create datasets with only selected features
                selected_cols = selected_features + [target]
                train_selected = train_df[selected_cols]
                
                # For tuning, we need to create a new validation split
                # Use 20% of training data for tuning validation
                tuning_train_size = int(0.8 * len(train_selected))
                tuning_indices = np.random.permutation(len(train_selected))
                tuning_train_df = train_selected.iloc[tuning_indices[:tuning_train_size]]
                tuning_valid_df = train_selected.iloc[tuning_indices[tuning_train_size:]]
                
                console.print(f"      Using {len(tuning_train_df)} samples for tuning train, {len(tuning_valid_df)} for tuning validation")
                
                # Create tuner
                tuner = ydf.RandomSearchTuner(
                    num_trials=tuning_trials,
                    automatic_search_space=True,
                    parallel_trials=1  # Keep it simple for now
                )
                
                # Create new learner with tuning
                tuned_learner = create_learner(model_type, target, task, **learner_params)
                tuned_learner.tuner = tuner
                
                # Train with tuning - use validation for tuning
                if model_type == 'rf':
                    # Random Forest doesn't support validation, will use OOB
                    model = _train_silently(tuned_learner, train_selected, verbose=0)
                else:
                    # GBT can use validation for tuning
                    model = _train_silently(tuned_learner, tuning_train_df, valid_data=tuning_valid_df, verbose=0)
                console.print(f"    → Tuning completed")
        else:
            # No feature selection, create learner normally
            learner = create_learner(model_type, target, task, **learner_params)
            
            # No feature selection, but maybe tuning
            if use_tuning:
                console.print(f"    → Tuning hyperparameters on all {len(df.columns)-1} features...")
                
                # Create validation split for tuning
                tuning_train_size = int(0.8 * len(train_df))
                tuning_indices = np.random.permutation(len(train_df))
                tuning_train_df = train_df.iloc[tuning_indices[:tuning_train_size]]
                tuning_valid_df = train_df.iloc[tuning_indices[tuning_train_size:]]
                
                console.print(f"      Using {len(tuning_train_df)} samples for tuning train, {len(tuning_valid_df)} for tuning validation")
                
                tuner = ydf.RandomSearchTuner(
                    num_trials=tuning_trials,
                    automatic_search_space=True,
                    parallel_trials=1
                )
                learner.tuner = tuner
                
                # Train with validation for tuning
                if model_type == 'rf':
                    # Random Forest uses OOB for tuning
                    model = _train_silently(learner, train_df, verbose=0)
                else:
                    # GBT uses validation for tuning
                    model = _train_silently(learner, tuning_train_df, valid_data=tuning_valid_df, verbose=0)
            else:
                # No tuning, just train silently
                model = _train_silently(learner, train_df, verbose=0)
        
        # Always try to get feature importances
        try:
            # Get feature importances
            importances = model.variable_importances()
            
            # For feature selection, show the selection process
            if use_feature_selection:
                # Check if model has feature_selection_logs method
                if hasattr(model, 'feature_selection_logs'):
                    try:
                        logs = model.feature_selection_logs()
                        if logs:
                            console.print(f"    → Feature selection completed")
                            # Try to access log details
                            if hasattr(logs, 'iterations'):
                                console.print(f"      Iterations: {len(logs.iterations)}")
                            if hasattr(logs, 'selected_features'):
                                console.print(f"      Final features: {len(logs.selected_features)} selected from {len(df.columns)-1}")
                    except Exception as e:
                        console.print(f"    → Could not get feature selection logs: {e}")
                
                # Count features actually used by the model
                input_features = model.input_features()
                original_features = len(df.columns) - 1
                selected_features = len(input_features)
                
                # Store the count for averaging
                n_selected_list.append(selected_features)
                
                if selected_features < original_features:
                    console.print(f"    → Feature selection: {selected_features} features selected (from {original_features})")
                    reduction_pct = (1 - selected_features/original_features) * 100
                    console.print(f"    → Reduction: {reduction_pct:.1f}%")
                else:
                    console.print(f"    → Feature selection: kept all {selected_features} features")
                    console.print(f"      (No improvement found by removing features)")
                
                # Show top important features from selected ones
                if isinstance(importances, dict) and 'NUM_AS_ROOT' in importances:
                    # The format is (value, feature_name) tuples
                    top_features = sorted(importances['NUM_AS_ROOT'], key=lambda x: x[0], reverse=True)[:5]
                    # Extract feature names from the second element of tuple
                    feature_names = []
                    for f in top_features:
                        if isinstance(f, tuple) and len(f) >= 2:
                            feature_names.append(str(f[1]))
                    if feature_names:
                        console.print(f"    → Top features: {', '.join(feature_names[:3])}...")
            
            # Store top features for summary
            if isinstance(importances, dict):
                if 'NUM_AS_ROOT' in importances and isinstance(importances['NUM_AS_ROOT'], list):
                    # The format is (value, feature_name) tuples
                    # Sort by importance value (first element of tuple)
                    sorted_features = sorted(importances['NUM_AS_ROOT'], key=lambda x: x[0], reverse=True)
                    # Extract feature names from second element
                    top_features = []
                    for feat_tuple in sorted_features[:20]:
                        if isinstance(feat_tuple, tuple) and len(feat_tuple) >= 2:
                            feat_name = feat_tuple[1]  # Feature name is second element
                            if isinstance(feat_name, str):
                                top_features.append(feat_name)
                    selected_features_list.append(top_features)
                else:
                    selected_features_list.append([])
            else:
                selected_features_list.append([])
            
        except Exception as e:
            # If we can't get importances, skip
            console.print(f"    → Warning: Could not get feature information: {e}")
            pass
        
        # Make predictions
        if needs_probabilities(metric_name):
            # Get probabilities
            predictions = model.predict(val_df)
            if problem_type == 'binary_classification':
                # For binary classification, get probability of positive class
                if hasattr(predictions, 'probability'):
                    # If predictions have probability attribute
                    pred_proba = predictions.probability(1)  # Probability of class 1
                else:
                    # If predictions are already probabilities
                    pred_proba = predictions
                y_pred = pred_proba
            else:
                # Multi-class - get all probabilities
                y_pred = predictions
        else:
            # Get class predictions
            predictions = model.predict(val_df)
            # Don't convert to int here - let calculate_metric handle the conversion
            # This allows proper threshold-based conversion for binary classification
            y_pred = predictions
        
        # Calculate metric
        y_true = val_df[target].values
        score = calculate_metric(y_true, y_pred, metric_name, problem_type)
        scores.append(score)
        
        # Log fold score
        console.print(f"    → Fold {fold_idx + 1} score: {score:.4f}")
    
    # Get consensus selected features across folds
    if selected_features_list:
        # Get features that appear in all folds
        from collections import Counter
        all_features = [f for features in selected_features_list for f in features]
        feature_counts = Counter(all_features)
        # Features that appear in at least half the folds
        consensus_features = [f for f, count in feature_counts.items() if count >= n_splits / 2]
    else:
        consensus_features = None
    
    # Calculate average number of selected features
    avg_n_selected = np.mean(n_selected_list) if n_selected_list else None
    
    return np.mean(scores), np.std(scores), scores, consensus_features, avg_n_selected


def cross_validate_ydf_simple(
    X: pd.DataFrame,
    y: Union[pd.Series, pd.DataFrame],
    learner_type: str = 'RANDOM_FOREST',
    learner_kwargs: Optional[Dict[str, Any]] = None,
    metric: str = 'accuracy',
    cv_folds: int = 5,
    random_state: int = 42,
    verbose: int = 0
) -> Dict[str, Any]:
    """
    Simple cross-validation with X and y separately.
    
    Returns:
        Dict with 'mean', 'std', and 'scores'
    """
    # Combine X and y
    if isinstance(y, pd.DataFrame):
        df = pd.concat([X, y], axis=1)
        target = list(y.columns)
    else:
        df = pd.concat([X, y], axis=1)
        target = y.name if hasattr(y, 'name') and y.name else 'target'
    
    # Determine problem type
    if isinstance(target, list):
        problem_type = 'multilabel_classification'
    elif y.dtype == 'object' or y.nunique() < 20:
        if y.nunique() == 2:
            problem_type = 'binary_classification'
        else:
            problem_type = 'multiclass_classification'
    else:
        problem_type = 'regression'
    
    # Map learner type
    model_type = 'rf' if 'FOREST' in learner_type.upper() else 'gbt'
    
    # Call the main function
    mean_score, std_score, scores, _, _ = cross_validate_ydf(
        df=df,
        target=target,
        model_type=model_type,
        problem_type=problem_type,
        metric_name=metric,
        n_splits=cv_folds,
        random_state=random_state,
        learner_params=learner_kwargs,
        use_feature_selection=False,
        use_tuning=False
    )
    
    return {
        'mean': mean_score,
        'std': std_score,
        'scores': scores
    }


def cross_validate_multilabel(
    df: pd.DataFrame,
    targets: List[str],
    model_type: str,
    metric_name: str,
    n_splits: int = 5,
    random_state: int = 42,
    learner_params: Optional[Dict[str, Any]] = None
) -> Tuple[float, float, List[float], Optional[List[str]]]:
    """
    Cross-validation for multi-label classification.
    
    Trains separate models for each label.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_scores = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(df)):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        
        # Predictions for all labels
        all_predictions = {}
        y_true_all = []
        
        # Train model for each label
        for label_idx, label in enumerate(targets):
            learner_params = learner_params or {}
            learner = create_learner(
                model_type, label, ydf.Task.CLASSIFICATION, **learner_params
            )
            
            model = _train_silently(learner, train_df)
            
            # Get predictions
            if needs_probabilities(metric_name):
                predictions = model.predict(val_df)
                if hasattr(predictions, 'probability'):
                    pred_proba = predictions.probability(1)
                else:
                    pred_proba = predictions
                all_predictions[label] = pred_proba
            else:
                all_predictions[label] = model.predict(val_df)
            
            y_true_all.append(val_df[label].values)
        
        # Convert to arrays
        y_true = np.column_stack(y_true_all)
        y_pred = np.column_stack([all_predictions[label] for label in targets])
        
        # Calculate metric
        score = calculate_metric(y_true, y_pred, metric_name, 'multilabel_classification')
        fold_scores.append(score)
    
    return np.mean(fold_scores), np.std(fold_scores), fold_scores, None, None


def tune_hyperparameters(
    df: pd.DataFrame,
    target: str,
    model_type: str,
    problem_type: str,
    metric_name: str,
    n_trials: int = 50,
    n_splits: int = 3,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Tune hyperparameters using random search.
    
    Returns:
        Best hyperparameters found
    """
    # Define search space
    if model_type == 'gbt':
        search_space = {
            'num_trees': [50, 100, 200, 300],
            'max_depth': [4, 6, 8, 10, 12],
            'shrinkage': [0.05, 0.1, 0.15, 0.2],
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'min_examples': [5, 10, 20]
        }
    else:  # Random Forest
        search_space = {
            'num_trees': [50, 100, 200, 300, 500],
            'max_depth': [8, 12, 16, 20, -1],  # -1 means no limit
            'min_examples': [5, 10, 20],
            'num_candidate_attributes_ratio': [0.5, 0.7, 1.0, -1.0]  # -1 means sqrt
        }
    
    best_score = -np.inf if metric_name != 'rmse' else np.inf
    best_params = {}
    
    # Random search
    for trial in range(n_trials):
        # Sample hyperparameters
        params = {}
        for param, values in search_space.items():
            params[param] = np.random.choice(values)
        
        # Evaluate with cross-validation
        try:
            mean_score, _, _ = cross_validate_ydf(
                df, target, model_type, problem_type, metric_name,
                n_splits=n_splits, random_state=random_state,
                learner_params=params
            )
            
            # Check if better
            is_better = (
                (metric_name in ['rmse', 'mae'] and mean_score < best_score) or
                (metric_name not in ['rmse', 'mae'] and mean_score > best_score)
            )
            
            if is_better:
                best_score = mean_score
                best_params = params.copy()
        
        except Exception as e:
            # Skip failed trials
            continue
    
    return best_params


def get_feature_importance(
    model: Any,
    top_k: Optional[int] = None
) -> pd.DataFrame:
    """
    Get feature importance from trained model.
    
    Args:
        model: Trained YDF model
        top_k: Return only top K features
    
    Returns:
        DataFrame with feature names and importance scores
    """
    importance = model.variable_importances()
    
    if top_k is not None:
        importance = importance.head(top_k)
    
    return importance


def select_top_features(
    df: pd.DataFrame,
    model: Any,
    target: str,
    top_k: int = 50
) -> pd.DataFrame:
    """
    Select top K features based on importance.
    
    Args:
        df: Input DataFrame
        model: Trained YDF model
        target: Target column name
        top_k: Number of top features to select
    
    Returns:
        DataFrame with selected features and target
    """
    importance = get_feature_importance(model, top_k)
    top_features = importance['feature'].tolist()
    
    # Ensure target is included
    if target not in top_features:
        top_features.append(target)
    
    # Handle case where we have fewer features than requested
    available_features = [f for f in top_features if f in df.columns]
    
    return df[available_features]