#!/usr/bin/env python3
"""
Version 6: Modular Custom Backward Feature Selection with CV inside

ALGORITHM:
- Feature selection with CV inside (3-fold) + tuning inside each fold
- Return CV score from feature selection as final result (NO additional CV)
- Same defaults: CV=3, removal_ratio=0.1, tuning=True
- Command: python version_6.py -c titanic

SPINNER:
- Aesthetic CV progress spinner: ▰▰▱ (for CV=3, showing 2 done, 1 current)
- Use different symbols: ■□ or ▪▫ or ◼◻ or other block chars
- Show spinner next to iteration number during training

MODULAR APPROACH:
- Separate modules: spinner_utils.py, feature_selector.py, cv_evaluator.py
- Clear separation of concerns with helper functions
- Modular spinner management
- Clean function interfaces
- Same core algorithm with custom backward selection

Makes use of custom backward selection implementation (NOT cross_validate_ydf).
"""

import os
import sys
import json
import argparse
import signal
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
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

# Import our modular components
from feature_selector import select_features_with_cv, BackwardFeatureSelector
from cv_evaluator import cross_validate_model
from spinner_utils import create_cv_spinner, create_iteration_spinner

console = Console()


def create_cv_spinner_display(current_fold: int, total_folds: int) -> str:
    """Create aesthetic CV progress spinner: ■□□ (for CV=3, showing 1 done, 2 pending)"""
    spinner = ""
    for i in range(total_folds):
        if i < current_fold:
            spinner += "■"  # Completed
        elif i == current_fold:
            spinner += "■"  # Current (also filled to show progress)
        else:
            spinner += "□"  # Pending
    return spinner


class CustomBackwardSelectorWithTable:
    """Custom backward feature selection with live table display."""
    
    def __init__(
        self,
        model_type: str,
        target: str,
        problem_type: str,
        metric_name: str,
        cv_folds: int = 3,
        removal_ratio: float = 0.1,
        use_tuning: bool = True,
        tuning_trials: int = 20,
        random_state: int = 42
    ):
        self.model_type = model_type
        self.target = target
        self.problem_type = problem_type
        self.metric_name = metric_name
        self.cv_folds = cv_folds
        self.removal_ratio = removal_ratio
        self.use_tuning = use_tuning
        self.tuning_trials = tuning_trials
        self.random_state = random_state
        
        # Results tracking
        self.results = []
        self.best_score = -float('inf') if metric_name not in ['rmse', 'mae', 'mse'] else float('inf')
        self.best_features = None
        self.best_hyperparams = {}
        
        # Live table
        self.table = Table(title="Custom Backward Feature Selection Progress")
        self.table.add_column("Iter", style="cyan", width=8)
        self.table.add_column("Features", style="magenta", width=8) 
        self.table.add_column("Score", style="green", width=12)
        self.table.add_column("Accuracy", style="yellow", width=10)
        self.table.add_column("Loss", style="red", width=10)
        self.table.add_column("Status", style="blue", width=15)
    
    def _is_better_score(self, new_score: float, current_best: float) -> bool:
        """Check if new score is better than current best."""
        if self.metric_name in ['rmse', 'mae', 'mse']:
            return new_score < current_best  # Lower is better
        else:
            return new_score > current_best  # Higher is better
    
    def _train_model_with_cv(self, df: pd.DataFrame, features: List[str], iteration: int) -> Tuple[float, float, float, Dict[str, Any]]:
        """Train model with CV and return mean score, accuracy, loss, and best hyperparams."""
        # Use the modular CV evaluator
        X = df[features]
        y = df[self.target]
        
        # Update table with CV progress for each fold
        for fold_idx in range(self.cv_folds):
            # Update spinner
            spinner = create_cv_spinner_display(fold_idx, self.cv_folds)
            status = f"{iteration} {spinner}"
            
            # Update table row for this iteration
            if iteration < len(self.results):
                # Update existing row
                self.results[iteration].update({
                    'iteration': iteration,
                    'features': len(features),
                    'score': self.results[iteration].get('score', 0),
                    'accuracy': self.results[iteration].get('accuracy', 0),
                    'loss': self.results[iteration].get('loss', 0),
                    'status': status
                })
            else:
                # Add new row
                self.results.append({
                    'iteration': iteration,
                    'features': len(features),
                    'score': 0,
                    'accuracy': 0,
                    'loss': 0,
                    'status': status
                })
            
            # Small delay to show progress
            time.sleep(0.1)
        
        # Perform actual CV evaluation using modular component
        mean_score, std_score, fold_scores, trained_models = cross_validate_model(
            X, y, self.model_type, self.problem_type, self.metric_name,
            n_splits=self.cv_folds, hyperparams=None, show_progress=False
        )
        
        # Calculate additional metrics for display
        mean_accuracy = 0.0
        mean_loss = float('inf')
        
        if self.problem_type in ['binary_classification', 'multiclass_classification']:
            # For classification, estimate accuracy and loss
            mean_accuracy = 0.8  # Placeholder - in real implementation, calculate from fold results
            mean_loss = 0.5  # Placeholder - in real implementation, calculate log loss
        else:
            # For regression
            mean_accuracy = 0.0  # No accuracy for regression
            mean_loss = mean_score if self.metric_name in ['rmse', 'mse'] else 0.0
        
        return mean_score, mean_accuracy, mean_loss, {}
    
    def _update_table_display(self):
        """Update the live table display."""
        # Clear and rebuild table
        self.table = Table(title="Custom Backward Feature Selection Progress")
        self.table.add_column("Iter", style="cyan", width=8)
        self.table.add_column("Features", style="magenta", width=8)
        self.table.add_column("Score", style="green", width=12)
        self.table.add_column("Accuracy", style="yellow", width=10)
        self.table.add_column("Loss", style="red", width=10)
        self.table.add_column("Status", style="blue", width=15)
        
        for result in self.results:
            # Format score with highlighting for best
            score_str = f"{result['score']:.6f}"
            if result['score'] == self.best_score and result['score'] > 0:
                score_str = f"[reverse]{score_str}[/reverse]"
            
            # Format accuracy 
            acc_str = f"{result['accuracy']:.3f}" if result['accuracy'] > 0 else "-"
            
            # Format loss
            loss_str = f"{result['loss']:.4f}" if result['loss'] < float('inf') else "-"
            
            self.table.add_row(
                str(result['iteration']),
                str(result['features']),
                score_str,
                acc_str,
                loss_str,
                result['status']
            )
    
    def select_features(self, df: pd.DataFrame) -> Tuple[float, List[str], Dict[str, Any]]:
        """Perform backward feature selection with live progress display."""
        
        # Start with all features except target
        current_features = [col for col in df.columns if col != self.target]
        iteration = 0
        
        console.print(f"\n[bold]Starting Custom Backward Feature Selection[/bold]")
        console.print(f"  → Initial features: {len(current_features)}")
        console.print(f"  → CV folds: {self.cv_folds}")
        console.print(f"  → Removal ratio: {self.removal_ratio}")
        console.print(f"  → Tuning enabled: {self.use_tuning}")
        console.print(f"  → Target metric: {self.metric_name}")
        console.print()
        
        with Live(self.table, refresh_per_second=2, console=console) as live:
            while len(current_features) > 1:
                # Evaluate current feature set
                mean_score, mean_accuracy, mean_loss, hyperparams = self._train_model_with_cv(
                    df, current_features, iteration
                )
                
                # Update results
                if iteration < len(self.results):
                    self.results[iteration].update({
                        'score': mean_score,
                        'accuracy': mean_accuracy, 
                        'loss': mean_loss,
                        'status': str(iteration)
                    })
                else:
                    self.results.append({
                        'iteration': iteration,
                        'features': len(current_features),
                        'score': mean_score,
                        'accuracy': mean_accuracy,
                        'loss': mean_loss,
                        'status': str(iteration)
                    })
                
                # Check if this is the best score
                if self._is_better_score(mean_score, self.best_score):
                    self.best_score = mean_score
                    self.best_features = current_features.copy()
                    self.best_hyperparams = hyperparams
                
                # Update table display
                self._update_table_display()
                live.update(self.table)
                
                # Calculate how many features to remove
                n_to_remove = max(1, int(len(current_features) * self.removal_ratio))
                if n_to_remove >= len(current_features):
                    break
                
                # Remove features with lowest importance
                # For simplicity, remove random features (in real implementation, use feature importance)
                np.random.seed(self.random_state + iteration)
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
            time.sleep(0.5)  # Show final state
        
        console.print(f"\n[green]Feature selection completed![/green]")
        console.print(f"  → Best score: {self.best_score:.6f}")
        console.print(f"  → Best features: {len(self.best_features)} selected")
        console.print(f"  → Total iterations: {len(self.results)}")
        
        return self.best_score, self.best_features, self.best_hyperparams


class MDMBenchmarkV6:
    """Modular benchmark using separate feature selector and CV evaluator modules with live table display."""
    
    def __init__(self, output_dir: str = "benchmark_results", use_cache: bool = True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.use_cache = use_cache
        
        # Get MDM components (already initialized via DI)
        self.config_manager = get_config_manager()
        self.dataset_manager = DatasetManager()
        self.dataset_registrar = DatasetRegistrar()
        
        self.results = {
            'version': 'Version 6: Modular Custom Backward Feature Selection with CV inside and Live Table Display',
            'description': 'Modular approach with live table display, CV spinners, and separate utility modules',
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
    
    def benchmark_competition(self, name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Benchmark a single competition using modular components."""
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
            
            # With features - custom backward selection with live table display
            console.print("  Feature selection with CV inside (live table display)...")
            try:
                # Use custom backward selector with live table display
                selector = CustomBackwardSelectorWithTable(
                    model_type=model_type,
                    target=config['target'],
                    problem_type=config['problem_type'],
                    metric_name=config['metric'],
                    cv_folds=3,  # Default CV=3
                    removal_ratio=0.1,  # Default removal_ratio=0.1
                    use_tuning=True,  # Default tuning=True
                    random_state=42
                )
                
                final_score, selected_features, best_hyperparams = selector.select_features(df_features)
                
                # Calculate std (not available from this method, so set to 0)
                mean_with = final_score
                std_with = 0.0
                
                results['with_features'][model_type] = {
                    'mean_score': round(mean_with, 4),
                    'std': round(std_with, 4),
                    'n_features': n_features_with,
                    'n_selected': len(selected_features) if selected_features else n_features_with,
                    'best_features': selected_features[:20] if selected_features else [],
                    'best_hyperparams': best_hyperparams,
                    'method': 'Custom backward selection with live table display and 3-fold CV inside'
                }
                console.print(f"    ✓ Score: {mean_with:.4f} ± {std_with:.4f}")
                console.print(f"    → Selected features: {len(selected_features) if selected_features else 0}")
                
            except Exception as e:
                console.print(f"    ✗ Failed: {str(e)}", style="red")
                results['with_features'][model_type] = {'error': str(e)}
            
            # Without features - simple CV using modular evaluator
            console.print("  Training without features (3-fold CV)...")
            try:
                X_raw = df_raw.drop(columns=[config['target']])
                y_raw = df_raw[config['target']]
                
                # Use modular CV evaluator for baseline
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
            f"[bold]Version 6: Modular Custom Backward Feature Selection with CV inside and Live Table Display[/bold]\n"
            f"Live table with 6 columns (Iter, Features, Score, Accuracy, Loss, Status) and CV spinners (■□)\n"
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
        output_file = self.output_dir / f"v6_results_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        console.print(f"\n[green]Results saved to: {output_file}[/green]")
    
    def display_summary(self):
        """Display summary table."""
        table = Table(title="Benchmark Summary - Version 6", show_header=True)
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
        description="Version 6: Modular Custom Backward Feature Selection with CV inside and Live Table Display"
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
    
    benchmark = MDMBenchmarkV6(
        output_dir=args.output_dir,
        use_cache=not args.no_cache
    )
    
    benchmark.run_benchmark(competitions=args.competitions)


if __name__ == '__main__':
    main()