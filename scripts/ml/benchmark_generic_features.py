#!/usr/bin/env python3
"""
Benchmark MDM generic features using YDF (Yggdrasil Decision Forests).

This script compares model performance with and without MDM's automatic 
feature engineering across multiple Kaggle competitions.
"""

import os
import sys
import json
import argparse
import signal
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
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.panel import Panel
from rich import print as rprint

# Add parent directory to path for MDM imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

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

from utils.competition_configs import get_all_competitions, get_competition_config
from utils.ydf_helpers import cross_validate_ydf, tune_hyperparameters, select_features_then_cv
from utils.custom_ml_helpers import custom_feature_selection_cv
from utils.metrics import needs_probabilities

console = Console()


class MDMBenchmark:
    """Benchmark MDM generic features across competitions."""
    
    def __init__(self, output_dir: str = "benchmark_results", use_cache: bool = True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.use_cache = use_cache
        
        # Get MDM components (already initialized via DI)
        self.config_manager = get_config_manager()
        self.dataset_manager = DatasetManager()
        self.dataset_registrar = DatasetRegistrar()
        
        self.results = {
            'benchmark_date': datetime.now().isoformat(),
            'mdm_version': mdm.__version__,
            'results': {},
            'summary': {}
        }
    
    def register_competition(self, name: str, config: Dict[str, Any]) -> bool:
        """
        Register a competition dataset in MDM.
        
        Returns:
            True if successful, False otherwise
        """
        dataset_name = name
        # MDM converts dashes to underscores in dataset names
        mdm_name = name.replace('-', '_')
        
        # Check if already registered
        try:
            existing = self.dataset_manager.get_dataset(mdm_name)
            if existing and self.use_cache:
                console.print(f"  ✓ Using cached dataset: {dataset_name}")
                return True
        except DatasetError:
            pass  # Dataset doesn't exist, proceed with registration
        
        # Register dataset
        # Use full directory path for Kaggle datasets to get proper structure
        dataset_path = Path(config['path'])
        
        try:
            # Prepare registration parameters
            reg_params = {
                'name': dataset_name,
                'path': dataset_path,  # Use directory path for auto-detection
                'problem_type': config['problem_type'],
                'force': True  # Overwrite if exists
            }
            
            # Handle different target types
            if isinstance(config['target'], list):
                # Multi-label case
                reg_params['target'] = config['target'][0]  # Use first target for now
                # TODO: MDM doesn't support multi-label targets yet
            else:
                reg_params['target'] = config['target']
            
            console.print(f"  → Registering {dataset_name}...")
            dataset_info = self.dataset_registrar.register(
                name=reg_params['name'],
                path=reg_params['path'],  # Now this is a Path object
                target=reg_params.get('target'),
                problem_type=reg_params.get('problem_type'),
                force=reg_params.get('force', False)
            )
            console.print(f"  ✓ Registered: {dataset_name}")
            return True
            
        except Exception as e:
            console.print(f"  ✗ Failed to register {dataset_name}: {str(e)}", style="red")
            return False
    
    def load_competition_data(
        self, name: str, config: Dict[str, Any], with_features: bool
    ) -> Optional[pd.DataFrame]:
        """Load competition data from MDM."""
        # MDM converts dashes to underscores in dataset names
        dataset_name = name.replace('-', '_')
        
        try:
            # Get dataset info
            dataset = self.dataset_manager.get_dataset(dataset_name)
            
            # Determine base table name
            # Check if dataset has train/test split (Kaggle structure)
            if 'train' in dataset.tables:
                base_table = 'train'
            elif 'data' in dataset.tables:
                base_table = 'data'
            else:
                # Use first available table
                base_table = list(dataset.tables.keys())[0]
            
            # Check if features exist by looking for feature tables
            has_features = f'{base_table}_features' in dataset.feature_tables
            
            # Determine final table name
            if with_features and has_features:
                table_name = f'{base_table}_features'
            else:
                table_name = base_table
            
            # Direct SQLite access for simplicity
            import sqlite3
            db_path = Path(dataset.database['path'])
            conn = sqlite3.connect(db_path)
            
            # Read table to DataFrame
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
            conn.close()
            
            # If loading without features and we still got feature columns,
            # filter them manually
            if not with_features:
                # Keep only original columns (those without feature suffixes)
                original_cols = []
                for col in df.columns:
                    # Keep target and ID columns
                    if col == config['target'] or col == config.get('id_column', 'id'):
                        original_cols.append(col)
                    # Keep columns without feature suffixes
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
    
    def benchmark_competition(self, name: str, config: Dict[str, Any], use_tuning: bool = False, proper_cv: bool = False, removal_ratio: float = 0.2, custom_selection: bool = False) -> Dict[str, Any]:
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
        
        # Skip multi-label for now (MDM limitation)
        if config['problem_type'] == 'multilabel_classification':
            console.print("  ⚠️  Skipping multi-label classification (not yet supported)", style="yellow")
            results['status'] = 'skipped'
            results['reason'] = 'Multi-label not supported'
            return results
        
        # Register dataset (only once)
        console.print("[bold]Registering dataset...")
        if not self.register_competition(name, config):
            results['status'] = 'failed'
            results['reason'] = 'Failed to register dataset'
            return results
        
        # Load data
        console.print("\n[bold]Loading data...")
        df_features = self.load_competition_data(name, config, with_features=True)
        df_raw = self.load_competition_data(name, config, with_features=False)
        
        if df_features is None or df_raw is None:
            results['status'] = 'failed'
            results['reason'] = 'Failed to load data'
            return results
        
        n_features_with = len(df_features.columns) - 1  # Exclude target
        n_features_without = len(df_raw.columns) - 1
        
        console.print(f"  → With features: {n_features_with} features")
        console.print(f"  → Without features: {n_features_without} features")
        
        # Benchmark models
        console.print("\n[bold]Training models...")
        model_types = ['gbt', 'rf']
        
        for model_type in model_types:
            console.print(f"\n[cyan]{model_type.upper()}:[/cyan]")
            
            # With features (with backward feature selection)
            if custom_selection:
                console.print("  Training with features (custom backward selection)...")
            elif use_tuning:
                console.print("  Training with features (backward selection + hyperparameter tuning)...")
            else:
                console.print("  Training with features (backward selection)...")
            try:
                # Use configured removal ratio
                
                if custom_selection:
                    # CUSTOM IMPLEMENTATION: Our own backward selection + CV
                    mean_with, std_with, _, selected_features, n_selected, best_hyperparams = custom_feature_selection_cv(
                        df_features,
                        config['target'],
                        model_type,
                        config['problem_type'],
                        config['metric'],
                        removal_ratio=removal_ratio,
                        n_splits=5,
                        use_tuning=use_tuning,
                        tuning_trials=30
                    )
                    avg_n_selected = n_selected
                    # For custom selection, selected_features is already a list of feature names
                    best_features = selected_features
                elif proper_cv:
                    # PROPER WAY: First select features, then do CV
                    mean_with, std_with, _, selected_features, n_selected, best_hyperparams = select_features_then_cv(
                        df_features,
                        config['target'],
                        model_type,
                        config['problem_type'],
                        config['metric'],
                        feature_removal_ratio=removal_ratio,
                        n_splits=5,
                        use_tuning=use_tuning,
                        tuning_trials=30
                    )
                    avg_n_selected = n_selected
                    # For proper CV, selected_features is a list of feature names
                    best_features = selected_features
                else:
                    # OLD WAY: Feature selection inside each CV fold
                    mean_with, std_with, _, selected_features, avg_n_selected, best_hyperparams = cross_validate_ydf(
                        df_features,
                        config['target'],
                        model_type,
                        config['problem_type'],
                        config['metric'],
                        n_splits=5,
                        use_feature_selection=True,
                        feature_removal_ratio=removal_ratio,
                        use_tuning=use_tuning,
                        tuning_trials=30
                    )
                    # For old way, selected_features is a list of lists (one per fold)
                    # Get the most common features across folds
                    if selected_features and len(selected_features) > 0:
                        # Flatten and get unique features
                        all_features = []
                        for fold_features in selected_features:
                            if isinstance(fold_features, list):
                                all_features.extend(fold_features)
                        # Get unique features
                        best_features = list(set(all_features))[:20]  # Top 20 most common
                    else:
                        best_features = []
                results['with_features'][model_type] = {
                    'mean_score': round(mean_with, 4),
                    'std': round(std_with, 4),
                    'n_features': n_features_with,
                    'n_selected': int(avg_n_selected) if avg_n_selected else n_features_with,
                    'best_features': best_features if 'best_features' in locals() else [],
                    'best_hyperparameters': best_hyperparams if 'best_hyperparams' in locals() else {}
                }
                console.print(f"    ✓ Score: {mean_with:.4f} ± {std_with:.4f}")
                
                # Show top features from last model
                if selected_features and len(selected_features) > 0:
                    # Count non-empty feature lists
                    non_empty_features = []
                    for sf in selected_features:
                        # Check if sf is a list and has length
                        if isinstance(sf, list) and len(sf) > 0:
                            non_empty_features.append(sf)
                    
                    if non_empty_features:
                        avg_features = np.mean([len(sf) for sf in non_empty_features])
                        console.print(f"    → Average top features tracked: {avg_features:.0f}")
                        if non_empty_features[0] and len(non_empty_features[0]) > 0:
                            # Convert to strings if needed
                            feature_names = [str(f) for f in non_empty_features[0][:5]]
                            console.print(f"    → Top 5 important: {', '.join(feature_names)}...")
            except Exception as e:
                console.print(f"    ✗ Failed: {str(e)}", style="red")
                results['with_features'][model_type] = {'error': str(e)}
            
            # Without features
            console.print("  Training without features...")
            try:
                # cross_validate_ydf always returns 6 values now
                mean_without, std_without, _, _, _, _ = cross_validate_ydf(
                    df_raw,
                    config['target'],
                    model_type,
                    config['problem_type'],
                    config['metric'],
                    n_splits=5
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
                    
                    # For metrics where lower is better (RMSE, MAE)
                    if config['metric'] in ['rmse', 'mae']:
                        improvement = ((score_without - score_with) / score_without) * 100
                    else:
                        improvement = ((score_with - score_without) / score_without) * 100
                    
                    results['improvement'][model_type] = f"{improvement:+.2f}%"
                    console.print(f"    [green]Improvement: {improvement:+.2f}%[/green]")
        
        results['status'] = 'completed'
        return results
    
    def run_benchmark(self, competitions: Optional[List[str]] = None, use_tuning: bool = False, proper_cv: bool = False, removal_ratio: float = 0.2, custom_selection: bool = False):
        """Run benchmark for specified competitions or all."""
        all_competitions = get_all_competitions()
        
        if competitions:
            # Filter to specified competitions
            selected = {k: v for k, v in all_competitions.items() if k in competitions}
        else:
            selected = all_competitions
        
        console.print(Panel.fit(
            f"[bold]MDM Generic Features Benchmark[/bold]\n"
            f"Competitions: {len(selected)}\n"
            f"MDM Version: {mdm.__version__}",
            title="Benchmark Info"
        ))
        
        # Run benchmarks
        for name, config in selected.items():
            try:
                results = self.benchmark_competition(name, config, use_tuning=use_tuning, proper_cv=proper_cv, removal_ratio=removal_ratio, custom_selection=custom_selection)
                self.results['results'][name] = results
            except Exception as e:
                console.print(f"\n[red]Error benchmarking {name}: {str(e)}[/red]")
                self.results['results'][name] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        # Calculate summary
        self.calculate_summary()
        
        # Save results
        self.save_results()
        
        # Display summary
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
        output_file = self.output_dir / f"benchmark_results_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        console.print(f"\n[green]Results saved to: {output_file}[/green]")
    
    def display_summary(self):
        """Display summary table."""
        table = Table(title="Benchmark Summary", show_header=True)
        table.add_column("Competition", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Features", justify="right")
        table.add_column("GBT Improvement", justify="right")
        table.add_column("RF Improvement", justify="right")
        
        for name, result in self.results['results'].items():
            status = result.get('status', 'unknown')
            gbt_imp = result.get('improvement', {}).get('gbt', 'N/A')
            rf_imp = result.get('improvement', {}).get('rf', 'N/A')
            
            # Get feature info
            if 'with_features' in result and 'gbt' in result['with_features']:
                n_total = result['with_features']['gbt'].get('n_features', 'N/A')
                n_selected = result['with_features']['gbt'].get('n_selected', n_total)
                if n_selected != n_total and n_selected != 'N/A':
                    features_str = f"{n_selected}/{n_total}"
                else:
                    features_str = str(n_total)
            else:
                features_str = 'N/A'
            
            # Color code improvements
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
        description="Benchmark MDM generic features using YDF"
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
    parser.add_argument(
        '--tune',
        action='store_true',
        help='Enable hyperparameter tuning after feature selection'
    )
    parser.add_argument(
        '--proper-cv',
        action='store_true',
        help='Use proper CV: first select features, then do CV on selected features'
    )
    parser.add_argument(
        '--removal-ratio', '-r',
        type=float,
        default=0.2,
        help='Feature removal ratio. If <1: remove that fraction per iteration (0.2 = 20%%). If >=1: remove exactly that many features per iteration (2 = remove 2 features)'
    )
    parser.add_argument(
        '--custom-selection',
        action='store_true',
        help='Use custom feature selection implementation instead of YDF BackwardSelectionFeatureSelector'
    )
    
    args = parser.parse_args()
    
    # Run benchmark
    benchmark = MDMBenchmark(
        output_dir=args.output_dir,
        use_cache=not args.no_cache
    )
    
    benchmark.run_benchmark(competitions=args.competitions, use_tuning=args.tune, proper_cv=args.proper_cv, removal_ratio=args.removal_ratio, custom_selection=args.custom_selection)


if __name__ == '__main__':
    main()