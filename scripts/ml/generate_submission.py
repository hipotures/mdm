#!/usr/bin/env python3
"""
Generate submission files for Kaggle competitions using MDM and custom ML pipeline.
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

import pandas as pd
import numpy as np
from rich.console import Console
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

from utils.competition_configs import get_competition_config
from utils.custom_ml_helpers import (
    backward_feature_selection,
    train_final_model
)

console = Console()


def load_test_data(competition_name: str, config: Dict[str, Any], 
                  selected_features: Optional[List[str]] = None) -> pd.DataFrame:
    """Load test data from MDM dataset."""
    # MDM converts dashes to underscores
    dataset_name = competition_name.replace('-', '_')
    dataset_manager = DatasetManager()
    
    try:
        # Get dataset info
        dataset = dataset_manager.get_dataset(dataset_name)
        
        # Load test data
        import sqlite3
        db_path = Path(dataset.database['path'])
        
        # Handle different database file extensions
        if not db_path.exists():
            # Try with .sqlite extension
            alt_path = db_path.with_suffix('.sqlite')
            if alt_path.exists():
                db_path = alt_path
        
        conn = sqlite3.connect(db_path)
        
        # Check which tables are available
        if 'test' in dataset.tables:
            table_name = 'test'
        else:
            # Some competitions might not have separate test set
            console.print("[yellow]Warning: No test table found. Using train for demo.[/yellow]")
            table_name = 'train'
        
        # For submission, use raw data without features to avoid mismatch
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        
        conn.close()
        
        # If selected features provided, filter to those
        if selected_features:
            # Include ID column if exists
            id_col = config.get('id_column', 'id')
            cols_to_keep = []
            
            # Only keep features that exist in test data
            for feat in selected_features:
                if feat in df.columns:
                    cols_to_keep.append(feat)
                else:
                    console.print(f"[yellow]Warning: Feature '{feat}' not in test data, skipping[/yellow]")
            
            # Always include ID column
            if id_col in df.columns and id_col not in cols_to_keep:
                cols_to_keep = [id_col] + cols_to_keep
                
            df = df[cols_to_keep]
        
        return df
        
    except Exception as e:
        console.print(f"[red]Error loading test data: {e}[/red]")
        return None


def generate_submission(
    competition_name: str,
    model,
    test_df: pd.DataFrame,
    config: Dict[str, Any],
    output_dir: str = "submissions"
) -> str:
    """Generate submission file in Kaggle format."""
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Get ID column
    id_col = config.get('id_column', 'id')
    if id_col not in test_df.columns:
        console.print(f"[red]Error: ID column '{id_col}' not found in test data[/red]")
        return None
    
    # Make predictions
    console.print("Making predictions...")
    predictions = model.predict(test_df)
    
    # Handle different problem types
    problem_type = config['problem_type']
    target = config['target']
    
    if problem_type == 'binary_classification':
        # Check if we need probabilities
        submission_format = config.get('submission_format', 'class')
        if submission_format == 'probability':
            # Keep probabilities as is
            y_pred = predictions
        else:
            # Convert to class predictions (0 or 1)
            # YDF returns probabilities as numpy array for binary classification
            if isinstance(predictions, np.ndarray) and predictions.dtype in [np.float32, np.float64]:
                # These are probabilities, convert to classes
                y_pred = (predictions > 0.5).astype(int)
            else:
                # Already class predictions
                y_pred = predictions
            
            # Map to original labels if needed
            if 'label_mapping' in config:
                mapping = config['label_mapping']
                y_pred = [mapping.get(int(p), p) for p in y_pred]
    
    elif problem_type == 'multiclass_classification':
        y_pred = predictions
        # Map to original labels if needed
        if 'label_mapping' in config:
            mapping = config['label_mapping']
            y_pred = [mapping.get(int(p), p) for p in y_pred]
    
    elif problem_type == 'regression':
        y_pred = predictions
        # Handle RMSLE - ensure non-negative predictions
        if config.get('metric') == 'rmsle':
            y_pred = np.maximum(0, y_pred)
    
    else:
        console.print(f"[red]Unsupported problem type: {problem_type}[/red]")
        return None
    
    # Create submission DataFrame
    submission = pd.DataFrame({
        id_col: test_df[id_col],
        target: y_pred
    })
    
    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{competition_name}_submission_{timestamp}.csv"
    filepath = output_path / filename
    
    # Save submission
    submission.to_csv(filepath, index=False)
    console.print(f"[green]✓ Submission saved to: {filepath}[/green]")
    
    # Show preview
    console.print("\n[bold]Submission Preview:[/bold]")
    preview_df = submission.head(10).copy()
    # Format floats nicely in preview
    if preview_df[target].dtype == np.float64:
        preview_df[target] = preview_df[target].round(4)
    console.print(preview_df)
    console.print(f"... ({len(submission)} total rows)")
    
    return str(filepath)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate submission files for Kaggle competitions"
    )
    parser.add_argument(
        'competition',
        help='Competition name (e.g., titanic, playground-s4e10)'
    )
    parser.add_argument(
        '--model-type', '-m',
        choices=['gbt', 'rf'],
        default='gbt',
        help='Model type to use (default: gbt)'
    )
    parser.add_argument(
        '--feature-selection', '-fs',
        action='store_true',
        help='Use feature selection before training'
    )
    parser.add_argument(
        '--removal-ratio', '-r',
        type=float,
        default=0.2,
        help='Feature removal ratio for selection (default: 0.2)'
    )
    parser.add_argument(
        '--tuning', '-t',
        action='store_true',
        help='Enable hyperparameter tuning'
    )
    parser.add_argument(
        '--tuning-trials',
        type=int,
        default=30,
        help='Number of tuning trials (default: 30)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default='submissions',
        help='Output directory for submissions (default: submissions)'
    )
    parser.add_argument(
        '--use-existing-model',
        help='Path to existing model file to use for predictions'
    )
    
    args = parser.parse_args()
    
    # Get competition config
    try:
        config = get_competition_config(args.competition)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        return 1
    
    # Display info
    console.print(Panel.fit(
        f"[bold]Generating Submission[/bold]\n"
        f"Competition: {args.competition}\n"
        f"Model: {args.model_type.upper()}\n"
        f"Feature Selection: {'Yes' if args.feature_selection else 'No'}\n"
        f"Hyperparameter Tuning: {'Yes' if args.tuning else 'No'}",
        title="Submission Generator"
    ))
    
    # Load data
    dataset_manager = DatasetManager()
    dataset_name = args.competition.replace('-', '_')
    
    try:
        # Load train data
        console.print("\n[bold]Loading training data...[/bold]")
        dataset = dataset_manager.get_dataset(dataset_name)
        
        # Determine table name
        if 'train' in dataset.tables:
            base_table = 'train'
        else:
            base_table = 'data'
        
        # For submission generation, use raw data without features
        # to avoid mismatch between train and test features
        table_name = base_table
        
        # Load train data
        import sqlite3
        db_path = Path(dataset.database['path'])
        
        # Handle different database file extensions
        if not db_path.exists():
            # Try with .sqlite extension
            alt_path = db_path.with_suffix('.sqlite')
            if alt_path.exists():
                db_path = alt_path
        
        conn = sqlite3.connect(db_path)
        train_df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        conn.close()
        
        console.print(f"✓ Loaded {len(train_df)} training samples with {len(train_df.columns)-1} features")
        
        # Feature selection if requested
        selected_features = None
        if args.feature_selection:
            selected_features, _ = backward_feature_selection(
                df=train_df,
                target=config['target'],
                model_type=args.model_type,
                problem_type=config['problem_type'],
                metric_name=config['metric'],
                removal_ratio=args.removal_ratio
            )
        
        # Train final model
        model = train_final_model(
            df=train_df,
            target=config['target'],
            model_type=args.model_type,
            problem_type=config['problem_type'],
            features=selected_features,
            with_tuning=args.tuning,
            tuning_trials=args.tuning_trials
        )
        
        # Load test data
        console.print("\n[bold]Loading test data...[/bold]")
        test_df = load_test_data(args.competition, config, selected_features)
        
        if test_df is None:
            return 1
        
        console.print(f"✓ Loaded {len(test_df)} test samples")
        
        # Generate submission
        filepath = generate_submission(
            competition_name=args.competition,
            model=model,
            test_df=test_df,
            config=config,
            output_dir=args.output_dir
        )
        
        if filepath:
            console.print(f"\n[bold green]Success! Submission ready for upload.[/bold green]")
            return 0
        else:
            return 1
            
    except DatasetError as e:
        console.print(f"[red]Dataset error: {e}[/red]")
        console.print("Make sure to register the dataset first using benchmark_generic_features.py")
        return 1
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())