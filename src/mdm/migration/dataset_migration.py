"""Dataset registration migration utilities.

Provides tools for migrating between old and new dataset registration systems.
"""
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import yaml
from datetime import datetime

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.panel import Panel

from ..core import feature_flags
from ..adapters import (
    get_dataset_registrar,
    get_dataset_manager,
    get_storage_backend
)
from ..interfaces.dataset import IDatasetRegistrar, IDatasetManager
from ..core.exceptions import DatasetError

logger = logging.getLogger(__name__)
console = Console()


class DatasetMigrator:
    """Migrates datasets between registration systems."""
    
    def __init__(self):
        """Initialize migrator."""
        self.legacy_registrar = get_dataset_registrar(force_new=False)
        self.new_registrar = get_dataset_registrar(force_new=True)
        self.legacy_manager = get_dataset_manager(force_new=False)
        self.new_manager = get_dataset_manager(force_new=True)
        self._migration_log = []
    
    def migrate_dataset(
        self,
        name: str,
        dry_run: bool = True,
        preserve_features: bool = True
    ) -> Dict[str, Any]:
        """Migrate a single dataset to new system.
        
        Args:
            name: Dataset name
            dry_run: If True, only simulate migration
            preserve_features: If True, preserve generated features
            
        Returns:
            Migration result dictionary
        """
        result = {
            'name': name,
            'status': 'pending',
            'errors': [],
            'warnings': [],
            'duration': 0.0
        }
        
        start_time = pd.Timestamp.now()
        
        try:
            # Get dataset info from legacy system
            console.print(f"\n[cyan]Migrating dataset: {name}[/cyan]")
            
            if not self.legacy_manager.dataset_exists(name):
                raise DatasetError(f"Dataset '{name}' not found in legacy system")
            
            # Load configuration
            config = self.legacy_manager.get_dataset_config(name)
            info = self.legacy_manager.get_dataset_info(name)
            
            console.print(f"Source: {config.get('source', {}).get('path', 'Unknown')}")
            console.print(f"Backend: {config.get('storage', {}).get('backend', 'Unknown')}")
            console.print(f"Tables: {', '.join(config.get('storage', {}).get('tables', {}).keys())}")
            
            if dry_run:
                console.print("\n[yellow]DRY RUN - No changes will be made[/yellow]")
                
                # Simulate registration
                result['simulation'] = {
                    'would_register': True,
                    'source_path': config.get('source', {}).get('path'),
                    'target': config.get('schema', {}).get('target_column'),
                    'problem_type': config.get('schema', {}).get('problem_type'),
                    'id_columns': config.get('schema', {}).get('id_columns', []),
                    'preserve_features': preserve_features and bool(config.get('features', {}))
                }
                
                result['status'] = 'simulated'
            else:
                # Perform actual migration
                console.print("\n[green]Performing migration...[/green]")
                
                # Temporarily switch to new registration system
                original_flag = feature_flags.get("use_new_dataset_registration", False)
                feature_flags.set("use_new_dataset_registration", True)
                
                try:
                    # Re-register with new system
                    source_path = config.get('source', {}).get('path')
                    if not source_path:
                        raise DatasetError("Source path not found in configuration")
                    
                    registration_result = self.new_registrar.register(
                        name=name,
                        path=source_path,
                        target=config.get('schema', {}).get('target_column'),
                        problem_type=config.get('schema', {}).get('problem_type'),
                        id_columns=config.get('schema', {}).get('id_columns'),
                        datetime_columns=config.get('metadata', {}).get('datetime_columns', []),
                        force=True  # Override existing
                    )
                    
                    result['registration'] = registration_result
                    result['status'] = 'migrated'
                    
                    # Log successful migration
                    self._migration_log.append({
                        'dataset': name,
                        'timestamp': datetime.now().isoformat(),
                        'status': 'success',
                        'duration': (pd.Timestamp.now() - start_time).total_seconds()
                    })
                    
                finally:
                    # Restore feature flag
                    feature_flags.set("use_new_dataset_registration", original_flag)
            
        except Exception as e:
            result['status'] = 'failed'
            result['errors'].append(str(e))
            logger.error(f"Migration failed for {name}: {e}")
            
            # Log failed migration
            self._migration_log.append({
                'dataset': name,
                'timestamp': datetime.now().isoformat(),
                'status': 'failed',
                'error': str(e),
                'duration': (pd.Timestamp.now() - start_time).total_seconds()
            })
        
        result['duration'] = (pd.Timestamp.now() - start_time).total_seconds()
        return result
    
    def migrate_all_datasets(
        self,
        dry_run: bool = True,
        batch_size: int = 5
    ) -> Dict[str, Any]:
        """Migrate all datasets to new system.
        
        Args:
            dry_run: If True, only simulate migration
            batch_size: Number of datasets to migrate in parallel
            
        Returns:
            Overall migration results
        """
        # Get all datasets
        datasets = self.legacy_manager.list_datasets()
        
        if not datasets:
            console.print("[yellow]No datasets found to migrate[/yellow]")
            return {'total': 0, 'migrated': 0, 'failed': 0}
        
        console.print(Panel.fit(
            f"[bold cyan]Dataset Migration[/bold cyan]\n\n"
            f"Found {len(datasets)} datasets to migrate\n"
            f"Mode: {'DRY RUN' if dry_run else 'LIVE'}",
            title="Migration Summary"
        ))
        
        results = {
            'total': len(datasets),
            'migrated': 0,
            'failed': 0,
            'skipped': 0,
            'datasets': {}
        }
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            
            task = progress.add_task(
                f"Migrating {len(datasets)} datasets...",
                total=len(datasets)
            )
            
            for dataset in datasets:
                name = dataset['name']
                
                # Migrate dataset
                result = self.migrate_dataset(name, dry_run=dry_run)
                results['datasets'][name] = result
                
                # Update counters
                if result['status'] == 'migrated' or result['status'] == 'simulated':
                    results['migrated'] += 1
                elif result['status'] == 'failed':
                    results['failed'] += 1
                else:
                    results['skipped'] += 1
                
                progress.update(task, advance=1)
        
        # Show summary
        self._show_migration_summary(results)
        
        return results
    
    def validate_migration(
        self,
        name: str,
        tolerance: float = 0.01
    ) -> Dict[str, Any]:
        """Validate dataset migration by comparing old and new.
        
        Args:
            name: Dataset name
            tolerance: Tolerance for numeric comparisons
            
        Returns:
            Validation results
        """
        validation = {
            'dataset': name,
            'valid': True,
            'checks': {},
            'differences': []
        }
        
        try:
            # Check existence in both systems
            legacy_exists = self.legacy_manager.dataset_exists(name)
            new_exists = self.new_manager.dataset_exists(name)
            
            validation['checks']['exists'] = {
                'legacy': legacy_exists,
                'new': new_exists,
                'match': legacy_exists and new_exists
            }
            
            if not validation['checks']['exists']['match']:
                validation['valid'] = False
                return validation
            
            # Compare configurations
            legacy_config = self.legacy_manager.get_dataset_config(name)
            new_config = self.new_manager.get_dataset_config(name)
            
            # Check schema
            legacy_schema = legacy_config.get('schema', {})
            new_schema = new_config.get('schema', {})
            
            schema_match = (
                legacy_schema.get('target_column') == new_schema.get('target_column') and
                set(legacy_schema.get('id_columns', [])) == set(new_schema.get('id_columns', [])) and
                legacy_schema.get('problem_type') == new_schema.get('problem_type')
            )
            
            validation['checks']['schema'] = {
                'match': schema_match,
                'legacy': legacy_schema,
                'new': new_schema
            }
            
            if not schema_match:
                validation['valid'] = False
                validation['differences'].append('Schema mismatch')
            
            # Compare data
            backend = get_storage_backend()
            
            for table_name in legacy_config.get('storage', {}).get('tables', {}):
                if table_name in new_config.get('storage', {}).get('tables', {}):
                    # Get row counts
                    legacy_stats = backend.get_table_stats(name, table_name)
                    new_stats = backend.get_table_stats(name, table_name)
                    
                    row_match = legacy_stats.get('row_count') == new_stats.get('row_count')
                    col_match = legacy_stats.get('column_count') == new_stats.get('column_count')
                    
                    validation['checks'][f'table_{table_name}'] = {
                        'row_count_match': row_match,
                        'column_count_match': col_match,
                        'legacy_rows': legacy_stats.get('row_count'),
                        'new_rows': new_stats.get('row_count'),
                        'legacy_cols': legacy_stats.get('column_count'),
                        'new_cols': new_stats.get('column_count')
                    }
                    
                    if not (row_match and col_match):
                        validation['valid'] = False
                        validation['differences'].append(f'Table {table_name} data mismatch')
            
        except Exception as e:
            validation['valid'] = False
            validation['error'] = str(e)
            logger.error(f"Validation failed for {name}: {e}")
        
        return validation
    
    def get_migration_status(self) -> Dict[str, Any]:
        """Get overall migration status."""
        # Count datasets in each system
        legacy_datasets = set(d['name'] for d in self.legacy_manager.list_datasets())
        new_datasets = set(d['name'] for d in self.new_manager.list_datasets())
        
        return {
            'legacy_only': list(legacy_datasets - new_datasets),
            'new_only': list(new_datasets - legacy_datasets),
            'migrated': list(legacy_datasets & new_datasets),
            'total_legacy': len(legacy_datasets),
            'total_new': len(new_datasets),
            'migration_log': self._migration_log
        }
    
    def _show_migration_summary(self, results: Dict[str, Any]) -> None:
        """Display migration summary."""
        table = Table(title="Migration Results")
        table.add_column("Dataset", style="cyan")
        table.add_column("Status", style="yellow")
        table.add_column("Duration", style="green")
        table.add_column("Notes", style="white")
        
        for name, result in results['datasets'].items():
            status = result['status']
            if status == 'migrated':
                status_str = "[green]✓ Migrated[/green]"
            elif status == 'simulated':
                status_str = "[yellow]⚠ Simulated[/yellow]"
            elif status == 'failed':
                status_str = "[red]✗ Failed[/red]"
            else:
                status_str = "[grey]- Skipped[/grey]"
            
            duration = f"{result.get('duration', 0):.2f}s"
            
            notes = []
            if result.get('errors'):
                notes.append(f"[red]{result['errors'][0]}[/red]")
            if result.get('warnings'):
                notes.append(f"[yellow]{result['warnings'][0]}[/yellow]")
            
            table.add_row(
                name,
                status_str,
                duration,
                '\n'.join(notes) if notes else '-'
            )
        
        console.print(table)
        
        # Summary panel
        console.print(Panel.fit(
            f"[bold]Migration Complete[/bold]\n\n"
            f"Total: {results['total']}\n"
            f"Migrated: [green]{results['migrated']}[/green]\n"
            f"Failed: [red]{results['failed']}[/red]\n"
            f"Skipped: [grey]{results['skipped']}[/grey]",
            title="Summary"
        ))


class DatasetValidator:
    """Validates dataset registration consistency."""
    
    def __init__(self):
        """Initialize validator."""
        self.console = Console()
    
    def validate_consistency(
        self,
        dataset_name: str,
        check_data: bool = True,
        check_features: bool = True
    ) -> Dict[str, Any]:
        """Validate dataset registration consistency.
        
        Args:
            dataset_name: Dataset to validate
            check_data: If True, validate data consistency
            check_features: If True, validate feature consistency
            
        Returns:
            Validation results
        """
        results = {
            'dataset': dataset_name,
            'timestamp': datetime.now().isoformat(),
            'passed': True,
            'checks': {}
        }
        
        # Get managers for both systems
        legacy_manager = get_dataset_manager(force_new=False)
        new_manager = get_dataset_manager(force_new=True)
        
        # Check 1: Dataset exists in both
        legacy_exists = legacy_manager.dataset_exists(dataset_name)
        new_exists = new_manager.dataset_exists(dataset_name)
        
        results['checks']['existence'] = {
            'legacy': legacy_exists,
            'new': new_exists,
            'passed': legacy_exists == new_exists
        }
        
        if not results['checks']['existence']['passed']:
            results['passed'] = False
            return results
        
        if not (legacy_exists and new_exists):
            return results
        
        # Check 2: Configuration consistency
        legacy_config = legacy_manager.get_dataset_config(dataset_name)
        new_config = new_manager.get_dataset_config(dataset_name)
        
        config_checks = {
            'target_column': (
                legacy_config.get('schema', {}).get('target_column') ==
                new_config.get('schema', {}).get('target_column')
            ),
            'problem_type': (
                legacy_config.get('schema', {}).get('problem_type') ==
                new_config.get('schema', {}).get('problem_type')
            ),
            'id_columns': (
                set(legacy_config.get('schema', {}).get('id_columns', [])) ==
                set(new_config.get('schema', {}).get('id_columns', []))
            ),
            'backend': (
                legacy_config.get('storage', {}).get('backend') ==
                new_config.get('storage', {}).get('backend')
            )
        }
        
        results['checks']['configuration'] = {
            'details': config_checks,
            'passed': all(config_checks.values())
        }
        
        if not results['checks']['configuration']['passed']:
            results['passed'] = False
        
        # Check 3: Data consistency (if requested)
        if check_data:
            data_checks = self._validate_data_consistency(
                dataset_name,
                legacy_manager,
                new_manager
            )
            results['checks']['data'] = data_checks
            
            if not data_checks['passed']:
                results['passed'] = False
        
        # Check 4: Feature consistency (if requested)
        if check_features:
            feature_checks = self._validate_feature_consistency(
                dataset_name,
                legacy_config,
                new_config
            )
            results['checks']['features'] = feature_checks
            
            if not feature_checks['passed']:
                results['passed'] = False
        
        return results
    
    def _validate_data_consistency(
        self,
        dataset_name: str,
        legacy_manager: IDatasetManager,
        new_manager: IDatasetManager
    ) -> Dict[str, Any]:
        """Validate data consistency between systems."""
        checks = {
            'passed': True,
            'tables': {}
        }
        
        try:
            # Get table lists
            legacy_info = legacy_manager.get_dataset_info(dataset_name)
            new_info = new_manager.get_dataset_info(dataset_name)
            
            legacy_tables = set(legacy_info.get('storage', {}).get('tables', {}).keys())
            new_tables = set(new_info.get('storage', {}).get('tables', {}).keys())
            
            # Check table consistency
            if legacy_tables != new_tables:
                checks['passed'] = False
                checks['table_mismatch'] = {
                    'legacy_only': list(legacy_tables - new_tables),
                    'new_only': list(new_tables - legacy_tables)
                }
            
            # Check each table's data
            for table_name in legacy_tables & new_tables:
                # Load samples
                legacy_sample = legacy_manager.load_dataset(
                    dataset_name, table_name, limit=1000
                )
                new_sample = new_manager.load_dataset(
                    dataset_name, table_name, limit=1000
                )
                
                table_check = {
                    'row_count_match': len(legacy_sample) == len(new_sample),
                    'column_match': list(legacy_sample.columns) == list(new_sample.columns),
                    'legacy_shape': legacy_sample.shape,
                    'new_shape': new_sample.shape
                }
                
                # Check data values (first few rows)
                if table_check['row_count_match'] and table_check['column_match']:
                    try:
                        pd.testing.assert_frame_equal(
                            legacy_sample.head(10),
                            new_sample.head(10),
                            check_dtype=False
                        )
                        table_check['data_match'] = True
                    except AssertionError:
                        table_check['data_match'] = False
                else:
                    table_check['data_match'] = False
                
                table_check['passed'] = (
                    table_check['row_count_match'] and
                    table_check['column_match'] and
                    table_check['data_match']
                )
                
                if not table_check['passed']:
                    checks['passed'] = False
                
                checks['tables'][table_name] = table_check
                
        except Exception as e:
            checks['passed'] = False
            checks['error'] = str(e)
        
        return checks
    
    def _validate_feature_consistency(
        self,
        dataset_name: str,
        legacy_config: Dict[str, Any],
        new_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate feature consistency between systems."""
        checks = {
            'passed': True,
            'feature_tables': {}
        }
        
        legacy_features = legacy_config.get('features', {})
        new_features = new_config.get('features', {})
        
        # Check feature table consistency
        legacy_tables = set(legacy_features.keys())
        new_tables = set(new_features.keys())
        
        if legacy_tables != new_tables:
            checks['passed'] = False
            checks['table_mismatch'] = {
                'legacy_only': list(legacy_tables - new_tables),
                'new_only': list(new_tables - legacy_tables)
            }
        
        # Check each feature table
        for table_name in legacy_tables & new_tables:
            legacy_feat = legacy_features[table_name]
            new_feat = new_features[table_name]
            
            table_check = {
                'n_features_match': (
                    legacy_feat.get('n_features') == new_feat.get('n_features')
                ),
                'source_match': (
                    legacy_feat.get('source_table') == new_feat.get('source_table')
                )
            }
            
            table_check['passed'] = all(table_check.values())
            
            if not table_check['passed']:
                checks['passed'] = False
            
            checks['feature_tables'][table_name] = table_check
        
        return checks
