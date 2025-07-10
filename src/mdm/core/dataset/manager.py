"""New dataset manager implementation.

Provides clean dataset management operations with improved architecture.
"""
import logging
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import yaml
from datetime import datetime

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from ...interfaces.dataset import IDatasetManager
from ...core.exceptions import DatasetError
from ...adapters import get_storage_backend
from ...config import get_config
from .loaders import loader_registry

logger = logging.getLogger(__name__)
console = Console()


class NewDatasetManager(IDatasetManager):
    """New implementation of dataset manager with clean architecture."""
    
    def __init__(self):
        """Initialize manager."""
        config = get_config()
        self.config_dir = Path(config.paths.datasets_config)
        self.datasets_dir = Path(config.paths.datasets_path)
        self._metrics = {
            "datasets_listed": 0,
            "datasets_loaded": 0,
            "datasets_removed": 0,
            "datasets_exported": 0,
            "total_data_loaded_mb": 0.0,
        }
        logger.info("Initialized NewDatasetManager")
    
    def list_datasets(
        self,
        limit: Optional[int] = None,
        sort_by: Optional[str] = None,
        backend: Optional[str] = None,
        tag: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List all datasets with filtering and sorting."""
        datasets = []
        
        # Read all dataset configs
        for config_file in self.config_dir.glob("*.yaml"):
            try:
                with open(config_file) as f:
                    config = yaml.safe_load(f)
                
                # Apply filters
                if backend and config.get('storage', {}).get('backend') != backend:
                    continue
                
                if tag and tag not in config.get('tags', []):
                    continue
                
                # Build dataset info
                dataset = {
                    'name': config['name'],
                    'registration_date': config.get('registration_date', 'Unknown'),
                    'backend': config.get('storage', {}).get('backend', 'Unknown'),
                    'tables': list(config.get('storage', {}).get('tables', {}).keys()),
                    'target': config.get('schema', {}).get('target_column'),
                    'problem_type': config.get('schema', {}).get('problem_type'),
                    'tags': config.get('tags', []),
                    'size': self._calculate_dataset_size(config['name'])
                }
                
                datasets.append(dataset)
                
            except Exception as e:
                logger.warning(f"Failed to read config {config_file}: {e}")
        
        # Sort datasets
        if sort_by:
            if sort_by == 'name':
                datasets.sort(key=lambda d: d['name'])
            elif sort_by == 'registration_date':
                datasets.sort(key=lambda d: d['registration_date'], reverse=True)
            elif sort_by == 'size':
                datasets.sort(key=lambda d: d['size'], reverse=True)
        
        # Apply limit
        if limit:
            datasets = datasets[:limit]
        
        self._metrics["datasets_listed"] += 1
        return datasets
    
    def get_dataset_info(self, name: str) -> Dict[str, Any]:
        """Get detailed dataset information."""
        config = self.get_dataset_config(name)
        
        # Add runtime information
        info = config.copy()
        info['size_bytes'] = self._calculate_dataset_size(name)
        info['location'] = str(self.datasets_dir / name)
        
        # Get table details from storage
        backend = get_storage_backend()
        table_details = {}
        
        for table_name in config.get('storage', {}).get('tables', {}):
            try:
                stats = backend.get_table_stats(name, table_name)
                table_details[table_name] = stats
            except Exception as e:
                logger.warning(f"Failed to get stats for {name}.{table_name}: {e}")
        
        info['table_details'] = table_details
        
        return info
    
    def dataset_exists(self, name: str) -> bool:
        """Check if dataset exists."""
        config_path = self.config_dir / f"{name}.yaml"
        return config_path.exists()
    
    def remove_dataset(self, name: str, force: bool = False) -> None:
        """Remove dataset and all associated data."""
        if not self.dataset_exists(name):
            raise DatasetError(f"Dataset '{name}' not found")
        
        if not force:
            # Confirm removal
            console.print(f"[yellow]Warning: This will permanently delete dataset '{name}'[/yellow]")
            confirm = console.input("Type 'yes' to confirm: ")
            if confirm.lower() != 'yes':
                console.print("[red]Removal cancelled[/red]")
                return
        
        try:
            # Remove from storage backend
            backend = get_storage_backend()
            backend.drop_dataset(name)
            
            # Remove config file
            config_path = self.config_dir / f"{name}.yaml"
            if config_path.exists():
                config_path.unlink()
            
            # Remove dataset directory if exists
            dataset_dir = self.datasets_dir / name
            if dataset_dir.exists():
                shutil.rmtree(dataset_dir)
            
            self._metrics["datasets_removed"] += 1
            logger.info(f"Removed dataset '{name}'")
            
        except Exception as e:
            raise DatasetError(f"Failed to remove dataset '{name}': {e}")
    
    def update_dataset(
        self,
        name: str,
        description: Optional[str] = None,
        target: Optional[str] = None,
        problem_type: Optional[str] = None,
        id_columns: Optional[List[str]] = None,
        tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Update dataset metadata."""
        if not self.dataset_exists(name):
            raise DatasetError(f"Dataset '{name}' not found")
        
        # Load current config
        config = self.get_dataset_config(name)
        
        # Update fields
        updates = {}
        
        if description is not None:
            config['description'] = description
            updates['description'] = description
        
        if target is not None:
            config.setdefault('schema', {})['target_column'] = target
            updates['target'] = target
        
        if problem_type is not None:
            config.setdefault('schema', {})['problem_type'] = problem_type
            updates['problem_type'] = problem_type
        
        if id_columns is not None:
            config.setdefault('schema', {})['id_columns'] = id_columns
            updates['id_columns'] = id_columns
        
        if tags is not None:
            config['tags'] = tags
            updates['tags'] = tags
        
        # Add update timestamp
        config['last_updated'] = datetime.now().isoformat()
        
        # Save updated config
        self.update_dataset_config(name, config)
        
        logger.info(f"Updated dataset '{name}': {updates}")
        return updates
    
    def export_dataset(
        self,
        name: str,
        output_dir: str,
        format: str = "csv",
        compression: Optional[str] = None,
        tables: Optional[List[str]] = None
    ) -> List[str]:
        """Export dataset to files."""
        if not self.dataset_exists(name):
            raise DatasetError(f"Dataset '{name}' not found")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get dataset info
        info = self.get_dataset_info(name)
        backend = get_storage_backend()
        
        # Determine tables to export
        if tables is None:
            tables = list(info.get('storage', {}).get('tables', {}).keys())
        
        exported_files = []
        
        for table_name in tables:
            # Load data
            data = backend.load_data(name, table_name)
            
            # Determine file name
            base_name = f"{name}_{table_name}"
            
            if format == "csv":
                file_name = f"{base_name}.csv"
                if compression == "gzip":
                    file_name += ".gz"
                elif compression == "zip":
                    file_name += ".zip"
                
                file_path = output_path / file_name
                data.to_csv(file_path, index=False, compression=compression)
                
            elif format == "parquet":
                file_name = f"{base_name}.parquet"
                file_path = output_path / file_name
                data.to_parquet(file_path, compression=compression)
                
            elif format == "json":
                file_name = f"{base_name}.json"
                if compression == "gzip":
                    file_name += ".gz"
                
                file_path = output_path / file_name
                data.to_json(file_path, orient='records', compression=compression)
                
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            exported_files.append(str(file_path))
            logger.info(f"Exported {table_name} to {file_path}")
        
        self._metrics["datasets_exported"] += 1
        return exported_files
    
    def get_dataset_stats(
        self,
        name: str,
        mode: str = "basic",
        tables: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Get dataset statistics."""
        if not self.dataset_exists(name):
            raise DatasetError(f"Dataset '{name}' not found")
        
        info = self.get_dataset_info(name)
        backend = get_storage_backend()
        
        # Determine tables
        if tables is None:
            tables = list(info.get('storage', {}).get('tables', {}).keys())
        
        stats = {
            'dataset_name': name,
            'mode': mode,
            'tables': {}
        }
        
        for table_name in tables:
            table_stats = backend.get_table_stats(name, table_name)
            
            if mode == "detailed":
                # Load sample for detailed analysis
                sample = backend.execute_query(
                    name,
                    f"SELECT * FROM {table_name} LIMIT 10000"
                )
                
                # Add detailed statistics
                table_stats['numeric_stats'] = sample.describe().to_dict()
                table_stats['missing_values'] = sample.isnull().sum().to_dict()
                table_stats['unique_counts'] = {col: sample[col].nunique() for col in sample.columns}
            
            stats['tables'][table_name] = table_stats
        
        # Add overall statistics
        total_rows = sum(t.get('row_count', 0) for t in stats['tables'].values())
        total_columns = sum(t.get('column_count', 0) for t in stats['tables'].values())
        
        stats['overall'] = {
            'total_rows': total_rows,
            'total_columns': total_columns,
            'n_tables': len(stats['tables']),
            'size_bytes': self._calculate_dataset_size(name)
        }
        
        return stats
    
    def search_datasets(
        self,
        pattern: str,
        search_in: List[str] = ["name", "description", "tags"],
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Search datasets by pattern."""
        all_datasets = self.list_datasets()
        results = []
        
        pattern_lower = pattern.lower()
        
        for dataset in all_datasets:
            # Load full config for description
            if 'description' in search_in:
                try:
                    config = self.get_dataset_config(dataset['name'])
                    dataset['description'] = config.get('description', '')
                except Exception:
                    dataset['description'] = ''
            
            # Check each field
            for field in search_in:
                value = dataset.get(field, '')
                
                if isinstance(value, list):
                    # For tags
                    value = ' '.join(value)
                
                if pattern_lower in str(value).lower():
                    results.append(dataset)
                    break
        
        # Apply limit
        if limit:
            results = results[:limit]
        
        return results
    
    def load_dataset(
        self,
        name: str,
        table_name: str = "data",
        columns: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """Load dataset data."""
        if not self.dataset_exists(name):
            raise DatasetError(f"Dataset '{name}' not found")
        
        backend = get_storage_backend()
        
        # Build query
        if columns:
            columns_str = ', '.join(columns)
        else:
            columns_str = '*'
        
        query = f"SELECT {columns_str} FROM {table_name}"
        
        if limit:
            query += f" LIMIT {limit}"
        
        # Execute query
        data = backend.execute_query(name, query)
        
        # Track metrics
        data_size_mb = data.memory_usage(deep=True).sum() / 1024 / 1024
        self._metrics["datasets_loaded"] += 1
        self._metrics["total_data_loaded_mb"] += data_size_mb
        
        logger.info(f"Loaded {len(data)} rows from {name}.{table_name}")
        return data
    
    def get_dataset_config(self, name: str) -> Dict[str, Any]:
        """Get dataset configuration from YAML file."""
        config_path = self.config_dir / f"{name}.yaml"
        
        if not config_path.exists():
            raise DatasetError(f"Dataset config not found: {name}")
        
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        return config
    
    def update_dataset_config(self, name: str, config: Dict[str, Any]) -> None:
        """Update dataset configuration YAML file."""
        config_path = self.config_dir / f"{name}.yaml"
        
        # Ensure directory exists
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write config
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Updated config for dataset '{name}'")
    
    def register_dataset(self, dataset_info: Any) -> None:
        """Register a new dataset (called by registrar)."""
        # Convert dataset info to config format
        config = {
            'name': dataset_info.name,
            'registration_date': datetime.now().isoformat(),
            'description': dataset_info.description,
            'source': dataset_info.source,
            'storage': {
                'backend': dataset_info.database['backend'],
                'tables': {t: {'path': p} for t, p in dataset_info.tables.items()}
            },
            'schema': {
                'target_column': dataset_info.target_column,
                'id_columns': dataset_info.id_columns,
                'problem_type': dataset_info.problem_type,
                'time_column': dataset_info.time_column,
                'group_column': dataset_info.group_column,
            },
            'features': dataset_info.feature_tables or {},
            'tags': dataset_info.tags,
            'metadata': dataset_info.metadata
        }
        
        # Save config
        self.update_dataset_config(dataset_info.name, config)
    
    def _calculate_dataset_size(self, name: str) -> int:
        """Calculate total dataset size in bytes."""
        try:
            backend = get_storage_backend()
            config = self.get_dataset_config(name)
            
            total_size = 0
            for table_name in config.get('storage', {}).get('tables', {}):
                try:
                    stats = backend.get_table_stats(name, table_name)
                    # Estimate size: rows * columns * 8 bytes average
                    size = stats.get('row_count', 0) * stats.get('column_count', 0) * 8
                    total_size += size
                except Exception:
                    pass
            
            return total_size
        except Exception:
            return 0
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get manager metrics."""
        return self._metrics.copy()
    
    def display_datasets(self, datasets: List[Dict[str, Any]], title: str = "Datasets") -> None:
        """Display datasets in a rich table."""
        if not datasets:
            console.print(f"[yellow]No {title.lower()} found[/yellow]")
            return
        
        table = Table(title=title)
        table.add_column("Name", style="cyan")
        table.add_column("Backend", style="green")
        table.add_column("Tables", style="yellow")
        table.add_column("Target", style="magenta")
        table.add_column("Type", style="blue")
        table.add_column("Registered", style="white")
        
        for dataset in datasets:
            table.add_row(
                dataset['name'],
                dataset.get('backend', 'Unknown'),
                ', '.join(dataset.get('tables', [])),
                dataset.get('target', '-'),
                dataset.get('problem_type', '-'),
                dataset.get('registration_date', 'Unknown')[:10]
            )
        
        console.print(table)
