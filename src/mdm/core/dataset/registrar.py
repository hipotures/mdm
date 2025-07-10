"""New dataset registrar implementation.

Provides a clean, modular implementation of the dataset registration process.
"""
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import yaml
from datetime import datetime

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.console import Console
from rich.panel import Panel

from ...interfaces.dataset import IDatasetRegistrar
from ...core.exceptions import DatasetError
from ...adapters import get_storage_backend, get_feature_generator, get_dataset_manager
from .validators import DatasetNameValidator, DatasetPathValidator, DatasetStructureDetector
from .loaders import loader_registry

logger = logging.getLogger(__name__)
console = Console()


class RegistrationStep:
    """Represents a step in the registration process."""
    
    def __init__(self, name: str, description: str, func):
        self.name = name
        self.description = description
        self.func = func
        self.result = None
        self.error = None
        self.duration = 0.0


class NewDatasetRegistrar(IDatasetRegistrar):
    """New implementation of dataset registrar with improved architecture."""
    
    def __init__(self):
        """Initialize registrar with dependencies."""
        self.name_validator = DatasetNameValidator()
        self.path_validator = DatasetPathValidator()
        self.structure_detector = DatasetStructureDetector()
        self._metrics = {
            "datasets_registered": 0,
            "total_registration_time": 0.0,
            "registration_errors": 0,
            "steps_executed": 0,
        }
        logger.info("Initialized NewDatasetRegistrar")
    
    def register(
        self,
        name: str,
        path: str,
        target: Optional[str] = None,
        problem_type: Optional[str] = None,
        id_columns: Optional[List[str]] = None,
        datetime_columns: Optional[List[str]] = None,
        force: bool = False
    ) -> Dict[str, Any]:
        """Register a new dataset with improved process.
        
        This implementation uses a modular approach with clear separation
        of concerns and better error handling.
        """
        start_time = pd.Timestamp.now()
        path_obj = Path(path)
        
        # Initialize registration context
        context = {
            'name': name,
            'path': path_obj,
            'target': target,
            'problem_type': problem_type,
            'id_columns': id_columns or [],
            'datetime_columns': datetime_columns or [],
            'force': force,
            'metadata': {}
        }
        
        # Define registration steps
        steps = [
            RegistrationStep("Validate Name", "Validating dataset name", self._step_validate_name),
            RegistrationStep("Check Exists", "Checking if dataset exists", self._step_check_exists),
            RegistrationStep("Validate Path", "Validating dataset path", self._step_validate_path),
            RegistrationStep("Detect Structure", "Detecting dataset structure", self._step_detect_structure),
            RegistrationStep("Discover Files", "Discovering data files", self._step_discover_files),
            RegistrationStep("Create Storage", "Creating storage backend", self._step_create_storage),
            RegistrationStep("Load Data", "Loading data files", self._step_load_data),
            RegistrationStep("Analyze Data", "Analyzing data types", self._step_analyze_data),
            RegistrationStep("Detect Features", "Detecting ID and target columns", self._step_detect_features),
            RegistrationStep("Generate Features", "Generating feature tables", self._step_generate_features),
            RegistrationStep("Compute Stats", "Computing statistics", self._step_compute_stats),
            RegistrationStep("Save Config", "Saving dataset configuration", self._step_save_config),
        ]
        
        # Show registration header
        console.print(Panel.fit(
            f"[bold cyan]Registering Dataset: {name}[/bold cyan]\n"
            f"Path: {path}",
            title="Dataset Registration"
        ))
        
        # Execute steps with progress tracking
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            
            overall_task = progress.add_task(
                "[cyan]Registration Progress",
                total=len(steps)
            )
            
            for i, step in enumerate(steps):
                step_start = pd.Timestamp.now()
                
                try:
                    # Execute step
                    step.result = step.func(context)
                    step.duration = (pd.Timestamp.now() - step_start).total_seconds()
                    self._metrics["steps_executed"] += 1
                    
                    # Update progress
                    progress.update(overall_task, advance=1)
                    
                except Exception as e:
                    step.error = e
                    step.duration = (pd.Timestamp.now() - step_start).total_seconds()
                    
                    # Log error
                    logger.error(f"Step '{step.name}' failed: {e}")
                    self._metrics["registration_errors"] += 1
                    
                    # Update progress and stop
                    progress.update(
                        overall_task,
                        description=f"[red]Failed at: {step.description}[/red]"
                    )
                    
                    raise DatasetError(f"Registration failed at step '{step.name}': {e}")
        
        # Registration successful
        duration = (pd.Timestamp.now() - start_time).total_seconds()
        self._metrics["datasets_registered"] += 1
        self._metrics["total_registration_time"] += duration
        
        # Build result
        result = {
            'name': context['normalized_name'],
            'path': str(context['path']),
            'backend': context['storage']['backend'],
            'tables': list(context.get('tables', {}).keys()),
            'features_generated': len(context.get('feature_tables', {})),
            'registration_time': duration,
            'metadata': context['metadata']
        }
        
        # Show success message
        console.print(Panel.fit(
            f"[bold green]✓ Dataset '{context['normalized_name']}' registered successfully![/bold green]\n\n"
            f"Backend: {result['backend']}\n"
            f"Tables: {', '.join(result['tables'])}\n"
            f"Features: {result['features_generated']} tables generated\n"
            f"Time: {duration:.2f}s",
            title="Registration Complete"
        ))
        
        return result
    
    def _step_validate_name(self, context: Dict[str, Any]) -> None:
        """Step 1: Validate dataset name."""
        normalized = self.name_validator.validate(context['name'])
        context['normalized_name'] = normalized
        logger.info(f"Validated name: {context['name']} -> {normalized}")
    
    def _step_check_exists(self, context: Dict[str, Any]) -> None:
        """Step 2: Check if dataset exists."""
        manager = get_dataset_manager()
        
        if manager.dataset_exists(context['normalized_name']):
            if context['force']:
                logger.info(f"Dataset exists, removing due to --force")
                manager.remove_dataset(context['normalized_name'], force=True)
            else:
                raise DatasetError(f"Dataset '{context['normalized_name']}' already exists")
    
    def _step_validate_path(self, context: Dict[str, Any]) -> None:
        """Step 3: Validate dataset path."""
        validated_path = self.path_validator.validate(context['path'])
        context['path'] = validated_path
        logger.info(f"Validated path: {validated_path}")
    
    def _step_detect_structure(self, context: Dict[str, Any]) -> None:
        """Step 4: Detect dataset structure."""
        structure = self.structure_detector.detect_structure(context['path'])
        context['structure'] = structure
        
        # Apply detected features
        if 'detected_features' in structure:
            features = structure['detected_features']
            if 'id_column' in features and not context['id_columns']:
                context['id_columns'] = [features['id_column']]
            if 'target_column' in features and not context['target']:
                context['target'] = features['target_column']
        
        logger.info(f"Detected structure: {structure['type']}")
    
    def _step_discover_files(self, context: Dict[str, Any]) -> None:
        """Step 5: Discover data files."""
        files = context['structure']['files']
        
        if not files:
            raise DatasetError("No data files found")
        
        context['files'] = files
        logger.info(f"Discovered {len(files)} data files")
    
    def _step_create_storage(self, context: Dict[str, Any]) -> None:
        """Step 6: Create storage backend."""
        backend = get_storage_backend()
        
        # Initialize dataset storage
        backend.initialize_dataset(context['normalized_name'])
        
        context['storage'] = {
            'backend': backend.__class__.__name__.replace('Adapter', ''),
            'instance': backend
        }
        
        logger.info(f"Created storage with backend: {context['storage']['backend']}")
    
    def _step_load_data(self, context: Dict[str, Any]) -> None:
        """Step 7: Load data files into storage."""
        backend = context['storage']['instance']
        tables = {}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=None,  # Use internal progress
        ) as progress:
            
            for table_name, file_path in context['files'].items():
                task = progress.add_task(
                    f"Loading {file_path.name}",
                    total=None
                )
                
                # Get appropriate loader
                loader = loader_registry.get_loader(file_path)
                
                # Load data in batches
                total_rows = 0
                for i, batch in enumerate(loader.load_batch(file_path)):
                    if i == 0:
                        # Create table on first batch
                        backend.create_table(
                            context['normalized_name'],
                            table_name,
                            batch
                        )
                    
                    # Insert batch
                    backend.insert_data(
                        context['normalized_name'],
                        table_name,
                        batch
                    )
                    
                    total_rows += len(batch)
                    progress.update(
                        task,
                        description=f"Loading {file_path.name} ({total_rows:,} rows)"
                    )
                
                tables[table_name] = {
                    'path': str(file_path),
                    'rows': total_rows,
                    'columns': loader.metadata.get('columns', [])
                }
                
                progress.update(task, completed=True)
        
        context['tables'] = tables
        logger.info(f"Loaded {len(tables)} tables")
    
    def _step_analyze_data(self, context: Dict[str, Any]) -> None:
        """Step 8: Analyze data types and columns."""
        backend = context['storage']['instance']
        column_info = {}
        
        for table_name in context['tables']:
            # Get sample data
            sample = backend.execute_query(
                context['normalized_name'],
                f"SELECT * FROM {table_name} LIMIT 1000"
            )
            
            # Detect column types
            types = self.detect_column_types(sample)
            column_info[table_name] = {
                'columns': list(sample.columns),
                'types': types
            }
        
        context['column_info'] = column_info
        logger.info("Analyzed column types")
    
    def _step_detect_features(self, context: Dict[str, Any]) -> None:
        """Step 9: Detect ID columns and problem type."""
        # Focus on main data table
        main_table = 'data' if 'data' in context['tables'] else list(context['tables'].keys())[0]
        backend = context['storage']['instance']
        
        # Get sample for detection
        sample = backend.execute_query(
            context['normalized_name'],
            f"SELECT * FROM {main_table} LIMIT 10000"
        )
        
        # Detect ID columns if not provided
        if not context['id_columns']:
            column_types = context['column_info'][main_table]['types']
            id_cols = self.structure_detector.detect_id_columns(sample, column_types)
            if id_cols:
                context['id_columns'] = id_cols
                logger.info(f"Detected ID columns: {id_cols}")
        
        # Detect target column if not provided
        if not context['target']:
            target = self.structure_detector.detect_target_column(
                sample,
                list(sample.columns)
            )
            if target:
                context['target'] = target
                logger.info(f"Detected target column: {target}")
        
        # Infer problem type if not provided
        if not context['problem_type'] and context['target']:
            problem_type = self.structure_detector.infer_problem_type(
                sample,
                context['target']
            )
            context['problem_type'] = problem_type
            logger.info(f"Inferred problem type: {problem_type}")
    
    def _step_generate_features(self, context: Dict[str, Any]) -> None:
        """Step 10: Generate feature tables."""
        # Check if feature generation is enabled
        from ...config import get_config
        config = get_config()
        
        if not config.feature_engineering.enabled:
            context['feature_tables'] = {}
            return
        
        backend = context['storage']['instance']
        generator = get_feature_generator()
        feature_tables = {}
        
        # Generate features for main table
        main_table = 'data' if 'data' in context['tables'] else list(context['tables'].keys())[0]
        
        # Load data for feature generation
        data = backend.load_data(context['normalized_name'], main_table)
        
        # Generate features
        features = generator.generate_features(
            data,
            target_column=context.get('target'),
            problem_type=context.get('problem_type'),
            datetime_columns=context.get('datetime_columns'),
            id_columns=context.get('id_columns')
        )
        
        # Save feature tables
        feature_table_name = f"{main_table}_features"
        backend.create_table(
            context['normalized_name'],
            feature_table_name,
            features
        )
        backend.insert_data(
            context['normalized_name'],
            feature_table_name,
            features
        )
        
        feature_tables[feature_table_name] = {
            'source_table': main_table,
            'n_features': len(features.columns),
            'feature_names': list(features.columns)
        }
        
        context['feature_tables'] = feature_tables
        logger.info(f"Generated {len(features.columns)} features")
    
    def _step_compute_stats(self, context: Dict[str, Any]) -> None:
        """Step 11: Compute dataset statistics."""
        backend = context['storage']['instance']
        stats = {}
        
        total_rows = 0
        total_columns = 0
        
        for table_name, table_info in context['tables'].items():
            # Get basic stats from backend
            table_stats = backend.get_table_stats(
                context['normalized_name'],
                table_name
            )
            
            stats[table_name] = table_stats
            total_rows += table_stats.get('row_count', 0)
            total_columns += table_stats.get('column_count', 0)
        
        # Add overall stats
        stats['_overall'] = {
            'total_rows': total_rows,
            'total_columns': total_columns,
            'n_tables': len(context['tables']),
            'n_feature_tables': len(context.get('feature_tables', {}))
        }
        
        context['stats'] = stats
        context['metadata']['statistics'] = stats
        logger.info(f"Computed statistics: {total_rows} rows, {total_columns} columns")
    
    def _step_save_config(self, context: Dict[str, Any]) -> None:
        """Step 12: Save dataset configuration."""
        # Build configuration
        config = {
            'name': context['normalized_name'],
            'registration_date': datetime.now().isoformat(),
            'source': {
                'path': str(context['path']),
                'type': context['structure']['type']
            },
            'storage': {
                'backend': context['storage']['backend'],
                'tables': context['tables']
            },
            'schema': {
                'target_column': context.get('target'),
                'id_columns': context.get('id_columns', []),
                'datetime_columns': context.get('datetime_columns', []),
                'problem_type': context.get('problem_type')
            },
            'features': context.get('feature_tables', {}),
            'statistics': context.get('stats', {}),
            'metadata': context.get('metadata', {})
        }
        
        # Save to manager
        manager = get_dataset_manager()
        manager.update_dataset_config(context['normalized_name'], config)
        
        logger.info("Saved dataset configuration")
    
    # Implement interface methods for compatibility
    
    def validate_dataset_name(self, name: str) -> None:
        """Validate dataset name format."""
        self.name_validator.validate(name)
    
    def detect_structure(self, path: str) -> Dict[str, Any]:
        """Auto-detect dataset structure."""
        return self.structure_detector.detect_structure(Path(path))
    
    def detect_file_format(self, file_path: str) -> str:
        """Detect file format from extension."""
        file_format, _ = self.path_validator.detect_format(Path(file_path))
        return file_format
    
    def load_data_file(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Load data from a single file."""
        loader = loader_registry.get_loader(Path(file_path))
        return loader.load(Path(file_path), **kwargs)
    
    def detect_column_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """Detect column data types."""
        type_map = {
            'int64': 'integer',
            'float64': 'numeric',
            'object': 'string',
            'datetime64[ns]': 'datetime',
            'bool': 'boolean',
            'category': 'categorical'
        }
        
        types = {}
        for col in df.columns:
            dtype_name = str(df[col].dtype)
            
            # Try to match known types
            for pattern, type_name in type_map.items():
                if pattern in dtype_name:
                    types[col] = type_name
                    break
            else:
                # Default to string for unknown
                types[col] = 'string'
        
        return types
    
    def detect_id_columns(self, df: pd.DataFrame) -> List[str]:
        """Auto-detect ID columns."""
        column_types = self.detect_column_types(df)
        return self.structure_detector.detect_id_columns(df, column_types)
    
    def detect_target_column(self, df: pd.DataFrame, columns: List[str]) -> Optional[str]:
        """Auto-detect target column."""
        return self.structure_detector.detect_target_column(df, columns)
    
    def detect_problem_type(self, df: pd.DataFrame, target_column: str) -> str:
        """Auto-detect problem type from target column."""
        return self.structure_detector.infer_problem_type(df, target_column)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get registrar metrics."""
        return self._metrics.copy()
