"""Dataset registrar implementing the 12-step registration process."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml
from loguru import logger
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from ydata_profiling import ProfileReport

from mdm.config import get_config_manager
from mdm.core.exceptions import DatasetError
from mdm.dataset.auto_detect import (
    detect_delimiter,
    detect_id_columns,
    detect_kaggle_structure,
    discover_data_files,
    extract_target_from_sample_submission,
    infer_problem_type,
    validate_kaggle_submission_format,
)
from mdm.dataset.manager import DatasetManager
from mdm.features.generator import FeatureGenerator
from mdm.models.dataset import DatasetInfo
from mdm.models.enums import ColumnType
from mdm.storage.factory import BackendFactory
from mdm.utils.serialization import serialize_for_yaml
from mdm.monitoring import SimpleMonitor, MetricType
from mdm.dataset.loaders import (
    FileLoaderRegistry,
    CSVLoader,
    ParquetLoader,
    JSONLoader,
    CompressedCSVLoader,
    ExcelLoader,
)


class DatasetRegistrar:
    """Handles the 12-step dataset registration process."""

    def __init__(self, manager: Optional[DatasetManager] = None):
        """Initialize dataset registrar.
        
        Args:
            manager: Optional DatasetManager instance
        """
        self.manager = manager or DatasetManager()
        config_manager = get_config_manager()
        self.config = config_manager.config
        self.base_path = config_manager.base_path
        self.feature_generator = FeatureGenerator()
        self._detected_datetime_columns = []
        self.monitor = SimpleMonitor()
        
        # Initialize file loader registry
        self._loader_registry = FileLoaderRegistry()
        self._setup_loaders()

    def register(
        self,
        name: str,
        path: Path,
        auto_detect: bool = True,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any
    ) -> DatasetInfo:
        """Register a dataset following the 12-step process.
        
        Args:
            name: Dataset name
            path: Path to dataset directory or file
            auto_detect: Enable auto-detection (default: True)
            description: Dataset description
            tags: Dataset tags
            **kwargs: Additional dataset metadata
            
        Returns:
            Registered DatasetInfo
            
        Raises:
            DatasetError: If registration fails
        """
        logger.info(f"Starting registration for dataset '{name}'")
        start_time = pd.Timestamp.now()
        
        # Steps 1-3: Validation and preparation
        normalized_name = self._prepare_registration(name, path, kwargs.get('force', False))
        
        # Steps 4-5: Detection and discovery
        detected_info, files = self._detect_and_discover(path, auto_detect)
        
        # Step 6: Create database
        db_info = self._create_database(normalized_name)
        
        # Steps 7-10.5: Load data and generate features
        table_mappings, column_info, id_columns, target_column, problem_type, feature_tables = \
            self._process_data(
                normalized_name, files, db_info, detected_info, 
                auto_detect, kwargs
            )
        
        # Steps 11-12: Create info and save
        dataset_info = self._finalize_registration(
            normalized_name, name, path, description, tags,
            db_info, table_mappings, column_info,
            id_columns, target_column, problem_type,
            feature_tables, detected_info, kwargs
        )
        
        # Record metrics
        self._record_registration_metrics(
            normalized_name, start_time, dataset_info,
            files, feature_tables
        )
        
        logger.info(f"Dataset '{normalized_name}' registered successfully")
        return dataset_info
    
    def _prepare_registration(self, name: str, path: Path, force: bool) -> str:
        """Steps 1-3: Validate name, check existence, validate path."""
        # Step 1: Validate dataset name
        normalized_name = self._validate_name(name)
        
        # Step 2: Check if dataset already exists
        if self.manager.dataset_exists(normalized_name):
            if force:
                self._remove_existing_dataset(normalized_name)
            else:
                raise DatasetError(f"Dataset '{normalized_name}' already exists")
        
        # Step 3: Validate path
        self._validate_path(path)
        
        return normalized_name
    
    def _remove_existing_dataset(self, normalized_name: str) -> None:
        """Remove existing dataset when --force is used."""
        logger.info(f"Dataset '{normalized_name}' exists, removing due to --force flag")
        from mdm.dataset.operations import RemoveOperation
        remove_op = RemoveOperation()
        try:
            remove_op.execute(normalized_name, force=True, dry_run=False)
        except Exception as e:
            logger.warning(f"Failed to remove existing dataset: {e}")
    
    def _detect_and_discover(self, path: Path, auto_detect: bool) -> tuple[Dict[str, Any], Dict[str, Path]]:
        """Steps 4-5: Auto-detect structure and discover files."""
        # Step 4: Auto-detect or manual mode
        if auto_detect:
            detected_info = self._auto_detect(path)
        else:
            detected_info = {}
        
        # Step 5: Discover data files
        files = self._discover_files(path, detected_info)
        
        return detected_info, files
    
    def _process_data(
        self,
        normalized_name: str,
        files: Dict[str, Path],
        db_info: Dict[str, Any],
        detected_info: Dict[str, Any],
        auto_detect: bool,
        kwargs: Dict[str, Any]
    ) -> tuple:
        """Steps 7-10.5: Load data, analyze, and generate features."""
        # Create a single progress context for all operations
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=None,
        ) as progress:
            # Step 7: Load data files
            table_mappings = self._load_data_files(files, db_info, progress)
            
            # Step 8: Detect columns and types
            column_info = self._analyze_columns_with_progress(db_info, table_mappings, progress)
            
            # Step 9: Detect ID columns
            id_columns = self._determine_id_columns(detected_info, column_info, auto_detect)
            
            # Step 10: Determine problem type and target
            target_column, problem_type = self._determine_target_and_type(
                detected_info, column_info, auto_detect, kwargs
            )
            
            # Step 10.5: Generate feature tables
            feature_tables = self._generate_features_if_enabled(
                normalized_name, db_info, table_mappings, column_info,
                target_column, id_columns, kwargs, progress
            )
            
            if feature_tables:
                table_mappings.update(feature_tables)
        
        return table_mappings, column_info, id_columns, target_column, problem_type, feature_tables
    
    def _analyze_columns_with_progress(
        self,
        db_info: Dict[str, Any],
        table_mappings: Dict[str, str],
        progress: Progress
    ) -> Dict[str, Any]:
        """Analyze columns with progress tracking."""
        task = progress.add_task("Analyzing columns and data types...", total=None)
        column_info = self._analyze_columns(db_info, table_mappings)
        progress.update(task, completed=True, visible=False)
        return column_info
    
    def _determine_id_columns(
        self,
        detected_info: Dict[str, Any],
        column_info: Dict[str, Any],
        auto_detect: bool
    ) -> List[str]:
        """Determine ID columns from detection or auto-detection."""
        id_columns = detected_info.get('id_columns', [])
        if not id_columns and auto_detect:
            id_columns = self._detect_id_columns(column_info)
        return id_columns
    
    def _determine_target_and_type(
        self,
        detected_info: Dict[str, Any],
        column_info: Dict[str, Any],
        auto_detect: bool,
        kwargs: Dict[str, Any]
    ) -> tuple[Optional[str], Optional[str]]:
        """Determine target column and problem type."""
        # User-provided values take precedence over auto-detected ones
        target_column = kwargs.get('target_column') or detected_info.get('target_column')
        problem_type = kwargs.get('problem_type') or detected_info.get('problem_type')
        
        if auto_detect and not problem_type and target_column:
            problem_type = self._infer_problem_type(column_info, target_column)
        
        return target_column, problem_type
    
    def _generate_features_if_enabled(
        self,
        normalized_name: str,
        db_info: Dict[str, Any],
        table_mappings: Dict[str, str],
        column_info: Dict[str, Any],
        target_column: Optional[str],
        id_columns: List[str],
        kwargs: Dict[str, Any],
        progress: Progress
    ) -> Dict[str, str]:
        """Generate features if enabled."""
        # Check configuration first, then kwargs override
        generate_features = self.config.feature_engineering.enabled
        if 'generate_features' in kwargs:
            generate_features = kwargs['generate_features']
        
        logger.debug(f"Feature generation config: enabled={self.config.feature_engineering.enabled}, "
                    f"kwargs override={kwargs.get('generate_features')}, final={generate_features}")
        
        if generate_features:
            return self._generate_features(
                normalized_name, db_info, table_mappings, column_info,
                target_column, id_columns, kwargs.get('type_schema'),
                progress
            )
        return {}
    
    def _finalize_registration(
        self,
        normalized_name: str,
        name: str,
        path: Path,
        description: Optional[str],
        tags: Optional[List[str]],
        db_info: Dict[str, Any],
        table_mappings: Dict[str, str],
        column_info: Dict[str, Any],
        id_columns: List[str],
        target_column: Optional[str],
        problem_type: Optional[str],
        feature_tables: Dict[str, str],
        detected_info: Dict[str, Any],
        kwargs: Dict[str, Any]
    ) -> DatasetInfo:
        """Steps 11-12: Create dataset info and save registration."""
        # Step 11: Create dataset info
        dataset_info = self._create_dataset_info(
            normalized_name, name, path, description, tags,
            db_info, table_mappings, id_columns,
            target_column, problem_type, feature_tables,
            detected_info, kwargs
        )
        
        # Step 11.5: Add datetime columns to metadata
        if self._detected_datetime_columns:
            dataset_info.metadata['datetime_columns'] = self._detected_datetime_columns
            logger.info(f"Saved datetime columns in metadata: {self._detected_datetime_columns}")
        
        # Step 11.6: Compute and add statistics
        statistics = self._compute_initial_statistics(normalized_name, db_info, table_mappings)
        if statistics:
            dataset_info.metadata['statistics'] = statistics
        
        # Step 12: Save registration
        self.manager.register_dataset(dataset_info)
        
        return dataset_info
    
    def _create_dataset_info(
        self,
        normalized_name: str,
        name: str,
        path: Path,
        description: Optional[str],
        tags: Optional[List[str]],
        db_info: Dict[str, Any],
        table_mappings: Dict[str, str],
        id_columns: List[str],
        target_column: Optional[str],
        problem_type: Optional[str],
        feature_tables: Dict[str, str],
        detected_info: Dict[str, Any],
        kwargs: Dict[str, Any]
    ) -> DatasetInfo:
        """Create DatasetInfo object."""
        return DatasetInfo(
            name=normalized_name,
            display_name=kwargs.get('display_name', name),
            description=description or detected_info.get('description', ''),
            database=db_info,
            tables=table_mappings,
            problem_type=problem_type,
            target_column=target_column,
            id_columns=id_columns,
            time_column=kwargs.get('time_column'),
            group_column=kwargs.get('group_column'),
            feature_tables=feature_tables,
            tags=tags or [],
            source=str(path),
            **{k: v for k, v in kwargs.items()
               if k not in ['display_name', 'target_column', 'problem_type', 
                           'id_columns', 'time_column', 'group_column']}
        )
    
    def _record_registration_metrics(
        self,
        normalized_name: str,
        start_time: pd.Timestamp,
        dataset_info: DatasetInfo,
        files: Dict[str, Path],
        feature_tables: Dict[str, str]
    ) -> None:
        """Record registration metrics."""
        duration_ms = (pd.Timestamp.now() - start_time).total_seconds() * 1000
        total_rows = 0
        if hasattr(dataset_info, 'statistics') and dataset_info.statistics:
            total_rows = dataset_info.statistics.get('row_count', 0)
        
        self.monitor.record_metric(
            MetricType.DATASET_REGISTER,
            f"register_{normalized_name}",
            duration_ms=duration_ms,
            success=True,
            dataset_name=normalized_name,
            row_count=total_rows,
            metadata={
                'files': len(files),
                'features_generated': len(feature_tables) if feature_tables else 0,
                'backend': self.config.database.default_backend
            }
        )

    def _validate_name(self, name: str) -> str:
        """Step 1: Validate dataset name."""
        return self.manager.validate_dataset_name(name)

    def _validate_path(self, path: Path) -> Path:
        """Step 3: Validate dataset path."""
        path = path.resolve()

        if not path.exists():
            raise DatasetError(f"Path does not exist: {path}")

        return path

    def _auto_detect(self, path: Path) -> Dict[str, Any]:
        """Step 4: Auto-detect dataset structure and metadata."""
        detected = {}

        # Check if it's a Kaggle dataset
        if path.is_dir() and detect_kaggle_structure(path):
            logger.info("Detected Kaggle competition structure")
            detected['structure'] = 'kaggle'

            # Extract target from sample submission
            sample_submission = path / 'sample_submission.csv'
            if sample_submission.exists():
                target = extract_target_from_sample_submission(sample_submission)
                if target:
                    detected['target_column'] = target
                    logger.info(f"Detected target column: {target}")

        return detected

    def _discover_files(self, path: Path, detected_info: Dict[str, Any]) -> Dict[str, Path]:
        """Step 5: Discover data files."""
        if path.is_file():
            # Single file dataset
            return {'data': path}

        # Directory - discover files
        files = discover_data_files(path)

        if not files:
            raise DatasetError(f"No data files found in {path}")

        # Validate Kaggle structure if detected
        if detected_info.get('structure') == 'kaggle':
            if 'test' in files and 'sample_submission' in files:
                # Read test columns for validation
                # Try to detect encoding first
                encoding = 'utf-8'
                try:
                    import chardet
                    with open(files['test'], 'rb') as fb:
                        raw_data = fb.read(10000)
                        result = chardet.detect(raw_data)
                        if result['encoding'] and result['confidence'] > 0.7:
                            encoding = result['encoding']
                except:
                    pass
                
                try:
                    logger.debug(f"Reading test file: {files['test']}")
                    logger.debug(f"Using encoding: {encoding}")
                    
                    # Check file extension to use appropriate reader
                    test_file = files['test']
                    if test_file.suffix.lower() == '.parquet':
                        test_df = pd.read_parquet(test_file).head(5)
                    elif test_file.suffix.lower() in ['.xlsx', '.xls']:
                        test_df = pd.read_excel(test_file, nrows=5)
                    else:
                        # CSV/TSV files
                        test_df = pd.read_csv(files['test'], nrows=5, encoding=encoding, encoding_errors='replace')
                    
                    logger.debug(f"Test columns: {list(test_df.columns)}")
                except Exception as e:
                    logger.error(f"Failed to read test file {files['test']}: {e}")
                    import traceback
                    logger.debug(f"Traceback: {traceback.format_exc()}")
                    raise
                is_valid, error = validate_kaggle_submission_format(
                    list(test_df.columns),
                    files['sample_submission']
                )
                if not is_valid:
                    logger.warning(f"Kaggle validation warning: {error}")

        return files

    def _create_database(self, name: str) -> Dict[str, Any]:
        """Step 6: Create database for dataset."""
        backend_type = self.config.database.default_backend
        backend_config = getattr(self.config.database, backend_type)

        # Create database info based on backend type
        db_info = {'backend': backend_type}
        
        # Include the full backend configuration (including SQLite pragmas)
        db_info.update(backend_config.model_dump())

        if backend_type in ['sqlite', 'duckdb']:
            # File-based backends
            db_path = self.base_path / self.config.paths.datasets_path / name / f"{name}.{backend_type}"
            db_path.parent.mkdir(parents=True, exist_ok=True)
            db_info['path'] = str(db_path)
        else:
            # Server-based backends (PostgreSQL)
            db_info.update({
                'host': backend_config.host,
                'port': backend_config.port,
                'database': f"mdm_{name}",
                'user': backend_config.user,
                'password': backend_config.password
            })

            # Create database if needed
            self._create_postgresql_database(db_info)

        return db_info

    def _create_postgresql_database(self, db_info: Dict[str, Any]) -> None:
        """Create PostgreSQL database if it doesn't exist."""
        try:
            import psycopg2
            from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

            # Connect to default database
            conn = psycopg2.connect(
                host=db_info['host'],
                port=db_info['port'],
                user=db_info['user'],
                password=db_info['password'],
                database='postgres'
            )
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)

            with conn.cursor() as cur:
                # Check if database exists
                cur.execute(
                    "SELECT 1 FROM pg_database WHERE datname = %s",
                    (db_info['database'],)
                )
                if not cur.fetchone():
                    # Create database
                    cur.execute(f"CREATE DATABASE {db_info['database']}")
                    logger.info(f"Created PostgreSQL database: {db_info['database']}")

            conn.close()

        except Exception as e:
            raise DatasetError(f"Failed to create PostgreSQL database: {e}")
    
    def _setup_loaders(self) -> None:
        """Set up file loaders in the registry."""
        batch_size = self.config.performance.batch_size
        
        # Register all loaders
        self._loader_registry.register(CSVLoader(batch_size))
        self._loader_registry.register(ParquetLoader(batch_size))
        self._loader_registry.register(JSONLoader(batch_size))
        self._loader_registry.register(CompressedCSVLoader(batch_size))
        self._loader_registry.register(ExcelLoader(batch_size))

    def _load_data_files(
        self,
        files: Dict[str, Path],
        db_info: Dict[str, Any],
        progress: Optional[Progress] = None
    ) -> Dict[str, str]:
        """Step 7: Load data files into database using file loaders.
        
        This implementation uses the Strategy pattern to handle different file types.
        Each loader handles its specific format and loads data in chunks.
        """
        backend = BackendFactory.create(db_info['backend'], db_info)
        table_mappings = {}
        
        # Store column types detected from first chunk for later use
        self._detected_column_types = {}
        # Store detected datetime columns for parsing
        self._detected_datetime_columns = []

        # Use passed progress or create new one
        if progress is None:
            progress_ctx = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeRemainingColumn(),
                console=None,
            )
            progress = progress_ctx.__enter__()
            own_progress = True
        else:
            own_progress = False

        try:
            # Get database path/connection string
            if 'path' in db_info:
                db_path = db_info['path']
            else:
                # For server-based backends
                db_path = f"{db_info['backend']}://{db_info['user']}:{db_info['password']}@{db_info['host']}:{db_info['port']}/{db_info['database']}"

            # Create database if needed
            if not backend.database_exists(db_path):
                backend.create_database(db_path)

            # Get engine
            engine = backend.get_engine(db_path)

            # Determine which tables to detect types for
            detect_types_for = ['train', 'data']  # Focus on main training data

            for file_key, file_path in files.items():
                table_name = file_key
                
                # Get appropriate loader for this file
                loader = self._loader_registry.get_loader(file_path)
                if loader is None:
                    logger.warning(f"No loader found for file type: {file_path}")
                    continue
                
                # Load the file using the appropriate loader
                loader.load_file(
                    file_path=file_path,
                    table_name=table_name,
                    backend=backend,
                    engine=engine,
                    progress=progress,
                    detect_types_for=detect_types_for
                )
                
                table_mappings[file_key] = table_name
                
                # Store detected column types and datetime columns
                if loader.detected_column_types:
                    self._detected_column_types.update(loader.detected_column_types)
                if loader.detected_datetime_columns:
                    self._detected_datetime_columns.extend(
                        col for col in loader.detected_datetime_columns 
                        if col not in self._detected_datetime_columns
                    )
                
                # Get final row count from database
                try:
                    if hasattr(backend, 'query'):
                        row_count_result = backend.query(f"SELECT COUNT(*) as count FROM {table_name}")
                        if row_count_result is not None and not row_count_result.empty:
                            row_count = int(row_count_result.iloc[0]['count'])
                            logger.info(f"Loaded {row_count} rows into '{table_name}'")
                    else:
                        # Alternative: use pandas to read and count
                        count_df = pd.read_sql_query(f"SELECT COUNT(*) as count FROM {table_name}", engine)
                        row_count = int(count_df.iloc[0]['count'])
                        logger.info(f"Loaded {row_count} rows into '{table_name}'")
                except Exception as e:
                    logger.info(f"Loaded data into '{table_name}'")

        except Exception as e:
            raise DatasetError(f"Failed to load data files: {e}")
        finally:
            if own_progress:
                progress_ctx.__exit__(None, None, None)
            backend.close_connections()

        return table_mappings
    
    def _convert_datetime_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect and convert datetime columns in DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with datetime columns converted
        """
        df_copy = df.copy()
        
        for col in df_copy.columns:
            if df_copy[col].dtype == 'object':
                
                try:
                    # Try to parse as datetime
                    # Use infer_datetime_format for better detection
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        parsed = pd.to_datetime(df_copy[col], errors='coerce')
                    
                    # Check success rate
                    success_rate = parsed.notna().sum() / len(parsed)
                    
                    # If more than 80% parsed successfully, convert the column
                    if success_rate >= 0.8:
                        df_copy[col] = parsed
                        logger.debug(f"Detected datetime column: {col} (success rate: {success_rate:.0%})")
                        # Store for later parsing in all chunks
                        if col not in self._detected_datetime_columns:
                            self._detected_datetime_columns.append(col)
                    
                except Exception as e:
                    # If parsing fails, skip this column
                    logger.debug(f"Could not parse {col} as datetime: {e}")
                    
        return df_copy
    
    def _detect_datetime_columns_from_sample(self, df: pd.DataFrame) -> None:
        """Detect datetime columns from a sample DataFrame.
        
        Args:
            df: Sample DataFrame
        """
        logger.debug(f"Detecting datetime columns from sample with {len(df)} rows")
        for col in df.columns:
            if df[col].dtype == 'object':
                logger.debug(f"Checking column {col} for datetime patterns")
                # For datetime detection, we don't skip based on unique values
                # Dates can be very unique (timestamps) but still valid dates
                logger.debug(f"Checking if column {col} contains datetime values")
                
                try:
                    # Try to parse as datetime
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        parsed = pd.to_datetime(df[col], errors='coerce')
                    
                    # Check success rate
                    success_rate = parsed.notna().sum() / len(parsed)
                    
                    # If more than 80% parsed successfully, mark as datetime
                    if success_rate >= 0.8:
                        self._detected_datetime_columns.append(col)
                        logger.debug(f"Detected datetime column: {col} (success rate: {success_rate:.0%})")
                    
                except Exception as e:
                    # If parsing fails, skip this column
                    logger.debug(f"Could not parse {col} as datetime: {e}")
    
    def _detect_and_store_column_types(self, df: pd.DataFrame, table_name: str) -> None:
        """Detect column types from first chunk using ydata-profiling.
        
        This runs on the first chunk only to save memory.
        Results are stored for later use in feature generation.
        """
        logger.info(f"Detecting column types from first chunk of {table_name}")
        
        # First, try to detect and convert datetime columns
        df = self._convert_datetime_columns(df)
        
        try:
            # Suppress ydata-profiling output
            import os
            import warnings
            old_tqdm = os.environ.get('TQDM_DISABLE')
            os.environ['TQDM_DISABLE'] = '1'
            
            # Also suppress warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Create minimal profile with completely suppressed output
                # Disable all tqdm globally
                import tqdm
                import sys
                from io import StringIO
                
                # Save original tqdm
                original_tqdm = tqdm.tqdm
                original_trange = tqdm.trange
                
                # Create dummy tqdm that does nothing
                def dummy_tqdm(iterable=None, *args, **kwargs):
                    # Return the iterable itself, or an empty iterator
                    # This properly handles progress bars without displaying anything
                    if iterable is not None:
                        # Convert to list to consume any generators/ranges
                        # but return an iterator to match tqdm behavior
                        try:
                            return iter(list(iterable))
                        except:
                            return iter([])
                    return iter([])
                
                # Replace tqdm temporarily
                tqdm.tqdm = dummy_tqdm
                tqdm.trange = dummy_tqdm
                
                # Also redirect stdout to suppress any remaining output
                old_stdout = sys.stdout
                sys.stdout = StringIO()
                
                try:
                    profile = ProfileReport(
                        df,
                        minimal=True,
                        progress_bar=False
                    )
                finally:
                    # Restore tqdm and stdout
                    tqdm.tqdm = original_tqdm
                    tqdm.trange = original_trange
                    sys.stdout = old_stdout
                
                # Extract column types
                description = profile.get_description()
                variables = description.variables
                
                for col_name, var_info in variables.items():
                    if isinstance(var_info, dict):
                        var_type = var_info.get('type', 'Unsupported')
                    else:
                        var_type = getattr(var_info, 'type', 'Unsupported')
                    
                    # Store the detected type
                    self._detected_column_types[col_name] = var_type
                    
            # Restore TQDM setting
            if old_tqdm is None:
                os.environ.pop('TQDM_DISABLE', None)
            else:
                os.environ['TQDM_DISABLE'] = old_tqdm
                
            logger.info(f"Detected {len(self._detected_column_types)} column types")
            
        except Exception as e:
            logger.warning(f"Failed to detect column types with ydata-profiling: {e}")
            # Store basic types as fallback
            for col in df.columns:
                dtype = str(df[col].dtype)
                if 'int' in dtype or 'float' in dtype:
                    self._detected_column_types[col] = 'Numeric'
                elif 'datetime' in dtype:
                    self._detected_column_types[col] = 'DateTime'
                else:
                    self._detected_column_types[col] = 'Categorical'

    def _analyze_columns(
        self,
        db_info: Dict[str, Any],
        tables: Dict[str, str]
    ) -> Dict[str, Dict[str, Any]]:
        """Step 8: Analyze columns and types."""
        backend = BackendFactory.create(db_info['backend'], db_info)
        column_info = {}

        try:
            # Get database path/connection string
            if 'path' in db_info:
                db_path = db_info['path']
            else:
                # For server-based backends
                db_path = f"{db_info['backend']}://{db_info['user']}:{db_info['password']}@{db_info['host']}:{db_info['port']}/{db_info['database']}"

            engine = backend.get_engine(db_path)

            for table_key, table_name in tables.items():
                # Get table info
                table_info = backend.get_table_info(table_name, engine)
                
                # Check if table_info is valid
                if not table_info or 'columns' not in table_info:
                    logger.error(f"Invalid table info for {table_name}: {table_info}")
                    raise DatasetError(f"Failed to get table info for {table_name}")

                # Get sample data for analysis
                sample_df = backend.read_table_to_dataframe(table_name, engine, limit=1000)

                # Convert column info to dict format
                columns_dict = {col['name']: col['type'] for col in table_info['columns']}

                column_info[table_key] = {
                    'columns': columns_dict,
                    'sample_data': sample_df.to_dict('list'),
                    'dtypes': sample_df.dtypes.to_dict()
                }

        except Exception as e:
            raise DatasetError(f"Failed to analyze columns: {e}")
        finally:
            backend.close_connections()

        return column_info

    def _detect_id_columns(self, column_info: Dict[str, Dict[str, Any]]) -> List[str]:
        """Step 9: Detect ID columns."""
        all_id_columns = set()

        for table_key, info in column_info.items():
            columns = list(info['columns'].keys())
            sample_data = info['sample_data']

            # Detect ID columns for this table
            id_cols = detect_id_columns(sample_data, columns)
            all_id_columns.update(id_cols)

        return sorted(list(all_id_columns))

    def _infer_problem_type(
        self,
        column_info: Dict[str, Dict[str, Any]],
        target_column: str
    ) -> Optional[str]:
        """Step 10: Infer problem type from target column."""
        # Find target column in tables
        for table_key, info in column_info.items():
            if target_column in info['columns']:
                sample_data = info['sample_data'].get(target_column, [])
                if sample_data:
                    n_unique = len(set(v for v in sample_data if v is not None))
                    return infer_problem_type(target_column, sample_data, n_unique)

        return None

    def update_from_auto_detect(
        self,
        dataset_name: str,
        force: bool = False
    ) -> DatasetInfo:
        """Re-run auto-detection on an existing dataset.
        
        Args:
            dataset_name: Name of dataset to update
            force: Force update even if auto-detected values exist
            
        Returns:
            Updated DatasetInfo
        """
        dataset_info = self.manager.get_dataset(dataset_name)
        if not dataset_info:
            raise DatasetError(f"Dataset '{dataset_name}' not found")

        # Re-run auto-detection on source path
        source_path = Path(dataset_info.source)
        detected = self._auto_detect(source_path)

        updates = {}

        # Update target column if not set or force
        if detected.get('target_column') and (force or not dataset_info.target_column):
            updates['target_column'] = detected['target_column']

        # Update problem type if not set or force
        if detected.get('problem_type') and (force or not dataset_info.problem_type):
            updates['problem_type'] = detected['problem_type']

        # Re-detect ID columns if needed
        if force or not dataset_info.id_columns:
            backend = self.manager.get_backend(dataset_name)
            column_info = self._analyze_columns(dataset_info.database, dataset_info.tables)
            id_columns = self._detect_id_columns(column_info)
            if id_columns:
                updates['id_columns'] = id_columns

        if updates:
            return self.manager.update_dataset(dataset_name, updates)

        return dataset_info

    def _generate_features(
        self,
        dataset_name: str,
        db_info: Dict[str, Any],
        table_mappings: Dict[str, str],
        column_info: Dict[str, Dict[str, Any]],
        target_column: Optional[str],
        id_columns: List[str],
        type_schema: Optional[Dict[str, str]] = None,
        progress: Optional[Progress] = None
    ) -> Dict[str, str]:
        """Generate feature tables for the dataset.
        
        NOTE: This is called AFTER all data is loaded, so feature generation
        happens on the complete dataset in the database, not in memory.
        
        Args:
            dataset_name: Name of the dataset
            db_info: Database connection info
            table_mappings: Mapping of table types to table names
            column_info: Column information for each table
            target_column: Target column name
            id_columns: List of ID columns
            
        Returns:
            Mapping of feature table types to table names
        """
        logger.info(f"Generating features for dataset '{dataset_name}'")

        backend = BackendFactory.create(db_info['backend'], db_info)
        feature_tables = {}

        try:
            # Get database path/connection string
            if 'path' in db_info:
                db_path = db_info['path']
            else:
                db_path = f"{db_info['backend']}://{db_info['user']}:{db_info['password']}@{db_info['host']}:{db_info['port']}/{db_info['database']}"

            engine = backend.get_engine(db_path)

            # Determine column types using ydata-profiling
            column_types = self._detect_column_types_with_profiling(
                column_info, 
                table_mappings, 
                engine,
                target_column,
                id_columns,
                type_schema
            )

            # Generate features for each table
            feature_tables = self.feature_generator.generate_feature_tables(
                engine=engine,
                dataset_name=dataset_name,
                source_tables=table_mappings,
                column_types=column_types,
                target_column=target_column,
                id_columns=id_columns,
                progress=progress,
                datetime_columns=self._detected_datetime_columns
            )

            logger.info(f"Generated {len(feature_tables)} feature tables")

        except Exception as e:
            logger.error(f"Failed to generate features: {e}")
            # Feature generation failure is not critical - continue without features

        finally:
            backend.close_connections()

        return feature_tables

    def _detect_column_types_with_profiling(
        self,
        column_info: Dict[str, Dict[str, Any]],
        table_mappings: Dict[str, str],
        engine: Any,
        target_column: Optional[str],
        id_columns: List[str],
        type_schema: Optional[Dict[str, str]] = None
    ) -> Dict[str, ColumnType]:
        """Detect column types using ydata-profiling.
        
        Args:
            column_info: Column information from database
            table_mappings: Table mappings
            engine: Database engine
            target_column: Target column name
            id_columns: List of ID columns
            type_schema: Optional type schema to override detection
            
        Returns:
            Dictionary mapping column names to ColumnType
        """
        column_types = {}
        
        # Process ALL tables to ensure we have column types for each
        for table_type, table_name in table_mappings.items():
            logger.info(f"Detecting column types for {table_type} table: {table_name}")
            
            # Create minimal profile for type detection
            try:
                # Read sample data for profiling
                backend = BackendFactory.create(engine.url.drivername, {})
                df = backend.read_table_to_dataframe(table_name, engine, limit=10000)
                
                # Suppress tqdm progress bars from ydata-profiling
                import os
                old_tqdm = os.environ.get('TQDM_DISABLE')
                os.environ['TQDM_DISABLE'] = '1'
                
                try:
                    # Use stored column types if available (only for first table)
                    if table_type == 'train' and hasattr(self, '_detected_column_types') and self._detected_column_types:
                        # We already have column types from first chunk
                        logger.info("Using column types detected during data loading for train table")
                        for col_name, var_type in self._detected_column_types.items():
                            if col_name not in df.columns:
                                continue
                                
                            # Map to MDM ColumnType
                            if col_name in id_columns:
                                column_types[col_name] = ColumnType.ID
                            elif col_name == target_column:
                                column_types[col_name] = ColumnType.TARGET
                            elif var_type == "Numeric":
                                column_types[col_name] = ColumnType.NUMERIC
                            elif var_type == "Categorical":
                                column_types[col_name] = ColumnType.CATEGORICAL
                            elif var_type == "DateTime":
                                column_types[col_name] = ColumnType.DATETIME
                            elif var_type in ["Text", "URL", "Path", "File"]:
                                # Check cardinality for Text columns
                                # If low cardinality, treat as categorical
                                n_unique = df[col_name].nunique() if col_name in df.columns else 0
                                threshold = self.config.feature_engineering.type_detection.categorical_threshold
                                
                                logger.info(f"Checking column {col_name}: type={var_type}, unique={n_unique}, threshold={threshold}")
                                
                                if n_unique > 0 and n_unique <= threshold:
                                    logger.info(f"Column {col_name} has {n_unique} unique values (<= {threshold}), treating as categorical")
                                    column_types[col_name] = ColumnType.CATEGORICAL
                                else:
                                    logger.info(f"Column {col_name} has {n_unique} unique values (> {threshold}), keeping as text")
                                    column_types[col_name] = ColumnType.TEXT
                            elif var_type == "Boolean":
                                column_types[col_name] = ColumnType.CATEGORICAL
                            else:
                                column_types[col_name] = ColumnType.CATEGORICAL
                    else:
                        # Fallback: run profiling on sample
                        # Suppress all output
                        import tqdm
                        import sys
                        from io import StringIO
                        
                        original_tqdm = tqdm.tqdm
                        original_trange = tqdm.trange
                        
                        def dummy_tqdm(iterable=None, *args, **kwargs):
                            if iterable is not None:
                                try:
                                    return iter(list(iterable))
                                except:
                                    return iter([])
                            return iter([])
                        
                        tqdm.tqdm = dummy_tqdm
                        tqdm.trange = dummy_tqdm
                        old_stdout = sys.stdout
                        sys.stdout = StringIO()
                        
                        try:
                            profile = ProfileReport(
                                df,
                                minimal=True,
                                type_schema=type_schema,
                                progress_bar=False
                            )
                        finally:
                            tqdm.tqdm = original_tqdm
                            tqdm.trange = original_trange
                            sys.stdout = old_stdout
                        
                        # Extract column types from profile
                        description = profile.get_description()
                        variables = description.variables
                        
                        for col_name, var_info in variables.items():
                            if isinstance(var_info, dict):
                                var_type = var_info.get('type', 'Unsupported')
                            else:
                                var_type = getattr(var_info, 'type', 'Unsupported')
                            
                            # Map ydata-profiling types to MDM ColumnType
                            if col_name in id_columns:
                                column_types[col_name] = ColumnType.ID
                            elif col_name == target_column:
                                column_types[col_name] = ColumnType.TARGET
                            elif var_type == "Numeric":
                                column_types[col_name] = ColumnType.NUMERIC
                            elif var_type == "Categorical":
                                column_types[col_name] = ColumnType.CATEGORICAL
                            elif var_type == "DateTime":
                                column_types[col_name] = ColumnType.DATETIME
                            elif var_type in ["Text", "URL", "Path", "File"]:
                                # Check cardinality for Text columns
                                # If low cardinality, treat as categorical
                                n_unique = df[col_name].nunique() if col_name in df.columns else 0
                                threshold = self.config.feature_engineering.type_detection.categorical_threshold
                                
                                logger.info(f"Checking column {col_name}: type={var_type}, unique={n_unique}, threshold={threshold}")
                                
                                if n_unique > 0 and n_unique <= threshold:
                                    logger.info(f"Column {col_name} has {n_unique} unique values (<= {threshold}), treating as categorical")
                                    column_types[col_name] = ColumnType.CATEGORICAL
                                else:
                                    logger.info(f"Column {col_name} has {n_unique} unique values (> {threshold}), keeping as text")
                                    column_types[col_name] = ColumnType.TEXT
                            elif var_type == "Boolean":
                                # Treat boolean as categorical
                                column_types[col_name] = ColumnType.CATEGORICAL
                            else:
                                # Default to categorical for unknown types
                                column_types[col_name] = ColumnType.CATEGORICAL
                finally:
                    # Restore original TQDM setting
                    if old_tqdm is None:
                        os.environ.pop('TQDM_DISABLE', None)
                    else:
                        os.environ['TQDM_DISABLE'] = old_tqdm
                        
                logger.info(f"Detected {len(column_types)} column types for {table_type} table")
                
            except Exception as e:
                logger.warning(f"Failed to use ydata-profiling for {table_type}, falling back to simple detection: {e}")
                # Fallback to simple detection for this table
                table_column_types = self._simple_column_type_detection(
                    {table_type: column_info.get(table_type, {})}, target_column, id_columns
                )
                column_types.update(table_column_types)
        
        # If no tables were processed or no column types detected, use simple detection
        if not column_types:
            logger.warning("No column types detected, using simple detection for all tables")
            column_types = self._simple_column_type_detection(
                column_info, target_column, id_columns
            )
            
        logger.info(f"Total column types detected: {len(column_types)}")
        return column_types
    
    def _simple_column_type_detection(
        self,
        column_info: Dict[str, Dict[str, Any]],
        target_column: Optional[str],
        id_columns: List[str]
    ) -> Dict[str, ColumnType]:
        """Simple fallback column type detection.
        
        Args:
            column_info: Column information
            target_column: Target column name
            id_columns: List of ID columns
            
        Returns:
            Dictionary mapping column names to ColumnType
        """
        column_types = {}
        
        # Use the first available table
        for table_key, info in column_info.items():
            for col_name, col_type in info['columns'].items():
                dtype = str(info['dtypes'].get(col_name, 'object'))
                
                if col_name in id_columns:
                    column_types[col_name] = ColumnType.ID
                elif col_name == target_column:
                    column_types[col_name] = ColumnType.TARGET
                elif 'datetime' in str(dtype):
                    column_types[col_name] = ColumnType.DATETIME
                elif dtype == 'object':
                    # Simple heuristic for categorical vs text
                    sample_data = info.get('sample_data', {}).get(col_name, [])
                    if sample_data:
                        avg_length = sum(len(str(v)) for v in sample_data if v is not None) / len(sample_data)
                        unique_ratio = len(set(sample_data)) / len(sample_data)
                        
                        if avg_length > 50 or unique_ratio > 0.8:
                            column_types[col_name] = ColumnType.TEXT
                        else:
                            column_types[col_name] = ColumnType.CATEGORICAL
                    else:
                        column_types[col_name] = ColumnType.CATEGORICAL
                else:
                    column_types[col_name] = ColumnType.NUMERIC
            break  # Only process first table
            
        return column_types
    
    def _compute_initial_statistics(
        self,
        dataset_name: str,
        db_info: Dict[str, Any],
        table_mappings: Dict[str, str]
    ) -> Optional[Dict[str, Any]]:
        """Compute initial statistics including memory size from ydata-profiling.
        
        Args:
            dataset_name: Name of the dataset
            db_info: Database connection info
            table_mappings: Mapping of table types to table names
        """
        from rich.progress import Progress, SpinnerColumn, TextColumn
        
        backend = None
        try:
            logger.info(f"Computing initial statistics for dataset '{dataset_name}'")
            
            backend = BackendFactory.create(db_info['backend'], db_info)
            
            # Get database path/connection string
            if 'path' in db_info:
                db_path = db_info['path']
            else:
                db_path = f"{db_info['backend']}://{db_info['user']}:{db_info['password']}@{db_info['host']}:{db_info['port']}/{db_info['database']}"
            
            engine = backend.get_engine(db_path)
            
            total_rows = 0
            total_memory_bytes = 0
            
            # Use progress indicator for statistics computation
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=None,
                transient=True,
            ) as progress:
                task = progress.add_task("Computing dataset statistics...", total=None)
                
                # Focus on main data tables for statistics
                for table_type, table_name in table_mappings.items():
                    if table_type in ['train', 'test', 'validation', 'data']:
                        try:
                            progress.update(task, description=f"Analyzing {table_name}...")
                            
                            # Get row count
                            try:
                                # Try using query method if available
                                if hasattr(backend, 'query'):
                                    row_count_result = backend.query(f"SELECT COUNT(*) as count FROM {table_name}")
                                    if row_count_result is not None and not row_count_result.empty:
                                        row_count = int(row_count_result.iloc[0]['count'])
                                        total_rows += row_count
                                else:
                                    # Fallback to reading dataframe
                                    df = backend.read_table_to_dataframe(table_name, engine)
                                    row_count = len(df)
                                    total_rows += row_count
                            except Exception as e:
                                logger.debug(f"Failed to get row count: {e}")
                                row_count = 0
                            
                            # Get sample data for memory estimation
                            sample_size = min(10000, row_count) if row_count > 0 else 0
                            if sample_size > 0:
                                progress.update(task, description=f"Estimating memory usage for {table_name}...")
                                df = backend.read_table_to_dataframe(table_name, engine, limit=sample_size)
                                
                                # Use ydata-profiling to get memory size
                                try:
                                    # Suppress all output
                                    import tqdm
                                    import sys
                                    from io import StringIO
                                    
                                    original_tqdm = tqdm.tqdm
                                    original_trange = tqdm.trange
                                    
                                    def dummy_tqdm(iterable=None, *args, **kwargs):
                                        if iterable is not None:
                                            try:
                                                return iter(list(iterable))
                                            except:
                                                return iter([])
                                        return iter([])
                                    
                                    tqdm.tqdm = dummy_tqdm
                                    tqdm.trange = dummy_tqdm
                                    old_stdout = sys.stdout
                                    sys.stdout = StringIO()
                                    
                                    try:
                                        profile = ProfileReport(
                                            df, 
                                            minimal=True,
                                            progress_bar=False
                                        )
                                    finally:
                                        tqdm.tqdm = original_tqdm
                                        tqdm.trange = original_trange
                                        sys.stdout = old_stdout
                                    description = profile.get_description()
                                    
                                    # Get memory size from profile
                                    table_stats = description.table if hasattr(description, 'table') else {}
                                    memory_size = getattr(table_stats, 'memory_size', df.memory_usage(deep=True).sum())
                                    
                                    # Scale memory size to full dataset if sampling
                                    if sample_size < row_count:
                                        memory_size = int(memory_size * (row_count / sample_size))
                                    
                                    total_memory_bytes += memory_size
                                    
                                except Exception as e:
                                    logger.debug(f"Failed to use ydata-profiling for memory estimation: {e}")
                                    # Fallback to pandas memory usage
                                    memory_per_row = df.memory_usage(deep=True).sum() / len(df)
                                    total_memory_bytes += int(memory_per_row * row_count)
                                    
                        except Exception as e:
                            logger.warning(f"Failed to compute statistics for table {table_name}: {e}")
            
            # Return statistics
            statistics = {
                'row_count': total_rows,
                'memory_size_bytes': total_memory_bytes,
                'computed_at': pd.Timestamp.now().isoformat()
            }
            
            logger.info(f"Computed initial statistics: {total_rows} rows, {total_memory_bytes} bytes memory")
            return statistics
                    
        except Exception as e:
            logger.error(f"Failed to compute initial statistics: {e}")
            # Not critical - continue without statistics
            return None
        
        finally:
            if backend and hasattr(backend, 'close_connections'):
                backend.close_connections()
