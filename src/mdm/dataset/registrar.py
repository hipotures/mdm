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

        # Step 1: Validate dataset name
        normalized_name = self._validate_name(name)

        # Step 2: Check if dataset already exists
        if self.manager.dataset_exists(normalized_name):
            if kwargs.get('force', False):
                logger.info(f"Dataset '{normalized_name}' exists, removing due to --force flag")
                # Remove existing dataset
                from mdm.dataset.operations import RemoveOperation
                remove_op = RemoveOperation()
                try:
                    remove_op.execute(normalized_name, force=True, dry_run=False)
                except Exception as e:
                    logger.warning(f"Failed to remove existing dataset: {e}")
            else:
                raise DatasetError(f"Dataset '{normalized_name}' already exists")

        # Step 3: Validate path
        path = self._validate_path(path)

        # Step 4: Auto-detect or manual mode
        if auto_detect:
            detected_info = self._auto_detect(path)
        else:
            detected_info = {}

        # Step 5: Discover data files
        files = self._discover_files(path, detected_info)

        # Step 6: Create database
        db_info = self._create_database(normalized_name)

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
            task = progress.add_task("Analyzing columns and data types...", total=None)
            column_info = self._analyze_columns(db_info, table_mappings)
            progress.update(task, completed=True, visible=False)

            # Step 9: Detect ID columns
            id_columns = detected_info.get('id_columns', [])
            if not id_columns and auto_detect:
                id_columns = self._detect_id_columns(column_info)

            # Step 10: Determine problem type and target
            target_column = detected_info.get('target_column') or kwargs.get('target_column')
            problem_type = detected_info.get('problem_type') or kwargs.get('problem_type')

            if auto_detect and not problem_type and target_column:
                problem_type = self._infer_problem_type(column_info, target_column)

            # Step 10.5: Generate feature tables
            feature_tables = {}
            if kwargs.get('generate_features', True):
                feature_tables = self._generate_features(
                    normalized_name, db_info, table_mappings, column_info,
                    target_column, id_columns, kwargs.get('type_schema'),
                    progress
                )
                table_mappings.update(feature_tables)

        # Step 11: Create dataset info
        dataset_info = DatasetInfo(
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

        # Step 11.5: Compute initial statistics including memory size
        statistics = self._compute_initial_statistics(normalized_name, db_info, table_mappings)
        if statistics:
            dataset_info.metadata['statistics'] = statistics

        # Step 12: Save registration
        self.manager.register_dataset(dataset_info)

        logger.info(f"Dataset '{normalized_name}' registered successfully")
        return dataset_info

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
                test_df = pd.read_csv(files['test'], nrows=5)
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

    def _load_data_files(
        self,
        files: Dict[str, Path],
        db_info: Dict[str, Any],
        progress: Optional[Progress] = None
    ) -> Dict[str, str]:
        """Step 7: Load data files into database with true batch processing.
        
        This implementation loads data in chunks to save memory. Each chunk is:
        1. Loaded from file
        2. Saved to database
        3. Released from memory
        
        Column type detection happens on the first chunk only.
        Feature generation happens later on the complete dataset in the database.
        """
        backend = BackendFactory.create(db_info['backend'], db_info)
        table_mappings = {}
        
        # Store column types detected from first chunk for later use
        self._detected_column_types = {}

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

            for file_key, file_path in files.items():
                table_name = file_key
                logger.info(f"Loading {file_path} as table '{table_name}'")

                # Read and load based on file type with batch processing
                batch_size = self.config.performance.batch_size
                
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
                    if file_path.suffix.lower() in ['.csv', '.tsv']:
                        delimiter = detect_delimiter(file_path)
                        
                        # First, get total row count for progress bar
                        total_rows = sum(1 for _ in open(file_path)) - 1  # Subtract header
                        
                        # Create progress task
                        task = progress.add_task(
                            f"Loading {file_path.name} into {table_name}",
                            total=total_rows
                        )
                        
                        # Read in chunks
                        first_chunk = True
                        chunk_count = 0
                        for chunk_df in pd.read_csv(
                            file_path, 
                            delimiter=delimiter, 
                            parse_dates=True,
                            chunksize=batch_size
                        ):
                            chunk_count += 1
                            logger.debug(f"Processing chunk {chunk_count} with {len(chunk_df)} rows")
                            
                            if first_chunk:
                                # Log column information from first chunk
                                logger.debug(f"Columns in {table_name}: {list(chunk_df.columns)}")
                                logger.debug(f"Data types: {chunk_df.dtypes.to_dict()}")
                                
                                # Detect column types on first chunk (for later feature generation)
                                if table_name == 'train':  # Focus on train table for type detection
                                    self._detect_and_store_column_types(chunk_df, table_name)
                                
                                # Create table with first chunk
                                backend.create_table_from_dataframe(
                                    chunk_df, table_name, engine, if_exists='replace'
                                )
                                first_chunk = False
                            else:
                                # Append subsequent chunks
                                backend.create_table_from_dataframe(
                                    chunk_df, table_name, engine, if_exists='append'
                                )
                            
                            # Update progress
                            progress.update(task, advance=len(chunk_df))
                            
                            # Explicitly free memory
                            del chunk_df
                            
                    elif file_path.suffix.lower() == '.parquet':
                        # Parquet files can be read in batches too
                        df = pd.read_parquet(file_path)
                        total_rows = len(df)
                        
                        task = progress.add_task(
                            f"Loading {file_path.name} into {table_name}",
                            total=total_rows
                        )
                        
                        # Detect column types on first batch
                        if table_name == 'train' and len(df) > 0:
                            first_batch = df.iloc[:min(batch_size, len(df))]
                            self._detect_and_store_column_types(first_batch, table_name)
                        
                        # Process in batches
                        for i in range(0, total_rows, batch_size):
                            batch_df = df.iloc[i:i + batch_size]
                            batch_num = (i // batch_size) + 1
                            logger.debug(f"Processing batch {batch_num} (rows {i}:{i+len(batch_df)}) for Parquet file")
                            
                            if i == 0:
                                logger.debug(f"Columns in {table_name}: {list(batch_df.columns)}")
                                logger.debug(f"Data types: {batch_df.dtypes.to_dict()}")
                                backend.create_table_from_dataframe(
                                    batch_df, table_name, engine, if_exists='replace'
                                )
                            else:
                                backend.create_table_from_dataframe(
                                    batch_df, table_name, engine, if_exists='append'
                                )
                            
                            progress.update(task, advance=len(batch_df))
                            
                    elif file_path.suffix.lower() == '.json':
                        # JSON files typically need to be loaded fully
                        df = pd.read_json(file_path)
                        total_rows = len(df)
                        
                        task = progress.add_task(
                            f"Loading {file_path.name} into {table_name}",
                            total=total_rows
                        )
                        
                        logger.debug(f"Columns in {table_name}: {list(df.columns)}")
                        logger.debug(f"Data types: {df.dtypes.to_dict()}")
                        
                        # Detect column types on first batch
                        if table_name == 'train' and len(df) > 0:
                            first_batch = df.iloc[:min(batch_size, len(df))]
                            self._detect_and_store_column_types(first_batch, table_name)
                        
                        # Process in batches
                        for i in range(0, total_rows, batch_size):
                            batch_df = df.iloc[i:i + batch_size]
                            
                            if i == 0:
                                backend.create_table_from_dataframe(
                                    batch_df, table_name, engine, if_exists='replace'
                                )
                            else:
                                backend.create_table_from_dataframe(
                                    batch_df, table_name, engine, if_exists='append'
                                )
                            
                            progress.update(task, advance=len(batch_df))
                            
                    else:
                        logger.warning(f"Unsupported file type: {file_path}")
                        continue
                    
                    table_mappings[file_key] = table_name
                    
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
                
                finally:
                    if own_progress:
                        progress_ctx.__exit__(None, None, None)

        except Exception as e:
            raise DatasetError(f"Failed to load data files: {e}")
        finally:
            backend.close_connections()

        return table_mappings
    
    def _detect_and_store_column_types(self, df: pd.DataFrame, table_name: str) -> None:
        """Detect column types from first chunk using ydata-profiling.
        
        This runs on the first chunk only to save memory.
        Results are stored for later use in feature generation.
        """
        logger.info(f"Detecting column types from first chunk of {table_name}")
        
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
                progress=progress
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
        
        # Focus on the main training table
        if 'train' in table_mappings:
            table_name = table_mappings['train']
            
            # Read sample data for profiling
            backend = BackendFactory.create(engine.url.drivername, {})
            df = backend.read_table_to_dataframe(table_name, engine, limit=10000)
            
            # Create minimal profile for type detection
            try:
                # Don't create a new progress if we already have one active
                logger.info("Analyzing column types with ydata-profiling...")
                
                # Suppress tqdm progress bars from ydata-profiling
                import os
                old_tqdm = os.environ.get('TQDM_DISABLE')
                os.environ['TQDM_DISABLE'] = '1'
                
                try:
                    # Use stored column types if available
                    if hasattr(self, '_detected_column_types') and self._detected_column_types:
                        # We already have column types from first chunk
                        logger.info("Using column types detected during data loading")
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
                        
                logger.info(f"Detected column types using ydata-profiling: {column_types}")
                
            except Exception as e:
                logger.warning(f"Failed to use ydata-profiling, falling back to simple detection: {e}")
                # Fallback to simple detection
                column_types = self._simple_column_type_detection(
                    column_info, target_column, id_columns
                )
        else:
            # No train table, use simple detection
            column_types = self._simple_column_type_detection(
                column_info, target_column, id_columns
            )
            
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
            if hasattr(backend, 'close_connections'):
                backend.close_connections()
