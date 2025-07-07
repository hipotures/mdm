"""Dataset registrar implementing the 12-step registration process."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
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

logger = logging.getLogger(__name__)


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

        # Step 7: Load data files
        table_mappings = self._load_data_files(files, db_info)

        # Step 8: Detect columns and types
        column_info = self._analyze_columns(db_info, table_mappings)

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
                target_column, id_columns, kwargs.get('type_schema')
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
               if k not in ['display_name', 'target_column', 'problem_type']}
        )

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
        db_info: Dict[str, Any]
    ) -> Dict[str, str]:
        """Step 7: Load data files into database."""
        backend = BackendFactory.create(db_info['backend'], db_info)
        table_mappings = {}

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
                if file_key == 'sample_submission':
                    continue  # Skip sample submission

                table_name = file_key
                logger.info(f"Loading {file_path} as table '{table_name}'")

                # Read and load based on file type
                if file_path.suffix.lower() in ['.csv', '.tsv']:
                    delimiter = detect_delimiter(file_path)
                    df = pd.read_csv(file_path, delimiter=delimiter, parse_dates=True)
                elif file_path.suffix.lower() == '.parquet':
                    df = pd.read_parquet(file_path)
                elif file_path.suffix.lower() == '.json':
                    df = pd.read_json(file_path)
                else:
                    logger.warning(f"Unsupported file type: {file_path}")
                    continue

                # Log column information
                logger.debug(f"Columns in {table_name}: {list(df.columns)}")
                logger.debug(f"Data types: {df.dtypes.to_dict()}")
                
                # Load into database
                backend.create_table_from_dataframe(df, table_name, engine, if_exists='replace')
                table_mappings[file_key] = table_name

                logger.info(f"Loaded {len(df)} rows into '{table_name}'")

        except Exception as e:
            raise DatasetError(f"Failed to load data files: {e}")
        finally:
            backend.close_connections()

        return table_mappings

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
        type_schema: Optional[Dict[str, str]] = None
    ) -> Dict[str, str]:
        """Generate feature tables for the dataset.
        
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
                id_columns=id_columns
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
                profile = ProfileReport(
                    df,
                    minimal=True,
                    type_schema=type_schema
                )
                
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
