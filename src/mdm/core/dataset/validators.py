"""Dataset validation utilities.

Provides validators for dataset names, paths, and structure detection.
"""
import re
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd

from ...core.exceptions import DatasetError

logger = logging.getLogger(__name__)


class DatasetNameValidator:
    """Validates dataset names according to MDM conventions."""
    
    # Valid name pattern: alphanumeric, underscores, hyphens
    NAME_PATTERN = re.compile(r'^[a-zA-Z0-9_-]+$')
    MAX_LENGTH = 100
    RESERVED_NAMES = {'config', 'system', 'metadata', 'temp', 'tmp'}
    
    @classmethod
    def validate(cls, name: str) -> str:
        """Validate and normalize dataset name.
        
        Args:
            name: Dataset name to validate
            
        Returns:
            Normalized dataset name
            
        Raises:
            DatasetError: If name is invalid
        """
        if not name:
            raise DatasetError("Dataset name cannot be empty")
        
        # Strip whitespace
        name = name.strip()
        
        # Check length
        if len(name) > cls.MAX_LENGTH:
            raise DatasetError(f"Dataset name too long (max {cls.MAX_LENGTH} chars)")
        
        # Check pattern
        if not cls.NAME_PATTERN.match(name):
            raise DatasetError(
                f"Invalid dataset name '{name}'. "
                "Only letters, numbers, underscores, and hyphens allowed."
            )
        
        # Check reserved names
        if name.lower() in cls.RESERVED_NAMES:
            raise DatasetError(f"'{name}' is a reserved name")
        
        # Normalize to lowercase
        normalized = name.lower()
        
        logger.debug(f"Validated dataset name: {name} -> {normalized}")
        return normalized


class DatasetPathValidator:
    """Validates dataset paths and file formats."""
    
    SUPPORTED_EXTENSIONS = {
        '.csv', '.tsv', '.txt',  # Text formats
        '.parquet', '.pq',       # Parquet
        '.xlsx', '.xls',         # Excel
        '.json', '.jsonl',       # JSON
        '.feather',              # Feather
        '.pkl', '.pickle',       # Pickle
    }
    
    COMPRESSED_EXTENSIONS = {'.gz', '.zip', '.bz2', '.xz'}
    
    @classmethod
    def validate(cls, path: Path) -> Path:
        """Validate dataset path.
        
        Args:
            path: Path to validate
            
        Returns:
            Resolved absolute path
            
        Raises:
            DatasetError: If path is invalid
        """
        # Ensure Path object
        if not isinstance(path, Path):
            path = Path(path)
        
        # Resolve to absolute path
        path = path.resolve()
        
        # Check existence
        if not path.exists():
            raise DatasetError(f"Path does not exist: {path}")
        
        # Check if it's a file or directory
        if path.is_file():
            # Validate file extension
            cls._validate_file_extension(path)
        elif not path.is_dir():
            raise DatasetError(f"Path is neither file nor directory: {path}")
        
        logger.debug(f"Validated dataset path: {path}")
        return path
    
    @classmethod
    def _validate_file_extension(cls, path: Path) -> None:
        """Validate file extension."""
        suffixes = path.suffixes
        
        if not suffixes:
            raise DatasetError(f"File has no extension: {path}")
        
        # Check for compressed files
        if suffixes[-1] in cls.COMPRESSED_EXTENSIONS:
            if len(suffixes) < 2:
                raise DatasetError(f"Compressed file has no data extension: {path}")
            data_ext = suffixes[-2]
        else:
            data_ext = suffixes[-1]
        
        if data_ext not in cls.SUPPORTED_EXTENSIONS:
            raise DatasetError(
                f"Unsupported file format: {data_ext}. "
                f"Supported: {', '.join(sorted(cls.SUPPORTED_EXTENSIONS))}"
            )
    
    @classmethod
    def detect_format(cls, path: Path) -> Tuple[str, Optional[str]]:
        """Detect file format and compression.
        
        Args:
            path: File path
            
        Returns:
            Tuple of (format, compression)
        """
        suffixes = path.suffixes
        
        compression = None
        if suffixes and suffixes[-1] in cls.COMPRESSED_EXTENSIONS:
            compression = suffixes[-1][1:]  # Remove dot
            format_ext = suffixes[-2] if len(suffixes) > 1 else None
        else:
            format_ext = suffixes[-1] if suffixes else None
        
        # Map extension to format
        format_map = {
            '.csv': 'csv',
            '.tsv': 'tsv',
            '.txt': 'csv',  # Assume CSV for .txt
            '.parquet': 'parquet',
            '.pq': 'parquet',
            '.xlsx': 'excel',
            '.xls': 'excel',
            '.json': 'json',
            '.jsonl': 'jsonl',
            '.feather': 'feather',
            '.pkl': 'pickle',
            '.pickle': 'pickle',
        }
        
        file_format = format_map.get(format_ext, 'unknown')
        
        return file_format, compression


class DatasetStructureDetector:
    """Detects dataset structure and metadata."""
    
    KAGGLE_FILES = {'train.csv', 'test.csv', 'sample_submission.csv'}
    COMMON_ID_PATTERNS = [
        r'^id$', r'^ID$', r'^Id$',
        r'.*_id$', r'.*_ID$', r'.*Id$',
        r'^idx$', r'^index$', r'^pk$',
        r'^row_?num$', r'^row_?id$',
    ]
    COMMON_TARGET_PATTERNS = [
        r'^target$', r'^label$', r'^class$',
        r'^y$', r'^Y$', r'^outcome$',
        r'^prediction$', r'^result$',
    ]
    
    @classmethod
    def detect_structure(cls, path: Path) -> Dict[str, Any]:
        """Detect dataset structure.
        
        Args:
            path: Dataset path
            
        Returns:
            Dictionary with structure information
        """
        structure = {
            'type': 'unknown',
            'files': {},
            'detected_features': {}
        }
        
        if path.is_file():
            structure['type'] = 'single_file'
            structure['files']['data'] = path
        elif path.is_dir():
            # Check for Kaggle structure
            if cls._is_kaggle_structure(path):
                structure['type'] = 'kaggle'
                structure['files'] = cls._get_kaggle_files(path)
                structure['detected_features'] = cls._detect_kaggle_features(path)
            else:
                # General directory structure
                structure['type'] = 'directory'
                structure['files'] = cls._discover_data_files(path)
        
        logger.info(f"Detected structure type: {structure['type']}")
        return structure
    
    @classmethod
    def _is_kaggle_structure(cls, path: Path) -> bool:
        """Check if directory has Kaggle competition structure."""
        files = {f.name for f in path.iterdir() if f.is_file()}
        return bool(cls.KAGGLE_FILES.intersection(files))
    
    @classmethod
    def _get_kaggle_files(cls, path: Path) -> Dict[str, Path]:
        """Get Kaggle competition files."""
        files = {}
        
        for filename in cls.KAGGLE_FILES:
            filepath = path / filename
            if filepath.exists():
                if filename == 'train.csv':
                    files['train'] = filepath
                elif filename == 'test.csv':
                    files['test'] = filepath
                elif filename == 'sample_submission.csv':
                    files['submission'] = filepath
        
        return files
    
    @classmethod
    def _detect_kaggle_features(cls, path: Path) -> Dict[str, Any]:
        """Detect features from Kaggle structure."""
        features = {}
        
        # Try to detect target from sample submission
        submission_path = path / 'sample_submission.csv'
        if submission_path.exists():
            try:
                df = pd.read_csv(submission_path, nrows=5)
                # First column is usually ID, second is target
                if len(df.columns) >= 2:
                    features['id_column'] = df.columns[0]
                    features['target_column'] = df.columns[1]
                    logger.info(f"Detected from submission: ID={features['id_column']}, target={features['target_column']}")
            except Exception as e:
                logger.warning(f"Failed to read sample submission: {e}")
        
        return features
    
    @classmethod
    def _discover_data_files(cls, path: Path) -> Dict[str, Path]:
        """Discover all data files in directory."""
        files = {}
        validator = DatasetPathValidator()
        
        # Look for data files
        for filepath in path.rglob('*'):
            if filepath.is_file():
                try:
                    # Check if it's a supported format
                    file_format, _ = validator.detect_format(filepath)
                    if file_format != 'unknown':
                        # Use relative path as key
                        rel_path = filepath.relative_to(path)
                        key = str(rel_path).replace('/', '_').replace('.', '_')
                        files[key] = filepath
                except Exception:
                    continue
        
        return files
    
    @classmethod
    def detect_id_columns(cls, df: pd.DataFrame, column_types: Dict[str, str]) -> List[str]:
        """Detect potential ID columns.
        
        Args:
            df: DataFrame to analyze
            column_types: Column type mapping
            
        Returns:
            List of potential ID column names
        """
        id_columns = []
        
        for col in df.columns:
            # Check name patterns
            for pattern in cls.COMMON_ID_PATTERNS:
                if re.match(pattern, col, re.IGNORECASE):
                    id_columns.append(col)
                    break
            
            # Check if column has unique values
            if col not in id_columns and column_types.get(col) in ['integer', 'string']:
                if df[col].nunique() == len(df):
                    id_columns.append(col)
        
        return id_columns
    
    @classmethod
    def detect_target_column(cls, df: pd.DataFrame, columns: List[str]) -> Optional[str]:
        """Detect potential target column.
        
        Args:
            df: DataFrame to analyze
            columns: List of column names
            
        Returns:
            Detected target column name or None
        """
        # Check name patterns
        for col in columns:
            for pattern in cls.COMMON_TARGET_PATTERNS:
                if re.match(pattern, col, re.IGNORECASE):
                    return col
        
        # Check last column (common convention)
        if columns:
            last_col = columns[-1]
            # Check if it's not an obvious ID column
            is_id = any(re.match(p, last_col, re.IGNORECASE) for p in cls.COMMON_ID_PATTERNS)
            if not is_id:
                return last_col
        
        return None
    
    @classmethod
    def infer_problem_type(cls, df: pd.DataFrame, target_column: str) -> str:
        """Infer ML problem type from target column.
        
        Args:
            df: DataFrame with target column
            target_column: Name of target column
            
        Returns:
            Problem type: 'regression', 'binary_classification', or 'multiclass_classification'
        """
        if target_column not in df.columns:
            return 'unknown'
        
        target = df[target_column]
        
        # Check if numeric
        if pd.api.types.is_numeric_dtype(target):
            n_unique = target.nunique()
            
            # Binary classification
            if n_unique == 2:
                return 'binary_classification'
            
            # Multiclass vs regression
            if n_unique < 20 and n_unique / len(df) < 0.05:
                return 'multiclass_classification'
            else:
                return 'regression'
        else:
            # String/object type
            n_unique = target.nunique()
            
            if n_unique == 2:
                return 'binary_classification'
            elif n_unique < 100:
                return 'multiclass_classification'
            else:
                return 'unknown'
