"""Auto-detection module for dataset structures and metadata."""

import csv
import logging
import re
from pathlib import Path
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)


def detect_kaggle_structure(path: Path) -> bool:
    """Detect if the directory follows Kaggle competition structure.
    
    Args:
        path: Directory path to check
        
    Returns:
        True if Kaggle structure is detected
    """
    # Check for common Kaggle files
    kaggle_indicators = [
        'sample_submission.csv',
        'train.csv',
        'test.csv',
        'data_description.txt',
        'submission.csv'
    ]

    found_indicators = sum(1 for file in kaggle_indicators if (path / file).exists())

    # Consider it Kaggle if we find at least 2 indicators, including sample_submission
    has_sample_submission = (path / 'sample_submission.csv').exists()
    return has_sample_submission and found_indicators >= 2


def extract_target_from_sample_submission(path: Path) -> Optional[str]:
    """Extract target column name from sample_submission.csv.
    
    Args:
        path: Path to sample_submission.csv
        
    Returns:
        Target column name or None if not found
    """
    if not path.exists():
        return None

    try:
        with open(path, encoding='utf-8') as f:
            reader = csv.reader(f)
            headers = next(reader, None)

            if headers and len(headers) >= 2:
                # Target is typically the last column that's not an ID
                # Filter out common ID patterns
                non_id_columns = [
                    col for col in headers
                    if not is_id_column(col)
                ]

                if non_id_columns:
                    # Return the last non-ID column
                    return non_id_columns[-1]
                if len(headers) == 2:
                    # If only 2 columns, second is likely the target
                    return headers[1]

    except Exception as e:
        logger.warning(f"Failed to read sample_submission.csv: {e}")

    return None


def detect_id_columns(df_sample: dict[str, Any], column_names: list[str]) -> list[str]:
    """Detect potential ID columns from dataframe sample.
    
    Args:
        df_sample: Sample of dataframe data (first few rows)
        column_names: List of column names
        
    Returns:
        List of detected ID column names
    """
    id_columns = []

    for col in column_names:
        if is_id_column(col):
            id_columns.append(col)
            continue

        # Check data patterns if column name doesn't indicate ID
        if col in df_sample:
            values = df_sample[col]
            if is_id_pattern(values):
                id_columns.append(col)

    return id_columns


def is_id_column(column_name: str) -> bool:
    """Check if column name indicates an ID column.
    
    Args:
        column_name: Column name to check
        
    Returns:
        True if column name matches ID patterns
    """
    col_lower = column_name.lower()

    # Common ID patterns
    id_patterns = [
        r'^id$',
        r'^.*_id$',
        r'^id_.*',
        r'^.*id$',
        r'^index$',
        r'^idx$',
        r'^.*_idx$',
        r'^row_?id$',
        r'^record_?id$',
        r'^customer_?id$',
        r'^user_?id$',
        r'^order_?id$',
        r'^transaction_?id$',
        r'^session_?id$',
        r'^patient_?id$',
        r'^product_?id$',
        r'^item_?id$',
        r'^key$',
        r'^.*_key$',
        r'^pk$',
        r'^.*_pk$'
    ]

    return any(re.match(pattern, col_lower) for pattern in id_patterns)


def is_id_pattern(values: list[Any]) -> bool:
    """Check if values follow ID-like patterns.
    
    Args:
        values: List of values to check
        
    Returns:
        True if values appear to be IDs
    """
    if not values or len(values) < 2:
        return False

    # Sample up to 100 values
    sample_size = min(100, len(values))
    sample_values = values[:sample_size]

    # Check if all values are unique (strong ID indicator)
    if len(set(sample_values)) != len(sample_values):
        return False

    # Check for common ID patterns
    string_values = [str(v) for v in sample_values if v is not None]
    if not string_values:
        return False

    # Pattern checks
    patterns = {
        'sequential_numeric': all(str(v).isdigit() for v in string_values),
        'uuid_like': all(len(v) > 20 and '-' in v for v in string_values),
        'alphanumeric': all(any(c.isalpha() for c in v) and any(c.isdigit() for c in v)
                           for v in string_values),
        'consistent_length': len(set(len(v) for v in string_values)) == 1
    }

    # If values are sequential integers
    if patterns['sequential_numeric']:
        try:
            int_values = sorted([int(v) for v in string_values])
            # Check if roughly sequential
            diffs = [int_values[i+1] - int_values[i] for i in range(len(int_values)-1)]
            if all(d > 0 and d < 100 for d in diffs):  # Reasonably sequential
                return True
        except:
            pass

    # UUID-like or consistent alphanumeric patterns
    return patterns['uuid_like'] or (patterns['alphanumeric'] and patterns['consistent_length'])


def infer_problem_type(
    target_column: Optional[str],
    target_values: Optional[list[Any]] = None,
    n_unique: Optional[int] = None
) -> Optional[str]:
    """Infer ML problem type from target column.
    
    Args:
        target_column: Name of target column
        target_values: Sample of target values
        n_unique: Number of unique values in target
        
    Returns:
        Problem type: 'binary_classification', 'multiclass_classification', 
        'regression', or None
    """
    if not target_column:
        return None

    # Check column name hints
    col_lower = target_column.lower()

    # Regression indicators
    regression_patterns = [
        'price', 'cost', 'amount', 'value', 'score', 'rating',
        'count', 'quantity', 'sales', 'revenue', 'profit',
        'temperature', 'weight', 'height', 'age', 'salary'
    ]
    if any(pattern in col_lower for pattern in regression_patterns):
        return 'regression'

    # Classification indicators
    classification_patterns = [
        'class', 'category', 'type', 'label', 'target',
        'is_', 'has_', 'flag', 'binary', 'status'
    ]

    # If we have values to inspect
    if target_values and n_unique is not None:
        # Binary classification
        if n_unique == 2:
            return 'binary_classification'

        # Likely classification if few unique values
        if n_unique < 20 and n_unique < len(target_values) * 0.1:
            return 'multiclass_classification'

        # Check if values are categorical
        sample_values = target_values[:100]
        if all(isinstance(v, (str, bool)) for v in sample_values if v is not None):
            if n_unique <= 20:
                return 'multiclass_classification'
            return None  # Too many categories

        # Numeric values - check if they're actually classes
        if all(isinstance(v, (int, float)) for v in sample_values if v is not None):
            numeric_values = [v for v in sample_values if v is not None]
            unique_nums = set(numeric_values)

            # Integer values 0,1 or close to that
            if unique_nums == {0, 1} or unique_nums == {0.0, 1.0}:
                return 'binary_classification'

            # Small set of integers (likely class labels)
            if all(isinstance(v, int) or v.is_integer() for v in numeric_values):
                if len(unique_nums) <= 20:
                    return 'multiclass_classification'

            # Otherwise, likely regression
            return 'regression'

    # Fall back to name patterns
    if any(pattern in col_lower for pattern in classification_patterns):
        return 'multiclass_classification'

    return None


def discover_data_files(path: Path, extensions: Optional[list[str]] = None) -> dict[str, Path]:
    """Discover data files in a directory.
    
    Args:
        path: Directory path to search
        extensions: List of file extensions to look for (default: csv, parquet, json)
        
    Returns:
        Dictionary mapping file type to path
    """
    if extensions is None:
        extensions = ['.csv', '.parquet', '.json', '.tsv', '.xlsx', '.xls']

    files = {}

    # Look for standard file names first
    standard_names = ['train', 'test', 'validation', 'val', 'dev',
                      'training', 'testing', 'sample_submission']

    for name in standard_names:
        for ext in extensions:
            file_path = path / f"{name}{ext}"
            if file_path.exists():
                # Normalize the key name
                key = name
                if name == 'training':
                    key = 'train'
                elif name == 'testing':
                    key = 'test'
                elif name in ['val', 'dev']:
                    key = 'validation'
                elif name == 'sample_submission':
                    key = 'sample_submission'

                files[key] = file_path

    # If no standard files found, get all data files
    if not files:
        for ext in extensions:
            for file_path in path.glob(f"*{ext}"):
                if file_path.is_file():
                    files[file_path.stem] = file_path

    return files


def validate_kaggle_submission_format(
    test_columns: list[str],
    sample_submission_path: Path
) -> Tuple[bool, Optional[str]]:
    """Validate that test data matches sample submission format.
    
    Args:
        test_columns: Columns in test dataset
        sample_submission_path: Path to sample submission file
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not sample_submission_path.exists():
        return True, None  # Can't validate without sample

    try:
        with open(sample_submission_path, encoding='utf-8') as f:
            reader = csv.reader(f)
            submission_columns = next(reader, None)

        if not submission_columns:
            return False, "Sample submission file is empty"

        # Check if submission ID column exists in test
        submission_id = submission_columns[0] if submission_columns else None
        if submission_id and submission_id not in test_columns:
            return False, f"Submission ID column '{submission_id}' not found in test data"

        return True, None

    except Exception as e:
        return False, f"Failed to validate submission format: {e}"


def detect_delimiter(file_path: Path) -> str:
    """Detect delimiter for CSV-like files.
    
    Args:
        file_path: Path to file
        
    Returns:
        Detected delimiter (default: ',')
    """
    if file_path.suffix.lower() == '.tsv':
        return '\t'

    try:
        with open(file_path, encoding='utf-8') as f:
            sample = f.read(1024)
            sniffer = csv.Sniffer()
            delimiter = sniffer.sniff(sample).delimiter
            return delimiter
    except:
        return ','
