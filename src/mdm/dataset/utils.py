"""Dataset utility functions."""

import hashlib
from pathlib import Path
from typing import Optional, Tuple

from mdm.core.exceptions import DatasetError


def normalize_dataset_name(name: str) -> str:
    """Normalize dataset name for case-insensitive handling.
    
    Args:
        name: Dataset name to normalize
    
    Returns:
        Normalized dataset name (lowercase)
    """
    if not name:
        raise DatasetError("Dataset name cannot be empty")

    # Convert to lowercase
    normalized = name.lower().strip()

    # Validate characters
    if not all(c.isalnum() or c in "_-" for c in normalized):
        raise DatasetError(
            "Dataset name can only contain alphanumeric characters, underscores, and dashes"
        )

    return normalized


def get_dataset_path(dataset_name: str) -> Path:
    """Get path to dataset directory.
    
    Args:
        dataset_name: Name of the dataset
    
    Returns:
        Path to dataset directory
    """
    from mdm.config import get_config_manager

    config_manager = get_config_manager()
    config = config_manager.config
    base_path = config_manager.base_path
    dataset_name = normalize_dataset_name(dataset_name)

    # First check if there's a pointer in the registry
    pointer_path = base_path / config.paths.configs_path / f"{dataset_name}.yaml"
    if pointer_path.exists():
        try:
            import yaml
            with open(pointer_path) as f:
                pointer_data = yaml.safe_load(f)
            return Path(pointer_data["path"]).expanduser().resolve()
        except Exception as e:
            raise DatasetError(f"Failed to load dataset pointer: {e}")

    # Otherwise, use default datasets directory
    return base_path / config.paths.datasets_path / dataset_name


def get_dataset_config_path(dataset_name: str) -> Path:
    """Get path to dataset YAML configuration file.
    
    Args:
        dataset_name: Name of the dataset
    
    Returns:
        Path to dataset.yaml file
    """
    dataset_path = get_dataset_path(dataset_name)
    return dataset_path / "dataset.yaml"


def validate_dataset_name(name: str) -> bool:
    """Validate dataset name format.
    
    Args:
        name: Dataset name to validate
    
    Returns:
        True if valid, False otherwise
    """
    if not name:
        return False

    # Must be alphanumeric with underscores or dashes only
    return all(c.isalnum() or c in "_-" for c in name)


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
    
    Returns:
        Human-readable size string
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def infer_file_format(file_path: Path) -> str:
    """Detect file format from extension.
    
    Args:
        file_path: Path to the file
    
    Returns:
        File format string (csv, parquet, json, etc.)
    """
    file_path = Path(file_path)
    extension = file_path.suffix.lower()

    # Map common data file extensions
    format_map = {
        ".csv": "csv",
        ".tsv": "tsv",
        ".parquet": "parquet",
        ".pq": "parquet",
        ".json": "json",
        ".jsonl": "jsonl",
        ".ndjson": "jsonl",
        ".xlsx": "excel",
        ".xls": "excel",
        ".feather": "feather",
        ".arrow": "arrow",
        ".h5": "hdf5",
        ".hdf5": "hdf5",
        ".pkl": "pickle",
        ".pickle": "pickle",
        ".gz": "gzip",
        ".zip": "zip",
        ".txt": "text",
        ".dat": "data",
    }

    # Check if compressed
    if extension == ".gz":
        # Check the extension before .gz
        stem_path = Path(file_path.stem)
        if stem_path.suffix:
            inner_format = format_map.get(stem_path.suffix.lower(), "unknown")
            return f"{inner_format}.gz"

    return format_map.get(extension, "unknown")


def get_file_encoding(file_path: Path, sample_size: int = 8192) -> str:
    """Detect file encoding.
    
    Args:
        file_path: Path to the file
        sample_size: Number of bytes to sample for detection
    
    Returns:
        Detected encoding (defaults to utf-8)
    """
    try:
        import chardet

        with open(file_path, "rb") as f:
            sample = f.read(sample_size)

        result = chardet.detect(sample)
        return result.get("encoding", "utf-8") or "utf-8"
    except ImportError:
        # If chardet is not available, default to utf-8
        return "utf-8"
    except Exception:
        return "utf-8"


def calculate_file_checksum(file_path: Path, algorithm: str = "sha256") -> str:
    """Calculate file checksum.
    
    Args:
        file_path: Path to the file
        algorithm: Hash algorithm to use (sha256, md5, etc.)
    
    Returns:
        Hexadecimal checksum string
    """
    file_path = Path(file_path).expanduser().resolve()

    if not file_path.exists():
        raise DatasetError(f"File not found: {file_path}")

    # Get hash function
    if algorithm == "sha256":
        hasher = hashlib.sha256()
    elif algorithm == "md5":
        hasher = hashlib.md5()
    elif algorithm == "sha1":
        hasher = hashlib.sha1()
    else:
        raise DatasetError(f"Unsupported hash algorithm: {algorithm}")

    # Calculate checksum
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


def dataset_exists(dataset_name: str) -> bool:
    """Check if a dataset exists in the registry.
    
    Args:
        dataset_name: Name of the dataset
    
    Returns:
        True if dataset exists, False otherwise
    """
    from mdm.config import get_config_manager

    config_manager = get_config_manager()
    config = config_manager.config
    base_path = config_manager.base_path
    dataset_name = normalize_dataset_name(dataset_name)

    # Check if pointer exists
    pointer_path = base_path / config.paths.configs_path / f"{dataset_name}.yaml"
    if pointer_path.exists():
        # Verify the pointed path exists
        try:
            import yaml
            with open(pointer_path) as f:
                pointer_data = yaml.safe_load(f)
            dataset_path = Path(pointer_data["path"]).expanduser().resolve()
            return dataset_path.exists() and dataset_path.is_dir()
        except Exception:
            return False

    # Check default location
    dataset_path = base_path / config.paths.datasets_path / dataset_name
    return dataset_path.exists() and dataset_path.is_dir()


def get_dataset_database_path(dataset_name: str) -> Optional[Path]:
    """Get path to dataset database file.
    
    Args:
        dataset_name: Name of the dataset
    
    Returns:
        Path to database file if it exists, None otherwise
    """
    dataset_path = get_dataset_path(dataset_name)

    # Check for common database files
    for db_name in ["dataset.db", "dataset.duckdb", f"{dataset_name}.db", f"{dataset_name}.duckdb"]:
        db_path = dataset_path / db_name
        if db_path.exists():
            return db_path

    return None


def parse_file_type(filename: str) -> Tuple[str, Optional[str]]:
    """Parse file type from filename.
    
    Args:
        filename: Name of the file
    
    Returns:
        Tuple of (file_type, format)
    """
    filename_lower = filename.lower()

    # Remove compression extensions
    if filename_lower.endswith(".gz"):
        filename_lower = filename_lower[:-3]
    elif filename_lower.endswith(".zip"):
        filename_lower = filename_lower[:-4]

    # Extract base name and extension
    path = Path(filename_lower)
    base_name = path.stem
    extension = path.suffix

    # Determine file type from name patterns
    if "train" in base_name:
        file_type = "train"
    elif "test" in base_name:
        file_type = "test"
    elif "valid" in base_name or "validation" in base_name:
        file_type = "validation"
    elif "submit" in base_name or "submission" in base_name:
        file_type = "submission"
    else:
        file_type = "data"

    # Get format from extension
    format_str = infer_file_format(Path(filename))

    return file_type, format_str


def is_data_file(file_path: Path) -> bool:
    """Check if a file is a data file based on extension.
    
    Args:
        file_path: Path to the file
    
    Returns:
        True if it's a recognized data file format
    """
    data_extensions = {
        ".csv", ".tsv", ".parquet", ".pq", ".json", ".jsonl", ".ndjson",
        ".xlsx", ".xls", ".feather", ".arrow", ".h5", ".hdf5",
        ".pkl", ".pickle"
    }

    file_path = Path(file_path)
    extension = file_path.suffix.lower()

    # Check if compressed data file
    if extension == ".gz":
        stem_path = Path(file_path.stem)
        if stem_path.suffix.lower() in data_extensions:
            return True

    return extension in data_extensions
