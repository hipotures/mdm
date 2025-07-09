"""Comprehensive unit tests for dataset utils to achieve 80%+ coverage."""

import pytest
from unittest.mock import Mock, patch, mock_open
from pathlib import Path
import yaml
import hashlib

from mdm.dataset.utils import (
    normalize_dataset_name,
    get_dataset_path,
    get_dataset_config_path,
    validate_dataset_name,
    format_file_size,
    infer_file_format,
    get_file_encoding,
    calculate_file_checksum,
    dataset_exists,
    get_dataset_database_path,
    parse_file_type,
    is_data_file,
)
from mdm.core.exceptions import DatasetError


class TestNormalizeDatasetName:
    """Test cases for normalize_dataset_name function."""

    def test_normalize_lowercase(self):
        """Test normalizing to lowercase."""
        assert normalize_dataset_name("TestDataset") == "testdataset"
        assert normalize_dataset_name("TEST_DATASET") == "test_dataset"

    def test_normalize_strip_whitespace(self):
        """Test stripping whitespace."""
        assert normalize_dataset_name("  dataset  ") == "dataset"
        assert normalize_dataset_name("\tdataset\n") == "dataset"

    def test_normalize_valid_characters(self):
        """Test valid characters (alphanumeric, underscore, dash)."""
        assert normalize_dataset_name("valid_dataset-123") == "valid_dataset-123"
        assert normalize_dataset_name("abc_123-xyz") == "abc_123-xyz"

    def test_normalize_empty_name(self):
        """Test error on empty name."""
        with pytest.raises(DatasetError, match="Dataset name cannot be empty"):
            normalize_dataset_name("")
        
        # Empty string after strip passes validation (all chars in empty string are valid)
        result = normalize_dataset_name("   ")
        assert result == ""

    def test_normalize_invalid_characters(self):
        """Test error on invalid characters."""
        with pytest.raises(DatasetError, match="can only contain alphanumeric"):
            normalize_dataset_name("dataset@name")
        
        with pytest.raises(DatasetError, match="can only contain alphanumeric"):
            normalize_dataset_name("dataset name")
        
        with pytest.raises(DatasetError, match="can only contain alphanumeric"):
            normalize_dataset_name("dataset.name")


class TestGetDatasetPath:
    """Test cases for get_dataset_path function."""

    @pytest.fixture
    def mock_config_manager(self):
        """Create mock config manager."""
        mock_manager = Mock()
        mock_manager.config.paths.configs_path = "config/datasets"
        mock_manager.config.paths.datasets_path = "datasets"
        mock_manager.base_path = Path("/home/user/.mdm")
        return mock_manager

    def test_get_path_with_pointer(self, mock_config_manager, tmp_path):
        """Test getting path from pointer file."""
        # Create pointer file
        pointer_path = tmp_path / "config" / "datasets" / "test_dataset.yaml"
        pointer_path.parent.mkdir(parents=True)
        
        pointer_data = {"path": "/custom/path/to/dataset"}
        with open(pointer_path, 'w') as f:
            yaml.dump(pointer_data, f)
        
        mock_config_manager.base_path = tmp_path
        
        with patch('mdm.config.get_config_manager', return_value=mock_config_manager):
            result = get_dataset_path("test_dataset")
            assert result == Path("/custom/path/to/dataset").resolve()

    def test_get_path_default(self, mock_config_manager):
        """Test getting default path when no pointer exists."""
        with patch('mdm.config.get_config_manager', return_value=mock_config_manager):
            result = get_dataset_path("test_dataset")
            assert result == Path("/home/user/.mdm/datasets/test_dataset")

    def test_get_path_invalid_pointer(self, mock_config_manager, tmp_path):
        """Test error handling for invalid pointer file."""
        # Create invalid pointer file
        pointer_path = tmp_path / "config" / "datasets" / "test_dataset.yaml"
        pointer_path.parent.mkdir(parents=True)
        pointer_path.write_text("invalid: yaml: content:")
        
        mock_config_manager.base_path = tmp_path
        
        with patch('mdm.config.get_config_manager', return_value=mock_config_manager):
            with pytest.raises(DatasetError, match="Failed to load dataset pointer"):
                get_dataset_path("test_dataset")

    def test_get_path_with_expanduser(self, mock_config_manager, tmp_path):
        """Test path expansion for home directory."""
        # Create pointer with ~ path
        pointer_path = tmp_path / "config" / "datasets" / "test_dataset.yaml"
        pointer_path.parent.mkdir(parents=True)
        
        pointer_data = {"path": "~/datasets/test_dataset"}
        with open(pointer_path, 'w') as f:
            yaml.dump(pointer_data, f)
        
        mock_config_manager.base_path = tmp_path
        
        with patch('mdm.config.get_config_manager', return_value=mock_config_manager):
            result = get_dataset_path("test_dataset")
            assert str(result).startswith(str(Path.home()))


class TestGetDatasetConfigPath:
    """Test cases for get_dataset_config_path function."""

    def test_get_config_path(self):
        """Test getting dataset config path."""
        with patch('mdm.dataset.utils.get_dataset_path', return_value=Path("/path/to/dataset")):
            result = get_dataset_config_path("test_dataset")
            assert result == Path("/path/to/dataset/dataset.yaml")


class TestValidateDatasetName:
    """Test cases for validate_dataset_name function."""

    def test_valid_names(self):
        """Test valid dataset names."""
        assert validate_dataset_name("valid_dataset") is True
        assert validate_dataset_name("dataset-123") is True
        assert validate_dataset_name("abc_123-xyz") is True
        assert validate_dataset_name("123") is True

    def test_invalid_names(self):
        """Test invalid dataset names."""
        assert validate_dataset_name("") is False
        assert validate_dataset_name("dataset name") is False
        assert validate_dataset_name("dataset@name") is False
        assert validate_dataset_name("dataset.name") is False
        assert validate_dataset_name("dataset/name") is False


class TestFormatFileSize:
    """Test cases for format_file_size function."""

    def test_bytes(self):
        """Test formatting bytes."""
        assert format_file_size(0) == "0.00 B"
        assert format_file_size(512) == "512.00 B"
        assert format_file_size(1023) == "1023.00 B"

    def test_kilobytes(self):
        """Test formatting kilobytes."""
        assert format_file_size(1024) == "1.00 KB"
        assert format_file_size(1536) == "1.50 KB"
        assert format_file_size(1024 * 1023) == "1023.00 KB"

    def test_megabytes(self):
        """Test formatting megabytes."""
        assert format_file_size(1024 * 1024) == "1.00 MB"
        assert format_file_size(1024 * 1024 * 5.5) == "5.50 MB"

    def test_gigabytes(self):
        """Test formatting gigabytes."""
        assert format_file_size(1024 * 1024 * 1024) == "1.00 GB"
        assert format_file_size(1024 * 1024 * 1024 * 2.25) == "2.25 GB"

    def test_terabytes(self):
        """Test formatting terabytes."""
        assert format_file_size(1024 * 1024 * 1024 * 1024) == "1.00 TB"

    def test_petabytes(self):
        """Test formatting petabytes."""
        assert format_file_size(1024 * 1024 * 1024 * 1024 * 1024) == "1.00 PB"
        assert format_file_size(1024 * 1024 * 1024 * 1024 * 1024 * 10) == "10.00 PB"


class TestInferFileFormat:
    """Test cases for infer_file_format function."""

    def test_common_formats(self):
        """Test common data file formats."""
        assert infer_file_format(Path("data.csv")) == "csv"
        assert infer_file_format(Path("data.tsv")) == "tsv"
        assert infer_file_format(Path("data.parquet")) == "parquet"
        assert infer_file_format(Path("data.pq")) == "parquet"
        assert infer_file_format(Path("data.json")) == "json"
        assert infer_file_format(Path("data.jsonl")) == "jsonl"
        assert infer_file_format(Path("data.xlsx")) == "excel"
        assert infer_file_format(Path("data.xls")) == "excel"

    def test_compressed_formats(self):
        """Test compressed file formats."""
        assert infer_file_format(Path("data.csv.gz")) == "csv.gz"
        assert infer_file_format(Path("data.json.gz")) == "json.gz"
        assert infer_file_format(Path("data.gz")) == "gzip"

    def test_other_formats(self):
        """Test other file formats."""
        assert infer_file_format(Path("data.feather")) == "feather"
        assert infer_file_format(Path("data.arrow")) == "arrow"
        assert infer_file_format(Path("data.h5")) == "hdf5"
        assert infer_file_format(Path("data.hdf5")) == "hdf5"
        assert infer_file_format(Path("data.pkl")) == "pickle"
        assert infer_file_format(Path("data.pickle")) == "pickle"
        assert infer_file_format(Path("data.zip")) == "zip"
        assert infer_file_format(Path("data.txt")) == "text"
        assert infer_file_format(Path("data.dat")) == "data"

    def test_unknown_format(self):
        """Test unknown file format."""
        assert infer_file_format(Path("data.xyz")) == "unknown"
        assert infer_file_format(Path("data")) == "unknown"

    def test_case_insensitive(self):
        """Test case-insensitive format detection."""
        assert infer_file_format(Path("DATA.CSV")) == "csv"
        assert infer_file_format(Path("Data.JSON")) == "json"


class TestGetFileEncoding:
    """Test cases for get_file_encoding function."""

    def test_with_chardet(self, tmp_path):
        """Test encoding detection with chardet."""
        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, world!", encoding="utf-8")
        
        # Mock the chardet module within the function scope
        mock_chardet = Mock()
        mock_chardet.detect.return_value = {"encoding": "utf-8", "confidence": 0.99}
        
        with patch.dict('sys.modules', {'chardet': mock_chardet}):
            result = get_file_encoding(test_file)
            assert result == "utf-8"
            
            # Verify chardet was called
            mock_chardet.detect.assert_called_once()

    def test_without_chardet(self, tmp_path):
        """Test encoding detection without chardet."""
        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, world!", encoding="utf-8")
        
        # Make chardet unavailable
        with patch.dict('sys.modules', {'chardet': None}):
            result = get_file_encoding(test_file)
            assert result == "utf-8"  # Default

    def test_chardet_returns_none(self, tmp_path):
        """Test when chardet returns None encoding."""
        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, world!", encoding="utf-8")
        
        # Mock the chardet module within the function scope
        mock_chardet = Mock()
        mock_chardet.detect.return_value = {"encoding": None}
        
        with patch.dict('sys.modules', {'chardet': mock_chardet}):
            result = get_file_encoding(test_file)
            assert result == "utf-8"  # Default

    def test_file_read_error(self):
        """Test handling file read error."""
        non_existent = Path("/non/existent/file.txt")
        
        result = get_file_encoding(non_existent)
        assert result == "utf-8"  # Default on error


class TestCalculateFileChecksum:
    """Test cases for calculate_file_checksum function."""

    def test_sha256_checksum(self, tmp_path):
        """Test SHA256 checksum calculation."""
        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, world!")
        
        result = calculate_file_checksum(test_file, "sha256")
        
        # Verify correct checksum
        expected = hashlib.sha256(b"Hello, world!").hexdigest()
        assert result == expected

    def test_md5_checksum(self, tmp_path):
        """Test MD5 checksum calculation."""
        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test data")
        
        result = calculate_file_checksum(test_file, "md5")
        
        # Verify correct checksum
        expected = hashlib.md5(b"Test data").hexdigest()
        assert result == expected

    def test_sha1_checksum(self, tmp_path):
        """Test SHA1 checksum calculation."""
        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test data")
        
        result = calculate_file_checksum(test_file, "sha1")
        
        # Verify correct checksum
        expected = hashlib.sha1(b"Test data").hexdigest()
        assert result == expected

    def test_file_not_found(self):
        """Test error when file not found."""
        non_existent = Path("/non/existent/file.txt")
        
        with pytest.raises(DatasetError, match="File not found"):
            calculate_file_checksum(non_existent)

    def test_unsupported_algorithm(self, tmp_path):
        """Test error for unsupported algorithm."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test data")
        
        with pytest.raises(DatasetError, match="Unsupported hash algorithm"):
            calculate_file_checksum(test_file, "sha512")

    def test_large_file_chunking(self, tmp_path):
        """Test checksum calculation for large file."""
        # Create larger test file
        test_file = tmp_path / "large.txt"
        with open(test_file, 'wb') as f:
            # Write 10KB of data
            for _ in range(10):
                f.write(b"x" * 1024)
        
        result = calculate_file_checksum(test_file, "sha256")
        
        # Verify it processes the entire file
        expected = hashlib.sha256(b"x" * 10240).hexdigest()
        assert result == expected


class TestDatasetExists:
    """Test cases for dataset_exists function."""

    @pytest.fixture
    def mock_config_manager(self):
        """Create mock config manager."""
        mock_manager = Mock()
        mock_manager.config.paths.configs_path = "config/datasets"
        mock_manager.config.paths.datasets_path = "datasets"
        mock_manager.base_path = Path("/home/user/.mdm")
        return mock_manager

    def test_exists_with_pointer(self, mock_config_manager, tmp_path):
        """Test dataset exists with pointer file."""
        # Create pointer file
        pointer_path = tmp_path / "config" / "datasets" / "test_dataset.yaml"
        pointer_path.parent.mkdir(parents=True)
        
        # Create actual dataset directory
        dataset_dir = tmp_path / "custom" / "dataset"
        dataset_dir.mkdir(parents=True)
        
        pointer_data = {"path": str(dataset_dir)}
        with open(pointer_path, 'w') as f:
            yaml.dump(pointer_data, f)
        
        mock_config_manager.base_path = tmp_path
        
        with patch('mdm.config.get_config_manager', return_value=mock_config_manager):
            assert dataset_exists("test_dataset") is True

    def test_exists_default_location(self, mock_config_manager, tmp_path):
        """Test dataset exists in default location."""
        # Create dataset directory
        dataset_dir = tmp_path / "datasets" / "test_dataset"
        dataset_dir.mkdir(parents=True)
        
        mock_config_manager.base_path = tmp_path
        
        with patch('mdm.config.get_config_manager', return_value=mock_config_manager):
            assert dataset_exists("test_dataset") is True

    def test_not_exists(self, mock_config_manager):
        """Test dataset does not exist."""
        with patch('mdm.config.get_config_manager', return_value=mock_config_manager):
            assert dataset_exists("nonexistent") is False

    def test_pointer_points_to_nonexistent(self, mock_config_manager, tmp_path):
        """Test pointer exists but points to non-existent directory."""
        # Create pointer file
        pointer_path = tmp_path / "config" / "datasets" / "test_dataset.yaml"
        pointer_path.parent.mkdir(parents=True)
        
        pointer_data = {"path": "/non/existent/path"}
        with open(pointer_path, 'w') as f:
            yaml.dump(pointer_data, f)
        
        mock_config_manager.base_path = tmp_path
        
        with patch('mdm.config.get_config_manager', return_value=mock_config_manager):
            assert dataset_exists("test_dataset") is False

    def test_invalid_pointer_file(self, mock_config_manager, tmp_path):
        """Test invalid pointer file."""
        # Create invalid pointer file
        pointer_path = tmp_path / "config" / "datasets" / "test_dataset.yaml"
        pointer_path.parent.mkdir(parents=True)
        pointer_path.write_text("invalid: yaml: content:")
        
        mock_config_manager.base_path = tmp_path
        
        with patch('mdm.config.get_config_manager', return_value=mock_config_manager):
            assert dataset_exists("test_dataset") is False


class TestGetDatasetDatabasePath:
    """Test cases for get_dataset_database_path function."""

    def test_find_dataset_db(self, tmp_path):
        """Test finding dataset.db file."""
        dataset_path = tmp_path / "dataset"
        dataset_path.mkdir()
        db_file = dataset_path / "dataset.db"
        db_file.touch()
        
        with patch('mdm.dataset.utils.get_dataset_path', return_value=dataset_path):
            result = get_dataset_database_path("test_dataset")
            assert result == db_file

    def test_find_dataset_duckdb(self, tmp_path):
        """Test finding dataset.duckdb file."""
        dataset_path = tmp_path / "dataset"
        dataset_path.mkdir()
        db_file = dataset_path / "dataset.duckdb"
        db_file.touch()
        
        with patch('mdm.dataset.utils.get_dataset_path', return_value=dataset_path):
            result = get_dataset_database_path("test_dataset")
            assert result == db_file

    def test_find_named_db(self, tmp_path):
        """Test finding {dataset_name}.db file."""
        dataset_path = tmp_path / "dataset"
        dataset_path.mkdir()
        db_file = dataset_path / "test_dataset.db"
        db_file.touch()
        
        with patch('mdm.dataset.utils.get_dataset_path', return_value=dataset_path):
            result = get_dataset_database_path("test_dataset")
            assert result == db_file

    def test_no_database_found(self, tmp_path):
        """Test when no database file found."""
        dataset_path = tmp_path / "dataset"
        dataset_path.mkdir()
        
        with patch('mdm.dataset.utils.get_dataset_path', return_value=dataset_path):
            result = get_dataset_database_path("test_dataset")
            assert result is None


class TestParseFileType:
    """Test cases for parse_file_type function."""

    def test_train_file(self):
        """Test parsing train file."""
        file_type, format_str = parse_file_type("train.csv")
        assert file_type == "train"
        assert format_str == "csv"
        
        file_type, format_str = parse_file_type("training_data.parquet")
        assert file_type == "train"
        assert format_str == "parquet"

    def test_test_file(self):
        """Test parsing test file."""
        file_type, format_str = parse_file_type("test.csv")
        assert file_type == "test"
        assert format_str == "csv"
        
        file_type, format_str = parse_file_type("test_set.json")
        assert file_type == "test"
        assert format_str == "json"

    def test_validation_file(self):
        """Test parsing validation file."""
        file_type, format_str = parse_file_type("valid.csv")
        assert file_type == "validation"
        assert format_str == "csv"
        
        file_type, format_str = parse_file_type("validation_data.parquet")
        assert file_type == "validation"
        assert format_str == "parquet"

    def test_submission_file(self):
        """Test parsing submission file."""
        file_type, format_str = parse_file_type("submit.csv")
        assert file_type == "submission"
        assert format_str == "csv"
        
        file_type, format_str = parse_file_type("submission_template.csv")
        assert file_type == "submission"
        assert format_str == "csv"

    def test_generic_data_file(self):
        """Test parsing generic data file."""
        file_type, format_str = parse_file_type("data.csv")
        assert file_type == "data"
        assert format_str == "csv"
        
        file_type, format_str = parse_file_type("sales_2023.xlsx")
        assert file_type == "data"
        assert format_str == "excel"

    def test_compressed_files(self):
        """Test parsing compressed files."""
        file_type, format_str = parse_file_type("train.csv.gz")
        assert file_type == "train"
        assert format_str == "csv.gz"
        
        file_type, format_str = parse_file_type("test.json.gz")
        assert file_type == "test"
        assert format_str == "json.gz"
        
        file_type, format_str = parse_file_type("data.zip")
        assert file_type == "data"
        assert format_str == "zip"


class TestIsDataFile:
    """Test cases for is_data_file function."""

    def test_data_files(self):
        """Test recognized data file formats."""
        assert is_data_file(Path("data.csv")) is True
        assert is_data_file(Path("data.tsv")) is True
        assert is_data_file(Path("data.parquet")) is True
        assert is_data_file(Path("data.pq")) is True
        assert is_data_file(Path("data.json")) is True
        assert is_data_file(Path("data.jsonl")) is True
        assert is_data_file(Path("data.xlsx")) is True
        assert is_data_file(Path("data.xls")) is True
        assert is_data_file(Path("data.feather")) is True
        assert is_data_file(Path("data.arrow")) is True
        assert is_data_file(Path("data.h5")) is True
        assert is_data_file(Path("data.hdf5")) is True
        assert is_data_file(Path("data.pkl")) is True
        assert is_data_file(Path("data.pickle")) is True

    def test_compressed_data_files(self):
        """Test compressed data files."""
        assert is_data_file(Path("data.csv.gz")) is True
        assert is_data_file(Path("data.json.gz")) is True
        assert is_data_file(Path("data.parquet.gz")) is True

    def test_non_data_files(self):
        """Test non-data files."""
        assert is_data_file(Path("README.md")) is False
        assert is_data_file(Path("script.py")) is False
        assert is_data_file(Path("config.yaml")) is False
        assert is_data_file(Path("image.png")) is False
        assert is_data_file(Path("data.txt")) is False
        assert is_data_file(Path("data.gz")) is False  # Just .gz without data extension

    def test_case_insensitive(self):
        """Test case-insensitive detection."""
        assert is_data_file(Path("DATA.CSV")) is True
        assert is_data_file(Path("Data.JSON")) is True
        assert is_data_file(Path("TEST.PARQUET")) is True