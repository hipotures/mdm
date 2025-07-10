"""Comprehensive unit tests for dataset utility functions."""

import hashlib
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

import pytest
import yaml

from mdm.core.exceptions import DatasetError
from mdm.dataset import utils


class TestDatasetUtilsComplete:
    """Comprehensive test coverage for dataset utilities."""

    def test_normalize_dataset_name_success(self):
        """Test successful dataset name normalization."""
        assert utils.normalize_dataset_name("TestDataset") == "testdataset"
        assert utils.normalize_dataset_name("Test-Dataset") == "test-dataset"
        assert utils.normalize_dataset_name("Test_Dataset") == "test_dataset"
        assert utils.normalize_dataset_name("  Test123  ") == "test123"
        assert utils.normalize_dataset_name("TEST_DATASET") == "test_dataset"

    def test_normalize_dataset_name_empty(self):
        """Test normalizing empty dataset name."""
        with pytest.raises(DatasetError, match="Dataset name cannot be empty"):
            utils.normalize_dataset_name("")
        
        # Spaces-only string becomes empty after normalization, but is valid in validation
        # The current implementation doesn't check for empty after strip
        assert utils.normalize_dataset_name("   ") == ""

    def test_normalize_dataset_name_invalid_chars(self):
        """Test normalizing dataset name with invalid characters."""
        with pytest.raises(DatasetError, match="can only contain alphanumeric"):
            utils.normalize_dataset_name("test dataset")  # Space
        
        with pytest.raises(DatasetError, match="can only contain alphanumeric"):
            utils.normalize_dataset_name("test/dataset")  # Slash
        
        with pytest.raises(DatasetError, match="can only contain alphanumeric"):
            utils.normalize_dataset_name("test@dataset")  # Special char

    def test_get_dataset_path_default(self):
        """Test getting dataset path with default location."""
        mock_config = Mock()
        mock_config.paths.configs_path = "config/datasets"
        mock_config.paths.datasets_path = "datasets"
        
        mock_config_manager = Mock()
        mock_config_manager.config = mock_config
        mock_config_manager.base_path = Path("/base")
        
        with patch('mdm.config.get_config_manager', return_value=mock_config_manager):
            # No pointer file exists
            pointer_path = Path("/base/config/datasets/test_dataset.yaml")
            with patch('pathlib.Path.exists', return_value=False):
                path = utils.get_dataset_path("Test_Dataset")
                
                assert path == Path("/base/datasets/test_dataset")

    def test_get_dataset_path_with_pointer(self):
        """Test getting dataset path from pointer file."""
        mock_config = Mock()
        mock_config.paths.configs_path = "config/datasets"
        mock_config.paths.datasets_path = "datasets"
        
        mock_config_manager = Mock()
        mock_config_manager.config = mock_config
        mock_config_manager.base_path = Path("/base")
        
        pointer_data = {"path": "/custom/path/to/dataset"}
        
        with patch('mdm.config.get_config_manager', return_value=mock_config_manager):
            with patch('pathlib.Path.exists', return_value=True):
                with patch('builtins.open', mock_open(read_data=yaml.dump(pointer_data))):
                    with patch('pathlib.Path.expanduser') as mock_expand:
                        with patch('pathlib.Path.resolve') as mock_resolve:
                            mock_expand.return_value = Path("/custom/path/to/dataset")
                            mock_resolve.return_value = Path("/custom/path/to/dataset")
                            
                            path = utils.get_dataset_path("test_dataset")
                            
                            assert str(path) == "/custom/path/to/dataset"

    def test_get_dataset_path_pointer_error(self):
        """Test getting dataset path with pointer load error."""
        mock_config = Mock()
        mock_config.paths.configs_path = "config/datasets"
        
        mock_config_manager = Mock()
        mock_config_manager.config = mock_config
        mock_config_manager.base_path = Path("/base")
        
        with patch('mdm.config.get_config_manager', return_value=mock_config_manager):
            with patch('pathlib.Path.exists', return_value=True):
                with patch('builtins.open', side_effect=Exception("Read error")):
                    with pytest.raises(DatasetError, match="Failed to load dataset pointer"):
                        utils.get_dataset_path("test_dataset")

    def test_get_dataset_config_path(self):
        """Test getting dataset config path."""
        with patch('mdm.dataset.utils.get_dataset_path', return_value=Path("/path/to/dataset")):
            config_path = utils.get_dataset_config_path("test_dataset")
            assert config_path == Path("/path/to/dataset/dataset.yaml")

    def test_validate_dataset_name(self):
        """Test dataset name validation."""
        # Valid names
        assert utils.validate_dataset_name("test_dataset") is True
        assert utils.validate_dataset_name("Test-Dataset123") is True
        assert utils.validate_dataset_name("dataset") is True
        
        # Invalid names
        assert utils.validate_dataset_name("") is False
        assert utils.validate_dataset_name("test dataset") is False
        assert utils.validate_dataset_name("test/dataset") is False
        assert utils.validate_dataset_name("test@dataset") is False

    def test_format_file_size(self):
        """Test file size formatting."""
        assert utils.format_file_size(0) == "0.00 B"
        assert utils.format_file_size(100) == "100.00 B"
        assert utils.format_file_size(1024) == "1.00 KB"
        assert utils.format_file_size(1536) == "1.50 KB"
        assert utils.format_file_size(1048576) == "1.00 MB"
        assert utils.format_file_size(1073741824) == "1.00 GB"
        assert utils.format_file_size(1099511627776) == "1.00 TB"
        assert utils.format_file_size(1125899906842624) == "1.00 PB"

    def test_infer_file_format(self):
        """Test file format inference."""
        assert utils.infer_file_format(Path("data.csv")) == "csv"
        assert utils.infer_file_format(Path("data.tsv")) == "tsv"
        assert utils.infer_file_format(Path("data.parquet")) == "parquet"
        assert utils.infer_file_format(Path("data.pq")) == "parquet"
        assert utils.infer_file_format(Path("data.json")) == "json"
        assert utils.infer_file_format(Path("data.jsonl")) == "jsonl"
        assert utils.infer_file_format(Path("data.ndjson")) == "jsonl"
        assert utils.infer_file_format(Path("data.xlsx")) == "excel"
        assert utils.infer_file_format(Path("data.xls")) == "excel"
        assert utils.infer_file_format(Path("data.feather")) == "feather"
        assert utils.infer_file_format(Path("data.arrow")) == "arrow"
        assert utils.infer_file_format(Path("data.h5")) == "hdf5"
        assert utils.infer_file_format(Path("data.pkl")) == "pickle"
        assert utils.infer_file_format(Path("data.unknown")) == "unknown"

    def test_infer_file_format_compressed(self):
        """Test file format inference for compressed files."""
        assert utils.infer_file_format(Path("data.csv.gz")) == "csv.gz"
        assert utils.infer_file_format(Path("data.json.gz")) == "json.gz"
        assert utils.infer_file_format(Path("data.gz")) == "gzip"

    def test_get_file_encoding_with_chardet(self):
        """Test file encoding detection with chardet."""
        sample_data = b"Hello, world!"
        
        with patch('builtins.open', mock_open(read_data=sample_data)):
            with patch('chardet.detect', return_value={"encoding": "ascii"}):
                encoding = utils.get_file_encoding(Path("/test/file.txt"))
                assert encoding == "ascii"

    def test_get_file_encoding_chardet_none(self):
        """Test file encoding detection when chardet returns None."""
        sample_data = b"Hello, world!"
        
        with patch('builtins.open', mock_open(read_data=sample_data)):
            with patch('chardet.detect', return_value={"encoding": None}):
                encoding = utils.get_file_encoding(Path("/test/file.txt"))
                assert encoding == "utf-8"

    def test_get_file_encoding_no_chardet(self):
        """Test file encoding detection without chardet."""
        with patch('builtins.open', mock_open()):
            # Simulate ImportError
            import sys
            chardet_module = sys.modules.get('chardet')
            if 'chardet' in sys.modules:
                del sys.modules['chardet']
            
            try:
                encoding = utils.get_file_encoding(Path("/test/file.txt"))
                assert encoding == "utf-8"
            finally:
                # Restore module if it was there
                if chardet_module:
                    sys.modules['chardet'] = chardet_module

    def test_get_file_encoding_error(self):
        """Test file encoding detection with error."""
        with patch('builtins.open', side_effect=Exception("Read error")):
            encoding = utils.get_file_encoding(Path("/test/file.txt"))
            assert encoding == "utf-8"

    def test_calculate_file_checksum_sha256(self, tmp_path):
        """Test SHA256 checksum calculation."""
        test_file = tmp_path / "test.txt"
        test_content = b"Hello, world!"
        test_file.write_bytes(test_content)
        
        checksum = utils.calculate_file_checksum(test_file, "sha256")
        
        # Calculate expected checksum
        expected = hashlib.sha256(test_content).hexdigest()
        assert checksum == expected

    def test_calculate_file_checksum_md5(self, tmp_path):
        """Test MD5 checksum calculation."""
        test_file = tmp_path / "test.txt"
        test_content = b"Hello, world!"
        test_file.write_bytes(test_content)
        
        checksum = utils.calculate_file_checksum(test_file, "md5")
        
        # Calculate expected checksum
        expected = hashlib.md5(test_content).hexdigest()
        assert checksum == expected

    def test_calculate_file_checksum_sha1(self, tmp_path):
        """Test SHA1 checksum calculation."""
        test_file = tmp_path / "test.txt"
        test_content = b"Hello, world!"
        test_file.write_bytes(test_content)
        
        checksum = utils.calculate_file_checksum(test_file, "sha1")
        
        # Calculate expected checksum
        expected = hashlib.sha1(test_content).hexdigest()
        assert checksum == expected

    def test_calculate_file_checksum_unsupported(self, tmp_path):
        """Test checksum calculation with unsupported algorithm."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")
        
        with pytest.raises(DatasetError, match="Unsupported hash algorithm"):
            utils.calculate_file_checksum(test_file, "unsupported")

    def test_calculate_file_checksum_not_found(self):
        """Test checksum calculation for non-existent file."""
        with pytest.raises(DatasetError, match="File not found"):
            utils.calculate_file_checksum(Path("/nonexistent/file.txt"))

    def test_dataset_exists_true_default_location(self):
        """Test checking dataset existence in default location."""
        mock_config = Mock()
        mock_config.paths.configs_path = "config/datasets"
        mock_config.paths.datasets_path = "datasets"
        
        mock_config_manager = Mock()
        mock_config_manager.config = mock_config
        mock_config_manager.base_path = Path("/base")
        
        with patch('mdm.config.get_config_manager', return_value=mock_config_manager):
            # No pointer file
            with patch('pathlib.Path.exists') as mock_exists:
                with patch('pathlib.Path.is_dir') as mock_is_dir:
                    # First call: pointer doesn't exist
                    # Second call: dataset directory exists
                    mock_exists.side_effect = [False, True]
                    mock_is_dir.return_value = True
                    
                    assert utils.dataset_exists("test_dataset") is True

    def test_dataset_exists_true_with_pointer(self):
        """Test checking dataset existence with pointer."""
        mock_config = Mock()
        mock_config.paths.configs_path = "config/datasets"
        
        mock_config_manager = Mock()
        mock_config_manager.config = mock_config
        mock_config_manager.base_path = Path("/base")
        
        pointer_data = {"path": "/custom/dataset"}
        
        with patch('mdm.config.get_config_manager', return_value=mock_config_manager):
            with patch('pathlib.Path.exists', return_value=True):
                with patch('pathlib.Path.is_dir', return_value=True):
                    with patch('builtins.open', mock_open(read_data=yaml.dump(pointer_data))):
                        with patch('pathlib.Path.expanduser') as mock_expand:
                            with patch('pathlib.Path.resolve') as mock_resolve:
                                mock_expand.return_value = Path("/custom/dataset")
                                mock_resolve.return_value = Path("/custom/dataset")
                                
                                assert utils.dataset_exists("test_dataset") is True

    def test_dataset_exists_false(self):
        """Test checking non-existent dataset."""
        mock_config = Mock()
        mock_config.paths.configs_path = "config/datasets"
        mock_config.paths.datasets_path = "datasets"
        
        mock_config_manager = Mock()
        mock_config_manager.config = mock_config
        mock_config_manager.base_path = Path("/base")
        
        with patch('mdm.config.get_config_manager', return_value=mock_config_manager):
            with patch('pathlib.Path.exists', return_value=False):
                assert utils.dataset_exists("nonexistent") is False

    def test_dataset_exists_pointer_error(self):
        """Test dataset existence check with pointer error."""
        mock_config = Mock()
        mock_config.paths.configs_path = "config/datasets"
        mock_config.paths.datasets_path = "datasets"
        
        mock_config_manager = Mock()
        mock_config_manager.config = mock_config
        mock_config_manager.base_path = Path("/base")
        
        with patch('mdm.config.get_config_manager', return_value=mock_config_manager):
            with patch('pathlib.Path.exists') as mock_exists:
                # Pointer exists but can't be read
                mock_exists.side_effect = [True, False]
                with patch('builtins.open', side_effect=Exception("Read error")):
                    assert utils.dataset_exists("test_dataset") is False

    def test_get_dataset_database_path_found(self, tmp_path):
        """Test finding dataset database path."""
        dataset_path = tmp_path / "test_dataset"
        dataset_path.mkdir()
        
        # Create database file
        db_file = dataset_path / "dataset.db"
        db_file.touch()
        
        with patch('mdm.dataset.utils.get_dataset_path', return_value=dataset_path):
            db_path = utils.get_dataset_database_path("test_dataset")
            assert db_path == db_file

    def test_get_dataset_database_path_duckdb(self, tmp_path):
        """Test finding DuckDB database path."""
        dataset_path = tmp_path / "test_dataset"
        dataset_path.mkdir()
        
        # Create DuckDB file
        db_file = dataset_path / "dataset.duckdb"
        db_file.touch()
        
        with patch('mdm.dataset.utils.get_dataset_path', return_value=dataset_path):
            db_path = utils.get_dataset_database_path("test_dataset")
            assert db_path == db_file

    def test_get_dataset_database_path_named(self, tmp_path):
        """Test finding database with dataset name."""
        dataset_path = tmp_path / "test_dataset"
        dataset_path.mkdir()
        
        # Create database with dataset name
        db_file = dataset_path / "test_dataset.db"
        db_file.touch()
        
        with patch('mdm.dataset.utils.get_dataset_path', return_value=dataset_path):
            db_path = utils.get_dataset_database_path("test_dataset")
            assert db_path == db_file

    def test_get_dataset_database_path_not_found(self, tmp_path):
        """Test when no database file is found."""
        dataset_path = tmp_path / "test_dataset"
        dataset_path.mkdir()
        
        with patch('mdm.dataset.utils.get_dataset_path', return_value=dataset_path):
            db_path = utils.get_dataset_database_path("test_dataset")
            assert db_path is None

    def test_parse_file_type_train(self):
        """Test parsing train file type."""
        file_type, format_str = utils.parse_file_type("train.csv")
        assert file_type == "train"
        assert format_str == "csv"
        
        file_type, format_str = utils.parse_file_type("training_data.parquet")
        assert file_type == "train"
        assert format_str == "parquet"

    def test_parse_file_type_test(self):
        """Test parsing test file type."""
        file_type, format_str = utils.parse_file_type("test.csv")
        assert file_type == "test"
        assert format_str == "csv"
        
        file_type, format_str = utils.parse_file_type("testing_data.json")
        assert file_type == "test"
        assert format_str == "json"

    def test_parse_file_type_validation(self):
        """Test parsing validation file type."""
        file_type, format_str = utils.parse_file_type("valid.csv")
        assert file_type == "validation"
        assert format_str == "csv"
        
        file_type, format_str = utils.parse_file_type("validation_set.csv")
        assert file_type == "validation"
        assert format_str == "csv"

    def test_parse_file_type_submission(self):
        """Test parsing submission file type."""
        file_type, format_str = utils.parse_file_type("submit.csv")
        assert file_type == "submission"
        assert format_str == "csv"
        
        file_type, format_str = utils.parse_file_type("submission_template.csv")
        assert file_type == "submission"
        assert format_str == "csv"

    def test_parse_file_type_data(self):
        """Test parsing generic data file type."""
        file_type, format_str = utils.parse_file_type("data.csv")
        assert file_type == "data"
        assert format_str == "csv"
        
        file_type, format_str = utils.parse_file_type("customers.parquet")
        assert file_type == "data"
        assert format_str == "parquet"

    def test_parse_file_type_compressed(self):
        """Test parsing compressed file type."""
        file_type, format_str = utils.parse_file_type("train.csv.gz")
        assert file_type == "train"
        assert format_str == "csv.gz"
        
        # .zip extension gets removed but format is still zip
        file_type, format_str = utils.parse_file_type("test.json.zip")
        assert file_type == "test"
        assert format_str == "zip"  # ZIP is recognized as zip format, not json

    def test_is_data_file_true(self):
        """Test identifying data files."""
        assert utils.is_data_file(Path("data.csv")) is True
        assert utils.is_data_file(Path("data.tsv")) is True
        assert utils.is_data_file(Path("data.parquet")) is True
        assert utils.is_data_file(Path("data.json")) is True
        assert utils.is_data_file(Path("data.xlsx")) is True
        assert utils.is_data_file(Path("data.pkl")) is True
        assert utils.is_data_file(Path("data.feather")) is True

    def test_is_data_file_compressed(self):
        """Test identifying compressed data files."""
        assert utils.is_data_file(Path("data.csv.gz")) is True
        assert utils.is_data_file(Path("data.parquet.gz")) is True
        assert utils.is_data_file(Path("train.json.gz")) is True

    def test_is_data_file_false(self):
        """Test identifying non-data files."""
        assert utils.is_data_file(Path("readme.txt")) is False
        assert utils.is_data_file(Path("config.yaml")) is False
        assert utils.is_data_file(Path("script.py")) is False
        assert utils.is_data_file(Path("image.png")) is False
        assert utils.is_data_file(Path("archive.zip")) is False
        assert utils.is_data_file(Path("data.gz")) is False  # Just .gz without data extension