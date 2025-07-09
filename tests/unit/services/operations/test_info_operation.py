"""Unit tests for InfoOperation."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import yaml
import tempfile

from mdm.dataset.operations import InfoOperation
from mdm.core.exceptions import DatasetError


class TestInfoOperation:
    """Test cases for InfoOperation."""

    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            (base / "config" / "datasets").mkdir(parents=True)
            (base / "datasets").mkdir(parents=True)
            yield base

    @pytest.fixture
    def mock_config(self):
        """Mock configuration."""
        config = Mock()
        config.paths.configs_path = "config/datasets"
        config.paths.datasets_path = "datasets"
        config.database.default_backend = "sqlite"
        return config

    @pytest.fixture
    def info_operation(self, mock_config, temp_dirs):
        """Create InfoOperation instance."""
        with patch('mdm.dataset.operations.get_config_manager') as mock_get_config:
            mock_manager = Mock()
            mock_manager.config = mock_config
            mock_manager.base_path = temp_dirs
            mock_get_config.return_value = mock_manager
            
            operation = InfoOperation()
            return operation

    def create_dataset_yaml(self, registry_dir: Path, name: str, **kwargs):
        """Helper to create dataset YAML file."""
        data = {
            'name': name,
            'problem_type': kwargs.get('problem_type', 'classification'),
            'target_column': kwargs.get('target_column', 'target'),
            'tables': kwargs.get('tables', {'train': 'train_table'}),
            'description': kwargs.get('description', f'Test dataset {name}'),
            'tags': kwargs.get('tags', []),
            'created_at': kwargs.get('created_at', '2024-01-01T00:00:00'),
            'database': kwargs.get('database', {'backend': 'sqlite'}),
            'metadata': {
                'statistics': {
                    'row_count': kwargs.get('row_count', 1000),
                    'column_count': kwargs.get('column_count', 10)
                }
            }
        }
        
        yaml_file = registry_dir / f"{name}.yaml"
        with open(yaml_file, 'w') as f:
            yaml.dump(data, f)
        
        return yaml_file

    def test_execute_success(self, info_operation):
        """Test successful dataset info retrieval."""
        # Create dataset
        self.create_dataset_yaml(
            info_operation.dataset_registry_dir,
            "test_dataset",
            problem_type="regression",
            tags=["test", "sample"]
        )

        # Execute
        result = info_operation.execute("test_dataset")

        # Assert
        assert result['name'] == 'test_dataset'
        assert result['problem_type'] == 'regression'
        assert result['tags'] == ['test', 'sample']
        assert result['database']['backend'] == 'sqlite'

    def test_execute_dataset_not_found(self, info_operation):
        """Test info for non-existent dataset."""
        # Execute & Assert
        with pytest.raises(DatasetError, match="Dataset 'nonexistent' not found"):
            info_operation.execute("nonexistent")

    def test_execute_backend_mismatch(self, info_operation):
        """Test error when backend doesn't match."""
        # Create dataset with different backend
        self.create_dataset_yaml(
            info_operation.dataset_registry_dir,
            "duckdb_dataset",
            database={'backend': 'duckdb'}
        )

        # Execute & Assert
        with pytest.raises(DatasetError, match="uses 'duckdb' backend.*current backend is 'sqlite'"):
            info_operation.execute("duckdb_dataset")

    def test_execute_with_dataset_directory(self, info_operation):
        """Test info with existing dataset directory."""
        # Create dataset and directory
        self.create_dataset_yaml(
            info_operation.dataset_registry_dir,
            "test_dataset"
        )
        
        dataset_dir = info_operation.datasets_dir / "test_dataset"
        dataset_dir.mkdir(parents=True)
        
        # Create some files
        (dataset_dir / "file1.txt").write_text("test content")
        (dataset_dir / "file2.txt").write_text("more content")

        # Execute
        result = info_operation.execute("test_dataset")

        # Assert
        assert 'dataset_path' in result
        assert result['dataset_path'] == str(dataset_dir)
        assert 'total_size' in result
        assert result['total_size'] > 0

    def test_execute_with_duckdb_file(self, info_operation):
        """Test info with DuckDB database file."""
        # Create dataset with duckdb backend
        info_operation.config.database.default_backend = "duckdb"
        self.create_dataset_yaml(
            info_operation.dataset_registry_dir,
            "duckdb_dataset",
            database={'backend': 'duckdb'}
        )
        
        dataset_dir = info_operation.datasets_dir / "duckdb_dataset"
        dataset_dir.mkdir(parents=True)
        
        # Create mock database file
        db_file = dataset_dir / "dataset.duckdb"
        db_file.write_text("mock database content")

        # Execute
        result = info_operation.execute("duckdb_dataset")

        # Assert
        assert 'database_file' in result
        assert result['database_file'] == str(db_file)
        assert 'database_size' in result
        assert result['database_size'] > 0

    def test_execute_with_details(self, info_operation):
        """Test info with detailed statistics."""
        # Create dataset
        self.create_dataset_yaml(
            info_operation.dataset_registry_dir,
            "test_dataset"
        )

        # Execute with details
        result = info_operation.execute("test_dataset", details=True)

        # Assert
        assert 'statistics' in result
        assert 'note' in result['statistics']

    def test_execute_yaml_parse_error(self, info_operation):
        """Test error handling for corrupted YAML."""
        # Create invalid YAML file
        yaml_file = info_operation.dataset_registry_dir / "invalid.yaml"
        yaml_file.write_text("invalid: yaml: content: {")

        # Execute & Assert
        with pytest.raises(DatasetError, match="Failed to get dataset info"):
            info_operation.execute("invalid")

    def test_execute_missing_backend_info(self, info_operation):
        """Test handling of missing backend information."""
        # Create dataset without backend info
        data = {
            'name': 'test_dataset',
            'problem_type': 'binary_classification',
            'database': {}  # Missing backend
        }
        
        yaml_file = info_operation.dataset_registry_dir / "test_dataset.yaml"
        with open(yaml_file, 'w') as f:
            yaml.dump(data, f)

        # Execute & Assert
        with pytest.raises(DatasetError, match="uses 'unknown' backend"):
            info_operation.execute("test_dataset")

    def test_execute_calculate_total_size(self, info_operation):
        """Test total size calculation with nested files."""
        # Create dataset
        self.create_dataset_yaml(
            info_operation.dataset_registry_dir,
            "test_dataset"
        )
        
        # Create nested directory structure
        dataset_dir = info_operation.datasets_dir / "test_dataset"
        subdir = dataset_dir / "subdir"
        subdir.mkdir(parents=True)
        
        (dataset_dir / "file1.txt").write_text("content1")
        (subdir / "file2.txt").write_text("content2")
        (subdir / "file3.txt").write_text("content3")

        # Execute
        result = info_operation.execute("test_dataset")

        # Assert
        assert result['total_size'] == len("content1") + len("content2") + len("content3")