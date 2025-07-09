"""Unit tests for ListOperation."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from datetime import datetime
import yaml
import tempfile

from mdm.dataset.operations import ListOperation


class TestListOperation:
    """Test cases for ListOperation."""

    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary config directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def mock_config(self, temp_config_dir):
        """Mock configuration."""
        config = Mock()
        config.paths.configs_path = "config/datasets"
        config.paths.datasets_path = "datasets"
        config.performance.max_concurrent_operations = 4
        config.database.default_backend = "sqlite"
        return config

    @pytest.fixture
    def list_operation(self, mock_config, temp_config_dir):
        """Create ListOperation instance."""
        with patch('mdm.dataset.operations.get_config_manager') as mock_get_config:
            mock_manager = Mock()
            mock_manager.config = mock_config
            mock_manager.base_path = temp_config_dir
            mock_get_config.return_value = mock_manager
            
            operation = ListOperation()
            # Ensure paths are properly set
            assert operation.base_path == temp_config_dir
            assert operation.dataset_registry_dir == temp_config_dir / "config/datasets"
            # Create the directory
            operation.dataset_registry_dir.mkdir(parents=True, exist_ok=True)
            return operation

    def create_dataset_yaml(self, path: Path, name: str, **kwargs):
        """Helper to create dataset YAML file."""
        data = {
            'name': name,
            'display_name': kwargs.get('display_name', name),
            'problem_type': kwargs.get('problem_type', 'classification'),
            'target_column': kwargs.get('target_column', 'target'),
            'tables': kwargs.get('tables', {'train': 'train_table'}),
            'description': kwargs.get('description', f'Test dataset {name}'),
            'tags': kwargs.get('tags', []),
            'created_at': kwargs.get('created_at', datetime.now().isoformat()),
            'database': kwargs.get('database', {'backend': 'sqlite'}),
            'source': kwargs.get('source', 'test'),
            'metadata': {
                'statistics': {
                    'row_count': kwargs.get('row_count', 1000),
                    'memory_size_bytes': kwargs.get('size', 1024000)
                }
            }
        }
        
        yaml_file = path / f"{name}.yaml"
        with open(yaml_file, 'w') as f:
            yaml.dump(data, f)
        
        return yaml_file

    def test_execute_empty_directory(self, list_operation):
        """Test listing with no datasets."""
        result = list_operation.execute()
        assert result == []

    def test_execute_single_dataset(self, list_operation):
        """Test listing single dataset."""
        # Create dataset
        self.create_dataset_yaml(
            list_operation.dataset_registry_dir,
            "test_dataset",
            problem_type="regression",
            tags=["test", "sample"]
        )

        # Execute
        result = list_operation.execute()

        # Assert
        assert len(result) == 1
        assert result[0]['name'] == 'test_dataset'
        assert result[0]['problem_type'] == 'regression'
        assert result[0]['tags'] == ['test', 'sample']
        assert result[0]['backend_compatible'] is True

    def test_execute_multiple_datasets(self, list_operation):
        """Test listing multiple datasets."""
        # Create datasets
        for i in range(3):
            self.create_dataset_yaml(
                list_operation.dataset_registry_dir,
                f"dataset_{i}",
                created_at=datetime(2024, 1, i+1).isoformat()
            )

        # Execute
        result = list_operation.execute()

        # Assert
        assert len(result) == 3
        names = [d['name'] for d in result]
        assert 'dataset_0' in names
        assert 'dataset_1' in names
        assert 'dataset_2' in names

    def test_execute_with_filter(self, list_operation):
        """Test listing with filters."""
        # Create datasets with different problem types
        self.create_dataset_yaml(
            list_operation.dataset_registry_dir,
            "classification_dataset",
            problem_type="classification"
        )
        self.create_dataset_yaml(
            list_operation.dataset_registry_dir,
            "regression_dataset",
            problem_type="regression"
        )

        # Execute with filter
        result = list_operation.execute(filter_str="problem_type=classification")

        # Assert
        assert len(result) == 1
        assert result[0]['name'] == 'classification_dataset'

    def test_execute_with_multiple_filters(self, list_operation):
        """Test listing with multiple filters."""
        # Create datasets
        self.create_dataset_yaml(
            list_operation.dataset_registry_dir,
            "dataset1",
            problem_type="classification",
            source="kaggle"
        )
        self.create_dataset_yaml(
            list_operation.dataset_registry_dir,
            "dataset2",
            problem_type="classification",
            source="custom"
        )
        self.create_dataset_yaml(
            list_operation.dataset_registry_dir,
            "dataset3",
            problem_type="regression",
            source="kaggle"
        )

        # Execute with multiple filters
        result = list_operation.execute(filter_str="problem_type=classification,source=kaggle")

        # Assert
        assert len(result) == 1
        assert result[0]['name'] == 'dataset1'

    def test_execute_sort_by_name(self, list_operation):
        """Test sorting by name."""
        # Create datasets with different names
        self.create_dataset_yaml(list_operation.dataset_registry_dir, "zebra")
        self.create_dataset_yaml(list_operation.dataset_registry_dir, "apple")
        self.create_dataset_yaml(list_operation.dataset_registry_dir, "banana")

        # Execute with sort
        result = list_operation.execute(sort_by="name")

        # Assert
        names = [d['name'] for d in result]
        assert names == ['apple', 'banana', 'zebra']

    def test_execute_sort_by_date(self, list_operation):
        """Test sorting by registration date."""
        # Create datasets with different dates
        self.create_dataset_yaml(
            list_operation.dataset_registry_dir,
            "old_dataset",
            created_at="2023-01-01T00:00:00"
        )
        self.create_dataset_yaml(
            list_operation.dataset_registry_dir,
            "new_dataset",
            created_at="2024-01-01T00:00:00"
        )
        self.create_dataset_yaml(
            list_operation.dataset_registry_dir,
            "middle_dataset",
            created_at="2023-06-01T00:00:00"
        )

        # Execute with sort by date
        result = list_operation.execute(sort_by="registration_date")

        # Assert (sorted by date descending)
        names = [d['name'] for d in result]
        assert names == ['new_dataset', 'middle_dataset', 'old_dataset']

    def test_execute_with_limit(self, list_operation):
        """Test listing with limit."""
        # Create 5 datasets
        for i in range(5):
            self.create_dataset_yaml(
                list_operation.dataset_registry_dir,
                f"dataset_{i}"
            )

        # Execute with limit
        result = list_operation.execute(limit=3)

        # Assert
        assert len(result) == 3

    def test_execute_backend_compatibility(self, list_operation):
        """Test backend compatibility check."""
        # Create datasets with different backends
        self.create_dataset_yaml(
            list_operation.dataset_registry_dir,
            "sqlite_dataset",
            database={'backend': 'sqlite'}
        )
        self.create_dataset_yaml(
            list_operation.dataset_registry_dir,
            "duckdb_dataset",
            database={'backend': 'duckdb'}
        )

        # Execute
        result = list_operation.execute()

        # Assert
        sqlite_ds = next(d for d in result if d['name'] == 'sqlite_dataset')
        duckdb_ds = next(d for d in result if d['name'] == 'duckdb_dataset')
        
        assert sqlite_ds['backend_compatible'] is True  # Current backend is sqlite
        assert duckdb_ds['backend_compatible'] is False

    def test_parse_yaml_file_error_handling(self, list_operation):
        """Test error handling in YAML parsing."""
        # Create invalid YAML file
        yaml_file = list_operation.dataset_registry_dir / "invalid.yaml"
        yaml_file.parent.mkdir(parents=True, exist_ok=True)
        yaml_file.write_text("invalid: yaml: content: {")

        # Execute (should not raise, just log error)
        result = list_operation.execute()

        # Assert
        assert result == []

    def test_concurrent_parsing(self, list_operation):
        """Test concurrent YAML file parsing."""
        # Create many datasets
        for i in range(10):
            self.create_dataset_yaml(
                list_operation.dataset_registry_dir,
                f"dataset_{i:02d}"
            )

        # Execute
        result = list_operation.execute()

        # Assert all datasets are parsed
        assert len(result) == 10
        names = {d['name'] for d in result}
        expected_names = {f"dataset_{i:02d}" for i in range(10)}
        assert names == expected_names