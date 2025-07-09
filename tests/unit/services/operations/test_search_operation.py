"""Unit tests for SearchOperation."""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path
import yaml
import tempfile

from mdm.dataset.operations import SearchOperation


class TestSearchOperation:
    """Test cases for SearchOperation."""

    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary config directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def mock_config(self):
        """Mock configuration."""
        config = Mock()
        config.paths.configs_path = "config/datasets"
        config.paths.datasets_path = "datasets"
        config.database.default_backend = "sqlite"
        return config

    @pytest.fixture
    def search_operation(self, mock_config, temp_config_dir):
        """Create SearchOperation instance."""
        # Create config/datasets directory
        config_path = temp_config_dir / "config" / "datasets"
        config_path.mkdir(parents=True)
        
        with patch('mdm.dataset.operations.get_config_manager') as mock_get_config:
            mock_manager = Mock()
            mock_manager.config = mock_config
            mock_manager.base_path = temp_config_dir
            mock_get_config.return_value = mock_manager
            
            operation = SearchOperation()
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
            'database': kwargs.get('database', {'backend': 'sqlite'}),
            'columns': kwargs.get('columns', [])
        }
        
        yaml_file = path / f"{name}.yaml"
        with open(yaml_file, 'w') as f:
            yaml.dump(data, f)
        
        return yaml_file

    def test_execute_simple_search(self, search_operation):
        """Test simple search by name."""
        # Create datasets
        self.create_dataset_yaml(
            search_operation.dataset_registry_dir,
            "iris_dataset",
            description="Iris flower dataset"
        )
        self.create_dataset_yaml(
            search_operation.dataset_registry_dir,
            "titanic_dataset",
            description="Titanic passenger survival"
        )

        # Execute search
        result = search_operation.execute("iris")

        # Assert
        assert len(result) == 1
        assert result[0]['name'] == 'iris_dataset'

    def test_execute_case_insensitive_search(self, search_operation):
        """Test case-insensitive search."""
        # Create dataset
        self.create_dataset_yaml(
            search_operation.dataset_registry_dir,
            "IrisDataset",
            description="Iris flowers"
        )

        # Execute search with different case
        result = search_operation.execute("irisdataset", case_sensitive=False)

        # Assert
        assert len(result) == 1
        assert result[0]['name'] == 'IrisDataset'

    def test_execute_case_sensitive_search(self, search_operation):
        """Test case-sensitive search."""
        # Create datasets
        self.create_dataset_yaml(
            search_operation.dataset_registry_dir,
            "IrisDataset"
        )
        self.create_dataset_yaml(
            search_operation.dataset_registry_dir,
            "irisdataset"
        )

        # Execute case-sensitive search
        result = search_operation.execute("IrisDataset", case_sensitive=True)

        # Assert
        assert len(result) == 1
        assert result[0]['name'] == 'IrisDataset'

    def test_execute_pattern_search(self, search_operation):
        """Test glob pattern search."""
        # Create datasets
        self.create_dataset_yaml(
            search_operation.dataset_registry_dir,
            "train_dataset_v1"
        )
        self.create_dataset_yaml(
            search_operation.dataset_registry_dir,
            "train_dataset_v2"
        )
        self.create_dataset_yaml(
            search_operation.dataset_registry_dir,
            "test_dataset"
        )

        # Execute pattern search
        result = search_operation.execute("train_*", pattern=True)

        # Assert
        assert len(result) == 2
        names = {d['name'] for d in result}
        assert names == {'train_dataset_v1', 'train_dataset_v2'}

    def test_execute_search_by_tag(self, search_operation):
        """Test search by tag only."""
        # Create datasets with tags
        self.create_dataset_yaml(
            search_operation.dataset_registry_dir,
            "dataset1",
            tags=["kaggle", "competition"]
        )
        self.create_dataset_yaml(
            search_operation.dataset_registry_dir,
            "dataset2",
            tags=["custom", "test"]
        )
        self.create_dataset_yaml(
            search_operation.dataset_registry_dir,
            "dataset3",
            tags=["kaggle", "tutorial"]
        )

        # Execute search by tag
        result = search_operation.execute("kaggle", tag="kaggle")

        # Assert
        assert len(result) == 2
        names = {d['name'] for d in result}
        assert names == {'dataset1', 'dataset3'}

    def test_execute_search_with_tag_filter(self, search_operation):
        """Test search with query and tag filter."""
        # Create datasets
        self.create_dataset_yaml(
            search_operation.dataset_registry_dir,
            "iris_competition",
            tags=["competition"],
            description="Iris classification competition"
        )
        self.create_dataset_yaml(
            search_operation.dataset_registry_dir,
            "iris_tutorial",
            tags=["tutorial"],
            description="Iris dataset tutorial"
        )

        # Execute search with tag filter
        result = search_operation.execute("iris", tag="competition")

        # Assert
        assert len(result) == 1
        assert result[0]['name'] == 'iris_competition'

    def test_execute_deep_search(self, search_operation):
        """Test deep search in metadata."""
        # Create datasets with columns
        self.create_dataset_yaml(
            search_operation.dataset_registry_dir,
            "dataset1",
            columns=["user_id", "user_name", "user_email"]
        )
        self.create_dataset_yaml(
            search_operation.dataset_registry_dir,
            "dataset2",
            columns=["product_id", "product_name", "price"]
        )

        # Execute deep search
        result = search_operation.execute("user", deep=True)

        # Assert
        assert len(result) == 1
        assert result[0]['name'] == 'dataset1'

    def test_execute_search_in_description(self, search_operation):
        """Test search in description."""
        # Create datasets
        self.create_dataset_yaml(
            search_operation.dataset_registry_dir,
            "dataset1",
            description="Customer purchase history"
        )
        self.create_dataset_yaml(
            search_operation.dataset_registry_dir,
            "dataset2",
            description="Product inventory data"
        )

        # Execute search
        result = search_operation.execute("purchase")

        # Assert
        assert len(result) == 1
        assert result[0]['name'] == 'dataset1'

    def test_execute_with_limit(self, search_operation):
        """Test search with result limit."""
        # Create multiple matching datasets
        for i in range(5):
            self.create_dataset_yaml(
                search_operation.dataset_registry_dir,
                f"test_dataset_{i}",
                description="Test dataset"
            )

        # Execute with limit
        result = search_operation.execute("test", limit=3)

        # Assert
        assert len(result) == 3

    def test_execute_no_matches(self, search_operation):
        """Test search with no matches."""
        # Create datasets
        self.create_dataset_yaml(
            search_operation.dataset_registry_dir,
            "dataset1"
        )

        # Execute search that won't match
        result = search_operation.execute("nonexistent")

        # Assert
        assert result == []

    def test_execute_backend_filter(self, search_operation):
        """Test that search returns all datasets regardless of backend."""
        # Create datasets with different backends
        self.create_dataset_yaml(
            search_operation.dataset_registry_dir,
            "sqlite_dataset",
            database={'backend': 'sqlite'}
        )
        self.create_dataset_yaml(
            search_operation.dataset_registry_dir,
            "duckdb_dataset",
            database={'backend': 'duckdb'}
        )

        # Execute search
        result = search_operation.execute("dataset")

        # Assert - both datasets should be returned (search doesn't filter by backend)
        assert len(result) == 2
        names = {d['name'] for d in result}
        assert names == {'sqlite_dataset', 'duckdb_dataset'}

    def test_execute_empty_query_with_tag(self, search_operation):
        """Test empty query with tag filter."""
        # Create datasets with tags
        self.create_dataset_yaml(
            search_operation.dataset_registry_dir,
            "dataset1",
            tags=["production"]
        )
        self.create_dataset_yaml(
            search_operation.dataset_registry_dir,
            "dataset2",
            tags=["test"]
        )

        # Execute with empty query but with tag
        result = search_operation.execute("", tag="production")

        # Assert
        assert len(result) == 1
        assert result[0]['name'] == 'dataset1'