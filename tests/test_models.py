"""Tests for data models."""

from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from mdm.models.dataset import DatasetInfo, DatasetStatistics


class TestDatasetInfo:
    """Test DatasetInfo model."""

    def test_minimal_dataset_info(self):
        """Test creating DatasetInfo with minimal fields."""
        info = DatasetInfo(
            name="test_dataset",
            database={"path": "~/.mdm/datasets/test_dataset/dataset.sqlite"},
        )

        assert info.name == "test_dataset"
        assert info.database["path"] == "~/.mdm/datasets/test_dataset/dataset.sqlite"
        assert info.display_name is None
        assert info.description is None
        assert info.tables == {}
        assert info.problem_type is None
        assert info.target_column is None
        assert info.id_columns == []
        assert info.tags == []
        assert info.version == "1.0.0"

    def test_full_dataset_info(self):
        """Test creating DatasetInfo with all fields."""
        registered_at = datetime.now(timezone.utc)
        updated_at = datetime.now(timezone.utc)

        info = DatasetInfo(
            name="titanic",
            display_name="Titanic Survival Dataset",
            description="Passenger survival prediction",
            database={"path": "~/.mdm/datasets/titanic/dataset.sqlite"},
            tables={"train": "train", "test": "test", "validation": "validation"},
            problem_type="binary_classification",
            target_column="survived",
            id_columns=["passenger_id"],
            registered_at=registered_at,
            last_updated_at=updated_at,
            tags=["competition", "kaggle", "binary"],
            source="kaggle.com/competitions/titanic",
            version="2.0.0",
        )

        assert info.name == "titanic"
        assert info.display_name == "Titanic Survival Dataset"
        assert info.problem_type == "binary_classification"
        assert info.target_column == "survived"
        assert info.id_columns == ["passenger_id"]
        assert "competition" in info.tags
        assert info.version == "2.0.0"

    def test_name_validation(self):
        """Test dataset name validation."""
        # Valid names
        valid_names = ["dataset1", "test_data", "my-dataset", "dataset123"]
        for name in valid_names:
            info = DatasetInfo(name=name, database={"path": "test.db"})
            assert info.name == name.lower()

        # Invalid names
        invalid_names = ["", "dataset with spaces", "dataset@123", "dataset.name"]
        for name in invalid_names:
            with pytest.raises(ValidationError):
                DatasetInfo(name=name, database={"path": "test.db"})

    def test_name_normalization(self):
        """Test that dataset names are normalized to lowercase."""
        info = DatasetInfo(
            name="MyDataSet", database={"path": "test.db"}
        )
        assert info.name == "mydataset"

    def test_problem_type_validation(self):
        """Test problem type validation."""
        valid_types = ["binary_classification", "multiclass_classification", "regression"]
        for ptype in valid_types:
            info = DatasetInfo(
                name="test", database={"path": "test.db"}, problem_type=ptype
            )
            assert info.problem_type == ptype

        # Invalid problem type
        with pytest.raises(ValidationError):
            DatasetInfo(
                name="test",
                database={"path": "test.db"},
                problem_type="invalid_type",
            )

    def test_has_train_test_split(self):
        """Test has_train_test_split property."""
        # No tables
        info = DatasetInfo(name="test", database={"path": "test.db"})
        assert not info.has_train_test_split

        # Only train
        info = DatasetInfo(
            name="test",
            database={"path": "test.db"},
            tables={"train": "train"},
        )
        assert not info.has_train_test_split

        # Train and test
        info = DatasetInfo(
            name="test",
            database={"path": "test.db"},
            tables={"train": "train", "test": "test"},
        )
        assert info.has_train_test_split

    def test_has_validation(self):
        """Test has_validation property."""
        # No validation
        info = DatasetInfo(
            name="test",
            database={"path": "test.db"},
            tables={"train": "train", "test": "test"},
        )
        assert not info.has_validation

        # With validation
        info = DatasetInfo(
            name="test",
            database={"path": "test.db"},
            tables={"train": "train", "test": "test", "validation": "validation"},
        )
        assert info.has_validation

    def test_get_database_path(self):
        """Test getting database file path."""
        # SQLite/DuckDB with path
        info = DatasetInfo(
            name="test",
            database={"path": "~/.mdm/datasets/test/dataset.sqlite"},
        )
        db_path = info.get_database_path()
        assert db_path == Path("~/.mdm/datasets/test/dataset.sqlite").expanduser()

        # PostgreSQL with connection string
        info = DatasetInfo(
            name="test",
            database={"connection_string": "postgresql://user:pass@localhost/test"},
        )
        assert info.get_database_path() is None

    def test_get_connection_string(self):
        """Test getting connection string."""
        # PostgreSQL
        info = DatasetInfo(
            name="test",
            database={"connection_string": "postgresql://user:pass@localhost/test"},
        )
        assert info.get_connection_string() == "postgresql://user:pass@localhost/test"

        # SQLite/DuckDB
        info = DatasetInfo(
            name="test",
            database={"path": "~/.mdm/datasets/test/dataset.sqlite"},
        )
        assert info.get_connection_string() is None


class TestDatasetStatistics:
    """Test DatasetStatistics model."""

    def test_minimal_statistics(self):
        """Test creating DatasetStatistics with minimal fields."""
        stats = DatasetStatistics(
            row_count=1000,
            column_count=10,
            memory_usage_mb=5.2,
        )

        assert stats.row_count == 1000
        assert stats.column_count == 10
        assert stats.memory_usage_mb == 5.2
        assert stats.missing_values == {}
        assert stats.column_types == {}
        assert stats.numeric_columns == []
        assert stats.categorical_columns == []
        assert stats.datetime_columns == []
        assert stats.text_columns == []
        assert isinstance(stats.computed_at, datetime)

    def test_full_statistics(self):
        """Test creating DatasetStatistics with all fields."""
        computed_at = datetime.now(timezone.utc)

        stats = DatasetStatistics(
            row_count=5000,
            column_count=15,
            memory_usage_mb=25.5,
            missing_values={"age": 100, "income": 250},
            column_types={
                "id": "int64",
                "name": "object",
                "age": "float64",
                "date": "datetime64[ns]",
            },
            numeric_columns=["id", "age", "income"],
            categorical_columns=["category", "status"],
            datetime_columns=["date", "timestamp"],
            text_columns=["name", "description"],
            computed_at=computed_at,
        )

        assert stats.row_count == 5000
        assert stats.missing_values["age"] == 100
        assert stats.column_types["name"] == "object"
        assert "age" in stats.numeric_columns
        assert "category" in stats.categorical_columns
        assert "date" in stats.datetime_columns
        assert "description" in stats.text_columns
        assert stats.computed_at == computed_at

    def test_field_types(self):
        """Test field type validation."""
        # Valid statistics
        stats = DatasetStatistics(
            row_count=100,
            column_count=5,
            memory_usage_mb=1.5,
        )
        assert isinstance(stats.row_count, int)
        assert isinstance(stats.memory_usage_mb, float)

        # Pydantic v2 does automatic type conversion for simple types
        # So string "100" converts to int 100 automatically