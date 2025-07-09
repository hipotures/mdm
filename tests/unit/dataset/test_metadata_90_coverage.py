"""Comprehensive unit tests for dataset metadata module to achieve 90%+ coverage."""

import json
import pytest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, call
import sqlalchemy as sa
from sqlalchemy.exc import SQLAlchemyError

from mdm.dataset.metadata import (
    get_dataset_engine,
    create_metadata_tables,
    store_dataset_metadata,
    get_dataset_metadata,
    store_column_metadata,
    get_column_metadata,
    store_dataset_statistics,
    get_latest_statistics,
    store_extended_info,
    get_extended_info,
)
from mdm.core.exceptions import DatasetError
from mdm.models.dataset import ColumnInfo, DatasetInfoExtended, DatasetStatistics, FileInfo
from mdm.models.enums import ColumnType, ProblemType, FileType


class TestMetadata90Coverage:
    """Comprehensive test cases for metadata module to achieve 90%+ coverage."""

    @pytest.fixture
    def mock_engine(self):
        """Create mock SQLAlchemy engine."""
        engine = Mock()
        engine.dialect.name = 'sqlite'
        engine.begin.return_value.__enter__ = Mock()
        engine.begin.return_value.__exit__ = Mock()
        engine.connect.return_value.__enter__ = Mock()
        engine.connect.return_value.__exit__ = Mock()
        return engine

    @pytest.fixture
    def mock_connection(self):
        """Create mock database connection."""
        conn = Mock()
        conn.execute = Mock()
        return conn

    @pytest.fixture
    def sample_column_info(self):
        """Create sample ColumnInfo objects."""
        return [
            ColumnInfo(
                name="id",
                dtype="int64",
                column_type=ColumnType.ID,
                nullable=False,
                unique=True,
                missing_count=0,
                missing_ratio=0.0,
                cardinality=100,
                min_value=1,
                max_value=100,
                mean_value=50.5,
                std_value=28.87,
                sample_values=[1, 2, 3, 4, 5],
                description="Primary key"
            ),
            ColumnInfo(
                name="name",
                dtype="object",
                column_type=ColumnType.TEXT,
                nullable=True,
                unique=False,
                missing_count=5,
                missing_ratio=0.05,
                cardinality=90,
                min_value=None,
                max_value=None,
                mean_value=None,
                std_value=None,
                sample_values=["Alice", "Bob", "Charlie"],
                description="Customer name"
            ),
        ]

    @pytest.fixture
    def sample_statistics(self):
        """Create sample DatasetStatistics object."""
        return DatasetStatistics(
            row_count=1000,
            column_count=10,
            memory_usage_mb=5.2,
            missing_values={"name": 5, "age": 10},
            column_types={"id": "int64", "name": "object"},
            numeric_columns=["id", "age", "salary"],
            categorical_columns=["category", "status"],
            datetime_columns=["created_at", "updated_at"],
            text_columns=["name", "description"],
            computed_at=datetime.now(timezone.utc)
        )

    @pytest.fixture
    def sample_extended_info(self):
        """Create sample DatasetInfoExtended object."""
        return DatasetInfoExtended(
            name="test_dataset",
            display_name="Test Dataset",
            description="A test dataset",
            problem_type=ProblemType.BINARY_CLASSIFICATION,
            target_column="target",
            id_columns=["id"],
            datetime_columns=["created_at"],
            feature_columns=["feature1", "feature2"],
            row_count=1000,
            memory_usage_mb=5.2,
            source="/path/to/data.csv",
            version="1.0.0",
            tags=["test", "example"],
            metadata={"custom": "value"},
            created_at=datetime.now(timezone.utc),
            updated_at=None,
            files={
                "train": FileInfo(
                    path=Path("/path/to/train.csv"),
                    name="train.csv",
                    size_bytes=1024000,
                    file_type=FileType.CSV,
                    format="csv",
                    encoding="utf-8",
                    row_count=800,
                    column_count=10,
                    created_at=datetime.now(timezone.utc),
                    modified_at=None,
                    checksum="abc123"
                )
            },
            columns={},
            statistics=None
        )

    def test_get_dataset_engine_sqlite(self):
        """Test getting SQLAlchemy engine for SQLite."""
        mock_config = Mock()
        mock_config.database = {
            "type": "sqlite",
            "path": "/tmp/test.db"
        }
        
        with patch('mdm.dataset.config.load_dataset_config', return_value=mock_config):
            with patch('mdm.dataset.metadata.create_engine') as mock_create:
                engine = get_dataset_engine("test_dataset")
                
                mock_create.assert_called_once()
                call_arg = mock_create.call_args[0][0]
                assert call_arg.startswith("sqlite:///")
                assert "test.db" in call_arg

    def test_get_dataset_engine_duckdb(self):
        """Test getting SQLAlchemy engine for DuckDB."""
        mock_config = Mock()
        mock_config.database = {
            "type": "duckdb",
            "path": "/tmp/test.duckdb"
        }
        
        with patch('mdm.dataset.config.load_dataset_config', return_value=mock_config):
            with patch('mdm.dataset.metadata.create_engine') as mock_create:
                engine = get_dataset_engine("test_dataset")
                
                mock_create.assert_called_once()
                call_arg = mock_create.call_args[0][0]
                assert call_arg.startswith("duckdb:///")
                assert "test.duckdb" in call_arg

    def test_get_dataset_engine_postgresql(self):
        """Test getting SQLAlchemy engine for PostgreSQL."""
        mock_config = Mock()
        mock_config.database = {
            "type": "postgresql",
            "connection_string": "postgresql://user:pass@localhost/db"
        }
        
        with patch('mdm.dataset.config.load_dataset_config', return_value=mock_config):
            with patch('mdm.dataset.metadata.create_engine') as mock_create:
                engine = get_dataset_engine("test_dataset")
                
                mock_create.assert_called_once_with("postgresql://user:pass@localhost/db")

    def test_get_dataset_engine_unsupported(self):
        """Test error for unsupported database type."""
        mock_config = Mock()
        mock_config.database = {
            "type": "mysql",
            "path": "/tmp/test.db"
        }
        
        with patch('mdm.dataset.config.load_dataset_config', return_value=mock_config):
            with pytest.raises(DatasetError, match="Unsupported database type: mysql"):
                get_dataset_engine("test_dataset")

    def test_create_metadata_tables(self, mock_engine):
        """Test creating metadata tables."""
        mock_metadata = Mock()
        
        with patch('mdm.dataset.metadata.sa.MetaData', return_value=mock_metadata):
            with patch('mdm.dataset.metadata.Table') as mock_table:
                create_metadata_tables(mock_engine)
                
                # Should create 3 tables
                assert mock_table.call_count == 3
                
                # Check table names
                table_calls = mock_table.call_args_list
                assert table_calls[0][0][0] == "_metadata"
                assert table_calls[1][0][0] == "_columns"
                assert table_calls[2][0][0] == "_statistics"
                
                # Should create all tables
                mock_metadata.create_all.assert_called_once_with(mock_engine)

    def test_store_dataset_metadata_new(self, mock_engine, mock_connection):
        """Test storing new metadata key-value pair."""
        mock_engine.begin.return_value.__enter__.return_value = mock_connection
        mock_metadata_table = Mock()
        mock_metadata_table.insert.return_value.values.return_value = Mock()
        mock_metadata_table.insert.return_value.values.return_value.on_conflict_do_update.return_value = Mock()
        
        with patch('mdm.dataset.metadata.sa.MetaData'):
            with patch('mdm.dataset.metadata.Table', return_value=mock_metadata_table):
                with patch('mdm.dataset.metadata.create_metadata_tables'):
                    store_dataset_metadata("test_dataset", "key1", {"value": "test"}, mock_engine)
                    
                    # Should execute insert
                    mock_connection.execute.assert_called_once()

    def test_store_dataset_metadata_update_sqlite(self, mock_engine, mock_connection):
        """Test updating existing metadata for SQLite."""
        mock_engine.dialect.name = 'sqlite'
        mock_engine.begin.return_value.__enter__.return_value = mock_connection
        
        mock_metadata_table = Mock()
        mock_stmt = Mock()
        mock_metadata_table.insert.return_value.values.return_value = mock_stmt
        
        with patch('mdm.dataset.metadata.sa.MetaData'):
            with patch('mdm.dataset.metadata.Table', return_value=mock_metadata_table):
                with patch('mdm.dataset.metadata.create_metadata_tables'):
                    store_dataset_metadata("test_dataset", "key1", {"value": "updated"}, mock_engine)
                    
                    # Should call on_conflict_do_update for SQLite
                    mock_stmt.on_conflict_do_update.assert_called_once()

    def test_store_dataset_metadata_update_other_db(self, mock_engine, mock_connection):
        """Test updating existing metadata for other databases."""
        mock_engine.dialect.name = 'mysql'
        mock_engine.begin.return_value.__enter__.return_value = mock_connection
        
        mock_metadata_table = Mock()
        mock_metadata_table.c.key = Mock()
        mock_metadata_table.delete.return_value.where.return_value = Mock()
        
        with patch('mdm.dataset.metadata.sa.MetaData'):
            with patch('mdm.dataset.metadata.Table', return_value=mock_metadata_table):
                with patch('mdm.dataset.metadata.create_metadata_tables'):
                    store_dataset_metadata("test_dataset", "key1", {"value": "updated"}, mock_engine)
                    
                    # Should delete then insert for other databases
                    mock_metadata_table.delete.assert_called_once()
                    assert mock_connection.execute.call_count == 2  # Delete + Insert

    def test_store_dataset_metadata_no_engine(self):
        """Test storing metadata without providing engine."""
        mock_engine = Mock()
        mock_engine.dialect.name = 'sqlite'
        mock_engine.begin.return_value.__enter__ = Mock(return_value=Mock())
        mock_engine.begin.return_value.__exit__ = Mock()
        
        with patch('mdm.dataset.metadata.get_dataset_engine', return_value=mock_engine) as mock_get_engine:
            with patch('mdm.dataset.metadata.sa.MetaData'):
                with patch('mdm.dataset.metadata.Table'):
                    with patch('mdm.dataset.metadata.create_metadata_tables'):
                        store_dataset_metadata("test_dataset", "key1", {"value": "test"})
                        
                        # Should get engine automatically
                        mock_get_engine.assert_called_once_with("test_dataset")

    def test_get_dataset_metadata_exists(self, mock_engine, mock_connection):
        """Test getting existing metadata value."""
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_result = Mock()
        mock_result.scalar.return_value = {"value": "test"}
        mock_connection.execute.return_value = mock_result
        
        mock_metadata_table = Mock()
        mock_metadata_table.c.value = Mock()
        mock_metadata_table.c.key = Mock()
        
        with patch('mdm.dataset.metadata.sa.MetaData'):
            with patch('mdm.dataset.metadata.Table', return_value=mock_metadata_table):
                with patch('mdm.dataset.metadata.select') as mock_select:
                    result = get_dataset_metadata("test_dataset", "key1", mock_engine)
                    
                    assert result == {"value": "test"}
                    mock_select.assert_called_once()

    def test_get_dataset_metadata_not_exists(self, mock_engine, mock_connection):
        """Test getting non-existent metadata value."""
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_result = Mock()
        mock_result.scalar.return_value = None
        mock_connection.execute.return_value = mock_result
        
        with patch('mdm.dataset.metadata.sa.MetaData'):
            with patch('mdm.dataset.metadata.Table'):
                with patch('mdm.dataset.metadata.select'):
                    result = get_dataset_metadata("test_dataset", "key1", mock_engine)
                    
                    assert result is None

    def test_get_dataset_metadata_table_not_exists(self, mock_engine):
        """Test getting metadata when table doesn't exist."""
        with patch('mdm.dataset.metadata.sa.MetaData'):
            with patch('mdm.dataset.metadata.Table', side_effect=SQLAlchemyError("Table not found")):
                result = get_dataset_metadata("test_dataset", "key1", mock_engine)
                
                assert result is None

    def test_store_column_metadata(self, mock_engine, mock_connection, sample_column_info):
        """Test storing column metadata."""
        mock_engine.begin.return_value.__enter__.return_value = mock_connection
        
        mock_columns_table = Mock()
        mock_columns_table.c.table_name = Mock()
        mock_columns_table.delete.return_value.where.return_value = Mock()
        mock_columns_table.insert.return_value = Mock()
        
        with patch('mdm.dataset.metadata.create_metadata_tables'):
            with patch('mdm.dataset.metadata.sa.MetaData'):
                with patch('mdm.dataset.metadata.Table', return_value=mock_columns_table):
                    store_column_metadata("test_dataset", "train", sample_column_info, mock_engine)
                    
                    # Should delete existing columns
                    mock_columns_table.delete.assert_called_once()
                    
                    # Should insert new columns
                    assert mock_connection.execute.call_count == 2  # Delete + Insert
                    
                    # Check insert data
                    insert_call = mock_connection.execute.call_args_list[1]
                    rows = insert_call[0][1]  # Second argument is the rows
                    assert len(rows) == 2
                    assert rows[0]["column_name"] == "id"
                    assert rows[0]["column_type"] == "id"
                    assert rows[1]["column_name"] == "name"
                    assert rows[1]["column_type"] == "text"

    def test_store_column_metadata_empty(self, mock_engine, mock_connection):
        """Test storing empty column list."""
        mock_engine.begin.return_value.__enter__.return_value = mock_connection
        
        mock_columns_table = Mock()
        mock_columns_table.c.table_name = Mock()
        mock_columns_table.delete.return_value.where.return_value = Mock()
        
        with patch('mdm.dataset.metadata.create_metadata_tables'):
            with patch('mdm.dataset.metadata.sa.MetaData'):
                with patch('mdm.dataset.metadata.Table', return_value=mock_columns_table):
                    store_column_metadata("test_dataset", "train", [], mock_engine)
                    
                    # Should only delete, not insert
                    assert mock_connection.execute.call_count == 1
                    mock_columns_table.delete.assert_called_once()

    def test_get_column_metadata_all_tables(self, mock_engine, mock_connection, sample_column_info):
        """Test getting column metadata for all tables."""
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        
        # Mock query results
        mock_rows = [
            Mock(
                column_name="id", dtype="int64", column_type="id",
                nullable=False, unique=True, missing_count=0,
                missing_ratio=0.0, cardinality=100, min_value="1",
                max_value="100", mean_value=50.5, std_value=28.87,
                sample_values=[1, 2, 3], description="Primary key"
            ),
            Mock(
                column_name="name", dtype="object", column_type="text",
                nullable=True, unique=False, missing_count=5,
                missing_ratio=0.05, cardinality=90, min_value=None,
                max_value=None, mean_value=None, std_value=None,
                sample_values=["Alice", "Bob"], description="Customer name"
            )
        ]
        mock_connection.execute.return_value.fetchall.return_value = mock_rows
        
        with patch('mdm.dataset.metadata.sa.MetaData'):
            with patch('mdm.dataset.metadata.Table'):
                with patch('mdm.dataset.metadata.select'):
                    result = get_column_metadata("test_dataset", engine=mock_engine)
                    
                    assert len(result) == 2
                    assert "id" in result
                    assert "name" in result
                    assert result["id"].column_type == ColumnType.ID
                    assert result["name"].column_type == ColumnType.TEXT

    def test_get_column_metadata_specific_table(self, mock_engine, mock_connection):
        """Test getting column metadata for specific table."""
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_connection.execute.return_value.fetchall.return_value = []
        
        mock_columns_table = Mock()
        mock_columns_table.c.table_name = Mock()
        
        with patch('mdm.dataset.metadata.sa.MetaData'):
            with patch('mdm.dataset.metadata.Table', return_value=mock_columns_table):
                with patch('mdm.dataset.metadata.select') as mock_select:
                    result = get_column_metadata("test_dataset", "train", mock_engine)
                    
                    # Should filter by table name
                    mock_select.return_value.where.assert_called_once()

    def test_get_column_metadata_table_not_exists(self, mock_engine):
        """Test getting columns when table doesn't exist."""
        with patch('mdm.dataset.metadata.sa.MetaData'):
            with patch('mdm.dataset.metadata.Table', side_effect=SQLAlchemyError("Table not found")):
                result = get_column_metadata("test_dataset", engine=mock_engine)
                
                assert result == {}

    def test_store_dataset_statistics(self, mock_engine, mock_connection, sample_statistics):
        """Test storing dataset statistics."""
        mock_engine.begin.return_value.__enter__.return_value = mock_connection
        
        mock_stats_table = Mock()
        mock_stats_table.insert.return_value.values.return_value = Mock()
        
        with patch('mdm.dataset.metadata.create_metadata_tables'):
            with patch('mdm.dataset.metadata.sa.MetaData'):
                with patch('mdm.dataset.metadata.Table', return_value=mock_stats_table):
                    store_dataset_statistics("test_dataset", "train", sample_statistics, mock_engine)
                    
                    # Should insert statistics
                    mock_stats_table.insert.assert_called_once()
                    mock_connection.execute.assert_called_once()

    def test_get_latest_statistics_exists(self, mock_engine, mock_connection, sample_statistics):
        """Test getting latest statistics when they exist."""
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        
        # Mock query result
        mock_row = Mock(
            row_count=1000, column_count=10, memory_usage_mb=5.2,
            missing_values={"name": 5}, column_types={"id": "int64"},
            numeric_columns=["id", "age"], categorical_columns=["category"],
            datetime_columns=["created_at"], text_columns=["name"],
            computed_at=datetime.now(timezone.utc)
        )
        mock_connection.execute.return_value.fetchone.return_value = mock_row
        
        mock_stats_table = Mock()
        mock_stats_table.c.table_name = Mock()
        mock_stats_table.c.computed_at = Mock()
        mock_stats_table.c.computed_at.desc.return_value = Mock()
        
        with patch('mdm.dataset.metadata.sa.MetaData'):
            with patch('mdm.dataset.metadata.Table', return_value=mock_stats_table):
                with patch('mdm.dataset.metadata.select') as mock_select:
                    result = get_latest_statistics("test_dataset", "train", mock_engine)
                    
                    assert isinstance(result, DatasetStatistics)
                    assert result.row_count == 1000
                    assert result.column_count == 10
                    
                    # Should order by computed_at desc and limit 1
                    mock_select.return_value.where.return_value.order_by.assert_called_once()
                    mock_select.return_value.where.return_value.order_by.return_value.limit.assert_called_once_with(1)

    def test_get_latest_statistics_not_exists(self, mock_engine, mock_connection):
        """Test getting statistics when none exist."""
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_connection.execute.return_value.fetchone.return_value = None
        
        with patch('mdm.dataset.metadata.sa.MetaData'):
            with patch('mdm.dataset.metadata.Table'):
                with patch('mdm.dataset.metadata.select'):
                    result = get_latest_statistics("test_dataset", "train", mock_engine)
                    
                    assert result is None

    def test_get_latest_statistics_table_not_exists(self, mock_engine):
        """Test getting statistics when table doesn't exist."""
        with patch('mdm.dataset.metadata.sa.MetaData'):
            with patch('mdm.dataset.metadata.Table', side_effect=SQLAlchemyError("Table not found")):
                result = get_latest_statistics("test_dataset", "train", mock_engine)
                
                assert result is None

    def test_store_extended_info(self, mock_engine, sample_extended_info):
        """Test storing extended dataset information."""
        with patch('mdm.dataset.metadata.store_dataset_metadata') as mock_store:
            store_extended_info("test_dataset", sample_extended_info, mock_engine)
            
            # Should store dataset info and files
            assert mock_store.call_count == 2
            
            # Check dataset info call
            dataset_call = mock_store.call_args_list[0]
            assert dataset_call[0][0] == "test_dataset"
            assert dataset_call[0][1] == "dataset_info"
            dataset_meta = dataset_call[0][2]
            assert dataset_meta["name"] == "test_dataset"
            assert dataset_meta["problem_type"] == "binary_classification"
            
            # Check files call
            files_call = mock_store.call_args_list[1]
            assert files_call[0][0] == "test_dataset"
            assert files_call[0][1] == "files"
            files_meta = files_call[0][2]
            assert "train" in files_meta
            assert files_meta["train"]["name"] == "train.csv"

    def test_get_extended_info_exists(self, mock_engine):
        """Test getting extended info when it exists."""
        # Mock metadata retrieval
        dataset_meta = {
            "name": "test_dataset",
            "display_name": "Test Dataset",
            "description": "A test dataset",
            "problem_type": "binary_classification",
            "target_column": "target",
            "id_columns": ["id"],
            "datetime_columns": ["created_at"],
            "feature_columns": ["feature1", "feature2"],
            "row_count": 1000,
            "memory_usage_mb": 5.2,
            "source": "/path/to/data.csv",
            "version": "1.0.0",
            "tags": ["test", "example"],
            "metadata": {"custom": "value"},
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": None
        }
        
        files_meta = {
            "train": {
                "path": "/path/to/train.csv",
                "name": "train.csv",
                "size_bytes": 1024000,
                "file_type": "csv",
                "format": "csv",
                "encoding": "utf-8",
                "row_count": 800,
                "column_count": 10,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "modified_at": None,
                "checksum": "abc123"
            }
        }
        
        with patch('mdm.dataset.metadata.get_dataset_metadata') as mock_get:
            mock_get.side_effect = [dataset_meta, files_meta]
            
            with patch('mdm.dataset.metadata.get_column_metadata', return_value={}):
                result = get_extended_info("test_dataset", mock_engine)
                
                assert isinstance(result, DatasetInfoExtended)
                assert result.name == "test_dataset"
                assert result.problem_type == ProblemType.BINARY_CLASSIFICATION
                assert "train" in result.files
                assert result.files["train"].name == "train.csv"

    def test_get_extended_info_not_exists(self, mock_engine):
        """Test getting extended info when it doesn't exist."""
        with patch('mdm.dataset.metadata.get_dataset_metadata', return_value=None):
            result = get_extended_info("test_dataset", mock_engine)
            
            assert result is None

    def test_get_extended_info_no_files(self, mock_engine):
        """Test getting extended info with no files metadata."""
        dataset_meta = {
            "name": "test_dataset",
            "display_name": "Test Dataset",
            "description": "A test dataset",
            "problem_type": None,
            "target_column": None,
            "id_columns": [],
            "datetime_columns": [],
            "feature_columns": [],
            "row_count": 0,
            "memory_usage_mb": 0,
            "source": "",
            "version": "1.0.0",
            "tags": [],
            "metadata": {},
            "created_at": datetime.now(timezone.utc).isoformat(),  # Required field
            "updated_at": None
        }
        
        with patch('mdm.dataset.metadata.get_dataset_metadata') as mock_get:
            mock_get.side_effect = [dataset_meta, None]  # No files metadata
            
            with patch('mdm.dataset.metadata.get_column_metadata', return_value={}):
                result = get_extended_info("test_dataset", mock_engine)
                
                assert isinstance(result, DatasetInfoExtended)
                assert result.name == "test_dataset"
                assert result.files == {}

    def test_datetime_handling_in_metadata(self, mock_engine, mock_connection):
        """Test proper datetime handling in metadata operations."""
        mock_engine.begin.return_value.__enter__.return_value = mock_connection
        
        # Test with timezone-aware datetime
        now = datetime.now(timezone.utc)
        
        mock_metadata_table = Mock()
        mock_metadata_table.insert.return_value.values.return_value = Mock()
        mock_metadata_table.insert.return_value.values.return_value.on_conflict_do_update.return_value = Mock()
        
        with patch('mdm.dataset.metadata.sa.MetaData'):
            with patch('mdm.dataset.metadata.Table', return_value=mock_metadata_table):
                with patch('mdm.dataset.metadata.create_metadata_tables'):
                    with patch('mdm.dataset.metadata.datetime') as mock_datetime:
                        mock_datetime.now.return_value = now
                        
                        store_dataset_metadata("test_dataset", "timestamp", now, mock_engine)
                        
                        # Should handle datetime properly
                        mock_connection.execute.assert_called_once()

    def test_column_type_enum_handling(self, mock_engine, mock_connection):
        """Test proper handling of ColumnType enum values."""
        mock_engine.begin.return_value.__enter__.return_value = mock_connection
        
        # Create column with enum type
        column = ColumnInfo(
            name="status",
            dtype="object",
            column_type=ColumnType.CATEGORICAL,  # Enum value
            nullable=True,
            unique=False,
            missing_count=0,
            missing_ratio=0.0,
            cardinality=3,
            sample_values=["active", "inactive", "pending"],
            description="Status column"
        )
        
        mock_columns_table = Mock()
        mock_columns_table.c.table_name = Mock()
        mock_columns_table.delete.return_value.where.return_value = Mock()
        mock_columns_table.insert.return_value = Mock()
        
        with patch('mdm.dataset.metadata.create_metadata_tables'):
            with patch('mdm.dataset.metadata.sa.MetaData'):
                with patch('mdm.dataset.metadata.Table', return_value=mock_columns_table):
                    store_column_metadata("test_dataset", "train", [column], mock_engine)
                    
                    # Check that enum value is properly converted
                    insert_call = mock_connection.execute.call_args_list[1]
                    rows = insert_call[0][1]
                    assert rows[0]["column_type"] == "categorical"

    def test_null_value_handling(self, mock_engine, mock_connection):
        """Test handling of null/None values in metadata."""
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        
        # Mock row with null values
        mock_row = Mock(
            row_count=1000, column_count=10, memory_usage_mb=5.2,
            missing_values=None,  # Null value
            column_types=None,    # Null value
            numeric_columns=None, # Null value
            categorical_columns=[],
            datetime_columns=[],
            text_columns=[],
            computed_at=datetime.now(timezone.utc)
        )
        mock_connection.execute.return_value.fetchone.return_value = mock_row
        
        with patch('mdm.dataset.metadata.sa.MetaData'):
            with patch('mdm.dataset.metadata.Table'):
                with patch('mdm.dataset.metadata.select'):
                    result = get_latest_statistics("test_dataset", "train", mock_engine)
                    
                    # Should handle null values gracefully
                    assert result.missing_values == {}
                    assert result.column_types == {}
                    assert result.numeric_columns == []

    def test_error_propagation(self, mock_engine):
        """Test that SQLAlchemy errors are properly handled."""
        mock_engine.begin.side_effect = SQLAlchemyError("Database connection failed")
        
        with patch('mdm.dataset.metadata.sa.MetaData'):
            with patch('mdm.dataset.metadata.Table'):
                with pytest.raises(SQLAlchemyError, match="Database connection failed"):
                    store_dataset_metadata("test_dataset", "key", "value", mock_engine)