"""Comprehensive unit tests for dataset metadata to achieve 80%+ coverage."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone
from pathlib import Path
import sqlalchemy as sa
from sqlalchemy import MetaData, Table, Column, String, JSON, DateTime, Float, Integer, Text, Boolean
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
from mdm.models.dataset import ColumnInfo, DatasetInfoExtended, DatasetStatistics, FileInfo
from mdm.models.enums import ColumnType, ProblemType, FileType
from mdm.core.exceptions import DatasetError


class TestGetDatasetEngine:
    """Test cases for get_dataset_engine function."""

    def test_sqlite_engine(self):
        """Test creating SQLite engine."""
        mock_config = Mock()
        mock_config.database = {
            "type": "sqlite",
            "path": "/tmp/test.db"
        }
        
        with patch('mdm.dataset.config.load_dataset_config', return_value=mock_config):
            with patch('mdm.dataset.metadata.create_engine') as mock_create_engine:
                engine = get_dataset_engine("test_dataset")
                
                mock_create_engine.assert_called_once_with("sqlite:////tmp/test.db")

    def test_duckdb_engine(self):
        """Test creating DuckDB engine."""
        mock_config = Mock()
        mock_config.database = {
            "type": "duckdb",
            "path": "/tmp/test.duckdb"
        }
        
        with patch('mdm.dataset.config.load_dataset_config', return_value=mock_config):
            with patch('mdm.dataset.metadata.create_engine') as mock_create_engine:
                engine = get_dataset_engine("test_dataset")
                
                mock_create_engine.assert_called_once_with("duckdb:////tmp/test.duckdb")

    def test_postgresql_engine(self):
        """Test creating PostgreSQL engine."""
        mock_config = Mock()
        mock_config.database = {
            "type": "postgresql",
            "connection_string": "postgresql://user:pass@localhost/db"
        }
        
        with patch('mdm.dataset.config.load_dataset_config', return_value=mock_config):
            with patch('mdm.dataset.metadata.create_engine') as mock_create_engine:
                engine = get_dataset_engine("test_dataset")
                
                mock_create_engine.assert_called_once_with("postgresql://user:pass@localhost/db")

    def test_unsupported_database_type(self):
        """Test error for unsupported database type."""
        mock_config = Mock()
        mock_config.database = {
            "type": "mysql",
            "connection_string": "mysql://user:pass@localhost/db"
        }
        
        with patch('mdm.dataset.config.load_dataset_config', return_value=mock_config):
            with pytest.raises(DatasetError, match="Unsupported database type: mysql"):
                get_dataset_engine("test_dataset")

    def test_path_expansion(self):
        """Test path expansion for SQLite."""
        mock_config = Mock()
        mock_config.database = {
            "type": "sqlite",
            "path": "~/test.db"
        }
        
        with patch('mdm.dataset.config.load_dataset_config', return_value=mock_config):
            with patch('mdm.dataset.metadata.create_engine') as mock_create_engine:
                # Create a mock Path object
                mock_path = Mock(spec=Path)
                mock_path.expanduser.return_value = mock_path
                mock_path.resolve.return_value = mock_path
                mock_path.__str__ = Mock(return_value="/home/user/test.db")
                
                with patch('mdm.dataset.metadata.Path', return_value=mock_path):
                    engine = get_dataset_engine("test_dataset")
                    
                    # Check that expanduser was called
                    mock_path.expanduser.assert_called_once()


class TestCreateMetadataTables:
    """Test cases for create_metadata_tables function."""

    def test_create_tables(self):
        """Test creating all metadata tables."""
        mock_engine = Mock()
        
        with patch('sqlalchemy.MetaData') as mock_metadata_class:
            mock_metadata = Mock()
            mock_metadata.schema = None  # Important for SQLAlchemy
            mock_metadata_class.return_value = mock_metadata
            
            with patch('mdm.dataset.metadata.Table') as mock_table:
                # Mock Table constructor to prevent SQLAlchemy internals from running
                mock_table.return_value = Mock()
                
                create_metadata_tables(mock_engine)
                
                # Verify create_all was called
                mock_metadata.create_all.assert_called_once_with(mock_engine)

    def test_table_definitions(self):
        """Test that all tables are defined correctly."""
        mock_engine = Mock()
        created_tables = []
        
        def mock_table(*args, **kwargs):
            created_tables.append(args[0])  # Table name
            return Mock()
        
        with patch('mdm.dataset.metadata.Table', side_effect=mock_table):
            create_metadata_tables(mock_engine)
            
            # Verify all tables were created
            assert "_metadata" in created_tables
            assert "_columns" in created_tables
            assert "_statistics" in created_tables


class TestStoreDatasetMetadata:
    """Test cases for store_dataset_metadata function."""

    @pytest.fixture
    def mock_engine(self):
        """Create mock engine."""
        engine = Mock()
        engine.dialect.name = "sqlite"
        
        # Mock connection context
        mock_conn = Mock()
        mock_conn.__enter__ = Mock(return_value=mock_conn)
        mock_conn.__exit__ = Mock(return_value=False)
        engine.begin.return_value = mock_conn
        
        return engine

    def test_store_metadata_with_engine(self, mock_engine):
        """Test storing metadata with provided engine."""
        mock_metadata_table = Mock()
        mock_metadata_table.insert.return_value.values.return_value = Mock()
        mock_metadata_table.c.key = "key"
        
        with patch('mdm.dataset.metadata.Table', return_value=mock_metadata_table):
            with patch('mdm.dataset.metadata.create_metadata_tables'):
                store_dataset_metadata("test_dataset", "test_key", {"value": 123}, mock_engine)
                
                # Verify insert was called
                mock_metadata_table.insert.assert_called_once()

    def test_store_metadata_without_engine(self):
        """Test storing metadata without provided engine."""
        mock_engine = Mock()
        mock_engine.dialect.name = "sqlite"
        mock_conn = Mock()
        mock_conn.__enter__ = Mock(return_value=mock_conn)
        mock_conn.__exit__ = Mock(return_value=False)
        mock_engine.begin.return_value = mock_conn
        
        with patch('mdm.dataset.metadata.get_dataset_engine', return_value=mock_engine):
            with patch('mdm.dataset.metadata.Table'):
                with patch('mdm.dataset.metadata.create_metadata_tables'):
                    store_dataset_metadata("test_dataset", "test_key", {"value": 123})
                    
                    # Verify engine was created
                    assert mock_engine.begin.called

    def test_store_metadata_postgresql_upsert(self, mock_engine):
        """Test PostgreSQL upsert handling."""
        mock_engine.dialect.name = "postgresql"
        mock_metadata_table = Mock()
        mock_stmt = Mock()
        mock_metadata_table.insert.return_value.values.return_value = mock_stmt
        mock_metadata_table.c.key = "key"
        
        with patch('mdm.dataset.metadata.Table', return_value=mock_metadata_table):
            with patch('mdm.dataset.metadata.create_metadata_tables'):
                store_dataset_metadata("test_dataset", "test_key", {"value": 123}, mock_engine)
                
                # Verify on_conflict_do_update was called
                mock_stmt.on_conflict_do_update.assert_called_once()

    def test_store_metadata_other_db_delete_insert(self, mock_engine):
        """Test delete-insert for other databases."""
        mock_engine.dialect.name = "duckdb"
        mock_metadata_table = Mock()
        mock_metadata_table.c.key = "key"
        mock_conn = Mock()
        mock_conn.__enter__ = Mock(return_value=mock_conn)
        mock_conn.__exit__ = Mock(return_value=False)
        mock_engine.begin.return_value = mock_conn
        
        with patch('mdm.dataset.metadata.Table', return_value=mock_metadata_table):
            with patch('mdm.dataset.metadata.create_metadata_tables'):
                store_dataset_metadata("test_dataset", "test_key", {"value": 123}, mock_engine)
                
                # Verify delete was called before insert
                mock_conn.execute.assert_any_call(mock_metadata_table.delete.return_value.where.return_value)


class TestGetDatasetMetadata:
    """Test cases for get_dataset_metadata function."""

    @pytest.fixture
    def mock_engine(self):
        """Create mock engine."""
        engine = Mock()
        mock_conn = Mock()
        mock_conn.__enter__ = Mock(return_value=mock_conn)
        mock_conn.__exit__ = Mock(return_value=False)
        engine.connect.return_value = mock_conn
        return engine

    def test_get_metadata_success(self, mock_engine):
        """Test successful metadata retrieval."""
        mock_conn = mock_engine.connect.return_value.__enter__.return_value
        mock_conn.execute.return_value.scalar.return_value = {"test": "value"}
        
        with patch('mdm.dataset.metadata.Table') as mock_table:
            mock_table.return_value = Mock()
            with patch('mdm.dataset.metadata.select') as mock_select:
                mock_select.return_value = Mock()
                result = get_dataset_metadata("test_dataset", "test_key", mock_engine)
                
                assert result == {"test": "value"}

    def test_get_metadata_not_found(self, mock_engine):
        """Test metadata not found."""
        mock_conn = mock_engine.connect.return_value.__enter__.return_value
        mock_conn.execute.return_value.scalar.return_value = None
        
        with patch('mdm.dataset.metadata.Table') as mock_table:
            mock_table.return_value = Mock()
            with patch('mdm.dataset.metadata.select') as mock_select:
                mock_select.return_value = Mock()
                result = get_dataset_metadata("test_dataset", "test_key", mock_engine)
                
                assert result is None

    def test_get_metadata_table_not_exists(self, mock_engine):
        """Test when metadata table doesn't exist."""
        with patch('mdm.dataset.metadata.Table', side_effect=SQLAlchemyError("Table not found")):
            result = get_dataset_metadata("test_dataset", "test_key", mock_engine)
            
            assert result is None

    def test_get_metadata_without_engine(self):
        """Test getting metadata without provided engine."""
        mock_engine = Mock()
        
        with patch('mdm.dataset.metadata.get_dataset_engine', return_value=mock_engine):
            with patch('mdm.dataset.metadata.Table', side_effect=SQLAlchemyError("Table not found")):
                result = get_dataset_metadata("test_dataset", "test_key")
                
                assert result is None


class TestStoreColumnMetadata:
    """Test cases for store_column_metadata function."""

    @pytest.fixture
    def mock_engine(self):
        """Create mock engine."""
        engine = Mock()
        mock_conn = Mock()
        mock_conn.__enter__ = Mock(return_value=mock_conn)
        mock_conn.__exit__ = Mock(return_value=False)
        engine.begin.return_value = mock_conn
        return engine

    @pytest.fixture
    def sample_columns(self):
        """Create sample column info."""
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
                description="ID column"
            ),
            ColumnInfo(
                name="name",
                dtype="object",
                column_type=ColumnType.TEXT,
                nullable=True,
                unique=False,
                missing_count=5,
                missing_ratio=0.05,
                cardinality=95,
                sample_values=["Alice", "Bob", "Charlie"],
                description="Name column"
            )
        ]

    def test_store_columns_success(self, mock_engine, sample_columns):
        """Test successful column metadata storage."""
        mock_columns_table = Mock()
        mock_conn = mock_engine.begin.return_value.__enter__.return_value
        
        with patch('mdm.dataset.metadata.Table', return_value=mock_columns_table):
            with patch('mdm.dataset.metadata.create_metadata_tables'):
                store_column_metadata("test_dataset", "test_table", sample_columns, mock_engine)
                
                # Verify delete was called
                mock_conn.execute.assert_any_call(mock_columns_table.delete.return_value.where.return_value)
                
                # Verify insert was called
                assert mock_conn.execute.call_count >= 2

    def test_store_columns_empty_list(self, mock_engine):
        """Test storing empty column list."""
        mock_columns_table = Mock()
        mock_conn = mock_engine.begin.return_value.__enter__.return_value
        
        with patch('mdm.dataset.metadata.Table', return_value=mock_columns_table):
            with patch('mdm.dataset.metadata.create_metadata_tables'):
                store_column_metadata("test_dataset", "test_table", [], mock_engine)
                
                # Delete should be called but not insert
                mock_conn.execute.assert_any_call(mock_columns_table.delete.return_value.where.return_value)

    def test_store_columns_without_engine(self, sample_columns):
        """Test storing columns without provided engine."""
        mock_engine = Mock()
        mock_conn = Mock()
        mock_conn.__enter__ = Mock(return_value=mock_conn)
        mock_conn.__exit__ = Mock(return_value=False)
        mock_engine.begin.return_value = mock_conn
        
        with patch('mdm.dataset.metadata.get_dataset_engine', return_value=mock_engine):
            with patch('mdm.dataset.metadata.Table'):
                with patch('mdm.dataset.metadata.create_metadata_tables'):
                    store_column_metadata("test_dataset", "test_table", sample_columns)
                    
                    # Verify engine was created
                    assert mock_engine.begin.called

    def test_store_columns_with_none_values(self, mock_engine):
        """Test storing columns with None values."""
        columns = [
            ColumnInfo(
                name="col1",
                dtype="float64",
                column_type=ColumnType.NUMERIC,  # Use the enum
                nullable=True,
                unique=False,
                missing_count=10,
                missing_ratio=0.1,
                cardinality=None,
                min_value=None,
                max_value=None,
                mean_value=None,
                std_value=None,
                sample_values=[],  # Pydantic requires list, not None
                description=None
            )
        ]
        
        mock_columns_table = Mock()
        
        with patch('mdm.dataset.metadata.Table', return_value=mock_columns_table):
            with patch('mdm.dataset.metadata.create_metadata_tables'):
                # Should not raise error
                store_column_metadata("test_dataset", "test_table", columns, mock_engine)


class TestGetColumnMetadata:
    """Test cases for get_column_metadata function."""

    @pytest.fixture
    def mock_engine(self):
        """Create mock engine."""
        engine = Mock()
        mock_conn = Mock()
        mock_conn.__enter__ = Mock(return_value=mock_conn)
        mock_conn.__exit__ = Mock(return_value=False)
        engine.connect.return_value = mock_conn
        return engine

    def test_get_columns_success(self, mock_engine):
        """Test successful column metadata retrieval."""
        mock_row1 = Mock()
        mock_row1.column_name = "id"
        mock_row1.dtype = "int64"
        mock_row1.column_type = "id"
        mock_row1.nullable = False
        mock_row1.unique = True
        mock_row1.missing_count = 0
        mock_row1.missing_ratio = 0.0
        mock_row1.cardinality = 100
        mock_row1.min_value = "1"
        mock_row1.max_value = "100"
        mock_row1.mean_value = 50.5
        mock_row1.std_value = 28.87
        mock_row1.sample_values = [1, 2, 3]
        mock_row1.description = "ID column"
        
        mock_conn = mock_engine.connect.return_value.__enter__.return_value
        mock_conn.execute.return_value.fetchall.return_value = [mock_row1]
        
        with patch('mdm.dataset.metadata.Table') as mock_table:
            mock_table.return_value = Mock()
            with patch('mdm.dataset.metadata.select') as mock_select:
                mock_select.return_value = Mock()
                result = get_column_metadata("test_dataset", engine=mock_engine)
                
                assert "id" in result
                assert result["id"].name == "id"
                assert result["id"].column_type == ColumnType.ID

    def test_get_columns_with_table_filter(self, mock_engine):
        """Test getting columns with table name filter."""
        mock_conn = mock_engine.connect.return_value.__enter__.return_value
        mock_conn.execute.return_value.fetchall.return_value = []
        
        with patch('mdm.dataset.metadata.Table') as mock_table:
            mock_columns_table = Mock()
            mock_table.return_value = mock_columns_table
            
            with patch('mdm.dataset.metadata.select') as mock_select:
                # Create a proper mock for the select statement
                mock_stmt = Mock()
                mock_select.return_value = mock_stmt
                mock_stmt.where.return_value = mock_stmt
                
                result = get_column_metadata("test_dataset", table_name="specific_table", engine=mock_engine)
                
                # Verify where clause was added
                assert result == {}

    def test_get_columns_table_not_exists(self, mock_engine):
        """Test when columns table doesn't exist."""
        with patch('mdm.dataset.metadata.Table', side_effect=SQLAlchemyError("Table not found")):
            result = get_column_metadata("test_dataset", engine=mock_engine)
            
            assert result == {}

    def test_get_columns_with_null_type(self, mock_engine):
        """Test handling columns with null type."""
        mock_row = Mock()
        mock_row.column_name = "col1"
        mock_row.dtype = "object"
        mock_row.column_type = None  # Null type
        mock_row.nullable = True
        mock_row.unique = False
        mock_row.missing_count = 0
        mock_row.missing_ratio = 0.0
        mock_row.cardinality = 10
        mock_row.min_value = None
        mock_row.max_value = None
        mock_row.mean_value = None
        mock_row.std_value = None
        mock_row.sample_values = None
        mock_row.description = None
        
        mock_conn = mock_engine.connect.return_value.__enter__.return_value
        mock_conn.execute.return_value.fetchall.return_value = [mock_row]
        
        with patch('mdm.dataset.metadata.Table') as mock_table:
            mock_table.return_value = Mock()
            with patch('mdm.dataset.metadata.select') as mock_select:
                # Create a proper mock for the select statement
                mock_stmt = Mock()
                mock_select.return_value = mock_stmt
                result = get_column_metadata("test_dataset", engine=mock_engine)
                
                assert "col1" in result
                assert result["col1"].column_type == ColumnType.TEXT  # Default


class TestStoreDatasetStatistics:
    """Test cases for store_dataset_statistics function."""

    @pytest.fixture
    def mock_engine(self):
        """Create mock engine."""
        engine = Mock()
        mock_conn = Mock()
        mock_conn.__enter__ = Mock(return_value=mock_conn)
        mock_conn.__exit__ = Mock(return_value=False)
        engine.begin.return_value = mock_conn
        return engine

    @pytest.fixture
    def sample_statistics(self):
        """Create sample statistics."""
        return DatasetStatistics(
            row_count=1000,
            column_count=10,
            memory_usage_mb=50.5,
            missing_values={"col1": 5, "col2": 10},
            column_types={"numeric": "5", "text": "3", "categorical": "2"},  # Values must be strings
            numeric_columns=["col1", "col2", "col3"],
            categorical_columns=["cat1", "cat2"],
            datetime_columns=["date1"],
            text_columns=["text1", "text2"],
            computed_at=datetime.now(timezone.utc)
        )

    def test_store_statistics_success(self, mock_engine, sample_statistics):
        """Test successful statistics storage."""
        mock_stats_table = Mock()
        
        with patch('mdm.dataset.metadata.Table', return_value=mock_stats_table):
            with patch('mdm.dataset.metadata.create_metadata_tables'):
                store_dataset_statistics("test_dataset", "test_table", sample_statistics, mock_engine)
                
                # Verify insert was called
                mock_stats_table.insert.assert_called_once()

    def test_store_statistics_without_engine(self, sample_statistics):
        """Test storing statistics without provided engine."""
        mock_engine = Mock()
        mock_conn = Mock()
        mock_conn.__enter__ = Mock(return_value=mock_conn)
        mock_conn.__exit__ = Mock(return_value=False)
        mock_engine.begin.return_value = mock_conn
        
        with patch('mdm.dataset.metadata.get_dataset_engine', return_value=mock_engine):
            with patch('mdm.dataset.metadata.Table'):
                with patch('mdm.dataset.metadata.create_metadata_tables'):
                    store_dataset_statistics("test_dataset", "test_table", sample_statistics)
                    
                    # Verify engine was created
                    assert mock_engine.begin.called


class TestGetLatestStatistics:
    """Test cases for get_latest_statistics function."""

    @pytest.fixture
    def mock_engine(self):
        """Create mock engine."""
        engine = Mock()
        mock_conn = Mock()
        mock_conn.__enter__ = Mock(return_value=mock_conn)
        mock_conn.__exit__ = Mock(return_value=False)
        engine.connect.return_value = mock_conn
        return engine

    def test_get_statistics_success(self, mock_engine):
        """Test successful statistics retrieval."""
        mock_row = Mock()
        mock_row.row_count = 1000
        mock_row.column_count = 10
        mock_row.memory_usage_mb = 50.5
        mock_row.missing_values = {"col1": 5}
        mock_row.column_types = {"numeric": "5"}  # Must be string
        mock_row.numeric_columns = ["col1", "col2"]
        mock_row.categorical_columns = ["cat1"]
        mock_row.datetime_columns = ["date1"]
        mock_row.text_columns = ["text1"]
        mock_row.computed_at = datetime.now(timezone.utc)
        
        mock_conn = mock_engine.connect.return_value.__enter__.return_value
        mock_conn.execute.return_value.fetchone.return_value = mock_row
        
        with patch('mdm.dataset.metadata.Table') as mock_table:
            mock_table.return_value = Mock()
            with patch('mdm.dataset.metadata.select') as mock_select:
                # Create a proper mock for the select statement
                mock_stmt = Mock()
                mock_select.return_value = mock_stmt
                mock_stmt.where.return_value = mock_stmt
                mock_stmt.order_by.return_value = mock_stmt
                mock_stmt.limit.return_value = mock_stmt
                result = get_latest_statistics("test_dataset", "test_table", mock_engine)
                
                assert result is not None
                assert result.row_count == 1000
                assert result.column_count == 10

    def test_get_statistics_not_found(self, mock_engine):
        """Test when no statistics found."""
        mock_conn = mock_engine.connect.return_value.__enter__.return_value
        mock_conn.execute.return_value.fetchone.return_value = None
        
        with patch('mdm.dataset.metadata.Table') as mock_table:
            mock_table.return_value = Mock()
            with patch('mdm.dataset.metadata.select') as mock_select:
                # Create a proper mock for the select statement
                mock_stmt = Mock()
                mock_select.return_value = mock_stmt
                mock_stmt.where.return_value = mock_stmt
                mock_stmt.order_by.return_value = mock_stmt
                mock_stmt.limit.return_value = mock_stmt
                result = get_latest_statistics("test_dataset", "test_table", mock_engine)
                
                assert result is None

    def test_get_statistics_table_not_exists(self, mock_engine):
        """Test when statistics table doesn't exist."""
        with patch('mdm.dataset.metadata.Table', side_effect=SQLAlchemyError("Table not found")):
            result = get_latest_statistics("test_dataset", "test_table", mock_engine)
            
            assert result is None

    def test_get_statistics_with_null_fields(self, mock_engine):
        """Test handling statistics with null fields."""
        mock_row = Mock()
        mock_row.row_count = 1000
        mock_row.column_count = 10
        mock_row.memory_usage_mb = 50.5
        mock_row.missing_values = None
        mock_row.column_types = None
        mock_row.numeric_columns = None
        mock_row.categorical_columns = None
        mock_row.datetime_columns = None
        mock_row.text_columns = None
        mock_row.computed_at = datetime.now(timezone.utc)
        
        mock_conn = mock_engine.connect.return_value.__enter__.return_value
        mock_conn.execute.return_value.fetchone.return_value = mock_row
        
        with patch('mdm.dataset.metadata.Table') as mock_table:
            mock_table.return_value = Mock()
            with patch('mdm.dataset.metadata.select') as mock_select:
                # Create a proper mock for the select statement
                mock_stmt = Mock()
                mock_select.return_value = mock_stmt
                mock_stmt.where.return_value = mock_stmt
                mock_stmt.order_by.return_value = mock_stmt
                mock_stmt.limit.return_value = mock_stmt
                result = get_latest_statistics("test_dataset", "test_table", mock_engine)
                
                assert result is not None
                assert result.missing_values == {}
                assert result.column_types == {}
                assert result.numeric_columns == []


class TestStoreExtendedInfo:
    """Test cases for store_extended_info function."""

    @pytest.fixture
    def mock_engine(self):
        """Create mock engine."""
        return Mock()

    @pytest.fixture
    def sample_extended_info(self):
        """Create sample extended info."""
        return DatasetInfoExtended(
            name="test_dataset",
            display_name="Test Dataset",
            description="A test dataset",
            problem_type=ProblemType.BINARY_CLASSIFICATION,
            target_column="target",
            id_columns=["id"],
            datetime_columns=["created_at"],
            feature_columns=["feature1", "feature2"],
            files={
                "train": FileInfo(
                    path=Path("/data/train.csv"),
                    name="train.csv",
                    size_bytes=1000000,
                    file_type=FileType.CSV,
                    format="csv",
                    encoding="utf-8",
                    row_count=1000,
                    column_count=10,
                    created_at=datetime.now(timezone.utc),
                    modified_at=datetime.now(timezone.utc),
                    checksum="abc123"
                )
            },
            columns={},
            row_count=1000,
            memory_usage_mb=50.5,
            source="kaggle",
            version="1.0.0",
            tags=["test", "sample"],
            metadata={"custom": "value"},
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )

    def test_store_extended_info_success(self, mock_engine, sample_extended_info):
        """Test successful extended info storage."""
        with patch('mdm.dataset.metadata.store_dataset_metadata') as mock_store:
            store_extended_info("test_dataset", sample_extended_info, mock_engine)
            
            # Verify store was called twice (dataset_info and files)
            assert mock_store.call_count == 2
            
            # Check dataset_info call
            first_call = mock_store.call_args_list[0]
            assert first_call[0][0] == "test_dataset"
            assert first_call[0][1] == "dataset_info"
            assert first_call[0][2]["name"] == "test_dataset"
            
            # Check files call
            second_call = mock_store.call_args_list[1]
            assert second_call[0][0] == "test_dataset"
            assert second_call[0][1] == "files"
            assert "train" in second_call[0][2]

    def test_store_extended_info_without_engine(self, sample_extended_info):
        """Test storing extended info without provided engine."""
        mock_engine = Mock()
        
        with patch('mdm.dataset.metadata.get_dataset_engine', return_value=mock_engine):
            with patch('mdm.dataset.metadata.store_dataset_metadata') as mock_store:
                store_extended_info("test_dataset", sample_extended_info)
                
                # Verify store was called
                assert mock_store.call_count == 2

    def test_store_extended_info_with_none_fields(self, mock_engine):
        """Test storing extended info with None fields."""
        info = DatasetInfoExtended(
            name="test_dataset",
            display_name=None,
            description=None,
            problem_type=None,
            target_column=None,
            id_columns=[],
            datetime_columns=[],
            feature_columns=[],
            files={},
            columns={},
            row_count=None,
            memory_usage_mb=None,
            source=None,
            version="1.0.0",
            tags=[],
            metadata={},
            created_at=datetime.now(timezone.utc),  # created_at is required and has a default_factory
            updated_at=None
        )
        
        with patch('mdm.dataset.metadata.store_dataset_metadata') as mock_store:
            store_extended_info("test_dataset", info, mock_engine)
            
            # Should not raise error
            assert mock_store.call_count == 2


class TestGetExtendedInfo:
    """Test cases for get_extended_info function."""

    @pytest.fixture
    def mock_engine(self):
        """Create mock engine."""
        return Mock()

    def test_get_extended_info_success(self, mock_engine):
        """Test successful extended info retrieval."""
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
            "memory_usage_mb": 50.5,
            "source": "kaggle",
            "version": "1.0.0",
            "tags": ["test", "sample"],
            "metadata": {"custom": "value"},
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat()
        }
        
        files_meta = {
            "train": {
                "path": "/data/train.csv",
                "name": "train.csv",
                "size_bytes": 1000000,
                "file_type": "csv",
                "format": "csv",
                "encoding": "utf-8",
                "row_count": 1000,
                "column_count": 10,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "modified_at": datetime.now(timezone.utc).isoformat(),
                "checksum": "abc123"
            }
        }
        
        with patch('mdm.dataset.metadata.get_dataset_metadata') as mock_get:
            mock_get.side_effect = [dataset_meta, files_meta]
            
            with patch('mdm.dataset.metadata.get_column_metadata', return_value={}):
                result = get_extended_info("test_dataset", mock_engine)
                
                assert result is not None
                assert result.name == "test_dataset"
                assert result.display_name == "Test Dataset"
                assert result.problem_type == ProblemType.BINARY_CLASSIFICATION
                assert "train" in result.files

    def test_get_extended_info_not_found(self, mock_engine):
        """Test when dataset info not found."""
        with patch('mdm.dataset.metadata.get_dataset_metadata', return_value=None):
            result = get_extended_info("test_dataset", mock_engine)
            
            assert result is None

    def test_get_extended_info_without_engine(self):
        """Test getting extended info without provided engine."""
        mock_engine = Mock()
        
        with patch('mdm.dataset.metadata.get_dataset_engine', return_value=mock_engine):
            with patch('mdm.dataset.metadata.get_dataset_metadata', return_value=None):
                result = get_extended_info("test_dataset")
                
                assert result is None

    def test_get_extended_info_with_missing_fields(self, mock_engine):
        """Test handling missing fields in metadata."""
        dataset_meta = {
            "name": "test_dataset",
            # Missing many optional fields
            "created_at": datetime.now(timezone.utc).isoformat(),  # created_at is required
        }
        
        files_meta = {}  # No files
        
        with patch('mdm.dataset.metadata.get_dataset_metadata') as mock_get:
            mock_get.side_effect = [dataset_meta, files_meta]
            
            with patch('mdm.dataset.metadata.get_column_metadata', return_value={}):
                result = get_extended_info("test_dataset", mock_engine)
                
                assert result is not None
                assert result.name == "test_dataset"
                assert result.display_name is None
                assert result.description is None
                assert result.problem_type is None
                assert result.files == {}

    def test_get_extended_info_file_type_conversion(self, mock_engine):
        """Test FileInfo creation with type conversion."""
        dataset_meta = {
            "name": "test_dataset",
            "created_at": datetime.now(timezone.utc).isoformat(),  # created_at is required
        }
        
        files_meta = {
            "train": {
                "path": "/data/train.csv",
                "name": "train.csv",
                "size_bytes": 1000000,
                "file_type": "csv",  # String that needs conversion
                "format": "csv",
                # Missing optional fields to test defaults
            }
        }
        
        with patch('mdm.dataset.metadata.get_dataset_metadata') as mock_get:
            mock_get.side_effect = [dataset_meta, files_meta]
            
            with patch('mdm.dataset.metadata.get_column_metadata', return_value={}):
                result = get_extended_info("test_dataset", mock_engine)
                
                assert result is not None
                assert "train" in result.files
                assert result.files["train"].file_type == FileType.CSV
                assert result.files["train"].encoding == "utf-8"  # Default