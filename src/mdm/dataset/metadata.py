"""Dataset metadata handling - stores metadata within each dataset's database."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import sqlalchemy as sa
from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Table,
    Text,
    create_engine,
    select,
)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from mdm.core.exceptions import DatasetError
from mdm.models.dataset import ColumnInfo, DatasetInfoExtended, DatasetStatistics
from mdm.models.enums import ColumnType


def get_dataset_engine(dataset_name: str) -> Engine:
    """Get SQLAlchemy engine for dataset database.
    
    Args:
        dataset_name: Name of the dataset
    
    Returns:
        SQLAlchemy engine
    """
    from mdm.dataset.config import load_dataset_config

    config = load_dataset_config(dataset_name)
    db_config = config.database

    if db_config["type"] == "sqlite":
        db_path = Path(db_config["path"]).expanduser().resolve()
        return create_engine(f"sqlite:///{db_path}")
    if db_config["type"] == "duckdb":
        db_path = Path(db_config["path"]).expanduser().resolve()
        # DuckDB requires duckdb-engine package
        return create_engine(f"duckdb:///{db_path}")
    if db_config["type"] == "postgresql":
        return create_engine(db_config["connection_string"])
    raise DatasetError(f"Unsupported database type: {db_config['type']}")


def create_metadata_tables(engine: Engine) -> None:
    """Create metadata tables in the dataset database.
    
    Args:
        engine: SQLAlchemy engine
    """
    metadata = sa.MetaData()

    # Dataset metadata table
    Table(
        "_metadata",
        metadata,
        Column("key", String(255), primary_key=True),
        Column("value", JSON, nullable=True),
        Column("created_at", DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)),
        Column("updated_at", DateTime(timezone=True), onupdate=lambda: datetime.now(timezone.utc)),
    )

    # Column metadata table
    Table(
        "_columns",
        metadata,
        Column("table_name", String(255), primary_key=True),
        Column("column_name", String(255), primary_key=True),
        Column("dtype", String(100)),
        Column("column_type", String(50)),
        Column("nullable", sa.Boolean, default=True),
        Column("unique", sa.Boolean, default=False),
        Column("missing_count", Integer, default=0),
        Column("missing_ratio", Float, default=0.0),
        Column("cardinality", Integer),
        Column("min_value", Text),
        Column("max_value", Text),
        Column("mean_value", Float),
        Column("std_value", Float),
        Column("sample_values", JSON),
        Column("description", Text),
        Column("metadata", JSON),
        Column("created_at", DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)),
        Column("updated_at", DateTime(timezone=True), onupdate=lambda: datetime.now(timezone.utc)),
    )

    # Statistics table
    Table(
        "_statistics",
        metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("table_name", String(255)),
        Column("row_count", Integer),
        Column("column_count", Integer),
        Column("memory_usage_mb", Float),
        Column("missing_values", JSON),
        Column("column_types", JSON),
        Column("numeric_columns", JSON),
        Column("categorical_columns", JSON),
        Column("datetime_columns", JSON),
        Column("text_columns", JSON),
        Column("computed_at", DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)),
    )

    # Create tables
    metadata.create_all(engine)


def store_dataset_metadata(
    dataset_name: str,
    key: str,
    value: Any,
    engine: Optional[Engine] = None
) -> None:
    """Store metadata key-value pair in dataset database.
    
    Args:
        dataset_name: Name of the dataset
        key: Metadata key
        value: Metadata value (will be JSON serialized)
        engine: Optional SQLAlchemy engine
    """
    if engine is None:
        engine = get_dataset_engine(dataset_name)

    metadata = sa.MetaData()
    metadata_table = Table("_metadata", metadata, autoload_with=engine)

    # Ensure tables exist
    create_metadata_tables(engine)

    with engine.begin() as conn:
        # Upsert the metadata
        stmt = metadata_table.insert().values(
            key=key,
            value=value,
            created_at=datetime.now(timezone.utc)
        )

        # Handle conflict (update if exists)
        if engine.dialect.name == "postgresql" or engine.dialect.name == "sqlite":
            stmt = stmt.on_conflict_do_update(
                index_elements=["key"],
                set_=dict(value=value, updated_at=datetime.now(timezone.utc))
            )
        else:
            # For other databases, delete and insert
            conn.execute(metadata_table.delete().where(metadata_table.c.key == key))

        conn.execute(stmt)


def get_dataset_metadata(
    dataset_name: str,
    key: str,
    engine: Optional[Engine] = None
) -> Optional[Any]:
    """Get metadata value from dataset database.
    
    Args:
        dataset_name: Name of the dataset
        key: Metadata key
        engine: Optional SQLAlchemy engine
    
    Returns:
        Metadata value or None if not found
    """
    if engine is None:
        engine = get_dataset_engine(dataset_name)

    metadata = sa.MetaData()

    try:
        metadata_table = Table("_metadata", metadata, autoload_with=engine)
    except SQLAlchemyError:
        # Table doesn't exist
        return None

    with engine.connect() as conn:
        stmt = select(metadata_table.c.value).where(metadata_table.c.key == key)
        result = conn.execute(stmt).scalar()
        return result


def store_column_metadata(
    dataset_name: str,
    table_name: str,
    columns: List[ColumnInfo],
    engine: Optional[Engine] = None
) -> None:
    """Store column metadata in dataset database.
    
    Args:
        dataset_name: Name of the dataset
        table_name: Name of the table
        columns: List of ColumnInfo objects
        engine: Optional SQLAlchemy engine
    """
    if engine is None:
        engine = get_dataset_engine(dataset_name)

    # Ensure tables exist
    create_metadata_tables(engine)

    metadata = sa.MetaData()
    columns_table = Table("_columns", metadata, autoload_with=engine)

    with engine.begin() as conn:
        # Delete existing columns for this table
        conn.execute(columns_table.delete().where(columns_table.c.table_name == table_name))

        # Insert new column metadata
        rows = []
        for col in columns:
            rows.append({
                "table_name": table_name,
                "column_name": col.name,
                "dtype": col.dtype,
                "column_type": col.column_type.value if isinstance(col.column_type, ColumnType) else col.column_type,
                "nullable": col.nullable,
                "unique": col.unique,
                "missing_count": col.missing_count,
                "missing_ratio": col.missing_ratio,
                "cardinality": col.cardinality,
                "min_value": str(col.min_value) if col.min_value is not None else None,
                "max_value": str(col.max_value) if col.max_value is not None else None,
                "mean_value": col.mean_value,
                "std_value": col.std_value,
                "sample_values": col.sample_values,
                "description": col.description,
                "metadata": {},
                "created_at": datetime.now(timezone.utc),
            })

        if rows:
            conn.execute(columns_table.insert(), rows)


def get_column_metadata(
    dataset_name: str,
    table_name: Optional[str] = None,
    engine: Optional[Engine] = None
) -> Dict[str, ColumnInfo]:
    """Get column metadata from dataset database.
    
    Args:
        dataset_name: Name of the dataset
        table_name: Optional table name filter
        engine: Optional SQLAlchemy engine
    
    Returns:
        Dictionary mapping column names to ColumnInfo objects
    """
    if engine is None:
        engine = get_dataset_engine(dataset_name)

    metadata = sa.MetaData()

    try:
        columns_table = Table("_columns", metadata, autoload_with=engine)
    except SQLAlchemyError:
        # Table doesn't exist
        return {}

    with engine.connect() as conn:
        stmt = select(columns_table)
        if table_name:
            stmt = stmt.where(columns_table.c.table_name == table_name)

        results = conn.execute(stmt).fetchall()

        columns = {}
        for row in results:
            col_info = ColumnInfo(
                name=row.column_name,
                dtype=row.dtype,
                column_type=ColumnType(row.column_type) if row.column_type else ColumnType.TEXT,
                nullable=row.nullable,
                unique=row.unique,
                missing_count=row.missing_count,
                missing_ratio=row.missing_ratio,
                cardinality=row.cardinality,
                min_value=row.min_value,
                max_value=row.max_value,
                mean_value=row.mean_value,
                std_value=row.std_value,
                sample_values=row.sample_values or [],
                description=row.description,
            )
            columns[row.column_name] = col_info

        return columns


def store_dataset_statistics(
    dataset_name: str,
    table_name: str,
    statistics: DatasetStatistics,
    engine: Optional[Engine] = None
) -> None:
    """Store dataset statistics in dataset database.
    
    Args:
        dataset_name: Name of the dataset
        table_name: Name of the table
        statistics: DatasetStatistics object
        engine: Optional SQLAlchemy engine
    """
    if engine is None:
        engine = get_dataset_engine(dataset_name)

    # Ensure tables exist
    create_metadata_tables(engine)

    metadata = sa.MetaData()
    stats_table = Table("_statistics", metadata, autoload_with=engine)

    with engine.begin() as conn:
        conn.execute(stats_table.insert().values(
            table_name=table_name,
            row_count=statistics.row_count,
            column_count=statistics.column_count,
            memory_usage_mb=statistics.memory_usage_mb,
            missing_values=statistics.missing_values,
            column_types=statistics.column_types,
            numeric_columns=statistics.numeric_columns,
            categorical_columns=statistics.categorical_columns,
            datetime_columns=statistics.datetime_columns,
            text_columns=statistics.text_columns,
            computed_at=statistics.computed_at,
        ))


def get_latest_statistics(
    dataset_name: str,
    table_name: str,
    engine: Optional[Engine] = None
) -> Optional[DatasetStatistics]:
    """Get latest statistics for a table.
    
    Args:
        dataset_name: Name of the dataset
        table_name: Name of the table
        engine: Optional SQLAlchemy engine
    
    Returns:
        DatasetStatistics object or None
    """
    if engine is None:
        engine = get_dataset_engine(dataset_name)

    metadata = sa.MetaData()

    try:
        stats_table = Table("_statistics", metadata, autoload_with=engine)
    except SQLAlchemyError:
        # Table doesn't exist
        return None

    with engine.connect() as conn:
        stmt = (
            select(stats_table)
            .where(stats_table.c.table_name == table_name)
            .order_by(stats_table.c.computed_at.desc())
            .limit(1)
        )

        result = conn.execute(stmt).fetchone()

        if result:
            return DatasetStatistics(
                row_count=result.row_count,
                column_count=result.column_count,
                memory_usage_mb=result.memory_usage_mb,
                missing_values=result.missing_values or {},
                column_types=result.column_types or {},
                numeric_columns=result.numeric_columns or [],
                categorical_columns=result.categorical_columns or [],
                datetime_columns=result.datetime_columns or [],
                text_columns=result.text_columns or [],
                computed_at=result.computed_at,
            )

        return None


def store_extended_info(
    dataset_name: str,
    info: DatasetInfoExtended,
    engine: Optional[Engine] = None
) -> None:
    """Store extended dataset information as metadata.
    
    Args:
        dataset_name: Name of the dataset
        info: DatasetInfoExtended object
        engine: Optional SQLAlchemy engine
    """
    if engine is None:
        engine = get_dataset_engine(dataset_name)

    # Store main dataset info
    dataset_meta = {
        "name": info.name,
        "display_name": info.display_name,
        "description": info.description,
        "problem_type": info.problem_type.value if info.problem_type else None,
        "target_column": info.target_column,
        "id_columns": info.id_columns,
        "datetime_columns": info.datetime_columns,
        "feature_columns": info.feature_columns,
        "row_count": info.row_count,
        "memory_usage_mb": info.memory_usage_mb,
        "source": info.source,
        "version": info.version,
        "tags": info.tags,
        "metadata": info.metadata,
        "created_at": info.created_at.isoformat() if info.created_at else None,
        "updated_at": info.updated_at.isoformat() if info.updated_at else None,
    }

    store_dataset_metadata(dataset_name, "dataset_info", dataset_meta, engine)

    # Store file information
    files_meta = {}
    for file_type, file_info in info.files.items():
        files_meta[file_type] = {
            "path": str(file_info.path),
            "name": file_info.name,
            "size_bytes": file_info.size_bytes,
            "file_type": file_info.file_type.value if hasattr(file_info.file_type, "value") else file_info.file_type,
            "format": file_info.format,
            "encoding": file_info.encoding,
            "row_count": file_info.row_count,
            "column_count": file_info.column_count,
            "created_at": file_info.created_at.isoformat() if file_info.created_at else None,
            "modified_at": file_info.modified_at.isoformat() if file_info.modified_at else None,
            "checksum": file_info.checksum,
        }

    store_dataset_metadata(dataset_name, "files", files_meta, engine)


def get_extended_info(
    dataset_name: str,
    engine: Optional[Engine] = None
) -> Optional[DatasetInfoExtended]:
    """Get extended dataset information from metadata.
    
    Args:
        dataset_name: Name of the dataset
        engine: Optional SQLAlchemy engine
    
    Returns:
        DatasetInfoExtended object or None
    """
    if engine is None:
        engine = get_dataset_engine(dataset_name)

    # Get main dataset info
    dataset_meta = get_dataset_metadata(dataset_name, "dataset_info", engine)
    if not dataset_meta:
        return None

    # Get files info
    files_meta = get_dataset_metadata(dataset_name, "files", engine) or {}

    # Get column info
    columns = get_column_metadata(dataset_name, engine=engine)

    # Build DatasetInfoExtended
    from mdm.models.dataset import FileInfo
    from mdm.models.enums import FileType, ProblemType

    files = {}
    for file_type, file_data in files_meta.items():
        files[file_type] = FileInfo(
            path=Path(file_data["path"]),
            name=file_data["name"],
            size_bytes=file_data["size_bytes"],
            file_type=FileType(file_data["file_type"]),
            format=file_data["format"],
            encoding=file_data.get("encoding", "utf-8"),
            row_count=file_data.get("row_count"),
            column_count=file_data.get("column_count"),
            created_at=datetime.fromisoformat(file_data["created_at"]) if file_data.get("created_at") else None,
            modified_at=datetime.fromisoformat(file_data["modified_at"]) if file_data.get("modified_at") else None,
            checksum=file_data.get("checksum"),
        )

    return DatasetInfoExtended(
        name=dataset_meta["name"],
        display_name=dataset_meta.get("display_name"),
        description=dataset_meta.get("description"),
        problem_type=ProblemType(dataset_meta["problem_type"]) if dataset_meta.get("problem_type") else None,
        target_column=dataset_meta.get("target_column"),
        id_columns=dataset_meta.get("id_columns", []),
        datetime_columns=dataset_meta.get("datetime_columns", []),
        feature_columns=dataset_meta.get("feature_columns", []),
        files=files,
        columns=columns,
        row_count=dataset_meta.get("row_count"),
        memory_usage_mb=dataset_meta.get("memory_usage_mb"),
        source=dataset_meta.get("source"),
        version=dataset_meta.get("version", "1.0.0"),
        tags=dataset_meta.get("tags", []),
        metadata=dataset_meta.get("metadata", {}),
        created_at=datetime.fromisoformat(dataset_meta["created_at"]) if dataset_meta.get("created_at") else None,
        updated_at=datetime.fromisoformat(dataset_meta["updated_at"]) if dataset_meta.get("updated_at") else None,
    )
