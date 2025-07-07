"""Base SQLAlchemy models for MDM."""

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import JSON, Boolean, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""

    pass


class DatasetMetadata(Base):
    """Metadata table for dataset information."""

    __tablename__ = "_metadata"

    key: Mapped[str] = mapped_column(String(255), primary_key=True)
    value: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc)
    )


class ColumnInfo(Base):
    """Column metadata table."""

    __tablename__ = "_columns"

    column_name: Mapped[str] = mapped_column(String(255), primary_key=True)
    data_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    is_target: Mapped[bool] = mapped_column(Boolean, default=False)
    is_id: Mapped[bool] = mapped_column(Boolean, default=False)
    null_percentage: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    unique_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    column_metadata: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)


class QualityMetrics(Base):
    """Data quality metrics table."""

    __tablename__ = "_quality_metrics"

    metric_name: Mapped[str] = mapped_column(String(255), primary_key=True)
    metric_value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    computed_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )

