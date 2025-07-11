"""Dataset export client."""

from pathlib import Path
from typing import Optional, List
import pandas as pd
from loguru import logger

from mdm.core.exceptions import DatasetError
from mdm.dataset.operations import ExportOperation

from .base import BaseClient


class ExportClient(BaseClient):
    """Client for dataset export operations."""
    
    def export_dataset(
        self,
        name: str,
        output_dir: str,
        format: str = "csv",
        compression: Optional[str] = None,
        include_features: bool = True,
        tables: Optional[List[str]] = None
    ) -> Path:
        """Export dataset to files.

        Args:
            name: Dataset name
            output_dir: Output directory path
            format: Export format (csv, parquet, json)
            compression: Compression type (gzip, zip, None)
            include_features: Whether to export feature tables
            tables: Specific tables to export (None = all)

        Returns:
            Path to output directory

        Raises:
            DatasetError: If dataset not found or export fails
        """
        export_op = ExportOperation()
        
        # Convert string path to Path object
        output_path = Path(output_dir)
        
        return export_op.execute(
            dataset_name=name,
            output_path=output_path,
            format=format,
            compression=compression,
            include_features=include_features,
            tables=tables
        )