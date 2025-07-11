"""Parquet file loader implementation."""

from pathlib import Path
from typing import Iterator
import pandas as pd
import logging

from .base import FileLoader

logger = logging.getLogger(__name__)


class ParquetLoader(FileLoader):
    """Loader for Parquet files."""
    
    def can_handle(self, file_path: Path) -> bool:
        """Check if this loader can handle the given file."""
        return file_path.suffix.lower() == '.parquet'
    
    def get_total_rows(self, file_path: Path) -> int:
        """Get total number of rows in the file."""
        # For Parquet, we need to read metadata
        df = pd.read_parquet(file_path)
        return len(df)
    
    def read_chunks(self, file_path: Path) -> Iterator[pd.DataFrame]:
        """Read Parquet file in chunks."""
        # Parquet files need to be loaded fully first
        df = pd.read_parquet(file_path)
        
        # Process in batches
        total_rows = len(df)
        for i in range(0, total_rows, self.batch_size):
            batch_df = df.iloc[i:i + self.batch_size]
            yield batch_df