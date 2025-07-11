"""Compressed CSV file loader implementation."""

from pathlib import Path
from typing import Iterator
import pandas as pd
import gzip
import logging

from .base import FileLoader

logger = logging.getLogger(__name__)


class CompressedCSVLoader(FileLoader):
    """Loader for compressed CSV files (.csv.gz, .tsv.gz)."""
    
    def can_handle(self, file_path: Path) -> bool:
        """Check if this loader can handle the given file."""
        return (file_path.suffix.lower() == '.gz' and 
                '.csv' in file_path.suffixes or '.tsv' in file_path.suffixes)
    
    def get_total_rows(self, file_path: Path) -> int:
        """Get total number of rows in the compressed file."""
        # Count lines in compressed file
        with gzip.open(file_path, 'rt') as f:
            return sum(1 for _ in f) - 1  # Subtract header
    
    def read_chunks(self, file_path: Path) -> Iterator[pd.DataFrame]:
        """Read compressed CSV file in chunks."""
        # Determine delimiter
        delimiter = ','  # Default to comma for .csv.gz
        if '.tsv' in file_path.suffixes:
            delimiter = '\t'
        
        # First, detect datetime columns on a small sample
        if not self._detected_datetime_columns:
            sample_df = pd.read_csv(
                file_path, 
                delimiter=delimiter, 
                compression='gzip', 
                nrows=100
            )
            self._detect_datetime_columns_from_sample(sample_df)
            logger.info(f"Detected datetime columns: {self._detected_datetime_columns}")
        
        # Read compressed file in chunks
        for chunk_df in pd.read_csv(
            file_path,
            delimiter=delimiter,
            compression='gzip',
            parse_dates=None,  # Don't parse dates on initial load
            chunksize=self.batch_size
        ):
            yield chunk_df