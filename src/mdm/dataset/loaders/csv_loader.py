"""CSV file loader implementation."""

from pathlib import Path
from typing import Iterator
import pandas as pd
import logging

from .base import FileLoader
from ...dataset.auto_detect import detect_delimiter

logger = logging.getLogger(__name__)


class CSVLoader(FileLoader):
    """Loader for CSV and TSV files."""
    
    def can_handle(self, file_path: Path) -> bool:
        """Check if this loader can handle the given file."""
        return file_path.suffix.lower() in ['.csv', '.tsv']
    
    def get_total_rows(self, file_path: Path) -> int:
        """Get total number of rows in the file."""
        # Count lines minus header
        with open(file_path, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f) - 1
    
    def read_chunks(self, file_path: Path) -> Iterator[pd.DataFrame]:
        """Read CSV file in chunks."""
        delimiter = detect_delimiter(file_path)
        
        # First, detect datetime columns on a small sample
        if not self._detected_datetime_columns:
            sample_df = pd.read_csv(file_path, delimiter=delimiter, nrows=100)
            self._detect_datetime_columns_from_sample(sample_df)
            logger.info(f"Detected datetime columns: {self._detected_datetime_columns}")
        
        # Read in chunks without parsing dates initially to avoid missing column errors
        for chunk_df in pd.read_csv(
            file_path,
            delimiter=delimiter,
            parse_dates=None,  # Don't parse dates on initial load
            chunksize=self.batch_size
        ):
            yield chunk_df