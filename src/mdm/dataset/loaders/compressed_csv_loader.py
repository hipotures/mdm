"""Compressed CSV file loader implementation."""

from pathlib import Path
from typing import Iterator
import pandas as pd
import gzip
import logging
import chardet

from .base import FileLoader

logger = logging.getLogger(__name__)


class CompressedCSVLoader(FileLoader):
    """Loader for compressed CSV files (.csv.gz, .tsv.gz)."""
    
    def __init__(self, batch_size: int = 10000):
        """Initialize compressed CSV loader."""
        super().__init__(batch_size)
        self._file_encodings: dict[Path, str] = {}
    
    def can_handle(self, file_path: Path) -> bool:
        """Check if this loader can handle the given file."""
        return (file_path.suffix.lower() == '.gz' and 
                '.csv' in file_path.suffixes or '.tsv' in file_path.suffixes)
    
    def detect_encoding(self, file_path: Path, sample_size: int = 65536) -> str:
        """Detect file encoding using chardet on compressed file."""
        if file_path in self._file_encodings:
            return self._file_encodings[file_path]
        
        try:
            with gzip.open(file_path, 'rb') as f:
                raw_data = f.read(sample_size)
                result = chardet.detect(raw_data)
                
                encoding = result.get('encoding', 'utf-8')
                confidence = result.get('confidence', 0)
                
                # Fallback to common encodings if detection fails
                if not encoding or confidence < 0.5:
                    encoding = 'utf-8'  # Default for compressed files
                
                logger.info(f"Detected encoding for {file_path.name}: {encoding} (confidence: {confidence:.2%})")
                self._file_encodings[file_path] = encoding
                return encoding
                
        except Exception as e:
            logger.warning(f"Error detecting encoding for {file_path}: {e}. Using utf-8.")
            self._file_encodings[file_path] = 'utf-8'
            return 'utf-8'
    
    def get_total_rows(self, file_path: Path) -> int:
        """Get total number of rows in the compressed file."""
        # Detect encoding first
        encoding = self.detect_encoding(file_path)
        
        # Count lines in compressed file
        with gzip.open(file_path, 'rt', encoding=encoding, errors='replace') as f:
            return sum(1 for _ in f) - 1  # Subtract header
    
    def read_chunks(self, file_path: Path) -> Iterator[pd.DataFrame]:
        """Read compressed CSV file in chunks."""
        # Detect encoding first
        encoding = self.detect_encoding(file_path)
        
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
                nrows=100,
                encoding=encoding,
                encoding_errors='replace'
            )
            self._detect_datetime_columns_from_sample(sample_df)
            logger.info(f"Detected datetime columns: {self._detected_datetime_columns}")
        
        # Read compressed file in chunks
        for chunk_df in pd.read_csv(
            file_path,
            delimiter=delimiter,
            compression='gzip',
            parse_dates=None,  # Don't parse dates on initial load
            chunksize=self.batch_size,
            encoding=encoding,
            encoding_errors='replace'
        ):
            yield chunk_df