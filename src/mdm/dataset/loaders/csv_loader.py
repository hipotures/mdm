"""CSV file loader implementation."""

from pathlib import Path
from typing import Iterator, Optional
import pandas as pd
import logging
import chardet

from .base import FileLoader
from ...dataset.auto_detect import detect_delimiter

logger = logging.getLogger(__name__)


class CSVLoader(FileLoader):
    """Loader for CSV and TSV files."""
    
    def __init__(self, batch_size: int = 10000):
        """Initialize CSV loader."""
        super().__init__(batch_size)
        self._file_encodings: dict[Path, str] = {}
    
    def can_handle(self, file_path: Path) -> bool:
        """Check if this loader can handle the given file."""
        return file_path.suffix.lower() in ['.csv', '.tsv']
    
    def detect_encoding(self, file_path: Path, sample_size: int = 65536) -> str:
        """Detect file encoding using chardet."""
        if file_path in self._file_encodings:
            return self._file_encodings[file_path]
        
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(sample_size)
                result = chardet.detect(raw_data)
                
                encoding = result.get('encoding', 'utf-8')
                confidence = result.get('confidence', 0)
                
                # If confidence is low or encoding is None, try with more data
                if confidence < 0.7 or not encoding:
                    raw_data = f.read(sample_size * 4)  # Read more data
                    result = chardet.detect(raw_data)
                    encoding = result.get('encoding', 'utf-8')
                    confidence = result.get('confidence', 0)
                
                # Fallback to common encodings if detection fails
                if not encoding or confidence < 0.5:
                    # Try common encodings
                    for test_encoding in ['utf-8', 'latin-1', 'windows-1252', 'iso-8859-1']:
                        try:
                            with open(file_path, 'r', encoding=test_encoding) as test_f:
                                test_f.read(1000)
                            encoding = test_encoding
                            break
                        except UnicodeDecodeError:
                            continue
                    else:
                        encoding = 'latin-1'  # Ultimate fallback
                
                logger.info(f"Detected encoding for {file_path.name}: {encoding} (confidence: {confidence:.2%})")
                self._file_encodings[file_path] = encoding
                return encoding
                
        except Exception as e:
            logger.warning(f"Error detecting encoding for {file_path}: {e}. Using latin-1.")
            self._file_encodings[file_path] = 'latin-1'
            return 'latin-1'
    
    def get_total_rows(self, file_path: Path) -> int:
        """Get total number of rows in the file."""
        # Detect encoding first
        encoding = self.detect_encoding(file_path)
        
        # Count lines minus header
        with open(file_path, 'r', encoding=encoding, errors='replace') as f:
            return sum(1 for _ in f) - 1
    
    def read_chunks(self, file_path: Path) -> Iterator[pd.DataFrame]:
        """Read CSV file in chunks."""
        # Detect encoding first
        encoding = self.detect_encoding(file_path)
        delimiter = detect_delimiter(file_path, encoding=encoding)
        
        # First, detect datetime columns on a small sample
        if not self._detected_datetime_columns:
            sample_df = pd.read_csv(
                file_path, 
                delimiter=delimiter, 
                nrows=100,
                encoding=encoding,
                encoding_errors='replace'
            )
            self._detect_datetime_columns_from_sample(sample_df)
            logger.info(f"Detected datetime columns: {self._detected_datetime_columns}")
        
        # Read in chunks without parsing dates initially to avoid missing column errors
        for chunk_df in pd.read_csv(
            file_path,
            delimiter=delimiter,
            parse_dates=None,  # Don't parse dates on initial load
            chunksize=self.batch_size,
            encoding=encoding,
            encoding_errors='replace'
        ):
            yield chunk_df