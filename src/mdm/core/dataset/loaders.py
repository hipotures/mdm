"""Data loading utilities for various file formats.

Provides efficient data loaders with batch processing support.
"""
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, Iterator, List
import pandas as pd
import json
from functools import lru_cache

from ...core.exceptions import DatasetError

logger = logging.getLogger(__name__)


class DataLoader(ABC):
    """Abstract base class for data loaders."""
    
    def __init__(self, batch_size: int = 10000):
        """Initialize loader.
        
        Args:
            batch_size: Number of rows to load per batch
        """
        self.batch_size = batch_size
        self._metadata: Dict[str, Any] = {}
    
    @abstractmethod
    def can_handle(self, path: Path) -> bool:
        """Check if this loader can handle the file.
        
        Args:
            path: File path
            
        Returns:
            True if loader can handle this file
        """
        pass
    
    @abstractmethod
    def load(self, path: Path, **kwargs) -> pd.DataFrame:
        """Load entire file into DataFrame.
        
        Args:
            path: File path
            **kwargs: Additional loader options
            
        Returns:
            Loaded DataFrame
        """
        pass
    
    @abstractmethod
    def load_batch(self, path: Path, **kwargs) -> Iterator[pd.DataFrame]:
        """Load file in batches.
        
        Args:
            path: File path
            **kwargs: Additional loader options
            
        Yields:
            DataFrame batches
        """
        pass
    
    def detect_metadata(self, path: Path) -> Dict[str, Any]:
        """Detect file metadata without loading data.
        
        Args:
            path: File path
            
        Returns:
            Metadata dictionary
        """
        return {
            'path': str(path),
            'size_bytes': path.stat().st_size,
            'loader': self.__class__.__name__
        }
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Get cached metadata."""
        return self._metadata


class CSVLoader(DataLoader):
    """CSV file loader with auto-detection capabilities."""
    
    SAMPLE_ROWS = 1000
    
    def can_handle(self, path: Path) -> bool:
        """Check if file is CSV format."""
        suffixes = path.suffixes
        base_suffix = suffixes[-2] if len(suffixes) > 1 and suffixes[-1] in {'.gz', '.zip', '.bz2'} else suffixes[-1]
        return base_suffix in {'.csv', '.tsv', '.txt'}
    
    def load(self, path: Path, **kwargs) -> pd.DataFrame:
        """Load CSV file."""
        # Auto-detect parameters if not provided
        params = self._detect_csv_params(path)
        params.update(kwargs)
        
        logger.info(f"Loading CSV file: {path}")
        
        try:
            df = pd.read_csv(path, **params)
            self._metadata['shape'] = df.shape
            self._metadata['columns'] = list(df.columns)
            return df
        except Exception as e:
            raise DatasetError(f"Failed to load CSV file {path}: {e}")
    
    def load_batch(self, path: Path, **kwargs) -> Iterator[pd.DataFrame]:
        """Load CSV in batches."""
        # Auto-detect parameters
        params = self._detect_csv_params(path)
        params.update(kwargs)
        params['chunksize'] = self.batch_size
        
        logger.info(f"Loading CSV file in batches: {path}")
        
        try:
            total_rows = 0
            for chunk in pd.read_csv(path, **params):
                total_rows += len(chunk)
                yield chunk
            
            self._metadata['total_rows'] = total_rows
        except Exception as e:
            raise DatasetError(f"Failed to load CSV file {path}: {e}")
    
    def _detect_csv_params(self, path: Path) -> Dict[str, Any]:
        """Auto-detect CSV parameters."""
        params = {}
        
        # Detect delimiter
        delimiter = self._detect_delimiter(path)
        if delimiter:
            params['sep'] = delimiter
        
        # Detect encoding
        encoding = self._detect_encoding(path)
        if encoding:
            params['encoding'] = encoding
        
        # Detect compression
        if path.suffix in {'.gz', '.zip', '.bz2'}:
            params['compression'] = 'infer'
        
        return params
    
    @lru_cache(maxsize=32)
    def _detect_delimiter(self, path: Path) -> Optional[str]:
        """Detect CSV delimiter."""
        try:
            # Read sample
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                sample = f.read(8192)
            
            # Count delimiter occurrences
            delimiters = [',', '\t', ';', '|']
            counts = {d: sample.count(d) for d in delimiters}
            
            # Return most common
            if counts:
                delimiter = max(counts, key=counts.get)
                if counts[delimiter] > 0:
                    logger.debug(f"Detected delimiter: {repr(delimiter)}")
                    return delimiter
        except Exception as e:
            logger.warning(f"Failed to detect delimiter: {e}")
        
        return None
    
    @lru_cache(maxsize=32)
    def _detect_encoding(self, path: Path) -> str:
        """Detect file encoding."""
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(path, 'r', encoding=encoding) as f:
                    f.read(8192)
                return encoding
            except UnicodeDecodeError:
                continue
        
        # Default to utf-8 with errors='replace'
        return 'utf-8'


class ParquetLoader(DataLoader):
    """Parquet file loader."""
    
    def can_handle(self, path: Path) -> bool:
        """Check if file is Parquet format."""
        return path.suffix in {'.parquet', '.pq'}
    
    def load(self, path: Path, **kwargs) -> pd.DataFrame:
        """Load Parquet file."""
        logger.info(f"Loading Parquet file: {path}")
        
        try:
            df = pd.read_parquet(path, **kwargs)
            self._metadata['shape'] = df.shape
            self._metadata['columns'] = list(df.columns)
            return df
        except Exception as e:
            raise DatasetError(f"Failed to load Parquet file {path}: {e}")
    
    def load_batch(self, path: Path, **kwargs) -> Iterator[pd.DataFrame]:
        """Load Parquet in batches."""
        # Parquet doesn't have native batch reading in pandas
        # Load entire file and yield in chunks
        df = self.load(path, **kwargs)
        
        for i in range(0, len(df), self.batch_size):
            yield df.iloc[i:i + self.batch_size]


class ExcelLoader(DataLoader):
    """Excel file loader."""
    
    def can_handle(self, path: Path) -> bool:
        """Check if file is Excel format."""
        return path.suffix in {'.xlsx', '.xls'}
    
    def load(self, path: Path, **kwargs) -> pd.DataFrame:
        """Load Excel file."""
        logger.info(f"Loading Excel file: {path}")
        
        try:
            # If sheet_name not specified, load first sheet
            if 'sheet_name' not in kwargs:
                kwargs['sheet_name'] = 0
            
            df = pd.read_excel(path, **kwargs)
            self._metadata['shape'] = df.shape
            self._metadata['columns'] = list(df.columns)
            return df
        except Exception as e:
            raise DatasetError(f"Failed to load Excel file {path}: {e}")
    
    def load_batch(self, path: Path, **kwargs) -> Iterator[pd.DataFrame]:
        """Load Excel in batches."""
        # Excel doesn't support batch reading
        # Load entire file and yield in chunks
        df = self.load(path, **kwargs)
        
        for i in range(0, len(df), self.batch_size):
            yield df.iloc[i:i + self.batch_size]
    
    def detect_metadata(self, path: Path) -> Dict[str, Any]:
        """Detect Excel metadata including sheet names."""
        metadata = super().detect_metadata(path)
        
        try:
            # Get sheet names without loading data
            xl_file = pd.ExcelFile(path)
            metadata['sheets'] = xl_file.sheet_names
            metadata['n_sheets'] = len(xl_file.sheet_names)
        except Exception as e:
            logger.warning(f"Failed to detect Excel metadata: {e}")
        
        return metadata


class JSONLoader(DataLoader):
    """JSON file loader."""
    
    def can_handle(self, path: Path) -> bool:
        """Check if file is JSON format."""
        return path.suffix in {'.json', '.jsonl'}
    
    def load(self, path: Path, **kwargs) -> pd.DataFrame:
        """Load JSON file."""
        logger.info(f"Loading JSON file: {path}")
        
        try:
            if path.suffix == '.jsonl':
                # JSON Lines format
                df = pd.read_json(path, lines=True, **kwargs)
            else:
                # Regular JSON
                df = pd.read_json(path, **kwargs)
            
            self._metadata['shape'] = df.shape
            self._metadata['columns'] = list(df.columns)
            return df
        except Exception as e:
            raise DatasetError(f"Failed to load JSON file {path}: {e}")
    
    def load_batch(self, path: Path, **kwargs) -> Iterator[pd.DataFrame]:
        """Load JSON in batches."""
        if path.suffix == '.jsonl':
            # JSON Lines can be read line by line
            logger.info(f"Loading JSON Lines file in batches: {path}")
            
            batch = []
            with open(path, 'r') as f:
                for line in f:
                    batch.append(json.loads(line))
                    
                    if len(batch) >= self.batch_size:
                        yield pd.DataFrame(batch)
                        batch = []
                
                # Yield remaining
                if batch:
                    yield pd.DataFrame(batch)
        else:
            # Regular JSON - load all and chunk
            df = self.load(path, **kwargs)
            
            for i in range(0, len(df), self.batch_size):
                yield df.iloc[i:i + self.batch_size]


class LoaderRegistry:
    """Registry for data loaders."""
    
    def __init__(self):
        self._loaders: List[DataLoader] = [
            CSVLoader(),
            ParquetLoader(),
            ExcelLoader(),
            JSONLoader(),
        ]
    
    def get_loader(self, path: Path) -> DataLoader:
        """Get appropriate loader for file.
        
        Args:
            path: File path
            
        Returns:
            Appropriate data loader
            
        Raises:
            DatasetError: If no loader found
        """
        for loader in self._loaders:
            if loader.can_handle(path):
                return loader
        
        raise DatasetError(f"No loader found for file: {path}")
    
    def register_loader(self, loader: DataLoader) -> None:
        """Register custom loader.
        
        Args:
            loader: DataLoader instance
        """
        self._loaders.insert(0, loader)  # Prepend for priority


# Global registry
loader_registry = LoaderRegistry()
