"""Base classes for file loaders."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, Optional, Dict, Any, List
import pandas as pd
from sqlalchemy import Engine
import logging

from rich.progress import Progress

logger = logging.getLogger(__name__)


class FileLoader(ABC):
    """Abstract base class for file loaders."""
    
    def __init__(self, batch_size: int = 10000):
        """Initialize loader with batch size."""
        self.batch_size = batch_size
        self._detected_column_types = {}
        self._detected_datetime_columns = []
    
    @abstractmethod
    def can_handle(self, file_path: Path) -> bool:
        """Check if this loader can handle the given file."""
        pass
    
    @abstractmethod
    def get_total_rows(self, file_path: Path) -> int:
        """Get total number of rows in the file for progress tracking."""
        pass
    
    @abstractmethod
    def read_chunks(self, file_path: Path) -> Iterator[pd.DataFrame]:
        """Read file in chunks."""
        pass
    
    def load_file(
        self,
        file_path: Path,
        table_name: str,
        backend: Any,  # StorageBackend
        engine: Engine,
        progress: Progress,
        detect_types_for: Optional[List[str]] = None
    ) -> None:
        """Load file into database using batch processing.
        
        Args:
            file_path: Path to the file to load
            table_name: Name of the table to create/append to
            backend: Storage backend instance
            engine: SQLAlchemy engine
            progress: Rich progress instance
            detect_types_for: List of table names to detect column types for
        """
        logger.info(f"Loading {file_path} as table '{table_name}'")
        
        # Get total rows for progress bar
        total_rows = self.get_total_rows(file_path)
        
        # Create progress task
        task = progress.add_task(
            f"Loading {file_path.name} into {table_name}",
            total=total_rows
        )
        
        # Read and process chunks
        first_chunk = True
        chunk_count = 0
        
        for chunk_df in self.read_chunks(file_path):
            chunk_count += 1
            logger.debug(f"Processing chunk {chunk_count} with {len(chunk_df)} rows")
            
            if first_chunk:
                # Log column information from first chunk
                logger.debug(f"Columns in {table_name}: {list(chunk_df.columns)}")
                logger.debug(f"Data types: {chunk_df.dtypes.to_dict()}")
                
                # Detect column types on first chunk if requested
                if detect_types_for and table_name in detect_types_for:
                    self._detect_and_store_column_types(chunk_df, table_name)
                
                # Create table with first chunk
                backend.create_table_from_dataframe(
                    chunk_df, table_name, engine, if_exists='replace'
                )
                first_chunk = False
            else:
                # Append subsequent chunks
                backend.create_table_from_dataframe(
                    chunk_df, table_name, engine, if_exists='append'
                )
            
            # Update progress
            progress.update(task, advance=len(chunk_df))
            
            # Explicitly free memory
            del chunk_df
    
    def _detect_and_store_column_types(self, df: pd.DataFrame, table_name: str) -> None:
        """Detect and store column types from dataframe sample."""
        self._detected_column_types[table_name] = {}
        for col in df.columns:
            dtype = str(df[col].dtype)
            self._detected_column_types[table_name][col] = dtype
            logger.debug(f"Detected type for {col}: {dtype}")
    
    def _detect_datetime_columns_from_sample(self, sample_df: pd.DataFrame) -> None:
        """Detect datetime columns from a sample dataframe."""
        for col in sample_df.columns:
            # Skip if column looks like ID or numeric
            if 'id' in col.lower() or sample_df[col].dtype in ['int64', 'float64']:
                continue
                
            # Try to parse as datetime
            try:
                # Check if values look like dates
                sample_values = sample_df[col].dropna().astype(str).head(10)
                if len(sample_values) > 0:
                    # Try parsing with common date formats first
                    common_formats = [
                        '%Y-%m-%d',           # 2023-01-15
                        '%Y/%m/%d',           # 2023/01/15
                        '%d-%m-%Y',           # 15-01-2023
                        '%d/%m/%Y',           # 15/01/2023
                        '%m-%d-%Y',           # 01-15-2023
                        '%m/%d/%Y',           # 01/15/2023
                        '%Y-%m-%d %H:%M:%S', # 2023-01-15 10:30:45
                        '%Y/%m/%d %H:%M:%S', # 2023/01/15 10:30:45
                        '%d-%m-%Y %H:%M:%S', # 15-01-2023 10:30:45
                        '%d/%m/%Y %H:%M:%S', # 15/01/2023 10:30:45
                    ]
                    
                    parsed = None
                    for fmt in common_formats:
                        try:
                            parsed = pd.to_datetime(sample_values, format=fmt, errors='coerce')
                            if parsed.notna().sum() > len(parsed) * 0.8:  # 80% success rate
                                break
                        except:
                            continue
                    
                    # If no common format worked, fall back to inference
                    if parsed is None or parsed.notna().sum() <= len(parsed) * 0.5:
                        parsed = pd.to_datetime(sample_values, errors='coerce', infer_datetime_format=True)
                    
                    # If more than 50% parsed successfully, it's likely a datetime
                    if parsed is not None and parsed.notna().sum() > len(parsed) * 0.5:
                        self._detected_datetime_columns.append(col)
                        logger.debug(f"Detected datetime column: {col}")
            except Exception:
                # Not a datetime column
                pass
    
    @property
    def detected_column_types(self) -> Dict[str, Dict[str, str]]:
        """Get detected column types."""
        return self._detected_column_types
    
    @property 
    def detected_datetime_columns(self) -> List[str]:
        """Get detected datetime columns."""
        return self._detected_datetime_columns


class FileLoaderRegistry:
    """Registry for file loaders."""
    
    def __init__(self):
        """Initialize empty registry."""
        self._loaders: List[FileLoader] = []
    
    def register(self, loader: FileLoader) -> None:
        """Register a file loader."""
        self._loaders.append(loader)
    
    def get_loader(self, file_path: Path) -> Optional[FileLoader]:
        """Get appropriate loader for file path."""
        for loader in self._loaders:
            if loader.can_handle(file_path):
                return loader
        return None
    
    def get_all_loaders(self) -> List[FileLoader]:
        """Get all registered loaders."""
        return self._loaders.copy()