"""Excel file loader implementation."""

from pathlib import Path
from typing import Iterator, Optional
import pandas as pd
import logging

from .base import FileLoader

logger = logging.getLogger(__name__)


class ExcelLoader(FileLoader):
    """Loader for Excel files (.xlsx, .xls)."""
    
    def __init__(self, batch_size: int = 10000, sheet_name: Optional[str] = None):
        """Initialize with batch size and optional sheet name."""
        super().__init__(batch_size)
        self.sheet_name = sheet_name or 0  # Default to first sheet
    
    def can_handle(self, file_path: Path) -> bool:
        """Check if this loader can handle the given file."""
        return file_path.suffix.lower() in ['.xlsx', '.xls']
    
    def get_total_rows(self, file_path: Path) -> int:
        """Get total number of rows in the file."""
        # Excel files need to be loaded to count rows
        df = pd.read_excel(file_path, sheet_name=self.sheet_name)
        return len(df)
    
    def read_chunks(self, file_path: Path) -> Iterator[pd.DataFrame]:
        """Read Excel file in chunks."""
        # Excel files need to be loaded fully first
        df = pd.read_excel(file_path, sheet_name=self.sheet_name)
        
        logger.info(f"Loaded Excel file with {len(df)} rows from sheet: {self.sheet_name}")
        
        # Process in batches
        total_rows = len(df)
        for i in range(0, total_rows, self.batch_size):
            batch_df = df.iloc[i:i + self.batch_size]
            yield batch_df