"""JSON file loader implementation."""

from pathlib import Path
from typing import Iterator
import pandas as pd
import logging

from .base import FileLoader

logger = logging.getLogger(__name__)


class JSONLoader(FileLoader):
    """Loader for JSON files."""
    
    def can_handle(self, file_path: Path) -> bool:
        """Check if this loader can handle the given file."""
        return file_path.suffix.lower() == '.json'
    
    def get_total_rows(self, file_path: Path) -> int:
        """Get total number of rows in the file."""
        # JSON files need to be loaded fully to count rows
        df = pd.read_json(file_path)
        return len(df)
    
    def read_chunks(self, file_path: Path) -> Iterator[pd.DataFrame]:
        """Read JSON file in chunks."""
        # JSON files typically need to be loaded fully
        df = pd.read_json(file_path)
        
        # Process in batches
        total_rows = len(df)
        for i in range(0, total_rows, self.batch_size):
            batch_df = df.iloc[i:i + self.batch_size]
            yield batch_df