"""File loaders for dataset registration."""

from .base import FileLoader, FileLoaderRegistry
from .csv_loader import CSVLoader
from .parquet_loader import ParquetLoader
from .json_loader import JSONLoader
from .compressed_csv_loader import CompressedCSVLoader
from .excel_loader import ExcelLoader

__all__ = [
    'FileLoader',
    'FileLoaderRegistry',
    'CSVLoader',
    'ParquetLoader',
    'JSONLoader',
    'CompressedCSVLoader',
    'ExcelLoader',
]