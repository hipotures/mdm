"""
Dataset management interfaces based on actual usage analysis.

These interfaces match the existing DatasetRegistrar and DatasetManager APIs
to ensure compatibility during migration.
"""
from typing import Protocol, Dict, Any, List, Optional, Tuple, runtime_checkable
from pathlib import Path
import pandas as pd


@runtime_checkable
class IDatasetRegistrar(Protocol):
    """Dataset registration interface based on actual implementation."""
    
    def register(
        self, 
        name: str, 
        path: str, 
        target: Optional[str] = None,
        problem_type: Optional[str] = None,
        id_columns: Optional[List[str]] = None,
        datetime_columns: Optional[List[str]] = None,
        force: bool = False
    ) -> Dict[str, Any]:
        """
        Register a new dataset.
        
        Args:
            name: Dataset name
            path: Path to data file(s)
            target: Target column name
            problem_type: Type of ML problem
            id_columns: List of ID columns
            datetime_columns: List of datetime columns
            force: Force overwrite if exists
            
        Returns:
            Registration result with dataset info
        """
        ...
    
    def validate_dataset_name(self, name: str) -> None:
        """Validate dataset name format."""
        ...
    
    def detect_structure(self, path: str) -> Dict[str, Any]:
        """Auto-detect dataset structure (Kaggle format, CSV, etc)."""
        ...
    
    def detect_file_format(self, file_path: str) -> str:
        """Detect file format from extension."""
        ...
    
    def load_data_file(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Load data from a single file."""
        ...
    
    def detect_column_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """Detect column data types."""
        ...
    
    def detect_id_columns(self, df: pd.DataFrame) -> List[str]:
        """Auto-detect ID columns."""
        ...
    
    def detect_target_column(self, df: pd.DataFrame, columns: List[str]) -> Optional[str]:
        """Auto-detect target column."""
        ...
    
    def detect_problem_type(self, df: pd.DataFrame, target_column: str) -> str:
        """Auto-detect problem type from target column."""
        ...


@runtime_checkable
class IDatasetManager(Protocol):
    """Dataset management interface based on actual implementation."""
    
    def list_datasets(
        self, 
        limit: Optional[int] = None,
        sort_by: Optional[str] = None,
        backend: Optional[str] = None,
        tag: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List all datasets.
        
        Args:
            limit: Maximum number of datasets to return
            sort_by: Sort by field (name, registration_date, size)
            backend: Filter by backend type
            tag: Filter by tag
            
        Returns:
            List of dataset information dictionaries
        """
        ...
    
    def get_dataset_info(self, name: str) -> Dict[str, Any]:
        """Get detailed dataset information."""
        ...
    
    def dataset_exists(self, name: str) -> bool:
        """Check if dataset exists."""
        ...
    
    def remove_dataset(self, name: str, force: bool = False) -> None:
        """Remove dataset and all associated data."""
        ...
    
    def update_dataset(
        self, 
        name: str, 
        description: Optional[str] = None,
        target: Optional[str] = None,
        problem_type: Optional[str] = None,
        id_columns: Optional[List[str]] = None,
        tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Update dataset metadata."""
        ...
    
    def export_dataset(
        self, 
        name: str, 
        output_dir: str,
        format: str = "csv",
        compression: Optional[str] = None,
        tables: Optional[List[str]] = None
    ) -> List[str]:
        """
        Export dataset to files.
        
        Args:
            name: Dataset name
            output_dir: Output directory path
            format: Export format (csv, parquet, json)
            compression: Compression type (gzip, zip)
            tables: Specific tables to export
            
        Returns:
            List of exported file paths
        """
        ...
    
    def get_dataset_stats(
        self, 
        name: str,
        mode: str = "basic",
        tables: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Get dataset statistics."""
        ...
    
    def search_datasets(
        self,
        pattern: str,
        search_in: List[str] = ["name", "description", "tags"],
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Search datasets by pattern."""
        ...
    
    def load_dataset(
        self,
        name: str,
        table_name: str = "data",
        columns: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """Load dataset data."""
        ...
    
    def get_dataset_config(self, name: str) -> Dict[str, Any]:
        """Get dataset configuration from YAML file."""
        ...
    
    def update_dataset_config(self, name: str, config: Dict[str, Any]) -> None:
        """Update dataset configuration YAML file."""
        ...