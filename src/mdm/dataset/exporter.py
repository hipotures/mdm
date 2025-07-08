"""Dataset export functionality for MDM."""

import gzip
import json
from loguru import logger
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

from mdm.config import get_config_manager
from mdm.core.exceptions import DatasetError, StorageError
from mdm.storage.factory import BackendFactory



class DatasetExporter:
    """Export datasets to various formats."""

    def __init__(self):
        """Initialize exporter."""
        config_manager = get_config_manager()
        self.config = config_manager.config
        self.base_path = config_manager.base_path
        self.dataset_registry_dir = self.base_path / self.config.paths.configs_path
        self.datasets_dir = self.base_path / self.config.paths.datasets_path

    def export(
        self,
        dataset_name: str,
        format: str,
        output_dir: Optional[Path] = None,
        table: Optional[str] = None,
        compression: Optional[str] = None,
        metadata_only: bool = False,
        no_header: bool = False,
    ) -> List[Path]:
        """Export dataset to specified format.

        Args:
            dataset_name: Name of the dataset
            format: Export format (csv, parquet, json)
            output_dir: Output directory (default: current directory)
            table: Specific table to export (default: all tables)
            compression: Compression type (zip, gzip, snappy, lz4, or None)
            metadata_only: Export only metadata without data
            no_header: Exclude header row (CSV only)

        Returns:
            List of exported file paths
        """
        # Validate format
        valid_formats = ["csv", "parquet", "json"]
        if format not in valid_formats:
            raise ValueError(f"Invalid format '{format}'. Must be one of: {valid_formats}")

        # Set default compression based on format
        if compression is None:
            if format == "csv" or format == "json":
                compression = "zip"
            # Parquet has built-in compression, default to None

        # Get dataset info
        dataset_info = self._load_dataset_info(dataset_name)

        # Set output directory
        if output_dir is None:
            output_dir = Path.cwd()
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        exported_files = []

        # Export metadata
        if metadata_only:
            metadata_file = self._export_metadata(dataset_name, dataset_info, output_dir)
            exported_files.append(metadata_file)
            return exported_files

        # Export metadata alongside data
        metadata_file = self._export_metadata(dataset_name, dataset_info, output_dir)
        exported_files.append(metadata_file)

        # Get backend
        backend = self._get_backend(dataset_name, dataset_info)

        try:
            # Determine which tables to export
            tables_to_export = self._get_tables_to_export(dataset_info, table)

            # Export each table
            for table_type, table_name in tables_to_export.items():
                logger.info(f"Exporting table '{table_type}' ({table_name}) for dataset '{dataset_name}'")

                # Read data from backend
                df = self._read_table(backend, table_name)

                if df is None or df.empty:
                    logger.warning(f"Table '{table_name}' is empty or not found")
                    continue

                # Generate output filename
                output_file = output_dir / f"{dataset_name}_{table_type}.{format}"

                # Export based on format
                if format == "csv":
                    exported_file = self._export_csv(df, output_file, compression, no_header)
                elif format == "parquet":
                    exported_file = self._export_parquet(df, output_file, compression)
                elif format == "json":
                    exported_file = self._export_json(df, output_file, compression)

                exported_files.append(exported_file)
                logger.info(f"Exported {table_type} to {exported_file}")

        finally:
            # Close backend connection
            if hasattr(backend, 'close_connections'):
                backend.close_connections()

        return exported_files

    def _load_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """Load dataset information from YAML."""
        yaml_file = self.dataset_registry_dir / f"{dataset_name}.yaml"

        if not yaml_file.exists():
            raise DatasetError(f"Dataset '{dataset_name}' not found")

        try:
            with open(yaml_file) as f:
                data = yaml.safe_load(f)
            
            # Check backend compatibility
            dataset_backend = data.get('database', {}).get('backend', 'unknown')
            current_backend = self.config.database.default_backend
            
            if dataset_backend != current_backend:
                raise DatasetError(
                    f"Dataset '{dataset_name}' uses '{dataset_backend}' backend, "
                    f"but current backend is '{current_backend}'. "
                    f"Change default_backend in ~/.mdm/mdm.yaml to '{dataset_backend}' "
                    f"or re-register the dataset with --force option."
                )
            
            return data
        except DatasetError:
            raise
        except Exception as e:
            raise DatasetError(f"Failed to load dataset info: {e}") from e

    def _get_backend(self, dataset_name: str, dataset_info: Dict[str, Any]) -> Any:
        """Get storage backend for dataset."""
        backend_type = dataset_info.get('database', {}).get('backend', 'duckdb')
        backend_config = dataset_info.get('database', {}).copy()

        # Get the database path from dataset_info if available
        if 'path' in dataset_info.get('database', {}):
            db_path = dataset_info['database']['path']
        else:
            # Add dataset-specific paths
            dataset_dir = self.datasets_dir / dataset_name
            if backend_type == 'duckdb':
                db_path = str(dataset_dir / f'{dataset_name}.duckdb')
            elif backend_type == 'sqlite':
                db_path = str(dataset_dir / f'{dataset_name}.sqlite')
            else:
                raise StorageError(f"Unsupported backend type for export: {backend_type}")

        try:
            backend = BackendFactory.create(backend_type, backend_config)
            # Initialize the engine
            backend.get_engine(db_path)
            return backend
        except Exception as e:
            raise StorageError(f"Failed to create backend: {e}") from e

    def _get_tables_to_export(
        self,
        dataset_info: Dict[str, Any],
        specific_table: Optional[str]
    ) -> Dict[str, str]:
        """Determine which tables to export."""
        all_tables = dataset_info.get('tables', {})

        if specific_table:
            # Export only the specified table
            if specific_table in all_tables:
                return {specific_table: all_tables[specific_table]}
            # Try to find by table name
            for table_type, table_name in all_tables.items():
                if table_name == specific_table:
                    return {table_type: table_name}
            raise DatasetError(f"Table '{specific_table}' not found in dataset")
        # Export all tables by default
        return all_tables

    def _read_table(self, backend: Any, table_name: str) -> Optional[pd.DataFrame]:
        """Read table from backend."""
        try:
            # Use backend's query method
            query = f"SELECT * FROM {table_name}"
            return backend.query(query)
        except Exception as e:
            logger.error(f"Failed to read table '{table_name}': {e}")
            return None

    def _export_metadata(
        self,
        dataset_name: str,
        dataset_info: Dict[str, Any],
        output_dir: Path
    ) -> Path:
        """Export dataset metadata."""
        metadata = {
            'dataset_name': dataset_name,
            'export_timestamp': pd.Timestamp.now().isoformat(),
            'dataset_info': dataset_info,
        }

        # Add statistics if available
        stats_file = self.datasets_dir / dataset_name / "metadata" / "statistics.json"
        if stats_file.exists():
            try:
                with open(stats_file) as f:
                    metadata['statistics'] = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load statistics: {e}")

        # Export to JSON
        output_file = output_dir / f"{dataset_name}_metadata.json"
        with open(output_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        return output_file

    def _export_csv(
        self,
        df: pd.DataFrame,
        output_file: Path,
        compression: Optional[str],
        no_header: bool
    ) -> Path:
        """Export DataFrame to CSV."""
        # Export to CSV
        csv_options = {
            'index': False,
            'header': not no_header,
        }

        if compression == "zip":
            # Export to temporary CSV first
            temp_csv = output_file.with_suffix('.csv')
            df.to_csv(temp_csv, **csv_options)

            # Create zip file
            zip_file = output_file.with_suffix('.csv.zip')
            with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zf:
                zf.write(temp_csv, temp_csv.name)

            # Remove temporary file
            temp_csv.unlink()
            return zip_file

        if compression == "gzip":
            gz_file = output_file.with_suffix('.csv.gz')
            df.to_csv(gz_file, compression='gzip', **csv_options)
            return gz_file

        # No compression
        df.to_csv(output_file, **csv_options)
        return output_file

    def _export_parquet(
        self,
        df: pd.DataFrame,
        output_file: Path,
        compression: Optional[str]
    ) -> Path:
        """Export DataFrame to Parquet."""
        # Validate compression for Parquet
        valid_compressions = [None, 'snappy', 'gzip', 'lz4', 'zstd']
        if compression not in valid_compressions:
            logger.warning(f"Invalid compression '{compression}' for Parquet. Using 'snappy'")
            compression = 'snappy'

        # Export to Parquet
        try:
            df.to_parquet(output_file, compression=compression, index=False)
            return output_file
        except ImportError:
            raise DatasetError(
                "Parquet export requires 'pyarrow' or 'fastparquet' package. "
                "Install with: pip install pyarrow"
            )

    def _export_json(
        self,
        df: pd.DataFrame,
        output_file: Path,
        compression: Optional[str]
    ) -> Path:
        """Export DataFrame to JSON."""
        # Convert DataFrame to records format
        json_data = df.to_json(orient='records', date_format='iso')

        if compression == "zip":
            # Write to temporary file first
            temp_json = output_file.with_suffix('.json')
            with open(temp_json, 'w') as f:
                f.write(json_data)

            # Create zip file
            zip_file = output_file.with_suffix('.json.zip')
            with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zf:
                zf.write(temp_json, temp_json.name)

            # Remove temporary file
            temp_json.unlink()
            return zip_file

        if compression == "gzip":
            gz_file = output_file.with_suffix('.json.gz')
            with gzip.open(gz_file, 'wt', encoding='utf-8') as f:
                f.write(json_data)
            return gz_file

        # No compression
        with open(output_file, 'w') as f:
            f.write(json_data)
        return output_file


def export_dataset(
    dataset_name: str,
    format: str = "csv",
    output_dir: Optional[Path] = None,
    table: Optional[str] = None,
    compression: Optional[str] = None,
    metadata_only: bool = False,
    no_header: bool = False,
) -> List[Path]:
    """Export dataset using DatasetExporter.

    Args:
        dataset_name: Name of the dataset
        format: Export format (csv, parquet, json)
        output_dir: Output directory
        table: Specific table to export
        compression: Compression type
        metadata_only: Export only metadata
        no_header: Exclude header (CSV only)

    Returns:
        List of exported file paths
    """
    exporter = DatasetExporter()
    return exporter.export(
        dataset_name=dataset_name,
        format=format,
        output_dir=output_dir,
        table=table,
        compression=compression,
        metadata_only=metadata_only,
        no_header=no_header,
    )
