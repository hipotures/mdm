"""Dataset statistics computation for MDM."""

import json
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from mdm.config import get_config_manager
from mdm.core.exceptions import DatasetError, StorageError
from mdm.storage.factory import BackendFactory

logger = logging.getLogger(__name__)


class DatasetStatistics:
    """Compute statistics for datasets."""

    def __init__(self):
        """Initialize statistics computer."""
        config_manager = get_config_manager()
        self.config = config_manager.config
        self.base_path = config_manager.base_path
        self.dataset_registry_dir = self.base_path / self.config.paths.configs_path
        self.datasets_dir = self.base_path / self.config.paths.datasets_path

    def compute_statistics(
        self,
        dataset_name: str,
        full: bool = False,
        save: bool = True,
    ) -> Dict[str, Any]:
        """Compute statistics for a dataset.

        Args:
            dataset_name: Name of the dataset
            full: Whether to compute full statistics including correlations
            save: Whether to save statistics to metadata

        Returns:
            Dictionary containing statistics
        """
        # Load dataset info
        dataset_info = self._load_dataset_info(dataset_name)

        # Get backend
        backend = self._get_backend(dataset_name, dataset_info)

        try:
            stats = {
                'dataset_name': dataset_name,
                'computed_at': pd.Timestamp.now().isoformat(),
                'mode': 'full' if full else 'basic',
                'tables': {},
            }

            # Compute statistics for each table
            for table_type, table_name in dataset_info.get('tables', {}).items():
                logger.info(f"Computing statistics for table '{table_type}' ({table_name})")

                table_stats = self._compute_table_statistics(
                    backend, table_name, table_type, full
                )

                if table_stats:
                    stats['tables'][table_type] = table_stats

            # Add dataset-level summary
            stats['summary'] = self._compute_summary(stats['tables'])

            # Save statistics if requested
            if save:
                self._save_statistics(dataset_name, stats)

            return stats

        finally:
            # Close backend connection
            if hasattr(backend, 'close'):
                backend.close()

    def _load_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """Load dataset information from YAML."""
        yaml_file = self.dataset_registry_dir / f"{dataset_name}.yaml"

        if not yaml_file.exists():
            raise DatasetError(f"Dataset '{dataset_name}' not found")

        try:
            with open(yaml_file) as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise DatasetError(f"Failed to load dataset info: {e}") from e

    def _get_backend(self, dataset_name: str, dataset_info: Dict[str, Any]) -> Any:
        """Get storage backend for dataset."""
        backend_type = dataset_info.get('database', {}).get('backend', 'duckdb')
        backend_config = dataset_info.get('database', {}).copy()

        # Add dataset-specific paths
        dataset_dir = self.datasets_dir / dataset_name
        if backend_type == 'duckdb':
            backend_config['database'] = str(dataset_dir / 'dataset.duckdb')
        elif backend_type == 'sqlite':
            backend_config['database'] = str(dataset_dir / 'dataset.db')

        try:
            return BackendFactory.create(backend_type, backend_config)
        except Exception as e:
            raise StorageError(f"Failed to create backend: {e}") from e

    def _compute_table_statistics(
        self,
        backend: Any,
        table_name: str,
        table_type: str,
        full: bool
    ) -> Optional[Dict[str, Any]]:
        """Compute statistics for a single table."""
        try:
            # Get basic info
            row_count = self._get_row_count(backend, table_name)
            columns_info = self._get_columns_info(backend, table_name)

            if not columns_info:
                return None

            stats = {
                'row_count': row_count,
                'column_count': len(columns_info),
                'columns': {},
                'missing_values': {},
            }

            # Get sample data for analysis
            sample_size = min(10000, row_count) if row_count > 0 else 0
            if sample_size > 0:
                # For large datasets, use sampling
                if row_count > 100000:
                    query = f"SELECT * FROM {table_name} USING SAMPLE {sample_size}"
                else:
                    query = f"SELECT * FROM {table_name}"

                df = backend.query(query)

                if df is not None and not df.empty:
                    # Compute column statistics
                    for col in df.columns:
                        col_stats = self._compute_column_statistics(df[col], full)
                        if col_stats:
                            stats['columns'][col] = col_stats

                    # Compute missing values
                    stats['missing_values'] = self._compute_missing_values(df)

                    # Compute correlations if full mode
                    if full:
                        stats['correlations'] = self._compute_correlations(df)
                        stats['data_quality'] = self._compute_data_quality(df)

            # Add memory usage estimate
            stats['estimated_memory_usage'] = self._estimate_memory_usage(
                row_count, columns_info
            )

            return stats

        except Exception as e:
            logger.error(f"Failed to compute statistics for table '{table_name}': {e}")
            return None

    def _get_row_count(self, backend: Any, table_name: str) -> int:
        """Get row count for a table."""
        try:
            result = backend.query(f"SELECT COUNT(*) as count FROM {table_name}")
            if result is not None and not result.empty:
                return int(result.iloc[0]['count'])
        except Exception as e:
            logger.error(f"Failed to get row count: {e}")
        return 0

    def _get_columns_info(self, backend: Any, table_name: str) -> List[Dict[str, Any]]:
        """Get column information for a table."""
        try:
            # This varies by backend, using a generic approach
            result = backend.query(f"SELECT * FROM {table_name} LIMIT 0")
            if result is not None:
                return [
                    {
                        'name': col,
                        'dtype': str(result[col].dtype),
                    }
                    for col in result.columns
                ]
        except Exception as e:
            logger.error(f"Failed to get columns info: {e}")
        return []

    def _compute_column_statistics(
        self,
        series: pd.Series,
        full: bool
    ) -> Dict[str, Any]:
        """Compute statistics for a single column."""
        stats = {
            'dtype': str(series.dtype),
            'null_count': int(series.isna().sum()),
            'null_percentage': float(series.isna().mean() * 100),
            'unique_count': int(series.nunique()),
        }

        # Numeric columns
        if pd.api.types.is_numeric_dtype(series):
            non_null = series.dropna()
            if len(non_null) > 0:
                stats.update({
                    'mean': float(non_null.mean()),
                    'std': float(non_null.std()),
                    'min': float(non_null.min()),
                    'max': float(non_null.max()),
                    'median': float(non_null.median()),
                })

                if full:
                    # Add percentiles
                    percentiles = [0.01, 0.05, 0.25, 0.75, 0.95, 0.99]
                    stats['percentiles'] = {
                        f"p{int(p*100)}": float(non_null.quantile(p))
                        for p in percentiles
                    }

                    # Add skewness and kurtosis
                    try:
                        stats['skewness'] = float(non_null.skew())
                        stats['kurtosis'] = float(non_null.kurtosis())
                    except:
                        pass

        # Categorical columns
        elif pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(series):
            value_counts = series.value_counts()
            stats.update({
                'unique_values': list(value_counts.index[:10].astype(str)),  # Top 10
                'value_counts': value_counts.head(10).to_dict(),
            })

            if full and len(value_counts) > 0:
                # Add cardinality metrics
                stats['cardinality'] = float(len(value_counts) / len(series))
                stats['most_frequent'] = str(value_counts.index[0])
                stats['most_frequent_count'] = int(value_counts.iloc[0])

        # Datetime columns
        elif pd.api.types.is_datetime64_any_dtype(series):
            non_null = series.dropna()
            if len(non_null) > 0:
                stats.update({
                    'min': non_null.min().isoformat(),
                    'max': non_null.max().isoformat(),
                    'range_days': (non_null.max() - non_null.min()).days,
                })

        return stats

    def _compute_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute missing value statistics."""
        missing_counts = df.isna().sum()
        total_cells = len(df) * len(df.columns)
        total_missing = missing_counts.sum()

        return {
            'total_missing_cells': int(total_missing),
            'total_missing_percentage': float(total_missing / total_cells * 100) if total_cells > 0 else 0,
            'columns_with_missing': {
                col: {
                    'count': int(count),
                    'percentage': float(count / len(df) * 100)
                }
                for col, count in missing_counts.items()
                if count > 0
            },
            'rows_with_missing': int((df.isna().any(axis=1)).sum()),
            'complete_rows': int((~df.isna().any(axis=1)).sum()),
        }

    def _compute_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute correlation matrix for numeric columns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) < 2:
            return {}

        try:
            # Compute correlation matrix
            corr_matrix = df[numeric_cols].corr()

            # Find high correlations
            high_corr_pairs = []
            for i in range(len(numeric_cols)):
                for j in range(i + 1, len(numeric_cols)):
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) > 0.8:  # High correlation threshold
                        high_corr_pairs.append({
                            'column1': numeric_cols[i],
                            'column2': numeric_cols[j],
                            'correlation': float(corr_value),
                        })

            return {
                'numeric_columns': numeric_cols,
                'correlation_matrix': corr_matrix.to_dict(),
                'high_correlations': high_corr_pairs,
            }

        except Exception as e:
            logger.warning(f"Failed to compute correlations: {e}")
            return {}

    def _compute_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute data quality metrics."""
        quality_metrics = {
            'completeness': float(1 - df.isna().mean().mean()),
            'duplicate_rows': int(df.duplicated().sum()),
            'duplicate_percentage': float(df.duplicated().mean() * 100),
        }

        # Check for potential ID columns with duplicates
        potential_issues = []
        for col in df.columns:
            if df[col].nunique() == len(df):
                # Potential ID column
                continue
            if df[col].nunique() / len(df) > 0.95:
                # High cardinality column with duplicates
                dup_count = len(df) - df[col].nunique()
                potential_issues.append({
                    'column': col,
                    'issue': 'high_cardinality_with_duplicates',
                    'duplicate_count': dup_count,
                })

        if potential_issues:
            quality_metrics['potential_issues'] = potential_issues

        return quality_metrics

    def _estimate_memory_usage(
        self,
        row_count: int,
        columns_info: List[Dict[str, Any]]
    ) -> int:
        """Estimate memory usage in bytes."""
        # Simple estimation based on column types
        bytes_per_column = {
            'int': 8,
            'float': 8,
            'object': 50,  # Average string length
            'datetime': 8,
            'bool': 1,
        }

        total_bytes = 0
        for col_info in columns_info:
            dtype = col_info.get('dtype', 'object').lower()

            # Map dtype to bytes
            if 'int' in dtype:
                bytes_estimate = bytes_per_column['int']
            elif 'float' in dtype:
                bytes_estimate = bytes_per_column['float']
            elif 'datetime' in dtype:
                bytes_estimate = bytes_per_column['datetime']
            elif 'bool' in dtype:
                bytes_estimate = bytes_per_column['bool']
            else:
                bytes_estimate = bytes_per_column['object']

            total_bytes += bytes_estimate * row_count

        return total_bytes

    def _compute_summary(self, tables_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Compute dataset-level summary from table statistics."""
        total_rows = 0
        total_columns = 0
        total_missing = 0
        total_cells = 0

        for table_name, stats in tables_stats.items():
            if stats:
                total_rows += stats.get('row_count', 0)
                total_columns += stats.get('column_count', 0)

                # Calculate missing cells
                missing_info = stats.get('missing_values', {})
                total_missing += missing_info.get('total_missing_cells', 0)
                total_cells += stats.get('row_count', 0) * stats.get('column_count', 0)

        return {
            'total_rows': total_rows,
            'total_columns': total_columns,
            'total_tables': len(tables_stats),
            'total_missing_cells': total_missing,
            'overall_completeness': float(1 - total_missing / total_cells) if total_cells > 0 else 1.0,
        }

    def _save_statistics(self, dataset_name: str, statistics: Dict[str, Any]) -> None:
        """Save statistics to metadata directory."""
        metadata_dir = self.datasets_dir / dataset_name / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)

        stats_file = metadata_dir / "statistics.json"

        try:
            with open(stats_file, 'w') as f:
                json.dump(statistics, f, indent=2, default=str)
            logger.info(f"Statistics saved to {stats_file}")
        except Exception as e:
            logger.error(f"Failed to save statistics: {e}")

    def load_statistics(self, dataset_name: str) -> Optional[Dict[str, Any]]:
        """Load pre-computed statistics for a dataset."""
        stats_file = self.datasets_dir / dataset_name / "metadata" / "statistics.json"

        if not stats_file.exists():
            return None

        try:
            with open(stats_file) as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load statistics: {e}")
            return None


def compute_dataset_statistics(
    dataset_name: str,
    full: bool = False,
    save: bool = True,
) -> Dict[str, Any]:
    """Compute statistics for a dataset.

    Args:
        dataset_name: Name of the dataset
        full: Whether to compute full statistics
        save: Whether to save statistics

    Returns:
        Dictionary containing statistics
    """
    stats_computer = DatasetStatistics()
    return stats_computer.compute_statistics(dataset_name, full, save)
