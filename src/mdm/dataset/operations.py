"""Dataset operations for MDM."""

import fnmatch
import json
import logging
import shutil
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from mdm.config import get_config_manager
from mdm.core.exceptions import DatasetError
from mdm.dataset.exporter import DatasetExporter
from mdm.dataset.statistics import DatasetStatistics
from mdm.storage.factory import BackendFactory

logger = logging.getLogger(__name__)


class DatasetOperation(ABC):
    """Base class for dataset operations."""

    def __init__(self):
        """Initialize operation."""
        config_manager = get_config_manager()
        self.config = config_manager.config
        self.base_path = config_manager.base_path
        self.dataset_registry_dir = self.base_path / self.config.paths.configs_path
        self.datasets_dir = self.base_path / self.config.paths.datasets_path

    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """Execute the operation."""
        pass


class ListOperation(DatasetOperation):
    """List datasets with filtering and sorting."""

    def execute(
        self,
        format: str = "rich",
        filter_str: Optional[str] = None,
        sort_by: str = "name",
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """List datasets.

        Args:
            format: Output format (rich, text, filename)
            filter_str: Filter string (e.g., "problem_type=classification")
            sort_by: Sort field (name, registration_date)
            limit: Maximum number of results

        Returns:
            List of dataset information dictionaries
        """
        start_time = time.time()
        datasets = []

        # Ensure registry directory exists
        self.dataset_registry_dir.mkdir(parents=True, exist_ok=True)

        # Parse YAML files in parallel
        yaml_files = list(self.dataset_registry_dir.glob("*.yaml"))

        if not yaml_files:
            return datasets

        # Use thread pool for parallel YAML parsing
        with ThreadPoolExecutor(max_workers=min(len(yaml_files), self.config.performance.max_concurrent_operations)) as executor:
            future_to_file = {
                executor.submit(self._parse_yaml_file, yaml_file): yaml_file
                for yaml_file in yaml_files
            }

            for future in as_completed(future_to_file):
                try:
                    dataset_info = future.result()
                    if dataset_info:
                        datasets.append(dataset_info)
                except Exception as e:
                    yaml_file = future_to_file[future]
                    logger.error(f"Failed to parse {yaml_file}: {e}")

        # Apply filters
        if filter_str:
            datasets = self._apply_filters(datasets, filter_str)

        # Sort datasets
        datasets = self._sort_datasets(datasets, sort_by)

        # Apply limit
        if limit:
            datasets = datasets[:limit]

        elapsed = time.time() - start_time
        logger.info(f"Listed {len(datasets)} datasets in {elapsed:.3f}s")

        return datasets

    def _parse_yaml_file(self, yaml_file: Path) -> Optional[Dict[str, Any]]:
        """Parse a single YAML file quickly."""
        try:
            with open(yaml_file) as f:
                data = yaml.safe_load(f)

            # Extract essential fields
            stats = data.get('metadata', {}).get('statistics', {})
            return {
                'name': data.get('name', yaml_file.stem),
                'display_name': data.get('display_name', data.get('name', yaml_file.stem)),
                'problem_type': data.get('problem_type'),
                'target_column': data.get('target_column'),
                'tables': data.get('tables', {}),
                'description': data.get('description', ''),
                'tags': data.get('tags', []),
                'created_at': data.get('created_at'),
                'registration_date': data.get('created_at'),  # Alias for sorting
                'database': data.get('database', {}),
                'source': data.get('source', 'Unknown'),
                'row_count': stats.get('row_count'),  # From saved statistics
                'size': stats.get('memory_size_bytes'),  # Memory size from statistics
            }
        except Exception as e:
            logger.error(f"Failed to parse {yaml_file}: {e}")
            return None

    def _apply_filters(self, datasets: List[Dict[str, Any]], filter_str: str) -> List[Dict[str, Any]]:
        """Apply filters to dataset list."""
        filtered = []

        # Parse filter string (e.g., "problem_type=classification")
        filters = {}
        for part in filter_str.split(','):
            if '=' in part:
                key, value = part.split('=', 1)
                filters[key.strip()] = value.strip()

        for dataset in datasets:
            match = True
            for key, value in filters.items():
                dataset_value = dataset.get(key)
                if dataset_value is None:
                    match = False
                    break
                if str(dataset_value).lower() != value.lower():
                    match = False
                    break
            if match:
                filtered.append(dataset)

        return filtered

    def _sort_datasets(self, datasets: List[Dict[str, Any]], sort_by: str) -> List[Dict[str, Any]]:
        """Sort datasets by specified field."""
        if sort_by == "registration_date":
            # Sort by date, handling None values
            return sorted(
                datasets,
                key=lambda d: d.get('registration_date') or datetime.min.isoformat(),
                reverse=True
            )
        # Default: sort by name
        return sorted(datasets, key=lambda d: d.get('name', ''))



class InfoOperation(DatasetOperation):
    """Get detailed dataset information."""

    def execute(self, name: str, details: bool = False) -> Dict[str, Any]:
        """Get dataset information.

        Args:
            name: Dataset name
            details: Whether to include detailed statistics

        Returns:
            Dataset information dictionary
        """
        yaml_file = self.dataset_registry_dir / f"{name}.yaml"

        if not yaml_file.exists():
            raise DatasetError(f"Dataset '{name}' not found")

        try:
            with open(yaml_file) as f:
                data = yaml.safe_load(f)

            # Add dataset directory info
            dataset_dir = self.datasets_dir / name
            if dataset_dir.exists():
                data['dataset_path'] = str(dataset_dir)

                # Calculate size
                size = sum(f.stat().st_size for f in dataset_dir.rglob('*') if f.is_file())
                data['total_size'] = size

                # Get database file info
                backend = data.get('database', {}).get('backend', 'duckdb')
                if backend == 'duckdb':
                    db_file = dataset_dir / 'dataset.duckdb'
                    if db_file.exists():
                        data['database_file'] = str(db_file)
                        data['database_size'] = db_file.stat().st_size

            # Add detailed statistics if requested
            if details:
                data['statistics'] = self._get_detailed_statistics(name, data)

            return data

        except Exception as e:
            raise DatasetError(f"Failed to get dataset info: {e}") from e

    def _get_detailed_statistics(self, name: str, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """Get detailed statistics for dataset."""
        # This will be implemented with the statistics module
        return {
            'note': 'Detailed statistics will be available after statistics module implementation'
        }


class SearchOperation(DatasetOperation):
    """Search for datasets."""

    def execute(
        self,
        query: str,
        deep: bool = False,
        pattern: bool = False,
        case_sensitive: bool = False,
        tag: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Search for datasets.

        Args:
            query: Search query
            deep: Whether to search in database metadata
            pattern: Whether to use glob patterns
            case_sensitive: Whether search is case-sensitive
            tag: Search for datasets with specific tag
            limit: Maximum number of results

        Returns:
            List of matching datasets
        """
        matches = []
        count = 0

        # Ensure registry directory exists
        self.dataset_registry_dir.mkdir(parents=True, exist_ok=True)

        # Process query
        query_lower = query.lower() if not case_sensitive else query

        for yaml_file in self.dataset_registry_dir.glob("*.yaml"):
            if limit and count >= limit:
                break

            try:
                with open(yaml_file) as f:
                    data = yaml.safe_load(f)
                
                # If tag is specified, check if dataset has that tag
                if tag:
                    dataset_tags = data.get('tags', [])
                    if tag not in dataset_tags:
                        continue
                
                # If searching by tag only (query == tag), we already found a match
                # Otherwise, check if file matches the query
                if query != tag and not self._matches_file(yaml_file, query, pattern, case_sensitive, deep):
                    continue
                
                match_info = {
                    'name': data.get('name', yaml_file.stem),
                    'display_name': data.get('display_name', data.get('name')),
                    'description': data.get('description', ''),
                    'problem_type': data.get('problem_type'),
                    'target_column': data.get('target_column'),
                    'tags': data.get('tags', []),
                    'file': str(yaml_file),
                }
                
                # Add match location info
                if self._matches_name(yaml_file, query, pattern, case_sensitive):
                    match_info['match_location'] = 'name'
                elif tag and query == tag:
                    match_info['match_location'] = 'tag'
                else:
                    match_info['match_location'] = 'content'
                
                matches.append(match_info)
                count += 1
            except Exception as e:
                logger.error(f"Error searching {yaml_file}: {e}")

        return matches

    def _matches_name(
        self,
        yaml_file: Path,
        query: str,
        pattern: bool,
        case_sensitive: bool
    ) -> bool:
        """Check if filename matches search criteria."""
        filename = yaml_file.stem
        check_query = query
        if not case_sensitive:
            filename = filename.lower()
            check_query = query.lower()

        if pattern:
            return fnmatch.fnmatch(filename, check_query)
        else:
            return check_query in filename

    def _matches_file(
        self,
        yaml_file: Path,
        query: str,
        pattern: bool,
        case_sensitive: bool,
        deep: bool
    ) -> bool:
        """Check if file matches search criteria."""
        # First check filename
        if self._matches_name(yaml_file, query, pattern, case_sensitive):
            return True

        # Then check file contents
        try:
            with open(yaml_file) as f:
                content = f.read()
                check_query = query
                if not case_sensitive:
                    content = content.lower()
                    check_query = query.lower()

                if check_query in content:
                    return True

            # Deep search would open database
            if deep:
                # TODO: Implement deep search in database metadata
                pass

        except Exception:
            pass

        return False


class ExportOperation(DatasetOperation):
    """Export dataset to various formats."""

    def execute(
        self,
        name: str,
        format: str = "csv",
        output_dir: Optional[Path] = None,
        table: Optional[str] = None,
        compression: Optional[str] = None,
        metadata_only: bool = False,
        no_header: bool = False,
    ) -> List[Path]:
        """Export dataset.

        Args:
            name: Dataset name
            format: Export format (csv, parquet, json)
            output_dir: Output directory
            table: Specific table to export (default: all)
            compression: Compression type
            metadata_only: Export only metadata
            no_header: Exclude header (CSV only)

        Returns:
            List of exported file paths
        """
        exporter = DatasetExporter()
        return exporter.export(
            dataset_name=name,
            format=format,
            output_dir=output_dir,
            table=table,
            compression=compression,
            metadata_only=metadata_only,
            no_header=no_header,
        )


class StatsOperation(DatasetOperation):
    """Compute and display dataset statistics."""

    def execute(
        self,
        name: str,
        full: bool = False,
        export: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Get dataset statistics.

        Args:
            name: Dataset name
            full: Whether to compute full statistics
            export: Path to export statistics

        Returns:
            Statistics dictionary
        """
        stats_computer = DatasetStatistics()

        # Try to load pre-computed statistics first
        stats = stats_computer.load_statistics(name)

        if not stats or full:
            # Compute fresh statistics
            stats = stats_computer.compute_statistics(name, full=full, save=True)

        # Export if requested
        if export:
            export_path = Path(export)
            try:
                if export_path.suffix.lower() == '.yaml':
                    with open(export_path, 'w') as f:
                        yaml.dump(stats, f, default_flow_style=False)
                else:
                    # Default to JSON
                    with open(export_path, 'w') as f:
                        json.dump(stats, f, indent=2, default=str)
                logger.info(f"Statistics exported to {export_path}")
            except Exception as e:
                logger.error(f"Failed to export statistics: {e}")

        return stats


class UpdateOperation(DatasetOperation):
    """Update dataset metadata."""

    def execute(
        self,
        name: str,
        description: Optional[str] = None,
        target: Optional[str] = None,
        problem_type: Optional[str] = None,
        id_columns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Update dataset metadata.

        Args:
            name: Dataset name
            description: New description
            target: New target column
            problem_type: New problem type
            id_columns: New ID columns

        Returns:
            Updated dataset information
        """
        yaml_file = self.dataset_registry_dir / f"{name}.yaml"

        if not yaml_file.exists():
            raise DatasetError(f"Dataset '{name}' not found")

        try:
            # Load existing data
            with open(yaml_file) as f:
                data = yaml.safe_load(f)

            # Update fields
            if description is not None:
                data['description'] = description
            if target is not None:
                data['target_column'] = target
            if problem_type is not None:
                data['problem_type'] = problem_type
            if id_columns is not None:
                data['id_columns'] = id_columns

            # Update timestamp
            data['last_updated_at'] = datetime.utcnow().isoformat()

            # Save updated data
            with open(yaml_file, 'w') as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)

            # Also update in database metadata if needed
            self._update_database_metadata(name, data)

            return data

        except Exception as e:
            raise DatasetError(f"Failed to update dataset: {e}") from e

    def _update_database_metadata(self, name: str, data: Dict[str, Any]) -> None:
        """Update metadata in database."""
        try:
            dataset_dir = self.datasets_dir / name
            if not dataset_dir.exists():
                return

            backend_type = data.get('database', {}).get('backend', 'duckdb')
            backend_config = data.get('database', {})

            # Create backend instance
            backend = BackendFactory.create(backend_type, backend_config)

            # Update metadata table
            # TODO: Implement metadata update in backend

        except Exception as e:
            logger.warning(f"Failed to update database metadata: {e}")


class RemoveOperation(DatasetOperation):
    """Remove dataset."""

    def execute(self, name: str, force: bool = False, dry_run: bool = False) -> Dict[str, Any]:
        """Remove dataset.

        Args:
            name: Dataset name
            force: Skip confirmation
            dry_run: Preview what would be deleted

        Returns:
            Removal information
        """
        yaml_file = self.dataset_registry_dir / f"{name}.yaml"
        dataset_dir = self.datasets_dir / name

        if not yaml_file.exists():
            raise DatasetError(f"Dataset '{name}' not found")

        removal_info = {
            'name': name,
            'config_file': str(yaml_file),
            'dataset_directory': str(dataset_dir) if dataset_dir.exists() else None,
            'size': 0,
        }

        # Calculate size if directory exists
        if dataset_dir.exists():
            removal_info['size'] = sum(
                f.stat().st_size for f in dataset_dir.rglob('*') if f.is_file()
            )

        # Check for PostgreSQL database
        try:
            with open(yaml_file) as f:
                data = yaml.safe_load(f)
                if data.get('database', {}).get('backend') == 'postgresql':
                    removal_info['postgresql_db'] = f"{data.get('database', {}).get('database_prefix', 'mdm_')}{name}"
        except:
            pass

        if dry_run:
            removal_info['dry_run'] = True
            return removal_info

        # Atomic removal: first remove YAML (prevents listing), then database
        try:
            # Step 1: Remove YAML config
            yaml_file.unlink()
            logger.info(f"Removed configuration: {yaml_file}")

            # Step 2: Remove dataset directory
            if dataset_dir.exists():
                shutil.rmtree(dataset_dir)
                logger.info(f"Removed dataset directory: {dataset_dir}")

            # Step 3: Remove PostgreSQL database if applicable
            if 'postgresql_db' in removal_info:
                # TODO: Implement PostgreSQL database removal
                pass

            removal_info['status'] = 'success'
            return removal_info

        except Exception as e:
            removal_info['status'] = 'failed'
            removal_info['error'] = str(e)
            raise DatasetError(f"Failed to remove dataset: {e}") from e
