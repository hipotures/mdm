"""Storage backend migration utilities.

This module provides tools for migrating datasets between different storage
backends during the refactoring process.
"""
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import logging
import time
from datetime import datetime

import pandas as pd
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskID
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from mdm.interfaces.storage import IStorageBackend
from mdm.adapters.storage_manager import get_storage_backend
from mdm.core.exceptions import StorageError, DatasetError

logger = logging.getLogger(__name__)
console = Console()


class StorageMigrator:
    """Handles migration of datasets between storage backends."""
    
    def __init__(self, source_type: str, target_type: str, config: Optional[Dict[str, Any]] = None):
        """Initialize migrator.
        
        Args:
            source_type: Source backend type
            target_type: Target backend type
            config: Optional configuration for backends
        """
        self.source_type = source_type
        self.target_type = target_type
        self.config = config or {}
        
        # Get backend instances
        self.source_backend = get_storage_backend(source_type, config.get(source_type, {}))
        self.target_backend = get_storage_backend(target_type, config.get(target_type, {}))
        
        # Migration statistics
        self.stats = {
            "datasets_migrated": 0,
            "datasets_failed": 0,
            "total_rows": 0,
            "total_size_bytes": 0,
            "start_time": None,
            "end_time": None,
        }
    
    def migrate_dataset(
        self, 
        dataset_name: str,
        verify: bool = True,
        batch_size: int = 10000
    ) -> Dict[str, Any]:
        """Migrate a single dataset.
        
        Args:
            dataset_name: Name of dataset to migrate
            verify: Whether to verify migration
            batch_size: Batch size for data transfer
            
        Returns:
            Migration result dictionary
        """
        result = {
            "dataset_name": dataset_name,
            "success": False,
            "error": None,
            "rows_migrated": 0,
            "tables_migrated": [],
            "duration_seconds": 0,
        }
        
        start_time = time.time()
        
        try:
            # Check if source dataset exists
            if not self.source_backend.dataset_exists(dataset_name):
                raise DatasetError(f"Source dataset '{dataset_name}' does not exist")
            
            # Create target dataset
            console.print(f"Creating target dataset: {dataset_name}")
            self.target_backend.create_dataset(dataset_name, {"source": self.source_type})
            
            # Get source engine
            source_path = self._get_dataset_path(dataset_name, self.source_backend)
            source_engine = self.source_backend.get_engine(source_path)
            
            # Get target engine
            target_path = self._get_dataset_path(dataset_name, self.target_backend)
            target_engine = self.target_backend.get_engine(target_path)
            
            # Get all tables
            tables = self.source_backend.get_table_names(source_engine)
            
            # Migrate each table
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=console
            ) as progress:
                
                for table in tables:
                    task_id = progress.add_task(f"Migrating {table}", total=None)
                    
                    # Get table info
                    table_info = self.source_backend.get_table_info(table, source_engine)
                    row_count = table_info.get("row_count", 0)
                    
                    if row_count > 0:
                        progress.update(task_id, total=row_count)
                        
                        # Migrate in batches
                        offset = 0
                        while offset < row_count:
                            # Read batch
                            batch_df = self.source_backend.read_table(
                                table,
                                limit=batch_size,
                                engine=source_engine
                            )
                            
                            if batch_df.empty:
                                break
                            
                            # Write batch
                            if_exists = "replace" if offset == 0 else "append"
                            self.target_backend.write_table(
                                table,
                                batch_df,
                                if_exists=if_exists,
                                engine=target_engine
                            )
                            
                            offset += len(batch_df)
                            result["rows_migrated"] += len(batch_df)
                            progress.update(task_id, advance=len(batch_df))
                    
                    result["tables_migrated"].append(table)
                    progress.remove_task(task_id)
            
            # Migrate metadata
            metadata = self.source_backend.get_metadata(dataset_name)
            metadata["migrated_from"] = self.source_type
            metadata["migrated_to"] = self.target_type
            metadata["migration_date"] = datetime.now().isoformat()
            self.target_backend.update_metadata(dataset_name, metadata)
            
            # Verify if requested
            if verify:
                console.print("Verifying migration...")
                verification = self._verify_migration(
                    dataset_name,
                    source_engine,
                    target_engine,
                    tables
                )
                result["verification"] = verification
                
                if not verification["matches"]:
                    raise DatasetError(
                        f"Verification failed: {verification['differences']}"
                    )
            
            result["success"] = True
            self.stats["datasets_migrated"] += 1
            self.stats["total_rows"] += result["rows_migrated"]
            
        except Exception as e:
            logger.error(f"Failed to migrate dataset {dataset_name}: {e}")
            result["error"] = str(e)
            self.stats["datasets_failed"] += 1
            
            # Clean up partial migration
            try:
                if self.target_backend.dataset_exists(dataset_name):
                    self.target_backend.drop_dataset(dataset_name)
            except:
                pass
        
        finally:
            result["duration_seconds"] = time.time() - start_time
        
        return result
    
    def migrate_all_datasets(
        self,
        dataset_names: Optional[List[str]] = None,
        verify: bool = True,
        parallel: bool = False
    ) -> Dict[str, Any]:
        """Migrate multiple datasets.
        
        Args:
            dataset_names: List of datasets to migrate (None for all)
            verify: Whether to verify each migration
            parallel: Whether to migrate in parallel
            
        Returns:
            Migration summary
        """
        self.stats["start_time"] = datetime.now()
        
        # Get list of datasets to migrate
        if dataset_names is None:
            # TODO: Implement listing all datasets
            raise NotImplementedError("Listing all datasets not yet implemented")
        
        results = []
        
        # Show migration plan
        self._show_migration_plan(dataset_names)
        
        # Migrate each dataset
        for dataset_name in dataset_names:
            console.print(f"\n[bold]Migrating dataset: {dataset_name}[/bold]")
            result = self.migrate_dataset(dataset_name, verify=verify)
            results.append(result)
            
            # Show result
            if result["success"]:
                console.print(f"[green]✓ Successfully migrated {dataset_name}[/green]")
            else:
                console.print(f"[red]✗ Failed to migrate {dataset_name}: {result['error']}[/red]")
        
        self.stats["end_time"] = datetime.now()
        
        # Show summary
        self._show_migration_summary(results)
        
        return {
            "stats": self.stats,
            "results": results
        }
    
    def _verify_migration(
        self,
        dataset_name: str,
        source_engine,
        target_engine,
        tables: List[str]
    ) -> Dict[str, Any]:
        """Verify that migration was successful.
        
        Args:
            dataset_name: Dataset name
            source_engine: Source database engine
            target_engine: Target database engine
            tables: List of tables to verify
            
        Returns:
            Verification result
        """
        verification = {
            "matches": True,
            "differences": [],
            "checks_performed": []
        }
        
        for table in tables:
            # Check row counts
            source_info = self.source_backend.get_table_info(table, source_engine)
            target_info = self.target_backend.get_table_info(table, target_engine)
            
            source_rows = source_info.get("row_count", 0)
            target_rows = target_info.get("row_count", 0)
            
            if source_rows != target_rows:
                verification["matches"] = False
                verification["differences"].append(
                    f"Table {table}: row count mismatch "
                    f"(source: {source_rows}, target: {target_rows})"
                )
            
            verification["checks_performed"].append({
                "table": table,
                "check": "row_count",
                "source": source_rows,
                "target": target_rows,
                "matches": source_rows == target_rows
            })
            
            # Check column names
            source_cols = set(col["name"] for col in source_info["columns"])
            target_cols = set(col["name"] for col in target_info["columns"])
            
            if source_cols != target_cols:
                verification["matches"] = False
                missing = source_cols - target_cols
                extra = target_cols - source_cols
                
                if missing:
                    verification["differences"].append(
                        f"Table {table}: missing columns {missing}"
                    )
                if extra:
                    verification["differences"].append(
                        f"Table {table}: extra columns {extra}"
                    )
            
            verification["checks_performed"].append({
                "table": table,
                "check": "columns",
                "matches": source_cols == target_cols
            })
        
        return verification
    
    def _get_dataset_path(self, dataset_name: str, backend: IStorageBackend) -> str:
        """Get dataset path for a backend."""
        base_path = Path.home() / ".mdm" / "datasets"
        
        if backend.backend_type == "postgresql":
            # PostgreSQL uses database names
            prefix = self.config.get(backend.backend_type, {}).get("database_prefix", "mdm_")
            return f"{prefix}{dataset_name}"
        else:
            # File-based backends
            dataset_dir = base_path / dataset_name
            ext = "db" if backend.backend_type == "sqlite" else "duckdb"
            return str(dataset_dir / f"{dataset_name}.{ext}")
    
    def _show_migration_plan(self, dataset_names: List[str]) -> None:
        """Show migration plan to user."""
        table = Table(title="Migration Plan")
        table.add_column("Dataset", style="cyan")
        table.add_column("Source", style="yellow")
        table.add_column("Target", style="green")
        
        for name in dataset_names:
            table.add_row(name, self.source_type, self.target_type)
        
        console.print(table)
    
    def _show_migration_summary(self, results: List[Dict[str, Any]]) -> None:
        """Show migration summary."""
        duration = (self.stats["end_time"] - self.stats["start_time"]).total_seconds()
        
        # Create summary table
        table = Table(title="Migration Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Datasets", str(len(results)))
        table.add_row("Successful", str(self.stats["datasets_migrated"]))
        table.add_row("Failed", str(self.stats["datasets_failed"]))
        table.add_row("Total Rows", f"{self.stats['total_rows']:,}")
        table.add_row("Duration", f"{duration:.2f} seconds")
        
        console.print("\n")
        console.print(table)
        
        # Show failed datasets
        if self.stats["datasets_failed"] > 0:
            console.print("\n[red]Failed Datasets:[/red]")
            for result in results:
                if not result["success"]:
                    console.print(f"  - {result['dataset_name']}: {result['error']}")


class StorageValidator:
    """Validates storage backend implementations."""
    
    def __init__(self, backend: IStorageBackend):
        """Initialize validator.
        
        Args:
            backend: Storage backend to validate
        """
        self.backend = backend
        self.results = {
            "passed": [],
            "failed": [],
            "warnings": []
        }
    
    def validate_all(self) -> Dict[str, Any]:
        """Run all validation tests.
        
        Returns:
            Validation results
        """
        tests = [
            self._test_basic_operations,
            self._test_dataset_operations,
            self._test_table_operations,
            self._test_metadata_operations,
            self._test_error_handling,
        ]
        
        for test in tests:
            try:
                test()
            except Exception as e:
                self.results["failed"].append({
                    "test": test.__name__,
                    "error": str(e)
                })
        
        return self.results
    
    def _test_basic_operations(self) -> None:
        """Test basic backend operations."""
        # Test backend type
        assert hasattr(self.backend, "backend_type")
        assert isinstance(self.backend.backend_type, str)
        self.results["passed"].append("backend_type property")
        
        # Test engine creation
        test_db = "/tmp/mdm_test.db"
        engine = self.backend.get_engine(test_db)
        assert engine is not None
        self.results["passed"].append("get_engine")
        
        # Test connection
        conn = self.backend.get_connection()
        assert conn is not None
        conn.close()
        self.results["passed"].append("get_connection")
    
    def _test_dataset_operations(self) -> None:
        """Test dataset-level operations."""
        test_dataset = "test_validation_dataset"
        
        # Clean up if exists
        if self.backend.dataset_exists(test_dataset):
            self.backend.drop_dataset(test_dataset)
        
        # Test create
        self.backend.create_dataset(test_dataset, {"test": True})
        assert self.backend.dataset_exists(test_dataset)
        self.results["passed"].append("create_dataset")
        
        # Test drop
        self.backend.drop_dataset(test_dataset)
        assert not self.backend.dataset_exists(test_dataset)
        self.results["passed"].append("drop_dataset")
    
    def _test_table_operations(self) -> None:
        """Test table operations."""
        # Create test dataset
        test_dataset = "test_table_ops"
        self.backend.create_dataset(test_dataset, {})
        
        try:
            # Test save data
            test_df = pd.DataFrame({
                "id": range(100),
                "value": [f"test_{i}" for i in range(100)]
            })
            self.backend.save_data(test_dataset, test_df)
            self.results["passed"].append("save_data")
            
            # Test load data
            loaded_df = self.backend.load_data(test_dataset)
            assert len(loaded_df) == 100
            assert list(loaded_df.columns) == ["id", "value"]
            self.results["passed"].append("load_data")
            
        finally:
            # Clean up
            self.backend.drop_dataset(test_dataset)
    
    def _test_metadata_operations(self) -> None:
        """Test metadata operations."""
        test_dataset = "test_metadata_ops"
        self.backend.create_dataset(test_dataset, {})
        
        try:
            # Test update metadata
            test_metadata = {
                "description": "Test dataset",
                "created_by": "validator",
                "version": "1.0"
            }
            self.backend.update_metadata(test_dataset, test_metadata)
            self.results["passed"].append("update_metadata")
            
            # Test get metadata
            loaded_metadata = self.backend.get_metadata(test_dataset)
            assert loaded_metadata["description"] == "Test dataset"
            self.results["passed"].append("get_metadata")
            
        finally:
            # Clean up
            self.backend.drop_dataset(test_dataset)
    
    def _test_error_handling(self) -> None:
        """Test error handling."""
        # Test non-existent dataset
        try:
            self.backend.load_data("non_existent_dataset")
            self.results["failed"].append({
                "test": "error_handling",
                "error": "Should raise error for non-existent dataset"
            })
        except:
            self.results["passed"].append("error_handling")