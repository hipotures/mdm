"""
Dataset management adapters for existing implementations.

These adapters wrap the legacy DatasetRegistrar and DatasetManager to provide
the interface protocols while maintaining full backward compatibility.
"""
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import pandas as pd
import time
import logging

from ..interfaces.dataset import IDatasetRegistrar, IDatasetManager
from ..dataset.registrar import DatasetRegistrar
from ..dataset.manager import DatasetManager

logger = logging.getLogger(__name__)


class DatasetRegistrarAdapter(IDatasetRegistrar):
    """Adapter for existing dataset registrar with metrics tracking."""
    
    def __init__(self):
        self._registrar = DatasetRegistrar()
        self._metrics = {
            "datasets_registered": 0,
            "registration_time": 0.0,
            "errors": 0,
            "call_count": {}
        }
        logger.info("Initialized DatasetRegistrarAdapter")
    
    def _track_call(self, method: str, success: bool = True, time_taken: float = 0.0):
        """Track method calls and metrics."""
        self._metrics["call_count"][method] = self._metrics["call_count"].get(method, 0) + 1
        if not success:
            self._metrics["errors"] += 1
        if method == "register" and success:
            self._metrics["datasets_registered"] += 1
            self._metrics["registration_time"] += time_taken
        
        logger.debug(f"Registrar method called: {method} "
                    f"(success: {success}, time: {time_taken:.2f}s)")
    
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
        """Register a new dataset with metrics tracking."""
        start_time = time.time()
        success = True
        
        try:
            result = self._registrar.register(
                name=name,
                path=path,
                target=target,
                problem_type=problem_type,
                id_columns=id_columns,
                datetime_columns=datetime_columns,
                force=force
            )
        except Exception as e:
            success = False
            raise
        finally:
            time_taken = time.time() - start_time
            self._track_call("register", success, time_taken)
        
        return result
    
    def validate_dataset_name(self, name: str) -> None:
        """Validate dataset name format."""
        self._track_call("validate_dataset_name")
        return self._registrar.validate_dataset_name(name)
    
    def detect_structure(self, path: str) -> Dict[str, Any]:
        """Auto-detect dataset structure."""
        self._track_call("detect_structure")
        return self._registrar.detect_structure(path)
    
    def detect_file_format(self, file_path: str) -> str:
        """Detect file format from extension."""
        self._track_call("detect_file_format")
        return self._registrar.detect_file_format(file_path)
    
    def load_data_file(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Load data from a single file."""
        self._track_call("load_data_file")
        return self._registrar.load_data_file(file_path, **kwargs)
    
    def detect_column_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """Detect column data types."""
        self._track_call("detect_column_types")
        return self._registrar.detect_column_types(df)
    
    def detect_id_columns(self, df: pd.DataFrame) -> List[str]:
        """Auto-detect ID columns."""
        self._track_call("detect_id_columns")
        return self._registrar.detect_id_columns(df)
    
    def detect_target_column(self, df: pd.DataFrame, columns: List[str]) -> Optional[str]:
        """Auto-detect target column."""
        self._track_call("detect_target_column")
        return self._registrar.detect_target_column(df, columns)
    
    def detect_problem_type(self, df: pd.DataFrame, target_column: str) -> str:
        """Auto-detect problem type from target column."""
        self._track_call("detect_problem_type")
        return self._registrar.detect_problem_type(df, target_column)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get adapter metrics."""
        return self._metrics.copy()


class DatasetManagerAdapter(IDatasetManager):
    """Adapter for existing dataset manager with metrics tracking."""
    
    def __init__(self):
        self._manager = DatasetManager()
        self._metrics = {
            "datasets_loaded": 0,
            "datasets_removed": 0,
            "datasets_exported": 0,
            "total_data_loaded_mb": 0.0,
            "call_count": {}
        }
        logger.info("Initialized DatasetManagerAdapter")
    
    def _track_call(self, method: str, data_size_mb: float = 0.0):
        """Track method calls and metrics."""
        self._metrics["call_count"][method] = self._metrics["call_count"].get(method, 0) + 1
        
        if method == "load_dataset":
            self._metrics["datasets_loaded"] += 1
            self._metrics["total_data_loaded_mb"] += data_size_mb
        elif method == "remove_dataset":
            self._metrics["datasets_removed"] += 1
        elif method == "export_dataset":
            self._metrics["datasets_exported"] += 1
        
        logger.debug(f"Manager method called: {method}")
    
    def list_datasets(
        self, 
        limit: Optional[int] = None,
        sort_by: Optional[str] = None,
        backend: Optional[str] = None,
        tag: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List all datasets."""
        self._track_call("list_datasets")
        
        # The original method might not support all these parameters
        kwargs = {"limit": limit}
        if hasattr(self._manager.list_datasets, '__code__'):
            params = self._manager.list_datasets.__code__.co_varnames
            if 'sort_by' in params:
                kwargs['sort_by'] = sort_by
            if 'backend' in params:
                kwargs['backend'] = backend
            if 'tag' in params:
                kwargs['tag'] = tag
        
        return self._manager.list_datasets(**kwargs)
    
    def get_dataset_info(self, name: str) -> Dict[str, Any]:
        """Get detailed dataset information."""
        self._track_call("get_dataset_info")
        return self._manager.get_dataset_info(name)
    
    def dataset_exists(self, name: str) -> bool:
        """Check if dataset exists."""
        self._track_call("dataset_exists")
        return self._manager.dataset_exists(name)
    
    def remove_dataset(self, name: str, force: bool = False) -> None:
        """Remove dataset and all associated data."""
        self._track_call("remove_dataset")
        return self._manager.remove_dataset(name, force)
    
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
        self._track_call("update_dataset")
        
        # Build update dict with only provided values
        updates = {}
        if description is not None:
            updates['description'] = description
        if target is not None:
            updates['target'] = target
        if problem_type is not None:
            updates['problem_type'] = problem_type
        if id_columns is not None:
            updates['id_columns'] = id_columns
        if tags is not None:
            updates['tags'] = tags
        
        return self._manager.update_dataset(name, updates)
    
    def export_dataset(
        self, 
        name: str, 
        output_dir: str,
        format: str = "csv",
        compression: Optional[str] = None,
        tables: Optional[List[str]] = None
    ) -> List[str]:
        """Export dataset to files."""
        self._track_call("export_dataset")
        
        # The original method might have different parameter names
        kwargs = {
            "dataset_name": name,
            "output_dir": output_dir,
            "format": format
        }
        
        if hasattr(self._manager.export_dataset, '__code__'):
            params = self._manager.export_dataset.__code__.co_varnames
            if 'compression' in params:
                kwargs['compression'] = compression
            if 'tables' in params:
                kwargs['tables'] = tables
        
        result = self._manager.export_dataset(**kwargs)
        
        # Ensure we return a list of paths
        if isinstance(result, str):
            return [result]
        elif isinstance(result, dict):
            return list(result.values())
        else:
            return result
    
    def get_dataset_stats(
        self, 
        name: str,
        mode: str = "basic",
        tables: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Get dataset statistics."""
        self._track_call("get_dataset_stats")
        
        # Check if the method exists
        if hasattr(self._manager, 'get_dataset_stats'):
            return self._manager.get_dataset_stats(name, mode, tables)
        else:
            # Fallback to basic info
            info = self.get_dataset_info(name)
            return {
                "dataset_name": name,
                "mode": mode,
                "tables": info.get("tables", {}),
                "total_rows": info.get("total_rows", 0),
                "total_columns": info.get("total_columns", 0)
            }
    
    def search_datasets(
        self,
        pattern: str,
        search_in: List[str] = ["name", "description", "tags"],
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Search datasets by pattern."""
        self._track_call("search_datasets")
        
        # Check if the method exists
        if hasattr(self._manager, 'search_datasets'):
            return self._manager.search_datasets(pattern, search_in, limit)
        else:
            # Simple fallback implementation
            all_datasets = self.list_datasets()
            results = []
            
            pattern_lower = pattern.lower()
            for dataset in all_datasets:
                for field in search_in:
                    value = dataset.get(field, "")
                    if isinstance(value, list):
                        value = " ".join(value)
                    if pattern_lower in str(value).lower():
                        results.append(dataset)
                        break
            
            if limit:
                results = results[:limit]
            
            return results
    
    def load_dataset(
        self,
        name: str,
        table_name: str = "data",
        columns: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """Load dataset data."""
        self._track_call("load_dataset")
        
        # Load the data
        if hasattr(self._manager, 'load_dataset'):
            data = self._manager.load_dataset(name, table_name, columns, limit)
        else:
            # Use storage backend directly
            from ..storage import get_storage_backend
            backend = get_storage_backend()
            data = backend.load_data(name, table_name)
            
            # Apply column filter
            if columns:
                data = data[columns]
            
            # Apply limit
            if limit:
                data = data.head(limit)
        
        # Track data size
        data_size_mb = data.memory_usage(deep=True).sum() / 1024 / 1024
        self._metrics["total_data_loaded_mb"] += data_size_mb
        
        return data
    
    def get_dataset_config(self, name: str) -> Dict[str, Any]:
        """Get dataset configuration from YAML file."""
        self._track_call("get_dataset_config")
        
        if hasattr(self._manager, 'get_dataset_config'):
            return self._manager.get_dataset_config(name)
        else:
            # Read from YAML file directly
            from ..config import get_config
            import yaml
            
            config = get_config()
            config_path = Path(config.paths.datasets_config) / f"{name}.yaml"
            
            if config_path.exists():
                with open(config_path) as f:
                    return yaml.safe_load(f)
            else:
                raise FileNotFoundError(f"Dataset config not found: {name}")
    
    def update_dataset_config(self, name: str, config: Dict[str, Any]) -> None:
        """Update dataset configuration YAML file."""
        self._track_call("update_dataset_config")
        
        if hasattr(self._manager, 'update_dataset_config'):
            return self._manager.update_dataset_config(name, config)
        else:
            # Write to YAML file directly
            from ..config import get_config
            import yaml
            
            mdm_config = get_config()
            config_path = Path(mdm_config.paths.datasets_config) / f"{name}.yaml"
            
            # Merge with existing config
            if config_path.exists():
                with open(config_path) as f:
                    existing = yaml.safe_load(f) or {}
                existing.update(config)
                config = existing
            
            # Write updated config
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get adapter metrics."""
        return self._metrics.copy()