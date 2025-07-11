"""Infrastructure implementation of feature repository."""

import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd

from loguru import logger

from mdm.config import get_config_manager
from mdm.features.domain import (
    IFeatureRepository,
    FeatureSet,
    FeatureMetadata
)
from mdm.models.enums import ColumnType


class FileSystemFeatureRepository(IFeatureRepository):
    """Repository implementation using file system for metadata storage."""
    
    def __init__(self, base_path: Optional[Path] = None):
        """Initialize file system repository.
        
        Args:
            base_path: Base path for dataset storage
        """
        if base_path is None:
            config_manager = get_config_manager()
            self.base_path = config_manager.base_path
            self.datasets_path = self.base_path / config_manager.config.paths.datasets_path
        else:
            self.base_path = base_path
            self.datasets_path = base_path / "datasets"
    
    def save_features(self, dataset_name: str, table_name: str, 
                     feature_set: FeatureSet) -> None:
        """Save a feature set to storage.
        
        Note: This implementation only saves metadata. 
        Actual feature data is handled by IFeatureWriter.
        
        Args:
            dataset_name: Name of the dataset
            table_name: Name of the feature table
            feature_set: FeatureSet to save
        """
        # Save metadata
        self.save_metadata(dataset_name, feature_set.metadata)
        
        # Save feature set info
        feature_info = {
            'table_name': table_name,
            'feature_count': feature_set.feature_count,
            'generation_time': feature_set.generation_time,
            'discarded_count': feature_set.discarded_count,
            'created_at': datetime.now().isoformat(),
            'feature_names': feature_set.get_feature_names()
        }
        
        # Save to JSON file
        feature_info_path = self._get_feature_info_path(dataset_name, table_name)
        feature_info_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(feature_info_path, 'w') as f:
            json.dump(feature_info, f, indent=2)
        
        logger.info(f"Saved feature set info for {table_name}")
    
    def load_features(self, dataset_name: str, table_name: str) -> Optional[FeatureSet]:
        """Load a feature set from storage.
        
        Note: This implementation only loads metadata.
        Actual feature data should be loaded via IFeatureLoader.
        
        Args:
            dataset_name: Name of the dataset
            table_name: Name of the feature table
            
        Returns:
            FeatureSet if found, None otherwise
        """
        # Load feature info
        feature_info_path = self._get_feature_info_path(dataset_name, table_name)
        
        if not feature_info_path.exists():
            return None
        
        try:
            with open(feature_info_path) as f:
                feature_info = json.load(f)
            
            # Load metadata
            metadata = self.load_metadata(dataset_name)
            
            # Filter metadata for this table's features
            table_metadata = {
                name: meta for name, meta in metadata.items()
                if name in feature_info.get('feature_names', [])
            }
            
            # Create empty DataFrame as placeholder
            # Actual data should be loaded via IFeatureLoader
            features_df = pd.DataFrame()
            
            return FeatureSet(
                features=features_df,
                metadata=table_metadata,
                generation_time=feature_info.get('generation_time', 0),
                feature_count=feature_info.get('feature_count', 0),
                discarded_count=feature_info.get('discarded_count', 0)
            )
            
        except Exception as e:
            logger.error(f"Failed to load feature set for {table_name}: {e}")
            return None
    
    def save_metadata(self, dataset_name: str, metadata: Dict[str, FeatureMetadata]) -> None:
        """Save feature metadata.
        
        Args:
            dataset_name: Name of the dataset
            metadata: Dictionary of feature metadata
        """
        metadata_path = self._get_metadata_path(dataset_name)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert metadata to serializable format
        metadata_dict = {}
        for name, meta in metadata.items():
            metadata_dict[name] = {
                'name': meta.name,
                'source_columns': meta.source_columns,
                'transformer_name': meta.transformer_name,
                'column_type': meta.column_type.value if hasattr(meta.column_type, 'value') else str(meta.column_type),
                'created_at': meta.created_at.isoformat()
            }
        
        # Save to JSON
        with open(metadata_path, 'w') as f:
            json.dump(metadata_dict, f, indent=2)
        
        logger.info(f"Saved metadata for {len(metadata)} features")
    
    def load_metadata(self, dataset_name: str) -> Dict[str, FeatureMetadata]:
        """Load feature metadata.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary of feature metadata
        """
        metadata_path = self._get_metadata_path(dataset_name)
        
        if not metadata_path.exists():
            return {}
        
        try:
            with open(metadata_path) as f:
                metadata_dict = json.load(f)
            
            # Convert back to FeatureMetadata objects
            metadata = {}
            for name, meta_dict in metadata_dict.items():
                # Convert column type string back to enum
                column_type = ColumnType(meta_dict['column_type'])
                
                metadata[name] = FeatureMetadata(
                    name=meta_dict['name'],
                    source_columns=meta_dict['source_columns'],
                    transformer_name=meta_dict['transformer_name'],
                    column_type=column_type,
                    created_at=datetime.fromisoformat(meta_dict['created_at'])
                )
            
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            return {}
    
    def list_feature_tables(self, dataset_name: str) -> List[str]:
        """List all feature tables for a dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            List of feature table names
        """
        features_dir = self.datasets_path / dataset_name / "metadata" / "features"
        
        if not features_dir.exists():
            return []
        
        feature_tables = []
        for path in features_dir.glob("*.json"):
            if path.name != "feature_metadata.json":
                # Extract table name from filename
                table_name = path.stem
                feature_tables.append(table_name)
        
        return feature_tables
    
    def delete_features(self, dataset_name: str, table_name: str) -> bool:
        """Delete a feature table.
        
        Note: This only deletes metadata. Actual table deletion
        should be handled by the storage backend.
        
        Args:
            dataset_name: Name of the dataset
            table_name: Name of the feature table
            
        Returns:
            True if deleted, False if not found
        """
        feature_info_path = self._get_feature_info_path(dataset_name, table_name)
        
        if feature_info_path.exists():
            feature_info_path.unlink()
            logger.info(f"Deleted feature metadata for {table_name}")
            return True
        
        return False
    
    def _get_metadata_path(self, dataset_name: str) -> Path:
        """Get path to feature metadata file.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Path to metadata file
        """
        return self.datasets_path / dataset_name / "metadata" / "features" / "feature_metadata.json"
    
    def _get_feature_info_path(self, dataset_name: str, table_name: str) -> Path:
        """Get path to feature table info file.
        
        Args:
            dataset_name: Name of the dataset
            table_name: Name of the feature table
            
        Returns:
            Path to feature info file
        """
        return self.datasets_path / dataset_name / "metadata" / "features" / f"{table_name}.json"