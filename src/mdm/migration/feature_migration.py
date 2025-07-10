"""
Feature engineering migration utilities.

This module provides tools for migrating from the legacy feature engineering
system to the new plugin-based architecture.
"""
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import logging

from mdm.core import feature_flags
from mdm.interfaces.features import IFeatureGenerator
from mdm.adapters.feature_manager import get_feature_generator
from mdm.features.registry import feature_registry as legacy_registry
from mdm.core.features.base import transformer_registry as new_registry

logger = logging.getLogger(__name__)


class FeatureMigrator:
    """Migrates feature engineering between legacy and new systems."""
    
    def __init__(self):
        """Initialize feature migrator."""
        self.legacy_generator = get_feature_generator(force_new=False)
        self.new_generator = get_feature_generator(force_new=True)
        self.comparison_results = []
        logger.info("Initialized FeatureMigrator")
    
    def migrate_transformers(self, output_dir: Path) -> Dict[str, Any]:
        """Migrate legacy transformers to new plugin format.
        
        Args:
            output_dir: Directory to write migrated transformers
            
        Returns:
            Migration summary
        """
        logger.info(f"Migrating transformers to {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        migrated = []
        failed = []
        
        # Get all legacy transformers
        for column_type in ["numeric", "categorical", "datetime", "text"]:
            try:
                # Get transformers from legacy registry
                legacy_transformers = legacy_registry.get_transformers(column_type)
                
                for transformer in legacy_transformers:
                    transformer_name = transformer.__class__.__name__
                    
                    # Generate plugin code
                    plugin_code = self._generate_plugin_code(transformer, column_type)
                    
                    # Write to file
                    plugin_file = output_dir / f"{transformer_name.lower()}_plugin.py"
                    plugin_file.write_text(plugin_code)
                    
                    migrated.append({
                        "name": transformer_name,
                        "type": column_type,
                        "file": str(plugin_file)
                    })
                    
                    logger.debug(f"Migrated transformer: {transformer_name}")
                    
            except Exception as e:
                failed.append({
                    "type": column_type,
                    "error": str(e)
                })
                logger.error(f"Failed to migrate {column_type} transformers: {e}")
        
        summary = {
            "migrated_count": len(migrated),
            "failed_count": len(failed),
            "migrated": migrated,
            "failed": failed,
            "output_directory": str(output_dir)
        }
        
        # Save summary
        summary_file = output_dir / "migration_summary.json"
        summary_file.write_text(json.dumps(summary, indent=2))
        
        return summary
    
    def compare_outputs(
        self, 
        test_data: pd.DataFrame,
        target_column: Optional[str] = None,
        id_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Compare outputs between legacy and new generators.
        
        Args:
            test_data: Test DataFrame
            target_column: Target column name
            id_columns: ID columns to exclude
            
        Returns:
            Comparison results
        """
        logger.info("Comparing feature generation outputs")
        
        # Generate features with both systems
        start_legacy = datetime.now()
        legacy_features = self.legacy_generator.generate_features(
            data=test_data,
            target_column=target_column,
            id_columns=id_columns
        )
        legacy_time = (datetime.now() - start_legacy).total_seconds()
        
        start_new = datetime.now()
        new_features = self.new_generator.generate_features(
            data=test_data,
            target_column=target_column,
            id_columns=id_columns
        )
        new_time = (datetime.now() - start_new).total_seconds()
        
        # Compare results
        comparison = {
            "timestamp": datetime.now().isoformat(),
            "test_data_shape": test_data.shape,
            "legacy": {
                "features_generated": len(legacy_features.columns) - len(test_data.columns),
                "total_columns": len(legacy_features.columns),
                "processing_time": legacy_time
            },
            "new": {
                "features_generated": len(new_features.columns) - len(test_data.columns),
                "total_columns": len(new_features.columns),
                "processing_time": new_time
            },
            "performance_ratio": new_time / legacy_time if legacy_time > 0 else 1.0
        }
        
        # Find common and unique features
        legacy_cols = set(legacy_features.columns) - set(test_data.columns)
        new_cols = set(new_features.columns) - set(test_data.columns)
        
        comparison["features"] = {
            "common": sorted(list(legacy_cols.intersection(new_cols))),
            "legacy_only": sorted(list(legacy_cols - new_cols)),
            "new_only": sorted(list(new_cols - legacy_cols))
        }
        
        # Compare values for common features
        value_differences = []
        for col in comparison["features"]["common"]:
            if col in legacy_features.columns and col in new_features.columns:
                # Check if values are close
                if pd.api.types.is_numeric_dtype(legacy_features[col]):
                    # Numeric comparison
                    if not np.allclose(
                        legacy_features[col].fillna(0), 
                        new_features[col].fillna(0),
                        rtol=1e-5,
                        equal_nan=True
                    ):
                        diff = np.abs(legacy_features[col] - new_features[col]).mean()
                        value_differences.append({
                            "feature": col,
                            "type": "numeric",
                            "mean_difference": float(diff)
                        })
                else:
                    # Categorical comparison
                    if not legacy_features[col].equals(new_features[col]):
                        mismatch_pct = (legacy_features[col] != new_features[col]).sum() / len(legacy_features)
                        value_differences.append({
                            "feature": col,
                            "type": "categorical",
                            "mismatch_percentage": float(mismatch_pct)
                        })
        
        comparison["value_differences"] = value_differences
        comparison["values_match"] = len(value_differences) == 0
        
        self.comparison_results.append(comparison)
        
        return comparison
    
    def validate_migration(
        self, 
        test_datasets: List[pd.DataFrame],
        tolerance: float = 0.95
    ) -> Dict[str, Any]:
        """Validate migration across multiple test datasets.
        
        Args:
            test_datasets: List of test DataFrames
            tolerance: Minimum feature overlap ratio required
            
        Returns:
            Validation results
        """
        logger.info(f"Validating migration with {len(test_datasets)} test datasets")
        
        all_comparisons = []
        
        for i, dataset in enumerate(test_datasets):
            logger.debug(f"Testing dataset {i+1}/{len(test_datasets)}")
            comparison = self.compare_outputs(dataset)
            all_comparisons.append(comparison)
        
        # Aggregate results
        total_tests = len(all_comparisons)
        values_match_count = sum(1 for c in all_comparisons if c["values_match"])
        
        feature_overlaps = []
        performance_ratios = []
        
        for comp in all_comparisons:
            common = len(comp["features"]["common"])
            legacy_total = comp["legacy"]["features_generated"]
            if legacy_total > 0:
                overlap_ratio = common / legacy_total
                feature_overlaps.append(overlap_ratio)
            
            performance_ratios.append(comp["performance_ratio"])
        
        validation_result = {
            "total_tests": total_tests,
            "values_match_count": values_match_count,
            "values_match_rate": values_match_count / total_tests if total_tests > 0 else 0,
            "avg_feature_overlap": np.mean(feature_overlaps) if feature_overlaps else 0,
            "min_feature_overlap": np.min(feature_overlaps) if feature_overlaps else 0,
            "avg_performance_ratio": np.mean(performance_ratios),
            "passed": (
                np.mean(feature_overlaps) >= tolerance if feature_overlaps else False
            ),
            "tolerance": tolerance,
            "timestamp": datetime.now().isoformat()
        }
        
        return validation_result
    
    def _generate_plugin_code(self, transformer: Any, column_type: str) -> str:
        """Generate plugin code for a legacy transformer.
        
        Args:
            transformer: Legacy transformer instance
            column_type: Type of columns it processes
            
        Returns:
            Python code for plugin
        """
        transformer_name = transformer.__class__.__name__
        
        template = f'''"""
Auto-generated plugin for {transformer_name}.

This plugin was migrated from the legacy feature engineering system.
Generated on: {datetime.now().isoformat()}
"""
from typing import Dict, List, Any
import pandas as pd
import numpy as np

from mdm.core.features.base import FeatureTransformer


class {transformer_name}Plugin(FeatureTransformer):
    """Migrated {transformer_name} transformer."""
    
    @property
    def name(self) -> str:
        return "{transformer_name.lower()}"
    
    @property
    def supported_types(self) -> List[str]:
        return ["{column_type}"]
    
    def _fit_impl(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Fit the transformer."""
        # Legacy transformers are typically stateless
        return {{}}
    
    def _transform_impl(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform the data."""
        result = data.copy()
        
        # TODO: Implement transformation logic from legacy transformer
        # This is a placeholder - actual logic needs to be migrated
        
        return result
'''
        
        return template


class FeatureValidator:
    """Validates feature engineering implementations."""
    
    def __init__(self):
        """Initialize validator."""
        self.legacy_generator = get_feature_generator(force_new=False)
        self.new_generator = get_feature_generator(force_new=True)
    
    def validate_consistency(
        self, 
        test_data: pd.DataFrame,
        feature_types: List[str] = None
    ) -> Dict[str, Any]:
        """Validate consistency between implementations.
        
        Args:
            test_data: Test DataFrame
            feature_types: Types to test (default: all)
            
        Returns:
            Validation results
        """
        if feature_types is None:
            feature_types = ["numeric", "categorical", "datetime", "text", "interaction"]
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "test_data_shape": test_data.shape,
            "feature_types": {}
        }
        
        for feature_type in feature_types:
            logger.debug(f"Validating {feature_type} features")
            
            try:
                # Get method name
                method_name = f"generate_{feature_type}_features"
                
                # Check if both generators have the method
                if not hasattr(self.legacy_generator, method_name):
                    results["feature_types"][feature_type] = {
                        "status": "skipped",
                        "reason": "Legacy generator missing method"
                    }
                    continue
                
                if not hasattr(self.new_generator, method_name):
                    results["feature_types"][feature_type] = {
                        "status": "skipped", 
                        "reason": "New generator missing method"
                    }
                    continue
                
                # Generate features
                legacy_method = getattr(self.legacy_generator, method_name)
                new_method = getattr(self.new_generator, method_name)
                
                legacy_result = legacy_method(test_data)
                new_result = new_method(test_data)
                
                # Compare
                legacy_new_cols = set(legacy_result.columns) - set(test_data.columns)
                new_new_cols = set(new_result.columns) - set(test_data.columns)
                
                results["feature_types"][feature_type] = {
                    "status": "tested",
                    "legacy_features": len(legacy_new_cols),
                    "new_features": len(new_new_cols),
                    "common_features": len(legacy_new_cols.intersection(new_new_cols)),
                    "match_rate": (
                        len(legacy_new_cols.intersection(new_new_cols)) / len(legacy_new_cols)
                        if legacy_new_cols else 1.0
                    )
                }
                
            except Exception as e:
                results["feature_types"][feature_type] = {
                    "status": "error",
                    "error": str(e)
                }
        
        # Overall assessment
        tested_types = [
            ft for ft, res in results["feature_types"].items() 
            if res["status"] == "tested"
        ]
        
        if tested_types:
            avg_match_rate = np.mean([
                results["feature_types"][ft]["match_rate"] 
                for ft in tested_types
            ])
            results["overall_match_rate"] = float(avg_match_rate)
            results["passed"] = avg_match_rate >= 0.9
        else:
            results["overall_match_rate"] = 0.0
            results["passed"] = False
        
        return results