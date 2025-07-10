"""Configuration comparison testing framework.

This module provides tools for comparing old and new configuration
systems to ensure functional equivalence during migration.
"""
from typing import Dict, Any, List, Callable, Optional
from pathlib import Path
import tempfile
import logging
from datetime import datetime

from mdm.config.config import ConfigManager, get_config as get_legacy_config
from mdm.core.config_new import NewMDMConfig, get_new_config
from mdm.adapters.config_adapters import (
    LegacyConfigAdapter, 
    NewConfigAdapter,
    get_config_manager
)
from mdm.interfaces.config import IConfiguration
from mdm.testing import ComparisonTester
from mdm.core import metrics_collector, feature_flags

logger = logging.getLogger(__name__)


class ConfigComparisonTester:
    """Test framework for comparing configuration implementations."""
    
    def __init__(self):
        """Initialize tester."""
        self.comparison_tester = ComparisonTester()
        self.test_results: List[Dict[str, Any]] = []
        
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all configuration comparison tests.
        
        Returns:
            Test results summary
        """
        logger.info("Running configuration comparison tests")
        
        tests = [
            self.test_basic_loading,
            self.test_path_resolution,
            self.test_environment_override,
            self.test_yaml_loading,
            self.test_backend_settings,
            self.test_performance_settings,
            self.test_feature_engineering,
            self.test_validation_settings,
        ]
        
        for test in tests:
            try:
                result = test()
                self.test_results.append(result)
            except Exception as e:
                logger.error(f"Test {test.__name__} failed: {e}")
                self.test_results.append({
                    "test_name": test.__name__,
                    "passed": False,
                    "error": str(e),
                })
        
        # Generate summary
        passed = sum(1 for r in self.test_results if r.get("passed", False))
        total = len(self.test_results)
        
        return {
            "total_tests": total,
            "passed": passed,
            "failed": total - passed,
            "success_rate": (passed / total * 100) if total > 0 else 0,
            "results": self.test_results,
        }
    
    def test_basic_loading(self) -> Dict[str, Any]:
        """Test basic configuration loading."""
        logger.info("Testing basic configuration loading")
        
        # Reset configs
        from mdm.config.config import reset_config_manager
        from mdm.core.config import reset_config
        from mdm.core.config_new import reset_new_config
        
        reset_config_manager()
        reset_config()
        reset_new_config()
        
        # Test with legacy
        feature_flags.set("use_new_config", False)
        legacy_config = get_config_manager().load()
        
        # Test with new
        feature_flags.set("use_new_config", True)
        new_config = get_config_manager().load()
        
        # Compare key properties
        passed = all([
            legacy_config.home_dir.name == new_config.home_dir.name,
            legacy_config.default_backend == new_config.default_backend,
            legacy_config.batch_size == new_config.batch_size,
        ])
        
        return {
            "test_name": "basic_loading",
            "passed": passed,
            "legacy_home": str(legacy_config.home_dir),
            "new_home": str(new_config.home_dir),
        }
    
    def test_path_resolution(self) -> Dict[str, Any]:
        """Test path resolution compatibility."""
        logger.info("Testing path resolution")
        
        feature_flags.set("use_new_config", False)
        legacy_config = get_config_manager().load()
        
        feature_flags.set("use_new_config", True)
        new_config = get_config_manager().load()
        
        path_types = ["datasets_path", "config_path", "logs_path"]
        differences = []
        
        for path_type in path_types:
            try:
                legacy_path = legacy_config.get_full_path(path_type)
                new_path = new_config.get_full_path(path_type)
                
                if legacy_path.name != new_path.name:
                    differences.append({
                        "path_type": path_type,
                        "legacy": str(legacy_path),
                        "new": str(new_path),
                    })
            except Exception as e:
                differences.append({
                    "path_type": path_type,
                    "error": str(e),
                })
        
        return {
            "test_name": "path_resolution",
            "passed": len(differences) == 0,
            "differences": differences,
        }
    
    def test_environment_override(self) -> Dict[str, Any]:
        """Test environment variable override."""
        logger.info("Testing environment variable override")
        
        import os
        
        # Set test environment variables
        test_vars = {
            "MDM_DATABASE_DEFAULT_BACKEND": "postgresql",
            "MDM_PERFORMANCE_BATCH_SIZE": "50000",
            "MDM_LOGGING_LEVEL": "DEBUG",
        }
        
        for key, value in test_vars.items():
            os.environ[key] = value
            
        try:
            # Reset and reload
            from mdm.config.config import reset_config_manager
            from mdm.core.config_new import reset_new_config
            
            reset_config_manager()
            reset_new_config()
            
            # Test both systems
            feature_flags.set("use_new_config", False)
            legacy_mgr = get_config_manager()
            legacy_config = legacy_mgr.load()
            
            feature_flags.set("use_new_config", True) 
            new_mgr = get_config_manager()
            new_config = new_mgr.load()
            
            # Check overrides
            checks = [
                ("database.default_backend", "postgresql"),
                ("performance.batch_size", 50000),
                ("logging.level", "DEBUG"),
            ]
            
            legacy_results = []
            new_results = []
            
            for key, expected in checks:
                legacy_val = legacy_mgr.get(key)
                new_val = new_mgr.get(key)
                
                legacy_results.append(legacy_val == expected)
                new_results.append(new_val == expected)
            
            passed = all(legacy_results) and all(new_results)
            
            return {
                "test_name": "environment_override",
                "passed": passed,
                "legacy_results": legacy_results,
                "new_results": new_results,
            }
            
        finally:
            # Clean up env vars
            for key in test_vars:
                os.environ.pop(key, None)
    
    def test_yaml_loading(self) -> Dict[str, Any]:
        """Test YAML file loading."""
        logger.info("Testing YAML file loading")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "test_config.yaml"
            
            # Create test YAML
            yaml_content = """
database:
  default_backend: duckdb
  duckdb:
    memory_limit: 16GB
    threads: 8

performance:
  batch_size: 20000
  
logging:
  level: WARNING
  file: test.log
"""
            yaml_path.write_text(yaml_content)
            
            # Test legacy loading
            from mdm.config.config import ConfigManager
            legacy_manager = ConfigManager(yaml_path)
            legacy_config = legacy_manager.load()
            
            # Test new loading
            new_config = NewMDMConfig.from_yaml(yaml_path)
            
            # Compare values
            checks = [
                (legacy_config.database.default_backend, 
                 new_config.database.default_backend, "duckdb"),
                (legacy_config.database.duckdb.memory_limit,
                 new_config.database.duckdb_memory_limit, "16GB"),
                (legacy_config.performance.batch_size,
                 new_config.performance.batch_size, 20000),
                (legacy_config.logging.level,
                 new_config.logging.level, "WARNING"),
            ]
            
            passed = all(legacy == new == expected 
                        for legacy, new, expected in checks)
            
            return {
                "test_name": "yaml_loading",
                "passed": passed,
                "checks": len(checks),
            }
    
    def test_backend_settings(self) -> Dict[str, Any]:
        """Test database backend settings."""
        logger.info("Testing backend settings")
        
        feature_flags.set("use_new_config", False)
        legacy_config = get_config_manager().load()
        
        feature_flags.set("use_new_config", True)
        new_config = get_config_manager().load()
        
        # Get backend configs
        backends = ["sqlite", "duckdb", "postgresql"]
        differences = []
        
        for backend in backends:
            if isinstance(new_config, NewConfigAdapter):
                new_backend_config = new_config._config.get_backend_config(backend)
            else:
                new_backend_config = {}
                
            # Compare some key settings
            if backend == "sqlite":
                legacy_val = legacy_config.to_dict()["database"]["sqlite"]["journal_mode"]
                new_val = new_backend_config.get("journal_mode")
                if legacy_val != new_val:
                    differences.append(f"{backend}.journal_mode: {legacy_val} != {new_val}")
                    
        return {
            "test_name": "backend_settings",
            "passed": len(differences) == 0,
            "differences": differences,
        }
    
    def test_performance_settings(self) -> Dict[str, Any]:
        """Test performance settings compatibility."""
        result = self.comparison_tester.compare(
            test_name="performance_settings",
            old_impl=lambda: get_legacy_config().performance.batch_size,
            new_impl=lambda: get_new_config().performance.batch_size,
            compare_func=lambda old, new: old == new
        )
        
        return {
            "test_name": "performance_settings", 
            "passed": result.passed,
            "old_value": result.old_result,
            "new_value": result.new_result,
        }
    
    def test_feature_engineering(self) -> Dict[str, Any]:
        """Test feature engineering settings."""
        logger.info("Testing feature engineering settings")
        
        feature_flags.set("use_new_config", False)
        legacy_config = get_config_manager().load()
        
        feature_flags.set("use_new_config", True)
        new_config = get_config_manager().load()
        
        # Extract settings
        legacy_dict = legacy_config.to_dict()
        new_dict = new_config.to_dict()
        
        # Check feature engineering enabled
        legacy_enabled = legacy_dict.get("feature_engineering", {}).get("enabled", True)
        new_enabled = new_dict.get("feature_engineering", {}).get("enabled", True)
        
        passed = legacy_enabled == new_enabled
        
        return {
            "test_name": "feature_engineering",
            "passed": passed,
            "legacy_enabled": legacy_enabled,
            "new_enabled": new_enabled,
        }
    
    def test_validation_settings(self) -> Dict[str, Any]:
        """Test validation settings."""
        logger.info("Testing validation settings")
        
        feature_flags.set("use_new_config", False)
        legacy_config = get_config_manager().load()
        
        feature_flags.set("use_new_config", True)
        new_config = get_config_manager().load()
        
        # Check enable_validation property
        passed = legacy_config.enable_validation == new_config.enable_validation
        
        return {
            "test_name": "validation_settings",
            "passed": passed,
            "legacy_validation": legacy_config.enable_validation,
            "new_validation": new_config.enable_validation,
        }
    
    def generate_report(self, output_path: Optional[Path] = None) -> Path:
        """Generate comparison report.
        
        Args:
            output_path: Optional output path
            
        Returns:
            Path to report
        """
        if output_path is None:
            output_path = Path.home() / ".mdm" / "comparison_results" / \
                         f"config_comparison_{datetime.now():%Y%m%d_%H%M%S}.json"
                         
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        summary = self.run_all_tests()
        
        import json
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
            
        logger.info(f"Configuration comparison report saved to: {output_path}")
        return output_path