"""Tests for integration testing framework."""
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile

from mdm.testing import (
    IntegrationTestFramework,
    IntegrationTestResult,
    MigrationTestSuite,
    MigrationTestResult,
    PerformanceBenchmark,
    PerformanceMetric,
    PerformanceComparison
)


class TestIntegrationTestResult:
    """Test IntegrationTestResult class."""
    
    def test_result_initialization(self):
        """Test result initialization."""
        result = IntegrationTestResult("test_name", "test_type")
        
        assert result.test_name == "test_name"
        assert result.test_type == "test_type"
        assert result.passed is False
        assert result.duration == 0.0
        assert result.error is None
        assert result.warnings == []
        assert result.metrics == {}
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = IntegrationTestResult("test", "unit")
        result.passed = True
        result.duration = 1.5
        result.metrics = {'key': 'value'}
        
        d = result.to_dict()
        
        assert d['test_name'] == "test"
        assert d['test_type'] == "unit"
        assert d['passed'] is True
        assert d['duration'] == 1.5
        assert d['metrics'] == {'key': 'value'}


class TestMigrationTestResult:
    """Test MigrationTestResult class."""
    
    def test_add_integrity_check(self):
        """Test adding integrity checks."""
        result = MigrationTestResult("migration_test")
        
        result.add_integrity_check("check1", True, "Details 1")
        result.add_integrity_check("check2", False, "Details 2")
        
        assert len(result.data_integrity_checks) == 2
        assert result.data_integrity_checks[0]['name'] == "check1"
        assert result.data_integrity_checks[0]['passed'] is True
        assert result.data_integrity_checks[1]['name'] == "check2"
        assert result.data_integrity_checks[1]['passed'] is False


class TestPerformanceMetric:
    """Test PerformanceMetric class."""
    
    def test_memory_calculation(self):
        """Test memory usage calculation."""
        metric = PerformanceMetric(
            operation="test_op",
            implementation="legacy",
            duration=1.0,
            memory_before=100.0,
            memory_after=150.0,
            cpu_percent=50.0,
            success=True
        )
        
        assert metric.memory_used == 50.0
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        metric = PerformanceMetric(
            operation="test_op",
            implementation="new",
            duration=2.0,
            memory_before=200.0,
            memory_after=250.0,
            cpu_percent=75.0,
            success=True,
            metadata={'test': 'data'}
        )
        
        d = metric.to_dict()
        
        assert d['operation'] == "test_op"
        assert d['implementation'] == "new"
        assert d['duration'] == 2.0
        assert d['memory_used'] == 50.0
        assert d['cpu_percent'] == 75.0
        assert d['success'] is True
        assert d['metadata'] == {'test': 'data'}


class TestPerformanceComparison:
    """Test PerformanceComparison class."""
    
    def test_speedup_calculation(self):
        """Test speedup calculation."""
        legacy_metric = PerformanceMetric(
            operation="test",
            implementation="legacy",
            duration=10.0,
            memory_before=100.0,
            memory_after=150.0,
            cpu_percent=50.0,
            success=True
        )
        
        new_metric = PerformanceMetric(
            operation="test",
            implementation="new",
            duration=5.0,
            memory_before=100.0,
            memory_after=140.0,
            cpu_percent=45.0,
            success=True
        )
        
        comparison = PerformanceComparison("test", legacy_metric, new_metric)
        
        assert comparison.speedup == 2.0  # 10/5
        assert comparison.memory_ratio == 0.8  # 40/50
        assert comparison.is_regression is False
    
    def test_regression_detection(self):
        """Test regression detection."""
        # Case 1: Performance regression
        legacy_metric = PerformanceMetric(
            "test", "legacy", 5.0, 100.0, 150.0, 50.0, True
        )
        new_metric = PerformanceMetric(
            "test", "new", 10.0, 100.0, 150.0, 50.0, True
        )
        
        comparison = PerformanceComparison("test", legacy_metric, new_metric)
        assert comparison.speedup == 0.5  # Slower
        assert comparison.is_regression is True
        
        # Case 2: Memory regression
        legacy_metric2 = PerformanceMetric(
            "test2", "legacy", 5.0, 100.0, 150.0, 50.0, True
        )
        new_metric2 = PerformanceMetric(
            "test2", "new", 5.0, 100.0, 200.0, 50.0, True
        )
        
        comparison2 = PerformanceComparison("test2", legacy_metric2, new_metric2)
        assert comparison2.memory_ratio == 2.0  # 100/50
        assert comparison2.is_regression is True


class TestIntegrationTestFramework:
    """Test IntegrationTestFramework class."""
    
    def test_initialization(self):
        """Test framework initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            framework = IntegrationTestFramework(Path(temp_dir))
            
            assert framework.test_dir == Path(temp_dir)
            assert framework._test_datasets == []
            assert framework._test_results == []
    
    @patch('mdm.testing.integration_framework.get_config_manager')
    @patch('mdm.testing.integration_framework.get_storage_backend')
    def test_config_storage_integration(self, mock_storage, mock_config):
        """Test config-storage integration test."""
        # Setup mocks
        mock_config_inst = Mock()
        mock_config_inst.get_storage_config.return_value = {'key': 'value'}
        mock_config.return_value = mock_config_inst
        
        mock_storage_inst = Mock()
        mock_storage_inst.get_connection.return_value = Mock()
        mock_storage.return_value = mock_storage_inst
        
        with tempfile.TemporaryDirectory() as temp_dir:
            framework = IntegrationTestFramework(Path(temp_dir))
            result = framework._test_config_storage_integration()
            
            assert result.passed is True
            assert result.test_name == "Config-Storage Integration"
            assert result.test_type == "component"
            
            # Verify mocks were called
            mock_config.assert_called()
            mock_storage.assert_called_with("sqlite")
            mock_storage_inst.close.assert_called()
    
    @patch('mdm.testing.integration_framework.pd.DataFrame.to_csv')
    def test_create_test_data(self, mock_to_csv):
        """Test test data creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            framework = IntegrationTestFramework(Path(temp_dir))
            
            # Should not fail even if mocked
            framework._create_benchmark_datasets()
            
            # Verify CSV was attempted to be saved
            assert mock_to_csv.called


class TestMigrationTestSuite:
    """Test MigrationTestSuite class."""
    
    def test_initialization(self):
        """Test suite initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            suite = MigrationTestSuite(Path(temp_dir))
            
            assert suite.test_dir == Path(temp_dir)
            assert suite._test_datasets == []
            assert suite._original_flags == {}
    
    def test_save_and_restore_flags(self):
        """Test feature flag save/restore."""
        from mdm.core import feature_flags
        
        with tempfile.TemporaryDirectory() as temp_dir:
            suite = MigrationTestSuite(Path(temp_dir))
            
            # Set some flags
            feature_flags.set("use_new_config", True)
            feature_flags.set("use_new_storage", False)
            
            # Save current state
            suite._save_current_flags()
            
            # Change flags
            feature_flags.set("use_new_config", False)
            feature_flags.set("use_new_storage", True)
            
            # Restore
            suite._restore_original_flags()
            
            # Verify restored
            assert feature_flags.get("use_new_config") is True
            assert feature_flags.get("use_new_storage") is False
    
    def test_migration_readiness_calculation(self):
        """Test migration readiness calculation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            suite = MigrationTestSuite(Path(temp_dir))
            
            # Mock results
            results = {
                'total': 100,
                'passed': 95,
                'failed': 5,
                'groups': {
                    'Configuration Migration': {'total': 20, 'passed': 20},
                    'Storage Backend Migration': {'total': 20, 'passed': 18},
                    'Data Integrity': {'total': 20, 'passed': 20}
                }
            }
            
            readiness = suite._calculate_migration_readiness(results)
            
            assert readiness['overall_score'] == 95.0
            assert readiness['status'] == 'ready'
            assert 'critical_components' in readiness
            assert 'recommendation' in readiness


class TestPerformanceBenchmark:
    """Test PerformanceBenchmark class."""
    
    def test_initialization(self):
        """Test benchmark initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            benchmark = PerformanceBenchmark(Path(temp_dir))
            
            assert benchmark.test_dir == Path(temp_dir)
            assert benchmark._test_datasets == []
    
    @patch('mdm.testing.performance_tests.psutil.Process')
    def test_system_info(self, mock_process_class):
        """Test system info gathering."""
        mock_process = Mock()
        mock_process.name.return_value = "python"
        mock_process.exe.return_value = "/usr/bin/python3"
        mock_process_class.return_value = mock_process
        
        with tempfile.TemporaryDirectory() as temp_dir:
            benchmark = PerformanceBenchmark(Path(temp_dir))
            info = benchmark._get_system_info()
            
            assert 'cpu_count' in info
            assert 'memory_total_gb' in info
            assert 'platform' in info
    
    def test_generate_summary(self):
        """Test performance summary generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            benchmark = PerformanceBenchmark(Path(temp_dir))
            
            # Create mock comparisons
            comparisons = []
            for i in range(5):
                legacy = PerformanceMetric(
                    f"op{i}", "legacy", 10.0, 100.0, 150.0, 50.0, True
                )
                new = PerformanceMetric(
                    f"op{i}", "new", 5.0 if i < 3 else 15.0, 
                    100.0, 140.0, 45.0, True
                )
                comparisons.append(PerformanceComparison(f"op{i}", legacy, new))
            
            results = {'comparisons': comparisons, 'regressions': []}
            summary = benchmark._generate_summary(results)
            
            assert summary['total_operations'] == 5
            assert 'average_speedup' in summary
            assert 'regression_count' in summary
            assert 'improvement_count' in summary
    
    @patch('mdm.testing.performance_tests.get_dataset_registrar')
    def test_register_dataset(self, mock_registrar):
        """Test dataset registration in benchmark."""
        mock_reg_inst = Mock()
        mock_registrar.return_value = mock_reg_inst
        
        with tempfile.TemporaryDirectory() as temp_dir:
            benchmark = PerformanceBenchmark(Path(temp_dir))
            
            benchmark._register_dataset(
                "test_dataset",
                "/path/to/data.csv",
                use_new=True,
                generate_features=False
            )
            
            # Verify registration was called
            mock_reg_inst.register.assert_called_once_with(
                name="test_dataset",
                path="/path/to/data.csv",
                force=True,
                generate_features=False
            )
            
            # Verify dataset was tracked
            assert "test_dataset" in benchmark._test_datasets


@pytest.fixture
def mock_feature_flags():
    """Mock feature flags for tests."""
    with patch('mdm.core.feature_flags') as mock:
        mock.get.return_value = False
        mock.set.return_value = None
        yield mock


def test_framework_cleanup():
    """Test that frameworks clean up properly."""
    # Test IntegrationTestFramework cleanup
    with tempfile.TemporaryDirectory() as temp_dir:
        framework = IntegrationTestFramework(Path(temp_dir))
        test_file = framework.test_dir / "test.txt"
        test_file.write_text("test")
        
        framework._cleanup_test_data()
        
        # Directory should be removed
        assert not framework.test_dir.exists()
    
    # Test MigrationTestSuite cleanup
    with tempfile.TemporaryDirectory() as temp_dir:
        suite = MigrationTestSuite(Path(temp_dir))
        test_file = suite.test_dir / "test.txt"
        test_file.write_text("test")
        
        suite._cleanup_test_data()
        
        # Directory should be removed
        assert not suite.test_dir.exists()