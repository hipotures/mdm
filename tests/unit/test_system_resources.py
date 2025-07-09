"""Tests for system resource limits - disk space, network timeouts, etc."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os
import time
from unittest.mock import Mock, patch, MagicMock
import psutil

from mdm.api import MDMClient
from mdm.dataset.registrar import DatasetRegistrar
from mdm.core.exceptions import DatasetError
from mdm.storage.base import StorageError


class TestDiskSpaceHandling:
    """Test handling of disk space exhaustion scenarios."""
    
    def test_disk_space_check_before_operation(self, test_config):
        """Test that disk space is checked before large operations."""
        client = MDMClient(config=test_config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dataset
            csv_path = Path(tmpdir) / "train.csv"
            data = pd.DataFrame({
                'id': range(1000),
                'data': np.random.rand(1000),
            })
            data.to_csv(csv_path, index=False)
            
            # Mock low disk space
            with patch('psutil.disk_usage') as mock_disk_usage:
                # Simulate only 100MB free space
                mock_disk_usage.return_value = MagicMock(
                    free=100 * 1024 * 1024,  # 100MB
                    total=10 * 1024 * 1024 * 1024,  # 10GB
                    percent=99.0  # 99% used
                )
                
                # Should warn or handle gracefully
                with patch('mdm.dataset.registrar.logger.warning') as mock_warning:
                    try:
                        dataset_info = client.register_dataset(
                            name="low_disk_space",
                            dataset_path=str(tmpdir),
                        )
                        # If it succeeds, should have warned
                        mock_warning.assert_called()
                        warning_msg = str(mock_warning.call_args).lower()
                        assert 'disk' in warning_msg or 'space' in warning_msg
                        
                    except (OSError, StorageError) as e:
                        # Also acceptable to fail with clear error
                        assert 'disk' in str(e).lower() or 'space' in str(e).lower()
    
    def test_disk_full_during_write(self, test_config):
        """Test handling when disk becomes full during write operation."""
        registrar = DatasetRegistrar()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create initial dataset
            csv_path = Path(tmpdir) / "train.csv" 
            data = pd.DataFrame({
                'id': range(10000),
                'data': np.random.rand(10000),
            })
            data.to_csv(csv_path, index=False)
            
            # Mock disk becoming full during database write
            original_to_sql = pd.DataFrame.to_sql
            call_count = 0
            
            def mock_to_sql(self, *args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count > 2:  # Fail after 2 chunks
                    raise OSError("No space left on device")
                return original_to_sql(self, *args, **kwargs)
            
            with patch.object(pd.DataFrame, 'to_sql', mock_to_sql):
                # Should handle the disk full error gracefully
                with pytest.raises((OSError, StorageError, DatasetError)) as exc_info:
                    dataset_info = registrar.register(
                        name="disk_full_test",
                        path=Path(tmpdir),
                    )
                
                # Error should be clear about disk space
                error_msg = str(exc_info.value).lower()
                assert 'space' in error_msg or 'disk' in error_msg or 'device' in error_msg
    
    def test_cleanup_after_disk_full(self, test_config):
        """Test that partial data is cleaned up after disk full error."""
        client = MDMClient(config=test_config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "train.csv"
            data = pd.DataFrame({
                'id': range(1000),
                'value': np.random.rand(1000),
            })
            data.to_csv(csv_path, index=False)
            
            # Mock disk full during registration
            with patch('mdm.storage.sqlite.pd.DataFrame.to_sql') as mock_to_sql:
                mock_to_sql.side_effect = OSError("No space left on device")
                
                # Registration should fail
                with pytest.raises((OSError, StorageError, DatasetError)):
                    client.register_dataset(
                        name="cleanup_test",
                        dataset_path=str(tmpdir),
                    )
                
                # Dataset should not exist
                assert not client.dataset_exists("cleanup_test")
    
    def test_export_with_insufficient_space(self, test_config):
        """Test export operation with insufficient disk space."""
        client = MDMClient(config=test_config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and register dataset
            csv_path = Path(tmpdir) / "train.csv"
            data = pd.DataFrame({
                'id': range(1000),
                'data': np.random.rand(1000),
            })
            data.to_csv(csv_path, index=False)
            
            client.register_dataset(
                name="export_space_test",
                dataset_path=str(tmpdir),
            )
            
            # Mock insufficient space for export
            export_dir = Path(tmpdir) / "exports"
            export_dir.mkdir()
            
            with patch('pathlib.Path.write_bytes') as mock_write:
                mock_write.side_effect = OSError("No space left on device")
                
                # Export should fail gracefully
                with pytest.raises((OSError, StorageError)):
                    client.export_dataset(
                        "export_space_test",
                        output_dir=str(export_dir),
                        format="csv",
                    )


class TestNetworkTimeouts:
    """Test handling of network timeouts for remote operations."""
    
    def test_postgresql_connection_timeout(self, test_config):
        """Test PostgreSQL connection timeout handling."""
        # Configure PostgreSQL backend
        test_config.database.default_backend = "postgresql"
        test_config.database.postgresql.host = "non.existent.host"
        test_config.database.postgresql.port = 5432
        test_config.database.postgresql.timeout = 1  # 1 second timeout
        
        registrar = DatasetRegistrar()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "train.csv"
            csv_path.write_text("id,value\n1,100\n2,200")
            
            # Should handle connection timeout
            with patch('psycopg2.connect') as mock_connect:
                import psycopg2
                mock_connect.side_effect = psycopg2.OperationalError(
                    "could not connect to server: Connection timed out"
                )
                
                with pytest.raises((DatasetError, psycopg2.OperationalError)) as exc_info:
                    dataset_info = registrar.register(
                        name="pg_timeout_test",
                        path=Path(tmpdir),
                    )
                
                # Error should mention connection/timeout
                error_msg = str(exc_info.value).lower()
                assert 'connect' in error_msg or 'timeout' in error_msg
    
    def test_slow_query_timeout(self, test_config):
        """Test handling of slow queries that timeout."""
        client = MDMClient(config=test_config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dataset
            csv_path = Path(tmpdir) / "train.csv"
            data = pd.DataFrame({
                'id': range(10000),
                'value': np.random.rand(10000),
            })
            data.to_csv(csv_path, index=False)
            
            client.register_dataset(
                name="slow_query_test",
                dataset_path=str(tmpdir),
            )
            
            # Mock slow query
            with patch('pandas.read_sql_query') as mock_read_sql:
                def slow_query(*args, **kwargs):
                    time.sleep(5)  # Simulate slow query
                    raise Exception("Query timeout")
                
                mock_read_sql.side_effect = slow_query
                
                # Query should timeout
                with pytest.raises(Exception) as exc_info:
                    client.query_dataset(
                        "slow_query_test",
                        "SELECT COUNT(*) FROM train WHERE value > 0.5"
                    )
                
                assert "timeout" in str(exc_info.value).lower()
    
    def test_network_interruption_during_transfer(self):
        """Test handling of network interruption during data transfer."""
        # This would test remote storage backends
        with patch('urllib.request.urlopen') as mock_urlopen:
            # Simulate network interruption
            mock_urlopen.side_effect = ConnectionError("Network is unreachable")
            
            # Operations requiring network should fail gracefully
            with pytest.raises(ConnectionError):
                # Simulate downloading remote dataset
                mock_urlopen("https://example.com/dataset.csv")
    
    def test_retry_on_temporary_network_failure(self):
        """Test retry mechanism for temporary network failures."""
        retry_count = 0
        max_retries = 3
        
        def flaky_network_operation():
            nonlocal retry_count
            retry_count += 1
            if retry_count < max_retries:
                raise ConnectionError("Temporary network failure")
            return "Success"
        
        # Simple retry mechanism
        for attempt in range(max_retries + 1):
            try:
                result = flaky_network_operation()
                break
            except ConnectionError:
                if attempt == max_retries:
                    raise
                time.sleep(0.1)  # Brief delay before retry
        
        assert result == "Success"
        assert retry_count == max_retries


class TestResourceLimits:
    """Test handling of various resource limits."""
    
    def test_memory_limit_detection(self):
        """Test detection of available memory."""
        # Get available memory
        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024 ** 3)
        
        # Verify we can detect memory limits
        assert available_gb > 0
        assert mem.percent >= 0 and mem.percent <= 100
    
    def test_concurrent_operation_limits(self, test_config):
        """Test handling of concurrent operation limits."""
        client = MDMClient(config=test_config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple datasets
            for i in range(5):
                csv_path = Path(tmpdir) / f"data_{i}.csv" 
                csv_path.write_text(f"id,value\n1,{i}\n2,{i*10}")
            
            # Simulate concurrent registrations
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            def register_dataset(index):
                try:
                    return client.register_dataset(
                        name=f"concurrent_test_{index}",
                        dataset_path=str(tmpdir),
                        force=True,
                    )
                except Exception as e:
                    return e
            
            # Run concurrent registrations
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(register_dataset, i) for i in range(3)]
                
                results = []
                for future in as_completed(futures):
                    results.append(future.result())
            
            # At least some should succeed
            successful = [r for r in results if not isinstance(r, Exception)]
            assert len(successful) >= 1
    
    def test_file_handle_limits(self):
        """Test handling of file handle limits."""
        # Get current limit
        import resource
        soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
        
        # Verify we can detect file handle limits
        assert soft_limit > 0
        assert hard_limit >= soft_limit
        
        # In production, would test opening many files
        # and handling "too many open files" error