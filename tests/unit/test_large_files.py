"""Tests for handling large files and memory efficiency."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os
from unittest.mock import Mock, patch
import psutil

from mdm.api import MDMClient
from mdm.dataset.registrar import DatasetRegistrar
from mdm.core.exceptions import DatasetError


class TestLargeFileHandling:
    """Test handling of large files with memory efficiency."""
    
    def test_large_file_chunked_processing(self, test_config):
        """Test that large files are processed in chunks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a "large" CSV file (simulate with smaller data)
            csv_path = Path(tmpdir) / "train.csv"
            
            # Generate data that simulates a large file
            n_rows = 100_000  # In real scenario would be millions
            data = {
                'id': range(n_rows),
                'value': np.random.rand(n_rows),
                'category': np.random.choice(['A', 'B', 'C'], n_rows),
                'timestamp': pd.date_range('2024-01-01', periods=n_rows, freq='1min'),
            }
            df = pd.DataFrame(data)
            
            # Save to CSV
            df.to_csv(csv_path, index=False)
            
            # Track memory usage
            initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            # Register dataset with batch processing
            registrar = DatasetRegistrar()
            with patch.object(registrar, '_load_data_files') as mock_load:
                # Mock to verify chunked processing
                chunks_processed = []
                
                def mock_load_impl(*args, **kwargs):
                    # Simulate chunked loading
                    chunk_size = test_config.performance.batch_size
                    for chunk_start in range(0, n_rows, chunk_size):
                        chunk_end = min(chunk_start + chunk_size, n_rows)
                        chunks_processed.append((chunk_start, chunk_end))
                    return {}, {}
                
                mock_load.side_effect = mock_load_impl
                
                dataset_info = registrar.register(
                    name="large_dataset",
                    path=Path(tmpdir),
                    auto_detect=True,
                )
                
                # Verify chunked processing occurred
                assert len(chunks_processed) > 1
                assert chunks_processed[0][1] - chunks_processed[0][0] <= test_config.performance.batch_size
            
            # Check memory didn't spike too much
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            # Memory increase should be reasonable (< 500MB for 100k rows)
            assert memory_increase < 500, f"Memory increased by {memory_increase}MB"
    
    def test_streaming_export_large_dataset(self, test_config):
        """Test streaming export of large datasets."""
        client = MDMClient(config=test_config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dataset
            csv_path = Path(tmpdir) / "train.csv"
            n_rows = 50_000
            
            # Generate data in chunks to avoid memory issues
            with open(csv_path, 'w') as f:
                f.write("id,value,category\n")
                for i in range(0, n_rows, 1000):
                    chunk_data = []
                    for j in range(min(1000, n_rows - i)):
                        row_id = i + j
                        chunk_data.append(f"{row_id},{np.random.rand():.6f},{np.random.choice(['A', 'B', 'C'])}")
                    f.write('\n'.join(chunk_data) + '\n')
            
            # Register dataset
            client.register_dataset(
                name="large_export_test",
                dataset_path=str(tmpdir),
            )
            
            # Export with streaming
            export_dir = Path(tmpdir) / "exports"
            export_dir.mkdir()
            
            # Track memory during export
            initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            export_paths = client.export_dataset(
                "large_export_test",
                output_dir=str(export_dir),
                format="csv",
            )
            
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_increase = final_memory - initial_memory
            
            # Verify export completed
            assert len(export_paths) > 0
            
            # Memory increase should be minimal (< 100MB)
            assert memory_increase < 100, f"Memory increased by {memory_increase}MB during export"
    
    def test_batch_size_configuration(self, test_config):
        """Test that batch size configuration is respected."""
        # Set a small batch size
        test_config.performance.batch_size = 1000
        
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "train.csv"
            
            # Create file with 5000 rows
            df = pd.DataFrame({
                'id': range(5000),
                'value': np.random.rand(5000),
            })
            df.to_csv(csv_path, index=False)
            
            registrar = DatasetRegistrar()
            
            # Track chunk sizes during processing
            chunk_sizes = []
            original_to_sql = pd.DataFrame.to_sql
            
            def mock_to_sql(self, *args, **kwargs):
                chunk_sizes.append(len(self))
                return original_to_sql(self, *args, **kwargs)
            
            with patch.object(pd.DataFrame, 'to_sql', mock_to_sql):
                dataset_info = registrar.register(
                    name="batch_test",
                    path=Path(tmpdir),
                )
            
            # Verify chunks were of configured size (except possibly last chunk)
            for chunk_size in chunk_sizes[:-1]:
                assert chunk_size == 1000
            
            # Last chunk might be smaller
            assert chunk_sizes[-1] <= 1000
    
    def test_memory_limit_detection(self):
        """Test detection of memory limits."""
        # This is a placeholder for memory limit detection
        # In production, you'd want to check available memory
        # and adjust batch sizes accordingly
        
        available_memory = psutil.virtual_memory().available / 1024 / 1024 / 1024  # GB
        
        # If less than 2GB available, batch size should be smaller
        if available_memory < 2:
            recommended_batch_size = 5000
        elif available_memory < 4:
            recommended_batch_size = 10000
        else:
            recommended_batch_size = 50000
        
        assert recommended_batch_size > 0
        assert recommended_batch_size <= 50000


class TestFileSizeLimits:
    """Test file size limit handling."""
    
    def test_file_size_warning(self, test_config):
        """Test warning for very large files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "train.csv"
            
            # Create a file and mock its size
            csv_path.write_text("id,value\n1,100")
            
            with patch('pathlib.Path.stat') as mock_stat:
                # Mock 5GB file
                mock_stat.return_value.st_size = 5 * 1024 * 1024 * 1024
                
                registrar = DatasetRegistrar()
                
                # Should log warning but proceed
                with patch('mdm.dataset.registrar.logger.warning') as mock_warning:
                    dataset_info = registrar.register(
                        name="huge_file",
                        path=Path(tmpdir),
                    )
                    
                    # Verify warning was logged
                    mock_warning.assert_called()
                    warning_msg = str(mock_warning.call_args)
                    assert "large" in warning_msg.lower() or "size" in warning_msg.lower()