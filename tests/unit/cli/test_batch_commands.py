"""Unit tests for batch CLI commands."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
import pytest
from typer.testing import CliRunner

from mdm.cli.batch import batch_app
from mdm.core.exceptions import DatasetError


class TestBatchExportCommand:
    """Test batch export command."""
    
    @pytest.fixture
    def runner(self):
        return CliRunner()
    
    @pytest.fixture(autouse=True)
    def mock_setup_logging(self):
        """Mock setup_logging for all tests."""
        with patch('mdm.cli.main.setup_logging'):
            yield
    
    @patch('mdm.cli.batch.DatasetManager')
    @patch('mdm.cli.batch.ExportOperation')
    def test_batch_export_success(self, mock_export_op_class, mock_manager_class, runner):
        """Test successful batch export."""
        # Setup mocks
        mock_manager = Mock()
        mock_manager.dataset_exists.side_effect = [True, True]  # Both datasets exist
        mock_manager_class.return_value = mock_manager
        
        mock_export_op = Mock()
        mock_export_op.execute.side_effect = [
            ['/tmp/exports/dataset1/train.csv'],  # Returns list of exported files
            ['/tmp/exports/dataset2/train.csv'],  # Returns list of exported files
        ]
        mock_export_op_class.return_value = mock_export_op
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(batch_app, [
                "export", "dataset1", "dataset2",
                "--output-dir", tmpdir
            ])
        
        assert result.exit_code == 0
        assert "Successfully exported: 2 datasets" in result.stdout
        assert mock_export_op.execute.call_count == 2
    
    @patch('mdm.cli.batch.DatasetManager')
    @patch('mdm.cli.batch.ExportOperation')
    def test_batch_export_partial_failure(self, mock_export_op_class, mock_manager_class, runner):
        """Test batch export with partial failures."""
        # Setup mocks
        mock_manager = Mock()
        mock_manager.dataset_exists.side_effect = [
            True,   # dataset1 exists
            False,  # dataset2 not found
            True,   # dataset3 exists
        ]
        mock_manager_class.return_value = mock_manager
        
        mock_export_op = Mock()
        mock_export_op.execute.side_effect = [
            ['/tmp/dataset1.csv'],  # dataset1 exports successfully
            Exception("Export failed"),  # dataset3 export fails
        ]
        mock_export_op_class.return_value = mock_export_op
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(batch_app, [
                "export", "dataset1", "dataset2", "dataset3",
                "--output-dir", tmpdir
            ])
        
        assert result.exit_code == 0
        assert "Successfully exported: 1 datasets" in result.stdout
        assert "Failed: 2 datasets" in result.stdout
        assert "dataset2: Dataset not found" in result.stdout
        assert "dataset3: Export failed" in result.stdout
    
    @patch('mdm.cli.batch.DatasetManager')
    @patch('mdm.cli.batch.ExportOperation')
    def test_batch_export_all_options(self, mock_export_op_class, mock_manager_class, runner):
        """Test batch export with all options."""
        mock_manager = Mock()
        mock_manager.dataset_exists.return_value = True
        mock_manager_class.return_value = mock_manager
        
        mock_export_op = Mock()
        mock_export_op.execute.return_value = ['/tmp/dataset1.parquet.gz']  # Returns list of exported files
        mock_export_op_class.return_value = mock_export_op
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(batch_app, [
                "export", "dataset1",
                "--output-dir", tmpdir,
                "--format", "parquet",
                "--compression", "gzip",
                "--metadata-only"
            ])
        
        assert result.exit_code == 0
        
        # Verify export was called with correct parameters  
        mock_export_op.execute.assert_called_with(
            name="dataset1",
            format="parquet",
            output_dir=Path(tmpdir) / "dataset1",
            compression="gzip",
            metadata_only=True
        )
    
    @patch('mdm.cli.batch.DatasetManager')
    def test_batch_export_no_datasets(self, mock_manager_class, runner):
        """Test batch export with no valid datasets."""
        mock_manager = Mock()
        mock_manager.dataset_exists.return_value = False
        mock_manager_class.return_value = mock_manager
        
        result = runner.invoke(batch_app, [
            "export", "nonexistent",
            "--output-dir", "/tmp"
        ])
        
        assert result.exit_code == 0
        assert "Successfully exported: 0 datasets" in result.stdout
        assert "Failed: 1 datasets" in result.stdout


class TestBatchStatsCommand:
    """Test batch stats command."""
    
    @pytest.fixture
    def runner(self):
        return CliRunner()
    
    @patch('mdm.cli.batch.DatasetManager')
    @patch('mdm.dataset.operations.StatsOperation')
    def test_batch_stats_success(self, mock_stats_op_class, mock_manager_class, runner):
        """Test successful batch stats."""
        # Setup mocks
        mock_manager = Mock()
        mock_manager.dataset_exists.side_effect = [True, True]
        mock_manager_class.return_value = mock_manager
        
        mock_stats_op = Mock()
        mock_stats_op.execute.side_effect = [
            {
                'dataset_name': 'dataset1',
                'summary': {
                    'total_rows': 1000,
                    'total_columns': 10,
                    'total_tables': 1,
                    'overall_completeness': 0.95
                }
            },
            {
                'dataset_name': 'dataset2',
                'summary': {
                    'total_rows': 2000,
                    'total_columns': 15,
                    'total_tables': 1,
                    'overall_completeness': 0.98
                }
            }
        ]
        mock_stats_op_class.return_value = mock_stats_op
        
        result = runner.invoke(batch_app, ["stats", "dataset1", "dataset2"])
        
        assert result.exit_code == 0
        assert "dataset1" in result.stdout
        assert "dataset2" in result.stdout
        assert "1,000" in result.stdout  # Row count formatting
        assert "2,000" in result.stdout
        assert "95.0%" in result.stdout  # Completeness percentage
        assert "98.0%" in result.stdout
    
    @patch('mdm.cli.batch.DatasetManager')
    @patch('mdm.dataset.operations.StatsOperation')
    def test_batch_stats_with_export(self, mock_stats_op_class, mock_manager_class, runner):
        """Test batch stats with CSV export."""
        mock_manager = Mock()
        mock_manager.dataset_exists.return_value = True
        mock_manager_class.return_value = mock_manager
        
        mock_stats_op = Mock()
        mock_stats_op.execute.return_value = {
            'dataset_name': 'dataset1',
            'summary': {
                'total_rows': 1000,
                'total_columns': 10,
                'total_tables': 1,
                'overall_completeness': 0.95
            }
        }
        mock_stats_op_class.return_value = mock_stats_op
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(batch_app, [
                "stats", "dataset1",
                "--export", tmpdir
            ])
        
        assert result.exit_code == 0
        assert "Stats exported to:" in result.stdout
    
    @patch('mdm.cli.batch.DatasetManager')
    @patch('mdm.dataset.operations.StatsOperation')
    def test_batch_stats_full(self, mock_stats_op_class, mock_manager_class, runner):
        """Test batch stats with full flag."""
        mock_manager = Mock()
        mock_manager.dataset_exists.return_value = True
        mock_manager_class.return_value = mock_manager
        
        mock_stats_op = Mock()
        mock_stats_op.execute.return_value = {
            'dataset_name': 'dataset1',
            'tables': {
                'train': {
                    'row_count': 1000,
                    'column_count': 10,
                    'completeness': 0.95
                }
            }
        }
        mock_stats_op_class.return_value = mock_stats_op
        
        result = runner.invoke(batch_app, ["stats", "dataset1", "--full"])
        
        assert result.exit_code == 0
        mock_stats_op.execute.assert_called_with(
            name="dataset1",
            full=True,
            export=None
        )
    
    @patch('mdm.cli.batch.DatasetManager')
    def test_batch_stats_empty_list(self, mock_manager_class, runner):
        """Test batch stats with no datasets found."""
        mock_manager = Mock()
        mock_manager.dataset_exists.return_value = False
        mock_manager_class.return_value = mock_manager
        
        result = runner.invoke(batch_app, ["stats", "nonexistent"])
        
        assert result.exit_code == 0
        assert "Warning" in result.stdout
        assert "not found" in result.stdout


class TestBatchRemoveCommand:
    """Test batch remove command."""
    
    @pytest.fixture
    def runner(self):
        return CliRunner()
    
    @patch('mdm.cli.batch.DatasetManager')
    @patch('mdm.dataset.operations.RemoveOperation')
    def test_batch_remove_with_confirmation(self, mock_remove_op_class, mock_manager_class, runner):
        """Test batch remove with user confirmation."""
        # Setup mocks
        mock_manager = Mock()
        mock_manager.dataset_exists.side_effect = [True, True]  # Both datasets exist
        mock_manager_class.return_value = mock_manager
        
        mock_remove_op = Mock()
        # dry_run calls to get info, then actual remove calls
        mock_remove_op.execute.side_effect = [
            {'name': 'dataset1', 'size': 1024},  # dry_run info for dataset1
            {'name': 'dataset2', 'size': 2048},  # dry_run info for dataset2
            None,  # actual remove dataset1
            None,  # actual remove dataset2
        ]
        mock_remove_op_class.return_value = mock_remove_op
        
        # Simulate user confirming
        result = runner.invoke(batch_app, [
            "remove", "dataset1", "dataset2"
        ], input="y\n")
        
        assert result.exit_code == 0
        assert "Are you sure" in result.stdout
        assert "dataset1" in result.stdout
        assert "dataset2" in result.stdout
        assert "Removed 2 datasets" in result.stdout
        assert mock_remove_op.execute.call_count == 4  # 2 dry runs + 2 actual removes
    
    @patch('mdm.cli.batch.DatasetManager')
    @patch('mdm.dataset.operations.RemoveOperation')
    def test_batch_remove_cancelled(self, mock_remove_op_class, mock_manager_class, runner):
        """Test batch remove cancelled by user."""
        mock_manager = Mock()
        mock_manager.dataset_exists.return_value = True
        mock_manager_class.return_value = mock_manager
        
        mock_remove_op = Mock()
        mock_remove_op.execute.return_value = {'name': 'dataset1', 'size': 1024}  # dry_run info
        mock_remove_op_class.return_value = mock_remove_op
        
        # Simulate user cancelling
        result = runner.invoke(batch_app, [
            "remove", "dataset1"
        ], input="n\n")
        
        assert result.exit_code == 0
        assert "Cancelled" in result.stdout
    
    @patch('mdm.cli.batch.DatasetManager')
    @patch('mdm.dataset.operations.RemoveOperation')
    def test_batch_remove_force(self, mock_remove_op_class, mock_manager_class, runner):
        """Test batch remove with force flag."""
        mock_manager = Mock()
        mock_manager.dataset_exists.return_value = True
        mock_manager_class.return_value = mock_manager
        
        mock_remove_op = Mock()
        mock_remove_op.execute.side_effect = [
            {'name': 'dataset1', 'size': 1024},  # dry_run info
            None,  # actual remove
        ]
        mock_remove_op_class.return_value = mock_remove_op
        
        result = runner.invoke(batch_app, [
            "remove", "dataset1", "--force"
        ])
        
        assert result.exit_code == 0
        assert "Are you sure" not in result.stdout
        assert mock_remove_op.execute.call_count == 2  # dry run + actual remove
    
    @patch('mdm.cli.batch.DatasetManager')
    @patch('mdm.dataset.operations.RemoveOperation')
    def test_batch_remove_partial_failure(self, mock_remove_op_class, mock_manager_class, runner):
        """Test batch remove with partial failures."""
        mock_manager = Mock()
        mock_manager.dataset_exists.side_effect = [
            True,   # dataset1 exists
            False,  # dataset2 not found
            True,   # dataset3 exists
        ]
        mock_manager_class.return_value = mock_manager
        
        mock_remove_op = Mock()
        mock_remove_op.execute.side_effect = [
            {'name': 'dataset1', 'size': 1024},  # dry_run info for dataset1
            {'name': 'dataset3', 'size': 3072},  # dry_run info for dataset3
            None,  # dataset1 removed successfully
            Exception("Permission denied"),  # dataset3 fails
        ]
        mock_remove_op_class.return_value = mock_remove_op
        
        result = runner.invoke(batch_app, [
            "remove", "dataset1", "dataset2", "dataset3",
            "--force"
        ])
        
        assert result.exit_code == 0
        assert "Removed 1 datasets" in result.stdout
        assert "Failed to remove 1 datasets" in result.stdout
        assert "Permission denied" in result.stdout
    
    @patch('mdm.cli.batch.DatasetManager')
    def test_batch_remove_no_valid_datasets(self, mock_manager_class, runner):
        """Test batch remove with no valid datasets."""
        mock_manager = Mock()
        mock_manager.dataset_exists.return_value = False
        mock_manager_class.return_value = mock_manager
        
        result = runner.invoke(batch_app, [
            "remove", "nonexistent", "--force"
        ])
        
        assert result.exit_code == 0
        assert "No valid datasets to remove" in result.stdout