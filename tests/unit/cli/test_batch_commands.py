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
        mock_manager.get_dataset.side_effect = [
            Mock(name="dataset1"),  # First dataset exists
            Mock(name="dataset2"),  # Second dataset exists
        ]
        mock_manager_class.return_value = mock_manager
        
        mock_export_op = Mock()
        mock_export_op.execute.side_effect = [
            {
                'dataset_name': 'dataset1',
                'output_path': '/tmp/exports/dataset1.csv',
                'format': 'csv',
                'total_rows': 1000
            },
            {
                'dataset_name': 'dataset2',
                'output_path': '/tmp/exports/dataset2.csv',
                'format': 'csv',
                'total_rows': 2000
            }
        ]
        mock_export_op_class.return_value = mock_export_op
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(batch_app, [
                "export", "dataset1", "dataset2",
                "--output-dir", tmpdir
            ])
        
        assert result.exit_code == 0
        assert "2 datasets exported successfully" in result.stdout
        assert mock_export_op.execute.call_count == 2
    
    @patch('mdm.cli.batch.DatasetManager')
    @patch('mdm.cli.batch.ExportOperation')
    def test_batch_export_partial_failure(self, mock_export_op_class, mock_manager_class, runner):
        """Test batch export with partial failures."""
        # Setup mocks
        mock_manager = Mock()
        mock_manager.get_dataset.side_effect = [
            Mock(name="dataset1"),
            None,  # dataset2 not found
            Mock(name="dataset3"),
        ]
        mock_manager_class.return_value = mock_manager
        
        mock_export_op = Mock()
        mock_export_op.execute.side_effect = [
            {'dataset_name': 'dataset1', 'output_path': '/tmp/dataset1.csv'},
            Exception("Export failed"),  # dataset3 export fails
        ]
        mock_export_op_class.return_value = mock_export_op
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(batch_app, [
                "export", "dataset1", "dataset2", "dataset3",
                "--output-dir", tmpdir
            ])
        
        assert result.exit_code == 0
        assert "1 datasets exported successfully" in result.stdout
        assert "2 datasets failed" in result.stdout
        assert "dataset2: Dataset not found" in result.stdout
        assert "dataset3: Export failed" in result.stdout
    
    @patch('mdm.cli.batch.DatasetManager')
    @patch('mdm.cli.batch.ExportOperation')
    def test_batch_export_all_options(self, mock_export_op_class, mock_manager_class, runner):
        """Test batch export with all options."""
        mock_manager = Mock()
        mock_manager.get_dataset.return_value = Mock(name="dataset1")
        mock_manager_class.return_value = mock_manager
        
        mock_export_op = Mock()
        mock_export_op.execute.return_value = {
            'dataset_name': 'dataset1',
            'output_path': '/tmp/dataset1.parquet.gz'
        }
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
            dataset_name="dataset1",
            output_path=Path(tmpdir),
            format="parquet",
            compression="gzip",
            tables=None,
            metadata_only=True
        )
    
    @patch('mdm.cli.batch.DatasetManager')
    def test_batch_export_no_datasets(self, mock_manager_class, runner):
        """Test batch export with no valid datasets."""
        mock_manager = Mock()
        mock_manager.get_dataset.return_value = None
        mock_manager_class.return_value = mock_manager
        
        result = runner.invoke(batch_app, [
            "export", "nonexistent",
            "--output-dir", "/tmp"
        ])
        
        assert result.exit_code == 0
        assert "0 datasets exported successfully" in result.stdout
        assert "1 datasets failed" in result.stdout


class TestBatchStatsCommand:
    """Test batch stats command."""
    
    @pytest.fixture
    def runner(self):
        return CliRunner()
    
    @patch('mdm.cli.batch.DatasetManager')
    @patch('mdm.cli.batch.StatsOperation')
    def test_batch_stats_success(self, mock_stats_op_class, mock_manager_class, runner):
        """Test successful batch stats."""
        # Setup mocks
        mock_manager = Mock()
        mock_manager.get_dataset.side_effect = [
            Mock(name="dataset1"),
            Mock(name="dataset2"),
        ]
        mock_manager_class.return_value = mock_manager
        
        mock_stats_op = Mock()
        mock_stats_op.execute.side_effect = [
            {
                'dataset_name': 'dataset1',
                'tables': {
                    'train': {'row_count': 1000, 'size_bytes': 1048576}
                },
                'total_row_count': 1000,
                'total_size_bytes': 1048576
            },
            {
                'dataset_name': 'dataset2',
                'tables': {
                    'train': {'row_count': 2000, 'size_bytes': 2097152}
                },
                'total_row_count': 2000,
                'total_size_bytes': 2097152
            }
        ]
        mock_stats_op_class.return_value = mock_stats_op
        
        result = runner.invoke(batch_app, ["stats", "dataset1", "dataset2"])
        
        assert result.exit_code == 0
        assert "dataset1" in result.stdout
        assert "dataset2" in result.stdout
        assert "1,000" in result.stdout  # Row count formatting
        assert "2,000" in result.stdout
    
    @patch('mdm.cli.batch.DatasetManager')
    @patch('mdm.cli.batch.StatsOperation')
    def test_batch_stats_with_export(self, mock_stats_op_class, mock_manager_class, runner):
        """Test batch stats with CSV export."""
        mock_manager = Mock()
        mock_manager.get_dataset.return_value = Mock(name="dataset1")
        mock_manager_class.return_value = mock_manager
        
        mock_stats_op = Mock()
        mock_stats_op.execute.return_value = {
            'dataset_name': 'dataset1',
            'total_row_count': 1000,
            'total_size_bytes': 1048576
        }
        mock_stats_op_class.return_value = mock_stats_op
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
            result = runner.invoke(batch_app, [
                "stats", "dataset1",
                "--export", tmp.name
            ])
            
            # Check file was created
            assert Path(tmp.name).exists()
            Path(tmp.name).unlink()  # Clean up
        
        assert result.exit_code == 0
        assert f"Statistics exported to {tmp.name}" in result.stdout
    
    @patch('mdm.cli.batch.DatasetManager')
    @patch('mdm.cli.batch.StatsOperation')
    def test_batch_stats_detailed(self, mock_stats_op_class, mock_manager_class, runner):
        """Test batch stats with detailed flag."""
        mock_manager = Mock()
        mock_manager.get_dataset.return_value = Mock(name="dataset1")
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
        
        result = runner.invoke(batch_app, ["stats", "dataset1", "--detailed"])
        
        assert result.exit_code == 0
        mock_stats_op.execute.assert_called_with(
            dataset_name="dataset1",
            detailed=True,
            tables=None
        )
    
    @patch('mdm.cli.batch.DatasetManager')
    def test_batch_stats_empty_list(self, mock_manager_class, runner):
        """Test batch stats with no datasets found."""
        mock_manager = Mock()
        mock_manager.get_dataset.return_value = None
        mock_manager_class.return_value = mock_manager
        
        result = runner.invoke(batch_app, ["stats", "nonexistent"])
        
        assert result.exit_code == 0
        assert "No valid datasets found" in result.stdout


class TestBatchRemoveCommand:
    """Test batch remove command."""
    
    @pytest.fixture
    def runner(self):
        return CliRunner()
    
    @patch('mdm.cli.batch.DatasetManager')
    @patch('mdm.cli.batch.RemoveOperation')
    def test_batch_remove_with_confirmation(self, mock_remove_op_class, mock_manager_class, runner):
        """Test batch remove with user confirmation."""
        # Setup mocks
        mock_manager = Mock()
        mock_manager.get_dataset.side_effect = [
            Mock(name="dataset1"),
            Mock(name="dataset2"),
        ]
        mock_manager_class.return_value = mock_manager
        
        mock_remove_op = Mock()
        mock_remove_op_class.return_value = mock_remove_op
        
        # Simulate user confirming
        result = runner.invoke(batch_app, [
            "remove", "dataset1", "dataset2"
        ], input="y\n")
        
        assert result.exit_code == 0
        assert "Are you sure" in result.stdout
        assert "dataset1" in result.stdout
        assert "dataset2" in result.stdout
        assert "2 datasets removed successfully" in result.stdout
        assert mock_remove_op.execute.call_count == 2
    
    @patch('mdm.cli.batch.DatasetManager')
    def test_batch_remove_cancelled(self, mock_manager_class, runner):
        """Test batch remove cancelled by user."""
        mock_manager = Mock()
        mock_manager.get_dataset.return_value = Mock(name="dataset1")
        mock_manager_class.return_value = mock_manager
        
        # Simulate user cancelling
        result = runner.invoke(batch_app, [
            "remove", "dataset1"
        ], input="n\n")
        
        assert result.exit_code == 0
        assert "Operation cancelled" in result.stdout
    
    @patch('mdm.cli.batch.DatasetManager')
    @patch('mdm.cli.batch.RemoveOperation')
    def test_batch_remove_force(self, mock_remove_op_class, mock_manager_class, runner):
        """Test batch remove with force flag."""
        mock_manager = Mock()
        mock_manager.get_dataset.return_value = Mock(name="dataset1")
        mock_manager_class.return_value = mock_manager
        
        mock_remove_op = Mock()
        mock_remove_op_class.return_value = mock_remove_op
        
        result = runner.invoke(batch_app, [
            "remove", "dataset1", "--force"
        ])
        
        assert result.exit_code == 0
        assert "Are you sure" not in result.stdout
        mock_remove_op.execute.assert_called_once()
    
    @patch('mdm.cli.batch.DatasetManager')
    @patch('mdm.cli.batch.RemoveOperation')
    def test_batch_remove_partial_failure(self, mock_remove_op_class, mock_manager_class, runner):
        """Test batch remove with partial failures."""
        mock_manager = Mock()
        mock_manager.get_dataset.side_effect = [
            Mock(name="dataset1"),
            None,  # dataset2 not found
            Mock(name="dataset3"),
        ]
        mock_manager_class.return_value = mock_manager
        
        mock_remove_op = Mock()
        mock_remove_op.execute.side_effect = [
            None,  # dataset1 removed successfully
            Exception("Permission denied"),  # dataset3 fails
        ]
        mock_remove_op_class.return_value = mock_remove_op
        
        result = runner.invoke(batch_app, [
            "remove", "dataset1", "dataset2", "dataset3",
            "--force"
        ])
        
        assert result.exit_code == 0
        assert "1 datasets removed successfully" in result.stdout
        assert "2 datasets failed" in result.stdout
        assert "dataset2: Dataset not found" in result.stdout
        assert "dataset3: Permission denied" in result.stdout
    
    @patch('mdm.cli.batch.DatasetManager')
    def test_batch_remove_no_valid_datasets(self, mock_manager_class, runner):
        """Test batch remove with no valid datasets."""
        mock_manager = Mock()
        mock_manager.get_dataset.return_value = None
        mock_manager_class.return_value = mock_manager
        
        result = runner.invoke(batch_app, [
            "remove", "nonexistent", "--force"
        ])
        
        assert result.exit_code == 0
        assert "No valid datasets found to remove" in result.stdout