"""Unit tests for dataset CLI commands."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
import pytest
from typer.testing import CliRunner
import pandas as pd

from mdm.cli.dataset import dataset_app, _format_size, _display_column_summary
from mdm.core.exceptions import DatasetError


class TestHelperFunctions:
    """Test dataset command helper functions."""
    
    def test_format_size(self):
        """Test size formatting."""
        assert _format_size(100) == "100.0 B"
        assert _format_size(1024) == "1.0 KB"
        assert _format_size(1048576) == "1.0 MB"
        assert _format_size(1073741824) == "1.0 GB"
    
    @patch('mdm.cli.dataset.BackendFactory')
    @patch('mdm.cli.dataset.console')
    def test_display_column_summary(self, mock_console, mock_factory):
        """Test column summary display."""
        # Setup mocks
        mock_backend = Mock()
        mock_backend.query.side_effect = [
            pd.DataFrame({'total_rows': [1000]}),  # Total rows query
            pd.DataFrame({'null_count': [50]}),     # Null count for col1
            pd.DataFrame({'null_count': [0]}),      # Null count for col2
        ]
        mock_backend.get_table_info.return_value = {
            'columns': [
                {'name': 'col1', 'type': 'VARCHAR(255)'},
                {'name': 'col2', 'type': 'INTEGER'},
            ]
        }
        mock_factory.create.return_value = mock_backend
        
        mock_dataset_info = Mock()
        mock_dataset_info.database = {'backend': 'sqlite', 'path': '/tmp/test.db'}
        mock_dataset_info.tables = {'train': 'train_table'}
        
        mock_manager = Mock()
        
        # Call function
        _display_column_summary(mock_dataset_info, mock_manager, 'train')
        
        # Verify table was created and displayed
        assert mock_console.print.called
        
    @patch('mdm.cli.dataset.BackendFactory')
    @patch('mdm.cli.dataset.console')
    def test_display_column_summary_with_error(self, mock_console, mock_factory):
        """Test column summary display with query error."""
        # Setup mock that raises exception
        mock_backend = Mock()
        mock_backend.query.side_effect = Exception("Database error")
        mock_factory.create.return_value = mock_backend
        
        mock_dataset_info = Mock()
        mock_dataset_info.database = {'backend': 'sqlite', 'path': '/tmp/test.db'}
        mock_dataset_info.tables = {'train': 'train_table'}
        
        # Should not raise, just print error
        _display_column_summary(mock_dataset_info, Mock(), 'train')
        
        # Verify error was printed
        mock_console.print.assert_called_with("[red]Error displaying column summary:[/red] Database error")


class TestDatasetRegisterCommand:
    """Test dataset register command."""
    
    @pytest.fixture
    def runner(self):
        return CliRunner()
    
    @pytest.fixture(autouse=True)
    def mock_setup_logging(self):
        """Mock setup_logging for all tests."""
        with patch('mdm.cli.main.setup_logging'):
            yield
    
    @patch('mdm.cli.dataset.DatasetRegistrar')
    @patch('mdm.cli.dataset.console')
    def test_register_basic(self, mock_console, mock_registrar_class, runner):
        """Test basic dataset registration."""
        # Setup mock registrar
        mock_registrar = Mock()
        mock_dataset_info = Mock()
        mock_dataset_info.name = "test_dataset"
        mock_dataset_info.target_column = "target"
        mock_dataset_info.problem_type = "classification"
        mock_dataset_info.id_columns = ["id"]
        mock_dataset_info.tables = {"train": "train_table"}
        mock_dataset_info.database = {'backend': 'sqlite', 'path': '/tmp/test.db'}
        
        mock_registrar.register.return_value = mock_dataset_info
        mock_registrar_class.return_value = mock_registrar
        
        with tempfile.NamedTemporaryFile(suffix='.csv') as tmp:
            result = runner.invoke(dataset_app, [
                "register", "test_dataset", tmp.name
            ])
        
        assert result.exit_code == 0
        assert "registered successfully" in result.stdout
        mock_registrar.register.assert_called_once()
    
    @patch('mdm.cli.dataset.DatasetRegistrar')
    def test_register_with_all_options(self, mock_registrar_class, runner):
        """Test registration with all options."""
        mock_registrar = Mock()
        mock_dataset_info = Mock()
        mock_dataset_info.name = "test_dataset"
        mock_dataset_info.target_column = "target"
        mock_dataset_info.problem_type = "regression"
        mock_dataset_info.id_columns = ["id1", "id2"]
        mock_dataset_info.tables = {"train": "train_table"}
        mock_dataset_info.database = {'backend': 'sqlite', 'path': '/tmp/test.db'}
        
        mock_registrar.register.return_value = mock_dataset_info
        mock_registrar_class.return_value = mock_registrar
        
        with tempfile.NamedTemporaryFile(suffix='.csv') as tmp:
            result = runner.invoke(dataset_app, [
                "register", "test_dataset", tmp.name,
                "--target", "target",
                "--problem-type", "regression",
                "--id-columns", "id1,id2",
                "--datetime-columns", "date",
                "--description", "Test dataset",
                "--tags", "test,sample",
                "--force"
            ])
        
        assert result.exit_code == 0
        
        # Verify all parameters were passed
        call_kwargs = mock_registrar.register.call_args[1]
        assert call_kwargs['target_column'] == "target"
        assert call_kwargs['problem_type'] == "regression"
        assert call_kwargs['id_columns'] == ["id1", "id2"]
        assert call_kwargs['datetime_columns'] == ["date"]
        assert call_kwargs['description'] == "Test dataset"
        assert call_kwargs['tags'] == ["test", "sample"]
        assert call_kwargs['force'] is True
    
    @patch('mdm.cli.dataset.DatasetRegistrar')
    def test_register_no_auto_mode(self, mock_registrar_class, runner):
        """Test registration with --no-auto flag."""
        result = runner.invoke(dataset_app, [
            "register", "test_dataset",
            "--no-auto",
            "--train", "train.csv"
        ])
        
        # Currently not implemented
        assert result.exit_code == 1
        assert "Manual registration not yet implemented" in result.stdout
    
    @patch('mdm.cli.dataset.DatasetRegistrar')
    def test_register_with_exception(self, mock_registrar_class, runner):
        """Test registration with exception."""
        mock_registrar = Mock()
        mock_registrar.register.side_effect = DatasetError("Registration failed")
        mock_registrar_class.return_value = mock_registrar
        
        with tempfile.NamedTemporaryFile(suffix='.csv') as tmp:
            result = runner.invoke(dataset_app, [
                "register", "test_dataset", tmp.name
            ])
        
        assert result.exit_code == 1
        assert "Registration failed" in result.stdout


class TestDatasetListCommand:
    """Test dataset list command."""
    
    @pytest.fixture
    def runner(self):
        return CliRunner()
    
    @patch('mdm.cli.dataset.ListOperation')
    def test_list_empty(self, mock_list_op_class, runner):
        """Test list with no datasets."""
        mock_list_op = Mock()
        mock_list_op.execute.return_value = []
        mock_list_op_class.return_value = mock_list_op
        
        result = runner.invoke(dataset_app, ["list"])
        
        assert result.exit_code == 0
        assert "No datasets registered yet" in result.stdout
    
    @patch('mdm.cli.dataset.ListOperation')
    def test_list_with_datasets(self, mock_list_op_class, runner):
        """Test list with datasets."""
        mock_list_op = Mock()
        mock_list_op.execute.return_value = [
            {
                'name': 'dataset1',
                'problem_type': 'classification',
                'target_column': 'target',
                'tables': ['train', 'test'],
                'row_count': 1000,
                'size_bytes': 1048576,
                'backend': 'sqlite'
            },
            {
                'name': 'dataset2',
                'problem_type': 'regression',
                'target_column': 'value',
                'tables': ['data'],
                'row_count': 5000,
                'size_bytes': 5242880,
                'backend': 'duckdb'
            }
        ]
        mock_list_op_class.return_value = mock_list_op
        
        result = runner.invoke(dataset_app, ["list"])
        
        assert result.exit_code == 0
        assert "dataset1" in result.stdout
        assert "dataset2" in result.stdout
        assert "classification" in result.stdout
        assert "regression" in result.stdout
    
    @patch('mdm.cli.dataset.ListOperation')
    def test_list_with_filters(self, mock_list_op_class, runner):
        """Test list with filters."""
        mock_list_op = Mock()
        mock_list_op.execute.return_value = []
        mock_list_op_class.return_value = mock_list_op
        
        result = runner.invoke(dataset_app, [
            "list",
            "--filter", "problem_type=classification",
            "--sort-by", "registration_date",
            "--limit", "10"
        ])
        
        assert result.exit_code == 0
        
        # Verify parameters were passed
        mock_list_op.execute.assert_called_with(
            format="rich",
            filter_str="problem_type=classification",
            sort_by="registration_date",
            limit=10
        )
    
    @patch('mdm.cli.dataset.ListOperation')
    def test_list_with_exception(self, mock_list_op_class, runner):
        """Test list with exception."""
        mock_list_op = Mock()
        mock_list_op.execute.side_effect = Exception("Database error")
        mock_list_op_class.return_value = mock_list_op
        
        result = runner.invoke(dataset_app, ["list"])
        
        assert result.exit_code == 1
        assert "Database error" in result.stdout


class TestDatasetInfoCommand:
    """Test dataset info command."""
    
    @pytest.fixture
    def runner(self):
        return CliRunner()
    
    @patch('mdm.cli.dataset.InfoOperation')
    def test_info_success(self, mock_info_op_class, runner):
        """Test info command success."""
        mock_info_op = Mock()
        mock_info_op.execute.return_value = {
            'name': 'test_dataset',
            'description': 'Test description',
            'target_column': 'target',
            'problem_type': 'classification',
            'id_columns': ['id'],
            'tables': {'train': 'train_table'},
            'database': {'backend': 'sqlite', 'path': '/tmp/test.db'},
            'tags': ['test', 'sample']
        }
        mock_info_op_class.return_value = mock_info_op
        
        result = runner.invoke(dataset_app, ["info", "test_dataset"])
        
        assert result.exit_code == 0
        assert "test_dataset" in result.stdout
        mock_info_op.execute.assert_called_with(dataset_name="test_dataset")
    
    @patch('mdm.cli.dataset.InfoOperation')
    def test_info_dataset_not_found(self, mock_info_op_class, runner):
        """Test info command with non-existent dataset."""
        mock_info_op = Mock()
        mock_info_op.execute.side_effect = DatasetError("Dataset not found")
        mock_info_op_class.return_value = mock_info_op
        
        result = runner.invoke(dataset_app, ["info", "nonexistent"])
        
        assert result.exit_code == 1
        assert "Dataset not found" in result.stdout


class TestDatasetSearchCommand:
    """Test dataset search command."""
    
    @pytest.fixture
    def runner(self):
        return CliRunner()
    
    @patch('mdm.cli.dataset.DatasetManager')
    def test_search_with_results(self, mock_manager_class, runner):
        """Test search with results."""
        mock_manager = Mock()
        mock_manager.search_datasets.return_value = [
            {
                'name': 'test_dataset1',
                'problem_type': 'classification',
                'target_column': 'target',
                'tags': ['test']
            },
            {
                'name': 'test_dataset2',
                'problem_type': 'regression',
                'target_column': 'value',
                'tags': ['test', 'sample']
            }
        ]
        mock_manager_class.return_value = mock_manager
        
        result = runner.invoke(dataset_app, ["search", "test"])
        
        assert result.exit_code == 0
        assert "test_dataset1" in result.stdout
        assert "test_dataset2" in result.stdout
        assert "Found 2 datasets" in result.stdout
    
    @patch('mdm.cli.dataset.DatasetManager')
    def test_search_no_results(self, mock_manager_class, runner):
        """Test search with no results."""
        mock_manager = Mock()
        mock_manager.search_datasets.return_value = []
        mock_manager_class.return_value = mock_manager
        
        result = runner.invoke(dataset_app, ["search", "nonexistent"])
        
        assert result.exit_code == 0
        assert "No datasets found" in result.stdout
    
    @patch('mdm.cli.dataset.DatasetManager')
    def test_search_with_tags(self, mock_manager_class, runner):
        """Test search with tag filter."""
        mock_manager = Mock()
        mock_manager.search_datasets.return_value = []
        mock_manager_class.return_value = mock_manager
        
        result = runner.invoke(dataset_app, ["search", "test", "--tag", "sample"])
        
        assert result.exit_code == 0
        mock_manager.search_datasets.assert_called_with(pattern="test", tag="sample")


class TestDatasetStatsCommand:
    """Test dataset stats command."""
    
    @pytest.fixture
    def runner(self):
        return CliRunner()
    
    @patch('mdm.cli.dataset.StatsOperation')
    def test_stats_success(self, mock_stats_op_class, runner):
        """Test stats command success."""
        mock_stats_op = Mock()
        mock_stats_op.execute.return_value = {
            'dataset_name': 'test_dataset',
            'tables': {
                'train': {
                    'row_count': 1000,
                    'column_count': 10,
                    'size_bytes': 1048576,
                    'completeness': 0.95,
                    'missing_cells': 500
                }
            },
            'total_size_bytes': 1048576
        }
        mock_stats_op_class.return_value = mock_stats_op
        
        result = runner.invoke(dataset_app, ["stats", "test_dataset"])
        
        assert result.exit_code == 0
        assert "test_dataset" in result.stdout
        mock_stats_op.execute.assert_called_with(
            dataset_name="test_dataset",
            detailed=False,
            tables=None
        )
    
    @patch('mdm.cli.dataset.StatsOperation')
    def test_stats_detailed(self, mock_stats_op_class, runner):
        """Test stats command with detailed flag."""
        mock_stats_op = Mock()
        mock_stats_op.execute.return_value = {
            'dataset_name': 'test_dataset',
            'tables': {
                'train': {
                    'row_count': 1000,
                    'column_count': 10,
                    'columns': {
                        'col1': {'type': 'INTEGER', 'null_count': 10},
                        'col2': {'type': 'VARCHAR', 'null_count': 0}
                    }
                }
            }
        }
        mock_stats_op_class.return_value = mock_stats_op
        
        result = runner.invoke(dataset_app, ["stats", "test_dataset", "--detailed"])
        
        assert result.exit_code == 0
        mock_stats_op.execute.assert_called_with(
            dataset_name="test_dataset",
            detailed=True,
            tables=None
        )


class TestDatasetUpdateCommand:
    """Test dataset update command."""
    
    @pytest.fixture
    def runner(self):
        return CliRunner()
    
    @patch('mdm.cli.dataset.DatasetManager')
    def test_update_description(self, mock_manager_class, runner):
        """Test updating dataset description."""
        mock_manager = Mock()
        mock_manager_class.return_value = mock_manager
        
        result = runner.invoke(dataset_app, [
            "update", "test_dataset",
            "--description", "New description"
        ])
        
        assert result.exit_code == 0
        assert "Updated dataset 'test_dataset'" in result.stdout
        mock_manager.update_dataset.assert_called_with(
            "test_dataset",
            description="New description",
            tags=None
        )
    
    @patch('mdm.cli.dataset.DatasetManager')
    def test_update_tags(self, mock_manager_class, runner):
        """Test updating dataset tags."""
        mock_manager = Mock()
        mock_manager_class.return_value = mock_manager
        
        result = runner.invoke(dataset_app, [
            "update", "test_dataset",
            "--tags", "new,tags,here"
        ])
        
        assert result.exit_code == 0
        mock_manager.update_dataset.assert_called_with(
            "test_dataset",
            description=None,
            tags=["new", "tags", "here"]
        )
    
    @patch('mdm.cli.dataset.DatasetManager')
    def test_update_no_changes(self, mock_manager_class, runner):
        """Test update with no changes."""
        result = runner.invoke(dataset_app, ["update", "test_dataset"])
        
        assert result.exit_code == 1
        assert "No updates specified" in result.stdout


class TestDatasetExportCommand:
    """Test dataset export command."""
    
    @pytest.fixture
    def runner(self):
        return CliRunner()
    
    @patch('mdm.cli.dataset.ExportOperation')
    def test_export_success(self, mock_export_op_class, runner):
        """Test successful export."""
        mock_export_op = Mock()
        mock_export_op.execute.return_value = {
            'dataset_name': 'test_dataset',
            'output_path': '/tmp/export.csv',
            'format': 'csv',
            'tables_exported': ['train'],
            'total_rows': 1000,
            'file_size': 1048576
        }
        mock_export_op_class.return_value = mock_export_op
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(dataset_app, [
                "export", "test_dataset",
                "--output", tmpdir
            ])
        
        assert result.exit_code == 0
        assert "Export completed successfully" in result.stdout
    
    @patch('mdm.cli.dataset.ExportOperation')
    def test_export_with_format(self, mock_export_op_class, runner):
        """Test export with specific format."""
        mock_export_op = Mock()
        mock_export_op.execute.return_value = {
            'dataset_name': 'test_dataset',
            'output_path': '/tmp/export.parquet',
            'format': 'parquet'
        }
        mock_export_op_class.return_value = mock_export_op
        
        result = runner.invoke(dataset_app, [
            "export", "test_dataset",
            "--format", "parquet"
        ])
        
        assert result.exit_code == 0
        mock_export_op.execute.assert_called_with(
            dataset_name="test_dataset",
            output_path=Path("."),
            format="parquet",
            compression=None,
            tables=None,
            metadata_only=False
        )


class TestDatasetRemoveCommand:
    """Test dataset remove command."""
    
    @pytest.fixture
    def runner(self):
        return CliRunner()
    
    @patch('mdm.cli.dataset.RemoveOperation')
    def test_remove_with_confirmation(self, mock_remove_op_class, runner):
        """Test remove with confirmation."""
        mock_remove_op = Mock()
        mock_remove_op_class.return_value = mock_remove_op
        
        # Simulate user confirming
        result = runner.invoke(dataset_app, [
            "remove", "test_dataset"
        ], input="y\n")
        
        assert result.exit_code == 0
        assert "Are you sure" in result.stdout
        assert "removed successfully" in result.stdout
        mock_remove_op.execute.assert_called_with(dataset_name="test_dataset")
    
    @patch('mdm.cli.dataset.RemoveOperation')
    def test_remove_cancelled(self, mock_remove_op_class, runner):
        """Test remove cancelled by user."""
        mock_remove_op = Mock()
        mock_remove_op_class.return_value = mock_remove_op
        
        # Simulate user cancelling
        result = runner.invoke(dataset_app, [
            "remove", "test_dataset"
        ], input="n\n")
        
        assert result.exit_code == 0
        assert "Are you sure" in result.stdout
        assert "Cancelled" in result.stdout
        mock_remove_op.execute.assert_not_called()
    
    @patch('mdm.cli.dataset.RemoveOperation')
    def test_remove_force(self, mock_remove_op_class, runner):
        """Test remove with force flag."""
        mock_remove_op = Mock()
        mock_remove_op_class.return_value = mock_remove_op
        
        result = runner.invoke(dataset_app, [
            "remove", "test_dataset", "--force"
        ])
        
        assert result.exit_code == 0
        assert "Are you sure" not in result.stdout
        mock_remove_op.execute.assert_called_with(dataset_name="test_dataset")