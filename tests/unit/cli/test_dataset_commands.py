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
        mock_console.print.assert_called_with("\n[yellow]Could not generate column summary: Database error[/yellow]")


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
    
    @patch('mdm.cli.dataset._display_column_summary')
    @patch('mdm.cli.dataset.DatasetRegistrar')
    def test_register_basic(self, mock_registrar_class, mock_display_summary, runner):
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
    
    @patch('mdm.cli.dataset._display_column_summary')
    @patch('mdm.cli.dataset.DatasetRegistrar')
    def test_register_with_all_options(self, mock_registrar_class, mock_display_summary, runner):
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
        assert 'type_schema' in call_kwargs
        assert call_kwargs['type_schema']['date'] == "datetime"
        assert call_kwargs['description'] == "Test dataset"
        assert call_kwargs['tags'] == ["test", "sample"]
        assert call_kwargs['force'] is True
    
    @patch('mdm.cli.dataset.DatasetRegistrar')
    def test_register_no_auto_mode(self, mock_registrar_class, runner):
        """Test registration with --no-auto flag."""
        result = runner.invoke(dataset_app, [
            "register", "test_dataset",
            "--no-auto",
            "--train", "train.csv",
            "--target", "target_col"
        ])
        
        # Currently not implemented
        assert result.exit_code == 1
        assert "Manual registration not yet implemented" in result.stdout
    
    @patch('mdm.cli.dataset._display_column_summary')
    @patch('mdm.cli.dataset.DatasetRegistrar')
    def test_register_with_exception(self, mock_registrar_class, mock_display_summary, runner):
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
                'tables': {'train': 'train_table', 'test': 'test_table'},
                'row_count': 1000,
                'size': 1048576,
                'backend': 'sqlite',
                'backend_compatible': True,
                'current_backend': 'sqlite'
            },
            {
                'name': 'dataset2',
                'problem_type': 'regression',
                'target_column': 'value',
                'tables': {'data': 'data_table'},
                'row_count': 5000,
                'size': 5242880,
                'backend': 'duckdb',
                'backend_compatible': True,
                'current_backend': 'sqlite'
            }
        ]
        mock_list_op_class.return_value = mock_list_op
        
        result = runner.invoke(dataset_app, ["list"])
        
        assert result.exit_code == 0
        assert "dataset1" in result.stdout
        assert "dataset2" in result.stdout
        # Check for truncated values in table
        assert "classificati" in result.stdout  # May be truncated
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
        mock_info_op.execute.assert_called_with("test_dataset", details=False)
    
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
    
    def test_search_with_results(self, runner):
        """Test search with results."""
        with patch('mdm.dataset.operations.SearchOperation') as mock_search_op_class:
            mock_search_op = Mock()
            mock_search_op.execute.return_value = [
                {
                    'name': 'test_dataset1',
                    'description': 'Test dataset 1',
                    'tags': ['test'],
                    'match_location': 'name'
                },
                {
                    'name': 'test_dataset2',
                    'description': 'Test dataset 2',
                    'tags': ['test', 'sample'],
                    'match_location': 'name'
                }
            ]
            mock_search_op_class.return_value = mock_search_op
        
            result = runner.invoke(dataset_app, ["search", "test"])
            
            assert result.exit_code == 0
            assert "test_dataset1" in result.stdout
            assert "test_dataset2" in result.stdout
            assert "Found 2 match(es)" in result.stdout
    
    def test_search_no_results(self, runner):
        """Test search with no results."""
        with patch('mdm.dataset.operations.SearchOperation') as mock_search_op_class:
            mock_search_op = Mock()
            mock_search_op.execute.return_value = []
            mock_search_op_class.return_value = mock_search_op
        
            result = runner.invoke(dataset_app, ["search", "nonexistent"])
            
            assert result.exit_code == 0
            assert "No datasets found" in result.stdout
    
    def test_search_with_tags(self, runner):
        """Test search with tag filter."""
        with patch('mdm.dataset.operations.SearchOperation') as mock_search_op_class:
            mock_search_op = Mock()
            mock_search_op.execute.return_value = []
            mock_search_op_class.return_value = mock_search_op
        
            result = runner.invoke(dataset_app, ["search", "test", "--tag", "sample"])
            
            assert result.exit_code == 0
            mock_search_op.execute.assert_called_with(
                query="test",
                deep=False,
                pattern=False,
                case_sensitive=False,
                tag="sample",
                limit=None
            )


class TestDatasetStatsCommand:
    """Test dataset stats command."""
    
    @pytest.fixture
    def runner(self):
        return CliRunner()
    
    def test_stats_success(self, runner):
        """Test stats command success."""
        with patch('mdm.dataset.operations.StatsOperation') as mock_stats_op_class:
            mock_stats_op = Mock()
            mock_stats_op.execute.return_value = {
                'dataset_name': 'test_dataset',
                'computed_at': '2025-01-09 10:00:00',
                'mode': 'basic',
                'summary': {
                    'total_tables': 1,
                    'total_rows': 1000,
                    'total_columns': 10,
                    'overall_completeness': 0.95
                },
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
                "test_dataset",
                full=False
            )
    
    def test_stats_detailed(self, runner):
        """Test stats command with detailed flag."""
        with patch('mdm.dataset.operations.StatsOperation') as mock_stats_op_class:
            mock_stats_op = Mock()
            mock_stats_op.execute.return_value = {
                'dataset_name': 'test_dataset',
                'computed_at': '2025-01-09 10:00:00',
                'mode': 'full',
                'summary': {
                    'total_tables': 1,
                    'total_rows': 1000,
                    'total_columns': 10,
                    'overall_completeness': 1.0
                },
                'tables': {
                    'train': {
                        'row_count': 1000,
                        'column_count': 10,
                        'columns': {
                            'col1': {'dtype': 'INTEGER', 'null_percentage': 1.0},
                            'col2': {'dtype': 'VARCHAR', 'null_percentage': 0.0}
                        }
                    }
                }
            }
            mock_stats_op_class.return_value = mock_stats_op
            
            result = runner.invoke(dataset_app, ["stats", "test_dataset", "--full"])
            
            assert result.exit_code == 0
            mock_stats_op.execute.assert_called_with(
                "test_dataset",
                full=True
            )


class TestDatasetUpdateCommand:
    """Test dataset update command."""
    
    @pytest.fixture
    def runner(self):
        return CliRunner()
    
    def test_update_description(self, runner):
        """Test updating dataset description."""
        with patch('mdm.dataset.operations.UpdateOperation') as mock_update_op_class:
            mock_update_op = Mock()
            mock_update_op_class.return_value = mock_update_op
        
            result = runner.invoke(dataset_app, [
                "update", "test_dataset",
                "--description", "New description"
            ])
            
            assert result.exit_code == 0
            assert "Dataset 'test_dataset' updated successfully" in result.stdout
            mock_update_op.execute.assert_called_with(
                "test_dataset",
                {"description": "New description"}
            )
    
    def test_update_tags(self, runner):
        """Test updating dataset tags."""
        with patch('mdm.dataset.operations.UpdateOperation') as mock_update_op_class:
            mock_update_op = Mock()
            mock_update_op_class.return_value = mock_update_op
        
            result = runner.invoke(dataset_app, [
                "update", "test_dataset",
                "--target", "new_target"
            ])
            
            assert result.exit_code == 0
            mock_update_op.execute.assert_called_with(
                "test_dataset",
                {"target_column": "new_target"}
            )
    
    def test_update_no_changes(self, runner):
        """Test update with no changes."""
        result = runner.invoke(dataset_app, ["update", "test_dataset"])
        
        assert result.exit_code == 0
        assert "No updates specified" in result.stdout


class TestDatasetExportCommand:
    """Test dataset export command."""
    
    @pytest.fixture
    def runner(self):
        return CliRunner()
    
    def test_export_success(self, runner):
        """Test successful export."""
        with patch('mdm.dataset.operations.ExportOperation') as mock_export_op_class:
            mock_export_op = Mock()
            mock_export_op.execute.return_value = [
                '/tmp/export/train.csv'
            ]
            mock_export_op_class.return_value = mock_export_op
        
            with tempfile.TemporaryDirectory() as tmpdir:
                result = runner.invoke(dataset_app, [
                    "export", "test_dataset",
                    "--output-dir", tmpdir
                ])
            
            assert result.exit_code == 0
            assert "exported successfully" in result.stdout
    
    def test_export_with_format(self, runner):
        """Test export with specific format."""
        with patch('mdm.dataset.operations.ExportOperation') as mock_export_op_class:
            mock_export_op = Mock()
            mock_export_op.execute.return_value = [
                '/tmp/export/train.parquet'
            ]
            mock_export_op_class.return_value = mock_export_op
        
            result = runner.invoke(dataset_app, [
                "export", "test_dataset",
                "--format", "parquet"
            ])
            
            assert result.exit_code == 0
            mock_export_op.execute.assert_called_with(
                name="test_dataset",
                format="parquet",
                output_dir=Path("."),
                table=None,
                compression=None,
                metadata_only=False,
                no_header=False
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
        # Mock the dry_run call that shows info
        mock_remove_op.execute.return_value = {
            'name': 'test_dataset',
            'config_file': '/home/user/.mdm/config/datasets/test_dataset.yaml',
            'dataset_directory': '/home/user/.mdm/datasets/test_dataset',
            'size': 1048576
        }
        mock_remove_op_class.return_value = mock_remove_op
        
        # Simulate user confirming
        result = runner.invoke(dataset_app, [
            "remove", "test_dataset"
        ], input="y\n")
        
        assert result.exit_code == 0
        assert "Are you sure" in result.stdout
        assert "removed successfully" in result.stdout
        # Should be called twice - once for dry run, once for actual removal
        assert mock_remove_op.execute.call_count == 2
        mock_remove_op.execute.assert_any_call("test_dataset", force=True, dry_run=True)
        mock_remove_op.execute.assert_any_call("test_dataset", force=True, dry_run=False)
    
    @patch('mdm.cli.dataset.RemoveOperation')
    def test_remove_cancelled(self, mock_remove_op_class, runner):
        """Test remove cancelled by user."""
        mock_remove_op = Mock()
        # Mock the dry_run call that shows info
        mock_remove_op.execute.return_value = {
            'name': 'test_dataset',
            'config_file': '/home/user/.mdm/config/datasets/test_dataset.yaml',
            'dataset_directory': '/home/user/.mdm/datasets/test_dataset',
            'size': 1048576
        }
        mock_remove_op_class.return_value = mock_remove_op
        
        # Simulate user cancelling
        result = runner.invoke(dataset_app, [
            "remove", "test_dataset"
        ], input="n\n")
        
        assert result.exit_code == 0
        assert "Are you sure" in result.stdout
        assert "Cancelled" in result.stdout
        # Should only be called once for dry run
        mock_remove_op.execute.assert_called_once_with("test_dataset", force=True, dry_run=True)
    
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
        mock_remove_op.execute.assert_called_with("test_dataset", force=True, dry_run=False)