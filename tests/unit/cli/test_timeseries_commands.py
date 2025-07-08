"""Unit tests for timeseries CLI commands."""

import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import pytest
from typer.testing import CliRunner
import pandas as pd

from mdm.cli.timeseries import app


class TestTimeseriesAnalyzeCommand:
    """Test timeseries analyze command."""
    
    @pytest.fixture
    def runner(self):
        return CliRunner()
    
    @pytest.fixture(autouse=True)
    def mock_setup_logging(self):
        """Mock setup_logging for all tests."""
        with patch('mdm.cli.main.setup_logging'):
            yield
    
    @patch('mdm.cli.timeseries.MDMClient')
    @patch('mdm.cli.timeseries.TimeSeriesAnalyzer')
    def test_analyze_success(self, mock_analyzer_class, mock_client_class, runner):
        """Test successful time series analysis."""
        # Setup mocks
        mock_client = Mock()
        mock_dataset_info = Mock()
        mock_dataset_info.time_column = "date"
        mock_dataset_info.target_column = "value"
        mock_client.get_dataset.return_value = mock_dataset_info
        
        # Create sample dataframe
        train_df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=100, freq='D'),
            'value': range(100)
        })
        mock_client.load_dataset_files.return_value = (train_df, None)
        mock_client_class.return_value = mock_client
        
        # Setup analyzer mock
        mock_analyzer = Mock()
        mock_analyzer.analyze.return_value = {
            'time_range': {
                'start': datetime(2023, 1, 1),
                'end': datetime(2023, 4, 10),
                'duration_days': 99
            },
            'frequency': 'daily',
            'missing_timestamps': {
                'count': 0,
                'percentage': 0.0,
                'dates': []
            },
            'trend': {
                'direction': 'increasing',
                'strength': 0.95
            },
            'seasonality': {
                'weekly': True,
                'monthly': False
            },
            'stationarity': {
                'is_stationary': False,
                'adf_statistic': -2.5,
                'p_value': 0.12
            }
        }
        mock_analyzer_class.return_value = mock_analyzer
        
        result = runner.invoke(app, ["analyze", "test_dataset"])
        
        assert result.exit_code == 0
        assert "Time Series Analysis: test_dataset" in result.stdout
        assert "Time Range:" in result.stdout
        assert "Start: 2023-01-01" in result.stdout
        assert "Duration: 99 days" in result.stdout
        assert "Frequency: daily" in result.stdout
        assert "Seasonality Detected:" in result.stdout
        assert "Weekly pattern" in result.stdout
    
    @patch('mdm.cli.timeseries.MDMClient')
    def test_analyze_dataset_not_found(self, mock_client_class, runner):
        """Test analyze with non-existent dataset."""
        mock_client = Mock()
        mock_client.get_dataset.return_value = None
        mock_client_class.return_value = mock_client
        
        result = runner.invoke(app, ["analyze", "nonexistent"])
        
        assert result.exit_code == 1
        assert "Dataset 'nonexistent' not found" in result.stdout
    
    @patch('mdm.cli.timeseries.MDMClient')
    def test_analyze_no_time_column(self, mock_client_class, runner):
        """Test analyze with dataset lacking time column."""
        mock_client = Mock()
        mock_dataset_info = Mock()
        mock_dataset_info.time_column = None
        mock_client.get_dataset.return_value = mock_dataset_info
        mock_client_class.return_value = mock_client
        
        result = runner.invoke(app, ["analyze", "test_dataset"])
        
        assert result.exit_code == 1
        assert "has no time column configured" in result.stdout
    
    @patch('mdm.cli.timeseries.MDMClient')
    @patch('mdm.cli.timeseries.TimeSeriesAnalyzer')
    def test_analyze_with_exception(self, mock_analyzer_class, mock_client_class, runner):
        """Test analyze with exception during analysis."""
        mock_client = Mock()
        mock_dataset_info = Mock()
        mock_dataset_info.time_column = "date"
        mock_dataset_info.target_column = "value"
        mock_client.get_dataset.return_value = mock_dataset_info
        mock_client.load_dataset_files.return_value = (pd.DataFrame(), None)
        mock_client_class.return_value = mock_client
        
        mock_analyzer = Mock()
        mock_analyzer.analyze.side_effect = Exception("Analysis failed")
        mock_analyzer_class.return_value = mock_analyzer
        
        result = runner.invoke(app, ["analyze", "test_dataset"])
        
        assert result.exit_code == 1
        assert "Analysis failed" in result.stdout


class TestTimeseriesSplitCommand:
    """Test timeseries split command."""
    
    @pytest.fixture
    def runner(self):
        return CliRunner()
    
    @patch('mdm.cli.timeseries.MDMClient')
    def test_split_success(self, mock_client_class, runner):
        """Test successful time series split."""
        # Setup mocks
        mock_client = Mock()
        mock_dataset_info = Mock()
        mock_dataset_info.time_column = "date"
        mock_client.get_dataset.return_value = mock_dataset_info
        
        # Create split dataframes
        train_df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=70, freq='D'),
            'value': range(70)
        })
        test_df = pd.DataFrame({
            'date': pd.date_range('2023-03-12', periods=30, freq='D'),
            'value': range(70, 100)
        })
        
        mock_client.split_time_series.return_value = {
            'train': train_df,
            'test': test_df
        }
        mock_client_class.return_value = mock_client
        
        result = runner.invoke(app, ["split", "test_dataset"])
        
        assert result.exit_code == 0
        assert "Splitting dataset 'test_dataset' by time" in result.stdout
        assert "Split Results:" in result.stdout
        assert "train: 70 rows" in result.stdout
        assert "test: 30 rows" in result.stdout
        assert "Time series split completed!" in result.stdout
    
    @patch('mdm.cli.timeseries.MDMClient')
    def test_split_with_validation(self, mock_client_class, runner):
        """Test time series split with validation set."""
        mock_client = Mock()
        mock_dataset_info = Mock()
        mock_dataset_info.time_column = "date"
        mock_client.get_dataset.return_value = mock_dataset_info
        
        # Create three splits
        splits = {
            'train': pd.DataFrame({
                'date': pd.date_range('2023-01-01', periods=60, freq='D'),
                'value': range(60)
            }),
            'val': pd.DataFrame({
                'date': pd.date_range('2023-03-02', periods=20, freq='D'),
                'value': range(60, 80)
            }),
            'test': pd.DataFrame({
                'date': pd.date_range('2023-03-22', periods=20, freq='D'),
                'value': range(80, 100)
            })
        }
        
        mock_client.split_time_series.return_value = splits
        mock_client_class.return_value = mock_client
        
        result = runner.invoke(app, [
            "split", "test_dataset",
            "--test-days", "20",
            "--val-days", "20"
        ])
        
        assert result.exit_code == 0
        assert "train: 60 rows" in result.stdout
        assert "val: 20 rows" in result.stdout
        assert "test: 20 rows" in result.stdout
        
        # Verify split_time_series was called with correct params
        mock_client.split_time_series.assert_called_with("test_dataset", 20, 20)
    
    @patch('mdm.cli.timeseries.MDMClient')
    def test_split_with_output(self, mock_client_class, runner):
        """Test time series split with output directory."""
        mock_client = Mock()
        mock_dataset_info = Mock()
        mock_dataset_info.time_column = "date"
        mock_client.get_dataset.return_value = mock_dataset_info
        
        splits = {
            'train': pd.DataFrame({'date': ['2023-01-01'], 'value': [1]}),
            'test': pd.DataFrame({'date': ['2023-02-01'], 'value': [2]})
        }
        mock_client.split_time_series.return_value = splits
        mock_client_class.return_value = mock_client
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(app, [
                "split", "test_dataset",
                "--output", tmpdir
            ])
            
            assert result.exit_code == 0
            
            # Check files were created
            train_file = Path(tmpdir) / "test_dataset_train.csv"
            test_file = Path(tmpdir) / "test_dataset_test.csv"
            assert f"Saved train to {train_file}" in result.stdout
            assert f"Saved test to {test_file}" in result.stdout
    
    @patch('mdm.cli.timeseries.MDMClient')
    def test_split_dataset_not_found(self, mock_client_class, runner):
        """Test split with non-existent dataset."""
        mock_client = Mock()
        mock_client.get_dataset.return_value = None
        mock_client_class.return_value = mock_client
        
        result = runner.invoke(app, ["split", "nonexistent"])
        
        assert result.exit_code == 1
        assert "Dataset 'nonexistent' not found" in result.stdout
    
    @patch('mdm.cli.timeseries.MDMClient')
    def test_split_with_exception(self, mock_client_class, runner):
        """Test split with exception."""
        mock_client = Mock()
        mock_dataset_info = Mock()
        mock_client.get_dataset.return_value = mock_dataset_info
        mock_client.split_time_series.side_effect = Exception("Split failed")
        mock_client_class.return_value = mock_client
        
        result = runner.invoke(app, ["split", "test_dataset"])
        
        assert result.exit_code == 1
        assert "Split failed" in result.stdout


class TestTimeseriesValidateCommand:
    """Test timeseries validate command."""
    
    @pytest.fixture
    def runner(self):
        return CliRunner()
    
    @patch('mdm.cli.timeseries.MDMClient')
    @patch('mdm.utils.time_series.TimeSeriesSplitter')
    def test_validate_success(self, mock_splitter_class, mock_client_class, runner):
        """Test successful time series cross-validation."""
        # Setup mocks
        mock_client = Mock()
        mock_dataset_info = Mock()
        mock_dataset_info.time_column = "date"
        mock_dataset_info.group_column = None
        mock_client.get_dataset.return_value = mock_dataset_info
        
        train_df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=100, freq='D'),
            'value': range(100)
        })
        mock_client.load_dataset_files.return_value = (train_df, None)
        mock_client_class.return_value = mock_client
        
        # Setup splitter mock
        mock_splitter = Mock()
        mock_splitter.split_by_folds.return_value = [
            {
                'fold': 1,
                'train_period': (datetime(2023, 1, 1), datetime(2023, 2, 15)),
                'test_period': (datetime(2023, 2, 16), datetime(2023, 3, 2)),
                'train': pd.DataFrame({'date': pd.date_range('2023-01-01', periods=46)}),
                'test': pd.DataFrame({'date': pd.date_range('2023-02-16', periods=15)})
            },
            {
                'fold': 2,
                'train_period': (datetime(2023, 1, 1), datetime(2023, 3, 2)),
                'test_period': (datetime(2023, 3, 3), datetime(2023, 3, 17)),
                'train': pd.DataFrame({'date': pd.date_range('2023-01-01', periods=61)}),
                'test': pd.DataFrame({'date': pd.date_range('2023-03-03', periods=15)})
            }
        ]
        mock_splitter_class.return_value = mock_splitter
        
        result = runner.invoke(app, ["validate", "test_dataset"])
        
        assert result.exit_code == 0
        assert "Time Series Cross-Validation Folds:" in result.stdout
        assert "Fold" in result.stdout
        assert "Train Period" in result.stdout
        assert "Test Period" in result.stdout
    
    @patch('mdm.cli.timeseries.MDMClient')
    @patch('mdm.utils.time_series.TimeSeriesSplitter')
    def test_validate_with_options(self, mock_splitter_class, mock_client_class, runner):
        """Test cross-validation with custom options."""
        mock_client = Mock()
        mock_dataset_info = Mock()
        mock_dataset_info.time_column = "date"
        mock_dataset_info.group_column = "group"
        mock_client.get_dataset.return_value = mock_dataset_info
        mock_client.load_dataset_files.return_value = (pd.DataFrame(), None)
        mock_client_class.return_value = mock_client
        
        mock_splitter = Mock()
        mock_splitter.split_by_folds.return_value = []
        mock_splitter_class.return_value = mock_splitter
        
        result = runner.invoke(app, [
            "validate", "test_dataset",
            "--folds", "10",
            "--gap", "7"
        ])
        
        assert result.exit_code == 0
        
        # Verify splitter was created with correct params
        mock_splitter_class.assert_called_with("date", "group")
        
        # Verify split_by_folds was called with correct params
        from unittest.mock import ANY
        mock_splitter.split_by_folds.assert_called_with(
            ANY,  # dataframe
            10,        # n_folds
            7          # gap_days
        )
    
    @patch('mdm.cli.timeseries.MDMClient')
    def test_validate_dataset_not_found(self, mock_client_class, runner):
        """Test validate with non-existent dataset."""
        mock_client = Mock()
        mock_client.get_dataset.return_value = None
        mock_client_class.return_value = mock_client
        
        result = runner.invoke(app, ["validate", "nonexistent"])
        
        assert result.exit_code == 1
        assert "Dataset 'nonexistent' not found" in result.stdout
    
    @patch('mdm.cli.timeseries.MDMClient')
    def test_validate_no_time_column(self, mock_client_class, runner):
        """Test validate with dataset lacking time column."""
        mock_client = Mock()
        mock_dataset_info = Mock()
        mock_dataset_info.time_column = None
        mock_client.get_dataset.return_value = mock_dataset_info
        mock_client_class.return_value = mock_client
        
        result = runner.invoke(app, ["validate", "test_dataset"])
        
        assert result.exit_code == 1
        assert "has no time column configured" in result.stdout
    
    @patch('mdm.cli.timeseries.MDMClient')
    @patch('mdm.utils.time_series.TimeSeriesSplitter')
    def test_validate_with_exception(self, mock_splitter_class, mock_client_class, runner):
        """Test validate with exception."""
        mock_client = Mock()
        mock_dataset_info = Mock()
        mock_dataset_info.time_column = "date"
        mock_dataset_info.group_column = None
        mock_client.get_dataset.return_value = mock_dataset_info
        mock_client.load_dataset_files.return_value = (pd.DataFrame(), None)
        mock_client_class.return_value = mock_client
        
        mock_splitter = Mock()
        mock_splitter.split_by_folds.side_effect = Exception("Validation failed")
        mock_splitter_class.return_value = mock_splitter
        
        result = runner.invoke(app, ["validate", "test_dataset"])
        
        assert result.exit_code == 1
        assert "Validation failed" in result.stdout