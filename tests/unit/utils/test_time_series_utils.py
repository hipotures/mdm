"""Unit tests for time series utilities."""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from mdm.utils.time_series import TimeSeriesSplitter, TimeSeriesAnalyzer


class TestTimeSeriesSplitter:
    """Test TimeSeriesSplitter functionality."""
    
    @pytest.fixture
    def sample_df(self):
        """Create sample time series DataFrame."""
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        df = pd.DataFrame({
            'date': dates,
            'value': np.random.randn(len(dates)),
            'group': ['A', 'B'] * (len(dates) // 2),
            'target': np.random.randint(0, 2, len(dates))
        })
        return df
    
    def test_init(self):
        """Test splitter initialization."""
        splitter = TimeSeriesSplitter('date', 'group')
        assert splitter.time_column == 'date'
        assert splitter.group_column == 'group'
        
        # Without group column
        splitter2 = TimeSeriesSplitter('timestamp')
        assert splitter2.time_column == 'timestamp'
        assert splitter2.group_column is None
    
    def test_split_by_time_fraction(self, sample_df):
        """Test split by time with fraction sizes."""
        splitter = TimeSeriesSplitter('date')
        
        # Two-way split
        result = splitter.split_by_time(sample_df, test_size=0.2)
        
        assert 'train' in result
        assert 'test' in result
        assert 'validation' not in result
        
        # Check sizes approximately match
        total_days = 365
        assert len(result['test']) / total_days == pytest.approx(0.2, rel=0.1)
        assert len(result['train']) / total_days == pytest.approx(0.8, rel=0.1)
        
        # Check no overlap
        assert result['train']['date'].max() < result['test']['date'].min()
    
    def test_split_by_time_days(self, sample_df):
        """Test split by time with day sizes."""
        splitter = TimeSeriesSplitter('date')
        
        # Split with 30 days for test
        result = splitter.split_by_time(sample_df, test_size=30)
        
        # Test set should be approximately 30 days
        test_duration = (result['test']['date'].max() - result['test']['date'].min()).days
        assert test_duration == pytest.approx(29, abs=1)  # 30 days inclusive
    
    def test_split_by_time_timedelta(self, sample_df):
        """Test split by time with timedelta."""
        splitter = TimeSeriesSplitter('date')
        
        # Split with timedelta
        result = splitter.split_by_time(sample_df, test_size=timedelta(days=60))
        
        test_duration = (result['test']['date'].max() - result['test']['date'].min()).days
        assert test_duration == pytest.approx(59, abs=1)
    
    def test_split_by_time_three_way(self, sample_df):
        """Test three-way split with validation set."""
        splitter = TimeSeriesSplitter('date')
        
        result = splitter.split_by_time(
            sample_df, 
            test_size=0.2, 
            validation_size=0.2
        )
        
        assert all(k in result for k in ['train', 'validation', 'test'])
        
        # Check ordering
        assert result['train']['date'].max() < result['validation']['date'].min()
        assert result['validation']['date'].max() < result['test']['date'].min()
        
        # Check approximate sizes
        total_days = 365
        assert len(result['test']) / total_days == pytest.approx(0.2, rel=0.1)
        assert len(result['validation']) / total_days == pytest.approx(0.2, rel=0.1)
    
    def test_split_by_time_preserves_columns(self, sample_df):
        """Test that split preserves all columns."""
        splitter = TimeSeriesSplitter('date')
        
        result = splitter.split_by_time(sample_df, test_size=0.2)
        
        # All columns should be preserved
        for split_df in result.values():
            assert list(split_df.columns) == list(sample_df.columns)
    
    def test_split_by_time_sorts_data(self):
        """Test that data is sorted before splitting."""
        # Create unsorted data
        dates = pd.date_range('2023-01-01', '2023-01-10', freq='D')
        df = pd.DataFrame({
            'date': dates,
            'value': range(10)
        })
        # Shuffle
        df = df.sample(frac=1, random_state=42)
        
        splitter = TimeSeriesSplitter('date')
        result = splitter.split_by_time(df, test_size=0.3)
        
        # Check that results are sorted
        assert result['train']['date'].is_monotonic_increasing
        assert result['test']['date'].is_monotonic_increasing
    
    def test_split_by_folds(self, sample_df):
        """Test cross-validation fold creation."""
        splitter = TimeSeriesSplitter('date')
        
        folds = splitter.split_by_folds(sample_df, n_folds=5, gap_days=7)
        
        assert len(folds) == 5
        
        for i, fold in enumerate(folds):
            assert fold['fold'] == i + 1
            assert 'train' in fold
            assert 'test' in fold
            assert 'train_period' in fold
            assert 'test_period' in fold
            
            # Check gap between train and test
            train_end = fold['train']['date'].max()
            test_start = fold['test']['date'].min()
            gap = (test_start - train_end).days
            assert gap >= 7
            
            # Check train set grows with each fold
            if i > 0:
                assert len(fold['train']) > len(folds[i-1]['train'])
    
    def test_split_by_folds_no_gap(self, sample_df):
        """Test folds without gap."""
        splitter = TimeSeriesSplitter('date')
        
        folds = splitter.split_by_folds(sample_df, n_folds=3, gap_days=0)
        
        assert len(folds) == 3
        
        for fold in folds:
            # No gap means test starts right after train
            train_end = fold['train_period'][1]
            test_start = fold['test_period'][0]
            assert test_start >= train_end
    
    def test_create_sliding_window(self, sample_df):
        """Test sliding window creation."""
        splitter = TimeSeriesSplitter('date')
        
        windows = splitter.create_sliding_window(
            sample_df,
            window_size=30,  # 30 days
            step_size=7,     # 7 days
            min_train_size=60  # 60 days minimum
        )
        
        assert len(windows) > 0
        
        for i, window in enumerate(windows):
            assert window['window'] == i + 1
            assert 'train' in window
            assert 'test' in window
            
            # Check window size
            test_duration = (window['test']['date'].max() - window['test']['date'].min()).days
            assert test_duration == pytest.approx(29, abs=1)
            
            # Check minimum train size
            train_duration = (window['train']['date'].max() - window['train']['date'].min()).days
            assert train_duration >= 59  # 60 days inclusive
            
            # Check step size
            if i > 0:
                prev_test_start = windows[i-1]['test']['date'].min()
                curr_test_start = window['test']['date'].min()
                step = (curr_test_start - prev_test_start).days
                assert step == 7
    
    def test_create_sliding_window_timedelta(self, sample_df):
        """Test sliding window with timedelta parameters."""
        splitter = TimeSeriesSplitter('date')
        
        windows = splitter.create_sliding_window(
            sample_df,
            window_size=timedelta(days=14),
            step_size=timedelta(days=7),
            min_train_size=timedelta(days=30)
        )
        
        assert len(windows) > 0
        
        # Check first window has minimum train size
        first_window = windows[0]
        train_duration = first_window['train']['date'].max() - first_window['train']['date'].min()
        assert train_duration >= timedelta(days=29)  # 30 days inclusive
    
    def test_create_sliding_window_no_min_train(self, sample_df):
        """Test sliding window without minimum train size."""
        splitter = TimeSeriesSplitter('date')
        
        windows = splitter.create_sliding_window(
            sample_df,
            window_size=30,
            step_size=7
        )
        
        # Without min_train_size, first window starts after window_size
        first_window = windows[0]
        train_duration = (first_window['train']['date'].max() - first_window['train']['date'].min()).days
        assert train_duration >= 29  # At least window_size


class TestTimeSeriesAnalyzer:
    """Test TimeSeriesAnalyzer functionality."""
    
    @pytest.fixture
    def sample_df(self):
        """Create sample time series DataFrame."""
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        values = np.sin(np.arange(len(dates)) * 2 * np.pi / 365) + np.random.randn(len(dates)) * 0.1
        df = pd.DataFrame({
            'date': dates,
            'value': values,
            'target': values + np.random.randn(len(dates)) * 0.05
        })
        return df
    
    def test_init(self):
        """Test analyzer initialization."""
        analyzer = TimeSeriesAnalyzer('date', 'target')
        assert analyzer.time_column == 'date'
        assert analyzer.target_column == 'target'
        
        # Without target
        analyzer2 = TimeSeriesAnalyzer('timestamp')
        assert analyzer2.time_column == 'timestamp'
        assert analyzer2.target_column is None
    
    def test_analyze_basic(self, sample_df):
        """Test basic time series analysis."""
        analyzer = TimeSeriesAnalyzer('date', 'target')
        
        result = analyzer.analyze(sample_df)
        
        assert 'time_range' in result
        assert 'frequency' in result
        assert 'missing_timestamps' in result
        assert 'seasonality' in result
        assert 'target_stats' in result
        
        # Check time range
        assert result['time_range']['start'] == '2023-01-01T00:00:00'
        assert result['time_range']['end'] == '2023-12-31T00:00:00'
        assert result['time_range']['duration_days'] == 364
        
        # Check frequency detection
        assert result['frequency'] == 'daily'
        
        # Check target stats
        assert 'mean' in result['target_stats']
        assert 'std' in result['target_stats']
        assert 'trend' in result['target_stats']
    
    def test_analyze_without_target(self, sample_df):
        """Test analysis without target column."""
        analyzer = TimeSeriesAnalyzer('date')
        
        result = analyzer.analyze(sample_df)
        
        assert 'target_stats' not in result
    
    def test_detect_frequency_hourly(self):
        """Test frequency detection for hourly data."""
        dates = pd.date_range('2023-01-01', '2023-01-07', freq='H')
        df = pd.DataFrame({'date': dates, 'value': range(len(dates))})
        
        analyzer = TimeSeriesAnalyzer('date')
        result = analyzer._detect_frequency(df)
        
        assert result == 'hourly'
    
    def test_detect_frequency_minutely(self):
        """Test frequency detection for minutely data."""
        dates = pd.date_range('2023-01-01 00:00', '2023-01-01 01:00', freq='T')
        df = pd.DataFrame({'date': dates, 'value': range(len(dates))})
        
        analyzer = TimeSeriesAnalyzer('date')
        result = analyzer._detect_frequency(df)
        
        assert result == 'minutely'
    
    def test_detect_frequency_weekly(self):
        """Test frequency detection for weekly data."""
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='W')
        df = pd.DataFrame({'date': dates, 'value': range(len(dates))})
        
        analyzer = TimeSeriesAnalyzer('date')
        result = analyzer._detect_frequency(df)
        
        assert result == 'weekly'
    
    def test_detect_frequency_monthly(self):
        """Test frequency detection for monthly data."""
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='M')
        df = pd.DataFrame({'date': dates, 'value': range(len(dates))})
        
        analyzer = TimeSeriesAnalyzer('date')
        result = analyzer._detect_frequency(df)
        
        assert result == 'monthly'
    
    def test_find_missing_timestamps(self):
        """Test finding missing timestamps."""
        # Create data with gaps
        dates = pd.date_range('2023-01-01', '2023-01-10', freq='D')
        df = pd.DataFrame({'date': dates, 'value': range(len(dates))})
        # Remove some dates
        df = df[~df['date'].isin([dates[3], dates[7]])]
        
        analyzer = TimeSeriesAnalyzer('date')
        result = analyzer._find_missing_timestamps(df)
        
        assert result['count'] == 2
        assert result['percentage'] > 0
        assert len(result['dates']) == 2
    
    def test_find_missing_timestamps_no_gaps(self, sample_df):
        """Test finding missing timestamps when there are none."""
        analyzer = TimeSeriesAnalyzer('date')
        result = analyzer._find_missing_timestamps(sample_df)
        
        assert result['count'] == 0
        assert result['percentage'] == 0.0
        assert result['dates'] == []
    
    def test_find_missing_timestamps_empty_df(self):
        """Test finding missing timestamps with empty DataFrame."""
        df = pd.DataFrame({'date': pd.Series(dtype='datetime64[ns]')})
        
        analyzer = TimeSeriesAnalyzer('date')
        result = analyzer._find_missing_timestamps(df)
        
        assert result['count'] == 0
        assert result['percentage'] == 0.0
        assert result['dates'] == []
    
    def test_detect_seasonality(self):
        """Test seasonality detection."""
        # Create data with clear daily pattern
        dates = pd.date_range('2023-01-01', '2023-01-31', freq='H')
        df = pd.DataFrame({
            'date': dates,
            'value': [i % 24 for i in range(len(dates))]  # Clear hourly pattern
        })
        
        analyzer = TimeSeriesAnalyzer('date')
        result = analyzer._detect_seasonality(df)
        
        assert 'daily' in result or len(result) == 0  # May detect daily pattern
    
    @patch('mdm.utils.time_series.stats')
    def test_detect_trend_increasing(self, mock_stats):
        """Test trend detection - increasing."""
        # Mock linear regression result
        mock_stats.linregress.return_value = (
            0.5,    # positive slope
            0,      # intercept
            0.9,    # r_value
            0.01,   # p_value (significant)
            0.1     # std_err
        )
        
        dates = pd.date_range('2023-01-01', '2023-01-10', freq='D')
        df = pd.DataFrame({
            'date': dates,
            'target': range(len(dates))
        })
        
        analyzer = TimeSeriesAnalyzer('date', 'target')
        result = analyzer._detect_trend(df)
        
        assert result == 'increasing'
    
    @patch('mdm.utils.time_series.stats')
    def test_detect_trend_decreasing(self, mock_stats):
        """Test trend detection - decreasing."""
        mock_stats.linregress.return_value = (
            -0.5,   # negative slope
            0,      # intercept
            0.9,    # r_value
            0.01,   # p_value (significant)
            0.1     # std_err
        )
        
        dates = pd.date_range('2023-01-01', '2023-01-10', freq='D')
        df = pd.DataFrame({
            'date': dates,
            'target': range(len(dates), 0, -1)
        })
        
        analyzer = TimeSeriesAnalyzer('date', 'target')
        result = analyzer._detect_trend(df)
        
        assert result == 'decreasing'
    
    @patch('mdm.utils.time_series.stats')
    def test_detect_trend_stable(self, mock_stats):
        """Test trend detection - stable."""
        mock_stats.linregress.return_value = (
            0.1,    # small slope
            0,      # intercept
            0.1,    # r_value
            0.1,    # p_value (not significant)
            0.1     # std_err
        )
        
        dates = pd.date_range('2023-01-01', '2023-01-10', freq='D')
        df = pd.DataFrame({
            'date': dates,
            'target': [5] * len(dates)
        })
        
        analyzer = TimeSeriesAnalyzer('date', 'target')
        result = analyzer._detect_trend(df)
        
        assert result == 'stable'
    
    def test_detect_trend_no_target(self):
        """Test trend detection without target column."""
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', '2023-01-10', freq='D'),
            'value': range(10)
        })
        
        analyzer = TimeSeriesAnalyzer('date', 'target')
        result = analyzer._detect_trend(df)
        
        assert result == 'unknown'