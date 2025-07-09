"""Unit tests for time series utilities."""

from datetime import datetime, timedelta

import pandas as pd
import pytest

from mdm.utils.time_series import TimeSeriesAnalyzer, TimeSeriesSplitter


class TestTimeSeriesSplitter:
    """Test time series splitting functionality."""

    def create_sample_data(self, n_days=100):
        """Create sample time series data."""
        dates = pd.date_range(start='2024-01-01', periods=n_days, freq='D')
        data = {
            'date': dates,
            'value': range(n_days),
            'group': ['A', 'B'] * (n_days // 2),
        }
        return pd.DataFrame(data)

    def test_split_by_time_fraction(self):
        """Test splitting by time fraction."""
        df = self.create_sample_data(100)
        splitter = TimeSeriesSplitter('date')
        
        # Split with 20% test
        splits = splitter.split_by_time(df, test_size=0.2)
        
        assert 'train' in splits
        assert 'test' in splits
        assert len(splits['train']) == 80
        assert len(splits['test']) == 20
        
        # Ensure chronological order
        assert splits['train']['date'].max() < splits['test']['date'].min()

    def test_split_by_time_days(self):
        """Test splitting by number of days."""
        df = self.create_sample_data(100)
        splitter = TimeSeriesSplitter('date')
        
        # Split with 10 days test
        splits = splitter.split_by_time(df, test_size=10)
        
        # When splitting by days, the boundary day is included, so we get 11 records
        assert len(splits['test']) == 11
        assert len(splits['train']) == 89

    def test_split_with_validation(self):
        """Test three-way split with validation set."""
        df = self.create_sample_data(100)
        splitter = TimeSeriesSplitter('date')
        
        # Split with validation
        splits = splitter.split_by_time(df, test_size=0.2, validation_size=0.1)
        
        assert 'train' in splits
        assert 'validation' in splits
        assert 'test' in splits
        
        # Check approximate sizes (may vary slightly due to date boundaries)
        assert 65 <= len(splits['train']) <= 75
        assert 5 <= len(splits['validation']) <= 15
        assert 15 <= len(splits['test']) <= 25
        
        # Check chronological order
        assert splits['train']['date'].max() < splits['validation']['date'].min()
        assert splits['validation']['date'].max() < splits['test']['date'].min()

    def test_split_by_folds(self):
        """Test creating cross-validation folds."""
        df = self.create_sample_data(100)
        splitter = TimeSeriesSplitter('date')
        
        # Create 5 folds
        folds = splitter.split_by_folds(df, n_folds=5)
        
        assert len(folds) == 5
        
        for i, fold in enumerate(folds):
            assert fold['fold'] == i + 1
            assert 'train' in fold
            assert 'test' in fold
            assert 'train_period' in fold
            assert 'test_period' in fold
            
            # Check that each fold's test is after its train
            assert fold['train']['date'].max() <= fold['test']['date'].min()

    def test_split_with_gap(self):
        """Test splitting with gap between train and test."""
        df = self.create_sample_data(100)
        splitter = TimeSeriesSplitter('date')
        
        # Create folds with 5-day gap
        folds = splitter.split_by_folds(df, n_folds=3, gap_days=5)
        
        for fold in folds:
            train_end = fold['train']['date'].max()
            test_start = fold['test']['date'].min()
            gap = (test_start - train_end).days
            assert gap >= 5

    def test_sliding_window(self):
        """Test sliding window splits."""
        df = self.create_sample_data(100)
        splitter = TimeSeriesSplitter('date')
        
        # Create sliding windows
        windows = splitter.create_sliding_window(
            df,
            window_size=10,  # 10-day test window
            step_size=5,     # 5-day step
            min_train_size=30  # At least 30 days of training
        )
        
        assert len(windows) > 0
        
        for i, window in enumerate(windows):
            assert window['window'] == i + 1
            # Test window should be 10 days
            test_days = (window['test']['date'].max() - window['test']['date'].min()).days + 1
            assert test_days == 10
            # Training set should be at least 30 days
            assert len(window['train']) >= 30


class TestTimeSeriesAnalyzer:
    """Test time series analysis functionality."""

    def test_analyze_daily_data(self):
        """Test analyzing daily time series data."""
        # Create daily data with some patterns
        dates = pd.date_range(start='2024-01-01', periods=365, freq='D')
        data = {
            'date': dates,
            'value': [100 + i + 10 * (i % 7) for i in range(365)],  # Weekly pattern
        }
        df = pd.DataFrame(data)
        
        analyzer = TimeSeriesAnalyzer('date', 'value')
        analysis = analyzer.analyze(df)
        
        assert 'time_range' in analysis
        assert analysis['time_range']['duration_days'] == 364
        assert analysis['frequency'] == 'daily'
        assert 'seasonality' in analysis
        assert 'target_stats' in analysis

    def test_detect_frequency(self):
        """Test frequency detection."""
        analyzer = TimeSeriesAnalyzer('timestamp')
        
        # Hourly data
        hourly_dates = pd.date_range(start='2024-01-01', periods=100, freq='h')
        hourly_df = pd.DataFrame({'timestamp': hourly_dates})
        analysis = analyzer.analyze(hourly_df)
        assert analysis['frequency'] == 'hourly'
        
        # Monthly data
        monthly_dates = pd.date_range(start='2024-01-01', periods=12, freq='ME')
        monthly_df = pd.DataFrame({'timestamp': monthly_dates})
        analysis = analyzer.analyze(monthly_df)
        assert analysis['frequency'] == 'monthly'

    def test_missing_timestamps(self):
        """Test detection of missing timestamps."""
        # Create data with gaps
        dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
        dates_with_gap = dates.delete([3, 4, 7])  # Remove some dates
        
        df = pd.DataFrame({'date': dates_with_gap, 'value': range(len(dates_with_gap))})
        
        analyzer = TimeSeriesAnalyzer('date')
        analysis = analyzer.analyze(df)
        
        missing = analysis['missing_timestamps']
        assert missing['count'] == 3
        assert missing['percentage'] > 0

    def test_trend_detection(self):
        """Test trend detection in target variable."""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        
        # Increasing trend
        df_increasing = pd.DataFrame({
            'date': dates,
            'target': [i * 2 + 10 for i in range(100)]  # Linear increase
        })
        
        analyzer = TimeSeriesAnalyzer('date', 'target')
        analysis = analyzer.analyze(df_increasing)
        assert analysis['target_stats']['trend'] == 'increasing'
        
        # Stable (no trend)
        df_stable = pd.DataFrame({
            'date': dates,
            'target': [50 + (i % 10) for i in range(100)]  # Oscillating around 50
        })
        
        analysis = analyzer.analyze(df_stable)
        assert analysis['target_stats']['trend'] == 'stable'