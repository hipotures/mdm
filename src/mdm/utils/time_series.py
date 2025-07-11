"""Time series utilities for MDM."""

from loguru import logger
from datetime import timedelta
from typing import Optional, Union, List

import pandas as pd
from scipy import stats



class TimeSeriesSplitter:
    """Split time series data for training and validation."""

    def __init__(self, time_column: str, group_column: Optional[str] = None):
        """Initialize time series splitter.
        
        Args:
            time_column: Name of the time column
            group_column: Optional column for grouped time series
        """
        self.time_column = time_column
        self.group_column = group_column

    def split_by_time(
        self,
        df: pd.DataFrame,
        test_size: Union[float, int, timedelta],
        validation_size: Optional[Union[float, int, timedelta]] = None,
    ) -> dict[str, pd.DataFrame]:
        """Split data by time.
        
        Args:
            df: DataFrame to split
            test_size: Size of test set (fraction, days, or timedelta)
            validation_size: Size of validation set
            
        Returns:
            Dictionary with train, validation (optional), and test DataFrames
        """
        # Ensure time column is datetime
        df = df.copy()
        df[self.time_column] = pd.to_datetime(df[self.time_column])

        # Sort by time
        df = df.sort_values(self.time_column)

        # Get time range
        time_min = df[self.time_column].min()
        time_max = df[self.time_column].max()
        time_range = time_max - time_min

        # Calculate split points
        if isinstance(test_size, float):
            # Fraction of time range
            test_duration = time_range * test_size
            test_start = time_max - test_duration
        elif isinstance(test_size, int):
            # Number of days
            test_start = time_max - timedelta(days=test_size)
        else:
            # Timedelta
            test_start = time_max - test_size

        if validation_size is not None:
            if isinstance(validation_size, float):
                val_duration = time_range * validation_size
                val_start = test_start - val_duration
            elif isinstance(validation_size, int):
                val_start = test_start - timedelta(days=validation_size)
            else:
                val_start = test_start - validation_size

            # Three-way split
            train = df[df[self.time_column] < val_start]
            validation = df[(df[self.time_column] >= val_start) & (df[self.time_column] < test_start)]
            test = df[df[self.time_column] >= test_start]

            return {
                'train': train,
                'validation': validation,
                'test': test
            }
        # Two-way split
        train = df[df[self.time_column] < test_start]
        test = df[df[self.time_column] >= test_start]

        return {
            'train': train,
            'test': test
        }

    def split_by_folds(
        self,
        df: pd.DataFrame,
        n_folds: int = 5,
        gap_days: int = 0,
    ) -> list[dict[str, pd.DataFrame]]:
        """Create time-based cross-validation folds.
        
        Args:
            df: DataFrame to split
            n_folds: Number of folds
            gap_days: Gap between train and test in days
            
        Returns:
            List of fold dictionaries with train and test DataFrames
        """
        # Ensure time column is datetime
        df = df.copy()
        df[self.time_column] = pd.to_datetime(df[self.time_column])

        # Sort by time
        df = df.sort_values(self.time_column)

        # Get time range
        time_min = df[self.time_column].min()
        time_max = df[self.time_column].max()
        time_range = time_max - time_min

        # Calculate fold size
        fold_duration = time_range / (n_folds + 1)
        gap = timedelta(days=gap_days)

        folds = []
        for i in range(n_folds):
            # Train end time
            train_end = time_min + fold_duration * (i + 1)

            # Test start time (with gap)
            test_start = train_end + gap
            test_end = test_start + fold_duration

            # Create fold
            train = df[df[self.time_column] <= train_end]
            test = df[(df[self.time_column] > test_start) & (df[self.time_column] <= test_end)]

            if len(train) > 0 and len(test) > 0:
                folds.append({
                    'fold': i + 1,
                    'train': train,
                    'test': test,
                    'train_period': (time_min, train_end),
                    'test_period': (test_start, test_end)
                })

        return folds
    
    def split(
        self,
        df: pd.DataFrame,
        n_splits: int = 5,
        test_size: Union[float, int] = 0.2,
        gap: int = 0,
        strategy: str = "expanding"
    ) -> List[tuple[pd.DataFrame, pd.DataFrame]]:
        """Split time series data for cross-validation.
        
        Args:
            df: DataFrame to split
            n_splits: Number of splits
            test_size: Test set size (fraction if float, days if int)
            gap: Gap between train and test in days
            strategy: Split strategy ('expanding' or 'sliding')
            
        Returns:
            List of (train, test) DataFrame tuples
        """
        # Ensure time column is datetime
        df = df.copy()
        df[self.time_column] = pd.to_datetime(df[self.time_column])
        df = df.sort_values(self.time_column)
        
        # Get time range
        time_min = df[self.time_column].min()
        time_max = df[self.time_column].max()
        time_range = time_max - time_min
        
        # Convert test_size to timedelta if needed
        if isinstance(test_size, float):
            test_duration = time_range * test_size / n_splits
        else:
            test_duration = timedelta(days=test_size)
            
        gap_duration = timedelta(days=gap)
        
        splits = []
        
        if strategy == "expanding":
            # Expanding window: train set grows, test set moves forward
            total_duration = time_range
            step_size = total_duration / n_splits
            
            for i in range(n_splits):
                # Test window end time
                test_end = time_min + step_size * (i + 1)
                test_start = test_end - test_duration
                
                # Train window (expanding)
                train_end = test_start - gap_duration
                train_start = time_min
                
                # Create split
                train_df = df[(df[self.time_column] >= train_start) & 
                              (df[self.time_column] < train_end)]
                test_df = df[(df[self.time_column] >= test_start) & 
                             (df[self.time_column] < test_end)]
                
                if len(train_df) > 0 and len(test_df) > 0:
                    splits.append((train_df, test_df))
                    
        elif strategy == "sliding":
            # Sliding window: both train and test windows move forward
            total_duration = time_range
            step_size = total_duration / n_splits
            
            # Fixed train window size (use remaining data after test and gap)
            train_duration = (total_duration - test_duration - gap_duration) / 2
            
            for i in range(n_splits):
                # Calculate window positions
                window_end = time_min + step_size * (i + 1)
                test_end = window_end
                test_start = test_end - test_duration
                train_end = test_start - gap_duration
                train_start = train_end - train_duration
                
                # Ensure train_start is not before data start
                if train_start < time_min:
                    train_start = time_min
                
                # Create split
                train_df = df[(df[self.time_column] >= train_start) & 
                              (df[self.time_column] < train_end)]
                test_df = df[(df[self.time_column] >= test_start) & 
                             (df[self.time_column] < test_end)]
                
                if len(train_df) > 0 and len(test_df) > 0:
                    splits.append((train_df, test_df))
                    
        else:
            raise ValueError(f"Unknown strategy: {strategy}. Use 'expanding' or 'sliding'")
            
        return splits

    def create_sliding_window(
        self,
        df: pd.DataFrame,
        window_size: Union[int, timedelta],
        step_size: Union[int, timedelta],
        min_train_size: Optional[Union[int, timedelta]] = None,
    ) -> list[dict[str, pd.DataFrame]]:
        """Create sliding window splits.
        
        Args:
            df: DataFrame to split
            window_size: Size of test window
            step_size: Step between windows
            min_train_size: Minimum training set size
            
        Returns:
            List of window dictionaries
        """
        # Ensure time column is datetime
        df = df.copy()
        df[self.time_column] = pd.to_datetime(df[self.time_column])

        # Sort by time
        df = df.sort_values(self.time_column)

        # Get time range
        time_min = df[self.time_column].min()
        time_max = df[self.time_column].max()

        # Convert sizes to timedelta if needed
        if isinstance(window_size, int):
            window_size = timedelta(days=window_size)
        if isinstance(step_size, int):
            step_size = timedelta(days=step_size)
        if min_train_size is not None and isinstance(min_train_size, int):
            min_train_size = timedelta(days=min_train_size)

        windows = []
        current_end = time_min + (min_train_size or window_size)

        while current_end + window_size <= time_max:
            # Define window
            test_start = current_end
            test_end = test_start + window_size

            # Create split
            train = df[df[self.time_column] < test_start]
            test = df[(df[self.time_column] >= test_start) & (df[self.time_column] < test_end)]

            if len(train) > 0 and len(test) > 0:
                windows.append({
                    'window': len(windows) + 1,
                    'train': train,
                    'test': test,
                    'train_period': (time_min, test_start),
                    'test_period': (test_start, test_end)
                })

            # Move to next window
            current_end += step_size

        return windows


class TimeSeriesAnalyzer:
    """Analyze time series data."""

    def __init__(self, time_column: str, target_column: Optional[str] = None):
        """Initialize analyzer.
        
        Args:
            time_column: Name of time column
            target_column: Optional target column for analysis
        """
        self.time_column = time_column
        self.target_column = target_column

    def analyze(self, df: pd.DataFrame) -> dict:
        """Analyze time series data.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Analysis results
        """
        df = df.copy()
        df[self.time_column] = pd.to_datetime(df[self.time_column])

        analysis = {
            'time_range': {
                'start': df[self.time_column].min().isoformat(),
                'end': df[self.time_column].max().isoformat(),
                'duration_days': (df[self.time_column].max() - df[self.time_column].min()).days
            },
            'frequency': self._detect_frequency(df),
            'missing_timestamps': self._find_missing_timestamps(df),
            'seasonality': self._detect_seasonality(df)
        }

        if self.target_column and self.target_column in df.columns:
            analysis['target_stats'] = {
                'mean': df[self.target_column].mean(),
                'std': df[self.target_column].std(),
                'trend': self._detect_trend(df)
            }

        return analysis

    def _detect_frequency(self, df: pd.DataFrame) -> str:
        """Detect time series frequency."""
        # Calculate time differences
        time_diffs = df[self.time_column].diff().dropna()

        # Get mode of time differences
        mode_diff = time_diffs.mode()[0]

        # Map to frequency string
        if mode_diff <= timedelta(minutes=1):
            return 'minutely'
        if mode_diff <= timedelta(hours=1):
            return 'hourly'
        if mode_diff <= timedelta(days=1):
            return 'daily'
        if mode_diff <= timedelta(days=7):
            return 'weekly'
        if mode_diff <= timedelta(days=31):
            return 'monthly'
        if mode_diff <= timedelta(days=92):
            return 'quarterly'
        return 'yearly'

    def _find_missing_timestamps(self, df: pd.DataFrame) -> dict:
        """Find missing timestamps in time series."""
        # Try to infer frequency from the data
        time_diffs = df[self.time_column].diff().dropna()
        
        if len(time_diffs) == 0:
            return {'count': 0, 'percentage': 0.0, 'dates': []}
            
        # Get mode of time differences
        mode_diff = time_diffs.mode()[0]
        
        # Create expected range based on mode
        expected_range = pd.date_range(
            start=df[self.time_column].min(),
            end=df[self.time_column].max(),
            freq=mode_diff
        )

        missing = expected_range.difference(df[self.time_column])

        return {
            'count': len(missing),
            'percentage': len(missing) / len(expected_range) * 100,
            'dates': [d.isoformat() for d in missing[:10]]  # First 10
        }

    def _detect_seasonality(self, df: pd.DataFrame) -> dict:
        """Detect seasonality patterns."""
        df = df.set_index(self.time_column).sort_index()

        seasonality = {}

        # Check for daily patterns
        if len(df) > 24:
            hourly_mean = df.groupby(df.index.hour).size()
            if hourly_mean.std() / hourly_mean.mean() > 0.1:
                seasonality['daily'] = True

        # Check for weekly patterns
        if len(df) > 7:
            daily_mean = df.groupby(df.index.dayofweek).size()
            if daily_mean.std() / daily_mean.mean() > 0.1:
                seasonality['weekly'] = True

        # Check for monthly patterns
        if len(df) > 30:
            monthly_mean = df.groupby(df.index.day).size()
            if monthly_mean.std() / monthly_mean.mean() > 0.1:
                seasonality['monthly'] = True

        return seasonality

    def _detect_trend(self, df: pd.DataFrame) -> str:
        """Detect trend in target variable."""
        if self.target_column not in df.columns:
            return 'unknown'

        # Simple linear regression
        from scipy import stats

        df = df.sort_values(self.time_column)
        x = range(len(df))
        y = df[self.target_column].values

        slope, _, r_value, p_value, _ = stats.linregress(x, y)

        if p_value < 0.05:  # Significant trend
            if slope > 0:
                return 'increasing'
            return 'decreasing'
        return 'stable'
