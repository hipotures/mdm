# Plugin Development Tutorial

## Welcome to MDM Plugin Development! 🚀

This tutorial will guide you through creating custom plugins for MDM's extensible architecture. Whether you're adding new feature transformers, storage backends, or custom validators, this guide covers everything you need to know.

## Table of Contents

1. [Plugin Architecture Overview](#plugin-architecture-overview)
2. [Setting Up Your Development Environment](#setting-up-your-development-environment)
3. [Creating Your First Feature Plugin](#creating-your-first-feature-plugin)
4. [Advanced Plugin Development](#advanced-plugin-development)
5. [Testing Your Plugin](#testing-your-plugin)
6. [Publishing and Distribution](#publishing-and-distribution)
7. [Best Practices](#best-practices)
8. [Plugin Examples](#plugin-examples)

## Plugin Architecture Overview

MDM uses a plugin architecture that allows developers to extend functionality without modifying core code:

```
MDM Core
├── Plugin Manager (Discovery & Loading)
├── Plugin Registry (Registration & Metadata)
├── Plugin Interfaces (Contracts)
└── Plugin Hooks (Extension Points)

Plugin Types:
├── Feature Transformers (Data Engineering)
├── Storage Backends (Custom Databases)
├── Validators (Data Quality)
├── Exporters (Output Formats)
└── Analyzers (Custom Analytics)
```

### Plugin Lifecycle

```python
# 1. Discovery → 2. Loading → 3. Validation → 4. Registration → 5. Execution

class PluginLifecycle:
    """Plugin lifecycle stages"""
    
    DISCOVERED = "discovered"    # Found in plugin directory
    LOADED = "loaded"           # Imported successfully
    VALIDATED = "validated"     # Passed validation checks
    REGISTERED = "registered"   # Added to registry
    ACTIVE = "active"          # Ready for use
    ERROR = "error"            # Failed at some stage
```

## Setting Up Your Development Environment

### 1. Create Plugin Project Structure

```bash
# Create your plugin directory
mkdir mdm-plugin-myfeature
cd mdm-plugin-myfeature

# Create the structure
mkdir -p src/mdm_plugin_myfeature tests docs examples
touch setup.py README.md LICENSE
touch src/mdm_plugin_myfeature/__init__.py
```

### 2. Setup Configuration

```python
# setup.py
from setuptools import setup, find_packages

setup(
    name="mdm-plugin-myfeature",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A custom feature plugin for MDM",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/mdm-plugin-myfeature",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "mdm>=2.0.0",  # Require MDM 2.0+ (refactored version)
        "pandas>=1.3.0",
        "numpy>=1.21.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "mypy>=0.910",
            "ruff>=0.0.261",
        ]
    },
    entry_points={
        "mdm.plugins": [
            "myfeature = mdm_plugin_myfeature:MyFeaturePlugin",
        ],
    },
)
```

### 3. Development Dependencies

```bash
# Install MDM in development mode
git clone https://github.com/your-org/mdm.git
cd mdm
pip install -e .

# Install your plugin in development mode
cd ../mdm-plugin-myfeature
pip install -e ".[dev]"
```

## Creating Your First Feature Plugin

Let's create a feature plugin that generates time-based features:

### 1. Define the Plugin Interface

```python
# src/mdm_plugin_myfeature/__init__.py
from mdm.plugins import FeaturePlugin, PluginMetadata
from mdm.features import FeatureDefinition, FeatureType
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime

class TimeSeriesFeaturePlugin(FeaturePlugin):
    """Generate time series features from datetime columns"""
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="timeseries_features",
            version="0.1.0",
            author="Your Name",
            description="Generate time-based features from datetime columns",
            tags=["time", "datetime", "feature-engineering"],
            requires_columns=["datetime"],  # Requires datetime columns
            produces_features=[
                "hour_of_day", "day_of_week", "month", "quarter",
                "is_weekend", "is_holiday", "time_since_start"
            ]
        )
    
    def discover_features(self, df: pd.DataFrame) -> List[FeatureDefinition]:
        """Discover which features can be generated"""
        features = []
        
        # Find datetime columns
        datetime_cols = df.select_dtypes(include=['datetime64']).columns
        
        for col in datetime_cols:
            # Hour of day
            features.append(FeatureDefinition(
                name=f"{col}_hour",
                description=f"Hour of day from {col}",
                input_columns=[col],
                output_type=FeatureType.NUMERIC,
                tags=["temporal", "hour"]
            ))
            
            # Day of week
            features.append(FeatureDefinition(
                name=f"{col}_dayofweek",
                description=f"Day of week (0=Monday, 6=Sunday) from {col}",
                input_columns=[col],
                output_type=FeatureType.NUMERIC,
                tags=["temporal", "dayofweek"]
            ))
            
            # Is weekend
            features.append(FeatureDefinition(
                name=f"{col}_is_weekend",
                description=f"Whether {col} falls on weekend",
                input_columns=[col],
                output_type=FeatureType.BOOLEAN,
                tags=["temporal", "weekend"]
            ))
            
            # Month
            features.append(FeatureDefinition(
                name=f"{col}_month",
                description=f"Month (1-12) from {col}",
                input_columns=[col],
                output_type=FeatureType.NUMERIC,
                tags=["temporal", "month"]
            ))
            
            # Quarter
            features.append(FeatureDefinition(
                name=f"{col}_quarter",
                description=f"Quarter (1-4) from {col}",
                input_columns=[col],
                output_type=FeatureType.NUMERIC,
                tags=["temporal", "quarter"]
            ))
            
            # Time since start
            features.append(FeatureDefinition(
                name=f"{col}_days_since_start",
                description=f"Days since earliest {col}",
                input_columns=[col],
                output_type=FeatureType.NUMERIC,
                tags=["temporal", "relative"]
            ))
            
        return features
    
    def generate_features(self, 
                         df: pd.DataFrame, 
                         feature_definitions: List[FeatureDefinition],
                         config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Generate the requested features"""
        result = pd.DataFrame(index=df.index)
        
        # Group features by input column for efficiency
        features_by_column = {}
        for feat in feature_definitions:
            col = feat.input_columns[0]
            if col not in features_by_column:
                features_by_column[col] = []
            features_by_column[col].append(feat)
        
        # Generate features for each datetime column
        for col, features in features_by_column.items():
            dt_series = pd.to_datetime(df[col])
            
            for feat in features:
                if feat.name.endswith("_hour"):
                    result[feat.name] = dt_series.dt.hour
                    
                elif feat.name.endswith("_dayofweek"):
                    result[feat.name] = dt_series.dt.dayofweek
                    
                elif feat.name.endswith("_is_weekend"):
                    result[feat.name] = dt_series.dt.dayofweek.isin([5, 6])
                    
                elif feat.name.endswith("_month"):
                    result[feat.name] = dt_series.dt.month
                    
                elif feat.name.endswith("_quarter"):
                    result[feat.name] = dt_series.dt.quarter
                    
                elif feat.name.endswith("_days_since_start"):
                    start_date = dt_series.min()
                    result[feat.name] = (dt_series - start_date).dt.days
        
        return result
    
    def validate_inputs(self, df: pd.DataFrame) -> List[str]:
        """Validate that input data is suitable"""
        errors = []
        
        # Check for datetime columns
        datetime_cols = df.select_dtypes(include=['datetime64']).columns
        if len(datetime_cols) == 0:
            errors.append("No datetime columns found in dataset")
        
        # Check for null values
        for col in datetime_cols:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                errors.append(f"Column {col} has {null_count} null values")
        
        return errors
    
    def estimate_memory_usage(self, 
                            df: pd.DataFrame, 
                            feature_definitions: List[FeatureDefinition]) -> int:
        """Estimate memory usage in bytes"""
        # Each numeric feature uses 8 bytes per row
        # Boolean features use 1 byte per row
        num_rows = len(df)
        memory_bytes = 0
        
        for feat in feature_definitions:
            if feat.output_type == FeatureType.NUMERIC:
                memory_bytes += num_rows * 8
            elif feat.output_type == FeatureType.BOOLEAN:
                memory_bytes += num_rows * 1
                
        return memory_bytes

# Export the plugin class
MyFeaturePlugin = TimeSeriesFeaturePlugin
```

### 2. Add Configuration Support

```python
# src/mdm_plugin_myfeature/config.py
from pydantic import BaseModel, Field
from typing import List, Optional

class TimeSeriesConfig(BaseModel):
    """Configuration for time series feature generation"""
    
    # Holiday detection
    country_code: str = Field(
        default="US",
        description="Country code for holiday detection"
    )
    include_holidays: bool = Field(
        default=True,
        description="Whether to generate holiday features"
    )
    
    # Cyclical encoding
    cyclical_encoding: bool = Field(
        default=False,
        description="Use sine/cosine encoding for cyclical features"
    )
    
    # Custom date ranges
    business_hours_start: int = Field(
        default=9,
        description="Start of business hours (0-23)"
    )
    business_hours_end: int = Field(
        default=17,
        description="End of business hours (0-23)"
    )
    
    # Performance
    chunk_size: int = Field(
        default=10000,
        description="Process data in chunks for memory efficiency"
    )

# Update the plugin to use configuration
class TimeSeriesFeaturePlugin(FeaturePlugin):
    def __init__(self):
        self.config = TimeSeriesConfig()
    
    def configure(self, config: Dict[str, Any]):
        """Update plugin configuration"""
        self.config = TimeSeriesConfig(**config)
    
    def generate_features(self, df, feature_definitions, config=None):
        if config:
            self.configure(config)
            
        # Use configuration in feature generation
        if self.config.cyclical_encoding:
            # Generate cyclical features
            pass
```

### 3. Add Advanced Features

```python
# src/mdm_plugin_myfeature/advanced.py
import holidays
from typing import Set

class AdvancedTimeFeatures:
    """Advanced time-based feature generation"""
    
    @staticmethod
    def get_holidays(country_code: str, years: Set[int]) -> holidays.HolidayBase:
        """Get holiday calendar for country"""
        country_holidays = holidays.CountryHoliday(country_code, years=list(years))
        return country_holidays
    
    @staticmethod
    def is_business_hour(dt_series: pd.Series, start: int, end: int) -> pd.Series:
        """Check if datetime is within business hours"""
        hour = dt_series.dt.hour
        return (hour >= start) & (hour < end) & (~dt_series.dt.dayofweek.isin([5, 6]))
    
    @staticmethod
    def cyclical_encode_hour(hours: pd.Series) -> pd.DataFrame:
        """Encode hour as sine/cosine for cyclical continuity"""
        hours_norm = 2 * np.pi * hours / 24
        return pd.DataFrame({
            'hour_sin': np.sin(hours_norm),
            'hour_cos': np.cos(hours_norm)
        })
    
    @staticmethod
    def cyclical_encode_day_of_week(days: pd.Series) -> pd.DataFrame:
        """Encode day of week cyclically"""
        days_norm = 2 * np.pi * days / 7
        return pd.DataFrame({
            'dow_sin': np.sin(days_norm),
            'dow_cos': np.cos(days_norm)
        })
    
    @staticmethod
    def time_until_next_holiday(dt_series: pd.Series, 
                               country_holidays: holidays.HolidayBase) -> pd.Series:
        """Calculate days until next holiday"""
        result = pd.Series(index=dt_series.index, dtype='float64')
        
        for idx, dt in dt_series.items():
            # Find next holiday
            current_date = dt.date()
            days_ahead = 0
            
            while days_ahead < 365:  # Look up to a year ahead
                check_date = current_date + pd.Timedelta(days=days_ahead)
                if check_date in country_holidays:
                    result[idx] = days_ahead
                    break
                days_ahead += 1
            else:
                result[idx] = np.nan  # No holiday found
                
        return result
```

## Advanced Plugin Development

### 1. Creating a Storage Backend Plugin

```python
# src/mdm_plugin_mongodb/backend.py
from mdm.storage import StorageBackend, StorageCapabilities
from pymongo import MongoClient
import pandas as pd
from typing import Dict, Any, Optional, List

class MongoDBBackend(StorageBackend):
    """MongoDB storage backend plugin"""
    
    @property
    def capabilities(self) -> StorageCapabilities:
        return StorageCapabilities(
            supports_transactions=True,
            supports_schemas=False,
            supports_partitioning=True,
            supports_compression=True,
            supports_concurrent_writes=True,
            max_columns=None,  # No column limit
            max_row_size=16_777_216,  # 16MB document limit
        )
    
    def __init__(self, connection_string: str, database: str, **kwargs):
        super().__init__()
        self.client = MongoClient(connection_string, **kwargs)
        self.db = self.client[database]
        self.metadata_collection = self.db['_metadata']
    
    def create_dataset(self, dataset_name: str, schema: Optional[Dict] = None):
        """Create a new dataset (collection)"""
        # MongoDB creates collections automatically
        # Store metadata
        self.metadata_collection.insert_one({
            '_id': dataset_name,
            'created_at': datetime.utcnow(),
            'schema': schema,
            'version': 1
        })
    
    def write_data(self, dataset_name: str, df: pd.DataFrame, 
                   mode: str = 'append', **kwargs):
        """Write DataFrame to MongoDB"""
        collection = self.db[dataset_name]
        
        # Convert DataFrame to records
        records = df.to_dict('records')
        
        # Add metadata to each record
        for record in records:
            record['_inserted_at'] = datetime.utcnow()
        
        if mode == 'overwrite':
            collection.delete_many({})
        
        # Insert in batches for performance
        batch_size = kwargs.get('batch_size', 1000)
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            collection.insert_many(batch)
    
    def read_data(self, dataset_name: str, 
                  columns: Optional[List[str]] = None,
                  filters: Optional[Dict] = None,
                  limit: Optional[int] = None) -> pd.DataFrame:
        """Read data from MongoDB"""
        collection = self.db[dataset_name]
        
        # Build query
        query = filters or {}
        projection = {col: 1 for col in columns} if columns else None
        
        # Execute query
        cursor = collection.find(query, projection)
        if limit:
            cursor = cursor.limit(limit)
        
        # Convert to DataFrame
        records = list(cursor)
        if records:
            df = pd.DataFrame(records)
            # Remove MongoDB's _id if not requested
            if '_id' in df.columns and columns and '_id' not in columns:
                df = df.drop('_id', axis=1)
            return df
        else:
            return pd.DataFrame()
    
    def get_dataset_stats(self, dataset_name: str) -> Dict[str, Any]:
        """Get dataset statistics"""
        collection = self.db[dataset_name]
        
        stats = {
            'row_count': collection.count_documents({}),
            'size_bytes': self.db.command('collStats', dataset_name)['size'],
            'index_count': len(collection.list_indexes()),
            'average_document_size': self.db.command('collStats', dataset_name).get('avgObjSize', 0)
        }
        
        return stats
    
    def optimize_dataset(self, dataset_name: str):
        """Optimize dataset storage"""
        collection = self.db[dataset_name]
        
        # Compact collection
        self.db.command('compact', dataset_name)
        
        # Create indexes for common queries
        # This would be based on query patterns
        collection.create_index('_inserted_at')
    
    def close(self):
        """Close connection"""
        self.client.close()
```

### 2. Creating a Validator Plugin

```python
# src/mdm_plugin_quality/validators.py
from mdm.plugins import ValidatorPlugin, ValidationResult, ValidationSeverity
import pandas as pd
from typing import List, Dict, Any
import re

class DataQualityValidator(ValidatorPlugin):
    """Comprehensive data quality validation plugin"""
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="data_quality_validator",
            version="0.1.0",
            description="Validate data quality across multiple dimensions"
        )
    
    def validate(self, df: pd.DataFrame, 
                 config: Dict[str, Any]) -> List[ValidationResult]:
        """Run all validations"""
        results = []
        
        # Completeness checks
        results.extend(self._check_completeness(df, config))
        
        # Uniqueness checks
        results.extend(self._check_uniqueness(df, config))
        
        # Validity checks
        results.extend(self._check_validity(df, config))
        
        # Consistency checks
        results.extend(self._check_consistency(df, config))
        
        # Statistical checks
        results.extend(self._check_statistical_properties(df, config))
        
        return results
    
    def _check_completeness(self, df: pd.DataFrame, 
                           config: Dict[str, Any]) -> List[ValidationResult]:
        """Check for missing values"""
        results = []
        
        # Check null percentages
        null_threshold = config.get('null_threshold', 0.05)  # 5% default
        
        for column in df.columns:
            null_ratio = df[column].isnull().sum() / len(df)
            
            if null_ratio > null_threshold:
                results.append(ValidationResult(
                    rule="completeness_check",
                    column=column,
                    severity=ValidationSeverity.WARNING if null_ratio < 0.1 else ValidationSeverity.ERROR,
                    message=f"Column {column} has {null_ratio:.1%} null values (threshold: {null_threshold:.1%})",
                    details={
                        'null_count': int(df[column].isnull().sum()),
                        'null_ratio': float(null_ratio),
                        'threshold': null_threshold
                    }
                ))
        
        return results
    
    def _check_uniqueness(self, df: pd.DataFrame, 
                         config: Dict[str, Any]) -> List[ValidationResult]:
        """Check for duplicate values"""
        results = []
        
        # Check for duplicate rows
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            results.append(ValidationResult(
                rule="duplicate_rows",
                column=None,
                severity=ValidationSeverity.WARNING,
                message=f"Found {duplicates} duplicate rows",
                details={'duplicate_count': int(duplicates)}
            ))
        
        # Check unique constraints
        unique_columns = config.get('unique_columns', [])
        for column in unique_columns:
            if column in df.columns:
                duplicates = df[column].duplicated().sum()
                if duplicates > 0:
                    results.append(ValidationResult(
                        rule="unique_constraint",
                        column=column,
                        severity=ValidationSeverity.ERROR,
                        message=f"Column {column} has {duplicates} duplicate values",
                        details={
                            'duplicate_count': int(duplicates),
                            'duplicate_values': df[df[column].duplicated()][column].value_counts().to_dict()
                        }
                    ))
        
        return results
    
    def _check_validity(self, df: pd.DataFrame, 
                       config: Dict[str, Any]) -> List[ValidationResult]:
        """Check data validity"""
        results = []
        
        # Email validation
        email_columns = config.get('email_columns', [])
        email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        
        for column in email_columns:
            if column in df.columns:
                invalid_emails = df[~df[column].astype(str).str.match(email_pattern)][column]
                if len(invalid_emails) > 0:
                    results.append(ValidationResult(
                        rule="email_format",
                        column=column,
                        severity=ValidationSeverity.ERROR,
                        message=f"Column {column} contains {len(invalid_emails)} invalid email addresses",
                        details={
                            'invalid_count': len(invalid_emails),
                            'sample_invalid': invalid_emails.head(5).tolist()
                        }
                    ))
        
        # Numeric range validation
        numeric_ranges = config.get('numeric_ranges', {})
        for column, (min_val, max_val) in numeric_ranges.items():
            if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
                out_of_range = df[(df[column] < min_val) | (df[column] > max_val)]
                if len(out_of_range) > 0:
                    results.append(ValidationResult(
                        rule="numeric_range",
                        column=column,
                        severity=ValidationSeverity.WARNING,
                        message=f"Column {column} has {len(out_of_range)} values outside range [{min_val}, {max_val}]",
                        details={
                            'out_of_range_count': len(out_of_range),
                            'min_value': float(df[column].min()),
                            'max_value': float(df[column].max()),
                            'expected_range': [min_val, max_val]
                        }
                    ))
        
        return results
    
    def _check_consistency(self, df: pd.DataFrame, 
                          config: Dict[str, Any]) -> List[ValidationResult]:
        """Check data consistency"""
        results = []
        
        # Cross-column validation
        consistency_rules = config.get('consistency_rules', [])
        
        for rule in consistency_rules:
            if rule['type'] == 'date_order':
                start_col = rule['start_column']
                end_col = rule['end_column']
                
                if start_col in df.columns and end_col in df.columns:
                    invalid = df[df[start_col] > df[end_col]]
                    if len(invalid) > 0:
                        results.append(ValidationResult(
                            rule="date_consistency",
                            column=f"{start_col}, {end_col}",
                            severity=ValidationSeverity.ERROR,
                            message=f"{start_col} is after {end_col} in {len(invalid)} rows",
                            details={
                                'invalid_count': len(invalid),
                                'sample_invalid': invalid[[start_col, end_col]].head(5).to_dict('records')
                            }
                        ))
        
        return results
    
    def _check_statistical_properties(self, df: pd.DataFrame, 
                                    config: Dict[str, Any]) -> List[ValidationResult]:
        """Check statistical properties"""
        results = []
        
        # Outlier detection
        outlier_config = config.get('outlier_detection', {})
        
        for column in df.select_dtypes(include=[np.number]).columns:
            if column in outlier_config:
                method = outlier_config[column].get('method', 'iqr')
                
                if method == 'iqr':
                    Q1 = df[column].quantile(0.25)
                    Q3 = df[column].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
                    
                    if len(outliers) > 0:
                        results.append(ValidationResult(
                            rule="outlier_detection",
                            column=column,
                            severity=ValidationSeverity.INFO,
                            message=f"Column {column} has {len(outliers)} potential outliers",
                            details={
                                'outlier_count': len(outliers),
                                'outlier_percentage': len(outliers) / len(df) * 100,
                                'lower_bound': float(lower_bound),
                                'upper_bound': float(upper_bound),
                                'method': method
                            }
                        ))
        
        return results
```

## Testing Your Plugin

### 1. Unit Tests

```python
# tests/test_timeseries_plugin.py
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from mdm_plugin_myfeature import TimeSeriesFeaturePlugin

class TestTimeSeriesPlugin:
    @pytest.fixture
    def sample_data(self):
        """Create sample data with datetime column"""
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        df = pd.DataFrame({
            'date': dates,
            'value': np.random.randn(len(dates)),
            'category': np.random.choice(['A', 'B', 'C'], size=len(dates))
        })
        return df
    
    @pytest.fixture
    def plugin(self):
        """Create plugin instance"""
        return TimeSeriesFeaturePlugin()
    
    def test_discover_features(self, plugin, sample_data):
        """Test feature discovery"""
        features = plugin.discover_features(sample_data)
        
        # Should discover 6 features per datetime column
        assert len(features) == 6
        
        # Check feature names
        feature_names = [f.name for f in features]
        assert 'date_hour' in feature_names
        assert 'date_dayofweek' in feature_names
        assert 'date_is_weekend' in feature_names
    
    def test_generate_hour_feature(self, plugin, sample_data):
        """Test hour feature generation"""
        features = [f for f in plugin.discover_features(sample_data) 
                   if f.name == 'date_hour']
        
        result = plugin.generate_features(sample_data, features)
        
        assert 'date_hour' in result.columns
        assert result['date_hour'].min() >= 0
        assert result['date_hour'].max() <= 23
    
    def test_weekend_detection(self, plugin, sample_data):
        """Test weekend detection"""
        # Create data with known weekend
        weekend_date = pd.Timestamp('2024-01-06')  # Saturday
        weekday_date = pd.Timestamp('2024-01-08')  # Monday
        
        df = pd.DataFrame({
            'date': [weekend_date, weekday_date]
        })
        
        features = [f for f in plugin.discover_features(df) 
                   if f.name == 'date_is_weekend']
        
        result = plugin.generate_features(df, features)
        
        assert result.loc[0, 'date_is_weekend'] == True
        assert result.loc[1, 'date_is_weekend'] == False
    
    def test_validation(self, plugin):
        """Test input validation"""
        # DataFrame without datetime columns
        df = pd.DataFrame({
            'value': [1, 2, 3],
            'category': ['A', 'B', 'C']
        })
        
        errors = plugin.validate_inputs(df)
        assert len(errors) > 0
        assert 'No datetime columns' in errors[0]
    
    def test_memory_estimation(self, plugin, sample_data):
        """Test memory usage estimation"""
        features = plugin.discover_features(sample_data)
        
        memory_bytes = plugin.estimate_memory_usage(sample_data, features)
        
        # 5 numeric features * 8 bytes + 1 boolean * 1 byte per row
        expected_bytes = len(sample_data) * (5 * 8 + 1)
        assert memory_bytes == expected_bytes
    
    @pytest.mark.parametrize("config", [
        {"cyclical_encoding": True},
        {"cyclical_encoding": False},
        {"country_code": "UK", "include_holidays": True}
    ])
    def test_configuration(self, plugin, sample_data, config):
        """Test different configurations"""
        features = plugin.discover_features(sample_data)
        
        # Should not raise exception
        result = plugin.generate_features(sample_data, features, config)
        assert isinstance(result, pd.DataFrame)
```

### 2. Integration Tests

```python
# tests/test_integration.py
import pytest
from mdm.api import MDMClient
from mdm.plugins import PluginManager

class TestPluginIntegration:
    @pytest.fixture
    def client(self):
        """Create MDM client with plugin"""
        client = MDMClient()
        
        # Register plugin
        manager = PluginManager()
        manager.register_plugin('mdm_plugin_myfeature.TimeSeriesFeaturePlugin')
        
        return client
    
    def test_plugin_in_pipeline(self, client, tmp_path):
        """Test plugin in full MDM pipeline"""
        # Create test dataset
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='H'),
            'value': np.random.randn(100)
        })
        
        csv_path = tmp_path / "test_data.csv"
        df.to_csv(csv_path, index=False)
        
        # Register dataset with feature generation
        client.register_dataset(
            name="test_timeseries",
            path=str(csv_path),
            features={
                'plugins': ['timeseries_features'],
                'config': {
                    'timeseries_features': {
                        'cyclical_encoding': True
                    }
                }
            }
        )
        
        # Check generated features
        dataset = client.get_dataset("test_timeseries")
        
        assert 'timestamp_hour' in dataset.columns
        assert 'timestamp_is_weekend' in dataset.columns
```

### 3. Performance Tests

```python
# tests/test_performance.py
import pytest
import time

class TestPluginPerformance:
    @pytest.mark.performance
    def test_large_dataset_performance(self, plugin):
        """Test performance with large dataset"""
        # Create large dataset (1M rows)
        dates = pd.date_range('2020-01-01', periods=1_000_000, freq='min')
        df = pd.DataFrame({
            'timestamp': dates,
            'value': np.random.randn(len(dates))
        })
        
        features = plugin.discover_features(df)
        
        start_time = time.time()
        result = plugin.generate_features(df, features)
        duration = time.time() - start_time
        
        # Should complete within reasonable time
        assert duration < 10.0  # 10 seconds for 1M rows
        assert len(result) == len(df)
    
    @pytest.mark.benchmark
    def test_feature_generation_benchmark(self, benchmark, plugin, sample_data):
        """Benchmark feature generation"""
        features = plugin.discover_features(sample_data)
        
        result = benchmark(plugin.generate_features, sample_data, features)
        
        assert isinstance(result, pd.DataFrame)
```

## Publishing and Distribution

### 1. Package Your Plugin

```bash
# Build distribution
python setup.py sdist bdist_wheel

# Check package
twine check dist/*
```

### 2. Publish to PyPI

```bash
# Test on TestPyPI first
twine upload --repository testpypi dist/*

# Install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ mdm-plugin-myfeature

# Publish to PyPI
twine upload dist/*
```

### 3. Create Documentation

```markdown
# MDM Time Series Feature Plugin

## Installation

```bash
pip install mdm-plugin-myfeature
```

## Configuration

Add to your MDM configuration:

```yaml
plugins:
  enabled:
    - timeseries_features
  
  config:
    timeseries_features:
      country_code: "US"
      include_holidays: true
      cyclical_encoding: false
```

## Usage

The plugin automatically discovers datetime columns and generates:

- Hour of day (0-23)
- Day of week (0=Monday, 6=Sunday)
- Is weekend (boolean)
- Month (1-12)  
- Quarter (1-4)
- Days since start

### Example

```python
from mdm import MDMClient

client = MDMClient()
client.register_dataset(
    name="sales_data",
    path="sales.csv",
    features={
        'plugins': ['timeseries_features']
    }
)
```
```

### 4. Plugin Registry Entry

```yaml
# plugin_registry.yaml
- name: mdm-plugin-myfeature
  type: feature
  author: Your Name
  description: Time series feature generation for MDM
  version: 0.1.0
  url: https://github.com/yourusername/mdm-plugin-myfeature
  tags:
    - time-series
    - feature-engineering
    - datetime
  requirements:
    mdm_version: ">=2.0.0"
    python_version: ">=3.8"
  installation:
    pip: mdm-plugin-myfeature
  documentation: https://mdm-plugin-myfeature.readthedocs.io
  examples:
    - https://github.com/yourusername/mdm-plugin-myfeature/tree/main/examples
```

## Best Practices

### 1. Plugin Design Principles

```python
# ✅ DO: Make plugins focused and composable
class SpecificFeaturePlugin:
    """Do one thing well"""
    pass

# ❌ DON'T: Create monolithic plugins
class EverythingPlugin:
    """Tries to do too much"""
    pass

# ✅ DO: Handle errors gracefully
def generate_features(self, df):
    try:
        return self._generate_features_impl(df)
    except Exception as e:
        logger.error(f"Feature generation failed: {e}")
        return pd.DataFrame()  # Return empty instead of crashing

# ✅ DO: Provide clear configuration
class MyPlugin:
    def __init__(self):
        self.config = PluginConfig(
            param1=Field(description="Clear description"),
            param2=Field(default=10, ge=1, le=100)
        )

# ✅ DO: Document extensively
def my_method(self, df: pd.DataFrame) -> pd.DataFrame:
    """
    Clear description of what this does.
    
    Args:
        df: Input dataframe with expected columns
        
    Returns:
        DataFrame with generated features
        
    Raises:
        ValueError: If required columns are missing
    """
    pass
```

### 2. Performance Guidelines

```python
# ✅ DO: Process data in chunks for memory efficiency
def process_large_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
    chunk_size = self.config.chunk_size
    results = []
    
    for start_idx in range(0, len(df), chunk_size):
        chunk = df.iloc[start_idx:start_idx + chunk_size]
        results.append(self._process_chunk(chunk))
        
    return pd.concat(results, ignore_index=True)

# ✅ DO: Use vectorized operations
def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
    # Good: Vectorized
    df['feature'] = df['col1'] * df['col2']
    
    # Bad: Loop
    # for i in range(len(df)):
    #     df.loc[i, 'feature'] = df.loc[i, 'col1'] * df.loc[i, 'col2']

# ✅ DO: Cache expensive computations
from functools import lru_cache

class MyPlugin:
    @lru_cache(maxsize=128)
    def expensive_computation(self, param: str) -> Any:
        # Expensive operation cached
        pass
```

### 3. Testing Guidelines

```python
# ✅ DO: Test edge cases
def test_empty_dataframe(plugin):
    df = pd.DataFrame()
    result = plugin.generate_features(df, [])
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0

# ✅ DO: Test with realistic data
def test_with_nulls(plugin):
    df = pd.DataFrame({
        'date': [pd.Timestamp('2024-01-01'), pd.NaT, pd.Timestamp('2024-01-03')]
    })
    # Plugin should handle nulls gracefully

# ✅ DO: Property-based testing
from hypothesis import given, strategies as st

@given(df=dataframe_strategy())
def test_property(plugin, df):
    result = plugin.generate_features(df, [])
    # Properties that should always hold
    assert len(result) == len(df)
    assert result.index.equals(df.index)
```

### 4. Documentation Standards

```python
"""
Plugin Module Title
==================

Brief description of what this plugin does.

Features
--------
- Feature 1: Description
- Feature 2: Description

Configuration
------------
.. code-block:: yaml

    plugins:
      myfeature:
        param1: value1
        param2: value2

Examples
--------
Basic usage:

.. code-block:: python

    from mdm_plugin_myfeature import MyFeaturePlugin
    
    plugin = MyFeaturePlugin()
    features = plugin.generate_features(df)

Advanced usage:

.. code-block:: python

    plugin.configure({'param1': 'custom_value'})
    features = plugin.generate_features(df, custom_features)

API Reference
------------
"""
```

## Plugin Examples

### Example 1: Anomaly Detection Plugin

```python
# Anomaly detection using isolation forest
class AnomalyDetectionPlugin(FeaturePlugin):
    def generate_features(self, df: pd.DataFrame, features: List[FeatureDefinition], config: Dict = None) -> pd.DataFrame:
        from sklearn.ensemble import IsolationForest
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return pd.DataFrame(index=df.index)
        
        # Train isolation forest
        iso_forest = IsolationForest(
            contamination=config.get('contamination', 0.1),
            random_state=42
        )
        
        # Fit and predict
        anomaly_scores = iso_forest.fit_predict(df[numeric_cols])
        
        result = pd.DataFrame(index=df.index)
        result['anomaly_score'] = iso_forest.score_samples(df[numeric_cols])
        result['is_anomaly'] = anomaly_scores == -1
        
        return result
```

### Example 2: Text Feature Plugin

```python
# Text analysis and feature extraction
class TextFeaturePlugin(FeaturePlugin):
    def generate_features(self, df: pd.DataFrame, features: List[FeatureDefinition], config: Dict = None) -> pd.DataFrame:
        import nltk
        from textblob import TextBlob
        
        text_cols = df.select_dtypes(include=['object']).columns
        result = pd.DataFrame(index=df.index)
        
        for col in text_cols:
            if df[col].astype(str).str.len().mean() > 10:  # Likely text
                # Basic stats
                result[f'{col}_word_count'] = df[col].astype(str).str.split().str.len()
                result[f'{col}_char_count'] = df[col].astype(str).str.len()
                
                # Sentiment analysis
                sentiments = df[col].astype(str).apply(
                    lambda x: TextBlob(x).sentiment.polarity if x else 0
                )
                result[f'{col}_sentiment'] = sentiments
                
        return result
```

### Example 3: Geospatial Feature Plugin

```python
# Geographic feature extraction
class GeospatialFeaturePlugin(FeaturePlugin):
    def generate_features(self, df: pd.DataFrame, features: List[FeatureDefinition], config: Dict = None) -> pd.DataFrame:
        from geopy.distance import geodesic
        
        result = pd.DataFrame(index=df.index)
        
        # Find lat/lon columns
        lat_cols = [c for c in df.columns if 'lat' in c.lower()]
        lon_cols = [c for c in df.columns if 'lon' in c.lower()]
        
        if lat_cols and lon_cols:
            # Distance from reference point
            ref_point = config.get('reference_point', (0, 0))
            
            for lat_col, lon_col in zip(lat_cols, lon_cols):
                distances = df.apply(
                    lambda row: geodesic(
                        (row[lat_col], row[lon_col]), 
                        ref_point
                    ).km if pd.notna(row[lat_col]) else np.nan,
                    axis=1
                )
                result[f'distance_from_reference_km'] = distances
                
        return result
```

## Troubleshooting

### Common Issues

1. **Plugin not discovered**
   - Check entry_points in setup.py
   - Verify plugin is installed: `pip show mdm-plugin-myfeature`
   - Check MDM plugin discovery: `mdm plugins list`

2. **Import errors**
   - Ensure all dependencies are installed
   - Check Python path: `python -c "import mdm_plugin_myfeature"`

3. **Performance issues**
   - Profile your code: `python -m cProfile -o profile.stats your_script.py`
   - Use chunking for large datasets
   - Consider using Numba or Cython for compute-intensive operations

4. **Configuration not loading**
   - Validate YAML syntax
   - Check configuration schema
   - Enable debug logging: `export MDM_LOG_LEVEL=DEBUG`

## Community Resources

- **MDM Plugin Gallery**: https://github.com/mdm-plugins
- **Plugin Template**: https://github.com/mdm-plugins/template
- **Community Forum**: https://discuss.mdm.io/c/plugins
- **Plugin Development Chat**: https://mdm-slack.com/channels/plugin-dev
- **Video Tutorials**: https://youtube.com/mdm-plugins

## Contributing to Core Plugins

If you've created a generally useful plugin, consider contributing it to the core MDM plugins:

1. Fork the MDM repository
2. Add your plugin to `src/mdm/plugins/contrib/`
3. Add tests to `tests/plugins/`
4. Update documentation
5. Submit a pull request

We welcome contributions that enhance MDM's capabilities!