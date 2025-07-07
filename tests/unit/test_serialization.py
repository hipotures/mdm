"""Unit tests for serialization utilities."""

import json
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mdm.utils.serialization import (
    MDMJSONEncoder,
    deserialize_datetime,
    is_serializable,
    json_dumps,
    json_loads,
    serialize_for_yaml,
)


class TestMDMJSONEncoder:
    """Test custom JSON encoder."""

    def test_numpy_types(self):
        """Test encoding NumPy types."""
        data = {
            "int64": np.int64(42),
            "float64": np.float64(3.14),
            "bool": np.bool_(True),
            "array": np.array([1, 2, 3]),
        }
        
        result = json.dumps(data, cls=MDMJSONEncoder)
        decoded = json.loads(result)
        
        assert decoded["int64"] == 42
        assert decoded["float64"] == 3.14
        assert decoded["bool"] is True
        assert decoded["array"] == [1, 2, 3]

    def test_datetime_types(self):
        """Test encoding datetime types."""
        now = datetime.now()
        today = date.today()
        
        data = {
            "datetime": now,
            "date": today,
        }
        
        result = json.dumps(data, cls=MDMJSONEncoder)
        decoded = json.loads(result)
        
        assert decoded["datetime"] == now.isoformat()
        assert decoded["date"] == today.isoformat()

    def test_pandas_types(self):
        """Test encoding pandas types."""
        data = {
            "timestamp": pd.Timestamp("2024-01-01"),
            "series": pd.Series([1, 2, 3]),
            "dataframe": pd.DataFrame({"a": [1, 2], "b": [3, 4]}),
        }
        
        result = json.dumps(data, cls=MDMJSONEncoder)
        decoded = json.loads(result)
        
        assert decoded["timestamp"] == "2024-01-01T00:00:00"
        assert decoded["series"] == [1, 2, 3]
        assert decoded["dataframe"] == [{"a": 1, "b": 3}, {"a": 2, "b": 4}]

    def test_other_types(self):
        """Test encoding other special types."""
        data = {
            "decimal": Decimal("123.45"),
            "path": Path("/home/user/data"),
            "set": {1, 2, 3},
            "bytes": b"hello",
        }
        
        result = json.dumps(data, cls=MDMJSONEncoder)
        decoded = json.loads(result)
        
        assert decoded["decimal"] == 123.45
        assert decoded["path"] == "/home/user/data"
        assert set(decoded["set"]) == {1, 2, 3}
        assert decoded["bytes"] == "hello"


class TestSerializeForYAML:
    """Test YAML serialization function."""

    def test_nested_structures(self):
        """Test serializing nested structures."""
        data = {
            "metrics": {
                "accuracy": np.float64(0.95),
                "samples": np.int32(1000),
            },
            "arrays": {
                "predictions": np.array([1, 2, 3]),
                "features": [np.int64(10), np.float32(20.5)],
            },
            "metadata": {
                "created": datetime(2024, 1, 1),
                "path": Path("/data/model"),
            },
        }
        
        result = serialize_for_yaml(data)
        
        assert isinstance(result["metrics"]["accuracy"], float)
        assert isinstance(result["metrics"]["samples"], int)
        assert result["arrays"]["predictions"] == [1, 2, 3]
        assert result["arrays"]["features"] == [10, 20.5]
        assert result["metadata"]["created"] == "2024-01-01T00:00:00"
        assert result["metadata"]["path"] == "/data/model"

    def test_pandas_dataframe(self):
        """Test serializing pandas DataFrame."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "value": [10.5, 20.5, 30.5],
            "category": ["A", "B", "A"],
        })
        
        result = serialize_for_yaml(df)
        
        assert isinstance(result, list)
        assert len(result) == 3
        assert result[0] == {"id": 1, "value": 10.5, "category": "A"}


class TestHelperFunctions:
    """Test helper functions."""

    def test_json_dumps_loads(self):
        """Test json_dumps and json_loads functions."""
        data = {
            "number": np.int64(42),
            "array": np.array([1, 2, 3]),
            "date": datetime.now(),
        }
        
        json_str = json_dumps(data)
        loaded = json_loads(json_str)
        
        assert loaded["number"] == 42
        assert loaded["array"] == [1, 2, 3]
        assert isinstance(loaded["date"], str)

    def test_deserialize_datetime(self):
        """Test datetime deserialization."""
        # ISO format
        dt = deserialize_datetime("2024-01-01T12:00:00")
        assert dt.year == 2024
        assert dt.month == 1
        assert dt.day == 1
        assert dt.hour == 12
        
        # With timezone
        dt_tz = deserialize_datetime("2024-01-01T12:00:00Z")
        assert dt_tz.tzinfo is not None

    def test_is_serializable(self):
        """Test serializable check."""
        # Serializable objects
        assert is_serializable({"a": 1, "b": [2, 3]})
        assert is_serializable([1, 2, 3])
        assert is_serializable("string")
        assert is_serializable(123)
        assert is_serializable(np.int64(42))
        assert is_serializable(datetime.now())
        
        # Non-serializable objects (without our encoder)
        class CustomClass:
            pass
        
        # This would fail with standard JSON encoder but passes with ours
        assert is_serializable(np.array([1, 2, 3]))  # Our encoder handles this