"""Enhanced serialization utilities for MDM."""

import json
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np


class MDMJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy and other special types."""

    def default(self, obj: Any) -> Any:
        """Convert special types to JSON-serializable formats."""
        # NumPy types
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        
        # Datetime types
        elif isinstance(obj, (datetime, date)):
            return obj.isoformat()
        
        # Decimal type
        elif isinstance(obj, Decimal):
            return float(obj)
        
        # Path type
        elif isinstance(obj, Path):
            return str(obj)
        
        # Pandas types
        try:
            import pandas as pd
            
            if isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            elif isinstance(obj, pd.Timedelta):
                return obj.total_seconds()
            elif isinstance(obj, pd.Series):
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict('records')
        except ImportError:
            pass
        
        # Bytes
        if isinstance(obj, bytes):
            return obj.decode('utf-8', errors='replace')
        
        # Sets
        elif isinstance(obj, set):
            return list(obj)
        
        # Default
        return super().default(obj)


def json_dumps(obj: Any, **kwargs) -> str:
    """Serialize object to JSON string with custom encoder."""
    kwargs.setdefault('cls', MDMJSONEncoder)
    kwargs.setdefault('indent', 2)
    kwargs.setdefault('sort_keys', True)
    return json.dumps(obj, **kwargs)


def json_loads(s: str) -> Any:
    """Load JSON string with custom handling."""
    return json.loads(s)


def serialize_for_yaml(obj: Any) -> Any:
    """Convert object to YAML-safe format."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (datetime, date)):
        return obj.isoformat()
    elif isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, bytes):
        return obj.decode('utf-8', errors='replace')
    elif isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, dict):
        return {k: serialize_for_yaml(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [serialize_for_yaml(item) for item in obj]
    
    # Pandas types
    try:
        import pandas as pd
        
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, pd.Timedelta):
            return obj.total_seconds()
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
    except ImportError:
        pass
    
    return obj


def deserialize_datetime(value: str) -> datetime:
    """Deserialize ISO datetime string."""
    return datetime.fromisoformat(value.replace('Z', '+00:00'))


def is_serializable(obj: Any) -> bool:
    """Check if object can be serialized to JSON/YAML."""
    try:
        json_dumps(obj)
        return True
    except (TypeError, ValueError):
        return False