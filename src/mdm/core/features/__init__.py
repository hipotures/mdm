"""
New feature engineering implementation.

This package provides the refactored feature engineering system with:
- Clean separation of concerns
- Plugin-based architecture
- Better performance and memory efficiency
- Comprehensive type support
"""

from .generator import NewFeatureGenerator
from .base import FeatureTransformer, TransformerRegistry
from .transformers import (
    NumericTransformer,
    CategoricalTransformer,
    DatetimeTransformer,
    TextTransformer,
    InteractionTransformer,
)

__all__ = [
    'NewFeatureGenerator',
    'FeatureTransformer',
    'TransformerRegistry',
    'NumericTransformer',
    'CategoricalTransformer',
    'DatetimeTransformer',
    'TextTransformer',
    'InteractionTransformer',
]