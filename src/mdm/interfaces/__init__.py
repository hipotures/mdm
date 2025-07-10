"""
Interfaces for MDM components.

These Protocol classes define the contracts that implementations must follow.
"""

from .storage import IStorageBackend

__all__ = ['IStorageBackend']