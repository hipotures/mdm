"""Caching layer for frequently accessed data."""
import time
import json
import hashlib
from typing import Dict, Any, Optional, Callable, Union, List
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import pickle
import threading
from functools import wraps

from ..core.logging import get_logger

logger = get_logger(__name__)


class CachePolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"  # Least Recently Used
    TTL = "ttl"  # Time To Live
    FIFO = "fifo"  # First In First Out


@dataclass
class CacheEntry:
    """Single cache entry."""
    key: str
    value: Any
    size: int
    created_at: float
    last_accessed: float
    access_count: int = 0
    ttl: Optional[float] = None
    
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
    
    def access(self) -> None:
        """Record access to this entry."""
        self.last_accessed = time.time()
        self.access_count += 1


class CacheManager:
    """Manages caching for MDM operations."""
    
    def __init__(self, 
                 max_size_mb: int = 100,
                 policy: CachePolicy = CachePolicy.LRU,
                 default_ttl: Optional[float] = None,
                 persist_to_disk: bool = False,
                 cache_dir: Optional[Path] = None):
        """Initialize cache manager.
        
        Args:
            max_size_mb: Maximum cache size in megabytes
            policy: Cache eviction policy
            default_ttl: Default time-to-live in seconds
            persist_to_disk: Whether to persist cache to disk
            cache_dir: Directory for disk cache
        """
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.policy = policy
        self.default_ttl = default_ttl
        self.persist_to_disk = persist_to_disk
        self.cache_dir = cache_dir or Path.home() / ".mdm" / "cache"
        
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._current_size = 0
        
        # Statistics
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_requests': 0
        }
        
        # Load persisted cache if enabled
        if self.persist_to_disk:
            self._load_from_disk()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        with self._lock:
            self._stats['total_requests'] += 1
            
            if key not in self._cache:
                self._stats['misses'] += 1
                return None
            
            entry = self._cache[key]
            
            # Check expiration
            if entry.is_expired():
                self._evict_entry(key)
                self._stats['misses'] += 1
                return None
            
            # Update access info
            entry.access()
            self._stats['hits'] += 1
            
            return entry.value
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (overrides default)
        """
        with self._lock:
            # Calculate size
            size = self._calculate_size(value)
            
            # Check if we need to make room
            if key in self._cache:
                # Remove old entry size
                self._current_size -= self._cache[key].size
            
            # Evict entries if needed
            while self._current_size + size > self.max_size_bytes and self._cache:
                self._evict_oldest()
            
            # Create new entry
            entry = CacheEntry(
                key=key,
                value=value,
                size=size,
                created_at=time.time(),
                last_accessed=time.time(),
                ttl=ttl or self.default_ttl
            )
            
            self._cache[key] = entry
            self._current_size += size
            
            # Persist if enabled
            if self.persist_to_disk:
                self._persist_entry(key, entry)
    
    def delete(self, key: str) -> bool:
        """Delete entry from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if entry was deleted, False if not found
        """
        with self._lock:
            if key in self._cache:
                self._evict_entry(key)
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._current_size = 0
            
            # Clear disk cache
            if self.persist_to_disk and self.cache_dir.exists():
                for file in self.cache_dir.glob("*.cache"):
                    file.unlink()
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate approximate size of value in bytes."""
        if isinstance(value, (str, bytes)):
            return len(value)
        elif isinstance(value, (dict, list)):
            # Rough estimate based on JSON serialization
            return len(json.dumps(value, default=str))
        else:
            # Use pickle for size estimation
            try:
                return len(pickle.dumps(value))
            except:
                return 1000  # Default size
    
    def _evict_oldest(self) -> None:
        """Evict oldest entry based on policy."""
        if not self._cache:
            return
        
        if self.policy == CachePolicy.LRU:
            # Find least recently used
            oldest_key = min(self._cache.keys(), 
                           key=lambda k: self._cache[k].last_accessed)
        elif self.policy == CachePolicy.FIFO:
            # Find oldest created
            oldest_key = min(self._cache.keys(), 
                           key=lambda k: self._cache[k].created_at)
        else:  # TTL
            # Find expired or oldest
            expired = [k for k, v in self._cache.items() if v.is_expired()]
            if expired:
                oldest_key = expired[0]
            else:
                oldest_key = min(self._cache.keys(), 
                               key=lambda k: self._cache[k].created_at)
        
        self._evict_entry(oldest_key)
    
    def _evict_entry(self, key: str) -> None:
        """Evict specific entry."""
        if key in self._cache:
            entry = self._cache[key]
            self._current_size -= entry.size
            del self._cache[key]
            self._stats['evictions'] += 1
            
            # Remove from disk
            if self.persist_to_disk:
                cache_file = self.cache_dir / f"{key}.cache"
                if cache_file.exists():
                    cache_file.unlink()
    
    def _persist_entry(self, key: str, entry: CacheEntry) -> None:
        """Persist cache entry to disk."""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = self.cache_dir / f"{key}.cache"
            
            with open(cache_file, 'wb') as f:
                pickle.dump(entry, f)
        except Exception as e:
            logger.warning(f"Failed to persist cache entry: {e}")
    
    def _load_from_disk(self) -> None:
        """Load cache from disk."""
        if not self.cache_dir.exists():
            return
        
        try:
            for cache_file in self.cache_dir.glob("*.cache"):
                try:
                    with open(cache_file, 'rb') as f:
                        entry = pickle.load(f)
                    
                    # Skip expired entries
                    if not entry.is_expired():
                        self._cache[entry.key] = entry
                        self._current_size += entry.size
                    else:
                        cache_file.unlink()
                        
                except Exception as e:
                    logger.debug(f"Failed to load cache file {cache_file}: {e}")
                    cache_file.unlink()
                    
        except Exception as e:
            logger.warning(f"Failed to load cache from disk: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            hit_rate = 0.0
            if self._stats['total_requests'] > 0:
                hit_rate = self._stats['hits'] / self._stats['total_requests']
            
            return {
                **self._stats,
                'hit_rate': hit_rate,
                'entries': len(self._cache),
                'size_mb': self._current_size / (1024 * 1024),
                'max_size_mb': self.max_size_bytes / (1024 * 1024)
            }
    
    # Decorator for caching function results
    def cached(self, key_func: Optional[Callable] = None, 
               ttl: Optional[float] = None):
        """Decorator for caching function results.
        
        Args:
            key_func: Function to generate cache key from arguments
            ttl: Time-to-live for cached result
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    # Default key generation
                    key_parts = [func.__name__]
                    key_parts.extend(str(arg) for arg in args)
                    key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                    cache_key = hashlib.md5(":".join(key_parts).encode()).hexdigest()
                
                # Check cache
                result = self.get(cache_key)
                if result is not None:
                    return result
                
                # Call function and cache result
                result = func(*args, **kwargs)
                self.set(cache_key, result, ttl=ttl)
                
                return result
            
            return wrapper
        return decorator


# Specialized cache managers for different data types

class DatasetCache(CacheManager):
    """Specialized cache for dataset metadata."""
    
    def __init__(self, max_size_mb: int = 50):
        """Initialize dataset cache."""
        super().__init__(
            max_size_mb=max_size_mb,
            policy=CachePolicy.LRU,
            persist_to_disk=True,
            cache_dir=Path.home() / ".mdm" / "cache" / "datasets"
        )
    
    def get_dataset_info(self, dataset_name: str) -> Optional[Dict[str, Any]]:
        """Get cached dataset info."""
        return self.get(f"dataset_info:{dataset_name}")
    
    def set_dataset_info(self, dataset_name: str, info: Dict[str, Any]) -> None:
        """Cache dataset info."""
        self.set(f"dataset_info:{dataset_name}", info, ttl=3600)  # 1 hour TTL
    
    def get_dataset_stats(self, dataset_name: str) -> Optional[Dict[str, Any]]:
        """Get cached dataset statistics."""
        return self.get(f"dataset_stats:{dataset_name}")
    
    def set_dataset_stats(self, dataset_name: str, stats: Dict[str, Any]) -> None:
        """Cache dataset statistics."""
        self.set(f"dataset_stats:{dataset_name}", stats, ttl=1800)  # 30 min TTL


class QueryResultCache(CacheManager):
    """Specialized cache for query results."""
    
    def __init__(self, max_size_mb: int = 200):
        """Initialize query result cache."""
        super().__init__(
            max_size_mb=max_size_mb,
            policy=CachePolicy.TTL,
            default_ttl=300,  # 5 minutes default
            persist_to_disk=False  # Don't persist query results
        )
    
    def get_query_result(self, query_hash: str) -> Optional[Any]:
        """Get cached query result."""
        return self.get(f"query:{query_hash}")
    
    def set_query_result(self, query_hash: str, result: Any, 
                        ttl: Optional[float] = None) -> None:
        """Cache query result."""
        self.set(f"query:{query_hash}", result, ttl=ttl)


class FeatureCache(CacheManager):
    """Specialized cache for computed features."""
    
    def __init__(self, max_size_mb: int = 100):
        """Initialize feature cache."""
        super().__init__(
            max_size_mb=max_size_mb,
            policy=CachePolicy.LRU,
            persist_to_disk=True,
            cache_dir=Path.home() / ".mdm" / "cache" / "features"
        )
    
    def get_features(self, dataset_name: str, feature_type: str) -> Optional[Any]:
        """Get cached features."""
        return self.get(f"features:{dataset_name}:{feature_type}")
    
    def set_features(self, dataset_name: str, feature_type: str, 
                    features: Any) -> None:
        """Cache computed features."""
        self.set(f"features:{dataset_name}:{feature_type}", features, ttl=7200)  # 2 hours