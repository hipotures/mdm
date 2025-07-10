"""Batch processing optimizations."""
import time
from typing import List, Dict, Any, Callable, Optional, Tuple, Iterator
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue, Empty
import pandas as pd
import numpy as np

from ..core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    batch_size: int = 10000
    max_workers: int = 4
    chunk_memory_limit_mb: int = 100
    enable_parallel: bool = True
    prefetch_batches: int = 2
    
    def validate(self) -> None:
        """Validate configuration."""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.max_workers <= 0:
            raise ValueError("max_workers must be positive")
        if self.chunk_memory_limit_mb <= 0:
            raise ValueError("chunk_memory_limit_mb must be positive")


class BatchOptimizer:
    """Optimizes batch processing operations."""
    
    def __init__(self, config: Optional[BatchConfig] = None):
        """Initialize batch optimizer.
        
        Args:
            config: Batch processing configuration
        """
        self.config = config or BatchConfig()
        self.config.validate()
        
        self._stats = {
            'total_batches': 0,
            'processed_rows': 0,
            'processing_time': 0.0,
            'parallel_speedup': 0.0
        }
    
    def process_dataframe_batches(self, 
                                 df: pd.DataFrame,
                                 process_func: Callable[[pd.DataFrame], pd.DataFrame],
                                 progress_callback: Optional[Callable[[int, int], None]] = None
                                ) -> pd.DataFrame:
        """Process DataFrame in optimized batches.
        
        Args:
            df: Input DataFrame
            process_func: Function to process each batch
            progress_callback: Optional callback for progress updates
            
        Returns:
            Processed DataFrame
        """
        start_time = time.time()
        total_rows = len(df)
        
        if total_rows <= self.config.batch_size or not self.config.enable_parallel:
            # Process single batch
            result = process_func(df)
            self._update_stats(1, total_rows, time.time() - start_time)
            return result
        
        # Split into batches
        batches = self._create_batches(df)
        
        if self.config.enable_parallel:
            result = self._process_parallel(batches, process_func, progress_callback)
        else:
            result = self._process_sequential(batches, process_func, progress_callback)
        
        self._update_stats(len(batches), total_rows, time.time() - start_time)
        return result
    
    def optimize_batch_inserts(self,
                              data: List[Dict[str, Any]],
                              insert_func: Callable[[List[Dict[str, Any]]], None],
                              batch_size: Optional[int] = None) -> None:
        """Optimize batch insert operations.
        
        Args:
            data: List of records to insert
            insert_func: Function to insert a batch of records
            batch_size: Optional batch size override
        """
        batch_size = batch_size or self.config.batch_size
        
        # Split data into batches
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            insert_func(batch)
            
            logger.debug(f"Inserted batch {i//batch_size + 1} ({len(batch)} records)")
    
    def stream_process(self,
                      data_generator: Iterator[Any],
                      process_func: Callable[[List[Any]], List[Any]],
                      output_func: Callable[[List[Any]], None],
                      batch_timeout: float = 1.0) -> None:
        """Process streaming data in optimized batches.
        
        Args:
            data_generator: Generator yielding data items
            process_func: Function to process a batch
            output_func: Function to handle processed batch
            batch_timeout: Timeout for incomplete batches
        """
        batch = []
        last_batch_time = time.time()
        
        for item in data_generator:
            batch.append(item)
            
            # Check if batch is ready
            if len(batch) >= self.config.batch_size or \
               (time.time() - last_batch_time) > batch_timeout:
                
                if batch:
                    # Process and output batch
                    processed = process_func(batch)
                    output_func(processed)
                    
                    # Reset batch
                    batch = []
                    last_batch_time = time.time()
        
        # Process remaining items
        if batch:
            processed = process_func(batch)
            output_func(processed)
    
    def parallel_map(self,
                    func: Callable[[Any], Any],
                    items: List[Any],
                    chunk_size: Optional[int] = None) -> List[Any]:
        """Parallel map operation with automatic chunking.
        
        Args:
            func: Function to apply to each item
            items: List of items to process
            chunk_size: Optional chunk size override
            
        Returns:
            List of results in original order
        """
        if not self.config.enable_parallel or len(items) < self.config.batch_size:
            return [func(item) for item in items]
        
        chunk_size = chunk_size or max(1, len(items) // (self.config.max_workers * 4))
        results = [None] * len(items)
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit chunks
            futures = {}
            for i in range(0, len(items), chunk_size):
                chunk = items[i:i + chunk_size]
                future = executor.submit(self._process_chunk, func, chunk)
                futures[future] = i
            
            # Collect results
            for future in as_completed(futures):
                start_idx = futures[future]
                chunk_results = future.result()
                for j, result in enumerate(chunk_results):
                    results[start_idx + j] = result
        
        return results
    
    def _create_batches(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        """Create optimized batches from DataFrame."""
        batches = []
        
        # Estimate memory per row
        memory_per_row = df.memory_usage(deep=True).sum() / len(df)
        
        # Calculate optimal batch size based on memory limit
        optimal_batch_size = min(
            self.config.batch_size,
            int(self.config.chunk_memory_limit_mb * 1024 * 1024 / memory_per_row)
        )
        
        # Create batches
        for i in range(0, len(df), optimal_batch_size):
            batch = df.iloc[i:i + optimal_batch_size]
            batches.append(batch)
        
        logger.debug(f"Created {len(batches)} batches of size ~{optimal_batch_size}")
        return batches
    
    def _process_parallel(self,
                         batches: List[pd.DataFrame],
                         process_func: Callable[[pd.DataFrame], pd.DataFrame],
                         progress_callback: Optional[Callable[[int, int], None]]
                        ) -> pd.DataFrame:
        """Process batches in parallel."""
        results = [None] * len(batches)
        total_batches = len(batches)
        completed = 0
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all batches
            futures = {
                executor.submit(process_func, batch): i 
                for i, batch in enumerate(batches)
            }
            
            # Collect results
            for future in as_completed(futures):
                batch_idx = futures[future]
                results[batch_idx] = future.result()
                
                completed += 1
                if progress_callback:
                    progress_callback(completed, total_batches)
        
        # Concatenate results
        return pd.concat(results, ignore_index=True)
    
    def _process_sequential(self,
                           batches: List[pd.DataFrame],
                           process_func: Callable[[pd.DataFrame], pd.DataFrame],
                           progress_callback: Optional[Callable[[int, int], None]]
                          ) -> pd.DataFrame:
        """Process batches sequentially."""
        results = []
        
        for i, batch in enumerate(batches):
            result = process_func(batch)
            results.append(result)
            
            if progress_callback:
                progress_callback(i + 1, len(batches))
        
        return pd.concat(results, ignore_index=True)
    
    def _process_chunk(self, func: Callable[[Any], Any], 
                      chunk: List[Any]) -> List[Any]:
        """Process a chunk of items."""
        return [func(item) for item in chunk]
    
    def _update_stats(self, batches: int, rows: int, time_taken: float) -> None:
        """Update processing statistics."""
        self._stats['total_batches'] += batches
        self._stats['processed_rows'] += rows
        self._stats['processing_time'] += time_taken
        
        # Calculate speedup
        if self.config.enable_parallel and batches > 1:
            theoretical_sequential_time = time_taken * self.config.max_workers
            self._stats['parallel_speedup'] = theoretical_sequential_time / time_taken
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get optimizer statistics."""
        avg_batch_time = 0.0
        if self._stats['total_batches'] > 0:
            avg_batch_time = self._stats['processing_time'] / self._stats['total_batches']
        
        return {
            **self._stats,
            'average_batch_time': avg_batch_time,
            'rows_per_second': self._stats['processed_rows'] / max(0.001, self._stats['processing_time'])
        }


class BatchQueue:
    """Queue for batch processing with prefetching."""
    
    def __init__(self, 
                 data_source: Iterator[Any],
                 batch_size: int = 1000,
                 prefetch_batches: int = 2):
        """Initialize batch queue.
        
        Args:
            data_source: Iterator providing data
            batch_size: Size of each batch
            prefetch_batches: Number of batches to prefetch
        """
        self.data_source = data_source
        self.batch_size = batch_size
        self.prefetch_batches = prefetch_batches
        
        self._queue = Queue(maxsize=prefetch_batches)
        self._exhausted = False
        self._prefetch_thread = None
        self._stop_event = threading.Event()
    
    def __enter__(self):
        """Start prefetching."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop prefetching."""
        self.stop()
    
    def start(self) -> None:
        """Start prefetching batches."""
        self._prefetch_thread = threading.Thread(target=self._prefetch_worker)
        self._prefetch_thread.daemon = True
        self._prefetch_thread.start()
    
    def stop(self) -> None:
        """Stop prefetching."""
        self._stop_event.set()
        if self._prefetch_thread:
            self._prefetch_thread.join()
    
    def get_batch(self, timeout: float = 1.0) -> Optional[List[Any]]:
        """Get next batch.
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            Batch of items or None if no more data
        """
        if self._exhausted and self._queue.empty():
            return None
        
        try:
            return self._queue.get(timeout=timeout)
        except Empty:
            return None if self._exhausted else []
    
    def _prefetch_worker(self) -> None:
        """Worker thread for prefetching batches."""
        batch = []
        
        try:
            for item in self.data_source:
                if self._stop_event.is_set():
                    break
                
                batch.append(item)
                
                if len(batch) >= self.batch_size:
                    self._queue.put(batch)
                    batch = []
            
            # Put remaining items
            if batch:
                self._queue.put(batch)
                
        finally:
            self._exhausted = True


# Specialized batch optimizers

class FeatureBatchOptimizer(BatchOptimizer):
    """Optimized batch processing for feature generation."""
    
    def __init__(self):
        """Initialize feature batch optimizer."""
        super().__init__(BatchConfig(
            batch_size=5000,  # Smaller batches for feature computation
            max_workers=4,
            chunk_memory_limit_mb=200,
            enable_parallel=True
        ))
    
    def generate_features_batch(self,
                               df: pd.DataFrame,
                               feature_generators: List[Callable],
                               progress_callback: Optional[Callable] = None
                              ) -> pd.DataFrame:
        """Generate features in optimized batches.
        
        Args:
            df: Input DataFrame
            feature_generators: List of feature generation functions
            progress_callback: Optional progress callback
            
        Returns:
            DataFrame with generated features
        """
        def process_batch(batch_df: pd.DataFrame) -> pd.DataFrame:
            result = batch_df.copy()
            
            for generator in feature_generators:
                try:
                    features = generator(batch_df)
                    if isinstance(features, pd.DataFrame):
                        result = pd.concat([result, features], axis=1)
                    elif isinstance(features, pd.Series):
                        result[features.name] = features
                except Exception as e:
                    logger.warning(f"Feature generation failed: {e}")
            
            return result
        
        return self.process_dataframe_batches(df, process_batch, progress_callback)