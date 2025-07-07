"""Performance optimization utilities for MDM."""

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Optional

import psutil

from mdm.config import get_config

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Monitor and log performance metrics."""

    def __init__(self):
        """Initialize performance monitor."""
        self.config = get_config()
        self.metrics = {}
        self.start_time = None

    @contextmanager
    def timer(self, operation: str) -> Iterator[None]:
        """Time an operation.
        
        Args:
            operation: Name of the operation
            
        Yields:
            None
        """
        start = time.time()
        logger.debug(f"Starting {operation}...")

        try:
            yield
        finally:
            elapsed = time.time() - start
            self.metrics[operation] = elapsed
            logger.info(f"{operation} completed in {elapsed:.2f}s")

    @contextmanager
    def memory_tracker(self, operation: str) -> Iterator[None]:
        """Track memory usage for an operation.
        
        Args:
            operation: Name of the operation
            
        Yields:
            None
        """
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB

        try:
            yield
        finally:
            mem_after = process.memory_info().rss / 1024 / 1024  # MB
            mem_used = mem_after - mem_before
            logger.info(f"{operation} used {mem_used:.1f} MB of memory")

    def get_metrics(self) -> dict[str, Any]:
        """Get collected metrics."""
        return self.metrics.copy()


class ChunkProcessor:
    """Process large datasets in chunks."""

    def __init__(self, chunk_size: Optional[int] = None):
        """Initialize chunk processor.
        
        Args:
            chunk_size: Size of chunks (default from config)
        """
        config = get_config()
        self.chunk_size = chunk_size or config.performance.batch_size

    def process_dataframe(self, df, process_func, show_progress: bool = True):
        """Process DataFrame in chunks.
        
        Args:
            df: DataFrame to process
            process_func: Function to apply to each chunk
            show_progress: Whether to show progress
            
        Returns:
            List of results from each chunk
        """

        total_rows = len(df)
        if total_rows <= self.chunk_size:
            # Process in one go
            return [process_func(df)]

        results = []
        chunks = []

        # Create chunks
        for start in range(0, total_rows, self.chunk_size):
            end = min(start + self.chunk_size, total_rows)
            chunks.append((start, end))

        if show_progress:
            from rich.progress import (
                BarColumn,
                Progress,
                SpinnerColumn,
                TaskProgressColumn,
                TextColumn,
            )

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            ) as progress:
                task = progress.add_task("Processing chunks", total=len(chunks))

                for i, (start, end) in enumerate(chunks):
                    chunk = df.iloc[start:end]
                    result = process_func(chunk)
                    results.append(result)
                    progress.update(task, advance=1)
        else:
            for start, end in chunks:
                chunk = df.iloc[start:end]
                result = process_func(chunk)
                results.append(result)

        return results

    def process_file(self, file_path: str, process_func, reader_func=None, **reader_kwargs):
        """Process file in chunks.
        
        Args:
            file_path: Path to file
            process_func: Function to apply to each chunk
            reader_func: Custom reader function
            **reader_kwargs: Arguments for reader
            
        Returns:
            List of results from each chunk
        """
        import pandas as pd

        if reader_func is None:
            # Determine reader based on file extension
            if file_path.endswith('.csv'):
                reader_func = pd.read_csv
            elif file_path.endswith('.parquet'):
                reader_func = pd.read_parquet
                # Parquet doesn't support chunking like CSV
                df = reader_func(file_path, **reader_kwargs)
                return self.process_dataframe(df, process_func)
            else:
                raise ValueError(f"Unsupported file type: {file_path}")

        # For CSV, use chunked reading
        if reader_func == pd.read_csv:
            reader_kwargs['chunksize'] = self.chunk_size
            results = []

            for chunk in reader_func(file_path, **reader_kwargs):
                result = process_func(chunk)
                results.append(result)

            return results
        # For other formats, read full and process in chunks
        df = reader_func(file_path, **reader_kwargs)
        return self.process_dataframe(df, process_func)


class QueryOptimizer:
    """Optimize database queries."""

    def __init__(self, backend_type: str):
        """Initialize query optimizer.
        
        Args:
            backend_type: Type of database backend
        """
        self.backend_type = backend_type.lower()

    def optimize_query(self, query: str) -> str:
        """Optimize a query for the specific backend.
        
        Args:
            query: Original SQL query
            
        Returns:
            Optimized query
        """
        # Basic optimizations
        query = query.strip()

        if self.backend_type == 'duckdb':
            # DuckDB-specific optimizations
            # Use EXPLAIN ANALYZE for query planning
            if query.upper().startswith('SELECT'):
                # Add sample clause for large tables
                if 'LIMIT' not in query.upper():
                    # Don't add LIMIT to aggregation queries
                    if not any(agg in query.upper() for agg in ['COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'GROUP BY']):
                        query += ' LIMIT 1000000'  # Default limit for safety

        elif self.backend_type == 'sqlite':
            # SQLite-specific optimizations
            # Enable query planner optimizations
            pass

        elif self.backend_type == 'postgresql':
            # PostgreSQL-specific optimizations
            # Use ANALYZE for better statistics
            pass

        return query

    def create_indexes(self, conn, table_name: str, columns: list[str]) -> None:
        """Create indexes for better query performance.
        
        Args:
            conn: Database connection
            table_name: Table name
            columns: Columns to index
        """
        for column in columns:
            index_name = f"idx_{table_name}_{column}"
            try:
                if self.backend_type == 'duckdb':
                    # DuckDB doesn't support traditional indexes
                    # It uses automatic indexing
                    pass
                elif self.backend_type in ['sqlite', 'postgresql']:
                    query = f"CREATE INDEX IF NOT EXISTS {index_name} ON {table_name}({column})"
                    conn.execute(query)
                    logger.debug(f"Created index {index_name}")
            except Exception as e:
                logger.warning(f"Could not create index {index_name}: {e}")


def estimate_memory_usage(df) -> float:
    """Estimate memory usage of a DataFrame in MB.
    
    Args:
        df: DataFrame
        
    Returns:
        Memory usage in MB
    """
    return df.memory_usage(deep=True).sum() / 1024 / 1024


def optimize_dtypes(df):
    """Optimize DataFrame dtypes to reduce memory usage.
    
    Args:
        df: DataFrame to optimize
        
    Returns:
        Optimized DataFrame
    """
    import numpy as np
    import pandas as pd

    # Store original memory usage
    mem_before = estimate_memory_usage(df)

    # Optimize numeric types
    for col in df.select_dtypes(include=['int']).columns:
        col_min = df[col].min()
        col_max = df[col].max()

        if col_min >= 0:
            if col_max < 255:
                df[col] = df[col].astype(np.uint8)
            elif col_max < 65535:
                df[col] = df[col].astype(np.uint16)
            elif col_max < 4294967295:
                df[col] = df[col].astype(np.uint32)
        else:
            if col_min > -128 and col_max < 127:
                df[col] = df[col].astype(np.int8)
            elif col_min > -32768 and col_max < 32767:
                df[col] = df[col].astype(np.int16)
            elif col_min > -2147483648 and col_max < 2147483647:
                df[col] = df[col].astype(np.int32)

    # Optimize float types
    for col in df.select_dtypes(include=['float']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')

    # Convert low-cardinality strings to categories
    for col in df.select_dtypes(include=['object']).columns:
        num_unique = df[col].nunique()
        num_total = len(df[col])
        if num_unique / num_total < 0.5:  # Less than 50% unique
            df[col] = df[col].astype('category')

    # Log memory savings
    mem_after = estimate_memory_usage(df)
    mem_saved = mem_before - mem_after
    if mem_saved > 0:
        logger.info(f"Memory usage reduced by {mem_saved:.1f} MB ({mem_saved/mem_before*100:.1f}%)")

    return df
