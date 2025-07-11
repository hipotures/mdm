"""Progress tracking implementation for feature generation."""

from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
import time

from rich.progress import (
    Progress, 
    SpinnerColumn, 
    TextColumn, 
    BarColumn, 
    TaskProgressColumn, 
    TimeRemainingColumn
)
from rich.console import Console
from loguru import logger


class IProgressTracker(ABC):
    """Interface for progress tracking."""
    
    @abstractmethod
    def start_task(self, name: str, total: Optional[int] = None) -> Any:
        """Start a new task.
        
        Args:
            name: Task name/description
            total: Total steps (None for indeterminate)
            
        Returns:
            Task ID or handle
        """
        pass
    
    @abstractmethod
    def update_task(self, task_id: Any, advance: int = 1, 
                   description: Optional[str] = None) -> None:
        """Update task progress.
        
        Args:
            task_id: Task ID from start_task
            advance: Number of steps to advance
            description: Optional new description
        """
        pass
    
    @abstractmethod
    def complete_task(self, task_id: Any) -> None:
        """Mark task as complete.
        
        Args:
            task_id: Task ID from start_task
        """
        pass


class RichProgressTracker(IProgressTracker):
    """Progress tracking using Rich library."""
    
    def __init__(self, console: Optional[Console] = None, 
                 transient: bool = True):
        """Initialize Rich progress tracker.
        
        Args:
            console: Rich console (None for default)
            transient: Whether progress should disappear when done
        """
        self.console = console
        self.transient = transient
        self._progress: Optional[Progress] = None
        self._tasks: Dict[Any, Dict[str, Any]] = {}
    
    def __enter__(self):
        """Enter context manager."""
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=self.console,
            transient=self.transient
        )
        self._progress.__enter__()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        if self._progress:
            self._progress.__exit__(exc_type, exc_val, exc_tb)
            self._progress = None
        self._tasks.clear()
    
    def start_task(self, name: str, total: Optional[int] = None) -> Any:
        """Start a new task.
        
        Args:
            name: Task description
            total: Total steps (None for indeterminate)
            
        Returns:
            Task ID
        """
        if not self._progress:
            raise RuntimeError("Progress tracker not initialized. Use with context manager.")
        
        task_id = self._progress.add_task(name, total=total)
        self._tasks[task_id] = {
            'name': name,
            'total': total,
            'start_time': time.time()
        }
        
        logger.debug(f"Started progress task: {name}")
        return task_id
    
    def update_task(self, task_id: Any, advance: int = 1, 
                   description: Optional[str] = None) -> None:
        """Update task progress.
        
        Args:
            task_id: Task ID from start_task
            advance: Number of steps to advance
            description: Optional new description
        """
        if not self._progress:
            return
        
        if description:
            self._progress.update(task_id, advance=advance, description=description)
        else:
            self._progress.update(task_id, advance=advance)
    
    def complete_task(self, task_id: Any) -> None:
        """Mark task as complete.
        
        Args:
            task_id: Task ID from start_task
        """
        if not self._progress or task_id not in self._tasks:
            return
        
        # Complete the task
        task_info = self._tasks[task_id]
        if task_info['total'] is not None:
            current = self._progress.tasks[task_id].completed
            remaining = task_info['total'] - current
            if remaining > 0:
                self._progress.update(task_id, advance=remaining)
        
        # Log completion time
        elapsed = time.time() - task_info['start_time']
        logger.debug(f"Completed task '{task_info['name']}' in {elapsed:.2f}s")


class NoOpProgressTracker(IProgressTracker):
    """No-operation progress tracker for when progress is disabled."""
    
    def start_task(self, name: str, total: Optional[int] = None) -> Any:
        """Start a new task (no-op)."""
        logger.debug(f"Progress tracking disabled. Task: {name}")
        return None
    
    def update_task(self, task_id: Any, advance: int = 1, 
                   description: Optional[str] = None) -> None:
        """Update task progress (no-op)."""
        pass
    
    def complete_task(self, task_id: Any) -> None:
        """Mark task as complete (no-op)."""
        pass


class BatchProgressTracker:
    """Specialized progress tracker for batch processing."""
    
    def __init__(self, total_rows: int, batch_size: int,
                 tracker: Optional[IProgressTracker] = None):
        """Initialize batch progress tracker.
        
        Args:
            total_rows: Total number of rows to process
            batch_size: Size of each batch
            tracker: Underlying progress tracker (None for Rich)
        """
        self.total_rows = total_rows
        self.batch_size = batch_size
        self.total_batches = (total_rows + batch_size - 1) // batch_size
        self.tracker = tracker
        self._task_id = None
        self._batches_processed = 0
        self._rows_processed = 0
    
    def __enter__(self):
        """Enter context manager."""
        if self.tracker is None:
            self.tracker = RichProgressTracker()
            self.tracker.__enter__()
        
        self._task_id = self.tracker.start_task(
            f"Processing {self.total_rows:,} rows in {self.total_batches} batches",
            total=self.total_rows
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        if self._task_id:
            self.tracker.complete_task(self._task_id)
        
        if isinstance(self.tracker, RichProgressTracker):
            self.tracker.__exit__(exc_type, exc_val, exc_tb)
    
    def update_batch(self, rows_in_batch: int) -> None:
        """Update progress for a completed batch.
        
        Args:
            rows_in_batch: Number of rows in the completed batch
        """
        self._batches_processed += 1
        self._rows_processed += rows_in_batch
        
        description = (f"Batch {self._batches_processed}/{self.total_batches} "
                      f"({self._rows_processed:,}/{self.total_rows:,} rows)")
        
        self.tracker.update_task(
            self._task_id, 
            advance=rows_in_batch,
            description=description
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current progress statistics.
        
        Returns:
            Dictionary with progress stats
        """
        return {
            'batches_processed': self._batches_processed,
            'rows_processed': self._rows_processed,
            'total_batches': self.total_batches,
            'total_rows': self.total_rows,
            'progress_percentage': (self._rows_processed / self.total_rows * 100) 
                                 if self.total_rows > 0 else 0
        }