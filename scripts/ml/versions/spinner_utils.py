#!/usr/bin/env python3
"""
Spinner utilities for aesthetic CV progress display.
"""

import time
import threading
from typing import Optional, List
from rich.console import Console
from rich.text import Text

console = Console()


class CVSpinner:
    """Aesthetic CV progress spinner with block characters."""
    
    def __init__(self, total_folds: int = 3, symbols: tuple = ("■", "□")):
        """
        Initialize CV spinner.
        
        Args:
            total_folds: Total number of CV folds
            symbols: Tuple of (filled, empty) symbols
        """
        self.total_folds = total_folds
        self.filled_symbol, self.empty_symbol = symbols
        self.current_fold = 0
        self.is_spinning = False
        self.spinner_thread: Optional[threading.Thread] = None
        self.current_message = ""
        
    def _create_progress_bar(self, current: int, total: int) -> str:
        """Create a visual progress bar with block characters."""
        filled = self.filled_symbol * current
        empty = self.empty_symbol * (total - current)
        return f"{filled}{empty}"
    
    def _spinner_worker(self):
        """Worker thread for the spinner animation."""
        spinner_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        spinner_idx = 0
        
        while self.is_spinning:
            progress_bar = self._create_progress_bar(self.current_fold, self.total_folds)
            spinner_char = spinner_chars[spinner_idx % len(spinner_chars)]
            
            # Clear line and show progress
            console.print(
                f"\r  {spinner_char} CV Progress: {progress_bar} ({self.current_fold}/{self.total_folds}) - {self.current_message}",
                end="",
                highlight=False
            )
            
            spinner_idx += 1
            time.sleep(0.1)
    
    def start(self, message: str = "Processing"):
        """Start the spinner with a message."""
        self.current_message = message
        self.is_spinning = True
        self.spinner_thread = threading.Thread(target=self._spinner_worker)
        self.spinner_thread.daemon = True
        self.spinner_thread.start()
    
    def update_fold(self, fold: int, message: str = ""):
        """Update current fold progress."""
        self.current_fold = fold
        if message:
            self.current_message = message
    
    def stop(self, final_message: str = "Completed"):
        """Stop the spinner and show final message."""
        if self.is_spinning:
            self.is_spinning = False
            if self.spinner_thread:
                self.spinner_thread.join(timeout=0.5)
            
            # Clear the line and show final result
            progress_bar = self._create_progress_bar(self.total_folds, self.total_folds)
            console.print(f"\r  ✓ CV Progress: {progress_bar} ({self.total_folds}/{self.total_folds}) - {final_message}")


class IterationSpinner:
    """Simple spinner for showing progress during iterations."""
    
    def __init__(self):
        self.is_spinning = False
        self.spinner_thread: Optional[threading.Thread] = None
        self.current_message = ""
    
    def _spinner_worker(self):
        """Worker thread for the spinner animation."""
        spinner_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        spinner_idx = 0
        
        while self.is_spinning:
            spinner_char = spinner_chars[spinner_idx % len(spinner_chars)]
            console.print(f"\r    {spinner_char} {self.current_message}", end="", highlight=False)
            spinner_idx += 1
            time.sleep(0.1)
    
    def start(self, message: str):
        """Start the spinner with a message."""
        self.current_message = message
        self.is_spinning = True
        self.spinner_thread = threading.Thread(target=self._spinner_worker)
        self.spinner_thread.daemon = True
        self.spinner_thread.start()
    
    def update(self, message: str):
        """Update the spinner message."""
        self.current_message = message
    
    def stop(self, final_message: str = ""):
        """Stop the spinner and show final message."""
        if self.is_spinning:
            self.is_spinning = False
            if self.spinner_thread:
                self.spinner_thread.join(timeout=0.5)
            
            if final_message:
                console.print(f"\r    ✓ {final_message}")
            else:
                console.print()  # Clear line


# Alternative symbol sets for different aesthetics
SYMBOL_SETS = {
    'blocks': ("■", "□"),           # Default solid blocks
    'squares': ("▪", "▫"),          # Small squares  
    'heavy': ("▰", "▱"),            # Heavy blocks
    'circles': ("●", "○"),          # Circles
    'diamonds': ("◆", "◇"),         # Diamonds
    'triangles': ("▲", "△"),        # Triangles
}


def create_cv_spinner(folds: int = 3, style: str = 'blocks') -> CVSpinner:
    """
    Create a CV spinner with specified style.
    
    Args:
        folds: Number of CV folds
        style: Style from SYMBOL_SETS
    
    Returns:
        CVSpinner instance
    """
    symbols = SYMBOL_SETS.get(style, SYMBOL_SETS['blocks'])
    return CVSpinner(total_folds=folds, symbols=symbols)


def create_iteration_spinner() -> IterationSpinner:
    """Create a simple iteration spinner."""
    return IterationSpinner()