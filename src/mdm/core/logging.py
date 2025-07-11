"""Logging configuration for MDM.

This module provides centralized logging configuration using loguru.
"""
import sys
from pathlib import Path
from typing import Optional
from loguru import logger

# Don't remove handlers here - let main.py manage them

# Global logger configuration
_configured = False


def configure_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format: Optional[str] = None
) -> None:
    """Configure logging for MDM.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for logging
        format: Optional custom format string
    """
    global _configured
    
    if _configured:
        return
    
    # Default format
    if format is None:
        format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )
    
    # Add console handler with level WARNING and above
    logger.add(
        sys.stderr,
        format=format,
        level="WARNING",
        colorize=True
    )
    
    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_file,
            format=format,
            level=level,
            rotation="10 MB",
            retention="7 days",
            compression="zip"
        )
    
    # Set level for all handlers
    logger.level(level)
    
    _configured = True


def get_logger(name: str) -> "logger":
    """Get a logger instance for the given name.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Logger instance
    """
    # Configure logging if not already done
    if not _configured:
        # Try to get config from environment or use defaults
        import os
        level = os.environ.get("MDM_LOGGING_LEVEL", "INFO")
        log_file = os.environ.get("MDM_LOGGING_FILE")
        configure_logging(level=level, log_file=log_file)
    
    # Loguru uses a single logger instance, but we can bind context
    return logger.bind(name=name)


def intercept_standard_logging():
    """Intercept standard library logging and redirect to loguru.
    
    This ensures that libraries using standard logging are captured.
    """
    import logging
    
    class InterceptHandler(logging.Handler):
        def emit(self, record):
            # Get corresponding Loguru level if it exists
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno

            # Find caller from where originated the logged message
            frame, depth = logging.currentframe(), 2
            while frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1

            logger.opt(depth=depth, exception=record.exc_info).log(
                level, record.getMessage()
            )

    # Install handler
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)


# Don't auto-configure on import - let main.py handle it


__all__ = ["logger", "get_logger", "configure_logging"]