"""Utility functions for CLI operations."""
from pathlib import Path
from datetime import datetime
from typing import Optional, Union


def format_size(size_bytes: int) -> str:
    """Format bytes as human-readable size.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    if size_bytes < 0:
        return "Unknown"
    
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    
    return f"{size_bytes:.1f} PB"


def format_datetime(dt: Optional[Union[str, datetime]]) -> str:
    """Format datetime for display.
    
    Args:
        dt: Datetime object or ISO string
        
    Returns:
        Formatted datetime string
    """
    if not dt:
        return "N/A"
    
    if isinstance(dt, str):
        try:
            dt = datetime.fromisoformat(dt.replace('Z', '+00:00'))
        except ValueError:
            return dt
    
    # Format as relative time if recent
    now = datetime.now(dt.tzinfo)
    delta = now - dt
    
    if delta.days == 0:
        if delta.seconds < 60:
            return "just now"
        elif delta.seconds < 3600:
            minutes = delta.seconds // 60
            return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
        else:
            hours = delta.seconds // 3600
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
    elif delta.days == 1:
        return "yesterday"
    elif delta.days < 7:
        return f"{delta.days} days ago"
    else:
        return dt.strftime("%Y-%m-%d %H:%M")


def validate_output_path(path: str) -> Path:
    """Validate and create output path if needed.
    
    Args:
        path: Output path string
        
    Returns:
        Validated Path object
        
    Raises:
        ValueError: If path is invalid
    """
    output_path = Path(path).resolve()
    
    # Create directory if it doesn't exist
    try:
        output_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise ValueError(f"Cannot create output directory: {e}")
    
    # Check if writable
    if not output_path.is_dir():
        raise ValueError(f"Not a directory: {output_path}")
    
    # Test write permissions
    test_file = output_path / ".mdm_test"
    try:
        test_file.touch()
        test_file.unlink()
    except Exception:
        raise ValueError(f"No write permission: {output_path}")
    
    return output_path


def truncate_string(s: str, max_length: int = 50, suffix: str = "...") -> str:
    """Truncate string to maximum length.
    
    Args:
        s: String to truncate
        max_length: Maximum length
        suffix: Suffix to add when truncated
        
    Returns:
        Truncated string
    """
    if len(s) <= max_length:
        return s
    
    return s[:max_length - len(suffix)] + suffix


def parse_tags(tags_str: Optional[str]) -> Optional[list]:
    """Parse comma-separated tags string.
    
    Args:
        tags_str: Comma-separated tags
        
    Returns:
        List of tags or None
    """
    if not tags_str:
        return None
    
    # Split by comma and clean up
    tags = [tag.strip() for tag in tags_str.split(',')]
    # Remove empty tags
    tags = [tag for tag in tags if tag]
    
    return tags if tags else None


def confirm_action(message: str, default: bool = False) -> bool:
    """Get confirmation for an action.
    
    Args:
        message: Confirmation message
        default: Default answer
        
    Returns:
        User confirmation
    """
    # In non-interactive environments, use default
    import sys
    if not sys.stdin.isatty():
        return default
    
    suffix = " [Y/n]" if default else " [y/N]"
    
    try:
        response = input(message + suffix + " ").strip().lower()
        if not response:
            return default
        return response in ['y', 'yes']
    except (EOFError, KeyboardInterrupt):
        print()  # New line after interrupt
        return False