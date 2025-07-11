#!/usr/bin/env python3
"""Check MDM installation and basic functionality."""

import subprocess
import sys
import os
from pathlib import Path

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None


def check_command_available(command: str) -> bool:
    """Check if a command is available in PATH."""
    try:
        result = subprocess.run([command, "--version"], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def check_mdm_version():
    """Check MDM version."""
    try:
        result = subprocess.run(["mdm", "version"], capture_output=True, text=True)
        if result.returncode == 0:
            return True, result.stdout.strip()
        else:
            return False, result.stderr.strip()
    except FileNotFoundError:
        return False, "MDM command not found in PATH"


def check_python_imports():
    """Check if MDM can be imported."""
    try:
        import mdm
        return True, f"MDM module found at: {mdm.__file__}"
    except ImportError as e:
        return False, f"Cannot import MDM: {e}"


def check_env_variables():
    """Check important environment variables."""
    env_vars = {
        "MDM_HOME_DIR": os.environ.get("MDM_HOME_DIR", "Not set"),
        "MDM_DATABASE_DEFAULT_BACKEND": os.environ.get("MDM_DATABASE_DEFAULT_BACKEND", "Not set"),
        "GITHUB_TOKEN": "Set" if os.environ.get("GITHUB_TOKEN") else "Not set",
        "GITHUB_REPO": os.environ.get("GITHUB_REPO", "Not set"),
    }
    return env_vars


def check_mdm_directories():
    """Check MDM directories."""
    mdm_home = Path(os.environ.get("MDM_HOME_DIR", Path.home() / ".mdm"))
    
    dirs = {
        "MDM Home": (mdm_home, mdm_home.exists()),
        "Datasets": (mdm_home / "datasets", (mdm_home / "datasets").exists()),
        "Config": (mdm_home / "config", (mdm_home / "config").exists()),
        "Logs": (mdm_home / "logs", (mdm_home / "logs").exists()),
    }
    
    return dirs


def run_basic_mdm_command():
    """Try to run a basic MDM command."""
    try:
        result = subprocess.run(["mdm", "dataset", "list"], capture_output=True, text=True)
        if result.returncode == 0:
            return True, "MDM dataset list command works"
        else:
            return False, f"MDM dataset list failed: {result.stderr}"
    except Exception as e:
        return False, f"Error running MDM: {e}"


def main():
    """Main function."""
    if RICH_AVAILABLE:
        console.print("\n[bold]MDM Installation Check[/bold]")
        console.rule()
        
        # Create results table
        table = Table(title="Check Results")
        table.add_column("Check", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Details")
        
        # Check MDM command
        mdm_available = check_command_available("mdm")
        table.add_row(
            "MDM in PATH",
            "✓" if mdm_available else "✗",
            "Found" if mdm_available else "Not found - run: uv pip install -e ."
        )
        
        # Check MDM version
        version_ok, version_info = check_mdm_version()
        table.add_row(
            "MDM Version",
            "✓" if version_ok else "✗",
            version_info
        )
        
        # Check Python import
        import_ok, import_info = check_python_imports()
        table.add_row(
            "Python Import",
            "✓" if import_ok else "✗",
            import_info
        )
        
        # Check basic command
        cmd_ok, cmd_info = run_basic_mdm_command()
        table.add_row(
            "Basic Command",
            "✓" if cmd_ok else "✗",
            cmd_info
        )
        
        console.print(table)
        
        # Environment variables
        console.print("\n[bold]Environment Variables[/bold]")
        env_table = Table()
        env_table.add_column("Variable", style="cyan")
        env_table.add_column("Value")
        
        for var, value in check_env_variables().items():
            style = "green" if value != "Not set" else "red"
            env_table.add_row(var, f"[{style}]{value}[/{style}]")
        
        console.print(env_table)
        
        # Directories
        console.print("\n[bold]MDM Directories[/bold]")
        dir_table = Table()
        dir_table.add_column("Directory", style="cyan")
        dir_table.add_column("Path")
        dir_table.add_column("Exists")
        
        for name, (path, exists) in check_mdm_directories().items():
            dir_table.add_row(
                name,
                str(path),
                "[green]✓[/green]" if exists else "[red]✗[/red]"
            )
        
        console.print(dir_table)
        
        # Summary
        all_ok = mdm_available and version_ok and import_ok and cmd_ok
        
        if all_ok:
            console.print("\n[bold green]✓ MDM is properly installed and working![/bold green]")
        else:
            console.print("\n[bold red]✗ Some checks failed. Please fix the issues above.[/bold red]")
            
    else:
        # Non-rich output
        print("\nMDM Installation Check")
        print("=" * 50)
        
        # Check MDM command
        mdm_available = check_command_available("mdm")
        print(f"MDM in PATH: {'Yes' if mdm_available else 'No'}")
        
        # Check MDM version
        version_ok, version_info = check_mdm_version()
        print(f"MDM Version: {version_info}")
        
        # Check Python import
        import_ok, import_info = check_python_imports()
        print(f"Python Import: {import_info}")
        
        # Check basic command
        cmd_ok, cmd_info = run_basic_mdm_command()
        print(f"Basic Command: {cmd_info}")
        
        # Environment variables
        print("\nEnvironment Variables:")
        for var, value in check_env_variables().items():
            print(f"  {var}: {value}")
        
        # Directories
        print("\nMDM Directories:")
        for name, (path, exists) in check_mdm_directories().items():
            print(f"  {name}: {path} ({'exists' if exists else 'missing'})")
        
        # Summary
        all_ok = mdm_available and version_ok and import_ok and cmd_ok
        
        if all_ok:
            print("\n✓ MDM is properly installed and working!")
        else:
            print("\n✗ Some checks failed. Please fix the issues above.")
    
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())