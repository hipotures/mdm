#!/usr/bin/env python3
"""Debug test environment issues."""

import os
import sys
import tempfile
import shutil
from pathlib import Path
import subprocess

try:
    from rich.console import Console
    from rich.panel import Panel
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None


def create_test_environment():
    """Create a clean test environment."""
    test_dir = Path(tempfile.mkdtemp(prefix="mdm_test_"))
    
    # Set up MDM directories
    (test_dir / "datasets").mkdir()
    (test_dir / "config" / "datasets").mkdir(parents=True)
    (test_dir / "logs").mkdir()
    
    # Create minimal config
    config_file = test_dir / "mdm.yaml"
    config_file.write_text("""
database:
  default_backend: sqlite
  sqlite:
    synchronous: NORMAL
    journal_mode: WAL

performance:
  batch_size: 10000

logging:
  level: DEBUG
  file: mdm.log
  format: console
""")
    
    return test_dir


def run_test_in_environment(test_env: Path, test_file: str = None):
    """Run a test in the isolated environment."""
    env = os.environ.copy()
    env["MDM_HOME_DIR"] = str(test_env)
    
    if test_file:
        cmd = [sys.executable, "-m", "pytest", test_file, "-xvs"]
    else:
        # Run a simple MDM command
        cmd = ["mdm", "info"]
    
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    
    return result


def check_dataset_isolation():
    """Check if datasets are properly isolated."""
    # Create two test environments
    env1 = create_test_environment()
    env2 = create_test_environment()
    
    results = []
    
    # Create dataset in env1
    env = os.environ.copy()
    env["MDM_HOME_DIR"] = str(env1)
    
    # Create a test CSV file
    test_csv = env1 / "test_data.csv"
    test_csv.write_text("id,value\n1,10\n2,20\n3,30\n")
    
    # Register dataset in env1
    result1 = subprocess.run(
        ["mdm", "dataset", "register", "test_dataset", str(test_csv)],
        env=env,
        capture_output=True,
        text=True
    )
    results.append(("Register in env1", result1.returncode == 0, result1.stderr))
    
    # Check if dataset exists in env1
    result2 = subprocess.run(
        ["mdm", "dataset", "list"],
        env=env,
        capture_output=True,
        text=True
    )
    results.append(("List in env1", "test_dataset" in result2.stdout, result2.stdout))
    
    # Check if dataset is NOT visible in env2
    env["MDM_HOME_DIR"] = str(env2)
    result3 = subprocess.run(
        ["mdm", "dataset", "list"],
        env=env,
        capture_output=True,
        text=True
    )
    results.append(("List in env2 (should be empty)", "test_dataset" not in result3.stdout, result3.stdout))
    
    # Cleanup
    shutil.rmtree(env1)
    shutil.rmtree(env2)
    
    return results


def check_config_loading():
    """Check configuration loading in test environment."""
    test_env = create_test_environment()
    
    # Test with different configs
    configs = [
        ("Default config", {}),
        ("SQLite backend", {"MDM_DATABASE_DEFAULT_BACKEND": "sqlite"}),
        ("DuckDB backend", {"MDM_DATABASE_DEFAULT_BACKEND": "duckdb"}),
        ("Debug logging", {"MDM_LOGGING_LEVEL": "DEBUG"}),
        ("Custom batch size", {"MDM_PERFORMANCE_BATCH_SIZE": "5000"}),
    ]
    
    results = []
    
    for name, extra_env in configs:
        env = os.environ.copy()
        env["MDM_HOME_DIR"] = str(test_env)
        env.update(extra_env)
        
        result = subprocess.run(
            ["mdm", "info"],
            env=env,
            capture_output=True,
            text=True
        )
        
        results.append((name, result.returncode == 0, result.stdout))
    
    # Cleanup
    shutil.rmtree(test_env)
    
    return results


def main():
    """Main function."""
    if RICH_AVAILABLE:
        console.print("\n[bold]MDM Test Environment Debugger[/bold]")
        console.rule()
        
        # Test environment creation
        console.print("\n[bold cyan]1. Testing environment creation...[/bold cyan]")
        try:
            test_env = create_test_environment()
            console.print(f"[green]✓[/green] Created test environment at: {test_env}")
            
            # Check contents
            console.print("\nEnvironment contents:")
            for item in sorted(test_env.rglob("*")):
                if item.is_file():
                    console.print(f"  [dim]file:[/dim] {item.relative_to(test_env)}")
                else:
                    console.print(f"  [dim]dir:[/dim]  {item.relative_to(test_env)}/")
            
            shutil.rmtree(test_env)
            
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to create test environment: {e}")
        
        # Test dataset isolation
        console.print("\n[bold cyan]2. Testing dataset isolation...[/bold cyan]")
        try:
            isolation_results = check_dataset_isolation()
            for name, passed, details in isolation_results:
                if passed:
                    console.print(f"[green]✓[/green] {name}")
                else:
                    console.print(f"[red]✗[/red] {name}")
                    if details:
                        console.print(f"  [dim]{details[:100]}...[/dim]")
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to test isolation: {e}")
        
        # Test configuration loading
        console.print("\n[bold cyan]3. Testing configuration loading...[/bold cyan]")
        try:
            config_results = check_config_loading()
            for name, passed, output in config_results:
                if passed:
                    console.print(f"[green]✓[/green] {name}")
                else:
                    console.print(f"[red]✗[/red] {name}")
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to test configuration: {e}")
        
        # Test specific test file if provided
        if len(sys.argv) > 1:
            test_file = sys.argv[1]
            console.print(f"\n[bold cyan]4. Testing specific file: {test_file}[/bold cyan]")
            
            test_env = create_test_environment()
            result = run_test_in_environment(test_env, test_file)
            
            if result.returncode == 0:
                console.print(f"[green]✓[/green] Test passed")
            else:
                console.print(f"[red]✗[/red] Test failed")
                console.print("\nOutput:")
                console.print(Panel(result.stdout + result.stderr, title="Test Output"))
            
            shutil.rmtree(test_env)
            
    else:
        # Non-rich output
        print("\nMDM Test Environment Debugger")
        print("=" * 50)
        
        # Test environment creation
        print("\n1. Testing environment creation...")
        try:
            test_env = create_test_environment()
            print(f"✓ Created test environment at: {test_env}")
            shutil.rmtree(test_env)
        except Exception as e:
            print(f"✗ Failed to create test environment: {e}")
        
        # Test dataset isolation
        print("\n2. Testing dataset isolation...")
        try:
            isolation_results = check_dataset_isolation()
            for name, passed, details in isolation_results:
                print(f"{'✓' if passed else '✗'} {name}")
        except Exception as e:
            print(f"✗ Failed to test isolation: {e}")
        
        # Test configuration loading
        print("\n3. Testing configuration loading...")
        try:
            config_results = check_config_loading()
            for name, passed, output in config_results:
                print(f"{'✓' if passed else '✗'} {name}")
        except Exception as e:
            print(f"✗ Failed to test configuration: {e}")
        
        # Test specific test file if provided
        if len(sys.argv) > 1:
            test_file = sys.argv[1]
            print(f"\n4. Testing specific file: {test_file}")
            
            test_env = create_test_environment()
            result = run_test_in_environment(test_env, test_file)
            
            if result.returncode == 0:
                print("✓ Test passed")
            else:
                print("✗ Test failed")
                print("\nOutput:")
                print(result.stdout + result.stderr)
            
            shutil.rmtree(test_env)


if __name__ == "__main__":
    main()