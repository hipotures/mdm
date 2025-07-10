"""Example demonstrating CLI migration functionality.

This example shows how to:
1. Use the new CLI implementation
2. Compare old and new CLI outputs
3. Migrate CLI configurations
4. Test command compatibility
5. Create custom CLI plugins
"""
import sys
from pathlib import Path
import tempfile
import shutil

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from mdm.core import feature_flags
from mdm.adapters import (
    get_dataset_commands,
    get_batch_commands,
    get_cli_formatter,
    get_cli_config,
    clear_cli_cache
)
from mdm.migration import CLIMigrator, CLIValidator
from mdm.testing import CLIComparisonTester
from mdm.core.cli.plugins import CLIPlugin, CLIPluginManager

console = Console()


def main():
    """Run CLI migration examples."""
    console.print(Panel.fit(
        "[bold cyan]CLI Migration Examples[/bold cyan]\n\n"
        "This demonstrates the CLI migration functionality",
        title="MDM CLI Migration"
    ))
    
    # Create temporary directory for examples
    temp_dir = Path(tempfile.mkdtemp(prefix="mdm_cli_example_"))
    
    try:
        # Example 1: Basic CLI usage comparison
        console.print("\n[bold]Example 1: Basic CLI Usage Comparison[/bold]")
        console.print("=" * 50 + "\n")
        example_basic_usage(temp_dir)
        
        # Example 2: Enhanced formatting
        console.print("\n[bold]Example 2: Enhanced Formatting[/bold]")
        console.print("=" * 50 + "\n")
        example_enhanced_formatting()
        
        # Example 3: CLI configuration
        console.print("\n[bold]Example 3: CLI Configuration[/bold]")
        console.print("=" * 50 + "\n")
        example_cli_configuration()
        
        # Example 4: Command compatibility testing
        console.print("\n[bold]Example 4: Command Compatibility Testing[/bold]")
        console.print("=" * 50 + "\n")
        example_compatibility_testing()
        
        # Example 5: CLI migration
        console.print("\n[bold]Example 5: CLI Migration[/bold]")
        console.print("=" * 50 + "\n")
        example_cli_migration()
        
        # Example 6: Custom CLI plugin
        console.print("\n[bold]Example 6: Custom CLI Plugin[/bold]")
        console.print("=" * 50 + "\n")
        example_custom_plugin()
        
        # Example 7: Performance comparison
        console.print("\n[bold]Example 7: Performance Comparison[/bold]")
        console.print("=" * 50 + "\n")
        example_performance_comparison(temp_dir)
        
        console.print("\n[bold green]Examples completed successfully![/bold green]")
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


def example_basic_usage(temp_dir: Path):
    """Example of basic CLI usage with old and new implementations."""
    # Create sample dataset
    console.print("Creating sample dataset...")
    data = pd.DataFrame({
        'user_id': range(1, 101),
        'age': np.random.randint(18, 65, 100),
        'score': np.random.uniform(0, 100, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    })
    
    csv_path = temp_dir / "users.csv"
    data.to_csv(csv_path, index=False)
    
    # Register with legacy CLI
    console.print("\n[yellow]Using legacy CLI:[/yellow]")
    feature_flags.set("use_new_cli", False)
    clear_cli_cache()
    
    legacy_commands = get_dataset_commands()
    legacy_result = legacy_commands.register(
        name="users_legacy",
        path=str(csv_path),
        target="category",
        force=True,
        generate_features=False
    )
    
    console.print("Legacy output:", legacy_result)
    
    # Register with new CLI
    console.print("\n[green]Using new CLI:[/green]")
    feature_flags.set("use_new_cli", True)
    clear_cli_cache()
    
    new_commands = get_dataset_commands()
    new_result = new_commands.register(
        name="users_new",
        path=str(csv_path),
        target="category",
        force=True,
        generate_features=False
    )
    
    # Note: New CLI prints rich output directly
    
    # List datasets with new CLI
    console.print("\n[green]Listing datasets with new CLI:[/green]")
    new_commands.list_datasets(limit=5)
    
    # Cleanup
    cleanup_datasets(["users_legacy", "users_new"])


def example_enhanced_formatting():
    """Example of enhanced formatting capabilities."""
    console.print("Demonstrating new formatter capabilities...")
    
    # Get new formatter
    feature_flags.set("use_new_cli", True)
    formatter = get_cli_formatter()
    
    # Sample data
    data = [
        {'name': 'dataset1', 'size': 1024000, 'status': True, 'created': '2024-01-15'},
        {'name': 'dataset2', 'size': 2048000, 'status': False, 'created': '2024-01-16'},
        {'name': 'dataset3', 'size': 512000, 'status': True, 'created': '2024-01-17'},
    ]
    
    # Table formatting
    console.print("\n[bold]Table Format:[/bold]")
    table_output = formatter.format_table(data, title="Sample Datasets")
    console.print(table_output)
    
    # JSON formatting
    console.print("\n[bold]JSON Format (with syntax highlighting):[/bold]")
    json_output = formatter.format_json(data[0])
    console.print(json_output)
    
    # Tree formatting
    console.print("\n[bold]Tree Format:[/bold]")
    tree_data = {
        'datasets': {
            'training': ['dataset1', 'dataset2'],
            'testing': ['dataset3'],
            'validation': []
        }
    }
    tree_output = formatter.format_tree(tree_data, title="Dataset Organization")
    console.print(tree_output)
    
    # Error formatting
    console.print("\n[bold]Error Format:[/bold]")
    try:
        raise ValueError("Sample error for demonstration")
    except Exception as e:
        error_output = formatter.format_error(e, verbose=False)
        console.print(error_output)


def example_cli_configuration():
    """Example of CLI configuration management."""
    console.print("Demonstrating CLI configuration...")
    
    # Get new config
    feature_flags.set("use_new_cli", True)
    config = get_cli_config()
    
    # Display current settings
    table = Table(title="Current CLI Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="yellow")
    
    table.add_row("Output Format", config.output_format)
    table.add_row("Color Enabled", str(config.color_enabled))
    table.add_row("Verbose", str(config.verbose))
    table.add_row("Quiet", str(config.quiet))
    
    console.print(table)
    
    # Get command-specific config
    console.print("\n[bold]Command Configuration:[/bold]")
    register_config = config.get_command_config("dataset.register")
    console.print(f"dataset.register config: {register_config}")
    
    # Get theme
    console.print("\n[bold]Theme Configuration:[/bold]")
    theme = config.get_theme()
    for key, value in list(theme.items())[:5]:
        console.print(f"  {key}: [dim]{value}[/dim]")


def example_compatibility_testing():
    """Example of command compatibility testing."""
    console.print("Testing command compatibility...")
    
    # Create validator
    validator = CLIValidator()
    
    # Test formatter compatibility
    console.print("\n[bold]Testing Formatter Compatibility:[/bold]")
    formatter_results = validator.validate_formatter_compatibility()
    
    # Test config compatibility
    console.print("\n[bold]Testing Configuration Compatibility:[/bold]")
    config_results = validator.validate_config_compatibility()
    
    # Summary
    if formatter_results['compatible'] and config_results['compatible']:
        console.print("\n[green]✓ All compatibility tests passed![/green]")
    else:
        console.print("\n[yellow]⚠ Some compatibility issues found[/yellow]")


def example_cli_migration():
    """Example of CLI migration process."""
    console.print("Demonstrating CLI migration...")
    
    # Create migrator
    migrator = CLIMigrator()
    
    # Check command compatibility
    console.print("\n[bold]Validating Dataset Commands:[/bold]")
    dataset_validation = migrator.validate_command_compatibility(
        command_group="dataset",
        commands=["register", "list_datasets", "info", "remove"]
    )
    
    # Test specific command output
    console.print("\n[bold]Testing Command Output Differences:[/bold]")
    
    # Create test data
    temp_file = Path(tempfile.mktemp(suffix=".csv"))
    pd.DataFrame({'id': range(10), 'value': range(10)}).to_csv(temp_file, index=False)
    
    output_test = migrator.test_command_output(
        command_group="dataset",
        command="register",
        name="test_migration",
        path=str(temp_file),
        force=True,
        generate_features=False
    )
    
    # Cleanup
    temp_file.unlink()
    cleanup_datasets(["test_migration"])
    
    # Migration recommendations
    console.print("\n[bold]Migration Recommendations:[/bold]")
    if dataset_validation['compatible'] == dataset_validation['total_commands']:
        console.print("  [green]✓[/green] All commands are compatible - safe to migrate")
    else:
        incompatible = dataset_validation['total_commands'] - dataset_validation['compatible']
        console.print(f"  [yellow]⚠[/yellow] Fix {incompatible} incompatible commands before migration")


def example_custom_plugin():
    """Example of creating a custom CLI plugin."""
    console.print("Creating custom CLI plugin...")
    
    # Define custom plugin
    class AnalyticsPlugin(CLIPlugin):
        """Custom analytics plugin for MDM."""
        
        def __init__(self):
            super().__init__()
            self.name = "analytics"
            self.version = "1.0.0"
            self.description = "Advanced analytics commands"
            self.commands = [
                {
                    'name': 'correlations',
                    'description': 'Compute correlations',
                    'handler': self.cmd_correlations
                },
                {
                    'name': 'outliers',
                    'description': 'Detect outliers',
                    'handler': self.cmd_outliers
                }
            ]
        
        def cmd_correlations(self, dataset: str, threshold: float = 0.7) -> dict:
            """Compute correlations above threshold."""
            # Simplified example
            return {
                'dataset': dataset,
                'threshold': threshold,
                'correlations': [
                    {'col1': 'age', 'col2': 'score', 'value': 0.85},
                    {'col1': 'score', 'col2': 'income', 'value': 0.72}
                ]
            }
        
        def cmd_outliers(self, dataset: str, method: str = "zscore") -> dict:
            """Detect outliers in dataset."""
            return {
                'dataset': dataset,
                'method': method,
                'outliers_found': 5,
                'columns_affected': ['score', 'age']
            }
    
    # Create and demonstrate plugin
    plugin = AnalyticsPlugin()
    
    console.print(f"\n[bold]Plugin: {plugin.name} v{plugin.version}[/bold]")
    console.print(f"Description: {plugin.description}")
    console.print("\nCommands:")
    for cmd in plugin.commands:
        console.print(f"  • {cmd['name']}: {cmd['description']}")
    
    # Execute plugin command
    console.print("\n[bold]Executing plugin command:[/bold]")
    result = plugin.cmd_correlations("test_dataset", threshold=0.8)
    
    table = Table(title="Correlation Results")
    table.add_column("Column 1", style="cyan")
    table.add_column("Column 2", style="cyan")
    table.add_column("Correlation", style="yellow")
    
    for corr in result['correlations']:
        table.add_row(corr['col1'], corr['col2'], f"{corr['value']:.2f}")
    
    console.print(table)
    
    # Plugin manager usage
    console.print("\n[bold]Plugin Manager:[/bold]")
    manager = CLIPluginManager()
    manager.register_plugin(
        plugin.name,
        plugin,
        [cmd['name'] for cmd in plugin.commands]
    )
    
    # List plugins
    plugins = manager.list_plugins()
    console.print(f"Registered plugins: {len(plugins)}")
    for p in plugins:
        console.print(f"  • {p['name']} - {p['description']}")


def example_performance_comparison(temp_dir: Path):
    """Example of performance comparison between implementations."""
    console.print("Running performance comparison...")
    
    # Create datasets of different sizes
    sizes = [100, 1000, 5000]
    results = []
    
    for size in sizes:
        console.print(f"\n[bold]Testing with {size:,} rows:[/bold]")
        
        # Create data
        data = pd.DataFrame({
            'id': range(size),
            'value1': np.random.randn(size),
            'value2': np.random.exponential(1, size),
            'category': np.random.choice(['A', 'B', 'C', 'D'], size)
        })
        
        csv_path = temp_dir / f"perf_{size}.csv"
        data.to_csv(csv_path, index=False)
        
        # Time legacy implementation
        import time
        feature_flags.set("use_new_cli", False)
        clear_cli_cache()
        
        legacy_start = time.time()
        legacy_cmds = get_dataset_commands()
        legacy_cmds.list_datasets()  # Warm up
        legacy_cmds.register(
            name=f"perf_legacy_{size}",
            path=str(csv_path),
            force=True,
            generate_features=False
        )
        legacy_time = time.time() - legacy_start
        
        # Time new implementation
        feature_flags.set("use_new_cli", True)
        clear_cli_cache()
        
        new_start = time.time()
        new_cmds = get_dataset_commands()
        new_cmds.list_datasets()  # Warm up
        new_cmds.register(
            name=f"perf_new_{size}",
            path=str(csv_path),
            force=True,
            generate_features=False
        )
        new_time = time.time() - new_start
        
        # Calculate speedup
        speedup = legacy_time / new_time if new_time > 0 else 0
        
        results.append({
            'size': size,
            'legacy_time': legacy_time,
            'new_time': new_time,
            'speedup': speedup
        })
        
        console.print(f"  Legacy: {legacy_time:.3f}s")
        console.print(f"  New: {new_time:.3f}s")
        console.print(f"  Speedup: {speedup:.2f}x")
        
        # Cleanup
        cleanup_datasets([f"perf_legacy_{size}", f"perf_new_{size}"])
    
    # Display summary
    console.print("\n[bold]Performance Summary:[/bold]")
    
    table = Table(title="CLI Performance Comparison")
    table.add_column("Dataset Size", style="cyan")
    table.add_column("Legacy (s)", style="yellow")
    table.add_column("New (s)", style="green")
    table.add_column("Speedup", style="magenta")
    
    for result in results:
        speedup_str = f"{result['speedup']:.2f}x"
        if result['speedup'] > 1:
            speedup_str = f"[green]{speedup_str}[/green]"
        elif result['speedup'] < 1:
            speedup_str = f"[red]{speedup_str}[/red]"
        
        table.add_row(
            f"{result['size']:,}",
            f"{result['legacy_time']:.3f}",
            f"{result['new_time']:.3f}",
            speedup_str
        )
    
    console.print(table)
    
    avg_speedup = sum(r['speedup'] for r in results) / len(results)
    console.print(f"\nAverage speedup: [bold]{avg_speedup:.2f}x[/bold]")


def cleanup_datasets(dataset_names: list):
    """Clean up test datasets."""
    for name in dataset_names:
        try:
            # Try both implementations
            for use_new in [False, True]:
                feature_flags.set("use_new_cli", use_new)
                cmds = get_dataset_commands()
                if hasattr(cmds, 'remove'):
                    try:
                        cmds.remove(name, force=True)
                        break
                    except Exception:
                        pass
        except Exception:
            pass  # Ignore cleanup errors


if __name__ == "__main__":
    main()