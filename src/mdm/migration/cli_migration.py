"""CLI migration utilities for MDM refactoring.

This module provides tools for migrating CLI configurations and
testing command compatibility between implementations.
"""
import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json
import yaml
from datetime import datetime

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..core import feature_flags
from ..adapters.cli_manager import (
    get_dataset_commands,
    get_batch_commands,
    get_timeseries_commands,
    get_stats_commands,
    get_cli_formatter,
    get_cli_config
)
from ..core.exceptions import MDMError

logger = logging.getLogger(__name__)
console = Console()


class CLIMigrator:
    """Migrates CLI configurations and validates command compatibility."""
    
    def __init__(self):
        """Initialize CLI migrator."""
        self.legacy_config_path = Path.home() / '.mdm' / 'config.yaml'
        self.new_config_path = Path.home() / '.mdm' / 'cli_config.yaml'
        self._validation_results = []
        logger.info("Initialized CLIMigrator")
    
    def migrate_cli_config(self, dry_run: bool = True) -> Dict[str, Any]:
        """Migrate CLI configuration from legacy to new format.
        
        Args:
            dry_run: If True, only simulate migration
            
        Returns:
            Migration result
        """
        console.print("[bold]Migrating CLI Configuration[/bold]")
        
        result = {
            'success': True,
            'dry_run': dry_run,
            'source': str(self.legacy_config_path),
            'target': str(self.new_config_path),
            'migrated_settings': {},
            'warnings': []
        }
        
        try:
            # Load legacy config
            if not self.legacy_config_path.exists():
                result['warnings'].append("No legacy config file found")
                return result
            
            with open(self.legacy_config_path, 'r') as f:
                legacy_config = yaml.safe_load(f) or {}
            
            # Extract CLI-related settings
            cli_settings = self._extract_cli_settings(legacy_config)
            result['migrated_settings'] = cli_settings
            
            if dry_run:
                console.print("\n[yellow]DRY RUN - Configuration to be migrated:[/yellow]")
                self._display_config_diff(cli_settings)
            else:
                # Create new config
                self.new_config_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Load existing config if present
                existing = {}
                if self.new_config_path.exists():
                    with open(self.new_config_path, 'r') as f:
                        existing = yaml.safe_load(f) or {}
                
                # Merge settings
                existing.update(cli_settings)
                
                # Save new config
                with open(self.new_config_path, 'w') as f:
                    yaml.dump(existing, f, default_flow_style=False)
                
                console.print(f"\n[green]✓[/green] Configuration migrated to: [cyan]{self.new_config_path}[/cyan]")
            
            return result
            
        except Exception as e:
            logger.error(f"CLI config migration failed: {e}")
            result['success'] = False
            result['error'] = str(e)
            return result
    
    def validate_command_compatibility(
        self,
        command_group: str,
        commands: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Validate command compatibility between implementations.
        
        Args:
            command_group: Command group to test (dataset, batch, etc.)
            commands: Specific commands to test (all if None)
            
        Returns:
            Validation results
        """
        console.print(f"[bold]Validating {command_group} commands[/bold]")
        
        results = {
            'command_group': command_group,
            'total_commands': 0,
            'compatible': 0,
            'incompatible': 0,
            'commands': {}
        }
        
        try:
            # Get command handlers
            legacy_handler = self._get_command_handler(command_group, force_new=False)
            new_handler = self._get_command_handler(command_group, force_new=True)
            
            # Get available commands
            available_commands = self._get_available_commands(legacy_handler)
            if commands:
                available_commands = [cmd for cmd in available_commands if cmd in commands]
            
            results['total_commands'] = len(available_commands)
            
            # Test each command
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True
            ) as progress:
                task = progress.add_task("Testing commands...", total=len(available_commands))
                
                for cmd in available_commands:
                    cmd_result = self._test_command_compatibility(
                        cmd, legacy_handler, new_handler
                    )
                    
                    results['commands'][cmd] = cmd_result
                    if cmd_result['compatible']:
                        results['compatible'] += 1
                    else:
                        results['incompatible'] += 1
                    
                    progress.update(task, advance=1)
            
            # Display results
            self._display_compatibility_results(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Command validation failed: {e}")
            console.print(f"[red]Validation failed: {e}[/red]")
            return results
    
    def test_command_output(
        self,
        command_group: str,
        command: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Test command output differences between implementations.
        
        Args:
            command_group: Command group
            command: Command name
            **kwargs: Command arguments
            
        Returns:
            Test results with output comparison
        """
        console.print(f"[bold]Testing command: {command_group}.{command}[/bold]")
        
        result = {
            'command': f"{command_group}.{command}",
            'arguments': kwargs,
            'legacy_output': None,
            'new_output': None,
            'differences': [],
            'compatible': False
        }
        
        try:
            # Get handlers
            legacy_handler = self._get_command_handler(command_group, force_new=False)
            new_handler = self._get_command_handler(command_group, force_new=True)
            
            # Execute with legacy
            console.print("\n[yellow]Legacy implementation:[/yellow]")
            try:
                legacy_method = getattr(legacy_handler, command)
                result['legacy_output'] = legacy_method(**kwargs)
                console.print("[green]✓ Success[/green]")
            except Exception as e:
                result['legacy_output'] = {'error': str(e)}
                console.print(f"[red]✗ Error: {e}[/red]")
            
            # Execute with new
            console.print("\n[green]New implementation:[/green]")
            try:
                new_method = getattr(new_handler, command)
                result['new_output'] = new_method(**kwargs)
                console.print("[green]✓ Success[/green]")
            except Exception as e:
                result['new_output'] = {'error': str(e)}
                console.print(f"[red]✗ Error: {e}[/red]")
            
            # Compare outputs
            if result['legacy_output'] and result['new_output']:
                differences = self._compare_outputs(
                    result['legacy_output'],
                    result['new_output']
                )
                result['differences'] = differences
                result['compatible'] = len(differences) == 0
                
                if differences:
                    console.print("\n[yellow]Differences found:[/yellow]")
                    for diff in differences[:5]:  # Show first 5
                        console.print(f"  • {diff}")
                else:
                    console.print("\n[green]✓ Outputs are compatible[/green]")
            
            return result
            
        except Exception as e:
            logger.error(f"Command test failed: {e}")
            console.print(f"[red]Test failed: {e}[/red]")
            result['error'] = str(e)
            return result
    
    def generate_migration_report(
        self,
        output_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive migration report.
        
        Args:
            output_file: Optional file to save report
            
        Returns:
            Migration report
        """
        console.print("[bold]Generating CLI Migration Report[/bold]")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'feature_flags': {
                'use_new_cli': feature_flags.get('use_new_cli', False)
            },
            'command_groups': {},
            'configuration': {},
            'recommendations': []
        }
        
        # Test all command groups
        for group in ['dataset', 'batch', 'timeseries', 'stats']:
            console.print(f"\n[bold]Testing {group} commands...[/bold]")
            validation = self.validate_command_compatibility(group)
            report['command_groups'][group] = validation
        
        # Check configuration
        config_result = self.migrate_cli_config(dry_run=True)
        report['configuration'] = config_result
        
        # Generate recommendations
        report['recommendations'] = self._generate_recommendations(report)
        
        # Display summary
        self._display_migration_summary(report)
        
        # Save report if requested
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                if output_file.endswith('.yaml'):
                    yaml.dump(report, f, default_flow_style=False)
                else:
                    json.dump(report, f, indent=2, default=str)
            
            console.print(f"\n[green]✓[/green] Report saved to: [cyan]{output_path}[/cyan]")
        
        return report
    
    def _extract_cli_settings(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract CLI-related settings from legacy config."""
        cli_settings = {}
        
        # Direct CLI settings
        if 'cli' in config:
            cli_settings.update(config['cli'])
        
        # Output preferences
        if 'output' in config:
            cli_settings['output_format'] = config['output'].get('format', 'table')
            cli_settings['color'] = config['output'].get('color', True)
        
        # Command defaults
        if 'commands' in config:
            cli_settings['commands'] = config['commands']
        
        # Aliases
        if 'aliases' in config:
            cli_settings['aliases'] = config['aliases']
        
        return cli_settings
    
    def _get_command_handler(self, command_group: str, force_new: bool):
        """Get command handler for a group."""
        if command_group == 'dataset':
            return get_dataset_commands(force_new=force_new)
        elif command_group == 'batch':
            return get_batch_commands(force_new=force_new)
        elif command_group == 'timeseries':
            return get_timeseries_commands(force_new=force_new)
        elif command_group == 'stats':
            return get_stats_commands(force_new=force_new)
        else:
            raise ValueError(f"Unknown command group: {command_group}")
    
    def _get_available_commands(self, handler) -> List[str]:
        """Get available commands from a handler."""
        commands = []
        
        for name in dir(handler):
            if not name.startswith('_'):
                attr = getattr(handler, name)
                if callable(attr):
                    commands.append(name)
        
        return commands
    
    def _test_command_compatibility(
        self,
        command: str,
        legacy_handler,
        new_handler
    ) -> Dict[str, Any]:
        """Test compatibility of a single command."""
        result = {
            'command': command,
            'legacy_exists': hasattr(legacy_handler, command),
            'new_exists': hasattr(new_handler, command),
            'compatible': False,
            'issues': []
        }
        
        # Check existence
        if not result['legacy_exists']:
            result['issues'].append("Command not found in legacy implementation")
        if not result['new_exists']:
            result['issues'].append("Command not found in new implementation")
        
        if result['legacy_exists'] and result['new_exists']:
            # Compare signatures
            import inspect
            
            legacy_sig = inspect.signature(getattr(legacy_handler, command))
            new_sig = inspect.signature(getattr(new_handler, command))
            
            # Check parameters
            legacy_params = set(legacy_sig.parameters.keys())
            new_params = set(new_sig.parameters.keys())
            
            if legacy_params != new_params:
                missing = legacy_params - new_params
                extra = new_params - legacy_params
                
                if missing:
                    result['issues'].append(f"Missing parameters: {missing}")
                if extra:
                    result['issues'].append(f"Extra parameters: {extra}")
            else:
                result['compatible'] = True
        
        return result
    
    def _compare_outputs(self, output1: Any, output2: Any) -> List[str]:
        """Compare two command outputs."""
        differences = []
        
        # Handle error cases
        if isinstance(output1, dict) and 'error' in output1:
            if not (isinstance(output2, dict) and 'error' in output2):
                differences.append("Legacy errored but new succeeded")
        elif isinstance(output2, dict) and 'error' in output2:
            differences.append("New errored but legacy succeeded")
        
        # Compare types
        if type(output1) != type(output2):
            differences.append(f"Type mismatch: {type(output1).__name__} vs {type(output2).__name__}")
            return differences
        
        # Compare dictionaries
        if isinstance(output1, dict) and isinstance(output2, dict):
            all_keys = set(output1.keys()) | set(output2.keys())
            
            for key in all_keys:
                if key not in output1:
                    differences.append(f"Key '{key}' only in new output")
                elif key not in output2:
                    differences.append(f"Key '{key}' only in legacy output")
                elif output1[key] != output2[key]:
                    differences.append(f"Value mismatch for '{key}'")
        
        # Compare lists
        elif isinstance(output1, list) and isinstance(output2, list):
            if len(output1) != len(output2):
                differences.append(f"Length mismatch: {len(output1)} vs {len(output2)}")
        
        return differences
    
    def _display_config_diff(self, settings: Dict[str, Any]) -> None:
        """Display configuration differences."""
        table = Table(title="CLI Configuration to Migrate")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="yellow")
        
        for key, value in settings.items():
            if isinstance(value, dict):
                table.add_row(key, "{ ... }")
            else:
                table.add_row(key, str(value))
        
        console.print(table)
    
    def _display_compatibility_results(self, results: Dict[str, Any]) -> None:
        """Display command compatibility results."""
        table = Table(title=f"{results['command_group'].title()} Command Compatibility")
        table.add_column("Command", style="cyan")
        table.add_column("Status", style="yellow")
        table.add_column("Issues", style="red")
        
        for cmd, info in results['commands'].items():
            status = "[green]Compatible[/green]" if info['compatible'] else "[red]Incompatible[/red]"
            issues = ', '.join(info['issues']) if info['issues'] else "None"
            table.add_row(cmd, status, issues)
        
        console.print(table)
        
        # Summary
        console.print(f"\nTotal: {results['total_commands']} commands")
        console.print(f"Compatible: [green]{results['compatible']}[/green]")
        console.print(f"Incompatible: [red]{results['incompatible']}[/red]")
    
    def _display_migration_summary(self, report: Dict[str, Any]) -> None:
        """Display migration summary."""
        console.print("\n[bold]CLI Migration Summary[/bold]")
        
        # Command compatibility
        table = Table(title="Command Compatibility")
        table.add_column("Group", style="cyan")
        table.add_column("Total", style="yellow")
        table.add_column("Compatible", style="green")
        table.add_column("Issues", style="red")
        
        for group, info in report['command_groups'].items():
            table.add_row(
                group,
                str(info['total_commands']),
                str(info['compatible']),
                str(info['incompatible'])
            )
        
        console.print(table)
        
        # Recommendations
        if report['recommendations']:
            console.print("\n[bold]Recommendations:[/bold]")
            for rec in report['recommendations']:
                console.print(f"  • {rec}")
    
    def _generate_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate migration recommendations based on report."""
        recommendations = []
        
        # Check command compatibility
        total_incompatible = sum(
            info['incompatible'] 
            for info in report['command_groups'].values()
        )
        
        if total_incompatible == 0:
            recommendations.append("All commands are compatible - safe to migrate")
        else:
            recommendations.append(f"Fix {total_incompatible} incompatible commands before full migration")
        
        # Check configuration
        if report['configuration'].get('migrated_settings'):
            recommendations.append("Run configuration migration to preserve CLI settings")
        
        # Feature flag recommendation
        if not report['feature_flags']['use_new_cli']:
            recommendations.append("Enable 'use_new_cli' feature flag for gradual rollout")
        
        return recommendations


class CLIValidator:
    """Validates CLI functionality between implementations."""
    
    def __init__(self):
        """Initialize CLI validator."""
        self._test_results = []
        logger.info("Initialized CLIValidator")
    
    def validate_formatter_compatibility(self) -> Dict[str, Any]:
        """Validate formatter compatibility between implementations."""
        console.print("[bold]Validating CLI Formatter Compatibility[/bold]")
        
        results = {
            'compatible': True,
            'tests': []
        }
        
        # Get formatters
        legacy_formatter = get_cli_formatter(force_new=False)
        new_formatter = get_cli_formatter(force_new=True)
        
        # Test data
        test_data = [
            {'name': 'test1', 'value': 100, 'status': True},
            {'name': 'test2', 'value': 200, 'status': False}
        ]
        
        # Test table formatting
        test_result = {
            'test': 'table_format',
            'compatible': False,
            'issues': []
        }
        
        try:
            legacy_table = legacy_formatter.format_table(test_data, title="Test")
            new_table = new_formatter.format_table(test_data, title="Test")
            
            # Basic check - both should produce output
            if legacy_table and new_table:
                test_result['compatible'] = True
            else:
                test_result['issues'].append("One formatter produced no output")
                
        except Exception as e:
            test_result['issues'].append(str(e))
        
        results['tests'].append(test_result)
        
        # Test JSON formatting
        test_result = {
            'test': 'json_format',
            'compatible': False,
            'issues': []
        }
        
        try:
            legacy_json = legacy_formatter.format_json(test_data)
            new_json = new_formatter.format_json(test_data)
            
            # Parse and compare
            import json
            legacy_parsed = json.loads(legacy_json)
            new_parsed = json.loads(new_json)
            
            if legacy_parsed == new_parsed:
                test_result['compatible'] = True
            else:
                test_result['issues'].append("JSON output differs")
                
        except Exception as e:
            test_result['issues'].append(str(e))
        
        results['tests'].append(test_result)
        
        # Update overall compatibility
        results['compatible'] = all(test['compatible'] for test in results['tests'])
        
        # Display results
        table = Table(title="Formatter Compatibility")
        table.add_column("Test", style="cyan")
        table.add_column("Status", style="yellow")
        table.add_column("Issues", style="red")
        
        for test in results['tests']:
            status = "[green]✓[/green]" if test['compatible'] else "[red]✗[/red]"
            issues = ', '.join(test['issues']) if test['issues'] else "None"
            table.add_row(test['test'], status, issues)
        
        console.print(table)
        
        return results
    
    def validate_config_compatibility(self) -> Dict[str, Any]:
        """Validate configuration compatibility."""
        console.print("[bold]Validating CLI Configuration Compatibility[/bold]")
        
        results = {
            'compatible': True,
            'properties': {}
        }
        
        # Get configs
        legacy_config = get_cli_config(force_new=False)
        new_config = get_cli_config(force_new=True)
        
        # Test properties
        properties = [
            'output_format',
            'color_enabled',
            'verbose',
            'quiet'
        ]
        
        for prop in properties:
            try:
                legacy_value = getattr(legacy_config, prop)
                new_value = getattr(new_config, prop)
                
                compatible = type(legacy_value) == type(new_value)
                results['properties'][prop] = {
                    'compatible': compatible,
                    'legacy_type': type(legacy_value).__name__,
                    'new_type': type(new_value).__name__
                }
                
                if not compatible:
                    results['compatible'] = False
                    
            except Exception as e:
                results['properties'][prop] = {
                    'compatible': False,
                    'error': str(e)
                }
                results['compatible'] = False
        
        # Display results
        table = Table(title="Configuration Compatibility")
        table.add_column("Property", style="cyan")
        table.add_column("Compatible", style="yellow")
        table.add_column("Details", style="dim")
        
        for prop, info in results['properties'].items():
            status = "[green]✓[/green]" if info['compatible'] else "[red]✗[/red]"
            details = info.get('error', f"{info.get('legacy_type')} → {info.get('new_type')}")
            table.add_row(prop, status, details)
        
        console.print(table)
        
        return results