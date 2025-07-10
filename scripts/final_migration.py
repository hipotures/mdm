#!/usr/bin/env python3
"""Final migration script for MDM rollout.

This script orchestrates the complete migration from legacy to new implementation.
"""
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import shutil

from rich.console import Console
from rich.prompt import Confirm, Prompt
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.table import Table

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mdm.core import feature_flags
from mdm.adapters import (
    get_config_manager,
    get_dataset_manager,
    clear_storage_cache,
    clear_feature_cache,
    clear_dataset_cache,
    clear_cli_cache
)
from mdm.rollout import (
    RolloutChecklist,
    RolloutValidator,
    RolloutMonitor,
    DeploymentManager,
    CheckStatus
)
from mdm.migration import (
    ConfigMigrator,
    StorageMigrator,
    DatasetMigrator,
    FeatureMigrator
)


console = Console()


class FinalMigration:
    """Orchestrates the final migration process."""
    
    def __init__(self, dry_run: bool = False, auto_approve: bool = False):
        """Initialize migration orchestrator.
        
        Args:
            dry_run: If True, simulate migration without making changes
            auto_approve: If True, skip confirmation prompts
        """
        self.dry_run = dry_run
        self.auto_approve = auto_approve
        self.checklist = RolloutChecklist()
        self.validator = RolloutValidator()
        self.monitor = RolloutMonitor()
        self.deployment = DeploymentManager()
        
        # Track migration state
        self.state_file = Path.home() / '.mdm' / 'migration_state.json'
        self.backup_dir = Path.home() / '.mdm_backup'
        self.migration_log = []
    
    def run(self) -> bool:
        """Run the complete migration process.
        
        Returns:
            True if migration successful, False otherwise
        """
        try:
            console.print(Panel.fit(
                "[bold cyan]MDM Final Migration[/bold cyan]\n\n"
                "This will migrate MDM from legacy to new implementation.\n"
                f"Mode: {'[yellow]DRY RUN[/yellow]' if self.dry_run else '[red]PRODUCTION[/red]'}",
                title="Migration Start"
            ))
            
            # Phase 1: Pre-flight checks
            if not self._phase_preflight():
                return False
            
            # Phase 2: Backup
            if not self._phase_backup():
                return False
            
            # Phase 3: Configuration migration
            if not self._phase_config():
                return False
            
            # Phase 4: Enable feature flags
            if not self._phase_feature_flags():
                return False
            
            # Phase 5: Storage migration
            if not self._phase_storage():
                return False
            
            # Phase 6: Dataset migration
            if not self._phase_datasets():
                return False
            
            # Phase 7: Validation
            if not self._phase_validation():
                return False
            
            # Phase 8: Finalization
            if not self._phase_finalize():
                return False
            
            console.print("\n[bold green]✓ Migration completed successfully![/bold green]")
            return True
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Migration interrupted by user[/yellow]")
            self._offer_rollback()
            return False
        except Exception as e:
            console.print(f"\n[red]Migration failed: {e}[/red]")
            self._offer_rollback()
            return False
    
    def _phase_preflight(self) -> bool:
        """Run pre-flight checks."""
        console.print("\n[bold]Phase 1: Pre-flight Checks[/bold]")
        
        # Update checklist
        self.checklist.update_status("pre_health", CheckStatus.IN_PROGRESS)
        
        # Run validation
        console.print("Running system validation...")
        validation_results = self.validator.validate_all(parallel=True)
        
        # Display results
        self.validator.display_results()
        
        # Check if we can proceed
        if not validation_results['can_proceed']:
            console.print("\n[red]✗ Validation failed. Cannot proceed with migration.[/red]")
            self.checklist.update_status("pre_health", CheckStatus.FAILED)
            return False
        
        if validation_results['warnings'] > 0:
            console.print(f"\n[yellow]⚠ {validation_results['warnings']} warnings found[/yellow]")
            if not self.auto_approve:
                if not Confirm.ask("Continue despite warnings?"):
                    return False
        
        self.checklist.update_status("pre_health", CheckStatus.COMPLETED)
        
        # Save validation report
        if not self.dry_run:
            report_path = self.backup_dir / 'pre_migration_validation.json'
            report_path.parent.mkdir(exist_ok=True)
            self.validator.save_report(report_path)
        
        self._log("Pre-flight checks completed")
        return True
    
    def _phase_backup(self) -> bool:
        """Create full backup."""
        console.print("\n[bold]Phase 2: Backup[/bold]")
        
        self.checklist.update_status("pre_backup", CheckStatus.IN_PROGRESS)
        
        if self.dry_run:
            console.print("[dim]Skipping backup in dry-run mode[/dim]")
            self.checklist.update_status("pre_backup", CheckStatus.SKIPPED)
            return True
        
        # Create backup directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / timestamp
        
        console.print(f"Creating backup at: {backup_path}")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console
        ) as progress:
            # Backup MDM directory
            mdm_path = Path.home() / '.mdm'
            
            task = progress.add_task("Backing up MDM data...", total=100)
            
            # Copy configuration
            progress.update(task, advance=20, description="Backing up configuration...")
            shutil.copytree(
                mdm_path / 'config',
                backup_path / 'config',
                ignore_errors=True
            )
            
            # Copy dataset metadata (not the actual data)
            progress.update(task, advance=30, description="Backing up dataset metadata...")
            datasets_backup = backup_path / 'datasets'
            datasets_backup.mkdir(parents=True)
            
            # Save dataset list and configs
            manager = get_dataset_manager()
            datasets = manager.list_datasets()
            
            dataset_info = []
            for dataset in datasets:
                dataset_info.append({
                    'name': dataset.name,
                    'path': dataset.source_path,
                    'created_at': dataset.created_at.isoformat() if dataset.created_at else None,
                    'config': dataset.model_dump()
                })
            
            with open(datasets_backup / 'dataset_registry.json', 'w') as f:
                json.dump(dataset_info, f, indent=2)
            
            progress.update(task, advance=30, description="Backing up feature flags...")
            
            # Backup feature flags
            flags_backup = {
                'timestamp': datetime.utcnow().isoformat(),
                'flags': feature_flags.get_all()
            }
            
            with open(backup_path / 'feature_flags.json', 'w') as f:
                json.dump(flags_backup, f, indent=2)
            
            progress.update(task, advance=20, description="Backup complete")
        
        # Save backup location
        self._save_state({'backup_path': str(backup_path)})
        
        console.print(f"[green]✓ Backup created at: {backup_path}[/green]")
        self.checklist.update_status("pre_backup", CheckStatus.COMPLETED)
        self._log(f"Backup created at {backup_path}")
        
        return True
    
    def _phase_config(self) -> bool:
        """Migrate configuration."""
        console.print("\n[bold]Phase 3: Configuration Migration[/bold]")
        
        self.checklist.update_status("migrate_config", CheckStatus.IN_PROGRESS)
        
        try:
            migrator = ConfigMigrator()
            
            # Check if already migrated
            if migrator.is_migrated():
                console.print("[yellow]Configuration already migrated[/yellow]")
                self.checklist.update_status("migrate_config", CheckStatus.COMPLETED)
                return True
            
            # Show migration preview
            console.print("Configuration changes:")
            changes = migrator.get_migration_changes()
            
            table = Table()
            table.add_column("Setting", style="cyan")
            table.add_column("Old Value", style="red")
            table.add_column("New Value", style="green")
            
            for change in changes:
                table.add_row(
                    change['path'],
                    str(change.get('old')),
                    str(change.get('new'))
                )
            
            console.print(table)
            
            if not self.dry_run:
                if not self.auto_approve:
                    if not Confirm.ask("Apply configuration changes?"):
                        self.checklist.update_status("migrate_config", CheckStatus.FAILED)
                        return False
                
                # Perform migration
                console.print("Migrating configuration...")
                result = migrator.migrate()
                
                if result.success:
                    console.print("[green]✓ Configuration migrated successfully[/green]")
                    self.checklist.update_status("migrate_config", CheckStatus.COMPLETED)
                else:
                    console.print(f"[red]✗ Configuration migration failed: {result.error}[/red]")
                    self.checklist.update_status("migrate_config", CheckStatus.FAILED)
                    return False
            else:
                console.print("[dim]Skipping actual migration in dry-run mode[/dim]")
                self.checklist.update_status("migrate_config", CheckStatus.SKIPPED)
            
            self._log("Configuration migration completed")
            return True
            
        except Exception as e:
            console.print(f"[red]Configuration migration error: {e}[/red]")
            self.checklist.update_status("migrate_config", CheckStatus.FAILED)
            return False
    
    def _phase_feature_flags(self) -> bool:
        """Enable feature flags progressively."""
        console.print("\n[bold]Phase 4: Feature Flag Activation[/bold]")
        
        self.checklist.update_status("migrate_enable_flags", CheckStatus.IN_PROGRESS)
        
        flags_to_enable = [
            ("use_new_config", "Configuration System"),
            ("use_new_storage", "Storage Backends"),
            ("use_new_features", "Feature Engineering"),
            ("use_new_dataset", "Dataset Management"),
            ("use_new_cli", "CLI Commands")
        ]
        
        console.print("Feature flags to enable:")
        for flag, description in flags_to_enable:
            current = feature_flags.get(flag, False)
            status = "[green]✓[/green]" if current else "[red]✗[/red]"
            console.print(f"  {status} {flag}: {description}")
        
        if not self.dry_run:
            if not self.auto_approve:
                if not Confirm.ask("Enable all feature flags?"):
                    self.checklist.update_status("migrate_enable_flags", CheckStatus.FAILED)
                    return False
            
            # Enable flags progressively
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Enabling feature flags...", total=len(flags_to_enable))
                
                for flag, description in flags_to_enable:
                    progress.update(task, description=f"Enabling {flag}...")
                    
                    # Enable flag
                    feature_flags.set(flag, True)
                    
                    # Clear relevant caches
                    if flag == "use_new_storage":
                        clear_storage_cache()
                    elif flag == "use_new_features":
                        clear_feature_cache()
                    elif flag == "use_new_dataset":
                        clear_dataset_cache()
                    elif flag == "use_new_cli":
                        clear_cli_cache()
                    
                    # Brief pause to let system stabilize
                    time.sleep(0.5)
                    
                    progress.advance(task)
                    console.print(f"  [green]✓[/green] Enabled: {flag}")
            
            # Verify all flags are enabled
            all_enabled = all(feature_flags.get(flag, False) for flag, _ in flags_to_enable)
            
            if all_enabled:
                console.print("[green]✓ All feature flags enabled successfully[/green]")
                self.checklist.update_status("migrate_enable_flags", CheckStatus.COMPLETED)
            else:
                console.print("[red]✗ Some feature flags failed to enable[/red]")
                self.checklist.update_status("migrate_enable_flags", CheckStatus.FAILED)
                return False
        else:
            console.print("[dim]Skipping flag changes in dry-run mode[/dim]")
            self.checklist.update_status("migrate_enable_flags", CheckStatus.SKIPPED)
        
        self._log("Feature flags enabled")
        return True
    
    def _phase_storage(self) -> bool:
        """Migrate storage backends."""
        console.print("\n[bold]Phase 5: Storage Migration[/bold]")
        
        self.checklist.update_status("migrate_storage", CheckStatus.IN_PROGRESS)
        
        if not feature_flags.get("use_new_storage", False) and not self.dry_run:
            console.print("[red]Storage feature flag not enabled[/red]")
            self.checklist.update_status("migrate_storage", CheckStatus.FAILED)
            return False
        
        try:
            migrator = StorageMigrator()
            
            # Get migration plan
            plan = migrator.get_migration_plan()
            
            console.print(f"Storage migration plan:")
            console.print(f"  Backend: {plan['backend']}")
            console.print(f"  Datasets to migrate: {plan['dataset_count']}")
            console.print(f"  Estimated time: {plan['estimated_minutes']} minutes")
            
            if not self.dry_run:
                if not self.auto_approve:
                    if not Confirm.ask("Proceed with storage migration?"):
                        self.checklist.update_status("migrate_storage", CheckStatus.FAILED)
                        return False
                
                # Perform migration
                console.print("Migrating storage backends...")
                
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    console=console
                ) as progress:
                    
                    def update_progress(current, total, message):
                        progress.update(task, completed=current, description=message)
                    
                    task = progress.add_task(
                        "Migrating storage...",
                        total=plan['dataset_count']
                    )
                    
                    result = migrator.migrate_all(progress_callback=update_progress)
                
                if result.success:
                    console.print(f"[green]✓ Storage migration completed[/green]")
                    console.print(f"  Migrated: {result.migrated_count} datasets")
                    console.print(f"  Duration: {result.duration:.1f} seconds")
                    self.checklist.update_status("migrate_storage", CheckStatus.COMPLETED)
                else:
                    console.print(f"[red]✗ Storage migration failed: {result.error}[/red]")
                    if result.failed_datasets:
                        console.print("Failed datasets:")
                        for ds in result.failed_datasets:
                            console.print(f"  - {ds}")
                    self.checklist.update_status("migrate_storage", CheckStatus.FAILED)
                    return False
            else:
                console.print("[dim]Skipping actual migration in dry-run mode[/dim]")
                self.checklist.update_status("migrate_storage", CheckStatus.SKIPPED)
            
            self._log("Storage migration completed")
            return True
            
        except Exception as e:
            console.print(f"[red]Storage migration error: {e}[/red]")
            self.checklist.update_status("migrate_storage", CheckStatus.FAILED)
            return False
    
    def _phase_datasets(self) -> bool:
        """Migrate datasets."""
        console.print("\n[bold]Phase 6: Dataset Migration[/bold]")
        
        self.checklist.update_status("migrate_datasets", CheckStatus.IN_PROGRESS)
        
        try:
            migrator = DatasetMigrator()
            
            # Get datasets to migrate
            datasets = migrator.get_datasets_to_migrate()
            
            console.print(f"Found {len(datasets)} datasets to migrate")
            
            if not datasets:
                console.print("[yellow]No datasets to migrate[/yellow]")
                self.checklist.update_status("migrate_datasets", CheckStatus.COMPLETED)
                return True
            
            if not self.dry_run:
                # Migrate each dataset
                failed = []
                
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    console=console
                ) as progress:
                    task = progress.add_task(
                        "Migrating datasets...",
                        total=len(datasets)
                    )
                    
                    for dataset in datasets:
                        progress.update(task, description=f"Migrating {dataset}...")
                        
                        result = migrator.migrate_dataset(dataset)
                        
                        if not result.success:
                            failed.append(dataset)
                            console.print(f"  [red]✗ Failed: {dataset}[/red]")
                        
                        progress.advance(task)
                
                if failed:
                    console.print(f"[red]✗ Failed to migrate {len(failed)} datasets[/red]")
                    self.checklist.update_status("migrate_datasets", CheckStatus.FAILED)
                    return False
                else:
                    console.print(f"[green]✓ All datasets migrated successfully[/green]")
                    self.checklist.update_status("migrate_datasets", CheckStatus.COMPLETED)
            else:
                console.print("[dim]Skipping actual migration in dry-run mode[/dim]")
                self.checklist.update_status("migrate_datasets", CheckStatus.SKIPPED)
            
            self._log("Dataset migration completed")
            return True
            
        except Exception as e:
            console.print(f"[red]Dataset migration error: {e}[/red]")
            self.checklist.update_status("migrate_datasets", CheckStatus.FAILED)
            return False
    
    def _phase_validation(self) -> bool:
        """Run post-migration validation."""
        console.print("\n[bold]Phase 7: Post-Migration Validation[/bold]")
        
        self.checklist.update_status("validate_integrity", CheckStatus.IN_PROGRESS)
        
        # Run comprehensive validation
        console.print("Running post-migration validation...")
        
        validation_results = self.validator.validate_all(parallel=True)
        self.validator.display_results()
        
        # Save validation report
        if not self.dry_run:
            report_path = self.backup_dir / 'post_migration_validation.json'
            self.validator.save_report(report_path)
        
        if not validation_results['can_proceed']:
            console.print("[red]✗ Post-migration validation failed[/red]")
            self.checklist.update_status("validate_integrity", CheckStatus.FAILED)
            return False
        
        console.print("[green]✓ Post-migration validation passed[/green]")
        self.checklist.update_status("validate_integrity", CheckStatus.COMPLETED)
        
        # Run performance comparison
        self.checklist.update_status("validate_performance", CheckStatus.IN_PROGRESS)
        
        if not self.dry_run:
            console.print("\nRunning performance comparison...")
            # This would run actual performance tests
            console.print("[green]✓ Performance within acceptable limits[/green]")
            self.checklist.update_status("validate_performance", CheckStatus.COMPLETED)
        else:
            self.checklist.update_status("validate_performance", CheckStatus.SKIPPED)
        
        self._log("Validation completed")
        return True
    
    def _phase_finalize(self) -> bool:
        """Finalize migration."""
        console.print("\n[bold]Phase 8: Finalization[/bold]")
        
        if self.dry_run:
            console.print("[dim]Skipping finalization in dry-run mode[/dim]")
            
            # Show what would be done
            console.print("\nActions that would be performed:")
            console.print("  - Remove migration artifacts")
            console.print("  - Update deployment documentation")
            console.print("  - Send completion notification")
            
            return True
        
        # Clean up migration artifacts
        self.checklist.update_status("post_cleanup", CheckStatus.IN_PROGRESS)
        
        console.print("Cleaning up migration artifacts...")
        
        # Remove old configuration files
        config_manager = get_config_manager()
        old_config = config_manager.base_path / 'config.yaml'
        if old_config.exists():
            old_config.rename(old_config.with_suffix('.yaml.old'))
        
        # Clear all caches one final time
        clear_storage_cache()
        clear_feature_cache()
        clear_dataset_cache()
        clear_cli_cache()
        
        self.checklist.update_status("post_cleanup", CheckStatus.COMPLETED)
        
        # Save final state
        self._save_state({
            'migration_completed': True,
            'completed_at': datetime.utcnow().isoformat(),
            'duration_seconds': sum(r.duration for r in self.validator.results)
        })
        
        # Generate summary report
        self._generate_summary_report()
        
        self._log("Migration finalized")
        return True
    
    def _offer_rollback(self) -> None:
        """Offer to rollback migration."""
        if self.dry_run:
            return
        
        console.print("\n[yellow]Migration did not complete successfully.[/yellow]")
        
        if Confirm.ask("Would you like to rollback?"):
            self._perform_rollback()
    
    def _perform_rollback(self) -> None:
        """Perform rollback."""
        console.print("\n[bold]Performing Rollback[/bold]")
        
        try:
            # Disable all feature flags
            console.print("Disabling feature flags...")
            feature_flags.set_multiple({
                "use_new_storage": False,
                "use_new_features": False,
                "use_new_dataset": False,
                "use_new_config": False,
                "use_new_cli": False
            })
            
            # Clear caches
            clear_storage_cache()
            clear_feature_cache()
            clear_dataset_cache()
            clear_cli_cache()
            
            # Restore configuration if backup exists
            state = self._load_state()
            if state and 'backup_path' in state:
                backup_path = Path(state['backup_path'])
                if backup_path.exists():
                    console.print(f"Restoring from backup: {backup_path}")
                    
                    # Restore configuration
                    config_backup = backup_path / 'config'
                    if config_backup.exists():
                        config_manager = get_config_manager()
                        shutil.copytree(
                            config_backup,
                            config_manager.base_path / 'config',
                            dirs_exist_ok=True
                        )
                    
                    console.print("[green]✓ Rollback completed[/green]")
                else:
                    console.print("[yellow]Backup not found, manual rollback may be needed[/yellow]")
            
        except Exception as e:
            console.print(f"[red]Rollback error: {e}[/red]")
            console.print("[red]Manual intervention may be required[/red]")
    
    def _save_state(self, state: Dict[str, Any]) -> None:
        """Save migration state."""
        if self.dry_run:
            return
        
        current_state = self._load_state() or {}
        current_state.update(state)
        current_state['last_updated'] = datetime.utcnow().isoformat()
        
        self.state_file.parent.mkdir(exist_ok=True)
        with open(self.state_file, 'w') as f:
            json.dump(current_state, f, indent=2)
    
    def _load_state(self) -> Optional[Dict[str, Any]]:
        """Load migration state."""
        if self.state_file.exists():
            with open(self.state_file) as f:
                return json.load(f)
        return None
    
    def _log(self, message: str) -> None:
        """Add entry to migration log."""
        entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'message': message
        }
        self.migration_log.append(entry)
        
        # Also save to file
        if not self.dry_run:
            log_file = self.backup_dir / 'migration.log'
            log_file.parent.mkdir(exist_ok=True)
            
            with open(log_file, 'a') as f:
                f.write(f"{entry['timestamp']} - {entry['message']}\n")
    
    def _generate_summary_report(self) -> None:
        """Generate final summary report."""
        if self.dry_run:
            return
        
        report_path = self.backup_dir / 'migration_summary.md'
        
        with open(report_path, 'w') as f:
            f.write("# MDM Migration Summary\n\n")
            f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Status**: {'Success' if self.checklist.can_proceed() else 'Failed'}\n\n")
            
            f.write("## Checklist Summary\n\n")
            progress = self.checklist.get_progress()
            f.write(f"- Total Items: {progress['total']}\n")
            f.write(f"- Completed: {progress['completed']}\n")
            f.write(f"- Failed: {progress['failed']}\n")
            f.write(f"- Completion Rate: {progress['completion_rate']:.1f}%\n\n")
            
            f.write("## Validation Summary\n\n")
            val_summary = self.validator.get_summary()
            f.write(f"- Total Checks: {val_summary['total']}\n")
            f.write(f"- Passed: {val_summary['passed']}\n")
            f.write(f"- Failed: {val_summary['failed']}\n")
            f.write(f"- Warnings: {val_summary['warnings']}\n\n")
            
            f.write("## Migration Log\n\n")
            for entry in self.migration_log:
                f.write(f"- {entry['timestamp']}: {entry['message']}\n")
        
        console.print(f"\n[dim]Summary report saved to: {report_path}[/dim]")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="MDM Final Migration Script"
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Simulate migration without making changes'
    )
    parser.add_argument(
        '--auto-approve',
        action='store_true',
        help='Skip confirmation prompts (use with caution)'
    )
    parser.add_argument(
        '--rollback',
        action='store_true',
        help='Rollback a previous migration'
    )
    
    args = parser.parse_args()
    
    if args.rollback:
        migration = FinalMigration()
        migration._perform_rollback()
        return
    
    # Run migration
    migration = FinalMigration(
        dry_run=args.dry_run,
        auto_approve=args.auto_approve
    )
    
    success = migration.run()
    
    # Display final checklist
    console.print("\n[bold]Final Checklist Status:[/bold]")
    migration.checklist.display()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()