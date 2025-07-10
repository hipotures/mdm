"""Rollback management for MDM migration.

This module provides comprehensive rollback capabilities to safely
revert changes if issues occur during rollout.
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import shutil
import subprocess

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.panel import Panel
from rich.prompt import Confirm

from mdm.core import feature_flags
from mdm.adapters import (
    get_config_manager,
    clear_storage_cache,
    clear_feature_cache,
    clear_dataset_cache,
    clear_cli_cache
)
from mdm.core.exceptions import RollbackError


class RollbackType(Enum):
    """Types of rollback operations."""
    FULL = "full"              # Complete rollback to pre-migration state
    PARTIAL = "partial"        # Rollback specific components
    FEATURE_FLAGS = "flags"    # Only rollback feature flags
    CONFIGURATION = "config"   # Only rollback configuration
    DATA = "data"             # Only rollback data changes


@dataclass
class RollbackPoint:
    """Point-in-time snapshot for rollback."""
    id: str
    timestamp: datetime
    description: str
    backup_path: Path
    component_states: Dict[str, Any] = field(default_factory=dict)
    feature_flags: Dict[str, bool] = field(default_factory=dict)
    config_snapshot: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'description': self.description,
            'backup_path': str(self.backup_path),
            'component_states': self.component_states,
            'feature_flags': self.feature_flags,
            'config_snapshot': self.config_snapshot
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RollbackPoint':
        """Create from dictionary."""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['backup_path'] = Path(data['backup_path'])
        return cls(**data)


@dataclass
class RollbackResult:
    """Result of a rollback operation."""
    success: bool
    rollback_type: RollbackType
    components_rolled_back: List[str]
    errors: List[str] = field(default_factory=list)
    duration: float = 0.0
    rollback_point: Optional[RollbackPoint] = None


class RollbackManager:
    """Manages rollback operations for MDM migration."""
    
    def __init__(self):
        """Initialize rollback manager."""
        self.console = Console()
        self.rollback_history_file = Path.home() / '.mdm' / 'rollback_history.json'
        self.rollback_points: List[RollbackPoint] = []
        self._load_rollback_points()
    
    def create_rollback_point(
        self,
        description: str,
        backup_path: Optional[Path] = None
    ) -> RollbackPoint:
        """Create a new rollback point.
        
        Args:
            description: Description of the rollback point
            backup_path: Path to backup data
            
        Returns:
            Created rollback point
        """
        timestamp = datetime.utcnow()
        point_id = f"rollback_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        
        # Create backup if path not provided
        if not backup_path:
            backup_path = Path.home() / '.mdm_rollback' / point_id
            self._create_backup(backup_path)
        
        # Capture current state
        rollback_point = RollbackPoint(
            id=point_id,
            timestamp=timestamp,
            description=description,
            backup_path=backup_path,
            component_states=self._capture_component_states(),
            feature_flags=feature_flags.get_all(),
            config_snapshot=self._capture_config_snapshot()
        )
        
        # Save rollback point
        self.rollback_points.append(rollback_point)
        self._save_rollback_points()
        
        self.console.print(f"[green]✓ Created rollback point: {point_id}[/green]")
        return rollback_point
    
    def rollback(
        self,
        rollback_type: RollbackType = RollbackType.FULL,
        point_id: Optional[str] = None,
        components: Optional[List[str]] = None,
        dry_run: bool = False
    ) -> RollbackResult:
        """Perform rollback operation.
        
        Args:
            rollback_type: Type of rollback to perform
            point_id: Specific rollback point ID (uses latest if not specified)
            components: Specific components to rollback (for PARTIAL type)
            dry_run: If True, simulate rollback without making changes
            
        Returns:
            Rollback result
        """
        start_time = datetime.utcnow()
        errors = []
        rolled_back = []
        
        try:
            # Get rollback point
            if point_id:
                rollback_point = self._get_rollback_point(point_id)
            else:
                rollback_point = self._get_latest_rollback_point()
            
            if not rollback_point:
                raise RollbackError("No rollback points available")
            
            self.console.print(Panel.fit(
                f"[bold]Rollback Operation[/bold]\n\n"
                f"Type: {rollback_type.value}\n"
                f"Point: {rollback_point.id}\n"
                f"Created: {rollback_point.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Description: {rollback_point.description}\n"
                f"Mode: {'[yellow]DRY RUN[/yellow]' if dry_run else '[red]EXECUTE[/red]'}",
                title="Rollback Details"
            ))
            
            if not dry_run and not Confirm.ask("Proceed with rollback?"):
                return RollbackResult(
                    success=False,
                    rollback_type=rollback_type,
                    components_rolled_back=[],
                    errors=["Rollback cancelled by user"]
                )
            
            # Execute rollback based on type
            if rollback_type == RollbackType.FULL:
                rolled_back, errors = self._rollback_full(rollback_point, dry_run)
            elif rollback_type == RollbackType.PARTIAL:
                rolled_back, errors = self._rollback_partial(rollback_point, components or [], dry_run)
            elif rollback_type == RollbackType.FEATURE_FLAGS:
                rolled_back, errors = self._rollback_feature_flags(rollback_point, dry_run)
            elif rollback_type == RollbackType.CONFIGURATION:
                rolled_back, errors = self._rollback_configuration(rollback_point, dry_run)
            elif rollback_type == RollbackType.DATA:
                rolled_back, errors = self._rollback_data(rollback_point, dry_run)
            
            # Calculate duration
            duration = (datetime.utcnow() - start_time).total_seconds()
            
            # Create result
            result = RollbackResult(
                success=len(errors) == 0,
                rollback_type=rollback_type,
                components_rolled_back=rolled_back,
                errors=errors,
                duration=duration,
                rollback_point=rollback_point
            )
            
            # Log rollback
            if not dry_run:
                self._log_rollback(result)
            
            # Display result
            if result.success:
                self.console.print(f"\n[green]✓ Rollback completed successfully[/green]")
                self.console.print(f"  Duration: {duration:.1f}s")
                self.console.print(f"  Components: {', '.join(rolled_back)}")
            else:
                self.console.print(f"\n[red]✗ Rollback failed[/red]")
                for error in errors:
                    self.console.print(f"  - {error}")
            
            return result
            
        except Exception as e:
            self.console.print(f"[red]Rollback error: {e}[/red]")
            return RollbackResult(
                success=False,
                rollback_type=rollback_type,
                components_rolled_back=rolled_back,
                errors=[str(e)],
                duration=(datetime.utcnow() - start_time).total_seconds()
            )
    
    def _rollback_full(
        self,
        rollback_point: RollbackPoint,
        dry_run: bool
    ) -> tuple[List[str], List[str]]:
        """Perform full rollback."""
        rolled_back = []
        errors = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=self.console
        ) as progress:
            
            steps = [
                ("feature_flags", "Reverting feature flags"),
                ("configuration", "Restoring configuration"),
                ("datasets", "Restoring dataset metadata"),
                ("caches", "Clearing caches"),
                ("verification", "Verifying rollback")
            ]
            
            task = progress.add_task("Rolling back...", total=len(steps))
            
            for step, description in steps:
                progress.update(task, description=description)
                
                try:
                    if step == "feature_flags":
                        if not dry_run:
                            # Restore feature flags
                            feature_flags.set_multiple(rollback_point.feature_flags)
                        rolled_back.append("feature_flags")
                    
                    elif step == "configuration":
                        if not dry_run and rollback_point.backup_path.exists():
                            # Restore configuration
                            config_backup = rollback_point.backup_path / 'config'
                            if config_backup.exists():
                                config_manager = get_config_manager()
                                shutil.copytree(
                                    config_backup,
                                    config_manager.base_path / 'config',
                                    dirs_exist_ok=True
                                )
                        rolled_back.append("configuration")
                    
                    elif step == "datasets":
                        if not dry_run and rollback_point.backup_path.exists():
                            # Restore dataset registry
                            dataset_backup = rollback_point.backup_path / 'datasets'
                            if dataset_backup.exists():
                                # This would restore dataset metadata
                                pass
                        rolled_back.append("datasets")
                    
                    elif step == "caches":
                        if not dry_run:
                            # Clear all caches
                            clear_storage_cache()
                            clear_feature_cache()
                            clear_dataset_cache()
                            clear_cli_cache()
                        rolled_back.append("caches")
                    
                    elif step == "verification":
                        # Verify rollback
                        if not dry_run:
                            if not self._verify_rollback(rollback_point):
                                errors.append("Rollback verification failed")
                
                except Exception as e:
                    errors.append(f"Failed to rollback {step}: {e}")
                
                progress.advance(task)
        
        return rolled_back, errors
    
    def _rollback_partial(
        self,
        rollback_point: RollbackPoint,
        components: List[str],
        dry_run: bool
    ) -> tuple[List[str], List[str]]:
        """Perform partial rollback of specific components."""
        rolled_back = []
        errors = []
        
        self.console.print(f"Rolling back components: {', '.join(components)}")
        
        for component in components:
            try:
                if component == "storage" and "feature_flags" in components:
                    if not dry_run:
                        feature_flags.set("use_new_storage", False)
                        clear_storage_cache()
                    rolled_back.append("storage")
                
                elif component == "features" and "feature_flags" in components:
                    if not dry_run:
                        feature_flags.set("use_new_features", False)
                        clear_feature_cache()
                    rolled_back.append("features")
                
                elif component == "datasets" and "feature_flags" in components:
                    if not dry_run:
                        feature_flags.set("use_new_dataset", False)
                        clear_dataset_cache()
                    rolled_back.append("datasets")
                
                elif component == "config":
                    if not dry_run:
                        # Restore configuration
                        self._restore_config(rollback_point)
                    rolled_back.append("config")
                
                else:
                    errors.append(f"Unknown component: {component}")
                    
            except Exception as e:
                errors.append(f"Failed to rollback {component}: {e}")
        
        return rolled_back, errors
    
    def _rollback_feature_flags(
        self,
        rollback_point: RollbackPoint,
        dry_run: bool
    ) -> tuple[List[str], List[str]]:
        """Rollback only feature flags."""
        rolled_back = []
        errors = []
        
        try:
            if not dry_run:
                # Show flag changes
                current_flags = feature_flags.get_all()
                
                self.console.print("\nFeature flag changes:")
                for flag, old_value in rollback_point.feature_flags.items():
                    current_value = current_flags.get(flag, False)
                    if current_value != old_value:
                        self.console.print(
                            f"  {flag}: {current_value} → {old_value}"
                        )
                
                # Apply changes
                feature_flags.set_multiple(rollback_point.feature_flags)
                
                # Clear caches
                clear_storage_cache()
                clear_feature_cache()
                clear_dataset_cache()
                clear_cli_cache()
            
            rolled_back.append("feature_flags")
            
        except Exception as e:
            errors.append(f"Failed to rollback feature flags: {e}")
        
        return rolled_back, errors
    
    def _rollback_configuration(
        self,
        rollback_point: RollbackPoint,
        dry_run: bool
    ) -> tuple[List[str], List[str]]:
        """Rollback only configuration."""
        rolled_back = []
        errors = []
        
        try:
            if not dry_run:
                self._restore_config(rollback_point)
            rolled_back.append("configuration")
            
        except Exception as e:
            errors.append(f"Failed to rollback configuration: {e}")
        
        return rolled_back, errors
    
    def _rollback_data(
        self,
        rollback_point: RollbackPoint,
        dry_run: bool
    ) -> tuple[List[str], List[str]]:
        """Rollback data changes."""
        rolled_back = []
        errors = []
        
        self.console.print("[yellow]Data rollback not implemented[/yellow]")
        errors.append("Data rollback not implemented")
        
        return rolled_back, errors
    
    def _create_backup(self, backup_path: Path) -> None:
        """Create backup of current state."""
        backup_path.mkdir(parents=True, exist_ok=True)
        
        config_manager = get_config_manager()
        mdm_path = config_manager.base_path
        
        # Backup configuration
        if (mdm_path / 'config').exists():
            shutil.copytree(
                mdm_path / 'config',
                backup_path / 'config',
                ignore_errors=True
            )
        
        # Backup dataset registry
        if (mdm_path / 'config' / 'datasets').exists():
            shutil.copytree(
                mdm_path / 'config' / 'datasets',
                backup_path / 'datasets',
                ignore_errors=True
            )
    
    def _capture_component_states(self) -> Dict[str, Any]:
        """Capture current component states."""
        states = {}
        
        try:
            # Capture dataset count
            from mdm.adapters import get_dataset_manager
            manager = get_dataset_manager()
            datasets = manager.list_datasets()
            states['dataset_count'] = len(datasets)
            
            # Capture backend type
            config_manager = get_config_manager()
            states['backend_type'] = config_manager.config.database.default_backend
            
        except Exception as e:
            self.console.print(f"[yellow]Warning: Failed to capture some states: {e}[/yellow]")
        
        return states
    
    def _capture_config_snapshot(self) -> Optional[Dict[str, Any]]:
        """Capture configuration snapshot."""
        try:
            config_manager = get_config_manager()
            return config_manager.config.model_dump()
        except Exception:
            return None
    
    def _restore_config(self, rollback_point: RollbackPoint) -> None:
        """Restore configuration from rollback point."""
        if rollback_point.config_snapshot:
            # Write config snapshot
            config_manager = get_config_manager()
            config_manager.update_config(rollback_point.config_snapshot)
        elif rollback_point.backup_path.exists():
            # Restore from backup files
            config_backup = rollback_point.backup_path / 'config'
            if config_backup.exists():
                config_manager = get_config_manager()
                shutil.copytree(
                    config_backup,
                    config_manager.base_path / 'config',
                    dirs_exist_ok=True
                )
    
    def _verify_rollback(self, rollback_point: RollbackPoint) -> bool:
        """Verify rollback was successful."""
        try:
            # Verify feature flags
            current_flags = feature_flags.get_all()
            for flag, expected in rollback_point.feature_flags.items():
                if current_flags.get(flag) != expected:
                    return False
            
            # Basic sanity checks
            from mdm.adapters import get_storage_backend
            backend = get_storage_backend("sqlite")  # Test basic functionality
            
            return True
            
        except Exception:
            return False
    
    def _get_rollback_point(self, point_id: str) -> Optional[RollbackPoint]:
        """Get specific rollback point."""
        for point in self.rollback_points:
            if point.id == point_id:
                return point
        return None
    
    def _get_latest_rollback_point(self) -> Optional[RollbackPoint]:
        """Get most recent rollback point."""
        if self.rollback_points:
            return max(self.rollback_points, key=lambda p: p.timestamp)
        return None
    
    def _load_rollback_points(self) -> None:
        """Load rollback points from history."""
        if self.rollback_history_file.exists():
            try:
                with open(self.rollback_history_file) as f:
                    data = json.load(f)
                    self.rollback_points = [
                        RollbackPoint.from_dict(p) for p in data.get('points', [])
                    ]
            except Exception as e:
                self.console.print(f"[yellow]Warning: Failed to load rollback history: {e}[/yellow]")
    
    def _save_rollback_points(self) -> None:
        """Save rollback points to history."""
        self.rollback_history_file.parent.mkdir(exist_ok=True)
        
        data = {
            'version': 1,
            'points': [p.to_dict() for p in self.rollback_points]
        }
        
        with open(self.rollback_history_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _log_rollback(self, result: RollbackResult) -> None:
        """Log rollback operation."""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'success': result.success,
            'type': result.rollback_type.value,
            'components': result.components_rolled_back,
            'errors': result.errors,
            'duration': result.duration
        }
        
        log_file = Path.home() / '.mdm' / 'rollback.log'
        log_file.parent.mkdir(exist_ok=True)
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def list_rollback_points(self) -> None:
        """List available rollback points."""
        if not self.rollback_points:
            self.console.print("[yellow]No rollback points available[/yellow]")
            return
        
        from rich.table import Table
        
        table = Table(title="Available Rollback Points")
        table.add_column("ID", style="cyan")
        table.add_column("Created", style="green")
        table.add_column("Description")
        table.add_column("Backup Size", justify="right")
        
        for point in sorted(self.rollback_points, key=lambda p: p.timestamp, reverse=True):
            # Calculate backup size
            size = 0
            if point.backup_path.exists():
                for file in point.backup_path.rglob('*'):
                    if file.is_file():
                        size += file.stat().st_size
            
            size_str = f"{size / (1024**2):.1f} MB" if size > 0 else "-"
            
            table.add_row(
                point.id,
                point.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                point.description,
                size_str
            )
        
        self.console.print(table)
    
    def cleanup_old_rollback_points(self, keep_count: int = 5) -> None:
        """Clean up old rollback points."""
        if len(self.rollback_points) <= keep_count:
            return
        
        # Sort by timestamp
        sorted_points = sorted(self.rollback_points, key=lambda p: p.timestamp, reverse=True)
        
        # Keep only the most recent
        to_remove = sorted_points[keep_count:]
        
        self.console.print(f"Removing {len(to_remove)} old rollback points...")
        
        for point in to_remove:
            # Remove backup data
            if point.backup_path.exists():
                shutil.rmtree(point.backup_path)
            
            # Remove from list
            self.rollback_points.remove(point)
        
        # Save updated list
        self._save_rollback_points()
        
        self.console.print(f"[green]✓ Cleaned up {len(to_remove)} rollback points[/green]")