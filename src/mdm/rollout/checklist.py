"""Rollout checklist management.

This module provides a comprehensive checklist system for managing
the final rollout of the MDM refactoring.
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import yaml

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich import box

from mdm.core.exceptions import RolloutError


class CheckStatus(Enum):
    """Status of a checklist item."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    BLOCKED = "blocked"


@dataclass
class ChecklistItem:
    """Individual checklist item."""
    id: str
    category: str
    description: str
    status: CheckStatus = CheckStatus.PENDING
    assignee: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    notes: str = ""
    dependencies: List[str] = field(default_factory=list)
    validation_command: Optional[str] = None
    rollback_command: Optional[str] = None
    critical: bool = False
    
    def is_ready(self, completed_items: List[str]) -> bool:
        """Check if item is ready to execute based on dependencies."""
        return all(dep in completed_items for dep in self.dependencies)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'category': self.category,
            'description': self.description,
            'status': self.status.value,
            'assignee': self.assignee,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'notes': self.notes,
            'dependencies': self.dependencies,
            'validation_command': self.validation_command,
            'rollback_command': self.rollback_command,
            'critical': self.critical
        }


class RolloutChecklist:
    """Manages the complete rollout checklist."""
    
    def __init__(self):
        """Initialize rollout checklist."""
        self.items: Dict[str, ChecklistItem] = {}
        self.console = Console()
        self._create_default_checklist()
    
    def _create_default_checklist(self) -> None:
        """Create the default rollout checklist."""
        # Pre-deployment checks
        self.add_item(ChecklistItem(
            id="pre_backup",
            category="Pre-Deployment",
            description="Create full system backup",
            critical=True,
            validation_command="mdm backup verify"
        ))
        
        self.add_item(ChecklistItem(
            id="pre_health",
            category="Pre-Deployment",
            description="Run system health checks",
            critical=True,
            dependencies=["pre_backup"],
            validation_command="mdm doctor --all"
        ))
        
        self.add_item(ChecklistItem(
            id="pre_tests",
            category="Pre-Deployment",
            description="Run all test suites",
            critical=True,
            dependencies=["pre_health"],
            validation_command="./scripts/run_tests.sh"
        ))
        
        self.add_item(ChecklistItem(
            id="pre_migration_test",
            category="Pre-Deployment",
            description="Test migration on staging environment",
            critical=True,
            dependencies=["pre_tests"],
            validation_command="mdm migration test --env staging"
        ))
        
        # Infrastructure preparation
        self.add_item(ChecklistItem(
            id="infra_monitoring",
            category="Infrastructure",
            description="Set up monitoring and alerting",
            critical=True,
            validation_command="mdm monitoring status"
        ))
        
        self.add_item(ChecklistItem(
            id="infra_logging",
            category="Infrastructure",
            description="Configure centralized logging",
            critical=False,
            validation_command="mdm logging test"
        ))
        
        self.add_item(ChecklistItem(
            id="infra_scaling",
            category="Infrastructure",
            description="Configure auto-scaling policies",
            critical=False
        ))
        
        # Migration execution
        self.add_item(ChecklistItem(
            id="migrate_config",
            category="Migration",
            description="Migrate configuration to new format",
            critical=True,
            dependencies=["pre_migration_test"],
            validation_command="mdm config validate",
            rollback_command="mdm config restore"
        ))
        
        self.add_item(ChecklistItem(
            id="migrate_enable_flags",
            category="Migration",
            description="Enable feature flags progressively",
            critical=True,
            dependencies=["migrate_config"],
            validation_command="mdm flags status",
            rollback_command="mdm flags reset"
        ))
        
        self.add_item(ChecklistItem(
            id="migrate_storage",
            category="Migration",
            description="Migrate storage backends",
            critical=True,
            dependencies=["migrate_enable_flags"],
            validation_command="mdm storage validate --all",
            rollback_command="mdm storage rollback"
        ))
        
        self.add_item(ChecklistItem(
            id="migrate_datasets",
            category="Migration",
            description="Migrate all datasets",
            critical=True,
            dependencies=["migrate_storage"],
            validation_command="mdm dataset validate --all",
            rollback_command="mdm dataset rollback --all"
        ))
        
        # Validation
        self.add_item(ChecklistItem(
            id="validate_integrity",
            category="Validation",
            description="Validate data integrity",
            critical=True,
            dependencies=["migrate_datasets"],
            validation_command="mdm validation integrity --deep"
        ))
        
        self.add_item(ChecklistItem(
            id="validate_performance",
            category="Validation",
            description="Run performance benchmarks",
            critical=False,
            dependencies=["validate_integrity"],
            validation_command="mdm benchmark run --compare"
        ))
        
        self.add_item(ChecklistItem(
            id="validate_api",
            category="Validation",
            description="Validate API compatibility",
            critical=True,
            dependencies=["validate_integrity"],
            validation_command="mdm api test --all"
        ))
        
        # Post-deployment
        self.add_item(ChecklistItem(
            id="post_monitoring",
            category="Post-Deployment",
            description="Verify monitoring and alerts",
            critical=True,
            dependencies=["validate_api"],
            validation_command="mdm monitoring verify"
        ))
        
        self.add_item(ChecklistItem(
            id="post_cleanup",
            category="Post-Deployment",
            description="Clean up migration artifacts",
            critical=False,
            dependencies=["post_monitoring"],
            validation_command="mdm cleanup --dry-run"
        ))
        
        self.add_item(ChecklistItem(
            id="post_documentation",
            category="Post-Deployment",
            description="Update deployment documentation",
            critical=False,
            dependencies=["post_cleanup"]
        ))
    
    def add_item(self, item: ChecklistItem) -> None:
        """Add item to checklist."""
        self.items[item.id] = item
    
    def update_status(
        self,
        item_id: str,
        status: CheckStatus,
        notes: Optional[str] = None
    ) -> None:
        """Update item status."""
        if item_id not in self.items:
            raise RolloutError(f"Checklist item '{item_id}' not found")
        
        item = self.items[item_id]
        item.status = status
        
        if status == CheckStatus.IN_PROGRESS:
            item.started_at = datetime.utcnow()
        elif status in [CheckStatus.COMPLETED, CheckStatus.FAILED, CheckStatus.SKIPPED]:
            item.completed_at = datetime.utcnow()
        
        if notes:
            item.notes = notes
    
    def get_ready_items(self) -> List[ChecklistItem]:
        """Get items ready for execution."""
        completed = [
            item_id for item_id, item in self.items.items()
            if item.status == CheckStatus.COMPLETED
        ]
        
        ready = []
        for item in self.items.values():
            if (item.status == CheckStatus.PENDING and
                item.is_ready(completed)):
                ready.append(item)
        
        return ready
    
    def get_progress(self) -> Dict[str, Any]:
        """Get checklist progress."""
        total = len(self.items)
        by_status = {}
        
        for item in self.items.values():
            status = item.status.value
            by_status[status] = by_status.get(status, 0) + 1
        
        completed = by_status.get(CheckStatus.COMPLETED.value, 0)
        failed = by_status.get(CheckStatus.FAILED.value, 0)
        
        return {
            'total': total,
            'completed': completed,
            'failed': failed,
            'in_progress': by_status.get(CheckStatus.IN_PROGRESS.value, 0),
            'pending': by_status.get(CheckStatus.PENDING.value, 0),
            'blocked': by_status.get(CheckStatus.BLOCKED.value, 0),
            'skipped': by_status.get(CheckStatus.SKIPPED.value, 0),
            'completion_rate': (completed / total * 100) if total > 0 else 0,
            'success_rate': (completed / (completed + failed) * 100) if (completed + failed) > 0 else 0
        }
    
    def get_critical_items(self) -> List[ChecklistItem]:
        """Get critical items that must succeed."""
        return [item for item in self.items.values() if item.critical]
    
    def can_proceed(self) -> bool:
        """Check if rollout can proceed."""
        critical_items = self.get_critical_items()
        
        for item in critical_items:
            if item.status == CheckStatus.FAILED:
                return False
            if item.status not in [CheckStatus.COMPLETED, CheckStatus.SKIPPED]:
                return False
        
        return True
    
    def display(self) -> None:
        """Display checklist in terminal."""
        # Group items by category
        categories = {}
        for item in self.items.values():
            if item.category not in categories:
                categories[item.category] = []
            categories[item.category].append(item)
        
        # Display each category
        for category, items in categories.items():
            table = Table(
                title=f"[bold]{category}[/bold]",
                box=box.ROUNDED,
                show_header=True,
                header_style="bold cyan"
            )
            
            table.add_column("ID", style="dim")
            table.add_column("Description")
            table.add_column("Status", justify="center")
            table.add_column("Critical", justify="center")
            table.add_column("Dependencies", style="dim")
            
            for item in sorted(items, key=lambda x: x.id):
                status_style = {
                    CheckStatus.PENDING: "yellow",
                    CheckStatus.IN_PROGRESS: "blue",
                    CheckStatus.COMPLETED: "green",
                    CheckStatus.FAILED: "red",
                    CheckStatus.BLOCKED: "magenta",
                    CheckStatus.SKIPPED: "dim"
                }.get(item.status, "white")
                
                deps = ", ".join(item.dependencies) if item.dependencies else "-"
                
                table.add_row(
                    item.id,
                    item.description,
                    f"[{status_style}]{item.status.value}[/{status_style}]",
                    "✓" if item.critical else "",
                    deps
                )
            
            self.console.print(table)
            self.console.print()
        
        # Display progress summary
        progress = self.get_progress()
        summary = Panel.fit(
            f"[bold]Progress Summary[/bold]\n\n"
            f"Total Items: {progress['total']}\n"
            f"Completed: [green]{progress['completed']}[/green]\n"
            f"Failed: [red]{progress['failed']}[/red]\n"
            f"In Progress: [blue]{progress['in_progress']}[/blue]\n"
            f"Pending: [yellow]{progress['pending']}[/yellow]\n"
            f"Completion: {progress['completion_rate']:.1f}%\n"
            f"Success Rate: {progress['success_rate']:.1f}%",
            title="[bold]Rollout Status[/bold]",
            border_style="cyan"
        )
        self.console.print(summary)
    
    def save(self, path: Path) -> None:
        """Save checklist to file."""
        data = {
            'timestamp': datetime.utcnow().isoformat(),
            'items': [item.to_dict() for item in self.items.values()],
            'progress': self.get_progress()
        }
        
        with open(path, 'w') as f:
            if path.suffix == '.yaml':
                yaml.dump(data, f, default_flow_style=False)
            else:
                json.dump(data, f, indent=2)
    
    def load(self, path: Path) -> None:
        """Load checklist from file."""
        with open(path) as f:
            if path.suffix == '.yaml':
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        
        self.items.clear()
        
        for item_data in data['items']:
            # Convert timestamps
            if item_data.get('started_at'):
                item_data['started_at'] = datetime.fromisoformat(item_data['started_at'])
            if item_data.get('completed_at'):
                item_data['completed_at'] = datetime.fromisoformat(item_data['completed_at'])
            
            # Convert status
            item_data['status'] = CheckStatus(item_data['status'])
            
            item = ChecklistItem(**item_data)
            self.add_item(item)
    
    def execute_validation(self, item_id: str) -> bool:
        """Execute validation command for an item."""
        if item_id not in self.items:
            raise RolloutError(f"Checklist item '{item_id}' not found")
        
        item = self.items[item_id]
        if not item.validation_command:
            return True
        
        import subprocess
        
        self.console.print(f"[cyan]Validating:[/cyan] {item.description}")
        self.console.print(f"[dim]Command:[/dim] {item.validation_command}")
        
        try:
            result = subprocess.run(
                item.validation_command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                self.console.print("[green]✓ Validation passed[/green]")
                return True
            else:
                self.console.print(f"[red]✗ Validation failed[/red]")
                if result.stderr:
                    self.console.print(f"[red]Error:[/red] {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.console.print("[red]✗ Validation timed out[/red]")
            return False
        except Exception as e:
            self.console.print(f"[red]✗ Validation error:[/red] {e}")
            return False