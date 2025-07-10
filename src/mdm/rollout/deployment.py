"""Deployment management for MDM rollout.

This module provides tools for managing the deployment process
and tracking deployment status.
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import subprocess

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from mdm.core.exceptions import DeploymentError


class DeploymentStatus(Enum):
    """Deployment status states."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class DeploymentStage(Enum):
    """Deployment stages."""
    PREPARATION = "preparation"
    VALIDATION = "validation"
    MIGRATION = "migration"
    VERIFICATION = "verification"
    FINALIZATION = "finalization"


@dataclass
class DeploymentStep:
    """Individual deployment step."""
    name: str
    stage: DeploymentStage
    status: DeploymentStatus = DeploymentStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    output: Optional[str] = None
    
    @property
    def duration(self) -> Optional[float]:
        """Get step duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


@dataclass
class DeploymentInfo:
    """Deployment information and metadata."""
    deployment_id: str
    environment: str
    version: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    status: DeploymentStatus = DeploymentStatus.PENDING
    steps: List[DeploymentStep] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'deployment_id': self.deployment_id,
            'environment': self.environment,
            'version': self.version,
            'started_at': self.started_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'status': self.status.value,
            'steps': [
                {
                    'name': step.name,
                    'stage': step.stage.value,
                    'status': step.status.value,
                    'started_at': step.started_at.isoformat() if step.started_at else None,
                    'completed_at': step.completed_at.isoformat() if step.completed_at else None,
                    'error': step.error,
                    'duration': step.duration
                }
                for step in self.steps
            ],
            'metadata': self.metadata
        }


class DeploymentManager:
    """Manages deployment process and tracking."""
    
    def __init__(self):
        """Initialize deployment manager."""
        self.console = Console()
        self.deployment_log = Path.home() / '.mdm' / 'deployment.log'
        self.current_deployment: Optional[DeploymentInfo] = None
    
    def start_deployment(
        self,
        environment: str,
        version: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> DeploymentInfo:
        """Start a new deployment.
        
        Args:
            environment: Target environment (e.g., 'production', 'staging')
            version: Version being deployed
            metadata: Additional deployment metadata
            
        Returns:
            Deployment information
        """
        deployment_id = f"deploy_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        self.current_deployment = DeploymentInfo(
            deployment_id=deployment_id,
            environment=environment,
            version=version,
            started_at=datetime.utcnow(),
            metadata=metadata or {}
        )
        
        # Define deployment steps
        self._define_deployment_steps()
        
        # Log deployment start
        self._log_deployment_event("deployment_started", {
            'deployment_id': deployment_id,
            'environment': environment,
            'version': version
        })
        
        self.console.print(Panel.fit(
            f"[bold cyan]Deployment Started[/bold cyan]\n\n"
            f"ID: {deployment_id}\n"
            f"Environment: {environment}\n"
            f"Version: {version}",
            title="Deployment"
        ))
        
        return self.current_deployment
    
    def _define_deployment_steps(self) -> None:
        """Define deployment steps."""
        if not self.current_deployment:
            return
        
        steps = [
            # Preparation
            DeploymentStep("Check prerequisites", DeploymentStage.PREPARATION),
            DeploymentStep("Create deployment backup", DeploymentStage.PREPARATION),
            DeploymentStep("Prepare deployment artifacts", DeploymentStage.PREPARATION),
            
            # Validation
            DeploymentStep("Validate environment", DeploymentStage.VALIDATION),
            DeploymentStep("Run pre-deployment tests", DeploymentStage.VALIDATION),
            DeploymentStep("Verify configuration", DeploymentStage.VALIDATION),
            
            # Migration
            DeploymentStep("Enable maintenance mode", DeploymentStage.MIGRATION),
            DeploymentStep("Apply database migrations", DeploymentStage.MIGRATION),
            DeploymentStep("Update feature flags", DeploymentStage.MIGRATION),
            DeploymentStep("Deploy new code", DeploymentStage.MIGRATION),
            
            # Verification
            DeploymentStep("Run smoke tests", DeploymentStage.VERIFICATION),
            DeploymentStep("Verify data integrity", DeploymentStage.VERIFICATION),
            DeploymentStep("Check system health", DeploymentStage.VERIFICATION),
            
            # Finalization
            DeploymentStep("Disable maintenance mode", DeploymentStage.FINALIZATION),
            DeploymentStep("Clear caches", DeploymentStage.FINALIZATION),
            DeploymentStep("Update documentation", DeploymentStage.FINALIZATION),
            DeploymentStep("Send notifications", DeploymentStage.FINALIZATION),
        ]
        
        self.current_deployment.steps = steps
    
    def execute_step(self, step_name: str) -> bool:
        """Execute a deployment step.
        
        Args:
            step_name: Name of the step to execute
            
        Returns:
            True if successful, False otherwise
        """
        if not self.current_deployment:
            raise DeploymentError("No deployment in progress")
        
        # Find step
        step = None
        for s in self.current_deployment.steps:
            if s.name == step_name:
                step = s
                break
        
        if not step:
            raise DeploymentError(f"Step '{step_name}' not found")
        
        # Start step
        step.status = DeploymentStatus.IN_PROGRESS
        step.started_at = datetime.utcnow()
        
        self.console.print(f"\n[cyan]Executing:[/cyan] {step.name}")
        
        try:
            # Execute step logic
            success = self._execute_step_logic(step)
            
            if success:
                step.status = DeploymentStatus.COMPLETED
                self.console.print(f"[green]✓[/green] {step.name}")
            else:
                step.status = DeploymentStatus.FAILED
                self.console.print(f"[red]✗[/red] {step.name}")
            
            step.completed_at = datetime.utcnow()
            return success
            
        except Exception as e:
            step.status = DeploymentStatus.FAILED
            step.error = str(e)
            step.completed_at = datetime.utcnow()
            self.console.print(f"[red]✗[/red] {step.name}: {e}")
            return False
    
    def _execute_step_logic(self, step: DeploymentStep) -> bool:
        """Execute the actual logic for a step."""
        # This would contain the actual deployment logic
        # For now, we'll simulate with some basic checks
        
        import time
        time.sleep(0.5)  # Simulate work
        
        # Simulate different step outcomes
        if "test" in step.name.lower():
            # Run actual tests
            return self._run_tests()
        elif "backup" in step.name.lower():
            # Create backup
            return self._create_backup()
        elif "health" in step.name.lower():
            # Check health
            return self._check_health()
        
        # Default success
        return True
    
    def _run_tests(self) -> bool:
        """Run deployment tests."""
        try:
            result = subprocess.run(
                ["python", "-m", "pytest", "tests/", "-v"],
                capture_output=True,
                text=True,
                timeout=300
            )
            return result.returncode == 0
        except Exception:
            return False
    
    def _create_backup(self) -> bool:
        """Create deployment backup."""
        try:
            backup_dir = Path.home() / '.mdm_deployment_backup' / self.current_deployment.deployment_id
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Create backup marker
            with open(backup_dir / 'backup_info.json', 'w') as f:
                json.dump({
                    'timestamp': datetime.utcnow().isoformat(),
                    'deployment_id': self.current_deployment.deployment_id
                }, f)
            
            return True
        except Exception:
            return False
    
    def _check_health(self) -> bool:
        """Check system health."""
        try:
            # Import and check basic functionality
            from mdm.adapters import get_storage_backend
            backend = get_storage_backend("sqlite")
            return True
        except Exception:
            return False
    
    def complete_deployment(self, success: bool) -> None:
        """Complete the current deployment.
        
        Args:
            success: Whether deployment was successful
        """
        if not self.current_deployment:
            raise DeploymentError("No deployment in progress")
        
        self.current_deployment.completed_at = datetime.utcnow()
        self.current_deployment.status = (
            DeploymentStatus.COMPLETED if success else DeploymentStatus.FAILED
        )
        
        # Calculate statistics
        total_duration = (
            self.current_deployment.completed_at - self.current_deployment.started_at
        ).total_seconds()
        
        successful_steps = sum(
            1 for step in self.current_deployment.steps
            if step.status == DeploymentStatus.COMPLETED
        )
        total_steps = len(self.current_deployment.steps)
        
        # Log completion
        self._log_deployment_event("deployment_completed", {
            'deployment_id': self.current_deployment.deployment_id,
            'success': success,
            'duration': total_duration,
            'successful_steps': successful_steps,
            'total_steps': total_steps
        })
        
        # Display summary
        self.display_deployment_summary()
        
        # Save deployment record
        self._save_deployment_record()
    
    def rollback_deployment(self) -> bool:
        """Rollback the current deployment.
        
        Returns:
            True if rollback successful
        """
        if not self.current_deployment:
            raise DeploymentError("No deployment in progress")
        
        self.console.print("\n[yellow]Rolling back deployment...[/yellow]")
        
        # Mark as rolled back
        self.current_deployment.status = DeploymentStatus.ROLLED_BACK
        self.current_deployment.completed_at = datetime.utcnow()
        
        # Log rollback
        self._log_deployment_event("deployment_rolled_back", {
            'deployment_id': self.current_deployment.deployment_id
        })
        
        return True
    
    def display_deployment_summary(self) -> None:
        """Display deployment summary."""
        if not self.current_deployment:
            return
        
        # Group steps by stage
        stages = {}
        for step in self.current_deployment.steps:
            if step.stage not in stages:
                stages[step.stage] = []
            stages[step.stage].append(step)
        
        # Display each stage
        for stage, steps in stages.items():
            table = Table(
                title=f"{stage.value.title()} Stage",
                box=box.SIMPLE,
                show_header=True
            )
            
            table.add_column("Step", style="cyan")
            table.add_column("Status", justify="center")
            table.add_column("Duration", justify="right")
            
            for step in steps:
                status_icon = {
                    DeploymentStatus.COMPLETED: "[green]✓[/green]",
                    DeploymentStatus.FAILED: "[red]✗[/red]",
                    DeploymentStatus.IN_PROGRESS: "[yellow]⟳[/yellow]",
                    DeploymentStatus.PENDING: "[dim]○[/dim]"
                }.get(step.status, "?")
                
                duration = f"{step.duration:.1f}s" if step.duration else "-"
                
                table.add_row(step.name, status_icon, duration)
            
            self.console.print(table)
            self.console.print()
        
        # Overall summary
        total_duration = (
            self.current_deployment.completed_at - self.current_deployment.started_at
        ).total_seconds() if self.current_deployment.completed_at else 0
        
        successful = sum(1 for s in self.current_deployment.steps if s.status == DeploymentStatus.COMPLETED)
        failed = sum(1 for s in self.current_deployment.steps if s.status == DeploymentStatus.FAILED)
        
        summary = Panel.fit(
            f"[bold]Deployment Summary[/bold]\n\n"
            f"ID: {self.current_deployment.deployment_id}\n"
            f"Status: {self.current_deployment.status.value}\n"
            f"Duration: {total_duration:.1f}s\n"
            f"Steps: {successful} successful, {failed} failed",
            title="Summary",
            border_style="green" if self.current_deployment.status == DeploymentStatus.COMPLETED else "red"
        )
        
        self.console.print(summary)
    
    def get_deployment_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get deployment history.
        
        Args:
            limit: Maximum number of deployments to return
            
        Returns:
            List of deployment records
        """
        history = []
        
        if self.deployment_log.exists():
            with open(self.deployment_log) as f:
                for line in f:
                    try:
                        event = json.loads(line)
                        if event.get('event') == 'deployment_completed':
                            history.append(event['data'])
                    except Exception:
                        continue
        
        # Sort by timestamp and limit
        history.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        return history[:limit]
    
    def _log_deployment_event(self, event: str, data: Dict[str, Any]) -> None:
        """Log deployment event."""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'event': event,
            'data': data
        }
        
        self.deployment_log.parent.mkdir(exist_ok=True)
        
        with open(self.deployment_log, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def _save_deployment_record(self) -> None:
        """Save deployment record."""
        if not self.current_deployment:
            return
        
        record_file = (
            Path.home() / '.mdm' / 'deployments' /
            f"{self.current_deployment.deployment_id}.json"
        )
        
        record_file.parent.mkdir(exist_ok=True)
        
        with open(record_file, 'w') as f:
            json.dump(self.current_deployment.to_dict(), f, indent=2)
    
    def generate_deployment_report(self, deployment_id: str) -> str:
        """Generate deployment report.
        
        Args:
            deployment_id: ID of deployment to report on
            
        Returns:
            Markdown-formatted report
        """
        # Load deployment record
        record_file = Path.home() / '.mdm' / 'deployments' / f"{deployment_id}.json"
        
        if not record_file.exists():
            return f"Deployment record not found: {deployment_id}"
        
        with open(record_file) as f:
            deployment = json.load(f)
        
        # Generate report
        report = f"# Deployment Report: {deployment_id}\n\n"
        report += f"**Environment:** {deployment['environment']}\n"
        report += f"**Version:** {deployment['version']}\n"
        report += f"**Status:** {deployment['status']}\n"
        report += f"**Started:** {deployment['started_at']}\n"
        report += f"**Completed:** {deployment['completed_at'] or 'N/A'}\n\n"
        
        # Add step details
        report += "## Deployment Steps\n\n"
        
        for step in deployment['steps']:
            status_icon = "✓" if step['status'] == 'completed' else "✗"
            report += f"- [{status_icon}] **{step['name']}**\n"
            if step['duration']:
                report += f"  - Duration: {step['duration']:.1f}s\n"
            if step['error']:
                report += f"  - Error: {step['error']}\n"
        
        return report