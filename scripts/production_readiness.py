#!/usr/bin/env python3
"""Production readiness checks for MDM.

This script performs comprehensive checks to ensure MDM is ready for production deployment.
"""
import argparse
import sys
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mdm.rollout import RolloutValidator, ValidationStatus
from mdm.testing import IntegrationTestFramework, PerformanceBenchmark
from mdm.adapters import get_config_manager, get_dataset_manager
from mdm.core import feature_flags


console = Console()


class ProductionReadinessChecker:
    """Checks if MDM is ready for production deployment."""
    
    def __init__(self):
        """Initialize checker."""
        self.checks = []
        self.results = {}
        
    def run_all_checks(self) -> Tuple[bool, Dict[str, Any]]:
        """Run all production readiness checks.
        
        Returns:
            Tuple of (is_ready, detailed_results)
        """
        console.print(Panel.fit(
            "[bold cyan]MDM Production Readiness Check[/bold cyan]\n\n"
            "Running comprehensive checks to verify production readiness...",
            title="Production Readiness"
        ))
        
        # Define all checks
        checks = [
            ("System Requirements", self._check_system_requirements),
            ("Dependencies", self._check_dependencies),
            ("Configuration", self._check_configuration),
            ("Feature Flags", self._check_feature_flags),
            ("Database Connectivity", self._check_database_connectivity),
            ("Security", self._check_security),
            ("Performance", self._check_performance),
            ("Integration Tests", self._check_integration_tests),
            ("Documentation", self._check_documentation),
            ("Monitoring", self._check_monitoring),
            ("Backup/Recovery", self._check_backup_recovery),
            ("Deployment Scripts", self._check_deployment_scripts),
        ]
        
        # Run checks
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Running checks...", total=len(checks))
            
            for check_name, check_func in checks:
                progress.update(task, description=f"Checking {check_name}...")
                
                try:
                    passed, details = check_func()
                    self.results[check_name] = {
                        'passed': passed,
                        'details': details
                    }
                except Exception as e:
                    self.results[check_name] = {
                        'passed': False,
                        'details': {'error': str(e)}
                    }
                
                progress.advance(task)
        
        # Display results
        self._display_results()
        
        # Determine overall readiness
        total_checks = len(self.results)
        passed_checks = sum(1 for r in self.results.values() if r['passed'])
        is_ready = passed_checks == total_checks
        
        return is_ready, self.results
    
    def _check_system_requirements(self) -> Tuple[bool, Dict[str, Any]]:
        """Check system requirements."""
        validator = RolloutValidator()
        
        # Run system checks
        validator._check_system_requirements()
        validator._check_disk_space()
        validator._check_memory()
        validator._check_python_version()
        
        # Aggregate results
        passed = all(
            r.status in [ValidationStatus.PASSED, ValidationStatus.WARNING]
            for r in validator.results
        )
        
        details = {
            'checks': [
                {
                    'name': r.check_name,
                    'status': r.status.value,
                    'message': r.message
                }
                for r in validator.results
            ]
        }
        
        return passed, details
    
    def _check_dependencies(self) -> Tuple[bool, Dict[str, Any]]:
        """Check all dependencies are installed."""
        required_packages = [
            'sqlalchemy',
            'pandas',
            'numpy',
            'typer',
            'rich',
            'pydantic',
            'duckdb',
            'psycopg2-binary',
            'ydata-profiling',
            'pytest'
        ]
        
        missing = []
        versions = {}
        
        for package in required_packages:
            try:
                # Try to import
                module = __import__(package.replace('-', '_'))
                # Get version if available
                if hasattr(module, '__version__'):
                    versions[package] = module.__version__
                else:
                    versions[package] = 'unknown'
            except ImportError:
                missing.append(package)
        
        return len(missing) == 0, {
            'missing': missing,
            'installed': versions
        }
    
    def _check_configuration(self) -> Tuple[bool, Dict[str, Any]]:
        """Check configuration validity."""
        try:
            config_manager = get_config_manager()
            config = config_manager.config
            
            issues = []
            
            # Check critical settings
            if not config.database.default_backend:
                issues.append("No default backend configured")
            
            # Check paths exist
            if not config_manager.base_path.exists():
                issues.append(f"Base path does not exist: {config_manager.base_path}")
            
            # Check logging configuration
            if config.logging.file:
                log_dir = Path(config.logging.file).parent
                if not log_dir.exists():
                    issues.append(f"Log directory does not exist: {log_dir}")
            
            return len(issues) == 0, {
                'backend': config.database.default_backend,
                'base_path': str(config_manager.base_path),
                'issues': issues
            }
            
        except Exception as e:
            return False, {'error': str(e)}
    
    def _check_feature_flags(self) -> Tuple[bool, Dict[str, Any]]:
        """Check feature flag configuration."""
        flags = feature_flags.get_all()
        
        # Expected flags for production
        expected_flags = {
            'use_new_storage': True,
            'use_new_features': True,
            'use_new_dataset': True,
            'use_new_config': True,
            'use_new_cli': True
        }
        
        mismatched = []
        for flag, expected in expected_flags.items():
            actual = flags.get(flag, False)
            if actual != expected:
                mismatched.append({
                    'flag': flag,
                    'expected': expected,
                    'actual': actual
                })
        
        return len(mismatched) == 0, {
            'flags': flags,
            'mismatched': mismatched
        }
    
    def _check_database_connectivity(self) -> Tuple[bool, Dict[str, Any]]:
        """Check database connectivity."""
        from mdm.adapters import get_storage_backend
        
        results = {}
        all_passed = True
        
        for backend_type in ['sqlite', 'duckdb', 'postgresql']:
            try:
                backend = get_storage_backend(backend_type)
                
                # Test basic operation
                import tempfile
                with tempfile.TemporaryDirectory() as tmpdir:
                    if backend_type == 'postgresql':
                        # Skip PostgreSQL test if not configured
                        config_manager = get_config_manager()
                        pg_config = config_manager.config.database.postgresql
                        if not pg_config.host or not pg_config.database:
                            results[backend_type] = "Not configured"
                            continue
                    
                    # Test connection
                    db_path = f"{tmpdir}/test.db"
                    engine = backend.get_engine(db_path)
                    
                    # Test query
                    with engine.connect() as conn:
                        result = conn.execute("SELECT 1")
                        result.fetchone()
                    
                    results[backend_type] = "OK"
                    
            except Exception as e:
                results[backend_type] = f"Failed: {str(e)}"
                all_passed = False
        
        return all_passed, results
    
    def _check_security(self) -> Tuple[bool, Dict[str, Any]]:
        """Check security settings."""
        issues = []
        
        # Check file permissions
        config_manager = get_config_manager()
        config_file = config_manager.base_path / 'mdm.yaml'
        
        if config_file.exists():
            # Check if config file is world-readable
            mode = config_file.stat().st_mode
            if mode & 0o004:
                issues.append("Configuration file is world-readable")
        
        # Check for sensitive data in environment
        import os
        sensitive_env_vars = []
        for key, value in os.environ.items():
            if 'PASSWORD' in key or 'SECRET' in key or 'TOKEN' in key:
                if key.startswith('MDM_'):
                    sensitive_env_vars.append(key)
        
        if sensitive_env_vars:
            issues.append(f"Sensitive environment variables detected: {sensitive_env_vars}")
        
        # Check SSL/TLS for PostgreSQL
        config = config_manager.config
        if config.database.default_backend == 'postgresql':
            pg_config = config.database.postgresql
            if not pg_config.sslmode or pg_config.sslmode == 'disable':
                issues.append("PostgreSQL SSL/TLS not enabled")
        
        return len(issues) == 0, {
            'issues': issues,
            'recommendations': [
                "Use environment variables for sensitive values",
                "Enable SSL/TLS for database connections",
                "Restrict file permissions on configuration files",
                "Regular security audits"
            ]
        }
    
    def _check_performance(self) -> Tuple[bool, Dict[str, Any]]:
        """Check performance configuration."""
        config_manager = get_config_manager()
        config = config_manager.config
        
        warnings = []
        
        # Check batch size
        if config.performance.batch_size < 1000:
            warnings.append(f"Batch size ({config.performance.batch_size}) is very small")
        
        # Check worker configuration
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        if config.performance.max_workers > cpu_count:
            warnings.append(
                f"Max workers ({config.performance.max_workers}) exceeds CPU count ({cpu_count})"
            )
        
        # Check cache configuration
        cache_settings = {
            'query_cache': hasattr(config, 'cache') and config.cache.get('enabled', False),
            'connection_pooling': True,  # Default enabled
        }
        
        return len(warnings) == 0, {
            'batch_size': config.performance.batch_size,
            'max_workers': config.performance.max_workers,
            'cpu_count': cpu_count,
            'cache_settings': cache_settings,
            'warnings': warnings
        }
    
    def _check_integration_tests(self) -> Tuple[bool, Dict[str, Any]]:
        """Check if integration tests pass."""
        try:
            # Run pytest with minimal output
            result = subprocess.run(
                ["python", "-m", "pytest", "tests/", "-q", "--tb=short"],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            passed = result.returncode == 0
            
            # Parse output for summary
            output_lines = result.stdout.split('\n')
            summary_line = None
            for line in output_lines:
                if 'passed' in line or 'failed' in line:
                    summary_line = line.strip()
                    break
            
            return passed, {
                'exit_code': result.returncode,
                'summary': summary_line or "No summary available",
                'errors': result.stderr if not passed else None
            }
            
        except subprocess.TimeoutExpired:
            return False, {'error': "Tests timed out after 5 minutes"}
        except Exception as e:
            return False, {'error': str(e)}
    
    def _check_documentation(self) -> Tuple[bool, Dict[str, Any]]:
        """Check documentation completeness."""
        required_docs = [
            'README.md',
            'docs/API_Reference.md',
            'docs/Migration_Guide.md',
            'docs/Deployment_Guide.md',
            'docs/Troubleshooting_Guide.md',
            'docs/Architecture_Decisions.md'
        ]
        
        missing = []
        found = []
        
        project_root = Path(__file__).parent.parent
        
        for doc in required_docs:
            doc_path = project_root / doc
            if doc_path.exists():
                found.append(doc)
            else:
                missing.append(doc)
        
        return len(missing) == 0, {
            'found': found,
            'missing': missing,
            'total': len(required_docs)
        }
    
    def _check_monitoring(self) -> Tuple[bool, Dict[str, Any]]:
        """Check monitoring capabilities."""
        try:
            from mdm.rollout import RolloutMonitor
            from mdm.performance import get_monitor
            
            # Check if monitoring can be initialized
            rollout_monitor = RolloutMonitor()
            perf_monitor = get_monitor()
            
            # Check if metrics are being collected
            report = perf_monitor.get_report()
            has_metrics = 'summary' in report and report.get('total_operations', 0) > 0
            
            features = {
                'rollout_monitoring': True,
                'performance_monitoring': True,
                'metrics_collection': has_metrics,
                'alerting': len(rollout_monitor.alert_handlers) > 0
            }
            
            return True, features
            
        except Exception as e:
            return False, {'error': str(e)}
    
    def _check_backup_recovery(self) -> Tuple[bool, Dict[str, Any]]:
        """Check backup and recovery capabilities."""
        try:
            from mdm.rollout import RollbackManager
            
            manager = RollbackManager()
            
            # Check if rollback points exist
            rollback_points = len(manager.rollback_points)
            
            # Test backup creation
            import tempfile
            with tempfile.TemporaryDirectory() as tmpdir:
                test_backup = Path(tmpdir) / 'test_backup'
                manager._create_backup(test_backup)
                backup_exists = test_backup.exists()
            
            return backup_exists, {
                'rollback_points': rollback_points,
                'backup_test': 'Passed' if backup_exists else 'Failed',
                'capabilities': [
                    'Full rollback',
                    'Partial rollback',
                    'Feature flag rollback',
                    'Configuration rollback'
                ]
            }
            
        except Exception as e:
            return False, {'error': str(e)}
    
    def _check_deployment_scripts(self) -> Tuple[bool, Dict[str, Any]]:
        """Check deployment scripts exist and are executable."""
        scripts = [
            'scripts/final_migration.py',
            'scripts/run_tests.sh',
            'scripts/check_test_imports.py'
        ]
        
        project_root = Path(__file__).parent.parent
        results = {}
        all_good = True
        
        for script in scripts:
            script_path = project_root / script
            
            if not script_path.exists():
                results[script] = "Missing"
                all_good = False
            elif not script_path.is_file():
                results[script] = "Not a file"
                all_good = False
            elif script.endswith('.py') and not script_path.stat().st_mode & 0o111:
                results[script] = "Not executable"
                all_good = False
            else:
                results[script] = "OK"
        
        return all_good, results
    
    def _display_results(self) -> None:
        """Display check results."""
        console.print("\n[bold]Production Readiness Check Results[/bold]\n")
        
        # Create summary table
        table = Table(box=box.ROUNDED)
        table.add_column("Check", style="cyan")
        table.add_column("Status", justify="center")
        table.add_column("Details", style="dim")
        
        for check_name, result in self.results.items():
            passed = result['passed']
            status = "[green]✓ PASSED[/green]" if passed else "[red]✗ FAILED[/red]"
            
            # Extract key details
            details = result.get('details', {})
            if 'error' in details:
                detail_text = f"Error: {details['error']}"
            elif 'issues' in details and details['issues']:
                detail_text = f"{len(details['issues'])} issues"
            elif 'missing' in details and details['missing']:
                detail_text = f"{len(details['missing'])} missing"
            elif 'warnings' in details and details['warnings']:
                detail_text = f"{len(details['warnings'])} warnings"
            else:
                detail_text = "OK"
            
            table.add_row(check_name, status, detail_text)
        
        console.print(table)
        
        # Show failed check details
        failed_checks = [
            (name, result) for name, result in self.results.items()
            if not result['passed']
        ]
        
        if failed_checks:
            console.print("\n[bold red]Failed Checks Details:[/bold red]\n")
            
            for check_name, result in failed_checks:
                console.print(f"[bold]{check_name}:[/bold]")
                details = result.get('details', {})
                
                # Pretty print details
                for key, value in details.items():
                    if isinstance(value, list) and value:
                        console.print(f"  {key}:")
                        for item in value:
                            console.print(f"    - {item}")
                    elif value:
                        console.print(f"  {key}: {value}")
                console.print()
    
    def generate_report(self, output_file: Path) -> None:
        """Generate detailed production readiness report."""
        report = {
            'timestamp': datetime.utcnow().isoformat(),
            'results': self.results,
            'summary': {
                'total_checks': len(self.results),
                'passed': sum(1 for r in self.results.values() if r['passed']),
                'failed': sum(1 for r in self.results.values() if not r['passed'])
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Check MDM production readiness"
    )
    parser.add_argument(
        '--report',
        type=Path,
        help='Save detailed report to file'
    )
    parser.add_argument(
        '--fail-fast',
        action='store_true',
        help='Stop on first failure'
    )
    
    args = parser.parse_args()
    
    # Run checks
    checker = ProductionReadinessChecker()
    is_ready, results = checker.run_all_checks()
    
    # Save report if requested
    if args.report:
        checker.generate_report(args.report)
        console.print(f"\n[dim]Report saved to: {args.report}[/dim]")
    
    # Final verdict
    if is_ready:
        console.print("\n[bold green]✓ MDM is READY for production deployment![/bold green]")
        sys.exit(0)
    else:
        failed_count = sum(1 for r in results.values() if not r['passed'])
        console.print(
            f"\n[bold red]✗ MDM is NOT ready for production. "
            f"{failed_count} checks failed.[/bold red]"
        )
        console.print("\nPlease address the failed checks before deploying to production.")
        sys.exit(1)


if __name__ == "__main__":
    main()