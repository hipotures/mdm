"""Simple HTML dashboard generator for MDM monitoring."""

import json
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from mdm.monitoring import SimpleMonitor
from mdm.utils.paths import PathManager


class DashboardGenerator:
    """Generate simple HTML dashboard for MDM statistics."""
    
    def __init__(self):
        self.monitor = SimpleMonitor()
        self.path_manager = PathManager()
        
    def generate(self, output_path: Optional[Path] = None) -> Path:
        """Generate HTML dashboard.
        
        Args:
            output_path: Optional path for output file.
                        Defaults to ~/.mdm/dashboard.html
            
        Returns:
            Path to generated dashboard
        """
        if output_path is None:
            output_path = self.path_manager.base_path / "dashboard.html"
        
        # Get data
        stats = self.monitor.get_summary_stats()
        recent_metrics = self.monitor.get_recent_metrics(limit=50)
        
        # Generate HTML
        html = self._generate_html(stats, recent_metrics)
        
        # Write file
        output_path.write_text(html, encoding='utf-8')
        
        return output_path
    
    def _generate_html(self, stats: Dict[str, Any], recent_metrics: List[Dict[str, Any]]) -> str:
        """Generate HTML content."""
        # Prepare data for charts
        metrics_by_type = {}
        for metric in recent_metrics:
            metric_type = metric['metric_type']
            if metric_type not in metrics_by_type:
                metrics_by_type[metric_type] = []
            metrics_by_type[metric_type].append(metric)
        
        # Calculate success rates by type
        type_stats = []
        for type_stat in stats.get('by_type', []):
            total = type_stat['count']
            errors = type_stat['error_count']
            success_rate = ((total - errors) / total * 100) if total > 0 else 100
            type_stats.append({
                'type': type_stat['metric_type'],
                'count': total,
                'success_rate': success_rate,
                'avg_duration': type_stat['avg_duration_ms'] or 0
            })
        
        html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MDM Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1 {{
            color: #333;
            margin-bottom: 30px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .stat-card h3 {{
            margin: 0 0 10px 0;
            color: #666;
            font-size: 14px;
            text-transform: uppercase;
        }}
        .stat-value {{
            font-size: 32px;
            font-weight: bold;
            color: #333;
        }}
        .stat-label {{
            color: #999;
            font-size: 14px;
        }}
        .chart-container {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        .chart-title {{
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 20px;
            color: #333;
        }}
        .recent-operations {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            text-align: left;
            padding: 12px;
            border-bottom: 1px solid #eee;
        }}
        th {{
            font-weight: 600;
            color: #666;
        }}
        .success {{
            color: #28a745;
        }}
        .error {{
            color: #dc3545;
        }}
        .timestamp {{
            color: #999;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>MDM Dashboard</h1>
        <p class="timestamp">Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="stats-grid">
            <div class="stat-card">
                <h3>Total Operations</h3>
                <div class="stat-value">{stats['overall'].get('total_operations', 0)}</div>
            </div>
            <div class="stat-card">
                <h3>Success Rate</h3>
                <div class="stat-value">{stats['overall'].get('successful_operations', 0) / max(stats['overall'].get('total_operations', 1), 1) * 100:.1f}%</div>
            </div>
            <div class="stat-card">
                <h3>Average Duration</h3>
                <div class="stat-value">{self._format_duration(stats['overall'].get('avg_duration_ms', 0))}</div>
            </div>
            <div class="stat-card">
                <h3>Total Datasets</h3>
                <div class="stat-value">{stats['dataset_stats'].get('total_datasets', 0)}</div>
            </div>
        </div>
        
        <div class="chart-container">
            <div class="chart-title">Operations by Type</div>
            <canvas id="operationsChart" height="100"></canvas>
        </div>
        
        <div class="chart-container">
            <div class="chart-title">Performance by Type</div>
            <canvas id="performanceChart" height="100"></canvas>
        </div>
        
        <div class="recent-operations">
            <div class="chart-title">Recent Operations</div>
            <table>
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Type</th>
                        <th>Operation</th>
                        <th>Dataset</th>
                        <th>Duration</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    {self._generate_operations_table(recent_metrics[:20])}
                </tbody>
            </table>
        </div>
    </div>
    
    <script>
        // Operations by Type Chart
        const ctx1 = document.getElementById('operationsChart').getContext('2d');
        new Chart(ctx1, {{
            type: 'bar',
            data: {{
                labels: {json.dumps([s['type'] for s in type_stats])},
                datasets: [{{
                    label: 'Operation Count',
                    data: {json.dumps([s['count'] for s in type_stats])},
                    backgroundColor: 'rgba(54, 162, 235, 0.5)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{
                        beginAtZero: true
                    }}
                }}
            }}
        }});
        
        // Performance Chart
        const ctx2 = document.getElementById('performanceChart').getContext('2d');
        new Chart(ctx2, {{
            type: 'bar',
            data: {{
                labels: {json.dumps([s['type'] for s in type_stats])},
                datasets: [{{
                    label: 'Avg Duration (ms)',
                    data: {json.dumps([s['avg_duration'] for s in type_stats])},
                    backgroundColor: 'rgba(75, 192, 192, 0.5)',
                    borderColor: 'rgba(75, 192, 192, 1)',
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{
                        beginAtZero: true
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>'''
        
        return html
    
    def _format_duration(self, ms: float) -> str:
        """Format duration for display."""
        if ms < 1000:
            return f"{ms:.0f}ms"
        elif ms < 60000:
            return f"{ms/1000:.1f}s"
        else:
            return f"{ms/60000:.1f}m"
    
    def _generate_operations_table(self, metrics: List[Dict[str, Any]]) -> str:
        """Generate table rows for recent operations."""
        rows = []
        for metric in metrics:
            timestamp = datetime.fromisoformat(metric['timestamp'].replace('Z', '+00:00'))
            time_str = timestamp.strftime('%H:%M:%S')
            
            status = '<span class="success">✓</span>' if metric['success'] else '<span class="error">✗</span>'
            duration = self._format_duration(metric['duration_ms']) if metric['duration_ms'] else '-'
            
            rows.append(f'''
                <tr>
                    <td class="timestamp">{time_str}</td>
                    <td>{metric['metric_type']}</td>
                    <td>{metric['operation']}</td>
                    <td>{metric['dataset_name'] or '-'}</td>
                    <td>{duration}</td>
                    <td>{status}</td>
                </tr>
            ''')
        
        return ''.join(rows)


def generate_dashboard(output_path: Optional[Path] = None, open_browser: bool = False) -> Path:
    """Generate dashboard and optionally open in browser.
    
    Args:
        output_path: Optional output path
        open_browser: Whether to open in browser
        
    Returns:
        Path to generated dashboard
    """
    generator = DashboardGenerator()
    dashboard_path = generator.generate(output_path)
    
    if open_browser:
        webbrowser.open(f"file://{dashboard_path.absolute()}")
    
    return dashboard_path