"""Visualization utilities for benchmark results."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime


def load_benchmark_results(file_path: str) -> Dict[str, Any]:
    """Load benchmark results from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def plot_improvement_summary(results: Dict[str, Any], output_path: Optional[str] = None):
    """Plot improvement summary across all competitions."""
    # Extract improvements
    competitions = []
    gbt_improvements = []
    rf_improvements = []
    
    for comp_name, comp_results in results['results'].items():
        if comp_results.get('status') != 'completed':
            continue
        
        competitions.append(comp_name.replace('playground-', 'ps-'))
        
        # Get improvements
        gbt_imp = comp_results.get('improvement', {}).get('gbt', '0%')
        rf_imp = comp_results.get('improvement', {}).get('rf', '0%')
        
        gbt_improvements.append(float(gbt_imp.replace('%', '').replace('+', '')))
        rf_improvements.append(float(rf_imp.replace('%', '').replace('+', '')))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(competitions))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, gbt_improvements, width, label='GBT', color='#2E86AB')
    bars2 = ax.bar(x + width/2, rf_improvements, width, label='RF', color='#A23B72')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3 if height >= 0 else -15),
                       textcoords="offset points",
                       ha='center', va='bottom' if height >= 0 else 'top',
                       fontsize=8)
    
    # Customize plot
    ax.set_xlabel('Competition', fontsize=12)
    ax.set_ylabel('Improvement (%)', fontsize=12)
    ax.set_title('MDM Generic Features Performance Improvement', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(competitions, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Add average line
    avg_improvement = np.mean(gbt_improvements + rf_improvements)
    ax.axhline(y=avg_improvement, color='red', linestyle='--', linewidth=1, 
               label=f'Average: {avg_improvement:.1f}%')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def plot_feature_counts(results: Dict[str, Any], output_path: Optional[str] = None):
    """Plot feature counts with and without MDM features."""
    competitions = []
    features_with = []
    features_without = []
    
    for comp_name, comp_results in results['results'].items():
        if comp_results.get('status') != 'completed':
            continue
        
        competitions.append(comp_name.replace('playground-', 'ps-'))
        
        # Get feature counts
        with_f = comp_results.get('with_features', {}).get('gbt', {}).get('n_features', 0)
        without_f = comp_results.get('without_features', {}).get('gbt', {}).get('n_features', 0)
        
        features_with.append(with_f)
        features_without.append(without_f)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(competitions))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, features_without, width, label='Original', color='#F18F01')
    bars2 = ax.bar(x + width/2, features_with, width, label='With MDM Features', color='#048A81')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{int(height)}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=8)
    
    # Customize plot
    ax.set_xlabel('Competition', fontsize=12)
    ax.set_ylabel('Number of Features', fontsize=12)
    ax.set_title('Feature Count Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(competitions, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def plot_metric_comparison(results: Dict[str, Any], output_path: Optional[str] = None):
    """Plot actual metric values for each competition."""
    # Group by metric type
    metric_groups = {}
    
    for comp_name, comp_results in results['results'].items():
        if comp_results.get('status') != 'completed':
            continue
        
        # Find metric type from config (would need to import competition_configs)
        # For now, we'll infer from the values
        gbt_with = comp_results.get('with_features', {}).get('gbt', {}).get('mean_score', 0)
        
        # Infer metric type based on value range
        if gbt_with > 1:
            metric_type = 'RMSE/MAE'
        else:
            metric_type = 'Accuracy/AUC'
        
        if metric_type not in metric_groups:
            metric_groups[metric_type] = {
                'competitions': [],
                'gbt_with': [],
                'gbt_without': [],
                'rf_with': [],
                'rf_without': []
            }
        
        metric_groups[metric_type]['competitions'].append(comp_name.replace('playground-', 'ps-'))
        
        # Get scores
        gbt_with = comp_results.get('with_features', {}).get('gbt', {}).get('mean_score', 0)
        gbt_without = comp_results.get('without_features', {}).get('gbt', {}).get('mean_score', 0)
        rf_with = comp_results.get('with_features', {}).get('rf', {}).get('mean_score', 0)
        rf_without = comp_results.get('without_features', {}).get('rf', {}).get('mean_score', 0)
        
        metric_groups[metric_type]['gbt_with'].append(gbt_with)
        metric_groups[metric_type]['gbt_without'].append(gbt_without)
        metric_groups[metric_type]['rf_with'].append(rf_with)
        metric_groups[metric_type]['rf_without'].append(rf_without)
    
    # Create subplots for each metric type
    n_groups = len(metric_groups)
    fig, axes = plt.subplots(n_groups, 1, figsize=(12, 6 * n_groups))
    
    if n_groups == 1:
        axes = [axes]
    
    for idx, (metric_type, data) in enumerate(metric_groups.items()):
        ax = axes[idx]
        
        x = np.arange(len(data['competitions']))
        width = 0.2
        
        # Plot bars
        ax.bar(x - 1.5*width, data['gbt_without'], width, label='GBT Original', color='#E63946', alpha=0.7)
        ax.bar(x - 0.5*width, data['gbt_with'], width, label='GBT + MDM', color='#E63946')
        ax.bar(x + 0.5*width, data['rf_without'], width, label='RF Original', color='#457B9D', alpha=0.7)
        ax.bar(x + 1.5*width, data['rf_with'], width, label='RF + MDM', color='#457B9D')
        
        # Customize
        ax.set_xlabel('Competition', fontsize=12)
        ax.set_ylabel(f'{metric_type} Score', fontsize=12)
        ax.set_title(f'{metric_type} Comparison', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(data['competitions'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def create_summary_report(results: Dict[str, Any], output_dir: str = "benchmark_results"):
    """Create a comprehensive visual report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create plots
    plot_improvement_summary(results, str(output_dir / f"improvement_summary_{timestamp}.png"))
    plot_feature_counts(results, str(output_dir / f"feature_counts_{timestamp}.png"))
    plot_metric_comparison(results, str(output_dir / f"metric_comparison_{timestamp}.png"))
    
    # Create summary DataFrame
    summary_data = []
    for comp_name, comp_results in results['results'].items():
        if comp_results.get('status') != 'completed':
            continue
        
        row = {
            'Competition': comp_name,
            'GBT Score (Original)': comp_results.get('without_features', {}).get('gbt', {}).get('mean_score', 'N/A'),
            'GBT Score (MDM)': comp_results.get('with_features', {}).get('gbt', {}).get('mean_score', 'N/A'),
            'GBT Improvement': comp_results.get('improvement', {}).get('gbt', 'N/A'),
            'RF Score (Original)': comp_results.get('without_features', {}).get('rf', {}).get('mean_score', 'N/A'),
            'RF Score (MDM)': comp_results.get('with_features', {}).get('rf', {}).get('mean_score', 'N/A'),
            'RF Improvement': comp_results.get('improvement', {}).get('rf', 'N/A'),
            'Original Features': comp_results.get('without_features', {}).get('gbt', {}).get('n_features', 'N/A'),
            'MDM Features': comp_results.get('with_features', {}).get('gbt', {}).get('n_features', 'N/A')
        }
        summary_data.append(row)
    
    # Save summary table
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(output_dir / f"summary_table_{timestamp}.csv", index=False)
    
    print(f"Visual report saved to: {output_dir}")
    
    return df_summary


if __name__ == '__main__':
    # Example usage
    import sys
    if len(sys.argv) > 1:
        results_file = sys.argv[1]
        results = load_benchmark_results(results_file)
        create_summary_report(results)