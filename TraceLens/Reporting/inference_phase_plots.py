#!/usr/bin/env python3
"""
Inference Phase Visualization Module for TraceLens

This module provides comprehensive visualization functions for analyzing
Prefill and Decode phase operations in LLM inference workloads.

Features:
- Phase distribution pie charts and bar plots
- Time duration analysis and histograms
- Operation frequency comparisons
- Performance bottleneck identification
- Timeline visualization
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.patches as mpatches

# Set style for better-looking plots
plt.style.use('default')
sns.set_palette("husl")

def plot_inference_phase_analysis(
    phase_summaries: Dict[str, pd.DataFrame],
    kernel_summaries: Optional[Dict[str, pd.DataFrame]] = None,
    output_dir: str = "inference_plots",
    trace_name: str = "trace",
    show_plots: bool = False,
    save_plots: bool = True,
    plot_format: str = "png",
    dpi: int = 300
) -> Dict[str, str]:
    """
    Generate comprehensive visualization plots for inference phase analysis.
    
    Args:
        phase_summaries (Dict[str, pd.DataFrame]): Dict with 'Prefill' and 'Decode' 
                                                  operation summaries
        kernel_summaries (Dict[str, pd.DataFrame], optional): Dict with 'Prefill' and 
                                                             'Decode' kernel summaries
        output_dir (str): Directory to save plots
        trace_name (str): Name of the trace for plot titles
        show_plots (bool): Whether to display plots interactively
        save_plots (bool): Whether to save plots to files
        plot_format (str): Format for saved plots ('png', 'pdf', 'svg')
        dpi (int): DPI for saved plots
        
    Returns:
        Dict[str, str]: Dictionary mapping plot names to file paths
    """
    
    # Create output directory
    if save_plots and not Path(output_dir).exists():
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    plot_files = {}
    
    # Check if we have data
    prefill_ops = phase_summaries.get('Prefill', pd.DataFrame())
    decode_ops = phase_summaries.get('Decode', pd.DataFrame())
    
    if prefill_ops.empty and decode_ops.empty:
        print("⚠ No phase data available for plotting")
        return plot_files
    
    # Set up the plotting style
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 14
    })
    
    # 1. Phase Overview Dashboard
    plot_files.update(_plot_phase_overview(
        prefill_ops, decode_ops, output_dir, trace_name, 
        show_plots, save_plots, plot_format, dpi
    ))
    
    # 2. Time Distribution Analysis
    plot_files.update(_plot_time_distributions(
        prefill_ops, decode_ops, output_dir, trace_name,
        show_plots, save_plots, plot_format, dpi
    ))
    
    # 3. Operation Frequency Analysis
    plot_files.update(_plot_operation_frequencies(
        prefill_ops, decode_ops, output_dir, trace_name,
        show_plots, save_plots, plot_format, dpi
    ))
    
    # 4. Performance Bottlenecks
    plot_files.update(_plot_performance_bottlenecks(
        prefill_ops, decode_ops, output_dir, trace_name,
        show_plots, save_plots, plot_format, dpi
    ))
    
    # 5. Detailed Operation Breakdown
    plot_files.update(_plot_detailed_breakdowns(
        prefill_ops, decode_ops, output_dir, trace_name,
        show_plots, save_plots, plot_format, dpi
    ))
    
    # 6. Kernel-level analysis (if available)
    if kernel_summaries:
        plot_files.update(_plot_kernel_analysis(
            kernel_summaries, output_dir, trace_name,
            show_plots, save_plots, plot_format, dpi
        ))
    
    # Generate summary report
    if save_plots:
        _generate_plot_summary(plot_files, output_dir, trace_name, phase_summaries)
    
    print(f"📊 Generated {len(plot_files)} visualization plots in '{output_dir}/'")
    return plot_files


def _plot_phase_overview(prefill_ops, decode_ops, output_dir, trace_name, 
                        show_plots, save_plots, plot_format, dpi):
    """Generate phase overview dashboard"""
    plot_files = {}
    
    # Calculate summary statistics
    prefill_total_time = prefill_ops['total_direct_kernel_time_ms'].sum() if not prefill_ops.empty else 0
    decode_total_time = decode_ops['total_direct_kernel_time_ms'].sum() if not decode_ops.empty else 0
    prefill_count = len(prefill_ops) if not prefill_ops.empty else 0
    decode_count = len(decode_ops) if not decode_ops.empty else 0
    
    total_time = prefill_total_time + decode_total_time
    total_ops = prefill_count + decode_count
    
    if total_time == 0:
        return plot_files
    
    # Create 2x2 subplot dashboard
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Inference Phase Analysis Overview - {trace_name}', fontsize=16, fontweight='bold')
    
    # 1. Phase Time Distribution (Pie Chart)
    if prefill_total_time > 0 or decode_total_time > 0:
        times = [prefill_total_time, decode_total_time]
        labels = ['Prefill', 'Decode']
        colors = ['#FF6B6B', '#4ECDC4']
        
        # Filter out zero values
        non_zero_times = [(t, l, c) for t, l, c in zip(times, labels, colors) if t > 0]
        if non_zero_times:
            times, labels, colors = zip(*non_zero_times)
            
            wedges, texts, autotexts = ax1.pie(times, labels=labels, colors=colors, autopct='%1.1f%%',
                                              startangle=90, textprops={'fontsize': 10})
            ax1.set_title('Total Execution Time Distribution', fontweight='bold')
            
            # Add time annotations
            for i, (time, label) in enumerate(zip(times, labels)):
                ax1.annotate(f'{time:.1f}ms', 
                           xy=(0, 0), xytext=(1.3, 0.8-i*0.2), 
                           transform=ax1.transAxes, fontsize=9,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor=colors[i], alpha=0.3))
    
    # 2. Operation Count Distribution (Bar Chart)
    counts = [prefill_count, decode_count]
    phases = ['Prefill', 'Decode']
    colors = ['#FF6B6B', '#4ECDC4']
    
    bars = ax2.bar(phases, counts, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    ax2.set_title('Number of Operations by Phase', fontweight='bold')
    ax2.set_ylabel('Operation Count')
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        if count > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                    str(count), ha='center', va='bottom', fontweight='bold')
    
    # 3. Average Operation Time Comparison
    avg_times = []
    phase_labels = []
    
    if prefill_count > 0:
        avg_prefill = prefill_total_time / prefill_count
        avg_times.append(avg_prefill)
        phase_labels.append(f'Prefill\n({prefill_count} ops)')
    
    if decode_count > 0:
        avg_decode = decode_total_time / decode_count  
        avg_times.append(avg_decode)
        phase_labels.append(f'Decode\n({decode_count} ops)')
    
    if avg_times:
        colors_subset = colors[:len(avg_times)]
        bars = ax3.bar(range(len(avg_times)), avg_times, color=colors_subset, alpha=0.7, 
                      edgecolor='black', linewidth=1)
        ax3.set_title('Average Time per Operation', fontweight='bold')
        ax3.set_ylabel('Average Time (ms)')
        ax3.set_xticks(range(len(avg_times)))
        ax3.set_xticklabels(phase_labels)
        
        # Add value labels
        for i, (bar, time) in enumerate(zip(bars, avg_times)):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(avg_times)*0.01,
                    f'{time:.1f}ms', ha='center', va='bottom', fontweight='bold')
    
    # 4. Summary Statistics Table
    ax4.axis('off')
    
    # Create summary data
    summary_data = [
        ['Metric', 'Prefill', 'Decode', 'Total/Ratio'],
        ['Total Time (ms)', f'{prefill_total_time:.1f}', f'{decode_total_time:.1f}', f'{total_time:.1f}'],
        ['Operation Count', str(prefill_count), str(decode_count), str(total_ops)],
        ['Avg Time/Op (ms)', 
         f'{prefill_total_time/prefill_count:.1f}' if prefill_count > 0 else 'N/A',
         f'{decode_total_time/decode_count:.1f}' if decode_count > 0 else 'N/A',
         f'{total_time/total_ops:.1f}' if total_ops > 0 else 'N/A'],
        ['Phase Percentage', 
         f'{100*prefill_total_time/total_time:.1f}%' if total_time > 0 else 'N/A',
         f'{100*decode_total_time/total_time:.1f}%' if total_time > 0 else 'N/A', 
         '100%'],
        ['Prefill:Decode Ratio',
         f'{prefill_total_time/decode_total_time:.2f}:1' if decode_total_time > 0 else 'N/A',
         f'1:{decode_total_time/prefill_total_time:.2f}' if prefill_total_time > 0 else 'N/A',
         'Time Ratio']
    ]
    
    # Create table
    table = ax4.table(cellText=summary_data[1:], colLabels=summary_data[0],
                     cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    # Style the table
    for i in range(len(summary_data)):
        for j in range(len(summary_data[0])):
            cell = table[(i, j)]
            if i == 0:  # Header row
                cell.set_facecolor('#E8E8E8')
                cell.set_text_props(weight='bold')
            elif j == 0:  # First column
                cell.set_facecolor('#F5F5F5')
                cell.set_text_props(weight='bold')
            else:
                cell.set_facecolor('white')
    
    ax4.set_title('Summary Statistics', fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_plots:
        filename = f"phase_overview.{plot_format}"
        filepath = Path(output_dir) / filename
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
        plot_files['phase_overview'] = str(filepath)
    
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    return plot_files


def _plot_time_distributions(prefill_ops, decode_ops, output_dir, trace_name,
                           show_plots, save_plots, plot_format, dpi):
    """Generate time distribution analysis plots"""
    plot_files = {}
    
    # Create 2x2 subplot for time analysis
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Operation Time Distribution Analysis - {trace_name}', fontsize=16, fontweight='bold')
    
    # 1. Top Operations by Time (Prefill)
    if not prefill_ops.empty and 'total_direct_kernel_time_ms' in prefill_ops.columns:
        top_prefill = prefill_ops.nlargest(10, 'total_direct_kernel_time_ms')
        
        # Truncate long operation names for readability
        op_names = [name[:30] + '...' if len(name) > 30 else name for name in top_prefill['name']]
        
        bars = ax1.barh(range(len(top_prefill)), top_prefill['total_direct_kernel_time_ms'], 
                       color='#FF6B6B', alpha=0.7, edgecolor='black', linewidth=0.5)
        ax1.set_yticks(range(len(top_prefill)))
        ax1.set_yticklabels(op_names)
        ax1.set_xlabel('Time (ms)')
        ax1.set_title('Top 10 Prefill Operations by Time', fontweight='bold')
        ax1.invert_yaxis()
        
        # Add value labels
        for i, (bar, time) in enumerate(zip(bars, top_prefill['total_direct_kernel_time_ms'])):
            ax1.text(bar.get_width() + max(top_prefill['total_direct_kernel_time_ms'])*0.01,
                    bar.get_y() + bar.get_height()/2, f'{time:.1f}ms',
                    va='center', fontsize=8)
    else:
        ax1.text(0.5, 0.5, 'No Prefill Data Available', ha='center', va='center', 
                transform=ax1.transAxes, fontsize=12)
        ax1.set_title('Top 10 Prefill Operations by Time', fontweight='bold')
    
    # 2. Top Operations by Time (Decode)
    if not decode_ops.empty and 'total_direct_kernel_time_ms' in decode_ops.columns:
        top_decode = decode_ops.nlargest(10, 'total_direct_kernel_time_ms')
        
        # Truncate long operation names
        op_names = [name[:30] + '...' if len(name) > 30 else name for name in top_decode['name']]
        
        bars = ax2.barh(range(len(top_decode)), top_decode['total_direct_kernel_time_ms'],
                       color='#4ECDC4', alpha=0.7, edgecolor='black', linewidth=0.5)
        ax2.set_yticks(range(len(top_decode)))
        ax2.set_yticklabels(op_names)
        ax2.set_xlabel('Time (ms)')
        ax2.set_title('Top 10 Decode Operations by Time', fontweight='bold')
        ax2.invert_yaxis()
        
        # Add value labels
        for i, (bar, time) in enumerate(zip(bars, top_decode['total_direct_kernel_time_ms'])):
            ax2.text(bar.get_width() + max(top_decode['total_direct_kernel_time_ms'])*0.01,
                    bar.get_y() + bar.get_height()/2, f'{time:.1f}ms',
                    va='center', fontsize=8)
    else:
        ax2.text(0.5, 0.5, 'No Decode Data Available', ha='center', va='center',
                transform=ax2.transAxes, fontsize=12)
        ax2.set_title('Top 10 Decode Operations by Time', fontweight='bold')
    
    # 3. Time Distribution Histogram (Prefill)
    if not prefill_ops.empty and 'total_direct_kernel_time_ms' in prefill_ops.columns:
        times = prefill_ops['total_direct_kernel_time_ms']
        ax3.hist(times, bins=20, color='#FF6B6B', alpha=0.7, edgecolor='black', linewidth=0.5)
        ax3.set_xlabel('Operation Time (ms)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Prefill Time Distribution', fontweight='bold')
        
        # Add statistics
        mean_time = times.mean()
        median_time = times.median()
        ax3.axvline(mean_time, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_time:.1f}ms')
        ax3.axvline(median_time, color='darkred', linestyle='-', linewidth=2, label=f'Median: {median_time:.1f}ms')
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, 'No Prefill Data Available', ha='center', va='center',
                transform=ax3.transAxes, fontsize=12)
        ax3.set_title('Prefill Time Distribution', fontweight='bold')
    
    # 4. Time Distribution Histogram (Decode)
    if not decode_ops.empty and 'total_direct_kernel_time_ms' in decode_ops.columns:
        times = decode_ops['total_direct_kernel_time_ms']
        ax4.hist(times, bins=20, color='#4ECDC4', alpha=0.7, edgecolor='black', linewidth=0.5)
        ax4.set_xlabel('Operation Time (ms)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Decode Time Distribution', fontweight='bold')
        
        # Add statistics
        mean_time = times.mean()
        median_time = times.median()
        ax4.axvline(mean_time, color='teal', linestyle='--', linewidth=2, label=f'Mean: {mean_time:.1f}ms')
        ax4.axvline(median_time, color='darkcyan', linestyle='-', linewidth=2, label=f'Median: {median_time:.1f}ms')
        ax4.legend()
    else:
        ax4.text(0.5, 0.5, 'No Decode Data Available', ha='center', va='center',
                transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Decode Time Distribution', fontweight='bold')
    
    plt.tight_layout()
    
    if save_plots:
        filename = f"time_distributions.{plot_format}"
        filepath = Path(output_dir) / filename
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
        plot_files['time_distributions'] = str(filepath)
    
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    return plot_files


def _plot_operation_frequencies(prefill_ops, decode_ops, output_dir, trace_name,
                              show_plots, save_plots, plot_format, dpi):
    """Generate operation frequency analysis plots"""
    plot_files = {}
    
    # Create 2x2 subplot for frequency analysis
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Operation Frequency Analysis - {trace_name}', fontsize=16, fontweight='bold')
    
    # 1. Most Frequent Operations (Prefill)
    if not prefill_ops.empty and 'Count' in prefill_ops.columns:
        top_freq_prefill = prefill_ops.nlargest(10, 'Count')
        
        op_names = [name[:25] + '...' if len(name) > 25 else name for name in top_freq_prefill['name']]
        
        bars = ax1.barh(range(len(top_freq_prefill)), top_freq_prefill['Count'],
                       color='#FF6B6B', alpha=0.7, edgecolor='black', linewidth=0.5)
        ax1.set_yticks(range(len(top_freq_prefill)))
        ax1.set_yticklabels(op_names)
        ax1.set_xlabel('Call Count')
        ax1.set_title('Most Frequent Prefill Operations', fontweight='bold')
        ax1.invert_yaxis()
        
        # Add value labels
        for i, (bar, count) in enumerate(zip(bars, top_freq_prefill['Count'])):
            ax1.text(bar.get_width() + max(top_freq_prefill['Count'])*0.01,
                    bar.get_y() + bar.get_height()/2, str(count),
                    va='center', fontsize=8)
    else:
        ax1.text(0.5, 0.5, 'No Prefill Data Available', ha='center', va='center',
                transform=ax1.transAxes, fontsize=12)
        ax1.set_title('Most Frequent Prefill Operations', fontweight='bold')
    
    # 2. Most Frequent Operations (Decode)
    if not decode_ops.empty and 'Count' in decode_ops.columns:
        top_freq_decode = decode_ops.nlargest(10, 'Count')
        
        op_names = [name[:25] + '...' if len(name) > 25 else name for name in top_freq_decode['name']]
        
        bars = ax2.barh(range(len(top_freq_decode)), top_freq_decode['Count'],
                       color='#4ECDC4', alpha=0.7, edgecolor='black', linewidth=0.5)
        ax2.set_yticks(range(len(top_freq_decode)))
        ax2.set_yticklabels(op_names)
        ax2.set_xlabel('Call Count')
        ax2.set_title('Most Frequent Decode Operations', fontweight='bold')
        ax2.invert_yaxis()
        
        # Add value labels
        for i, (bar, count) in enumerate(zip(bars, top_freq_decode['Count'])):
            ax2.text(bar.get_width() + max(top_freq_decode['Count'])*0.01,
                    bar.get_y() + bar.get_height()/2, str(count),
                    va='center', fontsize=8)
    else:
        ax2.text(0.5, 0.5, 'No Decode Data Available', ha='center', va='center',
                transform=ax2.transAxes, fontsize=12)
        ax2.set_title('Most Frequent Decode Operations', fontweight='bold')
    
    # 3. Frequency vs Time Scatter (Prefill)
    if not prefill_ops.empty and all(col in prefill_ops.columns for col in ['Count', 'total_direct_kernel_time_ms']):
        scatter = ax3.scatter(prefill_ops['Count'], prefill_ops['total_direct_kernel_time_ms'],
                            c='#FF6B6B', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        ax3.set_xlabel('Call Count')
        ax3.set_ylabel('Total Time (ms)')
        ax3.set_title('Prefill: Frequency vs Total Time', fontweight='bold')
        
        # Add trend line
        if len(prefill_ops) > 1:
            z = np.polyfit(prefill_ops['Count'], prefill_ops['total_direct_kernel_time_ms'], 1)
            p = np.poly1d(z)
            ax3.plot(prefill_ops['Count'], p(prefill_ops['Count']), "r--", alpha=0.8, linewidth=2)
    else:
        ax3.text(0.5, 0.5, 'No Prefill Data Available', ha='center', va='center',
                transform=ax3.transAxes, fontsize=12)
        ax3.set_title('Prefill: Frequency vs Total Time', fontweight='bold')
    
    # 4. Frequency vs Time Scatter (Decode)
    if not decode_ops.empty and all(col in decode_ops.columns for col in ['Count', 'total_direct_kernel_time_ms']):
        scatter = ax4.scatter(decode_ops['Count'], decode_ops['total_direct_kernel_time_ms'],
                            c='#4ECDC4', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        ax4.set_xlabel('Call Count')
        ax4.set_ylabel('Total Time (ms)')
        ax4.set_title('Decode: Frequency vs Total Time', fontweight='bold')
        
        # Add trend line
        if len(decode_ops) > 1:
            z = np.polyfit(decode_ops['Count'], decode_ops['total_direct_kernel_time_ms'], 1)
            p = np.poly1d(z)
            ax4.plot(decode_ops['Count'], p(decode_ops['Count']), "b--", alpha=0.8, linewidth=2)
    else:
        ax4.text(0.5, 0.5, 'No Decode Data Available', ha='center', va='center',
                transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Decode: Frequency vs Total Time', fontweight='bold')
    
    plt.tight_layout()
    
    if save_plots:
        filename = f"operation_frequencies.{plot_format}"
        filepath = Path(output_dir) / filename
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
        plot_files['operation_frequencies'] = str(filepath)
    
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    return plot_files


def _plot_performance_bottlenecks(prefill_ops, decode_ops, output_dir, trace_name,
                                show_plots, save_plots, plot_format, dpi):
    """Generate performance bottleneck identification plots"""
    plot_files = {}
    
    # Create figure for bottleneck analysis
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Performance Bottleneck Analysis - {trace_name}', fontsize=16, fontweight='bold')
    
    # 1. Efficiency Analysis (Time per Call)
    if not prefill_ops.empty and all(col in prefill_ops.columns for col in ['total_direct_kernel_time_ms', 'Count']):
        prefill_ops_copy = prefill_ops.copy()
        prefill_ops_copy['time_per_call'] = prefill_ops_copy['total_direct_kernel_time_ms'] / prefill_ops_copy['Count']
        top_inefficient_prefill = prefill_ops_copy.nlargest(8, 'time_per_call')
        
        op_names = [name[:20] + '...' if len(name) > 20 else name for name in top_inefficient_prefill['name']]
        
        bars = ax1.barh(range(len(top_inefficient_prefill)), top_inefficient_prefill['time_per_call'],
                       color='#FF6B6B', alpha=0.7, edgecolor='black', linewidth=0.5)
        ax1.set_yticks(range(len(top_inefficient_prefill)))
        ax1.set_yticklabels(op_names)
        ax1.set_xlabel('Time per Call (ms)')
        ax1.set_title('Slowest Prefill Operations (Time/Call)', fontweight='bold')
        ax1.invert_yaxis()
        
        # Add value labels
        for i, (bar, time) in enumerate(zip(bars, top_inefficient_prefill['time_per_call'])):
            ax1.text(bar.get_width() + max(top_inefficient_prefill['time_per_call'])*0.01,
                    bar.get_y() + bar.get_height()/2, f'{time:.2f}ms',
                    va='center', fontsize=8)
    else:
        ax1.text(0.5, 0.5, 'No Prefill Data Available', ha='center', va='center',
                transform=ax1.transAxes, fontsize=12)
        ax1.set_title('Slowest Prefill Operations (Time/Call)', fontweight='bold')
    
    # 2. Decode Efficiency Analysis
    if not decode_ops.empty and all(col in decode_ops.columns for col in ['total_direct_kernel_time_ms', 'Count']):
        decode_ops_copy = decode_ops.copy()
        decode_ops_copy['time_per_call'] = decode_ops_copy['total_direct_kernel_time_ms'] / decode_ops_copy['Count']
        top_inefficient_decode = decode_ops_copy.nlargest(8, 'time_per_call')
        
        op_names = [name[:20] + '...' if len(name) > 20 else name for name in top_inefficient_decode['name']]
        
        bars = ax2.barh(range(len(top_inefficient_decode)), top_inefficient_decode['time_per_call'],
                       color='#4ECDC4', alpha=0.7, edgecolor='black', linewidth=0.5)
        ax2.set_yticks(range(len(top_inefficient_decode)))
        ax2.set_yticklabels(op_names)
        ax2.set_xlabel('Time per Call (ms)')
        ax2.set_title('Slowest Decode Operations (Time/Call)', fontweight='bold')
        ax2.invert_yaxis()
        
        # Add value labels
        for i, (bar, time) in enumerate(zip(bars, top_inefficient_decode['time_per_call'])):
            ax2.text(bar.get_width() + max(top_inefficient_decode['time_per_call'])*0.01,
                    bar.get_y() + bar.get_height()/2, f'{time:.2f}ms',
                    va='center', fontsize=8)
    else:
        ax2.text(0.5, 0.5, 'No Decode Data Available', ha='center', va='center',
                transform=ax2.transAxes, fontsize=12)
        ax2.set_title('Slowest Decode Operations (Time/Call)', fontweight='bold')
    
    # 3. Cumulative Time Contribution (Prefill)
    if not prefill_ops.empty and 'Percentage (%)' in prefill_ops.columns:
        # Sort by percentage and get top contributors
        sorted_prefill = prefill_ops.sort_values('Percentage (%)', ascending=False).head(10)
        
        op_names = [name[:15] + '...' if len(name) > 15 else name for name in sorted_prefill['name']]
        
        bars = ax3.bar(range(len(sorted_prefill)), sorted_prefill['Percentage (%)'],
                      color='#FF6B6B', alpha=0.7, edgecolor='black', linewidth=0.5)
        ax3.set_xticks(range(len(sorted_prefill)))
        ax3.set_xticklabels(op_names, rotation=45, ha='right')
        ax3.set_ylabel('Percentage of Phase Time (%)')
        ax3.set_title('Top Prefill Time Contributors', fontweight='bold')
        
        # Add value labels
        for bar, pct in zip(bars, sorted_prefill['Percentage (%)']):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{pct:.1f}%', ha='center', va='bottom', fontsize=8)
    else:
        ax3.text(0.5, 0.5, 'No Prefill Data Available', ha='center', va='center',
                transform=ax3.transAxes, fontsize=12)
        ax3.set_title('Top Prefill Time Contributors', fontweight='bold')
    
    # 4. Cumulative Time Contribution (Decode)
    if not decode_ops.empty and 'Percentage (%)' in decode_ops.columns:
        # Sort by percentage and get top contributors
        sorted_decode = decode_ops.sort_values('Percentage (%)', ascending=False).head(10)
        
        op_names = [name[:15] + '...' if len(name) > 15 else name for name in sorted_decode['name']]
        
        bars = ax4.bar(range(len(sorted_decode)), sorted_decode['Percentage (%)'],
                      color='#4ECDC4', alpha=0.7, edgecolor='black', linewidth=0.5)
        ax4.set_xticks(range(len(sorted_decode)))
        ax4.set_xticklabels(op_names, rotation=45, ha='right')
        ax4.set_ylabel('Percentage of Phase Time (%)')
        ax4.set_title('Top Decode Time Contributors', fontweight='bold')
        
        # Add value labels
        for bar, pct in zip(bars, sorted_decode['Percentage (%)']):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{pct:.1f}%', ha='center', va='bottom', fontsize=8)
    else:
        ax4.text(0.5, 0.5, 'No Decode Data Available', ha='center', va='center',
                transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Top Decode Time Contributors', fontweight='bold')
    
    plt.tight_layout()
    
    if save_plots:
        filename = f"performance_bottlenecks.{plot_format}"
        filepath = Path(output_dir) / filename
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
        plot_files['performance_bottlenecks'] = str(filepath)
    
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    return plot_files


def _plot_detailed_breakdowns(prefill_ops, decode_ops, output_dir, trace_name,
                            show_plots, save_plots, plot_format, dpi):
    """Generate detailed operation breakdown plots"""
    plot_files = {}
    
    # Create figure for detailed breakdowns
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Detailed Operation Breakdown - {trace_name}', fontsize=16, fontweight='bold')
    
    # 1. Operation Type Analysis (Prefill)
    if not prefill_ops.empty:
        # Categorize operations by type
        def categorize_operation(name):
            name_lower = str(name).lower()
            if any(pattern in name_lower for pattern in ['mm', 'gemm', 'matmul']):
                return 'Matrix Ops'
            elif any(pattern in name_lower for pattern in ['attention', 'attn', 'flash']):
                return 'Attention'
            elif any(pattern in name_lower for pattern in ['norm', 'layer_norm', 'batch_norm']):
                return 'Normalization'
            elif any(pattern in name_lower for pattern in ['copy', 'clone', 'view', 'reshape']):
                return 'Memory Ops'
            elif any(pattern in name_lower for pattern in ['add', 'mul', 'div', 'sub']):
                return 'Elementwise'
            elif any(pattern in name_lower for pattern in ['embedding', 'embed']):
                return 'Embedding'
            elif any(pattern in name_lower for pattern in ['triton']):
                return 'Triton Kernels'
            else:
                return 'Other'
        
        prefill_ops_copy = prefill_ops.copy()
        prefill_ops_copy['op_type'] = prefill_ops_copy['name'].apply(categorize_operation)
        
        type_stats = prefill_ops_copy.groupby('op_type').agg({
            'total_direct_kernel_time_ms': 'sum',
            'Count': 'sum'
        }).reset_index()
        type_stats = type_stats.sort_values('total_direct_kernel_time_ms', ascending=False)
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(type_stats)))
        bars = ax1.bar(range(len(type_stats)), type_stats['total_direct_kernel_time_ms'],
                      color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax1.set_xticks(range(len(type_stats)))
        ax1.set_xticklabels(type_stats['op_type'], rotation=45, ha='right')
        ax1.set_ylabel('Total Time (ms)')
        ax1.set_title('Prefill Operations by Type', fontweight='bold')
        
        # Add value labels
        for bar, time in zip(bars, type_stats['total_direct_kernel_time_ms']):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(type_stats['total_direct_kernel_time_ms'])*0.01,
                    f'{time:.1f}ms', ha='center', va='bottom', fontsize=8, rotation=0)
    else:
        ax1.text(0.5, 0.5, 'No Prefill Data Available', ha='center', va='center',
                transform=ax1.transAxes, fontsize=12)
        ax1.set_title('Prefill Operations by Type', fontweight='bold')
    
    # 2. Operation Type Analysis (Decode)  
    if not decode_ops.empty:
        decode_ops_copy = decode_ops.copy()
        decode_ops_copy['op_type'] = decode_ops_copy['name'].apply(categorize_operation)
        
        type_stats = decode_ops_copy.groupby('op_type').agg({
            'total_direct_kernel_time_ms': 'sum',
            'Count': 'sum'
        }).reset_index()
        type_stats = type_stats.sort_values('total_direct_kernel_time_ms', ascending=False)
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(type_stats)))
        bars = ax2.bar(range(len(type_stats)), type_stats['total_direct_kernel_time_ms'],
                      color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax2.set_xticks(range(len(type_stats)))
        ax2.set_xticklabels(type_stats['op_type'], rotation=45, ha='right')
        ax2.set_ylabel('Total Time (ms)')
        ax2.set_title('Decode Operations by Type', fontweight='bold')
        
        # Add value labels
        for bar, time in zip(bars, type_stats['total_direct_kernel_time_ms']):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(type_stats['total_direct_kernel_time_ms'])*0.01,
                    f'{time:.1f}ms', ha='center', va='bottom', fontsize=8, rotation=0)
    else:
        ax2.text(0.5, 0.5, 'No Decode Data Available', ha='center', va='center',
                transform=ax2.transAxes, fontsize=12)
        ax2.set_title('Decode Operations by Type', fontweight='bold')
    
    # 3. Comparison: Total vs Average Time
    comparison_data = []
    
    if not prefill_ops.empty:
        prefill_total = prefill_ops['total_direct_kernel_time_ms'].sum()
        prefill_avg = prefill_ops['total_direct_kernel_time_ms'].mean()
        prefill_count = len(prefill_ops)
        comparison_data.append(['Prefill', prefill_total, prefill_avg, prefill_count])
    
    if not decode_ops.empty:
        decode_total = decode_ops['total_direct_kernel_time_ms'].sum()
        decode_avg = decode_ops['total_direct_kernel_time_ms'].mean()
        decode_count = len(decode_ops)
        comparison_data.append(['Decode', decode_total, decode_avg, decode_count])
    
    if comparison_data:
        phases = [row[0] for row in comparison_data]
        totals = [row[1] for row in comparison_data]
        averages = [row[2] for row in comparison_data]
        
        x = np.arange(len(phases))
        width = 0.35
        
        # Normalize for comparison (scale averages to be visible with totals)
        scale_factor = max(totals) / max(averages) * 0.3 if max(averages) > 0 else 1
        scaled_averages = [avg * scale_factor for avg in averages]
        
        bars1 = ax3.bar(x - width/2, totals, width, label='Total Time (ms)',
                       color=['#FF6B6B', '#4ECDC4'][:len(phases)], alpha=0.7, edgecolor='black', linewidth=0.5)
        bars2 = ax3.bar(x + width/2, scaled_averages, width, label=f'Avg Time (ms) ×{scale_factor:.1f}',
                       color=['#FFB3B3', '#B3E0DD'][:len(phases)], alpha=0.7, edgecolor='black', linewidth=0.5)
        
        ax3.set_xlabel('Phase')
        ax3.set_ylabel('Time (ms)')
        ax3.set_title('Total vs Average Operation Time', fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(phases)
        ax3.legend()
        
        # Add value labels
        for bars, values in [(bars1, totals), (bars2, averages)]:
            for bar, val in zip(bars, values):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(totals)*0.01,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    else:
        ax3.text(0.5, 0.5, 'No Data Available', ha='center', va='center',
                transform=ax3.transAxes, fontsize=12)
        ax3.set_title('Total vs Average Operation Time', fontweight='bold')
    
    # 4. Phase Efficiency Metrics
    ax4.axis('off')
    
    # Calculate efficiency metrics
    efficiency_data = [['Metric', 'Prefill', 'Decode', 'Comparison']]
    
    if not prefill_ops.empty and not decode_ops.empty:
        # Time efficiency (operations per ms)
        prefill_efficiency = len(prefill_ops) / prefill_ops['total_direct_kernel_time_ms'].sum()
        decode_efficiency = len(decode_ops) / decode_ops['total_direct_kernel_time_ms'].sum()
        
        # Memory efficiency (calls per operation type)
        prefill_diversity = prefill_ops['total_direct_kernel_time_ms'].sum() / len(prefill_ops)
        decode_diversity = decode_ops['total_direct_kernel_time_ms'].sum() / len(decode_ops)
        
        efficiency_data.extend([
            ['Ops/ms', f'{prefill_efficiency:.3f}', f'{decode_efficiency:.3f}',
             f'{"Prefill" if prefill_efficiency > decode_efficiency else "Decode"} more efficient'],
            ['Avg Time/Op (ms)', f'{prefill_diversity:.2f}', f'{decode_diversity:.2f}',
             f'{"Prefill" if prefill_diversity < decode_diversity else "Decode"} faster ops'],
            ['Total Operations', str(len(prefill_ops)), str(len(decode_ops)),
             f'{"Prefill" if len(prefill_ops) > len(decode_ops) else "Decode"} more diverse'],
            ['Phase Dominance', 
             f'{100*prefill_ops["total_direct_kernel_time_ms"].sum()/(prefill_ops["total_direct_kernel_time_ms"].sum() + decode_ops["total_direct_kernel_time_ms"].sum()):.1f}%',
             f'{100*decode_ops["total_direct_kernel_time_ms"].sum()/(prefill_ops["total_direct_kernel_time_ms"].sum() + decode_ops["total_direct_kernel_time_ms"].sum()):.1f}%',
             f'{"Prefill" if prefill_ops["total_direct_kernel_time_ms"].sum() > decode_ops["total_direct_kernel_time_ms"].sum() else "Decode"} dominates']
        ])
    elif not prefill_ops.empty:
        efficiency_data.extend([
            ['Total Operations', str(len(prefill_ops)), 'N/A', 'Only Prefill available'],
            ['Total Time (ms)', f'{prefill_ops["total_direct_kernel_time_ms"].sum():.1f}', 'N/A', 'Only Prefill available']
        ])
    elif not decode_ops.empty:
        efficiency_data.extend([
            ['Total Operations', 'N/A', str(len(decode_ops)), 'Only Decode available'],
            ['Total Time (ms)', 'N/A', f'{decode_ops["total_direct_kernel_time_ms"].sum():.1f}', 'Only Decode available']
        ])
    else:
        efficiency_data.append(['No data available', 'N/A', 'N/A', 'N/A'])
    
    # Create table
    table = ax4.table(cellText=efficiency_data[1:], colLabels=efficiency_data[0],
                     cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)
    
    # Style the table
    for i in range(len(efficiency_data)):
        for j in range(len(efficiency_data[0])):
            cell = table[(i, j)]
            if i == 0:  # Header row
                cell.set_facecolor('#E8E8E8')
                cell.set_text_props(weight='bold')
            elif j == 0:  # First column
                cell.set_facecolor('#F5F5F5')
                cell.set_text_props(weight='bold')
            else:
                cell.set_facecolor('white')
    
    ax4.set_title('Phase Efficiency Comparison', fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_plots:
        filename = f"detailed_breakdowns.{plot_format}"
        filepath = Path(output_dir) / filename
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
        plot_files['detailed_breakdowns'] = str(filepath)
    
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    return plot_files


def _plot_kernel_analysis(kernel_summaries, output_dir, trace_name,
                        show_plots, save_plots, plot_format, dpi):
    """Generate kernel-level analysis plots (if kernel data available)"""
    plot_files = {}
    
    # This is a placeholder for kernel-level analysis
    # Implementation would be similar to operation analysis but for individual kernels
    
    print("📊 Kernel-level plotting not yet implemented")
    return plot_files


def _generate_plot_summary(plot_files, output_dir, trace_name, phase_summaries):
    """Generate a summary HTML file with all plots"""
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Inference Phase Analysis - {trace_name}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #333; border-bottom: 2px solid #ddd; }}
            h2 {{ color: #666; margin-top: 30px; }}
            .plot {{ text-align: center; margin: 20px 0; }}
            .plot img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
            .summary {{ background: #f9f9f9; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <h1>Inference Phase Analysis Report: {trace_name}</h1>
        
        <div class="summary">
            <h3>Analysis Summary</h3>
            <ul>
                <li><strong>Prefill Operations:</strong> {len(phase_summaries.get('Prefill', pd.DataFrame()))} operations</li>
                <li><strong>Decode Operations:</strong> {len(phase_summaries.get('Decode', pd.DataFrame()))} operations</li>
                <li><strong>Total Plots Generated:</strong> {len(plot_files)}</li>
                <li><strong>Generated:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</li>
            </ul>
        </div>
        
        <h2>Visualization Plots</h2>
    """
    
    plot_descriptions = {
        'phase_overview': 'Complete overview of inference phase analysis with time distribution, operation counts, and summary statistics',
        'time_distributions': 'Detailed analysis of operation time distributions for both Prefill and Decode phases',
        'operation_frequencies': 'Analysis of operation call frequencies and their relationship to execution time',
        'performance_bottlenecks': 'Identification of performance bottlenecks and efficiency analysis for each phase',
        'detailed_breakdowns': 'Comprehensive breakdown of operations by type and detailed efficiency comparisons'
    }
    
    for plot_name, plot_path in plot_files.items():
        description = plot_descriptions.get(plot_name, f'Analysis plot: {plot_name}')
        plot_filename = Path(plot_path).name
        
        html_content += f"""
        <h3>{plot_name.replace('_', ' ').title()}</h3>
        <p>{description}</p>
        <div class="plot">
            <img src="{plot_filename}" alt="{plot_name}">
        </div>
        """
    
    html_content += """
        <div class="summary">
            <h3>How to Use This Analysis</h3>
            <ul>
                <li><strong>Phase Overview:</strong> Start here to understand the overall time distribution between Prefill and Decode</li>
                <li><strong>Time Distributions:</strong> Identify the most time-consuming operations in each phase</li>
                <li><strong>Operation Frequencies:</strong> Understand which operations are called most frequently</li>
                <li><strong>Performance Bottlenecks:</strong> Focus optimization efforts on the slowest operations per call</li>
                <li><strong>Detailed Breakdowns:</strong> Analyze operations by type and compare phase efficiency</li>
            </ul>
        </div>
    </body>
    </html>
    """
    
    # Save HTML summary
    html_path = Path(output_dir) / f"{trace_name}_inference_analysis.html"
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    print(f"📄 Generated plot summary: {html_path}")


# Helper function for easy integration
def generate_inference_plots_from_report(
    report_data: Dict[str, pd.DataFrame],
    output_dir: str = "inference_plots",
    trace_name: str = None,
    **plot_kwargs
) -> Dict[str, str]:
    """
    Generate inference phase plots directly from perf report data.
    
    Args:
        report_data: Dictionary from generate_perf_report_pytorch() 
        output_dir: Directory to save plots
        trace_name: Name for the trace (auto-detected if None)
        **plot_kwargs: Additional arguments for plot_inference_phase_analysis()
        
    Returns:
        Dictionary mapping plot names to file paths
    """
    
    # Extract phase summaries from report data
    phase_summaries = {}
    
    if 'ops_summary_prefill' in report_data:
        phase_summaries['Prefill'] = report_data['ops_summary_prefill']
    
    if 'ops_summary_decode' in report_data:
        phase_summaries['Decode'] = report_data['ops_summary_decode']
    
    # Extract kernel summaries if available
    kernel_summaries = {}
    
    if 'kernel_summary_prefill' in report_data:
        kernel_summaries['Prefill'] = report_data['kernel_summary_prefill']
        
    if 'kernel_summary_decode' in report_data:
        kernel_summaries['Decode'] = report_data['kernel_summary_decode']
    
    # Auto-detect trace name if not provided
    if trace_name is None:
        trace_name = f"trace_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Generate plots
    return plot_inference_phase_analysis(
        phase_summaries=phase_summaries,
        kernel_summaries=kernel_summaries if kernel_summaries else None,
        output_dir=output_dir,
        trace_name=trace_name,
        **plot_kwargs
    )


if __name__ == "__main__":
    # Example usage for testing
    print("📊 Inference Phase Visualization Module")
    print("Import this module and use plot_inference_phase_analysis() or")
    print("generate_inference_plots_from_report() to create visualizations.")