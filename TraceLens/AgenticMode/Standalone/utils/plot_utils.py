###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Plot generation and report embedding utilities for TraceLens AgenticMode.

Provides functions for:
- Generating the performance improvement plot from pre-computed plot data
- Embedding the plot as a base64 data URI in the markdown report
"""

import json
import os


def generate_perf_plot(output_dir: str, title: str) -> bool:
    """
    Generate the performance improvement plot from plot_data.json.

    Reads plot_data.json (produced by generate_plot_data), computes cumulative
    projections, renders a two-panel matplotlib chart (bar + line), and saves
    both the PNG image and a base64-encoded text file for report embedding.

    Args:
        output_dir: Base output directory containing plot_data.json
        title: Plot suptitle (e.g. '<Model> on <Platform> — Kernel Tuning Potential')

    Returns:
        True if the plot was generated successfully, False otherwise
    """
    plot_data_path = os.path.join(output_dir, 'plot_data.json')
    if not os.path.exists(plot_data_path):
        print(f'plot_data.json not found at {plot_data_path} - skipping plot')
        return False

    with open(plot_data_path, 'r') as f:
        plot_data = json.load(f)

    baseline_ms = plot_data.get('baseline_ms', 0)
    recommendations = plot_data.get('recommendations', [])

    if not recommendations or baseline_ms <= 0:
        print('No kernel tuning recommendations or invalid baseline - skipping plot')
        return False

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not installed - skipping plot generation')
        return False

    current_ms = baseline_ms
    steps = ['Baseline']
    e2e_ms = [baseline_ms]
    savings_list = [0]
    cumulative_rel = [100]

    for rec in recommendations:
        current_ms -= rec['savings_ms']
        current_ms = max(current_ms, 0.01)
        count = rec.get('operation_count', 1)
        label = rec['category'] + f'\n({count} ops)'
        steps.append(label)
        e2e_ms.append(current_ms)
        savings_list.append(rec['savings_ms'])
        cumulative_rel.append(round(baseline_ms / current_ms * 100))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5),
                                    gridspec_kw={'width_ratios': [1.1, 1]})

    colors = ['#4a90d9', '#e74c3c', '#e67e22', '#f1c40f', '#2ecc71',
              '#9b59b6', '#1abc9c'][:len(steps)]
    bars = ax1.bar(steps, e2e_ms, color=colors, edgecolor='white',
                   linewidth=1.2, width=0.65)
    for bar, val, sav in zip(bars, e2e_ms, savings_list):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                 f'{val:.1f} ms', ha='center', va='bottom', fontsize=10,
                 fontweight='bold')
        if sav > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                     f'-{sav:.1f} ms', ha='center', va='center',
                     fontsize=9, color='white', fontweight='bold')
    ax1.set_ylabel('E2E Latency (ms)', fontsize=11)
    ax1.set_title('Projected E2E Latency After Each Optimization',
                  fontsize=12, fontweight='bold', pad=12)
    ax1.set_ylim(0, max(e2e_ms) * 1.2)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.tick_params(axis='x', labelsize=9)

    ax2.plot(range(len(steps)), cumulative_rel, 'o-', color='#2ecc71',
             linewidth=2.5, markersize=9, markerfacecolor='white',
             markeredgewidth=2.5)
    for x, y in enumerate(cumulative_rel):
        ax2.annotate(f'{y}', (x, y), textcoords="offset points",
                     xytext=(0, 12), ha='center', fontsize=10,
                     fontweight='bold', color='#27ae60')
    ax2.set_xticks(range(len(steps)))
    ax2.set_xticklabels(steps, fontsize=9)
    ax2.set_ylabel('Relative Throughput (Baseline = 100)', fontsize=11)
    ax2.set_title('Cumulative Throughput Improvement',
                  fontsize=12, fontweight='bold', pad=12)
    ax2.set_ylim(80, max(cumulative_rel) * 1.15)
    ax2.axhline(y=100, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.tick_params(axis='x', labelsize=9)

    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()

    import base64

    output_path = os.path.join(output_dir, 'perf_improvement.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Plot saved to {output_path}')

    with open(output_path, 'rb') as f:
        b64_str = base64.b64encode(f.read()).decode('ascii')
    b64_path = os.path.join(output_dir, 'perf_improvement_base64.txt')
    with open(b64_path, 'w') as f:
        f.write(b64_str)
    print(f'Base64 written to {b64_path}')

    return True


def embed_plot_in_report(output_dir: str,
                         report_filename: str = 'standalone_analysis.md',
                         placeholder: str = '{{PERF_PLOT}}') -> bool:
    """
    Replace the plot placeholder in the report with a base64-embedded PNG data URI.

    Reads perf_improvement_base64.txt (written by generate_perf_plot) and substitutes
    the placeholder in the report file. If the base64 file is missing, the placeholder
    is removed so the report remains clean.

    Args:
        output_dir: Base output directory containing the report and base64 file
        report_filename: Name of the markdown report file
        placeholder: Placeholder string to replace

    Returns:
        True if the plot was embedded, False if the base64 file was missing or
        the report file does not exist
    """
    report_path = os.path.join(output_dir, report_filename)
    b64_path = os.path.join(output_dir, 'perf_improvement_base64.txt')

    if not os.path.exists(report_path):
        print(f'Report file not found at {report_path} - skipping embed')
        return False

    with open(report_path, 'r') as f:
        report = f.read()

    if os.path.exists(b64_path):
        with open(b64_path, 'r') as f:
            b64_str = f.read().strip()
        img_tag = f'![Performance Improvement](data:image/png;base64,{b64_str})'
        embedded = True
    else:
        img_tag = ''
        embedded = False

    report = report.replace(placeholder, img_tag)

    with open(report_path, 'w') as f:
        f.write(report)

    return embedded
