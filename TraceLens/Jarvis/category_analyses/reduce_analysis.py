#!/usr/bin/env python3
"""Reduce Operations Analysis"""

import pandas as pd
import json
import numpy as np
import os
import argparse


def main():
    parser = argparse.ArgumentParser(description='Analyze reduce operations')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    args = parser.parse_args()
    
    output_dir = args.output_dir
    
    # Load data
    ops_df = pd.read_csv(f'{output_dir}/category_data/reduce_ops.csv')
    with open(f'{output_dir}/metadata/reduce_metadata.json', 'r') as f:
        metadata = json.load(f)
    
    peak_hbm_bw = metadata['peak_hbm_bw_tbs']
    
    # Calculate total time
    if 'Kernel Time (µs)_sum' in ops_df.columns:
        total_time_us = ops_df['Kernel Time (µs)_sum'].sum()
        total_time_ms = total_time_us / 1000
        percent_of_compute = (total_time_ms / metadata['gpu_utilization']['total_time_ms']) * 100
    else:
        total_time_ms = 0
        percent_of_compute = 0
    
    # Calculate average efficiency
    avg_efficiency = 0
    eff_count = 0
    for _, row in ops_df.iterrows():
        if not pd.isna(row.get('TB/s_mean')):
            eff = (row.get('TB/s_mean', 0) / peak_hbm_bw) * 100
            avg_efficiency += eff
            eff_count += 1
    avg_efficiency = avg_efficiency / eff_count if eff_count > 0 else 0
    
    # Identify potential bottlenecks
    bottlenecks = []
    softmax_count = 0
    
    for idx, row in ops_df.iterrows():
        op_name = row.get('name', 'Unknown')
        time_ms = row.get('Kernel Time (µs)_sum', 0) / 1000
        count = row.get('count', 1)
        is_softmax = 'softmax' in op_name.lower()
        
        if is_softmax:
            softmax_count += 1
        
        # Calculate efficiency
        efficiency = None
        if not pd.isna(row.get('TB/s_mean')):
            efficiency = (row.get('TB/s_mean', 0) / peak_hbm_bw) * 100
        
        # Apply bottleneck criteria
        reasons = []
        if time_ms > 10 or (time_ms / total_time_ms * 100) > 5:
            reasons.append("High time")
        if efficiency and efficiency < 40:
            reasons.append("Low efficiency")
        if count > 1000:
            reasons.append("High count")
        if pd.isna(row.get('TB/s_mean')):
            reasons.append("Missing perf model")
        
        if reasons:
            bottlenecks.append({
                'op_name': op_name,
                'time_ms': time_ms,
                'percent_of_category': (time_ms / total_time_ms * 100) if total_time_ms > 0 else 0,
                'count': count,
                'efficiency': efficiency,
                'is_softmax': is_softmax,
                'reasons': reasons
            })
    
    bottlenecks.sort(key=lambda x: x['time_ms'], reverse=True)
    
    # Generate markdown output
    markdown = f"""# Reduce Operations Analysis Results

## Summary
- **Total operations:** {len(ops_df)}
- **Total time:** {total_time_ms:.2f} ms ({percent_of_compute:.1f}% of compute)
- **Average efficiency:** {avg_efficiency:.1f}% of peak HBM BW
- **Softmax operations:** {softmax_count}
- **Peak HBM BW:** {peak_hbm_bw} TB/s

## Operations Breakdown

| Operation | Count | Time (ms) | % of Category | Efficiency | Type |
|-----------|-------|-----------|---------------|------------|------|
"""
    
    top_ops = ops_df.nlargest(10, 'Kernel Time (µs)_sum') if 'Kernel Time (µs)_sum' in ops_df.columns else ops_df.head(10)
    for _, row in top_ops.iterrows():
        op_name = row.get('name', 'Unknown')
        count = row.get('count', 1)
        time_ms = row.get('Kernel Time (µs)_sum', 0) / 1000
        pct = (time_ms / total_time_ms * 100) if total_time_ms > 0 else 0
        
        efficiency = "N/A"
        if not pd.isna(row.get('TB/s_mean')):
            eff = (row.get('TB/s_mean', 0) / peak_hbm_bw) * 100
            efficiency = f"{eff:.1f}%"
        
        op_type = "Softmax" if 'softmax' in op_name.lower() else "Reduction"
        markdown += f"| {op_name[:50]} | {count} | {time_ms:.2f} | {pct:.1f}% | {efficiency} | {op_type} |\n"
    
    markdown += f"""
## Potential Bottlenecks

**{len(bottlenecks)} operations flagged** based on criteria

"""
    
    if bottlenecks:
        for i, b in enumerate(bottlenecks[:10], 1):
            markdown += f"""### {i}. {b['op_name']}
- **Time:** {b['time_ms']:.2f} ms ({b['percent_of_category']:.1f}% of category)
- **Count:** {b['count']}
- **Efficiency:** {f"{b['efficiency']:.1f}%" if b['efficiency'] else "N/A"}
- **Type:** {"Softmax" if b['is_softmax'] else "Reduction"}
- **Flagged for:** {", ".join(b['reasons'])}

"""
    else:
        markdown += "*No significant bottlenecks identified.*\n\n"
    
    markdown += """## Key Metrics
- **Memory-bound:** Reduce operations are typically memory-bound
- **Expected efficiency:** 50-70% of peak HBM BW
- **Softmax fusion:** Check if softmax can be fused with attention patterns

"""
    
    print(markdown)


if __name__ == "__main__":
    main()
