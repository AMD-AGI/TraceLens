#!/usr/bin/env python3
"""Generic/Other Operations Analysis"""

import pandas as pd
import json
import numpy as np
import os
import argparse


def main():
    parser = argparse.ArgumentParser(description='Analyze other/generic operations')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    args = parser.parse_args()
    
    output_dir = args.output_dir
    
    # Load data
    ops_df = pd.read_csv(f'{output_dir}/category_data/other_ops.csv')
    with open(f'{output_dir}/metadata/other_metadata.json', 'r') as f:
        metadata = json.load(f)
    
    peak_hbm_bw = metadata['peak_hbm_bw_tbs']
    peak_maf = metadata['peak_bf16_maf_tflops']
    
    # Calculate total time
    if 'Kernel Time (µs)_sum' in ops_df.columns:
        total_time_us = ops_df['Kernel Time (µs)_sum'].sum()
        total_time_ms = total_time_us / 1000
        percent_of_compute = (total_time_ms / metadata['gpu_utilization']['total_time_ms']) * 100
    else:
        total_time_ms = 0
        percent_of_compute = 0
    
    # Categorize operations (MoE, BatchNorm, Convolution are now separate categories)
    comm_ops = []
    graph_ops = []
    misc_ops = []
    
    for idx, row in ops_df.iterrows():
        op_name = row.get('name', 'Unknown')
        if 'all_reduce' in op_name.lower() or 'collective' in op_name.lower() or 'ncclKernel' in op_name:
            comm_ops.append(op_name)
        elif 'hipgraph' in op_name.lower() or 'graph' in op_name.lower():
            graph_ops.append(op_name)
        else:
            misc_ops.append(op_name)
    
    # Identify potential bottlenecks
    bottlenecks = []
    
    for idx, row in ops_df.iterrows():
        op_name = row.get('name', 'Unknown')
        time_ms = row.get('Kernel Time (µs)_sum', 0) / 1000
        count = row.get('count', 1)
        
        # Determine category
        category = "Other"
        if 'all_reduce' in op_name.lower() or 'collective' in op_name.lower() or 'ncclKernel' in op_name:
            category = "Communication"
        elif 'hipgraph' in op_name.lower() or 'graph' in op_name.lower():
            category = "Graph"
        
        # Calculate efficiency
        efficiency = None
        if not pd.isna(row.get('TB/s_mean')):
            efficiency = (row.get('TB/s_mean', 0) / peak_hbm_bw) * 100
        elif not pd.isna(row.get('TFLOPS/s_mean')):
            efficiency = (row.get('TFLOPS/s_mean', 0) / peak_maf) * 100
        
        # Apply bottleneck criteria
        reasons = []
        if time_ms > 100 or (time_ms / total_time_ms * 100) > 5:
            reasons.append("High time")
        if efficiency and efficiency < 40:
            reasons.append("Low efficiency")
        if count > 1000:
            reasons.append("High count")
        if efficiency is None:
            reasons.append("Missing perf model")
        
        if reasons:
            bottlenecks.append({
                'op_name': op_name,
                'time_ms': time_ms,
                'percent_of_category': (time_ms / total_time_ms * 100) if total_time_ms > 0 else 0,
                'count': count,
                'efficiency': efficiency,
                'category': category,
                'reasons': reasons
            })
    
    bottlenecks.sort(key=lambda x: x['time_ms'], reverse=True)
    
    # Generate markdown output
    markdown = f"""# Other/Generic Operations Analysis Results

## Summary
- **Total operations:** {len(ops_df)}
- **Total time:** {total_time_ms:.2f} ms ({percent_of_compute:.1f}% of compute)
- **Communication operations:** {len(comm_ops)}
- **Graph operations:** {len(graph_ops)}
- **Miscellaneous operations:** {len(misc_ops)}
- **Peak MAF:** {peak_maf} TFLOPS
- **Peak HBM BW:** {peak_hbm_bw} TB/s

**Note:** MoE, BatchNorm, and Convolution operations are now analyzed in separate categories.

## Operations Breakdown

| Operation | Count | Time (ms) | % of Category | Efficiency | Category |
|-----------|-------|-----------|---------------|------------|----------|
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
        elif not pd.isna(row.get('TFLOPS/s_mean')):
            eff = (row.get('TFLOPS/s_mean', 0) / peak_maf) * 100
            efficiency = f"{eff:.1f}%"
        
        # Determine category
        category = "Other"
        if 'all_reduce' in op_name.lower() or 'nccl' in op_name.lower():
            category = "Comm"
        elif 'graph' in op_name.lower():
            category = "Graph"
        
        markdown += f"| {op_name[:40]} | {count} | {time_ms:.2f} | {pct:.1f}% | {efficiency} | {category} |\n"
    
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
- **Category:** {b['category']}
- **Flagged for:** {", ".join(b['reasons'])}

"""
    else:
        markdown += "*No significant bottlenecks identified.*\n\n"
    
    markdown += """## Key Metrics
- **Communication:** Single rank limitation - can only observe collectives from one perspective
- **Graph operations:** CUDA/HIP graphs for kernel launch optimization
- **Miscellaneous:** Operations not fitting standard categories

## Additional Notes
"""
    
    if len(comm_ops) > 0:
        markdown += "- **Communication detected:** Review topology and collective types\n"
    if len(graph_ops) > 0:
        markdown += "- **Graph operations detected:** Check for graph capture and replay overhead\n"
    if len(misc_ops) > 0:
        markdown += f"- **{len(misc_ops)} miscellaneous operations:** May include memory operations, synchronization, etc.\n"
    
    print(markdown)


if __name__ == "__main__":
    main()
