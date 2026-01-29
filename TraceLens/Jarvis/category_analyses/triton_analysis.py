#!/usr/bin/env python3
"""Triton Kernel Analysis"""

import pandas as pd
import json
import numpy as np
import os
import argparse


def main():
    parser = argparse.ArgumentParser(description='Analyze Triton kernels')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    args = parser.parse_args()
    
    output_dir = args.output_dir
    
    # Load data
    ops_df = pd.read_csv(f'{output_dir}/category_data/triton_ops.csv')
    with open(f'{output_dir}/metadata/triton_metadata.json', 'r') as f:
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
    
    # Identify potential bottlenecks
    bottlenecks = []
    
    for idx, row in ops_df.iterrows():
        op_name = row.get('name', 'Unknown')
        time_ms = row.get('Kernel Time (µs)_sum', 0) / 1000
        count = row.get('count', 1)
        flops_byte = row.get('FLOPS/Byte', 0)
        
        # Calculate efficiency
        efficiency = None
        bound_type = "unknown"
        if not pd.isna(row.get('TB/s_mean')) and flops_byte < 50:
            tb_s = row.get('TB/s_mean', 0)
            efficiency = (tb_s / peak_hbm_bw) * 100
            bound_type = "memory-bound"
        elif not pd.isna(row.get('TFLOPS/s_mean')) and flops_byte > 100:
            tflops_s = row.get('TFLOPS/s_mean', 0)
            efficiency = (tflops_s / peak_maf) * 100
            bound_type = "compute-bound"
        
        # Apply bottleneck criteria
        reasons = []
        if time_ms > 10 or (time_ms / total_time_ms * 100) > 5:
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
                'bound_type': bound_type,
                'flops_byte': flops_byte,
                'reasons': reasons
            })
    
    bottlenecks.sort(key=lambda x: x['time_ms'], reverse=True)
    
    # Generate markdown output
    markdown = f"""# Triton Kernel Analysis Results

## Summary
- **Total operations:** {len(ops_df)}
- **Total time:** {total_time_ms:.2f} ms ({percent_of_compute:.1f}% of compute)
- **Peak MAF:** {peak_maf} TFLOPS
- **Peak HBM BW:** {peak_hbm_bw} TB/s

## Operations Breakdown

| Operation | Count | Time (ms) | % of Category | Efficiency | FLOPS/Byte | Bound Type |
|-----------|-------|-----------|---------------|------------|------------|------------|
"""
    
    top_ops = ops_df.nlargest(10, 'Kernel Time (µs)_sum') if 'Kernel Time (µs)_sum' in ops_df.columns else ops_df.head(10)
    for _, row in top_ops.iterrows():
        op_name = row.get('name', 'Unknown')
        count = row.get('count', 1)
        time_ms = row.get('Kernel Time (µs)_sum', 0) / 1000
        pct = (time_ms / total_time_ms * 100) if total_time_ms > 0 else 0
        flops_byte = row.get('FLOPS/Byte', 0)
        
        efficiency = "N/A"
        bound_type = "unknown"
        if not pd.isna(row.get('TB/s_mean')) and flops_byte < 50:
            eff = (row.get('TB/s_mean', 0) / peak_hbm_bw) * 100
            efficiency = f"{eff:.1f}%"
            bound_type = "memory"
        elif not pd.isna(row.get('TFLOPS/s_mean')) and flops_byte > 100:
            eff = (row.get('TFLOPS/s_mean', 0) / peak_maf) * 100
            efficiency = f"{eff:.1f}%"
            bound_type = "compute"
        
        fb_str = f"{flops_byte:.1f}" if not pd.isna(flops_byte) else "N/A"
        markdown += f"| {op_name[:40]} | {count} | {time_ms:.2f} | {pct:.1f}% | {efficiency} | {fb_str} | {bound_type} |\n"
    
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
- **Bound Type:** {b['bound_type']}
- **FLOPS/Byte:** {f"{b['flops_byte']:.1f}" if not pd.isna(b['flops_byte']) else "N/A"}
- **Flagged for:** {", ".join(b['reasons'])}

"""
    else:
        markdown += "*No significant bottlenecks identified.*\n\n"
    
    markdown += """## Key Metrics
- **Custom kernels:** Triton kernels are user-written, performance varies
- **Optimization potential:** Review tile sizes, memory access patterns
- **Comparison:** Validate performance against equivalent PyTorch operations

"""
    
    print(markdown)


if __name__ == "__main__":
    main()
