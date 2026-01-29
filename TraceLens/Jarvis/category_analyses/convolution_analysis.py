#!/usr/bin/env python3
"""Convolution Operations Analysis"""

import pandas as pd
import json
import numpy as np
import os
import argparse


def main():
    parser = argparse.ArgumentParser(description='Analyze Convolution operations')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    args = parser.parse_args()
    
    output_dir = args.output_dir
    
    # Load data
    ops_df = pd.read_csv(f'{output_dir}/category_data/convolution_ops.csv')
    with open(f'{output_dir}/metadata/convolution_metadata.json', 'r') as f:
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
    
    # Calculate average efficiency
    avg_efficiency = 0
    eff_count = 0
    for _, row in ops_df.iterrows():
        if not pd.isna(row.get('TFLOPS/s_mean')):
            eff = (row.get('TFLOPS/s_mean', 0) / peak_maf) * 100
            avg_efficiency += eff
            eff_count += 1
        elif not pd.isna(row.get('TB/s_mean')):
            eff = (row.get('TB/s_mean', 0) / peak_hbm_bw) * 100
            avg_efficiency += eff
            eff_count += 1
    avg_efficiency = avg_efficiency / eff_count if eff_count > 0 else 0
    
    # Check for transpose operations (layout issue indicator)
    transpose_ops = ops_df[ops_df['name'].str.contains('transpose', case=False, na=False)]
    transpose_time_ms = transpose_ops['Kernel Time (µs)_sum'].sum() / 1000 if len(transpose_ops) > 0 else 0
    transpose_overhead_pct = (transpose_time_ms / total_time_ms * 100) if total_time_ms > 0 else 0
    
    # Identify potential bottlenecks
    bottlenecks = []
    
    for idx, row in ops_df.iterrows():
        op_name = row.get('name', 'Unknown')
        time_ms = row.get('Kernel Time (µs)_sum', 0) / 1000
        count = row.get('count', 1)
        
        # Calculate efficiency
        efficiency = None
        if not pd.isna(row.get('TFLOPS/s_mean')):
            efficiency = (row.get('TFLOPS/s_mean', 0) / peak_maf) * 100
        elif not pd.isna(row.get('TB/s_mean')):
            efficiency = (row.get('TB/s_mean', 0) / peak_hbm_bw) * 100
        
        # Apply bottleneck criteria
        reasons = []
        if time_ms > 50 or (time_ms / total_time_ms * 100) > 5:
            reasons.append("High time")
        if efficiency and efficiency < 40:
            reasons.append("Low efficiency")
        if count > 1000:
            reasons.append("High count")
        if efficiency is None:
            reasons.append("Missing perf model")
        if 'transpose' in op_name.lower():
            reasons.append("Transpose overhead - layout issue")
        
        if reasons:
            is_transpose = 'transpose' in op_name.lower()
            bottlenecks.append({
                'op_name': op_name,
                'time_ms': time_ms,
                'percent_of_category': (time_ms / total_time_ms * 100) if total_time_ms > 0 else 0,
                'count': count,
                'efficiency': efficiency,
                'is_transpose': is_transpose,
                'reasons': reasons
            })
    
    bottlenecks.sort(key=lambda x: x['time_ms'], reverse=True)
    
    # Generate markdown output
    markdown = f"""# Convolution Analysis Results

## Summary
- **Total operations:** {len(ops_df)}
- **Total time:** {total_time_ms:.2f} ms ({percent_of_compute:.1f}% of compute)
- **Average efficiency:** {avg_efficiency:.1f}% of peak
- **Transpose operations:** {len(transpose_ops)} ({transpose_overhead_pct:.1f}% overhead)
- **Peak MAF:** {peak_maf} TFLOPS
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
        if not pd.isna(row.get('TFLOPS/s_mean')):
            eff = (row.get('TFLOPS/s_mean', 0) / peak_maf) * 100
            efficiency = f"{eff:.1f}%"
        elif not pd.isna(row.get('TB/s_mean')):
            eff = (row.get('TB/s_mean', 0) / peak_hbm_bw) * 100
            efficiency = f"{eff:.1f}%"
        
        op_type = "Transpose" if 'transpose' in op_name.lower() else "Convolution"
        markdown += f"| {op_name[:40]} | {count} | {time_ms:.2f} | {pct:.1f}% | {efficiency} | {op_type} |\n"
    
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
- **Type:** {"Transpose (Layout Issue)" if b['is_transpose'] else "Convolution"}
- **Flagged for:** {", ".join(b['reasons'])}

"""
    else:
        markdown += "*No significant bottlenecks identified.*\n\n"
    
    markdown += f"""## Key Metrics
- **MIOpen/cuDNN kernels prefer NHWC layout**
- **PyTorch defaults to NCHW layout**
- **Result:** batched_transpose kernels add 30-45% overhead
- **Solution:** `model.to(memory_format=torch.channels_last)`
- **Transpose overhead detected:** {transpose_overhead_pct:.1f}% of convolution time

## Additional Notes
"""
    
    if transpose_overhead_pct > 20:
        markdown += f"- **⚠️ High transpose overhead detected ({transpose_overhead_pct:.1f}%):** Consider using channels_last memory format\n"
    if len(transpose_ops) > 0:
        markdown += f"- **Transpose operations found:** {len(transpose_ops)} operations taking {transpose_time_ms:.2f} ms\n"
    if percent_of_compute > 30:
        markdown += f"- **Convolutions dominate compute ({percent_of_compute:.1f}%):** CNN-heavy workload\n"
    
    print(markdown)


if __name__ == "__main__":
    main()
