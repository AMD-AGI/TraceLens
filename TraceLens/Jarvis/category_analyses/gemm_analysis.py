#!/usr/bin/env python3
"""GEMM Analysis - Matrix Multiplications"""

import pandas as pd
import json
import numpy as np
import os
import argparse


def main():
    parser = argparse.ArgumentParser(description='Analyze GEMM operations')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    args = parser.parse_args()
    
    output_dir = args.output_dir
    
    # Load data
    ops_df = pd.read_csv(f'{output_dir}/category_data/gemm_ops.csv')
    with open(f'{output_dir}/metadata/gemm_metadata.json', 'r') as f:
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
        if not pd.isna(row.get('TFLOPS/s_mean')) and row.get('FLOPS/Byte', 0) > 100:
            eff = (row.get('TFLOPS/s_mean', 0) / peak_maf) * 100
            avg_efficiency += eff
            eff_count += 1
    avg_efficiency = avg_efficiency / eff_count if eff_count > 0 else 0
    
    # Identify potential bottlenecks
    bottlenecks = []
    for idx, row in ops_df.iterrows():
        op_name = row.get('name', 'Unknown')
        time_ms = row.get('Kernel Time (µs)_sum', 0) / 1000
        count = row.get('count', 1)
        flops_byte = row.get('FLOPS/Byte', 0)
        
        # Calculate efficiency
        efficiency = None
        if not pd.isna(row.get('TFLOPS/s_mean')) and flops_byte > 100:
            tflops_s = row.get('TFLOPS/s_mean', 0)
            efficiency = (tflops_s / peak_maf) * 100
        
        # Apply bottleneck criteria
        reasons = []
        if time_ms > 50 or (time_ms / total_time_ms * 100) > 5:
            reasons.append("High time")
        if efficiency and efficiency < 40:
            reasons.append("Low efficiency")
        if count > 1000:
            reasons.append("High count")
        if pd.isna(row.get('TFLOPS/s_mean')):
            reasons.append("Missing perf model")
        
        if reasons:
            is_quantized = 'w8a8' in op_name.lower() or 'int8' in op_name.lower() or 'fp8' in op_name.lower()
            bottlenecks.append({
                'op_name': op_name,
                'time_ms': time_ms,
                'percent_of_category': (time_ms / total_time_ms * 100) if total_time_ms > 0 else 0,
                'count': count,
                'efficiency': efficiency,
                'flops_byte': flops_byte,
                'is_quantized': is_quantized,
                'reasons': reasons
            })
    
    # Sort by time
    bottlenecks.sort(key=lambda x: x['time_ms'], reverse=True)
    
    # Generate markdown output
    markdown = f"""# GEMM Analysis Results

## Summary
- **Total operations:** {len(ops_df)}
- **Total time:** {total_time_ms:.2f} ms ({percent_of_compute:.1f}% of compute)
- **Average efficiency:** {avg_efficiency:.1f}% of peak MAF
- **Peak MAF:** {peak_maf} TFLOPS
- **Peak HBM BW:** {peak_hbm_bw} TB/s

## Operations Breakdown

| Operation | Count | Time (ms) | % of Category | Efficiency | FLOPS/Byte | Type |
|-----------|-------|-----------|---------------|------------|------------|------|
"""
    
    # Show top 10 operations by time
    top_ops = ops_df.nlargest(10, 'Kernel Time (µs)_sum') if 'Kernel Time (µs)_sum' in ops_df.columns else ops_df.head(10)
    for _, row in top_ops.iterrows():
        op_name = row.get('name', 'Unknown')
        count = row.get('count', 1)
        time_ms = row.get('Kernel Time (µs)_sum', 0) / 1000
        pct = (time_ms / total_time_ms * 100) if total_time_ms > 0 else 0
        flops_byte = row.get('FLOPS/Byte', 0)
        
        efficiency = "N/A"
        if not pd.isna(row.get('TFLOPS/s_mean')) and flops_byte > 100:
            eff = (row.get('TFLOPS/s_mean', 0) / peak_maf) * 100
            efficiency = f"{eff:.1f}%"
        
        op_type = "Quantized" if any(x in op_name.lower() for x in ['w8a8', 'int8', 'fp8']) else "Regular"
        fb_str = f"{flops_byte:.1f}" if not pd.isna(flops_byte) else "N/A"
        
        markdown += f"| {op_name[:50]} | {count} | {time_ms:.2f} | {pct:.1f}% | {efficiency} | {fb_str} | {op_type} |\n"
    
    markdown += f"""
## Potential Bottlenecks

**{len(bottlenecks)} operations flagged** based on criteria: High time (>50ms or >5% of category), Low efficiency (<40%), High count (>1000), or Missing perf model

"""
    
    if bottlenecks:
        for i, b in enumerate(bottlenecks[:10], 1):
            markdown += f"""### {i}. {b['op_name']}
- **Time:** {b['time_ms']:.2f} ms ({b['percent_of_category']:.1f}% of category)
- **Count:** {b['count']}
- **Efficiency:** {f"{b['efficiency']:.1f}%" if b['efficiency'] else "N/A"}
- **FLOPS/Byte:** {f"{b['flops_byte']:.1f}" if not pd.isna(b['flops_byte']) else "N/A"}
- **Type:** {"Quantized GEMM" if b['is_quantized'] else "Regular GEMM"}
- **Flagged for:** {", ".join(b['reasons'])}

"""
    else:
        markdown += "*No significant bottlenecks identified.*\n\n"
    
    markdown += f"""## Key Metrics
- **Compute-bound threshold:** FLOPS/Byte > 100
- **Expected efficiency:** GEMMs typically achieve 60-80% of peak MAF
- **Quantized GEMMs:** {sum(1 for b in bottlenecks if b.get('is_quantized', False))} detected

## Additional Notes
"""
    
    # Check for missing perf models
    missing_models = [row['name'] for _, row in ops_df.iterrows() if pd.isna(row.get('TFLOPS/s_mean'))]
    if missing_models:
        markdown += f"- **Missing perf models:** {len(missing_models)} operations lack performance models\n"
    
    # Output to stdout for LLM to read
    print(markdown)


if __name__ == "__main__":
    main()
