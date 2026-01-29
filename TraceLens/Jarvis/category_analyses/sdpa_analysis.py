#!/usr/bin/env python3
"""SDPA Analysis - Scaled Dot Product Attention"""

import pandas as pd
import json
import numpy as np
import os
import argparse


def main():
    parser = argparse.ArgumentParser(description='Analyze SDPA operations')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    args = parser.parse_args()
    
    output_dir = args.output_dir
    
    # Load data
    ops_df = pd.read_csv(f'{output_dir}/category_data/sdpa_fwd_ops.csv')
    with open(f'{output_dir}/metadata/sdpa_fwd_metadata.json', 'r') as f:
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
        has_perf_model = row.get('has_perf_model', False)
        if has_perf_model:
            if not pd.isna(row.get('TB/s_mean')):
                eff = (row.get('TB/s_mean', 0) / peak_hbm_bw) * 100
                avg_efficiency += eff
                eff_count += 1
            elif not pd.isna(row.get('TFLOPS/s_mean')):
                eff = (row.get('TFLOPS/s_mean', 0) / peak_maf) * 100
                avg_efficiency += eff
                eff_count += 1
    avg_efficiency = avg_efficiency / eff_count if eff_count > 0 else 0
    
    # Identify potential bottlenecks
    bottlenecks = []
    flash_attention_count = 0
    
    for idx, row in ops_df.iterrows():
        op_name = row.get('name', 'Unknown')
        time_ms = row.get('Kernel Time (µs)_sum', 0) / 1000
        count = row.get('count', 1)
        has_perf_model = row.get('has_perf_model', False)
        is_flash_attention = 'flash' in op_name.lower() or 'fmha' in op_name.lower()
        
        if is_flash_attention:
            flash_attention_count += 1
        
        # Calculate efficiency
        efficiency = None
        if has_perf_model:
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
        if not has_perf_model:
            reasons.append("Missing perf model")
        
        if reasons:
            bottlenecks.append({
                'op_name': op_name,
                'time_ms': time_ms,
                'percent_of_category': (time_ms / total_time_ms * 100) if total_time_ms > 0 else 0,
                'count': count,
                'efficiency': efficiency,
                'has_perf_model': has_perf_model,
                'is_flash_attention': is_flash_attention,
                'reasons': reasons
            })
    
    # Sort by time
    bottlenecks.sort(key=lambda x: x['time_ms'], reverse=True)
    
    # Generate markdown output
    markdown = f"""# SDPA Analysis Results

## Summary
- **Total operations:** {len(ops_df)}
- **Total time:** {total_time_ms:.2f} ms ({percent_of_compute:.1f}% of compute)
- **Average efficiency:** {avg_efficiency:.1f}% of peak
- **Flash Attention operations:** {flash_attention_count}
- **Peak MAF:** {peak_maf} TFLOPS
- **Peak HBM BW:** {peak_hbm_bw} TB/s

## Operations Breakdown

| Operation | Count | Time (ms) | % of Category | Efficiency | Type |
|-----------|-------|-----------|---------------|------------|------|
"""
    
    # Show top 10 operations by time
    top_ops = ops_df.nlargest(10, 'Kernel Time (µs)_sum') if 'Kernel Time (µs)_sum' in ops_df.columns else ops_df.head(10)
    for _, row in top_ops.iterrows():
        op_name = row.get('name', 'Unknown')
        count = row.get('count', 1)
        time_ms = row.get('Kernel Time (µs)_sum', 0) / 1000
        pct = (time_ms / total_time_ms * 100) if total_time_ms > 0 else 0
        
        efficiency = "N/A"
        has_perf_model = row.get('has_perf_model', False)
        if has_perf_model:
            if not pd.isna(row.get('TB/s_mean')):
                eff = (row.get('TB/s_mean', 0) / peak_hbm_bw) * 100
                efficiency = f"{eff:.1f}%"
            elif not pd.isna(row.get('TFLOPS/s_mean')):
                eff = (row.get('TFLOPS/s_mean', 0) / peak_maf) * 100
                efficiency = f"{eff:.1f}%"
        
        is_flash = 'flash' in op_name.lower() or 'fmha' in op_name.lower()
        op_type = "Flash Attention" if is_flash else "Standard SDPA"
        
        markdown += f"| {op_name[:50]} | {count} | {time_ms:.2f} | {pct:.1f}% | {efficiency} | {op_type} |\n"
    
    markdown += f"""
## Potential Bottlenecks

**{len(bottlenecks)} operations flagged** based on criteria: High time (>100ms or >5% of category), Low efficiency (<40%), High count (>1000), or Missing perf model

"""
    
    if bottlenecks:
        for i, b in enumerate(bottlenecks[:10], 1):
            markdown += f"""### {i}. {b['op_name']}
- **Time:** {b['time_ms']:.2f} ms ({b['percent_of_category']:.1f}% of category)
- **Count:** {b['count']}
- **Efficiency:** {f"{b['efficiency']:.1f}%" if b['efficiency'] else "N/A"}
- **Type:** {"Flash Attention" if b['is_flash_attention'] else "Standard/Unfused SDPA"}
- **Has Perf Model:** {b['has_perf_model']}
- **Flagged for:** {", ".join(b['reasons'])}

"""
    else:
        markdown += "*No significant bottlenecks identified.*\n\n"
    
    markdown += f"""## Key Metrics
- **Expected efficiency:** Flash Attention typically achieves 40-70% depending on sequence length
- **Short sequences (<1024):** 8-15% efficiency is expected
- **Long sequences (>2048):** 50-70% efficiency typical
- **Unfused attention:** Typically 3-10x slower than Flash Attention

## Additional Notes
"""
    
    # Check for missing perf models
    missing_models = [row['name'] for _, row in ops_df.iterrows() if not row.get('has_perf_model', False)]
    if missing_models:
        markdown += f"- **Missing perf models:** {len(missing_models)} operations lack performance models\n"
    
    if flash_attention_count == 0:
        markdown += "- **No Flash Attention detected:** Consider migrating to Flash Attention for 3-10x speedup\n"
    
    # Output to stdout for LLM to read
    print(markdown)


if __name__ == "__main__":
    main()
