#!/usr/bin/env python3
"""Triton Kernel Analysis

Computes metrics for Triton kernels and outputs JSON for subagent processing.
"""

import argparse
import sys
import os
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analysis_utils import (
    load_category_data,
    calculate_time_metrics,
    build_operation_metrics,
    calculate_average_efficiency,
    write_metrics_json
)


def get_triton_config():
    """Return Triton-specific configuration."""
    return {
        'efficiency_method': 'auto',  # Triton can be compute or memory bound
        'extra_fields': ['FLOPS/Byte'],
        'operation_classifier': classify_triton_operation
    }


def classify_triton_operation(op_name: str, row) -> dict:
    """Classify Triton kernel bound type."""
    flops_byte = row.get('FLOPS/Byte', 0) if not pd.isna(row.get('FLOPS/Byte')) else 0
    
    if flops_byte > 100:
        bound_type = 'compute'
    elif flops_byte < 50:
        bound_type = 'memory'
    else:
        bound_type = 'mixed'
    
    return {
        'bound_type': bound_type,
        'flops_per_byte': round(flops_byte, 2) if flops_byte else None
    }


def extract_category_specific(ops_df, metadata) -> dict:
    """Extract Triton-specific aggregate metrics."""
    # Count by bound type
    compute_bound = 0
    memory_bound = 0
    
    if 'FLOPS/Byte' in ops_df.columns:
        for _, row in ops_df.iterrows():
            fb = row.get('FLOPS/Byte', 0) if not pd.isna(row.get('FLOPS/Byte')) else 0
            if fb > 100:
                compute_bound += 1
            elif fb < 50:
                memory_bound += 1
    
    return {
        'compute_bound_count': compute_bound,
        'memory_bound_count': memory_bound,
        'peak_maf_tflops': metadata.get('peak_bf16_maf_tflops'),
        'peak_hbm_bw_tbs': metadata.get('peak_hbm_bw_tbs')
    }


def main():
    parser = argparse.ArgumentParser(description='Analyze Triton kernels')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    args = parser.parse_args()
    
    try:
        ops_df, metadata = load_category_data(args.output_dir, 'triton')
    except FileNotFoundError as e:
        error_metrics = {
            'category': 'triton',
            'status': 'ERROR',
            'error': str(e)
        }
        write_metrics_json(error_metrics, args.output_dir, 'triton')
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    config = get_triton_config()
    peak_hbm_bw = metadata.get('peak_hbm_bw_tbs', 1)
    peak_maf = metadata.get('peak_bf16_maf_tflops', 1)
    
    time_metrics = calculate_time_metrics(ops_df, metadata)
    avg_efficiency = calculate_average_efficiency(ops_df, peak_hbm_bw, peak_maf, 'auto')
    operations = build_operation_metrics(ops_df, metadata, config)
    category_specific = extract_category_specific(ops_df, metadata)
    
    metrics = {
        'category': 'triton',
        'status': 'OK',
        **time_metrics,
        'average_efficiency_percent': avg_efficiency,
        'operations': operations,
        'category_specific': category_specific
    }
    
    output_path = write_metrics_json(metrics, args.output_dir, 'triton')
    print(f"Metrics written to: {output_path}")


if __name__ == "__main__":
    main()
