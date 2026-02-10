#!/usr/bin/env python3
"""Convolution Operations Analysis

Computes metrics for Convolution operations and outputs JSON for subagent processing.
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
    write_metrics_json,
    detect_transpose
)


def get_convolution_config():
    """Return Convolution-specific configuration."""
    return {
        'efficiency_method': 'prefer_compute',  # Convolutions can be compute-bound
        'extra_fields': [],
        'operation_classifier': classify_convolution_operation
    }


def classify_convolution_operation(op_name: str, row) -> dict:
    """Classify Convolution operation type."""
    is_transpose = detect_transpose(op_name)
    
    op_lower = op_name.lower()
    if is_transpose:
        conv_type = 'transpose'
    elif 'conv2d' in op_lower:
        conv_type = 'conv2d'
    elif 'conv1d' in op_lower:
        conv_type = 'conv1d'
    elif 'conv3d' in op_lower:
        conv_type = 'conv3d'
    elif 'depthwise' in op_lower:
        conv_type = 'depthwise'
    else:
        conv_type = 'other'
    
    return {
        'is_transpose': is_transpose,
        'conv_type': conv_type
    }


def extract_category_specific(ops_df, metadata) -> dict:
    """Extract Convolution-specific aggregate metrics."""
    # Check for transpose operations (layout issue indicator)
    transpose_ops = ops_df[ops_df['name'].str.contains('transpose', case=False, na=False)]
    transpose_time_ms = transpose_ops['Kernel Time (µs)_sum'].sum() / 1000 if len(transpose_ops) > 0 else 0
    
    total_time_ms = 0
    if 'Kernel Time (µs)_sum' in ops_df.columns:
        total_time_ms = ops_df['Kernel Time (µs)_sum'].sum() / 1000
    
    transpose_overhead_pct = (transpose_time_ms / total_time_ms * 100) if total_time_ms > 0 else 0
    
    return {
        'transpose_count': len(transpose_ops),
        'transpose_time_ms': round(transpose_time_ms, 3),
        'transpose_overhead_percent': round(transpose_overhead_pct, 2),
        'peak_maf_tflops': metadata.get('max_achievable_tflops', {}).get('matrix_bf16') if isinstance(metadata.get('max_achievable_tflops'), dict) else metadata.get('peak_bf16_maf_tflops'),
        'peak_hbm_bw_tbs': metadata.get('peak_hbm_bw_tbs')
    }


def main():
    parser = argparse.ArgumentParser(description='Analyze Convolution operations')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    args = parser.parse_args()
    
    try:
        ops_df, metadata = load_category_data(args.output_dir, 'convolution')
    except FileNotFoundError as e:
        error_metrics = {
            'category': 'convolution',
            'status': 'ERROR',
            'error': str(e)
        }
        write_metrics_json(error_metrics, args.output_dir, 'convolution')
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    config = get_convolution_config()
    peak_hbm_bw = metadata.get('peak_hbm_bw_tbs', 1)
    maf = metadata.get('max_achievable_tflops', metadata.get('peak_bf16_maf_tflops', 1))
    
    time_metrics = calculate_time_metrics(ops_df, metadata)
    avg_efficiency = calculate_average_efficiency(ops_df, peak_hbm_bw, maf, 'prefer_compute')
    operations = build_operation_metrics(ops_df, metadata, config)
    category_specific = extract_category_specific(ops_df, metadata)
    
    metrics = {
        'category': 'convolution',
        'status': 'OK',
        **time_metrics,
        'average_efficiency_percent': avg_efficiency,
        'operations': operations,
        'category_specific': category_specific
    }
    
    output_path = write_metrics_json(metrics, args.output_dir, 'convolution')
    print(f"Metrics written to: {output_path}")


if __name__ == "__main__":
    main()
