#!/usr/bin/env python3
"""Reduce Operations Analysis

Computes metrics for reduce operations and outputs JSON for subagent processing.
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analysis_utils import (
    load_category_data,
    calculate_time_metrics,
    build_operation_metrics,
    calculate_average_efficiency,
    write_metrics_json,
    detect_softmax
)


def get_reduce_config():
    """Return reduce-specific configuration."""
    return {
        'efficiency_method': 'memory_bound',  # Reduce ops are memory-bound
        'extra_fields': [],
        'operation_classifier': classify_reduce_operation
    }


def classify_reduce_operation(op_name: str, row) -> dict:
    """Classify reduce operation type."""
    is_softmax = detect_softmax(op_name)
    
    op_lower = op_name.lower()
    if is_softmax:
        reduce_type = 'softmax'
    elif 'sum' in op_lower:
        reduce_type = 'sum'
    elif 'mean' in op_lower or 'avg' in op_lower:
        reduce_type = 'mean'
    elif 'max' in op_lower:
        reduce_type = 'max'
    elif 'min' in op_lower:
        reduce_type = 'min'
    else:
        reduce_type = 'other'
    
    return {
        'is_softmax': is_softmax,
        'reduce_type': reduce_type
    }


def extract_category_specific(ops_df, metadata) -> dict:
    """Extract reduce-specific aggregate metrics."""
    softmax_count = sum(1 for name in ops_df['name'] if detect_softmax(str(name)))
    
    return {
        'softmax_count': int(softmax_count),
        'peak_hbm_bw_tbs': metadata.get('peak_hbm_bw_tbs')
    }


def main():
    parser = argparse.ArgumentParser(description='Analyze reduce operations')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    args = parser.parse_args()
    
    try:
        ops_df, metadata = load_category_data(args.output_dir, 'reduce')
    except FileNotFoundError as e:
        error_metrics = {
            'category': 'reduce',
            'status': 'ERROR',
            'error': str(e)
        }
        write_metrics_json(error_metrics, args.output_dir, 'reduce')
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    config = get_reduce_config()
    peak_hbm_bw = metadata.get('peak_hbm_bw_tbs', 1)
    peak_maf = metadata.get('peak_bf16_maf_tflops', 1)
    
    time_metrics = calculate_time_metrics(ops_df, metadata)
    avg_efficiency = calculate_average_efficiency(ops_df, peak_hbm_bw, peak_maf, 'memory_bound')
    operations = build_operation_metrics(ops_df, metadata, config)
    category_specific = extract_category_specific(ops_df, metadata)
    
    metrics = {
        'category': 'reduce',
        'status': 'OK',
        **time_metrics,
        'average_efficiency_percent': avg_efficiency,
        'operations': operations,
        'category_specific': category_specific
    }
    
    output_path = write_metrics_json(metrics, args.output_dir, 'reduce')
    print(f"Metrics written to: {output_path}")


if __name__ == "__main__":
    main()
