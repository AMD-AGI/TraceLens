#!/usr/bin/env python3
"""Elementwise Operations Analysis

Computes metrics for elementwise operations and outputs JSON for subagent processing.
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


def get_elementwise_config():
    """Return elementwise-specific configuration."""
    return {
        'efficiency_method': 'memory_bound',  # Elementwise ops are memory-bound
        'extra_fields': ['FLOPS/Byte'],
        'operation_classifier': classify_elementwise_operation
    }


def classify_elementwise_operation(op_name: str, row) -> dict:
    """Classify elementwise operation type."""
    op_lower = op_name.lower()
    
    if 'add' in op_lower:
        op_type = 'add'
    elif 'mul' in op_lower:
        op_type = 'multiply'
    elif 'copy' in op_lower:
        op_type = 'copy'
    elif 'div' in op_lower:
        op_type = 'divide'
    elif 'sub' in op_lower:
        op_type = 'subtract'
    else:
        op_type = 'other'
    
    return {
        'op_type': op_type,
        'is_baseline': op_lower in ['aten::add_', 'aten::mul', 'aten::copy_']
    }


def extract_category_specific(ops_df, metadata) -> dict:
    """Extract elementwise-specific aggregate metrics."""
    peak_hbm_bw = metadata.get('peak_hbm_bw_tbs', 1)
    
    # Find baseline bandwidth from simple ops
    baseline_ops = ops_df[ops_df['name'].isin(['aten::add_', 'aten::mul', 'aten::copy_'])]
    if len(baseline_ops) > 0 and 'TB/s_mean' in baseline_ops.columns:
        baseline_bw = baseline_ops['TB/s_mean'].mean()
        baseline_efficiency = (baseline_bw / peak_hbm_bw) * 100 if peak_hbm_bw > 0 else 0
    else:
        baseline_bw = peak_hbm_bw * 0.7
        baseline_efficiency = 70.0
    
    return {
        'baseline_bandwidth_tbs': round(baseline_bw, 2) if not pd.isna(baseline_bw) else None,
        'baseline_efficiency_percent': round(baseline_efficiency, 2) if not pd.isna(baseline_efficiency) else None,
        'peak_hbm_bw_tbs': metadata.get('peak_hbm_bw_tbs')
    }


def main():
    parser = argparse.ArgumentParser(description='Analyze elementwise operations')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    args = parser.parse_args()
    
    try:
        ops_df, metadata = load_category_data(args.output_dir, 'elementwise')
    except FileNotFoundError as e:
        error_metrics = {
            'category': 'elementwise',
            'status': 'ERROR',
            'error': str(e)
        }
        write_metrics_json(error_metrics, args.output_dir, 'elementwise')
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    config = get_elementwise_config()
    peak_hbm_bw = metadata.get('peak_hbm_bw_tbs', 1)
    maf = metadata.get('max_achievable_tflops', metadata.get('peak_bf16_maf_tflops', 1))
    
    time_metrics = calculate_time_metrics(ops_df, metadata)
    avg_efficiency = calculate_average_efficiency(ops_df, peak_hbm_bw, maf, 'memory_bound')
    operations = build_operation_metrics(ops_df, metadata, config)
    category_specific = extract_category_specific(ops_df, metadata)
    
    metrics = {
        'category': 'elementwise',
        'status': 'OK',
        **time_metrics,
        'average_efficiency_percent': avg_efficiency,
        'operations': operations,
        'category_specific': category_specific
    }
    
    output_path = write_metrics_json(metrics, args.output_dir, 'elementwise')
    print(f"Metrics written to: {output_path}")


if __name__ == "__main__":
    main()
