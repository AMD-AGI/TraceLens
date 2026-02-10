#!/usr/bin/env python3
"""BatchNorm Operations Analysis

Computes metrics for BatchNorm operations and outputs JSON for subagent processing.
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
    write_metrics_json
)


def get_batchnorm_config():
    """Return BatchNorm-specific configuration."""
    return {
        'efficiency_method': 'memory_bound',  # BatchNorm is memory-bound
        'extra_fields': [],
        'operation_classifier': classify_batchnorm_operation
    }


def classify_batchnorm_operation(op_name: str, row) -> dict:
    """Classify BatchNorm operation type."""
    op_lower = op_name.lower()
    
    if 'batch_norm' in op_lower or 'batchnorm' in op_lower:
        bn_type = 'batch_norm'
    elif 'layer_norm' in op_lower or 'layernorm' in op_lower:
        bn_type = 'layer_norm'
    elif 'group_norm' in op_lower or 'groupnorm' in op_lower:
        bn_type = 'group_norm'
    elif 'instance_norm' in op_lower:
        bn_type = 'instance_norm'
    else:
        bn_type = 'other_norm'
    
    return {
        'norm_type': bn_type
    }


def extract_category_specific(ops_df, metadata) -> dict:
    """Extract BatchNorm-specific aggregate metrics."""
    return {
        'peak_hbm_bw_tbs': metadata.get('peak_hbm_bw_tbs')
    }


def main():
    parser = argparse.ArgumentParser(description='Analyze BatchNorm operations')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    args = parser.parse_args()
    
    try:
        ops_df, metadata = load_category_data(args.output_dir, 'batchnorm')
    except FileNotFoundError as e:
        error_metrics = {
            'category': 'batchnorm',
            'status': 'ERROR',
            'error': str(e)
        }
        write_metrics_json(error_metrics, args.output_dir, 'batchnorm')
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    config = get_batchnorm_config()
    peak_hbm_bw = metadata.get('peak_hbm_bw_tbs', 1)
    maf = metadata.get('max_achievable_tflops', metadata.get('peak_bf16_maf_tflops', 1))
    
    time_metrics = calculate_time_metrics(ops_df, metadata)
    avg_efficiency = calculate_average_efficiency(ops_df, peak_hbm_bw, maf, 'memory_bound')
    operations = build_operation_metrics(ops_df, metadata, config)
    category_specific = extract_category_specific(ops_df, metadata)
    
    metrics = {
        'category': 'batchnorm',
        'status': 'OK',
        **time_metrics,
        'average_efficiency_percent': avg_efficiency,
        'operations': operations,
        'category_specific': category_specific
    }
    
    output_path = write_metrics_json(metrics, args.output_dir, 'batchnorm')
    print(f"Metrics written to: {output_path}")


if __name__ == "__main__":
    main()
