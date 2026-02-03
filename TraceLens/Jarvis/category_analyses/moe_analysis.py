#!/usr/bin/env python3
"""MoE (Mixture of Experts) Analysis

Computes metrics for MoE fused operations and outputs JSON for subagent processing.
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


def get_moe_config():
    """Return MoE-specific configuration."""
    return {
        'efficiency_method': 'prefer_compute',  # MoE ops can be compute or memory bound
        'extra_fields': [],
        'operation_classifier': classify_moe_operation
    }


def classify_moe_operation(op_name: str, row) -> dict:
    """Classify MoE operation type."""
    op_lower = op_name.lower()
    
    if 'gate' in op_lower or 'router' in op_lower:
        moe_type = 'routing'
    elif 'expert' in op_lower:
        moe_type = 'expert'
    else:
        moe_type = 'fused'
    
    return {
        'moe_type': moe_type
    }


def extract_category_specific(ops_df, metadata) -> dict:
    """Extract MoE-specific aggregate metrics."""
    return {
        'peak_maf_tflops': metadata.get('peak_bf16_maf_tflops'),
        'peak_hbm_bw_tbs': metadata.get('peak_hbm_bw_tbs')
    }


def main():
    parser = argparse.ArgumentParser(description='Analyze MoE fused operations')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    args = parser.parse_args()
    
    # Check if MoE data exists
    moe_csv = f'{args.output_dir}/category_data/moe_fused_ops.csv'
    if not os.path.exists(moe_csv):
        # No MoE operations - write minimal metrics
        metrics = {
            'category': 'moe_fused',
            'status': 'NO_DATA',
            'message': 'No MoE operations detected in this trace',
            'total_time_ms': 0,
            'percent_of_compute': 0,
            'operation_count': 0,
            'operations': [],
            'category_specific': {}
        }
        output_path = write_metrics_json(metrics, args.output_dir, 'moe_fused')
        print(f"No MoE data. Metrics written to: {output_path}")
        return
    
    try:
        ops_df, metadata = load_category_data(args.output_dir, 'moe_fused')
    except FileNotFoundError as e:
        error_metrics = {
            'category': 'moe_fused',
            'status': 'ERROR',
            'error': str(e)
        }
        write_metrics_json(error_metrics, args.output_dir, 'moe_fused')
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    config = get_moe_config()
    peak_hbm_bw = metadata.get('peak_hbm_bw_tbs', 1)
    peak_maf = metadata.get('peak_bf16_maf_tflops', 1)
    
    time_metrics = calculate_time_metrics(ops_df, metadata)
    avg_efficiency = calculate_average_efficiency(ops_df, peak_hbm_bw, peak_maf, 'prefer_compute')
    operations = build_operation_metrics(ops_df, metadata, config)
    category_specific = extract_category_specific(ops_df, metadata)
    
    metrics = {
        'category': 'moe_fused',
        'status': 'OK',
        **time_metrics,
        'average_efficiency_percent': avg_efficiency,
        'operations': operations,
        'category_specific': category_specific
    }
    
    output_path = write_metrics_json(metrics, args.output_dir, 'moe_fused')
    print(f"Metrics written to: {output_path}")


if __name__ == "__main__":
    main()
