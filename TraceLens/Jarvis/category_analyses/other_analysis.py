#!/usr/bin/env python3
"""Generic/Other Operations Analysis

Computes metrics for other/generic operations and outputs JSON for subagent processing.
This script handles Communication, Graph, and miscellaneous operations.
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
    classify_other_operation
)


def get_other_config():
    """Return other-specific configuration."""
    return {
        'efficiency_method': 'prefer_memory',  # Most "other" ops are memory-bound
        'extra_fields': [],
        'operation_classifier': classify_other_op
    }


def classify_other_op(op_name: str, row) -> dict:
    """Classify other operation type."""
    category = classify_other_operation(op_name)
    
    return {
        'sub_category': category
    }


def extract_category_specific(ops_df, metadata) -> dict:
    """Extract other-specific aggregate metrics."""
    comm_count = 0
    graph_count = 0
    misc_count = 0
    
    for name in ops_df['name']:
        category = classify_other_operation(str(name))
        if category == 'communication':
            comm_count += 1
        elif category == 'graph':
            graph_count += 1
        else:
            misc_count += 1
    
    return {
        'communication_count': comm_count,
        'graph_count': graph_count,
        'miscellaneous_count': misc_count,
        'peak_maf_tflops': metadata.get('peak_bf16_maf_tflops'),
        'peak_hbm_bw_tbs': metadata.get('peak_hbm_bw_tbs')
    }


def main():
    parser = argparse.ArgumentParser(description='Analyze other/generic operations')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--category', default='other', help='Category to analyze (default: other)')
    args = parser.parse_args()
    
    category = args.category
    
    try:
        ops_df, metadata = load_category_data(args.output_dir, category)
    except FileNotFoundError as e:
        error_metrics = {
            'category': category,
            'status': 'ERROR',
            'error': str(e)
        }
        write_metrics_json(error_metrics, args.output_dir, category)
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    config = get_other_config()
    peak_hbm_bw = metadata.get('peak_hbm_bw_tbs', 1)
    peak_maf = metadata.get('peak_bf16_maf_tflops', 1)
    
    time_metrics = calculate_time_metrics(ops_df, metadata)
    avg_efficiency = calculate_average_efficiency(ops_df, peak_hbm_bw, peak_maf, 'prefer_memory')
    operations = build_operation_metrics(ops_df, metadata, config)
    category_specific = extract_category_specific(ops_df, metadata)
    
    metrics = {
        'category': category,
        'status': 'OK',
        **time_metrics,
        'average_efficiency_percent': avg_efficiency,
        'operations': operations,
        'category_specific': category_specific
    }
    
    output_path = write_metrics_json(metrics, args.output_dir, category)
    print(f"Metrics written to: {output_path}")


if __name__ == "__main__":
    main()
