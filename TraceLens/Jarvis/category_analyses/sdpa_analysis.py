#!/usr/bin/env python3
"""SDPA Analysis - Scaled Dot Product Attention

Computes metrics for SDPA operations and outputs JSON for subagent processing.
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
    detect_flash_attention
)


def get_sdpa_config():
    """Return SDPA-specific configuration."""
    return {
        'efficiency_method': 'prefer_compute',  # SDPA can be compute or memory bound
        'extra_fields': ['Input Dims', 'has_perf_model'],
        'operation_classifier': classify_sdpa_operation
    }


def classify_sdpa_operation(op_name: str, row) -> dict:
    """Classify SDPA operation type."""
    has_perf_model = row.get('has_perf_model', False) if 'has_perf_model' in row.index else False
    return {
        'is_flash_attention': detect_flash_attention(op_name),
        'has_perf_model': bool(has_perf_model),
        'attention_type': 'flash' if detect_flash_attention(op_name) else 'standard'
    }


def extract_category_specific(ops_df, metadata) -> dict:
    """Extract SDPA-specific aggregate metrics."""
    flash_attention_count = sum(1 for name in ops_df['name'] if detect_flash_attention(str(name)))
    
    # Count operations with perf models
    has_perf_model_count = 0
    if 'has_perf_model' in ops_df.columns:
        has_perf_model_count = ops_df['has_perf_model'].sum()
    
    return {
        'flash_attention_count': int(flash_attention_count),
        'has_perf_model_count': int(has_perf_model_count),
        'flash_attention_detected': flash_attention_count > 0,
        'peak_maf_tflops': metadata.get('peak_bf16_maf_tflops'),
        'peak_hbm_bw_tbs': metadata.get('peak_hbm_bw_tbs')
    }


def main():
    parser = argparse.ArgumentParser(description='Analyze SDPA operations')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    args = parser.parse_args()
    
    try:
        ops_df, metadata = load_category_data(args.output_dir, 'sdpa_fwd')
    except FileNotFoundError as e:
        error_metrics = {
            'category': 'sdpa_fwd',
            'status': 'ERROR',
            'error': str(e)
        }
        write_metrics_json(error_metrics, args.output_dir, 'sdpa_fwd')
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    config = get_sdpa_config()
    peak_hbm_bw = metadata.get('peak_hbm_bw_tbs', 1)
    peak_maf = metadata.get('peak_bf16_maf_tflops', 1)
    
    time_metrics = calculate_time_metrics(ops_df, metadata)
    avg_efficiency = calculate_average_efficiency(ops_df, peak_hbm_bw, peak_maf, 'prefer_compute')
    operations = build_operation_metrics(ops_df, metadata, config)
    category_specific = extract_category_specific(ops_df, metadata)
    
    metrics = {
        'category': 'sdpa_fwd',
        'status': 'OK',
        **time_metrics,
        'average_efficiency_percent': avg_efficiency,
        'operations': operations,
        'category_specific': category_specific
    }
    
    output_path = write_metrics_json(metrics, args.output_dir, 'sdpa_fwd')
    print(f"Metrics written to: {output_path}")


if __name__ == "__main__":
    main()
