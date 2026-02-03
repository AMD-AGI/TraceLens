#!/usr/bin/env python3
"""GEMM Analysis - Matrix Multiplications

Computes metrics for GEMM operations and outputs JSON for subagent processing.
"""

import argparse
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analysis_utils import (
    load_category_data,
    calculate_time_metrics,
    build_operation_metrics,
    calculate_average_efficiency,
    write_metrics_json,
    detect_quantized_gemm
)


def get_gemm_config():
    """Return GEMM-specific configuration."""
    return {
        'efficiency_method': 'auto',  # Use FLOPS/Byte to determine bound type
        'extra_fields': ['Input Dims', 'has_perf_model'],
        'operation_classifier': classify_gemm_operation
    }


def classify_gemm_operation(op_name: str, row) -> dict:
    """Classify GEMM operation type."""
    return {
        'is_quantized': detect_quantized_gemm(op_name),
        'gemm_type': 'quantized' if detect_quantized_gemm(op_name) else 'regular'
    }


def extract_category_specific(ops_df, metadata) -> dict:
    """Extract GEMM-specific aggregate metrics."""
    quantized_count = sum(1 for name in ops_df['name'] if detect_quantized_gemm(str(name)))
    
    # Count operations missing perf models
    missing_perf_model = 0
    if 'TFLOPS/s_mean' in ops_df.columns:
        import pandas as pd
        missing_perf_model = ops_df['TFLOPS/s_mean'].isna().sum()
    
    return {
        'quantized_count': int(quantized_count),
        'missing_perf_model_count': int(missing_perf_model),
        'peak_maf_tflops': metadata.get('peak_bf16_maf_tflops'),
        'peak_hbm_bw_tbs': metadata.get('peak_hbm_bw_tbs')
    }


def main():
    parser = argparse.ArgumentParser(description='Analyze GEMM operations')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    args = parser.parse_args()
    
    try:
        ops_df, metadata = load_category_data(args.output_dir, 'gemm')
    except FileNotFoundError as e:
        # Write error metrics file
        error_metrics = {
            'category': 'gemm',
            'status': 'ERROR',
            'error': str(e)
        }
        write_metrics_json(error_metrics, args.output_dir, 'gemm')
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    config = get_gemm_config()
    peak_hbm_bw = metadata.get('peak_hbm_bw_tbs', 1)
    peak_maf = metadata.get('peak_bf16_maf_tflops', 1)
    
    # Build metrics
    time_metrics = calculate_time_metrics(ops_df, metadata)
    avg_efficiency = calculate_average_efficiency(ops_df, peak_hbm_bw, peak_maf, 'auto')
    operations = build_operation_metrics(ops_df, metadata, config)
    category_specific = extract_category_specific(ops_df, metadata)
    
    metrics = {
        'category': 'gemm',
        'status': 'OK',
        **time_metrics,
        'average_efficiency_percent': avg_efficiency,
        'operations': operations,
        'category_specific': category_specific
    }
    
    output_path = write_metrics_json(metrics, args.output_dir, 'gemm')
    print(f"Metrics written to: {output_path}")


if __name__ == "__main__":
    main()
