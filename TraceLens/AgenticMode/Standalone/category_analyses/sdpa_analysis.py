#!/usr/bin/env python3
"""SDPA Analysis - Scaled Dot Product Attention

Computes metrics for SDPA operations and outputs JSON for subagent processing.
Supports both Flash Attention and Paged Attention (vLLM) analysis.
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
    detect_flash_attention,
    detect_paged_attention,
    parse_kernel_breakdown,
    parse_perf_params
)


def get_sdpa_config():
    """Return SDPA-specific configuration."""
    return {
        'efficiency_method': 'prefer_compute',  # SDPA can be compute or memory bound
        'extra_fields': ['Input Dims', 'has_perf_model', 'perf_params', 'kernel_details_summary'],
        'operation_classifier': classify_sdpa_operation
    }


def classify_sdpa_operation(op_name: str, row) -> dict:
    """Classify SDPA operation type with paged attention detection."""
    has_perf_model = row.get('has_perf_model', False) if 'has_perf_model' in row.index else False
    
    # Get kernel details for paged attention detection
    kernel_details = row.get('kernel_details_summary', '') if 'kernel_details_summary' in row.index else ''
    
    # Detect attention type
    is_flash = detect_flash_attention(op_name)
    is_paged = detect_paged_attention(op_name, kernel_details)
    
    # Determine attention type
    if is_paged:
        attention_type = 'paged'
    elif is_flash:
        attention_type = 'flash'
    else:
        attention_type = 'standard'
    
    # Parse kernel breakdown for paged attention
    kernel_breakdown = None
    if is_paged and kernel_details:
        kernel_breakdown = parse_kernel_breakdown(kernel_details)
    
    # Parse perf_params for workload profiling
    perf_params_str = row.get('perf_params', '') if 'perf_params' in row.index else ''
    workload_profile = None
    if perf_params_str:
        workload_profile = parse_perf_params(perf_params_str)
    
    result = {
        'is_flash_attention': is_flash,
        'is_paged_attention': is_paged,
        'has_perf_model': bool(has_perf_model),
        'attention_type': attention_type
    }
    
    # Add kernel breakdown if available
    if kernel_breakdown and kernel_breakdown.get('kernels'):
        result['kernel_breakdown'] = {
            'has_paged_attention_kernel': kernel_breakdown.get('has_paged_attention', False),
            'has_fwd_kernel': kernel_breakdown.get('has_fwd_kernel', False),
            'has_reshape_cache': kernel_breakdown.get('has_reshape_cache', False),
            'kernels': kernel_breakdown.get('kernels', [])
        }
    
    # Add workload profile if available
    if workload_profile:
        result['workload_profile'] = {
            'n_q': workload_profile.get('n_q'),
            'n_kv': workload_profile.get('n_kv'),
            'h_q': workload_profile.get('h_q'),
            'h_kv': workload_profile.get('h_kv'),
            'sum_ctx_tokens': workload_profile.get('sum_ctx_tokens'),
            'sum_gen_tokens': workload_profile.get('sum_gen_tokens'),
            'ctx_ratio': workload_profile.get('ctx_ratio'),
            'workload_type': workload_profile.get('workload_type'),
            'attention_pattern': workload_profile.get('attention_pattern'),
            'gqa_ratio': workload_profile.get('gqa_ratio')
        }
    
    return result


def extract_category_specific(ops_df, metadata) -> dict:
    """Extract SDPA-specific aggregate metrics including paged attention."""
    flash_attention_count = 0
    paged_attention_count = 0
    
    # Aggregate kernel breakdown stats
    total_reshape_cache_percent = 0
    total_fwd_kernel_percent = 0
    total_paged_attention_percent = 0
    kernel_breakdown_count = 0
    
    # Aggregate workload profile stats
    total_ctx_tokens = 0
    total_gen_tokens = 0
    
    for _, row in ops_df.iterrows():
        op_name = str(row.get('name', ''))
        kernel_details = row.get('kernel_details_summary', '') if 'kernel_details_summary' in row.index else ''
        
        if detect_flash_attention(op_name):
            flash_attention_count += 1
        
        if detect_paged_attention(op_name, kernel_details):
            paged_attention_count += 1
            
            # Parse kernel breakdown for aggregation
            breakdown = parse_kernel_breakdown(kernel_details)
            if breakdown.get('kernels'):
                kernel_breakdown_count += 1
                for k in breakdown['kernels']:
                    if k['kernel_type'] == 'reshape_cache':
                        total_reshape_cache_percent += k.get('percent', 0)
                    elif k['kernel_type'] == 'fwd_kernel':
                        total_fwd_kernel_percent += k.get('percent', 0)
                    elif k['kernel_type'] == 'paged_attention':
                        total_paged_attention_percent += k.get('percent', 0)
        
        # Parse perf_params for workload aggregation
        perf_params_str = row.get('perf_params', '') if 'perf_params' in row.index else ''
        if perf_params_str:
            profile = parse_perf_params(perf_params_str)
            total_ctx_tokens += profile.get('sum_ctx_tokens') or 0
            total_gen_tokens += profile.get('sum_gen_tokens') or 0
    
    # Count operations with perf models
    has_perf_model_count = 0
    if 'has_perf_model' in ops_df.columns:
        has_perf_model_count = int(ops_df['has_perf_model'].sum())
    
    result = {
        'flash_attention_count': int(flash_attention_count),
        'paged_attention_count': int(paged_attention_count),
        'has_perf_model_count': has_perf_model_count,
        'flash_attention_detected': flash_attention_count > 0,
        'paged_attention_detected': paged_attention_count > 0,
        'peak_maf_tflops': metadata.get('max_achievable_tflops', {}).get('matrix_bf16') if isinstance(metadata.get('max_achievable_tflops'), dict) else metadata.get('peak_bf16_maf_tflops'),
        'peak_hbm_bw_tbs': metadata.get('peak_hbm_bw_tbs')
    }
    
    # Add kernel breakdown aggregates if available
    if kernel_breakdown_count > 0:
        result['kernel_breakdown_avg'] = {
            'avg_reshape_cache_percent': round(total_reshape_cache_percent / kernel_breakdown_count, 2),
            'avg_fwd_kernel_percent': round(total_fwd_kernel_percent / kernel_breakdown_count, 2),
            'avg_paged_attention_percent': round(total_paged_attention_percent / kernel_breakdown_count, 2)
        }
    
    # Add workload profile aggregates if available
    total_tokens = total_ctx_tokens + total_gen_tokens
    if total_tokens > 0:
        ctx_ratio = total_ctx_tokens / total_tokens
        if ctx_ratio > 0.8:
            profile_type = 'prefill_heavy'
        elif ctx_ratio < 0.2:
            profile_type = 'decode_heavy'
        else:
            profile_type = 'mixed'
        
        result['workload_profile'] = {
            'total_ctx_tokens': total_ctx_tokens,
            'total_gen_tokens': total_gen_tokens,
            'ctx_ratio': round(ctx_ratio, 3),
            'profile_type': profile_type
        }
    
    return result


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
    maf = metadata.get('max_achievable_tflops', metadata.get('peak_bf16_maf_tflops', 1))
    
    time_metrics = calculate_time_metrics(ops_df, metadata)
    avg_efficiency = calculate_average_efficiency(ops_df, peak_hbm_bw, maf, 'prefer_compute')
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
