#!/usr/bin/env python3
"""
Shared utilities for category-specific analysis scripts.

This module provides common functions for:
- Loading category data (CSV, metadata JSON)
- Calculating time metrics
- Calculating efficiency metrics
- Building operation metrics for JSON output
- Writing metrics JSON files
"""

import pandas as pd
import json
import numpy as np
import os
from typing import Dict, List, Optional, Tuple, Any


def load_category_data(output_dir: str, category: str) -> Tuple[pd.DataFrame, dict]:
    """
    Load CSV operations data and metadata JSON for a category.
    
    Args:
        output_dir: Base output directory path
        category: Category name (e.g., 'gemm', 'sdpa_fwd', 'elementwise')
        
    Returns:
        Tuple of (operations DataFrame, metadata dict)
        
    Raises:
        FileNotFoundError: If required files don't exist
    """
    csv_path = f'{output_dir}/category_data/{category}_ops.csv'
    metadata_path = f'{output_dir}/metadata/{category}_metadata.json'
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Category CSV not found: {csv_path}")
    
    ops_df = pd.read_csv(csv_path)
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return ops_df, metadata


def calculate_time_metrics(ops_df: pd.DataFrame, metadata: dict) -> dict:
    """
    Calculate total time and percentage of compute for a category.
    
    Args:
        ops_df: Operations DataFrame with 'Kernel Time (µs)_sum' column
        metadata: Metadata dict with gpu_utilization info
        
    Returns:
        Dict with total_time_ms, percent_of_compute, operation_count
    """
    if 'Kernel Time (µs)_sum' in ops_df.columns:
        total_time_us = ops_df['Kernel Time (µs)_sum'].sum()
        total_time_ms = total_time_us / 1000
        total_compute_ms = metadata.get('gpu_utilization', {}).get('total_time_ms', 1)
        percent_of_compute = (total_time_ms / total_compute_ms) * 100 if total_compute_ms > 0 else 0
    else:
        total_time_ms = 0
        percent_of_compute = 0
    
    return {
        'total_time_ms': round(total_time_ms, 3),
        'percent_of_compute': round(percent_of_compute, 2),
        'operation_count': len(ops_df)
    }


def calculate_efficiency(
    row: pd.Series,
    peak_hbm_bw: float,
    peak_maf: float,
    method: str = 'auto'
) -> Dict[str, Optional[float]]:
    """
    Calculate efficiency metrics for an operation.
    
    Args:
        row: DataFrame row with operation metrics
        peak_hbm_bw: Peak HBM bandwidth in TB/s
        peak_maf: Peak MAF in TFLOPS
        method: Efficiency calculation method:
            - 'memory_bound': Use TB/s only
            - 'compute_bound': Use TFLOPS/s only  
            - 'auto': Use FLOPS/Byte to determine (default)
            - 'prefer_compute': Try TFLOPS first, fall back to TB/s
            - 'prefer_memory': Try TB/s first, fall back to TFLOPS
            
    Returns:
        Dict with tflops_achieved, tb_s_achieved, efficiency_percent, bound_type
    """
    result = {
        'tflops_achieved': None,
        'tb_s_achieved': None,
        'efficiency_percent': None,
        'bound_type': None,
        'flops_per_byte': None
    }
    
    flops_byte = row.get('FLOPS/Byte', 0) if not pd.isna(row.get('FLOPS/Byte')) else 0
    result['flops_per_byte'] = round(flops_byte, 2) if flops_byte else None
    
    tflops_s = row.get('TFLOPS/s_mean') if not pd.isna(row.get('TFLOPS/s_mean')) else None
    tb_s = row.get('TB/s_mean') if not pd.isna(row.get('TB/s_mean')) else None
    
    if tflops_s is not None:
        result['tflops_achieved'] = round(tflops_s, 2)
    if tb_s is not None:
        result['tb_s_achieved'] = round(tb_s, 2)
    
    # Determine bound type and calculate efficiency
    if method == 'memory_bound':
        if tb_s is not None:
            result['efficiency_percent'] = round((tb_s / peak_hbm_bw) * 100, 2)
            result['bound_type'] = 'memory'
    elif method == 'compute_bound':
        if tflops_s is not None:
            result['efficiency_percent'] = round((tflops_s / peak_maf) * 100, 2)
            result['bound_type'] = 'compute'
    elif method == 'auto':
        if flops_byte > 100 and tflops_s is not None:
            result['efficiency_percent'] = round((tflops_s / peak_maf) * 100, 2)
            result['bound_type'] = 'compute'
        elif flops_byte < 50 and tb_s is not None:
            result['efficiency_percent'] = round((tb_s / peak_hbm_bw) * 100, 2)
            result['bound_type'] = 'memory'
        elif tflops_s is not None:
            result['efficiency_percent'] = round((tflops_s / peak_maf) * 100, 2)
            result['bound_type'] = 'compute'
        elif tb_s is not None:
            result['efficiency_percent'] = round((tb_s / peak_hbm_bw) * 100, 2)
            result['bound_type'] = 'memory'
    elif method == 'prefer_compute':
        if tflops_s is not None:
            result['efficiency_percent'] = round((tflops_s / peak_maf) * 100, 2)
            result['bound_type'] = 'compute'
        elif tb_s is not None:
            result['efficiency_percent'] = round((tb_s / peak_hbm_bw) * 100, 2)
            result['bound_type'] = 'memory'
    elif method == 'prefer_memory':
        if tb_s is not None:
            result['efficiency_percent'] = round((tb_s / peak_hbm_bw) * 100, 2)
            result['bound_type'] = 'memory'
        elif tflops_s is not None:
            result['efficiency_percent'] = round((tflops_s / peak_maf) * 100, 2)
            result['bound_type'] = 'compute'
    
    return result


def build_operation_metrics(
    ops_df: pd.DataFrame,
    metadata: dict,
    category_config: dict
) -> List[dict]:
    """
    Build list of operation metrics for JSON output.
    
    Args:
        ops_df: Operations DataFrame
        metadata: Metadata dict with peak performance values
        category_config: Category-specific configuration with:
            - efficiency_method: How to calculate efficiency
            - extra_fields: Additional fields to extract (optional)
            - operation_classifier: Function to classify operations (optional)
            
    Returns:
        List of operation metric dicts
    """
    peak_hbm_bw = metadata.get('peak_hbm_bw_tbs', 1)
    peak_maf = metadata.get('peak_bf16_maf_tflops', 1)
    efficiency_method = category_config.get('efficiency_method', 'auto')
    
    # Calculate total time for percentage calculations
    total_time_ms = 0
    if 'Kernel Time (µs)_sum' in ops_df.columns:
        total_time_ms = ops_df['Kernel Time (µs)_sum'].sum() / 1000
    
    operations = []
    
    # Sort by time descending
    if 'Kernel Time (µs)_sum' in ops_df.columns:
        sorted_df = ops_df.nlargest(len(ops_df), 'Kernel Time (µs)_sum')
    else:
        sorted_df = ops_df
    
    for _, row in sorted_df.iterrows():
        op_name = row.get('name', 'Unknown')
        count = int(row.get('count', 1))
        time_ms = row.get('Kernel Time (µs)_sum', 0) / 1000
        percent_of_category = (time_ms / total_time_ms * 100) if total_time_ms > 0 else 0
        
        # Calculate efficiency
        efficiency = calculate_efficiency(row, peak_hbm_bw, peak_maf, efficiency_method)
        
        op_metric = {
            'name': op_name,
            'count': count,
            'time_ms': round(time_ms, 3),
            'percent_of_category': round(percent_of_category, 2),
            'efficiency': efficiency
        }
        
        # Add extra fields if specified
        extra_fields = category_config.get('extra_fields', [])
        for field in extra_fields:
            if field in row and not pd.isna(row[field]):
                op_metric[field] = row[field]
        
        # Apply operation classifier if provided
        classifier = category_config.get('operation_classifier')
        if classifier:
            op_metric['classification'] = classifier(op_name, row)
        
        operations.append(op_metric)
    
    return operations


def calculate_average_efficiency(
    ops_df: pd.DataFrame,
    peak_hbm_bw: float,
    peak_maf: float,
    method: str = 'auto'
) -> float:
    """
    Calculate average efficiency across all operations.
    
    Args:
        ops_df: Operations DataFrame
        peak_hbm_bw: Peak HBM bandwidth in TB/s
        peak_maf: Peak MAF in TFLOPS
        method: Efficiency calculation method
        
    Returns:
        Average efficiency percentage
    """
    total_eff = 0
    count = 0
    
    for _, row in ops_df.iterrows():
        eff = calculate_efficiency(row, peak_hbm_bw, peak_maf, method)
        if eff['efficiency_percent'] is not None:
            total_eff += eff['efficiency_percent']
            count += 1
    
    return round(total_eff / count, 2) if count > 0 else 0


def write_metrics_json(metrics: dict, output_dir: str, category: str) -> str:
    """
    Write metrics JSON to category_data folder.
    
    Args:
        metrics: Metrics dict to write
        output_dir: Base output directory
        category: Category name
        
    Returns:
        Path to written file
    """
    output_path = f'{output_dir}/category_data/{category}_metrics.json'
    
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return output_path


# Category-specific helper functions

def detect_quantized_gemm(op_name: str) -> bool:
    """Check if GEMM operation is quantized."""
    quantized_markers = ['w8a8', 'int8', 'fp8', 'w4a16', 'w4a4']
    return any(marker in op_name.lower() for marker in quantized_markers)


def detect_flash_attention(op_name: str) -> bool:
    """Check if SDPA operation uses Flash Attention."""
    flash_markers = ['flash', 'fmha', 'flash_attention', 'flashattn']
    return any(marker in op_name.lower() for marker in flash_markers)


def detect_softmax(op_name: str) -> bool:
    """Check if operation is a softmax."""
    return 'softmax' in op_name.lower()


def detect_transpose(op_name: str) -> bool:
    """Check if operation is a transpose (layout overhead indicator)."""
    return 'transpose' in op_name.lower()


def classify_other_operation(op_name: str) -> str:
    """Classify 'other' category operations."""
    op_lower = op_name.lower()
    
    # Communication operations (vendor-agnostic naming)
    if any(x in op_lower for x in ['all_reduce', 'collective', 'ncclkernel', 'rccl', 'broadcast', 'allgather']):
        return 'communication'
    
    # Graph operations
    if any(x in op_lower for x in ['graph', 'hipgraph', 'cudagraph']):
        return 'graph'
    
    return 'miscellaneous'


def detect_paged_attention(op_name: str, kernel_details: str = None) -> bool:
    """
    Check if SDPA operation uses Paged Attention (vLLM style).
    
    Args:
        op_name: Operation name
        kernel_details: Optional kernel_details_summary string from CSV
        
    Returns:
        True if paged attention is detected
    """
    # Check operation name for vLLM paged attention markers
    paged_markers = ['unified_attention', 'paged_attention', 'vllm']
    if any(marker in op_name.lower() for marker in paged_markers):
        return True
    
    # Check kernel details for paged attention kernels
    if kernel_details:
        if 'kernel_paged_attention' in str(kernel_details).lower():
            return True
        if 'paged_attention_2d' in str(kernel_details).lower():
            return True
    
    return False


def parse_kernel_breakdown(kernel_details_str: str) -> dict:
    """
    Parse kernel_details_summary to extract sub-kernel timing breakdown.
    
    Args:
        kernel_details_str: String representation of kernel details list
        
    Returns:
        Dict with kernel breakdown: {kernel_name: {mean_us, percent, total_us}}
    """
    result = {
        'kernels': [],
        'total_kernel_time_us': 0,
        'has_paged_attention': False,
        'has_fwd_kernel': False,
        'has_reshape_cache': False
    }
    
    if not kernel_details_str or pd.isna(kernel_details_str):
        return result
    
    try:
        # Parse the string representation of the list
        import ast
        # Handle numpy types in string
        kernel_str = str(kernel_details_str)
        kernel_str = kernel_str.replace('np.float64(', '').replace(')', '')
        
        # Try to extract kernel info using regex for robustness
        import re
        
        # Pattern to match kernel entries
        kernel_pattern = r"'name':\s*'([^']+)'.*?'mean_duration_us':\s*([0-9.]+)"
        matches = re.findall(kernel_pattern, kernel_str, re.DOTALL)
        
        total_time = 0
        kernels = []
        
        for name, mean_us in matches:
            mean_us_float = float(mean_us)
            total_time += mean_us_float
            
            # Classify kernel type
            kernel_type = 'other'
            if 'reshape_and_cache' in name.lower():
                kernel_type = 'reshape_cache'
                result['has_reshape_cache'] = True
            elif 'paged_attention' in name.lower():
                kernel_type = 'paged_attention'
                result['has_paged_attention'] = True
            elif '_fwd_kernel' in name.lower() or 'fwd_kernel' in name.lower():
                kernel_type = 'fwd_kernel'
                result['has_fwd_kernel'] = True
            
            kernels.append({
                'name': name,
                'mean_us': mean_us_float,
                'kernel_type': kernel_type
            })
        
        # Calculate percentages
        if total_time > 0:
            for k in kernels:
                k['percent'] = round((k['mean_us'] / total_time) * 100, 2)
        
        result['kernels'] = kernels
        result['total_kernel_time_us'] = round(total_time, 2)
        
    except Exception as e:
        # If parsing fails, return empty result
        pass
    
    return result


def parse_perf_params(perf_params_str: str) -> dict:
    """
    Parse perf_params to extract attention configuration and workload profile.
    
    Args:
        perf_params_str: String representation of perf_params dict
        
    Returns:
        Dict with parsed parameters
    """
    result = {
        'batch_size': None,
        'n_q': None,
        'h_q': None,
        'n_kv': None,
        'h_kv': None,
        'd_h_qk': None,
        'd_h_v': None,
        'dropout': None,
        'causal': None,
        'flash_impl': None,
        'sum_ctx_tokens': None,
        'sum_gen_tokens': None,
        'ctx_ratio': None,
        'workload_type': 'unknown'
    }
    
    if not perf_params_str or pd.isna(perf_params_str):
        return result
    
    try:
        import ast
        params = ast.literal_eval(str(perf_params_str))
        
        # Extract basic parameters
        result['batch_size'] = params.get('B')
        result['n_q'] = params.get('N_Q')
        result['h_q'] = params.get('H_Q')
        result['n_kv'] = params.get('N_KV')
        result['h_kv'] = params.get('H_KV')
        result['d_h_qk'] = params.get('d_h_qk')
        result['d_h_v'] = params.get('d_h_v')
        result['dropout'] = params.get('dropout')
        result['causal'] = params.get('causal')
        result['flash_impl'] = params.get('flash_impl')
        
        # Extract context/generation tokens for workload profiling
        ctx_tokens = params.get('sum_ctx_tokens', 0)
        gen_tokens = params.get('sum_gen_tokens', 0)
        result['sum_ctx_tokens'] = ctx_tokens
        result['sum_gen_tokens'] = gen_tokens
        
        # Calculate context ratio and determine workload type
        total_tokens = ctx_tokens + gen_tokens
        if total_tokens > 0:
            ctx_ratio = ctx_tokens / total_tokens
            result['ctx_ratio'] = round(ctx_ratio, 3)
            
            if ctx_ratio > 0.8:
                result['workload_type'] = 'prefill_heavy'
            elif ctx_ratio < 0.2:
                result['workload_type'] = 'decode_heavy'
            else:
                result['workload_type'] = 'mixed'
        
        # Detect GQA (Grouped Query Attention)
        if result['h_q'] and result['h_kv']:
            if result['h_kv'] < result['h_q']:
                result['attention_pattern'] = 'GQA'
                result['gqa_ratio'] = result['h_q'] // result['h_kv']
            elif result['h_kv'] == result['h_q']:
                result['attention_pattern'] = 'MHA'
            else:
                result['attention_pattern'] = 'unknown'
                
    except Exception as e:
        pass
    
    return result
