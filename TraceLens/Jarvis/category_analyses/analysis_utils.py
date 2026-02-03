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
