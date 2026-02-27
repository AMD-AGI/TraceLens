#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Data Extraction Module
Handles extraction of various data structures from GPU trace data for AI analysis

This module provides functions to extract:
- Critical path comparison data
- Timeline comparison data
- GEMM operation analysis
- Convolution operation analysis
- Unique operations per GPU
- Overall performance statistics
- Detailed per-operation comparisons
"""

from typing import Dict, List, Tuple, Optional
import pandas as pd



def extract_critical_path_comparison(baseline_data: Dict, target_data: Dict, 
                                    baseline_total: float) -> Tuple[List[Dict], List[str]]:
    """Extract comparison data from critical path nodes"""
    cp_comparison_data = []
    baseline_cp_ops = []
    
    try:
        # Use crit_ops_unique_args sheet for critical path analysis (has aggregated data)
        baseline_crit_ops = baseline_data.get('crit_ops_unique_args')
        target_crit_ops = target_data.get('crit_ops_unique_args')
        
        if baseline_crit_ops is None or baseline_crit_ops.empty:
            print("  Warning: No crit_ops_unique_args data for baseline")
            return cp_comparison_data, baseline_cp_ops
        
        if target_crit_ops is None or target_crit_ops.empty:
            print("  Warning: No crit_ops_unique_args data for target")
            return cp_comparison_data, baseline_cp_ops
        
        # crit_ops_unique_args has: name, category, operation_count, total_duration_us, avg_duration_us, etc.
        name_col = 'name'
        time_col = 'total_duration_us'
        
        if name_col not in baseline_crit_ops.columns or time_col not in baseline_crit_ops.columns:
            print(f"  Warning: Required columns not found in crit_ops_unique_args")
            return cp_comparison_data, baseline_cp_ops
        
        # Sort by total duration (already aggregated by name in the sheet)
        baseline_sorted = baseline_crit_ops.sort_values(time_col, ascending=False)
        
        # Get top 20 operations from critical path
        baseline_cp_ops = [str(n) for n in baseline_sorted[name_col].head(20)]
        
        # Build comparison for each operation
        for op_name in baseline_cp_ops:
            try:
                baseline_row = baseline_crit_ops[baseline_crit_ops[name_col] == op_name]
                if baseline_row.empty:
                    continue
                
                b_time_us = float(baseline_row[time_col].iloc[0])
                b_time = b_time_us / 1000.0  # Convert to milliseconds
                
                # Find matching operation in target
                target_row = target_crit_ops[target_crit_ops[name_col] == op_name]
                if not target_row.empty:
                    t_time_us = float(target_row[time_col].iloc[0])
                    t_time = t_time_us / 1000.0  # Convert to milliseconds
                else:
                    t_time = None
                
                gap = (b_time - t_time) if t_time is not None else None
                impact_pct = (b_time / baseline_total * 100.0) if baseline_total > 0 else 0.0
                
                cp_comparison_data.append({
                    'operation': op_name,
                    'baseline_time_ms': round(b_time, 2),
                    'target_time_ms': round(t_time, 2) if t_time is not None else None,
                    'gap_ms': round(gap, 2) if gap is not None else None,
                    'impact_pct': round(impact_pct, 2)
                })
            except Exception as e:
                print(f"  Warning: Failed to process operation {op_name}: {e}")
                continue
    
    except Exception as e:
        print(f"  Warning: Failed to extract critical path comparison: {e}")
    return cp_comparison_data, baseline_cp_ops


def extract_timeline_comparison(baseline_data: Dict, target_data: Dict,
                                baseline_total: float) -> Tuple[List[Dict], List[str]]:
    """Extract comparison data from ops_summary_by_category (timeline mode)"""
    cp_comparison_data = []
    baseline_cp_ops = []
    
    print("  Timeline mode: Building comparison from ops_summary_by_category")
    
    try:
        baseline_ops = baseline_data.get('ops_summary_by_category')
        target_ops = target_data.get('ops_summary_by_category')
        
        if baseline_ops is None or baseline_ops.empty:
            print("  Warning: ops_summary_by_category not available")
            return cp_comparison_data, baseline_cp_ops
        
        # Aggregate by category
        for _, row in baseline_ops.iterrows():
            cat_name = row.get('op category', row.get('category', row.get('Category', 'Unknown')))
            b_time = float(row.get('total_direct_kernel_time_ms', row.get('total_duration_ms', 0)))
            
            # Find matching target category
            t_time = None
            if target_ops is not None and not target_ops.empty:
                target_match = target_ops[target_ops['op category'] == cat_name]
                if not target_match.empty:
                    t_time = float(target_match.iloc[0].get('total_direct_kernel_time_ms', 
                                                            target_match.iloc[0].get('total_duration_ms', 0)))
            
            gap = (b_time - t_time) if t_time is not None else None
            impact_pct = (b_time / baseline_total * 100.0) if baseline_total > 0 else 0.0
            
            cp_comparison_data.append({
                'operation': cat_name,
                'baseline_time_ms': round(b_time, 2),
                'target_time_ms': round(t_time, 2) if t_time is not None else None,
                'gap_ms': round(gap, 2) if gap is not None else None,
                'impact_pct': round(impact_pct, 2)
            })
            baseline_cp_ops.append(cat_name)
        
        # Sort by impact
        cp_comparison_data.sort(key=lambda x: x.get('impact_pct', 0), reverse=True)
        print(f"  Timeline mode: Extracted {len(cp_comparison_data)} categories")
    
    except Exception as e:
        print(f"  Error building timeline comparison data: {e}")
        import traceback
        traceback.print_exc()
    
    return cp_comparison_data, baseline_cp_ops


# def extract_gemm_data(gpu1_data: Dict, gpu2_data: Dict, 
#                      gpu1_name: str, gpu2_name: str) -> Dict:
#     """Extract GEMM operation analysis from ops_summary_by_category"""
#     gemm_data = {}
    
#     try:
#         # Get baseline and target ops_summary_by_category
#         baseline_ops = gpu2_data.get('GEMM')
#         target_ops = gpu1_data.get('GEMM')
        
#         if baseline_ops is None or baseline_ops.empty or target_ops is None or target_ops.empty:
#             return gemm_data
        
#         # Find category column
#         cat_col = 'op category' if 'op category' in baseline_ops.columns else 'category'
#         time_col = 'total_direct_kernel_time_ms'
#         count_col = 'Count' if 'Count' in baseline_ops.columns else 'count'
        
#         # Look for GEMM category
#         baseline_gemm = baseline_ops[baseline_ops[cat_col] == 'GEMM']
#         target_gemm = target_ops[target_ops[cat_col] == 'GEMM']
        
#         if not baseline_gemm.empty and not target_gemm.empty:
#             baseline_time = float(baseline_gemm[time_col].iloc[0])
#             target_time = float(target_gemm[time_col].iloc[0])
#             baseline_count = int(baseline_gemm[count_col].iloc[0]) if count_col in baseline_gemm.columns else 0
#             target_count = int(target_gemm[count_col].iloc[0]) if count_col in target_gemm.columns else 0
            
#             gemm_data['baseline_total_ms'] = round(baseline_time, 2)
#             gemm_data['target_total_ms'] = round(target_time, 2)
#             gemm_data['gap_ms'] = round(baseline_time - target_time, 2)
#             gemm_data['baseline_count'] = baseline_count
#             gemm_data['target_count'] = target_count
#             gemm_data['gap_percentage'] = round(
#                 ((baseline_time - target_time) / baseline_time * 100) if baseline_time > 0 else 0, 2
#             )
#         elif not baseline_gemm.empty:
#             baseline_time = float(baseline_gemm[time_col].iloc[0])
#             baseline_count = int(baseline_gemm[count_col].iloc[0]) if count_col in baseline_gemm.columns else 0
#             gemm_data['baseline_total_ms'] = round(baseline_time, 2)
#             gemm_data['target_total_ms'] = 0.0
#             gemm_data['gap_ms'] = round(baseline_time, 2)
#             gemm_data['baseline_count'] = baseline_count
#             gemm_data['target_count'] = 0
#         elif not target_gemm.empty:
#             target_time = float(target_gemm[time_col].iloc[0])
#             target_count = int(target_gemm[count_col].iloc[0]) if count_col in target_gemm.columns else 0
#             gemm_data['baseline_total_ms'] = 0.0
#             gemm_data['target_total_ms'] = round(target_time, 2)
#             gemm_data['gap_ms'] = round(-target_time, 2)
#             gemm_data['baseline_count'] = 0
#             gemm_data['target_count'] = target_count
    
#     except Exception as e:
#         print(f"  Warning: Failed to extract GEMM data: {e}")
    
#     return gemm_data

def extract_shape(row, category):
    """Extract shape information from a row based on category type."""
    if category in ('CONV_fwd', 'CONV_bwd'):
        input_shape = row.get('param: input_shape', '')
        filter_shape = row.get('param: filter_shape', '')
        if input_shape and filter_shape:
            return f"in:{input_shape}, filt:{filter_shape}"
        return input_shape or filter_shape or 'N/A'
    
    elif category in ('BN_fwd', 'BN_bwd'):
        input_shape = row.get('param: input_shape', '')
        return input_shape or 'N/A'
    
    elif category == 'GEMM':
        input_dims = row.get('Input Dims_first', '')
        if input_dims:
            return input_dims
        return row.get('param: shape', 'N/A')
    
    elif category == 'flash_attention':
        input_dims = row.get('Input Dims_first', '')
        return input_dims or 'N/A'
    
    elif category == 'elementwise':
        # For UnaryElementwise
        op_shape = row.get('param: op_shape', '')
        if op_shape:
            return op_shape
        # For BinaryElementwise
        shape_in1 = row.get('param: shape_in1', '')
        shape_in2 = row.get('param: shape_in2', '')
        if shape_in1 and shape_in2:
            return f"in1:{shape_in1}, in2:{shape_in2}"
        return shape_in1 or shape_in2 or 'N/A'
    
    elif category in ('reduction', 'NCCL'):
        input_dims = row.get('Input Dims_first', '')
        return input_dims or 'N/A'
    
    return 'N/A'


def extract_data_by_category(category, gpu_data, top_ops=25):
    if category not in ('GEMM', 'CONV_fwd', 'CONV_bwd', 'BN_fwd', 'BN_bwd', 'flash_attention', 
                          'elementwise', 'reduction', 'NCCL'):
        raise ValueError(f"Unsupported category {category}")
    
    try:
        if category == 'elementwise':
            ops = gpu_data.get('UnaryElementwise')
            ops2 = gpu_data.get('BinaryElementwise')
            
            result = []
            if ops is not None and not ops.empty:
                df = ops[['name', 'Kernel Time (µs)_sum']].copy()
                df['shape'] = ops.apply(lambda row: extract_shape(row, category), axis=1)
                result.append(df)
            if ops2 is not None and not ops2.empty:
                df = ops2[['name', 'Kernel Time (µs)_sum']].copy()
                df['shape'] = ops2.apply(lambda row: extract_shape(row, category), axis=1)
                result.append(df)
            
            if not result:
                return None
            
            combined = pd.concat(result, ignore_index=True)
            return combined.nlargest(top_ops, 'Kernel Time (µs)_sum')
        else:
            ops = gpu_data.get(category)
            if ops is None or ops.empty:
                return None
            df = ops[['name', 'Kernel Time (µs)_sum']].copy()
            df['shape'] = ops.apply(lambda row: extract_shape(row, category), axis=1)
            return df.nlargest(top_ops, 'Kernel Time (µs)_sum')
    
    except Exception:
        print(f"Category {category} is not supported for automated analysis yet")
        return None
    
 


def extract_gemm_data(gpu1_data: Dict, gpu2_data: Dict, 
                     gpu1_name: str, gpu2_name: str) -> Dict:
    """
    Extract GEMM operation analysis - returns both summary stats and full dataframes.
    
    The GEMM sheet contains detailed per-operation data with columns like:
    - name, param: M/N/K, dtype, GTOPS_first, Data Moved (MB)_first
    - TB/_mean, TB/_median, OPS%_ms, BW%_ms, Time (µs), count
    
    Returns:
        dict with keys:
            - baseline_df: Full baseline GEMM dataframe
            - target_df: Full target GEMM dataframe
            - baseline_total_ms: Total baseline GEMM time
            - target_total_ms: Total target GEMM time
            - gap_ms: Time gap
            - baseline_count: Total baseline GEMM operations
            - target_count: Total target GEMM operations
            - gap_percentage: Gap as percentage of baseline
            - baseline_total_gflops: Total GFLOPS for baseline
            - target_total_gflops: Total GFLOPS for target
    """
    gemm = {}
    
    try:
        # Get baseline and target GEMM sheets (full dataframes)
        baseline_ops = gpu2_data.get('GEMM')
        target_ops = gpu1_data.get('GEMM')
        
        if baseline_ops is None or baseline_ops.empty:
            print(f"  Warning: No GEMM data for baseline ({gpu2_name})")
            return gemm
            
        if target_ops is None or target_ops.empty:
            print(f"  Warning: No GEMM data for target ({gpu1_name})")
            return gemm
        
        # Store the full GEMM dataframes
        gemm['baseline_df'] = baseline_ops.copy()
        gemm['target_df'] = target_ops.copy()
        
        # Identify time and count columns
        # Time can be in microseconds or milliseconds
        time_col = None
        for col in ['Kernel Time (µs)_mean', 'Kernel Time (us)_mean', 'Time (µs)_mean', 'Time (us)_mean', 
                    'Time (µs)', 'Time (us)', 'total_direct_kernel_time_ms', 'time_ms']:
            if col in baseline_ops.columns:
                time_col = col
                break
        
        count_col = None
        for col in ['name_count', 'count', 'Count', 'kernel_count']:
            if col in baseline_ops.columns:
                count_col = col
                break
        
        if time_col is None:
            print(f"  Warning: Could not find time column in GEMM sheet")
            print(f"  Available columns: {list(baseline_ops.columns[:10])}")
            return gemm
        
        # Determine if time is in microseconds or milliseconds
        time_is_us = 'µs' in time_col or 'us' in time_col
        
        # Calculate total time for baseline
        baseline_times = pd.to_numeric(baseline_ops[time_col], errors='coerce')
        if count_col and count_col in baseline_ops.columns:
            baseline_counts = pd.to_numeric(baseline_ops[count_col], errors='coerce')
            baseline_total_time_us = (baseline_times * baseline_counts).sum()
            baseline_total_count = baseline_counts.sum()
        else:
            baseline_total_time_us = baseline_times.sum()
            baseline_total_count = len(baseline_ops)
        
        # Calculate total time for target
        target_times = pd.to_numeric(target_ops[time_col], errors='coerce')
        if count_col and count_col in target_ops.columns:
            target_counts = pd.to_numeric(target_ops[count_col], errors='coerce')
            target_total_time_us = (target_times * target_counts).sum()
            target_total_count = target_counts.sum()
        else:
            target_total_time_us = target_times.sum()
            target_total_count = len(target_ops)
        
        # Convert to milliseconds if needed
        if time_is_us:
            baseline_total_ms = baseline_total_time_us / 1000.0
            target_total_ms = target_total_time_us / 1000.0
        else:
            baseline_total_ms = baseline_total_time_us
            target_total_ms = target_total_time_us
        
        # Calculate GFLOPS if available
        baseline_total_gflops = 0.0
        target_total_gflops = 0.0
        
        if 'GTOPS_first' in baseline_ops.columns:
            baseline_gflops = pd.to_numeric(baseline_ops['GTOPS_first'], errors='coerce')
            baseline_total_gflops = baseline_gflops.sum()
        
        if 'GTOPS_first' in target_ops.columns:
            target_gflops = pd.to_numeric(target_ops['GTOPS_first'], errors='coerce')
            target_total_gflops = target_gflops.sum()
        
        # Store summary statistics
        gemm['baseline_total_ms'] = round(baseline_total_ms, 2)
        gemm['target_total_ms'] = round(target_total_ms, 2)
        gemm['gap_ms'] = round(baseline_total_ms - target_total_ms, 2)
        gemm['baseline_count'] = int(baseline_total_count)
        gemm['target_count'] = int(target_total_count)
        gemm['gap_percentage'] = round(
            ((baseline_total_ms - target_total_ms) / baseline_total_ms * 100) if baseline_total_ms > 0 else 0, 2
        )
        
        if baseline_total_gflops > 0 or target_total_gflops > 0:
            gemm['baseline_total_gflops'] = round(baseline_total_gflops, 2)
            gemm['target_total_gflops'] = round(target_total_gflops, 2)
        
        print(f"  ✓ GEMM summary: {gpu2_name}={baseline_total_ms:.2f}ms, "
              f"{gpu1_name}={target_total_ms:.2f}ms, gap={gemm['gap_ms']:.2f}ms "
              f"({gemm['gap_percentage']:.1f}%)")
    
    except Exception as e:
        print(f"  Warning: Failed to extract GEMM data: {e}")
        import traceback
        traceback.print_exc()
    
    return gemm

# def extract_conv_data(gpu1_data: Dict, gpu2_data: Dict, 
#                      gpu1_name: str, gpu2_name: str) -> Dict:
#     """Extract convolution operation analysis from ops_summary_by_category"""
#     conv_data = {}
    
#     try:
#         # Get baseline and target ops_summary_by_category
#         baseline_ops = gpu2_data.get('ops_summary_by_category')
#         target_ops = gpu1_data.get('ops_summary_by_category')
        
#         if baseline_ops is None or baseline_ops.empty or target_ops is None or target_ops.empty:
#             return conv_data
        
#         # Find category column
#         cat_col = 'op category' if 'op category' in baseline_ops.columns else 'category'
#         time_col = 'total_direct_kernel_time_ms'
#         count_col = 'Count' if 'Count' in baseline_ops.columns else 'count'
        
#         # Look for CONV_fwd and CONV_bwd categories
#         conv_categories = ['CONV_fwd', 'CONV_bwd']
        
#         baseline_time_total = 0.0
#         target_time_total = 0.0
#         baseline_count_total = 0
#         target_count_total = 0
        
#         conv_data['by_direction'] = {}
        
#         for cat in conv_categories:
#             baseline_conv = baseline_ops[baseline_ops[cat_col] == cat]
#             target_conv = target_ops[target_ops[cat_col] == cat]
            
#             if not baseline_conv.empty:
#                 baseline_time = float(baseline_conv[time_col].iloc[0])
#                 baseline_count = int(baseline_conv[count_col].iloc[0]) if count_col in baseline_conv.columns else 0
#                 baseline_time_total += baseline_time
#                 baseline_count_total += baseline_count
#             else:
#                 baseline_time = 0.0
#                 baseline_count = 0
            
#             if not target_conv.empty:
#                 target_time = float(target_conv[time_col].iloc[0])
#                 target_count = int(target_conv[count_col].iloc[0]) if count_col in target_conv.columns else 0
#                 target_time_total += target_time
#                 target_count_total += target_count
#             else:
#                 target_time = 0.0
#                 target_count = 0
            
#             conv_data['by_direction'][cat] = {
#                 'baseline_time_ms': round(baseline_time, 2),
#                 'target_time_ms': round(target_time, 2),
#                 'gap_ms': round(baseline_time - target_time, 2),
#                 'baseline_count': baseline_count,
#                 'target_count': target_count
#             }
        
#         # Overall conv data
#         conv_data['baseline_total_ms'] = round(baseline_time_total, 2)
#         conv_data['target_total_ms'] = round(target_time_total, 2)
#         conv_data['gap_ms'] = round(baseline_time_total - target_time_total, 2)
#         conv_data['baseline_count'] = baseline_count_total
#         conv_data['target_count'] = target_count_total
#         conv_data['gap_percentage'] = round(
#             ((baseline_time_total - target_time_total) / baseline_time_total * 100) if baseline_time_total > 0 else 0, 2
#         )
    
#     except Exception as e:
#         print(f"  Warning: Failed to extract Conv data: {e}")
    
#     return conv_data

def extract_conv_data(gpu1_data: Dict, gpu2_data: Dict, 
                     gpu1_name: str, gpu2_name: str) -> Dict:
    """
    Extract convolution operation analysis from CONV_fwd and CONV_bwd sheets.
    
    The CONV sheets contain detailed per-operation data with columns like:
    - name, param: conv*, input_data_filter, dtype, length*, input_x*, weight
    - param: bias*, param: stride*, param: pad*, Transpose output, param: group
    - FLOPS, BKflowed, MB/PS/byte_fit/x_mem, TB/x_mean, TB/x_median, OPS%_ms, BW%_ms
    - Time (µs) columns, count
    
    Returns:
        dict with keys:
            - by_direction: Dict with 'CONV_fwd' and 'CONV_bwd' sub-dicts containing:
                - baseline_df, target_df: Full dataframes
                - baseline_total_ms, target_total_ms, gap_ms, counts, gap_percentage
            - baseline_total_ms: Combined total baseline conv time
            - target_total_ms: Combined total target conv time
            - gap_ms: Combined time gap
            - baseline_count: Combined operation count
            - target_count: Combined operation count
            - gap_percentage: Gap as percentage of baseline
    """
    conv_data = {}
    
    try:
        conv_categories = ['CONV_fwd', 'CONV_bwd']
        
        baseline_time_total = 0.0
        target_time_total = 0.0
        baseline_count_total = 0
        target_count_total = 0
        
        conv_data['by_direction'] = {}
        
        for cat in conv_categories:
            cat_data = {}
            
            # Get baseline and target sheets for this conv type
            baseline_ops = gpu2_data.get(cat)
            target_ops = gpu1_data.get(cat)
            
            if baseline_ops is None or baseline_ops.empty:
                print(f"  Warning: No {cat} data for baseline ({gpu2_name})")
                continue
                
            if target_ops is None or target_ops.empty:
                print(f"  Warning: No {cat} data for target ({gpu1_name})")
                continue
            
            # Store the full dataframes
            cat_data['baseline_df'] = baseline_ops.copy()
            cat_data['target_df'] = target_ops.copy()
            
            # Identify time and count columns
            time_col = None
            for col in ['Kernel Time (µs)_mean', 'Kernel Time (us)_mean', 'Time (µs)_mean', 'Time (us)_mean', 
                        'Time (µs)', 'Time (us)', 'total_direct_kernel_time_ms', 'time_ms']:
                if col in baseline_ops.columns:
                    time_col = col
                    break
            
            count_col = None
            for col in ['name_count', 'count', 'Count', 'kernel_count', 'count_UID']:
                if col in baseline_ops.columns:
                    count_col = col
                    break
            
            if time_col is None:
                print(f"  Warning: Could not find time column in {cat} sheet")
                continue
            
            # Determine if time is in microseconds or milliseconds
            time_is_us = 'µs' in time_col or 'us' in time_col
            
            # Calculate total time for baseline
            baseline_times = pd.to_numeric(baseline_ops[time_col], errors='coerce')
            if count_col and count_col in baseline_ops.columns:
                baseline_counts = pd.to_numeric(baseline_ops[count_col], errors='coerce')
                baseline_time_us = (baseline_times * baseline_counts).sum()
                baseline_count = baseline_counts.sum()
            else:
                baseline_time_us = baseline_times.sum()
                baseline_count = len(baseline_ops)
            
            # Calculate total time for target
            target_times = pd.to_numeric(target_ops[time_col], errors='coerce')
            if count_col and count_col in target_ops.columns:
                target_counts = pd.to_numeric(target_ops[count_col], errors='coerce')
                target_time_us = (target_times * target_counts).sum()
                target_count = target_counts.sum()
            else:
                target_time_us = target_times.sum()
                target_count = len(target_ops)
            
            # Convert to milliseconds if needed
            if time_is_us:
                baseline_time_ms = baseline_time_us / 1000.0
                target_time_ms = target_time_us / 1000.0
            else:
                baseline_time_ms = baseline_time_us
                target_time_ms = target_time_us
            
            # Store per-direction data
            cat_data['baseline_total_ms'] = round(baseline_time_ms, 2)
            cat_data['target_total_ms'] = round(target_time_ms, 2)
            cat_data['gap_ms'] = round(baseline_time_ms - target_time_ms, 2)
            cat_data['baseline_count'] = int(baseline_count)
            cat_data['target_count'] = int(target_count)
            cat_data['gap_percentage'] = round(
                ((baseline_time_ms - target_time_ms) / baseline_time_ms * 100) if baseline_time_ms > 0 else 0, 2
            )
            
            conv_data['by_direction'][cat] = cat_data
            
            # Add to totals
            baseline_time_total += baseline_time_ms
            target_time_total += target_time_ms
            baseline_count_total += baseline_count
            target_count_total += target_count
            
            print(f"  ✓ {cat}: {gpu2_name}={baseline_time_ms:.2f}ms, "
                  f"{gpu1_name}={target_time_ms:.2f}ms, gap={cat_data['gap_ms']:.2f}ms")
        
        # Overall conv data
        if baseline_time_total > 0 or target_time_total > 0:
            conv_data['baseline_total_ms'] = round(baseline_time_total, 2)
            conv_data['target_total_ms'] = round(target_time_total, 2)
            conv_data['gap_ms'] = round(baseline_time_total - target_time_total, 2)
            conv_data['baseline_count'] = int(baseline_count_total)
            conv_data['target_count'] = int(target_count_total)
            conv_data['gap_percentage'] = round(
                ((baseline_time_total - target_time_total) / baseline_time_total * 100) if baseline_time_total > 0 else 0, 2
            )
            
            print(f"  ✓ CONV total: {gpu2_name}={baseline_time_total:.2f}ms, "
                  f"{gpu1_name}={target_time_total:.2f}ms, gap={conv_data['gap_ms']:.2f}ms "
                  f"({conv_data['gap_percentage']:.1f}%)")
    
    except Exception as e:
        print(f"  Warning: Failed to extract Conv data: {e}")
        import traceback
        traceback.print_exc()
    
    return conv_data


def extract_ops_summary(gpu1_data: Dict, gpu2_data: Dict,
                              gpu1_name: str, gpu2_name: str) -> Dict:
    """Extract operations unique to each GPU based on ops_summary_by_category"""
    ops_summary = {
        f'{gpu1_name}_only': [],
        f'{gpu2_name}_only': []
    }
    
    try:
        baseline_ops = gpu2_data.get('ops_summary')
        target_ops = gpu1_data.get('ops_summary')
        
        if baseline_ops is None or baseline_ops.empty or target_ops is None or target_ops.empty:
            return ops_summary
        
        # Get category column name
        cat_col = None
        for col in ['name', 'total_direct_kernel_time_ms', 'Count', 'Percentage (%)']:
            if col in baseline_ops.columns:
                cat_col = col
                break
        
        if not cat_col:
            return ops_summary
        
        # Get unique categories
        baseline_categories = set(baseline_ops[cat_col].unique())
        target_categories = set(target_ops[cat_col].unique())
        
        # baseline_only = baseline_categories - target_categories
        # target_only = target_categories - baseline_categories

        baseline_only = baseline_categories
        target_only = target_categories
        
        # Get data for baseline-only categories
        if baseline_only:
            time_col = 'total_direct_kernel_time_ms'
            count_col = 'Count' if 'Count' in baseline_ops.columns else 'count'
            
            for cat in baseline_only:
                cat_data = baseline_ops[baseline_ops[cat_col] == cat]
                if not cat_data.empty:
                    total_time = float(cat_data[time_col].iloc[0]) if time_col in cat_data.columns else 0
                    count = int(cat_data[count_col].iloc[0]) if count_col in cat_data.columns else 0
                    ops_summary[f'{gpu2_name}_only'].append({
                        'name': cat,
                        'count': count,
                        'total_time_ms': round(total_time, 2)
                    })
            
            # Sort by total time descending and limit to top 10
            ops_summary[f'{gpu2_name}_only'].sort(key=lambda x: x['total_time_ms'], reverse=True)
            ops_summary[f'{gpu2_name}_only'] = ops_summary[f'{gpu2_name}_only'][:10]
        
        # Get data for target-only categories
        if target_only:
            time_col = 'total_direct_kernel_time_ms'
            count_col = 'Count' if 'Count' in target_ops.columns else 'count'
            
            for cat in target_only:
                cat_data = target_ops[target_ops[cat_col] == cat]
                if not cat_data.empty:
                    total_time = float(cat_data[time_col].iloc[0]) if time_col in cat_data.columns else 0
                    count = int(cat_data[count_col].iloc[0]) if count_col in cat_data.columns else 0
                    ops_summary[f'{gpu1_name}_only'].append({
                        'name': cat,
                        'count': count,
                        'total_time_ms': round(total_time, 2)
                    })
            
            # Sort by total time descending and limit to top 10
            ops_summary[f'{gpu1_name}_only'].sort(key=lambda x: x['total_time_ms'], reverse=True)
            ops_summary[f'{gpu1_name}_only'] = ops_summary[f'{gpu1_name}_only'][:10]
    
    except Exception as e:
        print(f"  Warning: Failed to extract unique operations: {e}")
    return ops_summary


def extract_overall_data(gpu1_data: Dict, gpu2_data: Dict) -> Dict:
    """Extract overall performance statistics"""
    overall_data = {}
    
    try:
        baseline_timeline = gpu2_data.get('gpu_timeline')
        target_timeline = gpu1_data.get('gpu_timeline')
        
        if baseline_timeline is not None:
            time_col = 'time ms' if 'time ms' in baseline_timeline.columns else 'duration_ms'
            
            # Get total time from 'total_time' row if available
            if 'type' in baseline_timeline.columns:
                total_time_row = baseline_timeline[baseline_timeline['type'] == 'total_time']
                if not total_time_row.empty:
                    overall_data['baseline_total_time_ms'] = round(float(total_time_row[time_col].values[0]), 2)
                # else:
                #     overall_data['baseline_total_time_ms'] = round(float(baseline_timeline[time_col].sum()), 2)
            else:
                overall_data['baseline_total_time_ms'] = 0.0
            #     overall_data['baseline_total_time_ms'] = round(float(baseline_timeline[time_col].sum()), 2)
            
            # overall_data['baseline_op_count'] = len(baseline_timeline)
            # overall_data['baseline_avg_op_time_ms'] = round(float(baseline_timeline[time_col].mean()), 2)
        
        if target_timeline is not None:
            time_col = 'time ms' if 'time ms' in target_timeline.columns else 'duration_ms'
            
            # Get total time from 'total_time' row if available
            if 'type' in target_timeline.columns:
                total_time_row = target_timeline[target_timeline['type'] == 'total_time']
                if not total_time_row.empty:
                    overall_data['target_total_time_ms'] = round(float(total_time_row[time_col].values[0]), 2)
                # else:
                #     overall_data['target_total_time_ms'] = round(float(target_timeline[time_col].sum()), 2)
            else:
                overall_data['target_total_time_ms'] = 0.0
            
            # overall_data['target_op_count'] = len(target_timeline)
            # overall_data['target_avg_op_time_ms'] = round(float(target_timeline[time_col].mean()), 2)
        
        # Calculate gaps
        if 'baseline_total_time_ms' in overall_data and 'target_total_time_ms' in overall_data:
            gap = overall_data['baseline_total_time_ms'] - overall_data['target_total_time_ms']
            overall_data['total_gap_ms'] = round(gap, 2)
            overall_data['gap_percentage'] = round(
                (gap / overall_data['baseline_total_time_ms'] * 100) 
                if overall_data['baseline_total_time_ms'] > 0 else 0, 2
            )
        
        # Category breakdown
        baseline_ops = gpu2_data.get('ops_summary_by_category')
        target_ops = gpu1_data.get('ops_summary_by_category')
        
        if baseline_ops is not None and not baseline_ops.empty:
            overall_data['baseline_category_count'] = len(baseline_ops)
            overall_data['baseline_categories'] = baseline_ops['op category'].tolist() if 'op category' in baseline_ops.columns else []
        
        if target_ops is not None and not target_ops.empty:
            overall_data['target_category_count'] = len(target_ops)
            overall_data['target_categories'] = target_ops['op category'].tolist() if 'op category' in target_ops.columns else []
    
    except Exception as e:
        print(f"  Warning: Failed to extract overall data: {e}")
    
    return overall_data


def extract_ops_unique_args_comparison(gpu1_data: Dict, gpu2_data: Dict) -> Optional[Dict]:
    """
    Extract detailed per-operation comparison from ops_unique_args sheet.
    
    Args:
        gpu1_data: Target GPU data dictionary
        gpu2_data: Baseline GPU data dictionary
        
    Returns:
        Dictionary with top slowdowns and speedups, or None if data unavailable
    """
    ops_comparison = {
        'top_slowdowns': [],
        'top_speedups': [],
        'total_operations': 0,
        'common_operations': 0
    }
    
    try:
        baseline_timeline = gpu2_data.get('ops_unique_args')
        target_timeline = gpu1_data.get('ops_unique_args')
        
        if baseline_timeline is None or baseline_timeline.empty:
            print("  Warning: No ops_unique_args data for baseline")
            return None
            
        if target_timeline is None or target_timeline.empty:
            print("  Warning: No ops_unique_args data for target")
            return None
        
        print(f"  Available columns in ops_unique_args: {list(baseline_timeline.columns[:10])}... (showing first 10)")
        
        # Find name column - standard column name is 'name'
        name_col = None
        for col in ['name', 'Name', 'operation', 'op_name']:
            if col in baseline_timeline.columns:
                name_col = col
                break
        
        if not name_col:
            print(f"  Warning: Could not find name column in ops_unique_args")
            return None
        
        # Find time column - may be 'col_kernel_t' or other variations
        time_col = None
        for col in ['total_direct_kernel_time_mean', 'total_direct_kernel_time_median', 
                    'col_kernel_t', 'total_time_ms', 'duration_ms', 'kernel_time_ms']:
            if col in baseline_timeline.columns:
                time_col = col
                break
        
        if not time_col:
            print(f"  Warning: Could not find time column in ops_unique_args")
            print(f"    Available columns: {list(baseline_timeline.columns)}")
            return None
        
        print(f"  Using columns: name='{name_col}', time='{time_col}'")
        
        # Find kernel_detail_summary column
        kernel_detail_col = None
        for col in ['kernel_detail_summary', 'Kernel_detail_summary', 'kernel_details', 'kernel_summary']:
            if col in baseline_timeline.columns:
                kernel_detail_col = col
                break
        
        # Find Input Dims column
        input_dims_col = None
        for col in ['Input Dims', 'input_dims', 'InputDims', 'input_dimensions']:
            if col in baseline_timeline.columns:
                input_dims_col = col
                break
        
        print(f"  Found optional columns: kernel_detail='{kernel_detail_col}', input_dims='{input_dims_col}'")
        
        # Get common operations
        baseline_ops = set(baseline_timeline[name_col].unique())
        target_ops = set(target_timeline[name_col].unique())
        common_ops = baseline_ops & target_ops
        
        ops_comparison['total_operations'] = len(baseline_ops)
        ops_comparison['common_operations'] = len(common_ops)
        
        print(f"  Comparing {len(common_ops)} common operations from ops_unique_args")
        
        comparisons = []
        for op in common_ops:
            baseline_rows = baseline_timeline[baseline_timeline[name_col] == op]
            target_rows = target_timeline[target_timeline[name_col] == op]
            
            baseline_time = float(baseline_rows[time_col].sum())
            target_time = float(target_rows[time_col].sum())
            
            if baseline_time > 0.1 or target_time > 0.1:  # Filter out negligible ops
                gap = baseline_time - target_time
                gap_pct = (gap / baseline_time * 100) if baseline_time > 0 else 0
                
                comp_entry = {
                    'operation': op,
                    'baseline_time_ms': round(baseline_time, 2),
                    'target_time_ms': round(target_time, 2),
                    'gap_ms': round(gap, 2),
                    'gap_percentage': round(gap_pct, 2)
                }
                
                # Add kernel_detail_summary if available (use first occurrence)
                if kernel_detail_col and not baseline_rows.empty:
                    kernel_detail = baseline_rows[kernel_detail_col].iloc[0]
                    if pd.notna(kernel_detail) and kernel_detail:
                        comp_entry['kernel_detail_summary'] = str(kernel_detail)
                
                # Add Input Dims if available (use first occurrence)
                if input_dims_col and not baseline_rows.empty:
                    input_dims = baseline_rows[input_dims_col].iloc[0]
                    if pd.notna(input_dims) and input_dims:
                        comp_entry['input_dims'] = str(input_dims)
                
                comparisons.append(comp_entry)
        
        # Sort and get top slowdowns and speedups
        slowdowns = [c for c in comparisons if c['gap_ms'] > 0]
        speedups = [c for c in comparisons if c['gap_ms'] < 0]
        
        ops_comparison['top_slowdowns'] = sorted(slowdowns, 
                                                 key=lambda x: x['gap_ms'], 
                                                 reverse=True)[:10]
        ops_comparison['top_speedups'] = sorted(speedups, 
                                                key=lambda x: x['gap_ms'])[:10]
        
        print(f"  ✓ Found {len(slowdowns)} slowdowns, {len(speedups)} speedups")
        
        return ops_comparison
    
    except Exception as e:
        print(f"  Warning: Failed to extract ops_unique_args comparison: {e}")
        import traceback
        traceback.print_exc()
        return None


def extract_ops_summary_by_category(gpu1_data: Dict, gpu2_data: Dict) -> Optional[Dict]:
    """
    Extract category-level summary comparison from ops_summary_by_category sheet.
    
    Args:
        gpu1_data: Target GPU data dictionary
        gpu2_data: Baseline GPU data dictionary
        
    Returns:
        Dictionary with category summary comparison, or None if data unavailable
    """
    category_comparison = {
        'categories': [],
        'total_categories': 0
    }
    
    try:
        baseline_cat = gpu2_data.get('ops_summary_by_category')
        target_cat = gpu1_data.get('ops_summary_by_category')
        
        if baseline_cat is None or baseline_cat.empty:
            print("  Warning: No ops_summary_by_category data for baseline")
            return None
            
        if target_cat is None or target_cat.empty:
            print("  Warning: No ops_summary_by_category data for target")
            return None
        
        print(f"  Available columns in ops_summary_by_category: {list(baseline_cat.columns)}")
        
        # Find category column - 'op category' is the standard name
        cat_col = None
        for col in ['op category', 'category', 'Category', 'op_category']:
            if col in baseline_cat.columns:
                cat_col = col
                break
        
        # Find time column
        time_col = None
        for col in ['total_direct_kernel_time_ms', 'ct_kernel', 'total_time_ms', 'duration_ms', 'time_ms']:
            if col in baseline_cat.columns:
                time_col = col
                break
        
        # Find count column
        count_col = None
        for col in ['Count', 'count', 'kernel_count', 'operation_count']:
            if col in baseline_cat.columns:
                count_col = col
                break
        
        # Find percentage column (optional)
        pct_col = None
        for col in ['Percentage (%)', 'rcentage', 'percentage', 'pct', 'Percentage']:
            if col in baseline_cat.columns:
                pct_col = col
                break
        
        if not cat_col or not time_col:
            print(f"  Warning: Required columns not found in ops_summary_by_category")
            print(f"    cat_col={cat_col}, time_col={time_col}")
            return None
        
        print(f"  Using columns: category='{cat_col}', time='{time_col}', count='{count_col}', pct='{pct_col}'")
        print(f"  Extracting category summary comparison")
        
        categories_list = []
        for _, row in baseline_cat.iterrows():
            cat_name = row.get(cat_col)
            baseline_time = float(row.get(time_col, 0))
            
            # Find matching category in target
            target_match = target_cat[target_cat[cat_col] == cat_name]
            if not target_match.empty:
                target_time = float(target_match.iloc[0].get(time_col, 0))
                gap = baseline_time - target_time
                
                category_info = {
                    'category': cat_name,
                    'baseline_time_ms': round(baseline_time, 2),
                    'target_time_ms': round(target_time, 2),
                    'gap_ms': round(gap, 2),
                    'gap_percentage': round((gap / baseline_time * 100) if baseline_time > 0 else 0, 2)
                }
                
                # Add count if available
                if count_col:
                    baseline_count = row.get(count_col)
                    target_count = target_match.iloc[0].get(count_col)
                    if pd.notna(baseline_count):
                        category_info['baseline_count'] = int(baseline_count)
                    if pd.notna(target_count):
                        category_info['target_count'] = int(target_count)
                
                # Add percentage of total if available
                if pct_col:
                    baseline_pct = row.get(pct_col)
                    target_pct = target_match.iloc[0].get(pct_col)
                    if pd.notna(baseline_pct):
                        category_info['baseline_pct'] = round(float(baseline_pct), 2)
                    if pd.notna(target_pct):
                        category_info['target_pct'] = round(float(target_pct), 2)
                
                categories_list.append(category_info)
        
        # Sort by absolute gap (descending)
        category_comparison['categories'] = sorted(categories_list, 
                                                   key=lambda x: abs(x['gap_ms']), 
                                                   reverse=True)
        category_comparison['total_categories'] = len(categories_list)
        
        print(f"  ✓ Extracted {len(categories_list)} category comparisons")
        
        return category_comparison
    
    except Exception as e:
        print(f"  Warning: Failed to extract ops_summary_by_category: {e}")
        import traceback
        traceback.print_exc()
        return None


def extract_detailed_comparison(gpu1_data: Dict, gpu2_data: Dict) -> Optional[Dict]:
    """
    DEPRECATED: This function is split into extract_ops_unique_args_comparison 
    and extract_ops_summary_by_category for better modularity.
    
    Kept for backward compatibility - combines both extractions.
    """
    detailed_comparison = {
        'top_slowdowns': [],
        'top_speedups': [],
        'category_summary': []
    }
    
    # Extract ops_unique_args comparison
    ops_comp = extract_ops_unique_args_comparison(gpu1_data, gpu2_data)
    if ops_comp:
        detailed_comparison['top_slowdowns'] = ops_comp['top_slowdowns']
        detailed_comparison['top_speedups'] = ops_comp['top_speedups']
    
    # Extract category summary
    cat_comp = extract_ops_summary_by_category(gpu1_data, gpu2_data)
    if cat_comp:
        detailed_comparison['category_summary'] = cat_comp['categories']
    
    return detailed_comparison if (ops_comp or cat_comp) else None


def extract_critical_path_categories(gpu_data: Dict) -> Dict[str, float]:
    """
    Extract category breakdown from critical path data.
    Uses crit_ops_unique_args sheet which has category and duration information.
    
    Args:
        gpu_data: Dictionary containing GPU trace data sheets
        
    Returns:
        Dict mapping category names to total duration in milliseconds
    """
    categories = {}
    
    try:
        # Try crit_ops_unique_args first (has category + duration info)
        crit_ops = gpu_data.get('crit_ops_unique_args')
        
        if crit_ops is not None and not crit_ops.empty:
            # Check for required columns
            if 'category' in crit_ops.columns and 'total_duration_us' in crit_ops.columns:
                # Aggregate by category
                for _, row in crit_ops.iterrows():
                    cat = row.get('category', 'Other')
                    duration_us = float(row.get('total_duration_us', 0))
                    duration_ms = duration_us / 1000.0  # Convert to milliseconds
                    
                    categories[cat] = categories.get(cat, 0) + duration_ms
                
                print(f"  Extracted {len(categories)} categories from crit_ops_unique_args")
                return categories
        
        # Fallback to critical_path_nodes if crit_ops_unique_args not available
        cp_nodes = gpu_data.get('critical_path_nodes')
        if cp_nodes is not None and not cp_nodes.empty:
            # Find time column
            time_col = None
            for col in ['duration_us', 'duration_ms', 'time ms', 'duration']:
                if col in cp_nodes.columns:
                    time_col = col
                    break
            
            if time_col and 'category' in cp_nodes.columns:
                for _, row in cp_nodes.iterrows():
                    cat = row.get('category', 'Other')
                    duration = float(row.get(time_col, 0))
                    
                    # Convert to milliseconds if needed
                    if time_col == 'duration_us':
                        duration = duration / 1000.0
                    
                    categories[cat] = categories.get(cat, 0) + duration
                
                print(f"  Extracted {len(categories)} categories from critical_path_nodes")
                return categories
        
        print("  Warning: No critical path category data available")
        
    except Exception as e:
        print(f"  Error extracting critical path categories: {e}")
    
    return categories


def extract_timeline_categories_comparison(baseline_data: Dict, target_data: Dict,
                                           baseline_total: float, target_total: float) -> List[Dict]:
    """
    Extract and compare timeline categories (computation_time, exposed_comm_time, etc.)
    from gpu_timeline sheet.
    
    Returns:
        List of dicts with category comparisons
    """
    gpu_timeline = []
    
    try:
        baseline_timeline = baseline_data.get('gpu_timeline')
        target_timeline = target_data.get('gpu_timeline')
        
        if baseline_timeline is None or baseline_timeline.empty:
            print("  Warning: No gpu_timeline data for baseline")
            return gpu_timeline
        
        if target_timeline is None or target_timeline.empty:
            print("  Warning: No gpu_timeline data for target")
            return gpu_timeline
        
        # Define timeline categories to extract (based on your attachment)
        categories_to_extract = [
            'computation_time',
            'exposed_comm_time',
            'exposed_memcpy_time',
            'busy_time',
            'idle_time',
            'total_time'
        ]
        
        time_col = 'time ms' if 'time ms' in baseline_timeline.columns else 'duration_ms'
        type_col = 'type'
        
        if type_col not in baseline_timeline.columns:
            print("  Warning: 'type' column not found in gpu_timeline")
            return gpu_timeline
        
        # Extract each category
        for category in categories_to_extract:
            baseline_row = baseline_timeline[baseline_timeline[type_col] == category]
            target_row = target_timeline[target_timeline[type_col] == category]
            
            if not baseline_row.empty:
                baseline_time = float(baseline_row[time_col].iloc[0])
                target_time = float(target_row[time_col].iloc[0]) if not target_row.empty else None
                
                gap = (baseline_time - target_time) if target_time is not None else None
                baseline_pct = (baseline_time / baseline_total * 100.0) if baseline_total > 0 else 0.0
                target_pct = (target_time / target_total * 100.0) if target_time is not None and target_total > 0 else None
                
                # Calculate pct_gap from unrounded percentages for accuracy
                pct_gap = round(baseline_pct - target_pct, 2) if target_pct is not None else None
                
                gpu_timeline.append({
                    'category': category,
                    'baseline_time_ms': round(baseline_time, 2),
                    'target_time_ms': round(target_time, 2) if target_time is not None else None,
                    'gap_ms': round(gap, 2) if gap is not None else None,
                    'baseline_pct': round(baseline_pct, 2),
                    'target_pct': round(target_pct, 2) if target_pct is not None else None,
                    'pct_gap': pct_gap
                })
        
        print(f"  Extracted {len(gpu_timeline)} timeline categories from gpu_timeline")
        
    except Exception as e:
        print(f"  Error extracting timeline categories: {e}")
        import traceback
        traceback.print_exc()
    
    return gpu_timeline


def extract_detailed_operations_for_category(baseline_data: Dict, target_data: Dict, 
                                             category: str, top_n: int = 10) -> List[Dict]:
    """
    Extract detailed operation-level data for a specific category from ops_unique_args sheet.
    Returns top N operations sorted by baseline time.
    
    Args:
        baseline_data: Baseline GPU data dictionary
        target_data: Target GPU data dictionary
        category: Category name (e.g., 'GEMM', 'CONV_fwd', 'BN_bwd')
        top_n: Number of top operations to return (default 10)
    
    Returns:
        List of dictionaries with operation details
    """
    operations = []
    
    try:
        # Get ops_unique_args sheet (has all operations with detailed info)
        baseline_ops = baseline_data.get('ops_unique_args')
        target_ops = target_data.get('ops_unique_args')
        
        if baseline_ops is None or baseline_ops.empty:
            print(f"  Warning: No ops_unique_args data for baseline")
            return operations
        
        if target_ops is None or target_ops.empty:
            print(f"  Warning: No ops_unique_args data for target")
            return operations
        
        # Check required columns
        required_cols = ['name', 'category']
        time_col = None
        for col in ['total_duration_us', 'duration_us', 'time_ms']:
            if col in baseline_ops.columns:
                time_col = col
                break
        
        if not all(col in baseline_ops.columns for col in required_cols) or time_col is None:
            print(f"  Warning: Required columns not found in ops_unique_args")
            return operations
        
        # Filter by category
        baseline_cat_ops = baseline_ops[baseline_ops['category'] == category].copy()
        
        if baseline_cat_ops.empty:
            print(f"  No operations found for category '{category}' in baseline")
            return operations
        
        # Sort by time (descending) and get top N
        baseline_cat_ops = baseline_cat_ops.sort_values(time_col, ascending=False).head(top_n)
        
        print(f"  Extracting top {top_n} operations for category '{category}'")
        
        # Build detailed comparison for each operation
        for idx, row in baseline_cat_ops.iterrows():
            op_name = row['name']
            baseline_time_val = float(row[time_col])
            
            # Convert to milliseconds if needed
            if 'us' in time_col:
                baseline_time = baseline_time_val / 1000.0
            else:
                baseline_time = baseline_time_val
            
            # Find matching operation in target
            target_row = target_ops[target_ops['name'] == op_name]
            if not target_row.empty:
                target_time_val = float(target_row[time_col].iloc[0])
                if 'us' in time_col:
                    target_time = target_time_val / 1000.0
                else:
                    target_time = target_time_val
                
                gap = baseline_time - target_time
                speedup = target_time / baseline_time if baseline_time > 0 else 1.0
            else:
                target_time = None
                gap = None
                speedup = None
            
            # Extract additional info if available
            op_details = {
                'name': op_name,
                'category': category,
                'baseline_time_ms': round(baseline_time, 2),
                'target_time_ms': round(target_time, 2) if target_time is not None else None,
                'gap_ms': round(gap, 2) if gap is not None else None,
                'speedup': round(speedup, 2) if speedup is not None else None
            }
            
            # Add operation count if available
            if 'operation_count' in row:
                op_details['operation_count'] = int(row['operation_count'])
            
            # Add average duration if available
            if 'avg_duration_us' in row:
                op_details['avg_duration_ms'] = round(float(row['avg_duration_us']) / 1000.0, 2)
            
            # Add arguments/shape info if available
            for col in ['args', 'arguments', 'shape', 'input_shape']:
                if col in row and pd.notna(row[col]):
                    op_details[col] = str(row[col])
            
            operations.append(op_details)
        
        print(f"  Extracted {len(operations)} operations for category '{category}'")
        
    except Exception as e:
        print(f"  Error extracting detailed operations for category '{category}': {e}")
        import traceback
        traceback.print_exc()
    
    return operations


def extract_prefill_decode_comparison(baseline_data: Dict, target_data: Dict) -> Optional[Dict]:
    """
    Extract prefill and decode phase comparison if available.
    Returns detailed summary of prefill vs decode performance with category breakdowns.
    
    Args:
        baseline_data: Baseline GPU data dictionary
        target_data: Target GPU data dictionary
    
    Returns:
        Dictionary with prefill/decode comparison or None if not available
    """
    try:
        # Check if prefill/decode sheets exist
        baseline_prefill = baseline_data.get('ops_summary_prefill')
        baseline_decode = baseline_data.get('ops_summary_decode')
        target_prefill = target_data.get('ops_summary_prefill')
        target_decode = target_data.get('ops_summary_decode')
        
        print(f"\n  Checking for prefill/decode sheets:")
        print(f"    baseline_prefill exists: {baseline_prefill is not None}")
        print(f"    baseline_decode exists: {baseline_decode is not None}")
        print(f"    target_prefill exists: {target_prefill is not None}")
        print(f"    target_decode exists: {target_decode is not None}")
        
        # If sheets don't exist, return None
        if baseline_prefill is None or baseline_decode is None:
            print("  Prefill/decode sheets not found - skipping phase analysis")
            return None
        
        print("  Extracting prefill/decode phase comparison...")
        
        # Show available columns for debugging
        if not baseline_prefill.empty:
            print(f"  Available columns in ops_summary_prefill: {list(baseline_prefill.columns)}")
        
        result = {
            'baseline_prefill': {},
            'baseline_decode': {},
            'target_prefill': {},
            'target_decode': {},
            'phase_comparison': {}
        }
        
        # Define column mappings based on actual sheet structure
        cat_col_names = ['Kernel categories', 'kernel_categories', 'category', 'op category']
        name_col_names = ['name', 'Name', 'operation']
        time_col_names = ['total_direct_kernel_time_sum', 'total_direct_kernel_time_ms', 
                         'time ms', 'duration_ms', 'total_time_ms']
        count_col_names = ['Count', 'count', 'operation_count']
        pct_col_names = ['rcentage', 'Percentage (%)', 'percentage', 'pct']
        
        # Helper function to extract phase data
        def extract_phase_data(df, phase_name):
            if df is None or df.empty:
                return {}
            
            # Find columns
            cat_col = next((col for col in cat_col_names if col in df.columns), None)
            name_col = next((col for col in name_col_names if col in df.columns), None)
            time_col = next((col for col in time_col_names if col in df.columns), None)
            count_col = next((col for col in count_col_names if col in df.columns), None)
            pct_col = next((col for col in pct_col_names if col in df.columns), None)
            
            if not time_col:
                print(f"  Warning: Could not find time column in {phase_name}")
                return {}
            
            print(f"  Using columns for {phase_name}: category='{cat_col}', name='{name_col}', time='{time_col}', count='{count_col}', pct='{pct_col}'")
            
            phase_data = {
                'total_time_ms': 0.0,
                'categories': []
            }
            
            # Calculate total time
            total_time = df[time_col].sum()
            phase_data['total_time_ms'] = round(total_time, 2)
            
            # Extract category breakdown
            categories_list = []
            for _, row in df.iterrows():
                cat_info = {}
                
                # Get category or name
                if cat_col:
                    cat_info['category'] = str(row[cat_col])
                elif name_col:
                    cat_info['name'] = str(row[name_col])
                else:
                    cat_info['category'] = 'Unknown'
                
                # Get time
                cat_info['time_ms'] = round(float(row[time_col]), 2)
                
                # Get count if available
                if count_col and pd.notna(row.get(count_col)):
                    cat_info['count'] = int(row[count_col])
                
                # Get percentage if available
                if pct_col and pd.notna(row.get(pct_col)):
                    cat_info['percentage'] = round(float(row[pct_col]), 2)
                
                categories_list.append(cat_info)
            
            # Sort by time descending
            phase_data['categories'] = sorted(categories_list, 
                                            key=lambda x: x['time_ms'], 
                                            reverse=True)
            
            return phase_data
        
        # Extract all four phase data
        result['baseline_prefill'] = extract_phase_data(baseline_prefill, 'baseline_prefill')
        result['baseline_decode'] = extract_phase_data(baseline_decode, 'baseline_decode')
        result['target_prefill'] = extract_phase_data(target_prefill, 'target_prefill')
        result['target_decode'] = extract_phase_data(target_decode, 'target_decode')
        
        # Calculate phase comparison metrics
        baseline_prefill_time = result['baseline_prefill'].get('total_time_ms', 0)
        baseline_decode_time = result['baseline_decode'].get('total_time_ms', 0)
        target_prefill_time = result['target_prefill'].get('total_time_ms', 0)
        target_decode_time = result['target_decode'].get('total_time_ms', 0)
        
        if baseline_prefill_time > 0 or baseline_decode_time > 0:
            result['phase_comparison'] = {
                'prefill_gap_ms': round(baseline_prefill_time - target_prefill_time, 2),
                'decode_gap_ms': round(baseline_decode_time - target_decode_time, 2),
                'prefill_gap_pct': round(((baseline_prefill_time - target_prefill_time) / baseline_prefill_time * 100) if baseline_prefill_time > 0 else 0, 2),
                'decode_gap_pct': round(((baseline_decode_time - target_decode_time) / baseline_decode_time * 100) if baseline_decode_time > 0 else 0, 2)
            }
        
        print(f"  ✓ Prefill/Decode data extracted:")
        print(f"    Baseline prefill: {baseline_prefill_time:.2f}ms ({len(result['baseline_prefill'].get('categories', []))} categories)")
        print(f"    Baseline decode: {baseline_decode_time:.2f}ms ({len(result['baseline_decode'].get('categories', []))} categories)")
        print(f"    Target prefill: {target_prefill_time:.2f}ms")
        print(f"    Target decode: {target_decode_time:.2f}ms")
        
        return result
        
    except Exception as e:
        print(f"  Error extracting prefill/decode comparison: {e}")
        import traceback
        traceback.print_exc()
        return None
