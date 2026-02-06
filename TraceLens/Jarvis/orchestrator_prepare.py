#!/usr/bin/env python3
"""
TraceLens Jarvis - Orchestrator Preparation Script
Steps 2-5: GPU Utilization, Top Ops, Tree Data Pre-computation, Category Filtering

TO DO: Prune out unnecessary segments
"""

import pandas as pd
import json
import os
import argparse
from pathlib import Path
import sys

# Add parent directory to path to import utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.platform_specs import PLATFORM_SPECS, CATEGORY_SKILL_MAP


def main():
    parser = argparse.ArgumentParser(description='Prepare category data for TraceLens analysis')
    parser.add_argument('--trace-path', required=True, help='Path to trace file')
    parser.add_argument('--platform', required=True, choices=list(PLATFORM_SPECS.keys()),
                        help='AMD platform (MI300X, MI325X, MI355X, MI400)')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--enable_pseudo_ops', action='store_true', 
                        help='Enable pseudo ops in TreePerfAnalyzer')
    
    args = parser.parse_args()
    
    trace_path = args.trace_path
    platform = args.platform
    output_dir = args.output_dir
    enable_pseudo_ops = args.enable_pseudo_ops
    csv_dir = f"{output_dir}/perf_report_csvs"
    
    print("="*80)
    print("TRACELENS JARVIS - ORCHESTRATOR PREPARATION")
    print("="*80)
    print(f"Platform: {platform}")
    print(f"Trace: {trace_path}")
    print(f"Output: {output_dir}")
    print(f"Pseudo Ops: {'Enabled' if enable_pseudo_ops else 'Disabled'}")
    print("="*80)
    
    # Create directory structure (chmod 777 so host user can write when running in container as root)
    for d in [output_dir, f"{output_dir}/metadata", f"{output_dir}/category_data", f"{output_dir}/category_findings"]:
        os.makedirs(d, exist_ok=True)
        os.chmod(d, 0o777)
    
    platform_specs = PLATFORM_SPECS[platform]
    
    # ============================================================================
    # STEP 2: Assess GPU Utilization
    # ============================================================================
    print("\n[STEP 2] Assessing GPU Utilization...")
    
    gpu_timeline = pd.read_csv(f'{csv_dir}/gpu_timeline.csv')
    
    # The CSV has columns: type, time ms, percent
    # Convert to dict for easy lookup
    gpu_data = {}
    for _, row in gpu_timeline.iterrows():
        gpu_data[row['type']] = {'time_ms': row['time ms'], 'percent': row['percent']}
    
    gpu_utilization_metrics = {
        "total_time_ms": gpu_data.get('total_time', {}).get('time_ms', 0),
        "computation_time_percent": gpu_data.get('computation_time', {}).get('percent', 0),
        "exposed_comm_time_percent": gpu_data.get('exposed_comm_time', {}).get('percent', 0),
        "exposed_memcpy_time_percent": gpu_data.get('exposed_memcpy_time', {}).get('percent', 0),
        "idle_time_percent": gpu_data.get('idle_time', {}).get('percent', 0)
    }
    
    print(f"\nGPU Utilization Metrics:")
    print(f"  Total Time: {gpu_utilization_metrics['total_time_ms']:.2f} ms")
    print(f"  Computation: {gpu_utilization_metrics['computation_time_percent']:.2f}%")
    print(f"  Communication: {gpu_utilization_metrics['exposed_comm_time_percent']:.4f}%")
    print(f"  MemCpy: {gpu_utilization_metrics['exposed_memcpy_time_percent']:.2f}%")
    print(f"  Idle: {gpu_utilization_metrics['idle_time_percent']:.2f}%")
    
    if gpu_utilization_metrics['computation_time_percent'] < 95:
        print(f"  ⚠️  WARNING: Compute utilization < 95%")
    
    # Check for critical idle time - flag for CPU/idle analysis
    cpu_idle_critical = gpu_utilization_metrics['idle_time_percent'] > 50
    if cpu_idle_critical:
        print(f"  🔴 CRITICAL: Idle time > 50% - CPU/idle analysis required")
    
    # ============================================================================
    # STEP 3: Identify Top Operations
    # ============================================================================
    print("\n[STEP 3] Identifying Top Operations...")
    
    ops_summary = pd.read_csv(f'{csv_dir}/ops_summary.csv')
    
    # Sort by total_direct_kernel_time_ms if available, else by time column
    if 'total_direct_kernel_time_ms' in ops_summary.columns:
        ops_summary_sorted = ops_summary.sort_values('total_direct_kernel_time_ms', ascending=False)
        time_col = 'total_direct_kernel_time_ms'
    elif 'Kernel Time (µs)_sum' in ops_summary.columns:
        ops_summary_sorted = ops_summary.sort_values('Kernel Time (µs)_sum', ascending=False)
        time_col = 'Kernel Time (µs)_sum'
    else:
        print(f"  ⚠️  Could not determine time column")
        print(f"  Available columns: {ops_summary.columns.tolist()}")
        ops_summary_sorted = ops_summary
        time_col = None
    
    print(f"\nTop 10 Operations by GPU Time:")
    print("="*80)
    if time_col:
        for idx, row in ops_summary_sorted.head(10).iterrows():
            op_name = row.get('name', 'Unknown')
            time_val = row[time_col]
            category = row.get('op category', 'N/A')
            print(f"  {op_name:50s} | {time_val:10.2f} | {category}")
    
    # ============================================================================
    # STEP 4: Pre-compute Tree Data (Optimization)
    # ============================================================================
    print("\n[STEP 4] Pre-computing Tree Data for Bottleneck Operations...")
    
    try:
        from TraceLens.TreePerf import TreePerfAnalyzer
        
        print(f"  Loading trace: {trace_path}")
        print(f"  Pseudo ops: {'enabled' if enable_pseudo_ops else 'disabled'}")
        analyzer = TreePerfAnalyzer.from_file(trace_path, add_python_func=True, enable_pseudo_ops=enable_pseudo_ops)
        tree = analyzer.tree
        print(f"  ✓ Trace loaded successfully")
        print(f"  ✓ Tree has {len(tree.events)} events")
        
        # Read unified performance summary
        unified_df = pd.read_csv(f'{csv_dir}/unified_perf_summary.csv')
        
        # Get unique categories
        categories = unified_df['op category'].unique()
        print(f"\n  Found {len(categories)} categories")
        
        # For each category, pre-compute tree data for bottlenecks
        for category in categories:
            if pd.isna(category) or category == '':
                category_name = 'other'
                display_name = 'Other'
            else:
                category_name = category.replace(' ', '_').replace('/', '_').lower()
                display_name = category
            
            # Filter operations for this category
            if pd.isna(category) or category == '':
                category_df = unified_df[unified_df['op category'].isna() | (unified_df['op category'] == '')]
            else:
                category_df = unified_df[unified_df['op category'] == category]
            
            if len(category_df) == 0:
                continue
            
            # Identify bottlenecks: ops with >10% of category time
            if 'Kernel Time (µs)_sum' in category_df.columns:
                category_total_time = category_df['Kernel Time (µs)_sum'].sum()
                category_df = category_df.copy()
                category_df['category_percent'] = (category_df['Kernel Time (µs)_sum'] / category_total_time) * 100 if category_total_time > 0 else 0
                
                bottleneck_ops = category_df[category_df['category_percent'] > 10]
                
                # If no ops > 10%, take top 5 by time
                if len(bottleneck_ops) == 0:
                    bottleneck_ops = category_df.nlargest(min(5, len(category_df)), 'Kernel Time (µs)_sum')
                
                # Simplified tree data (no full tree traversal to avoid complexity)
                tree_data = {}
                for idx, row in bottleneck_ops.iterrows():
                    target_uid = row.get('ex_UID', row.get('UID', None))
                    if pd.isna(target_uid):
                        continue
                    
                    tree_data[str(int(target_uid))] = {
                        'op_name': row.get('name', 'Unknown'),
                        'ex_uid': int(target_uid),
                        'input_dims': str(row.get('Input Dims', '')),
                        'parent_chain': [],  # Simplified
                        'subtree': [],
                        'fusion_opportunity': False,
                        'notes': 'Tree traversal simplified - using CSV data'
                    }
                
                # Save tree data
                tree_data_file = f"{output_dir}/category_data/{category_name}_tree_data.json"
                with open(tree_data_file, 'w') as f:
                    json.dump(tree_data, f, indent=2)
        
        print(f"  ✓ Pre-computed tree data for bottleneck operations")
        
    except ImportError as e:
        print(f"  ⚠️  Could not import TraceLens: {e}")
        print(f"  Skipping tree data pre-computation")
    except Exception as e:
        print(f"  ⚠️  Error during tree data pre-computation: {e}")
        import traceback
        traceback.print_exc()
    
    # ============================================================================
    # STEP 5: Filter and Export Category Data
    # ============================================================================
    print("\n[STEP 5] Filtering and Exporting Category Data...")
    
    unified_df = pd.read_csv(f'{csv_dir}/unified_perf_summary.csv')
    
    # Create enhanced categories with special detection for MoE, BatchNorm, Convolution
    def get_enhanced_category(row):
        """Determine category with special handling for MoE, BatchNorm, Convolution"""
        op_name = row.get('name', '')
        category = row.get('op category', '')
        
        # Check for special categories by operation name
        if 'moe' in op_name.lower() or 'fused_moe' in op_name.lower():
            return 'moe_fused', 'MoE Fused'
        elif 'batch_norm' in op_name.lower() or 'batchnorm' in op_name.lower():
            return 'batchnorm', 'BatchNorm'
        elif 'conv' in op_name.lower() and 'aten::' in op_name:  # Convolution operations
            return 'convolution', 'Convolution'
        
        # Use existing category
        if pd.isna(category) or category == '':
            return 'other', 'Other'
        else:
            category_name = category.replace(' ', '_').replace('/', '_').lower()
            display_name = category
            return category_name, display_name
    
    # Apply enhanced categorization
    unified_df['enhanced_category'], unified_df['display_name'] = zip(*unified_df.apply(get_enhanced_category, axis=1))
    
    categories = unified_df['enhanced_category'].unique()
    exported_categories = []
    
    for category_name in categories:
        category_df = unified_df[unified_df['enhanced_category'] == category_name]
        display_name = category_df.iloc[0]['display_name']
        
        print(f"\n  Category: {display_name} ({category_name})")
        
        if len(category_df) == 0:
            print(f"    No operations - skipping")
            continue
        
        # Export filtered CSV
        csv_file = f"{output_dir}/category_data/{category_name}_ops.csv"
        category_df.to_csv(csv_file, index=False)
        print(f"    ✓ Exported CSV: {len(category_df)} ops")
        
        # Create metadata JSON
        metadata = {
            "platform": platform,
            "peak_hbm_bw_tbs": platform_specs["peak_hbm_bw_tbs"],
            "peak_bf16_maf_tflops": platform_specs["peak_bf16_maf_tflops"],
            "memory_gb": platform_specs["memory_gb"],
            "trace_path": trace_path,
            "output_dir": output_dir,
            "category": display_name,
            "category_name": category_name,
            "gpu_utilization": gpu_utilization_metrics,
            "trace_loading_policy": "DO_NOT_LOAD_TRACE_use_precomputed_tree_data"
        }
        
        metadata_file = f"{output_dir}/metadata/{category_name}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"    ✓ Exported metadata")
        
        exported_categories.append({
            "name": category_name,
            "display_name": display_name,
            "skill": CATEGORY_SKILL_MAP.get(category_name, "generic-op-analysis"),
            "ops_count": len(category_df),
            "csv_file": csv_file,
            "metadata_file": metadata_file,
            "tree_data_file": f"{output_dir}/category_data/{category_name}_tree_data.json"
        })
    
    # ============================================================================
    # CPU/Idle Category Creation (when idle > 50%)
    # ============================================================================
    if cpu_idle_critical:
        print(f"\n  Category: CPU/Idle Analysis (cpu_idle)")
        print(f"    🔴 Creating CPU/idle category due to {gpu_utilization_metrics['idle_time_percent']:.1f}% idle time")
        
        # Create CPU idle metadata
        cpu_idle_metadata = {
            "platform": platform,
            "peak_hbm_bw_tbs": platform_specs["peak_hbm_bw_tbs"],
            "peak_bf16_maf_tflops": platform_specs["peak_bf16_maf_tflops"],
            "memory_gb": platform_specs["memory_gb"],
            "trace_path": trace_path,
            "output_dir": output_dir,
            "category": "CPU/Idle Analysis",
            "category_name": "cpu_idle",
            "gpu_utilization": gpu_utilization_metrics,
            "idle_critical": True,
            "severity": "CRITICAL" if gpu_utilization_metrics['idle_time_percent'] > 70 else "HIGH"
        }
        
        cpu_idle_metadata_file = f"{output_dir}/metadata/cpu_idle_metadata.json"
        with open(cpu_idle_metadata_file, 'w') as f:
            json.dump(cpu_idle_metadata, f, indent=2)
        
        # Create empty ops CSV (cpu_idle doesn't use ops, it uses gpu_timeline)
        cpu_idle_csv = f"{output_dir}/category_data/cpu_idle_ops.csv"
        pd.DataFrame().to_csv(cpu_idle_csv, index=False)
        
        print(f"    ✓ Exported metadata")
        
        # Insert at beginning of categories (highest priority)
        exported_categories.insert(0, {
            "name": "cpu_idle",
            "display_name": "CPU/Idle Analysis",
            "skill": "cpu-idle-analysis",
            "ops_count": 0,
            "csv_file": cpu_idle_csv,
            "metadata_file": cpu_idle_metadata_file,
            "tree_data_file": None,
            "priority": 0,
            "critical": True
        })
    
    # ============================================================================
    # STEP 5.5: Calculate Time Metric Breakdown per Category
    # ============================================================================
    print("\n[STEP 5.5] Calculating Time Metric Breakdown per Category...")
    
    # Calculate GPU kernel time vs CPU duration per category
    # GPU kernel time = actual GPU execution (use for bottleneck prioritization)
    # CPU duration = total operation time including sync/launch overhead
    # Sync time = operations where CPU duration >> GPU kernel time
    
    for cat_info in exported_categories:
        category_name = cat_info['name']
        if category_name == 'cpu_idle':
            continue  # Skip cpu_idle category - no ops
        
        category_df = unified_df[unified_df['enhanced_category'] == category_name]
        
        # Calculate GPU kernel time (ms)
        if 'Kernel Time (µs)_sum' in category_df.columns:
            gpu_kernel_time_ms = category_df['Kernel Time (µs)_sum'].sum() / 1000
        elif 'total_direct_kernel_time_ms' in category_df.columns:
            gpu_kernel_time_ms = category_df['total_direct_kernel_time_ms'].sum()
        else:
            gpu_kernel_time_ms = 0
        
        # Calculate CPU duration (ms) - total_duration_us if available
        if 'total_duration_us' in category_df.columns:
            cpu_duration_ms = category_df['total_duration_us'].sum() / 1000
        elif 'Duration (µs)_sum' in category_df.columns:
            cpu_duration_ms = category_df['Duration (µs)_sum'].sum() / 1000
        else:
            cpu_duration_ms = gpu_kernel_time_ms  # Fallback to kernel time
        
        # Calculate sync time (ops where CPU duration >> GPU kernel time)
        # Sync bottleneck = CPU duration - GPU kernel time when ratio > 5x
        sync_time_ms = 0
        sync_ops_count = 0
        
        for _, row in category_df.iterrows():
            if 'Kernel Time (µs)_sum' in row and 'total_duration_us' in row:
                kernel_us = row.get('Kernel Time (µs)_sum', 0) or 0
                duration_us = row.get('total_duration_us', 0) or 0
                if kernel_us > 0 and duration_us > kernel_us * 5:
                    sync_time_ms += (duration_us - kernel_us) / 1000
                    sync_ops_count += 1
        
        # Add time metrics to category info
        cat_info['gpu_kernel_time_ms'] = round(gpu_kernel_time_ms, 3)
        cat_info['cpu_duration_ms'] = round(cpu_duration_ms, 3)
        cat_info['sync_time_ms'] = round(sync_time_ms, 3)
        cat_info['sync_ops_count'] = sync_ops_count
        
        # Flag sync bottleneck if significant
        if sync_time_ms > 0.1 * gpu_kernel_time_ms and sync_time_ms > 1:
            cat_info['has_sync_bottleneck'] = True
            print(f"    ⚠️  {category_name}: Sync bottleneck detected ({sync_time_ms:.2f}ms sync time)")
        else:
            cat_info['has_sync_bottleneck'] = False
        
        # Also update the metadata file with time breakdown
        metadata_file = cat_info['metadata_file']
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            metadata['time_breakdown'] = {
                'gpu_kernel_time_ms': cat_info['gpu_kernel_time_ms'],
                'cpu_duration_ms': cat_info['cpu_duration_ms'],
                'sync_time_ms': cat_info['sync_time_ms'],
                'sync_ops_count': cat_info['sync_ops_count'],
                'has_sync_bottleneck': cat_info['has_sync_bottleneck'],
                'note': 'Use gpu_kernel_time_ms for bottleneck prioritization. sync_time_ms indicates host-device sync overhead.'
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
    
    print(f"  ✓ Time metrics calculated for all categories")
    
    # Save category manifest
    manifest = {
        "platform": platform,
        "trace_path": trace_path,
        "output_dir": output_dir,
        "gpu_utilization": gpu_utilization_metrics,
        "cpu_idle_critical": cpu_idle_critical,
        "categories": exported_categories,
        "time_metric_note": "Use gpu_kernel_time_ms for bottleneck prioritization. cpu_duration_ms includes sync/launch overhead."
    }
    
    manifest_file = f"{output_dir}/category_manifest.json"
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"✓ Orchestrator Preparation Complete (Steps 2-5)")
    print(f"✓ Exported {len(exported_categories)} categories")
    print(f"✓ Manifest saved: {manifest_file}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
