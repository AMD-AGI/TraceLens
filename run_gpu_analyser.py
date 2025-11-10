#!/usr/bin/env python3
"""
Standalone script to run GPUEventAnalyser on raw trace files.
Bypasses package-level imports to avoid dependency issues.
"""

import json
import gzip
import sys
import os
import importlib.util

def load_module_directly(module_path):
    """Load a Python module directly from a file path without package imports."""
    spec = importlib.util.spec_from_file_location("gpu_event_analyser", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def load_trace_file(trace_path):
    """
    Load a trace file (supports .json and .json.gz formats).
    Returns the list of events from the trace.
    """
    print(f"Loading trace from: {trace_path}")
    
    if trace_path.endswith('.gz'):
        with gzip.open(trace_path, 'rt', encoding='utf-8') as f:
            trace_data = json.load(f)
    else:
        with open(trace_path, 'r', encoding='utf-8') as f:
            trace_data = json.load(f)
    
    # PyTorch trace format has events under 'traceEvents' key
    if isinstance(trace_data, dict):
        events = trace_data.get('traceEvents', [])
    else:
        events = trace_data
    
    print(f"✓ Loaded {len(events)} events")
    return events

def analyze_pytorch_trace(analyser_class, trace_path, micro_idle_thresh_us=None, save_csv=None,
                          list_all_kernels=False, all_kernels_output=None):
    """
    Analyze a PyTorch trace file and print GPU metrics.
    """
    events = load_trace_file(trace_path)
    
    # Create analyzer instance
    print("\nCreating PytorchGPUEventAnalyser instance...")
    analyser = analyser_class(events)
    
    # Compute metrics
    print("\n" + "="*70)
    print("GPU METRICS")
    print("="*70)
    print("\nComputing metrics...")
    metrics = analyser.compute_metrics(micro_idle_thresh_us=micro_idle_thresh_us)
    
    print(f"\n{'Metric':<30} {'Time (µs)':>15} {'Time (ms)':>15} {'%':>8}")
    print("-"*70)
    total_time = metrics['total_time']
    for key, value in metrics.items():
        pct = (value / total_time * 100) if total_time > 0 else 0
        print(f"{key:<30} {value:>15.2f} {value/1000:>15.2f} {pct:>7.1f}%")
    
    # Get breakdown DataFrame
    print("\n" + "="*70)
    print("GPU TIME BREAKDOWN (DataFrame)")
    print("="*70)
    df_breakdown = analyser.get_breakdown_df(micro_idle_thresh_us=micro_idle_thresh_us)
    print(df_breakdown.to_string(index=False))
    
    # Save to CSV if requested
    if save_csv:
        df_breakdown.to_csv(save_csv, index=False)
        print(f"\n✓ Breakdown saved to: {save_csv}")
    
    # List all kernels with categories if requested
    if list_all_kernels:
        print("\n" + "="*70)
        print("ALL KERNELS WITH CATEGORIZATION")
        print("="*70)
        
        print("\nGenerating comprehensive kernel list...")
        kernel_reports = analyser.get_kernel_operations_report()
        
        # Get kernel summary first
        df_summary = analyser.get_kernel_summary_stats()
        print("\n--- Kernel Summary by Category ---")
        print(df_summary.to_string(index=False))
        
        # Combine all category DataFrames into one with category column
        import pandas as pd
        all_kernels_dfs = []
        for category, df in kernel_reports.items():
            if not df.empty:
                df_copy = df.copy()
                df_copy.insert(0, 'category', category)
                all_kernels_dfs.append(df_copy)
        
        if all_kernels_dfs:
            df_all_kernels = pd.concat(all_kernels_dfs, ignore_index=True)
            
            # Sort by total duration descending
            if 'total_duration_us' in df_all_kernels.columns:
                df_all_kernels = df_all_kernels.sort_values('total_duration_us', ascending=False)
            
            print(f"\n✓ Found {len(df_all_kernels)} unique kernels across all categories")
            print(f"\nTop 50 kernels by total duration:")
            print(df_all_kernels.head(50).to_string(index=False))
            
            # Save if output file specified
            if all_kernels_output:
                df_all_kernels.to_csv(all_kernels_output, index=False)
                print(f"\n✓ All kernels list saved to: {all_kernels_output}")
                # Also save summary
                summary_path = all_kernels_output.replace('.csv', '_summary.csv')
                df_summary.to_csv(summary_path, index=False)
                print(f"✓ Kernel summary saved to: {summary_path}")
        else:
            print("\n⚠ No kernels found in trace")
    
    return analyser, metrics, df_breakdown

def analyze_jax_trace(jax_analyser_class, trace_path, gpu_pid=1, save_csv=None,
                     list_all_kernels=False, all_kernels_output=None):
    """
    Analyze a JAX trace file and print GPU metrics.
    JAX traces can have multiple GPUs with different PIDs.
    """
    events = load_trace_file(trace_path)
    
    # Create analyzer instance
    print("\nCreating JaxGPUEventAnalyser instance...")
    analyser = jax_analyser_class(events)
    
    # For JAX, you can analyze specific GPU or get average across all GPUs
    print(f"\n{'='*70}")
    print(f"GPU {gpu_pid-1} METRICS (PID={gpu_pid})")
    print("="*70)
    metrics = analyser.compute_metrics(gpu_pid=gpu_pid)
    
    print(f"\n{'Metric':<30} {'Time (µs)':>15} {'Time (ms)':>15}")
    print("-"*70)
    for key, value in metrics.items():
        print(f"{key:<30} {value:>15.2f} {value/1000:>15.2f}")
    
    # Get breakdown for all GPUs
    print("\n" + "="*70)
    print("ALL GPUs BREAKDOWN")
    print("="*70)
    df_all_gpus = analyser.get_breakdown_df(gpu_pid=gpu_pid)
    print(df_all_gpus.to_string(index=False))
    
    # Get average across all GPUs (if multiple GPUs exist)
    try:
        print("\n" + "="*70)
        print("AVERAGE ACROSS ALL GPUs")
        print("="*70)
        df_avg = analyser.get_average_df()
        print(df_avg.to_string(index=False))
        
        if save_csv:
            df_avg.to_csv(save_csv, index=False)
            print(f"\n✓ Average breakdown saved to: {save_csv}")
    except Exception as e:
        print(f"Note: Could not compute average across GPUs: {e}")
        if save_csv:
            df_all_gpus.to_csv(save_csv, index=False)
            print(f"\n✓ Breakdown saved to: {save_csv}")
    
    # List all kernels with categories if requested
    if list_all_kernels:
        print("\n" + "="*70)
        print("ALL KERNELS WITH CATEGORIZATION")
        print("="*70)
        print("\nNote: For JAX, kernel reports show aggregate across all GPUs")
        
        print("\nGenerating comprehensive kernel list...")
        kernel_reports = analyser.get_kernel_operations_report()
        
        # Get kernel summary first
        df_summary = analyser.get_kernel_summary_stats()
        print("\n--- Kernel Summary by Category ---")
        print(df_summary.to_string(index=False))
        
        # Combine all category DataFrames into one with category column
        import pandas as pd
        all_kernels_dfs = []
        for category, df in kernel_reports.items():
            if not df.empty:
                df_copy = df.copy()
                df_copy.insert(0, 'category', category)
                all_kernels_dfs.append(df_copy)
        
        if all_kernels_dfs:
            df_all_kernels = pd.concat(all_kernels_dfs, ignore_index=True)
            
            # Sort by total duration descending
            if 'total_duration_us' in df_all_kernels.columns:
                df_all_kernels = df_all_kernels.sort_values('total_duration_us', ascending=False)
            
            print(f"\n✓ Found {len(df_all_kernels)} unique kernels across all categories")
            print(f"\nTop 50 kernels by total duration:")
            print(df_all_kernels.head(50).to_string(index=False))
            
            # Save if output file specified
            if all_kernels_output:
                df_all_kernels.to_csv(all_kernels_output, index=False)
                print(f"\n✓ All kernels list saved to: {all_kernels_output}")
                # Also save summary
                summary_path = all_kernels_output.replace('.csv', '_summary.csv')
                df_summary.to_csv(summary_path, index=False)
                print(f"✓ Kernel summary saved to: {summary_path}")
        else:
            print("\n⚠ No kernels found in trace")
    
    return analyser, metrics, df_all_gpus

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analyze GPU events from a trace file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze PyTorch trace
  python3 run_gpu_analyser.py trace.json --framework pytorch
  
  # Analyze gzipped trace
  python3 run_gpu_analyser.py "trace.json.gz" --framework pytorch
  
  # Analyze with micro-idle threshold of 50 microseconds
  python3 run_gpu_analyser.py trace.json --framework pytorch --micro-idle-threshold 50
  
  # Analyze JAX trace for GPU 0 (PID 1)
  python3 run_gpu_analyser.py trace.json --framework jax --gpu-pid 1
  
  # Save breakdown to CSV
  python3 run_gpu_analyser.py trace.json --framework pytorch --output breakdown.csv
  
  # List ALL kernels with categorization
  python3 run_gpu_analyser.py trace.json --framework pytorch --list-all-kernels --all-kernels-output my_kernels.csv
        """
    )
    
    parser.add_argument('trace_file', help='Path to trace file (.json or .json.gz)')
    parser.add_argument(
        '--framework',
        choices=['pytorch', 'jax'],
        default='pytorch',
        help='Framework that generated the trace (default: pytorch)'
    )
    parser.add_argument(
        '--gpu-pid',
        type=int,
        default=1,
        help='GPU PID for JAX traces (default: 1, which is GPU 0)'
    )
    parser.add_argument(
        '--output',
        '-o',
        help='Optional: Save breakdown DataFrame to CSV file'
    )
    parser.add_argument(
        '--micro-idle-threshold',
        type=float,
        default=None,
        help='Optional: Micro-idle threshold in microseconds (splits idle into micro/macro idle)'
    )
    parser.add_argument(
        '--list-all-kernels',
        action='store_true',
        help='Generate a comprehensive list of ALL kernels with their categories'
    )
    parser.add_argument(
        '--all-kernels-output',
        type=str,
        default=None,
        help='Output file for all kernels list (e.g., all_kernels.csv)'
    )
    
    args = parser.parse_args()
    
    try:
        # Load the gpu_event_analyser module directly
        script_dir = os.path.dirname(os.path.abspath(__file__))
        module_path = os.path.join(script_dir, 'TraceLens', 'TreePerf', 'gpu_event_analyser.py')
        
        if not os.path.exists(module_path):
            print(f"✗ Error: Could not find gpu_event_analyser.py at {module_path}")
            return 1
        
        print(f"Loading GPUEventAnalyser module from: {module_path}")
        gpu_module = load_module_directly(module_path)
        
        if args.framework == 'pytorch':
            analyser, metrics, df_breakdown = analyze_pytorch_trace(
                gpu_module.PytorchGPUEventAnalyser,
                args.trace_file, 
                micro_idle_thresh_us=args.micro_idle_threshold,
                save_csv=args.output,
                list_all_kernels=args.list_all_kernels,
                all_kernels_output=args.all_kernels_output
            )
        else:  # jax
            analyser, metrics, df_breakdown = analyze_jax_trace(
                gpu_module.JaxGPUEventAnalyser,
                args.trace_file, 
                args.gpu_pid,
                save_csv=args.output,
                list_all_kernels=args.list_all_kernels,
                all_kernels_output=args.all_kernels_output
            )
        
        print("\n" + "="*70)
        print("✓ Analysis complete!")
        print("="*70)
        
    except FileNotFoundError as e:
        print(f"✗ Error: Trace file not found: {args.trace_file}")
        print(f"  Details: {e}")
        return 1
    except json.JSONDecodeError as e:
        print(f"✗ Error: Invalid JSON in trace file: {e}")
        return 1
    except Exception as e:
        print(f"✗ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

