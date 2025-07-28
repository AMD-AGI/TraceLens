#!/usr/bin/env python3
"""
JAX xplane.pb Performance Report Generator

This script demonstrates how to generate performance reports from JAX xplane.pb files
using the TraceLens library. It creates the same analysis outputs as the PyTorch version
but ingests JAX trace data.

Usage:
    python generate_perf_report_jax_xplane.py --xplane_pb_path trace.xplane.pb --output_xlsx_path report.xlsx --output_dir ./results/

Requirements:
    - openpyxl for Excel output
    - tensorboard_plugin_profile for xplane.pb processing (only imported when needed)
"""

import argparse
import json
import pandas as pd
from pathlib import Path


def main():
    # Check openpyxl is installed
    try:
        import openpyxl
    except ImportError:
        raise ImportError("openpyxl is required to write Excel files for perf report gen. Please install it using 'pip install openpyxl'.")

    parser = argparse.ArgumentParser(description='Process a JAX xplane.pb trace file and generate performance report tables.')
    parser.add_argument('--xplane_pb_path', type=str, required=True, help='Path to the xplane.pb file')
    parser.add_argument('--output_xlsx_path', type=str, required=True, help='Path to the output Excel file')
    parser.add_argument('--output_dir', type=str, required=True, help='Path where to output csv files')
    parser.add_argument('--python_path', type=str, default=None, help='Path to the python executable')
    parser.add_argument('--gpu_arch_json_path', type=str, default=None, help='Path to the GPU architecture JSON file')
    args = parser.parse_args()

    # Load the arch json
    gpu_arch_json = None
    if args.gpu_arch_json_path:
        with open(args.gpu_arch_json_path, 'r') as f:
            gpu_arch_json = json.load(f)

    # Import JAX functionality only when needed
    from TraceLens import JaxTreePerfAnalyzer
    from TraceLens.PerfModel import dict_cat2names

    # Create TreePerfAnalyzer from JAX xplane.pb file
    print(f"Creating JAX TreePerfAnalyzer from {args.xplane_pb_path}")
    perf_analyzer = JaxTreePerfAnalyzer.from_xplane_pb(
        profile_filepath=args.xplane_pb_path, 
        arch=gpu_arch_json, 
        python_path=args.python_path
    )

    agg_metrics = ['mean', 'median', 'std', 'min', 'max']

    # Generate base DataFrames - same as PyTorch version
    print("Generating base DataFrames...")
    df_gpu_timeline = perf_analyzer.get_df_gpu_timeline()
    df_kernel_launchers = perf_analyzer.get_df_kernel_launchers(include_kernel_names=True)
    
    # Debug: Print kernel launcher information
    print(f"Debug: df_kernel_launchers shape: {df_kernel_launchers.shape}")
    print(f"Debug: df_kernel_launchers columns: {list(df_kernel_launchers.columns)}")
    print(f"Debug: df_kernel_launchers head:")
    print(df_kernel_launchers.head())
    
    df_kernel_launchers_summary = perf_analyzer.get_df_kernel_launchers_summary(df_kernel_launchers)
    df_kernel_launchers_unique_args = perf_analyzer.get_df_kernel_launchers_unique_args(
        df_kernel_launchers, 
        agg_metrics=agg_metrics, 
        include_pct=True
    )

    # Dictionary to hold the op-specific DataFrames
    op_dfs = {}

    print("Processing operation categories...")
    for op_cat, op_names in dict_cat2names.items():
        # Filter events belonging to the current category
        op_events = [event for event in perf_analyzer.tree.events if event['name'] in op_names]
        
        if op_events:
            print(f"  Processing {op_cat}: {len(op_events)} events")
            df_op = perf_analyzer.get_df_op_args(op_events, agg_metrics=agg_metrics, include_pct=True)
            op_dfs[op_cat] = df_op
        else:
            print(f"  Skipping {op_cat}: no events found")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write to Excel with multiple sheets
    print(f"Writing Excel report to {args.output_xlsx_path}")
    with pd.ExcelWriter(args.output_xlsx_path, engine='openpyxl') as writer:
        # Base sheets
        df_gpu_timeline.to_excel(writer, sheet_name='gpu_timeline', index=False)
        df_kernel_launchers.to_excel(writer, sheet_name='kernel_launchers', index=False)
        df_kernel_launchers_summary.to_excel(writer, sheet_name='kernel_summary', index=False)
        df_kernel_launchers_unique_args.to_excel(writer, sheet_name='kernel_unique_args', index=False)
        
        # Operation-specific sheets
        for op_cat, df_op in op_dfs.items():
            # Clean sheet name for Excel compatibility
            sheet_name = op_cat.replace(' ', '_').replace('/', '_')[:31]  # Excel sheet name limit
            df_op.to_excel(writer, sheet_name=sheet_name, index=False)

    # Write individual CSV files
    print(f"Writing CSV files to {output_dir}")
    df_gpu_timeline.to_csv(output_dir / 'gpu_timeline.csv', index=False)
    df_kernel_launchers.to_csv(output_dir / 'kernel_launchers.csv', index=False)
    df_kernel_launchers_summary.to_csv(output_dir / 'kernel_summary.csv', index=False)
    df_kernel_launchers_unique_args.to_csv(output_dir / 'kernel_unique_args.csv', index=False)
    
    for op_cat, df_op in op_dfs.items():
        safe_name = op_cat.replace(' ', '_').replace('/', '_')
        df_op.to_csv(output_dir / f'{safe_name}.csv', index=False)

    print("JAX performance report generation completed successfully!")
    print(f"Report saved to: {args.output_xlsx_path}")
    print(f"CSV files saved to: {output_dir}")


if __name__ == "__main__":
    main()