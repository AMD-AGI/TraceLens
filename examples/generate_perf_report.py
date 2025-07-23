import os
import argparse
import json
import pandas as pd
import numpy as np
from TraceLens import TraceToTree
from TraceLens import TreePerfAnalyzer
from TraceLens.PerfModel import dict_cat2names
import importlib.util
import subprocess
import sys

def request_install(package_name):
    choice = input(f"Do you want to install '{package_name}' via pip? [y/N]: ").strip().lower()
    if choice == 'y':
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        except subprocess.CalledProcessError:
            print(f"Failed to install '{package_name}'. Please install it manually. Exiting.")
            sys.exit(1)
    else:
        print(f"Skipping installation of '{package_name}' and exiting.")
        sys.exit(1)

def get_dfs_short_kernels(perf_analyzer, short_kernel_threshold_us=10, histogram_bins=100, topk=None):
    """
    TODO: move this to the TreePerfAnalyzer class
    Analyze short kernel events from the performance data and return two DataFrames:
    a histogram of short kernel durations and a summary of top short kernels.

    Args:
        perf_analyzer (TreePerfAnalyzer): The performance analyzer object containing kernel data.
        short_kernel_threshold_us (int, optional): Threshold in microseconds to classify a kernel as "short". Defaults to 10.
        histogram_bins (int, optional): Number of bins for the histogram of short kernel durations. Defaults to 100.
        topk (int, optional): Number of top short kernels to include in the summary. If None, include all. Defaults to None.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: Histogram of short kernel durations with columns ['bin_start', 'bin_end', 'count'].
            - pd.DataFrame: Summary of top short kernels with detailed statistics and percentage contribution to total time.
    """
    df_kernels = perf_analyzer.get_df_kernels()
    df_filtered = df_kernels[df_kernels['Kernel duration (µs)'] < short_kernel_threshold_us]

    # 1. get histogram of these short kernels
    vals = df_filtered['Kernel duration (µs)'].values
    counts, bin_edges = np.histogram(vals, bins=histogram_bins)
    df_hist = pd.DataFrame({
        "bin_start": bin_edges[:-1],
        "bin_end": bin_edges[1:],
        "count": counts
    })

    # 2. get df short kernels topk by total time
    agg_dict = {
        'Kernel duration (µs)': ['sum', 'count', 'mean'],
    }
    df_grouped = df_filtered.groupby(['Parent cpu_op', 'Input dims', 'Input strides', 'Concrete Inputs', 'Kernel name'], sort=False).agg(agg_dict)

    # Flatten multi-level column names
    df_grouped.columns = ['_'.join(col).strip() for col in df_grouped.columns]

    # Rename columns for clarity
    df_grouped.rename(columns={
        'Kernel duration (µs)_sum':  'Short Kernel duration (µs) sum',
        'Kernel duration (µs)_count': 'Short Kernel count',
        'Kernel duration (µs)_mean': 'Short Kernel duration (µs) mean'
    }, inplace=True)

    # Add percentage contribution to total time
    df_grouped['Short Kernel duration (µs) percent of total time'] = (
        df_grouped['Short Kernel duration (µs) sum'] / (perf_analyzer.total_time_ms * 1e3) * 100
    )

    # Sort and format
    df_grouped.sort_values(by='Short Kernel duration (µs) sum', ascending=False, inplace=True)
    df_grouped.reset_index(inplace=True)
    if topk is not None:
        df_grouped = df_grouped.head(topk)
    return df_hist, df_grouped


def main():

    parser = argparse.ArgumentParser(description='Process a JSON trace profile and generate performance report tables.')
    parser.add_argument('--profile_json_path', type=str, required=True, help='Path to the profile.json or .json.gz file')
    parser.add_argument('--output_xlsx_path', type=str, default=None,
                        help='Path to the output Excel file')
    parser.add_argument('--output_csvs_dir', type=str, default=None,
                        help='Directory to save output CSV files')

    # Optional arguments
    parser.add_argument('--short_kernel_study', action='store_true',
                        help='Include short kernel study in the report.')
    parser.add_argument('--short_kernel_threshold_us', type=int, default=10,
                        help='Threshold in microseconds to classify a kernel as "short". Defaults to 10 us.')
    parser.add_argument('--short_kernel_histogram_bins', type=int, default=100,
                        help='Number of bins for the short-kernel histogram.')
    parser.add_argument('--topk_short_kernels', type=int, default=None,
                        help='Rows to keep in the short-kernel table.')

    parser.add_argument('--topk_ops', type=int, default=None,
                        help='Rows to keep in the unique-args launcher table.')
    parser.add_argument('--topk_roofline_ops', type=int, default=None,
                        help='Rows to keep in the roofline table.')

    parser.add_argument('--python_path', type=str, default=None, help='Path to the python executable for gemmologist')
    parser.add_argument('--gpu_arch_json_path', type=str, default=None, help='Path to the GPU architecture JSON file')

    args = parser.parse_args()

    # Load the arch json
    gpu_arch_json = None
    if args.gpu_arch_json_path:
        with open(args.gpu_arch_json_path, 'r') as f:
            gpu_arch_json = json.load(f)

    perf_analyzer = TreePerfAnalyzer.from_file(profile_filepath=args.profile_json_path, arch=gpu_arch_json, python_path=args.python_path)

    agg_metrics = ['mean', 'median', 'std', 'min', 'max']

    # Generate base DataFrames
    df_gpu_timeline = perf_analyzer.get_df_gpu_timeline()

    # TODO: move this to the TreePerfAnalyzer class
    total_time_row = df_gpu_timeline[df_gpu_timeline['type'] == 'total_time']
    total_time_ms = total_time_row['time ms'].values[0]
    perf_analyzer.total_time_ms = total_time_ms

    df_kernel_launchers = perf_analyzer.get_df_kernel_launchers(include_kernel_names=True)
    df_kernel_launchers_summary = perf_analyzer.get_df_kernel_launchers_summary(df_kernel_launchers)
    df_kernel_launchers_summary_by_category = perf_analyzer.get_df_kernel_launchers_summary_by_category(df_kernel_launchers)
    df_kernel_launchers_unique_args = perf_analyzer.get_df_kernel_launchers_unique_args(df_kernel_launchers, 
                                                                                        agg_metrics=agg_metrics, 
                                                                                        include_pct=True)

    # Dictionary to hold the op-specific DataFrames
    perf_metrics_dfs = {}


    for op_cat, op_names in dict_cat2names.items():
        # Filter events belonging to the current category
        op_events = [event for event in perf_analyzer.tree.events if event['name'] in op_names]

        if op_cat in ['GEMM', 'UnaryElementwise', 'BinaryElementwise']:
            # For GEMM: create a single table that covers both fwd and bwd.
            df_ops = perf_analyzer.build_df_perf_metrics(op_events, bwd=False, include_kernel_names=True, include_args=True)
            df_ops = perf_analyzer.summarize_df_perf_metrics(df_ops, agg_metrics)
            perf_metrics_dfs[op_cat] = df_ops
        else:
            # For FLASH_ATTN and CONV: create separate tables for forward and backward passes.
            df_ops_fwd = perf_analyzer.build_df_perf_metrics(op_events, bwd=False, include_kernel_names=True, include_args=True)
            df_ops_fwd = perf_analyzer.summarize_df_perf_metrics(df_ops_fwd, agg_metrics)
            df_ops_bwd = perf_analyzer.build_df_perf_metrics(op_events, bwd=True, include_kernel_names=True, include_args=True)
            df_ops_bwd = perf_analyzer.summarize_df_perf_metrics(df_ops_bwd, agg_metrics)
            perf_metrics_dfs[f"{op_cat}_fwd"] = df_ops_fwd
            perf_metrics_dfs[f"{op_cat}_bwd"] = df_ops_bwd

    # Short kernel study
    df_hist, df_short_kernels = get_dfs_short_kernels(perf_analyzer, short_kernel_threshold_us=args.short_kernel_threshold_us,
                                                    histogram_bins=args.short_kernel_histogram_bins,
                                                    topk=args.topk_short_kernels)

    dict_name2df = {
        'gpu_timeline': df_gpu_timeline,
        'ops_summary_by_category': df_kernel_launchers_summary_by_category,
        'ops_summary': df_kernel_launchers_summary,
        'ops_unique_args': df_kernel_launchers_unique_args,
    }
    # update this dict with the perf_metrics_dfs
    dict_name2df.update(perf_metrics_dfs)
    if args.short_kernel_study:
        dict_name2df['short_kernel_histogram'] = df_hist
        dict_name2df['short_kernels_summary'] = df_short_kernels
    # Write all DataFrames to separate sheets in an Excel workbook
    if args.output_csvs_dir:
        # Ensure the output directory exists
        os.makedirs(args.output_csvs_dir, exist_ok=True)
        for sheet_name, df in dict_name2df.items():
            csv_path = os.path.join(args.output_csvs_dir, f"{sheet_name}.csv")
            df.to_csv(csv_path, index=False)
            print(f"DataFrame '{sheet_name}' written to {csv_path}")
    else:
        if args.output_xlsx_path is None:
            # split input path at 'json' and take the first part and append '.xlsx'
            base_path = args.profile_json_path.rsplit('.json', 1)[0]
            args.output_xlsx_path = base_path + '_perf_report.xlsx'
        try:
            import openpyxl
        except (ImportError, ModuleNotFoundError) as e:
            print(f"Error importing openpyxl: {e}")
            request_install('openpyxl')

        with pd.ExcelWriter(args.output_xlsx_path, engine='openpyxl') as writer:
            for sheet_name, df in dict_name2df.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"DataFrames successfully written to {args.output_xlsx_path}")

if __name__ == "__main__":
    main()
