import os
import argparse
import json
from typing import Optional, Dict, Tuple
import pandas as pd
import numpy as np
from TraceLens import TraceToTree
from TraceLens import TreePerfAnalyzer
from TraceLens import NcclAnalyser
import importlib.util
import warnings
import subprocess
import sys
from collections import defaultdict

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
    # If no short kernels, return empty summary to avoid KeyError on missing columns
    if df_filtered.empty:
        return df_hist, pd.DataFrame()
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

def apply_extension(perf_analyzer, extension_path):
    extension_path = os.path.abspath(extension_path)
    extension_name = os.path.splitext(os.path.basename(extension_path))[0]

    spec = importlib.util.spec_from_file_location(extension_name, extension_path)
    extension = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extension)

    if hasattr(extension, 'tree_postprocess_extension'):
        print(f"Applying tree postprocess extension from {extension_path}")
        tree_postprocess_extension = getattr(extension, 'tree_postprocess_extension')
        tree_postprocess_extension(perf_analyzer.tree)
        perf_analyzer.tree.label_non_gpu_paths()
    
    if hasattr(extension, 'perf_model_extension'):
        print(f"Applying perf model extension from {extension_path}")
        perf_model_extension = getattr(extension, 'perf_model_extension')
        if not isinstance(perf_model_extension, dict):
            raise ValueError(f"Expected perf_model_extension to be a dict, got {type(perf_model_extension)}")
        perf_analyzer.op_to_perf_model_class_map.update(perf_model_extension)
    if hasattr(extension, 'dict_cat2names_extension'):
        print(f"Updating dict_cat2names with extension from {extension_path}")
        if not isinstance(extension.dict_cat2names_extension, dict):
            raise ValueError(f"Expected dict_cat2names_extension to be a dict, got {type(extension.dict_cat2names_extension)}")

        # defaultdict(<class 'list'>,
        for cat, names in extension.dict_cat2names_extension.items():
            if cat not in perf_analyzer.dict_cat2names:
                perf_analyzer.dict_cat2names[cat] = []
            if not isinstance(names, list):
                raise ValueError(f"Expected names to be a list, got {type(names)}")
            perf_analyzer.dict_cat2names[cat].extend(names)

def trunc_kernel_details(row, kernel_detail_col, trunc_length=64):
    """
    Truncates the kernel details in a row to a specified length for readability.
    """
    if kernel_detail_col not in row or not row[kernel_detail_col]:
        return None  # No kernel details available

    truncated_details = []
    for detail in row[kernel_detail_col]:
        truncated_name = detail['name'][:trunc_length] + '...' if len(detail['name']) > trunc_length else detail['name']
        truncated_details.append({
            'name': truncated_name,
            'stream': detail.get('stream', None),
            'mean_duration_us': round(detail.get('mean_duration_us', 0), 2)
        })
    
    return truncated_details if truncated_details else None

def add_truncated_kernel_details(df: pd.DataFrame, 
                                 source_col: str = 'kernel_details', 
                                 new_col_name: str = None,
                                 trunc_length: int = 64) -> pd.DataFrame:
    """
    Applies the truncation logic to a DataFrame column and inserts the new
    truncated column immediately after the source column for easy comparison.

    Args:
        df (pd.DataFrame): The DataFrame to process.
        source_col (str): The name of the column containing the full kernel details.
        new_col_name (str): The name for the new truncated column.
        trunc_length (int): The character length to truncate kernel names to.

    Returns:
        pd.DataFrame: A new DataFrame with the added truncated column.
    """
    # First, ensure the source column exists. If not, do nothing.
    if source_col not in df.columns:
        warnings.warn(f"Source column '{source_col}' not found in DataFrame. Skipping truncation.", UserWarning)
        return df
    if new_col_name is None:
        new_col_name = f"trunc_{source_col}"
    # 1. Create the new column's data. It will be added to the end for now.
    df[new_col_name] = df.apply(
        lambda row: trunc_kernel_details(row, source_col, trunc_length=trunc_length),
        axis=1
    )

    # 2. Reorder the columns to place the new column next to its source.
    cols = df.columns.tolist()
    # Pop the new column from the end of the list
    new_col = cols.pop(cols.index(new_col_name))
    # Find the position of our source column and insert the new one after it
    source_col_idx = cols.index(source_col)
    cols.insert(source_col_idx + 1, new_col)
    
    # Return a new DataFrame with the desired column order
    return df[cols]

# Graph analysis helpers

def _make_graph_signature_factory(perf_analyzer):
    def _make_graph_signature(klist):
        if not isinstance(klist, list):
            return tuple()
        sig = []
        for kd in klist:
            uid = kd.get('uid') if isinstance(kd, dict) else None
            if uid is not None and uid in getattr(perf_analyzer.tree, 'events_by_uid', {}):
                ev = perf_analyzer.tree.get_UID2event(uid)
                args = ev.get('args', {}) if isinstance(ev, dict) else {}
                name = ev.get('name')
                grid = tuple(args.get('grid')) if isinstance(args.get('grid'), list) else args.get('grid')
                block = tuple(args.get('block')) if isinstance(args.get('block'), list) else args.get('block')
                smem = args.get('shared memory') if 'shared memory' in args else args.get('shared_memory')
                regs = args.get('registers per thread') if 'registers per thread' in args else args.get('regs_per_thread')
                stream = args.get('stream')
                device = args.get('device')
                context = args.get('context')
                sig.append((name, grid, block, smem, regs, stream, device, context))
            else:
                name = kd.get('name') if isinstance(kd, dict) else str(kd)
                stream = kd.get('stream') if isinstance(kd, dict) else None
                sig.append((name, None, None, None, None, stream, None, None))
        return tuple(sig)
    return _make_graph_signature


def build_graph_ops_summary(perf_analyzer, df_graph_launchers: pd.DataFrame, agg_metrics):
    df = df_graph_launchers.copy()
    make_sig = _make_graph_signature_factory(perf_analyzer)
    df['graph_signature'] = df['kernel_details'].apply(make_sig)

    grp = df.groupby('graph_signature', dropna=False).agg({
        'total_direct_kernel_time': ['sum', 'mean', 'count'],
        'direct_kernel_count': ['mean', 'max']
    })
    grp.columns = ['_'.join(col).strip() for col in grp.columns.values]
    grp = grp.reset_index()
    grp.rename(columns={
        'total_direct_kernel_time_sum': 'group_total_kernel_time_us',
        'total_direct_kernel_time_mean': 'avg_kernel_time_per_launch_us',
        'total_direct_kernel_time_count': 'launch_count',
        'direct_kernel_count_mean': 'avg_direct_kernel_count',
        'direct_kernel_count_max': 'max_direct_kernel_count'
    }, inplace=True)
    grp['group_total_kernel_time_ms'] = grp['group_total_kernel_time_us'] / 1000.0
    total_graph_ms = grp['group_total_kernel_time_ms'].sum()
    grp['Percentage (%)'] = (grp['group_total_kernel_time_ms'] / total_graph_ms * 100.0) if total_graph_ms > 0 else np.nan
    grp['Cumulative Percentage (%)'] = grp['Percentage (%)'].cumsum() if total_graph_ms > 0 else np.nan

    def _sig_preview(sig, max_items=4):
        first = list(sig)[:max_items]
        names = [(t[0] if isinstance(t, tuple) else str(t)) for t in first]
        return ' | '.join(names) + (' ...' if len(sig) > max_items else '')
    grp['graph_signature_preview'] = grp['graph_signature'].apply(_sig_preview)

    sig2total_us = {tuple(k): v for k, v in zip(grp['graph_signature'], grp['group_total_kernel_time_us'])}
    rows_group_kerns = []
    for sig, df_launches in df.groupby('graph_signature', dropna=False):
        index2durations = defaultdict(list)
        name_by_index = {}
        for _, r in df_launches.iterrows():
            klist = r.get('kernel_details') or []
            for idx, kd in enumerate(klist):
                index2durations[idx].append(kd.get('dur'))
                if idx not in name_by_index:
                    name_by_index[idx] = kd.get('name')
        group_total_us = sig2total_us.get(sig, 0)
        for idx in sorted(index2durations.keys()):
            dur_arr = np.array(index2durations[idx])
            total_us = float(np.sum(dur_arr))
            rows_group_kerns.append({
                'graph_signature': sig,
                'kernel_index': idx,
                'kernel_name': name_by_index.get(idx),
                'count': int(len(dur_arr)),
                'total_duration_us': total_us,
                'mean_duration_us': float(np.mean(dur_arr)) if dur_arr.size > 0 else np.nan,
                'median_duration_us': float(np.median(dur_arr)) if dur_arr.size > 0 else np.nan,
                'std_dev_duration_us': float(np.std(dur_arr)) if dur_arr.size > 0 else np.nan,
                'pct_of_graph_group_time': (total_us / group_total_us * 100.0) if group_total_us > 0 else np.nan
            })
    df_graph_group_kernels = pd.DataFrame(rows_group_kerns)

    combined_rows = []
    df_graph_group_summary = grp.sort_values(by='group_total_kernel_time_ms', ascending=False).reset_index(drop=True)
    for _, g in df_graph_group_summary.iterrows():
        sig = g['graph_signature']
        combined_rows.append({
            'level': 0,
            'graph_signature_preview': g['graph_signature_preview'],
            'launch_count': g['launch_count'],
            'group_total_kernel_time_us': g['group_total_kernel_time_us'],
            'group_total_kernel_time_ms': g['group_total_kernel_time_ms'],
            'Percentage (%)': g['Percentage (%)'],
            'Cumulative Percentage (%)': g['Cumulative Percentage (%)']
        })
        kdf = df_graph_group_kernels[df_graph_group_kernels['graph_signature'] == sig].sort_values('kernel_index')
        for _, kr in kdf.iterrows():
            combined_rows.append({
                'level': 1,
                'kernel_index': kr['kernel_index'],
                'kernel_name': kr['kernel_name'],
                'count': kr['count'],
                'total_duration_us': kr['total_duration_us'],
                'mean_duration_us': kr['mean_duration_us'],
                'median_duration_us': kr['median_duration_us'],
                'std_dev_duration_us': kr['std_dev_duration_us'],
                'pct_of_graph_group_time': kr['pct_of_graph_group_time']
            })
    return pd.DataFrame(combined_rows)


def build_graph_ops_unique_args(perf_analyzer, df_graph_launchers: pd.DataFrame, agg_metrics, include_pct=True):
    df_graph_unique_args = perf_analyzer.get_df_kernel_launchers_unique_args(
        df_graph_launchers,
        agg_metrics=agg_metrics,
        include_pct=include_pct
    )
    return add_truncated_kernel_details(
        df_graph_unique_args,
        source_col='kernel_details_summary',
        new_col_name='trunc_kernel_details'
    )

def generate_perf_report_pytorch(profile_json_path: str, 
                                output_xlsx_path: Optional[str] = None,
                                output_csvs_dir: Optional[str] = None,

                                # collective analysis
                                collective_analysis: bool = False,

                                # short kernel study options
                                short_kernel_study: bool = False,
                                short_kernel_threshold_us: int = 10,
                                short_kernel_histogram_bins: int = 100,
                                topk_short_kernels: Optional[int] = None, #include all below thresh by default

                                topk_ops: Optional[int] = None,
                                topk_roofline_ops: Optional[int] = None,

                                extension_file: Optional[str] = None,

                                # for gemmologist
                                python_path: Optional[str] = None,
                                gpu_arch_json_path: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    if gpu_arch_json_path:
        with open(gpu_arch_json_path, 'r') as f:
            gpu_arch_json = json.load(f)
    else:
        gpu_arch_json = None

    perf_analyzer = TreePerfAnalyzer.from_file(profile_filepath=profile_json_path, arch=gpu_arch_json, python_path=python_path)

    if extension_file:
        apply_extension(perf_analyzer, extension_file)

    agg_metrics = ['mean', 'median', 'std', 'min', 'max']

    # Generate base DataFrames
    df_gpu_timeline = perf_analyzer.get_df_gpu_timeline()

    # TODO: move this to the TreePerfAnalyzer class
    total_time_row = df_gpu_timeline[df_gpu_timeline['type'] == 'total_time']
    total_time_ms = total_time_row['time ms'].values[0]
    perf_analyzer.total_time_ms = total_time_ms

    df_kernel_launchers = perf_analyzer.get_df_kernel_launchers(include_kernel_details=True)
    df_kernel_launchers_summary = perf_analyzer.get_df_kernel_launchers_summary(df_kernel_launchers)
    df_kernel_launchers_summary_by_category = perf_analyzer.get_df_kernel_launchers_summary_by_category(df_kernel_launchers)
    df_kernel_launchers_unique_args = perf_analyzer.get_df_kernel_launchers_unique_args(df_kernel_launchers, 
                                                                                        agg_metrics=agg_metrics, 
                                                                                        include_pct=True)
    df_kernel_launchers_unique_args = add_truncated_kernel_details(df_kernel_launchers_unique_args,
                                                                   source_col='kernel_details_summary', new_col_name='trunc_kernel_details')

    # Graph-launch specific breakdowns (CUDA/ROCm)
    df_graph_launchers = df_kernel_launchers[df_kernel_launchers['op category'] == 'GRAPH'].copy()
    if not df_graph_launchers.empty:
        df_graph_summary = build_graph_ops_summary(perf_analyzer, df_graph_launchers, agg_metrics)
        df_graph_unique_args = build_graph_ops_unique_args(perf_analyzer, df_graph_launchers, agg_metrics, include_pct=True)
    else:
        df_graph_summary = pd.DataFrame()
        df_graph_unique_args = pd.DataFrame()

    # Dictionary to hold the op-specific DataFrames
    perf_metrics_dfs = {}


    for op_cat, op_names in perf_analyzer.dict_cat2names.items():
        # Filter events belonging to the current category
        op_events = [event for event in perf_analyzer.tree.events if event['name'] in op_names]

        if op_cat in ['GEMM', 'UnaryElementwise', 'BinaryElementwise']:
            # For GEMM: create a single table that covers both fwd and bwd.
            df_ops = perf_analyzer.build_df_perf_metrics(op_events, bwd=False, include_kernel_details=True, include_args=True)
            df_ops = perf_analyzer.summarize_df_perf_metrics(df_ops, agg_metrics)
            df_ops = add_truncated_kernel_details(df_ops, source_col='kernel_details__summarize_kernel_stats', new_col_name='trunc_kernel_details')
            perf_metrics_dfs[op_cat] = df_ops
        else:
            # For FLASH_ATTN and CONV: create separate tables for forward and backward passes.
            df_ops_fwd = perf_analyzer.build_df_perf_metrics(op_events, bwd=False, include_kernel_details=True, include_args=True)
            df_ops_fwd = perf_analyzer.summarize_df_perf_metrics(df_ops_fwd, agg_metrics)
            df_ops_fwd = add_truncated_kernel_details(df_ops_fwd, source_col='kernel_details__summarize_kernel_stats', new_col_name='trunc_kernel_details')
            df_ops_bwd = perf_analyzer.build_df_perf_metrics(op_events, bwd=True, include_kernel_details=True, include_args=True)
            df_ops_bwd = perf_analyzer.summarize_df_perf_metrics(df_ops_bwd, agg_metrics)
            df_ops_bwd = add_truncated_kernel_details(df_ops_bwd, source_col='kernel_details__summarize_kernel_stats', new_col_name='trunc_kernel_details')
            perf_metrics_dfs[f"{op_cat}_fwd"] = df_ops_fwd
            perf_metrics_dfs[f"{op_cat}_bwd"] = df_ops_bwd

    # Short kernel study
    df_hist, df_short_kernels = get_dfs_short_kernels(perf_analyzer, short_kernel_threshold_us=short_kernel_threshold_us,
                                                    histogram_bins=short_kernel_histogram_bins,
                                                    topk=topk_short_kernels)

    dict_name2df = {
        'gpu_timeline': df_gpu_timeline,
        'ops_summary_by_category': df_kernel_launchers_summary_by_category,
        'ops_summary': df_kernel_launchers_summary,
        'ops_unique_args': df_kernel_launchers_unique_args,
        'graph_ops_summary': df_graph_summary,
        'graph_ops_unique_args': df_graph_unique_args,
    }
    # update this dict with the perf_metrics_dfs
    dict_name2df.update(perf_metrics_dfs)
    if short_kernel_study:
        dict_name2df['short_kernel_histogram'] = df_hist
        dict_name2df['short_kernels_summary'] = df_short_kernels

    if collective_analysis:
        nccl_analyser = NcclAnalyser([profile_json_path], None)
        df_nccl_summary = nccl_analyser.build_df_summary_long()
        dict_name2df['coll_analysis'] = df_nccl_summary

    # Write all DataFrames to separate sheets in an Excel workbook
    if output_csvs_dir:
        # Ensure the output directory exists
        os.makedirs(output_csvs_dir, exist_ok=True)
        for sheet_name, df in dict_name2df.items():
            csv_path = os.path.join(output_csvs_dir, f"{sheet_name}.csv")
            df.to_csv(csv_path, index=False)
            print(f"DataFrame '{sheet_name}' written to {csv_path}")
    else:
        if output_xlsx_path is None:
            # split input path at 'json' and take the first part and append '.xlsx'
            base_path = profile_json_path.rsplit('.json', 1)[0]
            output_xlsx_path = base_path + '_perf_report.xlsx'
        try:
            import openpyxl
        except (ImportError, ModuleNotFoundError) as e:
            print(f"Error importing openpyxl: {e}")
            request_install('openpyxl')

        with pd.ExcelWriter(output_xlsx_path, engine='openpyxl') as writer:
            for sheet_name, df in dict_name2df.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"DataFrames successfully written to {output_xlsx_path}")
    
    return dict_name2df


def main():

    parser = argparse.ArgumentParser(description='Process a JSON trace profile and generate performance report tables.')
    parser.add_argument('--profile_json_path', type=str, required=True, help='Path to the profile.json or .json.gz file')
    parser.add_argument('--output_xlsx_path', type=str, default=None,
                        help='Path to the output Excel file')
    parser.add_argument('--output_csvs_dir', type=str, default=None,
                        help='Directory to save output CSV files')

    # Optional arguments
    parser.add_argument('--collective_analysis', action='store_true',
                        help='Include collective communication analysis in the report.')
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

    parser.add_argument('--extension_file', type=str, default=None,
                        help='Path to the extension file containing custom extensions for TraceTree and PerfModel.')

    parser.add_argument('--python_path', type=str, default=None, help='Path to the python executable for gemmologist')
    parser.add_argument('--gpu_arch_json_path', type=str, default=None, help='Path to the GPU architecture JSON file')

    args = parser.parse_args()
    generate_perf_report_pytorch(profile_json_path=args.profile_json_path,
                                 output_xlsx_path=args.output_xlsx_path,
                                 output_csvs_dir=args.output_csvs_dir,
                                 collective_analysis=args.collective_analysis,
                                 short_kernel_study=args.short_kernel_study,
                                 short_kernel_threshold_us=args.short_kernel_threshold_us,
                                 short_kernel_histogram_bins=args.short_kernel_histogram_bins,
                                 topk_short_kernels=args.topk_short_kernels,
                                 topk_ops=args.topk_ops,
                                 topk_roofline_ops=args.topk_roofline_ops,
                                 extension_file=args.extension_file,
                                 python_path=args.python_path,
                                 gpu_arch_json_path=args.gpu_arch_json_path)
if __name__ == "__main__":
    main()
