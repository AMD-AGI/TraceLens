import argparse, os, sys
import json
import jax
import pandas as pd
from pathlib import Path
from collections import defaultdict

from TraceLens import TraceToTree
from TraceLens import TreePerfAnalyzer
from TraceLens.PerfModel import dict_cat2names
from TraceLens.TreePerf import TreePerfAnalyzer, JaxPerfAnalyser, JaxAnalyses
from TraceLens.Reporting.reporting_utils import export_data_df
from TraceLens.PerfModel.jax_op_mapping import categorize_jax_op

def perf_analysis(profile_path: str, arch = None, agg_metrics = ['mean', 'median', 'std', 'min', 'max'], *args, **kwargs) -> dict:
    """
    Generates a performance report for Pytorch analysis from a given profile trace.json file.
    This function processes GPU event statistics and GEMM (General Matrix Multiply) performance data
    from the specified profile file, and exports the results into a dictionary of Dataframes.
    Args:
        profile_path (str): Path to the input XPlane profile protobuf file.
        # *args, **kwargs are passed to the TreePerfAnalyzer constructor.
    Outputs:
        Writes multiple DataFrames containing GPU event statistics and GEMM performance data
        to a dictionary of Dataframes with appropriate suffixes.
    """
    # Get input trace type
    if profile_path.endswith('.pt.trace.json'):
        perf_analyzer = TreePerfAnalyzer.from_file(profile_filepath=profile_path, arch=arch)
    elif profile_path.endswith('.xplane.pb'):
        perf_analyzer = JaxPerfAnalyser.from_file(profile_filepath=profile_path)
        dict_cat2names = defaultdict(list)
    else:
        print('Unsupported trace file format.')
        pass

    # Generate base DataFrames 
    dict_dfs = {}
    df_gpu_timeline = perf_analyzer.get_df_gpu_timeline() 
    df_kernel_launchers = perf_analyzer.get_df_kernel_launchers(include_kernel_names=True)
    df_kernel_launchers_summary = perf_analyzer.get_df_kernel_launchers_summary(df_kernel_launchers)
    df_kernel_launchers_summary_by_category = perf_analyzer.get_df_kernel_launchers_summary_by_category(df_kernel_launchers)
    df_kernel_launchers_unique_args = perf_analyzer.get_df_kernel_launchers_unique_args(df_kernel_launchers, 
                                                                                    agg_metrics=agg_metrics, 
                                                                                    include_pct=True)

    # Store base DataFrames
    dict_dfs['gpu_timeline']= df_gpu_timeline
    dict_dfs['kernel_launchers']= df_kernel_launchers
    dict_dfs['kernel_launchers_summary']= df_kernel_launchers_summary
    dict_dfs['kernel_launchers_summary_by_category']= df_kernel_launchers_summary_by_category 
    dict_dfs['kernel_launchers_unique_args']= df_kernel_launchers_unique_args
    return dict_dfs 

def perf_pytorch(profile_path: str, arch = None, agg_metrics = ['mean', 'median', 'std', 'min', 'max'], *args, **kwargs) -> dict:
    perf_analyzer = TreePerfAnalyzer.from_file(profile_filepath=profile_path, arch=arch)
    
    # Generate & store op-specific DataFrames
    dict_dfs = {}
    for op_cat, op_names in dict_cat2names.items():
        # Filter events belonging to the current category
        op_events = [event for event in perf_analyzer.tree.events if event['name'] in op_names]
        if op_cat in ['GEMM', 'UnaryElementwise', 'BinaryElementwise']: 
            # For GEMM: create a single table that covers both fwd and bwd.
            df_ops = perf_analyzer.build_df_perf_metrics(op_events, bwd=False, include_kernel_names=True, include_args=True)
            df_ops = perf_analyzer.summarize_df_perf_metrics(df_ops, agg_metrics)
            dict_dfs[f"op_{op_cat}"] = df_ops
        else:
            # For FLASH_ATTN and CONV: create separate tables for forward and backward passes.
            df_ops_fwd = perf_analyzer.build_df_perf_metrics(op_events, bwd=False, include_kernel_names=True, include_args=True)
            df_ops_fwd = perf_analyzer.summarize_df_perf_metrics(df_ops_fwd, agg_metrics)
            df_ops_bwd = perf_analyzer.build_df_perf_metrics(op_events, bwd=True, include_kernel_names=True, include_args=True)
            df_ops_bwd = perf_analyzer.summarize_df_perf_metrics(df_ops_bwd, agg_metrics)
            dict_dfs[f"op_{op_cat}_fwd"] = df_ops_fwd
            dict_dfs[f"op_{op_cat}_bwd"] = df_ops_bwd

    return dict_dfs

def perf_jax(profile_path: str, agg_metrics = ['mean', 'median', 'std', 'min', 'max'], *args, **kwargs) -> dict:
    """
    Generates a performance report for JAX analysis from a given XPlane profile protobuf file.
    This function processes GPU event statistics and GEMM (General Matrix Multiply) performance data
    from the specified profile file, and exports the results into a dictionary of Dataframes.

    It summarizes GPU events by calculating averages, categorizing events,
    and grouping XLA events by their base names. It also computes the overlapped communication
    time and appends it to the averages DataFrame.

    Args:
        profile_path (str): Path to the input XPlane profile protobuf file generated by JAX.
        num_cus (int, optional): Number of compute units (CUs) for the GPU architecture. Defaults to 304.

    Returns:
        dict:
            - df_gpu_events_averages (pd.DataFrame): DataFrame containing average times and percentages for various GPU event types, including overlapped communication.
            - df_gpu_events_categorized_mean (pd.DataFrame): DataFrame with categorized GPU event statistics, indexed and renamed for clarity.
            - df_xla_grouped (pd.DataFrame): DataFrame of XLA events grouped by base name, sorted by percentage of total time.
            - df_gemms_detailed(pd.DataFrame): DataFrame of GEMMs

    Usage:
        python3 TraceLens/Reporting/perf_report.py --profile_path $trace_dir/$trace_pytorch.pt.trace.json --output_path $log_dir/tracelens/pytorch  2>&1 | tee $log_dir/tracelens_pytorch.log
        python3 TraceLens/Reporting/perf_report.py --profile_path $trace_dir/$trace_jax.xplane.pb --output_path $log_dir/tracelens/jax 2>&1 | tee $log_dir/tracelens_jax.log

    """
    perf_analyzer = JaxPerfAnalyser.from_file(profile_filepath=profile_path)

    # Generate base DataFrames
    dict_dfs = {}
    df_gpu_events_averages = perf_analyzer.get_df_gpu_events_averages() 
    dict_dfs['gpu_events_averages']= df_gpu_events_averages
    
    if 0:
        # Generate & store op-specific DataFrames
        from TraceLens.PerfModel.jax_op_mapping import ClassCategories
        # TODO: what is dict_cat2names (pytorch)? how is it made? how to make one for jax trace?
        for op_cat, op_names in ClassCategories.items():
            # Filter events belonging to the current category
            op_events = [event for event in perf_analyzer.tree.events if categorize_jax_op(event) == op_cat]
            print(op_cat, "events", len(op_events))
            if op_cat in ['GEMM', 'CONV', 'TE', 'FA V3']: 
                # For GEMM: create a single table that covers both fwd and bwd.
                df_ops = perf_analyzer.build_df_perf_metrics(op_events, bwd=False, include_kernel_names=True, include_args=False)
                df_ops = perf_analyzer.summarize_df_perf_metrics(df_ops, agg_metrics)
                dict_dfs[f"op_{op_cat}"] = df_ops
            else:
                # For FA: bwd and fwd 
                pass

    if 0:
        # Gabe: GPU events stats (legacy from jax_analyses.py) To be replaced by JaxPerAnalyzer.
        averages, categorized, xla_events = JaxAnalyses.summarize_gpu_events(profile_path)
        overlapped_comm = averages[averages['type']=='total_comm_time']['time ms'].iloc[0] - averages[averages['type']=='exposed_comm_time']['time ms'].iloc[0]
        averages.loc[len(averages)] = ['overlapped_comm', overlapped_comm, overlapped_comm/averages[averages['type']=='total_time']['time ms'].iloc[0]*100]
        new_order = [0, 1, 2, 3, 4, 5, 8, 6, 7]
        df_gpu_events_averages = averages.reindex(new_order)
        df_gpu_events_categorized_mean = categorized.copy()
        df_gpu_events_categorized_mean = df_gpu_events_categorized_mean.reset_index().rename(columns={'index': 'name'})

        # XLA events by stripping digits and underscores
        xla_grouped = xla_events.groupby(xla_events.index.str.replace(r'\d+|_+$', '', regex=True)).sum(numeric_only=True)
        xla_grouped = xla_grouped.reset_index().rename(columns={'index': 'short_name_grouped'})
        df_xla_grouped = xla_grouped.sort_values(by='percent', ascending=False)

        # GEMMs
        num_cus = {"num_cus": num_cus}
        df_gemms = JaxAnalyses.summarize_gpu_gemm_events_from_pb(profile_path)
        df_gemms = df_gemms.reset_index().rename(columns={'index': 'name'})
        df_gemms_detailed = JaxAnalyses.gemm_performance_from_pb(profile_path, arch = num_cus)

        # Store DataFrames
        dict_dfs['gpu_events_averages_Gabe']= df_gpu_events_averages
        dict_dfs['gpu_events_categorized_mean']= df_gpu_events_categorized_mean
        dict_dfs['xla_grouped']= df_xla_grouped
        dict_dfs['gemms']= df_gemms
        dict_dfs['gemms_detailed'] = df_gemms_detailed
    return dict_dfs

def main():

    # check openpyxl is installed
    try:
        import openpyxl
    except ImportError:
        raise ImportError("openpyxl is required to write Excel files for perf report gen. Please install it using 'pip install openpyxl'.")

    parser = argparse.ArgumentParser(description='Process a pytroch JSON trace or JAX xplane.pb profile and generate performance report tables.')
    parser.add_argument('--profile_path', type=str, required=True, help='Path to the trace file: pytorch trace.json or jax xplane.pb')
    parser.add_argument('--gpu_arch_json_path', type=str, default=None, help='Path to the GPU architecture JSON file')
    parser.add_argument("--num_cus", type=str, default=304, help="Number of compute units, MI300X - 304; MI210: 104")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output folder")
    parser.add_argument("--output_table_formats", type=str, nargs="+", default=[".xlsx", ".csv"], choices=[".xlsx", ".csv"], help="Output table save formats. You can select one or both formats: .xlsx and/or .csv.")
    parser.add_argument("--output_filename", type=str, default="trace_analysis_results", help="Base name for output files")
    args = parser.parse_args()

    # Load the arch json
    gpu_arch_json = None
    if args.gpu_arch_json_path:
        with open(args.gpu_arch_json_path, 'r') as f:
            gpu_arch_json = json.load(f)

    # Analyze trace profile
    assert args.profile_path.endswith('.pt.trace.json') or args.profile_path.endswith('.xplane.pb')
    dict_dfs = perf_analysis(args.profile_path, arch=gpu_arch_json, num_cus=args.num_cus)
    if args.profile_path.endswith('.pt.trace.json'):
        _dfs = perf_pytorch(args.profile_path)
        dict_dfs.update(_dfs)
    # Additional analysis on Jax xplane.pb trace
    if args.profile_path.endswith('.xplane.pb'):
        _dfs = perf_jax(args.profile_path) 
        dict_dfs.update(_dfs)

    # Save the output
    output_folder = Path(args.output_path)
    output_filename = args.output_filename
    try:
        output_folder.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        print(f"Error: Insufficient permissions to create the directory at {args.output_path}.", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: The specified path {args.output_path} is invalid.", file=sys.stderr)
        sys.exit(1)

    for name_df, df in dict_dfs.items():
        export_data_df(df, 
            output_folder, 
            output_filename,
            output_table_format=args.output_table_formats,
            suffix=f'_{name_df}',)

    print(f"DataFrames successfully written to {args.output_path}")

if __name__ == "__main__":
    main()
