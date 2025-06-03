import argparse
import json
import os
import pandas as pd

from pathlib import Path

from TraceLens import TraceToTree
from TraceLens import TreePerfAnalyzer, NcclAnalyser
from TraceLens.PerfModel import dict_cat2names

from TraceLens import TraceFuse

def get_trace_files(
    directory: str, scan_subfolders: bool = False, ignore_files: list = []
) -> list:
    """
    Generate file paths for trace files in a given directory.

    Args:
        directory (str): The directory to search for trace files.
        scan_subfolders (bool, optional): If True, scan subfolders recursively. Defaults to False.
        ignore_files (list, optional): List of file names to ignore. Defaults to an empty list.

    Returns:
        list: A list of file paths for trace files.
    """
    trace_files = []
    if scan_subfolders:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if (
                    file.endswith("json") or file.endswith("json.gz")
                ) and file not in ignore_files:
                    trace_files.append(os.path.join(root, file))
    else:
        for file in os.listdir(directory):
            if (
                file.endswith("json") or file.endswith("json.gz")
            ) and file not in ignore_files:
                trace_files.append(os.path.join(directory, file))

    return trace_files


def main():
    parser = argparse.ArgumentParser(description='Process a JSON trace profile and generate performance report tables.')
    parser.add_argument("--profile_path", type=str, required=True, help="Path to the profile.json file")
    parser.add_argument("--scan_subfolders", help="Scan subfolders for trace files", choices=[True, False], default=True)
    parser.add_argument("--output_xlsx_path", type=str, required=True, help="Path to the output Excel file",)
    parser.add_argument('--gpu_arch_json_path', type=str, default=None, help='Path to the GPU architecture JSON file')
    args = parser.parse_args()

    output_folder = Path(os.path.dirname(args.output_xlsx_path))
    output_folder.mkdir(parents=True, exist_ok=True)

    # Load the arch json
    gpu_arch_json = None
    if args.gpu_arch_json_path:
        with open(args.gpu_arch_json_path, 'r') as f:
            gpu_arch_json = json.load(f)

    trace_files = get_trace_files(directory=args.profile_path, scan_subfolders=args.scan_subfolders)

    # Dictionary to hold the op-specific DataFrames
    op_dfs = {}
    df_ops = {}
    df_ops_fwd = {}
    df_ops_bwd = {}

    for op_cat, op_names in dict_cat2names.items():

        if op_cat in ['GEMM', 'UnaryElementwise', 'BinaryElementwise']:
            op_dfs[op_cat] = []
            df_ops[op_cat] = []

        else:
            op_dfs[f"{op_cat}_fwd"] = []
            op_dfs[f"{op_cat}_bwd"] = []
            df_ops_fwd[op_cat] = []
            df_ops_bwd[op_cat] = []
    

    df_gpu_timeline = []
    df_kernel_launchers = []
    for rank, trace_file in enumerate(sorted(trace_files)):

        print(trace_file)
        
        perf_analyzer = TreePerfAnalyzer.from_file(profile_filepath=trace_file, arch=gpu_arch_json)

        agg_metrics = ['mean', 'median', 'std', 'min', 'max']

        # Generate base DataFrames
        df_gpu_timeline.append(perf_analyzer.get_df_gpu_timeline(rank=rank))

        df_kernel_launchers.append(perf_analyzer.get_df_kernel_launchers(rank=rank, id_cols=True, include_kernel_names=True))

        for op_cat, op_names in dict_cat2names.items():
            # Filter events belonging to the current category
            op_events = [event for event in perf_analyzer.tree.events if event['name'] in op_names]

            if op_cat in ['GEMM', 'UnaryElementwise', 'BinaryElementwise']:
                # For GEMM: create a single table that covers both fwd and bwd.
                df_ops[op_cat].append(perf_analyzer.build_df_perf_metrics(op_events, bwd=False, include_kernel_names=True, include_args=True))
               
            else:
                # For FLASH_ATTN and CONV: create separate tables for forward and backward passes.
                df_ops_fwd[op_cat].append(perf_analyzer.build_df_perf_metrics(op_events, bwd=False, include_kernel_names=True, include_args=True))
                df_ops_bwd[op_cat].append(perf_analyzer.build_df_perf_metrics(op_events, bwd=True, include_kernel_names=True, include_args=True))
               


    df_kernel_launchers = pd.concat(df_kernel_launchers, ignore_index=True)
    df_gpu_timeline = pd.concat(df_gpu_timeline, ignore_index=True)
        
    df_kernel_launchers_summary = perf_analyzer.get_df_kernel_launchers_summary(df_kernel_launchers)
    df_kernel_launchers_unique_args = perf_analyzer.get_df_kernel_launchers_unique_args(df_kernel_launchers, 
                                                                                        agg_metrics=agg_metrics, 
                                                                                        include_pct=True)
    
    for op_cat, op_names in dict_cat2names.items():
        # Filter events belonging to the current category
        op_events = [event for event in perf_analyzer.tree.events if event['name'] in op_names]

        if op_cat in ['GEMM', 'UnaryElementwise', 'BinaryElementwise']:
            # For GEMM: create a single table that covers both fwd and bwd.
            df_ops_cat = df_ops[op_cat]
            df_ops_cat  = pd.concat(df_ops_cat, ignore_index=True)
            df_ops_cat  = perf_analyzer.summarize_df_perf_metrics(df_ops_cat, agg_metrics)
            op_dfs[op_cat].append(df_ops_cat)

        else:
            # For FLASH_ATTN and CONV: create separate tables for forward and backward passes.
            df_ops_cat_fwd = df_ops_fwd[op_cat]
            if df_ops_cat_fwd==[]:
                 continue
            df_ops_cat_fwd  = pd.concat(df_ops_cat_fwd, ignore_index=True)
            df_ops_cat_fwd = perf_analyzer.summarize_df_perf_metrics(df_ops_cat_fwd, agg_metrics)
            op_dfs[f"{op_cat}_fwd"].append(df_ops_cat_fwd)
            
            df_ops_cat_bwd = df_ops_bwd[op_cat]
            if df_ops_cat_bwd==[]:
                 continue
            df_ops_cat_bwd  = pd.concat(df_ops_cat_bwd, ignore_index=True)
            df_ops_cat_bwd = perf_analyzer.summarize_df_perf_metrics(df_ops_cat_bwd, agg_metrics)
            op_dfs[f"{op_cat}_bwd"].append(df_ops_cat_bwd)


    world_size = len(trace_files)
    my_nccl_analyser = NcclAnalyser(trace_files, world_size)

    df_nccl_summary = my_nccl_analyser.build_df_summary_nccl_implicit_sync_cat(agg_metrics=['mean'])
    df_nccl_long = my_nccl_analyser.build_df_long()
    df_nccl_implicit_sync_cat = my_nccl_analyser.build_df_nccl_implicit_sync_cat()

    df_gpu_timeline_summary = df_gpu_timeline.groupby(['type']).agg({'time ms': ['mean'], 'percent': ['mean']})
    df_gpu_timeline_summary.reset_index(inplace=True)
    df_gpu_timeline_summary.columns = [col[0] for col in df_gpu_timeline_summary.columns.values]
    

    # Write all DataFrames to separate sheets in an Excel workbook
    with pd.ExcelWriter(args.output_xlsx_path) as writer:
        df_gpu_timeline.to_excel(writer, sheet_name='gpu_timeline', index=False)
        
        df_gpu_timeline_summary.to_excel(writer, sheet_name='gpu_timeline_summary', index=False)

        df_kernel_launchers_summary.to_excel(writer, sheet_name='kernel_launchers_summary', index=False)
        df_kernel_launchers_unique_args.to_excel(writer, sheet_name='kernel_launchers_unique_args', index=False)
        
        # Write each op category DataFrame
        for sheet_name, df in op_dfs.items():
            df_cat  = pd.concat(df, ignore_index=True)
            df_cat.to_excel(writer, sheet_name=sheet_name, index=False)

        df_nccl_summary.to_excel(writer, sheet_name='nccl_summary', index=False)
        df_nccl_long.to_excel(writer, sheet_name='nccl_long', index=False)
        df_nccl_implicit_sync_cat.to_excel(writer, sheet_name='nccl_implicit_sync_cat', index=False)

    print(f"DataFrames successfully written to {args.output_xlsx_path}")

if __name__ == "__main__":
    main()
