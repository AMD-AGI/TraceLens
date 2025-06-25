import argparse
import json
import pandas as pd
from TraceLens import TraceToTree
from TraceLens import TreePerfAnalyzer
from TraceLens.PerfModel import dict_cat2names

def main():

    # check openpyxl is installed
    try:
        import openpyxl
    except ImportError:
        raise ImportError("openpyxl is required to write Excel files for perf report gen. Please install it using 'pip install openpyxl'.")

    parser = argparse.ArgumentParser(description='Process a JSON trace profile and generate performance report tables.')
    parser.add_argument('--profile_json_path', type=str, required=True, help='Path to the profile.json file')
    parser.add_argument('--output_xlsx_path', type=str, required=True, help='Path to the output Excel file')
    parser.add_argument('--python_path', type=str, default=None, help='Path to the python executable')
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
    df_kernel_launchers = perf_analyzer.get_df_kernel_launchers(include_kernel_names=True)
    df_kernel_launchers_summary = perf_analyzer.get_df_kernel_launchers_summary(df_kernel_launchers)
    df_kernel_launchers_unique_args = perf_analyzer.get_df_kernel_launchers_unique_args(df_kernel_launchers, 
                                                                                        agg_metrics=agg_metrics, 
                                                                                        include_pct=True)

    # Dictionary to hold the op-specific DataFrames
    op_dfs = {}


    for op_cat, op_names in dict_cat2names.items():
        # Filter events belonging to the current category
        op_events = [event for event in perf_analyzer.tree.events if event['name'] in op_names]

        if op_cat in ['GEMM', 'UnaryElementwise', 'BinaryElementwise']:
            # For GEMM: create a single table that covers both fwd and bwd.
            df_ops = perf_analyzer.build_df_perf_metrics(op_events, bwd=False, include_kernel_names=True, include_args=True)
            df_ops = perf_analyzer.summarize_df_perf_metrics(df_ops, agg_metrics)
            op_dfs[op_cat] = df_ops
        else:
            # For FLASH_ATTN and CONV: create separate tables for forward and backward passes.
            df_ops_fwd = perf_analyzer.build_df_perf_metrics(op_events, bwd=False, include_kernel_names=True, include_args=True)
            df_ops_fwd = perf_analyzer.summarize_df_perf_metrics(df_ops_fwd, agg_metrics)
            df_ops_bwd = perf_analyzer.build_df_perf_metrics(op_events, bwd=True, include_kernel_names=True, include_args=True)
            df_ops_bwd = perf_analyzer.summarize_df_perf_metrics(df_ops_bwd, agg_metrics)
            op_dfs[f"{op_cat}_fwd"] = df_ops_fwd
            op_dfs[f"{op_cat}_bwd"] = df_ops_bwd

    # Write all DataFrames to separate sheets in an Excel workbook
    with pd.ExcelWriter(args.output_xlsx_path) as writer:
        df_gpu_timeline.to_excel(writer, sheet_name='gpu_timeline', index=False)
        df_kernel_launchers_summary.to_excel(writer, sheet_name='kernel_launchers_summary', index=False)
        df_kernel_launchers_unique_args.to_excel(writer, sheet_name='kernel_launchers_unique_args', index=False)
        
        # Write each op category DataFrame
        for sheet_name, df in op_dfs.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"DataFrames successfully written to {args.output_xlsx_path}")

if __name__ == "__main__":
    main()
