import argparse
import json
import pandas as pd
from TraceLens import TraceToTree
from TraceLens import TreePerfAnalyzer

def main():
    parser = argparse.ArgumentParser(description='Process a JSON trace profile and generate performance report tables.')
    parser.add_argument('--profile_path', type=str, required=True, help='Path to the profile.json file')
    parser.add_argument('--output_xlsx_path', type=str, required=True, help='Path to the output Excel file')
    args = parser.parse_args()

    perf_analyzer = TreePerfAnalyzer.from_file(profile_filepath=args.profile_path)

    # Generate base DataFrames
    df_gpu_timeline = perf_analyzer.get_df_gpu_timeline()
    df_kernel_launchers = perf_analyzer.get_df_kernel_launchers()
    df_kernel_launchers_summary = perf_analyzer.get_df_kernel_launchers_summary(df_kernel_launchers)

    # Define operation categories and their associated operation names.
    # TODO: This mapping should be moved to another file and expanded to include more operations.
    op_category_to_op_name_map = {
        'GEMM': ['aten::mm', 'aten::addmm', 'aten::_scaled_mm'],
        'FLASH_ATTN': ['FlashAttnFunc'],
        'CONV': ['aten::convolution'],
    }

    unary_elemwise_op_names = [
        'aten::copy', 'aten::copy_',
        'atem::clamp_min', 'aten::clamp_min_', 
        'aten::sigmoid',
    ]

    binary_elemwise_op_names = [
        'aten::div', 'aten::div_',
        'aten::mul', 'aten::mul_',
        'aten::add', 'aten::add_',
        'aten::sigmoid_backward',
        'aten::threshold_backward',
    ]

    op_category_to_op_name_map['UNARY_ELEMWISE'] = unary_elemwise_op_names
    op_category_to_op_name_map['BINARY_ELEMWISE'] = binary_elemwise_op_names

    # Dictionary to hold the op-specific DataFrames
    op_dfs = {}

    for op_cat, op_names in op_category_to_op_name_map.items():
        # Filter events belonging to the current category
        op_events = [event for event in perf_analyzer.tree.events if event['name'] in op_names]

        if op_cat in ['GEMM', 'UNARY_ELEMWISE', 'BINARY_ELEMWISE']:
            # For GEMM: create a single table that covers both fwd and bwd.
            df_ops = perf_analyzer.build_df_perf_metrics(op_events, bwd=False, non_data_mov=True)
            df_ops = perf_analyzer.summarize_df_perf_metrics(df_ops, ['mean'])
            op_dfs[op_cat] = df_ops
        else:
            # For FLASH_ATTN and CONV: create separate tables for forward and backward passes.
            df_ops_fwd = perf_analyzer.build_df_perf_metrics(op_events, bwd=False, non_data_mov=True)
            df_ops_fwd = perf_analyzer.summarize_df_perf_metrics(df_ops_fwd, ['mean'])
            df_ops_bwd = perf_analyzer.build_df_perf_metrics(op_events, bwd=True, non_data_mov=True)
            df_ops_bwd = perf_analyzer.summarize_df_perf_metrics(df_ops_bwd, ['mean'])
            op_dfs[f"{op_cat}_fwd"] = df_ops_fwd
            op_dfs[f"{op_cat}_bwd"] = df_ops_bwd

    # Write all DataFrames to separate sheets in an Excel workbook
    with pd.ExcelWriter(args.output_xlsx_path) as writer:
        df_gpu_timeline.to_excel(writer, sheet_name='gpu_timeline', index=False)
        df_kernel_launchers_summary.to_excel(writer, sheet_name='kernel_launchers_summary', index=False)
        
        # Write each op category DataFrame
        for sheet_name, df in op_dfs.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"DataFrames successfully written to {args.output_xlsx_path}")

if __name__ == "__main__":
    main()
