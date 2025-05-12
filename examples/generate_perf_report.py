import argparse
import json
from pathlib import Path
import pandas as pd
from TraceLens import TraceToTree
from TraceLens import TreePerfAnalyzer
from TraceLens import TraceFuse


def main():
    parser = argparse.ArgumentParser(
        description="Process a JSON trace profile and generate performance report tables."
    )
    parser.add_argument(
        "--profile_path", type=str, required=True, help="Path to the profile.json file"
    )
    parser.add_argument(
        "--output_xlsx_path",
        type=str,
        required=True,
        help="Path to the output Excel file",
    )
    parser.add_argument(
        "--merge_trace",
        action="store_true",
        help="Merge trace files (profile_path should be a directory). ",
    )
    args = parser.parse_args()

    if args.merge_trace:
        if not Path(args.profile_path).exists() or not Path(args.profile_path).is_dir():
            parser.error(
                f"profile_path is not a valid directory or does not exist. "
                f"When --merge_trace is provided, profile_path should be a directory with trace files."
            )
        else:
            trace_files = sorted(Path(args.profile_path).glob("*.json"))
            if not trace_files:
                parser.error(f"No JSON trace files found in {args.profile_path}")
            elif len(trace_files) == 1:
                print(
                    f"Only one trace file found. Nothing to merge. "
                    f"Extracting stastistics from {trace_files[0]}"
                )
                args.profile_path = str(trace_files[0])
            else:
                # create fodler to save the merged trace file
                merge_path = Path(args.profile_path).joinpath("merged")
                merge_path.mkdir(parents=True, exist_ok=True)
                merge_trace_file = merge_path.joinpath("merged_trace.json")
                print(f"Merging traces...")
                fuser = TraceFuse([str(trace_file) for trace_file in trace_files])
                args.profile_path = fuser.merge_and_save(str(merge_trace_file))
    else:
        if not Path(args.profile_path).is_file():
            parser.error(
                f"profile_path is not a valid file path. "
                f"profile_path should be a file if trace merging is not required. "
                f"profile_path should be a directory if merging is required."
            )

    perf_analyzer = TreePerfAnalyzer.from_file(profile_filepath=args.profile_path)

    # Generate base DataFrames
    df_gpu_timeline = perf_analyzer.get_df_gpu_timeline()
    df_kernel_launchers = perf_analyzer.get_df_kernel_launchers()
    df_kernel_launchers_summary = perf_analyzer.get_df_kernel_launchers_summary(
        df_kernel_launchers
    )

    # Define operation categories and their associated operation names.
    # TODO: This mapping should be moved to another file and expanded to include more operations.
    op_category_to_op_name_map = {
        "GEMM": ["aten::mm", "aten::addmm", "aten::_scaled_mm"],
        "FLASH_ATTN": ["FlashAttnFunc"],
        "CONV": ["aten::convolution"],
    }

    unary_elemwise_op_names = [
        "aten::copy",
        "aten::copy_",
        "atem::clamp_min",
        "aten::clamp_min_",
        "aten::sigmoid",
    ]

    binary_elemwise_op_names = [
        "aten::div",
        "aten::div_",
        "aten::mul",
        "aten::mul_",
        "aten::add",
        "aten::add_",
        "aten::sigmoid_backward",
        "aten::threshold_backward",
    ]

    op_category_to_op_name_map["UNARY_ELEMWISE"] = unary_elemwise_op_names
    op_category_to_op_name_map["BINARY_ELEMWISE"] = binary_elemwise_op_names

    # Dictionary to hold the op-specific DataFrames
    op_dfs = {}

    for op_cat, op_names in op_category_to_op_name_map.items():
        # Filter events belonging to the current category
        op_events = [
            event for event in perf_analyzer.tree.events if event["name"] in op_names
        ]

        if op_cat in ["GEMM", "UNARY_ELEMWISE", "BINARY_ELEMWISE"]:
            # For GEMM: create a single table that covers both fwd and bwd.
            df_ops = perf_analyzer.build_df_perf_metrics(
                op_events, bwd=False, non_data_mov=True
            )
            df_ops = perf_analyzer.summarize_df_perf_metrics(df_ops, ["mean"])

            # sort dataframe rows and reorder columns
            # by M,K,K parameter to simplify comparative analysis
            # of GEMM performance wiht other vendor devices
            # TODO: Perhaps this can be moved to summarize_df_perf_metrics?
            if op_cat == "GEMM":
                sort_columns = ["name", "param: M", "param: N", "param: K"]
            elif op_cat == "UNARY_ELEMWISE":
                sort_columns = ["name", "param: op_shape"]
            elif op_cat == "BINARY_ELEMWISE":
                sort_columns = ["name", "param: shape_in1", "param: shape_in2"]
            else:
                sort_columns = ["name"]

            df_ops.sort_values(
                by=sort_columns,
                inplace=True,
                ignore_index=True,
                ascending=False,
            )

            # reorder columns
            move_to_end = ["TFLOPS/s_mean", "Kernel Time (µs)_sum"]
            remaining_cols = [col for col in df_ops.columns if col not in move_to_end]
            new_cols = remaining_cols + move_to_end
            df_ops = df_ops[new_cols]

            # add percentage of total time
            total_time = df_ops["Kernel Time (µs)_sum"].sum()
            df_ops["pct_total_time"] = (
                df_ops["Kernel Time (µs)_sum"] / total_time * 100
            ).round(3)

            op_dfs[op_cat] = df_ops
        else:
            # For FLASH_ATTN and CONV: create separate tables for forward and backward passes.
            df_ops_fwd = perf_analyzer.build_df_perf_metrics(
                op_events, bwd=False, non_data_mov=True
            )
            df_ops_fwd = perf_analyzer.summarize_df_perf_metrics(df_ops_fwd, ["mean"])
            df_ops_bwd = perf_analyzer.build_df_perf_metrics(
                op_events, bwd=True, non_data_mov=True
            )
            df_ops_bwd = perf_analyzer.summarize_df_perf_metrics(df_ops_bwd, ["mean"])
            op_dfs[f"{op_cat}_fwd"] = df_ops_fwd
            op_dfs[f"{op_cat}_bwd"] = df_ops_bwd

    # Write all DataFrames to separate sheets in an Excel workbook
    with pd.ExcelWriter(args.output_xlsx_path) as writer:
        df_gpu_timeline.to_excel(writer, sheet_name="gpu_timeline", index=False)
        df_kernel_launchers_summary.to_excel(
            writer, sheet_name="kernel_launchers_summary", index=False
        )

        # Write each op category DataFrame
        for sheet_name, df in op_dfs.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"DataFrames successfully written to {args.output_xlsx_path}")


if __name__ == "__main__":
    main()
