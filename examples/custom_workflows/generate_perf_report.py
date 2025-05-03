"""
Using TraceLens, this script produces a performance report in Excel format from .json traces produced by torch profiler.
Tested with main branch @ 8f40079.

Note that the trace .json files are located recursively - this script does not depend on a certain directory structure.
Outputs are saved in the same directory where the .json files are located.

Example directory tree:

profiling_fa_cudnn
    results_013_profile_nv_cudnn
        traces
            huvideo_traces_rank_0_step_3.json
            ...
            huvideo_traces_rank_7_step_3.json

    results_013_profile_nv_fa
        traces
            huvideo_traces_rank_0_step_3.json
            ...
            huvideo_traces_rank_7_step_3.json
    results_013_profile_rocm_fa
        traces
            huvideo_traces_rank_0_step_3.json
            ...
            huvideo_traces_rank_7_step_3.json

Example usage:

Analyze all .json trace files containing rocm_fa and step_3 in their full path.

Trace files include rank number using the pattern rank_0, rank_1, ..., rank_n.
Script uses this pattern to recognize rank numbers, thus we give -p rank_

python analyze_traces_tracelens.py -b profiling_fa_cudnn -f rocm_fa step_3 -p rank_

Targeting:

profiling_fa_cudnn\results_013_profile_rocm_fa\traces\huvideo_traces_rank_0_step_3.json
profiling_fa_cudnn\results_013_profile_rocm_fa\traces\huvideo_traces_rank_1_step_3.json
profiling_fa_cudnn\results_013_profile_rocm_fa\traces\huvideo_traces_rank_2_step_3.json
profiling_fa_cudnn\results_013_profile_rocm_fa\traces\huvideo_traces_rank_3_step_3.json
profiling_fa_cudnn\results_013_profile_rocm_fa\traces\huvideo_traces_rank_4_step_3.json
profiling_fa_cudnn\results_013_profile_rocm_fa\traces\huvideo_traces_rank_5_step_3.json
profiling_fa_cudnn\results_013_profile_rocm_fa\traces\huvideo_traces_rank_6_step_3.json
profiling_fa_cudnn\results_013_profile_rocm_fa\traces\huvideo_traces_rank_7_step_3.json

NOTE: add -r flag for running microbenchmarking comparison for the most important GEMMs and CONVs.
      add -d flag for launching a dry run (check if correct files will be targeted).
      add -e flag for specifying extension. json and gz are currently supported, json is used by default
      add -a flag for saving all kernels from all ranks (produces a lot of data).
"""

import argparse
import gc
import os.path as osp
import re
import time

import pandas as pd
from node_replay import run_node_replay
from perf_report_utils import (
    build_grouped_breakdown,
    build_kernel_launchers_summary,
    collect_df_perf_metrics_per_group,
    group2ops,
    parse_traces,
    ram_stats_gb,
)
from TraceLens import NcclAnalyser, TreePerfAnalyzer

# Static methods
summarize_df_perf_metrics = TreePerfAnalyzer.summarize_df_perf_metrics


def analyze_traces(
    base_dirpath,
    rank_pattern="rank_",
    ext="json",
    include_only=["rank_0"],
    node_replay=False,
    dry_run=False,
    save_all_kernels=False,
):
    all_traces_grouped = parse_traces(base_dirpath, ext, include_only)

    if all_traces_grouped is None:
        return

    pattern = f"{rank_pattern}(\\d+)"

    for parent_dirpath, filenames in all_traces_grouped.items():
        df_kernel_launchers_all = None
        df_gpu_timelines_all = None
        dfs_all = {group: None for group in group2ops}

        prefix = "_".join(include_only)
        xlsx_path = osp.join(parent_dirpath, f"{prefix}_performance_report.xlsx")

        print("==================== Creating performance report ====================")
        print(f"Parent directory: {parent_dirpath}")
        print(f"Excel file: {osp.basename(xlsx_path)}")
        print(f"Filters: {', '.join(include_only)}")
        print(f"Rank pattern: {rank_pattern}")
        print("Traces:")
        print(*filenames, sep="\n")

        if dry_run:
            print("==================== End of dry run ====================")
            continue

        if osp.exists(xlsx_path):
            print(f"Excel file already exists: ({xlsx_path})")
            print("Terminating...")
            return

        world_size = len(filenames)

        for filename in filenames:
            filepath = osp.join(parent_dirpath, filename)
            match = re.search(pattern, filename)
            if not match:
                print("==================== Failed to extract rank number from trace filename ====================")
                continue

            rank = int(match.group(1))

            print(f"Starting TreePerfAnalyzer with {filename}")
            print(ram_stats_gb())
            start_time = time.perf_counter()
            perf_analyzer = TreePerfAnalyzer.from_file(filepath)

            # Collect group-specific perf metrics from single rank
            dfs_per_group = collect_df_perf_metrics_per_group(perf_analyzer, group2ops)

            for group, df in dfs_per_group.items():
                dfs_all[group] = pd.concat([dfs_all[group], df]) if dfs_all[group] is not None else df

            gc.collect()

            # Collect kernel launcher metrics from single rank
            df_kernel_launchers = perf_analyzer.get_df_kernel_launchers()
            df_kernel_launchers_all = pd.concat([df_kernel_launchers_all, df_kernel_launchers]) if df_kernel_launchers_all is not None else df_kernel_launchers

            # Collect gpu timeline metrics from single rank
            df_gpu_timeline = perf_analyzer.get_df_gpu_timeline()
            df_gpu_timeline.columns = ["type", f"time_ms_{rank}", f"percent_{rank}"]
            df_gpu_timelines_all = pd.concat([df_gpu_timelines_all, df_gpu_timeline.iloc[:, 1:]], axis=1) if df_gpu_timelines_all is not None else df_gpu_timeline

            elapsed_time = time.perf_counter() - start_time
            print(f"Elapsed time: {elapsed_time:.2f} seconds.")
            print(ram_stats_gb())

            ## Collect and write all rank-specific kernels
            if save_all_kernels:
                with pd.ExcelWriter(xlsx_path, mode="a" if osp.exists(xlsx_path) else "w") as writer:
                    perf_analyzer.get_df_kernels().to_excel(writer, sheet_name=f"{rank}_kernels", index=False)

            del perf_analyzer, df_kernel_launchers, df_gpu_timeline
            gc.collect()

        # Build and write group-specific perf metrics summaries from all ranks
        for group, df_all in dfs_all.items():
            if df_all is None:
                print(f"No events to summarize performance metrics from group {group}")
                continue

            df_all_summary = summarize_df_perf_metrics(df_all, agg_metrics=["mean", "std"])
            df_all_summary["kernel_time_sum_cum_pct"] = df_all_summary["Kernel Time (µs)_sum"].cumsum() / df_all_summary["Kernel Time (µs)_sum"].sum()

            with pd.ExcelWriter(xlsx_path, mode="a" if osp.exists(xlsx_path) else "w") as writer:
                df_all_summary.to_excel(writer, sheet_name=group, index=False)

            if node_replay and group in ["gemm", "conv"]:
                df_node_replay_results = run_node_replay(group, df_all_summary)

                with pd.ExcelWriter(xlsx_path, mode="a" if osp.exists(xlsx_path) else "w") as writer:
                    df_node_replay_results.to_excel(writer, sheet_name=f"{group}_node_replay", index=True)

            gc.collect()

        print("Processing dataframes")
        start_time = time.perf_counter()

        # Add avg and std to gpu timelines
        df_gpu_timelines_all["time_ms_avg"] = df_gpu_timelines_all.loc[:, df_gpu_timelines_all.columns.str.contains("time_ms")].mean(axis=1)
        df_gpu_timelines_all["time_ms_std"] = df_gpu_timelines_all.loc[:, df_gpu_timelines_all.columns.str.contains("time_ms")].std(axis=1)

        # Build and write kernel launcher metrics summaries from all ranks
        # Add avg and std to kernel launchers
        # Write gpu timelines from all ranks
        with pd.ExcelWriter(xlsx_path, mode="a" if osp.exists(xlsx_path) else "w") as writer:
            df_kernel_launchers_all_summary = build_kernel_launchers_summary(df_kernel_launchers_all, world_size)

            df_kernel_launchers_all_summary.to_excel(writer, sheet_name="kernel_launchers", index=False)
            df_gpu_timelines_all.to_excel(writer, sheet_name="gpu_timelines", index=False)

        # Build and write high-level grouped breakdown
        df_grouped_breakdown = build_grouped_breakdown(df_kernel_launchers_all_summary, df_gpu_timelines_all)

        with pd.ExcelWriter(xlsx_path, mode="a" if osp.exists(xlsx_path) else "w") as writer:
            df_grouped_breakdown.to_excel(writer, sheet_name="grouped_breakdown", index=False)

        elapsed_time = time.perf_counter() - start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds.")
        print(ram_stats_gb())

        if world_size > 1:
            print(f"Starting NcclAnalyser with world_size {world_size}")
            start_time = time.perf_counter()
            all_filepaths = [osp.join(parent_dirpath, filename) for filename in filenames]
            nccl_analyser = NcclAnalyser(all_filepaths, world_size)

            # Build and write nccl related summaries from all ranks
            with pd.ExcelWriter(xlsx_path, mode="a") as writer:
                nccl_analyser.build_df_summary_nccl_implicit_sync_cat(agg_metrics=["mean"]).to_excel(writer, sheet_name="summary_nccl_implicit_sync_cat", index=False)
                nccl_analyser.build_df_long().to_excel(writer, sheet_name="long", index=False)
                nccl_analyser.build_df_nccl_implicit_sync_cat().to_excel(writer, sheet_name="nccl_implicit_sync_cat", index=False)
                nccl_analyser.build_df_nccl_implicit_sync_cat(detailed=True).to_excel(writer, sheet_name="nccl_implicit_sync_cat_detailed", index=False)
                df_all2allv = nccl_analyser.build_df_nccl_all2allv()
                if df_all2allv:
                    df_all2allv.to_excel(writer, sheet_name="nccl_all2allv", index=False)

            elapsed_time = time.perf_counter() - start_time
            print(f"Elapsed time: {elapsed_time:.2f} seconds.")
            print(ram_stats_gb())

            del nccl_analyser

        gc.collect()


def main():
    parser = argparse.ArgumentParser(description="Parse and summarize traces produced by torch profiler using TraceLens.")
    parser.add_argument("-b", type=str, required=True, help="Path to base directory which contains profiling experiments as subdirectories.")
    parser.add_argument("-p", type=str, default="rank_", help="Pattern to use for finding the rank of a trace. Supports <string><sep> where separator can be empty, - or _.")
    parser.add_argument("-e", type=str, default="json", help="Extension to use for identifying trace files. json and gz are supported.")
    parser.add_argument("-f", type=str, nargs='+', default=["rank_0"], help="Select files containing given substring(s) in their name.")
    parser.add_argument("-r", action="store_true", help="Run node replay for GEMMs and CONVs that contribute 99 pct to group-specific execution time.")
    parser.add_argument("-d", action="store_true", help="Dry run for checking if correct trace paths found.")
    parser.add_argument("-a", action="store_true", help="Save all individual kernels from all ranks (sheets kernels_0 ... kernels_n).")

    args = parser.parse_args()

    analyze_traces(args.b, args.p, args.e, args.f, args.r, args.d, args.a)

if __name__ == "__main__":
    main()
