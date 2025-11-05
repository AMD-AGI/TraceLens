import argparse
import gc
import os.path as osp
import re
import time
import pandas as pd

from concurrent.futures import ProcessPoolExecutor
from node_replay import run_node_replay
from perf_report_utils import (
    build_grouped_breakdown,
    build_kernel_launchers_summary,
    collect_df_perf_metrics_per_group,
    parse_traces,
)
from perf_report_configs import group2ops, kernel_categories
from TraceLens import NcclAnalyser, TreePerfAnalyzer

# Static methods
summarize_df_perf_metrics = TreePerfAnalyzer.summarize_df_perf_metrics


def process_single_trace(args):
    """Process a single trace file and return collected dataframes."""
    filepath, rank = args

    perf_analyzer = TreePerfAnalyzer.from_file(filepath)

    # Collect group-specific perf metrics
    dfs_per_group = collect_df_perf_metrics_per_group(perf_analyzer, group2ops)

    # Collect kernel launcher metrics
    df_kernel_launchers = perf_analyzer.get_df_kernel_launchers()

    # Collect gpu timeline metrics
    df_gpu_timeline = perf_analyzer.get_df_gpu_timeline()

    # Collect information about (un)linked and short kernels
    kernel_events = [e for e in perf_analyzer.tree.events if perf_analyzer.event_to_category(e) in kernel_categories]
    kernel_events_linked = [e for e in kernel_events if e.get("args", {}).get("External id") is not None]
    kernel_events_unlinked = [e for e in kernel_events if e.get("args", {}).get("External id") is None]

    short_kernel_events_linked = [e for e in kernel_events_linked if e.get("dur", 0) < 10]
    short_kernel_events_unlinked = [e for e in kernel_events_unlinked if e.get("dur", 0) < 10]

    # OBS! 'kernel' category only
    df_kernel_events = perf_analyzer.get_df_kernels()
    df_short_kernel_events = df_kernel_events[df_kernel_events["Kernel duration (µs)"] < 10]
    short_cpu_op_counts = df_short_kernel_events["Parent cpu_op"]

    del perf_analyzer

    return {
        'rank': rank,
        'dfs_per_group': dfs_per_group,
        'df_kernel_launchers': df_kernel_launchers,
        'df_gpu_timeline': df_gpu_timeline,
        'short_cpu_op_counts': short_cpu_op_counts,
        "num_kernel_events": len(kernel_events),
        "num_kernel_events_linked": len(kernel_events_linked),
        "num_kernel_events_unlinked": len(kernel_events_unlinked),
        "num_short_kernel_events_linked": len(short_kernel_events_linked),
        "num_short_kernel_events_unlinked": len(short_kernel_events_unlinked),
        "short_cpu_op_counts": short_cpu_op_counts.value_counts().to_dict(),
        "kernel_events_unlinked": kernel_events_unlinked,
    }


def analyze_traces(
    base_dirpath,
    rank_pattern="rank_",
    ext="json",
    include_only=["rank_0"],
    node_replay=False,
    dry_run=False,
    xlsx_path=None,
):
    all_traces_grouped_sorted = parse_traces(base_dirpath, ext, include_only, rank_pattern)

    pattern = f"{rank_pattern}(\\d+)"

    if all_traces_grouped_sorted is None:
        return

    for parent_dirpath, filenames in all_traces_grouped_sorted.items():
        if xlsx_path is not None and len(all_traces_grouped_sorted) > 1:
            print("Multiple parent directories with traces found, give a more specific base path for the report with custom Excel path.")
            return

        df_kernel_launchers_all = None
        df_gpu_timelines_all = None
        dfs_all = {group: None for group in group2ops}

        linked_short_kernels_all = None
        unlinked_kernel_events_all = None
        short_cpu_op_counts_all = {}

        parent_dirname = osp.basename(parent_dirpath)
        prefix = "_".join([parent_dirname, *include_only])

        xlsx_path = xlsx_path or osp.join(parent_dirpath, f"{prefix}_performance_report.xlsx")

        print("==================== Creating performance report ====================")
        print(f"Parent directory: {parent_dirpath}")
        print(f"Excel file: {osp.basename(xlsx_path)}")
        print(f"Filters: {', '.join(include_only)}")
        print(f"Rank pattern: {rank_pattern}")
        print("Traces:")
        print(*filenames, sep="\n")

        if dry_run:
            print("==================== End of dry run ====================")
            xlsx_path = None
            continue

        if osp.exists(xlsx_path):
            print(f"Excel file already exists: ({xlsx_path})")
            print("Terminating...")
            return

        world_size = len(filenames)

        # Prepare arguments for parallel processing
        process_args = []
        for filename in filenames:
            filepath = osp.join(parent_dirpath, filename)
            match = re.search(pattern, filename)
            rank = int(match.group(1))
            process_args.append((filepath, rank))

        # Process traces in parallel
        print("Parallel processing traces")
        start_time = time.perf_counter()

        MAX_WORKERS = min(len(filenames), 8)  # Adjust as needed
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            results = list(executor.map(process_single_trace, process_args))

        elapsed_time = time.perf_counter() - start_time
        print(f"Elapsed time for processing traces: {elapsed_time:.2f} seconds.")

        # Aggregate results
        for result in results:
            rank = result['rank']

            # Concat group-specific dataframes
            for group, df in result['dfs_per_group'].items():
                dfs_all[group] = pd.concat([dfs_all[group], df]) if dfs_all[group] is not None else df

            # Concat kernel launchers
            df_kernel_launchers_all = pd.concat([df_kernel_launchers_all, result['df_kernel_launchers']]) if df_kernel_launchers_all is not None else result['df_kernel_launchers']

            # Concat gpu timelines
            df_gpu_timeline = result['df_gpu_timeline'].copy()
            df_gpu_timeline.columns = ["type", f"time_ms_{rank}", f"percent_{rank}"]
            df_gpu_timelines_all = pd.concat([df_gpu_timelines_all, df_gpu_timeline.iloc[:, 1:]], axis=1) if df_gpu_timelines_all is not None else df_gpu_timeline

            # Collect (un)linked short kernel info
            linked_short_kernels_all.append({
                'rank': rank,
                'num_kernel_events': result['num_kernel_events'],
                'num_kernel_events_linked': result['num_kernel_events_linked'],
                'num_kernel_events_unlinked': result['num_kernel_events_unlinked'],
                'num_short_kernel_events_linked': result['num_short_kernel_events_linked'],
                'num_short_kernel_events_unlinked': result['num_short_kernel_events_unlinked'],
            })

            for e in result["kernel_events_unlinked"]:
                unlinked_kernel_events_all.append({
                    'name': e.get("name", "unknown"),
                    'cat': e.get("cat", "unknown"),
                    'dur': e.get("dur", 0),
                })

            for op_name, count in result["short_cpu_op_counts"].items():
                short_cpu_op_counts_all[op_name] += count

            gc.collect()

        print("Processing dataframes")
        start_time = time.perf_counter()

        # Add avg and std to gpu timelines
        df_gpu_timelines_all["time_ms_avg"] = df_gpu_timelines_all.loc[:, df_gpu_timelines_all.columns.str.contains("time_ms")].mean(axis=1)
        df_gpu_timelines_all["time_ms_std"] = df_gpu_timelines_all.loc[:, df_gpu_timelines_all.columns.str.contains("time_ms")].std(axis=1)

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
                df_node_replay_results = run_node_replay(group, df_all_summary, parent_dirpath)

                with pd.ExcelWriter(xlsx_path, mode="a" if osp.exists(xlsx_path) else "w") as writer:
                    df_node_replay_results.to_excel(writer, sheet_name=f"{group}_node_replay", index=True)

            gc.collect()

        # Build and write kernel launcher metrics summaries from all ranks
        # Add avg and std to kernel launchers
        # Write gpu timelines from all ranks
        with pd.ExcelWriter(xlsx_path, mode="a" if osp.exists(xlsx_path) else "w") as writer:
            df_kernel_launchers_all_summary = build_kernel_launchers_summary(df_kernel_launchers_all, world_size)

            df_kernel_launchers_all_summary.to_excel(writer, sheet_name="kernel_launchers", index=False)
            df_gpu_timelines_all.to_excel(writer, sheet_name="gpu_timelines", index=False)

        # Build and write (un)linked short kernel info
        df_unlinked_kernel_events_all = pd.DataFrame(unlinked_kernel_events_all)
        df_unlinked_summary = df_unlinked_kernel_events_all.groupby(['name', 'cat']).agg(
            count=('dur', 'size'),
            min_dur=('dur', 'min'),
            max_dur=('dur', 'max'),
            mean_dur=('dur', 'mean')
        ).reset_index()

        df_unlinked_summary.sort_values(by='count', ascending=False, inplace=True)
        with pd.ExcelWriter(xlsx_path, mode="a" if osp.exists(xlsx_path) else "w") as writer:
            df_unlinked_summary.to_excel(writer, sheet_name="unlinked_short_kernels", index=False)

        df_short_cpu_op_counts_all = pd.DataFrame.from_dict(short_cpu_op_counts_all, orient='index', columns=['count'])
        df_short_cpu_op_counts_all.sort_values(by='count', ascending=False, inplace=True)
        total_count = df_short_cpu_op_counts_all['count'].sum()
        df_short_cpu_op_counts_all.loc['Total'] = total_count
        with pd.ExcelWriter(xlsx_path, mode="a" if osp.exists(xlsx_path) else "w") as writer:
            df_short_cpu_op_counts_all.to_excel(writer, sheet_name="short_cpu_op_counts", index=False)

        # Build and write high-level grouped breakdown
        df_grouped_breakdown = build_grouped_breakdown(df_kernel_launchers_all_summary, df_gpu_timelines_all)

        with pd.ExcelWriter(xlsx_path, mode="a" if osp.exists(xlsx_path) else "w") as writer:
            df_grouped_breakdown.to_excel(writer, sheet_name="grouped_breakdown", index=False)

        elapsed_time = time.perf_counter() - start_time
        print(f"Elapsed time for processing dataframes: {elapsed_time:.2f} seconds.")

        if world_size > 1:
            print(f"Starting NcclAnalyser with world_size {world_size}")
            start_time = time.perf_counter()
            all_filepaths = [osp.join(parent_dirpath, filename) for filename in filenames]
            nccl_analyser = NcclAnalyser(all_filepaths, world_size)

            # Build and write nccl related summaries from all ranks
            with pd.ExcelWriter(xlsx_path, mode="a") as writer:
                nccl_analyser.build_df_summary_nccl_implicit_sync_cat(agg_metrics=["mean"], strict_metadata_check=False).to_excel(writer, sheet_name="summary_nccl_implicit_sync_cat", index=False)
                nccl_analyser.build_df_long().to_excel(writer, sheet_name="long", index=False)
                nccl_analyser.build_df_nccl_implicit_sync_cat(strict_metadata_check=False).to_excel(writer, sheet_name="nccl_implicit_sync_cat", index=False)
                nccl_analyser.build_df_nccl_implicit_sync_cat(detailed=True, strict_metadata_check=False).to_excel(writer, sheet_name="nccl_implicit_sync_cat_detailed", index=False)
                df_all2allv = nccl_analyser.build_df_nccl_all2allv(strict_metadata_check=False)
                if df_all2allv is not None:
                    df_all2allv.to_excel(writer, sheet_name="nccl_all2allv", index=False)

            elapsed_time = time.perf_counter() - start_time
            print(f"Elapsed time: {elapsed_time:.2f} seconds.")

            del nccl_analyser

        xlsx_path = None
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description="Parse and summarize traces produced by torch profiler using TraceLens.")
    parser.add_argument("-b", type=str, required=True, help="Path to base directory which contains profiling experiments as subdirectories.")
    parser.add_argument("-p", type=str, default="rank_", help="Pattern to use for finding the rank of a trace from filename. Supports <string><sep> where separator can be empty, - or _.")
    parser.add_argument("-e", type=str, default="json", help="Extension to use for identifying trace files. json and gz are supported.")
    parser.add_argument("-f", type=str, nargs='+', default=["rank_0"], help="Select files containing given substring(s) in their full filepaths.")
    parser.add_argument("-r", action="store_true", help="Run node replay for GEMMs and CONVs that contribute 99 pct to group-specific execution time.")
    parser.add_argument("-d", action="store_true", help="Dry run for checking if correct trace paths found.")
    parser.add_argument("-o", type=str, default=None, help="Filepath to save the Excel performance report. Note that this works only with a single base/parent directory containing one set of traces.")

    args = parser.parse_args()

    analyze_traces(args.b, args.p, args.e, args.f, args.r, args.d, args.o)

if __name__ == "__main__":
    main()
