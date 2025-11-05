import glob
import re
import numpy as np
import os.path as osp
from collections import defaultdict

import pandas as pd
import warnings
from typing import Callable
from perf_report_configs import all_ops_launchers, grouped_breakdown_mapping
from TraceLens import TreePerfAnalyzer

# Static methods
get_df_kernel_launchers_summary = TreePerfAnalyzer.get_df_kernel_launchers_summary


def parse_traces(base_dirpath, ext="json", include_only=["rank_0"], rank_pattern="rank_"):
    pattern = f"{rank_pattern}(\\d+)"
    
    if ext not in ["json", "gz"]:
        print(f"==================== Invalid extension {ext}, json and gz are supported ====================")

    all_traces = [
        filepath
        for filepath in glob.glob(osp.join(base_dirpath, "**", f"*.{ext}"), recursive=True)
        if not [s for s in include_only if s not in filepath]
    ]

    if not all_traces:
        print(f"==================== No {ext} files found, check filters and filepaths ====================")

    all_traces_grouped = defaultdict(list)
    for filepath in all_traces:
        all_traces_grouped[osp.abspath(osp.dirname(filepath))].append(osp.basename(filepath))
        
    all_traces_grouped_sorted = {}
    for parent_dirpath, filenames in all_traces_grouped.items():
        all_traces_grouped_sorted[parent_dirpath] = sorted(filenames, key=lambda x: int(re.search(pattern, x).group(1)))

    return all_traces_grouped_sorted


def collect_df_perf_metrics_per_group(perf_analyzer, group2ops):
    dfs_all = {group: None for group in group2ops}
    all_events_with_shapes = [event for event in perf_analyzer.tree.events if 'Input Dims' in event.get("args", {})]
    all_events_with_shapes_unique_names = set(event["name"] for event in all_events_with_shapes)

    for group, ops in group2ops.items():
        events = [event for event in all_events_with_shapes if event["name"] in ops]

        if not events:
            print(f"Failed to build performance metrics from group {group}.")
            print("Ensure 1) target op is present in the trace, 2) target op is included in group2ops and 3) profiler has record_shapes=True.")
            print("Available ops with input shapes:", all_events_with_shapes_unique_names)
            continue

        df_ops = perf_analyzer.build_df_perf_metrics(events, bwd=False, non_data_mov=True, include_kernel_details=True, include_args=True)
        dfs_all[group] = pd.concat([dfs_all[group], df_ops]) if dfs_all[group] is not None else df_ops

    return dfs_all


def build_kernel_launchers_summary(df_kernel_launchers, world_size):
    df_kernel_launchers_summary = get_df_kernel_launchers_summary(df_kernel_launchers)
    df_kernel_launchers_summary = df_kernel_launchers_summary.drop(columns=["total_direct_kernel_time_sum"])
    df_kernel_launchers_summary.columns = ["name", "count", "kernel_time_ms", "pct", "cum_pct"]
    df_kernel_launchers_summary["time_ms_avg"] = df_kernel_launchers_summary["kernel_time_ms"] / world_size
    df_kernel_launchers_summary["count_avg"] = df_kernel_launchers_summary["count"] / world_size

    return df_kernel_launchers_summary


def build_grouped_breakdown(df_kernel_launchers, df_gpu_timelines):
    df_grouped_breakdown = pd.DataFrame({
        "Group": list(grouped_breakdown_mapping.keys()) + ["Comms", "Memcpy", "Other", "Total"],
    })

    times = []
    assigned_to_group = np.zeros(len(df_kernel_launchers), dtype=bool)
    for ops_launchers in grouped_breakdown_mapping.values():
        if isinstance(ops_launchers, Callable):
            mask = np.array([ops_launchers(x) for x in df_kernel_launchers["name"]])
        else:
            mask = df_kernel_launchers["name"].isin(ops_launchers)
        assigned_to_group[mask] = True
        time = df_kernel_launchers[mask]["time_ms_avg"].sum() / 1000
        times.append(time)

    times.extend([
        df_gpu_timelines[df_gpu_timelines["type"] == "total_comm_time"]["time_ms_avg"].sum() / 1000,
        df_gpu_timelines[df_gpu_timelines["type"] == "total_memcpy_time"]["time_ms_avg"].sum() / 1000,
        df_kernel_launchers[~assigned_to_group]["time_ms_avg"].sum() / 1000,
    ])

    times.append(sum(times))

    df_grouped_breakdown["Time (s)"] = times

    if np.count_nonzero(~assigned_to_group) > 0:
        names_other_group = df_kernel_launchers[~assigned_to_group]["name"].unique().tolist()
        warnings.warn(f"{np.count_nonzero(~assigned_to_group)} kernel launchers were assigned to the \"Other\" group: {names_other_group} "
              "Check the grouped_breakdown_mapping to ensure all relevant kernel launchers are appropriately grouped.")

    return df_grouped_breakdown
