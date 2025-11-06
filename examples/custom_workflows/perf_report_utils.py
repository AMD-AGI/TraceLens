import glob
import os.path as osp
import re
import warnings
from collections import defaultdict
from typing import Callable

import numpy as np
import pandas as pd
from TraceLens import TreePerfAnalyzer

from perf_report_configs import grouped_breakdown_mapping

# Static methods
get_df_kernel_launchers_summary = TreePerfAnalyzer.get_df_kernel_launchers_summary


def parse_traces(
    base_dirpath, ext="json", include_only=["rank_0"], rank_pattern="rank_"
):
    pattern = f"{rank_pattern}(\\d+)"

    if ext not in ["json", "gz"]:
        print(
            f"==================== Invalid extension {ext}, json and gz are supported ===================="
        )
        return None

    all_traces = [
        filepath
        for filepath in glob.glob(
            osp.join(base_dirpath, "**", f"*.{ext}"), recursive=True
        )
        if not [s for s in include_only if s not in filepath]
    ]

    if not all_traces:
        print(
            f"==================== No {ext} files found, check filters and filepaths ===================="
        )
        return None

    all_traces_grouped = defaultdict(list)
    for filepath in all_traces:
        all_traces_grouped[osp.abspath(osp.dirname(filepath))].append(
            osp.basename(filepath)
        )

    all_traces_grouped_sorted = {}
    for parent_dirpath, filenames in all_traces_grouped.items():
        all_traces_grouped_sorted[parent_dirpath] = sorted(
            filenames, key=lambda x: int(re.search(pattern, x).group(1))
        )

    return all_traces_grouped_sorted


def collect_parent_child_hierarchy(perf_analyzer, events, full_stack=False):
    """
    Build a dictionary mapping root parents to their descendant children.

    Args:
        perf_analyzer: TreePerfAnalyzer instance
        events: List of events to analyze

    Returns:
        dict: {root_parent_name: [child1, child2, ...]}
    """
    tree = perf_analyzer.tree
    root_to_children_all = {}

    # Create a set of event names for quick lookup
    event_names = {event["name"] for event in events}

    # For each event, find its root ancestor
    for event in events:
        # Find root: traverse up until parent is not in our event set
        root = event
        parent = tree.get_parent_event(root)

        while parent is not None and parent["name"] in event_names:
            root = parent
            parent = tree.get_parent_event(root)

        # Collect children
        def collect_descendants(evt, full_stack=False):
            descendants = []
            children_dur = 0
            stack = [evt]

            while stack:
                current = stack.pop()
                for child_uid in current.get("children", []):
                    if not child_uid:
                        continue

                    child = tree.get_UID2event(child_uid)
                    # Skip if filtered mode and not in event_names
                    if not full_stack and child["name"] not in event_names:
                        continue

                    descendants.append(child["name"])
                    children_dur += child.get("dur", 0)
                    stack.append(child)

            return descendants, children_dur

        descendants, children_dur = collect_descendants(root, full_stack=full_stack)

        if root["name"] in root_to_children_all:
            match = False
            for _, root_to_children in root_to_children_all[root["name"]].items():
                if root_to_children["children"] == descendants:
                    match = True
                    break

            if not match:
                root_to_children_all[root["name"]][root["UID"]] = {
                    "children": descendants,
                    "total_duration_us": root.get("dur", 0),
                    "children_duration_us": children_dur,
                }
        else:
            root_to_children = {
                "children": descendants,
                "total_duration_us": root.get("dur", 0),
                "children_duration_us": children_dur,
            }

            root_to_children_all[root["name"]] = {root["UID"]: root_to_children}

    return root_to_children_all


def collect_df_perf_metrics_per_group(perf_analyzer, group2ops, rank):
    dfs_all = {group: None for group in group2ops}
    all_events_with_shapes = [
        event
        for event in perf_analyzer.tree.events
        if "Input Dims" in event.get("args", {})
    ]
    parent_child_hierarchy = collect_parent_child_hierarchy(
        perf_analyzer, all_events_with_shapes
    )

    for group, ops in group2ops.items():
        events = [event for event in all_events_with_shapes if event["name"] in ops]

        if not events:
            print(f"Failed to build performance metrics from group {group}.")
            print("Ensure:")
            print("  1) Target op is present in the trace")
            print("  2) Op is present in op_to_perf_model_class_map")
            print("  3) Op is included in group2ops")
            print("  4) Profiler has record_shapes=True")

            print(
                "Available ops with input shapes in the trace (grouped by root parent ops):\n"
            )
            for parent, data in sorted(parent_child_hierarchy.items()):
                for uid, root_to_children in data.items():
                    total_duration_us = root_to_children["total_duration_us"]
                    children_duration_us = root_to_children["children_duration_us"]
                    children = list(set(root_to_children["children"]))
                    print(
                        f"[rank{rank}] Parent: {parent}   UID: {uid}, Duration: {total_duration_us/1000:.3f} ms, "
                        f"   Children: ({children_duration_us/1000:.3f} ms):\n"
                        f"                {', '.join(children) if children else '(none)'}"
                    )
            continue

        df_ops = perf_analyzer.build_df_perf_metrics(
            events,
            bwd=False,
            non_data_mov=True,
            include_kernel_details=True,
            include_args=True,
        )
        dfs_all[group] = (
            pd.concat([dfs_all[group], df_ops])
            if dfs_all[group] is not None
            else df_ops
        )

    return dfs_all


def build_kernel_launchers_summary(df_kernel_launchers, world_size):
    df_kernel_launchers_summary = get_df_kernel_launchers_summary(df_kernel_launchers)
    df_kernel_launchers_summary = df_kernel_launchers_summary.drop(
        columns=["total_direct_kernel_time_sum"]
    )
    df_kernel_launchers_summary.columns = [
        "name",
        "count",
        "kernel_time_ms",
        "pct",
        "cum_pct",
    ]
    df_kernel_launchers_summary["time_ms_avg"] = (
        df_kernel_launchers_summary["kernel_time_ms"] / world_size
    )
    df_kernel_launchers_summary["count_avg"] = (
        df_kernel_launchers_summary["count"] / world_size
    )

    return df_kernel_launchers_summary


def build_grouped_breakdown(df_kernel_launchers, df_gpu_timelines):
    df_grouped_breakdown = pd.DataFrame(
        {
            "Group": list(grouped_breakdown_mapping.keys())
            + ["Comms", "Memcpy", "Other", "Total"],
        }
    )

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

    times.extend(
        [
            df_gpu_timelines[df_gpu_timelines["type"] == "total_comm_time"][
                "time_ms_avg"
            ].sum()
            / 1000,
            df_gpu_timelines[df_gpu_timelines["type"] == "total_memcpy_time"][
                "time_ms_avg"
            ].sum()
            / 1000,
            df_kernel_launchers[~assigned_to_group]["time_ms_avg"].sum() / 1000,
        ]
    )

    times.append(sum(times))

    df_grouped_breakdown["Time (s)"] = times

    if np.count_nonzero(~assigned_to_group) > 0:
        names_other_group = (
            df_kernel_launchers[~assigned_to_group]["name"].unique().tolist()
        )
        warnings.warn(
            f'{np.count_nonzero(~assigned_to_group)} kernel launchers were assigned to the "Other" group: {names_other_group} '
            "Check the grouped_breakdown_mapping to ensure all relevant kernel launchers are appropriately grouped."
        )

    return df_grouped_breakdown
