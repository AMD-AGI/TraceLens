###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
TraceLens Extension for Critical Path Analysis

This extension filters operations to include only those on the critical path,
allowing performance report generation focused on critical path operations.

Usage:
    1. Run critical_path_analysis.py to generate critical_trace_nodes.txt containing critical path UIDs
    2. Use this extension with generate_perf_report_pytorch.py:
       
       python generate_perf_report_pytorch.py \
           --profile_json_path <trace.json> \
           --extension_file critical_path_extension.py \
           --output_xlsx_path critical_path_report.xlsx

The extension will:
    - Load critical path UIDs from critical_trace_nodes.txt
    - Filter trace events to only include critical path operations
    - Generate performance reports focused on critical path
"""

import json
import logging
import os

from TraceLens.util import TraceEventUtils


# Global variable to store critical path UIDs
CRITICAL_PATH_UIDS = set()


def load_critical_path_uids(critical_path_file="critical_trace_nodes.json"):
    """
    Load critical path UIDs from a JSON file.

    Args:
        critical_path_file (str): Path to JSON file containing critical path events

    Returns:
        set: Set of critical path UIDs
    """
    global CRITICAL_PATH_UIDS

    if not os.path.exists(critical_path_file):
        logging.warning(f"Critical path file not found: {critical_path_file}")
        logging.warning("Extension will not filter any operations.")
        return set()

    uids = set()
    try:
        with open(critical_path_file, "r") as f:
            data = json.load(f)

        # Handle Chrome Trace Format (traceEvents array) or plain array
        if isinstance(data, dict) and "traceEvents" in data:
            # Chrome Trace Format
            critical_events = data["traceEvents"]
            logging.info(
                f"Loaded Chrome Trace Format with {len(critical_events)} events"
            )
        elif isinstance(data, list):
            # Plain array format (backward compatibility)
            critical_events = data
        else:
            logging.error(f"Unexpected JSON format in {critical_path_file}")
            return set()

        # Extract UIDs from the list of event dictionaries
        for event in critical_events:
            if isinstance(event, dict):
                uid = event.get("UID")
                if uid is not None:
                    uids.add(uid)

        logging.info(f"Loaded {len(uids)} critical path UIDs from {critical_path_file}")
        print(f"✓ Loaded {len(uids)} critical path UIDs from {critical_path_file}")
        CRITICAL_PATH_UIDS = uids
        return uids
    except Exception as e:
        logging.error(f"Error loading critical path UIDs: {e}")
        return set()


def is_on_critical_path(event: dict) -> bool:
    """
    Check if an event is on the critical path.

    Args:
        event (dict): Trace event to check

    Returns:
        bool: True if event is on critical path, False otherwise
    """
    if not CRITICAL_PATH_UIDS:
        # If no critical path UIDs loaded, include all events
        return True

    uid = event.get("UID")
    return uid in CRITICAL_PATH_UIDS


def tree_postprocess_extension(trace_tree):
    """
    Post-process the trace tree to mark events on the critical path.

    This function is called by generate_perf_report_pytorch.py after building
    the trace tree. It marks events that are on the critical path by adding
    a 'critical_path_node' field to each event.

    Args:
        trace_tree: TraceTree object from TraceLens
    """
    # Load critical path UIDs (look in current directory and parent directory)
    # Try multiple possible locations
    possible_paths = [
        "CritPath/output/critical_trace_nodes.json",  # Standard location
        "./CritPath/output/critical_trace_nodes.json",
        "./traces/h200_ddp/critical_trace_nodes.json",
        "../jarvis/traces/h200_ddp/critical_trace_nodes.json",
        "jarvis/traces/h200_ddp/critical_trace_nodes.json",
        "/home/spanmore/work/jarvis/traces/h200_ddp/critical_trace_nodes.json",
        "critical_trace_nodes.json",
        "../critical_trace_nodes.json",
        "critical_trace_nodes.txt",  # Fallback to old format
    ]

    critical_path_file = None
    for path in possible_paths:
        if os.path.exists(path):
            critical_path_file = path
            print(f"Found critical path file at: {path}")
            break

    if critical_path_file is None:
        print(
            f"Critical path file not found in any of these locations: {possible_paths[:3]}"
        )
        print("No events will be marked as critical path.")
        return

    uids = load_critical_path_uids(critical_path_file)

    if not uids:
        print(
            "No critical path UIDs loaded. No events will be marked as critical path."
        )
        return

    print("=" * 80)
    print("CRITICAL PATH EXTENSION: Marking critical path events")
    print(f"Total event count: {len(trace_tree.events)}")

    # Count how many events are on critical path
    critical_count = 0

    # Mark all events with critical_path_node field
    for event in trace_tree.events:
        uid = event.get("UID")
        if uid in uids:
            event["critical_path_node"] = 1
            critical_count += 1
        else:
            event["critical_path_node"] = 0

    print(f"Events on critical path: {critical_count}")
    print(f"Events not on critical path: {len(trace_tree.events) - critical_count}")
    print(
        f"Critical path percentage: {100 * critical_count / len(trace_tree.events):.1f}%"
    )
    print("=" * 80)


def categorize_extension(row, plugin):
    """
    Optional categorizer plugin for critical path operations.
    This is called by the performance analyzer to categorize operations.

    Args:
        row: DataFrame row containing operation information
        plugin: The categorizer plugin object

    Returns:
        str or None: Category name or None if default categorization should be used
    """
    # Add any custom categorization logic here if needed
    # For now, we'll use the default categorization
    return None


# Statistics and reporting
def print_critical_path_stats(trace_tree):
    """
    Print statistics about the critical path filtering.

    Args:
        trace_tree: Filtered TraceTree object
    """
    print("\n" + "=" * 80)
    print("CRITICAL PATH STATISTICS")
    print("=" * 80)

    # Count operations by category
    from collections import Counter

    categories = Counter(event.get("cat") for event in trace_tree.events)

    print(f"\nTotal critical path events: {len(trace_tree.events)}")
    print(f"\nEvents by category:")
    for cat, count in categories.most_common():
        print(f"  {cat}: {count}")

    # Count unique operation names
    cpu_ops = [e for e in trace_tree.events if e.get("cat") == "cpu_op"]
    op_names = Counter(e.get("name") for e in cpu_ops)

    print(f"\nTop 10 CPU operations on critical path:")
    for name, count in op_names.most_common(10):
        print(f"  {name}: {count}")

    print("=" * 80 + "\n")


# Optional: Additional utility functions for critical path analysis


def export_critical_path_summary(trace_tree, output_file="critical_path_summary.json"):
    """
    Export a summary of critical path operations to JSON.

    Args:
        trace_tree: Filtered TraceTree object
        output_file: Path to output JSON file
    """
    summary = {
        "total_events": len(trace_tree.events),
        "event_categories": {},
        "operation_counts": {},
        "critical_path_uids": list(CRITICAL_PATH_UIDS),
    }

    # Count by category
    from collections import Counter

    categories = Counter(event.get("cat") for event in trace_tree.events)
    summary["event_categories"] = dict(categories)

    # Count CPU operations
    cpu_ops = [e for e in trace_tree.events if e.get("cat") == "cpu_op"]
    op_counts = Counter(e.get("name") for e in cpu_ops)
    summary["operation_counts"] = dict(op_counts.most_common(20))

    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)

    logging.info(f"Critical path summary exported to {output_file}")


def _create_critical_path_unique_args(trace_tree, critical_events):
    """
    Create a DataFrame with unique argument combinations for critical path operations.
    Similar to ops_unique_args sheet but only for critical path nodes.

    Args:
        trace_tree: TraceTree object
        critical_events: List of events on the critical path

    Returns:
        pd.DataFrame: DataFrame with unique argument stats
    """
    from collections import defaultdict

    import pandas as pd

    if not critical_events:
        return None

    # Build list of operations with their arguments
    ops_data = []
    for event in critical_events:
        args = event.get("args", {})

        row = {
            "name": event.get("name", ""),
            "category": event.get("cat", ""),
            "UID": event.get("UID"),
            "duration_us": event.get("dur", 0),
        }

        # Add argument details for grouping
        row["Input Dims"] = str(args.get("Input Dims", ""))
        row["Input type"] = str(args.get("Input type", ""))
        row["Input Strides"] = str(args.get("Input Strides", ""))
        row["Concrete Inputs"] = str(args.get("Concrete Inputs", ""))

        # Add kernel details for GPU operations
        if event.get("cat") in ["kernel", "gpu_memcpy", "gpu_memset"]:
            if "gpu_events" in event or "children" in event:
                row["has_gpu_ops"] = True

        ops_data.append(row)

    df = pd.DataFrame(ops_data)

    if df.empty:
        return None

    # Group by name and argument signature
    grouping_cols = [
        "name",
        "category",
        "Input Dims",
        "Input type",
        "Input Strides",
        "Concrete Inputs",
    ]

    # Aggregate statistics
    agg_dict = {
        "UID": ["first", "count"],
        "duration_us": ["mean", "sum", "std", "min", "max"],
    }

    df_grouped = df.groupby(grouping_cols, dropna=False).agg(agg_dict)
    df_grouped.columns = ["_".join(col).strip() for col in df_grouped.columns.values]
    df_grouped.reset_index(inplace=True)

    # Rename columns
    rename_map = {
        "UID_first": "ex_UID",
        "UID_count": "operation_count",
        "duration_us_mean": "avg_duration_us",
        "duration_us_sum": "total_duration_us",
        "duration_us_std": "std_duration_us",
        "duration_us_min": "min_duration_us",
        "duration_us_max": "max_duration_us",
    }
    df_grouped.rename(columns=rename_map, inplace=True)

    # Sort by total duration (descending)
    if "total_duration_us" in df_grouped.columns:
        df_grouped = df_grouped.sort_values("total_duration_us", ascending=False)

    # Add percentage of total critical path time
    total_crit_time = df_grouped["total_duration_us"].sum()
    if total_crit_time > 0:
        df_grouped["pct_of_crit_path"] = (
            df_grouped["total_duration_us"] / total_crit_time * 100
        ).round(2)
        df_grouped["cumulative_pct"] = df_grouped["pct_of_crit_path"].cumsum().round(2)

    # Reorder columns for readability
    col_order = [
        "name",
        "category",
        "operation_count",
        "total_duration_us",
        "avg_duration_us",
        "pct_of_crit_path",
        "cumulative_pct",
        "std_duration_us",
        "min_duration_us",
        "max_duration_us",
        "ex_UID",
        "Input Dims",
        "Input type",
        "Input Strides",
        "Concrete Inputs",
    ]

    # Keep only columns that exist
    col_order = [c for c in col_order if c in df_grouped.columns]
    remaining_cols = [c for c in df_grouped.columns if c not in col_order]
    df_grouped = df_grouped[col_order + remaining_cols]

    return df_grouped


def get_additional_dataframes_extension(trace_tree):
    """
    Generate additional DataFrames to be added as sheets in the Excel report.

    This function is called by the report generator to get custom DataFrames
    that should be added as separate sheets.

    Args:
        trace_tree: TraceTree object from TraceLens

    Returns:
        dict: Dictionary mapping sheet names to pandas DataFrames
    """
    import pandas as pd

    additional_dfs = {}

    # Create critical path summary DataFrame
    critical_events = [e for e in trace_tree.events if e.get("critical_path_node") == 1]

    if not critical_events:
        logging.warning("No critical path events found, skipping critical path sheet")
        return additional_dfs

    # Build a list of critical path nodes with relevant information
    critical_path_data = []
    for event in critical_events:
        row = {
            "UID": event.get("UID"),
            "name": event.get("name", ""),
            "category": event.get("cat", ""),
            "duration_us": event.get("dur", 0),
            "timestamp_us": event.get("ts", 0),
            "pid": event.get("pid", ""),
            "tid": event.get("tid", ""),
        }

        # Add additional fields if available
        args = event.get("args", {})
        if "External id" in args:
            row["external_id"] = args["External id"]

        # Add device info for GPU events
        if event.get("cat") in ["kernel", "gpu_memcpy", "gpu_memset"]:
            row["device_id"] = args.get("device", "")
            row["stream_id"] = args.get("stream", "")

        critical_path_data.append(row)

    df_critical_path = pd.DataFrame(critical_path_data)

    # Sort by timestamp
    if not df_critical_path.empty and "timestamp_us" in df_critical_path.columns:
        df_critical_path = df_critical_path.sort_values("timestamp_us")

    additional_dfs["critical_path_nodes"] = df_critical_path

    # Create critical path summary statistics
    summary_data = []

    # Overall stats
    total_critical_events = len(critical_events)
    total_events = len(trace_tree.events)
    critical_percentage = (
        100 * total_critical_events / total_events if total_events > 0 else 0
    )

    summary_data.append({"Metric": "Total Events", "Value": total_events})
    summary_data.append(
        {"Metric": "Critical Path Events", "Value": total_critical_events}
    )
    summary_data.append(
        {"Metric": "Critical Path Percentage", "Value": f"{critical_percentage:.2f}%"}
    )

    # Category breakdown
    from collections import Counter

    categories = Counter(e.get("cat") for e in critical_events)

    summary_data.append({"Metric": "", "Value": ""})
    summary_data.append({"Metric": "Events by Category", "Value": ""})

    for cat, count in categories.most_common():
        summary_data.append({"Metric": f"  {cat}", "Value": count})

    df_critical_path_summary = pd.DataFrame(summary_data)
    additional_dfs["critical_path_summary"] = df_critical_path_summary

    # Create critical path unique args sheet (similar to ops_unique_args)
    df_crit_unique = _create_critical_path_unique_args(trace_tree, critical_events)
    if df_crit_unique is not None and not df_crit_unique.empty:
        additional_dfs["crit_ops_unique_args"] = df_crit_unique

    logging.info(
        f"Generated {len(additional_dfs)} additional DataFrames for critical path analysis"
    )

    return additional_dfs


# Extension metadata
__extension_name__ = "Critical Path Filter Extension"
__extension_version__ = "1.0.0"
__extension_description__ = "Filters trace operations to only include critical path"
