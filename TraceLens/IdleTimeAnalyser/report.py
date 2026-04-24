###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from collections import defaultdict


def get_grouping_key(rec):
    if rec["cpu_during_gap"] == "RUNTIME_DOMINATED":
        return rec["cpu_during_gap_detail"] or "OTHER_RUNTIME"
    if rec["cpu_during_gap"] == "CPU_DOMINATED":
        return rec["dominant_op"] or "unknown"
    if rec["cpu_during_gap"] == "CPU_UNTRACED":
        return rec["dominant_op"] or "(no_cpu_op_overlap)"
    if rec["cpu_during_gap"] == "LAUNCH_ANOMALY":
        return "prequeued" if rec.get("kernel_prequeued") else "launched_during_gap"
    if rec["cpu_during_gap"] == "LAUNCH_OVERHEAD_ONLY":
        return "prequeued" if rec.get("kernel_prequeued") else "launched_during_gap"
    return rec.get("cpu_during_gap_detail", "")


def compute_stats(items):
    import numpy as np
    durations = [r["duration"] for r in items]
    arr = np.array(durations)
    return {
        "count": len(items),
        "total_time_ms": arr.sum() / 1e3,
        "mean_us": arr.mean(),
        "median_us": float(np.median(arr)),
        "std_us": arr.std(),
        "min_us": arr.min(),
        "max_us": arr.max(),
    }


def build_idle_dataframes(classified, gpu_busy_time_us=None):
    """Build idle_overview, idle_summary, and idle_intervals DataFrames.

    When gpu_busy_time_us is set, idle_overview gains pct_of_trace and
    gpu_utilization_pct (idle vs full GPU timeline: busy + macro idle).
    """
    import pandas as pd

    macro = [r for r in classified if not r["label_noise"]]

    # --- Sheet 1: idle_summary ---
    summary_rows = []

    # Total row
    if macro:
        total_stats = compute_stats(macro)
        total_stats["drain_type"] = "ALL"
        total_stats["cpu_during_gap"] = "ALL"
        total_stats["dominant_op"] = "ALL"
        total_stats["pct_of_idle"] = 100.0
        summary_rows.append(total_stats)

    # Noise row — with actual stats
    noise_intervals = [r for r in classified if r["label_noise"]]
    if noise_intervals:
        noise_stats = compute_stats(noise_intervals)
        noise_stats["drain_type"] = "—"
        noise_stats["cpu_during_gap"] = "noise"
        noise_stats["dominant_op"] = "—"
        noise_stats["pct_of_idle"] = None
        summary_rows.append(noise_stats)
    else:
        summary_rows.append({
            "drain_type": "—", "cpu_during_gap": "noise", "dominant_op": "—",
            "count": 0, "total_time_ms": 0.0,
            "pct_of_idle": None, "mean_us": None, "median_us": None,
            "std_us": None, "min_us": None, "max_us": None,
        })

    total_macro_time = sum(r["duration"] for r in macro) if macro else 1.0

    combo_groups = defaultdict(list)
    for rec in macro:
        key = (rec["drain_type"], rec["cpu_during_gap"], get_grouping_key(rec))
        combo_groups[key].append(rec)

    # Sort by total time descending
    sorted_combos = sorted(combo_groups.items(), key=lambda x: -sum(r["duration"] for r in x[1]))
    for (drain, cpu_gap, dom_op), items in sorted_combos:
        stats = compute_stats(items)
        stats["drain_type"] = drain
        stats["cpu_during_gap"] = cpu_gap
        stats["dominant_op"] = dom_op
        stats["pct_of_idle"] = stats["total_time_ms"] / (total_macro_time / 1e3) * 100
        ids = sorted(r.get("idle_id", -1) for r in items)
        stats["idle_ids"] = ",".join(str(x) for x in ids)
        summary_rows.append(stats)

    summary_col_order = [
        "drain_type", "cpu_during_gap", "dominant_op", "count",
        "total_time_ms", "pct_of_idle", "cumulative_pct", "mean_us", "median_us",
        "std_us", "min_us", "max_us", "idle_ids",
    ]
    df_summary = pd.DataFrame(summary_rows, columns=summary_col_order)

    # Cumulative % only for the combo rows (skip ALL and noise rows)
    combo_mask = ~df_summary["cpu_during_gap"].isin(["ALL", "noise"])
    if combo_mask.any():
        df_summary.loc[combo_mask, "cumulative_pct"] = (
            df_summary.loc[combo_mask, "pct_of_idle"].cumsum()
        )

    # --- Sheet 2: idle_intervals ---
    interval_rows = []
    for i, rec in enumerate(macro):
        group = f"{rec['drain_type']} | {rec['cpu_during_gap']} | {get_grouping_key(rec)}"
        interval_rows.append({
            "idle_id": rec.get("idle_id", i),
            "group": group,
            "start_us": rec["start"],
            "end_us": rec["end"],
            "duration_us": rec["duration"],
            "drain_type": rec["drain_type"],
            "sync_type": rec["sync_type"],
            "sync_event_name": rec.get("sync_event_name"),
            "sync_event_correlation": rec.get("sync_event_correlation"),
            "sync_event_dur": rec.get("sync_event_dur"),
            "cpu_during_gap": rec["cpu_during_gap"],
            "cpu_during_gap_detail": rec["cpu_during_gap_detail"],
            "dominant_op": rec["dominant_op"],
            "preceding_gpu_event": rec.get("preceding_gpu_event"),
            "following_gpu_event": rec.get("following_gpu_event"),
            "following_launch_name": rec.get("following_launch_name"),
            "following_gpu_uid": rec.get("following_gpu_uid"),
            "following_launch_uid": rec.get("following_launch_uid"),
            "sync_event_uid": rec.get("sync_event_uid"),
            "launch_to_exec_us": rec.get("launch_to_exec_us"),
            "kernel_prequeued": rec.get("kernel_prequeued"),
        })

    df_intervals = pd.DataFrame(interval_rows)
    interval_col_order = [
        "idle_id", "group", "start_us", "end_us", "duration_us", "drain_type", "sync_type",
        "sync_event_name", "sync_event_correlation", "sync_event_dur", "cpu_during_gap",
        "cpu_during_gap_detail", "dominant_op", "preceding_gpu_event", "following_gpu_event",
        "following_launch_name", "following_gpu_uid", "following_launch_uid", "sync_event_uid",
        "launch_to_exec_us", "kernel_prequeued",
    ]
    if not df_intervals.empty:
        df_intervals = df_intervals[interval_col_order]
        df_intervals = df_intervals.sort_values("duration_us", ascending=False).reset_index(drop=True)

    # --- Sheet 0: idle_overview (coarse summary) ---
    overview_rows = []
    if macro:
        total_stats = compute_stats(macro)
        total_stats.update({"drain_type": "ALL", "cpu_during_gap": "ALL", "pct_of_idle": 100.0})
        overview_rows.append(total_stats)

    coarse_groups = defaultdict(list)
    for rec in macro:
        coarse_groups[(rec["drain_type"], rec["cpu_during_gap"])].append(rec)

    sorted_coarse = sorted(coarse_groups.items(), key=lambda x: -sum(r["duration"] for r in x[1]))
    for (drain, cpu_gap), items in sorted_coarse:
        stats = compute_stats(items)
        stats["drain_type"] = drain
        stats["cpu_during_gap"] = cpu_gap
        stats["pct_of_idle"] = stats["total_time_ms"] / (total_macro_time / 1e3) * 100
        overview_rows.append(stats)

    overview_col_order = [
        "drain_type", "cpu_during_gap", "count",
        "total_time_ms", "pct_of_idle", "mean_us", "median_us",
        "min_us", "max_us",
    ]
    if gpu_busy_time_us is not None and macro:
        trace_span_us = float(gpu_busy_time_us) + float(total_macro_time)
        if trace_span_us > 0:
            gpu_util = 100.0 * float(gpu_busy_time_us) / trace_span_us
            for row in overview_rows:
                row_us = row["total_time_ms"] * 1e3
                row["pct_of_trace"] = 100.0 * row_us / trace_span_us
                row["gpu_utilization_pct"] = gpu_util
        else:
            for row in overview_rows:
                row["pct_of_trace"] = None
                row["gpu_utilization_pct"] = None
        overview_col_order = [
            "drain_type", "cpu_during_gap", "count",
            "total_time_ms", "pct_of_idle", "pct_of_trace", "gpu_utilization_pct",
            "mean_us", "median_us", "min_us", "max_us",
        ]

    df_overview = pd.DataFrame(overview_rows, columns=overview_col_order)

    return {
        "idle_overview": df_overview,
        "idle_summary": df_summary,
        "idle_intervals": df_intervals,
    }


def write_idle_excel(dict_dfs, output_path):
    """Write idle DataFrames from build_idle_dataframes to an Excel workbook."""
    import pandas as pd

    df_overview = dict_dfs["idle_overview"]
    df_summary = dict_dfs["idle_summary"]
    df_intervals = dict_dfs["idle_intervals"]

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df_overview.to_excel(writer, sheet_name="idle_overview", index=False)
        df_summary.to_excel(writer, sheet_name="idle_summary", index=False)
        df_intervals.to_excel(writer, sheet_name="idle_intervals", index=False)

    print(f"Wrote Excel report: {output_path}")
    print(f"  idle_overview: {len(df_overview)} rows")
    print(f"  idle_summary: {len(df_summary)} rows")
    print(f"  idle_intervals: {len(df_intervals)} rows")
