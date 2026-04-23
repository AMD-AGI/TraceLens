"""
Extension for generate_perf_report_pytorch that adds idle time classification sheets.

Usage:
    python -m TraceLens.Reporting.generate_perf_report_pytorch \
        --profile_json_path trace.json \
        --extension_file /path/to/idle_time_extension.py

Adds three sheets to the perf report Excel:
    - idle_overview: coarse summary grouped by (drain_type, cpu_during_gap), ~5-8 rows
    - idle_summary: grouped idle time breakdown by (drain_type, cpu_during_gap, dominant_op)
    - idle_intervals: per-interval detail with timestamps, sync info, and classification
"""
import numpy as np
import pandas as pd
from collections import defaultdict
from TraceLens.Reporting.classify_idle_time import classify_idle_intervals, assign_idle_ids


def get_additional_dataframes_extension(tree):
    """Called by generate_perf_report_pytorch via the extension mechanism."""
    classified = classify_idle_intervals(tree)
    assign_idle_ids(classified)

    macro = [r for r in classified if not r["label_noise"]]
    noise_intervals = [r for r in classified if r["label_noise"]]

    # --- idle_summary ---
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

    summary_rows = []
    if macro:
        total_stats = compute_stats(macro)
        total_stats.update({"drain_type": "ALL", "cpu_during_gap": "ALL",
                           "dominant_op": "ALL", "pct_of_idle": 100.0})
        summary_rows.append(total_stats)

    if noise_intervals:
        noise_stats = compute_stats(noise_intervals)
        noise_stats.update({"drain_type": "-", "cpu_during_gap": "noise",
                           "dominant_op": "-", "pct_of_idle": None})
        summary_rows.append(noise_stats)

    total_macro_time = sum(r["duration"] for r in macro) if macro else 1.0
    combo_groups = defaultdict(list)
    for rec in macro:
        key = (rec["drain_type"], rec["cpu_during_gap"], get_grouping_key(rec))
        combo_groups[key].append(rec)

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

    col_order = [
        "drain_type", "cpu_during_gap", "dominant_op", "count",
        "total_time_ms", "pct_of_idle", "cumulative_pct", "mean_us", "median_us",
        "std_us", "min_us", "max_us", "idle_ids",
    ]
    df_summary = pd.DataFrame(summary_rows, columns=col_order)
    combo_mask = ~df_summary["cpu_during_gap"].isin(["ALL", "noise"])
    df_summary.loc[combo_mask, "cumulative_pct"] = (
        df_summary.loc[combo_mask, "pct_of_idle"].cumsum()
    )

    # --- idle_intervals ---
    interval_rows = []
    for rec in macro:
        group = f"{rec['drain_type']} | {rec['cpu_during_gap']} | {get_grouping_key(rec)}"
        interval_rows.append({
            "idle_id": rec.get("idle_id"),
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
    if not df_intervals.empty:
        df_intervals = df_intervals.sort_values("duration_us", ascending=False).reset_index(drop=True)

    # --- idle_overview (coarse summary) ---
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
    df_overview = pd.DataFrame(overview_rows, columns=overview_col_order)

    return {
        "idle_overview": df_overview,
        "idle_summary": df_summary,
        "idle_intervals": df_intervals,
    }
