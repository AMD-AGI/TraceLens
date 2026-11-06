###############################################################################
# Copyright (c) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Idle sheets: idle_overview (how much, and why), idle_groups (where in the code), and
idle_intervals (each gap). ``group_id`` links idle_groups rows to their idle_intervals rows.
"""

from collections import defaultdict
from dataclasses import asdict

from .classify import PARTS, IdleThresholds, full_label, strip_addresses

# A group is kept if it holds at least FOLD_MIN_PCT of idle or ranks in the top
# FOLD_KEEP_TOP of its label; the rest fold into one "other (N groups)" row per label
# and drain_type.
FOLD_MIN_PCT = 1.0
FOLD_KEEP_TOP = 3
# A group's top_runtime is left empty when that call holds less than this share of the
# group's idle time: it is then incidental (attribute queries, module loads), not a cause.
TOP_RUNTIME_MIN_PCT = 10.0

INTERVAL_COLUMNS = [
    "idle_id",
    "group_id",
    "label",
    "sub_label",
    "case",
    "drain_type",
    "stream_switch",
    "start_us",
    "end_us",
    "duration_us",
    *PARTS,
    "detail",
    "covering_op",
    "top_op",
    "top_runtime",
    "launch_parent_op",
    "following_launch_name",
    "launch_tid",
    "launch_to_start_us",
    "launch_lead_us",
    "launch_call_x_ref",
    "other_thread_top_runtime",
    "other_thread_runtime_us",
    "preceding_gpu_event",
    "following_gpu_event",
    "preceding_stream",
    "following_stream",
    "sync_event_name",
    "sync_event_dur",
    "sync_gain_us",
    "sync_event_correlation",
    "preceding_gpu_uid",
    "following_gpu_uid",
    "following_launch_uid",
    "sync_event_uid",
]
CALL_STACK_COLUMNS = [
    "sync_entry_point",
    "sync_call_stack_full",
    "covering_call_stack_full",
]
PART_MS_COLUMNS = [p.replace("_us", "_ms") for p in PARTS]
OVERVIEW_COLUMNS = [
    "section",
    "row",
    "gaps",
    "total_ms",
    "pct_of_idle",
    "pct_of_trace",
    "sync_drain_ms",
    "sync_gain_ms",
    "starved_ms",
    "stream_switch_ms",
    *PART_MS_COLUMNS,
]


def group_key(rec):
    """The CPU op covering the gap on the launching thread, the same rule for every label."""
    if rec["case"] == "no_launch":
        return "(no launch)"
    return strip_addresses(rec["covering_op"]) or "(no CPU op)"


def _runtime_of(rec):
    if rec["sub_label"] == "other_thread":
        return rec["other_thread_top_runtime"], rec["other_thread_runtime_us"] or 0.0
    return rec["top_runtime"], rec["cpu_runtime_us"]


def _top_with_time(pairs):
    totals = defaultdict(float)
    for name, us in pairs:
        if name:
            totals[name] += us
    if not totals:
        return None, 0.0
    top = max(totals, key=totals.get)
    return top, totals[top]


def _top_by_time(pairs):
    return _top_with_time(pairs)[0]


def drain_row_name(rec):
    """Overview drain row: starved, or the sync call (and its call site, if known)."""
    if rec["drain_type"] != "sync_drain":
        return "starved"
    name = f"sync_drain: {rec['sync_event_name']}"
    site = rec.get("sync_entry_point")
    return f"{name} @ {site}" if site and site != "Not found" else name


def _pctile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))]


def _duration_stats(items):
    import numpy as np

    d = np.array([r["duration"] for r in items])
    return {
        "gaps": len(items),
        "total_ms": d.sum() / 1e3,
        "mean_us": d.mean(),
        "median_us": float(np.median(d)),
        "p90_us": _pctile(d.tolist(), 0.9),
        "max_us": d.max(),
    }


def _fold_groups(groups, pct):
    """Keep large groups and the top few per label; fold the rest per label and drain."""
    per_label = defaultdict(list)
    for key, items in groups.items():
        per_label[key[0]].append((key, items))
    kept = []
    for label, entries in per_label.items():
        kept += _fold_label(label, entries, pct)
    kept.sort(key=lambda x: -sum(r["duration"] for r in x[1]))
    return kept


def _fold_label(label, entries, pct):
    entries.sort(key=lambda x: -sum(r["duration"] for r in x[1]))
    kept, folded = [], defaultdict(list)
    for rank, (key, items) in enumerate(entries):
        if (
            rank < FOLD_KEEP_TOP
            or pct(sum(r["duration"] for r in items)) >= FOLD_MIN_PCT
        ):
            kept.append((key, items, None))
        else:
            folded[key[1]].append((key, items))
    for drain, group in folded.items():
        if len(group) == 1:
            kept.append((group[0][0], group[0][1], None))
            continue
        items = [r for _, its in group for r in its]
        top_keys = "; ".join(k[2] for k, _ in group[:3])
        kept.append(((label, drain, f"other ({len(group)} groups)"), items, top_keys))
    return kept


def build_idle_dataframes(
    classified, gpu_busy_time_us=None, thresholds=None, include_call_stack=False
):
    """Build the idle_overview, idle_groups and idle_intervals DataFrames.

    Args:
        classified: output of classify_idle_intervals, with idle_id assigned.
        gpu_busy_time_us: GPU busy time; if given, the overview reports GPU
            utilization and pct_of_trace, consistent with the gpu_timeline sheet.
        thresholds: IdleThresholds used for classification, echoed in the overview.
        include_call_stack: add sync_entry_point, sync_call_stack_full and
            covering_call_stack_full to idle_intervals, and split the overview drain
            rows by sync call site.

    idle_overview rows (``section``): ``total`` (utilization, all / noise / macro idle),
    ``label`` and ``sub_label``, ``drain`` (starved, and sync_drain per sync call), and
    ``threshold``. Macro, label, sub-label and drain rows carry one ``*_ms`` column per
    part, so each row shows what its gaps were made of.
    """
    import pandas as pd

    th = thresholds or IdleThresholds()
    macro = [r for r in classified if not r["label_noise"]]
    noise = [r for r in classified if r["label_noise"]]
    macro_us = sum(r["duration"] for r in macro)
    noise_us = sum(r["duration"] for r in noise)
    all_idle_us = macro_us + noise_us
    span_us = gpu_busy_time_us + all_idle_us if gpu_busy_time_us is not None else None

    def pct(us):
        return 100.0 * us / all_idle_us if all_idle_us else 0.0

    def pct_trace(us):
        return 100.0 * us / span_us if span_us else None

    def by_duration(d):
        return sorted(d.items(), key=lambda x: -sum(r["duration"] for r in x[1]))

    # --- idle_overview ---
    overview = []
    if classified:
        if span_us:
            overview.append(
                {
                    "section": "total",
                    "row": "gpu_utilization_pct",
                    "pct_of_trace": 100.0 * gpu_busy_time_us / span_us,
                }
            )
        for row, us, n in (
            ("ALL idle", all_idle_us, len(classified)),
            (f"noise (< {th.noise_us:g} us)", noise_us, len(noise)),
        ):
            overview.append(
                {
                    "section": "total",
                    "row": row,
                    "gaps": n,
                    "total_ms": us / 1e3,
                    "pct_of_idle": pct(us),
                    "pct_of_trace": pct_trace(us),
                }
            )

        def ms_where(items, cond):
            return sum(r["duration"] for r in items if cond(r)) / 1e3

        def matrix_row(section, name, items):
            us = sum(r["duration"] for r in items)
            row = {
                "section": section,
                "row": name,
                "gaps": len(items),
                "total_ms": us / 1e3,
                "pct_of_idle": pct(us),
                "pct_of_trace": pct_trace(us),
                "sync_drain_ms": ms_where(
                    items, lambda r: r["drain_type"] == "sync_drain"
                ),
                "starved_ms": ms_where(items, lambda r: r["drain_type"] == "starved"),
                "stream_switch_ms": ms_where(items, lambda r: r["stream_switch"]),
            }
            for p, col in zip(PARTS, PART_MS_COLUMNS):
                row[col] = sum(r[p] for r in items) / 1e3
            return row

        overview.append(matrix_row("total", f"macro (>= {th.noise_us:g} us)", macro))
        by_label, by_sub, by_drain = defaultdict(list), defaultdict(list), {}
        for r in macro:
            by_label[r["label"]].append(r)
            if r["sub_label"]:
                by_sub[full_label(r)].append(r)
            by_drain.setdefault(drain_row_name(r), []).append(r)
        for name, items in by_duration(by_label):
            overview.append(matrix_row("label", name, items))
            for sub, sub_items in by_duration(by_sub):
                if sub.startswith(name + "/"):
                    overview.append(matrix_row("sub_label", sub, sub_items))
        for name, items in by_duration(by_drain):
            row = matrix_row("drain", name, items)
            if name != "starved":
                row["sync_gain_ms"] = sum(r["sync_gain_us"] or 0.0 for r in items) / 1e3
            overview.append(row)
        for name, value in asdict(th).items():
            overview.append({"section": "threshold", "row": f"{name} = {value}"})
    df_overview = pd.DataFrame(overview, columns=OVERVIEW_COLUMNS)
    df_overview["gaps"] = df_overview["gaps"].astype("Int64")

    # --- idle_groups ---
    groups = defaultdict(list)
    for r in macro:
        groups[(full_label(r), r["drain_type"], group_key(r))].append(r)
    group_rows, group_of = [], {}
    cumulative = 0.0
    for gid, ((label, drain, key), items, top_keys) in enumerate(
        _fold_groups(groups, pct)
    ):
        stats = _duration_stats(items)
        group_us = sum(r["duration"] for r in items)
        group_pct = pct(group_us)
        cumulative += group_pct
        top, _, sub = label.partition("/")
        top_runtime, runtime_us = _top_with_time(_runtime_of(r) for r in items)
        if runtime_us < TOP_RUNTIME_MIN_PCT / 100.0 * group_us:
            top_runtime = None
        row = {
            "group_id": gid,
            "label": top,
            "sub_label": sub or None,
            "drain_type": drain,
            "covering_op": key,
            "top_runtime": top_runtime,
            "top_launch": _top_by_time(
                (r["following_launch_name"], r["duration"]) for r in items
            ),
            "top_next_gpu_op": _top_by_time(
                (r["following_gpu_event"], r["duration"]) for r in items
            ),
            **stats,
            "pct_of_idle": group_pct,
            "cumulative_pct": cumulative,
            "worst_idle_id": max(items, key=lambda r: r["duration"]).get("idle_id"),
            "folded_top_ops": top_keys,
        }
        for p, col in zip(PARTS, PART_MS_COLUMNS):
            row[col] = sum(r[p] for r in items) / 1e3
        group_rows.append(row)
        for r in items:
            group_of[id(r)] = gid
    df_groups = pd.DataFrame(group_rows)

    # --- idle_intervals ---
    columns = INTERVAL_COLUMNS + (CALL_STACK_COLUMNS if include_call_stack else [])
    interval_rows = []
    for r in sorted(macro, key=lambda r: -r["duration"]):
        row = {c: r.get(c) for c in columns}
        row.update(
            group_id=group_of[id(r)],
            start_us=r["start"],
            end_us=r["end"],
            duration_us=r["duration"],
        )
        interval_rows.append(row)
    df_intervals = pd.DataFrame(interval_rows, columns=columns)

    return {
        "idle_overview": df_overview,
        "idle_groups": df_groups,
        "idle_intervals": df_intervals,
    }


def write_idle_excel(dict_dfs, output_path):
    """Write the DataFrames from build_idle_dataframes to an Excel workbook."""
    import pandas as pd

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for name in ("idle_overview", "idle_groups", "idle_intervals"):
            dict_dfs[name].to_excel(writer, sheet_name=name, index=False)
