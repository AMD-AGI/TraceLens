###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""TraceDiff-based comparison for PyTorch perf reports (extension workflow).

Use with::

    TraceLens_generate_perf_report_pytorch \\
        --profile_json_path trace1.json \\
        --comparison_json_path trace2.json

This runs TraceDiff in memory, then adds **speedup**, **delta**,
**lca_count_trace2**, and LCA columns (``lca_id``, ``lca_name``,
``lca_total_kernel_time_trace1_us``, ``lca_total_kernel_time_trace2_us``)
to ``unified_perf_summary``. A ``diff_stats`` sheet is also included.

Matching uses ``gpu_op_uid`` values from ``df_unified_perf``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd

_KERNEL_TIME_COL_FOR_SPEEDUP_DELTA = "Kernel Time (µs)_sum"

_GROUPING_COLS = [
    "name",
    "op category",
    "process_name",
    "process_label",
    "thread_name",
    "Input Dims",
    "Input type",
    "Input Strides",
    "Concrete Inputs",
]


def tracediff_perf_summary_from_diff_stats(diff_stats_df: pd.DataFrame) -> pd.DataFrame:
    """Build the per-LCA summary table from ``diff_stats`` (no perf-sheet merge).

    Used by tests and tooling that need kernel-time totals per
    ``lowest_common_ancestor_id`` without running the full extension enrich.
    """
    if diff_stats_df.empty:
        return pd.DataFrame()

    df = diff_stats_df.copy()
    df = df.dropna(subset=["lowest_common_ancestor_id"])
    if df.empty:
        return pd.DataFrame()

    trace1 = df[df["source"] == "trace1"]
    trace2 = df[df["source"] == "trace2"]

    rows = []
    for lca_id in df["lowest_common_ancestor_id"].unique():
        lca_names = df[df["lowest_common_ancestor_id"] == lca_id][
            "lowest_common_ancestor_name"
        ].unique()
        rows.append(
            _lca_summary_row(
                lca_id,
                lca_names[0] if len(lca_names) > 0 else "",
                trace1[trace1["lowest_common_ancestor_id"] == lca_id],
                trace2[trace2["lowest_common_ancestor_id"] == lca_id],
            )
        )

    df_summary = pd.DataFrame(rows)
    if df_summary.empty:
        return df_summary

    df_summary = df_summary.sort_values(
        by="kernel_time_trace1_us", ascending=False
    ).reset_index(drop=True)
    return _add_total_pct_columns(df_summary)


def _lca_op_name(t1: pd.DataFrame, t2: pd.DataFrame) -> str:
    """CPU op name for one LCA; several names are joined, none is "unknown"."""
    for frame in (t1, t2):
        names = frame["cpu_op_name"].unique()
        if len(names) == 1:
            return names[0]
        if len(names) > 1:
            return " | ".join(sorted(names))
    return "unknown"


def _first_value(frame: pd.DataFrame, col: str) -> Any:
    """First value of a column, or empty string when the frame has no rows."""
    return frame[col].iloc[0] if not frame.empty else ""


def _lca_summary_row(
    lca_id: Any, lca_name: Any, t1: pd.DataFrame, t2: pd.DataFrame
) -> Dict[str, Any]:
    """One per-LCA summary row carrying both traces' kernel times."""
    src = t1 if not t1.empty else t2
    time1 = t1["busy_time"].iloc[0] if not t1.empty else 0.0
    time2 = t2["busy_time"].iloc[0] if not t2.empty else 0.0
    return {
        "name": _lca_op_name(t1, t2),
        "lowest_common_ancestor_name": lca_name,
        "lowest_common_ancestor_id": lca_id,
        "nn_module_stack": _first_value(src, "nn_module_stack"),
        "nn_module_parent": _first_value(src, "nn_module_parent"),
        "Input Dims": _first_value(src, "Input Dims"),
        "Input type": _first_value(src, "Input type"),
        "Input Strides": _first_value(src, "Input Strides"),
        "Concrete Inputs": _first_value(src, "Concrete Inputs"),
        "kernel_time_trace1_us": time1,
        "kernel_time_trace2_us": time2,
        "num_kernels_trace1": len(t1),
        "num_kernels_trace2": len(t2),
        "kernel_names_trace1": t1["name"].tolist() if not t1.empty else [],
        "kernel_names_trace2": t2["name"].tolist() if not t2.empty else [],
        "speedup (trace2/trace1)": (time2 / time1 if time1 > 0 else float("nan")),
        "delta_us (trace2 - trace1)": time2 - time1,
    }


def _add_total_pct_columns(df_summary: pd.DataFrame) -> pd.DataFrame:
    """Share-of-total columns, added only for a trace that contributed time."""
    total_trace1 = df_summary["kernel_time_trace1_us"].sum()
    total_trace2 = df_summary["kernel_time_trace2_us"].sum()
    if total_trace1 > 0:
        df_summary["pct_of_trace1_total (%)"] = (
            df_summary["kernel_time_trace1_us"] / total_trace1
        ) * 100
        df_summary["cumulative_pct_trace1 (%)"] = df_summary[
            "pct_of_trace1_total (%)"
        ].cumsum()
    if total_trace2 > 0:
        df_summary["pct_of_trace2_total (%)"] = (
            df_summary["kernel_time_trace2_us"] / total_trace2
        ) * 100
    return df_summary


def _build_uid_to_row_idx(
    df_unified_perf: pd.DataFrame,
    unified_perf_summary: pd.DataFrame,
) -> Dict[Any, Any]:
    """Map every gpu_op_uid in df_unified_perf to its unified_perf_summary row index.

    Uses the same grouping columns as summarize_df_unified_perf_table so that
    each pre-summary row maps to the summary row it was aggregated into.  This
    replaces the old approach of mining gpu_op_uids back out of
    kernel_details_summary, which required _summarize_kernel_stats to embed
    internal join state in summary output.
    """
    if df_unified_perf is None or df_unified_perf.empty or unified_perf_summary.empty:
        return {}
    if "kernel_details" not in df_unified_perf.columns:
        return {}

    # Build group_key → row index from unified_perf_summary.
    present_grouping = [c for c in _GROUPING_COLS if c in unified_perf_summary.columns]
    group_key_to_row_idx: Dict[tuple, Any] = {}
    for idx, row in unified_perf_summary.iterrows():
        key = tuple(str(row.get(c, "")) for c in present_grouping)
        group_key_to_row_idx[key] = idx

    # Build gpu_op_uid → row index via df_unified_perf grouping key.
    present_pre = [c for c in _GROUPING_COLS if c in df_unified_perf.columns]
    uid_to_row_idx: Dict[Any, Any] = {}
    for _, row in df_unified_perf.iterrows():
        key = tuple(str(row.get(c, "")) for c in present_pre)
        summary_idx = group_key_to_row_idx.get(key)
        if summary_idx is None:
            continue
        kd = row.get("kernel_details")
        if not isinstance(kd, list):
            continue
        for entry in kd:
            if not isinstance(entry, dict):
                continue
            uid = entry.get("gpu_op_uid")
            if uid is not None and not (isinstance(uid, float) and np.isnan(uid)):
                uid_to_row_idx[uid] = summary_idx

    return uid_to_row_idx


def _trace2_gpu_op_uid_set_for_lca(trace2: pd.DataFrame, lca_id) -> Set[Any]:
    """Distinct gpu_op_uid values for trace2 diff_stats rows under lca_id."""
    sub = trace2[trace2["lowest_common_ancestor_id"] == lca_id]
    if sub.empty or "gpu_op_uid" not in sub.columns:
        return set()
    return set(sub["gpu_op_uid"].dropna().tolist())


def _resolve_diff_row_to_key(row, uid_to_row_idx: Dict[Any, Any]) -> Optional[Any]:
    """Row index for a diff_stats row via its gpu_op_uid."""
    if not uid_to_row_idx or "gpu_op_uid" not in row.index:
        return None
    gpu_uid = row.get("gpu_op_uid")
    if gpu_uid is None or (isinstance(gpu_uid, float) and pd.isna(gpu_uid)):
        return None
    return uid_to_row_idx.get(gpu_uid)


def _build_lca_metadata(
    diff_stats_df: pd.DataFrame,
    uid_to_row_idx: Dict[Any, Any],
) -> Dict[Any, Dict[str, Any]]:
    """Map unified_perf_summary row index → LCA metadata from diff_stats.

    Each mapped row carries lists of lca_ids and lca_names (one entry per LCA
    group that maps to this summary row), plus accumulated
    lca_total_kernel_time_trace1_us and lca_total_kernel_time_trace2_us.
    """
    out: Dict[Any, Dict[str, Any]] = {}
    if diff_stats_df.empty or not uid_to_row_idx:
        return out

    df = diff_stats_df.dropna(subset=["lowest_common_ancestor_id"])
    if df.empty:
        return out

    trace1 = df[df["source"] == "trace1"]
    trace2 = df[df["source"] == "trace2"]
    if trace1.empty:
        return out

    lca_trace2_time = trace2.groupby("lowest_common_ancestor_id")["busy_time"].first()

    for lca_id, grp in trace1.groupby("lowest_common_ancestor_id"):
        names = grp["lowest_common_ancestor_name"].dropna()
        lca_display_name = str(names.iloc[0]) if not names.empty else None
        entry = {
            "lca_ids": [lca_id],
            "lca_names": [lca_display_name or None],
            "lca_total_kernel_time_trace1_us": float(grp["busy_time"].iloc[0]),
            "lca_total_kernel_time_trace2_us": float(lca_trace2_time.get(lca_id, 0.0)),
            "lca_count_trace2": (
                1 if _trace2_gpu_op_uid_set_for_lca(trace2, lca_id) else 0
            ),
        }
        _accumulate_lca_rows(out, grp, uid_to_row_idx, lca_id, entry)

    return out


def _accumulate_lca_rows(
    out: Dict[Any, Dict[str, Any]],
    grp: pd.DataFrame,
    uid_to_row_idx: Dict[Any, Any],
    lca_id: Any,
    entry: Dict[str, Any],
) -> None:
    """Fold one LCA group's rows into the per-summary-row metadata."""
    seen: Set[Any] = set()
    for _, row in grp.iterrows():
        key = _resolve_diff_row_to_key(row, uid_to_row_idx)
        if key is None or key in seen:
            continue
        seen.add(key)
        existing = out.get(key)
        if existing is None:
            out[key] = dict(
                entry,
                lca_ids=list(entry["lca_ids"]),
                lca_names=list(entry["lca_names"]),
            )
        elif lca_id not in existing["lca_ids"]:
            existing["lca_ids"].append(lca_id)
            existing["lca_names"].append(entry["lca_names"][0])
            for key_name in (
                "lca_total_kernel_time_trace1_us",
                "lca_total_kernel_time_trace2_us",
                "lca_count_trace2",
            ):
                existing[key_name] += entry[key_name]


_TRACE2_OP_COUNT_COL = "lca_count_trace2"


def _enrich_sheet_with_trace2(
    df_sheet: pd.DataFrame,
    kernel_time_col: str,
    *,
    lca_metadata: Optional[Dict[Any, Dict[str, Any]]] = None,
) -> pd.DataFrame:
    """Add LCA columns, speedup, delta, and trace2 op counts to unified_perf_summary.

    Speedup is computed from LCA-level aggregates
    (lca_total_kernel_time_trace2_us / lca_total_kernel_time_trace1_us).
    lca_id and lca_name are semicolon-separated when a row maps to multiple
    LCA groups.
    """
    if df_sheet.empty or not lca_metadata:
        return df_sheet

    df = df_sheet.copy()
    counts, ids, names, t1_vals, t2_vals = _lca_columns(df, lca_metadata)

    lca_t1 = pd.Series(t1_vals, index=df.index, dtype=float)
    lca_t2 = pd.Series(t2_vals, index=df.index, dtype=float)
    row_t1 = df[kernel_time_col].astype(float)

    columns = (
        ("speedup (trace2/trace1)", lca_t2 / lca_t1.replace(0, np.nan)),
        ("delta_us (trace2 - trace1)", lca_t2 - row_t1),
        (_TRACE2_OP_COUNT_COL, pd.Series(counts, index=df.index, dtype=float)),
        ("lca_id", pd.Series(ids, index=df.index, dtype=object)),
        ("lca_name", pd.Series(names, index=df.index, dtype=object)),
        ("lca_total_kernel_time_trace1_us", lca_t1),
        ("lca_total_kernel_time_trace2_us", lca_t2),
    )
    col_idx = df.columns.get_loc(kernel_time_col)
    for offset, (name, series) in enumerate(columns):
        df.insert(col_idx + 1 + offset, name, series)

    return df


def _lca_columns(
    df: pd.DataFrame, lca_metadata: Dict[Any, Dict[str, Any]]
) -> tuple[List[Any], List[Any], List[Any], List[float], List[float]]:
    """Per-row LCA counts, ids, names and kernel-time totals, in sheet order."""
    counts: List[float] = []
    ids: List[Any] = []
    names: List[Any] = []
    t1_vals: List[float] = []
    t2_vals: List[float] = []
    for idx in df.index:
        meta = lca_metadata.get(idx)
        if not meta:
            for acc in (counts, ids, names, t1_vals, t2_vals):
                acc.append(np.nan)
            continue
        count = meta.get("lca_count_trace2")
        counts.append(np.nan if count is None else float(count))
        lca_ids = meta.get("lca_ids", [])
        lca_names = meta.get("lca_names", [])
        ids.append("; ".join(str(i) for i in lca_ids) if lca_ids else np.nan)
        names.append(
            "; ".join(str(n) for n in lca_names if n is not None)
            if lca_names
            else np.nan
        )
        total1 = meta.get("lca_total_kernel_time_trace1_us")
        total2 = meta.get("lca_total_kernel_time_trace2_us")
        t1_vals.append(float(total1) if total1 is not None else np.nan)
        t2_vals.append(float(total2) if total2 is not None else np.nan)
    return counts, ids, names, t1_vals, t2_vals


def enrich_perf_report_dict_inplace(
    perf_dfs: Dict[str, pd.DataFrame],
    diff_stats_df: pd.DataFrame,
    df_unified_perf: Optional[pd.DataFrame] = None,
) -> Dict[str, pd.DataFrame]:
    """Enrich unified_perf_summary with TraceDiff comparison columns.

    When df_unified_perf is provided (the pre-summary DataFrame from
    build_df_unified_perf_table), the uid→row mapping is built from its
    kernel_details column where each concrete GPU event has a single
    gpu_op_uid.
    """
    if diff_stats_df.empty:
        return perf_dfs

    ups = perf_dfs.get("unified_perf_summary")
    ups_df = ups if isinstance(ups, pd.DataFrame) else pd.DataFrame()

    uid_to_row_idx = _build_uid_to_row_idx(df_unified_perf, ups_df)

    if not uid_to_row_idx:
        return perf_dfs

    kt_col = _KERNEL_TIME_COL_FOR_SPEEDUP_DELTA
    if kt_col not in ups_df.columns:
        return perf_dfs

    lca_metadata = _build_lca_metadata(diff_stats_df, uid_to_row_idx)

    working = {k: v.copy() for k, v in perf_dfs.items()}
    working["unified_perf_summary"] = _enrich_sheet_with_trace2(
        ups_df,
        kt_col,
        lca_metadata=lca_metadata,
    )

    print(
        "[TraceDiff] Added speedup, delta, lca_count_trace2, "
        "and LCA columns (lca_id, lca_name, "
        "lca_total_kernel_time_trace1_us, lca_total_kernel_time_trace2_us) "
        "to unified_perf_summary; added diff_stats sheet."
    )
    return working
