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

from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

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
    lca_ids = df["lowest_common_ancestor_id"].unique()

    rows = []
    for lca_id in lca_ids:
        t1 = trace1[trace1["lowest_common_ancestor_id"] == lca_id]
        t2 = trace2[trace2["lowest_common_ancestor_id"] == lca_id]

        cpu_ops_t1 = t1["cpu_op_name"].unique()
        if len(cpu_ops_t1) == 1:
            op_name = cpu_ops_t1[0]
        elif len(cpu_ops_t1) > 1:
            op_name = " | ".join(sorted(cpu_ops_t1))
        else:
            cpu_ops_t2 = t2["cpu_op_name"].unique()
            if len(cpu_ops_t2) == 1:
                op_name = cpu_ops_t2[0]
            elif len(cpu_ops_t2) > 1:
                op_name = " | ".join(sorted(cpu_ops_t2))
            else:
                op_name = "unknown"

        lca_names = df[df["lowest_common_ancestor_id"] == lca_id][
            "lowest_common_ancestor_name"
        ].unique()
        lca_name = lca_names[0] if len(lca_names) > 0 else ""

        src = t1 if not t1.empty else t2
        nn_module_stack = src["nn_module_stack"].iloc[0] if not src.empty else ""
        nn_module_parent = src["nn_module_parent"].iloc[0] if not src.empty else ""

        kernel_time_trace1 = t1["busy_time"].iloc[0] if not t1.empty else 0.0
        kernel_time_trace2 = t2["busy_time"].iloc[0] if not t2.empty else 0.0
        num_kernels_trace1 = len(t1)
        num_kernels_trace2 = len(t2)
        kernel_names_trace1 = t1["name"].tolist() if not t1.empty else []
        kernel_names_trace2 = t2["name"].tolist() if not t2.empty else []

        input_dims = src["Input Dims"].iloc[0] if not src.empty else ""
        input_type = src["Input type"].iloc[0] if not src.empty else ""
        input_strides = src["Input Strides"].iloc[0] if not src.empty else ""
        concrete_inputs = src["Concrete Inputs"].iloc[0] if not src.empty else ""

        row = {
            "name": op_name,
            "lowest_common_ancestor_name": lca_name,
            "lowest_common_ancestor_id": lca_id,
            "nn_module_stack": nn_module_stack,
            "nn_module_parent": nn_module_parent,
            "Input Dims": input_dims,
            "Input type": input_type,
            "Input Strides": input_strides,
            "Concrete Inputs": concrete_inputs,
            "kernel_time_trace1_us": kernel_time_trace1,
            "kernel_time_trace2_us": kernel_time_trace2,
            "num_kernels_trace1": num_kernels_trace1,
            "num_kernels_trace2": num_kernels_trace2,
            "kernel_names_trace1": kernel_names_trace1,
            "kernel_names_trace2": kernel_names_trace2,
        }

        if kernel_time_trace1 > 0:
            row["speedup (trace2/trace1)"] = kernel_time_trace2 / kernel_time_trace1
        else:
            row["speedup (trace2/trace1)"] = float("nan")
        row["delta_us (trace2 - trace1)"] = kernel_time_trace2 - kernel_time_trace1
        rows.append(row)

    df_summary = pd.DataFrame(rows)
    if df_summary.empty:
        return df_summary

    df_summary = df_summary.sort_values(
        by="kernel_time_trace1_us", ascending=False
    ).reset_index(drop=True)

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
    if df_unified_perf.empty or unified_perf_summary.empty:
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
    # Number of trace2 kernel invocations per LCA group. lca_count_trace2 reports
    # the count of the *dominant* mapped LCA (the one contributing the most trace2
    # kernel time) so it is comparable to trace1's op-invocation count -- rather
    # than the number of matched LCA groups, which is meaningless as an op count
    # for ops that span multiple LCAs (e.g. a custom all-reduce whose op owns both
    # a MemCpy and an AllReduce LCA).
    lca_trace2_count = trace2.groupby("lowest_common_ancestor_id").size()

    for lca_id, grp in trace1.groupby("lowest_common_ancestor_id"):
        t1_total = float(grp["busy_time"].iloc[0])

        t2_total = float(lca_trace2_time.get(lca_id, 0.0))
        has_trace2_match = bool(_trace2_gpu_op_uid_set_for_lca(trace2, lca_id))
        t2_count_this_lca = (
            int(lca_trace2_count.get(lca_id, 0)) if has_trace2_match else 0
        )

        names = grp["lowest_common_ancestor_name"].dropna()
        lca_display_name = str(names.iloc[0]) if not names.empty else None
        if lca_display_name == "":
            lca_display_name = None

        seen: Set[Any] = set()
        for _, row in grp.iterrows():
            key = _resolve_diff_row_to_key(row, uid_to_row_idx)
            if key is None or key in seen:
                continue
            seen.add(key)
            if key not in out:
                out[key] = {
                    "lca_ids": [lca_id],
                    "lca_names": [lca_display_name],
                    "lca_total_kernel_time_trace1_us": t1_total,
                    "lca_total_kernel_time_trace2_us": t2_total,
                    "lca_count_trace2": t2_count_this_lca,
                    "_dominant_t2_time": t2_total,
                }
            else:
                existing = out[key]
                if lca_id not in existing["lca_ids"]:
                    existing["lca_ids"].append(lca_id)
                    existing["lca_names"].append(lca_display_name)
                    existing["lca_total_kernel_time_trace1_us"] += t1_total
                    existing["lca_total_kernel_time_trace2_us"] += t2_total
                    if t2_total > existing["_dominant_t2_time"]:
                        existing["_dominant_t2_time"] = t2_total
                        existing["lca_count_trace2"] = t2_count_this_lca

    # Drop the private dominant-time helper so the output schema is unchanged.
    for meta in out.values():
        meta.pop("_dominant_t2_time", None)

    return out


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

    trace2_counts: List[float] = []
    lca_id_vals: List[Any] = []
    lca_name_vals: List[Any] = []
    lca_t1_vals: List[float] = []
    lca_t2_vals: List[float] = []

    for idx, row in df.iterrows():
        meta = lca_metadata.get(idx)

        cv = meta.get("lca_count_trace2") if meta else None
        trace2_counts.append(np.nan if cv is None else float(cv))

        if meta:
            ids = meta.get("lca_ids", [])
            names = meta.get("lca_names", [])
            lca_id_vals.append("; ".join(str(i) for i in ids) if ids else np.nan)
            lca_name_vals.append(
                "; ".join(str(n) for n in names if n is not None) if names else np.nan
            )
            t1m = meta.get("lca_total_kernel_time_trace1_us")
            t2m = meta.get("lca_total_kernel_time_trace2_us")
            lca_t1_vals.append(float(t1m) if t1m is not None else np.nan)
            lca_t2_vals.append(float(t2m) if t2m is not None else np.nan)
        else:
            lca_id_vals.append(np.nan)
            lca_name_vals.append(np.nan)
            lca_t1_vals.append(np.nan)
            lca_t2_vals.append(np.nan)

    col_idx = df.columns.get_loc(kernel_time_col)

    lca_t1 = pd.Series(lca_t1_vals, index=df.index, dtype=float)
    lca_t2 = pd.Series(lca_t2_vals, index=df.index, dtype=float)
    c2 = pd.Series(trace2_counts, index=df.index, dtype=float)

    insert_at = col_idx + 1

    def _ins(name: str, series) -> None:
        nonlocal insert_at
        df.insert(insert_at, name, series)
        insert_at += 1

    row_t1 = df[kernel_time_col].astype(float)
    _ins("speedup (trace2/trace1)", lca_t2 / lca_t1.replace(0, np.nan))
    _ins("delta_us (trace2 - trace1)", lca_t2 - row_t1)
    _ins(_TRACE2_OP_COUNT_COL, c2)
    _ins("lca_id", pd.Series(lca_id_vals, index=df.index, dtype=object))
    _ins("lca_name", pd.Series(lca_name_vals, index=df.index, dtype=object))
    _ins("lca_total_kernel_time_trace1_us", lca_t1)
    _ins("lca_total_kernel_time_trace2_us", lca_t2)

    return df


def aligned_trace2_summary_by_category(
    diff_stats_df: pd.DataFrame,
    df_unified_perf: pd.DataFrame,
    unified_perf_summary: pd.DataFrame,
    category_col: str = "op category",
) -> List[Dict[str, Any]]:
    """Roll trace2 kernel time up to **trace1** categories via the LCA alignment.

    The per-trace ``ops_summary_by_category`` of trace2 is computed from
    trace2's *own* categorization, which does not survive a cross-framework
    comparison (e.g. a vLLM custom all-reduce attributed to a compute-kernel
    category vs. an SGLang NCCL collective counted as exposed communication).
    Joining the two per-trace breakdowns by category *name* therefore produces
    spurious near-zero trace2 times for categories that exist on only one side.

    This helper instead attributes each diff_stats LCA group's trace2 time to a
    single trace1 category -- the category that owns the largest share of that
    LCA's trace1 kernel time -- so the resulting per-category totals are aligned
    with the same semantic blocks the Detailed Analysis cards already use.

    Returns a list of dicts matching the ``trace2_ops_summary_by_category``
    schema (``op category``, ``Count``, ``total_direct_kernel_time_ms``,
    ``Percentage (%)``, ``Cumulative Percentage (%)``), sorted by time desc.
    Returns ``[]`` when alignment data is unavailable (caller should fall back).
    """
    if (
        diff_stats_df is None
        or diff_stats_df.empty
        or df_unified_perf is None
        or df_unified_perf.empty
        or unified_perf_summary is None
        or unified_perf_summary.empty
        or category_col not in unified_perf_summary.columns
    ):
        return []

    uid_to_row_idx = _build_uid_to_row_idx(df_unified_perf, unified_perf_summary)
    if not uid_to_row_idx:
        return []

    row_idx_to_cat: Dict[Any, str] = {
        idx: str(unified_perf_summary.at[idx, category_col])
        for idx in unified_perf_summary.index
    }

    df = diff_stats_df.dropna(subset=["lowest_common_ancestor_id"])
    if df.empty:
        return []
    trace1 = df[df["source"] == "trace1"]
    trace2 = df[df["source"] == "trace2"]
    if trace1.empty:
        return []

    lca_trace2_time = trace2.groupby("lowest_common_ancestor_id")["busy_time"].first()
    lca_trace2_count = trace2.groupby("lowest_common_ancestor_id").size()

    cat_time_us: Dict[str, float] = defaultdict(float)
    cat_count: Dict[str, int] = defaultdict(int)

    for lca_id, grp in trace1.groupby("lowest_common_ancestor_id"):
        # Pick the dominant trace1 category for this LCA, weighting each mapped
        # summary row by its diff_stats kernel_time.
        cat_weight: Dict[str, float] = defaultdict(float)
        for _, row in grp.iterrows():
            key = _resolve_diff_row_to_key(row, uid_to_row_idx)
            if key is None:
                continue
            cat = row_idx_to_cat.get(key)
            if cat is None:
                continue
            try:
                kt = float(row.get("kernel_time", 0.0) or 0.0)
            except (TypeError, ValueError):
                kt = 0.0
            cat_weight[cat] += kt
        if not cat_weight:
            continue
        dominant_cat = max(cat_weight.items(), key=lambda kv: kv[1])[0]

        t2_us = float(lca_trace2_time.get(lca_id, 0.0) or 0.0)
        cat_time_us[dominant_cat] += t2_us
        cat_count[dominant_cat] += int(lca_trace2_count.get(lca_id, 0))

    if not cat_time_us:
        return []

    total_ms = sum(cat_time_us.values()) / 1000.0
    rows = sorted(cat_time_us.items(), key=lambda kv: kv[1], reverse=True)
    summary: List[Dict[str, Any]] = []
    cumulative = 0.0
    for cat, time_us in rows:
        time_ms = time_us / 1000.0
        pct = (time_ms / total_ms * 100.0) if total_ms > 0 else 0.0
        cumulative += pct
        summary.append(
            {
                "op category": cat,
                "Count": cat_count.get(cat, 0),
                "total_direct_kernel_time_ms": round(time_ms, 10),
                "Percentage (%)": round(pct, 10),
                "Cumulative Percentage (%)": round(cumulative, 10),
            }
        )
    return summary


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
