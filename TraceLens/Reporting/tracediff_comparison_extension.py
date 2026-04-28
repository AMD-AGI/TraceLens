###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""TraceDiff-based comparison for PyTorch perf reports (extension workflow).

Use with::

    TraceLens_generate_perf_report_pytorch \\
        --profile_json_path trace1.json \\
        --extension-file <path/to/this/file.py> \\
        --extension-args /path/to/trace2.json

This runs TraceDiff in memory, then adds **speedup**, **delta**,
``Kernel Time (µs)_trace2_sum``, and **operation_count_trace2** (count of
distinct ``gpu_op_uid`` values on trace2 ``diff_stats`` rows aligned to the
same perf row) to ``unified_perf_summary`` (beside ``Kernel Time (µs)_sum`` /
``operation_count``). Op-category workbook tabs (``GEMM``, ``CONV_bwd``,
etc.) have their ``Kernel Time (µs)_sum`` consolidated when multi-op LCAs
are rolled up. A ``diff_stats`` sheet (TraceDiff per-kernel diff rows) is
included whenever ``diff_stats_df`` is non-empty; pass ``debug`` as an extra
extension arg for ``debug_lca_ids`` on ``unified_perf_summary`` only.

Matching is **gpu_op_uid only**. Every row in ``unified_perf_summary`` and in
the op-category sheets carries a ``kernel_details_summary`` / ``kernel_details``
list of kernels with ``gpu_op_uid``. Each diff_stats row carries the
``gpu_op_uid`` of its kernel. Rows are aligned by mapping every gpu_op_uid in
a perf row to a canonical uid (the minimum uid in that row), so all sheets
sharing the same underlying kernels resolve to the same key. There is no
fallback to CPU UID tree walks or to (name, args) string matching.
"""

from __future__ import annotations

import ast
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from TraceLens import TraceDiff, TreePerfAnalyzer

# Sheets enriched with speedup/delta must use this column (unified + op-category tabs).
_KERNEL_TIME_COL_FOR_SPEEDUP_DELTA = "Kernel Time (µs)_sum"

_KERNEL_TIME_SUM_COLS = [
    _KERNEL_TIME_COL_FOR_SPEEDUP_DELTA,
    "total_direct_kernel_time_sum",
]

_KERNEL_DETAIL_COLS = ("kernel_details_summary", "kernel_details")


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

        if not t1.empty and "busy_time" in t1.columns and t1["busy_time"].notna().any():
            kernel_time_trace1 = t1["busy_time"].iloc[0]
        else:
            kernel_time_trace1 = t1["kernel_time"].sum() if not t1.empty else 0.0
        if not t2.empty and "busy_time" in t2.columns and t2["busy_time"].notna().any():
            kernel_time_trace2 = t2["busy_time"].iloc[0]
        else:
            kernel_time_trace2 = t2["kernel_time"].sum() if not t2.empty else 0.0
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


def _trace2_gpu_op_uid_set_for_lca(trace2: pd.DataFrame, lca_id) -> Set[Any]:
    """Distinct ``gpu_op_uid`` values for trace2 ``diff_stats`` rows under ``lca_id``."""
    sub = trace2[trace2["lowest_common_ancestor_id"] == lca_id]
    if sub.empty or "gpu_op_uid" not in sub.columns:
        return set()
    return set(sub["gpu_op_uid"].dropna().tolist())


def _parse_kernel_details(kds):
    """Normalize ``kernel_details`` / ``kernel_details_summary`` into a list of dicts.

    Returns an empty list when the value is missing, unparseable, or the wrong type.
    Accepts both an already-parsed list (op-category sheets write it that way) and
    the string form that Excel/CSV round-trips yield for ``kernel_details_summary``.
    """
    if kds is None:
        return []
    if isinstance(kds, float) and pd.isna(kds):
        return []
    if isinstance(kds, list):
        return kds
    if isinstance(kds, str):
        try:
            parsed = ast.literal_eval(
                re.sub(r"np\.(?:float|int)\d*\((.*?)\)", r"\1", kds)
            )
        except (ValueError, SyntaxError):
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def _iter_row_gpu_op_uids(row) -> List[Any]:
    """All ``gpu_op_uid`` values found in a perf-row's kernel_details column(s)."""
    uids: List[Any] = []
    for col in _KERNEL_DETAIL_COLS:
        if col not in row.index:
            continue
        for kd in _parse_kernel_details(row.get(col)):
            if not isinstance(kd, dict):
                continue
            # Prefer the list of all-instance uids added by the tree_perf fix;
            # fall back to the singular uid (template instance only).
            multi = kd.get("gpu_op_uids")
            if isinstance(multi, list) and multi:
                uids.extend(u for u in multi if u is not None)
            else:
                uid = kd.get("gpu_op_uid")
                if uid is not None and not (isinstance(uid, float) and pd.isna(uid)):
                    uids.append(uid)
        if uids:
            break
    return uids


def _row_canonical_gpu_uid(row) -> Optional[Any]:
    """Stable per-row key: minimum ``gpu_op_uid`` across the row's kernel_details.

    Minimum (rather than first) makes the key order-independent so op-category
    sheets and ``unified_perf_summary`` resolve to the same key for rows that
    group the same set of kernels.
    """
    uids = _iter_row_gpu_op_uids(row)
    return min(uids) if uids else None


def _build_gpu_uid_to_canonical(ups: pd.DataFrame) -> Dict[Any, Any]:
    """Map every ``gpu_op_uid`` in ``unified_perf_summary`` to its row's canonical uid.

    This is the only index used for diff_stats → perf-row matching.
    """
    mapping: Dict[Any, Any] = {}
    if ups is None or ups.empty:
        return mapping
    if not any(c in ups.columns for c in _KERNEL_DETAIL_COLS):
        return mapping
    for _, row in ups.iterrows():
        uids = _iter_row_gpu_op_uids(row)
        if not uids:
            continue
        canonical = min(uids)
        for u in uids:
            mapping[u] = canonical
    return mapping


def _resolve_diff_row_to_key(row, gpu_uid_to_canonical: Dict[Any, Any]) -> Optional[Any]:
    """Canonical key for a diff_stats row via its ``gpu_op_uid``."""
    if not gpu_uid_to_canonical or "gpu_op_uid" not in row.index:
        return None
    gpu_uid = row.get("gpu_op_uid")
    if gpu_uid is None or (isinstance(gpu_uid, float) and pd.isna(gpu_uid)):
        return None
    return gpu_uid_to_canonical.get(gpu_uid)


def _build_lca_consolidation(
    diff_stats_df: pd.DataFrame,
    gpu_uid_to_canonical: Optional[Dict[Any, Any]] = None,
) -> Tuple[
    Dict[Any, float],
    Dict[Any, float],
    Dict[Any, float],
    Dict[Any, List],
    Dict[Any, int],
]:
    empty = ({}, {}, {}, {}, {})
    if diff_stats_df.empty:
        return empty
    if not gpu_uid_to_canonical:
        return empty

    df = diff_stats_df.dropna(subset=["lowest_common_ancestor_id"])
    if df.empty:
        return empty

    t1 = df[df["source"] == "trace1"]
    t2 = df[df["source"] == "trace2"]
    if t1.empty:
        return empty

    lca_ops = t1.groupby("lowest_common_ancestor_id")["cpu_op_name"].nunique()
    multi_lca_ids = set(lca_ops[lca_ops > 1].index)
    if not multi_lca_ids:
        return empty

    _has_busy = "busy_time" in t2.columns and t2["busy_time"].notna().any()
    if _has_busy:
        lca_t2_time = t2.groupby("lowest_common_ancestor_id")["busy_time"].first()
    else:
        lca_t2_time = t2.groupby("lowest_common_ancestor_id")["kernel_time"].sum()

    time_additions: Dict[Any, float] = {}
    time_subtractions: Dict[Any, float] = {}
    dominant_trace2_map: Dict[Any, float] = {}
    dominant_trace2_uid_sets: Dict[Any, Set[Any]] = {}
    dominant_lca_ids: Dict[Any, List] = {}

    for lca_id in multi_lca_ids:
        grp = t1[t1["lowest_common_ancestor_id"] == lca_id]
        op_times = grp.groupby("cpu_op_name")["kernel_time"].sum()
        dominant_op_name = op_times.idxmax()

        dominant_rows = grp[grp["cpu_op_name"] == dominant_op_name]
        dominant_key = _resolve_diff_row_to_key(
            dominant_rows.iloc[0], gpu_uid_to_canonical
        )
        if dominant_key is None:
            continue

        t2_time = float(lca_t2_time.get(lca_id, 0.0))
        t2_uid_set = _trace2_gpu_op_uid_set_for_lca(t2, lca_id)
        dominant_trace2_map[dominant_key] = (
            dominant_trace2_map.get(dominant_key, 0.0) + t2_time
        )
        dominant_trace2_uid_sets[dominant_key] = dominant_trace2_uid_sets.get(
            dominant_key, set()
        ).union(t2_uid_set)
        dominant_lca_ids.setdefault(dominant_key, []).append(lca_id)

        for op_name, op_total_time in op_times.items():
            if op_name == dominant_op_name:
                continue
            nd_rows = grp[grp["cpu_op_name"] == op_name]
            nd_key = _resolve_diff_row_to_key(nd_rows.iloc[0], gpu_uid_to_canonical)
            if nd_key is None:
                continue
            time_subtractions[nd_key] = (
                time_subtractions.get(nd_key, 0.0) + op_total_time
            )
            time_additions[dominant_key] = (
                time_additions.get(dominant_key, 0.0) + op_total_time
            )

    dominant_trace2_count = {k: len(v) for k, v in dominant_trace2_uid_sets.items()}

    return (
        time_additions,
        time_subtractions,
        dominant_trace2_map,
        dominant_lca_ids,
        dominant_trace2_count,
    )


def _apply_lca_consolidation(
    perf_dfs: Dict[str, pd.DataFrame],
    time_additions: Dict[Any, float],
    time_subtractions: Dict[Any, float],
) -> Dict[str, pd.DataFrame]:
    if not time_additions and not time_subtractions:
        return perf_dfs

    result: Dict[str, pd.DataFrame] = {}
    for sheet_name, df_sheet in perf_dfs.items():
        if "name" not in df_sheet.columns:
            result[sheet_name] = df_sheet
            continue

        kt_col = None
        for c in _KERNEL_TIME_SUM_COLS:
            if c in df_sheet.columns:
                kt_col = c
                break
        if kt_col is None:
            result[sheet_name] = df_sheet
            continue
        if not any(c in df_sheet.columns for c in _KERNEL_DETAIL_COLS):
            result[sheet_name] = df_sheet
            continue

        df = df_sheet.copy()
        rows_to_drop: List[int] = []

        for idx, row in df.iterrows():
            key = _row_canonical_gpu_uid(row)
            if key is None:
                continue

            if key in time_additions:
                df.at[idx, kt_col] = df.at[idx, kt_col] + time_additions[key]

            if key in time_subtractions:
                new_time = df.at[idx, kt_col] - time_subtractions[key]
                if new_time <= 0:
                    rows_to_drop.append(idx)
                else:
                    df.at[idx, kt_col] = new_time

        if rows_to_drop:
            df = df.drop(rows_to_drop).reset_index(drop=True)

        result[sheet_name] = df

    return result


def _build_trace2_time_lookup(
    diff_stats_df: pd.DataFrame,
    gpu_uid_to_canonical: Optional[Dict[Any, Any]] = None,
) -> Tuple[Dict[Any, float], Dict[Any, List], Dict[Any, int]]:
    if diff_stats_df.empty or not gpu_uid_to_canonical:
        return {}, {}, {}

    df = diff_stats_df.dropna(subset=["lowest_common_ancestor_id"])
    if df.empty:
        return {}, {}, {}

    trace1 = df[df["source"] == "trace1"]
    trace2 = df[df["source"] == "trace2"]

    if trace1.empty:
        return {}, {}, {}

    _has_busy = "busy_time" in trace2.columns and trace2["busy_time"].notna().any()
    if _has_busy:
        lca_trace2_time = trace2.groupby("lowest_common_ancestor_id")[
            "busy_time"
        ].first()
    else:
        lca_trace2_time = trace2.groupby("lowest_common_ancestor_id")[
            "kernel_time"
        ].sum()

    lca_groups = trace1.groupby("lowest_common_ancestor_id")

    lookup: Dict[Any, float] = {}
    lca_ids_lookup: Dict[Any, List] = {}
    lookup_uid_sets: Dict[Any, Set[Any]] = {}
    for lca_id, grp in lca_groups:
        t2_time = float(lca_trace2_time.get(lca_id, 0.0))
        t2_uid_set = _trace2_gpu_op_uid_set_for_lca(trace2, lca_id)

        unique_ops = grp["cpu_op_name"].unique()
        if len(unique_ops) > 1:
            op_times = grp.groupby("cpu_op_name")["kernel_time"].sum()
            dominant_rows = grp[grp["cpu_op_name"] == op_times.idxmax()]
            key_row = dominant_rows.iloc[0]
        else:
            key_row = grp.iloc[0]

        key = _resolve_diff_row_to_key(key_row, gpu_uid_to_canonical)
        if key is None:
            continue
        lookup[key] = lookup.get(key, 0.0) + t2_time
        lookup_uid_sets[key] = lookup_uid_sets.get(key, set()).union(t2_uid_set)
        lca_ids_lookup.setdefault(key, []).append(lca_id)

    lookup_count = {k: len(v) for k, v in lookup_uid_sets.items()}
    return lookup, lca_ids_lookup, lookup_count


_TRACE2_KERNEL_TIME_COL = "Kernel Time (µs)_trace2_sum"
_TRACE2_OP_COUNT_COL = "operation_count_trace2"


def _enrich_sheet_with_trace2(
    df_sheet: pd.DataFrame,
    lookup: Dict[Any, float],
    kernel_time_col: str,
    *,
    lca_ids_lookup: Optional[Dict[Any, List]] = None,
    count_lookup: Optional[Dict[Any, int]] = None,
    debug: bool = False,
) -> pd.DataFrame:
    """Add speedup, delta, trace2 kernel time, and trace2 distinct gpu_op_uid count."""
    if df_sheet.empty or not lookup:
        return df_sheet

    df = df_sheet.copy()
    clookup = count_lookup if count_lookup is not None else {}

    trace2_times = []
    trace2_counts: List[float] = []
    lca_id_lists: List[str] = []
    need_lca_debug = debug and lca_ids_lookup is not None
    for _, row in df.iterrows():
        key = _row_canonical_gpu_uid(row)
        trace2_times.append(lookup.get(key, np.nan) if key is not None else np.nan)
        cv = clookup.get(key) if key is not None else None
        trace2_counts.append(np.nan if cv is None else float(cv))
        if need_lca_debug:
            lca_id_lists.append(
                str(lca_ids_lookup.get(key, []) if key is not None else [])
            )

    col_idx = df.columns.get_loc(kernel_time_col)
    t1 = df[kernel_time_col]
    t2 = pd.Series(trace2_times, index=df.index)
    c2 = pd.Series(trace2_counts, index=df.index)

    df.insert(
        col_idx + 1,
        "speedup (trace2/trace1)",
        t2 / t1.replace(0, np.nan),
    )
    df.insert(
        col_idx + 2,
        "delta_us (trace2 - trace1)",
        t2 - t1,
    )
    df.insert(
        col_idx + 3,
        _TRACE2_KERNEL_TIME_COL,
        t2,
    )
    df.insert(
        col_idx + 4,
        _TRACE2_OP_COUNT_COL,
        c2,
    )
    if debug:
        df.insert(
            col_idx + 5,
            "debug_lca_ids",
            lca_id_lists if lca_id_lists else "[]",
        )

    return df


def _merge_dominant_into_lookup(
    diff_stats_lookup: Dict[Any, float],
    dominant_t2_map: Dict[Any, float],
    diff_stats_lca_ids: Optional[Dict[Any, List]] = None,
    dominant_lca_ids: Optional[Dict[Any, List]] = None,
    diff_count_lookup: Optional[Dict[Any, int]] = None,
    dominant_t2_count: Optional[Dict[Any, int]] = None,
) -> Tuple[Dict[Any, float], Dict[Any, List], Dict[Any, int]]:
    merged = dict(diff_stats_lookup)
    merged_lca: Dict[Any, List] = dict(diff_stats_lca_ids or {})
    merged_count: Dict[Any, int] = dict(diff_count_lookup or {})
    for key, val in dominant_t2_map.items():
        if key not in merged:
            merged[key] = val
            if dominant_lca_ids and key in dominant_lca_ids:
                merged_lca[key] = dominant_lca_ids[key]
            if dominant_t2_count and key in dominant_t2_count:
                merged_count[key] = int(dominant_t2_count[key])
    return merged, merged_lca, merged_count


def _enrich_consolidated_perf_sheets(
    consolidated_t1: Dict[str, pd.DataFrame],
    lookup: Dict[Any, float],
    *,
    lca_ids_lookup: Optional[Dict[Any, List]] = None,
    count_lookup: Optional[Dict[Any, int]] = None,
    debug: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Add speedup, delta, trace2 time, and ``operation_count_trace2`` to ``unified_perf_summary``."""
    result: Dict[str, pd.DataFrame] = {}
    kt_col = _KERNEL_TIME_COL_FOR_SPEEDUP_DELTA
    for sheet_name, df_sheet in consolidated_t1.items():
        if sheet_name != "unified_perf_summary":
            result[sheet_name] = df_sheet
            continue
        if "name" not in df_sheet.columns or not lookup:
            result[sheet_name] = df_sheet
            continue
        if kt_col not in df_sheet.columns:
            result[sheet_name] = df_sheet
            continue
        result[sheet_name] = _enrich_sheet_with_trace2(
            df_sheet,
            lookup,
            kt_col,
            lca_ids_lookup=lca_ids_lookup,
            count_lookup=count_lookup,
            debug=debug,
        )
    return result


def enrich_perf_report_dict_inplace(
    perf_dfs: Dict[str, pd.DataFrame],
    diff_stats_df: pd.DataFrame,
    trace1_tree=None,  # kept for backward-compat; no longer used
    debug: bool = False,
) -> Dict[str, pd.DataFrame]:
    del trace1_tree  # matching is gpu_op_uid-only; tree is not needed

    if diff_stats_df.empty:
        return perf_dfs

    ups = perf_dfs.get("unified_perf_summary")
    ups_df = ups if isinstance(ups, pd.DataFrame) else pd.DataFrame()
    gpu_uid_to_canonical = _build_gpu_uid_to_canonical(ups_df)
    if not gpu_uid_to_canonical:
        return perf_dfs

    (
        time_additions,
        time_subtractions,
        dominant_t2_map,
        dominant_lca_ids,
        dominant_t2_count,
    ) = _build_lca_consolidation(
        diff_stats_df,
        gpu_uid_to_canonical=gpu_uid_to_canonical,
    )
    consolidated_t1 = _apply_lca_consolidation(
        {k: v.copy() for k, v in perf_dfs.items()},
        time_additions,
        time_subtractions,
    )
    diff_lookup, diff_lca_ids, diff_count_lookup = _build_trace2_time_lookup(
        diff_stats_df,
        gpu_uid_to_canonical=gpu_uid_to_canonical,
    )
    lookup, merged_lca_ids, merged_count_lookup = _merge_dominant_into_lookup(
        diff_lookup,
        dominant_t2_map,
        diff_stats_lca_ids=diff_lca_ids,
        dominant_lca_ids=dominant_lca_ids,
        diff_count_lookup=diff_count_lookup,
        dominant_t2_count=dominant_t2_count,
    )
    return _enrich_consolidated_perf_sheets(
        consolidated_t1,
        lookup,
        lca_ids_lookup=merged_lca_ids if debug else None,
        count_lookup=merged_count_lookup,
        debug=debug,
    )


def postprocess_perf_report_dataframes_extension(
    dict_name2df: Dict[str, Any],
    perf_analyzer,
    *,
    extension_args: Optional[str] = None,
    extension_file: Optional[str] = None,
    enable_pseudo_ops: bool = False,
) -> Dict[str, Any]:
    if not extension_args or not str(extension_args).strip():
        return dict_name2df

    parts = str(extension_args).strip().split()
    trace2_path = parts[0]
    debug = "debug" in parts[1:]

    perf_analyzer2 = TreePerfAnalyzer.from_file(
        profile_filepath=trace2_path,
        arch=perf_analyzer.arch,
        python_path=perf_analyzer.python_path,
        include_unlinked_kernels=perf_analyzer.include_unlinked_kernels,
        enable_pseudo_ops=enable_pseudo_ops,
        add_python_func=perf_analyzer.add_python_func,
    )
    perf_analyzer2.tree.apply_annotation(
        name_filters=["vllm::unified_attention_with_output"]
    )
    if extension_file:
        from TraceLens.Reporting.generate_perf_report_pytorch import apply_extension

        apply_extension(perf_analyzer2, extension_file)

    td = TraceDiff(perf_analyzer.tree, perf_analyzer2.tree)
    td.generate_tracediff_report()

    diff_stats_df = td.diff_stats_df

    out = dict(dict_name2df)
    if not diff_stats_df.empty:
        enriched = enrich_perf_report_dict_inplace(
            dict_name2df, diff_stats_df, debug=debug
        )
        out.update(enriched)
        out["diff_stats"] = diff_stats_df
        print(
            "[TraceDiff] Added speedup, delta, Kernel Time (µs)_trace2_sum, "
            "and operation_count_trace2 to unified_perf_summary; "
            "added diff_stats sheet."
        )

    return out
