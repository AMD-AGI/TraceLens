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
distinct ``cpu_op_uid`` values on trace2 ``diff_stats`` rows aligned to the
same key as trace2 kernel time) to ``unified_perf_summary`` (beside
``Kernel Time (µs)_sum`` / ``operation_count``). Op-category workbook tabs
(``GEMM``, ``CONV_bwd``, etc.) and launcher sheets are unchanged—enough for
workflows that consume unified summary only (e.g. AgenticMode standalone).
When ``unified_perf_summary`` is non-empty, diff rows use ``cpu_op_uid`` and
enriched unified rows use ``ex_UID`` to align tuple keys via the same tree walk
as the trace2 time lookup. If no unified row matches a diff UID, that diff row
is skipped in the lookup. If the unified sheet is empty, keys use only the
diff row's ``cpu_op_name`` and arg columns.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from TraceLens import TraceDiff, TreePerfAnalyzer

_ARG_COLS = ["Input Dims", "Input type", "Input Strides", "Concrete Inputs"]

# Sheets enriched with speedup/delta must use this column (unified + op-category tabs).
_KERNEL_TIME_COL_FOR_SPEEDUP_DELTA = "Kernel Time (µs)_sum"

_KERNEL_TIME_SUM_COLS = [
    _KERNEL_TIME_COL_FOR_SPEEDUP_DELTA,
    "total_direct_kernel_time_sum",
]

# Cached in uid_cache when unified_perf matching was attempted and failed.
_NO_UNIFIED_PERF_MATCH = object()


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


def _normalize_val(v: str) -> str:
    return v.replace("[", "(").replace("]", ")").replace(",)", ")").strip()


def _trace2_cpu_op_uid_set_for_lca(trace2: pd.DataFrame, lca_id) -> Set[Any]:
    """Distinct ``cpu_op_uid`` values for trace2 ``diff_stats`` rows under ``lca_id``.

    If ``cpu_op_uid`` is missing or all null (e.g. synthetic test frames), fall
    back to row indices so each row still counts as distinct.
    """
    sub = trace2[trace2["lowest_common_ancestor_id"] == lca_id]
    if sub.empty:
        return set()
    if "cpu_op_uid" in sub.columns and sub["cpu_op_uid"].notna().any():
        return set(sub["cpu_op_uid"].dropna().tolist())
    return set(sub.index.tolist())


def _make_str_key(name, row, arg_cols=_ARG_COLS):
    parts = [str(name)]
    for c in arg_cols:
        val = row.get(c, None)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            val = row.get(f"{c}_first", "")
        parts.append(_normalize_val(str(val if val is not None else "")))
    tn = row.get("thread_name", None)
    if tn is None or (isinstance(tn, float) and np.isnan(tn)):
        tn = row.get("thread_name_first", "")
    parts.append(str(tn if tn is not None else ""))
    return tuple(parts)


def _str_key_from_cpu_op_node(node, tree=None) -> Tuple:
    """Same tuple key as perf sheets use for a trace-tree cpu_op node."""
    if node is None:
        return tuple()
    name = node.get("name", "")
    args = node.get("args", {}) or {}
    row = {
        c: str(args.get(c, "") if args.get(c) is not None else "") for c in _ARG_COLS
    }
    tn = ""
    if tree is not None:
        pid = node.get("pid")
        tid = node.get("tid")
        if pid is not None and tid is not None:
            tn = tree.metadata.get(pid, {}).get(tid, {}).get("thread_name", "")
    row["thread_name"] = tn
    return _make_str_key(name, row)


def _unified_perf_summary_key_set(df: pd.DataFrame) -> Set[tuple]:
    """Keys of rows in ``unified_perf_summary`` (for aligning diff_stats UIDs)."""
    if df is None or df.empty or "name" not in df.columns:
        return set()
    keys: Set[tuple] = set()
    for _, r in df.iterrows():
        keys.add(_make_str_key(r.get("name", ""), r))
    return keys


def _ensure_parent_to_children(tree) -> Dict[int, list]:
    if not hasattr(tree, "_parent_to_children"):
        ptc: Dict[int, list] = {}
        for uid, ev in tree.events_by_uid.items():
            p = ev.get("parent")
            if p is not None:
                ptc.setdefault(p, []).append(uid)
        tree._parent_to_children = ptc
    return tree._parent_to_children


def _find_perf_summary_matching_node(cpu_op_uid, tree, perf_summary_keys: Set[tuple]):
    """Walk ancestors then descendants until a cpu_op's key is in ``unified_perf_summary``."""
    if tree is None or cpu_op_uid is None or not perf_summary_keys:
        return None

    node = tree.events_by_uid.get(cpu_op_uid)
    if node is None:
        return None

    temp = node
    while temp is not None:
        if tree.event_to_category(temp) == "cpu_op":
            k = _str_key_from_cpu_op_node(temp, tree)
            if k in perf_summary_keys:
                return temp
        p_uid = temp.get("parent")
        temp = tree.events_by_uid.get(p_uid) if p_uid is not None else None

    ptc = _ensure_parent_to_children(tree)
    queue = deque(ptc.get(cpu_op_uid, []))
    while queue:
        child_uid = queue.popleft()
        child_node = tree.events_by_uid.get(child_uid)
        if child_node is None:
            continue
        if tree.event_to_category(child_node) == "cpu_op":
            k = _str_key_from_cpu_op_node(child_node, tree)
            if k in perf_summary_keys:
                return child_node
        queue.extend(ptc.get(child_uid, []))

    return None


def _row_cpu_op_uid(row) -> Optional[Any]:
    """UID for tree alignment: diff_stats uses ``cpu_op_uid``; perf sheets ``UID_first``; unified ``ex_UID``."""
    for col in ("cpu_op_uid", "UID_first", "ex_UID"):
        if col not in row.index:
            continue
        u = row.get(col)
        if u is not None and pd.notna(u):
            return u
    return None


def _tree_resolve_uid_to_key(
    uid: Any,
    tree,
    perf_summary_keys: Set[tuple],
    uid_cache: Dict,
) -> Optional[tuple]:
    """Return unified-aligned tuple key, or ``None`` if this UID does not match any unified row."""
    if uid not in uid_cache:
        anc = _find_perf_summary_matching_node(uid, tree, perf_summary_keys)
        uid_cache[uid] = (
            _str_key_from_cpu_op_node(anc, tree)
            if anc is not None
            else _NO_UNIFIED_PERF_MATCH
        )
    resolved = uid_cache[uid]
    if resolved is _NO_UNIFIED_PERF_MATCH:
        return None
    return resolved


def _resolve_row_to_key(
    row, tree=None, uid_cache=None, perf_summary_keys=None
) -> Optional[tuple]:
    if uid_cache is None:
        uid_cache = {}
    if perf_summary_keys is None:
        perf_summary_keys = set()

    if tree is not None and perf_summary_keys:
        uid = _row_cpu_op_uid(row)
        if uid is not None:
            k = _tree_resolve_uid_to_key(uid, tree, perf_summary_keys, uid_cache)
            if k is not None:
                return k
            return None

    return _make_str_key(row.get("cpu_op_name", ""), row)


def _enrichment_row_lookup_key(
    row: pd.Series,
    tree,
    perf_summary_keys: Set[tuple],
    uid_cache: Dict,
) -> tuple:
    """Key into trace2 time lookup for a perf-sheet row (UID-first when possible)."""
    if tree is not None and perf_summary_keys:
        uid = _row_cpu_op_uid(row)
        if uid is not None:
            k = _tree_resolve_uid_to_key(uid, tree, perf_summary_keys, uid_cache)
            if k is not None:
                return k
    return _make_str_key(row.get("name", ""), row)


def _build_lca_consolidation(
    diff_stats_df: pd.DataFrame,
    trace1_tree=None,
    perf_summary_keys: Optional[Set[tuple]] = None,
) -> Tuple[
    Dict[tuple, float],
    Dict[tuple, float],
    Dict[tuple, float],
    Dict[tuple, List],
    Dict[tuple, int],
]:
    empty = ({}, {}, {}, {}, {})
    if diff_stats_df.empty:
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

    uid_cache: Dict = {}
    time_additions: Dict[tuple, float] = {}
    time_subtractions: Dict[tuple, float] = {}
    dominant_trace2_map: Dict[tuple, float] = {}
    dominant_trace2_uid_sets: Dict[tuple, Set[Any]] = {}
    dominant_lca_ids: Dict[tuple, List] = {}
    psk = perf_summary_keys if perf_summary_keys is not None else set()

    for lca_id in multi_lca_ids:
        grp = t1[t1["lowest_common_ancestor_id"] == lca_id]
        op_times = grp.groupby("cpu_op_name")["kernel_time"].sum()
        dominant_op_name = op_times.idxmax()

        dominant_rows = grp[grp["cpu_op_name"] == dominant_op_name]
        dominant_key = _resolve_row_to_key(
            dominant_rows.iloc[0],
            tree=trace1_tree,
            uid_cache=uid_cache,
            perf_summary_keys=psk,
        )
        if dominant_key is None:
            continue

        t2_time = float(lca_t2_time.get(lca_id, 0.0))
        t2_uid_set = _trace2_cpu_op_uid_set_for_lca(t2, lca_id)
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
            nd_key = _resolve_row_to_key(
                nd_rows.iloc[0],
                tree=trace1_tree,
                uid_cache=uid_cache,
                perf_summary_keys=psk,
            )
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
    time_additions: Dict[tuple, float],
    time_subtractions: Dict[tuple, float],
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

        df = df_sheet.copy()
        rows_to_drop: List[int] = []

        for idx, row in df.iterrows():
            key = _make_str_key(row.get("name", ""), row)

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
    tree=None,
    perf_summary_keys: Optional[Set[tuple]] = None,
) -> Tuple[Dict[tuple, float], Dict[tuple, List], Dict[tuple, int]]:
    if diff_stats_df.empty:
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

    uid_cache: Dict = {}
    lca_groups = trace1.groupby("lowest_common_ancestor_id")
    psk = perf_summary_keys if perf_summary_keys is not None else set()

    lookup: Dict[tuple, float] = {}
    lca_ids_lookup: Dict[tuple, List] = {}
    lookup_uid_sets: Dict[tuple, Set[Any]] = {}
    for lca_id, grp in lca_groups:
        t2_time = float(lca_trace2_time.get(lca_id, 0.0))
        t2_uid_set = _trace2_cpu_op_uid_set_for_lca(trace2, lca_id)

        unique_ops = grp["cpu_op_name"].unique()
        if len(unique_ops) > 1:
            op_times = grp.groupby("cpu_op_name")["kernel_time"].sum()
            dominant_rows = grp[grp["cpu_op_name"] == op_times.idxmax()]
            key_row = dominant_rows.iloc[0]
        else:
            key_row = grp.iloc[0]

        key = _resolve_row_to_key(
            key_row, tree=tree, uid_cache=uid_cache, perf_summary_keys=psk
        )
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
    lookup: Dict[tuple, float],
    kernel_time_col: str,
    *,
    tree=None,
    perf_summary_keys: Optional[Set[tuple]] = None,
    lca_ids_lookup: Optional[Dict[tuple, List]] = None,
    count_lookup: Optional[Dict[tuple, int]] = None,
    debug: bool = False,
) -> pd.DataFrame:
    """Add speedup, delta, trace2 kernel time, and trace2 unique-``cpu_op_uid`` count."""
    if df_sheet.empty or not lookup:
        return df_sheet

    df = df_sheet.copy()

    psk = perf_summary_keys if perf_summary_keys is not None else set()
    uid_cache: Dict = {}
    clookup = count_lookup if count_lookup is not None else {}

    trace2_times = []
    trace2_counts: List[float] = []
    lca_id_lists: List[str] = []
    need_lca_debug = debug and lca_ids_lookup is not None
    for _, row in df.iterrows():
        key = _enrichment_row_lookup_key(row, tree, psk, uid_cache)
        trace2_times.append(lookup.get(key, np.nan))
        cv = clookup.get(key)
        trace2_counts.append(np.nan if cv is None else float(cv))
        if need_lca_debug:
            lca_id_lists.append(str(lca_ids_lookup.get(key, [])))

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
    diff_stats_lookup: Dict[tuple, float],
    dominant_t2_map: Dict[tuple, float],
    diff_stats_lca_ids: Optional[Dict[tuple, List]] = None,
    dominant_lca_ids: Optional[Dict[tuple, List]] = None,
    diff_count_lookup: Optional[Dict[tuple, int]] = None,
    dominant_t2_count: Optional[Dict[tuple, int]] = None,
) -> Tuple[Dict[tuple, float], Dict[tuple, List], Dict[tuple, int]]:
    merged = dict(diff_stats_lookup)
    merged_lca: Dict[tuple, List] = dict(diff_stats_lca_ids or {})
    merged_count: Dict[tuple, int] = dict(diff_count_lookup or {})
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
    lookup: Dict[tuple, float],
    *,
    trace1_tree=None,
    perf_summary_keys: Optional[Set[tuple]] = None,
    lca_ids_lookup: Optional[Dict[tuple, List]] = None,
    count_lookup: Optional[Dict[tuple, int]] = None,
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
            tree=trace1_tree,
            perf_summary_keys=perf_summary_keys,
            lca_ids_lookup=lca_ids_lookup,
            count_lookup=count_lookup,
            debug=debug,
        )
    return result


def enrich_perf_report_dict_inplace(
    perf_dfs: Dict[str, pd.DataFrame],
    diff_stats_df: pd.DataFrame,
    trace1_tree,
    debug: bool = False,
) -> Dict[str, pd.DataFrame]:
    if diff_stats_df.empty:
        return perf_dfs

    ups = perf_dfs.get("unified_perf_summary")
    perf_summary_keys = _unified_perf_summary_key_set(
        ups if isinstance(ups, pd.DataFrame) else pd.DataFrame()
    )

    (
        time_additions,
        time_subtractions,
        dominant_t2_map,
        dominant_lca_ids,
        dominant_t2_count,
    ) = _build_lca_consolidation(
        diff_stats_df,
        trace1_tree=trace1_tree,
        perf_summary_keys=perf_summary_keys,
    )
    consolidated_t1 = _apply_lca_consolidation(
        {k: v.copy() for k, v in perf_dfs.items()},
        time_additions,
        time_subtractions,
    )
    diff_lookup, diff_lca_ids, diff_count_lookup = _build_trace2_time_lookup(
        diff_stats_df,
        tree=trace1_tree,
        perf_summary_keys=perf_summary_keys,
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
        trace1_tree=trace1_tree,
        perf_summary_keys=perf_summary_keys,
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
    baseline = td.baseline

    out = dict(dict_name2df)
    if not diff_stats_df.empty:
        enriched = enrich_perf_report_dict_inplace(
            dict_name2df, diff_stats_df, baseline, debug=debug
        )
        out.update(enriched)
        if debug:
            out["diff_stats"] = diff_stats_df
        print(
            "[TraceDiff] Added speedup, delta, Kernel Time (µs)_trace2_sum, "
            "and operation_count_trace2 to unified_perf_summary only."
        )

    return out
