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
**lca_count_trace2**, and LCA columns (``lca_id``, ``lca_name``,
``lca_total_kernel_time_trace1_us``, ``lca_total_kernel_time_trace2_us``)
to ``unified_perf_summary`` (beside ``Kernel Time (µs)_sum`` /
``operation_count``). Speedup and delta are always computed from the LCA-level
totals (``trace2 / trace1``). For single-op LCAs, the totals equal the
per-row values. For multi-op LCAs (several trace1 CPU ops sharing one
``lowest_common_ancestor_id``), the totals aggregate across the group and
``lca_id`` identifies the shared group. A ``diff_stats`` sheet
(TraceDiff per-kernel diff rows) is included whenever ``diff_stats_df`` is
non-empty. ``lca_id`` and ``lca_name`` are semicolon-separated lists when a
row maps to multiple LCA groups.

Matching is **gpu_op_uid only**. Every row in ``unified_perf_summary`` carries
a ``kernel_details_summary`` / ``kernel_details`` list of kernels with
``gpu_op_uid``. Each diff_stats row carries the ``gpu_op_uid`` of its kernel.
Rows are aligned by mapping every gpu_op_uid to the row index of the
``unified_perf_summary`` row that contains it. There is no fallback to CPU UID
tree walks or to (name, args) string matching.
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


def _build_uid_to_row_idx(ups: pd.DataFrame) -> Dict[Any, Any]:
    """Map every ``gpu_op_uid`` in ``unified_perf_summary`` to its row index.

    This is the only index used for diff_stats → perf-row matching.
    """
    mapping: Dict[Any, Any] = {}
    if ups is None or ups.empty:
        return mapping
    if not any(c in ups.columns for c in _KERNEL_DETAIL_COLS):
        return mapping
    for idx, row in ups.iterrows():
        for uid in _iter_row_gpu_op_uids(row):
            mapping[uid] = idx
    return mapping


def _resolve_diff_row_to_key(row, uid_to_row_idx: Dict[Any, Any]) -> Optional[Any]:
    """Row index for a diff_stats row via its ``gpu_op_uid``."""
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
    """Map perf-summary row index → LCA metadata from ``diff_stats``.

    For each trace1 row, the row index is resolved via ``uid_to_row_idx``.
    Every mapped row carries lists of ``lca_ids`` and ``lca_names`` (one entry
    per LCA group that maps to this key), plus accumulated
    ``lca_total_kernel_time_trace1_us`` and ``lca_total_kernel_time_trace2_us``
    (summed across all LCA groups).

    A single perf-summary row can map to multiple LCA groups when its kernels
    participate in different TraceDiff matching scopes.
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

    _has_busy = "busy_time" in trace2.columns and trace2["busy_time"].notna().any()
    if _has_busy:
        lca_trace2_time = trace2.groupby("lowest_common_ancestor_id")[
            "busy_time"
        ].first()
    else:
        lca_trace2_time = trace2.groupby("lowest_common_ancestor_id")[
            "kernel_time"
        ].sum()

    for lca_id, grp in trace1.groupby("lowest_common_ancestor_id"):
        _has_busy_t1 = "busy_time" in grp.columns and grp["busy_time"].notna().any()
        if _has_busy_t1:
            t1_total = float(grp["busy_time"].iloc[0])
        elif "kernel_time" in grp.columns:
            t1_total = float(grp["kernel_time"].sum())
        else:
            t1_total = 0.0

        t2_total = float(lca_trace2_time.get(lca_id, 0.0))

        names = grp["lowest_common_ancestor_name"].dropna()
        lca_display_name = str(names.iloc[0]) if not names.empty else None
        if lca_display_name == "":
            lca_display_name = None

        for _, row in grp.iterrows():
            key = _resolve_diff_row_to_key(row, uid_to_row_idx)
            if key is None:
                continue
            if key not in out:
                out[key] = {
                    "lca_ids": [lca_id],
                    "lca_names": [lca_display_name],
                    "lca_total_kernel_time_trace1_us": t1_total,
                    "lca_total_kernel_time_trace2_us": t2_total,
                }
            else:
                existing = out[key]
                if lca_id not in existing["lca_ids"]:
                    existing["lca_ids"].append(lca_id)
                    existing["lca_names"].append(lca_display_name)
                    existing["lca_total_kernel_time_trace1_us"] += t1_total
                    existing["lca_total_kernel_time_trace2_us"] += t2_total

    return out


def _build_trace2_time_lookup(
    diff_stats_df: pd.DataFrame,
    uid_to_row_idx: Optional[Dict[Any, Any]] = None,
    gpu_uid_to_canonical: Optional[Dict[Any, Any]] = None,  # deprecated, ignored
) -> Tuple[Dict[Any, float], Dict[Any, List], Dict[Any, int]]:
    if diff_stats_df.empty or not uid_to_row_idx:
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

        all_keys: Set[Any] = set()
        for _, row in grp.iterrows():
            key = _resolve_diff_row_to_key(row, uid_to_row_idx)
            if key is not None:
                all_keys.add(key)

        for key in all_keys:
            lookup[key] = lookup.get(key, 0.0) + t2_time
            lookup_uid_sets[key] = lookup_uid_sets.get(key, set()).union(t2_uid_set)
            lca_ids_lookup.setdefault(key, [])
            if t2_uid_set:
                lca_ids_lookup[key].append(lca_id)

    lookup_count = {k: len(v) for k, v in lca_ids_lookup.items()}
    return lookup, lca_ids_lookup, lookup_count


_TRACE2_OP_COUNT_COL = "lca_count_trace2"


def _enrich_sheet_with_trace2(
    df_sheet: pd.DataFrame,
    kernel_time_col: str,
    *,
    count_lookup: Optional[Dict[Any, int]] = None,
    lca_metadata: Optional[Dict[Any, Dict[str, Any]]] = None,
) -> pd.DataFrame:
    """Add LCA columns, speedup, delta, and trace2 op counts to a perf sheet.

    Speedup is computed from LCA-level aggregates (``lca_total_kernel_time_trace2_us``
    / ``lca_total_kernel_time_trace1_us``). Delta uses the per-row
    ``kernel_time_col`` (``Kernel Time (µs)_sum``) as the trace1 reference so
    that each row's delta reflects the row's own kernel time rather than the
    full LCA group total.
    ``lca_id`` and ``lca_name`` are semicolon-separated lists when a row
    maps to multiple LCA groups.
    """
    if df_sheet.empty or not lca_metadata:
        return df_sheet

    df = df_sheet.copy()
    clookup = count_lookup if count_lookup is not None else {}
    lca_meta = lca_metadata

    trace2_counts: List[float] = []
    lca_id_vals: List[Any] = []
    lca_name_vals: List[Any] = []
    lca_t1_vals: List[float] = []
    lca_t2_vals: List[float] = []

    for idx, row in df.iterrows():
        meta = lca_meta.get(idx)

        cv = clookup.get(idx)
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


def _enrich_consolidated_perf_sheets(
    perf_dfs: Dict[str, pd.DataFrame],
    *,
    count_lookup: Optional[Dict[Any, int]] = None,
    lca_metadata: Optional[Dict[Any, Dict[str, Any]]] = None,
) -> Dict[str, pd.DataFrame]:
    """Add speedup, delta, LCA columns to ``unified_perf_summary``."""
    result: Dict[str, pd.DataFrame] = {}
    kt_col = _KERNEL_TIME_COL_FOR_SPEEDUP_DELTA
    for sheet_name, df_sheet in perf_dfs.items():
        if sheet_name != "unified_perf_summary":
            result[sheet_name] = df_sheet
            continue
        if "name" not in df_sheet.columns or not lca_metadata:
            result[sheet_name] = df_sheet
            continue
        if kt_col not in df_sheet.columns:
            result[sheet_name] = df_sheet
            continue
        result[sheet_name] = _enrich_sheet_with_trace2(
            df_sheet,
            kt_col,
            count_lookup=count_lookup,
            lca_metadata=lca_metadata,
        )
    return result


def enrich_perf_report_dict_inplace(
    perf_dfs: Dict[str, pd.DataFrame],
    diff_stats_df: pd.DataFrame,
    trace1_tree=None,  # kept for backward-compat; no longer used
    debug: bool = False,
) -> Dict[str, pd.DataFrame]:
    del trace1_tree  # matching is gpu_op_uid-only; tree is not needed
    del debug  # lca_id column now always carries the full list

    if diff_stats_df.empty:
        return perf_dfs

    ups = perf_dfs.get("unified_perf_summary")
    ups_df = ups if isinstance(ups, pd.DataFrame) else pd.DataFrame()
    uid_to_row_idx = _build_uid_to_row_idx(ups_df)
    if not uid_to_row_idx:
        return perf_dfs

    working = {k: v.copy() for k, v in perf_dfs.items()}
    lca_metadata = _build_lca_metadata(diff_stats_df, uid_to_row_idx)
    _, _, diff_count_lookup = _build_trace2_time_lookup(
        diff_stats_df,
        uid_to_row_idx=uid_to_row_idx,
    )
    return _enrich_consolidated_perf_sheets(
        working,
        count_lookup=diff_count_lookup,
        lca_metadata=lca_metadata,
    )


def postprocess_perf_report_dataframes_extension(
    dict_name2df: Dict[str, Any],
    perf_analyzer,
    *,
    extension_args: Optional[str] = None,
    extension_file: Optional[str] = None,
    enable_pseudo_ops: bool = True,
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
            "[TraceDiff] Added speedup, delta, lca_count_trace2, "
            "and LCA columns (lca_id, lca_name, "
            "lca_total_kernel_time_trace1_us, lca_total_kernel_time_trace2_us) "
            "to unified_perf_summary; added diff_stats sheet."
        )

    return out
