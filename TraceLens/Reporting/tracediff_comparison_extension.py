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

from collections import Counter
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
        t1_total = float(grp["busy_time"].iloc[0])

        t2_total = float(lca_trace2_time.get(lca_id, 0.0))
        has_trace2_match = bool(_trace2_gpu_op_uid_set_for_lca(trace2, lca_id))

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
                    "lca_count_trace2": 1 if has_trace2_match else 0,
                }
            else:
                existing = out[key]
                if lca_id not in existing["lca_ids"]:
                    existing["lca_ids"].append(lca_id)
                    existing["lca_names"].append(lca_display_name)
                    existing["lca_total_kernel_time_trace1_us"] += t1_total
                    existing["lca_total_kernel_time_trace2_us"] += t2_total
                    if has_trace2_match:
                        existing["lca_count_trace2"] += 1

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


_MULTI_OP_LCA_THRESHOLD = 5


def _rollup_multi_op_lcas(
    df: pd.DataFrame,
    diff_stats_df: pd.DataFrame,
    lca_metadata: Dict[Any, Dict[str, Any]],
) -> pd.DataFrame:
    """Replace summary rows from multi-op LCAs with a single LCA-level row.

    When an LCA contains kernels from many trace1 ops (>threshold), the
    per-op trace2 time is the full LCA busy_time — making individual op
    comparisons meaningless.  This function detects those rows, removes
    them, and inserts one rolled-up row per unique trace1 LCA name where
    both t1 and t2 are LCA-level busy_times.
    """
    if df.empty or diff_stats_df.empty or not lca_metadata:
        return df

    if "source" not in diff_stats_df.columns:
        return df

    ds = diff_stats_df.dropna(subset=["lowest_common_ancestor_id"])
    trace1_ds = ds[ds["source"] == "trace1"]
    trace2_ds = ds[ds["source"] == "trace2"]
    if trace1_ds.empty:
        return df

    # Step 1: LCA ID → trace1 row count; identify multi-op LCA IDs
    lca_t1_counts = trace1_ds.groupby("lowest_common_ancestor_id").size()
    multi_op_lca_ids = set(
        lca_t1_counts[lca_t1_counts > _MULTI_OP_LCA_THRESHOLD].index
    )
    if not multi_op_lca_ids:
        return df

    # Step 2: LCA ID → busy_times (for building rolled-up rows)
    lca_t1_busy = (
        trace1_ds.groupby("lowest_common_ancestor_id")["busy_time"].first()
    )
    lca_t2_busy = (
        trace2_ds.groupby("lowest_common_ancestor_id")["busy_time"].first()
        if not trace2_ds.empty
        else pd.Series(dtype=float)
    )

    # Step 3: LCA ID → trace1 LCA name (before " | ")
    lca_id_to_t1_name: Dict[Any, str] = {}
    for lca_id in multi_op_lca_ids:
        grp = trace1_ds[trace1_ds["lowest_common_ancestor_id"] == lca_id]
        full_name = grp["lowest_common_ancestor_name"].dropna()
        if full_name.empty:
            lca_id_to_t1_name[lca_id] = "unknown"
            continue
        raw = str(full_name.iloc[0])
        lca_id_to_t1_name[lca_id] = raw.split(" | ")[0]

    # Step 4: Identify affected summary rows
    affected_rows: Dict[Any, Set] = {}  # summary row idx → set of multi-op LCA IDs
    for row_idx, meta in lca_metadata.items():
        overlap = set(meta.get("lca_ids", [])) & multi_op_lca_ids
        if overlap:
            affected_rows[row_idx] = overlap

    if not affected_rows:
        return df

    # Step 5: Group affected rows by trace1 LCA name
    #   For each affected row, pick the most common trace1 LCA name among
    #   its multi-op LCA IDs as the group key.
    groups: Dict[str, Dict] = {}  # t1_name → {row_indices, lca_ids}
    for row_idx, overlap_lca_ids in affected_rows.items():
        names = [lca_id_to_t1_name[lid] for lid in overlap_lca_ids]
        group_name = Counter(names).most_common(1)[0][0]
        if group_name not in groups:
            groups[group_name] = {"row_indices": set(), "lca_ids": set()}
        groups[group_name]["row_indices"].add(row_idx)
        groups[group_name]["lca_ids"].update(overlap_lca_ids)

    # Step 6: Build rolled-up rows
    kt_col = _KERNEL_TIME_COL_FOR_SPEEDUP_DELTA
    rollup_rows = []
    all_drop_indices = set()

    for group_name, info in groups.items():
        row_indices = info["row_indices"]
        group_lca_ids = info["lca_ids"]
        all_drop_indices.update(row_indices)

        t1_total = sum(
            float(lca_t1_busy.get(lid, 0.0)) for lid in group_lca_ids
        )
        t2_total = sum(
            float(lca_t2_busy.get(lid, 0.0)) for lid in group_lca_ids
        )
        n_instances = len(group_lca_ids)
        t2_matched = sum(
            1 for lid in group_lca_ids if lid in lca_t2_busy.index
        )

        lca_id_str = "; ".join(str(int(lid)) for lid in sorted(group_lca_ids))
        lca_name_strs = []
        for lid in sorted(group_lca_ids):
            grp = trace1_ds[trace1_ds["lowest_common_ancestor_id"] == lid]
            fn = grp["lowest_common_ancestor_name"].dropna()
            if not fn.empty:
                lca_name_strs.append(str(fn.iloc[0]))
        lca_name_str = "; ".join(dict.fromkeys(lca_name_strs))

        categories = set()
        for ri in row_indices:
            if ri in df.index and "op category" in df.columns:
                categories.add(df.loc[ri, "op category"])
        op_cat = categories.pop() if len(categories) == 1 else "other"

        speedup = (t2_total / t1_total) if t1_total > 0 else np.nan

        new_row = {c: np.nan for c in df.columns}
        new_row["name"] = group_name
        new_row["op category"] = op_cat
        new_row["operation_count"] = n_instances
        if kt_col in df.columns:
            new_row[kt_col] = t1_total
        new_row["speedup (trace2/trace1)"] = speedup
        new_row["delta_us (trace2 - trace1)"] = t2_total - t1_total
        new_row[_TRACE2_OP_COUNT_COL] = float(t2_matched)
        new_row["lca_id"] = lca_id_str
        new_row["lca_name"] = lca_name_str
        new_row["lca_total_kernel_time_trace1_us"] = t1_total
        new_row["lca_total_kernel_time_trace2_us"] = t2_total

        rollup_rows.append(new_row)

    # Step 7: Drop affected rows, append rolled-up rows
    valid_drop = [idx for idx in all_drop_indices if idx in df.index]
    result = df.drop(index=valid_drop)

    if rollup_rows:
        rollup_df = pd.DataFrame(rollup_rows, columns=df.columns)
        result = pd.concat([result, rollup_df], ignore_index=True)

    n_removed = len(valid_drop)
    n_added = len(rollup_rows)
    n_multi = len(multi_op_lca_ids)
    print(
        f"[TraceDiff] Multi-op LCA rollup: {n_multi} LCAs (>{_MULTI_OP_LCA_THRESHOLD} "
        f"trace1 rows), {n_removed} summary rows replaced by {n_added} rolled-up rows."
    )

    return result


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

    working["unified_perf_summary"] = _rollup_multi_op_lcas(
        working["unified_perf_summary"],
        diff_stats_df,
        lca_metadata,
    )

    print(
        "[TraceDiff] Added speedup, delta, lca_count_trace2, "
        "and LCA columns (lca_id, lca_name, "
        "lca_total_kernel_time_trace1_us, lca_total_kernel_time_trace2_us) "
        "to unified_perf_summary; added diff_stats sheet."
    )
    return working
