###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Combine the perf report CSVs of several traces into one report.

Each input is a ``(phase, csv_dir)`` pair written by the perf report. Rows that
share a table's row key are combined: times and counts add up, means and
standard deviations are recomputed from the per-report counts, and min / max
take the overall extreme. Percentages are recomputed from the combined totals,
so they are shares of all reports together.

Policies:

* ``combine_only_same_phase`` — the phase is part of the row key. Reports of
  the same phase combine; rows of different phases sit side by side.
* ``combine`` — the phase is ignored and matching rows combine across all
  reports. The ``phase`` column lists every contributing phase.

``gpu_timeline`` is always one timeline summed over every report, so its
``busy_time`` is the denominator for shares of the whole run.

Medians and per-kernel summaries can't be combined exactly; they come from the
contributing row with the most instances. A column or CSV without a rule
raises ``AggregationError`` instead of being dropped.
"""

import ast
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from TraceLens.TreePerf.tree_perf import PERF_PARAM_PREFIX, UNIQUE_ARGS_GROUPING_COLS

# Constants
POLICY_SAME_PHASE = "combine_only_same_phase"
POLICY_COMBINE = "combine"
POLICIES = (POLICY_SAME_PHASE, POLICY_COMBINE)
PHASE_COL = "phase"
_PHASE_JOIN = "+"

_GPU_TIMELINE = "gpu_timeline"
_GPU_TIMELINE_TIME = "time ms"
_GPU_TIMELINE_PCT = "percent"
_GPU_TIMELINE_TOTAL = "total_time"
_TABLE_KEYS = {
    _GPU_TIMELINE: ("type",),
    "ops_summary": ("name",),
    "ops_summary_by_category": ("op category",),
    "ops_unique_args": UNIQUE_ARGS_GROUPING_COLS,
    "unified_perf_summary": UNIQUE_ARGS_GROUPING_COLS,
}
# Any other CSV with this column is a roofline sheet keyed by name and params.
_ROOFLINE_MARKER = "name_count"
_ROOFLINE_KEYS = ("name",)
# Group-by columns a summary has only when it was grouped by them.
_OPTIONAL_KEYS = (
    "parent_module",
    "overlapping_kernel_names",
    "num_kernels",
    "is_recompute",
)
# Instance counts that weight means and pool standard deviations.
_COUNT_COLS = ("operation_count", "name_count")

_SUM = "sum"
_MEAN = "mean"
_STD = "std"
_MIN = "min"
_MAX = "max"
_FIRST = "first"
_REPRESENTATIVE = "representative"
_UNION = "union"
_DERIVED = "derived"
_NAMED_RULES = {
    _GPU_TIMELINE_TIME: _SUM,
    "operation_count": _SUM,
    "name_count": _SUM,
    "Count": _SUM,
    "total_duration_us": _SUM,
    "mean_duration_us": _MEAN,
    "std_duration_us": _STD,
    "ex_UID": _FIRST,
    "GFLOPS": _FIRST,
    "Data Moved (MB)": _FIRST,
    "FLOPS/Byte": _FIRST,
    "Compute Spec": _FIRST,
    "Roofline Bound": _FIRST,
    "perf_params": _FIRST,
    "has_perf_model": _FIRST,
    "entry_point": _FIRST,
    "call_stack": _FIRST,
    "call_stack_full": _FIRST,
    "kernel_details_summary": _REPRESENTATIVE,
    "kernel_details__summarize_kernel_stats": _REPRESENTATIVE,
    "trunc_kernel_details": _REPRESENTATIVE,
    "Categories": _UNION,
    _GPU_TIMELINE_PCT: _DERIVED,
    "Percentage (%)": _DERIVED,
    "Cumulative Percentage (%)": _DERIVED,
}
_SUFFIX_RULES = (
    ("_sum", _SUM),
    ("_count", _SUM),
    ("_ms", _SUM),
    ("_mean", _MEAN),
    ("_std", _STD),
    ("_min", _MIN),
    ("_max", _MAX),
    ("_median", _REPRESENTATIVE),
    ("_first", _FIRST),
)
# Standard deviations whose mean column doesn't share their prefix.
_STD_TO_MEAN = {"std_duration_us": "mean_duration_us"}
# Time column that drives Percentage (%) and the row order; first present wins.
_PCT_SOURCE_COLS = (
    "Kernel Time (µs)_sum",
    "total_direct_kernel_time_sum",
    "total_direct_kernel_time_ms",
)
_PCT_COL = "Percentage (%)"
_CUM_PCT_COL = "Cumulative Percentage (%)"


class AggregationError(ValueError):
    """The reports can't be combined: unknown input or invalid arguments."""


def aggregate_perf_report_csvs(
    reports: Sequence[Tuple[str, str]],
    policy: str,
    output_dir: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """Combine ``reports`` (``(phase, csv_dir)`` pairs) under ``policy``.

    Returns one DataFrame per CSV name and writes them to ``output_dir`` when
    it is given. Reports are read in order; ``first`` columns take the first
    report that has the row.
    """
    if policy not in POLICIES:
        raise AggregationError(
            f"unknown aggregation policy {policy!r}; use one of: "
            + ", ".join(POLICIES)
            + "."
        )
    if not reports:
        raise AggregationError("at least one report is required.")

    tables: Dict[str, List[Tuple[str, pd.DataFrame]]] = {}
    for phase, csv_dir in reports:
        if not os.path.isdir(csv_dir):
            raise AggregationError(f"report CSV directory not found: {csv_dir}")
        for file_name in sorted(os.listdir(csv_dir)):
            if not file_name.endswith(".csv"):
                continue
            table = file_name[: -len(".csv")]
            df = _read_table(table, os.path.join(csv_dir, file_name))
            tables.setdefault(table, []).append((phase, df))

    result = {
        table: _aggregate_table(table, frames, policy)
        for table, frames in tables.items()
    }
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        for table, df in result.items():
            df.to_csv(os.path.join(output_dir, f"{table}.csv"), index=False)
    return result


def _row_keys(table: str, columns: Sequence[str]) -> List[str]:
    if table in _TABLE_KEYS:
        base = list(_TABLE_KEYS[table])
    elif _ROOFLINE_MARKER in columns:
        base = list(_ROOFLINE_KEYS) + [
            c for c in columns if c.startswith(PERF_PARAM_PREFIX)
        ]
    else:
        raise AggregationError(
            f"{table}.csv has no aggregation rule; add one to "
            "TraceLens/Reporting/aggregate_perf_reports.py."
        )
    keys = [c for c in base if c in columns]
    keys += [c for c in _OPTIONAL_KEYS if c in columns and c not in keys]
    return keys


def _column_rule(column: str) -> Optional[str]:
    if column in _NAMED_RULES:
        return _NAMED_RULES[column]
    for suffix, rule in _SUFFIX_RULES:
        if column.endswith(suffix):
            return rule
    return None


def _read_table(table: str, path: str) -> pd.DataFrame:
    # An aggregated report has a phase column; the caller's phase replaces it.
    columns = [c for c in pd.read_csv(path, nrows=0).columns if c != PHASE_COL]
    keys = _row_keys(table, columns)
    unknown = [c for c in columns if c not in keys and _column_rule(c) is None]
    if unknown:
        raise AggregationError(
            f"{table}.csv has columns without an aggregation rule: "
            + ", ".join(unknown)
            + "."
        )
    # Keys compare as text so 32 and 32.0 from different files don't split a row.
    return pd.read_csv(path, dtype={k: str for k in keys})


def _aggregate_table(
    table: str, frames: List[Tuple[str, pd.DataFrame]], policy: str
) -> pd.DataFrame:
    tagged = []
    for phase, df in frames:
        df = df.drop(columns=[PHASE_COL], errors="ignore").copy()
        df[PHASE_COL] = phase
        tagged.append(df)
    combined = pd.concat(tagged, ignore_index=True)
    columns = [c for c in combined.columns if c != PHASE_COL]
    keys = _row_keys(table, columns)
    split_phases = policy == POLICY_SAME_PHASE and table != _GPU_TIMELINE
    group_keys = keys + ([PHASE_COL] if split_phases else [])
    count_col = next((c for c in _COUNT_COLS if c in columns), None)

    rules = {c: _column_rule(c) for c in columns if c not in keys}
    if count_col is None and any(r in (_MEAN, _STD) for r in rules.values()):
        raise AggregationError(f"{table}.csv has means but no instance count column.")

    rows = []
    for _, positions in combined.groupby(
        group_keys, dropna=False, sort=False
    ).indices.items():
        group = combined.iloc[positions]
        row = {c: group[c].iloc[0] for c in keys}
        row[PHASE_COL] = _PHASE_JOIN.join(sorted(set(group[PHASE_COL])))
        weights = group[count_col].astype(float) if count_col else None
        rep = (
            group.loc[weights.idxmax()]
            if weights is not None and weights.notna().any()
            else group.iloc[0]
        )
        for col, rule in rules.items():
            row[col] = _combine(col, rule, group, weights, rep)
        rows.append(row)

    out = pd.DataFrame(rows, columns=[PHASE_COL] + columns)
    if table == _GPU_TIMELINE:
        return _with_timeline_percent(out)
    return _with_percentages(out)


def _combine(col, rule, group, weights, rep):
    values = group[col]
    if rule == _SUM:
        return pd.to_numeric(values).sum(min_count=1)
    if rule == _FIRST:
        present = values.dropna()
        return present.iloc[0] if len(present) else np.nan
    if rule == _REPRESENTATIVE:
        return rep[col]
    if rule == _MIN:
        return pd.to_numeric(values).min()
    if rule == _MAX:
        return pd.to_numeric(values).max()
    if rule == _UNION:
        merged = set()
        for value in values.dropna():
            merged.update(ast.literal_eval(value))
        return str(sorted(merged))
    if rule == _MEAN:
        return _weighted_mean(pd.to_numeric(values), weights)
    if rule == _STD:
        mean_col = _STD_TO_MEAN.get(col, col[: -len("_std")] + "_mean")
        if mean_col not in group.columns:
            raise AggregationError(f"{col} has no matching mean column {mean_col}.")
        return _pooled_std(
            pd.to_numeric(group[mean_col]), pd.to_numeric(values), weights
        )
    return np.nan


def _weighted_mean(means: pd.Series, counts: pd.Series) -> float:
    mask = means.notna() & counts.notna()
    total = counts[mask].sum()
    if total == 0:
        return np.nan
    return float((means[mask] * counts[mask]).sum() / total)


def _pooled_std(means: pd.Series, stds: pd.Series, counts: pd.Series) -> float:
    """Sample standard deviation (ddof=1) of the union of the groups."""
    mask = means.notna() & counts.notna()
    n, m = counts[mask], means[mask]
    # pandas leaves std NaN for a single instance; its spread is zero.
    s = stds[mask].fillna(0.0)
    total = n.sum()
    if total <= 1:
        return np.nan
    grand = (n * m).sum() / total
    ss = ((n - 1) * s**2 + n * (m - grand) ** 2).sum()
    return float(np.sqrt(ss / (total - 1)))


def _with_percentages(df: pd.DataFrame) -> pd.DataFrame:
    source = next((c for c in _PCT_SOURCE_COLS if c in df.columns), None)
    if source is None:
        return df
    df = df.sort_values(by=source, ascending=False, kind="mergesort")
    df = df.reset_index(drop=True)
    if _PCT_COL in df.columns:
        df[_PCT_COL] = df[source] / df[source].sum() * 100
    if _CUM_PCT_COL in df.columns:
        df[_CUM_PCT_COL] = df[_PCT_COL].cumsum()
    return df


def _with_timeline_percent(df: pd.DataFrame) -> pd.DataFrame:
    total = df.loc[df["type"] == _GPU_TIMELINE_TOTAL, _GPU_TIMELINE_TIME].sum()
    df[_GPU_TIMELINE_PCT] = df[_GPU_TIMELINE_TIME] / total * 100 if total else np.nan
    return df
