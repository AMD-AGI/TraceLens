###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unified trace quality checks for TraceLens.

Validates that a profiling trace is suitable for analysis, covering file-level
checks (size), event-level checks (kernel presence, call stacks, graph mode,
shapes), post-tree checks (dropped kernels via correlation IDs), and
post-report checks (report completeness, idle time, op consistency).

Basic checks run automatically via ``DataLoader.load_trace_events()`` and
``TreePerfAnalyzer``. Detailed checks run when
``TRACELENS_DETAILED_QUALITY_CHECKS=1`` is set. All checks can be skipped
with ``TRACELENS_SKIP_QUALITY_CHECK=1``.
"""

from __future__ import annotations

import enum
import json
import os
import statistics
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .util import (
    GPU_KERNEL_CATEGORIES,
    GRAPH_LAUNCH_NAMES,
    TraceEventUtils,
    merge_intervals,
    parse_bool_string,
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
class Status(enum.Enum):
    WARN = "warn"
    ERROR = "error"


@dataclass
class TraceCheck:
    status: Status
    message: str
    check_name: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TraceCheckReport:
    findings: List[TraceCheck] = field(default_factory=list)
    ran_checks: List[str] = field(default_factory=list)

    def log_findings(self) -> None:
        for f in self.findings:
            if f.status == Status.ERROR:
                raise ValueError(f"[trace_check] {f.message}")
            elif f.status == Status.WARN:
                warnings.warn(
                    f"[trace_check] {f.message}",
                    UserWarning,
                    stacklevel=2,
                )

    def merge(self, other: "TraceCheckReport") -> None:
        self.findings.extend(other.findings)
        self.ran_checks.extend(other.ran_checks)

    def to_json_dict(self) -> Dict[str, Dict[str, str]]:
        failed = {}
        for f in self.findings:
            failed[f.check_name] = {
                "status": f.status.value.upper(),
                "message": f.message,
            }
        result = {}
        for name in self.ran_checks:
            if name in failed:
                result[name] = failed[name]
            else:
                result[name] = {"status": "PASS", "message": ""}
        return result

    def write_json(self, output_dir: str) -> None:
        if not output_dir:
            return
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "trace_checks.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_json_dict(), f, indent=2)
            f.write("\n")


# ---------------------------------------------------------------------------
# Constants and thresholds
# ---------------------------------------------------------------------------
_MIN_TRACE_BYTES = 100_000
_MAX_TRACE_BYTES = 5_000_000_000
_MIN_CPU_OP_SHAPES = 10


@dataclass
class QualityThresholds:
    # runtime variability (spurious events)
    runtime_cv_warn: float = 0.25
    runtime_cv_fail: float = 0.75
    runtime_min_samples: int = 5
    runtime_min_dur_us: float = 2.0
    runtime_max_flagged_frac: float = 0.10

    # GPU busy / idle
    idle_pct_warn: float = 30.0
    idle_pct_fail: float = 60.0

    # shape / metadata coverage
    shape_coverage_warn: float = 0.80
    shape_coverage_fail: float = 0.40

    # CPU:GPU event ratio (PyTorch)
    cpu_gpu_ratio_min: float = 0.2
    cpu_gpu_ratio_max: float = 50.0

    # high-idle ops
    op_high_idle_pct: float = 40.0


@dataclass
class AttnCountConfig:
    """Config for the attention-kernel-count check.

    expected = num_gpus * num_steps * num_layers * calls * unique_kernels (per
    pass). num_gpus and unique_kernels are inferred; num_layers and num_steps
    must be supplied (else the check is skipped).
    """

    num_layers: Optional[int] = None
    num_steps: Optional[int] = None
    num_gpus: Optional[int] = None
    fwd_attn_calls: int = 1
    bwd_attn_calls: int = 1
    tol: float = 0.0

    @property
    def enabled(self) -> bool:
        return self.num_layers is not None and self.num_steps is not None


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def coefficient_of_variation(values: Sequence[float]) -> float:
    """std / mean (population std). 0.0 if empty or mean is 0."""
    n = len(values)
    if n == 0:
        return 0.0
    mean = sum(values) / n
    if mean == 0:
        return 0.0
    return statistics.pstdev(values) / abs(mean)


def busy_idle_from_events(events: Sequence[dict]) -> Dict[str, float]:
    """Compute busy/idle/total time (us) from events with ``ts``/``dur``."""
    intervals = [
        (float(e["ts"]), float(e["ts"]) + float(e["dur"]))
        for e in events
        if e.get("ts") is not None and e.get("dur") is not None
    ]
    if not intervals:
        return {"busy_time": 0.0, "idle_time": 0.0, "total_time": 0.0, "idle_pct": 0.0}
    merged = merge_intervals(intervals)
    total_time = merged[-1][1] - merged[0][0]
    busy_time = sum(e - s for s, e in merged)
    idle_time = max(0.0, total_time - busy_time)
    idle_pct = (100.0 * idle_time / total_time) if total_time > 0 else 0.0
    return {
        "busy_time": busy_time,
        "idle_time": idle_time,
        "total_time": total_time,
        "idle_pct": idle_pct,
    }


# ---------------------------------------------------------------------------
# Basic pre-report checks (always run)
# ---------------------------------------------------------------------------
def _check_trace_file_size(ctx) -> List[TraceCheck]:
    filepath = ctx.get("filepath")
    if filepath is None:
        return []
    findings = []
    try:
        size = os.path.getsize(filepath)
    except OSError:
        return []
    if size < _MIN_TRACE_BYTES:
        findings.append(
            TraceCheck(
                Status.WARN,
                f"Trace file is {size:,} bytes — may be empty or warmup-only.",
            )
        )
    if size > _MAX_TRACE_BYTES:
        findings.append(
            TraceCheck(
                Status.WARN,
                f"Trace file is {size / 1e9:.1f} GB — consider splitting.",
            )
        )
    return findings


def _check_kernels_present(ctx) -> List[TraceCheck]:
    events = ctx["events"]
    has_kernel = any(e.get("cat", "") in GPU_KERNEL_CATEGORIES for e in events)
    if not has_kernel:
        return [
            TraceCheck(
                Status.ERROR,
                "No GPU kernel events found in trace. "
                "Set TRACELENS_SKIP_QUALITY_CHECK=1 to bypass if intentional.",
            )
        ]
    return []


def _check_call_stack(ctx) -> List[TraceCheck]:
    events = ctx["events"]
    has_python_func = any(e.get("cat") == "python_function" for e in events)
    if not has_python_func:
        return [
            TraceCheck(
                Status.WARN,
                "Trace does not contain call stack data (with_stack=False or not set). "
                "Pass with_stack=True to the profiler for call-stack-based analysis. "
                "Cross-trace comparisons may also be affected.",
            )
        ]
    return []


def _check_graph_mode(ctx) -> List[TraceCheck]:
    events = ctx["events"]
    has_graph_launch = any(e.get("name") in GRAPH_LAUNCH_NAMES for e in events)
    if has_graph_launch and ctx.get("capture_trace_filepath") is None:
        return [
            TraceCheck(
                Status.WARN,
                "Trace contains graph launch event(s) but no capture trace "
                "was provided. Graph-mode analysis will be limited. "
                "Pass --capture_folder to the inference perf report generator.",
            )
        ]
    return []


def _check_cpu_op_shapes_basic(ctx) -> List[TraceCheck]:
    events = ctx["events"]
    cpu_op_count = 0
    cpu_op_with_shapes = 0
    for e in events:
        if e.get("cat") == "cpu_op":
            cpu_op_count += 1
            if "Input Dims" in (e.get("args") or {}):
                cpu_op_with_shapes += 1
    if cpu_op_count > 0 and cpu_op_with_shapes < _MIN_CPU_OP_SHAPES:
        return [
            TraceCheck(
                Status.WARN,
                f"Only {cpu_op_with_shapes} of {cpu_op_count} cpu_op events have "
                f"input shapes. Profile with record_shapes=True for shape-aware analysis.",
            )
        ]
    return []


def _check_cpu_op_shapes_detailed(ctx) -> List[TraceCheck]:
    events, thr = ctx["events"], ctx["thr"]
    cpu_ops = [e for e in events if e.get("cat") == "cpu_op"]
    if not cpu_ops:
        return []
    with_shapes = sum(
        1
        for e in cpu_ops
        if (e.get("args") or {}).get("Input Dims")
        and any(d for d in (e.get("args") or {}).get("Input Dims", []))
    )
    coverage = with_shapes / len(cpu_ops)
    if coverage < thr.shape_coverage_fail:
        return [
            TraceCheck(
                Status.WARN,
                f"Only {coverage:.0%} of cpu_ops have shapes (< {thr.shape_coverage_fail:.0%}).",
            )
        ]
    if coverage < thr.shape_coverage_warn:
        return [
            TraceCheck(
                Status.WARN,
                f"{coverage:.0%} of cpu_ops have shapes (< {thr.shape_coverage_warn:.0%}).",
            )
        ]
    return []


def _check_cpu_gpu_ratio(ctx) -> List[TraceCheck]:
    events, thr = ctx["events"], ctx["thr"]
    cpu_ops = sum(1 for e in events if e.get("cat") == "cpu_op")
    gpu_events = sum(1 for e in events if e.get("cat") in GPU_KERNEL_CATEGORIES)
    if gpu_events == 0 or cpu_ops == 0:
        return []
    ratio = cpu_ops / gpu_events
    if not (thr.cpu_gpu_ratio_min <= ratio <= thr.cpu_gpu_ratio_max):
        return [
            TraceCheck(
                Status.WARN,
                f"Suspicious CPU:GPU event ratio {ratio:.2f} "
                f"(expected {thr.cpu_gpu_ratio_min}–{thr.cpu_gpu_ratio_max}).",
            )
        ]
    return []


# ---------------------------------------------------------------------------
# Detailed pre-report checks (TRACELENS_DETAILED_QUALITY_CHECKS=1)
# ---------------------------------------------------------------------------
def _check_runtime_variability(ctx) -> List[TraceCheck]:
    """Flag kernel groups whose per-invocation duration varies too much."""
    events, thr = ctx["events"], ctx["thr"]
    groups: Dict[str, List[float]] = defaultdict(list)
    for e in events:
        if e.get("cat") not in GPU_KERNEL_CATEGORIES or e.get("dur") is None:
            continue
        args = e.get("args") or {}
        sig = (
            f"{e.get('name')}|grid={args.get('grid', '')}|block={args.get('block', '')}"
        )
        groups[sig].append(float(e["dur"]))

    worst_cv = 0.0
    worst_name = None
    failed_count = 0
    eligible = 0
    for key, durs in groups.items():
        if len(durs) < thr.runtime_min_samples:
            continue
        if sum(durs) / len(durs) < thr.runtime_min_dur_us:
            continue
        eligible += 1
        cv = coefficient_of_variation(durs)
        if cv > worst_cv:
            worst_cv, worst_name = cv, key
        if cv >= thr.runtime_cv_fail:
            failed_count += 1

    if eligible == 0:
        return []
    fail_frac = failed_count / eligible
    if fail_frac > thr.runtime_max_flagged_frac:
        return [
            TraceCheck(
                Status.WARN,
                f"{failed_count}/{eligible} kernel groups exceed CV>={thr.runtime_cv_fail} "
                f"(worst={worst_cv:.2f} '{worst_name}').",
                {"eligible_groups": eligible, "worst_cv": round(worst_cv, 3)},
            )
        ]
    return []


def _check_gpu_busy_idle(ctx) -> List[TraceCheck]:
    events, thr = ctx["events"], ctx["thr"]
    gpu_events = [e for e in events if e.get("cat") in GPU_KERNEL_CATEGORIES]
    if not gpu_events:
        return []
    m = busy_idle_from_events(gpu_events)
    idle_pct = m.get("idle_pct", 0.0)
    if idle_pct >= thr.idle_pct_fail:
        return [
            TraceCheck(
                Status.WARN,
                f"GPU idle {idle_pct:.1f}% (>= {thr.idle_pct_fail}%).",
                m,
            )
        ]
    if idle_pct >= thr.idle_pct_warn:
        return [
            TraceCheck(
                Status.WARN,
                f"GPU idle {idle_pct:.1f}% (>= {thr.idle_pct_warn}%).",
                m,
            )
        ]
    return []


def _check_jax_metadata_richness(ctx) -> List[TraceCheck]:
    """Fraction of GPU events carrying rich HLO metadata."""
    events, thr = ctx["events"], ctx["thr"]
    gpu_events = [e for e in events if e.get("cat") in GPU_KERNEL_CATEGORIES]
    n = len(gpu_events)
    if n == 0:
        return []
    fields = ("hlo_op", "hlo_module", "tf_op")
    field_counts = {f: 0 for f in fields}
    for e in gpu_events:
        args = e.get("args") or {}
        for f in fields:
            if args.get(f):
                field_counts[f] += 1
    coverage = {f: round(c / n, 3) for f, c in field_counts.items()}
    min_cov = min(coverage.values())
    if min_cov < thr.shape_coverage_fail:
        return [
            TraceCheck(
                Status.WARN,
                f"Sparse HLO metadata (min coverage {min_cov:.0%}): {coverage}.",
                coverage,
            )
        ]
    return []


def _check_kernels_dropped_correlation(ctx) -> List[TraceCheck]:
    """Check for dropped kernels via correlation ID after tree building."""
    tree = ctx.get("tree")
    if tree is None or not hasattr(tree, "runtime_event_uids"):
        return []
    dropped = 0
    for uid in tree.runtime_event_uids:
        event = tree.events_by_uid[uid]
        if "launch" not in event.get("name", "").lower():
            continue
        if not event.get("gpu_events"):
            dropped += 1
    if dropped == 0:
        return []
    return [
        TraceCheck(
            Status.WARN,
            f"{dropped} runtime launch event(s) have no corresponding "
            f"GPU kernel — kernels were likely dropped by the profiler.",
        )
    ]


# ---------------------------------------------------------------------------
# Post-report checks (run on generated perf report DataFrames)
# ---------------------------------------------------------------------------
_SHEETS = {
    "pytorch": {
        "timeline": "gpu_timeline",
        "op_summary": "ops_summary",
        "op_category": "ops_summary_by_category",
        "unique_args": "ops_unique_args",
        "count_col": "Count",
        "category_col": "op category",
    },
    "jax": {
        "timeline": "gpu_timeline",
        "op_summary": "kernel_launchers_summary",
        "op_category": "kernel_launchers_summary_by_category",
        "unique_args": "kernel_launchers_unique_args",
        "count_col": "Count",
        "category_col": "op category",
    },
}

_ATTN_FWD_SHEETS = ("op_fa_fwd",)
_ATTN_BWD_SHEETS = ("op_fa_bwd",)


def _check_ops_have_shapes(ctx) -> List[TraceCheck]:
    dfs, sheets, thr = ctx["dfs"], ctx["sheets"], ctx["thr"]
    df = dfs.get(sheets["unique_args"])
    if df is None or "Input Dims" not in getattr(df, "columns", []):
        return []
    total = len(df)
    if total == 0:
        return []

    def _has_shape(v) -> bool:
        if v is None:
            return False
        return str(v).strip() not in ("", "[]", "[[]]", "nan", "None", "()")

    with_shape = int(df["Input Dims"].apply(_has_shape).sum())
    coverage = with_shape / total
    if coverage < thr.shape_coverage_fail:
        return [
            TraceCheck(
                Status.WARN,
                f"Only {coverage:.0%} of ops have shapes in perf report.",
            )
        ]
    return []


def _check_high_idle_ops(ctx) -> List[TraceCheck]:
    """Flag ops with a large gap between subtree wall time and kernel time."""
    dfs, sheets, thr = ctx["dfs"], ctx["sheets"], ctx["thr"]
    df = dfs.get(sheets["unique_args"])
    cols = set(df.columns) if df is not None else set()
    need = {"total_subtree_kernel_time_sum", "total_direct_kernel_time_sum"}
    if df is None or not need.issubset(cols):
        return []
    offenders = 0
    for _, row in df.iterrows():
        subtree = float(row["total_subtree_kernel_time_sum"] or 0)
        direct = float(row["total_direct_kernel_time_sum"] or 0)
        if subtree <= 0:
            continue
        gap_pct = 100.0 * max(0.0, subtree - direct) / subtree
        if gap_pct >= thr.op_high_idle_pct:
            offenders += 1
    if offenders:
        return [
            TraceCheck(
                Status.WARN,
                f"{offenders} op(s) with >= {thr.op_high_idle_pct}% non-kernel time.",
            )
        ]
    return []


def _check_kernel_time_coverage(ctx) -> List[TraceCheck]:
    """Check that kernel time in reports accounts for GPU computation time."""
    dfs, sheets = ctx.get("dfs"), ctx.get("sheets")
    if not dfs or not sheets:
        return []
    tl = dfs.get(sheets.get("timeline", ""))
    if tl is None or "time ms" not in getattr(tl, "columns", []):
        return []
    comp_rows = tl[tl["type"] == "computation_time"]
    if comp_rows.empty:
        return []
    computation_time_us = float(comp_rows["time ms"].values[0]) * 1e3

    def _sum_time(df, *cols):
        for col in cols:
            if col in getattr(df, "columns", []):
                return df[col].sum()
        return None

    findings = []
    kl = dfs.get(sheets.get("op_summary", ""))
    if kl is not None:
        kl_us = _sum_time(
            kl, "total_direct_kernel_time_sum", "total_direct_kernel_time"
        )
        if kl_us is not None and kl_us < computation_time_us:
            findings.append(
                TraceCheck(
                    Status.WARN,
                    f"kernel_launchers kernel time ({kl_us:.0f} µs) is less than "
                    f"computation time ({computation_time_us:.0f} µs).",
                )
            )

    up = dfs.get("unified_perf_summary")
    if up is not None:
        up_us = _sum_time(up, "Kernel Time (µs)_sum", "Kernel Time (µs)")
        if up_us is not None and up_us < computation_time_us:
            findings.append(
                TraceCheck(
                    Status.WARN,
                    f"unified_perf kernel time ({up_us:.0f} µs) is less than "
                    f"computation time ({computation_time_us:.0f} µs).",
                )
            )

    return findings


def _check_kernel_count_alignment(ctx) -> List[TraceCheck]:
    """Check that kernel counts in report match ground truth from tree events."""
    tree, dfs, sheets = ctx.get("tree"), ctx.get("dfs"), ctx.get("sheets")
    if not tree or not dfs or not sheets:
        return []
    events = tree.events

    tree_counts = Counter(
        e["name"]
        for e in events
        if e.get("cat") in GPU_KERNEL_CATEGORIES
        and not TraceEventUtils.is_communication_string(e.get("name", ""))
    )
    if not tree_counts:
        return []
    tree_total = sum(tree_counts.values())

    def _extract_counts(df):
        for col in ["kernel_details_summary", "kernel_details"]:
            if col not in getattr(df, "columns", []):
                continue
            counts = Counter()
            for kd in df[col]:
                if isinstance(kd, list):
                    for d in kd:
                        counts[d["name"]] += d.get("count", 1)
            return counts
        return None

    findings = []
    for label, df in [
        ("kernel_launchers", dfs.get(sheets.get("op_summary", ""))),
        ("unified_perf", dfs.get("unified_perf_summary")),
    ]:
        if df is None:
            continue
        df_counts = _extract_counts(df)
        if df_counts is None:
            continue
        df_total = sum(df_counts.values())
        if df_total != tree_total:
            findings.append(
                TraceCheck(
                    Status.WARN,
                    f"{label} kernel count ({df_total}) does not match "
                    f"tree kernel count ({tree_total}).",
                )
            )
    return findings


def _infer_num_gpus(dfs: Dict) -> Optional[int]:
    """Number of GPU devices, from the perf-report ``gpu_timeline`` sheet."""
    tl = dfs.get("gpu_timeline")
    if tl is None or "gpu_pid" not in getattr(tl, "columns", []):
        return None
    try:
        n = int(tl["gpu_pid"].nunique())
    except Exception:
        return None
    return n or None


def _attention_observed(dfs: Dict, sheet_names: Sequence[str]) -> Dict[str, Any]:
    """Sum observed launches, count unique kernels, and list per-kernel counts."""
    launches = 0
    unique = 0
    found = []
    for name in sheet_names:
        df = dfs.get(name)
        if df is None or "name_count" not in getattr(df, "columns", []):
            continue
        found.append(name)
        unique += int(len(df))
        launches += int(df["name_count"].fillna(0).astype(int).sum())
    return {"launches": launches, "unique_kernels": unique, "sheets": found}


def _check_attention_kernel_count(ctx) -> List[TraceCheck]:
    """Sanity-check observed attention launches against the model config."""
    dfs, cfg = ctx["dfs"], ctx["attn_cfg"]
    if not cfg.enabled:
        return []
    fwd = _attention_observed(dfs, _ATTN_FWD_SHEETS)
    bwd = _attention_observed(dfs, _ATTN_BWD_SHEETS)
    if not fwd["sheets"] and not bwd["sheets"]:
        return []
    num_gpus = cfg.num_gpus or _infer_num_gpus(dfs)
    if not num_gpus:
        return []
    problems = []
    for label, obs, calls in (
        ("forward", fwd, cfg.fwd_attn_calls),
        ("backward", bwd, cfg.bwd_attn_calls),
    ):
        if not obs["sheets"]:
            continue
        expected = (
            num_gpus * cfg.num_steps * cfg.num_layers * calls * obs["unique_kernels"]
        )
        observed = obs["launches"]
        tol_abs = cfg.tol * expected
        if abs(observed - expected) > tol_abs:
            problems.append(f"{label}: expected {expected} != observed {observed}")
    if problems:
        return [
            TraceCheck(
                Status.WARN,
                "Attention launches do not match config: " + "; ".join(problems),
            )
        ]
    return []


# ---------------------------------------------------------------------------
# Check registry and public API
# ---------------------------------------------------------------------------
# Each entry: (fn, pre_report, detailed, framework)
# fn accepts a single ctx dict.
# pre_report: True = pre-report check, False = post-report check.
# detailed: True = requires TRACELENS_DETAILED_QUALITY_CHECKS=1.
_accumulated_reports: List[TraceCheckReport] = []

# framework: None = all, "pytorch" = PyTorch only, "jax" = JAX only.
CHECK_REGISTRY = [
    # Pre-report, basic
    (_check_trace_file_size, True, False, None),
    (_check_kernels_present, True, False, "pytorch"),
    (_check_call_stack, True, False, None),
    (_check_graph_mode, True, False, None),
    (_check_cpu_op_shapes_basic, True, False, "pytorch"),
    # Pre-report, detailed
    (_check_cpu_op_shapes_detailed, True, True, "pytorch"),
    (_check_cpu_gpu_ratio, True, True, "pytorch"),
    (_check_runtime_variability, True, True, None),
    (_check_gpu_busy_idle, True, True, None),
    (_check_jax_metadata_richness, True, True, "jax"),
    # Post-report, basic
    (_check_kernels_dropped_correlation, False, False, None),
    (_check_kernel_time_coverage, False, False, None),
    (_check_kernel_count_alignment, False, False, None),
    # Post-report, detailed
    (_check_ops_have_shapes, False, True, None),
    (_check_high_idle_ops, False, True, None),
    (_check_attention_kernel_count, False, True, "jax"),
]


def run_trace_checks(
    events: Sequence[Dict[str, Any]] = (),
    capture_trace_filepath: Optional[str] = None,
    filepath: Optional[str] = None,
    framework: Optional[str] = None,
    pre_report: bool = True,
    tree=None,
    dfs: Optional[Dict] = None,
    attn_cfg: Optional[AttnCountConfig] = None,
) -> TraceCheckReport:
    """Run trace checks.

    Args:
        pre_report: If True, run pre-report checks. If False, run post-report
            checks (including post-tree checks that need the built tree).
        tree: A built TraceToTree (for post-report correlation-ID kernel drop check).
        dfs: Report DataFrames (required for post-report DataFrame checks).

    Basic checks always run. Detailed checks run when
    ``TRACELENS_DETAILED_QUALITY_CHECKS=1`` is set. Checks are filtered
    by ``framework`` (None runs only general checks).
    """
    findings: List[TraceCheck] = []
    ran_checks: List[str] = []
    detailed = parse_bool_string(os.environ.get("TRACELENS_DETAILED_QUALITY_CHECKS"))
    thr = QualityThresholds()
    sheets = _SHEETS.get(framework or "pytorch", _SHEETS["pytorch"])
    ctx = {
        "events": events,
        "capture_trace_filepath": capture_trace_filepath,
        "filepath": filepath,
        "thr": thr,
        "tree": tree,
        "dfs": dfs,
        "sheets": sheets,
        "attn_cfg": attn_cfg or AttnCountConfig(),
    }

    for fn, is_pre_report, is_detailed, check_fw in CHECK_REGISTRY:
        if is_pre_report != pre_report:
            continue
        if is_detailed and not detailed:
            continue
        if check_fw is not None and check_fw != framework:
            continue
        name = fn.__name__
        ran_checks.append(name)
        check_findings = fn(ctx)
        for f in check_findings:
            f.check_name = name
        findings.extend(check_findings)

    report = TraceCheckReport(findings=findings, ran_checks=ran_checks)
    _accumulated_reports.append(report)
    return report


def write_accumulated_json(output_dir: str) -> None:
    """Write all accumulated check results to trace_check.json and clear."""
    if not output_dir or not _accumulated_reports:
        return
    merged = TraceCheckReport()
    for r in _accumulated_reports:
        merged.merge(r)
    merged.write_json(output_dir)
    _accumulated_reports.clear()
