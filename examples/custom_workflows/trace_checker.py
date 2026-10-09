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
``TRACELENS_DETAILED_HEALTH_CHECKS=1`` is set. All checks can be skipped
with ``TRACELENS_SKIP_HEALTH_CHECK=1``.
"""

from __future__ import annotations

import enum
import functools
import math
import os
import statistics
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .util import GPU_KERNEL_CATEGORIES, GRAPH_LAUNCH_NAMES, merge_intervals


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
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TraceCheckReport:
    findings: List[TraceCheck] = field(default_factory=list)

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


# ---------------------------------------------------------------------------
# Constants and thresholds
# ---------------------------------------------------------------------------
_MIN_TRACE_BYTES = 100_000
_MAX_TRACE_BYTES = 5_000_000_000
_MIN_CPU_OP_SHAPES = 10


@dataclass
class QualityThresholds:
    drop_num_windows: int = 20
    drop_empty_window_frac: float = 0.5
    drop_fail_frac: float = 0.15

    runtime_cv_warn: float = 0.25
    runtime_cv_fail: float = 0.75
    runtime_min_samples: int = 5
    runtime_min_dur_us: float = 2.0
    runtime_max_flagged_frac: float = 0.10

    idle_pct_warn: float = 30.0
    idle_pct_fail: float = 60.0

    shape_coverage_warn: float = 0.80
    shape_coverage_fail: float = 0.40

    cpu_gpu_ratio_min: float = 0.2
    cpu_gpu_ratio_max: float = 50.0

    attn_name_patterns: List[str] = field(
        default_factory=lambda: [
            "fmha",
            "flash_attn",
            "flash_fwd",
            "flash_bwd",
            "fused_attn",
            "sdpa",
            "scaled_dot_product",
            "attention",
            "_attn",
        ]
    )

    op_high_idle_pct: float = 40.0


@dataclass
class AttnCountConfig:
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
    n = len(values)
    if n == 0:
        return 0.0
    mean = sum(values) / n
    if mean == 0:
        return 0.0
    return statistics.pstdev(values) / abs(mean)


def gcd_of_counts(counts: Iterable[int]) -> int:
    positives = [int(c) for c in counts if c and int(c) > 0]
    if not positives:
        return 0
    return functools.reduce(math.gcd, positives)


def busy_idle_from_events(events: Sequence[dict]) -> Dict[str, float]:
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


def matches_any(name: str, patterns: Iterable[str]) -> bool:
    low = (name or "").lower()
    return any(p.lower() in low for p in patterns)


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
                "No GPU kernel events found in trace.",
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
# Detailed pre-report checks (TRACELENS_DETAILED_HEALTH_CHECKS=1)
# ---------------------------------------------------------------------------
def _check_kernels_dropped_windowed(ctx) -> List[TraceCheck]:
    events, thr = ctx["events"], ctx["thr"]
    kernels = [e for e in events if e.get("cat") in GPU_KERNEL_CATEGORIES]
    times = sorted(
        float(k["ts"]) for k in kernels if k.get("ts") is not None and k.get("dur")
    )
    if len(times) < thr.drop_num_windows:
        return []
    t0, t1 = times[0], times[-1]
    span = t1 - t0
    if span <= 0:
        return []
    nbins = thr.drop_num_windows
    width = span / nbins
    counts = [0] * nbins
    for t in times:
        counts[min(int((t - t0) / width), nbins - 1)] += 1
    nonzero = [c for c in counts if c > 0]
    median = statistics.median(nonzero) if nonzero else 0
    floor = thr.drop_empty_window_frac * median
    starved = [i for i in range(1, nbins - 1) if counts[i] < floor]
    num_interior = max(1, nbins - 2)
    starved_frac = len(starved) / num_interior
    if starved_frac > thr.drop_fail_frac:
        return [
            TraceCheck(
                Status.WARN,
                f"{len(starved)}/{num_interior} interior timeline windows starved of "
                f"kernels — kernels likely dropped intermittently.",
                {"starved_windows": starved, "starved_frac": round(starved_frac, 3)},
            )
        ]
    return []


def _check_runtime_variability(ctx) -> List[TraceCheck]:
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


def _check_kernel_counts(ctx) -> List[TraceCheck]:
    events, thr = ctx["events"], ctx["thr"]
    kernels = [e for e in events if e.get("cat") in GPU_KERNEL_CATEGORIES]
    if not kernels:
        return []
    name_counts = Counter(k.get("name", "") for k in kernels)
    attn_total = sum(
        c
        for name, c in name_counts.items()
        if matches_any(name, thr.attn_name_patterns)
    )
    if attn_total == 0:
        return [
            TraceCheck(
                Status.WARN,
                f"{len(name_counts)} unique kernels; no attention/FMHA kernel detected.",
                {
                    "num_unique_kernels": len(name_counts),
                    "count_gcd": gcd_of_counts(name_counts.values()),
                },
            )
        ]
    return []


def _check_jax_metadata_richness(ctx) -> List[TraceCheck]:
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


# ---------------------------------------------------------------------------
# Post-tree checks (run after tree building)
# ---------------------------------------------------------------------------
def _check_kernels_dropped_correlation(tree) -> List[TraceCheck]:
    if not hasattr(tree, "runtime_event_uids"):
        return []
    dropped = 0
    for uid in tree.runtime_event_uids:
        event = tree.events_by_uid[uid]
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


def _check_report_generated(ctx) -> List[TraceCheck]:
    dfs, sheets = ctx["dfs"], ctx["sheets"]
    expected = {sheets["timeline"], sheets["op_summary"], sheets["unique_args"]}
    missing = sorted(expected - set(dfs.keys()))
    if missing:
        return [
            TraceCheck(
                Status.WARN,
                f"Perf report missing expected sheets: {missing}.",
            )
        ]
    return []


def _check_idle_timeline(ctx) -> List[TraceCheck]:
    dfs, sheets, thr = ctx["dfs"], ctx["sheets"], ctx["thr"]
    tl = dfs.get(sheets["timeline"])
    if tl is None or "type" not in getattr(tl, "columns", []):
        return []
    idle_rows = tl[tl["type"] == "idle_time"]
    if idle_rows.empty:
        return []
    idle_pct = float(idle_rows["percent"].mean())
    if idle_pct >= thr.idle_pct_fail:
        return [
            TraceCheck(
                Status.WARN,
                f"GPU idle {idle_pct:.1f}% (>= {thr.idle_pct_fail}%).",
            )
        ]
    if idle_pct >= thr.idle_pct_warn:
        return [
            TraceCheck(
                Status.WARN,
                f"GPU idle {idle_pct:.1f}% (>= {thr.idle_pct_warn}%).",
            )
        ]
    return []


def _check_op_count_consistency(ctx) -> List[TraceCheck]:
    dfs, sheets = ctx["dfs"], ctx["sheets"]
    summary = dfs.get(sheets["op_summary"])
    col = sheets["count_col"]
    if summary is None or col not in getattr(summary, "columns", []):
        return []
    counts = [int(c) for c in summary[col].tolist() if c and int(c) > 0]
    if not counts:
        return []
    divisor = gcd_of_counts(counts)
    if divisor <= 1:
        return [
            TraceCheck(
                Status.WARN,
                f"{len(counts)} ops have gcd=1 (no common per-step multiple).",
            )
        ]
    return []


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


def _check_sdpa_count(ctx) -> List[TraceCheck]:
    dfs, sheets, thr = ctx["dfs"], ctx["sheets"], ctx["thr"]
    cat_df = dfs.get(sheets["op_category"])
    cat_col = sheets["category_col"]
    total = 0
    if cat_df is not None and cat_col in getattr(cat_df, "columns", []):
        count_col = (
            sheets["count_col"] if sheets["count_col"] in cat_df.columns else None
        )
        for _, row in cat_df.iterrows():
            cat = str(row.get(cat_col, ""))
            if matches_any(cat, thr.attn_name_patterns):
                total += int(row[count_col]) if count_col else 1
    if total == 0:
        return [
            TraceCheck(
                Status.WARN,
                "No SDPA / attention ops found in perf report.",
            )
        ]
    return []


def _infer_num_gpus(dfs: Dict) -> Optional[int]:
    tl = dfs.get("gpu_timeline")
    if tl is None or "gpu_pid" not in getattr(tl, "columns", []):
        return None
    try:
        n = int(tl["gpu_pid"].nunique())
    except Exception:
        return None
    return n or None


def _attention_observed(dfs: Dict, sheet_names: Sequence[str]) -> Dict[str, Any]:
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
# detailed: True = requires TRACELENS_DETAILED_HEALTH_CHECKS=1.
# framework: None = all, "pytorch" = PyTorch only, "jax" = JAX only.
CHECK_REGISTRY = [
    # Pre-report, basic
    (_check_trace_file_size, True, False, None),
    (_check_kernels_present, True, False, None),
    (_check_call_stack, True, False, None),
    (_check_graph_mode, True, False, None),
    (_check_cpu_op_shapes_basic, True, False, "pytorch"),
    # Pre-report, detailed
    (_check_cpu_op_shapes_detailed, True, True, "pytorch"),
    (_check_cpu_gpu_ratio, True, True, "pytorch"),
    (_check_kernels_dropped_windowed, True, True, None),
    (_check_runtime_variability, True, True, None),
    (_check_gpu_busy_idle, True, True, None),
    (_check_kernel_counts, True, True, None),
    (_check_jax_metadata_richness, True, True, "jax"),
    # Post-report, detailed
    (_check_report_generated, False, True, None),
    (_check_idle_timeline, False, True, None),
    (_check_op_count_consistency, False, True, None),
    (_check_ops_have_shapes, False, True, None),
    (_check_high_idle_ops, False, True, None),
    (_check_sdpa_count, False, True, None),
    (_check_attention_kernel_count, False, True, "jax"),
]


def run_trace_checks(
    events: Sequence[Dict[str, Any]] = (),
    trace_metadata: Optional[Dict[str, Any]] = None,
    capture_trace_filepath: Optional[str] = None,
    filepath: Optional[str] = None,
    framework: Optional[str] = None,
    pre_report: bool = True,
    dfs: Optional[Dict] = None,
    attn_cfg: Optional[AttnCountConfig] = None,
) -> TraceCheckReport:
    """Run trace checks.

    Args:
        pre_report: If True, run pre-report checks. If False, run post-report checks.
        dfs: Report DataFrames (required for post-report checks).

    Basic checks always run. Detailed checks run when
    ``TRACELENS_DETAILED_HEALTH_CHECKS=1`` is set. Checks are filtered
    by ``framework`` (None runs only general checks).
    """
    findings: List[TraceCheck] = []
    detailed = os.environ.get("TRACELENS_DETAILED_HEALTH_CHECKS") == "1"
    thr = QualityThresholds()
    sheets = _SHEETS.get(framework or "pytorch", _SHEETS["pytorch"])
    ctx = {
        "events": events,
        "capture_trace_filepath": capture_trace_filepath,
        "filepath": filepath,
        "thr": thr,
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
        findings.extend(fn(ctx))

    return TraceCheckReport(findings=findings)


def run_post_tree_checks(tree) -> TraceCheckReport:
    """Run checks that require a built trace tree.

    Called after ``tree.build_tree()`` in ``TreePerfAnalyzer``.
    """
    return TraceCheckReport(findings=_check_kernels_dropped_correlation(tree))
