###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Trace quality checks for PyTorch and JAX profiler traces.

Given a trace path and a framework (``"pytorch"`` or ``"jax"``), returns a
:class:`QualityReport`.

* Phase 1 -- kernel presence, drop detection, runtime variability, GPU
  busy/idle, kernel-count structure and metadata richness, off the raw JSON.
* Phase 2 -- checks derived from TraceLens perf reports, plus (JAX) an optional
  attention-kernel-count sanity check when the model config is supplied.

CLI (runs both phases by default)::

    python examples/custom_workflows/trace_checker.py TRACE.trace.json.gz --framework pytorch --phase all

    python examples/custom_workflows/trace_checker.py --json_trace TRACE.trace.json.gz --framework jax --phase all --xplane-pb-path TRACE.xplane.pb

Optional outputs and the JAX attention-kernel-count check
(expected = num_gpus * num_steps * num_layers * calls * unique_kernels; num_gpus
and unique_kernels are inferred, --num-layers and --num-steps enable the check)::

    python examples/custom_workflows/trace_checker.py --json_trace TRACE.trace.json.gz --framework jax --xplane-pb-path TRACE.xplane.pb \\
        --output-dir ./perf_csvs --json ./report.json \\
        --num-layers 4 --num-steps 3 --fwd-attn-calls 2 --bwd-attn-calls 1

Set ``TRACELENS_VERBOSE_NATIVE_LOGS=1`` to keep the raw native XLA/xprof logs.
"""

from __future__ import annotations

import argparse
import inspect
import enum
import functools
import json
import logging
import math
import os
import statistics
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from TraceLens import GPUEventAnalyser
from TraceLens.util import DataLoader

logger = logging.getLogger(__name__)

FRAMEWORK_PYTORCH = "pytorch"
FRAMEWORK_JAX = "jax"
VALID_FRAMEWORKS = (FRAMEWORK_PYTORCH, FRAMEWORK_JAX)

# Categories treated as GPU work in a PyTorch trace.
PT_GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}


# ===========================================================================
# Result data model
# ===========================================================================
class Status(enum.Enum):
    """Outcome of a single quality check (ordered PASS < SKIP < WARN < FAIL)."""

    PASS = "PASS"
    SKIP = "SKIP"
    WARN = "WARN"
    FAIL = "FAIL"

    @property
    def severity(self) -> int:
        return {"PASS": 0, "SKIP": 1, "WARN": 2, "FAIL": 3}[self.value]


@dataclass
class QualityResult:
    check_id: str
    name: str
    status: Status
    message: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "check_id": self.check_id,
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "metrics": self.metrics,
        }


@dataclass
class QualityReport:
    trace_path: str
    framework: str
    results: List[QualityResult] = field(default_factory=list)

    def add(self, result: QualityResult) -> QualityResult:
        self.results.append(result)
        return result

    def extend(self, results: List[QualityResult]) -> None:
        self.results.extend(results)

    @property
    def worst_status(self) -> Status:
        if not self.results:
            return Status.SKIP
        return max((r.status for r in self.results), key=lambda s: s.severity)

    @property
    def ok(self) -> bool:
        """True when no check FAILED (WARN/SKIP are tolerated)."""
        return all(r.status is not Status.FAIL for r in self.results)

    def get(self, check_id: str) -> Optional[QualityResult]:
        for r in self.results:
            if r.check_id == check_id:
                return r
        return None

    def counts(self) -> Dict[str, int]:
        out = {s.value: 0 for s in Status}
        for r in self.results:
            out[r.status.value] += 1
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_path": self.trace_path,
            "framework": self.framework,
            "ok": self.ok,
            "worst_status": self.worst_status.value,
            "counts": self.counts(),
            "results": [r.to_dict() for r in self.results],
        }

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        lines = [
            f"Trace quality report: {self.trace_path}",
            f"  framework={self.framework}  worst={self.worst_status.value}  "
            f"ok={self.ok}  counts={self.counts()}",
            "",
        ]
        width = max((len(r.check_id) for r in self.results), default=10)
        for r in self.results:
            lines.append(f"  [{r.status.value:<4}] {r.check_id:<{width}}  {r.message}")
        return "\n".join(lines)


# ===========================================================================
# Tunable thresholds
# ===========================================================================
@dataclass
class QualityThresholds:
    # kernel presence
    min_kernels: int = 100

    # dropped-kernel detection
    drop_num_windows: int = 20
    drop_empty_window_frac: float = 0.5
    drop_fail_frac: float = 0.15

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

    # attention / SDPA detection
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

    # phase 2: high-idle ops
    op_high_idle_pct: float = 40.0


@dataclass
class AttnCountConfig:
    """Config for the attention-kernel-count check.

    expected = num_gpus * num_steps * num_layers * calls * unique_kernels (per
    pass). num_gpus and unique_kernels are inferred; num_layers and num_steps
    must be supplied (else the check SKIPs and just reports observed counts).
    """

    num_layers: Optional[int] = None
    num_steps: Optional[int] = None
    num_gpus: Optional[int] = None
    fwd_attn_calls: int = 1
    bwd_attn_calls: int = 1
    tol: float = 0.0  # relative tolerance; 0.0 == exact

    @property
    def enabled(self) -> bool:
        return self.num_layers is not None and self.num_steps is not None


def coefficient_of_variation(values: Sequence[float]) -> float:
    """std / mean (population std). 0.0 if empty or mean is 0."""
    n = len(values)
    if n == 0:
        return 0.0
    mean = sum(values) / n
    if mean == 0:
        return 0.0
    return statistics.pstdev(values) / abs(mean)


def gcd_of_counts(counts: Iterable[int]) -> int:
    """GCD ("largest common denominator") of positive counts."""
    positives = [int(c) for c in counts if c and int(c) > 0]
    if not positives:
        return 0
    return functools.reduce(math.gcd, positives)


def merge_intervals(intervals):
    """Merge overlapping ``(start, end)`` intervals."""
    if not intervals:
        return []
    return GPUEventAnalyser.merge_intervals(list(intervals))


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


def matches_any(name: str, patterns: Iterable[str]) -> bool:
    low = (name or "").lower()
    return any(p.lower() in low for p in patterns)


def _round_metric(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return round(value, 3)
    return value


@dataclass
class PytorchTrace:
    path: str
    raw_events: List[dict]
    by_cat: Dict[str, List[dict]] = field(default_factory=dict)

    @property
    def kernels(self) -> List[dict]:
        return self.by_cat.get("kernel", [])

    @property
    def cpu_ops(self) -> List[dict]:
        return self.by_cat.get("cpu_op", [])

    @property
    def gpu_events(self) -> List[dict]:
        out: List[dict] = []
        for cat in PT_GPU_CATS:
            out.extend(self.by_cat.get(cat, []))
        return out


@dataclass
class JaxJsonTrace:
    path: str
    raw_events: List[dict]
    pid_name: Dict[str, str] = field(default_factory=dict)
    gpu_events: List[dict] = field(default_factory=list)
    gpu_events_by_device: Dict[str, List[dict]] = field(default_factory=dict)
    host_events: List[dict] = field(default_factory=list)


def load_pytorch(path: str) -> PytorchTrace:
    """Load a PyTorch trace, bucket events by ``cat`` and assign each a ``UID``."""
    data = DataLoader.load_data(path)
    events = data.get("traceEvents", [])
    by_cat: Dict[str, List[dict]] = defaultdict(list)
    for i, e in enumerate(events):
        if "UID" not in e:
            e["UID"] = i
        by_cat[e.get("cat", "unknown")].append(e)
    return PytorchTrace(path=path, raw_events=events, by_cat=dict(by_cat))


def load_jax_json(path: str) -> JaxJsonTrace:
    """Load a JAX xprof JSON export.

    GPU events are ``ph == "X"`` events on a ``/device:GPU:*`` process.
    """
    data = DataLoader.load_data(path)
    events = data.get("traceEvents", [])

    pid_name: Dict[str, str] = {}
    for e in events:
        if e.get("ph") == "M" and e.get("name") == "process_name":
            pid = e.get("pid")
            if pid is not None:
                pid_name[pid] = (e.get("args") or {}).get("name", "")

    gpu_events: List[dict] = []
    gpu_by_dev: Dict[str, List[dict]] = defaultdict(list)
    host_events: List[dict] = []
    for e in events:
        if e.get("ph") != "X":
            continue
        _coerce_numeric(e)  # xprof exports ts/dur as strings
        proc = pid_name.get(e.get("pid"), "")
        if proc.startswith("/device:GPU"):
            gpu_events.append(e)
            gpu_by_dev[proc].append(e)
        elif proc.startswith("/host"):
            host_events.append(e)
    return JaxJsonTrace(
        path=path,
        raw_events=events,
        pid_name=pid_name,
        gpu_events=gpu_events,
        gpu_events_by_device=dict(gpu_by_dev),
        host_events=host_events,
    )


def _coerce_numeric(event: dict) -> None:
    for key in ("ts", "dur"):
        val = event.get(key)
        if isinstance(val, str):
            try:
                event[key] = float(val)
            except ValueError:
                logger.debug("Skipping non-numeric %s value: %r", key, val)


def make_jax_analyzer(xplane_path: str):
    """Build a ``JaxTreePerfAnalyzer`` from an ``*.xplane.pb`` file."""
    from TraceLens.TreePerf.tree_perf import JaxTreePerfAnalyzer

    return JaxTreePerfAnalyzer.from_file(profile_filepath=xplane_path)


# Phase 1 checks (no external tooling)


def _check_kernels_present(
    kernels: Sequence[dict], thr: QualityThresholds
) -> QualityResult:
    n = len(kernels)
    if n == 0:
        status, msg = Status.FAIL, "no GPU kernel events found in trace"
    elif n < thr.min_kernels:
        status = Status.WARN
        msg = f"only {n} GPU kernels (< min_kernels={thr.min_kernels})"
    else:
        status, msg = Status.PASS, f"{n} GPU kernel events present"
    return QualityResult(
        "kernels_present", "GPU kernels present", status, msg, {"num_kernels": n}
    )


def _check_kernels_not_dropped(
    kernels: Sequence[dict], thr: QualityThresholds
) -> QualityResult:
    """Detect intermittent kernel dropping.

    Buckets the timeline into windows; an interior window that is nearly empty
    while its neighbors are busy indicates dropped kernels.
    """
    times = sorted(
        float(k["ts"]) for k in kernels if k.get("ts") is not None and k.get("dur")
    )
    if len(times) < thr.drop_num_windows:
        return QualityResult(
            "kernels_not_dropped",
            "Kernels not intermittently dropped",
            Status.SKIP,
            "too few kernels to bucket reliably",
            {"num_kernels": len(times)},
        )
    t0, t1 = times[0], times[-1]
    span = t1 - t0
    if span <= 0:
        return QualityResult(
            "kernels_not_dropped",
            "Kernels not intermittently dropped",
            Status.SKIP,
            "zero time span across kernels",
            {},
        )
    nbins = thr.drop_num_windows
    width = span / nbins
    counts = [0] * nbins
    for t in times:
        counts[min(int((t - t0) / width), nbins - 1)] += 1
    nonzero = [c for c in counts if c > 0]
    median = statistics.median(nonzero) if nonzero else 0
    floor = thr.drop_empty_window_frac * median
    # Only interior windows count as "drops" (ramp-up/tear-down live at edges).
    starved = [i for i in range(1, nbins - 1) if counts[i] < floor]
    num_interior = max(1, nbins - 2)
    starved_frac = len(starved) / num_interior
    metrics = {
        "num_windows": nbins,
        "window_counts": counts,
        "median_window_count": median,
        "starved_windows": starved,
        "starved_frac": round(starved_frac, 3),
    }
    if not starved:
        status = Status.PASS
        msg = f"kernel density even across {nbins} windows (median={median})"
    elif starved_frac > thr.drop_fail_frac:
        status = Status.FAIL
        msg = (
            f"{len(starved)}/{num_interior} interior windows starved of kernels "
            f"(idx={starved}); kernels likely dropped intermittently"
        )
    else:
        status = Status.WARN
        msg = (
            f"{len(starved)}/{num_interior} interior window(s) sparse "
            f"(idx={starved}); likely a phase boundary, verify not dropped kernels"
        )
    return QualityResult(
        "kernels_not_dropped",
        "Kernels not intermittently dropped",
        status,
        msg,
        metrics,
    )


def _check_runtime_variability(
    groups: Dict[str, List[float]], thr: QualityThresholds
) -> QualityResult:
    """Flag kernel groups whose per-invocation duration varies too much."""
    eligible = 0
    warned: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []
    worst_cv = 0.0
    worst_name = None
    for key, durs in groups.items():
        if len(durs) < thr.runtime_min_samples:
            continue
        if sum(durs) / len(durs) < thr.runtime_min_dur_us:
            continue
        eligible += 1
        cv = coefficient_of_variation(durs)
        if cv > worst_cv:
            worst_cv, worst_name = cv, key
        if cv >= thr.runtime_cv_warn:
            stats = {
                "group": key,
                "cv": round(cv, 3),
                "count": len(durs),
                "min_us": round(min(durs), 3),
                "median_us": round(statistics.median(durs), 3),
                "max_us": round(max(durs), 3),
                "mean_us": round(sum(durs) / len(durs), 3),
            }
            (failed if cv >= thr.runtime_cv_fail else warned).append(stats)

    failed.sort(key=lambda s: -s["cv"])
    warned.sort(key=lambda s: -s["cv"])
    metrics = {
        "eligible_groups": eligible,
        "num_warn": len(warned),
        "num_fail": len(failed),
        "worst_cv": round(worst_cv, 3),
        "worst_group": worst_name,
        "failed_groups": failed,
        "warned_groups": warned,
        "top_offenders": [
            {"group": s["group"], "cv": s["cv"]} for s in (failed + warned)[:10]
        ],
    }
    if eligible == 0:
        return QualityResult(
            "runtime_variability",
            "No spurious kernel-time variability",
            Status.SKIP,
            "no eligible kernel groups to assess",
            metrics,
        )
    fail_frac = len(failed) / eligible
    if fail_frac > thr.runtime_max_flagged_frac:
        status = Status.FAIL
        msg = (
            f"{len(failed)}/{eligible} groups exceed CV>={thr.runtime_cv_fail} "
            f"(worst={worst_cv:.2f} '{worst_name}')"
        )
        msg += _format_variability_detail(
            failed, "failed (CV>=%.2f)" % thr.runtime_cv_fail
        )
    elif failed or warned:
        status = Status.WARN
        msg = (
            f"{len(failed)} high / {len(warned)} moderate variability groups "
            f"(worst CV={worst_cv:.2f} '{worst_name}')"
        )
        if failed:
            msg += _format_variability_detail(
                failed, "failed (CV>=%.2f)" % thr.runtime_cv_fail
            )
    else:
        status = Status.PASS
        msg = f"kernel durations stable across {eligible} groups (worst CV={worst_cv:.2f})"
    return QualityResult(
        "runtime_variability",
        "No spurious kernel-time variability",
        status,
        msg,
        metrics,
    )


def _format_variability_detail(groups: List[Dict[str, Any]], header: str) -> str:
    """Multi-line listing of every flagged kernel group with duration stats."""
    if not groups:
        return ""
    lines = [
        f"\n    {len(groups)} kernel group(s) {header} "
        "[us: count / CV / min / median / max]:"
    ]
    for s in groups:
        lines.append(
            f"      - {s['group']}\n"
            f"          count={s['count']} CV={s['cv']} "
            f"min={s['min_us']} median={s['median_us']} max={s['max_us']} "
            f"mean={s['mean_us']}"
        )
    return "\n".join(lines)


def _check_busy_idle(
    metrics: Dict[str, float], thr: QualityThresholds
) -> QualityResult:
    idle_pct = metrics.get("idle_pct", 0.0)
    out = {k: _round_metric(v) for k, v in metrics.items()}
    if idle_pct >= thr.idle_pct_fail:
        status = Status.FAIL
        msg = f"GPU idle {idle_pct:.1f}% (>= {thr.idle_pct_fail}%)"
    elif idle_pct >= thr.idle_pct_warn:
        status = Status.WARN
        msg = f"GPU idle {idle_pct:.1f}% (>= {thr.idle_pct_warn}%)"
    else:
        status = Status.PASS
        msg = f"GPU idle {idle_pct:.1f}%, busy {100 - idle_pct:.1f}%"
    return QualityResult("gpu_busy_idle", "GPU busy/idle time", status, msg, out)


def _check_kernel_counts(name_counts: Counter, thr: QualityThresholds) -> QualityResult:
    """Report kernel-count structure: common divisor + attention kernel count."""
    if not name_counts:
        return QualityResult(
            "kernel_counts",
            "Kernel counts / attention presence",
            Status.FAIL,
            "no kernels to count",
            {},
        )
    divisor = gcd_of_counts(name_counts.values())
    attn = {
        name: c
        for name, c in name_counts.items()
        if matches_any(name, thr.attn_name_patterns)
    }
    attn_total = sum(attn.values())
    top_attn = sorted(attn.items(), key=lambda x: -x[1])[:5]
    metrics = {
        "num_unique_kernels": len(name_counts),
        "count_gcd": divisor,
        "attention_kernel_count": attn_total,
        "attention_kernels": [{"name": n, "count": c} for n, c in top_attn],
        "top_kernels": [{"name": n, "count": c} for n, c in name_counts.most_common(5)],
    }
    if attn_total == 0:
        status = Status.WARN
        msg = (
            f"{len(name_counts)} unique kernels (count gcd={divisor}); "
            "no attention/FMHA kernel detected"
        )
    else:
        status = Status.PASS
        msg = (
            f"{len(name_counts)} unique kernels (count gcd={divisor}); "
            f"attention kernels={attn_total}"
        )
    return QualityResult(
        "kernel_counts", "Kernel counts / attention presence", status, msg, metrics
    )


def run_pytorch_phase1(
    trace: PytorchTrace, thr: Optional[QualityThresholds] = None
) -> List[QualityResult]:
    thr = thr or QualityThresholds()
    kernels = trace.kernels
    results = [
        _check_kernels_present(kernels, thr),
        _check_kernels_not_dropped(kernels, thr),
    ]

    # Runtime variability grouped by (name, grid, block): GPU kernel events lack
    # tensor shapes, but grid/block launch dims disambiguate problem sizes so
    # only genuinely identical launches are compared.
    groups: Dict[str, List[float]] = defaultdict(list)
    for k in kernels:
        if k.get("dur") is None:
            continue
        args = k.get("args") or {}
        sig = (
            f"{k.get('name')}|grid={args.get('grid', '')}|block={args.get('block', '')}"
        )
        groups[sig].append(float(k["dur"]))
    results.append(_check_runtime_variability(groups, thr))

    results.append(_pytorch_busy_idle(trace, thr))
    results.append(
        _check_kernel_counts(Counter(k.get("name", "") for k in kernels), thr)
    )
    results.append(_check_cpu_op_shapes(trace, thr))
    results.append(_check_cpu_call_stack(trace, thr))
    return results


def _pytorch_busy_idle(trace: PytorchTrace, thr: QualityThresholds) -> QualityResult:
    gpu_events = trace.gpu_events
    if not gpu_events:
        return QualityResult(
            "gpu_busy_idle", "GPU busy/idle time", Status.FAIL, "no GPU events", {}
        )
    try:
        m = dict(GPUEventAnalyser(gpu_events).compute_metrics())
        total = m.get("total_time", 0.0)
        m["idle_pct"] = (100.0 * m.get("idle_time", 0.0) / total) if total else 0.0
    except Exception as exc:  # noqa: BLE001 - fall back to manual interval merge
        m = busy_idle_from_events(gpu_events)
        m["fallback_reason"] = f"GPUEventAnalyser failed: {exc}"
    return _check_busy_idle(m, thr)


def _check_cpu_op_shapes(trace: PytorchTrace, thr: QualityThresholds) -> QualityResult:
    cpu_ops = trace.cpu_ops
    if not cpu_ops:
        return QualityResult(
            "cpu_op_shapes",
            "CPU ops carry shapes (PyTorch, optional)",
            Status.WARN,
            "no cpu_op events in trace",
            {"num_cpu_ops": 0},
        )
    with_shapes = 0
    for op in cpu_ops:
        dims = (op.get("args") or {}).get("Input Dims")
        if dims and any(d for d in dims):
            with_shapes += 1
    coverage = with_shapes / len(cpu_ops)
    num_gpu = len(trace.gpu_events)
    cpu_gpu_ratio = (len(cpu_ops) / num_gpu) if num_gpu else float("inf")
    metrics = {
        "num_cpu_ops": len(cpu_ops),
        "cpu_ops_with_shapes": with_shapes,
        "shape_coverage": round(coverage, 3),
        "num_gpu_events": num_gpu,
        "cpu_gpu_ratio": round(cpu_gpu_ratio, 3),
    }
    msgs = []
    status = Status.PASS
    if coverage < thr.shape_coverage_fail:
        status = Status.FAIL
        msgs.append(f"only {coverage:.0%} of cpu_ops have shapes")
    elif coverage < thr.shape_coverage_warn:
        status = Status.WARN
        msgs.append(f"{coverage:.0%} of cpu_ops have shapes")
    else:
        msgs.append(f"{coverage:.0%} of cpu_ops have shapes")
    if not (thr.cpu_gpu_ratio_min <= cpu_gpu_ratio <= thr.cpu_gpu_ratio_max):
        status = max(status, Status.WARN, key=lambda s: s.severity)
        msgs.append(f"suspicious CPU:GPU ratio {cpu_gpu_ratio:.2f}")
    return QualityResult(
        "cpu_op_shapes",
        "CPU ops carry shapes (PyTorch, optional)",
        status,
        "; ".join(msgs),
        metrics,
    )


def _check_cpu_call_stack(trace: PytorchTrace, thr: QualityThresholds) -> QualityResult:
    py_funcs = trace.by_cat.get("python_function", [])
    metrics = {"num_python_function_events": len(py_funcs)}
    if py_funcs:
        status = Status.PASS
        msg = f"CPU call stack present ({len(py_funcs)} python_function events)"
    else:
        status = Status.WARN
        msg = "no python_function events (call stack not captured; with_stack=False)"
    return QualityResult(
        "cpu_call_stack",
        "CPU call stack present (PyTorch, optional)",
        status,
        msg,
        metrics,
    )


def run_jax_phase1(
    trace: JaxJsonTrace, thr: Optional[QualityThresholds] = None
) -> List[QualityResult]:
    thr = thr or QualityThresholds()
    gpu_events = trace.gpu_events
    results = [
        _check_kernels_present(gpu_events, thr),
        _check_kernels_not_dropped(gpu_events, thr),
    ]

    # Runtime variability grouped by (name, hlo_op).
    groups: Dict[str, List[float]] = defaultdict(list)
    for e in gpu_events:
        if e.get("dur") is None:
            continue
        hlo = (e.get("args") or {}).get("hlo_op", "")
        groups[f"{e.get('name')}|{hlo}"].append(float(e["dur"]))
    results.append(_check_runtime_variability(groups, thr))

    results.append(_jax_busy_idle(trace, thr))
    results.append(
        _check_kernel_counts(Counter(e.get("name", "") for e in gpu_events), thr)
    )
    results.append(_check_jax_metadata_richness(gpu_events, thr))
    return results


def _jax_busy_idle(trace: JaxJsonTrace, thr: QualityThresholds) -> QualityResult:
    by_dev = trace.gpu_events_by_device or {"all": trace.gpu_events}
    per_dev = {}
    worst = None
    for dev, evs in by_dev.items():
        m = busy_idle_from_events(evs)
        per_dev[dev] = round(m["idle_pct"], 3)
        if worst is None or m["idle_pct"] > worst[1]["idle_pct"]:
            worst = (dev, m)
    if worst is None:
        return QualityResult(
            "gpu_busy_idle", "GPU busy/idle time", Status.FAIL, "no GPU events", {}
        )
    dev, m = worst
    m = dict(m)
    m["worst_device"] = dev
    m["idle_pct_per_device"] = per_dev
    result = _check_busy_idle(m, thr)
    result.message = f"[{dev}] " + result.message
    return result


def _check_jax_metadata_richness(
    gpu_events: Sequence[dict], thr: QualityThresholds
) -> QualityResult:
    """Fraction of GPU events carrying rich HLO metadata."""
    n = len(gpu_events)
    if n == 0:
        return QualityResult(
            "jax_metadata_richness",
            "JAX GPU metadata richness (optional)",
            Status.FAIL,
            "no GPU events",
            {},
        )
    fields = ("hlo_op", "hlo_module", "tf_op")
    field_counts = {f: 0 for f in fields}
    for e in gpu_events:
        args = e.get("args") or {}
        for f in fields:
            if args.get(f):
                field_counts[f] += 1
    coverage = {f: round(c / n, 3) for f, c in field_counts.items()}
    min_cov = min(coverage.values())
    metrics = {"num_gpu_events": n, "coverage": coverage}
    if min_cov < thr.shape_coverage_fail:
        status = Status.FAIL
        msg = f"sparse HLO metadata (min coverage {min_cov:.0%}): {coverage}"
    elif min_cov < thr.shape_coverage_warn:
        status = Status.WARN
        msg = f"partial HLO metadata (min coverage {min_cov:.0%}): {coverage}"
    else:
        status = Status.PASS
        msg = f"rich HLO metadata: {coverage}"
    return QualityResult(
        "jax_metadata_richness",
        "JAX GPU metadata richness (optional)",
        status,
        msg,
        metrics,
    )


# Phase 2 checks (TraceLens perf reports)

# Map a logical sheet role to the concrete perf-report sheet per framework.
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


def find_megatron_extension() -> Optional[str]:
    """Locate the Megatron/TE extension (override via ``TRACE_QUALITY_MEGATRON_EXTENSION``)."""
    override = os.environ.get("TRACE_QUALITY_MEGATRON_EXTENSION")
    if override and os.path.exists(override):
        return override
    try:
        base = os.path.dirname(os.path.dirname(inspect.getfile(GPUEventAnalyser)))
        candidate = os.path.join(base, "examples", "example_megatron_extension.py")
        if os.path.exists(candidate):
            return candidate
    except Exception:  # noqa: BLE001
        logger.debug("Megatron extension lookup failed", exc_info=True)
    return None


def generate_report(
    path: str, framework: str, output_dir: Optional[str] = None
) -> Dict:
    """Generate the TraceLens perf report. PyTorch gets the Megatron extension."""
    out = output_dir or tempfile.mkdtemp(prefix="trace_quality_")
    if framework == FRAMEWORK_PYTORCH:
        from TraceLens.Reporting.generate_perf_report_pytorch import (
            generate_perf_report_pytorch,
        )

        return generate_perf_report_pytorch(
            profile_json_path=path,
            output_csvs_dir=out,
            collective_analysis=True,
            extension_file=find_megatron_extension(),
        )
    if framework == FRAMEWORK_JAX:
        from TraceLens.Reporting.generate_perf_report_jax import (
            generate_perf_report_jax,
        )

        return generate_perf_report_jax(profile_path=path, output_csvs_dir=out)
    raise ValueError(f"unknown framework: {framework}")


def _check_report_generated(dfs: Dict, sheets: Dict) -> QualityResult:
    present = sorted(dfs.keys())
    expected = {sheets["timeline"], sheets["op_summary"], sheets["unique_args"]}
    missing = sorted(expected - set(present))
    metrics = {"sheets": present, "missing_expected": missing}
    if missing:
        status = Status.FAIL
        msg = f"perf report missing expected sheets: {missing}"
    else:
        status = Status.PASS
        msg = f"perf report generated with {len(present)} sheets"
    return QualityResult(
        "perf_report_generated", "Perf report generated", status, msg, metrics
    )


def _check_idle_timeline(
    dfs: Dict, sheets: Dict, thr: QualityThresholds
) -> QualityResult:
    tl = dfs.get(sheets["timeline"])
    if tl is None or "type" not in getattr(tl, "columns", []):
        return QualityResult(
            "gpu_idle_timeline",
            "GPU idle timeline (perf report)",
            Status.SKIP,
            "no gpu_timeline sheet",
            {},
        )
    idle_rows = tl[tl["type"] == "idle_time"]
    if idle_rows.empty:
        return QualityResult(
            "gpu_idle_timeline",
            "GPU idle timeline (perf report)",
            Status.SKIP,
            "no idle_time row in timeline",
            {},
        )
    idle_pct = float(idle_rows["percent"].mean())
    metrics = {"idle_pct": round(idle_pct, 3), "num_devices": int(len(idle_rows))}
    if idle_pct >= thr.idle_pct_fail:
        status = Status.FAIL
        msg = f"GPU idle {idle_pct:.1f}% (>= {thr.idle_pct_fail}%)"
    elif idle_pct >= thr.idle_pct_warn:
        status = Status.WARN
        msg = f"GPU idle {idle_pct:.1f}% (>= {thr.idle_pct_warn}%)"
    else:
        status = Status.PASS
        msg = f"GPU idle {idle_pct:.1f}%"
    return QualityResult(
        "gpu_idle_timeline", "GPU idle timeline (perf report)", status, msg, metrics
    )


def _check_op_count_consistency(dfs: Dict, sheets: Dict) -> QualityResult:
    summary = dfs.get(sheets["op_summary"])
    col = sheets["count_col"]
    if summary is None or col not in getattr(summary, "columns", []):
        return QualityResult(
            "op_count_consistency",
            "Op-count consistency (perf report)",
            Status.SKIP,
            "no op-count column available",
            {},
        )
    counts = [int(c) for c in summary[col].tolist() if c and int(c) > 0]
    if not counts:
        return QualityResult(
            "op_count_consistency",
            "Op-count consistency (perf report)",
            Status.WARN,
            "no positive op counts",
            {},
        )
    divisor = gcd_of_counts(counts)
    metrics = {
        "num_ops": len(counts),
        "count_gcd": divisor,
        "min_count": min(counts),
        "max_count": max(counts),
    }
    if divisor > 1:
        status = Status.PASS
        msg = f"{len(counts)} ops share count gcd={divisor} (consistent per-step multiplicity)"
    else:
        status = Status.WARN
        msg = f"{len(counts)} ops have gcd=1 (no common per-step multiple)"
    return QualityResult(
        "op_count_consistency",
        "Op-count consistency (perf report)",
        status,
        msg,
        metrics,
    )


def _check_ops_have_shapes(
    dfs: Dict, sheets: Dict, thr: QualityThresholds
) -> QualityResult:
    df = dfs.get(sheets["unique_args"])
    if df is None or "Input Dims" not in getattr(df, "columns", []):
        return QualityResult(
            "ops_have_shapes",
            "Ops carry shapes (perf report)",
            Status.SKIP,
            "no Input Dims column",
            {},
        )
    total = len(df)
    if total == 0:
        return QualityResult(
            "ops_have_shapes",
            "Ops carry shapes (perf report)",
            Status.WARN,
            "no ops in unique_args sheet",
            {},
        )

    def _has_shape(v) -> bool:
        if v is None:
            return False
        return str(v).strip() not in ("", "[]", "[[]]", "nan", "None", "()")

    with_shape = int(df["Input Dims"].apply(_has_shape).sum())
    coverage = with_shape / total
    metrics = {
        "num_ops": total,
        "ops_with_shapes": with_shape,
        "shape_coverage": round(coverage, 3),
    }
    if coverage < thr.shape_coverage_fail:
        status = Status.FAIL
        msg = f"only {coverage:.0%} of ops have shapes"
    elif coverage < thr.shape_coverage_warn:
        status = Status.WARN
        msg = f"{coverage:.0%} of ops have shapes"
    else:
        status = Status.PASS
        msg = f"{coverage:.0%} of ops have shapes"
    return QualityResult(
        "ops_have_shapes", "Ops carry shapes (perf report)", status, msg, metrics
    )


def _check_high_idle_ops(
    dfs: Dict, sheets: Dict, thr: QualityThresholds
) -> QualityResult:
    """Flag ops with a large gap between subtree wall time and kernel time."""
    df = dfs.get(sheets["unique_args"])
    cols = set(df.columns) if df is not None else set()
    need = {"total_subtree_kernel_time_sum", "total_direct_kernel_time_sum"}
    if df is None or not need.issubset(cols):
        return QualityResult(
            "high_idle_ops",
            "High-idle ops (perf report)",
            Status.SKIP,
            "no subtree/direct kernel-time columns to estimate idle",
            {},
        )
    offenders = []
    for _, row in df.iterrows():
        subtree = float(row["total_subtree_kernel_time_sum"] or 0)
        direct = float(row["total_direct_kernel_time_sum"] or 0)
        if subtree <= 0:
            continue
        gap_pct = 100.0 * max(0.0, subtree - direct) / subtree
        if gap_pct >= thr.op_high_idle_pct:
            offenders.append({"name": row.get("name"), "gap_pct": round(gap_pct, 1)})
    offenders.sort(key=lambda x: -x["gap_pct"])
    metrics = {"num_high_idle_ops": len(offenders), "top": offenders[:10]}
    if offenders:
        status = Status.WARN
        msg = f"{len(offenders)} op(s) with >= {thr.op_high_idle_pct}% non-kernel time"
    else:
        status = Status.PASS
        msg = "no ops with excessive non-kernel (idle) time"
    return QualityResult(
        "high_idle_ops", "High-idle ops (perf report)", status, msg, metrics
    )


def _check_sdpa_count(dfs: Dict, sheets: Dict, thr: QualityThresholds) -> QualityResult:
    """Count SDPA / attention ops across category and dedicated op sheets."""
    counts: Dict[str, int] = {}
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
                c = int(row[count_col]) if count_col else 1
                counts[cat] = counts.get(cat, 0) + c
                total += c
    # Additional signal: dedicated TE / attention op sheets (e.g. JAX op_te).
    for sheet_name, df in dfs.items():
        if sheet_name.startswith("op_") and (
            "te" in sheet_name.lower()
            or matches_any(sheet_name, thr.attn_name_patterns)
        ):
            counts[sheet_name] = max(counts.get(sheet_name, 0), len(df))
    metrics = {"attention_breakdown": counts, "total_attention_ops": total}
    if total == 0 and not counts:
        status = Status.WARN
        msg = "no SDPA / attention ops found in perf report"
    else:
        status = Status.PASS
        msg = f"attention ops present: {counts}"
    return QualityResult(
        "sdpa_count", "SDPA / attention op count (perf report)", status, msg, metrics
    )


# Perf-report sheets that hold forward / backward attention kernels.
_ATTN_FWD_SHEETS = ("op_fa_fwd",)
_ATTN_BWD_SHEETS = ("op_fa_bwd",)


def _infer_num_gpus(dfs: Dict) -> Optional[int]:
    """Number of GPU devices, from the perf-report ``gpu_timeline`` sheet."""
    tl = dfs.get("gpu_timeline")
    if tl is None or "gpu_pid" not in getattr(tl, "columns", []):
        return None
    try:
        n = int(tl["gpu_pid"].nunique())
    except Exception:  # noqa: BLE001
        return None
    return n or None


def _attention_observed(dfs: Dict, sheet_names: Sequence[str]) -> Dict[str, Any]:
    """Sum observed launches, count unique kernels, and list per-kernel counts."""
    launches = 0
    unique = 0
    found = []
    kernels: List[Dict[str, Any]] = []
    for name in sheet_names:
        df = dfs.get(name)
        if df is None or "name_count" not in getattr(df, "columns", []):
            continue
        found.append(name)
        unique += int(len(df))
        launches += int(df["name_count"].fillna(0).astype(int).sum())
        names = df["name"] if "name" in df.columns else [None] * len(df)
        for kname, kcount in zip(names, df["name_count"].fillna(0).astype(int)):
            kernels.append({"name": kname, "count": int(kcount)})
    kernels.sort(key=lambda k: -k["count"])
    return {
        "launches": launches,
        "unique_kernels": unique,
        "sheets": found,
        "kernels": kernels,
    }


def _format_attn_failure(
    label: str,
    obs: Dict[str, Any],
    num_gpus: int,
    num_steps: int,
    num_layers: int,
    calls: int,
    expected: int,
) -> str:
    """FAIL detail: substituted formula and per-kernel counts."""
    lines = [
        f"\n    {label}: expected {expected} != observed {obs['launches']}",
        "      formula: num_gpus * num_steps * num_layers * calls * unique_kernels",
        f"               = {num_gpus} * {num_steps} * {num_layers} * {calls} "
        f"* {obs['unique_kernels']} = {expected}",
        f"      observed launches = {obs['launches']} "
        f"across {obs['unique_kernels']} unique kernel(s):",
    ]
    for k in obs["kernels"]:
        lines.append(f"        - count={k['count']:<8} {k['name']}")
    return "\n".join(lines)


def _check_attention_kernel_count(
    dfs: Dict,
    cfg: AttnCountConfig,
) -> QualityResult:
    """Sanity-check observed attention launches against the model config.

    expected = num_gpus * num_steps * num_layers * calls * unique_kernels,
    computed separately for the forward and backward passes.
    """
    check_id = "attention_kernel_count"
    name = "Attention kernel count matches config"

    fwd = _attention_observed(dfs, _ATTN_FWD_SHEETS)
    bwd = _attention_observed(dfs, _ATTN_BWD_SHEETS)
    if not fwd["sheets"] and not bwd["sheets"]:
        return QualityResult(
            check_id,
            name,
            Status.SKIP,
            "no attention (op_fa_fwd/op_fa_bwd) sheets in perf report",
            {},
        )

    num_gpus = cfg.num_gpus or _infer_num_gpus(dfs)
    num_steps = cfg.num_steps

    def _per_pass(obs: Dict[str, Any], calls: int) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "observed_launches": obs["launches"],
            "unique_kernels": obs["unique_kernels"],
            "calls": calls,
            "kernels": obs["kernels"],
        }
        # layers*calls implied by the trace (independent of config).
        denom = (num_gpus or 0) * (num_steps or 0) * (obs["unique_kernels"] or 0)
        if denom:
            info["implied_layers_x_calls"] = round(obs["launches"] / denom, 4)
        return info

    metrics: Dict[str, Any] = {
        "num_gpus": num_gpus,
        "num_steps": num_steps,
        "num_layers": cfg.num_layers,
        "forward": _per_pass(fwd, cfg.fwd_attn_calls),
        "backward": _per_pass(bwd, cfg.bwd_attn_calls),
    }

    if not cfg.enabled:
        return QualityResult(
            check_id,
            name,
            Status.SKIP,
            "num_layers and/or num_steps not provided; reporting observed counts "
            f"only (fwd={fwd['launches']}, bwd={bwd['launches']}, "
            f"num_gpus={num_gpus})",
            metrics,
        )
    if not num_gpus:
        return QualityResult(
            check_id,
            name,
            Status.SKIP,
            "could not infer num_gpus; pass --num-gpus to enable the check",
            metrics,
        )

    problems = []
    for label, obs, calls in (
        ("forward", fwd, cfg.fwd_attn_calls),
        ("backward", bwd, cfg.bwd_attn_calls),
    ):
        if not obs["sheets"]:
            continue
        expected = num_gpus * num_steps * cfg.num_layers * calls * obs["unique_kernels"]
        metrics[label]["expected_launches"] = expected
        observed = obs["launches"]
        tol_abs = cfg.tol * expected
        ok = abs(observed - expected) <= tol_abs
        if not ok:
            problems.append(
                _format_attn_failure(
                    label, obs, num_gpus, num_steps, cfg.num_layers, calls, expected
                )
            )

    if problems:
        summary = (
            f"attention launches do not match config "
            f"(num_gpus={num_gpus}, num_steps={num_steps}, num_layers={cfg.num_layers})"
        )
        return QualityResult(
            check_id, name, Status.FAIL, summary + "".join(problems), metrics
        )
    return QualityResult(
        check_id,
        name,
        Status.PASS,
        f"attention launches match config "
        f"(num_gpus={num_gpus}, num_steps={num_steps}, num_layers={cfg.num_layers}; "
        f"fwd={fwd['launches']}, bwd={bwd['launches']})",
        metrics,
    )


def run_phase2(
    path: str,
    framework: str,
    thr: Optional[QualityThresholds] = None,
    output_dir: Optional[str] = None,
    attn_cfg: Optional[AttnCountConfig] = None,
) -> List[QualityResult]:
    thr = thr or QualityThresholds()
    sheets = _SHEETS[framework]
    try:
        dfs = generate_report(path, framework, output_dir=output_dir)
    except Exception as exc:  # noqa: BLE001
        return [
            QualityResult(
                "perf_report_generated",
                "Perf report generated",
                Status.FAIL,
                f"perf report generation failed: {exc}",
                {},
            )
        ]
    results = [
        _check_report_generated(dfs, sheets),
        _check_idle_timeline(dfs, sheets, thr),
        _check_op_count_consistency(dfs, sheets),
        _check_ops_have_shapes(dfs, sheets, thr),
        _check_high_idle_ops(dfs, sheets, thr),
        _check_sdpa_count(dfs, sheets, thr),
    ]
    if framework == FRAMEWORK_JAX:
        results.append(
            _check_attention_kernel_count(dfs, attn_cfg or AttnCountConfig())
        )
    return results


# ===========================================================================
# Runner
# ===========================================================================
def run_quality_checks(
    path: str,
    framework: str,
    phases: Iterable[str] = ("1", "2"),
    thresholds: Optional[QualityThresholds] = None,
    xplane_pb_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    attn_cfg: Optional[AttnCountConfig] = None,
) -> QualityReport:
    """Run trace quality checks for a single trace.

    Args:
        path: Trace path. For phase 1 JAX, the xprof JSON export.
        framework: ``"pytorch"`` or ``"jax"``.
        phases: Subset of ``{"1", "2"}``.
        thresholds: Optional :class:`QualityThresholds` overrides.
        xplane_pb_path: For JAX, the ``*.xplane.pb`` used by phase 2; defaults to
            ``path`` (PyTorch) or a ``.xplane.pb`` sibling guess (JAX).
        output_dir: Optional directory for perf-report CSV output (phase 2).
        attn_cfg: Optional model config for the attention-kernel-count check.
    """
    framework = framework.lower()
    if framework not in VALID_FRAMEWORKS:
        raise ValueError(
            f"framework must be one of {VALID_FRAMEWORKS}, got {framework!r}"
        )
    thr = thresholds or QualityThresholds()
    phases = {str(p) for p in phases}
    report = QualityReport(trace_path=path, framework=framework)

    if "1" in phases:
        try:
            if framework == FRAMEWORK_PYTORCH:
                report.extend(run_pytorch_phase1(load_pytorch(path), thr))
            else:
                report.extend(run_jax_phase1(load_jax_json(path), thr))
        except Exception as exc:  # noqa: BLE001
            report.add(
                QualityResult(
                    "phase1_load",
                    "Phase 1 load/parse",
                    Status.FAIL,
                    f"failed to load/parse trace for phase 1: {exc}",
                    {},
                )
            )

    if "2" in phases:
        p2_path = xplane_pb_path or _default_xplane_pb_path(path, framework)
        report.extend(
            run_phase2(
                p2_path,
                framework,
                thr,
                output_dir=output_dir,
                attn_cfg=attn_cfg,
            )
        )

    return report


def _default_xplane_pb_path(path: str, framework: str) -> str:
    if framework == FRAMEWORK_JAX and not path.endswith(".xplane.pb"):
        for suffix in (".trace.json.gz", ".trace.json", ".json.gz", ".json"):
            if path.endswith(suffix):
                return path[: -len(suffix)] + ".xplane.pb"
    return path


# ===========================================================================
# CLI
# ===========================================================================
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="trace_quality.py",
        description="Check the quality of a PyTorch or JAX profiler trace.",
    )
    p.add_argument(
        "trace",
        nargs="?",
        default=None,
        help="Path to the trace file (can also be passed via --json_trace)",
    )
    p.add_argument(
        "--json_trace",
        default=None,
        help="Path to the trace file (alternative to the positional argument)",
    )
    p.add_argument(
        "--framework",
        required=True,
        choices=VALID_FRAMEWORKS,
        help="Trace framework (explicit; not auto-detected)",
    )
    p.add_argument(
        "--phase",
        default="all",
        choices=["1", "2", "all"],
        help="Which phase(s) of checks to run (default: all)",
    )
    p.add_argument(
        "--xplane-pb-path",
        default=None,
        help="JAX only: path to the .xplane.pb used for phase 2 perf report",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write perf-report CSVs (phase 2)",
    )
    p.add_argument("--json", default=None, help="Write the full report as JSON here")

    attn = p.add_argument_group(
        "attention kernel count (JAX phase 2)",
        "Sanity-check observed attention launches against the model config: "
        "expected = num_gpus * num_steps * num_layers * calls * unique_kernels "
        "(forward and backward computed separately). num_gpus and the "
        "unique-kernel counts are inferred from the trace; --num-layers and "
        "--num-steps are both required to enable the check (else it SKIPs and "
        "just reports counts).",
    )
    attn.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Number of transformer/attention layers (required to enable the check)",
    )
    attn.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Training steps in the trace (required to enable the check)",
    )
    attn.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Number of GPUs (default: inferred from the trace)",
    )
    attn.add_argument(
        "--fwd-attn-calls",
        type=int,
        default=1,
        help="Attention invocations per layer in the forward pass (default: 1; "
        "use 2 when forward attention is recomputed during backward)",
    )
    attn.add_argument(
        "--bwd-attn-calls",
        type=int,
        default=1,
        help="Attention invocations per layer in the backward pass (default: 1)",
    )
    attn.add_argument(
        "--attn-count-tol",
        type=float,
        default=0.0,
        help="Relative tolerance for the attention-count comparison (default: 0.0)",
    )
    return p


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    trace_path = args.trace or args.json_trace
    if not trace_path:
        parser.error("a trace path is required (positional argument or --json_trace)")
    phases = ("1", "2") if args.phase == "all" else (args.phase,)

    attn_cfg = AttnCountConfig(
        num_layers=args.num_layers,
        num_steps=args.num_steps,
        num_gpus=args.num_gpus,
        fwd_attn_calls=args.fwd_attn_calls,
        bwd_attn_calls=args.bwd_attn_calls,
        tol=args.attn_count_tol,
    )

    report = run_quality_checks(
        path=trace_path,
        framework=args.framework,
        phases=phases,
        thresholds=QualityThresholds(),
        xplane_pb_path=args.xplane_pb_path,
        output_dir=args.output_dir,
        attn_cfg=attn_cfg,
    )

    print(report)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"\nWrote JSON report to {args.json}")

    # Non-zero exit when any check FAILed, so CI can gate on it.
    return 1 if report.worst_status is Status.FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
