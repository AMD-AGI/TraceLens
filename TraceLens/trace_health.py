###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Lightweight trace health checks run before analysis begins.

Validates that a trace is analyzable and surfaces actionable warnings for
common profiling pitfalls (dropped kernels, missing call stacks, graph-mode
traces without a capture trace).

Bypass all checks by setting ``TRACELENS_SKIP_HEALTH_CHECK=1``.
"""

import logging
import os
import statistics
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

# Constants
_ENV_SKIP = "TRACELENS_SKIP_HEALTH_CHECK"
_GPU_CATS = frozenset({"kernel", "gpu_memcpy", "gpu_memset"})
_GRAPH_LAUNCH_PATTERN = "graphlaunch"
_MIN_KERNEL_COUNT = 10
_DROP_GAP_FACTOR = 10.0


@dataclass
class TraceHealthFinding:
    check_id: str
    level: str  # "info", "warn", or "error"
    message: str


@dataclass
class TraceHealthReport:
    findings: List[TraceHealthFinding] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(f.level != "error" for f in self.findings)

    def log_findings(self) -> None:
        for f in self.findings:
            if f.level in ("warn", "error"):
                warnings.warn(
                    f"[trace_health:{f.check_id}] {f.message}",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                logger.info("[trace_health:%s] %s", f.check_id, f.message)


def run_trace_health_check(
    events: Sequence[Dict[str, Any]],
    trace_metadata: Optional[Dict[str, Any]] = None,
    capture_trace_filepath: Optional[str] = None,
) -> TraceHealthReport:
    """Run lightweight health checks on a trace before analysis.

    Args:
        events: The ``traceEvents`` list from a loaded trace.
        trace_metadata: Top-level trace fields (everything except ``traceEvents``).
        capture_trace_filepath: Path to a graph-capture trace, if provided.

    Returns:
        A :class:`TraceHealthReport` with any findings.  Call
        ``report.log_findings()`` to emit warnings.
    """
    if os.environ.get(_ENV_SKIP, "").lower() in ("1", "true", "yes"):
        return TraceHealthReport()

    if trace_metadata is None:
        trace_metadata = {}

    kernel_timestamps: List[float] = []
    has_python_func = False
    graph_launch_count = 0

    for event in events:
        cat = event.get("cat", "")
        if cat in _GPU_CATS:
            ts = event.get("ts")
            if ts is not None:
                kernel_timestamps.append(float(ts))
        elif cat == "python_function":
            has_python_func = True
        name = event.get("name", "")
        if _GRAPH_LAUNCH_PATTERN in name.lower():
            graph_launch_count += 1

    findings: List[TraceHealthFinding] = []
    findings.extend(_check_kernels(kernel_timestamps))
    findings.extend(_check_profiler_options(trace_metadata, has_python_func))
    findings.extend(_check_graph_mode(graph_launch_count, capture_trace_filepath))
    return TraceHealthReport(findings=findings)


def _check_kernels(kernel_timestamps: List[float]) -> List[TraceHealthFinding]:
    findings: List[TraceHealthFinding] = []
    n = len(kernel_timestamps)

    if n == 0:
        findings.append(
            TraceHealthFinding(
                "kernels_present",
                "error",
                "No GPU kernel events found in trace.",
            )
        )
        return findings

    if n < _MIN_KERNEL_COUNT:
        findings.append(
            TraceHealthFinding(
                "kernels_present",
                "warn",
                f"Only {n} GPU kernel events found (expected >= {_MIN_KERNEL_COUNT}).",
            )
        )

    if n >= 3:
        kernel_timestamps.sort()
        gaps = [kernel_timestamps[i + 1] - kernel_timestamps[i] for i in range(n - 1)]
        median_gap = statistics.median(gaps)
        if median_gap > 0:
            threshold = _DROP_GAP_FACTOR * median_gap
            # Only interior gaps (skip first and last which may be ramp-up/tear-down)
            large_gaps = sum(1 for g in gaps[1:-1] if g > threshold)
            if large_gaps > 0:
                findings.append(
                    TraceHealthFinding(
                        "kernels_dropped",
                        "warn",
                        f"Possible kernel drop: {large_gaps} gap(s) exceed "
                        f"{_DROP_GAP_FACTOR:.0f}x the median inter-kernel interval.",
                    )
                )

    return findings


def _check_profiler_options(
    trace_metadata: Dict[str, Any], has_python_func: bool
) -> List[TraceHealthFinding]:
    if trace_metadata.get("with_stack") == 1:
        return []
    if has_python_func:
        return []
    return [
        TraceHealthFinding(
            "call_stack_missing",
            "warn",
            "Trace does not contain call stack data (with_stack=False or not set). "
            "Pass with_stack=True to the profiler for call-stack-based analysis.",
        )
    ]


def _check_graph_mode(
    graph_launch_count: int, capture_trace_filepath: Optional[str]
) -> List[TraceHealthFinding]:
    if graph_launch_count == 0:
        return []
    if capture_trace_filepath is not None:
        return []
    return [
        TraceHealthFinding(
            "graph_mode_no_capture",
            "warn",
            f"Trace contains {graph_launch_count} graph launch event(s) "
            f"but no capture trace was provided. "
            f"Graph-mode analysis will be limited.",
        )
    ]
