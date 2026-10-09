###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Lightweight trace health checks run before analysis begins.

Validates that a trace is analyzable and surfaces actionable warnings for
common profiling pitfalls (dropped kernels, missing call stacks, graph-mode
traces without a capture trace).

Checks are gated by ``TRACELENS_SKIP_HEALTH_CHECK`` in
``DataLoader.load_trace_events``.
"""

import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

from .util import GPU_KERNEL_CATEGORIES, GRAPH_LAUNCH_NAMES


class HealthCheckLevel(Enum):
    WARN = "warn"
    ERROR = "error"


@dataclass
class TraceHealthFinding:
    check_id: str
    level: HealthCheckLevel
    message: str


@dataclass
class TraceHealthReport:
    findings: List[TraceHealthFinding] = field(default_factory=list)

    def log_findings(self) -> None:
        for f in self.findings:
            if f.level == HealthCheckLevel.ERROR:
                raise ValueError(f"[trace_health:{f.check_id}] {f.message}")
            elif f.level == HealthCheckLevel.WARN:
                warnings.warn(
                    f"[trace_health:{f.check_id}] {f.message}",
                    UserWarning,
                    stacklevel=2,
                )


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
    has_kernel = False
    has_python_func = False
    has_graph_launch = False

    for event in events:
        if not has_kernel and event.get("cat", "") in GPU_KERNEL_CATEGORIES:
            has_kernel = True
        if not has_python_func and event.get("cat") == "python_function":
            has_python_func = True
        if not has_graph_launch and event.get("name") in GRAPH_LAUNCH_NAMES:
            has_graph_launch = True
        if has_kernel and has_python_func and has_graph_launch:
            break

    findings: List[TraceHealthFinding] = []
    if not has_kernel:
        findings.append(
            TraceHealthFinding(
                "kernels_present",
                HealthCheckLevel.ERROR,
                "No GPU kernel events found in trace.",
            )
        )
    if not has_python_func:
        findings.append(
            TraceHealthFinding(
                "call_stack_missing",
                HealthCheckLevel.WARN,
                "Trace does not contain call stack data (with_stack=False or not set). "
                "Pass with_stack=True to the profiler for call-stack-based analysis. "
                "Cross-trace comparisons may also be affected.",
            )
        )
    if has_graph_launch and capture_trace_filepath is None:
        findings.append(
            TraceHealthFinding(
                "graph_mode_no_capture",
                HealthCheckLevel.WARN,
                "Trace contains graph launch event(s) but no capture trace "
                "was provided. Graph-mode analysis will be limited. "
                "Pass --capture_folder to the inference perf report generator.",
            )
        )
    return TraceHealthReport(findings=findings)


def check_kernels_dropped(tree) -> TraceHealthReport:
    """Check for dropped GPU kernels by looking for runtime launch events
    that have no corresponding GPU kernel after tree building.

    Must be called after ``tree.build_tree()`` so that
    ``runtime_event_uids`` and ``gpu_events`` are populated.
    """
    if not hasattr(tree, "runtime_event_uids"):
        return TraceHealthReport()

    dropped = 0
    for uid in tree.runtime_event_uids:
        event = tree.events_by_uid[uid]
        if not event.get("gpu_events"):
            dropped += 1

    if dropped == 0:
        return TraceHealthReport()
    return TraceHealthReport(
        findings=[
            TraceHealthFinding(
                "kernels_dropped",
                HealthCheckLevel.WARN,
                f"{dropped} runtime launch event(s) have no corresponding "
                f"GPU kernel — kernels were likely dropped by the profiler.",
            )
        ]
    )
