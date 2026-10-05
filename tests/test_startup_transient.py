###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for the profiler-start transient trim.

Builds a ``RootSet`` and its kernels directly rather than going through the
detection cascade, so each case isolates one property of the collective-vs-
noise comparison in :func:`trim_startup_transient`.
"""

from TraceLens.TraceUtils.utils.detect_utils import (
    DetectStatus,
    EventIndex,
    PhaseConfidence,
    RootSet,
)
from TraceLens.TraceUtils.split_trace.startup_transient import trim_startup_transient

ITER_PERIOD = 1000
ITER_DUR = 500


def _root(index, pid=1, tid=10):
    return {
        "name": f"iter{index}",
        "pid": pid,
        "tid": tid,
        "ts": index * ITER_PERIOD,
        "dur": ITER_DUR,
    }


def _kernel(ts, dur, name, pid=1, tid=99):
    return {
        "name": name,
        "cat": "kernel",
        "ph": "X",
        "ts": ts,
        "dur": dur,
        "pid": pid,
        "tid": tid,
        "args": {},
    }


def _build(
    n_iterations, collective_durs, compute_durs, collective_name="ncclAllReduce"
):
    """``n_iterations`` roots, each with one compute and one collective kernel."""
    roots = [_root(i) for i in range(n_iterations)]
    kernels = []
    for i in range(n_iterations):
        base = i * ITER_PERIOD
        kernels.append(_kernel(base + 10, compute_durs[i], "gemm"))
        kernels.append(_kernel(base + 20, collective_durs[i], collective_name))
    return roots, kernels


def _root_set(roots, status=DetectStatus.SPLITTABLE, diagnostics=None):
    return RootSet(
        roots=roots,
        method="test",
        phase_confidence=PhaseConfidence.HIGH,
        status=status,
        diagnostics=diagnostics or {},
    )


class TestTrimStartupTransient:
    def test_collective_only_inflation_is_trimmed(self):
        """Iteration 0's collective op is far slower; compute is steady -> trimmed."""
        roots, kernels = _build(
            n_iterations=5,
            collective_durs=[900, 40, 40, 40, 40],
            compute_durs=[40, 40, 40, 40, 40],
        )
        result = trim_startup_transient(_root_set(roots), EventIndex(kernels))
        assert result.diagnostics["startup_transient_trimmed"] is True
        assert result.diagnostics["startup_transient_kernel"] == "ncclAllReduce"
        assert len(result.roots) == 4
        assert result.roots[0]["name"] == "iter1"

    def test_uniform_noise_is_not_trimmed(self):
        """Iteration 0 is slow across the board -> ordinary variance, not skew."""
        roots, kernels = _build(
            n_iterations=5,
            collective_durs=[200, 40, 40, 40, 40],
            compute_durs=[200, 40, 40, 40, 40],
        )
        result = trim_startup_transient(_root_set(roots), EventIndex(kernels))
        assert result.diagnostics["startup_transient_trimmed"] is False
        assert len(result.roots) == 5

    def test_non_collective_outlier_is_not_trimmed(self):
        """Only the compute kernel is inflated; the collective op is untouched."""
        roots, kernels = _build(
            n_iterations=5,
            collective_durs=[40, 40, 40, 40, 40],
            compute_durs=[900, 40, 40, 40, 40],
        )
        result = trim_startup_transient(_root_set(roots), EventIndex(kernels))
        assert result.diagnostics["startup_transient_trimmed"] is False
        assert len(result.roots) == 5

    def test_two_iterations_can_still_trim(self):
        """Even with just one later iteration, the comparison works."""
        roots, kernels = _build(
            n_iterations=2,
            collective_durs=[900, 40],
            compute_durs=[40, 40],
        )
        result = trim_startup_transient(_root_set(roots), EventIndex(kernels))
        assert result.diagnostics["startup_transient_trimmed"] is True
        assert len(result.roots) == 1

    def test_single_iteration_is_left_alone(self):
        roots, kernels = _build(
            n_iterations=1,
            collective_durs=[900],
            compute_durs=[40],
        )
        root_set = _root_set(roots)
        result = trim_startup_transient(root_set, EventIndex(kernels))
        assert result is root_set

    def test_not_splittable_is_left_alone(self):
        roots, kernels = _build(
            n_iterations=5,
            collective_durs=[900, 40, 40, 40, 40],
            compute_durs=[40, 40, 40, 40, 40],
        )
        root_set = _root_set(roots, status=DetectStatus.NOT_SPLITTABLE)
        result = trim_startup_transient(root_set, EventIndex(kernels))
        assert result is root_set

    def test_bookend_root_is_left_alone(self):
        roots, kernels = _build(
            n_iterations=5,
            collective_durs=[900, 40, 40, 40, 40],
            compute_durs=[40, 40, 40, 40, 40],
        )
        roots[0]["name"] = "warmup"
        root_set = _root_set(roots)
        result = trim_startup_transient(root_set, EventIndex(kernels))
        assert result is root_set

    def test_no_trace_index_is_left_alone(self):
        roots, _ = _build(
            n_iterations=5,
            collective_durs=[900, 40, 40, 40, 40],
            compute_durs=[40, 40, 40, 40, 40],
        )
        root_set = _root_set(roots)
        result = trim_startup_transient(root_set, None)
        assert result is root_set

    def test_no_collective_kernels_is_left_alone(self):
        """When no kernels match collective patterns, skip entirely."""
        roots = [_root(i) for i in range(5)]
        kernels = [_kernel(i * ITER_PERIOD + 10, 40, "gemm") for i in range(5)]
        root_set = _root_set(roots)
        result = trim_startup_transient(root_set, EventIndex(kernels))
        assert result is root_set

    def test_coverage_excludes_transient_kernel_duration(self):
        """After trimming, the transient kernel's duration is removed from gpu_busy."""
        roots, kernels = _build(
            n_iterations=5,
            collective_durs=[900, 40, 40, 40, 40],
            compute_durs=[40, 40, 40, 40, 40],
        )
        result = trim_startup_transient(_root_set(roots), EventIndex(kernels))
        assert result.diagnostics["startup_transient_trimmed"] is True
        total_all = sum(k["dur"] for k in kernels)
        assert result.coverage.gpu_busy < total_all
