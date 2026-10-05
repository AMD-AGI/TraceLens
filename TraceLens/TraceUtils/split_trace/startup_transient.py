###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Detect and drop a profiler-start transient from iteration 0.

In tensor-parallel inference, every rank starts its profiler independently.
The rank(s) that reach the first collective op earliest are forced to wait for
the slowest rank, and that wait is booked against the collective kernel's
duration wherever it happens to land -- inflating iteration 0's collective-op
GPU time and, with it, deflating the reported GPU-time share of every other
kernel for that iteration.

The signature that distinguishes this from ordinary run-to-run noise is that
it *selectively* inflates collective ops: legitimate variance (a slow first
iteration from cache warmup, allocator setup, etc.) touches compute and
collective kernels roughly alike. So rather than a blanket "iteration 0 is an
outlier" heuristic, :func:`trim_startup_transient` compares iteration 0's
per-kernel-name durations against the median of the same kernel in later
iterations, separately for collective and non-collective kernels, and only
drops iteration 0 when its worst collective-op inflation both clears an
absolute floor and dwarfs the worst non-collective inflation (the run's noise
ceiling).
"""

from bisect import bisect_left
from statistics import median
from typing import Dict, List, Optional, Tuple

from ..utils.detect_utils import (
    BOOKEND_NAMES,
    DetectStatus,
    EventIndex,
    GpuAttribution,
    RootSet,
    grade_coverage,
)

# Iteration 0's collective kernel must run at least this many times its later-
# iteration median before it is even considered a transient.
_COLLECTIVE_RATIO_FLOOR = 5.0
# ...and that inflation must dwarf the worst non-collective (compute) kernel's
# inflation by this factor, so ordinary noise that touches every op alike never
# qualifies.
_NOISE_RATIO_MARGIN = 5.0


def _kernel_durations_by_name(
    kernels: List[dict], starts: List[float], window: Tuple[float, float]
) -> Dict[str, float]:
    """Total duration per kernel name, for kernels starting inside ``window``."""
    lo, hi = window
    i = bisect_left(starts, lo)
    totals: Dict[str, float] = {}
    while i < len(starts) and starts[i] < hi:
        k = kernels[i]
        name = k.get("name", "")
        totals[name] = totals.get(name, 0.0) + k.get("dur", 0.0)
        i += 1
    return totals


def trim_startup_transient(
    root_set: RootSet, trace_index: Optional[EventIndex]
) -> RootSet:
    """Drop iteration 0 when profiler-start skew inflated its collective kernels.

    Compares iteration 0's per-kernel-name GPU durations to the median of the
    same names across later iterations, and drops iteration 0 only when a
    collective op is inflated far beyond the worst inflation seen among
    non-collective ops in the same trace -- the run's own noise ceiling. This
    keeps ordinary variance (which touches all op types) from ever qualifying,
    without any of the arbitrary sample-count/outlier-count safeguards a
    blanket "iteration 0 is slow" heuristic would need.
    """
    roots = root_set.roots
    if (
        trace_index is None
        or len(roots) < 2
        or root_set.status is DetectStatus.NOT_SPLITTABLE
        or roots[0].get("name") in BOOKEND_NAMES
        or not trace_index.collective_kernels
    ):
        return root_set

    kernels = trace_index.kernels
    starts = [k["ts"] for k in kernels]

    first_totals = _kernel_durations_by_name(
        kernels,
        starts,
        (roots[0].get("ts", 0), roots[0].get("ts", 0) + roots[0].get("dur", 0)),
    )
    later_totals = [
        _kernel_durations_by_name(
            kernels, starts, (r.get("ts", 0), r.get("ts", 0) + r.get("dur", 0))
        )
        for r in roots[1:]
    ]

    collective_ratio: Optional[float] = None
    collective_name: Optional[str] = None
    noise_ratio = 0.0
    for name, first_dur in first_totals.items():
        later_values = [t.get(name, 0.0) for t in later_totals]
        nonzero = [v for v in later_values if v > 0]
        if not nonzero:
            continue
        med = median(nonzero)
        ratio = first_dur / med
        if name in trace_index.collective_kernels:
            if collective_ratio is None or ratio > collective_ratio:
                collective_ratio, collective_name = ratio, name
        else:
            noise_ratio = max(noise_ratio, ratio)

    diagnostics = dict(root_set.diagnostics)
    if (
        collective_ratio is not None
        and collective_ratio >= _COLLECTIVE_RATIO_FLOOR
        and collective_ratio >= _NOISE_RATIO_MARGIN * max(noise_ratio, 1.0)
    ):
        diagnostics.update(
            {
                "startup_transient_trimmed": True,
                "startup_transient_kernel": collective_name,
                "startup_transient_collective_ratio": round(collective_ratio, 1),
                "startup_transient_noise_ratio": round(noise_ratio, 1),
            }
        )
        new_roots = roots[1:]
        coverage = GpuAttribution(trace_index).audit(new_roots)
        return RootSet(
            roots=new_roots,
            method=root_set.method,
            phase_confidence=root_set.phase_confidence,
            status=grade_coverage(coverage.covered_selected),
            coverage=coverage,
            diagnostics=diagnostics,
        )

    diagnostics["startup_transient_trimmed"] = False
    return RootSet(
        roots=root_set.roots,
        method=root_set.method,
        phase_confidence=root_set.phase_confidence,
        status=root_set.status,
        coverage=root_set.coverage,
        diagnostics=diagnostics,
    )
