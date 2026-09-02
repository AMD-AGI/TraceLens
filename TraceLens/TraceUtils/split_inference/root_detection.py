###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Annotation families and leaf detection helpers.

Step 1 groups every annotation into families, known and unknown alike. Step 1.5
chooses which nesting level is the iteration and where its metadata comes from.
``detect_from_graph_launches`` handles the graph-replay fast path.
"""

from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Dict, List, Optional, Sequence

from ...Trace2Tree.inference_iteration_roots import (
    GRAPH_LAUNCH_NAMES,
    MIN_LABEL_CHILDREN,
)
from ..annotation_utils import (
    ANNOTATION_CAT,
    is_parseable,
    name_skeleton,
)
from .detect_utils import (
    COVERAGE_FLOOR,
    COVERAGE_GATE,
    MIN_ROOTS,
    DetectStatus,
    GpuAttribution,
    PhaseConfidence,
    RootSet,
)

# A family must enclose more than this share of another family's instances for
# the nesting relation to hold. Majority rather than unanimity, because real
# traces have ragged edges: warmup instances outside the loop, a truncated final
# iteration. Requiring every instance rejects the correct relation nearly always.
NESTING_MAJORITY = 0.5


@dataclass
class AnnotationFamily:
    """All instances of one logical annotation, keyed by its skeleton.

    CPU-side only: projections duplicate one annotation across streams, so any
    count taken from them is inflated. They give the "has GPU work" signal only.
    """

    skeleton: str
    instances: List[dict]
    gpu_time: float = 0.0
    parseable: bool = False
    interarrival_cv: float = 0.0

    @property
    def count(self) -> int:
        return len(self.instances)

    @property
    def regular(self) -> bool:
        """Enough instances to be a per-iteration event rather than a one-off."""
        return self.count >= MIN_ROOTS

    @property
    def rank(self) -> tuple:
        """Sort key for choosing between families: most GPU work, steadiest."""
        return (-self.gpu_time, round(self.interarrival_cv, 3), -self.count)


def _interarrival_cv(instances: Sequence[dict]) -> float:
    """Variation in the spacing between consecutive instances.

    A once-per-iteration annotation arrives on a steady cadence. Used to rank
    families, not reject them, so it needs no threshold.
    """
    stamps = sorted(e.get("ts", 0) for e in instances)
    gaps = [b - a for a, b in zip(stamps, stamps[1:]) if b > a]
    if len(gaps) < 2:
        return 0.0
    average = mean(gaps)
    return pstdev(gaps) / average if average else 0.0


def collect_annotations(events: Sequence[dict]) -> List[dict]:
    """CPU-side *marker* annotations, in time order.

    A split point must be a semantic iteration marker, not an operation. Real
    tensor ops -- collectives especially (``nccl:*`` / ``gloo:*``) -- are emitted
    as annotations too, but they carry ``Input Dims`` because they act on
    tensors. Iteration markers (``step[DECODE]``, ``execute_...``, ``DataLoader``)
    never do. Excluding anything with ``Input Dims`` keeps a collective from being
    chosen as the iteration root, which is how an all-gather was winning before.
    """
    annotations = [
        e
        for e in events
        if e.get("cat") == ANNOTATION_CAT
        and e.get("ts") is not None
        and e.get("dur") is not None
        and "Input Dims" not in (e.get("args") or {})
    ]
    annotations.sort(key=lambda e: e["ts"])
    return annotations


def build_families(
    annotations: Sequence[dict], attribution: GpuAttribution
) -> List[AnnotationFamily]:
    """Group annotations into families and drop the ones with no GPU work.

    Pruning first is the point of the ordering: roughly half the families are
    pure CPU scheduling chatter that nesting analysis would otherwise carry.

    A family survives if *any* instance has GPU work, not all -- a scheduler
    family may have thousands of instances where only hundreds wrap real work.
    """
    grouped: Dict[str, List[dict]] = {}
    for event in annotations:
        grouped.setdefault(name_skeleton(event.get("name", "")), []).append(event)

    families = [
        AnnotationFamily(
            skeleton=skeleton,
            instances=instances,
            gpu_time=attribution.gpu_time_for_family(skeleton, instances),
            parseable=any(is_parseable(e.get("name", "")) for e in instances),
            interarrival_cv=_interarrival_cv(instances),
        )
        for skeleton, instances in grouped.items()
    ]

    # "No projection means no GPU work" is an inference, not a guarantee. If it
    # would delete every family on a trace that plainly has kernels, it is wrong
    # here -- fall back to following launch correlations instead.
    if families and attribution.kernels and not any(f.gpu_time for f in families):
        for family in families:
            family.gpu_time = attribution.gpu_time_by_correlation(family.instances)

    return [f for f in families if f.gpu_time > 0]


# --- steps ------------------------------------------------------------------
def detect_from_graph_launches(
    events: Sequence[dict],
    attribution: GpuAttribution,
    annotations: Sequence[dict],
) -> Optional[RootSet]:
    """Detect iteration roots from graph-replay launches.

    Each ``hipGraphLaunch`` / ``cudaGraphLaunch`` event maps 1:1 to a captured
    graph replay.  This signal survives graph capture (which erases the per-op
    python/kernel periodicity the tree traversal relies on) and needs no call
    tree, so it is tried before the expensive tree-based detectors.

    Returns a graded :class:`RootSet` when the launches explain enough GPU work
    (``COVERAGE_FLOOR``), or ``None`` when there are too few launches or the
    coverage is too poor to trust.
    """
    launches = sorted(
        (
            e
            for e in events
            if e.get("name") in GRAPH_LAUNCH_NAMES and e.get("ts") is not None
        ),
        key=lambda e: e["ts"],
    )
    if len(launches) < MIN_LABEL_CHILDREN:
        return None

    roots = [dict(e) for e in launches]
    coverage = attribution.audit(annotations, roots)
    if coverage.covered_selected < COVERAGE_FLOOR:
        return None

    status = (
        DetectStatus.SPLITTABLE
        if coverage.covered_selected >= COVERAGE_GATE
        else DetectStatus.DEGRADED
    )
    return RootSet(
        roots=roots,
        method="generic:graph_launch",
        phase_confidence=PhaseConfidence.UNKNOWN,
        status=status,
        coverage=coverage,
        diagnostics={"graph_launch_count": len(roots)},
    )
