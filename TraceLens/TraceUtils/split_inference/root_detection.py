###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Annotation families, and the detection steps built on them.

Step 1 groups every annotation into families, known and unknown alike. Step 1.5
chooses which nesting level is the iteration and where its metadata comes from.
Step 3 handles traces with no recognizable annotation, and step 5 falls back to
call-tree periodicity.
"""

from collections import Counter
from dataclasses import dataclass, field
from statistics import mean, pstdev
from typing import Dict, List, Optional, Sequence

from ...Trace2Tree.inference_iteration_roots import (
    GRAPH_LAUNCH_TIER,
    find_iteration_roots_generic,
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
    IntervalIndex,
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
    encloses: Counter = field(default_factory=Counter)

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

    def is_outer_to(self, inner: "AnnotationFamily") -> bool:
        """Whether this family encloses a majority of ``inner``'s instances."""
        if self.skeleton == inner.skeleton or not inner.count:
            return False
        return self.encloses.get(inner.skeleton, 0) / inner.count > NESTING_MAJORITY


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


def resolve_nesting(families: Sequence[AnnotationFamily], index: IntervalIndex) -> None:
    """How many instances of other families each family encloses.

    Computed per family, not per event, which would be quadratic.
    """
    for family in families:
        counts: Counter = Counter()
        for instance in family.instances:
            for inner in index.contained_in(instance):
                counts[name_skeleton(inner.get("name", ""))] += 1
        family.encloses = counts


# --- steps ------------------------------------------------------------------
def detect_from_unknown_family(
    events: Sequence[dict], attribution: GpuAttribution
) -> Optional[RootSet]:
    """Step 3: no parseable annotation anywhere, but a regular family exists.

    Adopts the family doing the most GPU work on the steadiest cadence. Phases
    are unknowable from an unrecognized name; the trace is still splittable.
    """
    annotations = collect_annotations(events)
    if not annotations:
        return None
    families = [f for f in build_families(annotations, attribution) if f.regular]
    if not families:
        return None

    family = min(families, key=lambda f: f.rank)
    return RootSet(
        roots=sorted(family.instances, key=lambda e: e.get("ts", 0)),
        method="family:unknown_only",
        phase_confidence=PhaseConfidence.UNKNOWN,
        diagnostics={
            "n_families": len(families),
            "root_family_skeleton": family.skeleton,
            "root_family_known": False,
        },
    )


def detect_generic(
    events: Sequence[dict], attribution: GpuAttribution
) -> Optional[RootSet]:
    """Step 5: grade the generic call-tree split into a :class:`RootSet`.

    The detectors in ``find_iteration_roots_generic`` all judge their own split by
    GPU coverage -- branch-descent and sibling-roots on tree-descendant kernels,
    graph launches on launch-window coverage here (they are gathered before a tree
    exists). So there is no kernel-stream period cross-check anymore: it duplicated
    the coverage guard and, worse, refused graph-replay splits because capture
    erases the per-step kernel period.
    """
    diagnostics: dict = {}
    roots = find_iteration_roots_generic(list(events), diagnostics)
    if not roots:
        return None

    tier = diagnostics.get("period_label_tier")
    method = f"generic:{tier}" if tier else "generic:python_function"

    # Graph launches are gathered off the flat event list (no tree), so their
    # coverage is audited here by projection / launch correlation rather than by
    # tree descendants. span_share is skipped because a launch is a point event
    # that explains ~no time on its own; kernel coverage is the real signal.
    if tier == GRAPH_LAUNCH_TIER:
        coverage = attribution.audit(collect_annotations(events), roots)
        if coverage.covered_selected >= COVERAGE_GATE:
            status = DetectStatus.SPLITTABLE
        elif coverage.covered_selected >= COVERAGE_FLOOR:
            status = DetectStatus.DEGRADED
        else:
            status = DetectStatus.NOT_SPLITTABLE
        diagnostics.update({"graph_launch_count": len(roots)})
        return RootSet(
            roots=roots,
            method=method,
            phase_confidence=PhaseConfidence.UNKNOWN,
            status=status,
            coverage=coverage,
            diagnostics=diagnostics,
        )

    # branch-descent and sibling-roots already gated on tree-descendant GPU
    # coverage inside the detector (cross-thread exact after reattachment), so
    # reaching here means the split cleared the bar -- accept it.
    return RootSet(
        roots=roots,
        method=method,
        phase_confidence=PhaseConfidence.UNKNOWN,
        status=DetectStatus.SPLITTABLE,
        diagnostics=diagnostics,
    )
