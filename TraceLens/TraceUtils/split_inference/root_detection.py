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
from typing import Dict, List, Optional, Sequence, Tuple

from ...Trace2Tree.inference_iteration_roots import (
    PERIOD_CONFLICT,
    compare_periods,
    find_iteration_roots_generic,
    find_period_candidates,
)
from ..annotation_utils import (
    ANNOTATION_CAT,
    inherit_identity,
    is_parseable,
    name_skeleton,
)
from .detect_utils import (
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
    """CPU-side annotation events, in time order."""
    annotations = [
        e
        for e in events
        if e.get("cat") == ANNOTATION_CAT
        and e.get("ts") is not None
        and e.get("dur") is not None
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


# --- step 1.5: choose the nesting level and where metadata comes from --------
def select_root_family(
    families: Sequence[AnnotationFamily],
) -> Tuple[Optional[AnnotationFamily], List[AnnotationFamily]]:
    """Outermost regular family that wraps something parseable.

    Separating "which span" from "which label" is the point: a scheduler span
    can be the right window while carrying no metadata, and the annotation that
    carries it can be too narrow. Returns the family and what it wraps.
    """
    parseable = [f for f in families if f.parseable]
    if not parseable:
        return None, []

    candidates = []
    for family in families:
        if not family.regular:
            continue
        wrapped = [p for p in parseable if family.is_outer_to(p)]
        if wrapped:
            candidates.append((family, wrapped))
    if not candidates:
        return None, []

    # Outermost means not enclosed by another candidate. Being outermost is not
    # sufficient on its own -- a whole-run wrapper would win every time -- which
    # is why only regular families were considered above.
    outermost = [
        (family, wrapped)
        for family, wrapped in candidates
        if not any(other.is_outer_to(family) for other, _ in candidates)
    ]
    pool = outermost or candidates
    return min(pool, key=lambda item: item[0].rank)


def enrich_roots(
    family: AnnotationFamily, index: IntervalIndex
) -> Tuple[List[dict], int, float]:
    """Roots from ``family``, each carrying its inner annotation's identity.

    Instances enclosing nothing parseable do no iteration work and are dropped.
    Their count and span share are returned because a large dropped share next
    to passing coverage means the wrong nesting level.
    """
    roots: List[dict] = []
    dropped_dur = 0.0
    total_dur = 0.0
    for instance in family.instances:
        total_dur += instance.get("dur", 0)
        inner = [
            e for e in index.contained_in(instance) if is_parseable(e.get("name", ""))
        ]
        if not inner:
            dropped_dur += instance.get("dur", 0)
            continue
        # Longest *parseable* child, not longest child: an unparseable winner
        # leaves the metadata fabricated and silently collapses batch size to 1.
        source = max(inner, key=lambda e: e.get("dur", 0))
        roots.append(inherit_identity(instance, source))
    roots.sort(key=lambda e: e.get("ts", 0))
    dropped = family.count - len(roots)
    return roots, dropped, (dropped_dur / total_dur if total_dur else 0.0)


# --- steps ------------------------------------------------------------------
def detect_from_families(
    events: Sequence[dict], attribution: GpuAttribution
) -> Optional[RootSet]:
    """Steps 1 and 1.5: families, then the chosen nesting level.

    ``None`` when no family wraps anything parseable, handing off to step 3.
    """
    annotations = collect_annotations(events)
    if not annotations:
        return None

    families = build_families(annotations, attribution)
    if not families:
        return None

    index = IntervalIndex(annotations)
    resolve_nesting(families, index)
    family, wrapped = select_root_family(families)

    diagnostics = {
        "n_families": len(families),
        "n_annotations": len(annotations),
    }
    if family is None:
        return None

    roots, dropped, dropped_share = enrich_roots(family, index)
    if not roots:
        return None

    inner = min(wrapped, key=lambda f: f.rank)
    diagnostics.update(
        {
            "root_family_skeleton": family.skeleton,
            "root_family_known": family.parseable,
            "inherited_from_skeleton": inner.skeleton,
            "n_root_instances_dropped": dropped,
            "dropped_gpu_time_share": round(dropped_share, 4),
            "suspiciously_few_roots": len(roots) < MIN_ROOTS,
        }
    )
    known = "known" if family.parseable else "unknown"
    return RootSet(
        roots=roots,
        method=f"family:{known}_outer+parseable_inner",
        phase_confidence=PhaseConfidence.HIGH,
        diagnostics=diagnostics,
    )


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
    """Step 5: call-tree periodicity, cross-checked against the kernel stream.

    Two independent detections make this trustworthy without a coverage gate.
    They are compared on *iteration count*, not raw period: the periods are
    measured in different units -- python frames versus kernels -- so a kernel
    loop at an exact multiple is the same loop at finer grain, and confirms it.
    """
    diagnostics: dict = {}
    roots = find_iteration_roots_generic(list(events), diagnostics)
    if not roots:
        return None

    kernel_names = [k.get("name", "") for k in attribution.kernels]
    kernel_candidates = find_period_candidates(kernel_names)
    kernel_blocks = kernel_candidates[0].repeats if kernel_candidates else None
    verdict, ratio = compare_periods(len(roots), kernel_blocks)

    diagnostics.update(
        {
            "kernel_loop_blocks": kernel_blocks,
            "period_agreement": verdict,
            "generic_ratio_k": ratio,
        }
    )
    if verdict == PERIOD_CONFLICT:
        return RootSet(
            roots=roots,
            method="generic:python_function",
            phase_confidence=PhaseConfidence.UNKNOWN,
            status=DetectStatus.NOT_SPLITTABLE,
            diagnostics=diagnostics,
        )
    return RootSet(
        roots=roots,
        method="generic:python_function",
        phase_confidence=PhaseConfidence.UNKNOWN,
        status=DetectStatus.SPLITTABLE,
        diagnostics=diagnostics,
    )
