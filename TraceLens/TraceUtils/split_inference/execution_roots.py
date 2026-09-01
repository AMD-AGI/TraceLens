###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Stage 1: find iteration execution roots in an inference trace.

Two entry points. :func:`find_iteration_roots` is the original first-tier-wins
lookup, kept exactly as it was for callers that just want a root list.
:func:`find_iteration_roots_ex` runs the coverage-gated flow and reports how much
of the GPU's work the roots actually account for, which is the only way to tell a
correct root set from one that locked onto a warmup loop.
"""

from typing import List, Optional, Sequence, Tuple

from ..annotation_utils import (
    ITERATION_BACKUP_PATTERNS,
    ITERATION_PATTERNS,
    PROVENANCE_KEY,
    find_events_by_patterns,
    find_iteration_roots_by_priority,
    inherit_identity,
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
from .root_detection import (
    NESTING_MAJORITY,
    AnnotationFamily,
    build_families,
    collect_annotations,
    detect_from_unknown_family,
    detect_generic,
    resolve_nesting,
)

__all__ = [
    "COVERAGE_FLOOR",
    "COVERAGE_GATE",
    "DetectStatus",
    "PhaseConfidence",
    "RootSet",
    "find_iteration_roots",
    "find_iteration_roots_ex",
]


def find_iteration_roots(events: list[dict]) -> list[dict] | None:
    """Return iteration-root events.

    Tries the primary annotation pattern first, then backup patterns, then
    falls back to generic call-tree traversal via Trace2Tree.
    """
    roots = find_events_by_patterns(
        events, ITERATION_PATTERNS, label="execution steps (iteration)", verbose=True
    )
    if len(roots) == 0:
        print("No primary annotations found; falling back to backup patterns...")
        roots = find_events_by_patterns(
            events,
            ITERATION_BACKUP_PATTERNS,
            label="execution steps (iteration, backup)",
            verbose=True,
        )
    if len(roots) == 0:
        print("No annotation patterns found; trying generic call-tree traversal...")
        from ...Trace2Tree.inference_iteration_roots import (
            find_iteration_roots_generic,
        )

        roots = find_iteration_roots_generic(events)
    return roots


# --- step 1.5: separate the extraction window from the metadata --------------
def _widen_to_outer_family(
    roots: Sequence[dict],
    families: Sequence[AnnotationFamily],
    index: IntervalIndex,
) -> Tuple[Optional[List[dict]], Optional[AnnotationFamily]]:
    """Replace ``roots`` with an enclosing family's instances, where one exists.

    Enclosing the known roots is how the family is *found*, but the whole family
    is adopted: instances enclosing no known root are iterations whose inner
    annotation is merely unrecognized. Keeping only the matching ones is how a
    512-iteration run gets split into the 15 a regex happened to know.

    Ranking by GPU time separates the iteration boundary from the bookkeeping
    spans that wrap it equally well but do almost no work.
    """
    root_ids = {id(r) for r in roots}
    own = {name_skeleton(r.get("name", "")) for r in roots}
    threshold = NESTING_MAJORITY * len(roots)
    best = None
    for family in families:
        if family.skeleton in own or not family.regular:
            continue
        # Count the *roots* that end up wrapped, not the instances doing the
        # wrapping. One outer span often holds several known roots, and counting
        # instances then reports half when every root is in fact covered.
        wrapped = {
            id(e)
            for instance in family.instances
            for e in index.contained_in(instance)
            if id(e) in root_ids
        }
        if len(wrapped) > threshold and (best is None or family.rank < best.rank):
            best = family
    if best is None:
        return None, None
    return sorted(best.instances, key=lambda e: e.get("ts", 0)), best


def _relabel_from_inner(roots: Sequence[dict], index: IntervalIndex) -> List[dict]:
    """Give each root the identity of the parseable annotation inside it.

    The longest *parseable* inner annotation wins: an unparseable one leaves the
    metadata fabricated and silently collapses batch size to one.
    """
    relabelled = []
    for root in roots:
        own = name_skeleton(root.get("name", ""))
        inner = [
            e
            for e in index.contained_in(root)
            if is_parseable(e.get("name", ""))
            and name_skeleton(e.get("name", "")) != own
        ]
        if inner:
            root = inherit_identity(root, max(inner, key=lambda e: e.get("dur", 0)))
        relabelled.append(root)
    return relabelled


def _detect_annotated(
    events: Sequence[dict], attribution: GpuAttribution
) -> Optional[RootSet]:
    """Steps 1 and 1.5 for a trace with at least one recognized annotation."""
    known = find_iteration_roots_by_priority(events)
    if not known:
        return None

    annotations = collect_annotations(events)
    index = IntervalIndex(annotations)
    families = build_families(annotations, attribution)
    resolve_nesting(families, index)

    diagnostics = {
        "n_families": len(families),
        "n_known_roots": len(known),
    }
    widened, outer = _widen_to_outer_family(known, families, index)
    roots = widened if widened else list(known)
    if outer is not None:
        diagnostics["root_family_skeleton"] = outer.skeleton
        diagnostics["root_family_known"] = outer.parseable

    roots = _relabel_from_inner(roots, index)
    inherited = {
        r[PROVENANCE_KEY]["identity_from"] for r in roots if PROVENANCE_KEY in r
    }
    if inherited:
        diagnostics["inherited_from_skeleton"] = sorted(
            {name_skeleton(n) for n in inherited}
        )
    diagnostics["suspiciously_few_roots"] = len(roots) < MIN_ROOTS

    # Trust the phase labels only as far as they were actually parsed. Adopting a
    # whole family means some of its iterations may carry no recognizable
    # annotation at all, and calling that "high" would launder a guess.
    labelled = sum(1 for r in roots if is_parseable(r.get("name", "")))
    diagnostics["n_roots_with_phase"] = labelled
    if labelled == len(roots):
        confidence = PhaseConfidence.HIGH
    elif labelled:
        confidence = PhaseConfidence.LOW
    else:
        confidence = PhaseConfidence.UNKNOWN

    return RootSet(
        roots=sorted(roots, key=lambda e: e.get("ts", 0)),
        method="annotation:widened" if widened else "annotation:tier",
        phase_confidence=confidence,
        diagnostics=diagnostics,
    )


def find_iteration_roots_ex(events: Sequence[dict]) -> RootSet:
    """Find iteration roots and report how much GPU work they account for.

    Escalates only when it has to: recognized annotations, the nesting level
    around them, a coverage audit, probes, then call-tree periodicity.
    """
    attribution = GpuAttribution(events)
    annotations = collect_annotations(events)

    root_set = _detect_annotated(events, attribution)
    if root_set is None:
        root_set = detect_from_unknown_family(events, attribution)
    if root_set is None:
        generic = detect_generic(events, attribution)
        if generic is not None:
            return generic
        return RootSet(
            roots=[],
            method="none",
            status=DetectStatus.NOT_SPLITTABLE,
            diagnostics={"reason": "no annotations and no repeating call pattern"},
        )

    coverage = attribution.audit(annotations, root_set.roots)
    root_set.coverage = coverage
    root_set.diagnostics["probes_run"] = []  # escalation probes removed

    # Trust recognized labels. When the split comes from a parsed annotation
    # (phase_confidence high) and its GPU coverage is good, a small root count is
    # a genuinely short run, not a warmup loop -- accept it rather than second-
    # guessing known-per-iteration labels. The warmup guard still applies to
    # unknown families, whose meaning we cannot vouch for.
    known_labels = root_set.phase_confidence is PhaseConfidence.HIGH
    if coverage.passes and (
        known_labels or not root_set.diagnostics.get("suspiciously_few_roots")
    ):
        root_set.status = DetectStatus.SPLITTABLE
        return root_set

    # No probes: grade directly on how much GPU work the roots explain.
    if coverage and coverage.covered_selected >= COVERAGE_FLOOR:
        root_set.status = DetectStatus.DEGRADED
    else:
        root_set.status = DetectStatus.NOT_SPLITTABLE
    return root_set
