###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Stage 1: find iteration execution roots in an inference trace."""

from typing import List, Optional, Sequence, Tuple

from ...Trace2Tree.inference_iteration_roots import _entry_roots, _reattach_worker_threads
from ...Trace2Tree.trace_to_tree import TraceToTree
from ..annotation_utils import (
    PROVENANCE_KEY,
    find_known_annotations,
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
    _total_gpu_time,
    build_families,
    collect_annotations,
    detect_from_branch_descent,
    detect_from_sibling_roots,
)

__all__ = [
    "COVERAGE_FLOOR",
    "COVERAGE_GATE",
    "DetectStatus",
    "PhaseConfidence",
    "RootSet",
    "find_iteration_roots",
]


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


def _detect_from_annotations(
    attribution: GpuAttribution, annotations: Sequence[dict]
) -> Optional[RootSet]:
    """Steps 1-3: roots from a recognized annotation, or an unknown regular family.

    A recognized iteration pattern goes through step 1.5 -- widen to the
    enclosing family, relabel from the inner parseable annotation, and grade
    phase confidence by how many roots actually parsed. With no recognized
    pattern but a regular family present, that family is adopted directly: the
    trace is still splittable, just without phase labels.

    ``known`` is the subset of ``annotations`` matching a recognized pattern --
    matched over the pre-gathered list, inheriting the same collective filter (a
    real iteration marker never carries Input Dims).
    """
    families = build_families(annotations, attribution)
    known = find_known_annotations(annotations)

    if known:
        index = IntervalIndex(annotations)
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

        # Trust the phase labels only as far as they were actually parsed.
        # Adopting a whole family means some iterations may carry no recognizable
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

    # No recognized pattern: adopt the top-ranked regular family (most GPU work,
    # steadiest cadence), unlabeled -- phases are unknowable from an unrecognized
    # name, but the trace is still splittable.
    regular = [f for f in families if f.regular]
    if not regular:
        return None
    family = min(regular, key=lambda f: f.rank)
    return RootSet(
        roots=sorted(family.instances, key=lambda e: e.get("ts", 0)),
        method="family:unknown_only",
        phase_confidence=PhaseConfidence.UNKNOWN,
        diagnostics={
            "n_families": len(regular),
            "root_family_skeleton": family.skeleton,
            "root_family_known": False,
        },
    )


def find_iteration_roots(events: Sequence[dict]) -> RootSet:
    """Find iteration roots and report how much GPU work they account for.

    Flat cascade -- each step is tried in order and returns as soon as a
    detector produces roots with acceptable GPU coverage:

    1. Recognized / unknown-family annotations
    2. Branch-descent on the call tree (after cross-thread reattachment)
    3. Sibling-root periodicity across top-level frames
    """
    attribution = GpuAttribution(events)
    annotations = collect_annotations(events)

    # --- 1. Annotations (known pattern or regular unknown family) -------------
    root_set = _detect_from_annotations(attribution, annotations)
    if root_set is not None:
        coverage = attribution.audit(annotations, root_set.roots)
        root_set.coverage = coverage

        known_labels = root_set.phase_confidence is PhaseConfidence.HIGH
        if coverage.passes and (
            known_labels
            or not root_set.diagnostics.get("suspiciously_few_roots")
        ):
            root_set.status = DetectStatus.SPLITTABLE
            return root_set

        if coverage.covered_selected >= COVERAGE_FLOOR:
            root_set.status = DetectStatus.DEGRADED
            return root_set

    # --- 2 & 3. Tree-based detectors (built once) ----------------------------
    try:
        tree = TraceToTree(list(events), prune_nongpu_paths=False)
        tree.build_tree(add_python_func=True)
    except Exception as exc:
        print(f"TraceToTree build failed ({exc}), skipping tree detectors.")
        if root_set is not None:
            root_set.status = DetectStatus.NOT_SPLITTABLE
            return root_set
        return RootSet(
            roots=[],
            method="none",
            status=DetectStatus.NOT_SPLITTABLE,
            diagnostics={"reason": "tree build failed and no annotations"},
        )

    tree = _reattach_worker_threads(tree)
    entry_roots = _entry_roots(tree)
    total_gpu = _total_gpu_time(tree)

    # --- 2. Branch descent ----------------------------------------------------
    branch_set = detect_from_branch_descent(tree, entry_roots, total_gpu)
    if branch_set is not None and branch_set.status is not DetectStatus.NOT_SPLITTABLE:
        return branch_set

    # --- 3. Sibling roots ----------------------------------------------------
    sibling_set = detect_from_sibling_roots(tree, entry_roots, total_gpu)
    if sibling_set is not None and sibling_set.status is not DetectStatus.NOT_SPLITTABLE:
        return sibling_set

    # --- Nothing worked -------------------------------------------------------
    for candidate in (branch_set, sibling_set):
        if candidate is not None:
            return candidate
    if root_set is not None:
        root_set.status = DetectStatus.NOT_SPLITTABLE
        return root_set
    return RootSet(
        roots=[],
        method="none",
        status=DetectStatus.NOT_SPLITTABLE,
        diagnostics={"reason": "no annotations and no repeating call pattern"},
    )
