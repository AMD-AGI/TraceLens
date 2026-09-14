###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Stage 1: find iteration execution roots in an inference trace."""

from typing import Optional, Sequence

from ...Trace2Tree.inference_iteration_roots import (
    _entry_roots,
    _reattach_worker_threads,
)
from ...Trace2Tree.trace_to_tree import TraceToTree
from ..annotation_utils import (
    find_known_annotations,
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
from .root_detection import (
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


def _detect_from_known_annotations(
    annotations: Sequence[dict],
) -> Optional[RootSet]:
    """Try to find roots from a recognized annotation pattern.

    Returns a RootSet if any known pattern matches, or None.  Phase confidence
    is HIGH when every root is parseable, LOW when some are, UNKNOWN otherwise.
    """
    known = find_known_annotations(annotations)
    if not known:
        return None
    # ToDo: We do not need this check, clean up in a follow up PR.
    labelled = sum(1 for r in known if is_parseable(r.get("name", "")))
    if labelled == len(known):
        confidence = PhaseConfidence.HIGH
    elif labelled:
        confidence = PhaseConfidence.LOW
    else:
        confidence = PhaseConfidence.UNKNOWN

    return RootSet(
        roots=sorted(known, key=lambda e: e.get("ts", 0)),
        method="annotation:tier",
        phase_confidence=confidence,
        diagnostics={
            "n_known_roots": len(known),
            "suspiciously_few_roots": len(known) < MIN_ROOTS,
        },
    )


def _detect_from_unknown_family(
    annotations: Sequence[dict], attribution: GpuAttribution
) -> Optional[RootSet]:
    """Adopt the top-ranked regular annotation family with GPU work.

    This catches iteration markers that TraceLens doesn't recognize by regex
    (e.g. ProfilerStep, scheduler.run_batch) but that repeat regularly and
    correlate with GPU work.
    """
    families = build_families(annotations, attribution)
    regular = [f for f in families if f.regular]
    if not regular:
        return None
    # Find the family with the lowest rank, which is the family with the most GPU work.
    # Tie-break by the family with lowest interarrival CV
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


# Enough to show a mixed root set without one pathological case flooding the log.
_LOGGED_SKELETONS = 3


def _log_attempt(step: str, root_set: Optional[RootSet]) -> None:
    """Report what one cascade step found, so the escalation is traceable."""
    if root_set is None:
        print(f"[roots] {step}: no candidate")
        return
    cov = root_set.coverage

    skeletons = sorted({name_skeleton(r.get("name", "")) for r in root_set.roots})
    shown = ", ".join(
        s if len(s) <= 60 else s[:57] + "..." for s in skeletons[:_LOGGED_SKELETONS]
    )
    if len(skeletons) > _LOGGED_SKELETONS:
        shown += f", +{len(skeletons) - _LOGGED_SKELETONS} more"
    head = (
        f"[roots] {step}: {len(root_set.roots)} roots via {root_set.method} "
        f"[{shown}]"
    )
    if cov is None:
        print(f"{head}, coverage not measured, status={root_set.status.name}")
        return
    print(
        f"{head}, covered_selected={cov.covered_selected:.1%}, "
        f"span_share={cov.span_share:.1%} "
        f"({cov.strategy}), status={root_set.status.name}"
    )


def find_iteration_roots(events: Sequence[dict]) -> RootSet:
    """Find iteration roots and report how much GPU work they account for.

    Flat cascade -- each step is tried in order and returns as soon as a
    detector produces roots with acceptable GPU coverage:

    1. Known annotation patterns (vLLM execute_*, SGLang step[*], etc.)
    2. Unknown annotation families (ProfilerStep, scheduler.run_batch, etc.)
    3. Branch-descent on the call tree (after cross-thread reattachment)
    4. Sibling-root periodicity across top-level frames
    """
    attribution = GpuAttribution(events)
    annotations = collect_annotations(events)
    best_fallback: Optional[RootSet] = None

    def _try(step: str, root_set: Optional[RootSet]) -> Optional[RootSet]:
        """Audit coverage; return the root_set if it passes, else save as fallback."""
        nonlocal best_fallback
        if root_set is None:
            _log_attempt(step, None)
            return None
        coverage = attribution.audit(root_set.roots)
        root_set.coverage = coverage

        known_labels = root_set.phase_confidence is PhaseConfidence.HIGH
        if coverage.passes and (
            known_labels or not root_set.diagnostics.get("suspiciously_few_roots")
        ):
            root_set.status = DetectStatus.SPLITTABLE
            _log_attempt(step, root_set)
            return root_set

        if coverage.covered_selected >= COVERAGE_FLOOR:
            root_set.status = DetectStatus.DEGRADED
        else:
            root_set.status = DetectStatus.NOT_SPLITTABLE
        _log_attempt(step, root_set)

        if best_fallback is None or (
            root_set.status.value < best_fallback.status.value
        ):
            best_fallback = root_set
        return None

    # --- 1. Known annotation patterns -----------------------------------------
    result = _try("1 known annotations", _detect_from_known_annotations(annotations))
    if result is not None:
        return result

    # --- 2. Unknown annotation families ---------------------------------------
    result = _try(
        "2 unknown families", _detect_from_unknown_family(annotations, attribution)
    )
    if result is not None:
        return result

    # --- 3 & 4. Tree-based detectors (built once) ----------------------------
    try:
        tree = TraceToTree(list(events), prune_nongpu_paths=True)
        tree.build_tree(add_python_func=True)
    except Exception as exc:
        print(f"TraceToTree build failed ({exc}), skipping tree detectors.")
        if best_fallback is not None:
            return best_fallback
        return RootSet(
            roots=[],
            method="none",
            status=DetectStatus.NOT_SPLITTABLE,
            diagnostics={"reason": "tree build failed and no annotations"},
        )

    tree = _reattach_worker_threads(tree)
    entry_roots = _entry_roots(tree)
    total_gpu = _total_gpu_time(tree)
    if total_gpu == 0:
        return RootSet(
            roots=[],
            method="none",
            status=DetectStatus.NOT_SPLITTABLE,
            diagnostics={"reason": "no GPU work"},
        )
    # --- 3. Branch descent ----------------------------------------------------
    branch_set = detect_from_branch_descent(tree, entry_roots, total_gpu)
    if branch_set is not None:
        branch_set.coverage = attribution.audit(branch_set.roots)
    _log_attempt("3 branch descent", branch_set)
    if branch_set is not None and branch_set.status is DetectStatus.SPLITTABLE:
        return branch_set

    # --- 4. Sibling roots ----------------------------------------------------
    sibling_set = detect_from_sibling_roots(tree, entry_roots, total_gpu)
    if sibling_set is not None:
        sibling_set.coverage = attribution.audit(sibling_set.roots)
    _log_attempt("4 sibling roots", sibling_set)
    if sibling_set is not None and sibling_set.status is DetectStatus.SPLITTABLE:
        return sibling_set

    # --- Return the best result across all detectors --------------------------
    for candidate in (branch_set, sibling_set, best_fallback):
        if (
            candidate is not None
            and candidate.status is not DetectStatus.NOT_SPLITTABLE
        ):
            _log_attempt("fallback (best usable)", candidate)
            return candidate
    for candidate in (branch_set, sibling_set, best_fallback):
        if candidate is not None:
            _log_attempt("fallback (last resort)", candidate)
            return candidate
    return RootSet(
        roots=[],
        method="none",
        status=DetectStatus.NOT_SPLITTABLE,
        diagnostics={"reason": "no annotations and no repeating call pattern"},
    )
