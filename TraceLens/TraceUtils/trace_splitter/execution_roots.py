###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Stage 1: find iteration execution roots in an inference trace."""

from typing import Optional, Sequence

from ...Trace2Tree.inference_iteration_roots import (
    _descendant_gpu_time,
    _entry_roots,
    _reattach_worker_threads,
    GPU_KERNEL_CATS,
)
from ...Trace2Tree.trace_to_tree import TraceToTree
from ..annotation_utils import (
    find_known_annotations,
    is_parseable,
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
    _grade,
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


def _try_bookend_enhancement(
    candidate: RootSet,
    tree: TraceToTree,
    total_gpu: float,
) -> Optional[RootSet]:
    """Add warmup and/or wrapup bookend roots to improve coverage.

    Uses the before_uids / after_uids lists stored in diagnostics by the
    branch_descent and sibling_roots detectors.  These are the UIDs of
    GPU-bearing siblings that fall outside the repeating pattern.
    Only adds a bookend if it contributes GPU time.
    """
    if not total_gpu or not candidate.roots:
        return None

    before_uids = candidate.diagnostics.get("before_uids", [])
    after_uids = candidate.diagnostics.get("after_uids", [])
    if not before_uids and not after_uids:
        return None

    before = [tree.events_by_uid[uid] for uid in before_uids if uid in tree.events_by_uid]
    after = [tree.events_by_uid[uid] for uid in after_uids if uid in tree.events_by_uid]

    before_gpu = _descendant_gpu_time(tree, before) if before else 0
    after_gpu = _descendant_gpu_time(tree, after) if after else 0

    iter_gpu = candidate.diagnostics.get("iter_gpu_time", 0)
    new_cov = (before_gpu + iter_gpu + after_gpu) / total_gpu

    new_roots = list(candidate.roots)
    if before and before_gpu > 0:
        before_sorted = sorted(before, key=lambda e: e["ts"])
        warmup = dict(before_sorted[0])
        warmup["name"] = "warmup"
        warmup["dur"] = (
            before_sorted[-1]["ts"] + before_sorted[-1].get("dur", 0) - before_sorted[0]["ts"]
        )
        new_roots.insert(0, warmup)
    if after and after_gpu > 0:
        after_sorted = sorted(after, key=lambda e: e["ts"])
        wrapup = dict(after_sorted[0])
        wrapup["name"] = "wrapup"
        wrapup["dur"] = (
            after_sorted[-1]["ts"] + after_sorted[-1].get("dur", 0) - after_sorted[0]["ts"]
        )
        new_roots.append(wrapup)

    diag = dict(candidate.diagnostics)
    diag["bookend_enhancement"] = True
    diag["branch_coverage"] = round(new_cov, 4)
    if before_gpu > 0:
        diag["warmup_gpu_pct"] = round(100 * before_gpu / total_gpu, 1)
    if after_gpu > 0:
        diag["wrapup_gpu_pct"] = round(100 * after_gpu / total_gpu, 1)

    return RootSet(
        roots=new_roots,
        method=candidate.method,
        phase_confidence=candidate.phase_confidence,
        status=_grade(new_cov),
        diagnostics=diag,
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

    def _try(root_set: Optional[RootSet]) -> Optional[RootSet]:
        """Audit coverage; return the root_set if it passes, else save as fallback."""
        nonlocal best_fallback
        if root_set is None:
            return None
        coverage = attribution.audit(annotations, root_set.roots)
        root_set.coverage = coverage

        known_labels = root_set.phase_confidence is PhaseConfidence.HIGH
        if coverage.passes and (
            known_labels or not root_set.diagnostics.get("suspiciously_few_roots")
        ):
            root_set.status = DetectStatus.SPLITTABLE
            return root_set

        if coverage.covered_selected >= COVERAGE_FLOOR:
            root_set.status = DetectStatus.DEGRADED
        else:
            root_set.status = DetectStatus.NOT_SPLITTABLE

        if best_fallback is None or (
            root_set.status.value < best_fallback.status.value
        ):
            best_fallback = root_set
        return None

    # --- 1. Known annotation patterns -----------------------------------------
    result = _try(_detect_from_known_annotations(annotations))
    if result is not None:
        return result

    # --- 2. Unknown annotation families ---------------------------------------
    result = _try(_detect_from_unknown_family(annotations, attribution))
    if result is not None:
        return result

    # --- 3 & 4. Tree-based detectors (built once) ----------------------------
    try:
        tree = TraceToTree(list(events), prune_nongpu_paths=False)
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
    uid_map = tree.events_by_uid

    def _attach_uid_map(root_set: RootSet) -> RootSet:
        root_set.diagnostics["_events_by_uid"] = uid_map
        return root_set

    # --- 3. Branch descent ----------------------------------------------------
    branch_set = detect_from_branch_descent(tree, entry_roots, total_gpu)
    if branch_set is not None and branch_set.status is DetectStatus.SPLITTABLE:
        return _attach_uid_map(branch_set)

    # --- 4. Sibling roots ----------------------------------------------------
    sibling_set = detect_from_sibling_roots(tree, entry_roots, total_gpu)
    if sibling_set is not None and sibling_set.status is DetectStatus.SPLITTABLE:
        return _attach_uid_map(sibling_set)

    # --- 5. Bookend enhancement ------------------------------------------------
    # If a generic detector found iterations covering >=50% of GPU time but
    # not enough to pass, check whether adding a warmup block (before first
    # iteration) and/or wrapup block (after last iteration) improves coverage.
    BOOKEND_FLOOR = 0.50
    bookend_set = None
    for candidate in (branch_set, sibling_set):
        if candidate is None or not candidate.roots:
            continue
        cov = candidate.diagnostics.get("branch_coverage", 0)
        if cov < BOOKEND_FLOOR:
            continue
        bookend_set = _try_bookend_enhancement(candidate, tree, total_gpu)
        if bookend_set is not None:
            break
    if bookend_set is not None and bookend_set.status is DetectStatus.SPLITTABLE:
        return _attach_uid_map(bookend_set)

    # --- Return the best result across all detectors --------------------------
    for candidate in (bookend_set, branch_set, sibling_set, best_fallback):
        if (
            candidate is not None
            and candidate.status is not DetectStatus.NOT_SPLITTABLE
        ):
            return _attach_uid_map(candidate)
    for candidate in (bookend_set, branch_set, sibling_set, best_fallback):
        if candidate is not None:
            return _attach_uid_map(candidate)
    return RootSet(
        roots=[],
        method="none",
        status=DetectStatus.NOT_SPLITTABLE,
        diagnostics={"reason": "no annotations and no repeating call pattern"},
    )
