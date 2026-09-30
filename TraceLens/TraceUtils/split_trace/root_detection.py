###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Annotation families and tree-based detection helpers.

Steps 1 and 2 group every annotation into families, known and unknown alike.
``detect_from_branch_descent`` and ``detect_from_sibling_roots`` handle the
tree-based detection path for traces no annotation family explains.
"""

from typing import Dict, List, Optional, Sequence

from collections import deque

from ...util import normalize_name_for_comparison
from ...Trace2Tree.trace_to_tree import TraceToTree
from ...Trace2Tree.util import (
    _descendant_gpu_time,
    _entry_roots,
    _reattach_worker_threads,
)
from ..utils.annotation_utils import (
    find_known_annotations,
    name_skeleton,
)
from ..utils.detect_utils import (
    COVERAGE_FLOOR,
    COVERAGE_GATE,
    MIN_ROOTS,
    DetectStatus,
    GpuAttribution,
    PhaseConfidence,
    RootSet,
    EventIndex,
)
from .period_detection import (
    _blocks_by_pattern,
    _find_repeating_period,
)

# Label sequences shorter than this are utility-function child lists, not loops.
MIN_LABEL_CHILDREN = 4

# The per-iteration windows must explain at least this share of GPU time; below
# it the repeating family is a sub-loop, not the iteration boundary.
BRANCH_COVERAGE_GATE = 0.95
# Bound the descent so a pathological tree cannot walk forever / explode a level.
BRANCH_MAX_NODES = 200000
# A branch/sibling candidate is extended with warmup/wrapup bookends only when it
# already explains at least this share of GPU time on its own.
BOOKEND_FLOOR = 0.50


# --- steps ------------------------------------------------------------------
def _grade(coverage: float) -> DetectStatus:
    if coverage >= COVERAGE_GATE:
        return DetectStatus.SPLITTABLE
    if coverage >= COVERAGE_FLOOR:
        return DetectStatus.DEGRADED
    return DetectStatus.NOT_SPLITTABLE


def _child_groups(tree: TraceToTree, ordered: Sequence[dict], norm: Dict) -> tuple:
    """Children grouped by normalized name, with the GPU time under each group.

    ``norm`` maps each child's UID to its normalized name (line numbers and hex
    addresses stripped), built once per node in :func:`find_pattern` so a name
    is normalized only once and shared with :func:`_periodic_candidate`.
    Normalizing lets otherwise-identical iteration frames whose call-site drifts
    still cluster together, matching how ``_periodic_candidate`` and
    ``_blocks_by_pattern`` compare names. Computed once per node and shared by
    both candidate paths, since walking the subtree is the expensive part of
    visiting a node.
    """
    groups: Dict[str, List[dict]] = {}
    for child in ordered:
        groups.setdefault(norm[child["UID"]], []).append(child)
    gpu = {name: _descendant_gpu_time(tree, inst) for name, inst in groups.items()}
    return groups, gpu


def _branch_candidate(
    roots: List[dict],
    total_gpu: float,
    depth: int,
    source: str,
    period: Optional[int],
    gpu_time: float,
    root_gpu_times: List[float],
) -> RootSet:
    cov = gpu_time / total_gpu
    diagnostics = {
        "period_label_tier": "branch_descent",
        "period": period,
        "period_depth": depth,
        "branch_source": source,
        "branch_coverage": round(cov, 4),
        "iter_gpu_time": gpu_time,
        "root_gpu_times": root_gpu_times,
    }
    return RootSet(
        roots=roots,
        method="generic:branch_descent",
        phase_confidence=PhaseConfidence.UNKNOWN,
        status=_grade(cov),
        diagnostics=diagnostics,
    )


def _bookend_diagnostics(
    blocked: Sequence[dict],
    prefix: Sequence[dict],
    suffix: Sequence[dict],
) -> Dict:
    """The unpromoted remainder, for the cascade to offer as bookend roots.

    Only UIDs: resolving them to events and pricing their GPU time is the
    cascade's job, and doing it here would walk the subtree a second time on the
    descent's hot path.
    """
    blocked_uids = {e.get("UID") for e in blocked}
    return {
        "before_uids": [
            e.get("UID") for e in prefix if e.get("UID") not in blocked_uids
        ],
        "after_uids": [
            e.get("UID") for e in suffix if e.get("UID") not in blocked_uids
        ],
    }


def _periodic_candidate(
    tree: TraceToTree,
    node: dict,
    ordered: Sequence[dict],
    gputime_by_name: Dict[str, float],
    total_gpu: float,
    depth: int,
    norm: Dict,
) -> Optional[RootSet]:
    """One candidate per repetition of a contiguous repeating child-name run.

    ``norm`` maps each child's UID to its normalized name (shared with
    :func:`_child_groups`). Frames whose *name* carries no GPU work anywhere are
    dropped before the period search.
    """
    live = [e for e in ordered if gputime_by_name.get(norm[e["UID"]], 0.0) > 0]
    period, pattern, start = _find_repeating_period([norm[e["UID"]] for e in live])
    if period is None or period == 1:
        return None
    unit_blocks = _blocks_by_pattern(live, pattern, start, norm)
    if len(unit_blocks) < MIN_LABEL_CHILDREN:
        return None
    prefix = list(live[:start])
    blocked_uids = {e.get("UID") for b in unit_blocks for e in b}
    last_block_end = unit_blocks[-1][-1]["ts"] + unit_blocks[-1][-1].get("dur", 0)
    suffix = [
        e
        for e in live
        if e["ts"] >= last_block_end and e.get("UID") not in blocked_uids
    ]
    iteration_roots: List[dict] = []
    root_gpu_times: List[float] = []
    blocked: List[dict] = []
    for block in unit_blocks:
        first, last = block[0], block[-1]
        event = dict(first)
        event["name"] = node.get("name", event.get("name", ""))
        event["dur"] = (last["ts"] + last.get("dur", 0)) - first["ts"]
        iteration_roots.append(event)
        root_gpu_times.append(_descendant_gpu_time(tree, block))
        blocked.extend(block)
    # Blocks own disjoint subtrees, so their GPU times sum without overlap.
    candidate = _branch_candidate(
        iteration_roots,
        total_gpu,
        depth,
        "period",
        period,
        sum(root_gpu_times),
        root_gpu_times,
    )
    candidate.diagnostics.update(_bookend_diagnostics(blocked, prefix, suffix))
    return candidate


def _grouped_candidate(
    tree: TraceToTree,
    ordered: Sequence[dict],
    groups: Dict[str, List[dict]],
    gputime_by_name: Dict[str, float],
    total_gpu: float,
    depth: int,
) -> Optional[RootSet]:
    """One candidate from the recurring child frame that carries the GPU work.

    A *conditional* loop body has no contiguous period. Grouping by name ignores
    the gaps, exactly as the unknown-family detector does one level up. The
    family carrying the most GPU time wins; work before its first instance and
    after its last is offered as warmup/wrapup bookends.
    """
    families = [
        (gputime_by_name[name], name, instances)
        for name, instances in groups.items()
        if len(instances) >= MIN_LABEL_CHILDREN
    ]
    families = [f for f in families if f[0] > 0]
    if not families:
        return None
    families.sort(key=lambda f: f[0], reverse=True)
    gpu_time, _, instances = families[0]

    blocked = list(instances)
    first_ts = instances[0].get("ts", 0)
    last_end = instances[-1].get("ts", 0) + instances[-1].get("dur", 0)
    prefix = [e for e in ordered if e.get("ts", 0) < first_ts]
    suffix = [e for e in ordered if e.get("ts", 0) >= last_end]
    root_gpu_times = [_descendant_gpu_time(tree, [e]) for e in instances]

    candidate = _branch_candidate(
        blocked,
        total_gpu,
        depth,
        "frame_family",
        None,
        gpu_time,
        root_gpu_times,
    )
    candidate.diagnostics.update(_bookend_diagnostics(blocked, prefix, suffix))
    return candidate


_GAP_RATIO_THRESHOLD = 10.0


def _filter_low_gpu_roots(
    candidate: RootSet, tree: TraceToTree, total_gpu: float
) -> RootSet:
    """Drop roots whose GPU time falls below a natural cluster gap.

    Sorts per-root GPU times, finds the largest multiplicative gap between
    consecutive values, and drops everything on the low side if the gap
    ratio exceeds ``_GAP_RATIO_THRESHOLD``.
    """
    gpu_times = candidate.diagnostics.get("root_gpu_times")
    if not gpu_times:
        return candidate

    nonzero = [(i, g) for i, g in enumerate(gpu_times) if g > 0]
    if len(nonzero) < MIN_LABEL_CHILDREN:
        return candidate
    indexed = sorted(nonzero, key=lambda x: x[1])
    max_ratio = 1.0
    split_at = -1
    for i in range(len(indexed) - 1):
        ratio = indexed[i + 1][1] / indexed[i][1]
        if ratio > max_ratio:
            max_ratio = ratio
            split_at = i

    if max_ratio < _GAP_RATIO_THRESHOLD:
        return candidate

    keep_indices = {idx for idx, _ in indexed[split_at + 1 :]}
    if len(keep_indices) < MIN_LABEL_CHILDREN:
        return candidate

    new_roots = [r for i, r in enumerate(candidate.roots) if i in keep_indices]
    new_gpu_times = [g for i, g in enumerate(gpu_times) if i in keep_indices]
    new_gpu_total = sum(new_gpu_times)
    cov = new_gpu_total / total_gpu if total_gpu else 0.0

    diag = dict(candidate.diagnostics)
    diag["root_gpu_times"] = new_gpu_times
    diag["branch_coverage"] = round(cov, 4)
    diag["iter_gpu_time"] = new_gpu_total
    diag["dropped_low_gpu_roots"] = len(candidate.roots) - len(new_roots)

    return RootSet(
        roots=new_roots,
        method=candidate.method,
        phase_confidence=candidate.phase_confidence,
        status=_grade(cov),
        diagnostics=diag,
    )


def find_pattern(
    tree: TraceToTree,
    ordered: Sequence[dict],
    total_gpu: float,
    node: Optional[dict] = None,
    depth: int = 0,
) -> Optional[RootSet]:
    """Find the best repeating pattern in an ordered list of children.

    Tries both a contiguous repeating name run (``_periodic_candidate``) and
    the best name-grouped child frame (``_grouped_candidate``), returning
    whichever covers more GPU work. Roots with negligible GPU time are
    dropped when a natural cluster gap separates them from the real
    iterations.

    When *node* is provided, iteration roots are named after it (branch
    descent). Otherwise each root keeps its own name (sibling roots).
    """
    norm = {e["UID"]: normalize_name_for_comparison(e.get("name", "")) for e in ordered}
    groups, gputime_by_name = _child_groups(tree, ordered, norm)
    best: Optional[RootSet] = None
    dummy_node = node if node is not None else {}
    for candidate in (
        _periodic_candidate(
            tree, dummy_node, ordered, gputime_by_name, total_gpu, depth, norm
        ),
        _grouped_candidate(tree, ordered, groups, gputime_by_name, total_gpu, depth),
    ):
        if candidate is None:
            continue
        cov = candidate.diagnostics["branch_coverage"]
        if cov > 0 and (best is None or cov > best.diagnostics["branch_coverage"]):
            best = candidate
    if best is not None:
        best = _filter_low_gpu_roots(best, tree, total_gpu)
    return best


def detect_from_branch_descent(
    tree: TraceToTree,
    entry_roots: List[dict],
    total_gpu: float,
) -> Optional[RootSet]:
    """Walk the call tree to find the frame whose children repeat and cover the GPU.

    At every node, :func:`find_pattern` tries both a contiguous repeating name
    run and the best name-grouped child frame. The BFS keeps descending past
    nodes whose candidates explain too little GPU work (sub-loops). Returns the
    best :class:`RootSet` found, or ``None``.
    """
    best: Optional[RootSet] = None
    queue = deque((r, 0) for r in entry_roots)
    visited = 0
    while queue:
        node, depth = queue.popleft()
        visited += 1
        if visited > BRANCH_MAX_NODES:
            break
        children = tree.get_children_events(node)
        if len(children) >= MIN_LABEL_CHILDREN:
            ordered = sorted(children, key=lambda e: e.get("ts", 0))
            candidate = find_pattern(tree, ordered, total_gpu, node=node, depth=depth)
            if candidate is not None and (
                best is None
                or candidate.diagnostics["branch_coverage"]
                > best.diagnostics["branch_coverage"]
            ):
                best = candidate
            if (
                best is not None
                and best.diagnostics["branch_coverage"] >= BRANCH_COVERAGE_GATE
            ):
                break
        for child in children:
            if not child.get("non_gpu_path", False):
                queue.append((child, depth + 1))

    if best is not None:
        best.diagnostics["_events_by_uid"] = tree.events_by_uid
    return best


def detect_from_sibling_roots(
    tree: TraceToTree,
    entry_roots: List[dict],
    total_gpu: float,
) -> Optional[RootSet]:
    """Detect iterations that are top-level sibling frames.

    Returns a :class:`RootSet` with coverage info, or ``None`` when there is
    no repeating pattern among the entry roots. The caller decides whether
    coverage is acceptable.
    """
    ordered = sorted(entry_roots, key=lambda e: e.get("ts", 0))
    candidate = find_pattern(tree, ordered, total_gpu)
    if candidate is None:
        return None
    candidate.method = "generic:sibling_roots"
    candidate.diagnostics["period_label_tier"] = "sibling_roots"
    candidate.diagnostics["_events_by_uid"] = tree.events_by_uid
    return candidate


# ---------------------------------------------------------------------------
# Cascade: find_iteration_roots
# ---------------------------------------------------------------------------


def _annotation_root_set(
    roots: Sequence[dict],
    attribution: GpuAttribution,
    method: str,
    phase_confidence: PhaseConfidence,
    root_family_known: bool,
) -> RootSet:
    """Build a graded RootSet from a set of annotation roots.

    Audits GPU coverage and sets status: SPLITTABLE when the roots explain
    enough GPU work -- unless the match is a suspiciously short (warmup-only)
    run and the labels aren't known -- DEGRADED above the floor, else
    NOT_SPLITTABLE.

    ``root_family`` lists every distinct name skeleton present: one for a
    single-family set (unknown families, vLLM), several when known iteration
    annotations span phases (SGLang EXTEND/DECODE, ATOM prefill/decode).
    """
    ordered = sorted(roots, key=lambda e: e.get("ts", 0))
    coverage = attribution.audit(ordered)
    few_roots = len(ordered) < MIN_ROOTS
    known_labels = phase_confidence is PhaseConfidence.HIGH
    if coverage.passes and (known_labels or not few_roots):
        status = DetectStatus.SPLITTABLE
    elif coverage.covered_selected >= COVERAGE_FLOOR:
        status = DetectStatus.DEGRADED
    else:
        status = DetectStatus.NOT_SPLITTABLE

    skeletons = sorted({name_skeleton(r.get("name", "")) for r in ordered})
    return RootSet(
        roots=ordered,
        method=method,
        phase_confidence=phase_confidence,
        status=status,
        coverage=coverage,
        diagnostics={
            "n_roots": len(ordered),
            "root_family": ", ".join(skeletons),
            "root_family_known": root_family_known,
            "suspiciously_few_roots": few_roots,
        },
    )


def _detect_from_known_annotations(
    annotations: Sequence[dict],
    attribution: GpuAttribution,
) -> Optional[RootSet]:
    known = find_known_annotations(annotations)
    if not known:
        return None
    return _annotation_root_set(
        known,
        attribution,
        method="annotation:tier",
        phase_confidence=PhaseConfidence.HIGH,
        root_family_known=True,
    )


def _detect_from_unknown_annotations(
    annotations: Sequence[dict],
    attribution: GpuAttribution,
) -> Optional[RootSet]:
    grouped: Dict[str, List[dict]] = {}
    for event in annotations:
        grouped.setdefault(name_skeleton(event.get("name", "")), []).append(event)

    candidates = []
    for instances in grouped.values():
        if len(instances) < MIN_ROOTS:
            continue
        gpu_time = attribution.gpu_time_for_family(instances)
        if gpu_time <= 0:
            continue
        candidates.append((gpu_time, instances))

    if not candidates:
        return None
    _, instances = max(candidates, key=lambda c: c[0])
    return _annotation_root_set(
        instances,
        attribution,
        method="family:unknown_only",
        phase_confidence=PhaseConfidence.UNKNOWN,
        root_family_known=False,
    )


def _log_attempt(step: str, root_set: Optional[RootSet]) -> None:
    if root_set is None:
        print(f"[roots] {step}: no candidate")
        return
    cov = root_set.coverage
    max_shown = 3

    skeletons = sorted({name_skeleton(r.get("name", "")) for r in root_set.roots})
    shown = ", ".join(
        s if len(s) <= 60 else s[:57] + "..." for s in skeletons[:max_shown]
    )
    if len(skeletons) > max_shown:
        shown += f", +{len(skeletons) - max_shown} more"
    head: str = (
        f"[roots] {step}: {len(root_set.roots)} roots via {root_set.method} "
        f"[{shown}]"
    )
    if cov is None:
        print(f"{head}, coverage not measured, status={root_set.status.name}")
        return
    print(
        f"{head}, GPU coverage={cov.covered_selected:.1%}, "
        f"{cov.span_share:.1%} inside root spans "
        f"({cov.strategy}), status={root_set.status.name}"
    )


def _try_bookend_enhancement(
    candidate: RootSet,
    tree: TraceToTree,
    total_gpu: float,
) -> Optional[RootSet]:
    if not total_gpu or not candidate.roots:
        return None

    uid_map = tree.events_by_uid
    before = [
        uid_map[uid]
        for uid in candidate.diagnostics.get("before_uids", ())
        if uid in uid_map
    ]
    after = [
        uid_map[uid]
        for uid in candidate.diagnostics.get("after_uids", ())
        if uid in uid_map
    ]
    if not before and not after:
        return None

    before_gpu = _descendant_gpu_time(tree, before) if before else 0.0
    after_gpu = _descendant_gpu_time(tree, after) if after else 0.0
    if before_gpu <= 0 and after_gpu <= 0:
        return None

    def _span(events: Sequence[dict], name: str) -> dict:
        ordered = sorted(events, key=lambda e: e["ts"])
        last = ordered[-1]
        root = dict(ordered[0])
        root["name"] = name
        root["dur"] = last["ts"] + last.get("dur", 0) - ordered[0]["ts"]
        return root

    roots = list(candidate.roots)
    diagnostics = dict(candidate.diagnostics)
    if before_gpu > 0:
        roots.insert(0, _span(before, "warmup"))
        diagnostics["warmup_gpu_pct"] = round(100 * before_gpu / total_gpu, 1)
    if after_gpu > 0:
        roots.append(_span(after, "wrapup"))
        diagnostics["wrapup_gpu_pct"] = round(100 * after_gpu / total_gpu, 1)

    coverage = (
        before_gpu + diagnostics.get("iter_gpu_time", 0.0) + after_gpu
    ) / total_gpu
    diagnostics["bookend_enhancement"] = True
    diagnostics["branch_coverage"] = round(coverage, 4)
    diagnostics["_events_by_uid"] = tree.events_by_uid
    return RootSet(
        roots=roots,
        method=candidate.method,
        phase_confidence=candidate.phase_confidence,
        status=_grade(coverage),
        diagnostics=diagnostics,
    )


def _candidate_coverage(candidate: RootSet) -> float:
    """GPU coverage for ranking, from whichever measure the candidate carries.

    Annotation and bookend candidates have an audited ``coverage`` report; the
    branch/sibling detectors record their estimate in ``branch_coverage``.
    """
    if candidate.coverage:
        return candidate.coverage.covered_selected
    return candidate.diagnostics.get("branch_coverage", 0.0)


def find_iteration_roots(
    events: Sequence[dict],
    trace_index: EventIndex = None,
) -> RootSet:
    """Find iteration roots and report how much GPU work they account for.

    Flat cascade -- each step is tried in order and returns as soon as a
    detector produces roots with acceptable GPU coverage:

    1. Known annotation patterns (vLLM execute_*, SGLang step[*], etc.)
    2. Unknown annotation families (ProfilerStep, scheduler.run_batch, etc.)
    3. Descend down call tree root, searching for repeating patterns.
    4. Search across roots, looking for periodicity across top-level frames.
    """
    attribution = GpuAttribution(trace_index)
    annotations = trace_index.annotations
    all_candidates: List[RootSet] = []

    def _check(step: str, root_set: Optional[RootSet]) -> bool:
        """Log the result and return True if SPLITTABLE."""
        _log_attempt(step, root_set)
        if root_set is not None:
            all_candidates.append(root_set)
        return root_set is not None and root_set.status is DetectStatus.SPLITTABLE

    # --- 1. Known annotation patterns -----------------------------------------
    step1 = _detect_from_known_annotations(annotations, attribution)
    if _check("1 known annotations", step1):
        return step1

    # --- 2. Unknown annotation families ---------------------------------------
    step2 = _detect_from_unknown_annotations(annotations, attribution)
    if _check("2 unknown families", step2):
        return step2

    # --- 3 & 4. Tree-based detectors (built once) ----------------------------
    try:
        tree = TraceToTree(list(events), prune_nongpu_paths=True)
        tree.build_tree(add_python_func=True)
    except Exception as exc:
        print(f"TraceToTree build failed ({exc}), skipping tree detectors.")
        if all_candidates:
            best = max(all_candidates, key=_candidate_coverage)
            print("Returning annotation family with best coverage as fallback")
            return best
        return RootSet(
            roots=[],
            method="none",
            status=DetectStatus.NOT_SPLITTABLE,
            diagnostics={"reason": "tree build failed and no annotations"},
        )

    tree = _reattach_worker_threads(tree)
    entry_roots = _entry_roots(tree)
    total_gpu = attribution.gpu_busy

    # --- 3. Branch descent ----------------------------------------------------
    branch_set = detect_from_branch_descent(tree, entry_roots, total_gpu)
    if _check("3 branch descent", branch_set):
        return branch_set

    # --- 4. Sibling roots ----------------------------------------------------
    sibling_set = detect_from_sibling_roots(tree, entry_roots, total_gpu)
    if _check("4 sibling roots", sibling_set):
        return sibling_set

    # --- 5. Bookend enhancement ----------------------------------------------
    bookend_set = None
    for candidate in (branch_set, sibling_set):
        if candidate is None:
            continue
        if candidate.diagnostics.get("branch_coverage", 0) < BOOKEND_FLOOR:
            continue
        bookend_set = _try_bookend_enhancement(candidate, tree, total_gpu)
        if bookend_set is not None:
            bookend_set.coverage = attribution.audit(bookend_set.roots)
            break
    if _check("5 bookend enhancement", bookend_set):
        return bookend_set

    # --- Return the best result across all detectors --------------------------
    usable = [c for c in all_candidates if c.status is not DetectStatus.NOT_SPLITTABLE]
    if usable:
        best = max(usable, key=_candidate_coverage)
        _log_attempt("fallback (best usable)", best)
        return best
    if all_candidates:
        best = max(all_candidates, key=_candidate_coverage)
        _log_attempt("fallback (last resort)", best)
        return best
    return RootSet(
        roots=[],
        method="none",
        status=DetectStatus.NOT_SPLITTABLE,
        diagnostics={
            "reason": "no splittable annotations and no repeating call pattern"
        },
    )
