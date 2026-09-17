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

from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Dict, List, Optional, Sequence

from collections import deque

from ...util import normalize_name_for_comparison
from ...Trace2Tree.inference_iteration_roots import (
    BRANCH_COVERAGE_GATE,
    BRANCH_DESCENT_TIER,
    BRANCH_MAX_NODES,
    GPU_KERNEL_CATS,
    MIN_LABEL_CHILDREN,
    _blocks_by_pattern,
    _descendant_gpu_time,
    _find_repeating_period,
)
from ...Trace2Tree.trace_to_tree import TraceToTree
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


@dataclass
class AnnotationFamily:
    """All instances of one logical annotation, keyed by its skeleton.

    CPU-side only: GPU annotation spans duplicate one annotation across streams,
    so any count from them is inflated. They give "has GPU work" signal only.
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
    """Variation in the spacing between consecutive instances."""
    stamps = sorted(e.get("ts", 0) for e in instances)
    gaps = [b - a for a, b in zip(stamps, stamps[1:]) if b > a]
    if len(gaps) < 2:
        return 0.0
    average = mean(gaps)
    return pstdev(gaps) / average if average else 0.0


def collect_annotations(events: Sequence[dict]) -> List[dict]:
    """CPU-side *marker* annotations, in time order.

    Input Dims are checked for to avoid selecting operation annotations as iteration roots.
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
    """Group annotations and drop the ones with no GPU work."""
    grouped: Dict[str, List[dict]] = {}
    for event in annotations:
        grouped.setdefault(name_skeleton(event.get("name", "")), []).append(event)

    families = [
        AnnotationFamily(
            skeleton=skeleton,
            instances=instances,
            gpu_time=attribution.gpu_time_for_family(instances),
            parseable=any(is_parseable(e.get("name", "")) for e in instances),
            interarrival_cv=_interarrival_cv(instances),
        )
        for skeleton, instances in grouped.items()
    ]

    return [f for f in families if f.gpu_time > 0]


# --- steps ------------------------------------------------------------------
def _total_gpu_time(tree: TraceToTree) -> float:
    return sum(
        e.get("dur", 0)
        for e in tree.events_by_uid.values()
        if e.get("cat") in GPU_KERNEL_CATS
    )


def _grade(coverage: float) -> DetectStatus:
    if coverage >= COVERAGE_GATE:
        return DetectStatus.SPLITTABLE
    if coverage >= COVERAGE_FLOOR:
        return DetectStatus.DEGRADED
    return DetectStatus.NOT_SPLITTABLE


def _child_groups(tree: TraceToTree, ordered: Sequence[dict]) -> tuple:
    """Children grouped by name, with the GPU time under each group.

    Computed once per node and shared by both candidate paths, since walking the
    subtree is the expensive part of visiting a node.
    """
    groups: Dict[str, List[dict]] = {}
    for child in ordered:
        groups.setdefault(child.get("name", ""), []).append(child)
    gpu = {name: _descendant_gpu_time(tree, inst) for name, inst in groups.items()}
    return groups, gpu


def _branch_candidate(
    tree: TraceToTree,
    roots: List[dict],
    blocked: List[dict],
    total_gpu: float,
    depth: int,
    source: str,
    period: Optional[int],
    gpu_time: Optional[float] = None,
    extra: Optional[Dict] = None,
) -> RootSet:
    if gpu_time is None:
        gpu_time = _descendant_gpu_time(tree, blocked)
    cov = gpu_time / total_gpu
    diagnostics = {
        "period_label_tier": BRANCH_DESCENT_TIER,
        "period": period,
        "period_depth": depth,
        "branch_source": source,
        "branch_coverage": round(cov, 4),
        "iter_gpu_time": gpu_time,
    }
    diagnostics.update(extra or {})
    return RootSet(
        roots=roots,
        method=f"generic:{BRANCH_DESCENT_TIER}",
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
) -> Optional[RootSet]:
    """One candidate per repetition of a contiguous repeating child-name run.

    Frames whose *name* carries no GPU work anywhere are dropped before the
    period search.
    """
    live = [e for e in ordered if gputime_by_name.get(e.get("name", ""), 0.0) > 0]
    if len(live) < MIN_LABEL_CHILDREN:
        return None
    period, pattern, start = _find_repeating_period(
        [normalize_name_for_comparison(e.get("name", "")) for e in live]
    )
    if period is None or period == 1:
        return None
    unit_blocks = _blocks_by_pattern(live, pattern, start)
    if len(unit_blocks) < MIN_LABEL_CHILDREN:
        return None
    prefix = list(live[:start])
    blocked_uids = {e.get("UID") for b in unit_blocks for e in b}
    last_block_end = (
        unit_blocks[-1][-1]["ts"] + unit_blocks[-1][-1].get("dur", 0)
    )
    suffix = [
        e for e in live
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
    extra = _bookend_diagnostics(blocked, prefix, suffix)
    extra["root_gpu_times"] = root_gpu_times
    return _branch_candidate(
        tree, iteration_roots, blocked, total_gpu, depth,
        "period", period, extra=extra,
    )


def _grouped_candidate(
    tree: TraceToTree,
    groups: Dict[str, List[dict]],
    gputime_by_name: Dict[str, float],
    total_gpu: float,
    depth: int,
) -> Optional[RootSet]:
    """One candidate from the recurring child frame that carries the GPU work.

    A *conditional* loop body has no contiguous period. Grouping by name ignores
    the gaps, exactly as ``build_families`` does one level up.

    Ranked by GPU time, then cadence, then count. Only the winning family becomes roots.
    """
    ranked = [
        (
            gputime_by_name[name],
            -_interarrival_cv(instances),
            len(instances),
            name,
            instances,
        )
        for name, instances in groups.items()
        if len(instances) >= MIN_LABEL_CHILDREN
    ]
    if not ranked:
        return None
    ranked.sort(key=lambda r: (r[0], r[1], r[2]), reverse=True)
    gpu_time, _, _, _, instances = ranked[0]
    if gpu_time <= 0:
        return None

    # Siblings own disjoint subtrees, so their GPU times add without overlap.
    with_gpu = [r for r in ranked if r[0] > 0]
    extra = {
        "branch_families_with_gpu": len(with_gpu),
        "branch_families_combined_coverage": round(
            sum(r[0] for r in with_gpu) / total_gpu, 4
        ),
        "branch_runner_up_families": [
            {"name": name, "count": count, "gpu_share": round(gpu / total_gpu, 4)}
            for gpu, _, count, name, _ in with_gpu[1:4]
        ],
    }
    root_gpu_times = [_descendant_gpu_time(tree, [e]) for e in instances]
    extra["root_gpu_times"] = root_gpu_times
    return _branch_candidate(
        tree, list(instances), list(instances), total_gpu, depth,
        "frame_family", None, gpu_time=gpu_time, extra=extra,
    )


_GAP_RATIO_THRESHOLD = 10.0


def _filter_low_gpu_roots(candidate: RootSet, tree: TraceToTree, total_gpu: float) -> RootSet:
    """Drop roots whose GPU time falls below a natural cluster gap.

    Sorts per-root GPU times, finds the largest multiplicative gap between
    consecutive values, and drops everything on the low side if the gap
    ratio exceeds ``_GAP_RATIO_THRESHOLD``.
    """
    gpu_times = candidate.diagnostics.get("root_gpu_times")
    if not gpu_times or len(gpu_times) < MIN_LABEL_CHILDREN:
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

    keep_indices = {idx for idx, _ in indexed[split_at + 1:]}
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
    if len(ordered) < MIN_LABEL_CHILDREN:
        return None
    groups, gputime_by_name = _child_groups(tree, ordered)
    best: Optional[RootSet] = None
    dummy_node = node if node is not None else {}
    for candidate in (
        _periodic_candidate(
            tree, dummy_node, ordered, gputime_by_name, total_gpu, depth
        ),
        _grouped_candidate(tree, groups, gputime_by_name, total_gpu, depth),
    ):
        if candidate is None:
            continue
        cov = candidate.diagnostics["branch_coverage"]
        if cov > 0 and (
            best is None or cov > best.diagnostics["branch_coverage"]
        ):
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
        print(
            f"[roots]   branch best: {len(best.roots)} roots via "
            f"{best.diagnostics['branch_source']} under "
            f"'{best.roots[0].get('name', '')[:70]}', "
            f"period={best.diagnostics['period']}, "
            f"depth={best.diagnostics['period_depth']}, "
            f"coverage={best.diagnostics['branch_coverage']:.1%} "
            f"-> {best.status.name}"
        )
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
    if len(entry_roots) < MIN_LABEL_CHILDREN:
        return None

    ordered = sorted(entry_roots, key=lambda e: e.get("ts", 0))
    candidate = find_pattern(tree, ordered, total_gpu)
    if candidate is None:
        return None
    candidate.method = "generic:sibling_roots"
    candidate.diagnostics["period_label_tier"] = "sibling_roots"
    return candidate
