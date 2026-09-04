###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Generic iteration-root detection via TraceToTree call-tree traversal."""

from collections import Counter, deque
from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Dict, List, Optional, Sequence, Tuple

from .trace_to_tree import TraceToTree

# A period must explain more than half the sequence, matching the original rule.
MIN_PERIOD_COVERAGE = 0.5
# Candidate periods come from gaps between recurrences of one label; cap the
# number verified so a pathological sequence cannot blow up the search.
MAX_PERIOD_CANDIDATES = 64
# A longer period is only preferred over a shorter one it is a multiple of when
# it explains meaningfully more of the sequence.
DIVISOR_COVERAGE_TOLERANCE = 0.05

# Label sequences shorter than this are utility-function child lists, not loops.
MIN_LABEL_CHILDREN = 6

# Preferred sources of labels for period detection, best first. Python frames
# correspond to semantic loop bodies; the rest are fallbacks for traces captured
# without stack recording.
PYTHON_TIER = "python_function"
CPU_OP_TIER = "cpu_op"
ALL_CHILDREN_TIER = "all_children"

PERIOD_EXACT = "exact"
PERIOD_INTEGER_RATIO = "integer_ratio"
PERIOD_CONFLICT = "conflict"


@dataclass
class PeriodCandidate:
    """A verified repeating period, with the evidence for it."""

    period: int
    start: int
    repeats: int
    coverage: float
    duration_cv: float

    @property
    def rank(self) -> tuple:
        """Sort key: explain the most sequence, most evenly, with the shortest unit.

        Coverage is rounded so float noise cannot outrank a steadier candidate,
        and period breaks ties downward since every multiple explains as much.
        """
        return (-round(self.coverage, 3), round(self.duration_cv, 3), self.period)


def _candidate_periods(codes: Sequence[int], min_repeats: int) -> List[int]:
    """Plausible periods, taken from the gaps between one label's recurrences.

    If a sequence has period ``p`` every label recurs every ``p`` positions, so
    any single label's gaps contain ``p``. Anchoring on the rarest eligible
    label keeps the list short and makes a non-repeating sequence cost nothing.
    """
    counts = Counter(codes)
    eligible = [(n, code) for code, n in counts.items() if n >= min_repeats]
    if not eligible:
        return []
    _, anchor = min(eligible)
    positions = [i for i, code in enumerate(codes) if code == anchor]
    gaps = {b - a for a, b in zip(positions, positions[1:]) if b > a}
    return sorted(gaps)[:MAX_PERIOD_CANDIDATES]


def _longest_periodic_run(codes: Sequence[int], period: int) -> Tuple[int, int]:
    """Start index and block count of the longest ``period``-aligned run.

    Scanning for the longest run finds the loop wherever it sits, so a warmup
    prefix is skipped without retrying the search from every offset.
    """
    limit = len(codes) - period
    best_start = best_len = 0
    i = 0
    while i < limit:
        if codes[i] != codes[i + period]:
            i += 1
            continue
        j = i
        while j < limit and codes[j] == codes[j + period]:
            j += 1
        if j - i > best_len:
            best_start, best_len = i, j - i
        i = j + 1
    return best_start, (best_len + period) // period


def _duration_cv(
    durations: Optional[Sequence[float]], start: int, period: int, repeats: int
) -> float:
    """Coefficient of variation of per-occurrence duration.

    A real iteration takes about the same time every time, which separates a
    genuine loop from a coincidental label match.
    """
    if not durations:
        return 0.0
    blocks = [
        sum(durations[start + i * period : start + (i + 1) * period])
        for i in range(repeats)
    ]
    blocks = [b for b in blocks if b > 0]
    if len(blocks) < 2:
        return 0.0
    average = mean(blocks)
    return pstdev(blocks) / average if average else 0.0


def _drop_multiples(candidates: List[PeriodCandidate]) -> List[PeriodCandidate]:
    """Keep primitive periods; any multiple of one is valid and explains no more."""
    kept: List[PeriodCandidate] = []
    for cand in candidates:
        if any(
            cand.period % k.period == 0
            and cand.coverage <= k.coverage + DIVISOR_COVERAGE_TOLERANCE
            for k in kept
        ):
            continue
        kept.append(cand)
    return kept


def find_period_candidates(
    labels: Sequence[str],
    durations: Optional[Sequence[float]] = None,
    min_repeats: int = 3,
) -> List[PeriodCandidate]:
    """Every qualifying repeating period in ``labels``, best first.

    Scored candidates rather than one answer let callers cross-check against an
    independent detection instead of trusting a single verdict.
    """
    # Labels as small ints, so comparisons are cheap in the verification loop.
    table: Dict[str, int] = {}
    codes = [table.setdefault(label, len(table)) for label in labels]
    total = len(codes)
    found: List[PeriodCandidate] = []
    for period in _candidate_periods(codes, min_repeats):
        if period * min_repeats > total:
            continue
        start, repeats = _longest_periodic_run(codes, period)
        if repeats < min_repeats:
            continue
        coverage = repeats * period / total
        if coverage <= MIN_PERIOD_COVERAGE:
            continue
        found.append(
            PeriodCandidate(
                period,
                start,
                repeats,
                coverage,
                _duration_cv(durations, start, period, repeats),
            )
        )
    return _drop_multiples(sorted(found, key=lambda c: c.rank))


def compare_periods(a: Optional[int], b: Optional[int]) -> Tuple[str, Optional[int]]:
    """Whether two independently detected periods agree.

    Differing by an exact integer factor means the same loop at different
    granularities -- a confirmation, not a conflict.
    """
    if not a or not b:
        return PERIOD_CONFLICT, None
    low, high = min(a, b), max(a, b)
    if low == high:
        return PERIOD_EXACT, 1
    if high % low == 0:
        return PERIOD_INTEGER_RATIO, high // low
    return PERIOD_CONFLICT, None


def _find_repeating_period(
    names: List[str], min_repeats: int = 3
) -> Tuple[Optional[int], Optional[List[str]], Optional[int]]:
    """Best repeating name sequence in ``names`` as ``(period, pattern, start)``."""
    candidates = find_period_candidates(names, min_repeats=min_repeats)
    if not candidates:
        return None, None, None
    best = candidates[0]
    return best.period, list(names[best.start : best.start + best.period]), best.start


def _nearest_descendants(tree: TraceToTree, node: dict, cat: str) -> List[dict]:
    """Nearest descendants of ``node`` in category ``cat``, in time order.

    Descent stops at each match, so the result is one abstraction layer. Direct
    children are not enough: a python frame's children are often ATen ops whose
    own children are the next python frames, so filtering them returns nothing.
    """
    found: List[dict] = []
    queue = deque(tree.get_children_events(node))
    while queue:
        child = queue.popleft()
        if child.get("cat") == cat:
            found.append(child)
        else:
            queue.extend(tree.get_children_events(child))
    found.sort(key=lambda e: e.get("ts", 0))
    return found


def _label_events(tree: TraceToTree, node: dict) -> Tuple[List[dict], str]:
    """Events under ``node`` to run period detection over, and which tier they are.

    Launches recur many times per iteration and swamp the iteration-level
    signal, so python frames -- the semantic loop bodies -- are preferred. The
    ladder exists because a capture without stack recording has none at all.
    """
    for cat in (PYTHON_TIER, CPU_OP_TIER):
        found = _nearest_descendants(tree, node, cat)
        if len(found) >= MIN_LABEL_CHILDREN:
            return found, cat
    return tree.get_children_events(node), ALL_CHILDREN_TIER


def _detect_iteration_roots_from_tree(
    tree: TraceToTree, roots, diagnostics: Optional[dict] = None
) -> Optional[List[dict]]:
    """BFS down from ``roots`` for a repeating block, returned as synthetic roots.

    Each returned event spans one block, from the first child's start to the last
    child's end, so CPU-only leading work stays inside the iteration.
    """
    if isinstance(roots, dict):
        roots = [roots]

    queue = deque((node, 0) for node in roots)
    while queue:
        current, depth = queue.popleft()
        labelled, tier = _label_events(tree, current)
        if not labelled:
            continue

        # Only recurse into GPU-bearing subtrees, tested on the events actually
        # being used as labels rather than on the raw child list.
        if not any(e.get("gpu_events") for e in labelled):
            continue

        period, _, start = _find_repeating_period([e.get("name", "") for e in labelled])
        if period is None:
            for event in labelled:
                if event.get("gpu_events"):
                    queue.append((event, depth + 1))
            continue

        blocks = (len(labelled) - start) // period
        iteration_roots = []
        for index in range(blocks):
            block = labelled[start + index * period : start + (index + 1) * period]
            first, last = block[0], block[-1]
            root_event = dict(first)
            root_event["dur"] = (last["ts"] + last.get("dur", 0)) - first["ts"]
            iteration_roots.append(root_event)

        print(
            f"Generic fallback: repeating pattern found under "
            f"'{current.get('name')}' at depth {depth} (tier={tier}, period={period})"
        )
        print(f"Generic fallback: identified {len(iteration_roots)} iterations.")
        if diagnostics is not None:
            diagnostics.update(
                {
                    "period_label_tier": tier,
                    "period": period,
                    "period_depth": depth,
                }
            )
        return iteration_roots or None

    return None


def find_iteration_roots_generic(
    events: List[dict], diagnostics: Optional[dict] = None
) -> Optional[List[dict]]:
    """Fallback: detect iteration roots from a repeating child pattern.

    Works for any workload (diffusion, training, etc.) where the iteration loop
    body is a repeating sequence of calls under a common parent.
    """
    try:
        tree = TraceToTree(events, prune_nongpu_paths=False)
        tree.build_tree(add_python_func=True)
    except Exception as e:
        print(f"Generic fallback: TraceToTree build failed ({e}), skipping.")
        return None

    # Walk every cpu_root_node upward to a parentless node -- those are the true
    # per-thread entry points.
    seen_roots: set = set()
    trace_roots = []
    for uid in tree.cpu_root_nodes:
        event = tree.get_UID2event(uid)
        while True:
            parent = tree.get_parent_event(event)
            if parent is None:
                break
            event = parent
        if id(event) not in seen_roots:
            seen_roots.add(id(event))
            trace_roots.append(event)

    if not trace_roots:
        print("Generic fallback: no root nodes found.")
        return None

    roots = _detect_iteration_roots_from_tree(tree, trace_roots, diagnostics)
    if roots is None:
        print("Generic fallback: no repeating child pattern found.")
    return roots
