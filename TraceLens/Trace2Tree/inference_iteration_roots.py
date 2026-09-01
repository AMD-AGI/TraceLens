###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Generic iteration-root detection via TraceToTree call-tree traversal."""

from bisect import bisect_right
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

# Graph replay: each engine step replays a captured graph with a single launch.
# That launch survives graph capture (which erases the per-op python/kernel
# signal) and sits at iteration granularity, so it is tried before everything.
GRAPH_LAUNCH_TIER = "graph_launch"
GRAPH_LAUNCH_NAMES = {"hipGraphLaunch", "cudaGraphLaunch"}

PERIOD_EXACT = "exact"
PERIOD_INTEGER_RATIO = "integer_ratio"
PERIOD_CONFLICT = "conflict"

# Branch-descent tier: walk down the call tree until a frame's own children form
# a repeating family whose per-iteration windows account for ~all the GPU work.
BRANCH_DESCENT_TIER = "branch_descent"
GPU_KERNEL_CATS = ("kernel", "gpu_memcpy", "gpu_memset")
# The per-iteration windows must explain at least this share of GPU time; below
# it the repeating family is a sub-loop, not the iteration boundary.
BRANCH_COVERAGE_GATE = 0.95
# Bound the descent so a pathological tree cannot walk forever / explode a level.
BRANCH_MAX_NODES = 200000


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
    (Graph-replay traces are handled earlier, in
    :func:`find_iteration_roots_generic`, straight off the flat event list -- one
    graph launch per replay is the per-iteration marker there.)
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

    # --- Periodicity ACROSS the sibling roots themselves: when each iteration is
    # its own top-level frame, the repeating unit is the sequence of roots, not a
    # pattern inside any one of them.
    if len(roots) >= MIN_LABEL_CHILDREN:
        ordered = sorted(roots, key=lambda e: e.get("ts", 0))
        period, _, start = _find_repeating_period([e.get("name", "") for e in ordered])
        if period is not None:
            blocks = (len(ordered) - start) // period
            sibling_roots = []
            for index in range(blocks):
                block = ordered[start + index * period : start + (index + 1) * period]
                first, last = block[0], block[-1]
                event = dict(first)
                event["dur"] = (last["ts"] + last.get("dur", 0)) - first["ts"]
                sibling_roots.append(event)
            if sibling_roots:
                print(
                    f"Generic: {len(sibling_roots)} sibling-root iterations "
                    f"(period={period})."
                )
                if diagnostics is not None:
                    diagnostics.update(
                        {"period_label_tier": "sibling_roots", "period": period}
                    )
                return sibling_roots

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


def _entry_roots(tree: TraceToTree) -> List[dict]:
    """Parentless per-thread entry nodes, deduped, honoring current parents."""
    seen: set = set()
    roots: List[dict] = []
    for uid in tree.cpu_root_nodes:
        event = tree.get_UID2event(uid)
        while True:
            parent = tree.get_parent_event(event)
            if parent is None:
                break
            event = parent
        if event["UID"] not in seen:
            seen.add(event["UID"])
            roots.append(event)
    return roots


def _reattach_worker_threads(tree: TraceToTree) -> int:
    """Fold roots living on a worker thread under the host frame that, in time,
    encloses them.

    An autograd-engine (or similar dispatch) thread runs work the host thread is
    blocked waiting on, so each of its roots sits entirely inside one host leaf,
    yet the profiler records no parent link across the thread boundary. Rebuilding
    that link reunites e.g. ``backward_step`` with the kernels it triggered and
    lifts the worker roots off the top level, where they otherwise swamp the
    repeating-pattern search. Returns the number of roots reattached.
    """
    pyf: Counter = Counter()
    for e in tree.events_by_uid.values():
        if e.get("cat") == PYTHON_TIER:
            pyf[(e.get("pid"), e.get("tid"))] += 1
    if not pyf:
        return 0
    host = pyf.most_common(1)[0][0]

    host_nodes = sorted(
        (
            e
            for e in tree.events_by_uid.values()
            if (e.get("pid"), e.get("tid")) == host
            and e.get("ts") is not None
            and e.get("t_end") is not None
        ),
        key=lambda e: e["ts"],
    )
    if not host_nodes:
        return 0
    starts = [e["ts"] for e in host_nodes]

    def deepest_container(lo: float, hi: float) -> Optional[dict]:
        # Host frames nest, so among those starting at/before ``lo`` the latest
        # one still ending at/after ``hi`` is the innermost enclosing frame.
        i = bisect_right(starts, lo) - 1
        while i >= 0:
            node = host_nodes[i]
            if node["t_end"] >= hi:
                return node
            i -= 1
        return None

    reattached = 0
    for root in _entry_roots(tree):
        if (root.get("pid"), root.get("tid")) == host:
            continue
        lo, hi = root.get("ts"), root.get("t_end")
        if lo is None or hi is None:
            continue
        host_node = deepest_container(lo, hi)
        if host_node is None:
            continue
        root["parent"] = host_node["UID"]
        host_node.setdefault("children", []).append(root["UID"])
        # Flag the new ancestry GPU-bearing so the descent will follow it: the
        # reattached subtree carries kernels the host frames previously lacked.
        ancestor: Optional[dict] = host_node
        while ancestor is not None and not ancestor.get("_kernel_bearing"):
            ancestor["_kernel_bearing"] = True
            ancestor = tree.get_parent_event(ancestor)
        reattached += 1
    return reattached


def _gpu_bearing(event: dict) -> bool:
    """Whether ``event`` has any GPU work under it (native or reattached)."""
    return bool(event.get("gpu_events") or event.get("_kernel_bearing"))


def _descendant_gpu_time(tree: TraceToTree, nodes: Sequence[dict]) -> float:
    """Total GPU time under ``nodes`` in the tree, each kernel counted once."""
    seen: set = set()
    total = 0.0
    stack = [n["UID"] for n in nodes]
    while stack:
        uid = stack.pop()
        if uid in seen:
            continue
        seen.add(uid)
        event = tree.get_UID2event(uid)
        if event.get("cat") in GPU_KERNEL_CATS:
            total += event.get("dur", 0)
        children = event.get("children")
        if children:
            stack.extend(children)
    return total


def _blocks_by_pattern(
    ordered: Sequence[dict], pattern: Sequence[str], start: int
) -> List[List[dict]]:
    """Split ``ordered`` into one block per repetition of ``pattern``.

    A fixed stride of ``len(pattern)`` smears an iteration across two blocks the
    moment a stray frame slips between two repetitions -- a context-manager, a
    timer -- because every later block is then shifted by one. Matching the
    pattern element by element and *skipping intruders that carry no kernels*
    keeps the stride aligned: the skipped frame is bookkeeping, not work, so no
    kernel is lost. Matching stops at the first kernel-bearing deviation, so the
    post-loop teardown does not become a phantom iteration.
    """
    period = len(pattern)
    if period == 0:
        return []
    blocks: List[List[dict]] = []
    i = start
    n = len(ordered)
    while i < n:
        block: List[dict] = []
        pos = 0
        j = i
        while pos < period and j < n:
            child = ordered[j]
            if child.get("name", "") == pattern[pos]:
                block.append(child)
                pos += 1
                j += 1
            elif not _gpu_bearing(child):
                j += 1  # skip a kernel-less intruder, keep matching this position
            else:
                break  # kernel-bearing deviation: a real break, stop matching
        if pos == period:
            blocks.append(block)
            i = j
        else:
            break  # cannot complete another unit -- past the end of the loop
    return blocks


def _detect_by_branch_descent(
    tree: TraceToTree, diagnostics: Optional[dict] = None
) -> Optional[List[dict]]:
    """Descend the call tree until a frame's own children repeat and cover the GPU.

    Category-agnostic: single-child scaffolding frames are traversed for free (a
    node with fewer than a loop's worth of children is skipped), and each node is
    judged on *its own* children rather than a pool merged across sibling branches
    -- pooling mixes the training loop with logging/optimizer branches and hides
    the period. The first frame whose per-iteration windows explain at least
    :data:`BRANCH_COVERAGE_GATE` of GPU time is the iteration boundary; grading on
    coverage keeps the walk from stopping on a deep sub-loop (a grad-norm poll,
    say) that repeats but does almost no work.
    """
    total_gpu = sum(
        e.get("dur", 0)
        for e in tree.events_by_uid.values()
        if e.get("cat") in GPU_KERNEL_CATS
    )
    if not total_gpu:
        return None

    queue = deque((r, 0) for r in _entry_roots(tree))
    visited = 0
    while queue:
        node, depth = queue.popleft()
        visited += 1
        if visited > BRANCH_MAX_NODES:
            break
        children = tree.get_children_events(node)
        if len(children) >= MIN_LABEL_CHILDREN:
            ordered = sorted(children, key=lambda e: e.get("ts", 0))
            period, pattern, start = _find_repeating_period(
                [e.get("name", "") for e in ordered]
            )
            if period is not None:
                unit_blocks = _blocks_by_pattern(ordered, pattern, start)
                if len(unit_blocks) >= MIN_LABEL_CHILDREN:
                    iteration_roots: List[dict] = []
                    blocked: List[dict] = []
                    for block in unit_blocks:
                        first, last = block[0], block[-1]
                        # One window per period, named for the loop frame that
                        # owns it (``node``) rather than the arbitrary first child
                        # in the block, which is just whatever call happens to
                        # start the iteration. Time span stays the block's own.
                        event = dict(first)
                        event["name"] = node.get("name", event.get("name", ""))
                        event["dur"] = (
                            last["ts"] + last.get("dur", 0)
                        ) - first["ts"]
                        iteration_roots.append(event)
                        blocked.extend(block)
                    cov = _descendant_gpu_time(tree, blocked) / total_gpu
                    if cov >= BRANCH_COVERAGE_GATE:
                        print(
                            f"Generic: branch-descent found {len(unit_blocks)} "
                            f"iterations under '{node.get('name')}' "
                            f"(period={period}, coverage={cov:.3f}, depth={depth})."
                        )
                        if diagnostics is not None:
                            diagnostics.update(
                                {
                                    "period_label_tier": BRANCH_DESCENT_TIER,
                                    "period": period,
                                    "period_depth": depth,
                                    "branch_coverage": round(cov, 4),
                                }
                            )
                        return iteration_roots
        for child in children:
            if _gpu_bearing(child):
                queue.append((child, depth + 1))
    return None


def find_iteration_roots_generic(
    events: List[dict], diagnostics: Optional[dict] = None
) -> Optional[List[dict]]:
    """Fallback: detect iteration roots from a repeating child pattern.

    Works for any workload (diffusion, training, etc.) where the iteration loop
    body is a repeating sequence of calls under a common parent.
    """
    # --- Graph launches first, straight off the flat event list (no tree needed
    # and thread-independent). One launch per graph replay = one iteration, and
    # this signal survives graph capture, which erases the per-op python/kernel
    # periodicity the tree traversal relies on.
    launches = sorted(
        (
            e
            for e in events
            if e.get("name") in GRAPH_LAUNCH_NAMES and e.get("ts") is not None
        ),
        key=lambda e: e["ts"],
    )
    if len(launches) >= MIN_LABEL_CHILDREN:
        iteration_roots = [dict(e) for e in launches]
        print(f"Generic: {len(iteration_roots)} graph-launch iteration roots.")
        if diagnostics is not None:
            diagnostics.update({"period_label_tier": GRAPH_LAUNCH_TIER, "period": 1})
        return iteration_roots

    try:
        tree = TraceToTree(events, prune_nongpu_paths=False)
        tree.build_tree(add_python_func=True)
    except Exception as e:
        print(f"Generic fallback: TraceToTree build failed ({e}), skipping.")
        return None

    # Reunite dispatch threads (e.g. the autograd engine) with the host frames
    # that spawned them, so a training loop's backward work is reachable from the
    # main thread rather than flooding the top level as thousands of stray roots.
    _reattach_worker_threads(tree)

    # Preferred path: walk down to the frame whose children repeat and cover the
    # GPU. It reads the semantic loop body (forward_step/backward_step, denoise
    # step, ...) directly and is graded by coverage, so it does not lock onto a
    # deep sub-loop the way the tier ladder below can.
    branch_roots = _detect_by_branch_descent(tree, diagnostics)
    if branch_roots is not None:
        return branch_roots

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
