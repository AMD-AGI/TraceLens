###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Generic iteration-root detection via TraceToTree call-tree traversal."""

from bisect import bisect_right
from collections import Counter
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

# The python_function event category, used when reuniting worker threads and
# when scanning a thread's frames.
PYTHON_TIER = "python_function"

# Graph replay: each engine step replays a captured graph with a single launch.
# That launch survives graph capture (which erases the per-op python/kernel
# signal) and sits at iteration granularity, so it is tried before everything.
GRAPH_LAUNCH_TIER = "graph_launch"
GRAPH_LAUNCH_NAMES = {"hipGraphLaunch", "cudaGraphLaunch"}

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


def _find_repeating_period(
    names: List[str], min_repeats: int = 3
) -> Tuple[Optional[int], Optional[List[str]], Optional[int]]:
    """Best repeating name sequence in ``names`` as ``(period, pattern, start)``."""
    candidates = find_period_candidates(names, min_repeats=min_repeats)
    if not candidates:
        return None, None, None
    best = candidates[0]
    return best.period, list(names[best.start : best.start + best.period]), best.start




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


def _reattach_worker_threads(tree: TraceToTree) -> TraceToTree:
    """Fold roots living on a worker thread under the host frame that, in time,
    encloses them.

    An autograd-engine (or similar dispatch) thread runs work the host thread is
    blocked waiting on, so each of its roots sits entirely inside one host leaf,
    yet the profiler records no parent link across the thread boundary. Rebuilding
    that link reunites e.g. ``backward_step`` with the kernels it triggered and
    lifts the worker roots off the top level, where they otherwise swamp the
    repeating-pattern search.
    """
    pyf: Counter = Counter()
    for e in tree.events_by_uid.values():
        if e.get("cat") == PYTHON_TIER:
            pyf[(e.get("pid"), e.get("tid"))] += 1
    if not pyf:
        return tree
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
        return tree
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
    return tree


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




