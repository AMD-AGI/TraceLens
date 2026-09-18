###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Generic iteration-root detection via TraceToTree call-tree traversal."""

from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from ..util import GPU_KERNEL_CATEGORIES
from .trace_to_tree import TraceToTree

# A period must explain more than half the sequence, matching the original rule.
MIN_PERIOD_COVERAGE = 0.5
# A longer period is only preferred over a shorter one it is a multiple of when
# it explains meaningfully more of the sequence.
DIVISOR_COVERAGE_TOLERANCE = 0.05

# Label sequences shorter than this are utility-function child lists, not loops.
MIN_LABEL_CHILDREN = 4

# The python_function event category, used when reuniting worker threads and
# when scanning a thread's frames.
PYTHON_TIER = "python_function"

# Branch-descent tier: walk down the call tree until a frame's own children form
# a repeating family whose per-iteration windows account for ~all the GPU work.
BRANCH_DESCENT_TIER = "branch_descent"
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
    coverage: float

    @property
    def rank(self) -> tuple:
        """Sort key: explain the most sequence, with the shortest unit.

        Coverage is rounded so float noise cannot reorder near-ties, and period
        breaks ties downward since every multiple explains as much.
        """
        return (-round(self.coverage, 3), self.period)


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
    return sorted(gaps)


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


def _find_repeating_period(
    names: List[str], min_repeats: int = 3
) -> Tuple[Optional[int], Optional[List[str]], Optional[int]]:
    """Best repeating name sequence in ``names`` as ``(period, pattern, start)``.

    Labels are encoded as small ints so the verification loop compares cheaply.
    Candidate periods come from the gaps between one label's recurrences; each is
    verified by its longest period-aligned run and kept only if it repeats enough
    (``min_repeats``) and explains enough of the sequence (``MIN_PERIOD_COVERAGE``).
    The primitive period that explains the most wins, shortest unit breaking ties.
    """
    table: Dict[str, int] = {}
    codes = [table.setdefault(name, len(table)) for name in names]
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
        found.append(PeriodCandidate(period, start, coverage))
    if not found:
        return None, None, None
    best = _drop_multiples(sorted(found, key=lambda c: c.rank))[0]
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
        gpu_uids = root.get("gpu_events", [])
        ancestor: Optional[dict] = host_node
        while ancestor is not None and ancestor.get("non_gpu_path", False):
            if gpu_uids:
                ancestor.setdefault("gpu_events", []).extend(gpu_uids)
            ancestor.pop("non_gpu_path", None)
            ancestor = tree.get_parent_event(ancestor)
        reattached += 1
    return tree


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
        if event.get("cat") in GPU_KERNEL_CATEGORIES:
            total += event.get("dur", 0)
        children = event.get("children")
        if children:
            stack.extend(children)
    return total


def _blocks_by_pattern(
    ordered: Sequence[dict], pattern: Sequence[str], start: int, norm: Dict
) -> List[List[dict]]:
    """Split ``ordered`` into one block per repetition of ``pattern``.

    ``norm`` maps each event's UID to its normalized name, so names are compared
    against ``pattern`` without re-normalizing on every step.

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
            if norm[child["UID"]] == pattern[pos]:
                block.append(child)
                pos += 1
                j += 1
            elif child.get("non_gpu_path", False):
                j += 1  # skip a kernel-less intruder, keep matching this position
            else:
                break  # GPU-bearing deviation: a real break, stop matching
        if pos == period:
            blocks.append(block)
            i = j
        else:
            break  # cannot complete another unit -- past the end of the loop
    return blocks
