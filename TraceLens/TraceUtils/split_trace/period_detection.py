###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Repeating-period detection over a label sequence.

A domain-agnostic algorithm: given an ordered list of names, find the best
repeating unit (``_find_repeating_period``) and, given that unit, carve the
events into one block per repetition (``_blocks_by_pattern``). No knowledge of
traces or the call tree -- callers hand in names and events and get periods back.
"""

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

# A period must explain more than half the sequence, matching the original rule.
MIN_PERIOD_COVERAGE = 0.5
# A longer period is only preferred over a shorter one it is a multiple of when
# it explains meaningfully more of the sequence.
DIVISOR_COVERAGE_TOLERANCE = 0.05


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
