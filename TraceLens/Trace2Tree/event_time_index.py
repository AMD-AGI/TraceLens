###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Time-sorted indexes for trace events (merge / capture helpers).

Avoids repeated O(N * num_windows) scans over large ``graph_tree.events`` lists
when answering ``[ts, te]`` containment queries.
"""

from __future__ import annotations

import bisect
from typing import Dict, List, Sequence


def collect_events_fully_inside_window(
    starts: Sequence[float],
    ends: Sequence[float],
    events: Sequence[Dict],
    win_lo: float,
    win_hi: float,
) -> List[Dict]:
    """Return event dicts whose intervals ``[start, end]`` lie fully inside ``[win_lo, win_hi]``.

    *starts*, *ends*, and *events* must have equal length; *starts* must be
    non-decreasing (sorted by interval start).
    """
    if len(starts) != len(ends) or len(starts) != len(events):
        raise ValueError("starts, ends, and events must have the same length")
    i = bisect.bisect_left(starts, win_lo)
    out: List[Dict] = []
    n = len(starts)
    while i < n:
        s = starts[i]
        e = ends[i]
        if s > win_hi:
            break
        if win_lo <= s and e <= win_hi:
            out.append(events[i])
        i += 1
    return out


class TimeSortedGraphLaunchIndex:
    """Graph-launch events sorted by start time for fast window queries."""

    __slots__ = ("_starts", "_ends", "_events")

    def __init__(self, graphlaunch_events: List[Dict]):
        triples: List[tuple[float, float, Dict]] = []
        for ev in graphlaunch_events:
            s = float(ev.get("ts", 0) or 0)
            e = s + float(ev.get("dur", 0) or 0)
            triples.append((s, e, ev))
        triples.sort(key=lambda t: t[0])
        self._starts = [t[0] for t in triples]
        self._ends = [t[1] for t in triples]
        self._events = [t[2] for t in triples]

    def roots_in_window(self, exec_ts: float, exec_te: float) -> List[Dict]:
        """Match ``find_graph_roots_under_execution`` semantics (fully inside window)."""
        found = collect_events_fully_inside_window(
            self._starts, self._ends, self._events, exec_ts, exec_te
        )
        found.sort(key=lambda x: x.get("ts", 0))
        return found


def build_sorted_time_arrays(
    events: Sequence[Dict],
) -> tuple[List[float], List[float], List[Dict]]:
    """Sort *events* by ``ts`` and return parallel ``starts``, ``ends``, sorted event refs."""
    sorted_refs = sorted(events, key=lambda ev: float(ev.get("ts", 0) or 0))
    starts = [float(ev.get("ts", 0) or 0) for ev in sorted_refs]
    ends = [s + float(ev.get("dur", 0) or 0) for s, ev in zip(starts, sorted_refs)]
    return starts, ends, sorted_refs
