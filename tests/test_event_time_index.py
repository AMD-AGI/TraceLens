###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import random

from TraceLens.Trace2Tree.event_time_index import (
    TimeSortedGraphLaunchIndex,
    collect_events_fully_inside_window,
)
from TraceLens.Trace2Tree.trace_capture_merge_experimental import (
    find_graph_roots_under_execution,
)


def _brute_graph_roots(exec_ts, exec_te, graphlaunch_events):
    return [
        e
        for e in graphlaunch_events
        if exec_ts <= e.get("ts", 0) and e.get("ts", 0) + e.get("dur", 0) <= exec_te
    ]


def test_graph_launch_index_matches_bruteforce_random():
    rng = random.Random(0)
    for _ in range(200):
        n = rng.randint(0, 80)
        events = []
        for i in range(n):
            ts = rng.uniform(0, 1e6)
            dur = rng.uniform(0, 500)
            events.append(
                {
                    "ts": ts,
                    "dur": dur,
                    "name": (
                        "hipGraphLaunch" if rng.random() > 0.1 else "cudaGraphLaunch"
                    ),
                    "_i": i,
                }
            )
        idx = TimeSortedGraphLaunchIndex(events)
        for _w in range(40):
            lo = rng.uniform(0, 1e6)
            hi = lo + rng.uniform(0, 2000)
            exec_root = {"ts": lo, "dur": hi - lo}
            got = find_graph_roots_under_execution(exec_root, idx)
            brute = _brute_graph_roots(lo, hi, events)
            assert sorted(e["_i"] for e in got) == sorted(e["_i"] for e in brute), (
                lo,
                hi,
                len(got),
                len(brute),
            )


def test_collect_events_fully_inside_window_sorted():
    ev_a = {"ts": 10.0, "dur": 5.0, "k": "a"}
    ev_b = {"ts": 12.0, "dur": 1.0, "k": "b"}
    ev_c = {"ts": 20.0, "dur": 2.0, "k": "c"}
    starts = [10.0, 12.0, 20.0]
    ends = [15.0, 13.0, 22.0]
    events = [ev_a, ev_b, ev_c]
    win = collect_events_fully_inside_window(starts, ends, events, 11.0, 16.0)
    assert [e["k"] for e in win] == ["b"]
    win2 = collect_events_fully_inside_window(starts, ends, events, 10.0, 22.0)
    assert set(e["k"] for e in win2) == {"a", "b", "c"}
