###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Test cases covered:
- No events / single event -> no overlaps
- Two non-overlapping events -> each has empty overlapping_uids
- Two overlapping events -> each has the other's UID
- Two touching at boundary -> counted as overlapping (sweep treats start before end at same time)
- Three events with one containing the other two -> all pairs overlap
- overlapping_uids are sets; self not in overlapping_uids; kernel + gpu_memcpy mix
"""

import pytest
from TraceLens.TreePerf import GPUEventAnalyser


def _make_event(uid, ts_us, dur_us, cat="kernel", name=""):
    """Minimal GPU event compatible with get_gpu_event_lists (PyTorch-style)."""
    return {
        "UID": uid,
        "ts": ts_us,
        "dur": dur_us,
        "cat": cat,
        "name": name or "kernel",  # required: event.get("name") used with "in", so must not be None
    }


def _get_overlapping_uids_by_uid(result):
    """Return dict UID -> set of overlapping UIDs from get_gpu_event_lists result."""
    gpu = result[GPUEventAnalyser.all_gpu_key]
    return {e["UID"]: set(e.get("overlapping_uids", set())) for e in gpu}


def test_empty_events():
    """No GPU events -> no crash, empty lists."""
    analyser = GPUEventAnalyser([])
    result = analyser.get_gpu_event_lists()
    assert result[GPUEventAnalyser.all_gpu_key] == []
    # verify_dict_gpu_event_lists raises when all_gpu is empty, so skip here


def test_single_event_has_no_overlaps():
    """Single event -> overlapping_uids is empty (no other events)."""
    e = _make_event(1, 0, 10)
    analyser = GPUEventAnalyser([e])
    result = analyser.get_gpu_event_lists()
    by_uid = _get_overlapping_uids_by_uid(result)
    assert by_uid[1] == set()
    assert result[GPUEventAnalyser.all_gpu_key][0]["t_end"] == 10


def test_two_non_overlapping_events():
    """Two disjoint intervals -> each has empty overlapping_uids."""
    a = _make_event(1, 0, 10)
    b = _make_event(2, 20, 10)
    analyser = GPUEventAnalyser([a, b])
    result = analyser.get_gpu_event_lists()
    by_uid = _get_overlapping_uids_by_uid(result)
    assert by_uid[1] == set()
    assert by_uid[2] == set()


def test_two_overlapping_events():
    """Two overlapping intervals -> each has the other's UID in overlapping_uids."""
    a = _make_event(1, 0, 10)   # [0, 10]
    b = _make_event(2, 5, 10)   # [5, 15]
    analyser = GPUEventAnalyser([a, b])
    result = analyser.get_gpu_event_lists()
    by_uid = _get_overlapping_uids_by_uid(result)
    assert by_uid[1] == {2}
    assert by_uid[2] == {1}


def test_two_touching_at_boundary():
    """Two events touching at t=10 -> treated as overlapping (start processed before end at same time)."""
    a = _make_event(1, 0, 10)   # [0, 10]
    b = _make_event(2, 10, 10)  # [10, 20]
    analyser = GPUEventAnalyser([a, b])
    result = analyser.get_gpu_event_lists()
    by_uid = _get_overlapping_uids_by_uid(result)
    assert by_uid[1] == {2}
    assert by_uid[2] == {1}


def test_three_events_one_contains_others():
    """A [0,100], B [10,20], C [15,25] -> each overlaps the other two."""
    a = _make_event(1, 0, 100)
    b = _make_event(2, 10, 10)
    c = _make_event(3, 15, 10)
    analyser = GPUEventAnalyser([a, b, c])
    result = analyser.get_gpu_event_lists()
    by_uid = _get_overlapping_uids_by_uid(result)
    assert by_uid[1] == {2, 3}
    assert by_uid[2] == {1, 3}
    assert by_uid[3] == {1, 2}


def test_self_not_in_overlapping_uids():
    """overlapping_uids must not contain the event's own UID."""
    a = _make_event(1, 0, 10)
    b = _make_event(2, 5, 10)
    analyser = GPUEventAnalyser([a, b])
    result = analyser.get_gpu_event_lists()
    for event in result[GPUEventAnalyser.all_gpu_key]:
        assert event["UID"] not in event.get("overlapping_uids", set())


def test_overlapping_uids_are_sets():
    """overlapping_uids should be a set (no duplicates, fast membership)."""
    a = _make_event(1, 0, 100)
    b = _make_event(2, 10, 10)
    c = _make_event(3, 15, 10)
    analyser = GPUEventAnalyser([a, b, c])
    result = analyser.get_gpu_event_lists()
    for event in result[GPUEventAnalyser.all_gpu_key]:
        ou = event.get("overlapping_uids")
        assert ou is not None
        assert isinstance(ou, set)


def test_memcpy_and_kernel_events_both_get_overlapping_uids():
    """Mix of cat kernel and gpu_memcpy still get overlapping_uids computed."""
    k = _make_event(1, 0, 10, "kernel")
    m = _make_event(2, 5, 10, "gpu_memcpy")
    analyser = GPUEventAnalyser([k, m])
    result = analyser.get_gpu_event_lists()
    by_uid = _get_overlapping_uids_by_uid(result)
    assert by_uid[1] == {2}
    assert by_uid[2] == {1}
    assert len(result[GPUEventAnalyser.memcpy_key]) == 1
    assert len(result[GPUEventAnalyser.computation_key]) == 1
