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
- Same cpu_op: overlapping kernels under the same cpu_op are NOT marked overlapping
- Different cpu_ops: overlapping kernels under different cpu_ops ARE marked overlapping
- Mixed: some kernels share a cpu_op, others don't
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
        "name": name
        or "kernel",  # required: event.get("name") used with "in", so must not be None
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
    a = _make_event(1, 0, 10)  # [0, 10]
    b = _make_event(2, 5, 10)  # [5, 15]
    analyser = GPUEventAnalyser([a, b])
    result = analyser.get_gpu_event_lists()
    by_uid = _get_overlapping_uids_by_uid(result)
    assert by_uid[1] == {2}
    assert by_uid[2] == {1}


def test_two_touching_at_boundary():
    """Two events touching at t=10 -> treated as overlapping (start processed before end at same time)."""
    a = _make_event(1, 0, 10)  # [0, 10]
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


# ── cpu_op-aware overlap tests ──────────────────────────────────────────


def _make_tree(cpu_ops, kernels):
    """Build a minimal event list with cpu_op -> runtime -> kernel parent chains.

    cpu_ops: list of dicts with at least {"UID", "cat": "cpu_op"}
    kernels: list of (kernel_event, runtime_uid, cpu_op_uid) tuples
    Returns the full event list suitable for GPUEventAnalyser.
    """
    events = list(cpu_ops)
    for kernel, runtime_uid, cpu_op_uid in kernels:
        runtime = {
            "UID": runtime_uid,
            "cat": "cuda_runtime",
            "name": "cudaLaunchKernel",
            "parent": cpu_op_uid,
        }
        kernel["parent"] = runtime_uid
        events.append(runtime)
        events.append(kernel)
    return events


def test_same_cpu_op_overlap_not_marked():
    """Two overlapping kernels under the SAME cpu_op should NOT be marked overlapping."""
    cpu = {"UID": 100, "cat": "cpu_op", "name": "aten::mm"}
    k1 = _make_event(1, 0, 10)
    k2 = _make_event(2, 5, 10)
    events = _make_tree([cpu], [(k1, 201, 100), (k2, 202, 100)])
    analyser = GPUEventAnalyser(events)
    result = analyser.get_gpu_event_lists()
    by_uid = _get_overlapping_uids_by_uid(result)
    assert by_uid[1] == set()
    assert by_uid[2] == set()


def test_different_cpu_ops_overlap_is_marked():
    """Overlapping kernels from DIFFERENT cpu_ops should be marked overlapping."""
    cpu_a = {"UID": 100, "cat": "cpu_op", "name": "aten::mm"}
    cpu_b = {"UID": 101, "cat": "cpu_op", "name": "aten::add"}
    k1 = _make_event(1, 0, 10)
    k2 = _make_event(2, 5, 10)
    events = _make_tree([cpu_a, cpu_b], [(k1, 201, 100), (k2, 202, 101)])
    analyser = GPUEventAnalyser(events)
    result = analyser.get_gpu_event_lists()
    by_uid = _get_overlapping_uids_by_uid(result)
    assert by_uid[1] == {2}
    assert by_uid[2] == {1}


def test_mixed_cpu_ops_partial_overlap():
    """Three kernels: k1,k2 share a cpu_op, k3 is under a different one.
    k1 and k2 overlap but share a cpu_op -> not marked.
    k1 and k3 overlap and differ in cpu_op -> marked.
    k2 and k3 don't overlap -> not marked."""
    cpu_a = {"UID": 100, "cat": "cpu_op", "name": "aten::mm"}
    cpu_b = {"UID": 101, "cat": "cpu_op", "name": "aten::add"}
    k1 = _make_event(1, 0, 20)  # [0, 20]
    k2 = _make_event(2, 5, 10)  # [5, 15]  - same cpu_op as k1
    k3 = _make_event(3, 10, 5)  # [10, 15] - different cpu_op, overlaps k1 only
    events = _make_tree(
        [cpu_a, cpu_b],
        [(k1, 201, 100), (k2, 202, 100), (k3, 203, 101)],
    )
    analyser = GPUEventAnalyser(events)
    result = analyser.get_gpu_event_lists()
    by_uid = _get_overlapping_uids_by_uid(result)
    assert by_uid[1] == {3}
    assert by_uid[2] == {3}
    assert by_uid[3] == {1, 2}


def test_no_parent_info_falls_back_to_overlap():
    """Kernels without parent info should still be marked overlapping (conservative)."""
    k1 = _make_event(1, 0, 10)
    k2 = _make_event(2, 5, 10)
    analyser = GPUEventAnalyser([k1, k2])
    result = analyser.get_gpu_event_lists()
    by_uid = _get_overlapping_uids_by_uid(result)
    assert by_uid[1] == {2}
    assert by_uid[2] == {1}
