###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Test cases covered:

overlapping_uids computation (GPUEventAnalyser):
- No events / single event -> no overlaps
- Two non-overlapping events -> each has empty overlapping_uids
- Two overlapping events -> each has the other's UID
- Two touching at boundary -> NOT overlapping (end processed before start at same time)
- Three events with one containing the other two -> all pairs overlap
- overlapping_uids are ordered sets; self not in overlapping_uids; kernel + gpu_memcpy mix
- Same cpu_op: overlapping kernels under the same cpu_op are NOT marked overlapping
- Different cpu_ops: overlapping kernels under different cpu_ops ARE marked overlapping
- Mixed: some kernels share a cpu_op, others don't
- Same stream: overlapping kernels on the same stream are NOT marked overlapping
- Different streams: overlapping kernels on different streams ARE marked overlapping
- Missing stream info: conservatively falls back to marking as overlapping
- Same stream + different cpu_ops: stream check takes priority, NOT marked overlapping
- Sub-microsecond overlap (<1µs) filtered as timestamp noise (Bug 1 regression)
- Exactly 1µs overlap meets threshold and IS marked (strict < 1.0 check)
- Above-threshold overlap (>1µs) correctly marked

overlap_pct computation (via kernel launchers and DataFrames):
- Partial overlap between kernels from different cpu_ops
- Full containment (one kernel entirely within another)
- No overlap -> no overlapping_kernels_details entries
- Exact same interval -> overlap_pct = 1.0
- Boundary touching -> not overlapping
- Multiple overlapping kernels with different overlap fractions
- Multi-kernel cpu_op with partial / spanning overlap (merged runtime denominator)
- overlap_pct appears as a top-level DataFrame column, rounded to hundredths
- overlap_pct survives aggregation in unique-args kl_overlap
"""

import os
import pytest
import pandas as pd
from copy import deepcopy
from ordered_set import OrderedSet

from TraceLens.TreePerf import GPUEventAnalyser
from TraceLens.Trace2Tree.trace_to_tree import TraceToTree
from TraceLens.TreePerf.tree_perf import TreePerfAnalyzer


def _make_event(uid, ts_us, dur_us, cat="kernel", name="", args=None):
    """Minimal GPU event compatible with get_gpu_event_lists (PyTorch-style)."""
    event = {
        "UID": uid,
        "ts": ts_us,
        "dur": dur_us,
        "cat": cat,
        "name": name
        or "kernel",  # required: event.get("name") used with "in", so must not be None
    }
    if args is not None:
        event["args"] = args
    return event


def _get_overlapping_uids_by_uid(result):
    """Return dict UID -> ordered set of overlapping UIDs from get_gpu_event_lists result."""
    gpu = result[GPUEventAnalyser.all_gpu_key]
    return {e["UID"]: OrderedSet(e.get("overlapping_uids", OrderedSet())) for e in gpu}


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
    """Two events touching at t=10 -> NOT overlapping (end processed before start at same time)."""
    a = _make_event(1, 0, 10)  # [0, 10]
    b = _make_event(2, 10, 10)  # [10, 20]
    analyser = GPUEventAnalyser([a, b])
    result = analyser.get_gpu_event_lists()
    by_uid = _get_overlapping_uids_by_uid(result)
    assert by_uid[1] == set()
    assert by_uid[2] == set()


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
        assert event["UID"] not in event.get("overlapping_uids", OrderedSet())


def test_overlapping_uids_are_ordered_sets():
    """overlapping_uids should be an OrderedSet (no duplicates, fast membership)."""
    a = _make_event(1, 0, 100)
    b = _make_event(2, 10, 10)
    c = _make_event(3, 15, 10)
    analyser = GPUEventAnalyser([a, b, c])
    result = analyser.get_gpu_event_lists()
    for event in result[GPUEventAnalyser.all_gpu_key]:
        ou = event.get("overlapping_uids")
        assert ou is not None
        assert isinstance(ou, OrderedSet)


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


def test_same_stream_overlap_not_marked():
    """Two temporally overlapping kernels on the SAME stream should NOT be
    marked overlapping — the GPU serialises work on a single stream."""
    k1 = _make_event(1, 0, 10, args={"stream": 7})
    k2 = _make_event(2, 5, 10, args={"stream": 7})
    analyser = GPUEventAnalyser([k1, k2])
    result = analyser.get_gpu_event_lists()
    by_uid = _get_overlapping_uids_by_uid(result)
    assert by_uid[1] == set(), "same-stream kernels must not be marked overlapping"
    assert by_uid[2] == set(), "same-stream kernels must not be marked overlapping"


def test_different_stream_overlap_is_marked():
    """Two temporally overlapping kernels on DIFFERENT streams should be
    marked overlapping."""
    k1 = _make_event(1, 0, 10, args={"stream": 7})
    k2 = _make_event(2, 5, 10, args={"stream": 8})
    analyser = GPUEventAnalyser([k1, k2])
    result = analyser.get_gpu_event_lists()
    by_uid = _get_overlapping_uids_by_uid(result)
    assert by_uid[1] == {2}, "different-stream kernels should be marked overlapping"
    assert by_uid[2] == {1}, "different-stream kernels should be marked overlapping"


def test_missing_stream_info_falls_back_to_overlap():
    """When stream info is absent from one or both kernels, conservatively
    assume they could be on different streams and mark as overlapping."""
    k1 = _make_event(1, 0, 10, args={"stream": 7})
    k2 = _make_event(2, 5, 10)  # no args / no stream
    analyser = GPUEventAnalyser([k1, k2])
    result = analyser.get_gpu_event_lists()
    by_uid = _get_overlapping_uids_by_uid(result)
    assert by_uid[1] == {2}, "missing stream should be treated conservatively"
    assert by_uid[2] == {1}, "missing stream should be treated conservatively"


def test_same_stream_different_cpu_ops_not_marked():
    """Even when kernels come from different cpu_ops, same-stream should
    prevent them from being marked overlapping (stream check comes first)."""
    cpu_a = {"UID": 100, "cat": "cpu_op", "name": "aten::mm"}
    cpu_b = {"UID": 101, "cat": "cpu_op", "name": "aten::add"}
    k1 = _make_event(1, 0, 10, args={"stream": 7})
    k2 = _make_event(2, 5, 10, args={"stream": 7})
    events = _make_tree([cpu_a, cpu_b], [(k1, 201, 100), (k2, 202, 101)])
    analyser = GPUEventAnalyser(events)
    result = analyser.get_gpu_event_lists()
    by_uid = _get_overlapping_uids_by_uid(result)
    assert by_uid[1] == set()
    assert by_uid[2] == set()


def test_sub_microsecond_overlap_not_marked():
    """Back-to-back kernels with <1µs overlap (timestamp noise) must NOT be
    marked overlapping.  Reproduces Bug 1: consecutive same-stream GEMMs with
    0.2µs boundary artifact."""
    # kernel A: [1000, 2000], kernel B starts 0.2µs before A ends
    a = _make_event(1, 1000, 1000)  # [1000, 2000]
    b = _make_event(2, 1999.8, 1000)  # [1999.8, 2999.8]  overlap = 0.2µs
    analyser = GPUEventAnalyser([a, b])
    result = analyser.get_gpu_event_lists()
    by_uid = _get_overlapping_uids_by_uid(result)
    assert by_uid[1] == set(), "0.2µs overlap should be filtered as noise"
    assert by_uid[2] == set(), "0.2µs overlap should be filtered as noise"


def test_exactly_one_microsecond_overlap_is_marked():
    """Overlap of exactly 1.0µs meets the threshold (< 1.0 check is strict)
    so it IS marked as genuine overlap."""
    a = _make_event(1, 1000, 1000)  # [1000, 2000]
    b = _make_event(2, 1999.0, 1000)  # [1999, 2999]  overlap = 1.0µs
    analyser = GPUEventAnalyser([a, b])
    result = analyser.get_gpu_event_lists()
    by_uid = _get_overlapping_uids_by_uid(result)
    assert by_uid[1] == {2}, "1.0µs overlap meets the threshold"
    assert by_uid[2] == {1}, "1.0µs overlap meets the threshold"


def test_above_threshold_overlap_is_marked():
    """Overlap of >1µs is genuine and must be marked."""
    a = _make_event(1, 1000, 1000)  # [1000, 2000]
    b = _make_event(2, 1998.0, 1000)  # [1998, 2998]  overlap = 2.0µs
    analyser = GPUEventAnalyser([a, b])
    result = analyser.get_gpu_event_lists()
    by_uid = _get_overlapping_uids_by_uid(result)
    assert by_uid[1] == {2}, "2.0µs overlap should be marked"
    assert by_uid[2] == {1}, "2.0µs overlap should be marked"


def _mk_event(cat, name, ts, dur, pid, tid, args=None):
    return {
        "ph": "X",
        "cat": cat,
        "name": name,
        "pid": pid,
        "tid": tid,
        "ts": ts,
        "dur": dur,
        "args": args or {},
    }


def _mk_ac2g(corr_id, pid, tid, ts, phase):
    evt = {
        "ph": phase,
        "id": corr_id,
        "pid": pid,
        "tid": tid,
        "ts": ts,
        "cat": "ac2g",
        "name": "ac2g",
    }
    if phase == "f":
        evt["bp"] = "e"
    return evt


def _build_two_cpu_op_trace(
    kernel_a_ts, kernel_a_dur, kernel_b_ts, kernel_b_dur, stream_a=7, stream_b=8
):
    """Build a trace with two sequential cpu_ops under a parent, each launching
    one kernel.  CPU ops are sequential so the tree builder keeps them as
    siblings; their GPU kernels may overlap on different streams."""
    corr_a, corr_b = 100, 200
    return [
        _mk_event("cpu_op", "parent_op", ts=800, dur=500, pid=100, tid=100, args={}),
        _mk_event("cpu_op", "aten::mm", ts=810, dur=20, pid=100, tid=100, args={}),
        _mk_event(
            "cuda_runtime",
            "hipLaunchKernel",
            ts=815,
            dur=5,
            pid=100,
            tid=100,
            args={"correlation": corr_a},
        ),
        _mk_event(
            "kernel",
            "gemm_kernel",
            ts=kernel_a_ts,
            dur=kernel_a_dur,
            pid=0,
            tid=stream_a,
            args={"correlation": corr_a, "stream": stream_a},
        ),
        _mk_ac2g(corr_a, pid=0, tid=stream_a, ts=kernel_a_ts, phase="s"),
        _mk_ac2g(corr_a, pid=0, tid=stream_a, ts=kernel_a_ts, phase="f"),
        _mk_event("cpu_op", "aten::add", ts=840, dur=20, pid=100, tid=100, args={}),
        _mk_event(
            "cuda_runtime",
            "hipLaunchKernel",
            ts=845,
            dur=5,
            pid=100,
            tid=100,
            args={"correlation": corr_b},
        ),
        _mk_event(
            "kernel",
            "add_kernel",
            ts=kernel_b_ts,
            dur=kernel_b_dur,
            pid=0,
            tid=stream_b,
            args={"correlation": corr_b, "stream": stream_b},
        ),
        _mk_ac2g(corr_b, pid=0, tid=stream_b, ts=kernel_b_ts, phase="s"),
        _mk_ac2g(corr_b, pid=0, tid=stream_b, ts=kernel_b_ts, phase="f"),
    ]


def _get_launcher_by_name(launchers, name):
    return next(l for l in launchers if l["name"] == name)


def _build_analyzer_and_launchers(events):
    """Build tree, pre-compute overlapping_uids (as the report pipeline does),
    then return kernel launchers."""
    tree = TraceToTree(deepcopy(events))
    analyzer = TreePerfAnalyzer(tree, add_python_func=False)
    GPUEventAnalyser(tree.events).get_gpu_event_lists()
    return analyzer.get_kernel_launchers()


# ── overlap_pct: kernel launcher tests ──────────────────────────────────


class TestOverlapPctInKernelLaunchers:
    """Test overlap_pct values produced by get_kernel_launchers."""

    def test_partial_overlap(self):
        """Kernels partially overlap -> correct overlap_pct for each side.

        kernel A: [1000, 1100] dur=100
        kernel B: [1050, 1200] dur=150
        overlap region: [1050, 1100] = 50µs
        A's per-detail overlap_pct = 50/100 = 0.5
        B's per-detail overlap_pct = 50/150 = 1/3
        Top-level overlap_pct same (single kernel per launcher).
        """
        events = _build_two_cpu_op_trace(
            kernel_a_ts=1000,
            kernel_a_dur=100,
            kernel_b_ts=1050,
            kernel_b_dur=150,
        )
        launchers = _build_analyzer_and_launchers(events)

        mm = _get_launcher_by_name(launchers, "aten::mm")
        add = _get_launcher_by_name(launchers, "aten::add")

        assert mm["overlapping_kernels_details"] is not None
        assert len(mm["overlapping_kernels_details"]) == 1
        assert mm["overlapping_kernels_details"][0]["name"] == "add_kernel"
        assert mm["overlapping_kernels_details"][0]["overlap_pct"] == pytest.approx(0.5)
        assert mm["overlap_pct"] == pytest.approx(0.5)

        assert add["overlapping_kernels_details"] is not None
        assert len(add["overlapping_kernels_details"]) == 1
        assert add["overlapping_kernels_details"][0]["name"] == "gemm_kernel"
        assert add["overlapping_kernels_details"][0]["overlap_pct"] == pytest.approx(
            0.33
        )
        assert add["overlap_pct"] == pytest.approx(0.33)

    def test_full_containment(self):
        """One kernel fully contains the other.

        kernel A: [1000, 1200] dur=200
        kernel B: [1050, 1100] dur=50
        overlap = 50µs
        A's overlap_pct = 50/200 = 0.25
        B's overlap_pct = 50/50  = 1.0
        """
        events = _build_two_cpu_op_trace(
            kernel_a_ts=1000,
            kernel_a_dur=200,
            kernel_b_ts=1050,
            kernel_b_dur=50,
        )
        launchers = _build_analyzer_and_launchers(events)

        mm = _get_launcher_by_name(launchers, "aten::mm")
        add = _get_launcher_by_name(launchers, "aten::add")

        assert mm["overlapping_kernels_details"][0]["overlap_pct"] == pytest.approx(
            0.25
        )
        assert add["overlapping_kernels_details"][0]["overlap_pct"] == pytest.approx(
            1.0
        )

    def test_no_overlap(self):
        """Non-overlapping kernels -> overlapping_kernels_details is None.

        kernel A: [1000, 1100]
        kernel B: [1200, 1300]
        """
        events = _build_two_cpu_op_trace(
            kernel_a_ts=1000,
            kernel_a_dur=100,
            kernel_b_ts=1200,
            kernel_b_dur=100,
        )
        launchers = _build_analyzer_and_launchers(events)

        mm = _get_launcher_by_name(launchers, "aten::mm")
        add = _get_launcher_by_name(launchers, "aten::add")

        assert mm["overlapping_kernels_details"] is None
        assert mm["overlap_pct"] is None
        assert add["overlapping_kernels_details"] is None
        assert add["overlap_pct"] is None

    def test_exact_same_interval(self):
        """Identical intervals -> overlap_pct = 1.0 for both.

        kernel A: [1000, 1100] dur=100
        kernel B: [1000, 1100] dur=100
        overlap = 100µs -> pct = 1.0 for both
        """
        events = _build_two_cpu_op_trace(
            kernel_a_ts=1000,
            kernel_a_dur=100,
            kernel_b_ts=1000,
            kernel_b_dur=100,
        )
        launchers = _build_analyzer_and_launchers(events)

        mm = _get_launcher_by_name(launchers, "aten::mm")
        add = _get_launcher_by_name(launchers, "aten::add")

        assert mm["overlapping_kernels_details"][0]["overlap_pct"] == pytest.approx(1.0)
        assert add["overlapping_kernels_details"][0]["overlap_pct"] == pytest.approx(
            1.0
        )

    def test_touching_at_boundary(self):
        """Kernels touch at boundary -> NOT overlapping.

        kernel A: [1000, 1100] dur=100
        kernel B: [1100, 1200] dur=100
        No temporal overlap -> no overlapping_kernels_details.
        """
        events = _build_two_cpu_op_trace(
            kernel_a_ts=1000,
            kernel_a_dur=100,
            kernel_b_ts=1100,
            kernel_b_dur=100,
        )
        launchers = _build_analyzer_and_launchers(events)

        mm = _get_launcher_by_name(launchers, "aten::mm")
        add = _get_launcher_by_name(launchers, "aten::add")

        assert mm["overlapping_kernels_details"] is None
        assert mm["overlap_pct"] is None
        assert add["overlapping_kernels_details"] is None
        assert add["overlap_pct"] is None

    def test_multiple_overlapping_kernels(self):
        """One cpu_op's kernel overlaps with kernels from two other cpu_ops.

        cpu_op A kernel: [1000, 1200] dur=200
        cpu_op B kernel: [1050, 1100] dur=50   -> overlap with A = 50µs, pct_A = 50/200 = 0.25
        cpu_op C kernel: [1150, 1250] dur=100  -> overlap with A = 50µs, pct_A = 50/200 = 0.25
        """
        corr_a, corr_b, corr_c = 100, 200, 300
        events = [
            _mk_event(
                "cpu_op", "parent_op", ts=700, dur=500, pid=100, tid=100, args={}
            ),
            _mk_event("cpu_op", "op_A", ts=710, dur=20, pid=100, tid=100, args={}),
            _mk_event(
                "cuda_runtime",
                "hipLaunchKernel",
                ts=715,
                dur=5,
                pid=100,
                tid=100,
                args={"correlation": corr_a},
            ),
            _mk_event(
                "kernel",
                "kernel_A",
                ts=1000,
                dur=200,
                pid=0,
                tid=7,
                args={"correlation": corr_a, "stream": 7},
            ),
            _mk_ac2g(corr_a, pid=0, tid=7, ts=1000, phase="s"),
            _mk_ac2g(corr_a, pid=0, tid=7, ts=1000, phase="f"),
            _mk_event("cpu_op", "op_B", ts=740, dur=20, pid=100, tid=100, args={}),
            _mk_event(
                "cuda_runtime",
                "hipLaunchKernel",
                ts=745,
                dur=5,
                pid=100,
                tid=100,
                args={"correlation": corr_b},
            ),
            _mk_event(
                "kernel",
                "kernel_B",
                ts=1050,
                dur=50,
                pid=0,
                tid=8,
                args={"correlation": corr_b, "stream": 8},
            ),
            _mk_ac2g(corr_b, pid=0, tid=8, ts=1050, phase="s"),
            _mk_ac2g(corr_b, pid=0, tid=8, ts=1050, phase="f"),
            _mk_event("cpu_op", "op_C", ts=770, dur=20, pid=100, tid=100, args={}),
            _mk_event(
                "cuda_runtime",
                "hipLaunchKernel",
                ts=775,
                dur=5,
                pid=100,
                tid=100,
                args={"correlation": corr_c},
            ),
            _mk_event(
                "kernel",
                "kernel_C",
                ts=1150,
                dur=100,
                pid=0,
                tid=9,
                args={"correlation": corr_c, "stream": 9},
            ),
            _mk_ac2g(corr_c, pid=0, tid=9, ts=1150, phase="s"),
            _mk_ac2g(corr_c, pid=0, tid=9, ts=1150, phase="f"),
        ]

        launchers = _build_analyzer_and_launchers(events)

        op_a = _get_launcher_by_name(launchers, "op_A")
        details = op_a["overlapping_kernels_details"]
        assert details is not None
        assert len(details) == 2

        pct_by_name = {d["name"]: d["overlap_pct"] for d in details}
        assert pct_by_name["kernel_B"] == pytest.approx(0.25)
        assert pct_by_name["kernel_C"] == pytest.approx(0.25)

    def test_multi_kernel_cpu_op_partial_overlap(self):
        """cpu_op A launches two sequential kernels; cpu_op B's kernel overlaps
        with only one of them.  overlap_pct denominator is total kernel runtime
        (sum of merged kernel durations), not the span.

        cpu_op A kernels: kernel_A1 [1000, 1100] dur=100, kernel_A2 [1200, 1300] dur=100
            total_kernel_runtime = 200 (not 300 — the gap doesn't count)
        cpu_op B kernel:  kernel_B  [1050, 1150] dur=100
            total_kernel_runtime = 100

        Overlap region (A1 vs B): [1050, 1100] = 50µs
        A's overlap_pct = 50 / 200 = 0.25
        B's overlap_pct = 50 / 100 = 0.5
        """
        corr_a1, corr_a2, corr_b = 100, 101, 200
        events = [
            _mk_event("cpu_op", "parent_op", ts=800, dur=500, pid=100, tid=100),
            _mk_event("cpu_op", "op_A", ts=810, dur=30, pid=100, tid=100),
            _mk_event(
                "cuda_runtime",
                "hipLaunchKernel",
                ts=812,
                dur=5,
                pid=100,
                tid=100,
                args={"correlation": corr_a1},
            ),
            _mk_event(
                "cuda_runtime",
                "hipLaunchKernel",
                ts=820,
                dur=5,
                pid=100,
                tid=100,
                args={"correlation": corr_a2},
            ),
            _mk_event(
                "kernel",
                "kernel_A1",
                ts=1000,
                dur=100,
                pid=0,
                tid=7,
                args={"correlation": corr_a1, "stream": 7},
            ),
            _mk_ac2g(corr_a1, pid=0, tid=7, ts=1000, phase="s"),
            _mk_ac2g(corr_a1, pid=0, tid=7, ts=1000, phase="f"),
            _mk_event(
                "kernel",
                "kernel_A2",
                ts=1200,
                dur=100,
                pid=0,
                tid=7,
                args={"correlation": corr_a2, "stream": 7},
            ),
            _mk_ac2g(corr_a2, pid=0, tid=7, ts=1200, phase="s"),
            _mk_ac2g(corr_a2, pid=0, tid=7, ts=1200, phase="f"),
            _mk_event("cpu_op", "op_B", ts=850, dur=20, pid=100, tid=100),
            _mk_event(
                "cuda_runtime",
                "hipLaunchKernel",
                ts=855,
                dur=5,
                pid=100,
                tid=100,
                args={"correlation": corr_b},
            ),
            _mk_event(
                "kernel",
                "kernel_B",
                ts=1050,
                dur=100,
                pid=0,
                tid=8,
                args={"correlation": corr_b, "stream": 8},
            ),
            _mk_ac2g(corr_b, pid=0, tid=8, ts=1050, phase="s"),
            _mk_ac2g(corr_b, pid=0, tid=8, ts=1050, phase="f"),
        ]

        launchers = _build_analyzer_and_launchers(events)

        op_a = _get_launcher_by_name(launchers, "op_A")
        op_b = _get_launcher_by_name(launchers, "op_B")

        assert op_a["overlapping_kernels_details"] is not None
        assert len(op_a["overlapping_kernels_details"]) == 1
        assert op_a["overlapping_kernels_details"][0]["name"] == "kernel_B"
        assert op_a["overlapping_kernels_details"][0]["overlap_pct"] == pytest.approx(
            0.25
        )
        assert op_a["overlap_pct"] == pytest.approx(0.25)

        assert op_b["overlapping_kernels_details"] is not None
        assert len(op_b["overlapping_kernels_details"]) == 1
        assert op_b["overlapping_kernels_details"][0]["name"] == "kernel_A1"
        assert op_b["overlap_pct"] == pytest.approx(0.5)

    def test_multi_kernel_cpu_op_spanning_overlap(self):
        """cpu_op A launches two sequential kernels; cpu_op B's kernel spans
        the gap and overlaps with both.

        cpu_op A kernels: kernel_A1 [1000, 1100] dur=100, kernel_A2 [1200, 1300] dur=100
            total_kernel_runtime = 200
        cpu_op B kernel:  kernel_B  [1050, 1250] dur=200
            total_kernel_runtime = 200

        Overlap regions:
            A1 vs B: [1050, 1100] = 50µs
            A2 vs B: [1200, 1250] = 50µs
            total overlap for A = 100µs
        A's overlap_pct = 100 / 200 = 0.5
        B's overlap_pct = 100 / 200 = 0.5
        """
        corr_a1, corr_a2, corr_b = 100, 101, 200
        events = [
            _mk_event("cpu_op", "parent_op", ts=800, dur=500, pid=100, tid=100),
            _mk_event("cpu_op", "op_A", ts=810, dur=30, pid=100, tid=100),
            _mk_event(
                "cuda_runtime",
                "hipLaunchKernel",
                ts=812,
                dur=5,
                pid=100,
                tid=100,
                args={"correlation": corr_a1},
            ),
            _mk_event(
                "cuda_runtime",
                "hipLaunchKernel",
                ts=820,
                dur=5,
                pid=100,
                tid=100,
                args={"correlation": corr_a2},
            ),
            _mk_event(
                "kernel",
                "kernel_A1",
                ts=1000,
                dur=100,
                pid=0,
                tid=7,
                args={"correlation": corr_a1, "stream": 7},
            ),
            _mk_ac2g(corr_a1, pid=0, tid=7, ts=1000, phase="s"),
            _mk_ac2g(corr_a1, pid=0, tid=7, ts=1000, phase="f"),
            _mk_event(
                "kernel",
                "kernel_A2",
                ts=1200,
                dur=100,
                pid=0,
                tid=7,
                args={"correlation": corr_a2, "stream": 7},
            ),
            _mk_ac2g(corr_a2, pid=0, tid=7, ts=1200, phase="s"),
            _mk_ac2g(corr_a2, pid=0, tid=7, ts=1200, phase="f"),
            _mk_event("cpu_op", "op_B", ts=850, dur=20, pid=100, tid=100),
            _mk_event(
                "cuda_runtime",
                "hipLaunchKernel",
                ts=855,
                dur=5,
                pid=100,
                tid=100,
                args={"correlation": corr_b},
            ),
            _mk_event(
                "kernel",
                "kernel_B",
                ts=1050,
                dur=200,
                pid=0,
                tid=8,
                args={"correlation": corr_b, "stream": 8},
            ),
            _mk_ac2g(corr_b, pid=0, tid=8, ts=1050, phase="s"),
            _mk_ac2g(corr_b, pid=0, tid=8, ts=1050, phase="f"),
        ]

        launchers = _build_analyzer_and_launchers(events)

        op_a = _get_launcher_by_name(launchers, "op_A")
        op_b = _get_launcher_by_name(launchers, "op_B")

        assert op_a["overlapping_kernels_details"] is not None
        assert op_a["overlap_pct"] == pytest.approx(0.5)

        assert op_b["overlapping_kernels_details"] is not None
        assert op_b["overlap_pct"] == pytest.approx(0.5)


# ── overlap_pct: DataFrame column tests ─────────────────────────────────


class TestOverlapPctDataFrameColumn:
    """Test that overlap_pct appears as a top-level DataFrame column."""

    def test_overlap_pct_in_df_kernel_launchers(self):
        """overlap_pct should be a column; rounded to hundredths."""
        events = _build_two_cpu_op_trace(
            kernel_a_ts=1000,
            kernel_a_dur=100,
            kernel_b_ts=1050,
            kernel_b_dur=150,
        )
        tree = TraceToTree(deepcopy(events))
        analyzer = TreePerfAnalyzer(tree, add_python_func=False)
        GPUEventAnalyser(tree.events).get_gpu_event_lists()
        df = analyzer.get_df_kernel_launchers(include_kernel_details=True)

        assert "overlap_pct" in df.columns
        mm_row = df[df["name"] == "aten::mm"].iloc[0]
        add_row = df[df["name"] == "aten::add"].iloc[0]
        assert mm_row["overlap_pct"] == pytest.approx(0.5)
        assert add_row["overlap_pct"] == pytest.approx(0.33)

    def test_overlap_pct_blank_when_no_overlap(self):
        """overlap_pct should be NaN in the DataFrame when no overlapping kernels."""
        events = _build_two_cpu_op_trace(
            kernel_a_ts=1000,
            kernel_a_dur=100,
            kernel_b_ts=1200,
            kernel_b_dur=100,
        )
        tree = TraceToTree(deepcopy(events))
        analyzer = TreePerfAnalyzer(tree, add_python_func=False)
        GPUEventAnalyser(tree.events).get_gpu_event_lists()
        df = analyzer.get_df_kernel_launchers(include_kernel_details=True)

        assert "overlap_pct" in df.columns
        assert pd.isna(df[df["name"] == "aten::mm"].iloc[0]["overlap_pct"])
        assert pd.isna(df[df["name"] == "aten::add"].iloc[0]["overlap_pct"])

    def test_overlap_pct_in_unique_args_kl_overlap(self):
        """overlap_pct should survive aggregation in get_df_kernel_launchers_unique_args."""
        events = _build_two_cpu_op_trace(
            kernel_a_ts=1000,
            kernel_a_dur=100,
            kernel_b_ts=1050,
            kernel_b_dur=150,
        )
        tree = TraceToTree(deepcopy(events))
        analyzer = TreePerfAnalyzer(tree, add_python_func=False)
        GPUEventAnalyser(tree.events).get_gpu_event_lists()
        df = analyzer.get_df_kernel_launchers(include_kernel_details=True)
        df_agg = TreePerfAnalyzer.get_df_kernel_launchers_unique_args(
            df, agg_metrics=["mean"], include_overlapping_kernels=True
        )

        overlap_cols = [c for c in df_agg.columns if "overlap_pct" in c]
        assert (
            len(overlap_cols) > 0
        ), f"No overlap_pct column found. Columns: {list(df_agg.columns)}"


# ── Bug 2 regression: overlap data must not depend on call order ────────

TRACE_PATH = os.path.join(
    "tests",
    "traces",
    "mi300",
    "llama_70b_fsdp",
    "rank0_trace_no_pyfn.json.gz",
)


@pytest.mark.skipif(
    not os.path.exists(TRACE_PATH), reason=f"Trace not found: {TRACE_PATH}"
)
class TestOverlapCallOrderIndependence:
    """get_df_kernel_launchers() must return identical overlap data regardless
    of whether get_df_gpu_timeline() was called first (Bug 2 regression)."""

    def test_kernel_launchers_overlap_without_prior_gpu_timeline(self):
        """Calling get_df_kernel_launchers() directly must produce the same
        overlap results as calling get_df_gpu_timeline() first."""
        # Path A: get_df_kernel_launchers() only
        analyzer_a = TreePerfAnalyzer.from_file(TRACE_PATH)
        df_a = analyzer_a.get_df_kernel_launchers()

        # Path B: get_df_gpu_timeline() first (previously required for overlap)
        analyzer_b = TreePerfAnalyzer.from_file(TRACE_PATH)
        analyzer_b.get_df_gpu_timeline()
        df_b = analyzer_b.get_df_kernel_launchers()

        count_a = df_a["overlap_pct"].notna().sum()
        count_b = df_b["overlap_pct"].notna().sum()

        assert count_a == count_b, (
            f"overlap_pct count differs: direct={count_a}, "
            f"after get_df_gpu_timeline={count_b}"
        )

    def test_overlap_values_match(self):
        """The actual overlap_pct values must match, not just the counts."""
        analyzer_a = TreePerfAnalyzer.from_file(TRACE_PATH)
        df_a = analyzer_a.get_df_kernel_launchers()

        analyzer_b = TreePerfAnalyzer.from_file(TRACE_PATH)
        analyzer_b.get_df_gpu_timeline()
        df_b = analyzer_b.get_df_kernel_launchers()

        pcts_a = df_a.set_index("UID")["overlap_pct"].dropna().sort_index()
        pcts_b = df_b.set_index("UID")["overlap_pct"].dropna().sort_index()

        assert list(pcts_a.index) == list(
            pcts_b.index
        ), "UIDs with overlap differ between paths"
        pd.testing.assert_series_equal(pcts_a, pcts_b, check_names=False)
