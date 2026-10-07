###############################################################################
# Copyright (c) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for TraceLens.IdleTimeAnalyser

Covers:
  - compute_self_times correctness (flat, nested, multi-level, disjoint)
  - classify_sync_event for each sync type
  - classify_runtime_event categories
  - extract_idle_intervals edge cases (no events, single event, overlapping)
  - classify_idle_intervals on synthetic mini-traces
  - assign_idle_ids sequencing
  - standalone CLI and --enable_idle_analysis perf report integration
  - Edge cases: empty traces, single GPU event, zero-duration events
"""

import gzip
import json
import os

import pytest

from TraceLens.IdleTimeAnalyser.classify import (
    compute_self_times,
    classify_sync_event,
    classify_runtime_event,
    extract_idle_intervals,
    assign_idle_ids,
    get_overlapping_events,
)

# ---------------------------------------------------------------------------
# compute_self_times
# ---------------------------------------------------------------------------


class TestComputeSelfTimes:
    def test_single_event(self):
        events = [{"ts": 10, "dur": 50, "t_end": 60, "name": "op_a", "cat": "cpu_op"}]
        result = compute_self_times(events, 10, 60)
        assert result == {"op_a": 50.0}

    def test_parent_child(self):
        events = [
            {"ts": 10, "dur": 100, "t_end": 110, "name": "parent", "cat": "cpu_op"},
            {"ts": 20, "dur": 60, "t_end": 80, "name": "child", "cat": "cpu_op"},
        ]
        result = compute_self_times(events, 10, 110)
        assert result["child"] == 60.0
        assert result["parent"] == pytest.approx(40.0)

    def test_three_levels(self):
        events = [
            {"ts": 0, "dur": 100, "t_end": 100, "name": "grandparent", "cat": "cpu_op"},
            {"ts": 10, "dur": 60, "t_end": 70, "name": "parent", "cat": "cpu_op"},
            {"ts": 20, "dur": 30, "t_end": 50, "name": "child", "cat": "cpu_op"},
        ]
        result = compute_self_times(events, 0, 100)
        assert result["child"] == 30.0
        assert result["parent"] == pytest.approx(30.0)  # 60 - 30
        assert result["grandparent"] == pytest.approx(40.0)  # 100 - 60

    def test_sibling_children(self):
        events = [
            {"ts": 0, "dur": 100, "t_end": 100, "name": "parent", "cat": "cpu_op"},
            {"ts": 10, "dur": 20, "t_end": 30, "name": "child_a", "cat": "cpu_op"},
            {"ts": 50, "dur": 20, "t_end": 70, "name": "child_b", "cat": "cpu_op"},
        ]
        result = compute_self_times(events, 0, 100)
        assert result["child_a"] == 20.0
        assert result["child_b"] == 20.0
        assert result["parent"] == pytest.approx(60.0)  # 100 - 20 - 20

    def test_clipping_to_gap(self):
        events = [
            {"ts": 0, "dur": 200, "t_end": 200, "name": "wide_op", "cat": "cpu_op"},
        ]
        result = compute_self_times(events, 50, 150)
        assert result["wide_op"] == pytest.approx(100.0)

    def test_no_overlap(self):
        events = [
            {"ts": 200, "dur": 50, "t_end": 250, "name": "far_op", "cat": "cpu_op"},
        ]
        result = compute_self_times(events, 0, 100)
        assert result == {}

    def test_empty_events(self):
        result = compute_self_times([], 0, 100)
        assert result == {}

    def test_zero_duration_event(self):
        events = [
            {"ts": 50, "dur": 0, "t_end": 50, "name": "instant", "cat": "cpu_op"},
        ]
        result = compute_self_times(events, 0, 100)
        assert result == {}

    def test_overlapping_children_merged(self):
        events = [
            {"ts": 0, "dur": 100, "t_end": 100, "name": "parent", "cat": "cpu_op"},
            {"ts": 10, "dur": 30, "t_end": 40, "name": "child_a", "cat": "cpu_op"},
            {"ts": 30, "dur": 30, "t_end": 60, "name": "child_b", "cat": "cpu_op"},
        ]
        result = compute_self_times(events, 0, 100)
        # children overlap at [30,40], union is [10,60] = 50
        assert result["parent"] == pytest.approx(50.0)  # 100 - 50
        assert result["child_a"] == pytest.approx(30.0)
        assert result["child_b"] == pytest.approx(30.0)

    def test_python_function_included(self):
        events = [
            {
                "ts": 0,
                "dur": 100,
                "t_end": 100,
                "name": "model.py: forward",
                "cat": "python_function",
            },
            {"ts": 10, "dur": 60, "t_end": 70, "name": "aten::conv2d", "cat": "cpu_op"},
        ]
        result = compute_self_times(events, 0, 100)
        assert "model.py: forward" in result
        assert result["model.py: forward"] == pytest.approx(40.0)
        assert result["aten::conv2d"] == 60.0

    def test_same_name_events_summed(self):
        events = [
            {"ts": 0, "dur": 10, "t_end": 10, "name": "aten::add", "cat": "cpu_op"},
            {"ts": 20, "dur": 10, "t_end": 30, "name": "aten::add", "cat": "cpu_op"},
        ]
        result = compute_self_times(events, 0, 50)
        assert result["aten::add"] == pytest.approx(20.0)

    def test_parent_child_both_span_gap(self):
        """Regression: when both parent and child clip to the full gap range,
        the child should still get credit (parent self-time = 0)."""
        events = [
            {
                "ts": 0,
                "dur": 200,
                "t_end": 200,
                "name": "FlashAttnFunc",
                "cat": "cpu_op",
            },
            {
                "ts": 10,
                "dur": 180,
                "t_end": 190,
                "name": "flash_forward",
                "cat": "cpu_op",
            },
        ]
        result = compute_self_times(events, 50, 150)
        assert "flash_forward" in result
        assert result["flash_forward"] == pytest.approx(100.0)
        assert "FlashAttnFunc" not in result or result.get("FlashAttnFunc", 0) == 0

    def test_item_local_scalar_dense_pattern(self):
        """aten::item contains aten::_local_scalar_dense with same start."""
        events = [
            {
                "ts": 10,
                "dur": 1000,
                "t_end": 1010,
                "name": "aten::item",
                "cat": "cpu_op",
            },
            {
                "ts": 10,
                "dur": 998,
                "t_end": 1008,
                "name": "aten::_local_scalar_dense",
                "cat": "cpu_op",
            },
        ]
        result = compute_self_times(events, 50, 100)
        assert "aten::_local_scalar_dense" in result
        assert result["aten::_local_scalar_dense"] == pytest.approx(50.0)

    def test_identical_events_not_mutual_containment(self):
        """Two events with identical boundaries should not zero each other out."""
        events = [
            {"ts": 10, "dur": 100, "t_end": 110, "name": "op_a", "cat": "cpu_op"},
            {"ts": 10, "dur": 100, "t_end": 110, "name": "op_b", "cat": "cpu_op"},
        ]
        result = compute_self_times(events, 20, 80)
        total = sum(result.values())
        assert total > 0


# ---------------------------------------------------------------------------
# classify_sync_event
# ---------------------------------------------------------------------------


class TestClassifySyncEvent:
    def test_device_sync(self):
        evt = {"name": "hipDeviceSynchronize", "args": {}}
        assert classify_sync_event(evt) == "DEVICE_SYNC"

    def test_cuda_device_sync(self):
        evt = {"name": "cudaDeviceSynchronize", "args": {}}
        assert classify_sync_event(evt) == "DEVICE_SYNC"

    def test_stream_sync(self):
        evt = {"name": "hipStreamSynchronize", "args": {}}
        assert classify_sync_event(evt) == "STREAM_SYNC"

    def test_event_sync(self):
        evt = {"name": "hipEventSynchronize", "args": {}}
        assert classify_sync_event(evt) == "EVENT_SYNC"

    def test_d2h_copy_by_kind(self):
        evt = {"name": "hipMemcpyWithStream", "args": {"kind": "2"}}
        assert classify_sync_event(evt) == "D2H_COPY"

    def test_h2d_copy_by_kind(self):
        evt = {"name": "hipMemcpyWithStream", "args": {"kind": "1"}}
        assert classify_sync_event(evt) == "H2D_COPY"

    def test_d2h_copy_by_string_kind(self):
        evt = {"name": "hipMemcpy", "args": {"kind": "DtoH"}}
        assert classify_sync_event(evt) == "D2H_COPY"

    def test_h2d_copy_by_string_kind(self):
        evt = {"name": "hipMemcpy", "args": {"kind": "HtoD"}}
        assert classify_sync_event(evt) == "H2D_COPY"

    def test_memcpy_unknown_kind(self):
        evt = {"name": "hipMemcpyWithStream", "args": {"kind": "3"}}
        assert classify_sync_event(evt) is None

    def test_non_sync_event(self):
        evt = {"name": "hipLaunchKernel", "args": {}}
        assert classify_sync_event(evt) is None

    def test_memcpy_kind_from_preceding_gpu(self):
        evt = {"name": "hipMemcpyWithStream", "args": {}}
        gpu_evt = {"args": {"kind": "DtoH"}}
        assert classify_sync_event(evt, gpu_evt) == "D2H_COPY"


# ---------------------------------------------------------------------------
# classify_runtime_event
# ---------------------------------------------------------------------------


class TestClassifyRuntimeEvent:
    def test_malloc(self):
        assert classify_runtime_event({"name": "hipMalloc"}) == "MEMORY_ALLOC"
        assert classify_runtime_event({"name": "cudaFree"}) == "MEMORY_ALLOC"

    def test_launch(self):
        assert classify_runtime_event({"name": "hipLaunchKernel"}) == "LAUNCH_STALL"
        assert classify_runtime_event({"name": "cudaLaunchKernel"}) == "LAUNCH_STALL"

    def test_sync(self):
        assert classify_runtime_event({"name": "hipDeviceSynchronize"}) == "SYNC_CALL"

    def test_other(self):
        assert classify_runtime_event({"name": "hipSomethingElse"}) == "OTHER_RUNTIME"


# ---------------------------------------------------------------------------
# extract_idle_intervals
# ---------------------------------------------------------------------------


class TestExtractIdleIntervals:
    def test_two_events_one_gap(self):
        events = [
            {"ts": 0, "t_end": 10},
            {"ts": 20, "t_end": 30},
        ]
        gaps = extract_idle_intervals(events)
        assert len(gaps) == 1
        assert gaps[0] == (10, 20)

    def test_adjacent_events_no_gap(self):
        events = [
            {"ts": 0, "t_end": 10},
            {"ts": 10, "t_end": 20},
        ]
        gaps = extract_idle_intervals(events)
        assert len(gaps) == 0

    def test_overlapping_events_merged(self):
        events = [
            {"ts": 0, "t_end": 15},
            {"ts": 10, "t_end": 25},
            {"ts": 30, "t_end": 40},
        ]
        gaps = extract_idle_intervals(events)
        assert len(gaps) == 1
        assert gaps[0] == (25, 30)

    def test_single_event_no_gaps(self):
        events = [{"ts": 0, "t_end": 100}]
        gaps = extract_idle_intervals(events)
        assert len(gaps) == 0

    def test_empty_events(self):
        gaps = extract_idle_intervals([])
        assert len(gaps) == 0


# ---------------------------------------------------------------------------
# assign_idle_ids
# ---------------------------------------------------------------------------


class TestAssignIdleIds:
    def test_basic_assignment(self):
        records = [
            {"label_noise": True},
            {"label_noise": False},
            {"label_noise": False},
            {"label_noise": True},
            {"label_noise": False},
        ]
        assign_idle_ids(records)
        assert records[0]["idle_id"] == -1  # noise
        assert records[1]["idle_id"] == 0  # macro
        assert records[2]["idle_id"] == 1  # macro
        assert records[3]["idle_id"] == -2  # noise
        assert records[4]["idle_id"] == 2  # macro

    def test_all_noise(self):
        records = [{"label_noise": True}, {"label_noise": True}]
        assign_idle_ids(records)
        assert records[0]["idle_id"] == -1
        assert records[1]["idle_id"] == -2

    def test_all_macro(self):
        records = [{"label_noise": False}, {"label_noise": False}]
        assign_idle_ids(records)
        assert records[0]["idle_id"] == 0
        assert records[1]["idle_id"] == 1


# ---------------------------------------------------------------------------
# get_overlapping_events
# ---------------------------------------------------------------------------


class TestGetOverlappingEvents:
    def test_basic_overlap(self):
        events = [
            {"ts": 0, "t_end": 10},
            {"ts": 5, "t_end": 15},
            {"ts": 20, "t_end": 30},
        ]
        result = get_overlapping_events(events, 8, 22)
        # All three overlap: [0,10] has t_end=10 > 8, [5,15] spans, [20,30] starts at 20 < 22
        assert len(result) == 3

    def test_no_overlap(self):
        events = [
            {"ts": 0, "t_end": 10},
            {"ts": 20, "t_end": 30},
        ]
        result = get_overlapping_events(events, 12, 18)
        assert len(result) == 0

    def test_event_spanning_interval(self):
        events = [{"ts": 0, "t_end": 100}]
        result = get_overlapping_events(events, 20, 30)
        assert len(result) == 1

    def test_empty_events(self):
        result = get_overlapping_events([], 0, 100)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Integration: classify_idle_intervals on a synthetic mini-trace
# ---------------------------------------------------------------------------


def _make_synthetic_tree():
    """Create a minimal mock tree object for classify_idle_intervals."""
    from types import SimpleNamespace

    events = []
    uid = 0

    # GPU kernels on stream 7
    for start in [100, 200, 500]:
        events.append(
            {
                "ph": "X",
                "cat": "kernel",
                "name": f"kernel_{start}",
                "ts": start,
                "dur": 50,
                "t_end": start + 50,
                "pid": 1,
                "tid": 7,
                "uid": uid,
                "UID": uid,
            }
        )
        uid += 1

    # CPU runtime: a launch for each kernel
    for start, kernel_uid in [(90, 0), (190, 1), (490, 2)]:
        events.append(
            {
                "ph": "X",
                "cat": "cuda_runtime",
                "name": "hipLaunchKernel",
                "ts": start,
                "dur": 5,
                "t_end": start + 5,
                "pid": 0,
                "tid": 0,
                "uid": uid,
                "UID": uid,
            }
        )
        events[kernel_uid]["parent"] = uid
        uid += 1

    # CPU op spanning the gap between kernel 1 (ends at 250) and kernel 2 (starts at 500)
    events.append(
        {
            "ph": "X",
            "cat": "cpu_op",
            "name": "aten::heavy_op",
            "ts": 260,
            "dur": 200,
            "t_end": 460,
            "pid": 0,
            "tid": 0,
            "uid": uid,
        }
    )
    uid += 1

    events_by_uid = {e["uid"]: e for e in events}

    tree = SimpleNamespace(
        events=events,
        events_by_uid=events_by_uid,
    )
    return tree


class TestClassifyIdleIntervalsIntegration:
    def test_synthetic_trace(self):
        from TraceLens.IdleTimeAnalyser.classify import classify_idle_intervals

        tree = _make_synthetic_tree()
        results = classify_idle_intervals(tree, micro_thresh_us=5.0)

        # Gap between kernel_100 (end=150) and kernel_200 (start=200) = 50us
        # Gap between kernel_200 (end=250) and kernel_500 (start=500) = 250us
        macro = [r for r in results if not r["label_noise"]]
        assert len(macro) >= 1

        big_gap = [r for r in macro if r["duration"] > 100]
        assert len(big_gap) == 1
        assert big_gap[0]["drain_type"] == "starved"
        assert big_gap[0]["cpu_during_gap"] in ("CPU_DOMINATED", "RUNTIME_DOMINATED")

    def test_empty_tree(self):
        from types import SimpleNamespace
        from TraceLens.IdleTimeAnalyser.classify import classify_idle_intervals

        tree = SimpleNamespace(events=[], events_by_uid={})
        results = classify_idle_intervals(tree)
        assert results == []


def _build_tree(kernels, runtime=(), ops=()):
    """Build a mock tree.

    kernels: list of (ts, dur, launch_idx_or_None) -- launch_idx indexes into runtime.
    runtime: list of (name, ts, dur).
    ops:     list of (name, ts, dur) cpu_op events.
    """
    from types import SimpleNamespace

    events = []

    def add(evt):
        evt["UID"] = len(events)
        evt["t_end"] = evt["ts"] + evt["dur"]
        evt.setdefault("ph", "X")
        events.append(evt)
        return evt

    rt_events = [
        add({"cat": "cuda_runtime", "name": n, "ts": ts, "dur": d, "pid": 0, "tid": 0})
        for n, ts, d in runtime
    ]
    for n, ts, d in ops:
        add({"cat": "cpu_op", "name": n, "ts": ts, "dur": d, "pid": 0, "tid": 0})
    for i, (ts, dur, launch_idx) in enumerate(kernels):
        k = add(
            {"cat": "kernel", "name": f"k{i}", "ts": ts, "dur": dur, "pid": 1, "tid": 7}
        )
        if launch_idx is not None:
            k["parent"] = rt_events[launch_idx]["UID"]

    return SimpleNamespace(events=events, events_by_uid={e["UID"]: e for e in events})


class TestClassificationScenarios:
    def _macro(self, tree):
        from TraceLens.IdleTimeAnalyser.classify import classify_idle_intervals

        return [r for r in classify_idle_intervals(tree) if not r["label_noise"]]

    def test_launch_anomaly_launched_during_gap(self):
        # gap [10, 100); next kernel launched at 40-45, starts at 100:
        # launch_to_exec = 55us > 10us and > 25% of the 90us gap.
        tree = _build_tree(
            kernels=[(0, 10, None), (100, 10, 0)],
            runtime=[("hipLaunchKernel", 40, 5)],
        )
        (rec,) = self._macro(tree)
        assert rec["kernel_prequeued"] is False
        assert rec["cpu_during_gap"] == "LAUNCH_ANOMALY"
        assert "launched_during_gap" in rec["cpu_during_gap_detail"]

    def test_small_launch_latency_in_long_gap_is_not_anomaly(self):
        # launch_to_exec = 15us exceeds the 10us floor but is < 25% of the 200us gap.
        tree = _build_tree(
            kernels=[(0, 10, None), (210, 10, 0)],
            runtime=[("hipLaunchKernel", 190, 5)],
        )
        (rec,) = self._macro(tree)
        assert rec["launch_to_exec_us"] == pytest.approx(15)
        assert rec["cpu_during_gap"] == "CPU_UNTRACED"
        assert rec["dominant_op"] == "(no_cpu_op_overlap)"

    def test_launch_anomaly_prequeued(self):
        # Kernel launched well before the gap, yet GPU still sat idle for 50us.
        tree = _build_tree(
            kernels=[(0, 100, None), (150, 10, 0)],
            runtime=[("hipLaunchKernel", 20, 5)],
        )
        (rec,) = self._macro(tree)
        assert rec["kernel_prequeued"] is True
        assert rec["cpu_during_gap"] == "LAUNCH_ANOMALY"
        assert "prequeued" in rec["cpu_during_gap_detail"]

    def test_sync_drain(self):
        # hipDeviceSynchronize starts while k0 is running and returns just after it
        # finishes; the next launch happens only after the sync returns.
        tree = _build_tree(
            kernels=[(0, 100, None), (130, 10, 1)],
            runtime=[("hipDeviceSynchronize", 50, 55), ("hipLaunchKernel", 110, 5)],
        )
        (rec,) = self._macro(tree)
        assert rec["drain_type"] == "sync_drain"
        assert rec["sync_type"] == "DEVICE_SYNC"
        assert rec["sync_event_name"] == "hipDeviceSynchronize"

    def test_sync_after_launch_is_not_causal(self):
        # Next kernel was launched before the sync returned, so the sync
        # did not block submission.
        tree = _build_tree(
            kernels=[(0, 100, None), (130, 10, 1)],
            runtime=[("hipDeviceSynchronize", 50, 55), ("hipLaunchKernel", 60, 5)],
        )
        (rec,) = self._macro(tree)
        assert rec["drain_type"] == "starved"
        assert rec["sync_type"] is None

    def test_runtime_dominated_memory_alloc(self):
        tree = _build_tree(
            kernels=[(0, 10, None), (200, 10, 1)],
            runtime=[("hipMalloc", 20, 150), ("hipLaunchKernel", 180, 5)],
        )
        (rec,) = self._macro(tree)
        assert rec["cpu_during_gap"] == "RUNTIME_DOMINATED"
        assert rec["cpu_during_gap_detail"] == "MEMORY_ALLOC: hipMalloc"

    def test_cpu_dominated_by_op_self_time(self):
        tree = _build_tree(
            kernels=[(0, 10, None), (200, 10, 0)],
            runtime=[("hipLaunchKernel", 180, 5)],
            ops=[("aten::index", 20, 150)],
        )
        (rec,) = self._macro(tree)
        assert rec["cpu_during_gap"] == "CPU_DOMINATED"
        assert rec["dominant_op"] == "aten::index"


class TestHelpers:
    def test_find_launch_for_kernel_non_runtime_parent(self):
        from types import SimpleNamespace

        from TraceLens.IdleTimeAnalyser.classify import find_launch_for_kernel

        tree = SimpleNamespace(events_by_uid={1: {"cat": "cpu_op"}})
        assert find_launch_for_kernel(tree, {}) is None
        assert find_launch_for_kernel(tree, {"parent": 99}) is None
        assert find_launch_for_kernel(tree, {"parent": 1}) is None

    def test_find_gpu_pid_falls_back_to_process_labels(self):
        from TraceLens.IdleTimeAnalyser.classify import find_gpu_pid

        events = [
            {"ph": "M", "name": "process_labels", "pid": 3, "args": {"labels": "CPU"}},
            {
                "ph": "M",
                "name": "process_labels",
                "pid": 5,
                "args": {"labels": "GPU 0"},
            },
        ]
        assert find_gpu_pid(events) == 5
        assert find_gpu_pid([]) == 0

    def test_overlap_index_empty(self):
        from TraceLens.IdleTimeAnalyser.classify import OverlapIndex

        assert get_overlapping_events(OverlapIndex([]), 0, 100) == []


class TestIdleTimeAnalyserWrapper:
    """Tests for the IdleTimeAnalyser class (submodule entry point)."""

    def test_classify_returns_list(self):
        from TraceLens.IdleTimeAnalyser import IdleTimeAnalyser

        tree = _make_synthetic_tree()
        analyser = IdleTimeAnalyser(tree)
        classified = analyser.classify()
        assert isinstance(classified, list)
        assert len(classified) > 0

    def test_classify_caches_results(self):
        from TraceLens.IdleTimeAnalyser import IdleTimeAnalyser

        tree = _make_synthetic_tree()
        analyser = IdleTimeAnalyser(tree)
        first = analyser.classify()
        second = analyser.classify()
        assert first is second

    def test_get_dataframes(self):
        from TraceLens.IdleTimeAnalyser import IdleTimeAnalyser

        tree = _make_synthetic_tree()
        analyser = IdleTimeAnalyser(tree)
        dfs = analyser.get_dataframes()
        assert "idle_overview" in dfs
        assert "idle_summary" in dfs
        assert "idle_intervals" in dfs
        assert len(dfs["idle_intervals"]) > 0

    def test_get_augmented_events(self):
        from TraceLens.IdleTimeAnalyser import IdleTimeAnalyser

        tree = _make_synthetic_tree()
        analyser = IdleTimeAnalyser(tree)
        events = analyser.get_augmented_events(gpu_pid=1)
        assert isinstance(events, list)
        assert len(events) > 0
        for evt in events:
            assert evt["pid"] == 1


class TestBuildIdleDataframes:
    """Tests for report.py build_idle_dataframes function."""

    def test_overview_columns(self):
        from TraceLens.IdleTimeAnalyser.report import build_idle_dataframes
        from TraceLens.IdleTimeAnalyser.classify import (
            classify_idle_intervals,
            assign_idle_ids,
        )

        tree = _make_synthetic_tree()
        classified = classify_idle_intervals(tree)
        assign_idle_ids(classified)
        dfs = build_idle_dataframes(classified)
        overview = dfs["idle_overview"]
        for col in [
            "drain_type",
            "cpu_during_gap",
            "count",
            "total_time_ms",
            "pct_of_idle",
        ]:
            assert col in overview.columns, f"Missing column: {col}"

    def test_intervals_columns(self):
        from TraceLens.IdleTimeAnalyser.report import build_idle_dataframes
        from TraceLens.IdleTimeAnalyser.classify import (
            classify_idle_intervals,
            assign_idle_ids,
        )

        tree = _make_synthetic_tree()
        classified = classify_idle_intervals(tree)
        assign_idle_ids(classified)
        dfs = build_idle_dataframes(classified)
        intervals = dfs["idle_intervals"]
        required_cols = [
            "idle_id",
            "group",
            "start_us",
            "end_us",
            "duration_us",
            "drain_type",
            "cpu_during_gap",
            "launch_to_exec_us",
        ]
        for col in required_cols:
            assert col in intervals.columns, f"Missing column: {col}"

    def test_empty_classification(self):
        from TraceLens.IdleTimeAnalyser.report import build_idle_dataframes

        dfs = build_idle_dataframes([])
        assert dfs["idle_overview"].empty
        assert dfs["idle_intervals"].empty


class TestRealTraceCoverage:
    """Regression tests on real traces from the repo to ensure all classification
    branches are exercised."""

    TRACE_DIR = os.path.join(os.path.dirname(__file__), "traces")

    @staticmethod
    def _classify_trace(path):
        from TraceLens import TreePerfAnalyzer
        from TraceLens.IdleTimeAnalyser import IdleTimeAnalyser

        pa = TreePerfAnalyzer.from_file(path)
        analyser = IdleTimeAnalyser(pa.tree)
        return analyser.classify()

    def test_ddp_resnet18_has_memory_alloc(self):
        """DDP resnet18 without CUDA memory pool has hipFree/hipMalloc dominated gaps."""
        path = os.path.join(
            self.TRACE_DIR, "mi300", "ddp_resnet18_no_pool", "train_resnet18.json.gz"
        )
        if not os.path.exists(path):
            pytest.skip("Trace not available")
        classified = self._classify_trace(path)
        macro = [r for r in classified if not r["label_noise"]]
        rt_dominated = [r for r in macro if r["cpu_during_gap"] == "RUNTIME_DOMINATED"]
        rt_details = [r["cpu_during_gap_detail"] for r in rt_dominated]
        has_memory_alloc = any("MEMORY_ALLOC" in d for d in rt_details)
        assert (
            has_memory_alloc
        ), "Expected MEMORY_ALLOC in RUNTIME_DOMINATED details, got: {}".format(
            set(rt_details)
        )

    def test_fsdp_rank0_covers_all_cpu_categories(self):
        """FSDP rank0 trace should exercise most cpu_during_gap categories."""
        path = os.path.join(
            self.TRACE_DIR, "mi300", "llama_70b_fsdp", "rank0_trace_no_pyfn.json.gz"
        )
        if not os.path.exists(path):
            pytest.skip("Trace not available")
        classified = self._classify_trace(path)
        macro = [r for r in classified if not r["label_noise"]]
        cpu_categories = set(r["cpu_during_gap"] for r in macro)
        expected = {
            "LAUNCH_ANOMALY",
            "CPU_DOMINATED",
            "LAUNCH_OVERHEAD_ONLY",
            "CPU_UNTRACED",
        }
        missing = expected - cpu_categories
        assert not missing, "Missing cpu_during_gap categories: {}".format(missing)


DDP_RESNET18_TRACE = os.path.join(
    os.path.dirname(__file__),
    "traces",
    "mi300",
    "ddp_resnet18_no_pool",
    "train_resnet18.json.gz",
)


@pytest.fixture
def ddp_trace_copy(tmp_path):
    """Copy the DDP resnet18 trace into tmp_path so outputs written next to it
    don't land in the repo."""
    if not os.path.exists(DDP_RESNET18_TRACE):
        pytest.skip("Trace not available")
    dst = tmp_path / "train_resnet18.json.gz"
    dst.write_bytes(open(DDP_RESNET18_TRACE, "rb").read())
    return dst


def _idle_annotation_events(trace_path):
    with gzip.open(trace_path, "rt") as f:
        events = json.load(f)["traceEvents"]
    return [e for e in events if e.get("cat") == "idle_classification"]


class TestStandaloneCli:
    def test_main_writes_excel_and_augmented_trace(self, ddp_trace_copy, tmp_path):
        import pandas as pd

        from TraceLens.Reporting.classify_idle_time import main

        out = tmp_path / "out_idle.json.gz"
        main([str(ddp_trace_copy), "-o", str(out)])

        assert _idle_annotation_events(out)
        sheets = pd.read_excel(tmp_path / "out_idle.xlsx", sheet_name=None)
        assert set(sheets) == {"idle_overview", "idle_summary", "idle_intervals"}
        assert not sheets["idle_intervals"].empty

    def test_default_output_path(self, ddp_trace_copy, tmp_path):
        from TraceLens.Reporting.classify_idle_time import main

        main([str(ddp_trace_copy)])

        assert (tmp_path / "train_resnet18_idle_classified.json.gz").exists()
        assert (tmp_path / "train_resnet18_idle_classified.xlsx").exists()


class TestPerfReportIntegration:
    def test_enable_idle_analysis_adds_sheets(self, ddp_trace_copy, tmp_path):
        from TraceLens.Reporting.generate_perf_report_pytorch import (
            generate_perf_report_pytorch,
        )

        dfs = generate_perf_report_pytorch(
            profile_json_path=str(ddp_trace_copy),
            output_csvs_dir=str(tmp_path / "csvs"),
            collective_analysis=False,
            enable_idle_analysis=True,
            enable_augmented_trace=True,
        )

        for name in ("idle_overview", "idle_summary", "idle_intervals"):
            assert name in dfs
            assert (tmp_path / "csvs" / f"{name}.csv").exists()
        assert "gpu_utilization_pct" in dfs["idle_overview"].columns
        assert _idle_annotation_events(
            tmp_path / "train_resnet18_idle_augmented.json.gz"
        )

    def test_idle_sheets_absent_by_default(self, ddp_trace_copy, tmp_path):
        from TraceLens.Reporting.generate_perf_report_pytorch import (
            generate_perf_report_pytorch,
        )

        dfs = generate_perf_report_pytorch(
            profile_json_path=str(ddp_trace_copy),
            output_csvs_dir=str(tmp_path / "csvs"),
            collective_analysis=False,
        )

        assert not any(name.startswith("idle_") for name in dfs)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
