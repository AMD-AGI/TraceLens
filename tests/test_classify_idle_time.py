#!/usr/bin/env python3
"""
Unit tests for classify_idle_time.py

Covers:
  - compute_self_times correctness (flat, nested, multi-level, disjoint)
  - classify_sync_event for each sync type
  - classify_runtime_event categories
  - extract_idle_intervals edge cases (no events, single event, overlapping)
  - classify_idle_intervals on synthetic mini-traces
  - assign_idle_ids sequencing
  - generate_excel_report smoke test
  - Edge cases: empty traces, single GPU event, zero-duration events
"""
import pytest
import tempfile

from TraceLens.Reporting.classify_idle_time import (
    compute_self_times,
    classify_sync_event,
    classify_runtime_event,
    extract_idle_intervals,
    assign_idle_ids,
    build_sorted_cpu_events,
    build_sorted_cpu_ops,
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
            {"ts": 0, "dur": 100, "t_end": 100, "name": "model.py: forward", "cat": "python_function"},
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
            {"ts": 0, "dur": 200, "t_end": 200, "name": "FlashAttnFunc", "cat": "cpu_op"},
            {"ts": 10, "dur": 180, "t_end": 190, "name": "flash_forward", "cat": "cpu_op"},
        ]
        result = compute_self_times(events, 50, 150)
        assert "flash_forward" in result
        assert result["flash_forward"] == pytest.approx(100.0)
        assert "FlashAttnFunc" not in result or result.get("FlashAttnFunc", 0) == 0

    def test_item_local_scalar_dense_pattern(self):
        """aten::item contains aten::_local_scalar_dense with same start."""
        events = [
            {"ts": 10, "dur": 1000, "t_end": 1010, "name": "aten::item", "cat": "cpu_op"},
            {"ts": 10, "dur": 998, "t_end": 1008, "name": "aten::_local_scalar_dense", "cat": "cpu_op"},
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
        assert records[1]["idle_id"] == 0   # macro
        assert records[2]["idle_id"] == 1   # macro
        assert records[3]["idle_id"] == -2  # noise
        assert records[4]["idle_id"] == 2   # macro

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
        events.append({
            "ph": "X", "cat": "kernel", "name": f"kernel_{start}",
            "ts": start, "dur": 50, "t_end": start + 50,
            "pid": 1, "tid": 7, "uid": uid, "UID": uid,
        })
        uid += 1

    # CPU runtime: a launch for each kernel
    for start, kernel_uid in [(90, 0), (190, 1), (490, 2)]:
        events.append({
            "ph": "X", "cat": "cuda_runtime", "name": "hipLaunchKernel",
            "ts": start, "dur": 5, "t_end": start + 5,
            "pid": 0, "tid": 0, "uid": uid, "UID": uid,
        })
        events[kernel_uid]["parent"] = uid
        uid += 1

    # CPU op spanning the gap between kernel 1 (ends at 250) and kernel 2 (starts at 500)
    events.append({
        "ph": "X", "cat": "cpu_op", "name": "aten::heavy_op",
        "ts": 260, "dur": 200, "t_end": 460,
        "pid": 0, "tid": 0, "uid": uid,
    })
    uid += 1

    events_by_uid = {e["uid"]: e for e in events}

    tree = SimpleNamespace(
        events=events,
        events_by_uid=events_by_uid,
    )
    return tree


class TestClassifyIdleIntervalsIntegration:
    def test_synthetic_trace(self):
        from TraceLens.Reporting.classify_idle_time import classify_idle_intervals

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
        from TraceLens.Reporting.classify_idle_time import classify_idle_intervals

        tree = SimpleNamespace(events=[], events_by_uid={})
        results = classify_idle_intervals(tree)
        assert results == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
