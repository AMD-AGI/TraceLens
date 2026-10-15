###############################################################################
# Copyright (c) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for TraceLens.IdleTimeAnalyser

Covers:
  - exclusive_split (nesting, clipping, same-start parents, annotations, covering op)
  - OverlapIndex, extract_idle_intervals, gpu_busy_time_us, small helpers
  - is_sync_call / classify_runtime_event
  - classify_idle_intervals scenarios: unqueued (CPU / runtime / untraced / launch),
    queued, slow start after launch, stream_switch, no_launch, other_thread,
    sync_drain causality (including blocking "async" copies), sync call stacks,
    thresholds
  - build_idle_dataframes: overview totals, label x part matrix, drain section,
    group_id links, folding
  - real traces (including the same model on AMD and NVIDIA), standalone CLI, and
    --enable_idle_analysis perf report integration
"""

import gzip
import json
import os
from types import SimpleNamespace

import pandas as pd
import pytest

from TraceLens.IdleTimeAnalyser import IdleThresholds, IdleTimeAnalyser
from TraceLens.IdleTimeAnalyser.classify import (
    PARTS,
    OverlapIndex,
    _new_record,
    assign_idle_ids,
    build_thread_index,
    call_stack_full,
    classify_idle_intervals,
    classify_runtime_event,
    exclusive_split,
    extract_idle_intervals,
    find_gpu_pid,
    find_launch_for_kernel,
    gpu_busy_time_us,
    is_enqueue_call,
    is_sync_call,
    strip_addresses,
)
from TraceLens.IdleTimeAnalyser.report import build_idle_dataframes

# ---------------------------------------------------------------------------
# Synthetic trees
# ---------------------------------------------------------------------------


def _build_tree(kernels, runtime=(), ops=()):
    """Build a mock tree.

    kernels: (ts, dur, launch_idx_or_None[, stream[, cat]]); launch_idx indexes runtime.
    runtime: (name, ts, dur[, tid[, parent_op_idx]]).
    ops:     (name, ts, dur[, cat[, tid]]); cat defaults to cpu_op.
    """
    events = []

    def add(evt):
        evt["UID"] = len(events)
        evt["t_end"] = evt["ts"] + evt["dur"]
        evt.setdefault("ph", "X")
        events.append(evt)
        return evt

    op_events = []
    for name, ts, dur, *rest in ops:
        cat = rest[0] if rest else "cpu_op"
        tid = rest[1] if len(rest) > 1 else 0
        op_events.append(
            add({"cat": cat, "name": name, "ts": ts, "dur": dur, "pid": 0, "tid": tid})
        )
    rt_events = []
    for name, ts, dur, *rest in runtime:
        tid = rest[0] if rest else 0
        evt = add(
            {
                "cat": "cuda_runtime",
                "name": name,
                "ts": ts,
                "dur": dur,
                "pid": 0,
                "tid": tid,
            }
        )
        if len(rest) > 1:
            evt["parent"] = op_events[rest[1]]["UID"]
        rt_events.append(evt)
    for i, (ts, dur, launch_idx, *rest) in enumerate(kernels):
        stream = rest[0] if rest else 7
        cat = rest[1] if len(rest) > 1 else "kernel"
        k = add(
            {
                "cat": cat,
                "name": f"k{i}",
                "ts": ts,
                "dur": dur,
                "pid": 1,
                "tid": stream,
                "args": {"stream": stream},
            }
        )
        if launch_idx is not None:
            k["parent"] = rt_events[launch_idx]["UID"]
    return SimpleNamespace(events=events, events_by_uid={e["UID"]: e for e in events})


def _make_synthetic_tree():
    """Three kernels; a 250 us gap covered by aten::heavy_op, a 50 us gap untraced."""
    return _build_tree(
        kernels=[(100, 50, 0), (200, 50, 1), (500, 50, 2)],
        runtime=[
            ("hipLaunchKernel", 90, 5),
            ("hipLaunchKernel", 190, 5),
            ("hipLaunchKernel", 490, 5),
        ],
        ops=[("aten::heavy_op", 260, 200)],
    )


def _macro(tree, thresholds=None):
    recs = [
        r for r in classify_idle_intervals(tree, thresholds) if not r["label_noise"]
    ]
    for r in recs:
        assert sum(r[p] for p in PARTS) == pytest.approx(r["duration"])
    return recs


def _thread_index(events):
    for i, e in enumerate(events):
        e.setdefault("UID", i)
        e.setdefault("ph", "X")
        e.setdefault("pid", 0)
        e.setdefault("tid", 0)
        e.setdefault("cat", "cpu_op")
        e["t_end"] = e["ts"] + e["dur"]
    return build_thread_index(SimpleNamespace(events=events))


# ---------------------------------------------------------------------------
# exclusive_split
# ---------------------------------------------------------------------------


class TestExclusiveSplit:
    def _split(self, events, w0, w1, exclude_uid=None):
        return exclusive_split(_thread_index(events), (0, 0), w0, w1, exclude_uid)

    def test_parent_child(self):
        parts, names, covering = self._split(
            [
                {"name": "parent", "ts": 10, "dur": 100},
                {"name": "child", "ts": 20, "dur": 60},
            ],
            0,
            120,
        )
        assert names[("cpu_op_us", "child")] == pytest.approx(60)
        assert names[("cpu_op_us", "parent")] == pytest.approx(40)
        assert parts["cpu_op_us"] == pytest.approx(100)
        assert parts["cpu_untraced_us"] == pytest.approx(20)
        assert covering == "child"

    def test_clipping_and_enclosing_event(self):
        parts, names, covering = self._split(
            [{"name": "wide_op", "ts": 0, "dur": 200}], 50, 150
        )
        assert parts == {
            "cpu_op_us": 100.0,
            "cpu_runtime_us": 0.0,
            "cpu_untraced_us": 0.0,
        }
        assert covering == "wide_op"

    def test_same_start_child_gets_credit(self):
        # aten::item wraps aten::_local_scalar_dense with the same start.
        _, names, covering = self._split(
            [
                {"name": "aten::item", "ts": 10, "dur": 1000},
                {"name": "aten::_local_scalar_dense", "ts": 10, "dur": 998},
            ],
            50,
            100,
        )
        assert names[("cpu_op_us", "aten::_local_scalar_dense")] == pytest.approx(50)
        assert covering == "aten::_local_scalar_dense"

    def test_return_event(self):
        events = [
            {"name": "parent", "ts": 10, "dur": 100},
            {"name": "child", "ts": 20, "dur": 60},
        ]
        idx = _thread_index(events)
        _, _, covering = exclusive_split(idx, (0, 0), 0, 120, return_event=True)
        assert covering is events[1]
        _, _, covering = exclusive_split(idx, (0, 0), 200, 300, return_event=True)
        assert covering is None

    def test_identical_bounds_counted_once(self):
        parts, _, _ = self._split(
            [
                {"name": "op_a", "ts": 10, "dur": 100},
                {"name": "op_b", "ts": 10, "dur": 100},
            ],
            20,
            80,
        )
        assert parts["cpu_op_us"] == pytest.approx(60)

    def test_runtime_and_annotation_parts(self):
        parts, names, covering = self._split(
            [
                {
                    "name": "ProfilerStep#3",
                    "ts": 0,
                    "dur": 100,
                    "cat": "user_annotation",
                },
                {"name": "aten::copy_", "ts": 10, "dur": 30},
                {"name": "hipMalloc", "ts": 20, "dur": 10, "cat": "cuda_runtime"},
            ],
            0,
            100,
        )
        assert parts["cpu_runtime_us"] == pytest.approx(10)
        assert parts["cpu_op_us"] == pytest.approx(20)
        assert parts["cpu_untraced_us"] == pytest.approx(70)
        assert names[("cpu_runtime_us", "hipMalloc")] == pytest.approx(10)
        # Annotations and runtime calls never cover; no op spans half, so the largest.
        assert covering == "aten::copy_"

    def test_exclude_uid(self):
        parts, _, _ = self._split(
            [{"name": "hipLaunchKernel", "ts": 0, "dur": 50, "cat": "cuda_runtime"}],
            0,
            50,
            exclude_uid=0,
        )
        assert parts["cpu_untraced_us"] == pytest.approx(50)

    def test_empty_window_and_unknown_thread(self):
        idx = _thread_index([{"name": "op", "ts": 0, "dur": 10}])
        parts, _, _ = exclusive_split(idx, (0, 0), 5, 5)
        assert sum(parts.values()) == 0
        parts, _, _ = exclusive_split(idx, (9, 9), 0, 10)
        assert parts["cpu_untraced_us"] == pytest.approx(10)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


class TestOverlapIndex:
    def test_basic_overlap(self):
        events = [
            {"ts": 0, "t_end": 10},
            {"ts": 5, "t_end": 15},
            {"ts": 20, "t_end": 30},
        ]
        assert len(OverlapIndex(events).query(8, 22)) == 3

    def test_no_overlap(self):
        events = [{"ts": 0, "t_end": 10}, {"ts": 20, "t_end": 30}]
        assert OverlapIndex(events).query(12, 18) == []

    def test_event_spanning_interval(self):
        assert len(OverlapIndex([{"ts": 0, "t_end": 100}]).query(20, 30)) == 1

    def test_empty(self):
        assert OverlapIndex([]).query(0, 100) == []


class TestExtractIdleIntervals:
    def test_gap_and_busy_time(self):
        events = [
            {"ts": 0, "t_end": 15},
            {"ts": 10, "t_end": 25},
            {"ts": 30, "t_end": 40},
        ]
        assert extract_idle_intervals(events) == [(25, 30)]
        assert gpu_busy_time_us(events) == pytest.approx(35)

    def test_adjacent_and_empty(self):
        assert (
            extract_idle_intervals([{"ts": 0, "t_end": 10}, {"ts": 10, "t_end": 20}])
            == []
        )
        assert extract_idle_intervals([]) == []


class TestHelpers:
    def test_is_enqueue_call(self):
        assert is_enqueue_call("hipLaunchKernel")
        assert is_enqueue_call("hipExtLaunchKernel")
        assert is_enqueue_call("hipMemcpyAsync")
        assert not is_enqueue_call("hipMemcpyWithStream")
        assert not is_enqueue_call("hipMalloc")

    def test_strip_addresses(self):
        assert (
            strip_addresses("<built-in method foo of Bar object at 0x7f06c2a1b3d0>")
            == "<built-in method foo of Bar object>"
        )
        assert strip_addresses(None) is None

    def test_find_launch_for_kernel_non_runtime_parent(self):
        tree = SimpleNamespace(events_by_uid={1: {"cat": "cpu_op"}})
        assert find_launch_for_kernel(tree, {}) is None
        assert find_launch_for_kernel(tree, {"parent": 99}) is None
        assert find_launch_for_kernel(tree, {"parent": 1}) is None

    def test_find_gpu_pid_falls_back_to_process_labels(self):
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

    def test_assign_idle_ids(self):
        records = [{"label_noise": n} for n in (True, False, False, True, False)]
        assign_idle_ids(records)
        assert [r["idle_id"] for r in records] == [-1, 0, 1, -2, 2]


class TestIsSyncCall:
    @pytest.mark.parametrize(
        "name,expected",
        [
            ("hipDeviceSynchronize", True),
            ("cudaDeviceSynchronize", True),
            ("hipStreamSynchronize", True),
            ("cudaEventSynchronize", True),
            ("hipFree", True),
            ("hipHostFree", True),
            ("cudaFreeHost", True),
            ("hipMemcpyWithStream", True),
            ("cudaMemcpyAsync", True),
            ("hipLaunchKernel", False),
            ("hipMalloc", False),
            ("hipStreamWaitEvent", False),
        ],
    )
    def test_names(self, name, expected):
        assert is_sync_call(name) is expected


class TestClassifyRuntimeEvent:
    def test_categories(self):
        assert classify_runtime_event({"name": "hipMalloc"}) == "MEMORY_ALLOC"
        assert classify_runtime_event({"name": "cudaFree"}) == "MEMORY_ALLOC"
        assert classify_runtime_event({"name": "hipLaunchKernel"}) == "LAUNCH_STALL"
        assert classify_runtime_event({"name": "hipDeviceSynchronize"}) == "SYNC_CALL"
        assert classify_runtime_event({"name": "hipSomethingElse"}) == "OTHER_RUNTIME"


# ---------------------------------------------------------------------------
# classify_idle_intervals scenarios
# ---------------------------------------------------------------------------


class TestUnqueued:
    def test_cpu_op_dominated(self):
        # gap [10, 200); aten::index covers 150 us; launch 180-185.
        tree = _build_tree(
            kernels=[(0, 10, None), (200, 10, 0)],
            runtime=[("hipLaunchKernel", 180, 5)],
            ops=[("aten::index", 20, 150)],
        )
        (rec,) = _macro(tree)
        assert rec["case"] == "unqueued"
        assert rec["label"] == "CPU_OP_DOMINATED"
        assert rec["cpu_op_us"] == pytest.approx(150)
        assert rec["cpu_untraced_us"] == pytest.approx(20)
        assert rec["launch_call_us"] == pytest.approx(5)
        assert rec["submit_latency_us"] == pytest.approx(15)
        assert rec["covering_op"] == "aten::index"
        assert rec["top_op"] == "aten::index"

    def test_runtime_dominated(self):
        tree = _build_tree(
            kernels=[(0, 10, None), (200, 10, 1)],
            runtime=[("hipMalloc", 20, 150), ("hipLaunchKernel", 180, 5)],
        )
        (rec,) = _macro(tree)
        assert rec["label"] == "RUNTIME_DOMINATED"
        assert rec["sub_label"] is None
        assert rec["top_runtime"] == "hipMalloc"
        assert rec["cpu_runtime_us"] == pytest.approx(150)
        assert rec["covering_op"] is None

    def test_blocking_copy_counts_as_runtime(self):
        # The next GPU op is a copy whose call (hipMemcpyWithStream) blocks for 175 us.
        tree = _build_tree(
            kernels=[(0, 10, None), (200, 10, 0, 7, "gpu_memcpy")],
            runtime=[("hipMemcpyWithStream", 20, 175)],
        )
        (rec,) = _macro(tree)
        assert rec["label"] == "RUNTIME_DOMINATED"
        assert rec["top_runtime"] == "hipMemcpyWithStream"
        assert rec["launch_call_us"] == 0
        assert rec["cpu_runtime_us"] == pytest.approx(175)

    def test_untraced_dominated(self):
        tree = _build_tree(
            kernels=[(0, 10, None), (210, 10, 0)],
            runtime=[("hipLaunchKernel", 190, 5)],
        )
        (rec,) = _macro(tree)
        assert rec["label"] == "UNTRACED_DOMINATED"
        assert rec["cpu_untraced_us"] == pytest.approx(180)
        assert rec["covering_op"] is None

    def test_annotation_time_is_untraced(self):
        tree = _build_tree(
            kernels=[(0, 10, None), (210, 10, 0)],
            runtime=[("hipLaunchKernel", 190, 5)],
            ops=[("ProfilerStep#1", 0, 300, "user_annotation")],
        )
        (rec,) = _macro(tree)
        assert rec["label"] == "UNTRACED_DOMINATED"
        assert rec["covering_op"] is None

    def test_launch_dominated_covered_by_parent_op(self):
        # 20 us gap: launch call 12 us inside aten::addmm, then 8 us to start.
        tree = _build_tree(
            kernels=[(0, 10, None), (30, 10, 0)],
            runtime=[("hipLaunchKernel", 10, 12, 0, 0)],
            ops=[("aten::addmm", 5, 20)],
        )
        (rec,) = _macro(tree)
        assert rec["label"] == "LAUNCH_DOMINATED"
        assert rec["launch_call_us"] == pytest.approx(12)
        assert rec["submit_latency_us"] == pytest.approx(8)
        assert rec["launch_parent_op"] == "aten::addmm"
        assert rec["covering_op"] == "aten::addmm"

    def test_covering_op_falls_back_to_largest_op(self):
        # gap [10, 210): no op spans half of it; aten::mul (50 us) is the largest.
        tree = _build_tree(
            kernels=[(0, 10, None), (210, 10, 0)],
            runtime=[("hipLaunchKernel", 190, 5)],
            ops=[("aten::view", 20, 20), ("aten::mul", 100, 50)],
        )
        (rec,) = _macro(tree)
        assert rec["label"] == "UNTRACED_DOMINATED"
        assert rec["covering_op"] == "aten::mul"

    def test_covering_op_spans_whole_gap_when_queued(self):
        # Launched at 20-25, long before the gap [100, 200); aten::copy_ covers the gap.
        tree = _build_tree(
            kernels=[(0, 100, None), (200, 10, 0)],
            runtime=[("hipLaunchKernel", 20, 5)],
            ops=[("aten::copy_", 90, 120)],
        )
        (rec,) = _macro(tree)
        assert rec["case"] == "queued"
        assert rec["covering_op"] == "aten::copy_"

    def test_launch_returned_just_before_gap_is_unqueued(self):
        # Launch returned 5 us before the gap (< 15 us launch latency): the whole gap
        # is submit latency, not unexplained GPU wait.
        tree = _build_tree(
            kernels=[(0, 100, None), (110, 10, 0)],
            runtime=[("hipLaunchKernel", 90, 5)],
        )
        (rec,) = _macro(tree)
        assert rec["case"] == "unqueued"
        assert rec["label"] == "LAUNCH_DOMINATED"
        assert rec["submit_latency_us"] == pytest.approx(10)
        assert rec["launch_lead_us"] == pytest.approx(5)

    def test_slow_start_after_launch(self):
        # Launch returned at 45 but the kernel only started at 100.
        tree = _build_tree(
            kernels=[(0, 10, None), (100, 10, 0)],
            runtime=[("hipLaunchKernel", 40, 5)],
        )
        (rec,) = _macro(tree)
        assert rec["case"] == "unqueued"
        assert (rec["label"], rec["sub_label"]) == (
            "UNEXPLAINED_GPU_WAIT",
            "unqueued",
        )
        assert rec["launch_to_start_us"] == pytest.approx(55)

    def test_launch_call_x_ref(self):
        # A 500 us graph launch is 10x the 50 us reference, a 5 us kernel launch 0.1x;
        # the label still comes from the parts. Copies can block, so they get none.
        tree = _build_tree(
            kernels=[(0, 10, None), (600, 10, 0), (800, 10, 1), (1000, 10, 2)],
            runtime=[
                ("hipGraphLaunch", 90, 500),
                ("hipLaunchKernel", 700, 5),
                ("hipMemcpyWithStream", 900, 80),
            ],
        )
        graph, kernel, copy = _macro(tree)
        assert graph["launch_call_x_ref"] == pytest.approx(10)
        assert graph["label"] == "LAUNCH_DOMINATED"
        assert kernel["launch_call_x_ref"] == pytest.approx(0.1)
        assert copy["launch_call_x_ref"] is None
        graph, _, _ = _macro(tree, IdleThresholds(launch_call_ref_us=100))
        assert graph["launch_call_x_ref"] == pytest.approx(5)

    def test_other_thread_runtime(self):
        # Launching thread shows nothing; another thread is in hipFree for 150 of 200 us.
        tree = _build_tree(
            kernels=[(0, 10, None), (210, 10, 0)],
            runtime=[("hipLaunchKernel", 190, 5), ("hipFree", 20, 150, 1)],
        )
        (rec,) = _macro(tree)
        assert (rec["label"], rec["sub_label"]) == ("RUNTIME_DOMINATED", "other_thread")
        assert rec["other_thread_top_runtime"] == "hipFree"
        assert rec["cpu_untraced_us"] == pytest.approx(180)


class TestQueuedAndNoLaunch:
    def test_queued(self):
        tree = _build_tree(
            kernels=[(0, 100, None), (150, 10, 0)],
            runtime=[("hipLaunchKernel", 20, 5)],
        )
        (rec,) = _macro(tree)
        assert rec["case"] == "queued"
        assert (rec["label"], rec["sub_label"]) == ("UNEXPLAINED_GPU_WAIT", "queued")
        assert rec["gpu_wait_us"] == pytest.approx(50)
        assert rec["launch_lead_us"] == pytest.approx(75)
        assert rec["stream_switch"] is False

    @pytest.mark.parametrize("runtime_ts", [20, 140])
    def test_stream_switch_independent_of_case(self, runtime_ts):
        tree = _build_tree(
            kernels=[(0, 100, None), (150, 10, 0, 8)],
            runtime=[("hipLaunchKernel", runtime_ts, 5)],
        )
        (rec,) = _macro(tree)
        assert rec["case"] == ("queued" if runtime_ts == 20 else "unqueued")
        assert rec["stream_switch"] is True

    def test_no_launch(self):
        tree = _build_tree(kernels=[(0, 10, None), (100, 10, None)])
        (rec,) = _macro(tree)
        assert rec["case"] == "no_launch"
        assert rec["label"] == "UNATTRIBUTED"
        assert rec["unattributed_us"] == pytest.approx(90)
        assert rec["drain_type"] == "starved"

    def test_empty_tree(self):
        assert (
            classify_idle_intervals(SimpleNamespace(events=[], events_by_uid={})) == []
        )


class TestDrain:
    def _tree(
        self, sync_name="hipDeviceSynchronize", sync_ts=50, sync_tid=0, launch_ts=110
    ):
        # k0 runs 0-100; the sync returns at 105; next launch at launch_ts.
        return _build_tree(
            kernels=[(0, 100, None), (130, 10, 1)],
            runtime=[
                (sync_name, sync_ts, 105 - sync_ts, sync_tid),
                ("hipLaunchKernel", launch_ts, 5),
            ],
        )

    def test_sync_drain(self):
        (rec,) = _macro(self._tree())
        assert rec["drain_type"] == "sync_drain"
        assert rec["sync_event_name"] == "hipDeviceSynchronize"

    def test_sync_gain(self):
        # 55 us sync, 30 us gap: removing the sync saves at most the gap.
        (rec,) = _macro(self._tree())
        assert rec["sync_gain_us"] == pytest.approx(30)
        # 25 us sync (80-105), 200 us gap: at most the sync; the rest is CPU work after.
        tree = _build_tree(
            kernels=[(0, 100, None), (300, 10, 1)],
            runtime=[("hipDeviceSynchronize", 80, 25), ("hipLaunchKernel", 290, 5)],
        )
        (rec,) = _macro(tree)
        assert rec["drain_type"] == "sync_drain"
        assert rec["sync_gain_us"] == pytest.approx(25)
        (rec,) = _macro(self._tree(sync_tid=1))
        assert rec["sync_gain_us"] is None

    @pytest.mark.parametrize("name", ["hipFree", "hipHostFree", "cudaFreeHost"])
    def test_implicit_syncs(self, name):
        (rec,) = _macro(self._tree(sync_name=name))
        assert rec["drain_type"] == "sync_drain"
        assert rec["sync_event_name"] == name

    def test_blocking_async_copy_without_kind(self):
        # A CUDA cudaMemcpyAsync to pageable memory blocks; its runtime event has no
        # copy direction, and none is needed.
        (rec,) = _macro(self._tree(sync_name="cudaMemcpyAsync"))
        assert rec["drain_type"] == "sync_drain"
        assert rec["sync_event_name"] == "cudaMemcpyAsync"

    def test_sync_after_launch_is_not_causal(self):
        (rec,) = _macro(self._tree(launch_ts=60))
        assert rec["drain_type"] == "starved"
        assert rec["sync_event_name"] is None

    def test_sync_on_other_thread_ignored(self):
        (rec,) = _macro(self._tree(sync_tid=1))
        assert rec["drain_type"] == "starved"
        assert rec["sync_event_name"] is None

    def test_call_stack_only_when_requested(self):
        tree = _build_tree(
            kernels=[(0, 100, None), (130, 10, 1)],
            runtime=[
                ("hipMemcpyWithStream", 50, 55, 0, 1),
                ("hipLaunchKernel", 110, 5),
            ],
            ops=[("aten::item", 40, 70), ("aten::_local_scalar_dense", 45, 62)],
        )
        tree.events[1]["parent"] = tree.events[0]["UID"]
        (rec,) = _macro(tree)
        assert rec["sync_call_stack_full"] is None
        assert rec["sync_entry_point"] is None
        (rec,) = [
            r
            for r in classify_idle_intervals(tree, include_call_stack=True)
            if not r["label_noise"]
        ]
        assert rec["sync_call_stack_full"] == str(
            ["aten::item", "aten::_local_scalar_dense", "hipMemcpyWithStream"]
        )
        assert rec["sync_entry_point"] == "Not found"

    def test_covering_call_stack_only_when_requested(self):
        # aten::heavy (inside a module) covers the gap [100, 300).
        tree = _build_tree(
            kernels=[(0, 100, None), (300, 10, 0)],
            runtime=[("hipLaunchKernel", 292, 5)],
            ops=[("nn.Module: Block_0", 0, 300), ("aten::heavy", 105, 185)],
        )
        tree.events[1]["parent"] = tree.events[0]["UID"]
        (rec,) = _macro(tree)
        assert rec["covering_op"] == "aten::heavy"
        assert rec["covering_call_stack_full"] is None
        (rec,) = [
            r
            for r in classify_idle_intervals(tree, include_call_stack=True)
            if not r["label_noise"]
        ]
        assert rec["covering_call_stack_full"] == str(
            ["nn.Module: Block", "aten::heavy"]
        )

    def test_call_stack_strips_numeric_suffixes(self):
        tree = SimpleNamespace(
            events_by_uid={0: {"name": "nn.Module: Linear_0"}},
        )
        evt = {"name": "hipFree", "parent": 0}
        assert call_stack_full(tree, evt) == str(["nn.Module: Linear", "hipFree"])

    def test_sync_started_as_gpu_finished_is_not_drain(self):
        (rec,) = _macro(self._tree(sync_ts=97))
        assert rec["drain_type"] == "starved"

    def test_preceding_found_by_end_time(self):
        # k0 (0-100) outlasts k1 (50-60, other stream): the gap starts when k0 ends.
        tree = _build_tree(
            kernels=[(0, 100, None), (50, 10, None, 8), (130, 10, 0)],
            runtime=[("hipLaunchKernel", 110, 5)],
        )
        (rec,) = _macro(tree)
        assert rec["preceding_gpu_event"] == "k0"


class TestThresholds:
    def test_noise(self):
        tree = _build_tree(kernels=[(0, 10, None), (100, 10, None)])
        (rec,) = classify_idle_intervals(tree, IdleThresholds(noise_us=100))
        assert rec["label_noise"] is True
        assert rec["label"] is None

    def test_launch_latency_moves_queued_to_unqueued(self):
        tree = _build_tree(
            kernels=[(0, 100, None), (150, 10, 0)],
            runtime=[("hipLaunchKernel", 20, 5)],
        )
        (rec,) = _macro(tree, IdleThresholds(launch_latency_us=100))
        assert rec["case"] == "unqueued"
        assert rec["sub_label"] == "unqueued"

    def test_launch_latency_rel(self):
        # launch -> start 55 us; with rel=1 the limit becomes the 55 us median: not slow.
        tree = _build_tree(
            kernels=[(0, 10, None), (100, 10, 0)],
            runtime=[("hipLaunchKernel", 40, 5)],
        )
        (rec,) = _macro(tree, IdleThresholds(launch_latency_rel=1.0))
        assert rec["label"] == "LAUNCH_DOMINATED"

    def test_micro_thresh_override(self):
        an = IdleTimeAnalyser(_make_synthetic_tree(), micro_thresh_us=100)
        assert an.thresholds.noise_us == 100
        assert sum(not r["label_noise"] for r in an.classify()) == 1


# ---------------------------------------------------------------------------
# IdleTimeAnalyser and dataframes
# ---------------------------------------------------------------------------


class TestIdleTimeAnalyser:
    def test_classify_caches_results(self):
        an = IdleTimeAnalyser(_make_synthetic_tree())
        assert an.classify() is an.classify()

    def test_get_dataframes(self):
        dfs = IdleTimeAnalyser(_make_synthetic_tree()).get_dataframes()
        assert set(dfs) == {"idle_overview", "idle_groups", "idle_intervals"}
        ov = dfs["idle_overview"].set_index("row")
        # busy 150 us, idle 50 + 250 us
        assert ov.loc["gpu_utilization_pct", "pct_of_trace"] == pytest.approx(
            100 * 150 / 450
        )
        assert ov.loc["ALL idle", "total_ms"] == pytest.approx(0.3)

    def test_get_augmented_events(self):
        events = IdleTimeAnalyser(_make_synthetic_tree()).get_augmented_events(
            gpu_pid=1
        )
        assert events and all(e["pid"] == 1 for e in events)
        labels = [e for e in events if e.get("tid") == 9002 and e["ph"] == "X"]
        assert labels and all("case" in e["args"] for e in labels)


def _rec(duration, label="CPU_OP_DOMINATED", key="op", start=0.0, noise=False):
    rec = _new_record(start, start + duration)
    rec.update(label_noise=noise, label=label, case="unqueued", drain_type="starved")
    rec["covering_op"] = key
    rec["cpu_op_us"] = duration
    return rec


class TestBuildIdleDataframes:
    def test_totals_include_noise(self):
        recs = [_rec(100), _rec(3, noise=True)]
        assign_idle_ids(recs)
        dfs = build_idle_dataframes(recs, gpu_busy_time_us=897)
        ov = dfs["idle_overview"].set_index("row")
        assert ov.loc["ALL idle", "total_ms"] == pytest.approx(0.103)
        assert ov.loc["noise (< 5 us)", "gaps"] == 1
        assert ov.loc["gpu_utilization_pct", "pct_of_trace"] == pytest.approx(89.7)
        assert ov.loc["CPU_OP_DOMINATED", "starved_ms"] == pytest.approx(0.1)
        assert "launch_latency_us = 15.0" in ov.index

    def test_overview_matrix_and_drain_section(self):
        a, b = _rec(100, start=0), _rec(60, label="LAUNCH_DOMINATED", start=200)
        a.update(cpu_op_us=70, launch_call_us=30)
        b.update(
            cpu_op_us=0,
            launch_call_us=60,
            drain_type="sync_drain",
            sync_event_name="hipMemcpyWithStream",
            stream_switch=True,
        )
        recs = [a, b]
        assign_idle_ids(recs)
        ov = build_idle_dataframes(recs)["idle_overview"].set_index("row")
        assert set(ov["section"]) == {"total", "label", "drain", "threshold"}
        assert ov.loc["CPU_OP_DOMINATED", "cpu_op_ms"] == pytest.approx(0.07)
        assert ov.loc["CPU_OP_DOMINATED", "launch_call_ms"] == pytest.approx(0.03)
        assert ov.loc["macro (>= 5 us)", "launch_call_ms"] == pytest.approx(0.09)
        assert ov.loc["macro (>= 5 us)", "stream_switch_ms"] == pytest.approx(0.06)
        assert ov.loc["CPU_OP_DOMINATED", "stream_switch_ms"] == 0
        drain = ov[ov["section"] == "drain"]
        assert list(drain.index) == ["starved", "sync_drain: hipMemcpyWithStream"]
        assert drain["total_ms"].tolist() == pytest.approx([0.1, 0.06])
        assert drain.loc["sync_drain: hipMemcpyWithStream", "launch_call_ms"] == (
            pytest.approx(0.06)
        )

    def test_call_stack_column_only_when_requested(self):
        recs = [_rec(100)]
        recs[0]["sync_call_stack_full"] = str(["aten::item", "hipMemcpyWithStream"])
        assign_idle_ids(recs)
        assert (
            "sync_call_stack_full" not in build_idle_dataframes(recs)["idle_intervals"]
        )
        intervals = build_idle_dataframes(recs, include_call_stack=True)[
            "idle_intervals"
        ]
        assert (
            intervals.loc[0, "sync_call_stack_full"] == recs[0]["sync_call_stack_full"]
        )
        cols = list(intervals.columns)
        assert cols.index("sync_entry_point") + 1 == cols.index("sync_call_stack_full")

    def test_covering_call_stack_column_only_when_requested(self):
        recs = [_rec(100)]
        recs[0]["covering_call_stack_full"] = str(["nn.Module: Block", "op"])
        assign_idle_ids(recs)
        assert (
            "covering_call_stack_full"
            not in build_idle_dataframes(recs)["idle_intervals"]
        )
        intervals = build_idle_dataframes(recs, include_call_stack=True)[
            "idle_intervals"
        ]
        assert intervals.loc[0, "covering_call_stack_full"] == str(
            ["nn.Module: Block", "op"]
        )

    def test_sync_gain_on_drain_rows(self):
        a, b, c = _rec(100, start=0), _rec(60, start=200), _rec(40, start=400)
        for r, gain in ((a, 2.0), (b, 60.0)):
            r.update(
                drain_type="sync_drain",
                sync_event_name="hipMemcpyWithStream",
                sync_gain_us=gain,
            )
        recs = [a, b, c]
        assign_idle_ids(recs)
        dfs = build_idle_dataframes(recs)
        ov = dfs["idle_overview"].set_index("row")
        row = "sync_drain: hipMemcpyWithStream"
        assert ov.loc[row, "sync_gain_ms"] == pytest.approx(0.062)
        assert ov.loc[row, "sync_drain_ms"] == pytest.approx(0.16)
        assert pd.isna(ov.loc["starved", "sync_gain_ms"])
        assert pd.isna(ov.loc["CPU_OP_DOMINATED", "sync_gain_ms"])
        intervals = dfs["idle_intervals"].set_index("idle_id")
        assert intervals.loc[a["idle_id"], "sync_gain_us"] == pytest.approx(2.0)

    def test_drain_rows_split_by_call_site(self):
        recs = [_rec(100, start=0), _rec(60, start=200), _rec(40, start=400)]
        for r, site in zip(recs, ["model.py(10): step", "model.py(20): log", None]):
            r.update(
                drain_type="sync_drain",
                sync_event_name="hipMemcpyWithStream",
                sync_entry_point=site,
            )
        assign_idle_ids(recs)
        ov = build_idle_dataframes(recs, include_call_stack=True)["idle_overview"]
        assert list(ov.loc[ov["section"] == "drain", "row"]) == [
            "sync_drain: hipMemcpyWithStream @ model.py(10): step",
            "sync_drain: hipMemcpyWithStream @ model.py(20): log",
            "sync_drain: hipMemcpyWithStream",
        ]

    def test_group_id_links_intervals(self):
        recs = [
            _rec(100, key="a", start=0),
            _rec(50, key="b", start=200),
            _rec(30, key="a", start=400),
        ]
        assign_idle_ids(recs)
        dfs = build_idle_dataframes(recs)
        groups = dfs["idle_groups"].set_index("group_id")
        intervals = dfs["idle_intervals"]
        for gid, n in intervals.groupby("group_id").size().items():
            assert groups.loc[gid, "gaps"] == n
        assert groups.loc[0, "covering_op"] == "a"
        assert groups.loc[0, "total_ms"] == pytest.approx(0.13)
        assert groups.loc[0, "cpu_op_ms"] == pytest.approx(0.13)

    def test_group_columns_same_rule_for_every_label(self):
        a, b = _rec(100, start=0), _rec(80, label="RUNTIME_DOMINATED", start=200)
        b.update(top_runtime="hipFree", cpu_runtime_us=80, cpu_op_us=0)
        c = _rec(50, label="UNATTRIBUTED", key=None, start=400)
        c.update(case="no_launch", cpu_op_us=0, unattributed_us=50)
        for r, k in zip((a, b, c), ("k1", "k2", "k2")):
            r["following_gpu_event"] = k
        recs = [a, b, c]
        assign_idle_ids(recs)
        groups = build_idle_dataframes(recs)["idle_groups"].set_index("label")
        assert groups.loc["CPU_OP_DOMINATED", "covering_op"] == "op"
        assert groups.loc["RUNTIME_DOMINATED", "covering_op"] == "op"
        assert groups.loc["RUNTIME_DOMINATED", "top_runtime"] == "hipFree"
        assert groups.loc["RUNTIME_DOMINATED", "top_next_gpu_op"] == "k2"
        assert groups.loc["UNATTRIBUTED", "covering_op"] == "(no launch)"
        assert not isinstance(groups.loc["CPU_OP_DOMINATED", "top_runtime"], str)

    def test_top_runtime_blank_under_ten_pct(self):
        # 9 us of an attribute query in a 100 us CPU-op group is incidental; 10 us is
        # kept. An other_thread call is judged by the other thread's time.
        a = _rec(100, key="small", start=0)
        a.update(top_runtime="hipGetDevicePropertiesR0600", cpu_runtime_us=9)
        b = _rec(100, key="edge", start=200)
        b.update(top_runtime="hipMalloc", cpu_runtime_us=10)
        c = _rec(100, label="RUNTIME_DOMINATED", key="ot", start=400)
        c.update(
            sub_label="other_thread",
            top_runtime="hipEventQuery",
            cpu_runtime_us=1,
            other_thread_top_runtime="hipFree",
            other_thread_runtime_us=60,
        )
        recs = [a, b, c]
        assign_idle_ids(recs)
        groups = build_idle_dataframes(recs)["idle_groups"].set_index("covering_op")
        assert not isinstance(groups.loc["small", "top_runtime"], str)
        assert groups.loc["edge", "top_runtime"] == "hipMalloc"
        assert groups.loc["ot", "top_runtime"] == "hipFree"

    def test_top_launch(self):
        recs = [_rec(100, start=0), _rec(30, start=200), _rec(50, start=400)]
        for r, name in zip(
            recs, ("hipGraphLaunch", "hipLaunchKernel", "hipLaunchKernel")
        ):
            r["following_launch_name"] = name
        assign_idle_ids(recs)
        groups = build_idle_dataframes(recs)["idle_groups"]
        assert groups.loc[0, "top_launch"] == "hipGraphLaunch"
        cols = list(groups.columns)
        assert cols.index("top_runtime") + 1 == cols.index("top_launch")

    def test_small_groups_fold_per_label(self):
        recs = [
            _rec(d, key=f"op{i}", start=i * 2000)
            for i, d in enumerate([1000, 500, 300] + [5] * 7)
        ]
        assign_idle_ids(recs)
        groups = build_idle_dataframes(recs)["idle_groups"]
        assert len(groups) == 4
        other = groups.iloc[-1]
        assert other["covering_op"] == "other (7 groups)"
        assert other["gaps"] == 7
        assert other["folded_top_ops"].startswith("op3")

    def test_fold_keeps_drain_types_apart(self):
        recs = [
            _rec(d, key=f"op{i}", start=i * 2000)
            for i, d in enumerate([1000, 500, 300] + [5] * 6)
        ]
        for r in recs[3:6]:
            r["drain_type"] = "sync_drain"
        assign_idle_ids(recs)
        groups = build_idle_dataframes(recs)["idle_groups"]
        other = groups[groups["covering_op"] == "other (3 groups)"]
        assert sorted(other["drain_type"]) == ["starved", "sync_drain"]

    def test_empty(self):
        dfs = build_idle_dataframes([])
        assert all(df.empty for df in dfs.values())


# ---------------------------------------------------------------------------
# Real traces, CLI, perf report
# ---------------------------------------------------------------------------

TRACE_DIR = os.path.join(os.path.dirname(__file__), "traces")
DDP_RESNET18_TRACE = os.path.join(
    TRACE_DIR, "mi300", "ddp_resnet18_no_pool", "train_resnet18.json.gz"
)
FSDP_TRACE = os.path.join(
    TRACE_DIR, "mi300", "llama_70b_fsdp", "rank0_trace_no_pyfn.json.gz"
)
OWLV2 = "google_owlv2-large-patch14-ensemble__1016001.json.gz"


def _classify_trace(path):
    from TraceLens import TreePerfAnalyzer

    if not os.path.exists(path):
        pytest.skip("Trace not available")
    return IdleTimeAnalyser(TreePerfAnalyzer.from_file(path).tree).classify()


class TestRealTraces:
    def test_ddp_resnet18(self):
        """No memory pool: hipFree/hipMalloc dominate, partly from another thread."""
        macro = [r for r in _classify_trace(DDP_RESNET18_TRACE) if not r["label_noise"]]
        for r in macro:
            assert sum(r[p] for p in PARTS) == pytest.approx(r["duration"], abs=1e-6)
        runtime = [r for r in macro if r["label"] == "RUNTIME_DOMINATED"]
        tops = {r["top_runtime"] for r in runtime} | {
            r["other_thread_top_runtime"] for r in runtime
        }
        assert tops & {"hipFree", "hipMalloc"}
        assert any(r["sub_label"] == "other_thread" for r in runtime)

    def test_fsdp_rank0(self):
        macro = [r for r in _classify_trace(FSDP_TRACE) if not r["label_noise"]]
        for r in macro:
            assert sum(r[p] for p in PARTS) == pytest.approx(r["duration"], abs=1e-6)
        labels = {r["label"] for r in macro}
        assert {
            "CPU_OP_DOMINATED",
            "UNTRACED_DOMINATED",
            "LAUNCH_DOMINATED",
            "UNEXPLAINED_GPU_WAIT",
        } <= labels
        assert {r["case"] for r in macro} >= {"unqueued", "queued"}

    @pytest.mark.parametrize(
        "platform,copy_call",
        [("mi300", "hipMemcpyWithStream"), ("h100", "cudaMemcpyAsync")],
    )
    def test_owlv2_copy_drains_queue_on_both_vendors(self, platform, copy_call):
        """Same model on AMD and NVIDIA: host copies drain the queue before ~0.1 s of
        CPU work. On CUDA the blocking call is cudaMemcpyAsync to pageable memory."""
        macro = [
            r
            for r in _classify_trace(os.path.join(TRACE_DIR, platform, OWLV2))
            if not r["label_noise"]
        ]
        total = sum(r["duration"] for r in macro)
        drained = [r for r in macro if r["drain_type"] == "sync_drain"]
        assert sum(r["duration"] for r in drained) > 0.9 * total
        assert copy_call in {r["sync_event_name"] for r in drained}


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
        main(
            [
                str(ddp_trace_copy),
                "-o",
                str(out),
                "--launch-latency",
                "20",
                "--include-call-stack",
            ]
        )

        assert _idle_annotation_events(out)
        sheets = pd.read_excel(tmp_path / "out_idle.xlsx", sheet_name=None)
        assert set(sheets) == {"idle_overview", "idle_groups", "idle_intervals"}
        intervals = sheets["idle_intervals"]
        assert not intervals.empty
        assert "launch_latency_us = 20.0" in set(sheets["idle_overview"]["row"])
        cols = list(intervals.columns)
        assert cols.index("sync_entry_point") + 1 == cols.index("sync_call_stack_full")
        hipfree = intervals[intervals["sync_event_name"] == "hipFree"]
        assert hipfree["sync_call_stack_full"].str.endswith("'hipFree']").all()

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

        for name in ("idle_overview", "idle_groups", "idle_intervals"):
            assert name in dfs
            assert (tmp_path / "csvs" / f"{name}.csv").exists()
        ov = dfs["idle_overview"].set_index("row")
        timeline = dfs["gpu_timeline"].set_index("type")
        # Idle sheets and gpu_timeline report the same idle total.
        assert ov.loc["ALL idle", "total_ms"] == pytest.approx(
            timeline.loc["idle_time", "time ms"], rel=1e-6
        )
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
