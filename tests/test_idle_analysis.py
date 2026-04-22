###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for IdleTimeAnalyser and the TreePerfAnalyzer idle analysis
delegation methods.

Each test constructs a minimal synthetic trace that triggers a specific
gap classification, builds a TraceToTree + TreePerfAnalyzer, and asserts
the reason column in the resulting DataFrame.
"""

import pytest
from copy import deepcopy

from TraceLens.Trace2Tree.trace_to_tree import TraceToTree
from TraceLens.TreePerf.tree_perf import TreePerfAnalyzer
from TraceLens.TreePerf.idle_time_analysis import (
    IdleTimeAnalyser,
    REASON_LAUNCH_OVERHEAD,
    REASON_EXPLICIT_SYNC,
    REASON_CROSS_STREAM_DEP,
    REASON_CPU_BOTTLENECK,
    REASON_MEMORY_OP_STALL,
    REASON_SCHEDULER_SATURATION,
    REASON_UNKNOWN,
)

CPU_PID, CPU_TID = 100, 100
GPU_PID, GPU_TID = 0, 7
STREAM = 7


def _ev(cat, name, ts, dur, pid=CPU_PID, tid=CPU_TID, args=None):
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


def _kernel(name, ts, dur, corr, stream=STREAM):
    return _ev(
        "kernel",
        name,
        ts,
        dur,
        pid=GPU_PID,
        tid=stream,
        args={"correlation": corr, "stream": stream},
    )


def _launch(ts, dur, corr, name="hipLaunchKernel"):
    return _ev("cuda_runtime", name, ts, dur, args={"correlation": corr})


def _ac2g(corr, ts, phase, stream=STREAM):
    evt = {
        "ph": phase,
        "id": corr,
        "pid": GPU_PID,
        "tid": stream,
        "ts": ts,
        "cat": "ac2g",
        "name": "ac2g",
    }
    if phase == "f":
        evt["bp"] = "e"
    return evt


def _build(events):
    """Build tree and analyzer from events, return TreePerfAnalyzer."""
    trace = {"traceEvents": deepcopy(events)}
    tree = TraceToTree(trace["traceEvents"])
    return TreePerfAnalyzer(tree, add_python_func=False)


def _two_kernel_trace(
    corr1, k1_ts, k1_dur,
    corr2, launch2_ts, launch2_dur, k2_ts, k2_dur,
    extra_events=None,
    stream=STREAM,
    cpu_op_ts=None, cpu_op_dur=None,
):
    """Build a complete two-kernel trace with a wrapping cpu_op.

    The cpu_op is sized to encompass both launches by default.
    Pass extra_events (list) to inject additional runtime events (sync, etc.)
    between the two kernels.
    """
    if cpu_op_ts is None:
        cpu_op_ts = k1_ts - 100
    if cpu_op_dur is None:
        cpu_op_dur = (k2_ts + k2_dur) - cpu_op_ts + 100

    events = [
        _ev("cpu_op", "aten::op", ts=cpu_op_ts, dur=cpu_op_dur),
        _launch(ts=k1_ts - 50, dur=5, corr=corr1),
        _kernel("kernel_A", k1_ts, k1_dur, corr1, stream=stream),
        _ac2g(corr1, k1_ts, "s", stream=stream),
        _ac2g(corr1, k1_ts, "f", stream=stream),
    ]
    if extra_events:
        events.extend(extra_events)
    events.extend([
        _launch(ts=launch2_ts, dur=launch2_dur, corr=corr2),
        _kernel("kernel_B", k2_ts, k2_dur, corr2, stream=stream),
        _ac2g(corr2, k2_ts, "s", stream=stream),
        _ac2g(corr2, k2_ts, "f", stream=stream),
    ])
    return events


class TestLaunchOverhead:
    def test_small_pipelined_gap(self):
        """Gap < threshold with pipelined launch → launch_overhead."""
        # launch for kernel_B starts BEFORE kernel_A ends (pipelined),
        # kernel_B starts shortly after kernel_A ends → small gap.
        events = _two_kernel_trace(
            1, k1_ts=1000, k1_dur=100,
            corr2=2, launch2_ts=1050, launch2_dur=5, k2_ts=1105, k2_dur=100,
        )
        perf = _build(events)
        df = perf.get_df_idle_analysis(launch_overhead_thresh_us=10.0)
        assert len(df) == 1
        assert df.iloc[0]["reason"] == REASON_LAUNCH_OVERHEAD
        assert df.iloc[0]["duration_us"] == 5.0  # 1105 - 1100


class TestExplicitSync:
    def test_sync_blocks_launch_thread(self):
        """Sync API on the same thread as the launch → explicit_sync."""
        # Sync blocks the launch thread for 500us after kernel_A ends.
        events = _two_kernel_trace(
            1, k1_ts=1000, k1_dur=100,
            corr2=2, launch2_ts=1560, launch2_dur=5, k2_ts=1600, k2_dur=100,
            extra_events=[
                _ev("cuda_runtime", "hipEventSynchronize", ts=1050, dur=500),
            ],
        )
        perf = _build(events)
        df = perf.get_df_idle_analysis()
        assert len(df) == 1
        assert df.iloc[0]["reason"] == REASON_EXPLICIT_SYNC

    def test_sync_on_different_thread_not_blamed(self):
        """Sync on a different thread should NOT be blamed."""
        events = _two_kernel_trace(
            1, k1_ts=1000, k1_dur=100,
            corr2=2, launch2_ts=1200, launch2_dur=5, k2_ts=1300, k2_dur=100,
            extra_events=[
                # Sync on a DIFFERENT thread (tid=999).
                _ev("cuda_runtime", "hipEventSynchronize", ts=1050, dur=500, tid=999),
            ],
        )
        perf = _build(events)
        df = perf.get_df_idle_analysis()
        assert len(df) == 1
        # Should be cpu_bottleneck, NOT explicit_sync.
        assert df.iloc[0]["reason"] == REASON_CPU_BOTTLENECK


class TestCrossStreamDep:
    def test_stream_wait_event(self):
        """hipStreamWaitEvent during gap → cross_stream_dep."""
        # Stream wait issued during the gap interval [1100, 1200].
        events = _two_kernel_trace(
            1, k1_ts=1000, k1_dur=100,
            corr2=2, launch2_ts=1050, launch2_dur=5, k2_ts=1200, k2_dur=100,
            extra_events=[
                _ev("cuda_runtime", "hipStreamWaitEvent", ts=1100, dur=50),
            ],
        )
        perf = _build(events)
        df = perf.get_df_idle_analysis()
        assert len(df) == 1
        assert df.iloc[0]["reason"] == REASON_CROSS_STREAM_DEP


class TestCpuBottleneck:
    def test_late_launch(self):
        """Launch issued after prev kernel ended, no blocking API → cpu_bottleneck."""
        events = _two_kernel_trace(
            1, k1_ts=1000, k1_dur=100,
            corr2=2, launch2_ts=1200, launch2_dur=5, k2_ts=1300, k2_dur=100,
        )
        perf = _build(events)
        df = perf.get_df_idle_analysis()
        assert len(df) == 1
        assert df.iloc[0]["reason"] == REASON_CPU_BOTTLENECK
        assert "cpu_op_ancestor_name" in df.columns
        assert df.iloc[0]["cpu_op_ancestor_name"] == "aten::op"


class TestMemoryOpStall:
    def test_memcpy_blocks_launch_thread(self):
        """Memory API on launch thread during gap → memory_op_stall."""
        events = _two_kernel_trace(
            1, k1_ts=1000, k1_dur=100,
            corr2=2, launch2_ts=1160, launch2_dur=5, k2_ts=1200, k2_dur=100,
            extra_events=[
                _ev("cuda_runtime", "hipMemcpy", ts=1050, dur=100),
            ],
        )
        perf = _build(events)
        df = perf.get_df_idle_analysis()
        assert len(df) == 1
        assert df.iloc[0]["reason"] == REASON_MEMORY_OP_STALL


class TestSchedulerSaturation:
    def test_pipelined_large_gap(self):
        """Pipelined launch but large gap with no stream wait → scheduler_saturation."""
        events = _two_kernel_trace(
            1, k1_ts=1000, k1_dur=100,
            corr2=2, launch2_ts=1050, launch2_dur=5, k2_ts=1200, k2_dur=100,
        )
        perf = _build(events)
        df = perf.get_df_idle_analysis(launch_overhead_thresh_us=10.0)
        assert len(df) == 1
        assert df.iloc[0]["reason"] == REASON_SCHEDULER_SATURATION
        assert df.iloc[0]["duration_us"] == 100.0  # 1200 - 1100


class TestUnknown:
    def test_unlinked_gpu_op(self):
        """GPU event with no launch parent → unknown."""
        events = [
            _ev("cpu_op", "aten::op", ts=900, dur=500),
            _kernel("kernel_A", 1000, 100, corr=1),
            _ac2g(1, 1000, "s"),
            _ac2g(1, 1000, "f"),
            _launch(ts=950, dur=5, corr=1),
            # A memset with no launch parent or ac2g links.
            {
                "ph": "X",
                "cat": "gpu_memset",
                "name": "Memset (Device)",
                "pid": GPU_PID,
                "tid": STREAM,
                "ts": 1200,
                "dur": 10,
                "args": {"stream": STREAM},
            },
        ]
        perf = _build(events)
        df = perf.get_df_idle_analysis()
        assert len(df) == 1
        assert df.iloc[0]["reason"] == REASON_UNKNOWN


class TestDelegationMethods:
    """Verify the TreePerfAnalyzer thin delegation methods work."""

    def _make_analyzer(self):
        events = _two_kernel_trace(
            1, k1_ts=1000, k1_dur=100,
            corr2=2, launch2_ts=1200, launch2_dur=5, k2_ts=1300, k2_dur=100,
        )
        return _build(events)

    def test_get_df_idle_analysis(self):
        perf = self._make_analyzer()
        df = perf.get_df_idle_analysis()
        assert not df.empty
        assert "reason" in df.columns

    def test_get_idle_summary_df(self):
        perf = self._make_analyzer()
        summary = perf.get_idle_summary_df()
        assert not summary.empty
        assert "pct_of_total_idle" in summary.columns
        assert summary["pct_of_total_idle"].sum() == pytest.approx(100.0)

    def test_get_top_idle_gaps(self):
        perf = self._make_analyzer()
        top = perf.get_top_idle_gaps(n=5)
        assert len(top) <= 5

    def test_stream_filter(self):
        perf = self._make_analyzer()
        df_all = perf.get_df_idle_analysis()
        df_filtered = perf.get_df_idle_analysis(stream_id=STREAM)
        assert len(df_filtered) <= len(df_all)
        if not df_filtered.empty:
            assert (df_filtered["stream"] == STREAM).all()


class TestEmptyTrace:
    def test_no_gpu_events(self):
        events = [_ev("cpu_op", "aten::op", ts=1000, dur=100)]
        perf = _build(events)
        df = perf.get_df_idle_analysis()
        assert df.empty

    def test_single_kernel(self):
        events = [
            _ev("cpu_op", "aten::op", ts=900, dur=300),
            _launch(ts=950, dur=5, corr=1),
            _kernel("kernel_A", 1000, 100, corr=1),
            _ac2g(1, 1000, "s"),
            _ac2g(1, 1000, "f"),
        ]
        perf = _build(events)
        df = perf.get_df_idle_analysis()
        assert df.empty
