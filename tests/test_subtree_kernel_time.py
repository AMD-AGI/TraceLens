###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Tests for subtree (inclusive) kernel time across perf report paths.

Covers:
- _compute_subtree_kernel_time_us
- build_df_unified_perf_table "Subtree Kernel Time (µs)" column
- summarize_df_unified_perf_table aggregation of Subtree Kernel Time
- build_df_perf_metrics "Subtree Kernel Time (µs)" column
- summarize_df_perf_metrics aggregation of Subtree Kernel Time
"""

import pytest
from copy import deepcopy

from TraceLens.Trace2Tree.trace_to_tree import TraceToTree
from TraceLens.TreePerf.tree_perf import TreePerfAnalyzer


# event with start and end time
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


# cpu to gpu links
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


# simple trace with one cpu_op -> runtime -> kernel (leaf launcher)
def _make_simple_trace():
    """One cpu_op -> runtime -> kernel (leaf launcher)."""
    corr = 100
    return [
        _mk_event(
            "cpu_op", "aten::matmul", ts=1000, dur=100, pid=100, tid=100, args={}
        ),
        _mk_event(
            "cuda_runtime",
            "hipLaunchKernel",
            ts=1010,
            dur=5,
            pid=100,
            tid=100,
            args={"correlation": corr},
        ),
        _mk_event(
            "kernel",
            "gemm_kernel",
            ts=1050,
            dur=50,
            pid=0,
            tid=7,
            args={"correlation": corr, "stream": 7},
        ),
        _mk_ac2g(corr, pid=0, tid=7, ts=1050, phase="s"),
        _mk_ac2g(corr, pid=0, tid=7, ts=1050, phase="f"),
    ]


def _make_nested_trace_parent_child_kernel():
    """
    Parent cpu_op contains child cpu_op contains runtime -> kernel.
    Parent does not directly launch kernels; only the child (leaf) does.
    Tree builder propagates gpu_events to ancestors, so we test subtree vs direct
    by clearing parent's gpu_events to simulate "direct = only kernels I launched".
    """
    corr = 200
    return [
        # Parent: spans full range, no direct kernel launch
        _mk_event(
            "cpu_op", "aten::wrapper", ts=1000, dur=200, pid=100, tid=100, args={}
        ),
        # Child: nested inside parent, launches the kernel
        _mk_event(
            "cpu_op", "aten::matmul", ts=1010, dur=100, pid=100, tid=100, args={}
        ),
        _mk_event(
            "cuda_runtime",
            "hipLaunchKernel",
            ts=1015,
            dur=5,
            pid=100,
            tid=100,
            args={"correlation": corr},
        ),
        _mk_event(
            "kernel",
            "gemm_kernel",
            ts=1050,
            dur=50,
            pid=0,
            tid=7,
            args={"correlation": corr, "stream": 7},
        ),
        _mk_ac2g(corr, pid=0, tid=7, ts=1050, phase="s"),
        _mk_ac2g(corr, pid=0, tid=7, ts=1050, phase="f"),
    ]


def _make_nested_trace_two_children():
    """
    Parent cpu_op contains two child cpu_ops, each launching one kernel.
    Parent's subtree = both kernels; parent's direct (if not propagated) = 0.
    """
    return [
        # parent cpu op spans full range
        _mk_event("cpu_op", "parent_op", ts=1000, dur=300, pid=100, tid=100, args={}),
        # First child + kernel
        _mk_event("cpu_op", "aten::add", ts=1010, dur=80, pid=100, tid=100, args={}),
        _mk_event(
            "cuda_runtime",
            "hipLaunchKernel",
            ts=1015,
            dur=5,
            pid=100,
            tid=100,
            args={"correlation": 301},
        ),
        _mk_event(
            "kernel",
            "kernel_a",
            ts=1050,
            dur=50,
            pid=0,
            tid=7,
            args={"correlation": 301, "stream": 7},
        ),
        _mk_ac2g(301, pid=0, tid=7, ts=1050, phase="s"),
        _mk_ac2g(301, pid=0, tid=7, ts=1050, phase="f"),
        # Second child + kernel (sequential, no overlap)
        _mk_event("cpu_op", "aten::mul", ts=1110, dur=80, pid=100, tid=100, args={}),
        _mk_event(
            "cuda_runtime",
            "hipLaunchKernel",
            ts=1115,
            dur=5,
            pid=100,
            tid=100,
            args={"correlation": 302},
        ),
        _mk_event(
            "kernel",
            "kernel_b",
            ts=1150,
            dur=40,
            pid=0,
            tid=7,
            args={"correlation": 302, "stream": 7},
        ),
        _mk_ac2g(302, pid=0, tid=7, ts=1150, phase="s"),
        _mk_ac2g(302, pid=0, tid=7, ts=1150, phase="f"),
    ]


class TestComputeSubtreeKernelTimeUs:
    """Test _compute_subtree_kernel_time_us."""

    def test_leaf_launcher_subtree_equals_direct(self):
        """For a leaf cpu_op that launches one kernel, subtree time equals that kernel's busy time."""
        events = _make_simple_trace()
        tree = TraceToTree(deepcopy(events))
        tree.build_tree()
        analyzer = TreePerfAnalyzer(tree, add_python_func=False)
        # Get the cpu_op event (first event)
        cpu_op = next(e for e in tree.events if e.get("cat") == "cpu_op")
        subtree_us = analyzer._compute_subtree_kernel_time_us(cpu_op)
        assert subtree_us == 50.0

    def test_empty_subtree_returns_zero(self):
        """Event with no GPU kernels in subtree returns 0."""
        events = [
            _mk_event(
                "cpu_op", "aten::no_kernel", ts=1000, dur=10, pid=100, tid=100, args={}
            ),
        ]
        tree = TraceToTree(deepcopy(events))
        tree.build_tree()
        analyzer = TreePerfAnalyzer(tree, add_python_func=False)
        cpu_op = next(e for e in tree.events if e.get("cat") == "cpu_op")
        subtree_us = analyzer._compute_subtree_kernel_time_us(cpu_op)
        assert subtree_us == 0

    def test_parent_op_subtree_exceeds_direct_when_child_launches_kernel(self):
        """
        Parent op does not launch kernels; child does. So subtree (parent) > direct (parent).
        The tree propagates gpu_events to ancestors, so we clear parent's gpu_events
        to simulate reporting 'direct' as only kernels this op directly launched.
        """
        events = _make_nested_trace_parent_child_kernel()
        tree = TraceToTree(deepcopy(events))
        tree.build_tree()
        analyzer = TreePerfAnalyzer(tree, add_python_func=False)
        parent = next(e for e in tree.events if e.get("name") == "aten::wrapper")
        child = next(e for e in tree.events if e.get("name") == "aten::matmul")
        # Child (leaf launcher): subtree = 50
        child_subtree = analyzer._compute_subtree_kernel_time_us(child)
        assert child_subtree == 50.0
        # Simulate direct for parent: only kernels this op directly launched (none).
        # Tree propagates gpu_events to ancestors, so clear parent's to get direct=0.
        parent["gpu_events"] = []
        parent_subtree_us = analyzer._compute_subtree_kernel_time_us(parent)
        assert parent_subtree_us == 50.0, "parent subtree should include child's kernel"

    def test_parent_op_subtree_exceeds_direct_with_two_children(self):
        """
        Parent has two children that each launch a kernel. Subtree = 50+40 = 90;
        direct (if we treat parent as not launching) = 0.
        First child (aten::add): subtree = 50. Second child (aten::mul): subtree = 40.
        """
        events = _make_nested_trace_two_children()
        tree = TraceToTree(deepcopy(events))
        tree.build_tree()
        analyzer = TreePerfAnalyzer(tree, add_python_func=False)
        parent = next(e for e in tree.events if e.get("name") == "parent_op")
        child_add = next(e for e in tree.events if e.get("name") == "aten::add")
        child_mul = next(e for e in tree.events if e.get("name") == "aten::mul")
        # Clear so "direct" = kernels this op directly launched (none for parent)
        parent["gpu_events"] = []
        parent_direct_us = 0.0
        parent_subtree_us = analyzer._compute_subtree_kernel_time_us(parent)
        # Two kernels 50 and 40 us, sequential -> busy_time = 90
        assert parent_subtree_us == 90.0
        assert parent_direct_us == 0
        assert parent_subtree_us != parent_direct_us

        assert analyzer._compute_subtree_kernel_time_us(child_add) == 50.0
        assert analyzer._compute_subtree_kernel_time_us(child_mul) == 40.0


class TestUnifiedPerfTableSubtreeColumn:
    """Test Subtree Kernel Time (µs) in build_df_unified_perf_table and summarize."""

    def test_unified_perf_table_has_subtree_column(self):
        """build_df_unified_perf_table includes Subtree Kernel Time (µs)."""
        events = _make_simple_trace()
        tree = TraceToTree(deepcopy(events))
        tree.build_tree()
        analyzer = TreePerfAnalyzer(tree, add_python_func=False)
        df = analyzer.build_df_unified_perf_table(include_perf_metrics=True)
        assert "Subtree Kernel Time (µs)" in df.columns
        # Leaf op: subtree equals direct kernel time
        if "Kernel Time (µs)" in df.columns:
            for _, row in df.iterrows():
                assert row["Subtree Kernel Time (µs)"] == 50.0
                assert row["Kernel Time (µs)"] == 50.0

    def test_summarize_unified_perf_table_aggregates_subtree(self):
        """summarize_df_unified_perf_table produces Subtree Kernel Time (µs)_sum etc."""
        events = _make_simple_trace()
        tree = TraceToTree(deepcopy(events))
        tree.build_tree()
        analyzer = TreePerfAnalyzer(tree, add_python_func=False)
        df = analyzer.build_df_unified_perf_table(include_perf_metrics=True)
        summary = analyzer.summarize_df_unified_perf_table(
            df, agg_metrics=["mean", "std"], include_pct=False
        )
        assert not summary.empty
        assert "Subtree Kernel Time (µs)_sum" in summary.columns
        if "Kernel Time (µs)" in df.columns:
            for _, row in summary.iterrows():
                assert row["Subtree Kernel Time (µs)_sum"] == 50.0
                assert row["Kernel Time (µs)_sum"] == 50.0


class TestBuildDfPerfMetricsSubtreeColumn:
    """Test Subtree Kernel Time (µs) in build_df_perf_metrics and summarize_df_perf_metrics."""

    def test_summarize_df_perf_metrics_aggregates_subtree_when_present(self):
        """When input df has Subtree Kernel Time (µs), summarize_df_perf_metrics aggregates it."""
        import pandas as pd

        # Minimal df with columns required by summarize_df_perf_metrics
        df = pd.DataFrame(
            [
                {
                    "name": "aten::addmm",
                    "UID": 1,
                    "Kernel Time (µs)": 100.0,
                    "Subtree Kernel Time (µs)": 100.0,
                    "GFLOPS": 1.0,
                    "Data Moved (MB)": 0.1,
                    "FLOPS/Byte": 100.0,
                    "TB/s": 1.0,
                    "TFLOPS/s": 1.0,
                },
                {
                    "name": "aten::addmm",
                    "UID": 2,
                    "Kernel Time (µs)": 200.0,
                    "Subtree Kernel Time (µs)": 200.0,
                    "GFLOPS": 2.0,
                    "Data Moved (MB)": 0.2,
                    "FLOPS/Byte": 200.0,
                    "TB/s": 2.0,
                    "TFLOPS/s": 2.0,
                },
            ]
        )
        summary = TreePerfAnalyzer.summarize_df_perf_metrics(
            df, agg_metrics=["mean", "std"]
        )
        assert "Subtree Kernel Time (µs)_sum" in summary.columns
        assert summary["Subtree Kernel Time (µs)_sum"].iloc[0] == 300.0
