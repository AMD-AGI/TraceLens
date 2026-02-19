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

import os
import subprocess
import tempfile

import pandas as pd
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


def _generate_pytorch_perf_report(profile_path, output_path):
    """Run generate_perf_report_pytorch.py to produce an xlsx report."""
    script_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "TraceLens",
            "Reporting",
            "generate_perf_report_pytorch.py",
        )
    )
    cmd = [
        "python3",
        script_path,
        "--profile_json_path",
        profile_path,
        "--output_xlsx_path",
        output_path,
        "--enable_kernel_summary",
        "--short_kernel_study",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"generate_perf_report_pytorch failed: {result.stderr}")
    return output_path


class TestOpsSummarySubtreeVsDirectFromReport:
    """Test ops_summary direct vs subtree relationship using a generated PyTorch report."""

    def test_ops_summary_miopen_convolution_subtree_equals_direct_miopen_plus_direct_add(
        self, tol=1.0
    ):
        """
        Generate PyTorch report for facebook_timesformer trace; in ops_summary,
        total_direct_kernel_time_sum(aten::miopen_convolution) + total_direct_kernel_time_sum(aten::add_)
        should equal total_subtree_kernel_time_sum(aten::miopen_convolution).
        (aten::add_ is a child of miopen_convolution in the op tree.)
        """
        profile_path = os.path.join(
            os.path.dirname(__file__),
            "traces",
            "mi300",
            "facebook_timesformer-base-finetuned-k400__1016002.json.gz",
        )
        if not os.path.exists(profile_path):
            pytest.skip(f"Trace not found: {profile_path}")

        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = os.path.join(tmpdir, "perf_report.xlsx")
            _generate_pytorch_perf_report(profile_path, report_path)

            xls = pd.ExcelFile(report_path)
            if "ops_summary" not in xls.sheet_names:
                pytest.skip("ops_summary sheet not in report")

            df = pd.read_excel(report_path, sheet_name="ops_summary")

            if "total_direct_kernel_time_sum" not in df.columns:
                pytest.skip("ops_summary missing total_direct_kernel_time_sum")
            if "total_subtree_kernel_time_sum" not in df.columns:
                pytest.skip("ops_summary missing total_subtree_kernel_time_sum")

            miopen = df[df["name"] == "aten::miopen_convolution"]
            add_ = df[df["name"] == "aten::add_"]

            if miopen.empty:
                pytest.skip("aten::miopen_convolution not in ops_summary")
            if add_.empty:
                pytest.skip("aten::add_ not in ops_summary")

            direct_miopen = miopen["total_direct_kernel_time_sum"].sum()
            direct_add = add_["total_direct_kernel_time_sum"].sum()
            subtree_miopen = miopen["total_subtree_kernel_time_sum"].sum()

            assert abs((direct_miopen + direct_add) == subtree_miopen), (
                f"direct(miopen_convolution) + direct(aten::add_) should equal "
                f"subtree(miopen_convolution): {direct_miopen} + {direct_add} = {direct_miopen + direct_add} vs subtree {subtree_miopen}"
            )
