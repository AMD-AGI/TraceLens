###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for get_kernel_launchers in TreePerfAnalyzer.

Tests cover the following edge cases:
1. Basic case: cpu_op -> runtime -> kernel
2. With python functions: cpu_op -> python_function -> runtime -> kernel
3. No cpu_op in parent stack (unlinked runtime events)
4. NCCL kernel handling (include_nccl=True/False)
5. Graph launch: cpu_op -> cudaGraphLaunch -> multiple kernels
6. Nested cpu_ops with "execute" pattern
"""

import pytest
from typing import Dict, List
from copy import deepcopy

from TraceLens.Trace2Tree.trace_to_tree import TraceToTree
from TraceLens.TreePerf.tree_perf import TreePerfAnalyzer


def _mk_event(
    cat: str, name: str, ts: float, dur: float, pid: int, tid: int, args: Dict = None
) -> Dict:
    """Helper to create a trace event."""
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


def _mk_ac2g(corr_id: int, pid: int, tid: int, ts: float, phase: str) -> Dict:
    """Helper to create ac2g (async CPU to GPU) linking events."""
    evt = {
        "ph": phase,  # "s" for start, "f" for finish
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


class TestBasicKernelLauncher:
    """Test basic cpu_op -> runtime -> kernel pattern."""

    def test_simple_cpu_op_launches_kernel(self):
        """Basic case: cpu_op -> hipLaunchKernel -> kernel"""
        corr = 100
        events = [
            _mk_event(
                "cpu_op",
                "aten::matmul",
                ts=1000,
                dur=100,
                pid=100,
                tid=100,
                args={"Input Dims": [[32, 64], [64, 128]]},
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

        trace = {"traceEvents": events}
        tree = TraceToTree(deepcopy(trace["traceEvents"]))
        analyzer = TreePerfAnalyzer(tree, add_python_func=False)

        launchers = analyzer.get_kernel_launchers()

        assert len(launchers) == 1
        assert launchers[0]["name"] == "aten::matmul"
        assert launchers[0]["direct_kernel_count"] == 1
        assert launchers[0]["total_direct_kernel_time"] == 50.0
        assert launchers[0]["kernel_details"][0]["name"] == "gemm_kernel"

    def test_multiple_kernels_under_one_cpu_op(self):
        """cpu_op launches multiple kernels through multiple runtime calls."""
        events = [
            _mk_event(
                "cpu_op", "aten::conv2d", ts=1000, dur=200, pid=100, tid=100, args={}
            ),
            _mk_event(
                "cuda_runtime",
                "hipLaunchKernel",
                ts=1010,
                dur=5,
                pid=100,
                tid=100,
                args={"correlation": 101},
            ),
            _mk_event(
                "kernel",
                "conv_kernel_1",
                ts=1050,
                dur=30,
                pid=0,
                tid=7,
                args={"correlation": 101, "stream": 7},
            ),
            _mk_ac2g(101, pid=0, tid=7, ts=1050, phase="s"),
            _mk_ac2g(101, pid=0, tid=7, ts=1050, phase="f"),
            _mk_event(
                "cuda_runtime",
                "hipLaunchKernel",
                ts=1100,
                dur=5,
                pid=100,
                tid=100,
                args={"correlation": 102},
            ),
            _mk_event(
                "kernel",
                "conv_kernel_2",
                ts=1150,
                dur=40,
                pid=0,
                tid=7,
                args={"correlation": 102, "stream": 7},
            ),
            _mk_ac2g(102, pid=0, tid=7, ts=1150, phase="s"),
            _mk_ac2g(102, pid=0, tid=7, ts=1150, phase="f"),
        ]

        tree = TraceToTree(deepcopy(events))
        analyzer = TreePerfAnalyzer(tree, add_python_func=False)

        launchers = analyzer.get_kernel_launchers()

        assert len(launchers) == 1
        assert launchers[0]["name"] == "aten::conv2d"
        assert launchers[0]["direct_kernel_count"] == 2
        assert launchers[0]["total_direct_kernel_time"] == 70.0


class TestPythonFunctionSkipping:
    """Test that python functions are skipped when finding leaf cpu_op."""

    def test_cpu_op_with_python_func_between(self):
        """cpu_op -> python_function -> runtime -> kernel should find cpu_op as launcher."""
        corr = 200
        events = [
            _mk_event(
                "cpu_op", "aten::linear", ts=1000, dur=150, pid=100, tid=100, args={}
            ),
            _mk_event(
                "python_function",
                "torch/nn/linear.py:forward",
                ts=1005,
                dur=140,
                pid=100,
                tid=100,
                args={},
            ),
            _mk_event(
                "cuda_runtime",
                "hipLaunchKernel",
                ts=1020,
                dur=5,
                pid=100,
                tid=100,
                args={"correlation": corr},
            ),
            _mk_event(
                "kernel",
                "linear_kernel",
                ts=1060,
                dur=60,
                pid=0,
                tid=7,
                args={"correlation": corr, "stream": 7},
            ),
            _mk_ac2g(corr, pid=0, tid=7, ts=1060, phase="s"),
            _mk_ac2g(corr, pid=0, tid=7, ts=1060, phase="f"),
        ]

        # Test WITHOUT python_func
        tree_no_py = TraceToTree(deepcopy(events))
        analyzer_no_py = TreePerfAnalyzer(tree_no_py, add_python_func=False)
        launchers_no_py = analyzer_no_py.get_kernel_launchers()

        # Test WITH python_func
        tree_with_py = TraceToTree(deepcopy(events))
        analyzer_with_py = TreePerfAnalyzer(tree_with_py, add_python_func=True)
        launchers_with_py = analyzer_with_py.get_kernel_launchers()

        # Both should find the same cpu_op as launcher
        assert len(launchers_no_py) == 1
        assert len(launchers_with_py) == 1
        assert launchers_no_py[0]["name"] == "aten::linear"
        assert launchers_with_py[0]["name"] == "aten::linear"
        assert (
            launchers_no_py[0]["direct_kernel_count"]
            == launchers_with_py[0]["direct_kernel_count"]
        )

    def test_deeply_nested_python_functions(self):
        """cpu_op -> py_func -> py_func -> py_func -> runtime -> kernel."""
        corr = 300
        events = [
            _mk_event(
                "cpu_op", "aten::bmm", ts=1000, dur=200, pid=100, tid=100, args={}
            ),
            _mk_event(
                "python_function",
                "module1.py:forward",
                ts=1005,
                dur=190,
                pid=100,
                tid=100,
                args={},
            ),
            _mk_event(
                "python_function",
                "module2.py:call",
                ts=1010,
                dur=180,
                pid=100,
                tid=100,
                args={},
            ),
            _mk_event(
                "python_function",
                "module3.py:compute",
                ts=1015,
                dur=170,
                pid=100,
                tid=100,
                args={},
            ),
            _mk_event(
                "cuda_runtime",
                "hipLaunchKernel",
                ts=1050,
                dur=5,
                pid=100,
                tid=100,
                args={"correlation": corr},
            ),
            _mk_event(
                "kernel",
                "bmm_kernel",
                ts=1100,
                dur=80,
                pid=0,
                tid=7,
                args={"correlation": corr, "stream": 7},
            ),
            _mk_ac2g(corr, pid=0, tid=7, ts=1100, phase="s"),
            _mk_ac2g(corr, pid=0, tid=7, ts=1100, phase="f"),
        ]

        tree = TraceToTree(deepcopy(events))
        analyzer = TreePerfAnalyzer(tree, add_python_func=True)

        launchers = analyzer.get_kernel_launchers()

        assert len(launchers) == 1
        assert launchers[0]["name"] == "aten::bmm"


class TestUnlinkedRuntimeEvents:
    """Test runtime events with no cpu_op parent (unlinked)."""

    def test_runtime_without_cpu_op_parent(self):
        """Runtime event with no cpu_op ancestor should be captured as launcher."""
        corr = 400
        events = [
            # No cpu_op parent - runtime is top level
            _mk_event(
                "cuda_runtime",
                "hipLaunchKernel",
                ts=1000,
                dur=5,
                pid=100,
                tid=100,
                args={"correlation": corr},
            ),
            _mk_event(
                "kernel",
                "orphan_kernel",
                ts=1050,
                dur=25,
                pid=0,
                tid=7,
                args={"correlation": corr, "stream": 7},
            ),
            _mk_ac2g(corr, pid=0, tid=7, ts=1050, phase="s"),
            _mk_ac2g(corr, pid=0, tid=7, ts=1050, phase="f"),
        ]

        tree = TraceToTree(deepcopy(events))
        analyzer = TreePerfAnalyzer(tree, add_python_func=False)

        launchers = analyzer.get_kernel_launchers()

        # The runtime event itself should be the launcher
        assert len(launchers) == 1
        assert launchers[0]["name"] == "hipLaunchKernel"
        assert launchers[0]["direct_kernel_count"] == 1

    def test_mixed_linked_and_unlinked(self):
        """Mix of linked cpu_op and unlinked runtime events."""
        events = [
            # Linked: cpu_op -> runtime -> kernel
            _mk_event(
                "cpu_op", "aten::relu", ts=1000, dur=50, pid=100, tid=100, args={}
            ),
            _mk_event(
                "cuda_runtime",
                "hipLaunchKernel",
                ts=1010,
                dur=5,
                pid=100,
                tid=100,
                args={"correlation": 501},
            ),
            _mk_event(
                "kernel",
                "relu_kernel",
                ts=1050,
                dur=15,
                pid=0,
                tid=7,
                args={"correlation": 501, "stream": 7},
            ),
            _mk_ac2g(501, pid=0, tid=7, ts=1050, phase="s"),
            _mk_ac2g(501, pid=0, tid=7, ts=1050, phase="f"),
            # Unlinked: just runtime -> kernel
            _mk_event(
                "cuda_runtime",
                "hipLaunchKernel",
                ts=2000,
                dur=5,
                pid=100,
                tid=100,
                args={"correlation": 502},
            ),
            _mk_event(
                "kernel",
                "unlinked_kernel",
                ts=2050,
                dur=20,
                pid=0,
                tid=7,
                args={"correlation": 502, "stream": 7},
            ),
            _mk_ac2g(502, pid=0, tid=7, ts=2050, phase="s"),
            _mk_ac2g(502, pid=0, tid=7, ts=2050, phase="f"),
        ]

        tree = TraceToTree(deepcopy(events))
        analyzer = TreePerfAnalyzer(tree, add_python_func=False)

        launchers = analyzer.get_kernel_launchers()

        assert len(launchers) == 2
        launcher_names = {l["name"] for l in launchers}
        assert "aten::relu" in launcher_names
        assert "hipLaunchKernel" in launcher_names


class TestNCCLKernelHandling:
    """Test NCCL kernel inclusion/exclusion."""

    def test_nccl_excluded_by_default(self):
        """NCCL kernels should be excluded when include_nccl=False."""
        events = [
            _mk_event(
                "cpu_op", "c10d::allreduce", ts=1000, dur=100, pid=100, tid=100, args={}
            ),
            _mk_event(
                "cuda_runtime",
                "hipLaunchKernel",
                ts=1010,
                dur=5,
                pid=100,
                tid=100,
                args={"correlation": 601},
            ),
            _mk_event(
                "kernel",
                "ncclKernel_AllReduce",
                ts=1050,
                dur=50,
                pid=0,
                tid=7,
                args={"correlation": 601, "stream": 7},
            ),
            _mk_ac2g(601, pid=0, tid=7, ts=1050, phase="s"),
            _mk_ac2g(601, pid=0, tid=7, ts=1050, phase="f"),
        ]

        tree = TraceToTree(deepcopy(events))
        analyzer = TreePerfAnalyzer(tree, add_python_func=False)

        launchers_excl = analyzer.get_kernel_launchers(include_nccl=False)

        # NCCL should be excluded
        assert len(launchers_excl) == 0

    def test_nccl_included_when_requested(self):
        """NCCL kernels should be included when include_nccl=True."""
        events = [
            _mk_event(
                "cpu_op", "c10d::allreduce", ts=1000, dur=100, pid=100, tid=100, args={}
            ),
            _mk_event(
                "cuda_runtime",
                "hipLaunchKernel",
                ts=1010,
                dur=5,
                pid=100,
                tid=100,
                args={"correlation": 602},
            ),
            _mk_event(
                "kernel",
                "ncclKernel_AllReduce",
                ts=1050,
                dur=50,
                pid=0,
                tid=7,
                args={"correlation": 602, "stream": 7},
            ),
            _mk_ac2g(602, pid=0, tid=7, ts=1050, phase="s"),
            _mk_ac2g(602, pid=0, tid=7, ts=1050, phase="f"),
        ]

        tree = TraceToTree(deepcopy(events))
        analyzer = TreePerfAnalyzer(tree, add_python_func=False)

        launchers_incl = analyzer.get_kernel_launchers(include_nccl=True)

        # NCCL should be included
        assert len(launchers_incl) == 1
        assert launchers_incl[0]["name"] == "c10d::allreduce"
        assert "nccl" in launchers_incl[0]["kernel_details"][0]["name"].lower()

    def test_mixed_nccl_and_regular_kernels(self):
        """cpu_op with both NCCL and regular kernels."""
        events = [
            _mk_event(
                "cpu_op", "distributed_op", ts=1000, dur=150, pid=100, tid=100, args={}
            ),
            # Regular kernel
            _mk_event(
                "cuda_runtime",
                "hipLaunchKernel",
                ts=1010,
                dur=5,
                pid=100,
                tid=100,
                args={"correlation": 701},
            ),
            _mk_event(
                "kernel",
                "compute_kernel",
                ts=1050,
                dur=30,
                pid=0,
                tid=7,
                args={"correlation": 701, "stream": 7},
            ),
            _mk_ac2g(701, pid=0, tid=7, ts=1050, phase="s"),
            _mk_ac2g(701, pid=0, tid=7, ts=1050, phase="f"),
            # NCCL kernel
            _mk_event(
                "cuda_runtime",
                "hipLaunchKernel",
                ts=1100,
                dur=5,
                pid=100,
                tid=100,
                args={"correlation": 702},
            ),
            _mk_event(
                "kernel",
                "ncclKernel_Broadcast",
                ts=1150,
                dur=40,
                pid=0,
                tid=7,
                args={"correlation": 702, "stream": 7},
            ),
            _mk_ac2g(702, pid=0, tid=7, ts=1150, phase="s"),
            _mk_ac2g(702, pid=0, tid=7, ts=1150, phase="f"),
        ]

        tree = TraceToTree(deepcopy(events))
        analyzer = TreePerfAnalyzer(tree, add_python_func=False)

        # Without NCCL
        launchers_excl = analyzer.get_kernel_launchers(include_nccl=False)
        assert len(launchers_excl) == 1
        assert launchers_excl[0]["direct_kernel_count"] == 1  # Only compute kernel
        assert launchers_excl[0]["total_direct_kernel_time"] == 30.0

        # Need to rebuild tree for fresh state
        tree2 = TraceToTree(deepcopy(events))
        analyzer2 = TreePerfAnalyzer(tree2, add_python_func=False)

        # With NCCL
        launchers_incl = analyzer2.get_kernel_launchers(include_nccl=True)
        assert len(launchers_incl) == 1
        assert launchers_incl[0]["direct_kernel_count"] == 2  # Both kernels
        assert launchers_incl[0]["total_direct_kernel_time"] == 70.0


class TestGraphLaunch:
    """Test CUDA/HIP graph launch scenarios."""

    def test_graph_launch_multiple_kernels(self):
        """cudaGraphLaunch should capture all correlated kernels."""
        corr = 800
        events = [
            _mk_event(
                "cpu_op",
                "aten::graph_wrapper",
                ts=1000,
                dur=200,
                pid=100,
                tid=100,
                args={},
            ),
            _mk_event(
                "cuda_runtime",
                "cudaGraphLaunch",
                ts=1010,
                dur=10,
                pid=100,
                tid=100,
                args={"correlation": corr},
            ),
            # Multiple kernels with same correlation (graph replay)
            _mk_event(
                "kernel",
                "kernel_A",
                ts=1050,
                dur=20,
                pid=0,
                tid=7,
                args={"correlation": corr, "stream": 7},
            ),
            _mk_event(
                "kernel",
                "kernel_B",
                ts=1080,
                dur=30,
                pid=0,
                tid=7,
                args={"correlation": corr, "stream": 7},
            ),
            _mk_event(
                "kernel",
                "kernel_C",
                ts=1120,
                dur=40,
                pid=0,
                tid=7,
                args={"correlation": corr, "stream": 7},
            ),
            _mk_ac2g(corr, pid=0, tid=7, ts=1050, phase="s"),
            _mk_ac2g(corr, pid=0, tid=7, ts=1050, phase="f"),
        ]

        tree = TraceToTree(deepcopy(events))
        analyzer = TreePerfAnalyzer(tree, add_python_func=False)

        launchers = analyzer.get_kernel_launchers()

        assert len(launchers) == 1
        assert launchers[0]["name"] == "aten::graph_wrapper"
        assert launchers[0]["direct_kernel_count"] == 3
        assert launchers[0]["total_direct_kernel_time"] == 90.0

    def test_hip_graph_launch(self):
        """hipGraphLaunch should also work for graph mode."""
        corr = 850
        events = [
            _mk_event(
                "cpu_op", "graph_op", ts=1000, dur=150, pid=100, tid=100, args={}
            ),
            _mk_event(
                "cuda_runtime",
                "hipGraphLaunch",
                ts=1010,
                dur=8,
                pid=100,
                tid=100,
                args={"correlation": corr},
            ),
            _mk_event(
                "kernel",
                "graph_kernel_1",
                ts=1050,
                dur=25,
                pid=0,
                tid=7,
                args={"correlation": corr, "stream": 7},
            ),
            _mk_event(
                "kernel",
                "graph_kernel_2",
                ts=1085,
                dur=35,
                pid=0,
                tid=7,
                args={"correlation": corr, "stream": 7},
            ),
            _mk_ac2g(corr, pid=0, tid=7, ts=1050, phase="s"),
            _mk_ac2g(corr, pid=0, tid=7, ts=1050, phase="f"),
        ]

        tree = TraceToTree(deepcopy(events))
        analyzer = TreePerfAnalyzer(tree, add_python_func=False)

        launchers = analyzer.get_kernel_launchers()

        assert len(launchers) == 1
        assert launchers[0]["direct_kernel_count"] == 2
        assert launchers[0]["total_direct_kernel_time"] == 60.0


class TestExecutePattern:
    """Test the special 'execute' cpu_op pattern."""

    def test_execute_cpu_op_is_leaf_launcher(self):
        """
        cpu_op -> cpu_op(execute) -> runtime -> kernel

        The 'execute' cpu_op should be detected as the kernel launcher since
        it's the first cpu_op ancestor when walking up from the kernel.
        """
        corr = 950
        events = [
            _mk_event(
                "cpu_op",
                "outer_wrapper_op",
                ts=1000,
                dur=200,
                pid=100,
                tid=100,
                args={"Input Dims": [[32, 64]]},
            ),
            _mk_event("cpu_op", "execute", ts=1050, dur=100, pid=100, tid=100, args={}),
            _mk_event(
                "cuda_runtime",
                "hipLaunchKernel",
                ts=1060,
                dur=5,
                pid=100,
                tid=100,
                args={"correlation": corr},
            ),
            _mk_event(
                "kernel",
                "execute_kernel",
                ts=1100,
                dur=50,
                pid=0,
                tid=7,
                args={"correlation": corr, "stream": 7},
            ),
            _mk_ac2g(corr, pid=0, tid=7, ts=1100, phase="s"),
            _mk_ac2g(corr, pid=0, tid=7, ts=1100, phase="f"),
        ]

        tree = TraceToTree(deepcopy(events))
        analyzer = TreePerfAnalyzer(tree, add_python_func=False)

        launchers = analyzer.get_kernel_launchers()

        # The 'execute' cpu_op should be the launcher (it's the leaf/first cpu_op from kernel)
        assert len(launchers) == 1
        assert launchers[0]["name"] == "execute"
        assert launchers[0]["direct_kernel_count"] == 1

    def test_execute_with_python_func(self):
        """
        cpu_op -> cpu_op(execute) -> python_func -> runtime -> kernel

        Even with python functions, 'execute' should still be detected as launcher.
        """
        corr = 951
        events = [
            _mk_event(
                "cpu_op", "outer_op", ts=1000, dur=250, pid=100, tid=100, args={}
            ),
            _mk_event("cpu_op", "execute", ts=1050, dur=150, pid=100, tid=100, args={}),
            _mk_event(
                "python_function",
                "graph/execute.py:run",
                ts=1060,
                dur=130,
                pid=100,
                tid=100,
                args={},
            ),
            _mk_event(
                "cuda_runtime",
                "hipLaunchKernel",
                ts=1080,
                dur=5,
                pid=100,
                tid=100,
                args={"correlation": corr},
            ),
            _mk_event(
                "kernel",
                "graph_kernel",
                ts=1120,
                dur=60,
                pid=0,
                tid=7,
                args={"correlation": corr, "stream": 7},
            ),
            _mk_ac2g(corr, pid=0, tid=7, ts=1120, phase="s"),
            _mk_ac2g(corr, pid=0, tid=7, ts=1120, phase="f"),
        ]

        # Test WITHOUT python func
        tree1 = TraceToTree(deepcopy(events))
        analyzer1 = TreePerfAnalyzer(tree1, add_python_func=False)
        launchers1 = analyzer1.get_kernel_launchers()

        # Test WITH python func
        tree2 = TraceToTree(deepcopy(events))
        analyzer2 = TreePerfAnalyzer(tree2, add_python_func=True)
        launchers2 = analyzer2.get_kernel_launchers()

        # Both should find 'execute' as the launcher
        assert len(launchers1) == 1
        assert len(launchers2) == 1
        assert launchers1[0]["name"] == "execute"
        assert launchers2[0]["name"] == "execute"


class TestNestedCpuOps:
    """Test nested cpu_op scenarios."""

    def test_nested_cpu_ops_leaf_is_launcher(self):
        """Inner (leaf) cpu_op should be the kernel launcher, not outer."""
        events = [
            _mk_event(
                "cpu_op", "outer_op", ts=1000, dur=200, pid=100, tid=100, args={}
            ),
            _mk_event(
                "cpu_op", "inner_op", ts=1050, dur=100, pid=100, tid=100, args={}
            ),
            _mk_event(
                "cuda_runtime",
                "hipLaunchKernel",
                ts=1060,
                dur=5,
                pid=100,
                tid=100,
                args={"correlation": 901},
            ),
            _mk_event(
                "kernel",
                "compute_kernel",
                ts=1100,
                dur=50,
                pid=0,
                tid=7,
                args={"correlation": 901, "stream": 7},
            ),
            _mk_ac2g(901, pid=0, tid=7, ts=1100, phase="s"),
            _mk_ac2g(901, pid=0, tid=7, ts=1100, phase="f"),
        ]

        tree = TraceToTree(deepcopy(events))
        analyzer = TreePerfAnalyzer(tree, add_python_func=False)

        launchers = analyzer.get_kernel_launchers()

        # The inner cpu_op should be the launcher
        assert len(launchers) == 1
        assert launchers[0]["name"] == "inner_op"

    def test_sibling_cpu_ops_each_launch_kernels(self):
        """Multiple sibling cpu_ops each launching their own kernels."""
        events = [
            _mk_event(
                "cpu_op", "parent_op", ts=1000, dur=300, pid=100, tid=100, args={}
            ),
            # First child
            _mk_event(
                "cpu_op", "child_op_1", ts=1010, dur=80, pid=100, tid=100, args={}
            ),
            _mk_event(
                "cuda_runtime",
                "hipLaunchKernel",
                ts=1020,
                dur=5,
                pid=100,
                tid=100,
                args={"correlation": 1001},
            ),
            _mk_event(
                "kernel",
                "kernel_1",
                ts=1060,
                dur=30,
                pid=0,
                tid=7,
                args={"correlation": 1001, "stream": 7},
            ),
            _mk_ac2g(1001, pid=0, tid=7, ts=1060, phase="s"),
            _mk_ac2g(1001, pid=0, tid=7, ts=1060, phase="f"),
            # Second child
            _mk_event(
                "cpu_op", "child_op_2", ts=1100, dur=80, pid=100, tid=100, args={}
            ),
            _mk_event(
                "cuda_runtime",
                "hipLaunchKernel",
                ts=1110,
                dur=5,
                pid=100,
                tid=100,
                args={"correlation": 1002},
            ),
            _mk_event(
                "kernel",
                "kernel_2",
                ts=1150,
                dur=40,
                pid=0,
                tid=7,
                args={"correlation": 1002, "stream": 7},
            ),
            _mk_ac2g(1002, pid=0, tid=7, ts=1150, phase="s"),
            _mk_ac2g(1002, pid=0, tid=7, ts=1150, phase="f"),
        ]

        tree = TraceToTree(deepcopy(events))
        analyzer = TreePerfAnalyzer(tree, add_python_func=False)

        launchers = analyzer.get_kernel_launchers()

        # Both child ops should be launchers
        assert len(launchers) == 2
        launcher_names = {l["name"] for l in launchers}
        assert "child_op_1" in launcher_names
        assert "child_op_2" in launcher_names


class TestMemoryOperations:
    """Test gpu_memcpy and gpu_memset are also captured."""

    def test_memcpy_captured_as_kernel(self):
        """gpu_memcpy should be captured as a kernel event."""
        # Note: Use "External id" since there's no launch event to set linking_key to "correlation"
        ext_id = 1100
        events = [
            _mk_event(
                "cpu_op", "aten::copy_", ts=1000, dur=50, pid=100, tid=100, args={}
            ),
            _mk_event(
                "cuda_runtime",
                "hipMemcpyAsync",
                ts=1010,
                dur=5,
                pid=100,
                tid=100,
                args={"External id": ext_id},
            ),
            _mk_event(
                "gpu_memcpy",
                "Memcpy DtoD",
                ts=1050,
                dur=15,
                pid=0,
                tid=7,
                args={"External id": ext_id, "stream": 7},
            ),
            _mk_ac2g(ext_id, pid=0, tid=7, ts=1050, phase="s"),
            _mk_ac2g(ext_id, pid=0, tid=7, ts=1050, phase="f"),
        ]

        tree = TraceToTree(deepcopy(events))
        analyzer = TreePerfAnalyzer(tree, add_python_func=False)

        launchers = analyzer.get_kernel_launchers()

        assert len(launchers) == 1
        assert launchers[0]["name"] == "aten::copy_"
        assert launchers[0]["kernel_details"][0]["name"] == "Memcpy DtoD"

    def test_memset_captured_as_kernel(self):
        """gpu_memset should be captured as a kernel event."""
        # Note: Use "External id" since there's no launch event to set linking_key to "correlation"
        ext_id = 1200
        events = [
            _mk_event(
                "cpu_op", "aten::zero_", ts=1000, dur=40, pid=100, tid=100, args={}
            ),
            _mk_event(
                "cuda_runtime",
                "hipMemsetAsync",
                ts=1010,
                dur=5,
                pid=100,
                tid=100,
                args={"External id": ext_id},
            ),
            _mk_event(
                "gpu_memset",
                "Memset (Device)",
                ts=1050,
                dur=10,
                pid=0,
                tid=7,
                args={"External id": ext_id, "stream": 7},
            ),
            _mk_ac2g(ext_id, pid=0, tid=7, ts=1050, phase="s"),
            _mk_ac2g(ext_id, pid=0, tid=7, ts=1050, phase="f"),
        ]

        tree = TraceToTree(deepcopy(events))
        analyzer = TreePerfAnalyzer(tree, add_python_func=False)

        launchers = analyzer.get_kernel_launchers()

        assert len(launchers) == 1
        assert launchers[0]["name"] == "aten::zero_"
        assert launchers[0]["kernel_details"][0]["name"] == "Memset (Device)"


class TestConsistencyWithPythonFunc:
    """Test that results are identical with/without add_python_func."""

    def test_consistency_simple_case(self):
        """Simple case should give same results."""
        corr = 1300
        events = [
            _mk_event(
                "cpu_op", "aten::add", ts=1000, dur=60, pid=100, tid=100, args={}
            ),
            _mk_event(
                "python_function",
                "torch/add.py",
                ts=1005,
                dur=50,
                pid=100,
                tid=100,
                args={},
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
                "add_kernel",
                ts=1050,
                dur=20,
                pid=0,
                tid=7,
                args={"correlation": corr, "stream": 7},
            ),
            _mk_ac2g(corr, pid=0, tid=7, ts=1050, phase="s"),
            _mk_ac2g(corr, pid=0, tid=7, ts=1050, phase="f"),
        ]

        # Without python func
        tree1 = TraceToTree(deepcopy(events))
        analyzer1 = TreePerfAnalyzer(tree1, add_python_func=False)
        launchers1 = analyzer1.get_kernel_launchers()

        # With python func
        tree2 = TraceToTree(deepcopy(events))
        analyzer2 = TreePerfAnalyzer(tree2, add_python_func=True)
        launchers2 = analyzer2.get_kernel_launchers()

        assert len(launchers1) == len(launchers2)
        assert launchers1[0]["name"] == launchers2[0]["name"]
        assert (
            launchers1[0]["direct_kernel_count"] == launchers2[0]["direct_kernel_count"]
        )
        assert (
            launchers1[0]["total_direct_kernel_time"]
            == launchers2[0]["total_direct_kernel_time"]
        )

    def test_consistency_complex_nesting(self):
        """Complex nesting should give same results."""
        events = [
            _mk_event(
                "cpu_op",
                "vllm::moe_forward",
                ts=1000,
                dur=500,
                pid=100,
                tid=100,
                args={},
            ),
            _mk_event(
                "python_function",
                "moe/layer.py:forward",
                ts=1010,
                dur=480,
                pid=100,
                tid=100,
                args={},
            ),
            _mk_event(
                "python_function",
                "moe/routing.py:route",
                ts=1020,
                dur=100,
                pid=100,
                tid=100,
                args={},
            ),
            _mk_event(
                "cuda_runtime",
                "hipLaunchKernel",
                ts=1030,
                dur=5,
                pid=100,
                tid=100,
                args={"correlation": 1401},
            ),
            _mk_event(
                "kernel",
                "topk_kernel",
                ts=1070,
                dur=15,
                pid=0,
                tid=7,
                args={"correlation": 1401, "stream": 7},
            ),
            _mk_ac2g(1401, pid=0, tid=7, ts=1070, phase="s"),
            _mk_ac2g(1401, pid=0, tid=7, ts=1070, phase="f"),
            _mk_event(
                "python_function",
                "moe/experts.py:compute",
                ts=1150,
                dur=300,
                pid=100,
                tid=100,
                args={},
            ),
            _mk_event(
                "cuda_runtime",
                "hipLaunchKernel",
                ts=1200,
                dur=5,
                pid=100,
                tid=100,
                args={"correlation": 1402},
            ),
            _mk_event(
                "kernel",
                "gemm_kernel",
                ts=1250,
                dur=100,
                pid=0,
                tid=7,
                args={"correlation": 1402, "stream": 7},
            ),
            _mk_ac2g(1402, pid=0, tid=7, ts=1250, phase="s"),
            _mk_ac2g(1402, pid=0, tid=7, ts=1250, phase="f"),
        ]

        # Without python func
        tree1 = TraceToTree(deepcopy(events))
        analyzer1 = TreePerfAnalyzer(tree1, add_python_func=False)
        launchers1 = analyzer1.get_kernel_launchers()

        # With python func
        tree2 = TraceToTree(deepcopy(events))
        analyzer2 = TreePerfAnalyzer(tree2, add_python_func=True)
        launchers2 = analyzer2.get_kernel_launchers()

        # Should have same launcher
        assert len(launchers1) == 1
        assert len(launchers2) == 1
        assert launchers1[0]["name"] == "vllm::moe_forward"
        assert launchers2[0]["name"] == "vllm::moe_forward"
        assert launchers1[0]["direct_kernel_count"] == 2
        assert launchers2[0]["direct_kernel_count"] == 2
