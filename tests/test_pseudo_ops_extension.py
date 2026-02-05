###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for pseudo ops created by extensions.

Tests verify that:
1. Pseudo ops created by extensions appear in get_kernel_launchers()
2. Pseudo ops appear in ops_summary when generating perf reports
3. Parent pointers are properly rewired (pseudo ops are in parent chain)
"""

import pytest
from typing import Dict
from copy import deepcopy
import sys
import os

# Add examples to path to import extension
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

from TraceLens.Trace2Tree.trace_to_tree import TraceToTree
from TraceLens.TreePerf.tree_perf import TreePerfAnalyzer
from example_megatron_extension import tree_postprocess_extension


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


class TestPseudoOpsExtension:
    """Test that pseudo ops created by extensions work correctly."""

    def test_pseudo_ops_appear_in_kernel_launchers(self):
        """Test that pseudo ops appear in get_kernel_launchers() after extension."""
        corr_fwd = 100
        corr_xgrad = 101
        corr_wgrad = 102

        # Forward pass
        # For _Linear: args[0] = weight [out_features, in_features], args[1] = input [batch, in_features]
        # Extension checks: inp_shape[-1] == W_shape[1], so [20, 512] and [1024, 512] works
        fwd_op = _mk_event(
            "cpu_op",
            "_Linear",
            ts=1000,
            dur=100,
            pid=100,
            tid=100,
            args={
                "Input Dims": [
                    [1024, 512],
                    [20, 512],
                ],  # [weight [out,in], input [batch,in]]
                "Input type": ["c10::BFloat16", "c10::BFloat16"],
                "Input Strides": [[512, 1], [512, 1]],
                "Concrete Inputs": ["", "", "False", "True", "False"],
                "Sequence number": 1,
                "External id": 1,
            },
        )

        # Backward pass (needed for extension to create pseudo ops)
        bwd_op = _mk_event(
            "cpu_op",
            "_LinearBackward",
            ts=2000,
            dur=150,
            pid=100,
            tid=100,
            args={
                "Input Dims": [
                    [1024, 512],
                    [20, 512],
                ],  # Same shape structure as forward
                "Input type": ["c10::BFloat16", "c10::BFloat16"],
                "Input Strides": [[512, 1], [512, 1]],
                "Concrete Inputs": ["", "", "False", "True", "False"],
                "Sequence number": 1,
                "External id": 2,
            },
        )

        events = [
            fwd_op,
            _mk_event(
                "cuda_runtime",
                "cuLaunchKernelEx",
                ts=1010,
                dur=5,
                pid=100,
                tid=100,
                args={"correlation": corr_fwd},
            ),
            _mk_event(
                "kernel",
                "nvjet_gemm_kernel",
                ts=1050,
                dur=50,
                pid=0,
                tid=7,
                args={"correlation": corr_fwd, "stream": 7},
            ),
            _mk_ac2g(corr_fwd, pid=0, tid=7, ts=1050, phase="s"),
            _mk_ac2g(corr_fwd, pid=0, tid=7, ts=1050, phase="f"),
            bwd_op,
            _mk_event(
                "cuda_runtime",
                "cuLaunchKernelEx",
                ts=2010,
                dur=5,
                pid=100,
                tid=100,
                args={"correlation": corr_xgrad},
            ),
            _mk_event(
                "kernel",
                "nvjet_gemm_kernel_xgrad",
                ts=2050,
                dur=60,
                pid=0,
                tid=7,
                args={"correlation": corr_xgrad, "stream": 7},
            ),
            _mk_ac2g(corr_xgrad, pid=0, tid=7, ts=2050, phase="s"),
            _mk_ac2g(corr_xgrad, pid=0, tid=7, ts=2050, phase="f"),
            _mk_event(
                "cuda_runtime",
                "cuLaunchKernelEx",
                ts=2110,
                dur=5,
                pid=100,
                tid=100,
                args={"correlation": corr_wgrad},
            ),
            _mk_event(
                "kernel",
                "nvjet_gemm_kernel_wgrad",
                ts=2150,
                dur=70,
                pid=0,
                tid=7,
                args={"correlation": corr_wgrad, "stream": 7},
            ),
            _mk_ac2g(corr_wgrad, pid=0, tid=7, ts=2150, phase="s"),
            _mk_ac2g(corr_wgrad, pid=0, tid=7, ts=2150, phase="f"),
        ]

        trace = {"traceEvents": events}
        tree = TraceToTree(deepcopy(trace["traceEvents"]))
        analyzer = TreePerfAnalyzer(tree, add_python_func=False)

        # Link backward to forward (extension needs this)
        # Find the actual events in the tree
        fwd_events = [e for e in analyzer.tree.events if e.get("name") == "_Linear"]
        bwd_events = [
            e for e in analyzer.tree.events if e.get("name") == "_LinearBackward"
        ]
        if fwd_events and bwd_events:
            fwd_events[0]["bwd_events"] = [bwd_events[0]["UID"]]

        # Apply extension
        tree_postprocess_extension(analyzer.tree)

        # Get kernel launchers
        kernel_launchers = analyzer.get_kernel_launchers()

        # Verify pseudo op was created
        pseudo_ops_in_tree = [
            e
            for e in analyzer.tree.events
            if e.get("args", {}).get("Pseudo op") == True
        ]
        assert len(pseudo_ops_in_tree) > 0, "Pseudo ops should be created by extension"

        # Verify pseudo ops appear in kernel_launchers
        pseudo_launchers = [
            kl for kl in kernel_launchers if kl.get("args", {}).get("Pseudo op") == True
        ]
        assert (
            len(pseudo_launchers) > 0
        ), "Pseudo ops should appear in get_kernel_launchers()"

        # Verify pseudo op names
        pseudo_names = {kl["name"] for kl in pseudo_launchers}
        assert (
            "_Linear_yfwd_mm" in pseudo_names
            or "_LinearBackward_xgrad_mm" in pseudo_names
            or "_LinearBackward_wgrad_mm" in pseudo_names
        ), f"Should have pseudo ops. Found: {pseudo_names}"

    def test_pseudo_ops_in_parent_chain(self):
        """Test that pseudo ops are in the parent chain when walking up from kernels."""
        # Use the same setup as test_pseudo_ops_appear_in_kernel_launchers
        corr_fwd = 100
        corr_xgrad = 101
        corr_wgrad = 102

        fwd_op = _mk_event(
            "cpu_op",
            "_Linear",
            ts=1000,
            dur=100,
            pid=100,
            tid=100,
            args={
                "Input Dims": [
                    [1024, 512],
                    [20, 512],
                ],  # [weight [out,in], input [batch,in]]
                "Input type": ["c10::BFloat16", "c10::BFloat16"],
                "Input Strides": [[512, 1], [512, 1]],
                "Concrete Inputs": ["", "", "False", "True", "False"],
                "Sequence number": 1,
                "External id": 1,
            },
        )

        bwd_op = _mk_event(
            "cpu_op",
            "_LinearBackward",
            ts=2000,
            dur=150,
            pid=100,
            tid=100,
            args={
                "Input Dims": [
                    [1024, 512],
                    [20, 512],
                ],  # Same shape structure as forward
                "Input type": ["c10::BFloat16", "c10::BFloat16"],
                "Input Strides": [[512, 1], [512, 1]],
                "Concrete Inputs": ["", "", "False", "True", "False"],
                "Sequence number": 1,
                "External id": 2,
            },
        )

        events = [
            fwd_op,
            _mk_event(
                "cuda_runtime",
                "cuLaunchKernelEx",
                ts=1010,
                dur=5,
                pid=100,
                tid=100,
                args={"correlation": corr_fwd},
            ),
            _mk_event(
                "kernel",
                "nvjet_gemm_kernel",
                ts=1050,
                dur=50,
                pid=0,
                tid=7,
                args={"correlation": corr_fwd, "stream": 7},
            ),
            _mk_ac2g(corr_fwd, pid=0, tid=7, ts=1050, phase="s"),
            _mk_ac2g(corr_fwd, pid=0, tid=7, ts=1050, phase="f"),
            bwd_op,
            _mk_event(
                "cuda_runtime",
                "cuLaunchKernelEx",
                ts=2010,
                dur=5,
                pid=100,
                tid=100,
                args={"correlation": corr_xgrad},
            ),
            _mk_event(
                "kernel",
                "nvjet_gemm_kernel_xgrad",
                ts=2050,
                dur=60,
                pid=0,
                tid=7,
                args={"correlation": corr_xgrad, "stream": 7},
            ),
            _mk_ac2g(corr_xgrad, pid=0, tid=7, ts=2050, phase="s"),
            _mk_ac2g(corr_xgrad, pid=0, tid=7, ts=2050, phase="f"),
            _mk_event(
                "cuda_runtime",
                "cuLaunchKernelEx",
                ts=2110,
                dur=5,
                pid=100,
                tid=100,
                args={"correlation": corr_wgrad},
            ),
            _mk_event(
                "kernel",
                "nvjet_gemm_kernel_wgrad",
                ts=2150,
                dur=70,
                pid=0,
                tid=7,
                args={"correlation": corr_wgrad, "stream": 7},
            ),
            _mk_ac2g(corr_wgrad, pid=0, tid=7, ts=2150, phase="s"),
            _mk_ac2g(corr_wgrad, pid=0, tid=7, ts=2150, phase="f"),
        ]

        trace = {"traceEvents": events}
        tree = TraceToTree(deepcopy(trace["traceEvents"]))
        analyzer = TreePerfAnalyzer(tree, add_python_func=False)

        # Link backward to forward
        fwd_events = [e for e in analyzer.tree.events if e.get("name") == "_Linear"]
        bwd_events = [
            e for e in analyzer.tree.events if e.get("name") == "_LinearBackward"
        ]
        if fwd_events and bwd_events:
            fwd_events[0]["bwd_events"] = [bwd_events[0]["UID"]]

        # Apply extension
        tree_postprocess_extension(analyzer.tree)

        # Find a kernel that should be launched by a pseudo op
        kernels = [e for e in analyzer.tree.events if e.get("cat") == "kernel"]
        assert len(kernels) > 0, "Should have kernels"
        kernel = kernels[0]

        # Find pseudo op
        pseudo_ops = [
            e
            for e in analyzer.tree.events
            if e.get("args", {}).get("Pseudo op") == True
        ]
        assert len(pseudo_ops) > 0, "Pseudo op should be created"
        pseudo_op = pseudo_ops[0]
        pseudo_uid = pseudo_op["UID"]

        # Walk up parent chain from kernel and collect UIDs
        parent_uids = []
        current = kernel
        while current:
            parent = analyzer.tree.get_parent_event(current)
            if not parent:
                break
            parent_uids.append(parent["UID"])
            current = parent
            if len(parent_uids) > 10:
                break

        # Verify pseudo op UID is in parent chain
        assert (
            pseudo_uid in parent_uids
        ), f"Pseudo op (UID: {pseudo_uid}) should be in parent chain. Chain UIDs: {parent_uids}, Chain names: {[analyzer.tree.get_UID2event(uid).get('name') for uid in parent_uids]}"

    def test_pseudo_ops_in_ops_summary(self):
        """Test that pseudo ops appear in ops_summary DataFrame."""
        # Use the same setup as test_pseudo_ops_appear_in_kernel_launchers
        corr_fwd = 100
        corr_xgrad = 101
        corr_wgrad = 102

        fwd_op = _mk_event(
            "cpu_op",
            "_Linear",
            ts=1000,
            dur=100,
            pid=100,
            tid=100,
            args={
                "Input Dims": [
                    [1024, 512],
                    [20, 512],
                ],  # [weight [out,in], input [batch,in]]
                "Input type": ["c10::BFloat16", "c10::BFloat16"],
                "Input Strides": [[512, 1], [512, 1]],
                "Concrete Inputs": ["", "", "False", "True", "False"],
                "Sequence number": 1,
                "External id": 1,
            },
        )

        bwd_op = _mk_event(
            "cpu_op",
            "_LinearBackward",
            ts=2000,
            dur=150,
            pid=100,
            tid=100,
            args={
                "Input Dims": [
                    [1024, 512],
                    [20, 512],
                ],  # Same shape structure as forward
                "Input type": ["c10::BFloat16", "c10::BFloat16"],
                "Input Strides": [[512, 1], [512, 1]],
                "Concrete Inputs": ["", "", "False", "True", "False"],
                "Sequence number": 1,
                "External id": 2,
            },
        )

        events = [
            fwd_op,
            _mk_event(
                "cuda_runtime",
                "cuLaunchKernelEx",
                ts=1010,
                dur=5,
                pid=100,
                tid=100,
                args={"correlation": corr_fwd},
            ),
            _mk_event(
                "kernel",
                "nvjet_gemm_kernel",
                ts=1050,
                dur=50,
                pid=0,
                tid=7,
                args={"correlation": corr_fwd, "stream": 7},
            ),
            _mk_ac2g(corr_fwd, pid=0, tid=7, ts=1050, phase="s"),
            _mk_ac2g(corr_fwd, pid=0, tid=7, ts=1050, phase="f"),
            bwd_op,
            _mk_event(
                "cuda_runtime",
                "cuLaunchKernelEx",
                ts=2010,
                dur=5,
                pid=100,
                tid=100,
                args={"correlation": corr_xgrad},
            ),
            _mk_event(
                "kernel",
                "nvjet_gemm_kernel_xgrad",
                ts=2050,
                dur=60,
                pid=0,
                tid=7,
                args={"correlation": corr_xgrad, "stream": 7},
            ),
            _mk_ac2g(corr_xgrad, pid=0, tid=7, ts=2050, phase="s"),
            _mk_ac2g(corr_xgrad, pid=0, tid=7, ts=2050, phase="f"),
            _mk_event(
                "cuda_runtime",
                "cuLaunchKernelEx",
                ts=2110,
                dur=5,
                pid=100,
                tid=100,
                args={"correlation": corr_wgrad},
            ),
            _mk_event(
                "kernel",
                "nvjet_gemm_kernel_wgrad",
                ts=2150,
                dur=70,
                pid=0,
                tid=7,
                args={"correlation": corr_wgrad, "stream": 7},
            ),
            _mk_ac2g(corr_wgrad, pid=0, tid=7, ts=2150, phase="s"),
            _mk_ac2g(corr_wgrad, pid=0, tid=7, ts=2150, phase="f"),
        ]

        trace = {"traceEvents": events}
        tree = TraceToTree(deepcopy(trace["traceEvents"]))
        analyzer = TreePerfAnalyzer(tree, add_python_func=False)

        # Link backward to forward
        fwd_events = [e for e in analyzer.tree.events if e.get("name") == "_Linear"]
        bwd_events = [
            e for e in analyzer.tree.events if e.get("name") == "_LinearBackward"
        ]
        if fwd_events and bwd_events:
            fwd_events[0]["bwd_events"] = [bwd_events[0]["UID"]]

        # Apply extension
        tree_postprocess_extension(analyzer.tree)

        # Generate ops_summary
        df_kernel_launchers = analyzer.get_df_kernel_launchers()
        df_ops_summary = analyzer.get_df_kernel_launchers_summary(df_kernel_launchers)

        # Verify pseudo ops appear in ops_summary
        pseudo_ops_in_summary = df_ops_summary[
            df_ops_summary["name"].str.contains(
                "_yfwd_mm|_xgrad_mm|_wgrad_mm", na=False, regex=True
            )
        ]

        assert (
            len(pseudo_ops_in_summary) > 0
        ), "Pseudo ops should appear in ops_summary DataFrame"

        # Verify specific pseudo op names
        pseudo_names = set(pseudo_ops_in_summary["name"].values)
        assert (
            "_Linear_yfwd_mm" in pseudo_names
            or "_LinearBackward_xgrad_mm" in pseudo_names
            or "_LinearBackward_wgrad_mm" in pseudo_names
        ), f"Should have pseudo ops in ops_summary. Found: {pseudo_names}"
