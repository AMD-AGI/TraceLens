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
from example_megatron_extension import (
    tree_postprocess_extension,
    _link_checkpoint_fwd_bwd,
    perf_model_extension,
    dict_cat2names_extension,
    te_layer_norm_fwd,
    te_layer_norm_bwd,
)


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


class TestFusedAttnFuncBackwardCategorization:
    """Test that FusedAttnFuncBackward is categorized as SDPA_bwd."""

    def test_categorization_via_dict_cat2names(self):
        """FusedAttnFuncBackward must be in SDPA category."""
        assert "FusedAttnFuncBackward" in dict_cat2names_extension["SDPA"]

    def test_categorize_as_sdpa_bwd(self):
        """Core categorizer must return SDPA_bwd for FusedAttnFuncBackward."""
        from TraceLens.PerfModel.torch_op_mapping import (
            categorize_torch_op,
            dict_cat2names,
        )

        dict_cat2names["SDPA"].extend(dict_cat2names_extension["SDPA"])
        try:
            result = categorize_torch_op({"name": "FusedAttnFuncBackward"})
            assert result == "SDPA_bwd", f"Expected SDPA_bwd, got {result}"
        finally:
            for name in dict_cat2names_extension["SDPA"]:
                if name in dict_cat2names["SDPA"]:
                    dict_cat2names["SDPA"].remove(name)

    def test_fused_attn_fwd_still_sdpa_fwd(self):
        """FusedAttnFunc (forward) must still be SDPA_fwd."""
        from TraceLens.PerfModel.torch_op_mapping import (
            categorize_torch_op,
            dict_cat2names,
        )

        dict_cat2names["SDPA"].extend(dict_cat2names_extension["SDPA"])
        try:
            result = categorize_torch_op({"name": "FusedAttnFunc"})
            assert result == "SDPA_fwd", f"Expected SDPA_fwd, got {result}"
        finally:
            for name in dict_cat2names_extension["SDPA"]:
                if name in dict_cat2names["SDPA"]:
                    dict_cat2names["SDPA"].remove(name)


class TestLayerNormFnPerfModel:
    """Test LayerNormFn / LayerNormFnBackward perf model and categorization."""

    def test_layer_norm_fn_fwd_perf_model(self):
        """te_layer_norm_fwd must parse TE's arg layout and compute flops/bytes."""
        event = {
            "args": {
                "Input Dims": [(2048, 4, 2048), (2048,), ()],
                "Input type": ["c10::BFloat16", "c10::BFloat16", ""],
                "Input Strides": [(8192, 2048, 1), (1,), ()],
                "Concrete Inputs": ["", "", "", "", "1e-05", "256", "False", "True"],
                "Sequence number": 1,
                "External id": 1,
            }
        }
        model = te_layer_norm_fwd(event)
        assert model.num_channels == 2048
        assert model.num_elems == 2048 * 4 * 2048
        assert model.flops() > 0
        assert model.bytes() > 0

    def test_layer_norm_fn_bwd_perf_model(self):
        """te_layer_norm_bwd must parse gradient tensor and compute flops/bytes."""
        event = {
            "args": {
                "Input Dims": [(2048, 4, 2048)],
                "Input type": ["c10::BFloat16"],
                "Input Strides": [(8192, 2048, 1)],
                "Concrete Inputs": [""],
                "Sequence number": 1,
                "External id": 1,
            }
        }
        model = te_layer_norm_bwd(event)
        assert model.num_channels == 2048
        assert model.num_elems == 2048 * 4 * 2048
        assert model.flops() > 0
        assert model.bytes() > 0

    def test_categorization_normalization(self):
        """LayerNormFn/LayerNormFnBackward must be in Normalization category."""
        assert "LayerNormFn" in dict_cat2names_extension["Normalization"]
        assert "LayerNormFnBackward" in dict_cat2names_extension["Normalization"]

    def test_categorize_as_norm_fwd_bwd(self):
        """Core categorizer must return NORM_fwd and NORM_bwd."""
        from TraceLens.PerfModel.torch_op_mapping import (
            categorize_torch_op,
            dict_cat2names,
        )

        dict_cat2names["Normalization"].extend(
            dict_cat2names_extension["Normalization"]
        )
        try:
            assert categorize_torch_op({"name": "LayerNormFn"}) == "NORM_fwd"
            assert categorize_torch_op({"name": "LayerNormFnBackward"}) == "NORM_bwd"
        finally:
            for name in dict_cat2names_extension["Normalization"]:
                if name in dict_cat2names["Normalization"]:
                    dict_cat2names["Normalization"].remove(name)

    def test_perf_model_extension_registration(self):
        """LayerNormFn/LayerNormFnBackward must be registered in perf_model_extension."""
        assert "LayerNormFn" in perf_model_extension
        assert "LayerNormFnBackward" in perf_model_extension
        assert perf_model_extension["LayerNormFn"] is te_layer_norm_fwd
        assert perf_model_extension["LayerNormFnBackward"] is te_layer_norm_bwd


class TestActivationCheckpointingPseudoOps:
    """Test pseudo-op creation under CheckpointFunctionBackward."""

    def _build_checkpoint_trace(self):
        """Build a trace with _Linear inside CheckpointFunctionBackward.

        Simulates activation checkpointing where the forward ops are recomputed
        inside the backward context, and bwd_events is NOT wired (the bug).

        Mirrors real trace structure where backward ops are wrapped in
        ``autograd::engine::evaluate_function:`` nodes::

            CheckpointFunctionBackward
            ├── _Linear                               (direct child)
            ├── autograd::engine::evaluate_function: _LinearBackward
            │   └── _LinearBackward                   (grandchild)
        """
        corr_fwd = 200
        corr_xgrad = 201
        corr_wgrad = 202

        checkpoint_op = _mk_event(
            "cpu_op",
            "CheckpointFunctionBackward",
            ts=900,
            dur=2000,
            pid=100,
            tid=100,
            args={"Sequence number": 0, "External id": 0},
        )

        fwd_op = _mk_event(
            "cpu_op",
            "_Linear",
            ts=1000,
            dur=100,
            pid=100,
            tid=100,
            args={
                "Input Dims": [[1024, 512], [20, 512]],
                "Input type": ["c10::BFloat16", "c10::BFloat16"],
                "Input Strides": [[512, 1], [512, 1]],
                "Concrete Inputs": ["", "", "False", "True", "False"],
                "Sequence number": 1,
                "External id": 1,
            },
        )

        autograd_wrapper = _mk_event(
            "cpu_op",
            "autograd::engine::evaluate_function: _LinearBackward",
            ts=1990,
            dur=200,
            pid=100,
            tid=100,
            args={"Sequence number": 0, "External id": 0},
        )

        bwd_op = _mk_event(
            "cpu_op",
            "_LinearBackward",
            ts=2000,
            dur=150,
            pid=100,
            tid=100,
            args={
                "Input Dims": [[1024, 512], [20, 512]],
                "Input type": ["c10::BFloat16", "c10::BFloat16"],
                "Input Strides": [[512, 1], [512, 1]],
                "Concrete Inputs": ["", "", "False", "True", "False"],
                "Sequence number": 1,
                "External id": 2,
            },
        )

        events = [
            checkpoint_op,
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
            autograd_wrapper,
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

        # Wire tree structure to match real traces:
        # _Linear is a direct child of CheckpointFunctionBackward
        # _LinearBackward is a child of autograd_wrapper, which is a child of checkpoint
        fwd_events = [e for e in analyzer.tree.events if e.get("name") == "_Linear"]
        bwd_events = [
            e for e in analyzer.tree.events if e.get("name") == "_LinearBackward"
        ]
        ckpt_events = [
            e
            for e in analyzer.tree.events
            if e.get("name") == "CheckpointFunctionBackward"
        ]
        wrapper_events = [
            e
            for e in analyzer.tree.events
            if e.get("name") == "autograd::engine::evaluate_function: _LinearBackward"
        ]

        assert len(ckpt_events) == 1
        assert len(wrapper_events) == 1
        ckpt = ckpt_events[0]
        wrapper = wrapper_events[0]
        ckpt.setdefault("children", [])

        for evt in fwd_events:
            evt["parent"] = ckpt["UID"]
            if evt["UID"] not in ckpt["children"]:
                ckpt["children"].append(evt["UID"])

        wrapper["parent"] = ckpt["UID"]
        if wrapper["UID"] not in ckpt["children"]:
            ckpt["children"].append(wrapper["UID"])

        wrapper.setdefault("children", [])
        for evt in bwd_events:
            evt["parent"] = wrapper["UID"]
            if evt["UID"] not in wrapper["children"]:
                wrapper["children"].append(evt["UID"])

        # Do NOT wire bwd_events — this is the bug we're testing
        return analyzer

    def test_checkpoint_pseudo_ops_created(self):
        """Pseudo ops must be created even when bwd_events is not pre-wired."""
        analyzer = self._build_checkpoint_trace()

        fwd_events = [e for e in analyzer.tree.events if e.get("name") == "_Linear"]
        assert not fwd_events[0].get(
            "bwd_events"
        ), "bwd_events should NOT be pre-wired for this test"

        tree_postprocess_extension(analyzer.tree)

        pseudo_ops = [
            e
            for e in analyzer.tree.events
            if e.get("args", {}).get("Pseudo op") is True
        ]
        assert len(pseudo_ops) > 0, (
            "Pseudo ops should be created for _Linear inside "
            "CheckpointFunctionBackward"
        )

        pseudo_names = {p["name"] for p in pseudo_ops}
        assert "_Linear_yfwd_mm" in pseudo_names
        assert "_LinearBackward_xgrad_mm" in pseudo_names
        assert "_LinearBackward_wgrad_mm" in pseudo_names

    def test_checkpoint_bwd_events_linked(self):
        """_link_checkpoint_fwd_bwd must wire bwd_events for checkpoint contexts."""
        analyzer = self._build_checkpoint_trace()

        fwd_events = [e for e in analyzer.tree.events if e.get("name") == "_Linear"]
        assert not fwd_events[0].get("bwd_events")

        _link_checkpoint_fwd_bwd(analyzer.tree)

        assert fwd_events[0].get(
            "bwd_events"
        ), "bwd_events should be wired after _link_checkpoint_fwd_bwd"
        assert len(fwd_events[0]["bwd_events"]) == 1
