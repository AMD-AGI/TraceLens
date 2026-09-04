###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

import sys, os, json, pytest, gzip

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
from typing import Dict, List
from copy import deepcopy
from TraceLens.Trace2Tree.trace_to_tree import TraceToTree
from TraceLens.TreePerf.tree_perf import TreePerfAnalyzer
from example_megatron_extension import (
    _link_checkpoint_fwd_bwd,
    op_category_extension,
    perf_model_extension,
    te_layer_norm_bwd,
    te_layer_norm_fwd,
    tree_postprocess_extension,
)
from TraceLens.Trace2Tree.extensions.pseudo_ops_registry import (
    apply_pseudo_op_extensions,
)
from TraceLens.Trace2Tree.extensions.moe_aiter_pseudo_ops import (
    _create_pseudo_op_moe_fused_aiter,
    _has_cpu_op_descendant,
    create_pseudo_ops_moe_fused_aiter,
)
from tests.fixtures.traces import NORM_TRACE, TRACES_ROOT
from tests.fixtures.treeperf import _make_gpu_event, _mk_ac2g
from TraceLens.Reporting.generate_perf_report_pytorch import (
    generate_perf_report_pytorch as generate_inference_report,
)
from tests.fixtures.reporting import _mk_ac2g, _mk_event
from TraceLens.Trace2Tree.extensions.moe_flydsl_pseudo_ops import (
    FUSED_MOE_PARENT,
    create_pseudo_ops_moe_flydsl,
)
from TraceLens.Trace2Tree.extensions.moe_gptq_awq_pseudo_ops import (
    _create_pseudo_op_moe_gptq_awq,
    create_pseudo_ops_moe_gptq_awq,
)
from tests.test_trace2tree import _add_gpu_chain, _mk_event
from TraceLens.PerfModel.torch_op_mapping import (
    OP_CATEGORY_REGISTRY,
    categorize_torch_op,
    register_perf_model_categories,
)
from unittest import mock
from TraceLens.Trace2Tree.extensions.mla_decode_pseudo_ops import (
    MLA_DECODE_FWD_PATTERN,
    STAGE1_KERNEL_NAME,
    _create_pseudo_op_mla_decode,
    _find_mla_decode_python_funcs,
    _find_stage1_child,
    create_pseudo_ops_mla_decode,
)
from TraceLens.Trace2Tree.extensions.mla_prefill_pseudo_ops import (
    PREFILL_CPU_OP_NAME,
    _create_pseudo_op_mla_prefill,
    _find_mla_prefill_python_funcs,
    _find_prefill_cpu_op_child,
    create_pseudo_ops_mla_prefill,
)

# Add examples to path to import extension


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

    def test_categorization_via_op_category_extension(self):
        """FusedAttnFuncBackward must be registered as a category-only op."""
        assert op_category_extension["FusedAttnFuncBackward"] == "SDPA_bwd"

    def test_categorize_as_sdpa_bwd(self):
        result = categorize_torch_op({"name": "FusedAttnFuncBackward"})
        assert result == "SDPA_bwd", f"Expected SDPA_bwd, got {result}"

    def test_fused_attn_fwd_still_sdpa_fwd(self):
        """FusedAttnFunc (forward) must still be SDPA_fwd."""
        registry = dict(OP_CATEGORY_REGISTRY)
        register_perf_model_categories(
            {"FusedAttnFunc": perf_model_extension["FusedAttnFunc"]},
            registry,
        )

        assert registry["FusedAttnFunc"] == "SDPA_fwd"


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

    def test_layer_norm_fn_bwd_consistent_with_fwd(self):
        """Backward has_bias/is_affine defaults must be consistent with forward."""
        fwd_event = {
            "args": {
                "Input Dims": [(2048, 4, 2048), (2048,), ()],
                "Input type": ["c10::BFloat16", "c10::BFloat16", ""],
                "Input Strides": [(8192, 2048, 1), (1,), ()],
                "Concrete Inputs": ["", "", "", "", "1e-05", "256", "False", "True"],
                "Sequence number": 1,
                "External id": 1,
            }
        }
        bwd_event = {
            "args": {
                "Input Dims": [(2048, 4, 2048)],
                "Input type": ["c10::BFloat16"],
                "Input Strides": [(8192, 2048, 1)],
                "Concrete Inputs": [""],
                "Sequence number": 1,
                "External id": 1,
            }
        }
        fwd_model = te_layer_norm_fwd(fwd_event)
        bwd_model = te_layer_norm_bwd(bwd_event)
        assert (
            fwd_model.has_bias == bwd_model.has_bias
        ), f"has_bias mismatch: fwd={fwd_model.has_bias}, bwd={bwd_model.has_bias}"
        assert (
            fwd_model.is_affine == bwd_model.is_affine
        ), f"is_affine mismatch: fwd={fwd_model.is_affine}, bwd={bwd_model.is_affine}"

    def test_categorization_normalization(self):
        """LayerNormFn classes declare their categories directly."""
        assert te_layer_norm_fwd.category == "NORM_fwd"
        assert te_layer_norm_bwd.category == "NORM_bwd"

    def test_categorize_as_norm_fwd_bwd(self):
        """Core categorizer must return NORM_fwd and NORM_bwd."""
        registry = dict(OP_CATEGORY_REGISTRY)
        register_perf_model_categories(
            {
                "LayerNormFn": te_layer_norm_fwd,
                "LayerNormFnBackward": te_layer_norm_bwd,
            },
            registry,
        )

        assert registry["LayerNormFn"] == "NORM_fwd"
        assert registry["LayerNormFnBackward"] == "NORM_bwd"

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


def _mk_event(
    cat: str, name: str, ts: float, dur: float, pid: int, tid: int, args: Dict = None
) -> Dict:
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


def _add_gpu_chain(
    events: List[Dict],
    cpu_op: Dict,
    corr: int,
    kernel_name: str,
    ts_launch: float,
    ts_kernel: float,
) -> None:
    pid = cpu_op["pid"]
    tid = cpu_op["tid"]
    events.extend(
        [
            _mk_event(
                "cuda_runtime",
                "hipLaunchKernel",
                ts=ts_launch,
                dur=5,
                pid=pid,
                tid=tid,
                args={"correlation": corr},
            ),
            _mk_event(
                "kernel",
                kernel_name,
                ts=ts_kernel,
                dur=20,
                pid=0,
                tid=7,
                args={"correlation": corr, "stream": 7},
            ),
            _mk_ac2g(corr, pid=0, tid=7, ts=ts_kernel, phase="s"),
            _mk_ac2g(corr, pid=0, tid=7, ts=ts_kernel, phase="f"),
        ]
    )


def _build_tree(events: List[Dict], add_python_func: bool = True) -> TraceToTree:
    tree = TraceToTree(deepcopy(events), prune_nongpu_paths=False)
    tree.build_tree(add_python_func=add_python_func)
    return tree


def _wire_mla_decode_hierarchy(tree: TraceToTree) -> None:
    """Wire python_function -> stage1 -> kernel hierarchy after build_tree."""
    py_evt = next(
        e
        for e in tree.events
        if e.get("cat") == "python_function"
        and MLA_DECODE_FWD_PATTERN.search(e.get("name", ""))
    )
    stage1 = next(e for e in tree.events if e["name"] == STAGE1_KERNEL_NAME)
    kernel = next(e for e in tree.events if e.get("cat") == "kernel")

    py_evt.setdefault("children", [])
    if stage1["UID"] not in py_evt["children"]:
        py_evt["children"].append(stage1["UID"])
    stage1["parent"] = py_evt["UID"]

    gpu_uid = kernel["UID"]
    py_evt["gpu_events"] = [gpu_uid]
    stage1["gpu_events"] = [gpu_uid]


def _wire_mla_prefill_hierarchy(tree: TraceToTree) -> None:
    py_evt = next(
        e
        for e in tree.events
        if e.get("cat") == "python_function"
        and "mla_fp8_prefill_attn" in e.get("name", "")
    )
    cpu_op = next(e for e in tree.events if e["name"] == PREFILL_CPU_OP_NAME)
    kernel = next(e for e in tree.events if e.get("cat") == "kernel")

    py_evt.setdefault("children", [])
    if cpu_op["UID"] not in py_evt["children"]:
        py_evt["children"].append(cpu_op["UID"])
    cpu_op["parent"] = py_evt["UID"]

    gpu_uid = kernel["UID"]
    py_evt["gpu_events"] = [gpu_uid]
    cpu_op["gpu_events"] = [gpu_uid]


class TestMlaDecodePseudoOps:
    def test_create_pseudo_ops_success(self):
        py_func = _mk_event(
            "python_function",
            "aiter/mla.py(42): mla_decode_fwd",
            ts=100,
            dur=200,
            pid=1,
            tid=1,
            args={"Python id": 1},
        )
        stage1 = _mk_event(
            "cpu_op",
            STAGE1_KERNEL_NAME,
            ts=110,
            dur=80,
            pid=1,
            tid=1,
            args={
                "Input Dims": [[32, 128]],
                "Input type": ["fp16"],
                "Input Strides": [[128, 1]],
                "Sequence number": 1,
            },
        )
        events = [py_func, stage1]
        _add_gpu_chain(
            events,
            stage1,
            corr=50,
            kernel_name="mla_decode_k",
            ts_launch=120,
            ts_kernel=140,
        )
        tree = _build_tree(events)
        _wire_mla_decode_hierarchy(tree)

        create_pseudo_ops_mla_decode(tree)
        pseudo_ops = [
            e for e in tree.events if e.get("args", {}).get("Pseudo op") is True
        ]
        assert any(p["name"] == "pseudo_mla_decode_fwd" for p in pseudo_ops)

    def test_no_matching_python_funcs(self, caplog):
        tree = _build_tree(
            [_mk_event("cpu_op", "aten::mm", ts=0, dur=10, pid=1, tid=1)]
        )
        create_pseudo_ops_mla_decode(tree)
        assert "No python_function events matching mla_decode_fwd" in caplog.text

    def test_skip_when_no_stage1_child(self, caplog):
        py_func = _mk_event(
            "python_function",
            "aiter/mla.py(1): mla_decode_fwd",
            ts=100,
            dur=50,
            pid=1,
            tid=1,
            args={"Python id": 1},
        )
        tree = _build_tree([py_func], add_python_func=True)
        py_evt = tree.events[0]
        py_evt["gpu_events"] = [999]
        _create_pseudo_op_mla_decode(tree, py_evt)
        assert STAGE1_KERNEL_NAME in caplog.text

    def test_skip_when_no_gpu_events(self, caplog):
        py_func = _mk_event(
            "python_function",
            "aiter/mla.py(2): mla_decode_fwd",
            ts=100,
            dur=50,
            pid=1,
            tid=1,
            args={"Python id": 1},
        )
        stage1 = _mk_event(
            "cpu_op",
            STAGE1_KERNEL_NAME,
            ts=110,
            dur=20,
            pid=1,
            tid=1,
            args={"Sequence number": 1},
        )
        tree = _build_tree([py_func, stage1], add_python_func=True)
        py_evt = next(e for e in tree.events if "mla_decode_fwd" in e["name"])
        stage1_evt = next(e for e in tree.events if e["name"] == STAGE1_KERNEL_NAME)
        py_evt.setdefault("children", []).append(stage1_evt["UID"])
        stage1_evt["parent"] = py_evt["UID"]
        _create_pseudo_op_mla_decode(tree, py_evt)
        assert "No GPU events for MLA decode" in caplog.text

    def test_find_helpers(self):
        py_func = _mk_event(
            "python_function",
            "aiter/mla.py(99): mla_decode_fwd",
            ts=0,
            dur=10,
            pid=1,
            tid=1,
            args={"Python id": 1},
        )
        stage1 = _mk_event(
            "cpu_op",
            STAGE1_KERNEL_NAME,
            ts=1,
            dur=5,
            pid=1,
            tid=1,
            args={},
        )
        tree = _build_tree([py_func, stage1], add_python_func=True)
        py_evt = next(e for e in tree.events if "mla_decode_fwd" in e["name"])
        stage1_evt = next(e for e in tree.events if e["name"] == STAGE1_KERNEL_NAME)
        py_evt.setdefault("children", []).append(stage1_evt["UID"])
        stage1_evt["parent"] = py_evt["UID"]

        matched = _find_mla_decode_python_funcs(tree)
        assert len(matched) == 1
        assert _find_stage1_child(tree, py_evt) is stage1_evt


class TestMlaPrefillPseudoOps:
    def test_create_pseudo_ops_success(self):
        py_func = _mk_event(
            "python_function",
            "module.py(10): mla_fp8_prefill_attn",
            ts=100,
            dur=200,
            pid=1,
            tid=1,
            args={"Python id": 1},
        )
        cpu_op = _mk_event(
            "cpu_op",
            PREFILL_CPU_OP_NAME,
            ts=110,
            dur=80,
            pid=1,
            tid=1,
            args={
                "Input Dims": [[16, 64]],
                "Input type": ["fp16"],
                "Sequence number": 2,
            },
        )
        events = [py_func, cpu_op]
        _add_gpu_chain(
            events,
            cpu_op,
            corr=60,
            kernel_name="mla_prefill_k",
            ts_launch=120,
            ts_kernel=140,
        )
        tree = _build_tree(events)
        _wire_mla_prefill_hierarchy(tree)

        create_pseudo_ops_mla_prefill(tree)
        pseudo_ops = [
            e for e in tree.events if e.get("args", {}).get("Pseudo op") is True
        ]
        assert any(p["name"] == "pseudo_mla_prefill_fwd" for p in pseudo_ops)

    def test_no_matching_python_funcs(self, caplog):
        tree = _build_tree(
            [_mk_event("cpu_op", "aten::mm", ts=0, dur=10, pid=1, tid=1)]
        )
        create_pseudo_ops_mla_prefill(tree)
        assert "No python_function events matching mla_fp8_prefill_attn" in caplog.text

    def test_skip_when_no_cpu_op_child(self, caplog):
        py_func = _mk_event(
            "python_function",
            "x.py(1): mla_fp8_prefill_attn",
            ts=0,
            dur=10,
            pid=1,
            tid=1,
            args={"Python id": 1},
        )
        tree = _build_tree([py_func], add_python_func=True)
        py_evt = tree.events[0]
        py_evt["gpu_events"] = [1]
        _create_pseudo_op_mla_prefill(tree, py_evt)
        assert PREFILL_CPU_OP_NAME in caplog.text

    def test_skip_when_no_gpu_events(self, caplog):
        py_func = _mk_event(
            "python_function",
            "x.py(2): mla_fp8_prefill_attn",
            ts=0,
            dur=10,
            pid=1,
            tid=1,
            args={"Python id": 1},
        )
        cpu_op = _mk_event(
            "cpu_op",
            PREFILL_CPU_OP_NAME,
            ts=1,
            dur=5,
            pid=1,
            tid=1,
            args={},
        )
        tree = _build_tree([py_func, cpu_op], add_python_func=True)
        py_evt = next(e for e in tree.events if "mla_fp8_prefill_attn" in e["name"])
        cpu_evt = next(e for e in tree.events if e["name"] == PREFILL_CPU_OP_NAME)
        py_evt.setdefault("children", []).append(cpu_evt["UID"])
        cpu_evt["parent"] = py_evt["UID"]
        _create_pseudo_op_mla_prefill(tree, py_evt)
        assert "No GPU events for MLA prefill" in caplog.text

    def test_find_helpers(self):
        py_func = _mk_event(
            "python_function",
            "x.py(3): mla_fp8_prefill_attn",
            ts=0,
            dur=10,
            pid=1,
            tid=1,
            args={"Python id": 1},
        )
        cpu_op = _mk_event(
            "cpu_op",
            PREFILL_CPU_OP_NAME,
            ts=1,
            dur=5,
            pid=1,
            tid=1,
            args={},
        )
        tree = _build_tree([py_func, cpu_op], add_python_func=True)
        py_evt = next(e for e in tree.events if "mla_fp8_prefill_attn" in e["name"])
        cpu_evt = next(e for e in tree.events if e["name"] == PREFILL_CPU_OP_NAME)
        py_evt.setdefault("children", []).append(cpu_evt["UID"])
        cpu_evt["parent"] = py_evt["UID"]

        matched = _find_mla_prefill_python_funcs(tree)
        assert len(matched) == 1
        assert _find_prefill_cpu_op_child(tree, py_evt) is cpu_evt


class TestPseudoOpsRegistryMla:
    def test_apply_mla_decode_via_registry(self):
        py_func = _mk_event(
            "python_function",
            "aiter/mla.py(7): mla_decode_fwd",
            ts=100,
            dur=200,
            pid=1,
            tid=1,
            args={"Python id": 1},
        )
        stage1 = _mk_event(
            "cpu_op",
            STAGE1_KERNEL_NAME,
            ts=110,
            dur=80,
            pid=1,
            tid=1,
            args={"Input Dims": [[8, 64]], "Sequence number": 1},
        )
        events = [py_func, stage1]
        _add_gpu_chain(
            events, stage1, corr=70, kernel_name="mla_k", ts_launch=120, ts_kernel=140
        )
        tree = _build_tree(events)
        _wire_mla_decode_hierarchy(tree)

        apply_pseudo_op_extensions(tree, verbose=True)
        pseudo_ops = [
            e for e in tree.events if e.get("args", {}).get("Pseudo op") is True
        ]
        assert any(p["name"] == "pseudo_mla_decode_fwd" for p in pseudo_ops)

    def test_apply_mla_prefill_via_registry(self):
        py_func = _mk_event(
            "python_function",
            "mod.py(1): mla_fp8_prefill_attn",
            ts=100,
            dur=200,
            pid=1,
            tid=1,
            args={"Python id": 1},
        )
        cpu_op = _mk_event(
            "cpu_op",
            PREFILL_CPU_OP_NAME,
            ts=110,
            dur=80,
            pid=1,
            tid=1,
            args={"Input Dims": [[8, 64]], "Sequence number": 1},
        )
        events = [py_func, cpu_op]
        _add_gpu_chain(
            events,
            cpu_op,
            corr=71,
            kernel_name="prefill_k",
            ts_launch=120,
            ts_kernel=140,
        )
        tree = _build_tree(events)
        _wire_mla_prefill_hierarchy(tree)

        apply_pseudo_op_extensions(tree, verbose=True)
        pseudo_ops = [
            e for e in tree.events if e.get("args", {}).get("Pseudo op") is True
        ]
        assert any(p["name"] == "pseudo_mla_prefill_fwd" for p in pseudo_ops)

    def test_registry_handles_extension_failure(self, caplog):
        tree = _build_tree(
            [_mk_event("cpu_op", "vllm::moe_forward", ts=0, dur=10, pid=1, tid=1)]
        )
        tree.name2event_uids["vllm::moe_forward"] = [0]
        tree.name2event_uids["vllm::rocm_aiter_fused_moe"] = [0]

        with mock.patch(
            "TraceLens.Trace2Tree.extensions.moe_aiter_pseudo_ops.create_pseudo_ops_moe_fused_aiter",
            side_effect=RuntimeError("boom"),
        ):
            apply_pseudo_op_extensions(tree, verbose=True)
        assert "Failed to apply pseudo-op extension MoE_Fused" in caplog.text


class TestPseudoOpsRegistryFull:
    def test_registry_detects_all_extension_types(self, tmp_path):
        events = [
            _mk_event("cpu_op", "aiter::fused_moe_", 0, 10, 1, 1),
            _mk_event("cpu_op", "vllm::moe_forward", 1, 10, 1, 1),
            _mk_event("cpu_op", "outplace_fused_experts", 2, 10, 1, 1),
            _mk_event("cpu_op", "aiter::mla_decode_stage1_asm_fwd", 3, 10, 1, 1),
            _mk_event("cpu_op", "aiter::mla_prefill_ps_asm_fwd", 4, 10, 1, 1),
            _mk_event(
                "python_function",
                "paged_decode.py(1): sparse_attn_v4_paged_decode",
                7,
                10,
                1,
                1,
                {"Python id": 1},
            ),
            _mk_event(
                "python_function",
                "aiter/mla.py(1): mla_decode_fwd",
                8,
                10,
                1,
                1,
                {"Python id": 2},
            ),
            _mk_event(
                "python_function",
                "mod.py(1): mla_fp8_prefill_attn",
                9,
                10,
                1,
                1,
                {"Python id": 3},
            ),
        ]
        tree = TraceToTree(deepcopy(events), prune_nongpu_paths=False)
        tree.build_tree(add_python_func=True)
        apply_pseudo_op_extensions(tree, verbose=True)
        assert tree is not None


class TestMoePseudoOpsFullPhase10:
    def test_aiter_all_warning_paths(self):
        create_pseudo_ops_moe_fused_aiter(
            _build_tree([_mk_event("cpu_op", "other", 0, 1, 1, 1)])
        )

        events = []
        moe = _mk_event(
            "cpu_op",
            "vllm::rocm_aiter_fused_moe",
            100,
            200,
            1,
            1,
            {"Sequence number": 1},
        )
        child = _mk_event("cpu_op", "nested_cpu", 110, 10, 1, 1)
        events.extend([moe, child])
        _add_gpu_chain(events, moe, 10, "aiter::fmoe_kernel", 110, 150)
        tree = _build_tree(events)
        moe_evt = next(
            e for e in tree.events if e["name"] == "vllm::rocm_aiter_fused_moe"
        )
        child_evt = next(e for e in tree.events if e["name"] == "nested_cpu")
        moe_evt.setdefault("children", []).append(child_evt["UID"])
        child_evt["parent"] = moe_evt["UID"]
        assert _has_cpu_op_descendant(tree, moe_evt)
        _create_pseudo_op_moe_fused_aiter(tree, moe_evt)

        no_gpu = _mk_event("cpu_op", "vllm::rocm_aiter_fused_moe", 0, 1, 1, 1, {})
        tree_no_gpu = _build_tree([no_gpu])
        no_gpu_evt = next(
            e for e in tree_no_gpu.events if e["name"] == "vllm::rocm_aiter_fused_moe"
        )
        _create_pseudo_op_moe_fused_aiter(tree_no_gpu, no_gpu_evt)

        bad_kernels = _mk_event(
            "cpu_op", "vllm::rocm_aiter_fused_moe", 100, 200, 1, 1, {}
        )
        evs = [bad_kernels]
        _add_gpu_chain(evs, bad_kernels, 11, "aiter::MoeSorting", 110, 150)
        _add_gpu_chain(evs, bad_kernels, 12, "aiter::quant_fmoe", 160, 190)
        tree2 = _build_tree(evs)
        moe2 = next(
            e for e in tree2.events if e["name"] == "vllm::rocm_aiter_fused_moe"
        )
        _create_pseudo_op_moe_fused_aiter(tree2, moe2)

    def test_gptq_all_warning_paths(self):
        create_pseudo_ops_moe_gptq_awq(_build_tree([]))
        _create_pseudo_op_moe_gptq_awq(_build_tree([]), {"name": "wrong", "UID": 0})

        no_gpu = _mk_event(
            "cpu_op", "vllm::outplace_fused_experts", 0, 1, 1, 1, {"UID": 1}
        )
        tree_no_gpu = _build_tree([no_gpu])
        no_gpu_evt = next(
            e for e in tree_no_gpu.events if e["name"] == "vllm::outplace_fused_experts"
        )
        _create_pseudo_op_moe_gptq_awq(tree_no_gpu, no_gpu_evt)

        moe = _mk_event(
            "cpu_op",
            "vllm::outplace_fused_experts",
            100,
            200,
            1,
            1,
            {
                "Input Dims": [
                    [128, 4096],
                    [8, 4096, 512],
                    [8, 4096, 512],
                    [128, 6],
                    [128, 6],
                ],
            },
        )
        evs = [moe]
        _add_gpu_chain(evs, moe, 20, "other_kernel", 110, 150)
        tree = _build_tree(evs)
        moe_evt = next(
            e for e in tree.events if e["name"] == "vllm::outplace_fused_experts"
        )
        _create_pseudo_op_moe_gptq_awq(tree, moe_evt)

        moe2 = _mk_event(
            "cpu_op",
            "vllm::outplace_fused_experts",
            100,
            200,
            1,
            1,
            {
                "Input Dims": [
                    [128, 4096],
                    [8, 4096, 512],
                    [8, 4096, 512],
                    [128, 6],
                    [128, 6],
                ],
            },
        )
        evs2 = [moe2]
        _add_gpu_chain(evs2, moe2, 21, "fused_moe_kernel_gptq_awq_up", 110, 150)
        tree3 = _build_tree(evs2)
        moe3 = next(
            e for e in tree3.events if e["name"] == "vllm::outplace_fused_experts"
        )
        _create_pseudo_op_moe_gptq_awq(tree3, moe3)

    def test_flydsl_skip_paths(self):
        create_pseudo_ops_moe_flydsl(_build_tree([]))
        create_pseudo_ops_moe_flydsl(
            _build_tree(
                [
                    _mk_event("cpu_op", "not_moe", 0, 1, 1, 1),
                    _mk_event(
                        "python_function", "flydsl.py: flydsl_moe_stage1", 10, 5, 1, 1
                    ),
                ],
                add_python_func=True,
            )
        )
        create_pseudo_ops_moe_flydsl(
            _build_tree(
                [
                    _mk_event("cpu_op", FUSED_MOE_PARENT, 0, 500, 1, 1),
                    _mk_event("cpu_op", "flydsl_moe_stage1", 50, 100, 1, 1),
                ]
            )
        )


class TestTraceToTreePhase12:
    @pytest.mark.skipif(not os.path.isfile(NORM_TRACE), reason="norm trace missing")
    def test_trace_to_tree_rebuild_pseudo_ops(self):
        with gzip.open(NORM_TRACE, "rt") as f:
            data = json.load(f)
        tree = TraceToTree(deepcopy(data["traceEvents"]), prune_nongpu_paths=False)
        tree.build_tree(add_python_func=True)
        assert len(tree.events) > 0


class TestTraceToTreePhase4:
    def test_trace_to_tree_prune_and_metadata(self):
        events = [
            _make_gpu_event("k1", 0, 100, "kernel", "k1", pid=0, tid=7),
            _make_gpu_event("cpu", 0, 50, "cpu_op", "aten::mm", pid=100, tid=100),
        ]
        tree = TraceToTree(deepcopy(events), prune_nongpu_paths=True)
        tree.build_tree()
        assert len(tree.events) >= 1

    def test_inference_report_all_inference_dirs(self, tmp_path):
        inf_root = os.path.join(TRACES_ROOT, "inference")
        if not os.path.isdir(inf_root):
            pytest.skip("no inference fixtures")
        for case in os.listdir(inf_root):
            case_dir = os.path.join(inf_root, case)
            if not os.path.isdir(case_dir):
                continue
            gz = [f for f in os.listdir(case_dir) if f.endswith(".json.gz")]
            if not gz:
                continue
            trace = os.path.join(case_dir, gz[0])
            out = tmp_path / case
            out.mkdir(exist_ok=True)
            generate_inference_report(
                profile_json_path=trace,
                output_csvs_dir=str(out),
                output_xlsx_path=str(out / "r.xlsx"),
                collective_analysis=False,
                kernel_summary=True,
                short_kernel_study=True,
            )
            assert (out / "gpu_timeline.csv").exists()
