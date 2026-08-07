###############################################################################
# Copyright (c) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for TraceLens.Trace2Tree and related extension modules."""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, List

import pytest

from TraceLens.Trace2Tree.inference_iteration_roots import (
    _detect_iteration_roots_from_tree,
    _find_repeating_period,
    find_iteration_roots_generic,
)
from TraceLens.Trace2Tree.trace_capture_merge_experimental import (
    _align_capture_to_graph,
    _align_graph_to_capture_by_group,
    _capture_kernel_name,
    _names_match,
    _stream_of,
    align_streams,
    capture_has_kernel_names,
    get_subtree_events,
    is_multistream,
    verify_subtree_events,
)
from TraceLens.Trace2Tree.trace_to_tree import TraceToTree
from TraceLens.Trace2Tree.extensions.pseudo_ops_registry import (
    apply_pseudo_op_extensions,
)
from TraceLens.Trace2Tree.extensions.pseudo_ops_utils import (
    inject_pseudo_op,
    inject_pseudo_op_above_event,
    inject_pseudo_op_wrap_children,
    normalize_sglang_profiler_op_names,
)
from TraceLens.Trace2Tree.extensions.moe_aiter_pseudo_ops import (
    create_pseudo_ops_moe_fused_aiter,
    is_aiter_fused_moe_kernel,
)
from TraceLens.Trace2Tree.extensions.moe_unfused_triton_pseudo_ops import (
    _is_gated_kernel,
    is_matmul_ogs_kernel,
)
from TraceLens.Trace2Tree.extensions.moe_gptq_awq_pseudo_ops import (
    _extract_topk_from_outplace,
    is_fused_moe_gptq_awq_kernel,
)
from TraceLens.Trace2Tree.extensions.moe_flydsl_pseudo_ops import (
    FUSED_MOE_PARENT,
    _find_fused_moe_ancestor,
)
from TraceLens.Trace2Tree.extensions.v4_paged_decode_pseudo_ops import (
    _detect_v4_mode,
    _parse_geometry,
    _safe_int,
)


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


def _build_tree(events: List[Dict], add_python_func: bool = False) -> TraceToTree:
    tree = TraceToTree(deepcopy(events), prune_nongpu_paths=False)
    tree.build_tree(add_python_func=add_python_func)
    return tree


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
            cpu_op,
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


class TestInferenceIterationRoots:
    def test_find_repeating_period_skips_prefix(self):
        names = ["setup", "fwd", "bwd", "fwd", "bwd", "fwd", "bwd"]
        period, pattern, start = _find_repeating_period(names)
        assert period == 2
        assert pattern == ["fwd", "bwd"]
        assert start == 1

    def test_find_repeating_period_no_match(self):
        period, pattern, start = _find_repeating_period(["a", "b", "c", "d"])
        assert period is None
        assert pattern is None
        assert start is None

    def test_find_iteration_roots_from_synthetic_tree(self):
        events: List[Dict] = []
        loop = _mk_event(
            "cpu_op",
            "training_loop",
            ts=0,
            dur=7000,
            pid=1,
            tid=1,
            args={"Sequence number": 0},
        )
        events.append(loop)
        corr = 100
        for iteration in range(3):
            base_ts = 100 + iteration * 2000
            for step_name, offset in [("step_fwd", 0), ("step_bwd", 400)]:
                op = _mk_event(
                    "cpu_op",
                    step_name,
                    ts=base_ts + offset,
                    dur=300,
                    pid=1,
                    tid=1,
                    args={"Sequence number": iteration},
                )
                _add_gpu_chain(
                    events,
                    op,
                    corr,
                    f"kernel_{step_name}",
                    ts_launch=base_ts + offset + 10,
                    ts_kernel=base_ts + offset + 50,
                )
                corr += 1

        tree = _build_tree(events)
        loop_evt = next(e for e in tree.events if e["name"] == "training_loop")
        roots = _detect_iteration_roots_from_tree(tree, loop_evt)
        assert roots is not None
        assert len(roots) == 3
        assert all(root["dur"] > 0 for root in roots)

    def test_find_iteration_roots_generic_end_to_end(self):
        events: List[Dict] = []
        events.append(
            _mk_event(
                "cpu_op",
                "training_loop",
                ts=0,
                dur=7000,
                pid=1,
                tid=1,
                args={"Sequence number": 0},
            )
        )
        corr = 200
        for iteration in range(3):
            base_ts = 100 + iteration * 2000
            for step_name, offset in [("iter_fwd", 0), ("iter_bwd", 400)]:
                op = _mk_event(
                    "cpu_op",
                    step_name,
                    ts=base_ts + offset,
                    dur=300,
                    pid=1,
                    tid=1,
                    args={"Sequence number": iteration},
                )
                _add_gpu_chain(
                    events,
                    op,
                    corr,
                    f"{step_name}_kernel",
                    ts_launch=base_ts + offset + 10,
                    ts_kernel=base_ts + offset + 50,
                )
                corr += 1

        roots = find_iteration_roots_generic(events)
        assert roots is not None
        assert len(roots) >= 1


class TestPseudoOpsUtils:
    def test_normalize_sglang_profiler_op_names(self):
        events = [
            _mk_event(
                "cpu_op",
                "sglang_profiler::step_42",
                ts=0,
                dur=10,
                pid=1,
                tid=1,
                args={},
            )
        ]
        tree = _build_tree(events)
        normalize_sglang_profiler_op_names(tree)
        assert "sglang_profiler::step" in tree.name2event_uids
        assert "sglang_profiler::step_42" not in tree.name2event_uids
        renamed = tree.get_UID2event(tree.name2event_uids["sglang_profiler::step"][0])
        assert renamed["name"] == "sglang_profiler::step"

    def test_inject_pseudo_op_reparents_kernel(self):
        corr = 50
        events: List[Dict] = []
        cpu_op = _mk_event(
            "cpu_op",
            "aten::mm",
            ts=100,
            dur=100,
            pid=1,
            tid=1,
            args={
                "Input Dims": [[32, 64]],
                "Input type": ["fp16"],
                "Input Strides": [[64, 1]],
                "Concrete Inputs": [],
                "Sequence number": 1,
            },
        )
        _add_gpu_chain(events, cpu_op, corr, "gemm_kernel", 110, 150)
        tree = _build_tree(events)

        kernel = next(e for e in tree.events if e["cat"] == "kernel")
        inject_pseudo_op(tree, kernel, "pseudo::gemm", seq_num=99)

        pseudo_ops = [
            e for e in tree.events if e.get("args", {}).get("Pseudo op") is True
        ]
        assert len(pseudo_ops) == 1
        pseudo = pseudo_ops[0]
        assert pseudo["name"] == "pseudo::gemm"
        assert kernel["UID"] in pseudo.get("gpu_events", [])
        cpu_op_evt = next(e for e in tree.events if e["name"] == "aten::mm")
        launcher = tree.get_parent_event(kernel)
        assert launcher["UID"] in pseudo.get("children", [])
        assert pseudo["UID"] in cpu_op_evt.get("children", [])
        assert launcher["UID"] not in cpu_op_evt.get("children", [])

    def test_inject_pseudo_op_wrap_children(self):
        events: List[Dict] = []
        parent = _mk_event(
            "cpu_op",
            "parent_op",
            ts=0,
            dur=500,
            pid=1,
            tid=1,
            args={"Input Dims": [[4, 8]], "Sequence number": 1},
        )
        events.append(parent)
        corr = 300
        for idx, child_name in enumerate(["child_a", "child_b"]):
            child = _mk_event(
                "cpu_op",
                child_name,
                ts=50 + idx * 100,
                dur=80,
                pid=1,
                tid=1,
                args={"Sequence number": idx},
            )
            _add_gpu_chain(
                events,
                child,
                corr + idx,
                f"{child_name}_kernel",
                60 + idx * 100,
                90 + idx * 100,
            )

        tree = _build_tree(events)
        parent_evt = next(e for e in tree.events if e["name"] == "parent_op")
        inject_pseudo_op_wrap_children(tree, parent_evt, "pseudo::wrapped")

        pseudo_ops = [
            e for e in tree.events if e.get("args", {}).get("Pseudo op") is True
        ]
        assert len(pseudo_ops) == 1
        assert tree.get_children_events(parent_evt)[0]["name"] == "pseudo::wrapped"

    def test_inject_pseudo_op_above_event(self):
        events = [
            _mk_event("cpu_op", "grandparent", ts=0, dur=500, pid=1, tid=1, args={}),
            _mk_event(
                "python_function",
                "module.py(1): stage",
                ts=50,
                dur=200,
                pid=1,
                tid=1,
                args={"Python id": 1},
            ),
        ]
        tree = _build_tree(events, add_python_func=True)
        stage_evt = next(
            e for e in tree.events if e.get("name", "").endswith(": stage")
        )
        result = inject_pseudo_op_above_event(tree, stage_evt, "pseudo::above")
        assert result is not None
        assert result["name"] == "pseudo::above"
        assert tree.get_parent_event(stage_evt)["UID"] == result["UID"]


class TestPseudoOpsRegistry:
    def test_apply_moe_aiter_extension(self):
        events: List[Dict] = []
        events.append(
            _mk_event(
                "cpu_op",
                "vllm::moe_forward",
                ts=0,
                dur=10,
                pid=1,
                tid=1,
                args={"Sequence number": 0},
            )
        )
        moe_op = _mk_event(
            "cpu_op",
            "vllm::rocm_aiter_fused_moe",
            ts=100,
            dur=200,
            pid=1,
            tid=1,
            args={
                "Input Dims": [[8, 1024]],
                "Input type": ["fp16"],
                "Input Strides": [[1024, 1]],
                "Concrete Inputs": [],
                "Sequence number": 1,
            },
        )
        _add_gpu_chain(
            events,
            moe_op,
            corr=10,
            kernel_name="aiter::fmoe_kernel",
            ts_launch=110,
            ts_kernel=150,
        )
        tree = _build_tree(events)
        apply_pseudo_op_extensions(tree)
        pseudo_ops = [
            e for e in tree.events if e.get("args", {}).get("Pseudo op") is True
        ]
        assert any(p["name"] == "pseudo_op::moe_aiter_fused_1stage" for p in pseudo_ops)

    def test_create_pseudo_ops_moe_fused_aiter_direct(self):
        events: List[Dict] = []
        moe_op = _mk_event(
            "cpu_op",
            "vllm::rocm_aiter_fused_moe",
            ts=100,
            dur=200,
            pid=1,
            tid=1,
            args={
                "Input Dims": [[8, 1024]],
                "Input type": ["fp16"],
                "Input Strides": [[1024, 1]],
                "Concrete Inputs": [],
                "Sequence number": 1,
            },
        )
        _add_gpu_chain(
            events,
            moe_op,
            corr=11,
            kernel_name="aiter::fmoe_kernel",
            ts_launch=110,
            ts_kernel=150,
        )
        tree = _build_tree(events)
        create_pseudo_ops_moe_fused_aiter(tree)
        pseudo_ops = [
            e for e in tree.events if e.get("args", {}).get("Pseudo op") is True
        ]
        assert len(pseudo_ops) == 1


class TestExtensionHelpers:
    def test_is_aiter_fused_moe_kernel(self):
        assert is_aiter_fused_moe_kernel(
            {"cat": "kernel", "name": "aiter::fmoe_kernel"}
        )
        assert not is_aiter_fused_moe_kernel(
            {"cat": "kernel", "name": "aiter::MoeSorting"}
        )
        assert not is_aiter_fused_moe_kernel({"cat": "cpu_op", "name": "aiter::fmoe"})

    def test_is_matmul_ogs_kernel(self):
        assert is_matmul_ogs_kernel({"cat": "kernel", "name": "matmul_ogs_fwd"})
        assert not is_matmul_ogs_kernel({"cat": "kernel", "name": "other_kernel"})

    def test_is_gated_kernel(self):
        assert _is_gated_kernel("matmul_ogs_swiglu_kernel")
        assert _is_gated_kernel("matmul_ogs_glu_kernel")
        assert not _is_gated_kernel("matmul_ogs_plain")

    def test_is_fused_moe_gptq_awq_kernel(self):
        assert is_fused_moe_gptq_awq_kernel(
            {"cat": "kernel", "name": "fused_moe_kernel_gptq_awq"}
        )

    def test_extract_topk_from_outplace(self):
        event = {
            "args": {
                "Input Dims": [
                    [128, 4096],
                    [8, 4096, 512],
                    [8, 4096, 512],
                    [128, 6],
                    [128, 6],
                ]
            }
        }
        assert _extract_topk_from_outplace(event) == 6

    def test_find_fused_moe_ancestor(self):
        events = [
            _mk_event("cpu_op", FUSED_MOE_PARENT, ts=0, dur=500, pid=1, tid=1, args={}),
            _mk_event(
                "python_function",
                "flydsl.py(10): flydsl_moe_stage1",
                ts=50,
                dur=100,
                pid=1,
                tid=1,
                args={"Python id": 1},
            ),
        ]
        tree = _build_tree(events, add_python_func=True)
        stage_evt = next(
            e for e in tree.events if e.get("name", "").endswith("flydsl_moe_stage1")
        )
        ancestor = _find_fused_moe_ancestor(tree, stage_evt)
        assert ancestor is not None
        assert ancestor["name"] == FUSED_MOE_PARENT

    def test_v4_mode_and_geometry_helpers(self):
        assert _detect_v4_mode(["hca_compress_forward_kernel"]) == "hca"
        assert _detect_v4_mode(["fused_compress_attn_kernel"]) == "csa"
        assert _detect_v4_mode(["qk_norm_rope_H8_D128_RD64"]) == "swa"
        assert _parse_geometry(["qk_norm_rope_H8_D128_RD64"]) == (8, 128)
        assert _safe_int("4", default=1) == 4
        assert _safe_int(None, default=1) == 1


class TestTraceCaptureMergeExperimental:
    def test_capture_kernel_name_prefers_args(self):
        event = {"name": "dispatch", "args": {"kernel": "actual_kernel"}}
        assert _capture_kernel_name(event) == "actual_kernel"

    def test_align_capture_to_graph_greedy(self):
        capture = [
            {"name": "dispatch", "args": {"kernel": "k1"}},
            {"name": "dispatch", "args": {"kernel": "noise"}},
            {"name": "dispatch", "args": {"kernel": "k2"}},
        ]
        graph = [
            {"name": "k1", "args": {}},
            {"name": "k2", "args": {}},
        ]
        aligned = _align_capture_to_graph(capture, graph)
        assert aligned is not None
        assert [_capture_kernel_name(e) for e in aligned] == ["k1", "k2"]

    def test_align_graph_to_capture_by_group(self):
        capture = [
            {"name": "dispatch", "args": {"kernel": "k2"}},
            {"name": "dispatch", "args": {"kernel": "k1"}},
        ]
        graph = [
            {"name": "k1", "args": {}},
            {"name": "k2", "args": {}},
        ]
        aligned = _align_graph_to_capture_by_group(capture, graph)
        assert aligned is not None
        assert [e["name"] for e in aligned] == ["k2", "k1"]

    def test_verify_subtree_events_direct_match(self):
        capture = [
            {"name": "dispatch", "args": {"kernel": "k1"}},
            {"name": "Memcpy HtoD", "args": {}},
        ]
        graph = [
            {"name": "k1", "args": {}},
            {"name": "Memcpy HtoD", "args": {}},
        ]
        success, aligned_capture, aligned_graph = verify_subtree_events(capture, graph)
        assert success == 1
        assert aligned_capture == capture
        assert aligned_graph == graph

    def test_verify_subtree_events_greedy_on_count_mismatch(self):
        capture = [
            {"name": "dispatch", "args": {"kernel": "k1"}},
            {"name": "dispatch", "args": {"kernel": "extra"}},
            {"name": "dispatch", "args": {"kernel": "k2"}},
        ]
        graph = [
            {"name": "k1", "args": {}},
            {"name": "k2", "args": {}},
        ]
        success, aligned_capture, _ = verify_subtree_events(capture, graph)
        assert success == 2
        assert len(aligned_capture) == 2

    def test_names_match_memcpy_and_memset(self):
        capture_memcpy = {"name": "Memcpy HtoD", "args": {}}
        assert _names_match(capture_memcpy, "Memcpy DtoH")
        capture_memset = {"name": "Memset", "args": {}}
        assert _names_match(capture_memset, "fillBuffer")

    def test_stream_helpers_and_alignment(self):
        graph_events = [
            {"name": "k1", "args": {"stream": 1}},
            {"name": "k2", "args": {"stream": 2}},
        ]
        assert is_multistream(graph_events)
        assert _stream_of(graph_events[0]) == 1

        capture_events = [
            {"name": "dispatch", "args": {"kernel": "k1"}},
            {"name": "dispatch", "args": {"kernel": "k2"}},
        ]
        assert capture_has_kernel_names(capture_events)
        aligned = align_streams(graph_events, capture_events)
        assert aligned is not None
        assert len(aligned) == 2

    def test_get_subtree_events_filters(self):
        root = _mk_event("cpu_op", "root", ts=0, dur=100, pid=1, tid=1, args={})
        child = _mk_event("cpu_op", "child_mm", ts=10, dur=20, pid=1, tid=1, args={})
        events = [root, child]
        tree = _build_tree(events)
        root_evt = next(e for e in tree.events if e["name"] == "root")
        all_events, filtered = get_subtree_events(
            tree, root_evt, cat_filter={"cpu_op"}, name_filter=["mm"]
        )
        assert len(all_events) >= 1
        assert any("mm" in e["name"] for e in filtered)


class TestTraceToTreeUtilities:
    def test_parent_and_children_navigation(self):
        events = [
            _mk_event("cpu_op", "root", ts=0, dur=200, pid=1, tid=1, args={}),
            _mk_event("cpu_op", "child", ts=10, dur=50, pid=1, tid=1, args={}),
        ]
        tree = _build_tree(events)
        child = next(e for e in tree.events if e["name"] == "child")
        parent = tree.get_parent_event(child)
        assert parent is not None
        assert parent["name"] == "root"
        assert tree.get_children_events(parent)[0]["UID"] == child["UID"]

    def test_label_non_gpu_paths(self):
        events: List[Dict] = []
        gpu_op = _mk_event("cpu_op", "gpu_op", ts=0, dur=100, pid=1, tid=1, args={})
        _add_gpu_chain(
            events, gpu_op, corr=77, kernel_name="k", ts_launch=5, ts_kernel=20
        )
        events.append(
            _mk_event("cpu_op", "cpu_only", ts=200, dur=50, pid=1, tid=1, args={})
        )
        tree = TraceToTree(deepcopy(events), prune_nongpu_paths=True)
        tree.build_tree(add_python_func=False)
        gpu_op_evt = next(e for e in tree.events if e["name"] == "gpu_op")
        cpu_only = next(e for e in tree.events if e["name"] == "cpu_only")
        assert "non_gpu_path" not in gpu_op_evt
        assert cpu_only.get("non_gpu_path") is True

    def test_linking_key_uses_correlation_when_present(self):
        events = [
            _mk_event("cpu_op", "aten::add", ts=0, dur=10, pid=1, tid=1, args={}),
            _mk_event(
                "cuda_runtime",
                "hipLaunchKernel",
                ts=5,
                dur=1,
                pid=1,
                tid=1,
                args={"correlation": 7},
            ),
        ]
        tree = TraceToTree(deepcopy(events))
        assert tree.linking_key == "correlation"

    def test_linking_key_falls_back_to_external_id(self):
        events = [
            _mk_event(
                "cpu_op",
                "aten::add",
                ts=0,
                dur=10,
                pid=1,
                tid=1,
                args={"External id": 3},
            ),
        ]
        tree = TraceToTree(deepcopy(events))
        assert tree.linking_key == "External id"

    def test_nn_module_stack_strips_numeric_suffix(self):
        events = [
            _mk_event(
                "python_function",
                "nn.Module: Linear_3",
                ts=0,
                dur=10,
                pid=1,
                tid=1,
                args={"Python id": 1},
            ),
        ]
        tree = _build_tree(events, add_python_func=True)
        nn_evt = next(e for e in tree.events if "nn.Module" in e.get("name", ""))
        assert tree._nn_module_stack_name_for_event(nn_evt) == "nn.Module: Linear"

    def test_seq_num_index_populated(self):
        events = [
            _mk_event(
                "cpu_op",
                "op_a",
                ts=0,
                dur=10,
                pid=1,
                tid=1,
                args={"Sequence number": 5},
            ),
        ]
        tree = TraceToTree(deepcopy(events))
        assert 5 in tree.seq_num2event_uids_map
        assert len(tree.seq_num2event_uids_map[5]) == 1


class TestTraceToTreeGpuMarker:
    @pytest.mark.gpu
    def test_build_tree_with_cuda_events(self):
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            pytest.skip("Requires CUDA/HIP with at least one visible GPU")

        corr = 999
        events = [
            _mk_event("cpu_op", "aten::zeros", ts=0, dur=50, pid=1, tid=1, args={}),
            _mk_event(
                "cuda_runtime",
                "cudaLaunchKernel",
                ts=5,
                dur=2,
                pid=1,
                tid=1,
                args={"correlation": corr},
            ),
            _mk_event(
                "kernel",
                "void_kernel",
                ts=10,
                dur=5,
                pid=0,
                tid=7,
                args={"correlation": corr, "stream": 0},
            ),
            _mk_ac2g(corr, pid=0, tid=7, ts=10, phase="s"),
            _mk_ac2g(corr, pid=0, tid=7, ts=10, phase="f"),
        ]
        tree = _build_tree(events)
        kernel = next(e for e in tree.events if e["cat"] == "kernel")
        assert kernel.get("parent") is not None
