###############################################################################
# Copyright (c) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for TraceLens.Trace2Tree and related extension modules."""

from __future__ import annotations

import pytest
from copy import deepcopy
from typing import Dict, List
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
    append_subtree_to_event,
    capture_has_kernel_names,
    find_capture_roots,
    find_closest_batch_size,
    find_execution_details,
    get_subtree_events,
    is_multistream,
    make_connections,
    update_subtree_uids_and_timestamps,
    verify_subtree_events,
)
from TraceLens.Trace2Tree.trace_to_tree import JaxTraceToTree, TraceToTree
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
    create_pseudo_ops_moe_flydsl,
)
from TraceLens.Trace2Tree.extensions.v4_paged_decode_pseudo_ops import (
    _detect_v4_mode,
    _parse_geometry,
    _safe_int,
    create_pseudo_ops_v4_paged_decode,
)
from TraceLens.Trace2Tree import trace_to_tree as ttt


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
    kernel_dur: float = 20.0,
) -> None:
    pid = cpu_op["pid"]
    tid = cpu_op["tid"]
    if cpu_op not in events:
        events.append(cpu_op)
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
                dur=kernel_dur,
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

    def test_links_kernel_when_ac2g_start_is_missing(self):
        def _launch_events(corr, kernel_names):
            events = [
                _mk_event("cpu_op", "aten::mm", ts=0, dur=100, pid=1, tid=1, args={}),
                _mk_event(
                    "cuda_runtime",
                    "hipDrvLaunchKernelEx",
                    ts=5,
                    dur=5,
                    pid=1,
                    tid=1,
                    args={"correlation": corr},
                ),
            ]
            for idx, name in enumerate(kernel_names):
                events.append(
                    _mk_event(
                        "kernel",
                        name,
                        ts=20 + idx * 20,
                        dur=10,
                        pid=0,
                        tid=7,
                        args={"correlation": corr, "stream": 7},
                    )
                )
            events.append(_mk_ac2g(corr, pid=0, tid=7, ts=20, phase="f"))
            return events

        unique = _build_tree(_launch_events(26391, ["Cijk_Alik_Bljk"]))
        mm = next(e for e in unique.events if e["name"] == "aten::mm")
        gpu_events = unique.get_gpu_events(mm)
        assert len(gpu_events) == 1
        assert gpu_events[0]["name"] == "Cijk_Alik_Bljk"

        ambiguous = _build_tree(_launch_events(42, ["kernel_a", "kernel_b"]))
        mm = next(e for e in ambiguous.events if e["name"] == "aten::mm")
        assert ambiguous.get_gpu_events(mm) == []

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


class TestTraceToTreeExtended:
    def test_get_gpu_events_and_apply_annotation(self):
        events = [
            _mk_event(
                "user_annotation",
                "phase_a",
                ts=0,
                dur=500,
                pid=1,
                tid=1,
                args={},
            ),
            _mk_event("cpu_op", "root", ts=10, dur=100, pid=1, tid=1, args={}),
        ]
        corr = 500
        gpu_op = _mk_event("cpu_op", "gpu_op", ts=20, dur=80, pid=1, tid=1, args={})
        _add_gpu_chain(events, gpu_op, corr, "k1", 25, 40)
        tree = _build_tree(events)
        gpu_op_evt = next(e for e in tree.events if e["name"] == "gpu_op")
        gpu_events = tree.get_gpu_events(gpu_op_evt)
        assert len(gpu_events) == 1
        tree.apply_annotation(name_filters=["gpu_op"])
        assert tree.events[gpu_op_evt["UID"]]["annotation"] == "phase_a"

    def test_link_fwd_bwd_and_subtree_bwd_events(self):
        events = [
            _mk_event(
                "cpu_op",
                "aten::mm",
                ts=100,
                dur=50,
                pid=1,
                tid=1,
                args={"Sequence number": 7, "Input Dims": [[32, 64]]},
            ),
            _mk_event(
                "cpu_op",
                "autograd::engine::evaluate_function: MmBackward0",
                ts=200,
                dur=50,
                pid=1,
                tid=2,
                args={"Sequence number": 7},
            ),
        ]
        tree = _build_tree(events)
        fwd = next(e for e in tree.events if e["name"] == "aten::mm")
        bwd_uids = tree.get_subtree_bwd_events(fwd["UID"])
        assert len(bwd_uids) == 1
        bwd = tree.get_UID2event(bwd_uids[0])
        assert bwd["fwd_event"] == fwd["UID"]
        assert fwd["bwd_events"] == [bwd["UID"]]

    def test_traverse_parents_and_get_callstack(self):
        events: List[Dict] = []
        root = _mk_event("cpu_op", "root", ts=0, dur=200, pid=1, tid=1, args={})
        events.append(root)
        corr = 600
        child = _mk_event("cpu_op", "child_mm", ts=10, dur=50, pid=1, tid=1, args={})
        _add_gpu_chain(events, child, corr, "k1", 15, 40)
        tree = _build_tree(events)
        child_evt = next(e for e in tree.events if e["name"] == "child_mm")
        stack = tree.traverse_parents_and_get_callstack(child_evt, filter=None)
        assert stack[0] == "child_mm"
        assert "root" in stack

    def test_get_node_by_ext_id_pid_tid(self):
        events = [
            _mk_event(
                "cpu_op",
                "op_a",
                ts=0,
                dur=10,
                pid=1,
                tid=1,
                args={"External id": 42},
            ),
        ]
        tree = TraceToTree(deepcopy(events))
        found = tree.get_node_by_ext_id_pid_tid(42, 1, 1)
        assert found is not None
        assert found["name"] == "op_a"
        assert tree.get_node_by_ext_id_pid_tid(99, 1, 1) is None

    def test_traverse_subtree_and_print(self, capsys):
        events = [
            _mk_event("cpu_op", "root", ts=0, dur=100, pid=1, tid=1, args={}),
            _mk_event(
                "cpu_op",
                "leaf",
                ts=10,
                dur=20,
                pid=1,
                tid=1,
                args={"Input Dims": [[4, 8]]},
            ),
        ]
        tree = _build_tree(events)
        root = next(e for e in tree.events if e["name"] == "root")
        tree.traverse_subtree_and_print(
            root, prune_non_gpu=False, cpu_op_fields=("Input Dims",)
        )
        output = capsys.readouterr().out
        assert "leaf" in output
        assert "Input Dims" in output


class TestJaxTraceToTree:
    def test_is_gpu_event_static(self):
        gpu_evt = {"process": {"process_name": "/device:GPU:0"}}
        host_evt = {"process": {"process_name": "/host"}}
        assert JaxTraceToTree._is_gpu_event(gpu_evt) is True
        assert JaxTraceToTree._is_gpu_event(host_evt) is False

    def test_linking_key_after_init(self):
        events = [
            {
                "ph": "X",
                "name": "host",
                "pid": 701,
                "tid": 1,
                "ts": 0,
                "dur": 10,
                "args": {"correlation_id": 1},
                "process": {"process_name": "/host"},
            },
            {
                "ph": "X",
                "name": "Cijk_gemm",
                "pid": 2,
                "tid": 1,
                "ts": 5,
                "dur": 5,
                "parent": 0,
                "args": {"correlation_id": 1},
                "process": {"process_name": "/device:GPU:0"},
            },
        ]
        tree = JaxTraceToTree(
            events,
            compute_end_times=True,
            event_to_category=lambda e: (
                "kernel"
                if "/device:GPU" in e.get("process", {}).get("process_name", "")
                else "cpu_op"
            ),
        )
        assert tree.linking_key == "correlation_id"
        tree.add_gpu_ops_to_tree()
        assert tree.events[0].get("gpu_events")


class TestTraceCaptureMergeHelpers:
    def test_find_closest_batch_size_and_execution_details(self):
        assert find_closest_batch_size(32, [16, 32, 64]) == 32
        assert find_closest_batch_size(40, [16, 32, 64]) == 64
        assert find_closest_batch_size(100, [16, 32]) is None
        root = {
            "name": "execute_0_context_3(sq128sk256sqsq1sqsk1)_generation_2(sq1sk300sqsq1sqsk1)"
        }
        assert find_execution_details(root) == "129"

    def test_update_subtree_uids_and_make_connections(self):
        events = [
            _mk_event("cpu_op", "graph_root", ts=0, dur=200, pid=1, tid=1, args={}),
            _mk_event(
                "kernel",
                "k1",
                ts=50,
                dur=20,
                pid=0,
                tid=7,
                args={"stream": 1, "correlation": 1},
            ),
        ]
        tree = _build_tree(events)
        root = next(e for e in tree.events if e["name"] == "graph_root")
        kernel = next(e for e in tree.events if e["name"] == "k1")
        capture_events = [
            {
                "UID": 100,
                "name": "dispatch",
                "ts": 10,
                "dur": 5,
                "cat": "cuda_runtime",
                "args": {"kernel": "k1", "correlation": 99},
                "children": [],
            }
        ]
        updated, _, cpu_roots = update_subtree_uids_and_timestamps(
            tree,
            capture_events,
            capture_events,
            start_uid=len(tree.events),
            new_start_ts=root["ts"],
            c_root={"name": "CaptureRoot"},
            g_root_dur=root["dur"],
        )
        assert updated[0]["UID"] == len(tree.events)
        graph_filtered = [kernel]
        capture_filtered = updated
        tree = append_subtree_to_event(tree, updated, root, cpu_roots)
        tree = make_connections(tree, graph_filtered, capture_filtered)
        assert kernel["parent"] == updated[0]["UID"]
        assert capture_filtered[0]["args"]["correlation"] == 99

    def test_align_streams_multistream(self):
        graph_events = [
            {"name": "k1", "args": {"stream": 1}},
            {"name": "k2", "args": {"stream": 2}},
        ]
        capture_events = [
            {"name": "dispatch", "args": {"kernel": "k1"}},
            {"name": "dispatch", "args": {"kernel": "k2"}},
        ]
        assert is_multistream(graph_events)
        assert capture_has_kernel_names(capture_events)
        aligned = align_streams(graph_events, capture_events)
        assert aligned is not None
        assert [_capture_kernel_name(e) for e in aligned] == ["k1", "k2"]

    def test_verify_subtree_group_alignment(self):
        capture = [
            {"name": "dispatch", "args": {"kernel": "k2"}},
            {"name": "dispatch", "args": {"kernel": "k1"}},
        ]
        graph = [
            {"name": "k1", "args": {}},
            {"name": "k2", "args": {}},
        ]
        success, _, aligned_graph = verify_subtree_events(capture, graph)
        assert success == 3
        assert [e["name"] for e in aligned_graph] == ["k2", "k1"]

    def test_find_capture_roots_synthetic(self):
        events = [
            _mk_event(
                "cuda_runtime",
                "cudaStreamBeginCapture",
                ts=0,
                dur=10,
                pid=1,
                tid=1,
                args={},
            ),
            _mk_event(
                "cuda_runtime",
                "cudaLaunchKernel",
                ts=20,
                dur=5,
                pid=1,
                tid=1,
                args={"kernel": "k1"},
            ),
            _mk_event(
                "cuda_runtime",
                "cudaStreamEndCapture",
                ts=30,
                dur=5,
                pid=1,
                tid=1,
                args={},
            ),
        ]
        tree = TraceToTree(deepcopy(events))
        tree.build_tree(add_python_func=True)
        roots = find_capture_roots(tree)
        assert len(roots) == 1
        assert roots[0]["name"] == "CaptureRoot"


class TestPseudoOpsExtensionsExtended:
    def test_apply_triton_moe_extension(self):
        events: List[Dict] = []
        moe_fwd = _mk_event(
            "cpu_op",
            "vllm::moe_forward",
            ts=0,
            dur=300,
            pid=1,
            tid=1,
            args={"Sequence number": 0},
        )
        topk_op = _mk_event(
            "cpu_op",
            "aten::topk",
            ts=10,
            dur=20,
            pid=1,
            tid=1,
            args={"Concrete Inputs": ["", "6"]},
        )
        events.extend([moe_fwd, topk_op])
        corr = 400
        for idx, kernel_name in enumerate(
            ["matmul_ogs_fwd_kernel", "matmul_ogs_down_kernel"]
        ):
            _add_gpu_chain(
                events,
                moe_fwd,
                corr + idx,
                kernel_name,
                60 + idx * 100,
                90 + idx * 100,
            )
        tree = _build_tree(events)
        apply_pseudo_op_extensions(tree, verbose=True)
        pseudo_ops = [
            e for e in tree.events if e.get("args", {}).get("Pseudo op") is True
        ]
        assert any("moe_triton_unfused" in p["name"] for p in pseudo_ops)

    def test_apply_gptq_awq_extension(self):
        moe_op = _mk_event(
            "cpu_op",
            "vllm::outplace_fused_experts",
            ts=100,
            dur=200,
            pid=1,
            tid=1,
            args={
                "Input Dims": [
                    [128, 4096],
                    [8, 4096, 512],
                    [8, 4096, 512],
                    [128, 6],
                    [128, 6],
                ],
                "Sequence number": 1,
            },
        )
        events: List[Dict] = [moe_op]
        corr = 501
        for idx, kernel_name in enumerate(
            ["fused_moe_kernel_gptq_awq_up", "fused_moe_kernel_gptq_awq_down"]
        ):
            pid = moe_op["pid"]
            tid = moe_op["tid"]
            events.extend(
                [
                    _mk_event(
                        "cuda_runtime",
                        "hipLaunchKernel",
                        ts=110 + idx * 10,
                        dur=5,
                        pid=pid,
                        tid=tid,
                        args={"correlation": corr + idx},
                    ),
                    _mk_event(
                        "kernel",
                        kernel_name,
                        ts=150 + idx * 10,
                        dur=20,
                        pid=0,
                        tid=7,
                        args={"correlation": corr + idx, "stream": 7},
                    ),
                    _mk_ac2g(corr + idx, pid=0, tid=7, ts=150 + idx * 10, phase="s"),
                    _mk_ac2g(corr + idx, pid=0, tid=7, ts=150 + idx * 10, phase="f"),
                ]
            )
        tree = _build_tree(events)
        apply_pseudo_op_extensions(tree, verbose=True)
        pseudo_ops = [
            e for e in tree.events if e.get("args", {}).get("Pseudo op") is True
        ]
        assert any("moe_gptq_awq" in p["name"] for p in pseudo_ops)

    def test_apply_flydsl_extension(self):
        events = [
            _mk_event("cpu_op", FUSED_MOE_PARENT, ts=0, dur=500, pid=1, tid=1, args={}),
            _mk_event(
                "python_function",
                "flydsl.py(10): flydsl_moe_stage1",
                ts=50,
                dur=100,
                pid=1,
                tid=1,
                args={"Python id": 1, "Input Dims": [[8, 1024]]},
            ),
            _mk_event(
                "python_function",
                "flydsl.py(20): flydsl_moe_stage2",
                ts=160,
                dur=100,
                pid=1,
                tid=1,
                args={"Python id": 2, "Input Dims": [[8, 1024]]},
            ),
        ]
        tree = _build_tree(events, add_python_func=True)
        create_pseudo_ops_moe_flydsl(tree)
        pseudo_ops = [
            e for e in tree.events if e.get("args", {}).get("Pseudo op") is True
        ]
        assert len(pseudo_ops) == 2

    def test_apply_v4_paged_decode_extension(self, monkeypatch):
        monkeypatch.setenv("TL_MODEL", "deepseek-v4")
        monkeypatch.setenv("TL_TP", "4")
        events = [
            _mk_event(
                "cpu_op",
                "aiter::v4_attention_with_output",
                ts=0,
                dur=500,
                pid=1,
                tid=1,
                args={"Input Dims": [[1, 8, 128]]},
            ),
            _mk_event(
                "python_function",
                "paged_decode.py(12): sparse_attn_v4_paged_decode",
                ts=50,
                dur=200,
                pid=1,
                tid=1,
                args={"Python id": 1},
            ),
        ]
        corr = 700
        child = _mk_event(
            "cpu_op",
            "child",
            ts=60,
            dur=80,
            pid=1,
            tid=1,
            args={"Sequence number": 1},
        )
        _add_gpu_chain(
            events,
            child,
            corr,
            "qk_norm_rope_H8_D128_RD64",
            70,
            100,
        )
        tree = _build_tree(events, add_python_func=True)
        attn_evt = next(
            e for e in tree.events if e["name"] == "aiter::v4_attention_with_output"
        )
        py_evt = next(
            e
            for e in tree.events
            if e.get("name", "").endswith("sparse_attn_v4_paged_decode")
        )
        child_evt = next(e for e in tree.events if e["name"] == "child")
        py_evt["parent"] = attn_evt["UID"]
        attn_evt.setdefault("children", []).append(py_evt["UID"])
        child_evt["parent"] = py_evt["UID"]
        py_evt.setdefault("children", []).append(child_evt["UID"])
        attn_evt.setdefault("gpu_events", []).append(
            next(e for e in tree.events if e["cat"] == "kernel")["UID"]
        )
        py_evt.setdefault("gpu_events", []).extend(attn_evt["gpu_events"])
        create_pseudo_ops_v4_paged_decode(tree)
        pseudo_ops = [
            e for e in tree.events if e.get("args", {}).get("Pseudo op") is True
        ]
        assert any("pseudo_v4_paged_decode" in p["name"] for p in pseudo_ops)


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


class TestTraceToTreeTraversal:
    def test_traverse_subtree_with_bwd_events(self, capsys):
        fwd = _mk_event(
            "cpu_op",
            "_Linear",
            ts=100,
            dur=50,
            pid=1,
            tid=1,
            args={"Sequence number": 1},
        )
        bwd = _mk_event(
            "cpu_op",
            "_LinearBackward",
            ts=200,
            dur=60,
            pid=1,
            tid=1,
            args={"Sequence number": 1},
        )
        corr = 77
        events = [
            fwd,
            _mk_event(
                "cuda_runtime",
                "hipLaunchKernel",
                ts=110,
                dur=5,
                pid=1,
                tid=1,
                args={"correlation": corr},
            ),
            _mk_event(
                "kernel",
                "gemm_k",
                ts=120,
                dur=20,
                pid=0,
                tid=7,
                args={"correlation": corr, "stream": 7},
            ),
            _mk_ac2g(corr, pid=0, tid=7, ts=120, phase="s"),
            _mk_ac2g(corr, pid=0, tid=7, ts=120, phase="f"),
            bwd,
        ]
        tree = _build_tree(events)
        fwd_evt = next(e for e in tree.events if e["name"] == "_Linear")
        bwd_evt = next(e for e in tree.events if e["name"] == "_LinearBackward")
        fwd_evt["bwd_events"] = [bwd_evt["UID"]]
        bwd_evt.setdefault("children", [])

        tree.traverse_subtree_and_print(fwd_evt, include_bwd=True)
        out = capsys.readouterr().out
        assert "[BWD]" in out

    def test_traverse_parents_and_get_callstack_with_filter(self):
        events = [
            _mk_event(
                "cpu_op",
                "root_op",
                ts=0,
                dur=100,
                pid=1,
                tid=1,
                args={"Sequence number": 0},
            ),
            _mk_event(
                "cpu_op",
                "child_op",
                ts=10,
                dur=20,
                pid=1,
                tid=1,
                args={"Sequence number": 1},
            ),
        ]
        tree = _build_tree(events)
        child = next(e for e in tree.events if e["name"] == "child_op")
        root = next(e for e in tree.events if e["name"] == "root_op")
        child["parent"] = root["UID"]
        root.setdefault("children", []).append(child["UID"])

        frames = tree.traverse_parents_and_get_callstack(child, filter=("root",))
        assert "child_op" in frames[0]
        assert any("root_op" in f for f in frames)

    def test_traverse_parents_follow_fwd_link(self):
        fwd = _mk_event(
            "cpu_op", "fwd_op", ts=0, dur=50, pid=1, tid=1, args={"Sequence number": 0}
        )
        wrapper = _mk_event(
            "cpu_op",
            "autograd::evaluate_function: fwd_op",
            ts=50,
            dur=40,
            pid=1,
            tid=1,
            args={"Sequence number": 0},
        )
        bwd = _mk_event(
            "cpu_op",
            "bwd_op",
            ts=100,
            dur=50,
            pid=1,
            tid=1,
            args={"Sequence number": 0},
        )
        tree = _build_tree([fwd, wrapper, bwd])
        fwd_evt = next(e for e in tree.events if e["name"] == "fwd_op")
        wrapper_evt = next(e for e in tree.events if e["name"].startswith("autograd"))
        bwd_evt = next(e for e in tree.events if e["name"] == "bwd_op")

        wrapper_evt["fwd_event"] = fwd_evt["UID"]
        bwd_evt["parent"] = wrapper_evt["UID"]
        wrapper_evt.setdefault("children", []).append(bwd_evt["UID"])

        frames = tree.traverse_parents_and_get_callstack(
            bwd_evt, filter=None, follow_fwd_link=True
        )
        assert "[FWD]" in frames
        assert "fwd_op" in frames

    def test_traverse_parents_and_print(self, capsys):
        events = [
            _mk_event(
                "cpu_op",
                "mm_op",
                ts=0,
                dur=10,
                pid=1,
                tid=1,
                args={
                    "Input Dims": [[4, 8]],
                    "Input type": ["fp16"],
                    "Sequence number": 0,
                },
            )
        ]
        tree = _build_tree(events)
        evt = tree.events[0]
        tree.traverse_parents_and_print(evt, cpu_op_fields=("Input Dims",))
        out = capsys.readouterr().out
        assert "Node:" in out
        assert "Input Dims" in out

    def test_get_gpu_events_missing_field(self):
        tree = _build_tree([_mk_event("cpu_op", "op", ts=0, dur=1, pid=1, tid=1)])
        evt = tree.events[0]
        assert tree.get_gpu_events(evt) == []

    def test_traverse_subtree_prune_non_gpu(self, capsys):
        events = [
            _mk_event(
                "cpu_op",
                "parent",
                ts=0,
                dur=100,
                pid=1,
                tid=1,
                args={"Sequence number": 0},
            ),
            _mk_event(
                "cpu_op",
                "gpu_child",
                ts=10,
                dur=20,
                pid=1,
                tid=1,
                args={"Sequence number": 1},
            ),
            _mk_event(
                "cpu_op",
                "non_gpu_child",
                ts=30,
                dur=20,
                pid=1,
                tid=1,
                args={"Sequence number": 2},
            ),
        ]
        tree = _build_tree(events)
        parent = next(e for e in tree.events if e["name"] == "parent")
        gpu_child = next(e for e in tree.events if e["name"] == "gpu_child")
        non_gpu = next(e for e in tree.events if e["name"] == "non_gpu_child")
        parent.setdefault("children", []).extend([gpu_child["UID"], non_gpu["UID"]])
        gpu_child["parent"] = parent["UID"]
        non_gpu["parent"] = parent["UID"]
        non_gpu["non_gpu_path"] = True

        tree.traverse_subtree_and_print(parent, prune_non_gpu=True)
        out = capsys.readouterr().out
        assert "gpu_child" in out
        assert "non_gpu_child" not in out

    def test_long_name_truncation_in_callstack(self):
        long_name = "x" * 300
        child = _mk_event("cpu_op", "child_op", ts=10, dur=1, pid=1, tid=1)
        parent = _mk_event("cpu_op", long_name, ts=0, dur=20, pid=1, tid=1)
        tree = _build_tree([parent, child])
        child_evt = next(e for e in tree.events if e["name"] == "child_op")
        parent_evt = next(e for e in tree.events if e["name"] == long_name)
        child_evt["parent"] = parent_evt["UID"]
        parent_evt.setdefault("children", []).append(child_evt["UID"])

        frames = tree.traverse_parents_and_get_callstack(child_evt, filter=None)
        assert any(f.endswith("..") for f in frames)
        assert any(len(f) <= 258 for f in frames if f.endswith(".."))

    def test_traverse_parents_and_print_with_fwd_link(self, capsys):
        fwd = _mk_event("cpu_op", "fwd_op", ts=0, dur=50, pid=1, tid=1)
        wrapper = _mk_event(
            "cpu_op",
            "autograd::evaluate_function: fwd_op",
            ts=50,
            dur=40,
            pid=1,
            tid=1,
        )
        bwd = _mk_event("cpu_op", "bwd_op", ts=100, dur=50, pid=1, tid=1)
        tree = _build_tree([fwd, wrapper, bwd])
        fwd_evt = next(e for e in tree.events if e["name"] == "fwd_op")
        wrapper_evt = next(e for e in tree.events if e["name"].startswith("autograd"))
        bwd_evt = next(e for e in tree.events if e["name"] == "bwd_op")
        wrapper_evt["fwd_event"] = fwd_evt["UID"]
        bwd_evt["parent"] = wrapper_evt["UID"]
        wrapper_evt.setdefault("children", []).append(bwd_evt["UID"])

        tree.traverse_parents_and_print(bwd_evt, follow_fwd_link=True)
        out = capsys.readouterr().out
        assert "Following fwd_event link" in out
        assert "fwd_op" in out

    def test_traverse_parents_and_print_kernel_duration(self, capsys):
        corr = 88
        events = [
            _mk_event("cpu_op", "parent", ts=0, dur=100, pid=1, tid=1),
            _mk_event(
                "cuda_runtime",
                "hipLaunchKernel",
                ts=10,
                dur=5,
                pid=1,
                tid=1,
                args={"correlation": corr},
            ),
            _mk_event(
                "kernel",
                "gemm_k",
                ts=20,
                dur=30,
                pid=0,
                tid=7,
                args={"correlation": corr, "stream": 7},
            ),
            _mk_ac2g(corr, 0, 7, 20, "s"),
            _mk_ac2g(corr, 0, 7, 50, "f"),
        ]
        tree = _build_tree(events)
        kernel = next(e for e in tree.events if e["cat"] == "kernel")
        tree.traverse_parents_and_print(kernel)
        assert "Duration:" in capsys.readouterr().out

    def test_get_seq_nums_for_node_subtree(self):
        events = [
            _mk_event(
                "cpu_op",
                "root",
                ts=0,
                dur=100,
                pid=1,
                tid=1,
                args={"Sequence number": 1},
            ),
            _mk_event(
                "cpu_op",
                "child",
                ts=10,
                dur=20,
                pid=1,
                tid=1,
                args={"Sequence number": 2},
            ),
        ]
        tree = _build_tree(events)
        root = next(e for e in tree.events if e["name"] == "root")
        child = next(e for e in tree.events if e["name"] == "child")
        child["parent"] = root["UID"]
        root.setdefault("children", []).append(child["UID"])
        seqs = tree.get_seq_nums_for_node_subtree(root["UID"])
        assert seqs == {1, 2}

    def test_link_all_fwd_bwd_events(self):
        fwd = _mk_event(
            "cpu_op",
            "aten::mm",
            ts=0,
            dur=50,
            pid=1,
            tid=1,
            args={"Sequence number": 5},
        )
        bwd_autograd = _mk_event(
            "cpu_op",
            "autograd::engine::evaluate_function: MulBackward0",
            ts=100,
            dur=50,
            pid=1,
            tid=2,
            args={"Sequence number": 5},
        )
        tree = _build_tree([fwd, bwd_autograd])
        tree.link_all_fwd_bwd_events()
        fwd_evt = next(e for e in tree.events if e["name"] == "aten::mm")
        bwd_evt = next(
            e for e in tree.events if e["name"].startswith("autograd::engine")
        )
        assert bwd_evt["UID"] in fwd_evt.get("bwd_events", [])
        assert bwd_evt.get("fwd_event") == fwd_evt["UID"]

    def test_get_subtree_bwd_events(self):
        fwd = _mk_event(
            "cpu_op",
            "aten::add",
            ts=0,
            dur=10,
            pid=1,
            tid=1,
            args={"Sequence number": 1},
        )
        bwd_autograd = _mk_event(
            "cpu_op",
            "autograd::engine::evaluate_function: AddBackward0",
            ts=20,
            dur=10,
            pid=1,
            tid=2,
            args={"Sequence number": 1},
        )
        tree = _build_tree([fwd, bwd_autograd])
        tree.link_all_fwd_bwd_events()
        fwd_evt = next(e for e in tree.events if e["name"] == "aten::add")
        uids = tree.get_subtree_bwd_events(fwd_evt["UID"])
        assert len(uids) >= 1


def test_trace_to_tree_edge_helpers():

    events = [
        {
            "ph": "X",
            "name": "aten::mm",
            "cat": "cpu_op",
            "ts": 0,
            "dur": 10,
            "pid": 1,
            "tid": 1,
        },
        {
            "ph": "X",
            "name": "gemm",
            "cat": "kernel",
            "ts": 20,
            "dur": 5,
            "pid": 0,
            "tid": 7,
        },
    ]
    tree = ttt.TraceToTree(events)
    tree.build_tree(add_python_func=True)
    assert len(tree.events) >= 2
