###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""CPU-only coverage for MLA pseudo-op extensions and registry detection paths."""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, List
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
from TraceLens.Trace2Tree.extensions.pseudo_ops_registry import (
    apply_pseudo_op_extensions,
)
from TraceLens.Trace2Tree.trace_to_tree import TraceToTree


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
