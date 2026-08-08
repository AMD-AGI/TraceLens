###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""CPU-only coverage for TraceToTree helper and traversal paths."""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, List

import pytest

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


def _build_tree(events: List[Dict], add_python_func: bool = False) -> TraceToTree:
    tree = TraceToTree(deepcopy(events), prune_nongpu_paths=False)
    tree.build_tree(add_python_func=add_python_func)
    return tree


class TestTraceToTreeTraversal:
    def test_traverse_subtree_with_bwd_events(self, capsys):
        fwd = _mk_event("cpu_op", "_Linear", ts=100, dur=50, pid=1, tid=1, args={"Sequence number": 1})
        bwd = _mk_event("cpu_op", "_LinearBackward", ts=200, dur=60, pid=1, tid=1, args={"Sequence number": 1})
        corr = 77
        events = [
            fwd,
            _mk_event("cuda_runtime", "hipLaunchKernel", ts=110, dur=5, pid=1, tid=1, args={"correlation": corr}),
            _mk_event("kernel", "gemm_k", ts=120, dur=20, pid=0, tid=7, args={"correlation": corr, "stream": 7}),
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
            _mk_event("cpu_op", "root_op", ts=0, dur=100, pid=1, tid=1, args={"Sequence number": 0}),
            _mk_event("cpu_op", "child_op", ts=10, dur=20, pid=1, tid=1, args={"Sequence number": 1}),
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
        fwd = _mk_event("cpu_op", "fwd_op", ts=0, dur=50, pid=1, tid=1, args={"Sequence number": 0})
        wrapper = _mk_event(
            "cpu_op",
            "autograd::evaluate_function: fwd_op",
            ts=50,
            dur=40,
            pid=1,
            tid=1,
            args={"Sequence number": 0},
        )
        bwd = _mk_event("cpu_op", "bwd_op", ts=100, dur=50, pid=1, tid=1, args={"Sequence number": 0})
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
            _mk_event("cpu_op", "parent", ts=0, dur=100, pid=1, tid=1, args={"Sequence number": 0}),
            _mk_event("cpu_op", "gpu_child", ts=10, dur=20, pid=1, tid=1, args={"Sequence number": 1}),
            _mk_event("cpu_op", "non_gpu_child", ts=30, dur=20, pid=1, tid=1, args={"Sequence number": 2}),
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
