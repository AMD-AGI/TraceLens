###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for semantic_analyses/extract_tree_context.py matching logic."""

import os, sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SEM = os.path.join(REPO_ROOT, "TraceLens", "Agent", "Analysis", "semantic_analyses")
sys.path.insert(0, SEM)
import extract_tree_context


class FakeTree:
    """Minimal tree stub exposing only what the matching functions read."""

    def __init__(self, events):
        self.events = events

    def get_parent_event(self, event):
        return event.get("_parent")

    def event_to_category(self, event):
        return event.get("_cat")

    def traverse_parents_and_get_callstack(self, event):
        return "callstack:" + event.get("name", "")


def test_build_ts_index():
    events = [
        {"name": "a", "ts": 1},
        {"name": "a", "ts": 1},
        {"name": "b", "ts": 2},
    ]
    index = extract_tree_context._build_ts_index(FakeTree(events))
    assert len(index[("a", 1)]) == 2
    assert len(index[("b", 2)]) == 1


def test_find_tree_event_single_candidate():
    e = {"name": "a", "ts": 1, "dur": 3, "UID": 1}
    index = {("a", 1): [e]}
    assert extract_tree_context._find_tree_event(index, "a", 1, 3) is e


def test_find_tree_event_uid_disambiguates_collision():
    e1 = {"name": "a", "ts": 1, "dur": 3, "UID": 10}
    e2 = {"name": "a", "ts": 1, "dur": 3, "UID": 20}
    index = {("a", 1): [e1, e2]}
    got = extract_tree_context._find_tree_event(index, "a", 1, 3, uid=20)
    assert got is e2


def test_find_tree_event_dur_fallback_on_collision():
    e1 = {"name": "a", "ts": 1, "dur": 3, "UID": 10}
    e2 = {"name": "a", "ts": 1, "dur": 9, "UID": 20}
    index = {("a", 1): [e1, e2]}
    # No uid given -> disambiguate by dur.
    got = extract_tree_context._find_tree_event(index, "a", 1, 9)
    assert got is e2


def test_find_tree_event_first_when_no_match():
    e1 = {"name": "a", "ts": 1, "dur": 3, "UID": 10}
    e2 = {"name": "a", "ts": 1, "dur": 4, "UID": 20}
    index = {("a", 1): [e1, e2]}
    # uid + dur both fail to match -> first candidate.
    got = extract_tree_context._find_tree_event(index, "a", 1, 99, uid=999)
    assert got is e1


def test_find_tree_event_none_when_missing():
    assert extract_tree_context._find_tree_event({}, "a", 1, 3) is None


def test_find_cpu_op_ancestor_walks_to_cpu_op():
    cpu_op = {
        "name": "aten::mm",
        "_cat": "cpu_op",
        "args": {"Input Dims": [[2, 2]]},
        "nn_module_stack": ["mod"],
    }
    mid = {"name": "py_fn", "_cat": "python_function", "_parent": cpu_op}
    gpu = {"name": "gemm", "_cat": "kernel", "_parent": mid}
    tree = FakeTree([cpu_op, mid, gpu])

    name, callstack, nn_module, input_dims = extract_tree_context._find_cpu_op_ancestor(
        tree, gpu
    )
    assert name == "aten::mm"
    assert callstack == "callstack:aten::mm"
    assert nn_module == ["mod"]
    assert input_dims == [[2, 2]]


def test_find_cpu_op_ancestor_no_parent():
    gpu = {"name": "gemm", "_cat": "kernel"}
    tree = FakeTree([gpu])
    name, callstack, nn_module, input_dims = extract_tree_context._find_cpu_op_ancestor(
        tree, gpu
    )
    assert name == ""
    assert callstack == ""
    assert nn_module == []
    assert input_dims == []


def test_extract_tree_context_mixed():
    cpu_op = {
        "name": "aten::mm",
        "ts": 1,
        "UID": 10,
        "_cat": "cpu_op",
        "args": {"Input Dims": [[2, 2]]},
        "nn_module_stack": ["m"],
    }
    gpu = {
        "name": "gemm",
        "ts": 5,
        "dur": 3,
        "UID": 11,
        "_cat": "kernel",
        "_parent": cpu_op,
    }
    solo = {"name": "solo", "ts": 9, "dur": 2, "UID": 12, "_cat": "kernel"}
    tree = FakeTree([cpu_op, gpu, solo])

    extracted = {
        "source_file": "f.json",
        "kernels": [
            {"name": "gemm", "ts": 5, "dur": 3, "gpu_op_uid": 11},
            {"name": "solo", "ts": 9, "dur": 2, "gpu_op_uid": 12},
            {"name": "missing", "ts": 99, "dur": 1, "gpu_op_uid": 99},
        ],
    }

    result = extract_tree_context.extract_tree_context(tree, extracted)

    assert result["total_kernels"] == 3
    assert result["labeled_count"] == 1
    assert result["unlabeled_count"] == 2
    assert result["labeled_indices"] == [0]
    assert result["unlabeled_indices"] == [1, 2]
    assert result["coverage"] == round(1 / 3, 4)

    # Matched kernel with cpu_op ancestor.
    ctx0 = result["kernels"][0]
    assert ctx0["gpu_op_uid"] == 11
    assert ctx0["cpu_op_name"] == "aten::mm"
    assert ctx0["input_dims"] == [[2, 2]]

    # Matched kernel with no cpu_op ancestor.
    assert result["kernels"][1]["cpu_op_name"] == ""
    assert result["kernels"][1]["gpu_op_uid"] == 12

    # Unmatched kernel.
    assert result["kernels"][2]["gpu_op_uid"] is None
    assert result["kernels"][2]["cpu_op_name"] == ""


def test_extract_tree_context_empty_uses_prebuilt_index():
    tree = FakeTree([])
    ts_index = extract_tree_context._build_ts_index(tree)
    result = extract_tree_context.extract_tree_context(
        tree, {"kernels": []}, ts_index=ts_index
    )
    assert result["total_kernels"] == 0
    assert result["coverage"] == 0.0
    assert result["source_file"] == ""
