###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Tests for megatron_perf_model_extensions helper functions.
"""


class _FakeTree:
    """Minimal mock of TraceToTree for testing."""

    def __init__(self):
        self.events = []
        self.events_by_uid = {}
        self.name2event_uids = {}
        self.seq_num2event_uids_map = {}

    def add_event(self, event):
        uid = len(self.events)
        event["UID"] = uid
        self.events.append(event)
        self.events_by_uid[uid] = event
        name = event.get("name", "")
        self.name2event_uids.setdefault(name, []).append(uid)
        return uid

    def get_UID2event(self, uid):
        return self.events_by_uid[uid]

    def get_children_events(self, event):
        if "children" not in event:
            return []
        return [self.get_UID2event(uid) for uid in event["children"]]

    def get_parent_event(self, event):
        parent_uid = event.get("parent")
        if parent_uid is None:
            return None
        return self.events_by_uid.get(parent_uid)

    def get_subtree_bwd_events(self, uid):
        return self.events_by_uid[uid].get("bwd_events", [])


def test_find_layernorm_backward_targets_direct():
    """_find_layernorm_backward_targets returns bwd_op itself when it matches."""
    from TraceLens.PerfModel.extensions.megatron_perf_model_extensions import (
        _find_layernorm_backward_targets,
    )

    tree = _FakeTree()
    bwd = {"name": "_LayerNormLinearBackward", "args": {}}
    tree.add_event(bwd)
    result = _find_layernorm_backward_targets(tree, bwd)
    assert result == [bwd]


def test_find_layernorm_backward_targets_in_children():
    """_find_layernorm_backward_targets finds target in children."""
    from TraceLens.PerfModel.extensions.megatron_perf_model_extensions import (
        _find_layernorm_backward_targets,
    )

    tree = _FakeTree()
    parent = {"name": "autograd_wrapper", "children": []}
    tree.add_event(parent)
    child = {"name": "_LayerNormLinearBackward", "args": {}}
    child_uid = tree.add_event(child)
    parent["children"].append(child_uid)
    result = _find_layernorm_backward_targets(tree, parent)
    assert len(result) == 1
    assert result[0]["name"] == "_LayerNormLinearBackward"


def test_find_layernorm_backward_targets_no_match():
    """_find_layernorm_backward_targets returns empty when no match."""
    from TraceLens.PerfModel.extensions.megatron_perf_model_extensions import (
        _find_layernorm_backward_targets,
    )

    tree = _FakeTree()
    parent = {"name": "other_op", "children": []}
    tree.add_event(parent)
    child = {"name": "aten::add", "args": {}}
    child_uid = tree.add_event(child)
    parent["children"].append(child_uid)
    result = _find_layernorm_backward_targets(tree, parent)
    assert result == []


def test_extract_gemm_shapes():
    """_extract_gemm_shapes extracts correct shapes and dtype."""
    from TraceLens.PerfModel.extensions.megatron_perf_model_extensions import (
        _extract_gemm_shapes,
    )

    event = {
        "args": {
            "Input Dims": [
                [4096, 1024],
                [8, 512, 1024],
            ],
            "Input type": ["c10::BFloat16", "c10::BFloat16"],
            "Concrete Inputs": ["", "", "False"],
        },
    }
    W_shape, X_shape, Y_grad_shape, inp_dtype = _extract_gemm_shapes(event, 0, 1, 2)
    assert W_shape == [4096, 1024]
    assert X_shape == (4096, 1024)
    assert Y_grad_shape == (4096, 4096)
    assert inp_dtype == "c10::BFloat16"


def test_extract_gemm_shapes_fp8():
    """_extract_gemm_shapes sets fp8 dtype when fp8 flag is True."""
    from TraceLens.PerfModel.extensions.megatron_perf_model_extensions import (
        _extract_gemm_shapes,
    )

    event = {
        "args": {
            "Input Dims": [
                [4096, 1024],
                [8, 512, 1024],
            ],
            "Input type": ["c10::BFloat16", "c10::BFloat16"],
            "Concrete Inputs": ["", "", "True"],
        },
    }
    _, _, _, inp_dtype = _extract_gemm_shapes(event, 0, 1, 2)
    assert inp_dtype == "fp8"


def test_find_single_fwd_gemm_kernel_no_gpu_events():
    """_find_single_fwd_gemm_kernel returns None when no GPU events."""
    from TraceLens.PerfModel.extensions.megatron_perf_model_extensions import (
        _find_single_fwd_gemm_kernel,
    )

    tree = _FakeTree()
    fwd = {"gpu_events": [], "UID": 0}
    tree.add_event(fwd)
    assert _find_single_fwd_gemm_kernel(tree, fwd) is None


def test_find_single_fwd_gemm_kernel_no_gemm():
    """_find_single_fwd_gemm_kernel returns None when no GEMM kernel found."""
    from TraceLens.PerfModel.extensions.megatron_perf_model_extensions import (
        _find_single_fwd_gemm_kernel,
    )

    tree = _FakeTree()
    non_gemm = {"cat": "kernel", "name": "some_other_kernel", "args": {}}
    gpu_uid = tree.add_event(non_gemm)
    fwd = {"gpu_events": [gpu_uid], "UID": 1}
    tree.add_event(fwd)
    assert _find_single_fwd_gemm_kernel(tree, fwd) is None


def test_is_gemm_kernel_rocm():
    """is_gemm_kernel identifies ROCm GEMM kernels."""
    from TraceLens.PerfModel.extensions.megatron_perf_model_extensions import (
        is_gemm_kernel,
    )

    assert is_gemm_kernel(
        {"cat": "kernel", "name": "Cijk_Ailk_Bljk_something", "args": {}}
    )
    assert not is_gemm_kernel(
        {"cat": "kernel", "name": "some_other_kernel", "args": {}}
    )
    assert not is_gemm_kernel({"cat": "cpu_op", "name": "Cijk_test", "args": {}})
