###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""A leaf's role is the category the perf report would assign."""

from __future__ import annotations

from TraceLens.PerfModel.torch_op_mapping import categorize_torch_op
from TraceLens.Visualizer.model_explorer_export.computation_roles import (
    ATTENTION_ROLE,
    ROLE_BASIS_COMPUTATION,
    stamp_computation_roles,
)


def _node(node_id, label, role="other", kernel_name=None):
    attrs = [{"key": "role", "value": role}]
    if kernel_name:
        attrs.append({"key": "kernel_name", "value": kernel_name})
    return {"id": node_id, "label": label, "attrs": attrs}


def _attr(node, key):
    return next((a["value"] for a in node["attrs"] if a["key"] == key), None)


def test_report_attention_ops_are_stamped():
    nodes = [
        _node("sdpa", "aten::_scaled_dot_product_flash_attention"),
        _node("cache", "_C_cache_ops::reshape_and_cache_flash"),
        _node("gemm", "aten::mm"),
        _node("softmax", "aten::softmax"),
    ]
    assert categorize_torch_op(
        {"name": "aten::_scaled_dot_product_flash_attention"}
    ) == ("SDPA_fwd")
    stamped = stamp_computation_roles({"nodes": nodes})
    assert stamped == 2
    by_id = {node["id"]: node for node in nodes}
    for node_id in ("sdpa", "cache"):
        assert _attr(by_id[node_id], "role") == ATTENTION_ROLE
        assert _attr(by_id[node_id], "role_basis") == ROLE_BASIS_COMPUTATION
    for node_id in ("gemm", "softmax"):
        assert _attr(by_id[node_id], "role") == "other"
        assert _attr(by_id[node_id], "role_basis") is None


def test_an_eager_matmul_chain_is_not_relabeled():
    nodes = [
        _node("scores", "MatMul"),
        _node("softmax", "Softmax"),
        _node("values", "MatMul"),
    ]
    assert stamp_computation_roles({"nodes": nodes}) == 0


def test_a_role_already_read_from_the_module_is_kept():
    nodes = [_node("sdpa", "aten::_scaled_dot_product_flash_attention", role="norm")]
    assert stamp_computation_roles({"nodes": nodes}) == 0
    assert _attr(nodes[0], "role") == "norm"
    assert _attr(nodes[0], "role_basis") is None


def test_a_kernel_detail_uses_the_report_fallback():
    kernel = "void at::native::elementwise_kernel"
    nodes = [_node("ew", "unknown_op", kernel_name=kernel)]
    assert (
        categorize_torch_op(
            {"name": "unknown_op", "kernel_details": [{"name": kernel}]}
        )
        == "elementwise"
    )
    assert stamp_computation_roles({"nodes": nodes}) == 0
    assert _attr(nodes[0], "role") == "other"
