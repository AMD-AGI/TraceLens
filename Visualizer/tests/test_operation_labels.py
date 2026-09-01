###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for inline forward-operation display labels and MoE expansion."""

from __future__ import annotations

import pytest

from model_explorer_export.merge import build_merged_model_graph
from visualizer.block_tree import build_block_node
from visualizer.loader import load_model_spec


def test_decoder_spine_skips_inline_forward_ops():
    pytest.importorskip("huggingface_hub")
    spec = load_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
    attrs = {component.attr_name for component in spec.block_components}
    assert "@op_l1316_c85_matmul" not in attrs
    assert "@op_l1316_c24_add" not in attrs


def test_mhc_residual_merge_matmul_is_two_activation_matmul():
    pytest.importorskip("huggingface_hub")
    spec = load_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
    op = spec.class_registry["Glm5NextTextDecoderLayer"].forward_operations[
        "@op_l1316_c85_matmul"
    ]
    assert op.label == "MatMul"
    assert op.class_name == "MatMul"
    assert not op.external_inputs


def test_matmul_with_parameter_displays_as_linear():
    import ast

    from visualizer.ast_analyze import (
        _forward_operations_from_forward,
        _self_config_values,
    )

    source = """
import torch
class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(4, 8))
    def forward(self, hidden_states):
        return torch.matmul(hidden_states, self.weight)
"""
    tree = ast.parse(source)
    cls = tree.body[-1]
    forward = next(
        item
        for item in cls.body
        if isinstance(item, ast.FunctionDef) and item.name == "forward"
    )
    init = next(
        item
        for item in cls.body
        if isinstance(item, ast.FunctionDef) and item.name == "__init__"
    )
    ops = _forward_operations_from_forward(
        forward,
        self_values=_self_config_values(init, {}),
        all_tensor_ops=False,
    )
    assert len(ops.operations) == 1
    op = ops.operations[0]
    assert op.label == "Linear"
    assert "weight" in op.external_inputs


def test_two_activation_matmul_displays_as_matmul():
    import ast

    from visualizer.ast_analyze import _forward_operations_from_forward

    source = """
import torch
class Block(torch.nn.Module):
    def forward(self, query, key):
        return torch.matmul(query, key.transpose(-1, -2))
"""
    tree = ast.parse(source)
    cls = tree.body[-1]
    forward = next(
        item
        for item in cls.body
        if isinstance(item, ast.FunctionDef) and item.name == "forward"
    )
    ops = _forward_operations_from_forward(
        forward, self_values={}, all_tensor_ops=False
    )
    assert len(ops.operations) == 1
    op = ops.operations[0]
    assert op.label == "MatMul"
    assert not op.external_inputs


def test_rotary_pos_emb_shows_multiply_not_buffer():
    pytest.importorskip("huggingface_hub")
    spec = load_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
    rotary = spec.class_registry["Glm5NextVisionRotaryEmbedding"]
    assert any(op.label == "Multiply" for op in rotary.forward_operations.values())

    _, tree = next(
        item for item in spec.export_block_trees if item[0] == "rotary_pos_emb"
    )
    child_labels = [child.label for child in tree.children]
    assert "Multiply" in child_labels
    assert "Buffer" not in child_labels

    graph = build_merged_model_graph(spec)
    rotary_labels = {
        node.get("label")
        for node in graph["nodes"]
        if node.get("namespace") == "rotary_pos_emb"
    }
    assert "Multiply" in rotary_labels
    assert "Buffer" not in rotary_labels


def test_merged_graph_uses_operator_labels_not_op_ids():
    pytest.importorskip("huggingface_hub")
    spec = load_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
    graph = build_merged_model_graph(spec)
    bad = [
        node
        for node in graph["nodes"]
        if node.get("label", "").startswith("@op ")
        or node.get("label", "").startswith("@op_")
    ]
    assert bad == []


def test_glm_experts_keeps_inline_computation_when_inferred():
    pytest.importorskip("huggingface_hub")
    spec = load_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
    tree = build_block_node(
        attr_name="experts",
        class_name="Glm5NextTextExperts",
        registry=spec.class_registry,
        basic_ops=spec.basic_ops,
        infer_init_steps=True,
    )
    labels = [child.label for child in tree.children]
    assert labels.count("Linear") >= 2
    assert "Sum" in labels
    assert "Multiply" in labels
