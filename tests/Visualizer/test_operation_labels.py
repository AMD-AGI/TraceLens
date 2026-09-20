###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for inline forward-operation display labels and MoE expansion."""

from __future__ import annotations

import pytest

from TraceLens.Visualizer.model_explorer_export.merge import build_merged_model_graph
from TraceLens.ModelUtils.block_tree import build_block_node
from TraceLens.ModelUtils.loader import load_model_spec


def test_decoder_spine_skips_inline_forward_ops():
    pytest.importorskip("huggingface_hub")
    spec = load_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
    attrs = {component.attr_name for component in spec.block_components}
    assert "@op_l1319_c85_matmul" not in attrs
    assert "@op_l1319_c24_add" not in attrs


def test_mhc_residual_merge_matmul_is_two_activation_matmul():
    pytest.importorskip("huggingface_hub")
    spec = load_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
    op = spec.class_registry["Glm5NextTextDecoderLayer"].forward_operations[
        "@op_l1319_c85_matmul"
    ]
    assert op.label == "MatMul"
    assert op.class_name == "MatMul"
    assert not op.external_inputs


def test_matmul_with_parameter_displays_as_linear():
    import ast

    from TraceLens.ModelUtils.ast_analyze import (
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


def test_sibling_method_only_forward_keeps_inline_math():
    """A forward whose only calls are sibling helpers still owns its tensor math.

    Mirrors Glm5NextVisionRotaryEmbedding: forward computes cos/sin inline, then
    hands each to a sibling method (recomposition). The delegation must not hide
    the inline multiplies, and pruning must bridge the sibling call to keep them.
    """
    import ast

    from TraceLens.ModelUtils.ast_analyze import (
        _apply_forward_analysis,
        _forward_delegates_only_to_sibling_methods,
        _forward_delegates_to_nothing,
        _forward_operations_from_forward,
        _forward_owns_tensor_math,
        _self_config_values,
    )

    source = """
import torch
class Rotary(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = 1.0
    def forward(self, freqs):
        cos = freqs.cos() * self.scale
        sin = freqs.sin() * self.scale
        cos = self.recompose(cos)
        sin = self.recompose(sin)
        return cos, sin
    def recompose(self, freq):
        return torch.cat([freq, freq], dim=-1)
"""
    tree = ast.parse(source)
    cls = tree.body[-1]
    forward = next(
        item for item in cls.body
        if isinstance(item, ast.FunctionDef) and item.name == "forward"
    )
    init = next(
        item for item in cls.body
        if isinstance(item, ast.FunctionDef) and item.name == "__init__"
    )
    method_names = {
        item.name for item in cls.body if isinstance(item, ast.FunctionDef)
    }

    forward_calls = ["recompose"]
    init_assignments = {}

    # forward delegates only to a sibling method: neither "owns math" (no
    # pointwise submodules) nor "delegates to nothing" (there IS a call).
    assert not _forward_owns_tensor_math(forward_calls, init_assignments)
    assert not _forward_delegates_to_nothing(cls.name, forward_calls)
    assert _forward_delegates_only_to_sibling_methods(
        forward_calls, init_assignments, method_names
    )

    analysis = _forward_operations_from_forward(
        forward, self_values=_self_config_values(init, {}), all_tensor_ops=False
    )
    _calls, operations, *_ = _apply_forward_analysis(
        forward, analysis, forward_calls=forward_calls, init_assignments=init_assignments
    )
    # The inline multiplies survive pruning even though the returned values are
    # produced by the sibling call.
    assert any(op.label == "Multiply" for op in operations.values())


def test_sibling_method_delegation_rejects_submodule_and_free_calls():
    """The sibling-only heuristic must not fire when a real submodule is called."""
    from TraceLens.ModelUtils.ast_analyze import (
        _forward_delegates_only_to_sibling_methods,
    )

    method_names = {"forward", "helper"}
    # A submodule call (in init_assignments) means the math lives in the child.
    assert not _forward_delegates_only_to_sibling_methods(
        ["sub"], {"sub": "SubModule"}, method_names
    )
    # An unknown free call is not a recognized sibling method.
    assert not _forward_delegates_only_to_sibling_methods(
        ["mystery"], {}, method_names
    )
    # No calls at all is handled by _forward_delegates_to_nothing, not here.
    assert not _forward_delegates_only_to_sibling_methods([], {}, method_names)


def test_two_activation_matmul_displays_as_matmul():
    import ast

    from TraceLens.ModelUtils.ast_analyze import _forward_operations_from_forward

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

    # The text stack never calls this vision module, so it stays out of the graph
    # rather than hanging off the embedding as a branch that feeds nothing.
    graph = build_merged_model_graph(spec)
    assert not any(
        node["id"] == "rotary_pos_emb" or node["id"].startswith("rotary_pos_emb/")
        for node in graph["nodes"]
    )
    assert not any(node.get("namespace") == "rotary_pos_emb" for node in graph["nodes"])


def test_glm_attention_expand_kv_assembles_key_states_from_split_and_expand():
    pytest.importorskip("huggingface_hub")
    spec = load_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
    graph = build_merged_model_graph(spec)

    def synthetic(node):
        return next(
            (
                attr["value"]
                for attr in node.get("attrs", [])
                if attr.get("key") == "synthetic"
            ),
            None,
        )

    nodes = [node for node in graph["nodes"] if "expand_kv" in node["id"]]
    op_nodes = [node for node in nodes if synthetic(node) is None]

    # ``expand_kv`` splits ``kv_nope`` and expands ``k_rot``, then assembles
    # ``key_states`` with two in-place ``copy_`` writes into slices. Those copies
    # are the real consumers of the Split and Expand — without them modelled, both
    # dangle with no consumer. This makes the block a genuine (branchy) assembly.
    assert [node["label"] for node in op_nodes] == [
        "View",
        "Transpose",
        "Split",
        "Expand",
        "Copy",
        "Copy",
    ]
    by_id = {node["id"]: node for node in graph["nodes"]}
    split = next(node for node in op_nodes if node["label"] == "Split")
    expand = next(node for node in op_nodes if node["label"] == "Expand")
    copies = [node for node in op_nodes if node["label"] == "Copy"]

    def consumes(consumer, producer_id):
        return any(
            edge["sourceNodeId"] == producer_id
            for edge in consumer.get("incomingEdges", [])
        )

    def consumes_split_slice(consumer, split_id):
        # A multi-output Split surfaces each slice as its own named ``@slice_out``
        # passthrough tile; a consumer reads the slice via that tile, not the Split
        # node directly.
        slice_prefix = split_id + "^@slice_out:"
        return any(
            edge["sourceNodeId"].startswith(slice_prefix)
            for edge in consumer.get("incomingEdges", [])
        )

    # Each of Split and Expand feeds a Copy that writes it into ``key_states``.
    # (The Split reaches its Copy through one of its named slice tiles.)
    assert any(consumes_split_slice(copy, split["id"]) for copy in copies)
    assert any(consumes(copy, expand["id"]) for copy in copies)
    # The final Copy assembles ``key_states`` and feeds the ``expand_kv`` submodule's
    # own ``@output:key_states`` boundary. That crossing exits a module (expand_kv ->
    # the parent self_attn), so the hierarchy-aware same-name collapse KEEPS the
    # boundary tile rather than folding it onto the kernel port. Walking the kept
    # ``@output``/``@output_mirror`` chain reaches the attention kernel's key port.
    out_boundary = next(
        node
        for node in graph["nodes"]
        if synthetic(node) == "@output"
        and node["label"] == "key_states"
        and consumes(node, copies[-1]["id"])
    )
    out_mirror = next(
        node
        for node in graph["nodes"]
        if synthetic(node) == "@output_mirror"
        and consumes(node, out_boundary["id"])
    )
    key_port = next(
        node
        for node in graph["nodes"]
        if synthetic(node) == "@kernel_port_in" and consumes(node, out_mirror["id"])
    )
    assert key_port["label"] == "key_states"
    assert any(
        ":@attention:" in node["id"] and consumes(node, key_port["id"])
        for node in graph["nodes"]
    )


def test_glm_experts_expands_router_boundary_into_named_parameters():
    pytest.importorskip("huggingface_hub")
    spec = load_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
    graph = build_merged_model_graph(spec)

    def synthetic(node):
        return next(
            (a["value"] for a in node.get("attrs", []) if a.get("key") == "synthetic"),
            None,
        )

    # The expert dispatch flattens into the MoE scope, so the router boundary is
    # expanded into the named parameters ``topk_indices``/``topk_weights`` rather
    # than a single opaque "router" tile. After the same-name boundary collapse the
    # experts loop's redundant same-name ``@input:topk_*`` tiles fold away, so those
    # names now surface as the router's ``@output`` ports, still consumed in the MoE.
    moe_boundaries = [
        node
        for node in graph["nodes"]
        if "Glm5NextTextMoE" in node.get("namespace", "")
        and synthetic(node) in ("@input", "@output", "@output_mirror")
    ]
    labels = {node["label"] for node in moe_boundaries}
    assert "router" not in labels
    assert {"topk_indices", "topk_weights"} <= labels

    # Each named router output is really consumed inside the MoE scope.
    consumers: dict[str, list] = {}
    for node in graph["nodes"]:
        for edge in node.get("incomingEdges", []) or []:
            consumers.setdefault(edge["sourceNodeId"], []).append(node)
    for name in ("topk_indices", "topk_weights"):
        producer = next(
            node
            for node in moe_boundaries
            if node["label"] == name and synthetic(node) == "@output"
        )
        assert consumers.get(producer["id"]), name


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
