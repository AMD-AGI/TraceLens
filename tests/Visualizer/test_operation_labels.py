###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for inline forward-operation display labels and MoE expansion."""

from __future__ import annotations

from build_graph_in_env import pinned_model_spec

import pytest

from TraceLens.Visualizer.model_explorer_export.merge import build_merged_model_graph
from TraceLens.ModelUtils.block_tree import build_block_node


def test_decoder_spine_skips_inline_forward_ops():
    pytest.importorskip("huggingface_hub")
    spec = pinned_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
    attrs = {component.attr_name for component in spec.block_components}
    assert not any(name.startswith("@op_") for name in attrs)


def test_mhc_residual_merge_matmul_is_two_activation_matmul():
    pytest.importorskip("huggingface_hub")
    spec = pinned_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
    ops = spec.class_registry["Glm5NextTextDecoderLayer"].forward_operations.values()
    matmul = [op for op in ops if op.label == "MatMul" and not op.external_inputs]
    assert matmul
    assert all(op.class_name == "MatMul" for op in matmul)


def test_rotary_pos_emb_shows_multiply_not_buffer():
    pytest.importorskip("huggingface_hub")
    spec = pinned_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
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
    spec = pinned_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
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

    # ``expand_kv`` projects ``kv_nope`` through ``kv_b_proj``, splits it and
    # expands ``k_rot``, ALLOCATES the key buffer, then assembles it with two
    # in-place ``copy_`` writes into slices. Those copies are the real consumers
    # of the Split and Expand — without them modelled, both dangle with no
    # consumer. This makes the block a genuine (branchy) assembly. ``kv_b_proj``
    # itself is a real, materialised submodule call (not a dangling reference),
    # so it is the block's genuine first step.
    #
    # The allocation (``kv_nope.new_empty(*kv_nope.shape[:-1], qk_nope +
    # qk_rope)``) is an op like any other constructor. Leaving it out did not
    # merely omit a box: ``key_states`` then had no producer, so the first
    # ``copy_`` spine-fell onto the preceding op and the key reached sdpa at the
    # latent width instead of the head width.
    assert [node["label"] for node in op_nodes] == [
        "Linear",
        "View",
        "Transpose",
        "Split",
        "New empty",
        "Expand",
        "Copy",
        "Copy",
    ]
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
        if synthetic(node) == "@output_mirror" and consumes(node, out_boundary["id"])
    )
    # The attention the model dispatches to is its own implementation, expanded
    # into the ops it performs, so what the mirror reaches is that frame's own
    # parameter boundary -- the port named for the parameter ``key_states`` is
    # bound to -- not a port on a kernel nobody saw it call.
    key_port = next(
        node
        for node in graph["nodes"]
        if synthetic(node) == "@input"
        and ":@attention:" in node["id"]
        and consumes(node, out_mirror["id"])
    )
    assert key_port["label"] == "key_states"
    # And the expansion's own ops read it: the tensor crosses into the frame.
    assert any(
        ":@attention:" in node["id"]
        and synthetic(node) is None
        and consumes(node, key_port["id"])
        for node in graph["nodes"]
    )


def test_glm_experts_expands_router_boundary_into_named_parameters():
    pytest.importorskip("huggingface_hub")
    spec = pinned_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
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
    spec = pinned_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
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
    spec = pinned_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
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
