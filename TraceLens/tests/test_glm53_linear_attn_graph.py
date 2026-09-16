###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Regression tests for GLM-5.3-Flash linear-attention graph wiring."""

from __future__ import annotations

import pytest

from TraceLens.Visualizer.model_explorer_export.merge import build_merged_model_graph
from TraceLens.ModelUtils.computation_graph import add_forward_output, build_computation_graph
from TraceLens.ModelUtils.loader import load_model_spec
from TraceLens.ModelUtils.shape_inference import ShapeInferencer


def _linear_attn_tree(spec):
    return next((item for item in spec.export_block_trees if "Linear Attn" in item[0]))


def _graph_key(graph, suffix: str) -> str:
    matches = [node.key for node in graph.nodes if node.key.endswith(suffix)]
    assert len(matches) == 1, matches
    return matches[0]


def _graph_key_for_op(graph, fragment: str) -> str:
    """Locate a node by block and op identity, ignoring its slot within the block."""
    matches = [node.key for node in graph.nodes if fragment in node.key]
    assert len(matches) == 1, matches
    return matches[0]


def _linear_attn_variant_prefix(spec) -> str:
    variant = next(
        v for v in spec.layer_variants if "LinearAttention" in (v.attention_class or "")
    )
    return f"decoder/{variant.count}x_{variant.attention_class}_{variant.ffn_class}"


def _has_computation_path(graph, source: int, target: int) -> bool:
    pending = [source]
    visited: set[int] = set()
    while pending:
        current = pending.pop()
        if current == target:
            return True
        if current in visited:
            continue
        visited.add(current)
        pending.extend(
            destination for start, destination in graph.links if start == current
        )
    return False


def _has_export_path(nodes, source_id: str, target_id: str) -> bool:
    outgoing: dict[str, list[str]] = {}
    for node in nodes:
        for edge in node.get("incomingEdges", []):
            outgoing.setdefault(edge["sourceNodeId"], []).append(node["id"])
    pending = [source_id]
    visited: set[str] = set()
    while pending:
        current = pending.pop()
        if current == target_id:
            return True
        if current in visited:
            continue
        visited.add(current)
        pending.extend(outgoing.get(current, []))
    return False


def _assert_export_is_acyclic(nodes) -> None:
    # Loop-carried feedback edges (out→in) are intentionally cyclic.
    loop_carried_in_ids = {
        node["id"] for node in nodes if "@loop_carried_in:" in node["id"]
    }
    outgoing: dict[str, list[str]] = {}
    for node in nodes:
        for edge in node.get("incomingEdges", []):
            target_id = node["id"]
            source_id = edge["sourceNodeId"]
            if target_id in loop_carried_in_ids and "@loop_carried_out:" in source_id:
                continue
            outgoing.setdefault(source_id, []).append(target_id)

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in visiting:
            raise AssertionError(f"cycle reaches {node_id}")
        if node_id in visited:
            return
        visiting.add(node_id)
        for target_id in outgoing.get(node_id, []):
            visit(target_id)
        visiting.remove(node_id)
        visited.add(node_id)

    for node in nodes:
        visit(node["id"])


def test_glm53_linear_attention_has_single_output_exit():
    pytest.importorskip("huggingface_hub")
    spec = load_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
    title, tree = _linear_attn_tree(spec)
    graph = build_computation_graph(tree, basic_ops=spec.basic_ops)
    source_indices = {src for src, _target in graph.links}
    exits = [
        index
        for index, node in enumerate(graph.nodes)
        if index not in source_indices
        and node.synthetic not in {"@input", "@hidden_states", "@tensor"}
    ]
    assert exits, f"Expected at least one exit for {title}"

    add_forward_output(graph)
    output_sources = [
        src for src, tgt in graph.links if graph.nodes[tgt].label == "Output"
    ]
    assert len(output_sources) == 1
    assert graph.nodes[output_sources[0]].label == "Linear"


def test_glm53_linear_attention_gate_chain_is_not_short_circuited():
    pytest.importorskip("huggingface_hub")
    spec = load_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
    _title, tree = _linear_attn_tree(spec)
    graph = build_computation_graph(tree, basic_ops=spec.basic_ops)
    keys = {node.key for node in graph.nodes}

    g_a_key = _graph_key(graph, ":g_a_proj")
    g_b_key = _graph_key(graph, ":g_b_proj")
    o_norm_power_key = _graph_key_for_op(graph, ":o_norm:@op_l353_c19_power:")
    o_norm_mean_key = _graph_key_for_op(graph, ":o_norm:@op_l353_c19_mean:")
    o_norm_rsqrt_key = _graph_key_for_op(graph, ":o_norm:@op_l354_c40_reciprocal_sqrt:")
    o_norm_gate_key = _graph_key_for_op(graph, ":o_norm:@op_l358_c40_sigmoid:")
    o_norm_mul_key = _graph_key_for_op(graph, ":o_norm:@op_l358_c24_multiply:")

    assert g_a_key in keys
    assert g_b_key in keys
    assert o_norm_power_key in keys

    key_to_index = {node.key: index for index, node in enumerate(graph.nodes)}
    assert graph.nodes[key_to_index[o_norm_power_key]].label == "Power"
    assert graph.nodes[key_to_index[o_norm_mean_key]].label == "Mean"
    assert graph.nodes[key_to_index[o_norm_rsqrt_key]].label == "Reciprocal sqrt"
    assert graph.nodes[key_to_index[o_norm_gate_key]].label == "Sigmoid"
    assert graph.nodes[key_to_index[o_norm_mul_key]].label == "Multiply"
    links = set(graph.links)

    assert (key_to_index[g_a_key], key_to_index[g_b_key]) in links
    assert _has_computation_path(
        graph, key_to_index[g_b_key], key_to_index[o_norm_gate_key]
    )
    assert (key_to_index[o_norm_gate_key], key_to_index[o_norm_mul_key]) in links
    assert (key_to_index["@input"], key_to_index[g_b_key]) not in links


def test_glm53_gate_view_has_no_recurrent_state_cast_edge():
    """The gate ``view`` reads only its ``g_b_proj`` producer.

    ``last_recurrent_state.to(torch.float32)`` (source line 726) is an unconsumed
    cache-update side-effect that sits on the spine just before the gate
    ``view = self.g_b_proj(...).view(hidden_shape)``. Neither the sequential
    fallback nor the multi-input forward-link bridge may fabricate an edge from
    that terminal Cast into the view (or into the ``g_a_proj``/``g_b_proj`` side
    producers that already read ``hidden_states``).
    """
    pytest.importorskip("huggingface_hub")
    spec = load_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
    _title, tree = _linear_attn_tree(spec)
    graph = build_computation_graph(tree, basic_ops=spec.basic_ops)

    key_to_index = {node.key: index for index, node in enumerate(graph.nodes)}
    view_key = _graph_key_for_op(graph, ":@op_l732_c15_view:")
    g_b_key = _graph_key(graph, ":g_b_proj")
    view_index = key_to_index[view_key]
    incoming = [source for source, target in graph.links if target == view_index]
    # Exactly one real producer: the g_b_proj Linear.
    assert incoming == [key_to_index[g_b_key]]

    # The recurrent-state Cast never feeds anything inside the block.
    cast_indices = [
        index
        for index, node in enumerate(graph.nodes)
        if "@op_l729_c48_cast" in node.key
    ]
    for cast_index in cast_indices:
        assert not any(source == cast_index for source, _target in graph.links)


def test_glm53_spine_hyperconnection_stays_on_variant_namespace():
    pytest.importorskip("huggingface_hub")
    spec = load_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
    graph = build_merged_model_graph(spec)
    prefix = _linear_attn_variant_prefix(spec)
    variant_namespace = next(
        node["namespace"]
        for node in graph["nodes"]
        if node["id"].startswith(f"{prefix}/input_layernorm/")
    ).rsplit("/", 1)[0]

    attn_nodes = [
        node for node in graph["nodes"] if node["id"].startswith(f"{prefix}/attn_hc/")
    ]
    ffn_nodes = [
        node for node in graph["nodes"] if node["id"].startswith(f"{prefix}/ffn_hc/")
    ]
    assert attn_nodes and ffn_nodes
    for component, component_nodes in (("attn_hc", attn_nodes), ("ffn_hc", ffn_nodes)):
        output_prefix = f"{prefix}/{component}/@output:"
        mirrors = [
            node
            for node in component_nodes
            if node["id"].startswith(output_prefix)
            and any(
                attr.get("key") == "synthetic" and attr.get("value") == "@output_mirror"
                for attr in node.get("attrs", [])
            )
        ]
        internal_nodes = [node for node in component_nodes if node not in mirrors]
        assert all(
            f"/{component}" in node.get("namespace", "") for node in internal_nodes
        )
        assert {node["label"] for node in mirrors} == {"post", "comb", "collapsed"}
        assert all(node.get("namespace") == variant_namespace for node in mirrors)


def test_glm53_hyperconnection_expands_mhc_math():
    pytest.importorskip("huggingface_hub")
    from TraceLens.ModelUtils.computation_graph import build_computation_graph
    from TraceLens.ModelUtils.block_tree import build_block_node

    spec = load_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
    tree = build_block_node(
        attr_name="attn_hc",
        class_name="Glm5NextTextHyperConnection",
        registry=spec.class_registry,
        basic_ops=spec.basic_ops,
        infer_init_steps=True,
    )
    assert len(tree.children) >= 20
    labels = [child.label for child in tree.children]
    assert labels.index("RMSNorm") < labels.index("Linear")

    graph = build_computation_graph(tree, basic_ops=spec.basic_ops)
    labels = {node.label for node in graph.nodes}
    assert {"Square", "Mean", "Reciprocal sqrt"} <= labels
    assert "Sigmoid" in labels
    assert "Softmax" in labels
    assert "Sum" in labels
    assert any(frame.label == "Loop · 19 iterations" for frame in graph.inline_frames)
    assert set(graph.output_ports) == {"post", "comb", "collapsed"}
    carried_out_index = graph.loop_carried_nodes["@op_l291_c19_divide"]
    assert graph.nodes[carried_out_index].label == "Loop carried dependencies out"
    carried_out_inputs = {
        graph.link_port_labels[(source, carried_out_index)]
        for source, target in graph.links
        if target == carried_out_index
    }
    assert carried_out_inputs == {"updated"}
    carried_in_indices = [
        index
        for index, node in enumerate(graph.nodes)
        if node.label == "Loop carried dependencies in"
    ]
    assert len(carried_in_indices) >= 1
    carried_in_inputs = {
        graph.link_port_labels[(source, carried_in_indices[0])]
        for source, target in graph.links
        if target == carried_in_indices[0]
    }
    assert carried_in_inputs == {"initial", "next iteration"}
    assert graph.output_ports["comb"] == carried_out_index

    multiply_index = next(
        index
        for index, node in enumerate(graph.nodes)
        if node.block is not None and node.block.attr_name == "@op_l295_c21_multiply"
    )
    incoming_labels = {
        graph.nodes[source].label
        for source, dest in graph.links
        if dest == multiply_index
    }
    # `collapsed = (pre.unsqueeze(-1) * hidden_streams)` reshapes `pre` first.
    assert incoming_labels == {"Unsqueeze", "hidden_streams"}
    assert "Divide" not in incoming_labels


@pytest.mark.parametrize(
    ("class_name", "required_labels"),
    [
        (
            "Glm5NextRMSNorm",
            {"Power", "Mean", "Add", "Reciprocal sqrt", "Multiply"},
        ),
        (
            "Glm5NextTextRMSNorm",
            {"Power", "Mean", "Add", "Reciprocal sqrt", "Multiply"},
        ),
        (
            "Glm5NextTextRMSNormGated",
            {"Power", "Mean", "Add", "Reciprocal sqrt", "Sigmoid", "Multiply"},
        ),
        (
            "Glm5NextTextUnweightedRMSNorm",
            {"Square", "Mean", "Add", "Reciprocal sqrt", "Multiply"},
        ),
    ],
)
def test_glm53_rmsnorm_classes_expand_real_math(class_name, required_labels):
    pytest.importorskip("huggingface_hub")
    from TraceLens.ModelUtils.block_tree import build_block_node
    from TraceLens.ModelUtils.computation_graph import build_computation_graph

    spec = load_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
    tree = build_block_node(
        attr_name="norm",
        class_name=class_name,
        registry=spec.class_registry,
        basic_ops=spec.basic_ops,
        infer_init_steps=True,
    )
    graph = build_computation_graph(tree, basic_ops=spec.basic_ops)

    assert required_labels <= {node.label for node in graph.nodes}
    assert "RMSNorm" not in {node.label for node in graph.nodes}


def test_glm53_gated_norm_gives_each_forward_parameter_its_own_input():
    """`forward(hidden_states, gate)` reads two tensors, so it shows two inputs."""
    pytest.importorskip("huggingface_hub")
    from TraceLens.ModelUtils.block_tree import build_block_node
    from TraceLens.ModelUtils.computation_graph import build_computation_graph

    spec = load_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
    tree = build_block_node(
        attr_name="o_norm",
        class_name="Glm5NextTextRMSNormGated",
        registry=spec.class_registry,
        basic_ops=spec.basic_ops,
        infer_init_steps=True,
    )
    graph = build_computation_graph(tree, basic_ops=spec.basic_ops)

    inputs = {
        index: node.label
        for index, node in enumerate(graph.nodes)
        if node.synthetic == "@input"
    }
    assert set(inputs.values()) == {"hidden_states", "gate"}

    gate_index = next(index for index, label in inputs.items() if label == "gate")
    consumers = [target for source, target in graph.links if source == gate_index]
    assert consumers, "the gate input has to feed the step that reads it"
    # `sigmoid(gate.to(torch.float32))` retypes the gate before activating it.
    assert {graph.nodes[index].label for index in consumers} == {"Cast"}

    # The two parameters meet at the multiply that scales the norm by the gate.
    sigmoid = next(
        index for index, node in enumerate(graph.nodes) if node.label == "Sigmoid"
    )
    combine = next(target for source, target in graph.links if source == sigmoid)
    assert graph.nodes[combine].label == "Multiply"
    assert len([1 for _source, target in graph.links if target == combine]) == 2


def test_glm53_gated_norm_boundary_inputs_come_from_their_own_producers():
    """Each boundary tile is fed by the producer of the tensor it is named for."""
    pytest.importorskip("huggingface_hub")
    spec = load_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
    graph = build_merged_model_graph(spec, shape_inferencer=ShapeInferencer(spec))
    prefix = _linear_attn_variant_prefix(spec)

    namespace = next(
        node["namespace"]
        for node in graph["nodes"]
        if (node.get("namespace") or "").endswith("Glm5NextTextRMSNormGated")
        and node["id"].startswith(prefix)
    )
    group = [node for node in graph["nodes"] if node.get("namespace") == namespace]
    inputs = {
        node["label"]: node
        for node in group
        if any(
            attr.get("key") == "synthetic" and attr.get("value") == "@input"
            for attr in node.get("attrs", [])
        )
    }
    # The gated norm is called ``self.o_norm(core_attn_out, gate)``: its first
    # boundary tile is named for the actual argument ``core_attn_out`` (the
    # attention output), not the module's own parameter name ``hidden_states``.
    assert set(inputs) == {"core_attn_out", "gate"}

    node_by_id = {node["id"]: node for node in graph["nodes"]}

    def outer_sources(input_node):
        sources = []
        for edge in input_node.get("incomingEdges", []):
            source = node_by_id[edge["sourceNodeId"]]
            if any(
                attr.get("key") == "synthetic" and attr.get("value") == "@input_mirror"
                for attr in source.get("attrs", [])
            ):
                sources.extend(
                    incoming["sourceNodeId"]
                    for incoming in source.get("incomingEdges", [])
                )
            else:
                sources.append(edge["sourceNodeId"])
        return sources

    # The gate arrives from the projection that computes it (or its View
    # reshape), not from the spine.
    gate_sources = outer_sources(inputs["gate"])
    assert gate_sources and all(
        "g_b_proj" in source or "view" in source for source in gate_sources
    )
    hidden_sources = outer_sources(inputs["core_attn_out"])
    assert hidden_sources and not any("g_b_proj" in item for item in hidden_sources)

    gate_consumers = {
        str(node.get("label"))
        for node in group
        if any(
            edge["sourceNodeId"] == inputs["gate"]["id"]
            for edge in node.get("incomingEdges", [])
        )
    }
    assert gate_consumers == {"Cast"}


def test_glm53_o_norm_expands_rmsnorm_math_in_merged_graph():
    pytest.importorskip("huggingface_hub")
    spec = load_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
    graph = build_merged_model_graph(spec, shape_inferencer=ShapeInferencer(spec))
    prefix = _linear_attn_variant_prefix(spec)
    o_norm_nodes = [
        node
        for node in graph["nodes"]
        if node["id"].startswith(f"{prefix}/self_attn/") and "o_norm" in node["id"]
    ]
    labels = {node.get("label") for node in o_norm_nodes}
    assert {"Power", "Mean", "Reciprocal sqrt", "Multiply"} <= labels
    assert "RMSNorm" not in labels


def test_glm53_forget_gate_expands_internal_computation():
    pytest.importorskip("huggingface_hub")
    from TraceLens.ModelUtils.block_tree import build_block_node, collect_function_steps
    from TraceLens.ModelUtils.computation_graph import build_computation_graph

    spec = load_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
    tree = build_block_node(
        attr_name="self_attn",
        class_name="Glm5NextTextLinearAttention",
        registry=spec.class_registry,
        basic_ops=spec.basic_ops,
        infer_init_steps=True,
    )
    forget = next(child for child in tree.children if child.attr_name == "forget_gate")
    assert forget.class_name == "Glm5NextTextForgetGate"
    assert len(collect_function_steps(forget)) >= 6

    graph = build_computation_graph(forget, basic_ops=spec.basic_ops)
    labels = {node.label for node in graph.nodes}
    assert "Linear" in labels
    assert "Sigmoid" in labels
    assert "Multiply" in labels

    merged = build_merged_model_graph(spec)
    prefix = _linear_attn_variant_prefix(spec)
    forget_nodes = [
        node
        for node in merged["nodes"]
        if node["id"].startswith(f"{prefix}/self_attn/") and "forget_gate" in node["id"]
    ]
    forget_labels = {node.get("label") for node in forget_nodes}
    assert "Sigmoid" in forget_labels
    assert "Linear" in forget_labels


def test_glm53_concat_and_forget_gate_branch_ops_have_outgoing_edges():
    pytest.importorskip("huggingface_hub")
    from TraceLens.ModelUtils.block_tree import build_block_node
    from TraceLens.ModelUtils.computation_graph import build_computation_graph

    spec = load_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
    tree = build_block_node(
        attr_name="self_attn",
        class_name="Glm5NextTextLinearAttention",
        registry=spec.class_registry,
        basic_ops=spec.basic_ops,
        infer_init_steps=True,
    )
    graph = build_computation_graph(tree, basic_ops=spec.basic_ops)
    key_to_index = {node.key: index for index, node in enumerate(graph.nodes)}
    sources = {source for source, _target in graph.links}

    concat_key = _graph_key(graph, ":@op_l645_c20_concat:@op_l645_c20_concat:0")
    conv_update_key = _graph_key_for_op(graph, ":@op_l662_c24_causal_conv1d_update:")
    conv_key = _graph_key_for_op(graph, ":@op_l677_c24_causal_conv1d:")
    split_key = _graph_key_for_op(graph, ":@op_l688_c28_split:")
    forget_entry_key = _graph_key(graph, ":forget_gate:f_a_proj:0")
    branch_mul_key = _graph_key_for_op(graph, ":forget_gate:@op_l330_c19_multiply:")
    branch_add_key = _graph_key_for_op(graph, ":forget_gate:@op_l324_c13_add:")
    input_index = next(
        index for index, node in enumerate(graph.nodes) if node.synthetic == "@input"
    )

    assert key_to_index[concat_key] in sources
    # The decode/update and prefill convolution alternatives both consume mixed_qkv;
    # the selected convolution output is then split into query, key, and value.
    links = set(graph.links)
    assert _has_computation_path(
        graph, key_to_index[concat_key], key_to_index[conv_update_key]
    )
    assert _has_computation_path(
        graph, key_to_index[concat_key], key_to_index[conv_key]
    )
    assert _has_computation_path(graph, key_to_index[conv_key], key_to_index[split_key])
    # The block input reaches the forget gate through the now-visible
    # ``apply_mask_to_padding_states`` op (C4b), rather than a direct edge.
    assert _has_computation_path(
        graph, input_index, key_to_index[forget_entry_key]
    )
    assert key_to_index[branch_mul_key] in sources
    assert key_to_index[branch_add_key] in sources


def test_glm53_linear_attention_projections_are_parallel_off_masked_input():
    """q/k/v/b_proj each read *only* the masked-input producer, not a sibling.

    ``q_proj``, ``k_proj``, ``v_proj`` and ``b_proj`` are parallel calls on the
    same ``apply_mask_to_padding_states(hidden_states, ...)`` result -- none of
    them consumes another projection. The SeqSegment sequential-source fallback
    used to spine-chain consecutive submodule calls, fabricating a spurious
    ``q_proj -> k_proj`` edge that rendered a Linear with two tensor inputs.
    Each projection must therefore have exactly one incoming edge, and that edge
    must originate from the shared masked-input op rather than a sibling Linear.
    """
    pytest.importorskip("huggingface_hub")
    from TraceLens.ModelUtils.block_tree import build_block_node
    from TraceLens.ModelUtils.computation_graph import build_computation_graph

    spec = load_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
    tree = build_block_node(
        attr_name="self_attn",
        class_name="Glm5NextTextLinearAttention",
        registry=spec.class_registry,
        basic_ops=spec.basic_ops,
        infer_init_steps=True,
    )
    graph = build_computation_graph(tree, basic_ops=spec.basic_ops)

    proj_indices = {
        proj: next(
            index
            for index, node in enumerate(graph.nodes)
            if node.key.endswith(f":{proj}:{proj}:0")
        )
        for proj in ("q_proj", "k_proj", "v_proj", "b_proj")
    }

    for proj, target in proj_indices.items():
        incoming = [source for source, dest in graph.links if dest == target]
        # A Linear takes a single tensor input; two inputs would render as a
        # spurious two-input node.
        assert len(incoming) == 1, (proj, [graph.nodes[s].key for s in incoming])
        source_key = graph.nodes[incoming[0]].key
        # The one producer is the masked-input op, never a sibling projection.
        assert "apply_mask_to_padding_states" in source_key, (proj, source_key)
        assert not any(
            f":{sibling}:" in source_key for sibling in proj_indices
        ), (proj, source_key)


def test_glm53_flinear_weight_operand_is_hidden():
    """Every rendered ``Linear`` has a single input; learned weights are hidden.

    ``F.linear(input, weight)`` records both operands, so the hyper-connection,
    MoE-experts and router linears each carried a second tensor input -- the
    learned weight (a ``self.fn.float()`` cast, a per-expert ``gate_up_proj``
    gather, a ``self.weight.type(float32)`` cast). Just like the absorbed weight
    of an ``nn.Linear`` submodule, that operand must not be drawn: a ``Linear``
    node has exactly one input, the activation. Zero inputs would mean the
    activation edge was wrongly removed instead.
    """
    pytest.importorskip("huggingface_hub")

    spec = load_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
    graph = build_merged_model_graph(spec, shape_inferencer=ShapeInferencer(spec))

    linear_nodes = [node for node in graph["nodes"] if node.get("label") == "Linear"]
    # The export contains hyper-connection, experts and router F.linear ops.
    assert linear_nodes
    for node in linear_nodes:
        incoming = node.get("incomingEdges", [])
        assert len(incoming) == 1, (node["id"], [e.get("sourceNodeId") for e in incoming])


def test_glm53_hyperconnection_linear_keeps_activation_drops_weight():
    """The surviving hyper-connection linear input is the activation, not ``fn``.

    Removing the weight operand must leave the activation (``flat``) edge intact
    and disconnect the ``self.fn.float()`` weight producer -- confirming the pass
    hides the learned weight rather than the activation. (The disconnected weight
    cast may linger as a dangling leaf in this intermediate per-class graph when
    an inline frame protects it from the leaf strip; the merged export drops it,
    which ``test_glm53_flinear_weight_operand_is_hidden`` covers.)
    """
    pytest.importorskip("huggingface_hub")
    from TraceLens.ModelUtils.block_tree import build_block_node
    from TraceLens.ModelUtils.computation_graph import build_computation_graph

    spec = load_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
    tree = build_block_node(
        attr_name="attn_hc",
        class_name="Glm5NextTextHyperConnection",
        registry=spec.class_registry,
        basic_ops=spec.basic_ops,
        infer_init_steps=True,
    )
    graph = build_computation_graph(tree, basic_ops=spec.basic_ops)

    linear_index = next(
        index
        for index, node in enumerate(graph.nodes)
        if node.block is not None and node.block.attr_name.endswith("_linear")
    )
    incoming = [source for source, dest in graph.links if dest == linear_index]
    assert len(incoming) == 1, [graph.nodes[s].key for s in incoming]
    # The surviving producer is an activation op, never a param-reading weight op.
    survivor = graph.nodes[incoming[0]].block
    assert not (survivor is not None and survivor.external_inputs), (
        graph.nodes[incoming[0]].key,
        survivor.external_inputs if survivor else None,
    )
    # The learned-weight producer that read ``self.fn`` no longer feeds the linear.
    assert not any(
        graph.nodes[source].block is not None
        and "fn" in graph.nodes[source].block.external_inputs
        for source in incoming
    )


def test_glm53_dead_code_elimination_is_idempotent_for_hyperconnection():
    pytest.importorskip("huggingface_hub")
    from TraceLens.ModelUtils.block_tree import build_block_node
    from TraceLens.ModelUtils.computation_graph import (
        _apply_dead_code_elimination,
        build_computation_graph,
    )

    spec = load_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
    tree = build_block_node(
        attr_name="ffn_hc",
        class_name="Glm5NextTextHyperConnection",
        registry=spec.class_registry,
        basic_ops=spec.basic_ops,
        infer_init_steps=True,
    )
    graph = build_computation_graph(
        tree,
        basic_ops=spec.basic_ops,
        strip_unused_return_branches=True,
    )
    once = _apply_dead_code_elimination(
        graph,
        tree,
        strip_unused_return_branches=True,
    )
    twice = _apply_dead_code_elimination(
        once,
        tree,
        strip_unused_return_branches=True,
    )
    assert len(once.nodes) == len(twice.nodes)
    assert set(once.links) == set(twice.links)
    assert not twice.dead_node_indices


def test_glm53_ffn_hc_input_norm_precedes_linear():
    pytest.importorskip("huggingface_hub")
    from TraceLens.Visualizer.model_explorer_export.merge import _resolve_section_tree_for_component
    from TraceLens.ModelUtils.basic_ops import BasicOpFilter
    from TraceLens.ModelUtils.block_tree import build_block_node

    spec = load_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
    hc = spec.class_registry["Glm5NextTextHyperConnection"]
    assert hc.forward_calls.index("input_norm") < next(
        index
        for index, step in enumerate(hc.forward_calls)
        if step.startswith("@op_") and hc.forward_operations[step].label == "Linear"
    )
    assert "@functional_linear" not in hc.forward_calls

    tree = build_block_node(
        attr_name="ffn_hc",
        class_name="Glm5NextTextHyperConnection",
        registry=spec.class_registry,
        basic_ops=spec.basic_ops,
        infer_init_steps=True,
    )
    labels = [child.label for child in tree.children]
    assert labels.index("RMSNorm") < labels.index("Linear")
    assert len(tree.children) >= 20

    variant = next(
        v for v in spec.layer_variants if "LinearAttention" in (v.attention_class or "")
    )
    ffn_hc = next(c for c in spec.block_components if c.attr_name == "ffn_hc")
    title, section_tree = _resolve_section_tree_for_component(
        spec,
        ffn_hc,
        variant=variant,
        basic_ops=BasicOpFilter.for_detailed(),
    )
    assert title == "FFN"
    section_labels = [child.label for child in section_tree.children]
    assert section_labels.index("RMSNorm") < section_labels.index("Linear")


def test_glm53_hyperconnection_feeds_single_output_to_next_norm():
    pytest.importorskip("huggingface_hub")
    spec = load_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
    graph = build_merged_model_graph(spec)
    prefix = _linear_attn_variant_prefix(spec)
    node_by_id = {node["id"]: node for node in graph["nodes"]}

    source_component = {
        "input_layernorm": "attn_hc",
        "post_attention_layernorm": "ffn_hc",
    }
    for target in ("input_layernorm", "post_attention_layernorm"):
        node = next(
            candidate
            for candidate in graph["nodes"]
            if candidate["id"].startswith(f"{prefix}/{target}/")
            and candidate.get("label") == "Power"
        )
        incoming = node.get("incomingEdges", [])
        assert len(incoming) == 1, (target, incoming)
        output_id = f"{prefix}/{source_component[target]}/@output:collapsed"
        source = node_by_id[output_id]
        # The boundary inside the block reads the same as its mirror outside.
        assert source.get("label") == "collapsed"
        assert any(
            attr.get("key") == "synthetic" and attr.get("value") == "@output"
            for attr in source.get("attrs", [])
        )
        assert [item["id"] for item in source["outputsMetadata"]] == ["collapsed"]
        mirror_id = f"{output_id}^collapsed"
        norm_input_id = f"{prefix}/{target}/@input"
        norm_output_id = f"{prefix}/{target}/@output"
        mirror = node_by_id[mirror_id]
        assert mirror.get("label") == "collapsed"
        assert any(
            attr.get("key") == "synthetic" and attr.get("value") == "@output_mirror"
            for attr in mirror.get("attrs", [])
        )
        assert (
            node_by_id[norm_input_id]["incomingEdges"][0]["sourceNodeId"] == mirror_id
        )
        # RMSNorm casts to float32 before squaring, so the cast is what the block
        # boundary hands the norm math.
        cast_id = incoming[0]["sourceNodeId"]
        assert node_by_id[cast_id].get("label") == "Cast"
        assert node_by_id[cast_id]["incomingEdges"][0]["sourceNodeId"] == norm_input_id
        assert _has_export_path(graph["nodes"], output_id, node["id"])
        assert norm_output_id in node_by_id
        norm_output = node_by_id[norm_output_id]
        assert [item["id"] for item in norm_output["outputsMetadata"]] == [
            "hidden_states"
        ]
        assert norm_output["label"] == "hidden_states"
        # A single-output block needs no mirror; the Output already names the tensor.
        assert f"{norm_output_id}^hidden_states" not in node_by_id

        output_prefix = f"{prefix}/{source_component[target]}/@output:"
        output_nodes = [
            candidate
            for candidate in graph["nodes"]
            if candidate["id"].startswith(output_prefix)
            and any(
                attr.get("key") == "synthetic" and attr.get("value") == "@output"
                for attr in candidate.get("attrs", [])
            )
        ]
        assert {
            candidate["id"].removeprefix(output_prefix) for candidate in output_nodes
        } == {
            "post",
            "comb",
            "collapsed",
        }
        for output in output_nodes:
            port = output["outputsMetadata"][0]["id"]
            output_mirror = node_by_id[f"{output['id']}^{port}"]
            assert output_mirror["label"] == port
            assert output_mirror["incomingEdges"][0]["sourceNodeId"] == output["id"]
            assert output_mirror["incomingEdges"][0]["sourceNodeOutputId"] == port

    hc = spec.class_registry["Glm5NextTextHyperConnection"]
    assert hc.primary_return_slot == "collapsed"
    assert set(hc.forward_return_order) == {"post", "comb", "collapsed"}

    ffn_nodes = [
        node for node in graph["nodes"] if node["id"].startswith(f"{prefix}/ffn_hc/")
    ]
    ffn_labels = {node.get("label") for node in ffn_nodes}
    assert {"Softmax", "Divide"} <= ffn_labels
    for slot in ("post", "comb"):
        producer = hc.forward_return_slots[slot]
        assert any(producer in node["id"] for node in ffn_nodes)

    input_namespaces = {
        node.get("namespace", "")
        for node in graph["nodes"]
        if any(
            attr.get("key") == "synthetic" and attr.get("value") == "@input"
            for attr in node.get("attrs", [])
        )
    }
    output_namespaces = {
        node.get("namespace", "")
        for node in graph["nodes"]
        if any(
            attr.get("key") == "synthetic" and attr.get("value") == "@output"
            for attr in node.get("attrs", [])
        )
    }
    # Every real sub-module boundary that owns an Output also owns an Input.
    assert output_namespaces <= input_namespaces


def test_glm53_decoder_residual_ops_use_return_slot_producers():
    pytest.importorskip("huggingface_hub")
    spec = load_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
    decoder = spec.class_registry["Glm5NextTextDecoderLayer"]
    attn_hc = spec.class_registry["Glm5NextTextHyperConnection"]
    matmul = decoder.forward_operations["@op_l1319_c85_matmul"]
    multiply = decoder.forward_operations["@op_l1319_c24_multiply"]
    operations = decoder.forward_operations

    def depends_on(operation, producer):
        pending = list(operation.predecessors)
        visited = set()
        while pending:
            predecessor = pending.pop()
            if predecessor == producer:
                return True
            if predecessor in visited:
                continue
            visited.add(predecessor)
            nested = operations.get(predecessor)
            if nested is not None:
                pending.extend(nested.predecessors)
        return False

    assert depends_on(matmul, attn_hc.forward_return_slots["comb"])
    assert depends_on(multiply, attn_hc.forward_return_slots["post"])
    assert "attn_hc" not in matmul.predecessors
    assert "attn_hc" not in multiply.predecessors


def test_glm53_ffn_hc_expands_hyperconnection_not_moe():
    pytest.importorskip("huggingface_hub")
    from TraceLens.Visualizer.model_explorer_export.merge import _resolve_section_tree_for_component
    from TraceLens.Visualizer.model_explorer_export.overview import component_has_detail_section
    from TraceLens.ModelUtils.basic_ops import BasicOpFilter

    spec = load_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
    variant = next(
        v for v in spec.layer_variants if "LinearAttention" in (v.attention_class or "")
    )
    prefix = _linear_attn_variant_prefix(spec)
    ffn_hc = next(c for c in spec.block_components if c.attr_name == "ffn_hc")

    assert component_has_detail_section(ffn_hc, spec)
    title, _tree = _resolve_section_tree_for_component(
        spec,
        ffn_hc,
        variant=variant,
        basic_ops=BasicOpFilter.for_detailed(),
    )
    assert title == "FFN"

    graph = build_merged_model_graph(spec, shape_inferencer=ShapeInferencer(spec))
    ffn_nodes = [
        node for node in graph["nodes"] if node["id"].startswith(f"{prefix}/ffn_hc/")
    ]
    labels = {node.get("label") for node in ffn_nodes}
    assert {"Square", "Mean", "Reciprocal sqrt"} <= labels
    assert "Linear" in labels
    assert all("TopkRouter" not in node["id"] for node in ffn_nodes)
    assert all(
        "/ffn_hc" in node.get("namespace", "")
        for node in ffn_nodes
        if not any(
            attr.get("key") == "synthetic" and attr.get("value") == "@output_mirror"
            for attr in node.get("attrs", [])
        )
    )
    outputs = [
        node
        for node in ffn_nodes
        if "/@output:" in node["id"]
        and any(
            attr.get("key") == "synthetic" and attr.get("value") == "@output"
            for attr in node.get("attrs", [])
        )
    ]
    assert len(outputs) == 3
    assert all(len(node["outputsMetadata"]) == 1 for node in outputs)
    node_by_id = {node["id"]: node for node in graph["nodes"]}
    for output in outputs:
        port = output["outputsMetadata"][0]["id"]
        mirror = node_by_id[f"{output['id']}^{port}"]
        assert mirror["label"] == port
        # Boundary and mirror name the same tensor on both sides of the block.
        assert output["label"] == port
        assert mirror["incomingEdges"][0]["sourceNodeId"] == output["id"]
    assert {node["label"] for node in outputs} == {"post", "comb", "collapsed"}
    boundary = graph["groupNodeAttributes"][outputs[0]["namespace"]]
    assert boundary["input_shape"] == "[B, S, 4, 4096] bfloat16"
    assert boundary["output_shape"] == (
        "post: [B, S, 4] float32, comb: [B, S, 4, 4] float32, "
        "collapsed: [B, S, 4096] bfloat16"
    )


def test_glm53_expert_helper_stays_inside_loop_without_cycle():
    pytest.importorskip("huggingface_hub")
    spec = load_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
    graph = build_merged_model_graph(spec)
    prefix = _linear_attn_variant_prefix(spec)

    helper_nodes = [
        node
        for node in graph["nodes"]
        if node["id"].startswith(f"{prefix}/mlp/")
        and ":experts:_apply_gate:" in node["id"]
        and not any(
            attr.get("key") == "synthetic" and attr.get("value") == "@output_mirror"
            for attr in node.get("attrs", [])
        )
    ]
    assert helper_nodes
    assert all(
        "/Glm5NextTextMoE/Loop_288_iterations/_apply_gate" in node.get("namespace", "")
        for node in helper_nodes
    )
    _assert_export_is_acyclic(graph["nodes"])


def test_glm53_moe_keeps_only_the_live_residual_add():
    pytest.importorskip("huggingface_hub")
    spec = load_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
    graph = build_merged_model_graph(spec)
    prefix = _linear_attn_variant_prefix(spec)
    nodes = graph["nodes"]

    assert not any(
        node["id"].startswith(f"{prefix}/mlp/residual_add:") for node in nodes
    )
    add = next(
        node
        for node in nodes
        if node["id"].startswith(f"{prefix}/mlp/")
        and ":@op_l207_c24_add:" in node["id"]
    )
    assert any(
        edge["sourceNodeId"] == add["id"]
        for node in nodes
        for edge in node.get("incomingEdges", [])
    )


def test_glm53_expert_loop_inputs_are_separate_and_index_add_is_basic():
    pytest.importorskip("huggingface_hub")
    from TraceLens.ModelUtils.block_tree import build_block_node
    from TraceLens.ModelUtils.computation_graph import build_computation_graph

    spec = load_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
    graph = build_merged_model_graph(spec)
    prefix = _linear_attn_variant_prefix(spec)
    loop_inputs = [
        node
        for node in graph["nodes"]
        if node["id"].startswith(f"{prefix}/mlp/")
        and node.get("namespace", "").endswith("/Glm5NextTextMoE/Loop_288_iterations")
        and any(
            attr.get("key") == "synthetic" and attr.get("value") == "@input"
            for attr in node.get("attrs", [])
        )
    ]
    # The loop-carried value ``final`` renders as a carried-dependency boundary
    # (in/out) nested inside the loop frame, not as a plain ``@input``. Its id
    # carries the loop id and variable name.
    carried = [
        node
        for node in graph["nodes"]
        if node["id"].startswith(f"{prefix}/mlp/")
        and node.get("namespace", "").endswith("/Glm5NextTextMoE/Loop_288_iterations")
        and any(
            attr.get("key") == "synthetic" and attr.get("value") == "@loop_carried"
            for attr in node.get("attrs", [])
        )
    ]
    assert any(node["id"].endswith(":final") for node in carried)
    assert all(len(node.get("incomingEdges", [])) <= 1 for node in loop_inputs)
    assert not any(
        "nonzero" in edge["sourceNodeId"]
        for node in loop_inputs
        for edge in node.get("incomingEdges", [])
    )
    # The expert dispatch flattens into the MoE scope, so its boundary inputs are
    # mirrored directly under ``Glm5NextTextMoE``. The routed activations
    # (``hidden_states``) cross into the loop from *outside* the MoE, so they keep a
    # boundary mirror. ``topk_weights`` is produced by the sibling router in this
    # same MoE scope: its output mirror feeds the loop input directly, and the
    # redundant 1:1 ``@input_mirror`` pass-through is collapsed away (D4 -- a
    # ``topk_weights`` input tile that went into no block). ``topk_indices`` feeds
    # the MoE-scope ``one_hot`` preamble at the router's scope, so it never needed a
    # mirror.
    expert_input_mirrors = [
        node
        for node in graph["nodes"]
        if node["id"].startswith(f"{prefix}/mlp/")
        and any(
            attr.get("key") == "synthetic" and attr.get("value") == "@input_mirror"
            for attr in node.get("attrs", [])
        )
        and node.get("namespace", "").endswith("/Glm5NextTextMoE")
    ]
    mirror_labels = {node["label"] for node in expert_input_mirrors}
    assert "hidden_states" in mirror_labels
    assert "topk_weights" not in mirror_labels

    experts = build_block_node(
        attr_name="experts",
        class_name="Glm5NextTextExperts",
        registry=spec.class_registry,
        basic_ops=spec.basic_ops,
        infer_init_steps=True,
    )
    computation = build_computation_graph(experts, basic_ops=spec.basic_ops)
    index_add = next(node for node in computation.nodes if node.label == "Index add")
    assert index_add.block is not None
    assert index_add.block.is_basic


def test_glm53_loop_carried_pairs_are_well_formed():
    """Every loop-carried variable is one nested in/out pair with a single back edge.

    General loop-rendering invariant (not GLM-specific): for each ``@loop_carried``
    variable the export must emit exactly one ``@loop_carried_in`` and one
    ``@loop_carried_out`` node, both nested in the *same* loop namespace (not
    siblings of the loop), joined by exactly one back edge (out → in). No orphan
    ``out`` without its ``in``. The whole export stays acyclic once those single
    back edges are removed. The expert loop specifically carries ``final`` (not the
    ``mask`` intermediate) under a counted ``Loop_288_iterations`` frame.
    """
    pytest.importorskip("huggingface_hub")
    spec = load_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
    graph = build_merged_model_graph(spec)
    nodes = graph["nodes"]

    def is_carried(node) -> bool:
        return any(
            attr.get("key") == "synthetic" and attr.get("value") == "@loop_carried"
            for attr in node.get("attrs", [])
        )

    # key = (prefix, loop_key, var) -> {"in": node, "out": node}
    pairs: dict[tuple[str, str, str], dict[str, dict]] = {}
    for node in nodes:
        node_id = node["id"]
        for marker, direction in (
            ("@loop_carried_in:", "in"),
            ("@loop_carried_out:", "out"),
        ):
            if marker in node_id:
                assert is_carried(node), node_id
                prefix, rest = node_id.split(marker, 1)
                loop_key, var = rest.split(":", 1)
                slot = pairs.setdefault((prefix, loop_key, var), {})
                assert direction not in slot, f"duplicate {direction} for {node_id}"
                slot[direction] = node
                break

    assert pairs, "expected loop-carried boundaries in the export"

    for (prefix, loop_key, var), slot in pairs.items():
        # Exactly one in and one out — no orphan boundary.
        assert set(slot) == {"in", "out"}, (loop_key, var, sorted(slot))
        in_node, out_node = slot["in"], slot["out"]
        # Both boundaries live in the *same* namespace — nested together inside
        # the loop body, never split so that one is a sibling of the other. (A
        # compactly-rendered loop puts them under a ``Loop_N_iterations`` frame;
        # an inline-expanded loop such as the vision block shares the block's own
        # namespace. Either way, in and out agree.)
        assert in_node.get("namespace") == out_node.get("namespace")
        # Exactly one back edge: the in node is fed by its matching out node.
        back_edges = [
            edge
            for edge in in_node.get("incomingEdges", [])
            if edge["sourceNodeId"] == out_node["id"]
        ]
        assert len(back_edges) == 1, (loop_key, var, back_edges)

    # The expert loop carries ``final`` under a counted 288-iteration frame, and
    # never the ``mask`` intermediate that a scope collision used to mis-resolve.
    carried_vars_by_loop_ns = {
        (var, slot["in"].get("namespace", "").rsplit("/", 1)[-1])
        for (_prefix, _loop_key, var), slot in pairs.items()
    }
    assert ("final", "Loop_288_iterations") in carried_vars_by_loop_ns
    assert not any(var == "mask" for var, _ns in carried_vars_by_loop_ns)

    _assert_export_is_acyclic(nodes)


def test_glm53_decoder_boundary_keeps_hyper_stream_shape():
    pytest.importorskip("huggingface_hub")
    spec = load_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
    graph = build_merged_model_graph(spec, shape_inferencer=ShapeInferencer(spec))
    boundary = graph["groupNodeAttributes"]["45x_Glm5NextTextDecoderLayer"]

    assert boundary["input_shape"] == "[B, S, 4, 4096] bfloat16"
    # The hyper-stream shape dominates; the standard attention variant also
    # sends its collapsed output (B x S x 4096) across the boundary.
    assert "[B, S, 4, 4096] bfloat16" in boundary["output_shape"]

    prefix = _linear_attn_variant_prefix(spec)
    node_by_id = {node["id"]: node for node in graph["nodes"]}
    for operation_id in ("@op_l1319_c24_add", "@op_l1328_c24_add"):
        residual_add = node_by_id[f"{prefix}/{operation_id}"]
        output_shape = next(
            attr["value"]
            for attr in residual_add["attrs"]
            if attr["key"] == "output_shape"
        )
        assert output_shape == "[B, S, 4, 4096] bfloat16"


def test_glm53_spine_norms_do_not_share_a_namespace():
    """Merging both norms into one group made the spine look like it loops."""
    pytest.importorskip("huggingface_hub")
    spec = load_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
    graph = build_merged_model_graph(spec)
    prefix = _linear_attn_variant_prefix(spec)
    namespace_of = {node["id"]: node.get("namespace", "") for node in graph["nodes"]}

    def section_namespace(attr: str) -> str:
        return namespace_of[f"{prefix}/{attr}/@input"]

    input_ns = section_namespace("input_layernorm")
    post_ns = section_namespace("post_attention_layernorm")
    assert input_ns != post_ns

    # A shared namespace made a norm both feed and be fed by the same group, which
    # is what read as a loop.
    reaches: dict[str, set[str]] = {}
    for node in graph["nodes"]:
        target = namespace_of.get(node["id"], "")
        for edge in node.get("incomingEdges", []):
            source = namespace_of.get(edge["sourceNodeId"], "")
            if source != target:
                reaches.setdefault(source, set()).add(target)
    for norm_ns in (input_ns, post_ns):
        both_ways = {
            other
            for other in reaches.get(norm_ns, set())
            if norm_ns in reaches.get(other, set())
        }
        assert not both_ways, (norm_ns, both_ways)

    # The real order is attn_hc -> input_layernorm -> self_attn.
    norm_output = f"{prefix}/input_layernorm/@output"
    attention_input = next(
        node for node in graph["nodes"] if node["id"] == f"{prefix}/self_attn/@input"
    )
    assert attention_input["incomingEdges"][0]["sourceNodeId"] == norm_output


def test_glm53_attention_lora_norms_do_not_share_a_namespace():
    """q_a_layernorm and kv_a_layernorm are two RMSNorms, not one looping group."""
    pytest.importorskip("huggingface_hub")
    spec = load_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
    graph = build_merged_model_graph(spec)

    def namespace_for(attr: str) -> str:
        node = next(
            candidate
            for candidate in graph["nodes"]
            if candidate.get("label") == "Power"
            and (
                f":{attr}:" in candidate["id"]
                or str(candidate.get("namespace", "")).endswith(f"/{attr}")
            )
        )
        return str(node.get("namespace", ""))

    q_ns = namespace_for("q_a_layernorm")
    kv_ns = namespace_for("kv_a_layernorm")
    assert q_ns.endswith("/q_a_layernorm")
    assert kv_ns.endswith("/kv_a_layernorm")
    assert q_ns != kv_ns

    residual = next(
        node
        for node in graph["nodes"]
        if node["id"].endswith("/@op_l1319_c55_unsqueeze")
        and "Glm5NextTextAttention_Glm5NextTextMoE" in node["id"]
    )
    assert residual["incomingEdges"][0]["sourceNodeId"].endswith(
        "/self_attn/@output:attn_output"
    )


def test_glm53_operation_tile_colors_are_consistent_per_label():
    """One op must not render gray in one block and white in another."""
    pytest.importorskip("huggingface_hub")
    spec = load_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
    graph = build_merged_model_graph(spec)

    fills: dict[str, set[str]] = {}
    for node in graph["nodes"]:
        style = node.get("style")
        assert isinstance(style, dict), node["id"]
        # Boundary ports are deliberately colored by direction, not by op identity,
        # and they carry tensor names rather than operation names.
        if any(attr.get("key") == "synthetic" for attr in node.get("attrs", [])):
            continue
        label = node.get("label", "")
        fills.setdefault(label, set()).add(style.get("backgroundColor"))

    inconsistent = {label: colors for label, colors in fills.items() if len(colors) > 1}
    assert not inconsistent, inconsistent

    # Computation is gray; layout-only data movement is white.
    for label in ("Multiply", "Add", "MatMul", "Power", "Mean", "Linear"):
        assert fills[label] == {"#bdc3c7"}, (label, fills[label])
    for label in (
        "Unsqueeze",
        "Expand",
        "Contiguous",
        "Cast",
        "Transpose",
        "Split",
        "Concat",
    ):
        assert fills[label] == {"#ffffff"}, (label, fills[label])


def test_glm53_decoder_input_uses_source_data_movement_chain():
    pytest.importorskip("huggingface_hub")
    spec = load_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
    graph = build_merged_model_graph(spec, shape_inferencer=ShapeInferencer(spec))
    model_ops = [
        node for node in graph["nodes"] if node["id"].startswith("@model_forward/")
    ]

    assert [node["label"] for node in model_ops] == [
        "Unsqueeze",
        "Expand",
        "Contiguous",
    ]
    # The decoder consumes the vision/text combine (masked_scatter), not the raw
    # token embeddings — the combine is the true entry to the language stack.
    assert (
        model_ops[0]["incomingEdges"][0]["sourceNodeId"]
        == "@vision_language_combine"
    )
    combine = next(
        node for node in graph["nodes"] if node["id"] == "@vision_language_combine"
    )
    combine_sources = {
        edge["sourceNodeId"] for edge in combine["incomingEdges"]
    }
    assert combine_sources == {"embed_tokens", "@image_mask", "visual/@output"}
    assert model_ops[1]["incomingEdges"][0]["sourceNodeId"] == model_ops[0]["id"]
    assert model_ops[2]["incomingEdges"][0]["sourceNodeId"] == model_ops[1]["id"]
    assert not any(node["id"] == "rotary_pos_emb" for node in graph["nodes"])
    assert [
        next(attr["value"] for attr in node["attrs"] if attr["key"] == "output_shape")
        for node in model_ops
    ] == [
        "[B, S, 1, 4096] bfloat16",
        "[B, S, 4, 4096] bfloat16",
        "[B, S, 4, 4096] bfloat16",
    ]


def test_glm53_visual_loop_carried_in_is_consumed_and_precedes_body():
    pytest.importorskip("huggingface_hub")
    spec = load_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
    graph = build_merged_model_graph(spec)
    nodes = graph["nodes"]

    # The graph must never ship a cycle even with the vision loop inlined.
    _assert_export_is_acyclic(nodes)

    lc_in = next(
        node
        for node in nodes
        if "visual/@loop_carried_in:" in node["id"]
    )
    # Wiring: the loop-carried-in must actually feed the loop body (mirroring the
    # decoder LC nodes), not sit dead like ``patch_embed/@output``-only.
    consumers = [
        node["id"]
        for node in nodes
        for edge in node.get("incomingEdges", [])
        if edge["sourceNodeId"] == lc_in["id"]
    ]
    assert consumers, "visual @loop_carried_in has no consumer"
    assert any(
        consumer.startswith("visual/seq:3:blocks") for consumer in consumers
    ), consumers

    # Topological order: the LC-in floats above every loop-body node even though
    # the body lives in child namespaces (``visual/Block/norm1`` etc.).
    positions = {node["id"]: index for index, node in enumerate(nodes)}
    body_positions = [
        index
        for node in nodes
        if node["id"].startswith("visual/seq:3:blocks")
        and "@loop_carried" not in node["id"]
        for index in (positions[node["id"]],)
    ]
    assert body_positions
    assert positions[lc_in["id"]] < min(body_positions)


def _output_shape(node) -> str | None:
    return next(
        (attr["value"] for attr in node.get("attrs", []) if attr["key"] == "output_shape"),
        None,
    )


def test_glm53_vision_tower_carries_patch_axis_not_text_seq():
    """The vision tower must use the ``Pv`` patch axis, never the text ``B*S``.

    Bug 3B: the whole vision stack conflated its patch axis with the LLM
    sequence, stamping text ``B*S``/``B*S/4`` symbols onto vision activations.
    Every vision activation should now read ``Pv`` (raw patches) or ``Pv/4``
    (after the 2x2 spatial merge), while the text/decoder path is untouched.
    """
    pytest.importorskip("huggingface_hub")
    spec = load_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
    graph = build_merged_model_graph(spec, shape_inferencer=ShapeInferencer(spec))
    nodes = graph["nodes"]
    node_by_id = {node["id"]: node for node in nodes}

    _assert_export_is_acyclic(nodes)

    # No vision node may carry the text sequence symbol.
    vision_nodes = [n for n in nodes if str(n["id"]).startswith("visual")]
    assert vision_nodes
    for node in vision_nodes:
        shape = _output_shape(node)
        if shape is None:
            continue
        assert "B*S" not in shape, (node["id"], shape)
        assert not shape.lstrip("[").startswith("B,"), (node["id"], shape)

    # At least some vision activations carry the distinct patch axis.
    assert any("Pv" in (_output_shape(n) or "") for n in vision_nodes)

    # Patch-embed pipeline: [Pv,1176] -> view [Pv,3,2,14,14] -> Conv3d
    # [Pv,1024,1,1,1] -> view [Pv,1024].
    pe = "visual/seq:1:patch_embed:patch_embed:0"
    assert _output_shape(node_by_id[f"{pe}/@input"]) == "[Pv, 1176] bfloat16"
    assert (
        _output_shape(node_by_id[f"{pe}/seq:0:@op_l1713_c24_view:@op_l1713_c24_view:0"])
        == "[Pv, 3, 2, 14, 14] bfloat16"
    )
    assert (
        _output_shape(node_by_id[f"{pe}/seq:2:proj:proj:0"])
        == "[Pv, 1024, 1, 1, 1] bfloat16"
    )
    assert _output_shape(node_by_id[f"{pe}/@output"]) == "[Pv, 1024] bfloat16"

    # After the spatial merge the merger projects the pooled [Pv/4, 4096] rows.
    merger_out = node_by_id["visual/seq:9:merger/@output"]
    assert _output_shape(merger_out) == "[Pv/4, 4096] bfloat16"
    assert len(merger_out.get("incomingEdges", [])) == 1

    # The vision-language combine reconciles the [Pv/4,4096] image rows with the
    # independent text sequence and stays [B,S,4096] with three sources.
    combine = node_by_id["@vision_language_combine"]
    assert _output_shape(combine) == "[B, S, 4096] bfloat16"
    assert len(combine.get("incomingEdges", [])) == 3

    # Decoder path is untouched: the text RMSNorm still speaks [B, S, H].
    text_norm = next(
        n
        for n in nodes
        if "kv_a_layernorm:@op_l78_c24_cast" in n["id"]
    )
    assert "Pv" not in (_output_shape(text_norm) or "")
    assert (_output_shape(text_norm) or "").startswith("[B, S,")


def test_glm53_forget_gate_has_real_boundary_nodes():
    pytest.importorskip("huggingface_hub")
    spec = load_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
    graph = build_merged_model_graph(spec)
    prefix = _linear_attn_variant_prefix(spec)
    forget_nodes = [
        node
        for node in graph["nodes"]
        if node["id"].startswith(f"{prefix}/self_attn/")
        and "Glm5NextTextForgetGate" in node.get("namespace", "")
    ]

    assert forget_nodes
    forget_input = next(
        node
        for node in forget_nodes
        if any(
            attr.get("key") == "synthetic" and attr.get("value") == "@input"
            for attr in node.get("attrs", [])
        )
    )
    forget_output = next(
        node
        for node in forget_nodes
        if any(
            attr.get("key") == "synthetic" and attr.get("value") == "@output"
            for attr in node.get("attrs", [])
        )
    )
    assert forget_input["label"] == "hidden_states"
    assert [item["id"] for item in forget_output["outputsMetadata"]] == ["g"]
    terminal_multiply = next(
        node
        for node in forget_nodes
        if node.get("label") == "Multiply"
        and forget_output["incomingEdges"][0]["sourceNodeId"] == node["id"]
    )
    # One returned tensor, so the Output carries the name on its own.
    assert forget_output["label"] == "g"
    assert not any(node["id"] == f"{forget_output['id']}^g" for node in graph["nodes"])
    attention = next(
        node
        for node in graph["nodes"]
        if node["id"].startswith(prefix) and ":@attention:" in node["id"]
    )
    assert _has_export_path(graph["nodes"], terminal_multiply["id"], attention["id"])


def test_glm53_norm_boundary_connects_directly_to_attention_input():
    """One-tensor blocks hand off Output to Input without a mirror in between."""
    pytest.importorskip("huggingface_hub")
    spec = load_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
    graph = build_merged_model_graph(spec)
    prefix = _linear_attn_variant_prefix(spec)
    node_by_id = {node["id"]: node for node in graph["nodes"]}
    norm_prefix = f"{prefix}/input_layernorm/"

    norm_input = node_by_id[f"{norm_prefix}@input"]
    norm_output = node_by_id[f"{norm_prefix}@output"]
    assert norm_input["label"] == "hidden_states"
    assert norm_output["label"] == "hidden_states"
    assert [item["id"] for item in norm_output["outputsMetadata"]] == ["hidden_states"]
    assert f"{norm_prefix}@output^hidden_states" not in node_by_id

    attention_input = node_by_id[f"{prefix}/self_attn/@input"]
    assert attention_input["incomingEdges"][0]["sourceNodeId"] == norm_output["id"]
    assert _has_export_path(graph["nodes"], norm_input["id"], attention_input["id"])


def test_glm53_hyper_head_precedes_final_norm():
    pytest.importorskip("huggingface_hub")
    spec = load_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
    assert [component.attr_name for component in spec.stack_tail] == ["hc_head", "norm"]

    graph = build_merged_model_graph(spec, shape_inferencer=ShapeInferencer(spec))
    node_by_id = {node["id"]: node for node in graph["nodes"]}
    norm_input = node_by_id["norm/@input"]
    norm_output = node_by_id["norm/@output"]
    assert norm_input["incomingEdges"][0]["sourceNodeId"] == "hc_head"
    assert [item["id"] for item in norm_output["outputsMetadata"]] == ["hidden_states"]
    assert norm_output["label"] == "hidden_states"
    assert "norm/@output^hidden_states" not in node_by_id
    assert _has_export_path(graph["nodes"], "hc_head", norm_output["id"])


def test_glm53_vision_attention_resolves_single_kernel_branch():
    """The vision attention flash-vs-fallback branch resolves to one kernel.

    ``Glm5NextVisionAttention.forward`` branches on
    ``if is_flash_attention_requested(self.config): ...`` between a fused flash
    kernel and a per-chunk ``torch.cat`` fallback. GLM-5.3-Flash ships
    ``_attn_implementation`` unset, which transformers defaults to ``sdpa`` — not
    flash — so only the fallback branch runs. Resolving the predicate from the
    checkpoint config (C2) must leave exactly one attention kernel node, one real
    per-chunk ``Concat`` fed by that kernel (no phantom ``Concat`` from the
    untaken branch, no ``Concat`` self-loop, no duplicated ``@kernel_out`` node),
    and an output reshape reading that single ``Concat``. The graph stays acyclic.
    """
    pytest.importorskip("huggingface_hub")
    spec = load_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
    graph = build_merged_model_graph(spec)
    nodes = graph["nodes"]
    node_by_id = {node["id"]: node for node in nodes}

    _assert_export_is_acyclic(nodes)

    attn_nodes = [n for n in nodes if "VisionAttention" in n.get("namespace", "")]
    assert attn_nodes

    # Exactly one attention kernel node, tagged with the resolved implementation.
    kernels = [n for n in attn_nodes if n.get("label") == "Attention"]
    assert len(kernels) == 1, [n["id"] for n in kernels]
    kernel = kernels[0]
    assert any(
        attr.get("key") == "attn_implementation" and attr.get("value") == "sdpa"
        for attr in kernel.get("attrs", [])
    )

    # No duplicated kernel-output node survives from the untaken flash branch.
    assert not [n for n in attn_nodes if "@kernel_out" in n["id"]]

    # No per-chunk reassembly Concat survives: ``torch.cat`` over the windowed
    # comprehension collapses to one representative kernel output, so the cat has a
    # single input and restores the same ``[Pv, 1024]`` it received -- a provable
    # no-op that ``_elide_noop_single_input_concat`` folds onto the kernel (the
    # block-diagonal windowing is already carried by the kernel's ``cu_seqlens``).
    # The rope helper's own ``rotate_half`` Concats live in the expanded
    # ``apply_rotary_pos_emb_vision`` frame (a separate computation) and are
    # excluded here.
    concats = [
        n
        for n in attn_nodes
        if n.get("label") == "Concat"
        and "apply_rotary_pos_emb_vision" not in n["id"]
    ]
    assert concats == [], [n["id"] for n in concats]

    # The output reshape now reads the single kernel directly (the elided cat was
    # a pass-through between them).
    reshape = node_by_id["visual/seq:3:blocks:attn:@op_l1665_c22_reshape:14"]
    reshape_sources = [e["sourceNodeId"] for e in reshape.get("incomingEdges", [])]
    assert reshape_sources == [kernel["id"]]


def test_glm53_vision_attention_flags_impl_dead_interface_input():
    """``max_seqlen`` is declared interface but dead under sdpa, so it is flagged.

    ``Glm5NextVisionAttention.forward`` takes packed-attention metadata by keyword
    (``cu_seqlens``, ``max_seqlen``). Only the flash branch reads ``max_seqlen``
    (``get_max_seqlen(...)`` / ``max_length_q=``); the resolved sdpa branch consumes
    ``cu_seqlens`` (``cu_seqlens[1:] - cu_seqlens[:-1]``) but never ``max_seqlen``. So
    ``max_seqlen`` is part of the module *interface* yet dead in this implementation:
    the kernel must declare ``cu_seqlens`` as a live input and surface ``max_seqlen``
    as an ``unused_interface_inputs`` flag rather than a wired input port.

    General: keys off forward-signature params referenced only in dropped branches.
    """
    pytest.importorskip("huggingface_hub")
    spec = load_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
    graph = build_merged_model_graph(spec)
    nodes = graph["nodes"]

    _assert_export_is_acyclic(nodes)

    node_by_id = {node["id"]: node for node in nodes}
    kernel = node_by_id["visual/seq:3:blocks:attn:@attention:12"]
    details = next(
        attr["value"]
        for attr in kernel.get("attrs", [])
        if attr.get("key") == "details"
    )

    # cu_seqlens is a live declared input; max_seqlen is not.
    assert "inputs:" in details
    inputs_segment = [
        seg for seg in details.split(";") if seg.strip().startswith("inputs:")
    ][0]
    assert "cu_seqlens" in inputs_segment
    assert "max_seqlen" not in inputs_segment

    # max_seqlen is surfaced as an interface input dead in this implementation.
    assert "unused_interface_inputs: max_seqlen" in details


def test_glm53_vision_cu_seqlens_producer_visible_and_wired():
    """The ``cu_seqlens`` producer is a visible node wired into the kernel.

    ``Glm5NextVisionModel.forward`` computes packed-attention metadata via
    ``get_vision_attention_seqlens(...)`` *before* the block loop and hands the
    result to each block's attention as the ``cu_seqlens`` keyword. That producer
    must render as its own node and its output must reach the attention kernel
    across the loop-body boundary — not be dropped (which would leave the kernel
    port sourced from a raw model input) nor stripped as a dangling leaf.

    General: a boundary attention input (empty provenance chain) becomes a kernel
    ``param_inputs`` entry, giving the caller's predecessor edge a docking point;
    the kernel-input port then names the crossing boundary after the tensor.
    """
    pytest.importorskip("huggingface_hub")
    spec = load_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
    graph = build_merged_model_graph(spec)
    nodes = graph["nodes"]

    _assert_export_is_acyclic(nodes)

    node_by_id = {node["id"]: node for node in nodes}

    # The free-function producer renders as a visible node (not stripped).
    producer_id = (
        "visual/seq:0:@fn_l1840_get_vision_attention_seqlens:"
        "@fn_l1840_get_vision_attention_seqlens:0"
    )
    assert producer_id in node_by_id

    # Its output crosses the block-loop boundary named after the tensor it feeds
    # (``cu_seqlens``), not a generic ``hidden_states_2`` fallback.
    cu_mirror = node_by_id["visual/@input_mirror:cu_seqlens^cu_seqlens"]
    assert [e["sourceNodeId"] for e in cu_mirror["incomingEdges"]] == [producer_id]

    # The kernel's cu_seqlens input port is fed through that boundary, and the
    # crossing carries no back edge. (Looked up by suffix: the ``@kernel_in``
    # ordinal shifts as unrelated nodes are added/removed.)
    cu_port = next(
        n
        for n in nodes
        if n["id"].startswith("visual/@kernel_in:")
        and n["id"].endswith(":cu_seqlens")
    )
    assert [e["sourceNodeId"] for e in cu_port["incomingEdges"]] == [
        "visual/@input:cu_seqlens"
    ]


def test_glm53_vision_attention_kernel_reads_all_qkv_no_orphans():
    """The vision attention kernel reads query/key/value_states plus cu_seqlens.

    ``Glm5NextVisionAttention.forward`` computes ``query_states``, ``key_states``
    and ``value_states`` (each ending in an ``unsqueeze``) and hands all three to
    the attention interface alongside ``cu_seqlens``. The taken sdpa branch spells
    that call inside an ``else``-branch comprehension the provenance walk never
    descends into, and ``value_states`` never passes through a submodule call, so
    the kernel previously wired only ``query_states``/``cu_seqlens`` — dropping
    ``key_states``/``value_states`` and leaving their ``unsqueeze`` ops as orphan
    leaves.

    General: when the taken branch hides the attention call, the kernel's q/k/v
    predecessors are reconstructed from the first visible attention-interface call
    by resolving each positional ``Name`` arg through the topology producer map
    (``var_producer``) — which tracks inline ops and non-module tensors alike — so
    every declared kernel input is labeled and wired, with no orphaned tensor ops.
    """
    pytest.importorskip("huggingface_hub")
    spec = load_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
    graph = build_merged_model_graph(spec, shape_inferencer=ShapeInferencer(spec))
    nodes = graph["nodes"]
    node_by_id = {node["id"]: node for node in nodes}

    _assert_export_is_acyclic(nodes)

    kernel = node_by_id["visual/seq:3:blocks:attn:@attention:12"]
    kernel_sources = {e["sourceNodeId"] for e in kernel.get("incomingEdges", [])}

    # Every declared kernel input is a distinct, correctly-labeled port node, and
    # the three tensor ports source their own unsqueeze producer (not one shared).
    expected_producers = {
        "query_states": "visual/seq:3:blocks:attn:@op_l1616_c23_unsqueeze:7",
        "key_states": "visual/seq:3:blocks:attn:@op_l1617_c21_unsqueeze:9",
        "value_states": "visual/seq:3:blocks:attn:@op_l1618_c23_unsqueeze:11",
        "cu_seqlens": "visual/@input:cu_seqlens",
    }
    for label, producer_id in expected_producers.items():
        port = next(
            n
            for n in nodes
            if n["id"].startswith("visual/@kernel_in:")
            and n["id"].endswith(f":{label}")
        )
        assert port["id"] in kernel_sources, label
        assert [e["sourceNodeId"] for e in port["incomingEdges"]] == [producer_id]

    # The q/k/v unsqueeze ops are no longer orphan leaves: each feeds a kernel port.
    fed = {e["sourceNodeId"] for n in nodes for e in n.get("incomingEdges", [])}
    for producer_id in expected_producers.values():
        if producer_id.startswith("visual/seq:3:blocks:attn:"):
            assert producer_id in fed, producer_id


def test_glm53_vision_rotary_position_embeddings_wired_across_loop():
    """``position_embeddings`` reaches ``apply_rotary`` across the block loop.

    ``Glm5NextVisionModel.forward`` builds ``position_embeddings`` from
    ``rotary_pos_emb(...)`` *before* the block loop and hands it to each block's
    attention, which unpacks ``cos, sin = position_embeddings`` and feeds them to
    ``apply_rotary_pos_emb_vision``. The rotary path previously severed here: the
    positional synthetic declared no forward-param input, so its ``cos``/``sin``
    operands defaulted to ``hidden_states`` and the real producer was orphaned.

    General: a positional synthetic that reads a forward param (resolved through
    the ``cos, sin = position_embeddings`` unpack to its origin) declares that
    param as a ``param_inputs`` entry, so the module-call predecessor edge docks
    it onto the deep consumer and the producer's shape flows across the boundary
    (no text ``(B, S, H)`` re-stamp).
    """
    pytest.importorskip("huggingface_hub")
    spec = load_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
    graph = build_merged_model_graph(spec, shape_inferencer=ShapeInferencer(spec))
    nodes = graph["nodes"]
    node_by_id = {node["id"]: node for node in nodes}

    _assert_export_is_acyclic(nodes)

    # The rope helper renders like any module call (D3): its own ``@input``
    # boundary. The ``cos, sin = position_embeddings`` unpack is honoured at the
    # boundary -- the two tuple slots stay DISTINCT tiles (``@input:cos`` and
    # ``@input:sin``) named after the rotary block's return slots, rather than
    # collapsing back onto one ``position_embeddings`` tile that fans ports 0/1.
    # Each slice feeds its own unsqueeze; neither reads hidden_states.
    frame_prefix = "@positional_l1615_apply_rotary_pos_emb_vision:@/@input:"
    for slot in ("cos", "sin"):
        assert any(
            n["id"].endswith(frame_prefix + slot) for n in nodes
        ), frame_prefix + slot
    assert not any(
        n["id"].endswith(frame_prefix + "position_embeddings") for n in nodes
    ), "cos/sin must not re-merge onto a position_embeddings tile"

    frame_cos_id = next(n["id"] for n in nodes if n["id"].endswith(frame_prefix + "cos"))
    frame_sin_id = next(n["id"] for n in nodes if n["id"].endswith(frame_prefix + "sin"))

    def _sole_slot_source(node_suffix: str) -> str:
        node = next(n for n in nodes if n["id"].endswith(node_suffix))
        assert not any(
            "hidden_states" in e["sourceNodeId"] for e in node["incomingEdges"]
        ), node["id"]
        slot_edges = [
            e
            for e in node["incomingEdges"]
            if e["sourceNodeId"] in {frame_cos_id, frame_sin_id}
        ]
        assert len(slot_edges) == 1, node["id"]
        return slot_edges[0]["sourceNodeId"]

    # cos -> the c15 unsqueeze, sin -> the c42 unsqueeze (per D1 ordinal wiring).
    assert (
        _sole_slot_source(
            "@positional_l1615_apply_rotary_pos_emb_vision:@op_l1574_c15_unsqueeze:2"
        )
        == frame_cos_id
    )
    assert (
        _sole_slot_source(
            "@positional_l1615_apply_rotary_pos_emb_vision:@op_l1574_c42_unsqueeze:4"
        )
        == frame_sin_id
    )

    # Each frame slice traces back across the loop -- through its frame mirror to
    # the vision model's own ``@input:<slot>`` tile, and on to the pre-loop rotary
    # producer's distinct per-slot ``@output:<slot>`` (cos -> cos, sin -> sin; no
    # cross-alias).
    def _sole_source(node_id: str) -> str:
        edges = node_by_id[node_id]["incomingEdges"]
        assert len(edges) == 1, (node_id, edges)
        return edges[0]["sourceNodeId"]

    for slot, frame_id in (("cos", frame_cos_id), ("sin", frame_sin_id)):
        frame_mirror = _sole_source(frame_id)
        vision_input = _sole_source(frame_mirror)
        assert vision_input == f"visual/@input:{slot}", (slot, vision_input)
        output_mirror = _sole_source(vision_input)
        producer = _sole_source(output_mirror)
        assert producer == f"visual/sidefeed:2:rotary_pos_emb/@output:{slot}", (
            slot,
            producer,
        )

        # The crossing carries the vision producer's shape (Pv), not the text
        # sequence axis, at every hop of the split boundary.
        for boundary_id in (frame_id, frame_mirror, vision_input):
            shape = _output_shape(node_by_id[boundary_id])
            assert shape is not None and "Pv" in shape, (boundary_id, shape)
            assert "B, S" not in shape, (boundary_id, shape)


def test_glm53_vision_rotary_cos_and_sin_have_distinct_producers():
    """The rotary embedding keeps *both* a Cosine and a Sine branch.

    ``Glm5NextVisionRotaryEmbedding.forward`` calls the *same* child twice --
    ``cos = self.recomposition_frequencies(cos)`` then
    ``sin = self.recomposition_frequencies(sin)`` -- and returns ``(cos, sin)``.
    A submodule call's step key is the bare child attr, so the second call used to
    overwrite the first: the ``cos`` branch was dead-code-eliminated, the real
    Cosine op orphaned, and both return slots aliased onto the single (sin)
    producer.

    General: a child called more than once in a forward gets each call site
    disambiguated with an ``@l{lineno}`` suffix (``submodule_callsite_attr`` /
    ``base_submodule_attr``), so the two calls are two distinct steps. The result
    here: a live Cosine *and* Sine op, two separate ``recomposition_frequencies``
    frames, and ``@output:cos`` / ``@output:sin`` backed by *different* producers
    that trace back to the Cosine and Sine ops respectively (no cross-alias).
    """
    pytest.importorskip("huggingface_hub")
    spec = load_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
    graph = build_merged_model_graph(spec, shape_inferencer=ShapeInferencer(spec))
    nodes = graph["nodes"]
    node_by_id = {node["id"]: node for node in nodes}

    _assert_export_is_acyclic(nodes)

    rotary_ns = "Glm5NextVisionRotaryEmbedding"

    def _sole_labelled(label: str) -> dict:
        matches = [
            n
            for n in nodes
            if n.get("label") == label and rotary_ns in str(n.get("namespace", ""))
        ]
        assert len(matches) == 1, [n["id"] for n in matches]
        return matches[0]

    # Both trig branches survive as distinct live ops (cos is no longer DCE'd).
    cosine = _sole_labelled("Cosine")
    sine = _sole_labelled("Sine")

    # The repeated child produced two disambiguated call-site frames.
    import re

    recomp_frames = {
        m.group(0)
        for n in nodes
        for m in [re.search(r"recomposition_frequencies@l\d+", str(n["id"]))]
        if m
    }
    assert len(recomp_frames) == 2, sorted(recomp_frames)

    # ``@output:cos`` and ``@output:sin`` each have one producer, and the two
    # producers are different nodes in different recomposition frames.
    def _sole_producer(slot: str) -> str:
        out = node_by_id[f"visual/sidefeed:2:rotary_pos_emb/@output:{slot}"]
        edges = out["incomingEdges"]
        assert len(edges) == 1, [e["sourceNodeId"] for e in edges]
        return edges[0]["sourceNodeId"]

    cos_producer = _sole_producer("cos")
    sin_producer = _sole_producer("sin")
    assert cos_producer != sin_producer, cos_producer

    def _frame_of(node_id: str) -> str:
        m = re.search(r"recomposition_frequencies@l\d+", node_id)
        assert m, node_id
        return m.group(0)

    assert _frame_of(cos_producer) != _frame_of(sin_producer)

    # The Cosine op reaches cos's output and the Sine op reaches sin's output --
    # and crucially NOT the swapped pairing (no alias onto the single branch).
    assert _has_export_path(nodes, cosine["id"], cos_producer)
    assert _has_export_path(nodes, sine["id"], sin_producer)
    assert not _has_export_path(nodes, cosine["id"], sin_producer)
    assert not _has_export_path(nodes, sine["id"], cos_producer)


def test_glm53_vision_rotary_recomposition_slice_concat_tile_widths():
    """``recomposition_frequencies`` renders as Slice x2 -> Concat -> Tile.

    ``Glm5NextVisionRotaryEmbedding.recomposition_frequencies`` does
    ``freq_h, freq_w = freq[:, 0], freq[:, 1]`` (two single-index slices that each
    drop the indexed axis), ``freq_hw = torch.cat([freq_h, freq_w], -1)`` (CAT #1),
    then ``torch.cat([freq_hw, freq_hw], -1)`` (CAT #2, a self-concat).

    Three coupled defects used to make every recomposition op a no-op: the slice
    operands aliased the single ``freqs`` producer (CAT #1 had one input), the
    self-concat collapsed its two identical operands (CAT #2 had one input), and
    the whole chain mis-inferred to the flat patch width ``[Pv, 1176]``. General
    fixes -- materialize concat slice operands as real ``Slice`` ops, relabel a cat
    of identical operands as ``Tile``, and flow the buffer-derived trailing width
    (``inv_freq`` length 16, vision ``head_dim`` 64) into ``freqs`` -- give the
    honest chain: Slice ``[Pv, 16]`` x2 -> Concat ``[Pv, 32]`` -> Tile ``[Pv, 64]``.
    """
    pytest.importorskip("huggingface_hub")
    import re

    spec = load_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
    graph = build_merged_model_graph(spec, shape_inferencer=ShapeInferencer(spec))
    nodes = graph["nodes"]
    node_by_id = {node["id"]: node for node in nodes}

    _assert_export_is_acyclic(nodes)

    def _incoming(node: dict) -> list[dict]:
        return node.get("incomingEdges", []) or []

    frames = sorted(
        {
            match.group(0)
            for node in nodes
            for match in [re.search(r"recomposition_frequencies@l\d+", str(node["id"]))]
            if match
        }
    )
    # One recomposition frame per trig branch (cos and sin).
    assert len(frames) == 2, frames

    for frame in frames:
        frame_nodes = [node for node in nodes if frame in str(node["id"])]
        slices = [n for n in frame_nodes if n.get("label") == "Slice"]
        concats = [n for n in frame_nodes if n.get("label") == "Concat"]
        tiles = [n for n in frame_nodes if n.get("label") == "Tile"]

        # CAT #1: two distinct Slice inputs [Pv, 16] -> Concat [Pv, 32].
        assert len(slices) == 2, [n["id"] for n in slices]
        for sliced in slices:
            assert str(_attr_value(sliced, "output_shape")).startswith("[Pv, 16]"), (
                sliced["id"],
                _attr_value(sliced, "output_shape"),
            )
        assert len(concats) == 1, [n["id"] for n in concats]
        concat = concats[0]
        concat_sources = {edge["sourceNodeId"] for edge in _incoming(concat)}
        assert len(_incoming(concat)) == 2, sorted(concat_sources)
        assert concat_sources == {n["id"] for n in slices}, concat_sources
        assert str(_attr_value(concat, "output_shape")).startswith("[Pv, 32]"), (
            _attr_value(concat, "output_shape")
        )

        # CAT #2: the self-concat becomes a single-input Tile [Pv, 32] -> [Pv, 64].
        assert len(tiles) == 1, [n["id"] for n in tiles]
        tile = tiles[0]
        assert len(_incoming(tile)) == 1, [e["sourceNodeId"] for e in _incoming(tile)]
        assert _incoming(tile)[0]["sourceNodeId"] == concat["id"]
        assert str(_attr_value(tile, "output_shape")).startswith("[Pv, 64]"), (
            _attr_value(tile, "output_shape")
        )

    # The rotary outputs carry the vision head_dim [Pv, 64], not the patch width.
    for slot in ("cos", "sin"):
        out = node_by_id[f"visual/sidefeed:2:rotary_pos_emb/@output:{slot}"]
        assert str(_attr_value(out, "output_shape")).startswith("[Pv, 64]"), (
            slot,
            _attr_value(out, "output_shape"),
        )

    # No recomposition Concat is a degenerate single-input / identity cat.
    for node in nodes:
        if node.get("label") != "Concat":
            continue
        if not re.search(r"recomposition_frequencies@l\d+", str(node["id"])):
            continue
        assert len(_incoming(node)) >= 2, node["id"]


def test_glm53_no_single_input_concat_survives_anywhere():
    """The owner invariant holds graph-wide: every ``Concat`` has >1 input.

    ``torch.cat([x], dim=d)`` is ``x`` -- a Concat left with a single incoming edge
    concatenates nothing and computes nothing. The vision attention sdpa fallback's
    per-chunk reassembly cat
    (``torch.cat([interface(q,k,v) for q,k,v in zip(*splits)], dim=1)``) collapses
    to one representative kernel output, so it reads exactly one edge and restores
    the same ``[Pv, 1024]`` it received; the block-diagonal windowing it re-stitches
    is already carried by the kernel's ``cu_seqlens``. ``_elide_noop_single_input_concat``
    folds every such no-op onto its producer, so none survives.
    """
    pytest.importorskip("huggingface_hub")
    spec = load_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
    graph = build_merged_model_graph(spec, shape_inferencer=ShapeInferencer(spec))
    nodes = graph["nodes"]
    node_by_id = {node["id"]: node for node in nodes}

    _assert_export_is_acyclic(nodes)

    single_input_concats = [
        node["id"]
        for node in nodes
        if node.get("label") == "Concat"
        and len(node.get("incomingEdges", []) or []) == 1
    ]
    assert single_input_concats == [], single_input_concats

    # The known vision fallback reassembly cat is gone; its consumer (the output
    # reshape) now reads the attention kernel directly.
    assert "visual/seq:3:blocks:attn:@op_l1663_c26_concat:13" not in node_by_id
    reshape = node_by_id["visual/seq:3:blocks:attn:@op_l1665_c22_reshape:14"]
    reshape_sources = [e["sourceNodeId"] for e in reshape.get("incomingEdges", [])]
    assert reshape_sources == ["visual/seq:3:blocks:attn:@attention:12"]


def test_glm53_vision_rotary_output_edges_reference_real_producer_ports():
    """``@output:sin`` must read an existing port of its producer, not the ordinal.

    ``cos, sin = self.recomposition_frequencies(...)`` traces each slot to a
    *distinct* single-output op. The ordinal-1 (``sin``) slot's producer happens
    to coincide with the frame tail, so the fan-out used to tag its edge with the
    tuple ordinal ``"1"`` -- but that op has only output port ``"0"``. The
    dangling ``sourceNodeOutputId`` left the viewer unable to resolve a shape,
    rendering ``sin`` as ``?``. Both slot edges must reference a port that exists
    on the producer's ``outputsMetadata``.
    """
    pytest.importorskip("huggingface_hub")
    spec = load_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
    graph = build_merged_model_graph(spec, shape_inferencer=ShapeInferencer(spec))
    nodes = graph["nodes"]
    node_by_id = {node["id"]: node for node in nodes}

    for slot in ("cos", "sin"):
        out = node_by_id[f"visual/sidefeed:2:rotary_pos_emb/@output:{slot}"]
        edges = out["incomingEdges"]
        assert len(edges) == 1, [e["sourceNodeId"] for e in edges]
        edge = edges[0]
        producer = node_by_id[edge["sourceNodeId"]]
        producer_ports = {
            str(port.get("id")) for port in producer.get("outputsMetadata", [])
        }
        assert str(edge.get("sourceNodeOutputId", "0")) in producer_ports, (
            slot,
            edge.get("sourceNodeOutputId"),
            sorted(producer_ports),
        )


def test_glm53_vision_apply_rotary_tuple_returns_dock_per_ordinal():
    """``q_embed, k_embed = apply_rotary_pos_emb_vision(...)`` exits per slot.

    The rope helper now renders like any module call (D3): the tuple return exits
    through the frame's own per-slot ``@output`` boundary. ``q_embed`` is the
    ordinal-0 producer (an internal cast, l1577) and ``k_embed`` the ordinal-1
    producer (l1578); each backs its own frame output tile. The two consumers sit
    next in source order -- ``q_embed``'s transpose (l1616) then ``k_embed``'s
    (l1617) -- and each reads only its own slot's output (no cross-slot edge,
    ``q_embed`` not orphaned).
    """
    pytest.importorskip("huggingface_hub")
    spec = load_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
    graph = build_merged_model_graph(spec, shape_inferencer=ShapeInferencer(spec))
    nodes = graph["nodes"]
    node_by_id = {node["id"]: node for node in nodes}

    _assert_export_is_acyclic(nodes)

    def _one(suffix: str) -> dict:
        return next(n for n in nodes if n["id"].endswith(suffix))

    q_embed = _one(":@op_l1577_c14_cast:14")   # rotary tuple slot 0
    k_embed = _one(":@op_l1578_c14_cast:15")   # rotary tuple slot 1

    # Each tuple slot exits through its own frame @output tile, backed by that
    # slot's internal producer.
    q_out = _one("apply_rotary_pos_emb_vision:@/@output:result_1")
    k_out = _one("apply_rotary_pos_emb_vision:@/@output:result_2")
    assert [e["sourceNodeId"] for e in q_out["incomingEdges"]] == [q_embed["id"]]
    assert [e["sourceNodeId"] for e in k_out["incomingEdges"]] == [k_embed["id"]]

    # Downstream, each transpose reads only its own slot's output -- through that
    # slot's own output mirror, with no stale cross edge between the q/k paths.
    # (The frame @output tile is re-exposed to the parent via an ``^`` mirror.)
    q_out_mirror = q_out["id"] + "^result_1"
    k_out_mirror = k_out["id"] + "^result_2"
    assert q_out_mirror in node_by_id and k_out_mirror in node_by_id
    q_transpose = _one(":@op_l1616_c23_transpose:6")
    k_transpose = _one(":@op_l1617_c21_transpose:8")
    assert [e["sourceNodeId"] for e in q_transpose["incomingEdges"]] == [q_out_mirror]
    assert [e["sourceNodeId"] for e in k_transpose["incomingEdges"]] == [k_out_mirror]

    # ``q_embed`` (ordinal-0 slot) is consumed, not orphaned.
    all_sources = {
        e["sourceNodeId"] for n in nodes for e in n.get("incomingEdges", [])
    }
    assert q_embed["id"] in all_sources


def test_glm53_vision_rotary_frame_has_module_like_boundaries():
    """The rope-helper frame gets real @input/@output boundaries like a module (D3).

    A traced free-function call used to skip boundary injection: its ops docked
    straight onto external producers, so the frame had no @input/@output tiles and
    did not read like a module. The owner rule is that a free-function call renders
    exactly like any other module call -- so the frame now carries one @input tile
    per forward argument it reads (``q``, ``k``, and the two ``position_embeddings``
    tuple slots ``cos``/``sin``, which stay distinct tiles rather than collapsing
    onto one) and one @output tile per tuple return slot. The boundary set here is
    what makes the frame a first-class group.
    """
    pytest.importorskip("huggingface_hub")
    spec = load_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
    graph = build_merged_model_graph(spec, shape_inferencer=ShapeInferencer(spec))
    nodes = graph["nodes"]

    _assert_export_is_acyclic(nodes)

    frame_ns = "visual/Glm5NextVisionBlock/Glm5NextVisionAttention/apply_rotary_pos_emb_vision"
    frame_nodes = [n for n in nodes if n.get("namespace", "") == frame_ns]
    assert frame_nodes, sorted(
        {n.get("namespace", "") for n in nodes if "apply_rotary" in n.get("namespace", "")}
    )

    def _boundary_labels(kind: str) -> set[str]:
        return {
            n.get("label")
            for n in frame_nodes
            if f"/{kind}:" in n["id"] and "_mirror:" not in n["id"]
        }

    # One @input tile per forward argument the helper reads; the position_embeddings
    # tuple contributes its two distinct slots (cos, sin), not a merged tile.
    assert _boundary_labels("@input") == {"q", "k", "cos", "sin"}
    # One @output tile per tuple return slot (q_embed, k_embed) -- a group cannot
    # expose an output without an entry boundary, so the frame has both.
    assert len(_boundary_labels("@output")) == 2


def test_glm53_vision_attention_qk_norm_read_distinct_unbind_slices():
    """q_norm/k_norm boundaries are named after the qkv slice each one reads.

    ``Glm5NextVisionAttention`` carves a fused qkv projection into
    ``query_states, key_states, value_states`` via
    ``qkv(h).reshape(...).permute(...).unbind(0)`` and runs q_norm over the query
    slice, k_norm over the key slice. Both norms otherwise showed a generic
    ``hidden_states`` @input reading as if they consumed the same tensor.

    The trailing ``unbind`` is a bare view-split hop; it is folded onto its
    layout-only producer (the ``permute``), which now exposes the three slices as
    named output ports. Each norm boundary is named after the slice it reads
    (``query_states`` on producer port 0, ``key_states`` on port 1), so the two
    siblings are visibly distinct, and no redundant unbind tile survives.

    The per-slice shapes match the whole -- each slice keeps every non-split axis
    (``[Pv, 1024]``), it does not shed ``Pv`` down to ``[1024]``.
    """
    pytest.importorskip("huggingface_hub")
    spec = load_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
    graph = build_merged_model_graph(spec, shape_inferencer=ShapeInferencer(spec))
    nodes = graph["nodes"]

    _assert_export_is_acyclic(nodes)

    # The bare unbind hop is gone: its slices moved onto the permute producer.
    assert not any(
        n["id"].endswith("attn:@op_l1608_c12_unbind:2") for n in nodes
    ), "expected the qkv unbind to be folded onto its producer"

    producer = next(
        n
        for n in nodes
        if n["id"].endswith("attn:@op_l1608_c12_permute:1")
    )
    # The producer carries the three fused-qkv slices as named output ports.
    port_labels = {
        str(item["id"]): next(
            (a["value"] for a in item["attrs"] if a["key"] == "port_label"), None
        )
        for item in producer["outputsMetadata"]
    }
    assert port_labels == {
        "0": "query_states",
        "1": "key_states",
        "2": "value_states",
    }, port_labels
    port_shapes = {
        str(item["id"]): next(
            a["value"] for a in item["attrs"] if a["key"] == "shape"
        )
        for item in producer["outputsMetadata"]
    }
    assert all(shape.startswith("[Pv, 1024]") for shape in port_shapes.values()), (
        port_shapes
    )

    def _is_input(node) -> bool:
        return any(
            a.get("key") == "synthetic" and a.get("value") == "@input"
            for a in node.get("attrs", [])
        )

    def _norm_input(norm: str, slot: str, port: str) -> None:
        tile = next(
            n
            for n in nodes
            if n.get("namespace", "").endswith(f"/{norm}") and _is_input(n)
        )
        assert tile.get("label") == slot, (norm, tile.get("label"))
        # Boundary is fed by the matching producer slice port.
        sources = {
            (e["sourceNodeId"], e.get("sourceNodeOutputId", "0"))
            for e in tile["incomingEdges"]
        }
        assert sources == {(producer["id"], port)}, (norm, sources)
        shape = _output_shape(tile)
        assert shape is not None and shape.startswith("[Pv, 1024]"), (norm, shape)

    _norm_input("q_norm", "query_states", "0")
    _norm_input("k_norm", "key_states", "1")


def test_glm53_vision_rotary_frame_named_after_source_function():
    """The rope-helper frame reads like any module call, not a synthetic attr.

    A traced free-function call (``apply_rotary_pos_emb_vision(q, k, cos, sin)``)
    expands into a frame carrying the raw synthetic call attr
    (``@positional_l1615_apply_rotary_pos_emb_vision``) as both class and attr name.
    The frame's *namespace segment* must be the clean source function name, exactly
    as a module class would render; the raw attr still survives inside each op's id
    so the rename is id-stable.
    """
    pytest.importorskip("huggingface_hub")
    spec = load_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
    graph = build_merged_model_graph(spec, shape_inferencer=ShapeInferencer(spec))
    nodes = graph["nodes"]

    # The clean function name appears as a namespace segment...
    segments = {
        part
        for node in nodes
        for part in str(node.get("namespace", "")).split("/")
    }
    assert "apply_rotary_pos_emb_vision" in segments
    # ...and the ugly synthetic-attr segment never does.
    assert not any(
        seg.startswith("_positional_l") or seg.startswith("_fn_l")
        for seg in segments
    ), sorted(s for s in segments if "positional" in s or s.startswith("_fn_l"))

    # The raw synthetic attr is still embedded in the frame's op ids (id-stable).
    assert any(
        "@positional_l1615_apply_rotary_pos_emb_vision" in str(node.get("id", ""))
        for node in nodes
    )


def test_glm53_vision_index_helpers_labelled_cpu_ops():
    """Host-side index helpers carry a ``device: cpu`` label; tensor ops do not.

    ``get_vision_position_ids`` materialises grid metadata into Python
    (``grid_thw.tolist()`` + a loop) and ``get_vision_attention_seqlens`` reaches a
    ``.item()`` through ``get_max_seqlen`` -- both run on the host. The label is
    derived generally by AST-introspecting the transformers callables (following
    cross-file imports and callees), not from a hardcoded name list, so the rope
    helper ``apply_rotary_pos_emb_vision`` -- which has no host-materialisation idiom
    -- stays unlabelled.
    """
    pytest.importorskip("huggingface_hub")
    spec = load_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
    graph = build_merged_model_graph(spec, shape_inferencer=ShapeInferencer(spec))
    nodes = graph["nodes"]

    def _attr_name(node: dict) -> str:
        return str(_attr_value(node, "attr_name") or "")

    def _find(fragment: str) -> dict:
        matches = [n for n in nodes if fragment in _attr_name(n)]
        assert len(matches) == 1, [n["id"] for n in matches]
        return matches[0]

    position_ids = _find("get_vision_position_ids")
    attention_seqlens = _find("get_vision_attention_seqlens")
    assert _attr_value(position_ids, "device") == "cpu"
    assert _attr_value(attention_seqlens, "device") == "cpu"

    # The label is targeted, not blanket: only the two genuine host helpers carry
    # it. The pure-tensor rope helper (``apply_rotary_pos_emb_vision``), which has no
    # host-materialisation idiom, is absent from this set -- proving it is not
    # mislabelled.
    cpu_nodes = [n for n in nodes if _attr_value(n, "device") == "cpu"]
    assert {_attr_name(n) for n in cpu_nodes} == {
        "@fn_l1839_get_vision_position_ids",
        "@fn_l1840_get_vision_attention_seqlens",
    }
    assert not any(
        "apply_rotary_pos_emb_vision" in str(n.get("id", "")) for n in cpu_nodes
    )

    # Host ops render pale purple (not the gray GPU-op fill) so they read as
    # distinct in the viewer.
    for node in cpu_nodes:
        assert node.get("style", {}).get("backgroundColor") == "#e8daef", (
            _attr_name(node),
            node.get("style"),
        )


def test_glm53_router_outputs_no_redundant_mirror_passthrough():
    """The router's ``topk_weights`` reaches the experts without a duplicate tile.

    ``topk_weights`` is produced in the router and consumed inside the sibling
    experts loop, both under the ``Glm5NextTextMoE`` group. The boundary machinery
    otherwise leaves two identically named mirror tiles adjacent in the MoE scope --
    an ``@output_mirror`` feeding an input-styled ``@input_mirror`` -- which reads as
    a dangling ``topk_weights`` input node that goes into no block. The pass-through
    collapse drops the redundant ``@input_mirror`` and repoints the loop input onto
    the ``@output_mirror`` directly. Dataflow is unchanged: the experts still gather
    ``topk_weights`` and one-hot ``topk_indices``.
    """
    pytest.importorskip("huggingface_hub")
    spec = load_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
    graph = build_merged_model_graph(spec, shape_inferencer=ShapeInferencer(spec))
    nodes = graph["nodes"]
    node_by_id = {node["id"]: node for node in nodes}

    _assert_export_is_acyclic(nodes)

    def _synth(node: dict) -> str | None:
        return _attr_value(node, "synthetic")

    # No single-source @input_mirror is fed by a same-namespace @output_mirror: the
    # redundant 1:1 mirror-to-mirror handoff is collapsed everywhere it occurred. A
    # multi-source tuple fan-in (the vision ``position_embeddings`` gathering cos+sin
    # output mirrors) is a distinct, meaningful structure and stays.
    mirror_by_id = {n["id"]: n for n in nodes}
    for node in nodes:
        if _synth(node) != "@input_mirror":
            continue
        sources = {str(e.get("sourceNodeId")) for e in node.get("incomingEdges", [])}
        if len(sources) != 1:
            continue
        producer = mirror_by_id.get(next(iter(sources)))
        if producer is None:
            continue
        assert not (
            _synth(producer) == "@output_mirror"
            and producer.get("namespace") == node.get("namespace")
        ), node["id"]

    # The experts loop's topk_weights input now reads straight from the router's
    # output mirror (no interposed @input_mirror), and the gather still consumes it.
    loop_input = node_by_id[
        "decoder/11x_Glm5NextTextAttention_Glm5NextTextMoE/mlp/@input:topk_weights"
    ]
    sources = [e["sourceNodeId"] for e in loop_input["incomingEdges"]]
    assert sources == [
        "decoder/11x_Glm5NextTextAttention_Glm5NextTextMoE/mlp/"
        "sideproducer:1:gate:@op_l1/@output:topk_weights^topk_weights"
    ], sources
    assert _synth(node_by_id[sources[0]]) == "@output_mirror"

    gather = node_by_id[
        "decoder/11x_Glm5NextTextAttention_Glm5NextTextMoE/mlp/"
        "sidefeed:1:experts:@op_l134_c70_gather:12"
    ]
    assert loop_input["id"] in {e["sourceNodeId"] for e in gather["incomingEdges"]}
    # topk_indices wiring into the experts one-hot is untouched.
    one_hot = node_by_id[
        "decoder/11x_Glm5NextTextAttention_Glm5NextTextMoE/mlp/"
        "sidefeed:1:experts:@op_l126_c19_one_hot:1"
    ]
    assert any(
        "topk_indices" in e["sourceNodeId"] for e in one_hot["incomingEdges"]
    )


def _attr_value(node: dict, key: str) -> str | None:
    for attr in node.get("attrs", []):
        if attr.get("key") == key:
            return attr.get("value")
    return None


def _port_metadata(node: dict, port: str) -> dict | None:
    for metadata in node.get("outputsMetadata", []):
        if str(metadata.get("id")) == str(port):
            return metadata
    return None


def _port_attr(metadata: dict, key: str) -> str | None:
    for attr in metadata.get("attrs", []):
        if attr.get("key") == key:
            return attr.get("value")
    return None


def test_glm53_vision_attention_qkv_unbind_fans_out_three_ports():
    """The 3-way ``q, k, v = qkv(h)...unbind(0)`` fans out into three real ports.

    ``Glm5NextVisionAttention`` unpacks ``query_states, key_states, value_states``
    from ``qkv(h).reshape(...).permute(...).unbind(0)`` and then feeds
    ``q_norm(query_states)`` and ``k_norm(key_states)`` — two submodule calls that
    each read a *distinct* slot. The trailing ``unbind`` is a bare view-split hop,
    so it is folded onto its layout-only producer (the ``permute``): that producer
    now exposes one named output port per slot (no unbind tile, no synthetic
    per-slice tiles). Each consumer reads its own ordinal via ``sourceNodeOutputId``
    and each port carries its own slice shape. ``q_norm`` reads ordinal 0, ``k_norm``
    ordinal 1, ``value_states`` ordinal 2, with no direct ``q_norm``→``k_norm`` edge.
    """
    pytest.importorskip("huggingface_hub")
    spec = load_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
    # Per-slice port shapes are only stamped when shape inference runs.
    graph = build_merged_model_graph(spec, shape_inferencer=ShapeInferencer(spec))
    nodes = graph["nodes"]

    _assert_export_is_acyclic(nodes)

    # The bare unbind hop is gone, and there are no synthetic per-slice tiles.
    assert not any("@split_out:" in node["id"] for node in nodes)
    assert not any(node["id"].endswith("@op_l1608_c12_unbind:2") for node in nodes)

    # The slices live on the layout-only producer the unbind was folded into.
    producer = next(
        node
        for node in nodes
        if "VisionAttention" in node.get("namespace", "")
        and (_attr_value(node, "output_names") or "").split(",")
        == ["query_states", "key_states", "value_states"]
    )
    assert producer["id"].endswith("@op_l1608_c12_permute:1"), producer["id"]

    # Three ordinal-keyed output ports, each labeled with its slot and carrying a
    # per-slice shape (the head-dim slot axis dropped by the unbind).
    for ordinal, slot in enumerate(
        ("query_states", "key_states", "value_states")
    ):
        port = _port_metadata(producer, str(ordinal))
        assert port is not None, ordinal
        assert _port_attr(port, "port_label") == slot
        assert _port_attr(port, "shape"), slot

    # Each norm's input boundary reads the producer and selects its own ordinal —
    # k_norm no longer docks to query_states (ordinal 0).
    def _norm_input(kind: str) -> dict:
        return next(
            node
            for node in nodes
            if node.get("namespace", "").endswith(f"/{kind}")
            and "VisionAttention" in node.get("namespace", "")
            and node["id"].endswith("/@input")
        )

    q_norm_input = _norm_input("q_norm")
    k_norm_input = _norm_input("k_norm")
    q_edges = [
        e for e in q_norm_input["incomingEdges"] if e["sourceNodeId"] == producer["id"]
    ]
    k_edges = [
        e for e in k_norm_input["incomingEdges"] if e["sourceNodeId"] == producer["id"]
    ]
    assert [e["sourceNodeOutputId"] for e in q_edges] == ["0"]
    assert [e["sourceNodeOutputId"] for e in k_edges] == ["1"]

    # value_states (ordinal 2) is consumed directly by its transpose.
    value_consumers = [
        node
        for node in nodes
        if any(
            edge["sourceNodeId"] == producer["id"]
            and edge.get("sourceNodeOutputId") == "2"
            for edge in node.get("incomingEdges", [])
        )
    ]
    assert value_consumers and all(
        node.get("label") == "Transpose" for node in value_consumers
    )

    # No edge crosses between the two norms in either direction.
    q_norm_ids = {
        node["id"] for node in nodes if node.get("namespace", "").endswith("/q_norm")
    }
    k_norm_ids = {
        node["id"] for node in nodes if node.get("namespace", "").endswith("/k_norm")
    }
    for node in nodes:
        sources = {e["sourceNodeId"] for e in node.get("incomingEdges", [])}
        if node["id"] in k_norm_ids:
            assert not (sources & q_norm_ids), node["id"]
        if node["id"] in q_norm_ids:
            assert not (sources & k_norm_ids), node["id"]


def test_glm53_view_split_folds_onto_layout_producer_not_compute():
    """A trailing view-split collapses onto a layout-only producer, never a compute one.

    A ``reshape/transpose/permute(...).unbind|split(...)`` chain ends in a pure
    slice op that computes nothing. When its sole producer is *itself* a
    layout-only view op that feeds no one else, the split is a redundant hop and
    is folded away — its named slice ports move onto that producer (general rule,
    not a vision special case). The text ``self_attn`` ``query, key, value`` split
    (fed by a ``Transpose``) folds exactly like the vision qkv unbind.

    A split whose producer genuinely transforms values must NOT fold: the
    ``pre_w/post_w/comb_w`` activation split reads ``F.linear(flat, self.fn)``, so
    it stays its own tile with the ``Linear`` producer untouched.
    """
    pytest.importorskip("huggingface_hub")
    spec = load_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
    graph = build_merged_model_graph(spec, shape_inferencer=ShapeInferencer(spec))
    nodes = graph["nodes"]
    node_by_id = {node["id"]: node for node in nodes}

    _assert_export_is_acyclic(nodes)

    # --- Folded: the text self_attn query/key/value split (producer = Transpose).
    qkv_split_ids = [
        n["id"]
        for n in nodes
        if n["id"].endswith("self_attn/seq:11:@op_l688_c28_split:@op_l688_c28_split:0")
    ]
    assert not qkv_split_ids, ("split should have folded", qkv_split_ids)
    transpose = next(
        n
        for n in nodes
        if n["id"].endswith("self_attn/seq:10:@op_l689_c12_transpose:@op_l689_c12_transpose:0")
    )
    labels = [
        _port_attr(port, "port_label") for port in transpose["outputsMetadata"]
    ]
    assert labels == ["query", "key", "value"], labels

    # --- Preserved: the F.linear-fed pre_w/post_w/comb_w split keeps its tile,
    # and its Linear producer is left single-port.
    fn_split = next(
        n
        for n in nodes
        if "comb_w" in (_attr_value(n, "output_names") or "")
    )
    (producer_edge,) = fn_split["incomingEdges"]
    producer = node_by_id[producer_edge["sourceNodeId"]]
    assert producer.get("label") == "Linear", producer.get("label")
    assert len(producer["outputsMetadata"]) == 1, producer["outputsMetadata"]


def test_glm53_vision_mlp_gate_and_up_are_parallel():
    """``gate_proj`` and ``up_proj`` both read the block input, not each other.

    ``Glm5NextVisionMLP.forward`` computes ``gate = gate_proj(hidden_state)`` and
    ``up = up_proj(hidden_state)`` — two leaf linears in parallel off the same
    block input (clamp-based swiglu). The pipeline chain fed each leaf from the
    previous sibling by default, rendering ``up_proj <- gate_proj`` (two linears
    in a row). Consulting the AST-recorded predecessor (C3) must feed both from
    the block input with no edge between them.
    """
    pytest.importorskip("huggingface_hub")
    spec = load_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
    graph = build_merged_model_graph(spec)
    nodes = graph["nodes"]

    _assert_export_is_acyclic(nodes)

    mlp_nodes = [n for n in nodes if "VisionMLP" in n.get("namespace", "")]
    assert mlp_nodes

    gate = next(n for n in mlp_nodes if n["id"].endswith(":gate_proj:0"))
    up = next(n for n in mlp_nodes if n["id"].endswith(":up_proj:1"))
    block_input = next(n for n in mlp_nodes if n["id"].endswith("/@input"))

    gate_sources = {e["sourceNodeId"] for e in gate.get("incomingEdges", [])}
    up_sources = {e["sourceNodeId"] for e in up.get("incomingEdges", [])}

    # Both source the block input.
    assert gate_sources == {block_input["id"]}
    assert up_sources == {block_input["id"]}
    # No edge between the two sibling linears in either direction.
    assert gate["id"] not in up_sources
    assert up["id"] not in gate_sources


def test_glm53_hyperconnection_weight_only_ops_are_hidden():
    """The mHC mapping's weight-only unpacks are not drawn.

    ``pre_b, post_b, comb_b = self.base.split(...)`` and
    ``pre_scale, post_scale, comb_scale = self.scale.unbind(0)`` unpack a raw
    ``nn.Parameter`` with no activation flowing through, so — like any learned
    weight — the op and any ``@tensor:external`` operand for ``base``/``scale``
    must not appear. The ``pre_w/post_w/comb_w`` split stays: it unpacks
    ``F.linear(flat, self.fn)``, which transforms the real activation ``flat``,
    and its consumers keep that real operand after the weight operands vanish.
    """
    pytest.importorskip("huggingface_hub")
    spec = load_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
    graph = build_merged_model_graph(spec, shape_inferencer=ShapeInferencer(spec))
    nodes = graph["nodes"]
    _assert_export_is_acyclic(nodes)

    attn_hc = [n for n in nodes if "attn_hc" in n["id"]]
    assert attn_hc, "expected the attn_hc mHC mapping nodes"

    # The learned base/scale weights and their weight-only unpacks are gone.
    externals = [n for n in attn_hc if ":external:" in n["id"]]
    assert not externals, [n["id"] for n in externals]
    base_split = [n for n in attn_hc if n["id"].endswith(":@op_l281_c32_split:0")]
    scale_unbind = [n for n in attn_hc if n["id"].endswith(":@op_l282_c44_unbind:0")]
    assert not base_split, [n["id"] for n in base_split]
    assert not scale_unbind, [n["id"] for n in scale_unbind]

    # The activation-derived split (F.linear(flat, self.fn)) survives with its
    # three named output ports, and a real edge still feeds its consumers.
    fn_split = [
        n
        for n in attn_hc
        if "comb_w" in (_attr_value(n, "output_names") or "")
    ]
    assert fn_split, "expected the pre_w/post_w/comb_w activation split to survive"
    split_id = fn_split[0]["id"]
    consumers = [
        n
        for n in attn_hc
        if any(e["sourceNodeId"] == split_id for e in n.get("incomingEdges", []))
    ]
    assert consumers, "the surviving split must still feed downstream ops"
