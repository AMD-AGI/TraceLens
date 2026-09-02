###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Regression tests for GLM-5.3-Flash linear-attention graph wiring."""

from __future__ import annotations

import pytest

from model_explorer_export.merge import build_merged_model_graph
from visualizer.computation_graph import add_forward_output, build_computation_graph
from visualizer.loader import load_model_spec
from visualizer.shape_inference import ShapeInferencer


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
    outgoing: dict[str, list[str]] = {}
    for node in nodes:
        for edge in node.get("incomingEdges", []):
            outgoing.setdefault(edge["sourceNodeId"], []).append(node["id"])

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
    o_norm_power_key = _graph_key_for_op(graph, ":o_norm:@op_l351_c19_power:")
    o_norm_mean_key = _graph_key_for_op(graph, ":o_norm:@op_l351_c19_mean:")
    o_norm_rsqrt_key = _graph_key_for_op(graph, ":o_norm:@op_l352_c40_reciprocal_sqrt:")
    o_norm_gate_key = _graph_key_for_op(graph, ":o_norm:@op_l356_c40_sigmoid:")
    o_norm_mul_key = _graph_key_for_op(graph, ":o_norm:@op_l356_c24_multiply:")

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
    from visualizer.computation_graph import build_computation_graph
    from visualizer.block_tree import build_block_node

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
    carried_index = graph.loop_carried_nodes["@op_l290_c19_divide"]
    assert graph.nodes[carried_index].label == "Loop carried dependency"
    carried_inputs = {
        graph.link_port_labels[(source, carried_index)]
        for source, target in graph.links
        if target == carried_index
    }
    assert carried_inputs == {"initial", "updated"}
    assert graph.output_ports["comb"] == carried_index

    multiply_index = next(
        index
        for index, node in enumerate(graph.nodes)
        if node.block is not None and node.block.attr_name == "@op_l294_c21_multiply"
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
    from visualizer.block_tree import build_block_node
    from visualizer.computation_graph import build_computation_graph

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
    from visualizer.block_tree import build_block_node
    from visualizer.computation_graph import build_computation_graph

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
    assert set(inputs) == {"hidden_states", "gate"}

    # The gate arrives from the projection that computes it, not from the spine.
    gate_sources = [
        edge["sourceNodeId"] for edge in inputs["gate"].get("incomingEdges", [])
    ]
    assert gate_sources and all("g_b_proj" in source for source in gate_sources)
    hidden_sources = [
        edge["sourceNodeId"]
        for edge in inputs["hidden_states"].get("incomingEdges", [])
    ]
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
    from visualizer.block_tree import build_block_node, collect_function_steps
    from visualizer.computation_graph import build_computation_graph

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
    from visualizer.block_tree import build_block_node
    from visualizer.computation_graph import build_computation_graph

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

    concat_key = _graph_key(graph, ":@op_l642_c20_concat:@op_l642_c20_concat:0")
    conv_update_key = _graph_key_for_op(graph, ":@op_l659_c24_causal_conv1d_update:")
    conv_key = _graph_key_for_op(graph, ":@op_l674_c24_causal_conv1d:")
    split_key = _graph_key_for_op(graph, ":@op_l685_c28_split:")
    forget_entry_key = _graph_key(graph, ":forget_gate:f_a_proj:0")
    branch_mul_key = _graph_key_for_op(graph, ":forget_gate:@op_l329_c19_multiply:")
    branch_add_key = _graph_key_for_op(graph, ":forget_gate:@op_l323_c13_add:")
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
    assert (input_index, key_to_index[forget_entry_key]) in links
    assert key_to_index[branch_mul_key] in sources
    assert key_to_index[branch_add_key] in sources


def test_glm53_dead_code_elimination_is_idempotent_for_hyperconnection():
    pytest.importorskip("huggingface_hub")
    from visualizer.block_tree import build_block_node
    from visualizer.computation_graph import (
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
    from model_explorer_export.merge import _resolve_section_tree_for_component
    from visualizer.basic_ops import BasicOpFilter
    from visualizer.block_tree import build_block_node

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
    matmul = decoder.forward_operations["@op_l1316_c85_matmul"]
    multiply = decoder.forward_operations["@op_l1316_c24_multiply"]
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
    from model_explorer_export.merge import _resolve_section_tree_for_component
    from model_explorer_export.overview import component_has_detail_section
    from visualizer.basic_ops import BasicOpFilter

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
    assert boundary["input_shape"] == "B x S x 4 x 4096"
    assert boundary["output_shape"] == (
        "post: B x S x 4, comb: B x S x 4 x 4, " "collapsed: B x S x 4096"
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
        "/Glm5NextTextExperts/Loop_repeated/_apply_gate" in node.get("namespace", "")
        for node in helper_nodes
    )
    _assert_export_is_acyclic(graph["nodes"])


def test_glm53_decoder_boundary_keeps_hyper_stream_shape():
    pytest.importorskip("huggingface_hub")
    spec = load_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
    graph = build_merged_model_graph(spec, shape_inferencer=ShapeInferencer(spec))
    boundary = graph["groupNodeAttributes"]["45x_Glm5NextTextDecoderLayer"]

    assert boundary["input_shape"] == "B x S x 4 x 4096"
    assert boundary["output_shape"] == "B x S x 4 x 4096"

    prefix = _linear_attn_variant_prefix(spec)
    node_by_id = {node["id"]: node for node in graph["nodes"]}
    for operation_id in ("@op_l1316_c24_add", "@op_l1325_c24_add"):
        residual_add = node_by_id[f"{prefix}/{operation_id}"]
        output_shape = next(
            attr["value"]
            for attr in residual_add["attrs"]
            if attr["key"] == "output_shape"
        )
        assert output_shape == "B x S x 4 x 4096"


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
        if node["id"].endswith("/@op_l1316_c55_unsqueeze")
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
    assert model_ops[0]["incomingEdges"][0]["sourceNodeId"] == "embed_tokens"
    assert model_ops[1]["incomingEdges"][0]["sourceNodeId"] == model_ops[0]["id"]
    assert model_ops[2]["incomingEdges"][0]["sourceNodeId"] == model_ops[1]["id"]
    assert not any(node["id"] == "rotary_pos_emb" for node in graph["nodes"])
    assert [
        next(attr["value"] for attr in node["attrs"] if attr["key"] == "output_shape")
        for node in model_ops
    ] == [
        "B x S x 1 x 4096",
        "B x S x 4 x 4096",
        "B x S x 4 x 4096",
    ]


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
