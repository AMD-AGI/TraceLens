"""Tests for symbolic shape inference and operator export."""

from __future__ import annotations

import json
from pathlib import Path

from visualizer.ast_analyze import SYNTHETIC_ATTENTION
from visualizer.basic_ops import BasicOpFilter
from visualizer.block_tree import BlockNode
from visualizer.extract import load_architecture
from visualizer.model_graph import build_model_graph
from visualizer.shape_inference import (
    ModuleDimRegistry,
    ShapeContext,
    ShapeInferencer,
    build_operator_export,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _mla_fixture_root() -> BlockNode:
    def leaf(name: str) -> BlockNode:
        return BlockNode(
            attr_name=name,
            class_name="Linear",
            role="other",
            label="Linear",
            is_basic=True,
        )

    return BlockNode(
        attr_name="attn",
        class_name="Attn",
        role="attention",
        label="Attn",
        parallel_gates=["g_proj"],
        children=[
            leaf("q_a_proj"),
            leaf("q_a_layernorm"),
            leaf("q_b_proj"),
            leaf("kv_a_proj_with_mqa"),
            leaf("kv_a_layernorm"),
            leaf("kv_b_proj"),
            BlockNode(
                attr_name=SYNTHETIC_ATTENTION,
                class_name="AttentionOp",
                role="attention",
                label="Attention",
                is_basic=True,
                details=["kernel: flash_attn"],
            ),
            leaf("g_proj"),
            leaf("o_proj"),
        ],
    )


def test_module_dim_registry_parses_linear_from_ast():
    spec = load_architecture(
        FIXTURES / "custom_model",
        detailed=True,
        basic_ops=BasicOpFilter.for_detailed(),
    )
    context = ShapeContext.from_spec(spec)
    registry = ModuleDimRegistry.from_registry(
        spec.class_registry,
        config=spec.raw_config,
        context=context,
    )
    q_spec = registry.linear.get(("CustomLatentAttention", "q_proj"))
    assert q_spec is not None
    assert q_spec.in_features == 4096
    assert q_spec.out_features == 4096
    kv_spec = registry.linear.get(("CustomLatentAttention", "kv_proj"))
    assert kv_spec is not None
    assert kv_spec.out_features == 512
    router_spec = registry.linear.get(("CustomSharedExpertMoE", "router"))
    assert router_spec is not None
    assert router_spec.out_features == 64


def test_shape_inferencer_mla_fixture_linear_shapes():
    root = _mla_fixture_root()
    spec = load_architecture(
        FIXTURES / "custom_model",
        detailed=True,
        basic_ops=BasicOpFilter.for_detailed(),
    )
    inferencer = ShapeInferencer(spec)
    graph = build_model_graph(root, title="MLA")
    operators = inferencer.export_operators(graph, root=root)

    by_name = {op.name: op for op in operators}
    assert by_name["q_a_proj"].output.shape == ("B", "T", 4096)
    assert by_name["o_proj"].output.shape == ("B", "T", 4096)
    assert by_name["Attention"].computation == "AttentionOp"
    assert by_name["Attention"].output.shape == ("B", "T", 4096)
    assert by_name["×"].computation == "elementwise_mul"
    assert "input" in by_name["q_a_proj"].inputs
    assert by_name["input"].operation == "input"
    assert by_name["input"].name == "input"
    assert by_name["input"].inputs == []


def test_model_output_operator():
    spec = load_architecture(
        FIXTURES / "custom_model",
        detailed=True,
        basic_ops=BasicOpFilter.for_detailed(),
    )
    inferencer = ShapeInferencer(spec)
    output = inferencer.model_output_operator()
    assert output is not None
    assert output.name == "output"
    assert output.operation == "output"
    assert output.computation == "output"
    assert output.output.shape == ("B", "T", 32000)


def test_build_operator_export_custom_model():
    spec = load_architecture(
        FIXTURES / "custom_model",
        name="Custom MLA MoE",
        detailed=True,
        basic_ops=BasicOpFilter.for_detailed(),
    )
    payload = build_operator_export(spec)
    assert payload["name"] == "Custom MLA MoE"
    assert payload["dtype"] == "float16"
    assert payload["dimensions"]["H"] == 4096
    assert payload["sections"]
    assert payload.get("checkpoint_source")
    assert payload.get("code_sources")

    all_ops = []
    for section in payload["sections"]:
        all_ops.extend(section["operators"])
    assert any(op["name"] == "q_proj" for op in all_ops)
    assert any(op["name"] == "input" and op["operation"] == "input" for op in all_ops)
    assert any(op["name"] == "router" for op in all_ops)
    assert any(op["name"] == "embed_tokens" for op in all_ops)
    assert any(op["name"] == "lm_head" for op in all_ops)
    output_ops = [op for op in all_ops if op["name"] == "output"]
    assert len(output_ops) == 1
    assert output_ops[0]["inputs"] == ["lm_head"]
    json.dumps(payload)


def test_infer_forward_steps_from_init():
    from visualizer.ast_analyze import effective_forward_calls, infer_forward_steps_from_init

    spec = load_architecture(
        FIXTURES / "custom_model",
        detailed=True,
        basic_ops=BasicOpFilter.for_detailed(),
    )
    attn = spec.class_registry["CustomLatentAttention"]
    assert infer_forward_steps_from_init(attn) == ["q_proj", "kv_proj"]
    assert effective_forward_calls(attn) == ["q_proj", "kv_proj"]
    decoder = spec.class_registry["CustomDecoderLayer"]
    assert effective_forward_calls(decoder) == [
        "input_layernorm",
        "self_attn",
        "post_attention_layernorm",
        "block_sparse_moe",
    ]


def test_export_block_trees_expand_init_only_modules():
    spec = load_architecture(
        FIXTURES / "custom_model",
        detailed=True,
        basic_ops=BasicOpFilter.for_detailed(),
    )
    export_titles = [title for title, _ in spec.export_block_trees]
    detailed_titles = [title for title, _ in spec.detailed_block_trees]
    assert "Token Embedding" in export_titles
    assert "Linear" in export_titles
    assert detailed_titles == ["CustomLatent Attn", "CustomSharedExpertMoE"]

    attn_tree = next(tree for title, tree in spec.export_block_trees if "CustomLatent" in title)
    assert [child.attr_name for child in attn_tree.children] == ["q_proj", "kv_proj"]
