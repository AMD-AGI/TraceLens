"""Tests for symbolic shape inference and operator export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from visualizer.ast_analyze import SYNTHETIC_ATTENTION, analyze_source
from visualizer.basic_ops import BasicOpFilter
from visualizer.block_tree import BlockNode, build_block_node
from visualizer.extract import ArchitectureSpec, load_architecture
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
    section_titles = [section["title"] for section in payload["sections"]]
    assert "Token Embedding" in section_titles
    assert "Linear" in section_titles
    assert "CustomRotaryEmbedding" in section_titles
    assert section_titles.count("RMSNorm") == 1
    assert any(op["name"] == "q_proj" for op in all_ops)
    assert any(op["name"] == "input" and op["operation"] == "input" for op in all_ops)
    assert any(op["name"] == "router" for op in all_ops)
    output_ops = [op for op in all_ops if op["name"] == "output"]
    assert len(output_ops) == 1
    assert output_ops[0]["inputs"] == ["lm_head"]
    json.dumps(payload)


def test_subgraph_warrants_export_filters_opaque_single_ops():
    from visualizer.block_tree import (
        BlockNode,
        forward_operation_count,
        subgraph_expands_on_export,
        subgraph_warrants_export,
    )

    embed = BlockNode(
        attr_name="embed_tokens",
        class_name="Embedding",
        role="embedding",
        label="Embedding",
        is_basic=True,
    )
    rope = BlockNode(
        attr_name="rotary_emb",
        class_name="RotaryEmbedding",
        role="positional",
        label="RoPE",
        details=["positional encoding (RoPE)"],
    )
    attn = BlockNode(
        attr_name="attn",
        class_name="Attn",
        role="attention",
        label="Attn",
        children=[
            BlockNode(attr_name="q_proj", class_name="Linear", role="other", label="Linear", is_basic=True),
            BlockNode(attr_name="k_proj", class_name="Linear", role="other", label="Linear", is_basic=True),
        ],
    )
    inline_mlp = BlockNode(
        attr_name="mlp",
        class_name="MLP",
        role="ffn",
        label="MLP",
        children=[
            BlockNode(attr_name="down_proj", class_name="Linear", role="other", label="Linear", is_basic=True),
        ],
    )

    assert forward_operation_count(embed) == 1
    assert forward_operation_count(rope) == 1
    assert forward_operation_count(attn) == 2
    assert forward_operation_count(inline_mlp) == 1
    assert not subgraph_expands_on_export(rope)
    assert subgraph_expands_on_export(inline_mlp)
    assert not subgraph_warrants_export(embed)
    assert not subgraph_warrants_export(rope)
    assert subgraph_warrants_export(attn)
    assert subgraph_warrants_export(inline_mlp)


def test_build_operator_export_deduplicates_same_shape_subgraphs():
    from visualizer.shape_inference import ShapeInferencer, subgraph_boundary_signature

    spec = load_architecture(
        FIXTURES / "custom_model",
        name="Custom MLA MoE",
        detailed=True,
        basic_ops=BasicOpFilter.for_detailed(),
    )
    payload = build_operator_export(spec)
    section_titles = [section["title"] for section in payload["sections"]]
    assert section_titles.count("RMSNorm") == 1
    assert "CustomRotaryEmbedding" in section_titles
    assert "Token Embedding" in section_titles
    assert "Linear" in section_titles
    assert "CustomLatent Attn" in section_titles
    assert "CustomSharedExpertMoE" in section_titles

    inferencer = ShapeInferencer(spec)
    rmsnorm_signatures: list[tuple[Any, ...]] = []
    for title, tree in spec.export_block_trees:
        if title != "RMSNorm":
            continue
        operators = inferencer.export_operators(
            build_model_graph(tree, title=title, basic_ops=BasicOpFilter.for_detailed()),
            root=tree,
        )
        signature = subgraph_boundary_signature(operators, class_name=tree.class_name)
        assert signature is not None
        rmsnorm_signatures.append(signature)
    assert len(rmsnorm_signatures) >= 2
    assert len(set(rmsnorm_signatures)) == 1


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


def test_kimi_router_export_uses_real_ops_and_topk_shapes():
    config = {
        "hidden_size": 7168,
        "num_experts": 896,
        "num_experts_per_token": 16,
        "num_expert_group": 1,
        "topk_group": 1,
        "moe_router_activation_func": "sigmoid",
        "moe_renormalize": True,
        "routed_scaling_factor": 1.0,
    }
    analysis = analyze_source(
        (FIXTURES / "kimi_moe_gate.py").read_text(),
        config=config,
    )
    basic = BasicOpFilter.for_detailed()
    gate = build_block_node(
        attr_name="gate",
        class_name="KimiMoEGate",
        registry=analysis.class_registry,
        basic_ops=basic,
    )
    spec = ArchitectureSpec(
        name="Kimi gate",
        model_type="kimi",
        hidden_size=7168,
        num_experts=896,
        num_experts_per_tok=16,
        raw_config=config,
        class_registry=analysis.class_registry,
        basic_ops=basic,
        export_block_trees=[("KimiMoEGate", gate)],
    )
    exported = build_operator_export(spec)
    operators = exported["sections"][0]["operators"]
    by_computation = {item["computation"]: item for item in operators}
    assert by_computation["Linear"]["output"]["shape"] == ["B", "T", 896]
    assert by_computation["TopK"]["output"] == {
        "shape": ["B", "T", 16],
        "dtype": "int64",
    }
    assert by_computation["Gather"]["output"]["shape"] == ["B", "T", 16]
    assert by_computation["Sum"]["output"]["shape"] == ["B", "T", 1]
    assert all(
        item["operation"] == "torch_functional"
        for item in operators
        if item["name"].startswith("@op_")
    )
