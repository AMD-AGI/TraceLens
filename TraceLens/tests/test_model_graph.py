###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for serializable model graph IR and operation classification."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

from TraceLens.ModelUtils.ast_analyze import SYNTHETIC_ATTENTION, analyze_source
from TraceLens.ModelUtils.basic_ops import BasicOpFilter
from TraceLens.ModelUtils.block_tree import BlockNode, build_decoder_block_trees
from TraceLens.ModelUtils.computation_graph import (
    SYNTHETIC_INPUT,
    SYNTHETIC_OUTPUT,
    build_computation_graph,
)
from TraceLens.ModelUtils.extract import load_architecture
from TraceLens.ModelUtils.model_graph import (
    NodeKind,
    OperationKind,
    assert_operations_reduced,
    build_architecture_model_graphs,
    build_model_graph,
    classify_operation,
    collect_non_reduced_operations,
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


def test_classify_operation_kinds():
    linear = BlockNode(
        attr_name="q_proj",
        class_name="Linear",
        role="other",
        label="Linear",
        is_basic=True,
    )
    assert classify_operation(linear) == OperationKind.NN_MODULE

    functional = BlockNode(
        attr_name="@functional_linear",
        class_name="Linear",
        role="other",
        label="Linear",
        is_basic=True,
    )
    assert classify_operation(functional) == OperationKind.TORCH_FUNCTIONAL

    kernel = BlockNode(
        attr_name=SYNTHETIC_ATTENTION,
        class_name="AttentionOp",
        role="attention",
        label="Attention",
        details=["kernel: chunk_kda"],
    )
    assert classify_operation(kernel) == OperationKind.GPU_KERNEL

    kernel_subop = BlockNode(
        attr_name="tl_dot",
        class_name="KernelSubOp",
        role="other",
        label="tl.dot",
    )
    assert classify_operation(kernel_subop) == OperationKind.GPU_KERNEL

    torch_exp = BlockNode(
        attr_name="fused_sub_0",
        class_name="KernelSubOp",
        role="other",
        label="Exp",
    )
    assert classify_operation(torch_exp) == OperationKind.TORCH_FUNCTIONAL

    torch_cumsum = BlockNode(
        attr_name="gate_sub_1",
        class_name="KernelSubOp",
        role="other",
        label="CumSum",
    )
    assert classify_operation(torch_cumsum) == OperationKind.TORCH_FUNCTIONAL

    fused_sigmoid = BlockNode(
        attr_name="fused_beta_sigmoid",
        class_name="KernelOp",
        role="other",
        label="Fused beta sigmoid",
    )
    assert classify_operation(fused_sigmoid) == OperationKind.GPU_KERNEL

    intra_chunk = BlockNode(
        attr_name="chunk_kda_fwd_intra",
        class_name="KernelOp",
        role="other",
        label="Intra-chunk WY",
    )
    assert classify_operation(intra_chunk) == OperationKind.GPU_KERNEL

    nn_attention = BlockNode(
        attr_name="@attention",
        class_name="AttentionOp",
        role="attention",
        label="Attention",
        details=["kernel: sdpa_attention_forward"],
    )
    assert classify_operation(nn_attention) == OperationKind.TORCH_FUNCTIONAL

    assert (
        classify_operation(None, synthetic=SYNTHETIC_INPUT, label="hidden_states")
        == OperationKind.SYNTHETIC
    )
    assert (
        classify_operation(None, synthetic=SYNTHETIC_OUTPUT, label="Output")
        == OperationKind.SYNTHETIC
    )
    assert (
        classify_operation(None, synthetic=None, label="×") == OperationKind.SYNTHETIC
    )


def test_library_attention_is_a_kernel_but_torch_attention_is_not():
    """Flash-attn and friends are fused library kernels; SDPA and eager are torch."""

    def attention(kernel: str) -> BlockNode:
        return BlockNode(
            attr_name="@attention",
            class_name="AttentionOp",
            role="attention",
            label="Attention",
            details=[f"kernel: {kernel}"],
        )

    for kernel in (
        "sdpa",
        "eager",
        "sdpa_attention_forward",
        "torch.nn.attention.flex_attention",
    ):
        assert (
            classify_operation(attention(kernel)) == OperationKind.TORCH_FUNCTIONAL
        ), kernel

    for kernel in (
        "flash_attention_2",
        "flash_attn_varlen_func",
        "xformers",
        "transformer_engine",
    ):
        assert classify_operation(attention(kernel)) == OperationKind.GPU_KERNEL, kernel

    unresolved = BlockNode(
        attr_name="@attention",
        class_name="AttentionOp",
        role="attention",
        label="Attention",
    )
    assert (
        classify_operation(unresolved) == OperationKind.GPU_KERNEL
    ), "attention nobody could resolve to a torch call stays a kernel"


def test_dispatched_attention_resolves_through_the_checkpoint_config():
    """A forward that calls an attention variable gets its kernel from the config."""
    from TraceLens.ModelUtils.ast_analyze import kernel_name_from_step_details

    source = textwrap.dedent("""
        class Attention(nn.Module):
            def forward(self, hidden_states):
                query_states = self.q_proj(hidden_states)
                attention_interface = eager_attention_forward
                if self.config._attn_implementation != "eager":
                    attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
                attn_output, _ = attention_interface(self, query_states, k, v, scaling=self.scaling)
                return attn_output
        """)

    def kernel_for(config: dict[str, str]) -> str | None:
        analysis = analyze_source(source, config=config)
        details = analysis.class_registry["Attention"].forward_step_details[
            SYNTHETIC_ATTENTION
        ]
        return kernel_name_from_step_details(details)

    assert kernel_for({}) == "attention_interface", "no config leaves the variable name"
    assert (
        kernel_for({"_attn_implementation": "flash_attention_2"}) == "flash_attention_2"
    )
    assert kernel_for({"_attn_implementation": "sdpa"}) == "sdpa"


def test_build_model_graph_matches_computation_graph_topology():
    root = _mla_fixture_root()
    basic = BasicOpFilter.for_detailed()
    computation = build_computation_graph(root, basic_ops=basic)
    model_graph = build_model_graph(root, title="MLA", basic_ops=basic)

    assert model_graph.title == "MLA"
    assert len(model_graph.nodes) == len(computation.nodes)
    assert len(model_graph.edges) == len(computation.links)

    labels = {node.label for node in model_graph.nodes}
    assert "hidden_states" in labels
    assert "Multiply" in labels
    assert "Attention" in labels

    operations = {node.label: node.operation for node in model_graph.nodes}
    assert (
        operations["Attention"] == OperationKind.GPU_KERNEL
    ), "the fixture runs a flash-attn kernel"


def test_expanded_kernel_pipeline_is_composite_but_its_kernel_stays_low_level():
    pipeline = BlockNode(
        attr_name="@attn_pipeline",
        class_name="KernelPipeline",
        role="attention",
        label="sparse_attn pipeline",
        children=[
            BlockNode(
                attr_name="@kernel",
                class_name="KernelOp",
                role="other",
                label="Sparse attn kernel",
                details=["kernel: sparse_attn_kernel"],
            )
        ],
    )

    assert classify_operation(pipeline) == OperationKind.COMPOSITE
    assert classify_operation(pipeline.children[0]) == OperationKind.GPU_KERNEL


def test_contiguous_kernel_step_is_a_torch_operation():
    contiguous = BlockNode(
        attr_name="@pipeline_contiguous",
        class_name="KernelOp",
        role="other",
        label="Contiguous",
        details=["kernel: contiguous"],
    )
    assert classify_operation(contiguous) == OperationKind.TORCH_FUNCTIONAL


def test_model_graph_json_roundtrip():
    root = _mla_fixture_root()
    model_graph = build_model_graph(root, title="MLA")
    payload = json.loads(model_graph.to_json())

    assert payload["title"] == "MLA"
    assert payload["nodes"]
    assert payload["edges"]
    for node in payload["nodes"]:
        assert {"id", "kind", "label"} <= set(node)
        assert node["kind"] in {kind.value for kind in NodeKind}
    for edge in payload["edges"]:
        assert {"source", "target", "style"} <= set(edge)


def test_assert_operations_reduced_for_mla_fixture():
    root = _mla_fixture_root()
    model_graph = build_model_graph(root, title="MLA")
    assert_operations_reduced(model_graph)


def test_collect_non_reduced_operations_flags_unexpanded_module():
    unexpanded = BlockNode(
        attr_name="self_attn",
        class_name="CustomLatentAttention",
        role="attention",
        label="Latent Attention",
        is_basic=False,
    )
    graph = build_model_graph(unexpanded, title="Latent Attention")
    issues = collect_non_reduced_operations(graph)
    assert issues
    assert issues[0].operation in {OperationKind.UNKNOWN, OperationKind.COMPOSITE}


def test_build_architecture_model_graphs_custom_model():
    spec = load_architecture(
        FIXTURES / "custom_model",
        name="Custom MLA MoE",
        detailed=True,
        basic_ops=BasicOpFilter.from_cli(add=[r"(?i)^Linear$"]),
    )
    payload = build_architecture_model_graphs(spec)
    assert payload["name"] == "Custom MLA MoE"
    assert payload["sections"]

    for section in payload["sections"]:
        graph = section["graph"]
        assert graph["nodes"]
        assert graph["edges"]

    labels = {
        node["label"]
        for section in payload["sections"]
        for node in section["graph"]["nodes"]
    }
    assert "hidden_states" in labels or "input_ids" in labels


def test_decoder_block_tree_graph_has_nn_and_functional_ops():
    source = (FIXTURES / "custom_model" / "modeling_custom.py").read_text(
        encoding="utf-8"
    )
    analysis = analyze_source(source, filename="modeling_custom.py")
    basic_ops = BasicOpFilter.from_cli(add=[r"(?i)^Linear$"])
    trees = build_decoder_block_trees(
        analysis.block_components,
        analysis.class_registry,
        basic_ops,
    )
    moe_tree = next(tree for title, tree in trees if "MoE" in title)
    graph = build_model_graph(moe_tree, title="MoE")

    op_kinds = {node.operation for node in graph.nodes if node.operation is not None}
    assert OperationKind.NN_MODULE in op_kinds
    assert OperationKind.SYNTHETIC in op_kinds

    reduced = {
        node.operation
        for node in graph.nodes
        if node.operation not in {OperationKind.SYNTHETIC, None}
    }
    assert reduced <= {
        OperationKind.NN_MODULE,
        OperationKind.TORCH_FUNCTIONAL,
        OperationKind.GPU_KERNEL,
        OperationKind.COMPOSITE,
    }


def test_model_graph_inline_frames_use_node_ids():
    root = BlockNode(
        attr_name="mlp",
        class_name="MLP",
        role="ffn",
        label="MLP",
        children=[
            BlockNode(
                attr_name="gate_proj",
                class_name="Linear",
                role="other",
                label="Linear",
                is_basic=True,
            ),
            BlockNode(
                attr_name="up_proj",
                class_name="Linear",
                role="other",
                label="Linear",
                is_basic=True,
            ),
            BlockNode(
                attr_name="down_proj",
                class_name="Linear",
                role="other",
                label="Linear",
                is_basic=True,
            ),
        ],
    )
    computation = build_computation_graph(root)
    model_graph = build_model_graph(root, title="MLP")

    if computation.inline_frames:
        frame = model_graph.inline_frames[0]
        assert frame.node_ids
        node_ids = {node.id for node in model_graph.nodes}
        assert set(frame.node_ids) <= node_ids


def test_model_graph_edge_styles():
    root = _mla_fixture_root()
    model_graph = build_model_graph(root, title="MLA")

    assert all(edge.style == "solid" for edge in model_graph.edges)
    assert not any(edge.style == "side" for edge in model_graph.edges)


def test_model_graph_subgraphs_for_nested_composites():
    pipeline = BlockNode(
        attr_name="kernel",
        class_name="KernelPipeline",
        role="attention",
        label="chunk_kda pipeline",
        children=[
            BlockNode(
                attr_name="step0",
                class_name="KernelOp",
                role="other",
                label="l2norm_fwd",
                details=["kernel: chunk_kda"],
            ),
            BlockNode(
                attr_name="out",
                class_name="KernelOutput",
                role="other",
                label="output",
            ),
        ],
    )
    root = BlockNode(
        attr_name="attn",
        class_name="Attn",
        role="attention",
        label="Attn",
        children=[pipeline],
    )
    graph = build_model_graph(root, title="Attn", include_subgraphs=True)
    assert graph.subgraphs or any(
        node.operation == OperationKind.GPU_KERNEL for node in graph.nodes
    )
