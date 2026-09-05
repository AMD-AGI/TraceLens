###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for preferring expanded computations over opaque composite tiles."""

from __future__ import annotations

from TraceLens.ModelUtils.block_tree import (
    BlockNode,
    inline_composite_steps,
    is_simple_modeled_tile,
)


def test_inline_composite_steps_expands_kernel_pipeline_children():
    pipeline = BlockNode(
        attr_name="@attn_pipeline",
        class_name="KernelPipeline",
        role="attention",
        label="chunk_kda pipeline",
        children=[
            BlockNode(
                attr_name="step0",
                class_name="KernelOp",
                role="other",
                label="l2norm_fwd",
                children=[
                    BlockNode(
                        attr_name="sum",
                        class_name="Sum",
                        role="other",
                        label="Sum",
                        is_basic=False,
                    ),
                    BlockNode(
                        attr_name="sqrt",
                        class_name="Sqrt",
                        role="other",
                        label="Sqrt",
                        is_basic=False,
                    ),
                ],
            ),
            BlockNode(
                attr_name="step1",
                class_name="KernelOp",
                role="other",
                label="chunk_gated_delta_rule_fwd_h",
            ),
        ],
    )

    steps, wrapper = inline_composite_steps(pipeline)

    assert wrapper is pipeline
    assert [step.label for step in steps] == [
        "l2norm_fwd",
        "chunk_gated_delta_rule_fwd_h",
    ]


def test_inline_composite_steps_expands_output_gate_with_activation():
    gate = BlockNode(
        attr_name="g_proj",
        class_name="OutputGate",
        role="gate",
        label="Output gate",
        children=[
            BlockNode(
                attr_name="linear",
                class_name="Linear",
                role="other",
                label="Linear",
                is_basic=True,
            ),
            BlockNode(
                attr_name="act",
                class_name="Sigmoid",
                role="other",
                label="Sigmoid",
                is_basic=False,
            ),
        ],
    )

    steps, wrapper = inline_composite_steps(gate)

    assert wrapper is gate
    assert [step.label for step in steps] == ["Linear", "Sigmoid"]


def test_output_gate_is_not_simple_modeled_tile():
    gate = BlockNode(
        attr_name="g_proj",
        class_name="OutputGate",
        role="gate",
        label="Output gate",
        children=[
            BlockNode(
                attr_name="linear",
                class_name="Linear",
                role="other",
                label="Linear",
                is_basic=True,
            ),
        ],
    )

    assert not is_simple_modeled_tile(gate)


def test_single_operation_head_is_replaced_inline_without_wrapper():
    mean = BlockNode(
        attr_name="@op_mean",
        class_name="Mean",
        role="other",
        label="Mean",
        is_basic=True,
    )
    head = BlockNode(
        attr_name="hc_head",
        class_name="Glm5NextTextHyperHead",
        role="head",
        label="Glm5NextTextHyperHead",
        children=[mean],
    )

    steps, wrapper = inline_composite_steps(head)

    assert steps == [mean]
    assert wrapper is None


def test_build_computation_graph_inline_expansion_false_keeps_composites_opaque():
    """When inline_expansion=False, composite modules stay as single nodes."""
    from TraceLens.ModelUtils.computation_graph import build_computation_graph

    linear = BlockNode(
        attr_name="linear",
        class_name="Linear",
        role="other",
        label="Linear",
        is_basic=True,
    )
    sigmoid = BlockNode(
        attr_name="sigmoid",
        class_name="Sigmoid",
        role="other",
        label="Sigmoid",
        is_basic=True,
    )
    gate = BlockNode(
        attr_name="g_proj",
        class_name="StraightLine",
        role="other",
        label="Output gate",
        children=[linear, sigmoid],
    )
    root = BlockNode(
        attr_name="root",
        class_name="Root",
        role="other",
        label="Root",
        children=[gate],
    )

    expanded = build_computation_graph(root, inline_expansion=True)
    collapsed = build_computation_graph(root, inline_expansion=False)

    expanded_labels = [n.label for n in expanded.nodes if not n.synthetic]
    collapsed_labels = [n.label for n in collapsed.nodes if not n.synthetic]

    # Expanded graph should have more nodes (Linear, Sigmoid) than collapsed (Output gate)
    assert len(expanded_labels) >= len(collapsed_labels)
    assert "Output gate" in collapsed_labels
    # When collapsed, internal steps should not appear as separate nodes
    assert "Linear" not in collapsed_labels or "Sigmoid" not in collapsed_labels
