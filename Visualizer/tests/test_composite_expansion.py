"""Tests for preferring expanded computations over opaque composite tiles."""

from __future__ import annotations

from visualizer.block_tree import BlockNode, inline_composite_steps, is_simple_modeled_tile


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
    assert [step.label for step in steps] == ["l2norm_fwd", "chunk_gated_delta_rule_fwd_h"]


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
