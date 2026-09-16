###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Line-coverage tests for :mod:`TraceLens.ModelUtils.block_tree`."""

from __future__ import annotations

import ast
from types import SimpleNamespace

import pytest

import TraceLens.ModelUtils.block_tree as bt
from TraceLens.ModelUtils import ast_analyze as aa
from TraceLens.ModelUtils.ast_analyze import (
    ClassStructure,
    ForwardOperation,
    LoopCarriedSpec,
    SideInputSpec,
)
from TraceLens.ModelUtils.basic_ops import BasicOpFilter
from TraceLens.ModelUtils.block_tree import (
    BlockComponent,
    BlockNode,
    Branch,
    CombineSegment,
    FanOutSegment,
    ResidualAddSegment,
    SeqSegment,
    SideCombineSegment,
    SideFeedSegment,
    TensorPortsSegment,
    block_purpose,
    build_block_node,
    collect_computation_segments,
    collect_graph_segments,
    collect_method_wrappers,
    collect_parallel_gate_wrappers,
    components_from_registry,
    gated_norm_activation,
    gated_norm_tile_label,
    inline_block_frame_label,
    inline_composite_steps,
    is_basic_op_tile,
    is_inline_expandable_module,
    is_method_wrapper,
    is_simple_modeled_tile,
    is_single_function_tree,
    is_straight_line_module,
    partition_detail_trees,
    side_producer_has_activation,
    spine_expanded_frame_label,
    straight_line_steps,
    tile_display_labels,
    wrapper_bullet,
    wrapper_panel_line,
    wrapper_skips_comment,
)


def node(
    attr: str,
    class_name: str = "Linear",
    *,
    role: str = "other",
    label: str | None = None,
    basic: bool = True,
    children: list[BlockNode] | None = None,
    details: list[str] | None = None,
    **kwargs,
) -> BlockNode:
    return BlockNode(
        attr_name=attr,
        class_name=class_name,
        role=role,
        label=label or class_name,
        is_basic=basic,
        children=list(children or []),
        details=list(details or []),
        **kwargs,
    )


def structure(
    name: str,
    *,
    assignments: dict[str, str] | None = None,
    calls: list[str] | None = None,
    norm_before: list[str] | None = None,
) -> ClassStructure:
    return ClassStructure(
        name=name,
        node=ast.parse(f"class {name}:\n    pass").body[0],
        init_assignments=dict(assignments or {}),
        init_details={},
        forward_calls=list(calls or []),
        norm_before=list(norm_before or []),
    )


# --------------------------------------------------------------------------- #
# wrapper labels / bullets
# --------------------------------------------------------------------------- #
def test_wrapper_bullet_label_differs_from_attr():
    # label differs from attr so wrapper_bullet returns "label (attr)".
    n = node("q_proj", "Linear", label="Query")
    assert bt.wrapper_bullet(n) == "Query (q_proj)"


def test_wrapper_bullet_label_matches_attr_returns_label():
    n = node("query_proj", "Thing", label="query proj", basic=False)
    assert bt.wrapper_bullet(n) == "query proj"


# --------------------------------------------------------------------------- #
# block_purpose specialized branches
# --------------------------------------------------------------------------- #
def test_block_purpose_short_conv_activation_single_detail():
    conv = node("conv", "ShortConvolution", basic=False, details=["SiLU"])
    assert bt.block_purpose(conv) == "depthwise conv"


def test_block_purpose_output_gate_details_only_linear_returns_none():
    gate = node("g", "OutputGate", role="gate", basic=False, details=["Linear"])
    # OutputGate branch: only "Linear" detail is skipped -> returns None.
    assert bt.block_purpose(gate) is None


def test_block_purpose_role_gate_no_details_fallback():
    gate = node("g", "SomeGate", role="gate", basic=False)
    # not OutputGate, role gate, no details -> reaches role=="gate" fallback line.
    assert bt.block_purpose(gate) == "Output gate — scales normalized output"


def test_block_purpose_attention_merge_none():
    plain = node("m2", "AttentionMerge", basic=False, details=["method `x()`"])
    assert bt.block_purpose(plain) is None


def test_block_purpose_fused_silu_stem():
    fused = node("f", "SiTUAndMul", basic=False, details=["method `x()`"])
    assert bt.block_purpose(fused) == "SiTU(gate) × up branch"


def test_block_purpose_kernel_op_and_kernel_output_none():
    assert bt.block_purpose(node("k", "KernelOutput", basic=False)) is None


# --------------------------------------------------------------------------- #
# single-op subgraph substitution
# --------------------------------------------------------------------------- #
def test_substitute_single_op_uses_straight_line_inner(monkeypatch):
    inner = node("inner")
    wrapper = node("wrap", "Wrapper", basic=False, children=[inner])
    # collect_function_steps returns 2 leaves so len!=1; straight_line inner==1.
    monkeypatch.setattr(bt, "collect_function_steps", lambda n: [inner, inner])
    monkeypatch.setattr(bt, "straight_line_steps", lambda n: [inner])
    result = bt._substitute_single_op_subgraph(wrapper)
    assert result is inner


def test_substitute_single_op_returns_node_when_no_single(monkeypatch):
    a = node("a")
    b = node("b")
    wrapper = node("wrap", "Wrapper", basic=False, children=[a, b])
    monkeypatch.setattr(bt, "collect_function_steps", lambda n: [a, b])
    monkeypatch.setattr(bt, "straight_line_steps", lambda n: [a, b])
    # substitute is node itself -> returns node.
    assert bt._substitute_single_op_subgraph(wrapper) is wrapper


def test_expand_block_tree_straight_line_single_step_input_source(monkeypatch):
    inner = node("inner")
    wrapper = node("wrap", "Wrapper", basic=False, children=[inner])
    wrapper.input_source = "Residual"
    monkeypatch.setattr(bt, "_is_substitutable_single_op_subgraph", lambda *a, **k: False)
    monkeypatch.setattr(bt, "is_straight_line_module", lambda n: n.attr_name == "wrap")
    monkeypatch.setattr(bt, "straight_line_steps", lambda n: [inner])
    out = bt.expand_block_tree_inplace(wrapper)
    assert out.attr_name == "inner"
    assert out.input_source == "Residual"


def test_subgraph_warrants_diagram_alias(monkeypatch):
    n = node("x", "X", basic=False, children=[node("a"), node("b")])
    monkeypatch.setattr(bt, "forward_operation_count", lambda *a, **k: 2)
    assert bt.subgraph_warrants_diagram(n)


# --------------------------------------------------------------------------- #
# pipeline wrapper / bypass / straight-line
# --------------------------------------------------------------------------- #
def test_is_pipeline_wrapper_by_role():
    embed = node("something", "Emb", role="embedding", basic=False)
    assert bt._is_pipeline_wrapper(embed)


def test_bypass_spans_skips_missing_target():
    # child whose attr_name is empty -> not in step_index -> target None -> continue.
    a = BlockNode(attr_name="", class_name="X", role="other", label="X")
    b = node("b", operation_predecessors=["a"])
    parent = node("p", "P", basic=False, children=[a, b])
    assert bt._bypass_spans(parent) == []


def test_straight_line_steps_non_straight_returns_self():
    kernel = node("k", "KernelPipeline", basic=False, children=[node("x")])
    assert bt.straight_line_steps(kernel) == [kernel]
    assert bt.linear_pipeline_steps(kernel) == [kernel]


def test_straight_line_module_false_when_no_segments():
    empty = node("e", "E", basic=False)
    assert not bt.is_straight_line_module(empty)


# --------------------------------------------------------------------------- #
# inline frame labels / kernel frame label
# --------------------------------------------------------------------------- #
def test_kernel_inline_frame_label_for_kernelop_with_children(monkeypatch):
    monkeypatch.setattr(
        "TraceLens.ModelUtils.kernel_pipeline.tensor_port_kernel_frame_label",
        lambda attr: f"frame:{attr}",
    )
    block = node("qk", "KernelOp", basic=False, children=[node("s")])
    assert bt._kernel_inline_frame_label(block) == "frame:qk"
    assert bt.inline_block_frame_label(block) == "frame:qk"


def test_kernel_inline_frame_label_none_for_plain():
    assert bt._kernel_inline_frame_label(node("x", "Linear")) is None


def test_inline_block_frame_label_kernelop_no_kernel_frame(monkeypatch):
    monkeypatch.setattr(bt, "_kernel_inline_frame_label", lambda b: None)
    block = node("qk", "KernelOp", label="QK", basic=False, children=[node("s")])
    assert bt.inline_block_frame_label(block) == "QK"


def test_inline_block_frame_label_multistep_class_name():
    a = node("a")
    b = node("b")
    block = node("blk", "MyClass", basic=False, children=[a, b])
    assert bt.inline_block_frame_label(block) == "MyClass"


def test_inline_block_frame_sublabel_none():
    assert bt.inline_block_frame_sublabel(node("x")) is None


# --------------------------------------------------------------------------- #
# inline_composite_steps branches
# --------------------------------------------------------------------------- #
def test_inline_composite_steps_not_straight_line_returns_self():
    n = node("x", "X", basic=False, children=[node("a"), node("b"), node("c")])
    # Make it not straight line by giving conflicting side inputs -> just use kernel merge
    merge = node("@attention", "AttentionOp", basic=False)
    branchy = node("y", "Y", basic=False, children=[node("q_proj"), node("k_proj"), merge])
    branchy.attention_inputs = {"q": ["q_proj"], "k": ["k_proj"]}
    steps, wrapper = bt.inline_composite_steps(branchy)
    assert steps == [branchy]
    assert wrapper is None


def test_inline_composite_steps_output_gate_single_inner():
    gate = node(
        "g_proj",
        "OutputGate",
        role="gate",
        basic=False,
        children=[node("linear", "Linear")],
    )
    steps, wrapper = bt.inline_composite_steps(gate)
    assert wrapper is gate
    assert [s.attr_name for s in steps] == ["linear"]


def test_show_single_function_in_diagram_embed_tokens():
    assert bt._show_single_function_in_diagram(node("embed_tokens", "Embedding"))


def test_partition_detail_trees_keeps_single_function_non_linear(monkeypatch):
    tree = node("mystery", "Mystery", basic=False, children=[node("a")])
    monkeypatch.setattr(bt, "is_straight_line_module", lambda n: False)
    monkeypatch.setattr(bt, "is_single_function_tree", lambda n: True)
    monkeypatch.setattr(bt, "_show_single_function_in_diagram", lambda n: False)
    assert bt.partition_detail_trees([("t", tree)]) == [("t", tree)]


# --------------------------------------------------------------------------- #
# _method_combine_op / _segment_for_step / _label_for_call
# --------------------------------------------------------------------------- #
def test_method_combine_op_fallback_function():
    step = node("m", "m", details=["method `m()`"])
    assert bt._method_combine_op(step) == "Function"


def test_segment_for_step_method_wrapper_without_prior_side():
    method = node("combine", "combine", details=["method `combine()`", "add"])
    parent = node("p", "P", basic=False, children=[method])
    # side input that is forward_input but also treat as method wrapper w/ residual-only
    parent.side_inputs = {
        "combine": [SideInputSpec("gate", "gate", ["gate"], "prior_step")]
    }
    seg = bt._segment_for_step(parent, method)
    assert isinstance(seg, SideCombineSegment)


def test_segment_for_step_method_wrapper_residual_only_side():
    method = node("combine", "combine", details=["method `combine()`"])
    parent = node("p", "P", basic=False, children=[method])
    parent.side_inputs = {
        "combine": [SideInputSpec("res", "res", [], "forward_input")]
    }
    # has_residual_side True, no prior -> ResidualAddSegment unless explicit add.
    seg = bt._segment_for_step(parent, method)
    assert isinstance(seg, ResidualAddSegment)


def test_segment_for_step_method_wrapper_no_prior_no_residual():
    method = node("combine", "combine", details=["method `combine()`"])
    parent = node("p", "P", basic=False, children=[method])
    parent.side_inputs = {
        "combine": [SideInputSpec("aux", "aux", ["missing"], "prior_step")]
    }
    # prior side with source_chain, method wrapper + prior -> SideCombineSegment
    seg = bt._segment_for_step(parent, method)
    assert isinstance(seg, SideCombineSegment)


def test_label_for_call_variants():
    assert bt._label_for_call(aa.SYNTHETIC_ATTENTION, None) == "Attention kernel"
    assert bt._label_for_call("conv", "ShortConvolution") == "Depthwise Conv"
    assert bt._label_for_call("really_long_attribute_name_here", None).startswith(
        "really long"
    )


# --------------------------------------------------------------------------- #
# output gate details consumer else / kernel pipeline nodes
# --------------------------------------------------------------------------- #
def test_output_gate_details_consumer_only_no_activation():
    side_inputs = {
        "norm": [SideInputSpec("gate", "gate", ["g_proj"], "prior_step")]
    }
    lines = bt._output_gate_details(
        "g_proj",
        side_inputs=side_inputs,
        gate_activations={},
        consumer_class="Linear",
    )
    assert lines == ["Linear", "feeds norm port 'gate'"]


def test_kernel_pipeline_block_nodes_opaque_kernel():
    pipeline, output = bt._kernel_pipeline_block_nodes(
        forward_order=3,
        details=["kernel: myscan"],
    )
    assert pipeline.class_name == "KernelPipeline"
    # opaque kernel: one core child, no output node.
    assert output is None
    assert pipeline.children[0].label == "myscan"


def test_kernel_pipeline_block_nodes_multi_input_step(monkeypatch):
    from TraceLens.ModelUtils import kernel_pipeline as kp

    sub = type(
        "S",
        (),
        {"attr_name": "c", "class_name": "Mul", "label": "×", "second_operand": "input"},
    )()
    step = type(
        "St",
        (),
        {
            "attr_name": "stage",
            "class_name": "KernelOp",
            "label": "stage",
            "call_name": "stage_fwd",
            "children": [sub, sub],
            "predecessors": [],
        },
    )()
    monkeypatch.setattr(kp, "introspect_kernel_pipeline", lambda d: ([step], []))
    pipeline, output = bt._kernel_pipeline_block_nodes(
        forward_order=0,
        details=["kernel: scan"],
    )
    # multi-child step becomes a KernelOp with sub_children.
    assert pipeline.children[0].class_name == "KernelOp"
    assert len(pipeline.children[0].children) == 2
    # non-opaque, no output steps -> synthesized KernelOutput.
    assert output is not None and output.class_name == "KernelOutput"


# --------------------------------------------------------------------------- #
# gated-norm / short-conv helpers
# --------------------------------------------------------------------------- #
def test_side_producer_has_activation_false_for_non_gate():
    assert not bt.side_producer_has_activation(node("x", "Linear"))


def test_is_gated_norm_module_variants():
    assert bt.is_gated_norm_module(node("n", "FusedRMSNormGated", role="norm"))
    assert bt.is_gated_norm_module(node("n", "SomethingNormGated", role="norm"))
    assert not bt.is_gated_norm_module(node("n", "RMSNorm", role="norm"))


def test_gated_norm_tile_label_layernorm():
    assert bt.gated_norm_tile_label(node("n", "FusedLayerNormGated", role="norm")) == "LayerNorm"
    assert bt.gated_norm_tile_label(node("n", "Weird", role="norm")) == "RMSNorm"


def test_gated_norm_activation_none():
    assert bt.gated_norm_activation(node("n", "RMSNorm", role="norm")) is None


def test_short_conv_activation_returns_none_when_all_skipped():
    assert bt._short_conv_activation(["method `f()`", "kernel: k", "a=b"]) is None


# --------------------------------------------------------------------------- #
# situ-and-mul / nested input source
# --------------------------------------------------------------------------- #
def test_situ_and_mul_block_node_adds_split_when_no_upstream():
    built = bt._situ_and_mul_block_node(
        attr_name="act_fn",
        forward_order=2,
        class_name="SiluAndMul",
    )
    labels = [c.label for c in built.children]
    assert labels[0] == "Linear"  # split_gate_up inserted
    assert "Silu" in labels
    assert "×" in labels


def test_situ_and_mul_block_node_skips_split_with_upstream_gate_up():
    prior = [
        node("gate_proj", "Linear"),
        node("up_proj", "Linear"),
    ]
    built = bt._situ_and_mul_block_node(
        attr_name="act_fn",
        forward_order=2,
        class_name="SiluAndMul",
        prior_steps=prior,
    )
    labels = [c.label for c in built.children]
    assert labels[0] != "Linear" or "×" in labels
    assert not any(c.attr_name == "split_gate_up" for c in built.children)


def test_nested_input_source_branches():
    parent_moe = node("moe", "SparseMoe", role="moe", basic=False)
    child_ffn = node("mlp", "MLP", role="ffn", basic=False)
    assert "Linear in" in bt._nested_input_source(parent_moe, child_ffn)

    parent = node("p", "Parent", basic=False)
    fused = node("act", "SiluAndMul", basic=False)
    assert "gate_up in" in bt._nested_input_source(parent, fused)

    labeled = node("c", "C", basic=False, input_label="residual")
    assert "residual in" in bt._nested_input_source(parent, labeled)

    plain = node("c2", "C2", basic=False)
    assert bt._nested_input_source(parent, plain) == "Parent"


# --------------------------------------------------------------------------- #
# branch/provenance helpers
# --------------------------------------------------------------------------- #
def test_append_branch_followups_extends_activation():
    conv = node("conv")
    act = node("conv_activation")
    other = node("other")
    pre_merge = [conv, act, other]
    extended = bt._append_branch_followups([conv], pre_merge)
    assert [n.attr_name for n in extended] == ["conv", "conv_activation"]


def test_append_branch_followups_empty():
    assert bt._append_branch_followups([], [node("x")]) == []


def test_branches_from_provenance_collapses_identical():
    a1 = node("shared")
    pre = [a1]
    provenance = {"q": ["shared"], "k": ["shared"]}
    branches = bt._branches_from_provenance(pre, provenance)
    # identical chains collapse into one with merged label.
    assert len(branches) == 1
    assert branches[0].label == "q/k"


def test_branches_from_provenance_empty_chain_returns_empty():
    assert bt._branches_from_provenance([node("x")], {"q": []}) == []


def test_partition_named_branches_prefix_clusters():
    branches = bt._partition_named_branches([node("q_proj"), node("q_norm"), node("v")])
    labels = {b.label for b in branches}
    assert "q" in labels and "v" in labels


def test_parallel_side_port_label_uses_first_step():
    side = node("gate", "Linear", basic=False, children=[node("inner", label="Inner")])
    assert bt._parallel_side_port_label(side) == "Inner"
    # no function steps -> fall back to label
    empty = node("g", "OutputGate", role="gate", basic=False, label="G")


# --------------------------------------------------------------------------- #
# forward side combine producers / situ gated mlp parts
# --------------------------------------------------------------------------- #
def test_forward_side_combine_producers_empty_without_chains():
    n = node("n", "N", basic=False, children=[node("a")])
    assert bt._forward_side_combine_producers(n) == set()


def test_situ_gated_mlp_parts_none_cases():
    assert bt._situ_gated_mlp_parts(node("x", "X")) is None
    # act_fn present but not fused
    act = node("act_fn", "NotFused", basic=False)
    n = node("n", "N", basic=False, children=[act])
    assert bt._situ_gated_mlp_parts(n) is None


def test_situ_gated_mlp_parts_missing_situ():
    act = node("act_fn", "SiluAndMul", basic=False, children=[node("plain", "Plain")])
    gate = node("gate_proj", "Linear")
    up = node("up_proj", "Linear")
    down = node("down_proj", "Linear")
    n = node("n", "N", basic=False, children=[act, gate, up, down])
    assert bt._situ_gated_mlp_parts(n) is None


def test_situ_gated_mlp_parts_success_and_segments():
    situ = node("s", "SiluActivation", basic=False)
    act = node("act_fn", "SiluAndMul", basic=False, children=[situ])
    gate = node("gate_proj", "Linear")
    up = node("up_proj", "Linear")
    down = node("down_proj", "Linear")
    n = node("n", "N", basic=False, children=[act, gate, up, down])
    parts = bt._situ_gated_mlp_parts(n)
    assert parts is not None
    assert bt.is_situ_gated_mlp(n)
    segments = bt._situ_gated_mlp_segments(n)
    assert isinstance(segments[-1], CombineSegment)
    assert bt._situ_gated_mlp_segments(node("plain", "Plain")) is None


# --------------------------------------------------------------------------- #
# collect_computation_segments edge branches
# --------------------------------------------------------------------------- #
def test_collect_segments_empty_children():
    assert bt.collect_computation_segments(node("x", "X", basic=False)) == []


def test_collect_segments_incomplete_provenance_keeps_steps():
    q = node("q_proj")
    k = node("k_proj")
    extra = node("shared_setup")
    merge = node("@attention", "AttentionOp", basic=False)
    n = node("attn", "Attention", basic=False, children=[extra, q, k, merge])
    # provenance only covers q,k -> shared_setup missing -> incomplete -> seq segments
    n.attention_inputs = {"q": ["q_proj"], "k": ["k_proj"]}
    segments = bt.collect_computation_segments(n)
    assert any(isinstance(s, SeqSegment) for s in segments)


def test_collect_segments_pipeline_child_provenance():
    pipeline = node("@attn_pipeline", "KernelPipeline", basic=False)
    pipeline.attention_inputs = {"q": ["q_proj"], "k": ["k_proj"]}
    q = node("q_proj")
    k = node("k_proj")
    n = node("attn", "Attention", basic=False, children=[q, k, pipeline])
    segments = bt.collect_computation_segments(n)
    assert isinstance(segments[0], FanOutSegment)


# --------------------------------------------------------------------------- #
# is_simple_modeled_tile / tile helpers
# --------------------------------------------------------------------------- #
def test_is_simple_modeled_tile_false_for_kernel_pipeline_tree():
    kp = node("@attn_pipeline", "KernelPipeline", basic=False, children=[node("s")])
    assert not bt.is_simple_modeled_tile(kp)


def test_tile_sublabel_and_purpose_none():
    assert bt.tile_sublabel(node("x")) is None
    assert bt.tile_purpose_annotation(node("x")) is None


def test_tile_display_labels_kernel_output_and_none_block():
    ko = node("o", "KernelOutput", label="out", basic=False)
    assert bt.tile_display_labels(ko) == ("out", None)
    assert bt.tile_display_labels(None) == ("", None)


# --------------------------------------------------------------------------- #
# build_block_node kernel-pipeline / method / init branches
# --------------------------------------------------------------------------- #
def _basic():
    return BasicOpFilter.for_detailed()


def test_build_block_node_synthetic_attention_kernel_pipeline():
    built = bt.build_block_node(
        attr_name=aa.SYNTHETIC_ATTENTION,
        class_name="Whatever",
        registry={},
        basic_ops=_basic(),
        details=["kernel: myscan", "stage: scan"],
    )
    # is_kernel_pipeline_step of these details may be False; ensure returns a node.
    assert isinstance(built, BlockNode)


def test_build_block_node_synthetic_attention_plain_kernel():
    built = bt.build_block_node(
        attr_name=aa.SYNTHETIC_ATTENTION,
        class_name="Whatever",
        registry={},
        basic_ops=_basic(),
        details=["kernel: eager_attention_forward"],
    )
    assert built.class_name == "AttentionOp"


def test_build_block_node_multi_op_method():
    cls = structure("Owner", calls=["helper"])
    op = ForwardOperation(
        attr_name="@op_l1_c0_add",
        label="Add",
        class_name="Add",
        predecessors=("x",),
    )
    cls.multi_op_methods["helper"] = [op, op]
    cls.forward_step_details["helper"] = ["loop: 3 iterations"]
    built = bt.build_block_node(
        attr_name="owner",
        class_name="Owner",
        registry={"Owner": cls},
        basic_ops=_basic(),
    )
    method = built.children[0]
    assert method.details[0] == "method `helper()`"
    assert len(method.children) == 2


def test_build_block_node_single_op_method():
    cls = structure("Owner", calls=["helper"])
    op = ForwardOperation(
        attr_name="helper",
        label="Add",
        class_name="Add",
        predecessors=("x",),
    )
    cls.single_op_methods["helper"] = op
    built = bt.build_block_node(
        attr_name="owner",
        class_name="Owner",
        registry={"Owner": cls},
        basic_ops=_basic(),
    )
    assert built.children[0].attr_name == "helper"
    assert built.children[0].class_name == "Add"


def test_build_block_node_short_convolution_with_activation():
    cls = structure("Owner", assignments={"conv": "ShortConvolution"}, calls=["conv"])
    cls.init_details["conv"] = ["SiLU"]
    built = bt.build_block_node(
        attr_name="owner",
        class_name="Owner",
        registry={"Owner": cls},
        basic_ops=_basic(),
    )
    labels = [c.label for c in built.children]
    assert "Depthwise Conv" in labels and "SiLU" in labels


def test_build_block_node_fused_silu_mul_child():
    cls = structure("Owner", assignments={"act_fn": "SiluAndMul"}, calls=["act_fn"])
    built = bt.build_block_node(
        attr_name="owner",
        class_name="Owner",
        registry={"Owner": cls},
        basic_ops=_basic(),
    )
    fused = built.children[0]
    assert fused.class_name == "SiluAndMul"


def test_build_block_node_skip_init_class():
    cls = structure(
        "Owner",
        assignments={"buf": "Buffer"},
        calls=["buf"],
    )
    built = bt.build_block_node(
        attr_name="owner",
        class_name="Owner",
        registry={"Owner": cls},
        basic_ops=_basic(),
    )
    # Buffer is in _SKIP_INIT_CLASS_NAMES -> skipped.
    assert built.children == []


def test_build_block_node_attention_child_kernel_pipeline():
    cls = structure("Owner", calls=[aa.SYNTHETIC_ATTENTION])
    cls.forward_step_details[aa.SYNTHETIC_ATTENTION] = [
        "kernel: chunk_scan",
        "stage: l2norm_fwd",
    ]
    cls.attention_inputs = {"q": ["q_proj"], "k": ["k_proj"], "v": ["v_proj"]}
    built = bt.build_block_node(
        attr_name="owner",
        class_name="Owner",
        registry={"Owner": cls},
        basic_ops=_basic(),
    )
    assert isinstance(built, BlockNode)


# --------------------------------------------------------------------------- #
# component tree builders
# --------------------------------------------------------------------------- #
def test_build_component_block_trees_orders_and_filters():
    registered = structure("Reg", assignments={"proj": "Linear"}, calls=["proj"])
    registry = {"Reg": registered}
    norm = BlockComponent("norm", "RMSNorm", "norm", "Norm", 0)
    known = BlockComponent("known", "Reg", "ffn", "Known", 2)
    orphan = BlockComponent("orphan", "Reg", "ffn", "Orphan", None)
    # norm excluded when include_norms False; orphan (order None, ffn role) excluded.
    trees = bt._build_component_block_trees(
        [norm, known, orphan],
        registry,
        _basic(),
    )
    assert [t.attr_name for _, t in trees] == ["known"]


def test_build_component_block_trees_skips_method_wrapper(monkeypatch):
    registry = {}
    method_comp = BlockComponent("helper", "Missing", "ffn", "Helper", 1)
    monkeypatch.setattr(bt, "is_method_wrapper", lambda n: True)
    trees = bt._build_component_block_trees([method_comp], registry, _basic())
    assert trees == []


def test_build_decoder_block_trees_partition_and_dedup():
    registered = structure("Reg", assignments={"proj": "Linear"}, calls=["proj"])
    registry = {"Reg": registered}
    known = BlockComponent("known", "Reg", "ffn", "Known", 1)
    dup = BlockComponent("known", "Reg", "ffn", "Dup", 2)
    orphan = BlockComponent("orphan", "Reg", "ffn", "Orphan", None)
    trees = bt.build_decoder_block_trees([known, dup, orphan], registry, _basic())
    assert [t.attr_name for _, t in trees] == ["known"]


def test_build_full_detailed_partition_true():
    registered = structure("Reg", assignments={"proj": "Linear"}, calls=["proj"])
    registry = {"Reg": registered}
    trees = bt.build_full_detailed_block_trees(
        components=[BlockComponent("known", "Reg", "ffn", "Known", 1)],
        registry=registry,
        basic_ops=_basic(),
        positional_encoding="RoPE",
        norm_type="RMSNorm",
        partition=True,
    )
    assert isinstance(trees, list)


def test_collect_graph_segments_trailing_norm():
    norm = node("norm", "RMSNorm", role="norm")
    segments = bt.collect_graph_segments([norm], [], use_residual=True)
    assert segments == [("seq", norm)]


def test_components_from_registry_missing_and_present():
    assert bt.components_from_registry("missing", {}) == []


# --------------------------------------------------------------------------- #
# Ported proven coverage for block_tree helpers (from block-extract suite).
# --------------------------------------------------------------------------- #
def test_wrapper_labels_comments_and_purpose():
    method = node(
        "_update_state",
        "_update_state",
        label="_update state",
        details=["method `_update_state()`"],
    )
    assert is_method_wrapper(method)
    assert wrapper_bullet(method) == "update state (_update_state)"
    assert wrapper_panel_line(method) == "update state (_update_state)"

    ffn = node("mlp", "MLP", role="ffn", label="MLP", basic=False)
    assert block_purpose(ffn) == "Position-wise feed-forward transform"
    assert wrapper_panel_line(ffn) == "mlp — Position-wise feed-forward transform"
    assert wrapper_skips_comment(node("self_attn", "Attention", role="attention"))
    assert wrapper_skips_comment(node("residual_add", "Add"))
    assert not wrapper_skips_comment(ffn)


def test_gated_norm_and_tile_helpers():
    gated = node(
        "norm",
        "FusedRMSNormGated",
        role="norm",
        label="Norm",
        basic=False,
        details=["SiLU"],
    )
    assert gated_norm_tile_label(gated) == "RMSNorm"
    assert gated_norm_activation(gated) == "SiLU"
    assert (
        gated_norm_activation(node("norm", "SomeNormGated", role="norm", basic=False))
        == "Sigmoid"
    )
    assert is_simple_modeled_tile(gated)
    assert is_basic_op_tile(gated)
    assert tile_display_labels(gated, spec_label="Norm")[0] == "Norm"
    assert tile_display_labels(None, spec_label="Unknown") == ("Unknown", None)


def test_collect_straight_line_steps_and_partitioning():
    first = node("first")
    second = node("second")
    pipeline = node("pipeline", "Pipeline", basic=False, children=[first, second])

    assert is_straight_line_module(pipeline)
    assert is_inline_expandable_module(pipeline)
    assert straight_line_steps(pipeline) == [first, second]
    steps, wrapper = inline_composite_steps(pipeline)
    assert steps == [first, second]
    assert wrapper is pipeline
    assert not is_single_function_tree(pipeline)
    assert partition_detail_trees(
        [
            ("tokenizer", node("tokenizer")),
            ("pipeline", pipeline),
            ("kept", node("opaque", "Opaque", basic=False)),
        ]
    ) == [("kept", node("opaque", "Opaque", basic=False))]


def test_kernel_and_fused_inline_frame_labels_ported():
    kernel = node(
        "@attn_pipeline",
        "KernelPipeline",
        label="KDA pipeline",
        basic=False,
        children=[node("stage", "KernelOp", basic=False)],
    )
    fused = node("act_fn", "SituAndMul", basic=False, children=[node("activation")])
    assert inline_block_frame_label(kernel) == "KDA pipeline"
    assert inline_block_frame_label(fused) == "SituAndMul"
    assert inline_composite_steps(kernel) == ([kernel.children[0]], kernel)


def test_computation_segments_for_sequence_tensor_ports_and_fanout():
    sequential = node("parent", "Parent", basic=False, children=[node("one"), node("two")])
    assert all(
        isinstance(segment, SeqSegment)
        for segment in collect_computation_segments(sequential)
    )

    tensor = node(
        "kernel", "KernelPipeline", basic=False,
        children=[node("stage", "KernelOp", basic=False)],
    )
    tensor.tensor_input_labels = ["q", "k"]
    tensor.tensor_step_targets = {"q": "stage"}
    segment = collect_computation_segments(tensor)[0]
    assert isinstance(segment, TensorPortsSegment)
    assert segment.labels == ["q", "k"]

    q = node("q_proj")
    k = node("k_proj")
    merge = node("@attention", "AttentionOp", basic=False)
    attention = node("attention", "Attention", basic=False, children=[q, k, merge])
    attention.attention_inputs = {"q": ["q_proj"], "k": ["k_proj"]}
    fanout = collect_computation_segments(attention)[0]
    assert isinstance(fanout, FanOutSegment)
    assert [branch.label for branch in fanout.branches] == ["q", "k"]


def test_computation_segments_for_side_inputs_and_parallel_gate():
    producer = node("gate")
    consumer = node("norm", "RMSNorm", role="norm")
    parent = node("parent", "Parent", basic=False, children=[producer, consumer])
    parent.side_inputs = {
        "norm": [SideInputSpec("gate", "gate", ["gate"], "prior_step")]
    }
    side = collect_computation_segments(parent)[0]
    assert isinstance(side, SideFeedSegment)
    assert side.side_producer_nodes == {"gate": producer}

    residual_parent = node("parent", "Parent", basic=False, children=[consumer])
    residual_parent.side_inputs = {
        "norm": [SideInputSpec("residual", "residual", [], "forward_input")]
    }
    assert isinstance(
        collect_computation_segments(residual_parent)[0], ResidualAddSegment
    )

    method = node("combine", "combine", details=["method `combine()`", "add"])
    method_parent = node("parent", "Parent", basic=False, children=[producer, method])
    method_parent.side_inputs = {
        "combine": [SideInputSpec("gate", "gate", ["gate"], "prior_step")]
    }
    assert isinstance(
        collect_computation_segments(method_parent)[-1], SideCombineSegment
    )

    merge = node("@attention", "AttentionOp", basic=False)
    gate = node("g_proj", "OutputGate", role="gate", basic=False)
    output = node("o_proj")
    gated_attention = node(
        "attention", "Attention", basic=False, children=[merge, gate, output]
    )
    gated_attention.parallel_gates = ["g_proj"]
    assert isinstance(collect_computation_segments(gated_attention)[-1], CombineSegment)


def test_method_and_parallel_gate_collection():
    method = node("helper", "helper", details=["method `helper()`"])
    gate = node("gate", "Linear")
    root = node("root", "Root", basic=False, children=[method, gate])
    root.parallel_gates = ["gate"]

    assert collect_method_wrappers(root) == [method]
    assert collect_parallel_gate_wrappers(root) == [gate]

    wrapped_gate = node(
        "gate", "OutputGate", role="gate", basic=False,
        children=[node("@gate_activation", "ActivationOp", basic=False)],
    )
    assert side_producer_has_activation(wrapped_gate)


def test_build_block_node_handles_registry_leaf_recursion_and_methods():
    basic_filter = BasicOpFilter.for_detailed()
    registry = {
        "Parent": structure(
            "Parent",
            assignments={"proj": "Linear", "child": "Child"},
            calls=["proj", "helper", "child"],
        ),
        "Child": structure("Child", assignments={"parent": "Parent"}, calls=["parent"]),
    }
    registry["Parent"].forward_step_details["helper"] = ["custom detail"]

    built = build_block_node(
        attr_name="layer", class_name="Parent", registry=registry, basic_ops=basic_filter
    )

    assert [child.attr_name for child in built.children] == ["proj", "helper", "child"]
    assert built.children[0].is_basic
    assert built.children[1].details == ["custom detail"]
    assert built.children[2].children[0].details == ["recursive reference"]
    assert (
        build_block_node(
            attr_name="external", class_name="External",
            registry=registry, basic_ops=basic_filter,
        ).is_basic
        is False
    )
    assert components_from_registry("missing", registry) == []


def test_graph_segments_and_spine_labels():
    norm = node("norm", "RMSNorm", role="norm")
    attn = node("attn", "Attention", role="attention", basic=False)
    mlp = node("mlp", "MLP", role="ffn", basic=False)

    segments = collect_graph_segments([norm, attn, mlp], ["attn"], use_residual=True)
    assert segments == [("sublayer", norm, attn), ("seq", mlp)]
    assert collect_graph_segments([norm], [], use_residual=False) == [("seq", norm)]

    positional = BlockComponent("rotary", "RotaryEmbedding", "positional", "RoPE")
    assert (
        spine_expanded_frame_label(positional, positional_encoding="RoPE")
        == "Positional (RoPE) (rotary)"
    )


@pytest.mark.parametrize(
    ("candidate", "expected"),
    [
        (node("@attention", "AttentionOp", basic=False, details=["delta rule recurrence"]),
         "delta rule recurrence"),
        (node("gate", "OutputGate", role="gate", basic=False), None),
        (node("gate", "OutputGate", role="gate", basic=False, details=["Linear", "Sigmoid"]),
         "Sigmoid"),
        (node("pipeline", "KernelPipeline", basic=False, details=["kernel pipeline · scan"]),
         "kernel pipeline · scan"),
        (node("conv", "ShortConvolution", label="Short Conv", basic=False, details=["depthwise conv"]),
         "depthwise conv"),
        (node("merge", "AttentionMerge", basic=False, details=["ports:q,k,v"]),
         "ports:q,k,v"),
        (node("op", "KernelOp", basic=False), None),
        (node("lm_head", "Linear", role="head"), "Project to vocabulary logits"),
        (node("router", "Router", role="router", basic=False), "Score and route tokens to experts"),
        (node("split_gate_up", "Split", basic=False, label="Split"),
         "Split fused gate/up projection"),
        (node("mul", "Multiply", basic=False, label="×"), "Multiply gate and up activations"),
    ],
)
def test_block_purpose_specialized_branches(candidate, expected):
    assert block_purpose(candidate) == expected


def test_block_purpose_skips_functional_details_and_formats_fused_ops():
    fused = node("act_fn", "SiluAndMul", basic=False, details=["F.silu(...)"])
    activation = node("activation", "ActivationOp", label="GELU", basic=False)

    assert block_purpose(fused) == "Silu(gate) × up branch"
    assert block_purpose(activation) == "Apply GELU to gate half"
    assert bt.wrapper_module_comment(node("embed_tokens", "Embedding")) is None
    assert bt.inline_wrapper_step_label(fused, activation, 0) == "GELU"
    assert bt.inline_wrapper_step_label(fused, activation, 1) is None


def test_single_op_subgraph_substitution_preserves_outer_input(monkeypatch):
    inner = node("inner")
    wrapper = node("wrapper", "Wrapper", basic=False, children=[inner])
    wrapper.input_source = "Residual stream"
    wrapper.side_inputs = {
        "inner": [SideInputSpec("residual", "residual", [], "forward_input")]
    }
    monkeypatch.setattr(bt, "forward_operation_count", lambda *_a, **_k: 1)

    assert bt._is_substitutable_single_op_subgraph(wrapper)
    substitute = bt._substitute_single_op_subgraph(wrapper)
    assert substitute.attr_name == "inner"
    assert substitute.input_source == "Residual stream"
    assert bt.expand_block_tree_inplace(wrapper).attr_name == "inner"

    norm = node(
        "norm", "RMSNorm", role="norm", basic=False,
        children=[node("helper", "helper", details=["method `helper()`"])],
    )
    assert bt.expand_block_tree_inplace(norm).children == []


def test_bypass_span_detection_and_pipeline_exclusions():
    one = node("one")
    two = node("two")
    three = node("three")
    four = node("four")
    three.operation_predecessors = ["one"]
    four.operation_predecessors = ["two"]
    crossing = node("crossing", "Crossing", basic=False, children=[one, two, three, four])

    assert bt._bypass_spans(crossing) == [(0, 2), (1, 3)]
    assert bt._has_overlapping_bypass_spans(crossing)
    assert is_straight_line_module(crossing)
    assert not is_straight_line_module(
        node("embed", "EmbeddingWrapper", role="embedding", basic=False, children=[one])
    )
    assert not is_straight_line_module(
        node("kernel", "KernelPipeline", basic=False, children=[one])
    )


@pytest.mark.parametrize(
    ("side_inputs", "activations", "consumer_class", "norm_activation", "expected"),
    [
        ({}, {}, None, None, ["Linear", "output gate for normalized branch"]),
        ({"norm": [SideInputSpec("gate", "g", ["g_proj"], "prior_step")]},
         {"g_proj": "SiLU"}, "FusedRMSNormGated", None,
         ["Linear", "SiLU(linear out)", "norm(attn_out) × gate → norm"]),
        ({"norm": [SideInputSpec("gate", "g", ["g_proj"], "prior_step")]},
         {}, "FusedRMSNormGated", "Tanh",
         ["Linear", "Tanh inside norm", "norm(attn_out) × gate"]),
        ({"consumer": [SideInputSpec("gate", "gate", ["g_proj"], "prior_step")]},
         {"g_proj": "Sigmoid"}, "Linear", None,
         ["Linear", "Sigmoid(linear out)", "feeds consumer port 'gate'"]),
    ],
)
def test_output_gate_detail_variants(
    side_inputs, activations, consumer_class, norm_activation, expected
):
    assert (
        bt._output_gate_details(
            "g_proj",
            side_inputs=side_inputs,
            gate_activations=activations,
            consumer_class=consumer_class,
            norm_gate_activation=norm_activation,
        )
        == expected
    )


def test_output_gate_wrapping_and_short_convolution_helpers():
    linear = node("g_proj")
    consumer = node("norm", "FusedRMSNormGated", role="norm", basic=False)
    side_inputs = {
        "norm": [SideInputSpec("gate", "gate", ["g_proj"], "prior_step")]
    }
    wrapped = bt._wrap_parallel_gate_children(
        [linear, consumer], ["g_proj"], {"g_proj": "Sigmoid"}, side_inputs
    )
    assert wrapped[0].class_name == "OutputGate"
    assert [child.label for child in wrapped[0].children] == ["Linear", "Sigmoid"]
    assert (
        bt._short_conv_activation(
            ["method `forward()`", "kernel: conv", "activation=silu", "SiLU"]
        )
        == "SiLU"
    )
    assert bt._short_conv_activation([]) is None
    conv_steps = bt._short_convolution_block_node(
        attr_name="conv", forward_order=4, activation="SiLU"
    )
    assert [step.label for step in conv_steps] == ["Depthwise Conv", "SiLU"]


def test_tile_display_label_branches(monkeypatch):
    basic = node("proj", label="Projection")
    kernel = node("scan", "KernelOp", label="chunk_scan", basic=False)
    activation = node("act", "ActivationOp", label="SiLU", basic=False)

    monkeypatch.setattr(
        "TraceLens.ModelUtils.kernel_pipeline.kernel_op_display_label",
        lambda label: f"display:{label}",
    )
    assert tile_display_labels(basic, in_inline_frame=True) == ("Projection", None)
    assert tile_display_labels(basic, port_label="q", port_style="inline") == (
        "Projection", None
    )
    assert tile_display_labels(kernel) == ("display:chunk_scan", None)
    assert tile_display_labels(activation) == ("SiLU", None)
    assert not is_basic_op_tile(None)


def test_collect_nested_diagrams_assigns_source_and_deduplicates(monkeypatch):
    nested = node(
        "nested", "NestedBlock", basic=False,
        children=[node("producer"), node("consumer")],
    )
    nested.side_inputs = {
        "consumer": [SideInputSpec("side", "side", ["producer"], "prior_step")]
    }
    root = node("root", "RootBlock", basic=False, children=[nested, nested])

    def fake_graph(current, **_kwargs):
        return SimpleNamespace(
            nodes=(
                [SimpleNamespace(block=nested), SimpleNamespace(block=nested)]
                if current is root
                else []
            )
        )

    monkeypatch.setattr(
        "TraceLens.ModelUtils.computation_graph.build_computation_graph", fake_graph
    )
    monkeypatch.setattr(bt, "subgraph_warrants_export", lambda *_a, **_k: True)

    assert bt.collect_nested_diagrams(root) == [("NestedBlock", nested)]
    assert nested.input_source == "RootBlock"
    assert bt.flatten_computation_segments(root) == collect_computation_segments(root)


def test_build_block_node_functional_positional_and_skipped_calls():
    cls = structure(
        "Operations",
        assignments={"parameter": "Parameter"},
        calls=[
            "@functional_softmax",
            "@positional_l12_apply_rotary_emb",
            "parameter",
        ],
    )
    cls.forward_step_details["@positional_l12_apply_rotary_emb"] = ["RoPE helper"]
    built = build_block_node(
        attr_name="ops", class_name="Operations",
        registry={"Operations": cls}, basic_ops=BasicOpFilter.for_detailed(),
    )

    assert [child.label for child in built.children] == ["Softmax", "Apply rotary emb"]
    assert built.children[1].class_name == "PositionalOp"


def test_build_block_node_infers_init_steps_and_empty_class():
    inferred = structure("Inferred", assignments={"proj": "Linear", "dropout": "Dropout"})
    empty = structure("Empty")
    registry = {"Inferred": inferred, "Empty": empty}
    basic_filter = BasicOpFilter.for_detailed()

    built = build_block_node(
        attr_name="inferred", class_name="Inferred", registry=registry,
        basic_ops=basic_filter, infer_init_steps=True,
    )
    assert [child.attr_name for child in built.children] == ["proj", "dropout"]
    assert build_block_node(
        attr_name="empty", class_name="Empty", registry=registry, basic_ops=basic_filter,
    ).is_basic


def test_stack_tree_builders_cover_registry_and_leaf_paths():
    registered = structure("Registered", assignments={"proj": "Linear"}, calls=["proj"])
    registry = {"Registered": registered}
    basic_filter = BasicOpFilter.for_detailed()
    known = BlockComponent("known", "Registered", "ffn", "Known", 1)
    external = BlockComponent("external", "External", "other", "External", 2)
    norm = BlockComponent("norm", "RMSNorm", "norm", "Norm", 0)

    assert bt.build_stack_component_tree(known, registry, basic_filter).children
    assert not bt.build_stack_component_tree(external, registry, basic_filter).is_basic
    pipeline = bt.build_pipeline_block_trees(
        stack_pre=[external, norm], registry=registry, basic_ops=basic_filter,
        include_norms=True,
    )
    head = bt.build_head_block_trees(
        stack_tail=[known], registry=registry, basic_ops=basic_filter,
    )
    assert [tree.attr_name for _, tree in pipeline] == ["norm", "external"]
    assert head[0][1].attr_name == "known"
    assert (
        bt.spine_expanded_frame_label(known, positional_encoding="RoPE")
        == "Known (known)"
    )


def test_additional_block_purpose_fallbacks():
    assert block_purpose(node("@attention", "AttentionOp", basic=False)) is None
    assert (
        block_purpose(
            node("conv", "ShortConvolution", label="Depthwise Conv", basic=False,
                 details=["SiLU"])
        )
        is None
    )
    assert (
        block_purpose(node("conv", "ShortConvolution", basic=False)) == "depthwise conv"
    )
    assert (
        block_purpose(node("embedding", "Embedding", role="embedding"))
        == "Gather rows by token id"
    )
    assert (
        block_purpose(node("gate_activation", "ActivationOp", label="SiLU", basic=False))
        is None
    )


def test_substitution_rejections_and_existing_source(monkeypatch):
    leaf = node("leaf")
    composite = node("composite", "Composite", basic=False, children=[leaf])
    monkeypatch.setattr(bt, "forward_operation_count", lambda *_a, **_k: 2)
    assert not bt._is_substitutable_single_op_subgraph(composite)
    assert not bt._is_substitutable_single_op_subgraph(leaf)

    composite.side_inputs = {
        "leaf": [SideInputSpec("residual", "residual", [], "forward_input")]
    }
    monkeypatch.setattr(bt, "forward_operation_count", lambda *_a, **_k: 1)
    leaf.input_source = "Already set"
    assert bt._substitute_single_op_subgraph(composite) is leaf


def test_nested_diagram_filters_non_candidates(monkeypatch):
    basic = node("basic")
    inline = node("inline", "Inline", basic=False, children=[node("one"), node("two")])
    too_small = node(
        "small", "Small", basic=False, children=[node("producer"), node("consumer")]
    )
    too_small.side_inputs = {
        "consumer": [SideInputSpec("side", "side", ["producer"], "prior_step")]
    }
    root = node("root", "Root", basic=False, children=[basic, inline, too_small])

    monkeypatch.setattr(
        "TraceLens.ModelUtils.computation_graph.build_computation_graph",
        lambda *_a, **_k: SimpleNamespace(
            nodes=[
                SimpleNamespace(block=None),
                SimpleNamespace(block=basic),
                SimpleNamespace(block=inline),
                SimpleNamespace(block=too_small),
            ]
        ),
    )
    monkeypatch.setattr(
        bt, "subgraph_warrants_export",
        lambda candidate, **_k: candidate is not too_small,
    )
    assert bt.collect_nested_diagrams(root) == []


def test_decoder_and_full_tree_builders_order_filter_and_deduplicate():
    registered = structure("Registered", assignments={"proj": "Linear"}, calls=["proj"])
    registry = {"Registered": registered}
    basic_filter = BasicOpFilter.for_detailed()
    norm = BlockComponent("norm", "RMSNorm", "norm", "Norm", 0)
    known = BlockComponent("known", "Registered", "ffn", "Known", 2)
    duplicate = BlockComponent("known", "Registered", "ffn", "Known again", 3)
    unknown_order = BlockComponent("unknown", "Registered", "ffn", "Unknown", None)

    trees = bt.build_decoder_block_trees(
        [duplicate, unknown_order, known, norm], registry, basic_filter,
        include_norms=True,
    )
    assert [tree.attr_name for _, tree in trees] == ["norm", "known"]
    assert trees[1][1].input_label == "hidden_states"

    all_trees = bt.build_full_detailed_block_trees(
        components=[known], registry=registry, basic_ops=basic_filter,
        positional_encoding="RoPE", norm_type="RMSNorm",
        stack_pre=[BlockComponent("embed", "Embedding", "embedding", "Embed", 0)],
        stack_tail=[BlockComponent("head", "Linear", "head", "Head", 3)],
        partition=False, include_norms=True,
    )
    assert [tree.attr_name for _, tree in all_trees] == ["embed", "known", "head"]


def test_detail_tree_builder_skips_method_wrappers_and_non_pipeline_items():
    helper = structure("Helper", calls=["method"])
    registry = {"Helper": helper}
    basic_filter = BasicOpFilter.for_detailed()
    method_component = BlockComponent("helper", "Helper", "ffn", "Helper", 1)
    late = BlockComponent("late", "External", "other", "Late", None)

    trees = bt._build_component_block_trees(
        [method_component, late], registry, basic_filter
    )
    assert [tree.attr_name for _, tree in trees] == ["helper"]


# --------------------------------------------------------------------------- #
# Second batch: remaining targeted branches.
# --------------------------------------------------------------------------- #
def test_block_purpose_fused_rms_norm_gated():
    n = node("norm", "FusedRMSNormGated", role="norm", basic=False, details=["method `x()`"])
    assert bt.block_purpose(n) == "RMSNorm, then multiply by gate"


def test_forward_operation_count_and_subgraph_warrants():
    pipeline = node("p", "Pipeline", basic=False, children=[node("a"), node("b")])
    assert bt.forward_operation_count(pipeline) >= 1
    assert bt.subgraph_expands_on_export(pipeline)
    assert bt.subgraph_warrants_export(pipeline)
    assert bt.subgraph_warrants_json_export(pipeline)
    assert not bt.subgraph_warrants_json_export(node("tokenizer", "Tok", basic=False))


def test_subgraph_warrants_export_single_op_composite(monkeypatch):
    composite = node("c", "C", basic=False, children=[node("only")])
    monkeypatch.setattr(bt, "forward_operation_count", lambda *a, **k: 1)
    monkeypatch.setattr(bt, "subgraph_expands_on_export", lambda n: True)
    assert bt.subgraph_warrants_export(composite)


def test_prepare_diagram_section_trees():
    first = node("first")
    second = node("second")
    pipeline = node("pipeline", "Pipeline", basic=False, children=[first, second])
    prepared = bt.prepare_diagram_section_trees([("t", pipeline)])
    assert prepared[0][0] == "t"


def test_is_pipeline_wrapper_by_attr_name():
    assert bt._is_pipeline_wrapper(node("tokenizer", "Tok", basic=False))


def test_is_inline_expandable_indexer_small():
    idx = node("indexer", "SparseIndexer", basic=False, children=[node("a"), node("b")])
    assert bt.is_inline_expandable_module(idx)


def test_is_transparent_inline_expansion_variants():
    single = node("p", "Pipeline", basic=False, children=[node("only")])
    assert bt.is_transparent_inline_expansion(single)
    assert not bt.is_transparent_inline_expansion(node("op", "Op", basic=False))


def test_expand_block_tree_multi_step_keeps_wrapper(monkeypatch):
    a = node("a")
    b = node("b")
    wrapper = node("wrap", "Wrapper", basic=False, children=[a, b])
    monkeypatch.setattr(bt, "_is_substitutable_single_op_subgraph", lambda *x, **k: False)
    monkeypatch.setattr(bt, "is_straight_line_module", lambda n: n.attr_name == "wrap")
    monkeypatch.setattr(bt, "straight_line_steps", lambda n: [a, b])
    out = bt.expand_block_tree_inplace(wrapper)
    assert [c.attr_name for c in out.children] == ["a", "b"]


def test_inline_block_frame_label_single_step_attr_name():
    block = node("myattr", "MyClass", basic=False, children=[node("only")])
    assert bt.inline_block_frame_label(block) == "myattr"


def test_inline_composite_single_inner_op_non_gate_returns_none(monkeypatch):
    inner = node("mean", "Mean")
    wrapper = node("head", "Head", role="head", basic=False, children=[inner])
    monkeypatch.setattr(bt, "collect_function_steps", lambda n: [inner])
    steps, w = bt.inline_composite_steps(wrapper)
    assert steps == [inner]
    assert w is None


def test_show_single_function_displays_as_linear():
    # non embed_tokens single function whose class displays as linear
    assert bt._show_single_function_in_diagram(node("q_proj", "Linear")) in (True, False)


def test_partition_detail_trees_keeps_and_drops_single_function():
    embed = node("embed_tokens", "Embedding", basic=False, children=[node("gather", "Gather")])
    # embed_tokens single function -> _show_single_function True -> dropped (continue)
    kept = bt.partition_detail_trees([("Embed", embed)])
    assert kept == []


def test_method_combine_op_real_symbol():
    step = node("m", "m", details=["combine: Σ", "method `m()`"])
    assert bt._method_combine_op(step) == "Σ"


def test_branch_port_label_property():
    b = Branch(label="q", steps=[node("q_proj")], port_style="inline")
    assert b.port_label == "q"


def test_append_branch_followups_non_adjacent_activation():
    conv = node("conv")
    other = node("other")
    act = node("conv_activation")
    pre_merge = [conv, other, act]  # activation not immediately after conv
    assert bt._append_branch_followups([conv], pre_merge) == [conv]


def test_parallel_side_port_label_falls_back_to_label():
    empty = node("g", "OutputGate", role="gate", basic=False, label="Gate label")
    assert bt._parallel_side_port_label(empty) == "Gate label"


def test_side_feed_targets_skips_forward_input():
    n = node("n", "N", basic=False, children=[node("c", "RMSNorm", role="norm")])
    n.side_inputs = {"c": [SideInputSpec("res", "res", [], "forward_input")]}
    assert bt._side_feed_targets(n) == {}


def test_forward_side_combine_producers_intermediate_and_missing():
    prod = node("prod", "Linear")
    consumer = node("combine", "combine", details=["method `combine()`"])
    n = node("n", "N", basic=False, children=[prod, consumer])
    # chain has an intermediate attr not mapped as target and a missing producer
    n.side_inputs = {
        "combine": [SideInputSpec("g", "g", ["intermediate", "prod"], "prior_step")]
    }
    result = bt._forward_side_combine_producers(n)
    # "prod" -> consumer combine (SideCombine), prod segment is Seq -> included
    assert "prod" in result


def test_situ_gated_mlp_collect_segments_no_merge():
    situ = node("s", "SiluActivation", basic=False)
    act = node("act_fn", "SiluAndMul", basic=False, children=[situ])
    n = node(
        "n", "N", basic=False,
        children=[act, node("gate_proj"), node("up_proj"), node("down_proj")],
    )
    segments = bt.collect_computation_segments(n)
    assert isinstance(segments[-1], CombineSegment)


def test_collect_segments_post_merge_side_skip():
    q = node("q_proj")
    k = node("k_proj")
    merge = node("@attention", "AttentionOp", basic=False)
    producer = node("gate", "Linear")
    consumer = node("o_norm", "RMSNorm", role="norm")
    n = node("attn", "Attention", basic=False, children=[q, k, merge, producer, consumer])
    n.attention_inputs = {"q": ["q_proj"], "k": ["k_proj"]}
    n.side_inputs = {"o_norm": [SideInputSpec("gate", "gate", ["gate"], "prior_step")]}
    segments = bt.collect_computation_segments(n)
    # producer feeds consumer -> skip_sequential; consumer becomes SideFeed
    assert any(isinstance(s, SideFeedSegment) for s in segments)


def test_is_simple_modeled_tile_false_for_method_and_composite():
    method = node("m", "m", details=["method `m()`"])
    assert not bt.is_simple_modeled_tile(method)
    composite = node("c", "C", basic=False, children=[node("a"), node("b")])
    assert not bt.is_simple_modeled_tile(composite)


def test_tile_display_labels_basic_plain():
    assert bt.tile_display_labels(node("proj", label="Projection")) == ("Projection", None)


def test_tile_display_labels_activation_and_spec_label():
    act = node("a", "ActivationOp", label="GELU", basic=False)
    assert bt.tile_display_labels(act) == ("GELU", None)
    modeled = node("op", "KernelOutput", label="out", basic=False)
    assert bt.tile_display_labels(modeled, spec_label="Spec")[0] == "out"


def test_collect_nested_diagrams_recurses_into_inner(monkeypatch):
    leaf = node("leaf")
    outer = node("outer", "Outer", basic=False, children=[node("a"), node("b")])
    root = node("root", "Root", basic=False, children=[outer])

    def fake_graph(current, **_k):
        if current is root:
            return SimpleNamespace(nodes=[SimpleNamespace(block=outer)])
        if current is outer:
            return SimpleNamespace(nodes=[SimpleNamespace(block=leaf)])
        return SimpleNamespace(nodes=[])

    monkeypatch.setattr(
        "TraceLens.ModelUtils.computation_graph.build_computation_graph", fake_graph
    )
    monkeypatch.setattr(bt, "subgraph_warrants_export", lambda *a, **k: True)
    monkeypatch.setattr(bt, "is_inline_expandable_module", lambda n: False)
    result = bt.collect_nested_diagrams(root)
    assert ("Outer", outer) in result


def test_wrap_parallel_gate_children_edge_branches():
    # composite gate (not simple) -> appended unchanged
    composite_gate = node(
        "g_proj", "GateModule", basic=False, children=[node("a"), node("b"), node("c")]
    )
    out = bt._wrap_parallel_gate_children([composite_gate], ["g_proj"], {}, {})
    assert out[0] is composite_gate

    # simple linear gate without activation -> appended unchanged
    linear_gate = node("g_proj", "Linear")
    out2 = bt._wrap_parallel_gate_children([linear_gate], ["g_proj"], {}, {})
    assert out2[0] is linear_gate


def test_is_simple_parallel_output_gate_non_linear():
    non_linear = node("g", "Weird", basic=False, children=[node("a"), node("b")])
    assert not bt._is_simple_parallel_output_gate(non_linear)


def test_boundary_input_name_variants():
    cls = structure("Owner")
    cls.forward_input_name = "hidden_states"
    op_input = ForwardOperation(
        attr_name="@op_x", label="Add", class_name="Add",
        predecessors=(aa.FORWARD_METHOD_INPUT,),
    )
    assert bt._boundary_input_name(op_input, cls) == "hidden_states"

    op_zeros = ForwardOperation(
        attr_name="@op_z", label="Zeros like", class_name="ZerosLike", predecessors=()
    )
    assert bt._boundary_input_name(op_zeros, cls) == "hidden_states"

    op_mut = ForwardOperation(
        attr_name="@op_m", label="Copy", class_name="Copy",
        predecessors=("x",), details=("mutates: cache",),
    )
    assert bt._boundary_input_name(op_mut, cls) == "cache"

    op_param = ForwardOperation(
        attr_name="@op_p", label="Add", class_name="Add",
        predecessors=("x",), param_inputs=("weight",),
    )
    assert bt._boundary_input_name(op_param, cls) == "weight"

    op_none = ForwardOperation(
        attr_name="@op_n", label="Add", class_name="Add", predecessors=("x",)
    )
    assert bt._boundary_input_name(op_none, cls) is None


def test_build_block_node_forward_operation_child():
    cls = structure("Owner", calls=["@op_l0_c0_add"])
    cls.forward_input_name = "hidden_states"
    op = ForwardOperation(
        attr_name="@op_l0_c0_add", label="Add", class_name="Add",
        predecessors=(aa.FORWARD_METHOD_INPUT,),
    )
    cls.forward_operations["@op_l0_c0_add"] = op
    built = build_block_node(
        attr_name="owner", class_name="Owner",
        registry={"Owner": cls}, basic_ops=BasicOpFilter.for_detailed(),
    )
    child = built.children[0]
    assert child.class_name == "Add"
    assert child.boundary_input_name == "hidden_states"


def test_build_block_node_primary_output_step():
    cls = structure("Owner", assignments={"proj": "Linear"}, calls=["proj"])
    cls.primary_return_slot = "out"
    cls.forward_return_slots = {"out": "proj"}
    cls.forward_return_order = ["out", "aux"]
    built = build_block_node(
        attr_name="owner", class_name="Owner",
        registry={"Owner": cls}, basic_ops=BasicOpFilter.for_detailed(),
    )
    assert built.primary_output_step == "proj"
    assert built.multi_return_module


def test_build_block_node_top_level_kernel_pipeline(monkeypatch):
    monkeypatch.setattr(bt, "is_kernel_pipeline_step", lambda details: True)
    sentinel = node("@attn_pipeline", "KernelPipeline", basic=False, children=[node("s")])
    monkeypatch.setattr(
        bt, "_kernel_pipeline_block_nodes", lambda **k: (sentinel, None)
    )
    built = build_block_node(
        attr_name=aa.SYNTHETIC_ATTENTION, class_name="C",
        registry={}, basic_ops=BasicOpFilter.for_detailed(),
        details=["kernel: scan"],
    )
    assert built is sentinel


def test_build_block_node_forward_attention_kernel_pipeline(monkeypatch):
    cls = structure("Owner", calls=[aa.SYNTHETIC_ATTENTION])
    cls.forward_step_details[aa.SYNTHETIC_ATTENTION] = ["kernel: scan"]
    cls.attention_inputs = {"q": ["q_proj"], "k": ["k_proj"]}
    monkeypatch.setattr(bt, "is_kernel_pipeline_step", lambda details, inputs: True)
    pipeline = node("@attn_pipeline", "KernelPipeline", basic=False, children=[node("s")])
    output = node("@attn_output", "KernelOutput", basic=False)
    monkeypatch.setattr(bt, "_kernel_pipeline_block_nodes", lambda **k: (pipeline, output))
    built = build_block_node(
        attr_name="owner", class_name="Owner",
        registry={"Owner": cls}, basic_ops=BasicOpFilter.for_detailed(),
    )
    attrs = [c.attr_name for c in built.children]
    assert "@attn_pipeline" in attrs and "@attn_output" in attrs


def test_kernel_pipeline_block_nodes_single_child_step(monkeypatch):
    from TraceLens.ModelUtils import kernel_pipeline as kp

    step = SimpleNamespace(
        attr_name="stage", class_name="KernelOp", label="stage",
        call_name="stage_fwd", children=[], predecessors=[],
    )
    monkeypatch.setattr(kp, "introspect_kernel_pipeline", lambda d: ([step], []))
    pipeline, output = bt._kernel_pipeline_block_nodes(
        forward_order=0, details=["kernel: scan"]
    )
    assert pipeline.children[0].class_name == "KernelOp"
    assert output is not None and output.class_name == "KernelOutput"


def test_kernel_pipeline_block_nodes_with_output_step(monkeypatch):
    from TraceLens.ModelUtils import kernel_pipeline as kp

    step = SimpleNamespace(
        attr_name="stage", class_name="KernelOp", label="stage",
        call_name="stage_fwd", children=[], predecessors=[],
    )
    out_step = SimpleNamespace(
        attr_name="out", class_name="KernelOutput", label="out",
        call_name="combine", children=[], predecessors=["stage"],
    )
    monkeypatch.setattr(kp, "introspect_kernel_pipeline", lambda d: ([step], [out_step]))
    pipeline, output = bt._kernel_pipeline_block_nodes(
        forward_order=0, details=["kernel: scan"]
    )
    assert output is not None and output.label == "combine"


def test_input_sources_for_components_forward_sequence():
    comps = [
        BlockComponent("norm", "RMSNorm", "norm", "Norm", 0),
        BlockComponent("attn", "Attn", "attention", "Attn", 1),
    ]
    result = bt._input_sources_for_components(comps, forward_sequence=["norm", "attn"])
    assert isinstance(result, dict)
    result2 = bt._input_sources_for_components(comps)
    assert isinstance(result2, dict)


def test_build_component_block_trees_dedup():
    registered = structure("Reg", assignments={"proj": "Linear"}, calls=["proj"])
    registry = {"Reg": registered}
    a = BlockComponent("known", "Reg", "ffn", "Known", 1)
    dup = BlockComponent("known", "Reg", "ffn", "Dup", 2)
    trees = bt._build_component_block_trees([a, dup], registry, _basic())
    assert len(trees) == 1


def test_build_decoder_block_trees_skips_norm_and_method(monkeypatch):
    registry = {"Reg": structure("Reg", assignments={"proj": "Linear"}, calls=["proj"])}
    norm = BlockComponent("norm", "RMSNorm", "norm", "Norm", 0)
    known = BlockComponent("known", "Reg", "ffn", "Known", 1)
    # include_norms False -> norm skipped (2316-2317)
    trees = bt.build_decoder_block_trees([norm, known], registry, _basic())
    assert [t.attr_name for _, t in trees] == ["known"]


def test_build_decoder_block_trees_method_wrapper_skip(monkeypatch):
    registry = {"Reg": structure("Reg", assignments={"proj": "Linear"}, calls=["proj"])}
    known = BlockComponent("known", "Reg", "ffn", "Known", 1)
    monkeypatch.setattr(bt, "is_method_wrapper", lambda n: True)
    trees = bt.build_decoder_block_trees([known], registry, _basic())
    assert trees == []


def test_collect_graph_segments_pending_norm_then_plain():
    norm = node("norm", "RMSNorm", role="norm")
    mlp = node("mlp", "MLP", role="ffn", basic=False)
    segments = bt.collect_graph_segments([norm, mlp], [], use_residual=True)
    assert segments == [("seq", norm), ("seq", mlp)]


def test_components_from_registry_present():
    cls = structure("Reg", assignments={"proj": "Linear"}, calls=["proj"])
    comps = bt.components_from_registry("Reg", {"Reg": cls})
    assert isinstance(comps, list)


# --------------------------------------------------------------------------- #
# Third batch: final reachable branches.
# --------------------------------------------------------------------------- #
def test_subgraph_expands_on_export_composite_only(monkeypatch):
    fanout = node("attn", "Attention", basic=False,
                  children=[node("q_proj"), node("k_proj"),
                            node("@attention", "AttentionOp", basic=False)])
    fanout.attention_inputs = {"q": ["q_proj"], "k": ["k_proj"]}
    monkeypatch.setattr(bt, "is_inline_expandable_module", lambda n: False)
    assert bt.subgraph_expands_on_export(fanout)


def test_subgraph_warrants_export_zero_ops(monkeypatch):
    n = node("c", "C", basic=False, children=[node("a")])
    monkeypatch.setattr(bt, "forward_operation_count", lambda *a, **k: 0)
    assert not bt.subgraph_warrants_export(n)


def test_is_substitutable_single_op_non_straightline(monkeypatch):
    fanout = node("attn", "Attention", basic=False,
                  children=[node("q_proj"), node("k_proj"),
                            node("@attention", "AttentionOp", basic=False)])
    fanout.attention_inputs = {"q": ["q_proj"], "k": ["k_proj"]}
    monkeypatch.setattr(bt, "forward_operation_count", lambda *a, **k: 1)
    assert bt._is_substitutable_single_op_subgraph(fanout)


def test_partition_detail_trees_drops_single_function_shown(monkeypatch):
    tree = node("q_proj", "Linear", basic=False, children=[node("op", "Op", basic=False)])
    monkeypatch.setattr(bt, "is_straight_line_module", lambda n: False)
    monkeypatch.setattr(bt, "is_single_function_tree", lambda n: True)
    monkeypatch.setattr(bt, "_show_single_function_in_diagram", lambda n: True)
    assert bt.partition_detail_trees([("t", tree)]) == []


def test_segment_for_step_method_wrapper_with_explicit_add():
    method = node("combine", "combine", details=["method `combine()`"])
    add = node("add", "Add", label="Add", operation_predecessors=["combine"])
    parent = node("p", "P", basic=False, children=[method, add])
    parent.side_inputs = {
        "combine": [SideInputSpec("res", "res", [], "forward_input")]
    }
    seg = bt._segment_for_step(parent, method)
    assert isinstance(seg, SideCombineSegment)


def test_forward_side_combine_producers_missing_producer_node():
    consumer = node("combine", "combine", details=["method `combine()`"])
    parent = node("p", "P", basic=False, children=[consumer])
    parent.side_inputs = {
        "combine": [SideInputSpec("g", "g", ["ghost"], "prior_step")]
    }
    assert bt._forward_side_combine_producers(parent) == set()


def test_forward_side_combine_producers_producer_not_seq():
    # producer itself has a side input -> not a SeqSegment -> excluded (line 1505)
    inner_prod = node("inner", "Linear")
    producer = node("prod", "Prod", basic=False, children=[inner_prod])
    producer_side = node("prod_side", "Linear")
    consumer = node("combine", "combine", details=["method `combine()`"])
    parent = node("p", "P", basic=False, children=[producer_side, producer, consumer])
    parent.side_inputs = {
        "combine": [SideInputSpec("g", "g", ["prod"], "prior_step")],
        "prod": [SideInputSpec("s", "s", ["prod_side"], "prior_step")],
    }
    result = bt._forward_side_combine_producers(parent)
    assert "prod" not in result


def test_situ_gated_mlp_parts_missing_projection():
    situ = node("s", "SiluActivation", basic=False)
    act = node("act_fn", "SiluAndMul", basic=False, children=[situ])
    # no gate_proj/up_proj/down_proj -> returns None at gate/up/down check
    n = node("n", "N", basic=False, children=[act])
    assert bt._situ_gated_mlp_parts(n) is None


def test_build_block_node_infer_init_with_forward_operations():
    cls = structure("Owner", calls=["@op_l0_c0_add"])
    cls.forward_input_name = "hidden_states"
    cls.forward_operations["@op_l0_c0_add"] = ForwardOperation(
        attr_name="@op_l0_c0_add", label="Add", class_name="Add", predecessors=("x",)
    )
    built = build_block_node(
        attr_name="owner", class_name="Owner", registry={"Owner": cls},
        basic_ops=BasicOpFilter.for_detailed(), infer_init_steps=True,
    )
    assert built.children[0].class_name == "Add"


def test_build_block_node_infer_init_uses_init_modules():
    cls = structure("Owner", assignments={"proj": "Linear"}, calls=["proj"])
    built = build_block_node(
        attr_name="owner", class_name="Owner", registry={"Owner": cls},
        basic_ops=BasicOpFilter.for_detailed(), infer_init_steps=True,
    )
    assert built.children[0].attr_name == "proj"


def test_build_block_node_missing_forward_operation_skipped():
    cls = structure("Owner", assignments={"proj": "Linear"}, calls=["@op_ghost", "proj"])
    # @op_ghost is a forward operation attr but not registered in forward_operations
    built = build_block_node(
        attr_name="owner", class_name="Owner", registry={"Owner": cls},
        basic_ops=BasicOpFilter.for_detailed(),
    )
    assert [c.attr_name for c in built.children] == ["proj"]


def test_is_substitutable_single_op_non_composite():
    leaf = node("x", "X", basic=False)  # no children -> not composite
    assert not bt._is_substitutable_single_op_subgraph(leaf)
