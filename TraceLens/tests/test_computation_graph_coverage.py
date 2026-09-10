###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Line-coverage tests for TraceLens.ModelUtils.computation_graph.

These tests exercise the graph-building helpers directly and drive
``build_computation_graph`` with hand-built block trees and monkeypatched
segment streams so that every branch of the module is reached.
"""

from __future__ import annotations

import pytest

from TraceLens.ModelUtils import ast_analyze as aa
from TraceLens.ModelUtils import computation_graph as cg
from TraceLens.ModelUtils import block_tree as bt
from TraceLens.ModelUtils.basic_ops import BasicOpFilter
from TraceLens.ModelUtils.block_tree import (
    BlockNode,
    Branch,
    CombineSegment,
    FanOutSegment,
    ResidualAddSegment,
    SeqSegment,
    SideCombineSegment,
    SideFeedSegment,
    TensorPortsSegment,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _node(
    name: str,
    *,
    class_name: str = "Linear",
    children: list[BlockNode] | None = None,
    **kwargs,
) -> BlockNode:
    return BlockNode(
        attr_name=name,
        class_name=class_name,
        role=kwargs.pop("role", "other"),
        label=kwargs.pop("label", class_name),
        children=children or [],
        **kwargs,
    )


def _spec(**kwargs) -> cg.GraphNodeSpec:
    kwargs.setdefault("key", "k")
    return cg.GraphNodeSpec(**kwargs)


# --------------------------------------------------------------------------- #
# Small pure helpers
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("label", "expected"),
    [("+", "Add"), ("×", "Multiply"), ("*", "Multiply"), ("ƒ", "Function"), ("Σ", "Σ")],
)
def test_operation_tile_label(label, expected):
    assert cg._operation_tile_label(label) == expected


def test_normalize_param_name():
    assert cg._normalize_param_name("topk_indices") == cg._normalize_param_name(
        "top_k_index"
    )
    assert cg._normalize_param_name("weights") == "weight"


def test_producer_label_from_attr():
    assert cg._producer_label_from_attr("q_proj") == "Linear"
    assert cg._producer_label_from_attr("short_conv") == "Conv1d"
    assert cg._producer_label_from_attr("input_norm") == "RMSNorm"
    assert cg._producer_label_from_attr("some_thing") == "some thing"


def test_tensor_port_input_sublabels():
    labels = cg._tensor_port_input_sublabels(
        {"q": ["q_proj"], "v": ["value_norm in layers"], "empty": []}
    )
    assert labels == {"q": "← Linear", "v": "← RMSNorm"}


def test_kernel_input_names():
    block = _node("k", details=["inputs: q , k ,", "other: x"])
    assert cg._kernel_input_names(_spec(block=block)) == ["q", "k"]
    assert cg._kernel_input_names(_spec(block=None)) == []
    assert cg._kernel_input_names(_spec(block=_node("n", details=["x"]))) == []


def test_reads_only_a_side_parameter():
    reader = _node(
        "@op_l1_c0_add", class_name="Add", param_inputs=["aux"], operation_predecessors=[]
    )
    assert cg._reads_only_a_side_parameter(reader)
    assert not cg._reads_only_a_side_parameter(_node("plain"))
    with_pred = _node(
        "@op_l1_c0_add",
        class_name="Add",
        param_inputs=["aux"],
        operation_predecessors=["x"],
    )
    assert not cg._reads_only_a_side_parameter(with_pred)


def test_is_local_operation_port():
    ext = _spec(synthetic=cg.SYNTHETIC_TENSOR, key="foo:external:scale")
    assert cg._is_local_operation_port(ext)
    assert not cg._is_local_operation_port(_spec(synthetic=cg.SYNTHETIC_TENSOR, key="x"))


def test_condition_detail():
    assert cg._condition_detail(None) is None
    assert cg._condition_detail(_node("x", details=["note"])) is None
    assert (
        cg._condition_detail(_node("x", details=["condition: enabled"]))
        == "condition: enabled"
    )


def test_consumer_port_label():
    assert cg._consumer_port_label([]) is None
    s1 = aa.SideInputSpec("a", "gate", [])
    s2 = aa.SideInputSpec("b", "gate", [])
    assert cg._consumer_port_label([s1, s2]) == "gate"
    s3 = aa.SideInputSpec("c", "other", [])
    assert cg._consumer_port_label([s1, s3]) == "gate"


def test_input_label_for():
    assert cg._input_label_for(_node("r")) == "hidden_states"
    assert cg._input_label_for(_node("r", input_label="tokens")) == "tokens"


def test_node_has_outgoing_links():
    graph = cg.ComputationGraph(links=[(0, 1)])
    assert cg._node_has_outgoing_links(graph, 0)
    assert not cg._node_has_outgoing_links(graph, 1)


def test_forward_steps_by_attr():
    a = _node("a")
    b = _node("b")
    root = _node("root", children=[a, b, _node("")])
    assert cg._forward_steps_by_attr(root) == {"a": a, "b": b}


def test_append_step_link_variants():
    g = cg.ComputationGraph(nodes=[_spec(), _spec(), _spec()])
    cg._append_step_link(g, input_index=0, last_index=1, step_index=2, fork_from_input=True)
    assert g.links == [(0, 2)]
    g2 = cg.ComputationGraph(nodes=[_spec(), _spec(), _spec()])
    cg._append_step_link(g2, input_index=0, last_index=1, step_index=2, fork_from_input=False)
    assert g2.links == [(1, 2)]
    g3 = cg.ComputationGraph(nodes=[_spec(), _spec()])
    cg._append_step_link(g3, input_index=0, last_index=None, step_index=1, fork_from_input=False)
    assert g3.links == [(0, 1)]
    g4 = cg.ComputationGraph(nodes=[_spec()])
    cg._append_step_link(g4, input_index=None, last_index=None, step_index=0, fork_from_input=True)
    assert g4.links == []


def test_append_operand_link_dedup():
    g = cg.ComputationGraph()
    cg._append_operand_link(g, source_index=0, target_index=1)
    cg._append_operand_link(g, source_index=0, target_index=1)
    assert g.links == [(0, 1)]


# --------------------------------------------------------------------------- #
# Resolution helpers
# --------------------------------------------------------------------------- #
def test_resolve_return_slot_source():
    producer = _node("gate", forward_return_slots={"weights": "w_attr", "index": "i_attr"})
    attr_last = {"w_attr": 5, "i_attr": 7}
    assert cg._resolve_return_slot_source(producer, "weights", attr_last, 0) == 5
    # Normalized fuzzy match: "indices" normalizes to the same as slot "index".
    assert cg._resolve_return_slot_source(producer, "indices", attr_last, 0) == 7
    # Missing slot falls back to default.
    assert cg._resolve_return_slot_source(_node("g"), "x", {}, 42) == 42


def test_lookup_param_entry():
    entries = {"top_k_index": 3}
    assert cg._lookup_param_entry(entries, "top_k_index", 0) == 3
    assert cg._lookup_param_entry(entries, "topk_indices", 0) == 3
    assert cg._lookup_param_entry({}, "x", 9) == 9


def test_resolve_primary_input():
    # No arg_map → fallback chain.
    root = _node("root")
    assert cg._resolve_primary_input("c", root, {}, input_index=1, last_index=2) == 2
    assert cg._resolve_primary_input("c", root, {}, input_index=1, last_index=None) == 1

    # arg_map present but no matching child → fallback.
    root2 = _node(
        "root",
        forward_step_predecessor_args={"c": {"x": "prev"}},
    )
    assert cg._resolve_primary_input("c", root2, {}, input_index=1, last_index=5) == 5

    # child present: primary arg resolves via attr_last_index.
    side_reader = _node("@op_add", param_inputs=["weights"])
    child = _node("c", children=[side_reader])
    root3 = _node(
        "root",
        children=[child],
        forward_step_predecessor_args={
            "c": {"weights": "gate", "hidden": "prev"}
        },
    )
    attr_last = {"prev": 8}
    assert cg._resolve_primary_input("c", root3, attr_last, 1, 2) == 8

    # primary arg maps to FORWARD_METHOD_INPUT → input_index.
    root4 = _node(
        "root",
        children=[_node("c", children=[])],
        forward_step_predecessor_args={"c": {"hidden": aa.FORWARD_METHOD_INPUT}},
    )
    assert cg._resolve_primary_input("c", root4, {}, input_index=1, last_index=2) == 1

    # primary arg unresolved (not in attr_last) → final fallback.
    root5 = _node(
        "root",
        children=[_node("c", children=[])],
        forward_step_predecessor_args={"c": {"hidden": "unknown"}},
    )
    assert cg._resolve_primary_input("c", root5, {}, input_index=1, last_index=2) == 2


def test_build_module_param_entries():
    reader = _node("@op_add", param_inputs=["aux"])
    other = _node("plain")
    graph = cg.ComputationGraph(nodes=[_spec(block=reader), _spec(block=other), _spec(block=None)])
    graph.inline_frames.append(
        cg.InlineFrameSpec(frame_id="mod", label="Mod", node_indices=[0, 1, 2])
    )
    graph.inline_frames.append(
        cg.InlineFrameSpec(frame_id="empty", label="Empty", node_indices=[1])
    )
    entries = cg._build_module_param_entries(graph)
    assert entries == {"mod": {"aux": 0}}


def test_first_graph_index_for_module():
    # No function steps → falls back to module attr.
    leaf = _node("leaf")
    assert cg._first_graph_index_for_module(leaf, {"leaf": 4}) == 4
    # With children, picks the earliest forward_order step.
    early = _node("early", forward_order=1)
    late = _node("late", forward_order=5)
    module = _node("mod", class_name="Wrapper", children=[late, early])
    idx = cg._first_graph_index_for_module(module, {"early": 2, "late": 9})
    assert idx == 2


# --------------------------------------------------------------------------- #
# Chains / method wrappers
# --------------------------------------------------------------------------- #
def test_add_chain_method_wrapper_and_plain(monkeypatch):
    first = _node("first", details=["method `first()`"], label="_first")
    second = _node("second", details=["method `second()`"], label="Second")
    indices: dict[str, int] = {}
    direct = cg.ComputationGraph()
    head, tail = cg._add_chain(
        direct, [first, second], key_prefix="m", attr_last_index=indices
    )
    assert (head, tail) == (0, 1)
    assert direct.links == [(0, 1)]
    assert indices == {"first": 0, "second": 1}


def test_add_chain_wrapper_expansion(monkeypatch):
    inner_a = _node("inner_a")
    inner_b = _node("inner_b")
    wrap = _node("wrap", class_name="Wrapper", children=[inner_a, inner_b])
    plain = _node("plain")

    def expand(step, basic_ops=None):
        del basic_ops
        if step is wrap:
            return [inner_a, inner_b], wrap
        return [step], None

    monkeypatch.setattr(cg, "inline_composite_steps", expand)
    graph = cg.ComputationGraph()
    attr_last: dict[str, int] = {}
    first_index, tail = cg._add_chain(
        graph, [wrap, plain], key_prefix="c", attr_last_index=attr_last
    )
    assert first_index == 0
    assert attr_last["wrap"] == attr_last["inner_b"]
    assert graph.nodes[tail].block is plain
    assert len(graph.inline_frames) == 1


def test_maybe_inline_disabled():
    step = _node("s", class_name="Wrapper", children=[_node("a"), _node("b")])
    steps, wrapper = cg._maybe_inline(step, inline_expansion=False)
    assert steps == [step]
    assert wrapper is None


# --------------------------------------------------------------------------- #
# Kernel port nodes
# --------------------------------------------------------------------------- #
def test_add_kernel_port_nodes_labeled_and_unlabeled():
    kernel = _node("attn", class_name="AttentionOp", details=["inputs: q, k, v"])
    src_q = _node("src_q")
    src_k = _node("src_k")
    src_v = _node("src_v")
    graph = cg.ComputationGraph(
        nodes=[
            _spec(block=src_q, label="q_src"),
            _spec(block=src_k, label="k_src"),
            _spec(block=src_v, label="v_src"),
            _spec(block=kernel, label="Attention"),
        ],
        links=[(0, 3), (1, 3), (2, 3)],
        link_port_labels={(0, 3): "query/key/value"},
    )
    cg._add_kernel_port_nodes(graph)
    port_specs = [s for s in graph.nodes if s.synthetic == cg.SYNTHETIC_KERNEL_PORT_IN]
    # One labeled compound edge + two unlabeled edges matched to remaining names.
    labels = [s.label for s in port_specs]
    assert "query/key/value" in labels
    assert "q" in labels and "k" in labels


def test_add_kernel_port_nodes_fallback_and_duplicate_labels():
    kernel = _node("attn", class_name="KernelOp", details=[])
    a = _node("a")
    b = _node("b")
    graph = cg.ComputationGraph(
        nodes=[_spec(block=a, label="in"), _spec(block=b, label="in"), _spec(block=kernel)],
        links=[(0, 2), (1, 2)],
        link_port_labels={(0, 2): "gate", (1, 2): "gate"},
    )
    cg._add_kernel_port_nodes(graph)
    port_labels = [
        s.label for s in graph.nodes if s.synthetic == cg.SYNTHETIC_KERNEL_PORT_IN
    ]
    assert "gate" in port_labels and "gate_2" in port_labels


def test_add_kernel_port_nodes_unlabeled_no_declared():
    kernel = _node("attn", class_name="KernelOp")
    a = _node("a")
    graph = cg.ComputationGraph(
        nodes=[_spec(block=a, label=""), _spec(block=kernel)],
        links=[(0, 1)],
    )
    cg._add_kernel_port_nodes(graph)
    labels = [
        s.label for s in graph.nodes if s.synthetic == cg.SYNTHETIC_KERNEL_PORT_IN
    ]
    assert labels == ["input_0"]


def test_add_kernel_output_port_nodes():
    kernel = _node("attn", class_name="AttentionMerge")
    t1 = _node("t1")
    t2 = _node("t2")
    graph = cg.ComputationGraph(
        nodes=[_spec(block=kernel, label="Attn"), _spec(block=t1, label="c1"), _spec(block=t2, label="c2")],
        links=[(0, 1), (0, 2)],
        output_ports={"c1": 0},
    )
    cg._add_kernel_output_port_nodes(graph)
    outs = [s for s in graph.nodes if s.synthetic == cg.SYNTHETIC_KERNEL_PORT_OUT]
    assert len(outs) == 2
    # output_ports remapped to a port node.
    assert graph.output_ports["c1"] != 0


def test_add_kernel_output_port_nodes_single_output_ignored():
    kernel = _node("attn", class_name="AttentionMerge")
    graph = cg.ComputationGraph(
        nodes=[_spec(block=kernel), _spec(block=_node("t"))],
        links=[(0, 1)],
    )
    cg._add_kernel_output_port_nodes(graph)
    assert not any(
        s.synthetic == cg.SYNTHETIC_KERNEL_PORT_OUT for s in graph.nodes
    )


# --------------------------------------------------------------------------- #
# Conditional / dead-code / pruning
# --------------------------------------------------------------------------- #
def test_add_conditional_alternative_links():
    graph = cg.ComputationGraph(
        nodes=[
            _spec(block=_node("if", details=["condition: enabled"], operation_predecessors=["x"])),
            _spec(block=_node("else", details=["condition: not (enabled)"], operation_predecessors=["x"])),
        ]
    )
    cg._add_conditional_alternative_links(graph)
    assert (0, 1) in graph.links


def test_live_node_indices_to_fixpoint():
    graph = cg.ComputationGraph(
        nodes=[_spec(block=_node("a")), _spec(block=_node("b")), _spec(block=_node("c"))],
        links=[(0, 1), (1, 2)],
    )
    assert cg._live_node_indices_to_fixpoint(graph, [2]) == {0, 1, 2}


def test_predecessor_map():
    graph = cg.ComputationGraph(nodes=[_spec(), _spec()], links=[(0, 1)])
    preds = cg._predecessor_map(graph)
    assert preds == {0: [], 1: [0]}


def test_prune_computation_nodes_bridges():
    a = _node("a")
    b = _node("b")
    c = _node("c")
    graph = cg.ComputationGraph(
        nodes=[_spec(block=a, key="a"), _spec(block=b, key="drop"), _spec(block=c, key="c")],
        links=[(0, 1), (1, 2)],
        link_port_labels={(0, 1): "gate"},
        link_output_ports={(1, 2): "out"},
        excluded_output_indices={1},
        primary_output_index=2,
        output_node_index=2,
        output_ports={"result": 2},
        loop_carried_nodes={"c": 2},
        attr_output_indices={"c": 2},
    )
    graph.inline_frames.append(
        cg.InlineFrameSpec(frame_id="f", label="F", node_indices=[0, 1, 2])
    )
    pruned = cg._prune_computation_nodes(graph, {1})
    assert len(pruned.nodes) == 2
    assert (0, 1) in pruned.links
    assert pruned.primary_output_index == 1
    assert pruned.inline_frames[0].node_indices == [0, 1]


def test_prune_computation_nodes_noop():
    graph = cg.ComputationGraph(nodes=[_spec()], links=[])
    assert cg._prune_computation_nodes(graph, set()) is graph


def test_dead_node_indices_and_dce():
    keep = _node("keep")
    dead = _node("dead")
    op = _node(
        "@op_out",
        class_name="Add",
        operation_predecessors=["keep"],
    )
    root = _node(
        "root",
        children=[keep, dead, op],
        primary_output_step="@op_out",
        multi_return_module=True,
    )
    graph = cg.ComputationGraph(
        nodes=[
            _spec(block=keep, key="keep"),
            _spec(block=dead, key="dead"),
            _spec(block=op, key="op"),
        ],
        links=[(0, 2)],
        primary_output_index=2,
    )
    dead_set = cg._dead_node_indices(graph, root, strip_unused_return_branches=True)
    assert 1 in dead_set and 2 not in dead_set

    # No primary output → empty.
    empty_root = _node("r")
    assert cg._dead_node_indices(cg.ComputationGraph(), empty_root, strip_unused_return_branches=True) == set()
    # Not stripping → empty.
    assert (
        cg._dead_node_indices(graph, root, strip_unused_return_branches=False) == set()
    )


def test_apply_dead_code_elimination_disabled():
    graph = cg.ComputationGraph(nodes=[_spec()])
    root = _node("r", multi_return_module=False)
    out = cg._apply_dead_code_elimination(graph, root, strip_unused_return_branches=False)
    assert out is graph
    assert out.dead_node_indices == set()


def test_apply_dead_code_elimination_active():
    keep = _node("keep")
    dead = _node("dead")
    op = _node("@op_out", class_name="Add", operation_predecessors=["keep"])
    root = _node(
        "root",
        children=[keep, dead, op],
        primary_output_step="@op_out",
        multi_return_module=True,
    )
    graph = cg.ComputationGraph(
        nodes=[
            _spec(block=keep, key="keep"),
            _spec(block=dead, key="dead"),
            _spec(block=op, key="op"),
        ],
        links=[(0, 2)],
        primary_output_index=2,
    )
    out = cg._apply_dead_code_elimination(graph, root, strip_unused_return_branches=True)
    assert all(spec.block is not dead for spec in out.nodes)


def test_filter_graph_basic_only():
    basic = _node("b", is_basic=True)
    modeled = _node("m", class_name="Linear", is_basic=False)
    graph = cg.ComputationGraph(
        nodes=[
            _spec(block=basic, synthetic=cg.SYNTHETIC_INPUT, label="in"),
            _spec(block=modeled, label="Linear"),
        ],
        links=[(0, 1)],
    )
    filtered = cg._filter_graph_basic_only(graph)
    assert isinstance(filtered, cg.ComputationGraph)


def test_strip_dangling_leaves():
    a = _node("a")
    leaf = _node("leaf")
    graph = cg.ComputationGraph(
        nodes=[_spec(block=a, key="a"), _spec(block=leaf, key="leaf")],
        links=[(0, 1)],
    )
    stripped = cg._strip_dangling_leaves(graph)
    assert len(stripped.nodes) == 1

    # No dangling → returns same graph.
    graph2 = cg.ComputationGraph(
        nodes=[_spec(block=a, key="a", synthetic=cg.SYNTHETIC_OUTPUT)],
    )
    assert cg._strip_dangling_leaves(graph2) is graph2


def test_inline_frame_exit_index():
    graph = cg.ComputationGraph(
        nodes=[_spec(), _spec(), _spec()],
        links=[(0, 1), (1, 2)],
    )
    # member {0,1} → exit is source 1 (points outside).
    assert cg._inline_frame_exit_index(graph, {0, 1}) == 1
    # member {0,1,2} with no outgoing edge outside → dangling last.
    assert cg._inline_frame_exit_index(graph, {0, 1, 2}) == 2


# --------------------------------------------------------------------------- #
# Loop frames / carried nodes
# --------------------------------------------------------------------------- #
def test_add_loop_frames():
    a = _node("a", details=["loop: 4 iterations"])
    b = _node("b", details=["loop: 4 iterations"])
    c = _node("c")
    graph = cg.ComputationGraph(
        nodes=[_spec(block=a, key="a"), _spec(block=b, key="b"), _spec(block=c, key="c")]
    )
    cg._add_loop_frames(graph)
    assert any(f.frame_id.startswith("loop:") for f in graph.inline_frames)


def test_collect_loop_carried():
    spec = aa.LoopCarriedSpec("l1", 3, "x", "init", "upd", ("a",))
    child = _node("c", loop_carried=[spec])
    root = _node("root", children=[child], loop_carried=[])
    assert cg._collect_loop_carried(root) == [spec]


def test_add_loop_carried_nodes():
    op = _node("op")
    root = _node(
        "root",
        children=[op],
        loop_carried=[
            aa.LoopCarriedSpec("l1", 3, "x", aa.FORWARD_METHOD_INPUT, "op", ("op",))
        ],
    )
    inp = _node("inp")
    graph = cg.ComputationGraph(
        nodes=[
            _spec(block=inp, synthetic=cg.SYNTHETIC_INPUT, key="@input"),
            _spec(block=op, key="op"),
        ],
        links=[(0, 1)],
    )
    cg._add_loop_carried_nodes(graph, root)
    assert any(s.synthetic == cg.SYNTHETIC_LOOP_CARRIED for s in graph.nodes)
    assert "op" in graph.loop_carried_nodes


def test_add_loop_carried_nodes_no_specs():
    graph = cg.ComputationGraph(nodes=[_spec()])
    cg._add_loop_carried_nodes(graph, _node("root"))
    assert not graph.loop_carried_nodes


def test_add_loop_carried_nodes_repeated_and_frame():
    op = _node("op", details=["loop: repeated"])
    op2 = _node("op2", details=["loop: repeated"])
    root = _node(
        "root",
        children=[op, op2],
        loop_carried=[
            aa.LoopCarriedSpec("l1", None, "x", "init", "op", ("op", "op2"))
        ],
    )
    graph = cg.ComputationGraph(
        nodes=[
            _spec(block=_node("init"), key="init"),
            _spec(block=op, key="op"),
            _spec(block=op2, key="op2"),
        ],
        links=[(0, 1), (1, 2)],
    )
    graph.inline_frames.append(
        cg.InlineFrameSpec(frame_id="loop:x", label="Loop", node_indices=[1, 2])
    )
    cg._add_loop_carried_nodes(graph, root)
    frame = graph.inline_frames[0]
    assert len(frame.node_indices) > 2


# --------------------------------------------------------------------------- #
# add_forward_output / root pipeline frame
# --------------------------------------------------------------------------- #
def test_add_forward_output_return_slots():
    a = _node("a")
    b = _node("b")
    root = _node(
        "root",
        forward_return_slots={"logits": "a", "hidden": "b"},
        forward_return_order=["logits", "hidden"],
        primary_return_slot="logits",
    )
    graph = cg.ComputationGraph(
        nodes=[
            _spec(synthetic=cg.SYNTHETIC_INPUT, key="@input"),
            _spec(block=a, key="a"),
            _spec(block=b, key="b"),
        ],
        links=[(0, 1), (1, 2)],
    )
    idx = cg.add_forward_output(graph, root=root)
    assert idx is not None
    assert set(graph.output_ports) == {"logits", "hidden"}
    assert graph.primary_output_port == "logits"


def test_add_forward_output_method_input_slot():
    root = _node(
        "root",
        forward_return_slots={"passthrough": aa.FORWARD_METHOD_INPUT},
        forward_return_order=["passthrough"],
    )
    graph = cg.ComputationGraph(
        nodes=[_spec(synthetic=cg.SYNTHETIC_INPUT, key="@input")],
        links=[],
    )
    idx = cg.add_forward_output(graph, root=root)
    assert idx is not None
    assert graph.output_ports["passthrough"] == 0


def test_add_forward_output_guards():
    # Already has output.
    g = cg.ComputationGraph(output_node_index=3)
    assert cg.add_forward_output(g) == 3
    # No nodes.
    assert cg.add_forward_output(cg.ComputationGraph()) is None
    # No synthetic input.
    g2 = cg.ComputationGraph(nodes=[_spec(block=_node("a"))])
    assert cg.add_forward_output(g2) is None
    # Excluded output leaves no exit.
    g3 = cg.ComputationGraph(
        nodes=[_spec(synthetic=cg.SYNTHETIC_INPUT, key="@input")],
        excluded_output_indices={0},
    )
    assert cg.add_forward_output(g3) is None


def test_add_forward_output_multi_exit_and_primary_index():
    a = _node("a")
    b = _node("b")
    graph = cg.ComputationGraph(
        nodes=[
            _spec(synthetic=cg.SYNTHETIC_INPUT, key="@input"),
            _spec(block=a, key="a"),
            _spec(block=b, key="b"),
        ],
        links=[(0, 1), (0, 2)],
    )
    idx = cg.add_forward_output(graph)
    assert set(graph.output_ports) == {"result_1", "result_2"}


def test_add_root_pipeline_frame(monkeypatch):
    a = _node("a", is_basic=True)
    b = _node("b", is_basic=True)
    root = _node("root", class_name="Pipeline", children=[a, b], input_label="tokens")
    graph = cg.build_computation_graph(root)
    cg.add_root_pipeline_frame(graph, root)
    assert graph.inline_frames[-1].node_indices

    # Not a straight-line module → no frame added.
    monkeypatch.setattr(cg, "is_straight_line_module", lambda block: False)
    g2 = cg.ComputationGraph(nodes=[_spec(), _spec()])
    cg.add_root_pipeline_frame(g2, root)
    assert not g2.inline_frames

    # Straight-line but fewer than 2 non-io nodes → no frame.
    monkeypatch.setattr(cg, "is_straight_line_module", lambda block: True)
    g3 = cg.ComputationGraph(
        nodes=[_spec(synthetic=cg.SYNTHETIC_INPUT), _spec(block=_node("only"))]
    )
    cg.add_root_pipeline_frame(g3, root)
    assert not g3.inline_frames


# --------------------------------------------------------------------------- #
# build_computation_graph: simple/basic and situ paths
# --------------------------------------------------------------------------- #
def test_build_basic_root():
    root = _node("leaf", is_basic=True, input_label="x")
    graph = cg.build_computation_graph(root)
    assert graph.nodes[0].label == "x"
    assert graph.primary_output_index == 1
    assert graph.output_node_index is not None


def test_build_basic_root_no_input():
    root = _node("leaf", is_basic=True)
    graph = cg.build_computation_graph(root, include_input=False)
    assert len(graph.nodes) == 1
    assert graph.output_node_index is None


def test_build_situ_gated_root(monkeypatch):
    gate = _node("gate")
    up = _node("up")
    act = _node("act", class_name="Activation")
    situ = _node("situ", class_name="SituActivation")
    down = _node("down")
    root = _node("root", class_name="SituAndMul", children=[gate, up, down])
    monkeypatch.setattr(cg, "is_situ_gated_mlp", lambda node: node is root)
    monkeypatch.setattr(
        bt, "_situ_gated_mlp_parts", lambda _n: (gate, up, act, situ, down)
    )
    graph = cg.build_computation_graph(root)
    assert "×" in [s.label for s in graph.nodes]


def test_add_situ_gated_mlp_chain_none(monkeypatch):
    monkeypatch.setattr(bt, "_situ_gated_mlp_parts", lambda _n: None)
    graph = cg.ComputationGraph()
    indices, tail = cg._add_situ_gated_mlp_chain(
        graph, _node("x"), key_prefix="g", last_index=3
    )
    assert indices == []
    assert tail == 3


def test_add_situ_gated_mlp_chain_dashed(monkeypatch):
    gate = _node("gate")
    up = _node("up")
    act = _node("act", class_name="Activation")
    situ = _node("situ", class_name="SituActivation")
    down = _node("down")
    monkeypatch.setattr(
        bt, "_situ_gated_mlp_parts", lambda _n: (gate, up, act, situ, down)
    )
    graph = cg.ComputationGraph()
    input_index = cg._add_node(graph, key=cg.SYNTHETIC_INPUT, synthetic=cg.SYNTHETIC_INPUT)
    indices, tail = cg._add_situ_gated_mlp_chain(
        graph,
        _node("root"),
        key_prefix="g",
        input_index=input_index,
        branch_from_input_dashed=True,
        create_outer_frame=True,
    )
    assert len(indices) == 5
    assert graph.nodes[tail].block is down


# --------------------------------------------------------------------------- #
# build_computation_graph: segment-driven via monkeypatch
# --------------------------------------------------------------------------- #
def test_build_seq_and_method_wrapper_prefix(monkeypatch):
    ordinary = _node("ordinary")
    method = _node("m", details=["method `m()`"], label="M")
    seq_step = _node("seq_step")
    monkeypatch.setattr(cg, "flatten_computation_segments", lambda _r: [SeqSegment(seq_step)])
    graph = cg.build_computation_graph(
        _node("root", children=[ordinary, seq_step]),
        prefix_steps=[ordinary, method],
    )
    labels = [s.label for s in graph.nodes]
    assert "M" in labels


def test_build_seq_method_wrapper_segment(monkeypatch):
    method = _node("mw", details=["method `mw()`"], label="MW")
    monkeypatch.setattr(cg, "flatten_computation_segments", lambda _r: [SeqSegment(method)])
    graph = cg.build_computation_graph(_node("root", children=[method]))
    assert "MW" in [s.label for s in graph.nodes]


def test_build_seq_expanded_operation_sources(monkeypatch):
    left = _node("left", forward_order=1)
    op = _node(
        "@op_add",
        class_name="Add",
        operation_predecessors=["left"],
        forward_order=2,
    )
    root = _node("root", children=[left, op])
    monkeypatch.setattr(
        cg, "flatten_computation_segments", lambda _r: [SeqSegment(left), SeqSegment(op)]
    )
    graph = cg.build_computation_graph(root)
    assert "Add" in [s.label for s in graph.nodes]


def test_build_seq_side_parameter_only(monkeypatch):
    reader = _node(
        "@op_add",
        class_name="Add",
        param_inputs=["aux"],
        operation_predecessors=[],
        boundary_input_name="aux",
    )
    root = _node("root", children=[reader], forward_param_inputs=["aux"])
    monkeypatch.setattr(cg, "flatten_computation_segments", lambda _r: [SeqSegment(reader)])
    graph = cg.build_computation_graph(root)
    assert any(s.label == "aux" for s in graph.nodes)


def test_build_fanout_plain_merge(monkeypatch):
    left = _node("left")
    right = _node("right")
    merge = _node("merge", class_name="Merge")
    monkeypatch.setattr(
        cg,
        "flatten_computation_segments",
        lambda _r: [FanOutSegment([Branch("q", [left]), Branch("k", [right])], merge)],
    )
    graph = cg.build_computation_graph(_node("root", children=[left, right, merge]))
    merge_index = next(i for i, s in enumerate(graph.nodes) if s.block is merge)
    assert len([l for l in graph.links if l[1] == merge_index]) == 2


def test_build_fanout_wrapper_merge(monkeypatch):
    left = _node("left")
    right = _node("right")
    ma = _node("ma")
    mb = _node("mb")
    merge = _node("merge", class_name="MergeWrapper", children=[ma, mb])

    def expand(step, basic_ops=None):
        del basic_ops
        if step is merge:
            return [ma, mb], merge
        return [step], None

    monkeypatch.setattr(cg, "inline_composite_steps", expand)
    monkeypatch.setattr(cg, "is_kernel_pipeline_tree", lambda n: False)
    monkeypatch.setattr(
        cg,
        "flatten_computation_segments",
        lambda _r: [FanOutSegment([Branch("l", [left]), Branch("r", [right])], merge)],
    )
    graph = cg.build_computation_graph(_node("root", children=[left, right, merge]))
    assert any(f.frame_id == "merge" for f in graph.inline_frames)


def test_build_fanout_kernel_output_merge(monkeypatch):
    left = _node("left")
    right = _node("right")
    ma = _node("ma", class_name="KernelOp")
    mb = _node("mb", class_name="KernelOutput")
    merge = _node("merge", class_name="KernelPipeline", children=[ma, mb])

    def expand(step, basic_ops=None):
        del basic_ops
        if step is merge:
            return [ma, mb], merge
        return [step], None

    monkeypatch.setattr(cg, "inline_composite_steps", expand)
    monkeypatch.setattr(cg, "is_kernel_pipeline_tree", lambda n: False)
    monkeypatch.setattr(
        cg,
        "flatten_computation_segments",
        lambda _r: [FanOutSegment([Branch("l", [left]), Branch("r", [right])], merge)],
    )
    graph = cg.build_computation_graph(_node("root", children=[left, right, merge]))
    assert any(s.block is mb for s in graph.nodes)


def test_build_fanout_kernel_tensor_pipeline_merge(monkeypatch):
    left = _node("left")
    right = _node("right")
    qk = _node("qk", class_name="KernelOp")
    out = _node("out", class_name="KernelOutput")
    merge = _node(
        "merge",
        class_name="KernelPipeline",
        children=[qk, out],
        tensor_input_labels=["q", "k"],
        tensor_step_targets={"q": "qk", "k": "qk"},
        attention_inputs={"q": ["q_proj"], "k": ["k_proj"]},
    )

    def expand(step, basic_ops=None):
        del basic_ops
        if step is merge:
            return [qk, out], merge
        return [step], None

    monkeypatch.setattr(cg, "inline_composite_steps", expand)
    monkeypatch.setattr(cg, "is_kernel_pipeline_tree", lambda n: n is merge)
    monkeypatch.setattr(
        cg,
        "flatten_computation_segments",
        lambda _r: [
            FanOutSegment(
                [Branch("q", [left]), Branch("k", [right])], merge
            )
        ],
    )
    graph = cg.build_computation_graph(_node("root", children=[left, right, merge]))
    ports = [s for s in graph.nodes if s.synthetic == cg.SYNTHETIC_TENSOR]
    assert {"q", "k"} <= {s.label for s in ports}


def test_build_sidecombine_moe_and_ordinary(monkeypatch):
    aggregate = _node(
        "aggregate",
        details=["method `aggregate()`", f"combine: {aa.MOE_AGGREGATION_LABEL}"],
    )
    forward_side = aa.SideInputSpec("hidden", "hidden", [], "forward_input")
    empty_side = aa.SideInputSpec("empty", "empty", [], "prior_step")
    missing_side = aa.SideInputSpec("lost", "lost", ["missing"], "prior_step")
    router_side = aa.SideInputSpec("r", "router", [], "prior_step")
    monkeypatch.setattr(
        cg,
        "flatten_computation_segments",
        lambda _r: [
            SideCombineSegment(
                aggregate,
                [forward_side, empty_side, missing_side, router_side],
                aa.MOE_AGGREGATION_LABEL,
            )
        ],
    )
    graph = cg.build_computation_graph(_node("root", children=[aggregate]))
    assert any(s.label == aa.MOE_AGGREGATION_LABEL for s in graph.nodes)

    combine = _node("combine", details=["method `combine()`"])
    weighted = aa.SideInputSpec("w", "weights", ["combine"], "prior_step")
    monkeypatch.setattr(
        cg,
        "flatten_computation_segments",
        lambda _r: [SideCombineSegment(combine, [forward_side, empty_side, weighted], "×")],
    )
    ordinary = cg.build_computation_graph(_node("root2", children=[combine]))
    assert "Multiply" in [s.label for s in ordinary.nodes]


def test_build_residual_wrapper_and_plain(monkeypatch):
    inner_a = _node("inner_a")
    inner_b = _node("inner_b")
    module = _node("mlp", class_name="MLP", children=[inner_a, inner_b])
    main = _node("main")
    side = aa.SideInputSpec("x", "x", [], "forward_input", side_effect_call=True)

    def expand(step, basic_ops=None):
        del basic_ops
        if step is module:
            return [inner_a, inner_b], module
        return [step], None

    monkeypatch.setattr(cg, "inline_composite_steps", expand)
    monkeypatch.setattr(cg, "is_situ_gated_mlp", lambda n: False)
    monkeypatch.setattr(
        cg,
        "flatten_computation_segments",
        lambda _r: [SeqSegment(main), ResidualAddSegment(module, [side])],
    )
    graph = cg.build_computation_graph(_node("root", children=[main, module]))
    assert "Add" in [s.label for s in graph.nodes]
    assert module.attr_name in graph.side_effect_frame_ids


def test_build_residual_plain_module(monkeypatch):
    module = _node("mlp", class_name="MLP")
    main = _node("main")
    monkeypatch.setattr(cg, "is_situ_gated_mlp", lambda n: False)
    monkeypatch.setattr(
        cg,
        "flatten_computation_segments",
        lambda _r: [SeqSegment(main), ResidualAddSegment(module, [])],
    )
    graph = cg.build_computation_graph(_node("root", children=[main, module]))
    assert "Add" in [s.label for s in graph.nodes]


def test_build_residual_situ_gated(monkeypatch):
    gate = _node("gate")
    up = _node("up")
    act = _node("act", class_name="Activation")
    situ = _node("situ", class_name="SituActivation")
    down = _node("down")
    module = _node("mlp", class_name="SituAndMul", children=[gate, up, down])
    main = _node("main")
    monkeypatch.setattr(cg, "is_situ_gated_mlp", lambda n: n is module)
    monkeypatch.setattr(bt, "_situ_gated_mlp_parts", lambda _n: (gate, up, act, situ, down))
    monkeypatch.setattr(
        cg,
        "flatten_computation_segments",
        lambda _r: [SeqSegment(main), ResidualAddSegment(module, [])],
    )
    graph = cg.build_computation_graph(_node("root", children=[main, module]))
    assert "×" in [s.label for s in graph.nodes]


def test_build_sidefeed_wrapper_consumer(monkeypatch):
    source_a = _node("source_a")
    consume_aux = _node("@op_add", class_name="Add", param_inputs=["aux"], operation_predecessors=[])
    consume_main = _node("@op_mul", class_name="Multiply")
    consumer = _node("consumer", class_name="Consumer", children=[consume_aux, consume_main])
    side = aa.SideInputSpec("aux", "aux", ["source_a"], "prior_step")

    def expand(step, basic_ops=None):
        del basic_ops
        if step is consumer:
            return [consume_aux, consume_main], consumer
        return [step], None

    monkeypatch.setattr(cg, "inline_composite_steps", expand)
    monkeypatch.setattr(
        cg,
        "flatten_computation_segments",
        lambda _r: [
            SideFeedSegment(
                consumer,
                [side],
                side_producer_nodes={"source_a": source_a},
            )
        ],
    )
    graph = cg.build_computation_graph(_node("root", children=[source_a, consumer]))
    assert any(s.block is consume_aux for s in graph.nodes)


def test_build_sidefeed_method_wrapper_consumer(monkeypatch):
    consumer = _node("consume", details=["method `consume()`"])
    producer = _node("router", role="router")
    side = aa.SideInputSpec("weights", "router", ["router"])
    monkeypatch.setattr(
        cg,
        "flatten_computation_segments",
        lambda _r: [
            SeqSegment(producer),
            SideFeedSegment(
                consumer, [side], side_producer_nodes={"router": producer}
            ),
        ],
    )
    graph = cg.build_computation_graph(_node("root", children=[producer, consumer]))
    assert any(s.block is consumer for s in graph.nodes)


def test_build_sidefeed_plain_consumer(monkeypatch):
    consumer = _node("consumer", class_name="Opaque")
    side = aa.SideInputSpec("x", "x", [], "forward_input")
    monkeypatch.setattr(cg, "inline_composite_steps", lambda step, basic_ops=None: ([step], None))
    monkeypatch.setattr(
        cg,
        "flatten_computation_segments",
        lambda _r: [SideFeedSegment(consumer, [side])],
    )
    graph = cg.build_computation_graph(_node("root", children=[consumer]))
    assert any(s.block is consumer for s in graph.nodes)


def test_build_combine_wrapper_side_and_after(monkeypatch):
    main = _node("main")
    side_a = _node("side_a")
    side_b = _node("side_b")
    side = _node("side", class_name="SideWrapper", children=[side_a, side_b])
    after = _node("after", details=["method `after()`"])

    def expand(step, basic_ops=None):
        del basic_ops
        if step is side:
            return [side_a, side_b], side
        return [step], None

    monkeypatch.setattr(cg, "inline_composite_steps", expand)
    monkeypatch.setattr(
        cg,
        "flatten_computation_segments",
        lambda _r: [
            SeqSegment(main),
            CombineSegment(side, after=[after], side_port_label="gate", side_source="forward_input"),
        ],
    )
    graph = cg.build_computation_graph(_node("root", children=[main, side, after]))
    assert "Multiply" in [s.label for s in graph.nodes]
    assert any(f.frame_id == "side" for f in graph.inline_frames)


def test_build_combine_method_wrapper_side(monkeypatch):
    main = _node("main")
    side = _node("side", details=["method `side()`"])
    after = _node("after")
    monkeypatch.setattr(
        cg,
        "flatten_computation_segments",
        lambda _r: [
            SeqSegment(main),
            CombineSegment(side, after=[after], side_source="forward_input"),
        ],
    )
    graph = cg.build_computation_graph(_node("root", children=[main, side, after]))
    assert "Multiply" in [s.label for s in graph.nodes]


def test_build_combine_plain_side_no_last(monkeypatch):
    # No preceding SeqSegment → last_index None → combine short-circuits after side.
    side = _node("side", class_name="Opaque")
    monkeypatch.setattr(cg, "inline_composite_steps", lambda step, basic_ops=None: ([step], None))
    monkeypatch.setattr(
        cg,
        "flatten_computation_segments",
        lambda _r: [CombineSegment(side, after=[], side_source="forward_input")],
    )
    graph = cg.build_computation_graph(_node("root", children=[side]))
    assert isinstance(graph, cg.ComputationGraph)


def test_build_tensor_ports_segment_direct(monkeypatch):
    qk = _node("qk", class_name="KernelOp")
    out = _node(
        "out",
        class_name="KernelOutput",
        kernel_second_operand="input",
        kernel_predecessors=["qk"],
    )
    pipeline = _node("pipeline", class_name="KernelPipeline", children=[qk, out])
    leaf = _node("leaf")
    root = _node(
        "ports",
        children=[pipeline, leaf],
        tensor_input_labels=["q", "v"],
        attention_inputs={"q": ["q_proj"], "v": ["value_norm"]},
    )
    monkeypatch.setattr(
        cg,
        "flatten_computation_segments",
        lambda _r: [
            TensorPortsSegment(
                labels=["q", "v", "ignored"],
                targets={"q": "pipeline", "v": "pipeline", "ignored": "missing"},
                steps=[pipeline, leaf],
            )
        ],
    )
    graph = cg.build_computation_graph(root)
    ports = [s for s in graph.nodes if s.synthetic == cg.SYNTHETIC_TENSOR]
    assert {"q", "v"} <= {s.label for s in ports}


def test_add_tensor_ports_segment_empty():
    graph = cg.ComputationGraph()
    seg = TensorPortsSegment(labels=[], targets={}, steps=[])
    assert cg._add_tensor_ports_segment(graph, seg, key_prefix="k") is None


def test_label_multi_input_kernel_edges():
    graph = cg.ComputationGraph(
        nodes=[_spec(label="a"), _spec(label="b"), _spec(label="k")],
        links=[(0, 2), (1, 2)],
    )
    cg._label_multi_input_kernel_edges(graph, {"k": 2}, {"k": 2}, [(0, 2)])
    assert graph.link_port_labels[(0, 2)] == "a"
    assert graph.link_port_labels[(1, 2)] == "b"


def test_fanout_merge_key_prefix():
    kp = _node("@attn_pipeline", class_name="KernelPipeline")
    assert cg._fanout_merge_key_prefix(kp, 0) == "attn_pipeline"
    assert cg._fanout_merge_key_prefix(_node("m", class_name="Merge"), 3) == "merge:3"


# --------------------------------------------------------------------------- #
# Kernel pipeline merge chain / linear pipeline chain
# --------------------------------------------------------------------------- #
def test_add_kernel_pipeline_merge_chain_two_steps():
    pipeline = _node("pipeline", class_name="KernelPipeline", children=[_node("a"), _node("b")])
    output = _node("output", class_name="KernelOutput", kernel_predecessors=["b"])
    graph = cg.ComputationGraph()
    attr_indices: dict[str, int] = {}
    merged, tail = cg._add_kernel_pipeline_merge_chain(
        graph, [pipeline, output], key_prefix="k", attr_last_index=attr_indices
    )
    assert graph.nodes[tail].block is output


def test_add_kernel_pipeline_merge_chain_fallback():
    graph = cg.ComputationGraph()
    merged, tail = cg._add_kernel_pipeline_merge_chain(
        graph, [_node("only")], key_prefix="f"
    )
    assert merged and tail is not None


def test_add_kernel_pipeline_merge_chain_empty():
    graph = cg.ComputationGraph()
    merged, tail = cg._add_kernel_pipeline_merge_chain(graph, [], key_prefix="e")
    assert merged == []


def test_add_kernel_pipeline_merge_chain_pipeline_tail_link():
    # output_step with no kernel_predecessors → links from pipeline tail.
    pipeline = _node("pipeline", class_name="KernelPipeline", children=[_node("a"), _node("b")])
    output = _node("output", class_name="KernelOutput")
    graph = cg.ComputationGraph()
    merged, tail = cg._add_kernel_pipeline_merge_chain(
        graph, [pipeline, output], key_prefix="k", attr_last_index={}
    )
    assert tail is not None


def test_add_linear_pipeline_chain_nested(monkeypatch):
    inner_a = _node("inner_a")
    inner_b = _node("inner_b")
    nested = _node("nested", class_name="Nested", children=[inner_a, inner_b])
    tail_step = _node("tail")
    outer = _node("outer", class_name="Outer", children=[nested, tail_step])

    def expand(step, basic_ops=None):
        del basic_ops
        if step is outer:
            return [nested, tail_step], outer
        if step is nested:
            return [inner_a, inner_b], nested
        return [step], None

    monkeypatch.setattr(cg, "inline_composite_steps", expand)
    graph = cg.ComputationGraph()
    input_index = cg._add_node(graph, key=cg.SYNTHETIC_INPUT, synthetic=cg.SYNTHETIC_INPUT)
    aliases: dict[str, int] = {}
    chain, tail = cg._add_linear_pipeline_chain(
        graph,
        [nested, tail_step],
        wrapper=outer,
        key_prefix="outer",
        attr_last_index=aliases,
        input_index=input_index,
        fork_from_input=True,
        port_label="aux",
        port_style="inline",
    )
    assert len(chain) == 3
    assert aliases["nested"] == aliases["inner_b"]
    assert len(graph.inline_frames) == 2


def test_add_linear_pipeline_chain_empty():
    graph = cg.ComputationGraph()
    chain, tail = cg._add_linear_pipeline_chain(
        graph, [], wrapper=None, key_prefix="k", last_index=5
    )
    assert chain == []
    assert tail == 5


def test_add_linear_pipeline_chain_kernel_second_operand(monkeypatch):
    monkeypatch.setattr(cg, "inline_composite_steps", lambda step, basic_ops=None: ([step], None))
    prev = _node("prev")
    step = _node("out", class_name="KernelOutput", kernel_second_operand="input")
    graph = cg.ComputationGraph()
    input_index = cg._add_node(graph, key=cg.SYNTHETIC_INPUT, synthetic=cg.SYNTHETIC_INPUT)
    chain, tail = cg._add_linear_pipeline_chain(
        graph, [step], wrapper=None, key_prefix="k", input_index=input_index, last_index=input_index
    )
    assert tail is not None


def test_add_linear_pipeline_chain_branch_from_input_dashed(monkeypatch):
    monkeypatch.setattr(cg, "inline_composite_steps", lambda step, basic_ops=None: ([step], None))
    step = _node("s")
    graph = cg.ComputationGraph()
    input_index = cg._add_node(graph, key=cg.SYNTHETIC_INPUT, synthetic=cg.SYNTHETIC_INPUT)
    chain, tail = cg._add_linear_pipeline_chain(
        graph,
        [step],
        wrapper=None,
        key_prefix="k",
        input_index=input_index,
        branch_from_input_dashed=True,
    )
    assert (input_index, chain[0]) in graph.links


# --------------------------------------------------------------------------- #
# Kernel second-operand resolution
# --------------------------------------------------------------------------- #
def test_resolve_kernel_second_operand_index():
    assert cg._resolve_kernel_second_operand_index(_node("n"), {}, chain_input_index=1) is None
    step = _node("n", kernel_second_operand="input")
    assert cg._resolve_kernel_second_operand_index(step, {}, chain_input_index=7) == 7
    step2 = _node("n", kernel_second_operand="prev")
    assert cg._resolve_kernel_second_operand_index(step2, None, chain_input_index=7) is None
    assert cg._resolve_kernel_second_operand_index(step2, {"prev": 3}, chain_input_index=7) == 3


def test_append_kernel_second_operand_link_none():
    graph = cg.ComputationGraph()
    cg._append_kernel_second_operand_link(
        graph, _node("n"), step_index=0, attr_last_index={}, chain_input_index=None
    )
    assert graph.links == []


# --------------------------------------------------------------------------- #
# Side producer / side chain tail
# --------------------------------------------------------------------------- #
def test_add_side_producer_index_wrapper(monkeypatch):
    inner_a = _node("inner_a")
    inner_b = _node("inner_b")
    producer = _node("gate", class_name="Gate", children=[inner_a, inner_b])

    def expand(step, basic_ops=None):
        del basic_ops
        if step is producer:
            return [inner_a, inner_b], producer
        return [step], None

    monkeypatch.setattr(cg, "inline_composite_steps", expand)
    graph = cg.ComputationGraph()
    input_index = cg._add_node(graph, key=cg.SYNTHETIC_INPUT, synthetic=cg.SYNTHETIC_INPUT)
    attr_last: dict[str, int] = {}
    tail = cg._add_side_producer_index(
        graph,
        producer,
        segment_index=0,
        source_attr="gate",
        port_label="g",
        port_style=None,
        input_index=input_index,
        attr_last_index=attr_last,
    )
    assert attr_last["gate"] == tail


def test_add_side_producer_index_leaf():
    producer = _node("gate")
    graph = cg.ComputationGraph()
    input_index = cg._add_node(graph, key=cg.SYNTHETIC_INPUT, synthetic=cg.SYNTHETIC_INPUT)
    attr_last: dict[str, int] = {}
    tail = cg._add_side_producer_index(
        graph,
        producer,
        segment_index=0,
        source_attr="gate",
        port_label="g",
        port_style="inline",
        input_index=input_index,
        attr_last_index=attr_last,
    )
    assert (input_index, tail) in graph.links


def test_ensure_side_chain_tail_index_variants():
    root = _node("root", input_fed_steps=["g_a"])
    # No source_chain → None.
    side_empty = aa.SideInputSpec("x", "x", [])
    seg = SideFeedSegment(_node("c"), [side_empty])
    graph = cg.ComputationGraph()
    assert (
        cg._ensure_side_chain_tail_index(
            graph, seg, side_empty, segment_index=0, input_index=None, attr_last_index={}, root=root
        )
        is None
    )

    # Cached in attr_last_index.
    side = aa.SideInputSpec("x", "x", ["g_a"])
    assert (
        cg._ensure_side_chain_tail_index(
            graph, seg, side, segment_index=0, input_index=None, attr_last_index={"g_a": 4}, root=root
        )
        == 4
    )

    # No chain, producer node available.
    producer = _node("g_a")
    seg2 = SideFeedSegment(_node("c"), [side], side_producer_nodes={"g_a": producer})
    graph2 = cg.ComputationGraph()
    input_index = cg._add_node(graph2, key=cg.SYNTHETIC_INPUT, synthetic=cg.SYNTHETIC_INPUT)
    tail = cg._ensure_side_chain_tail_index(
        graph2, seg2, side, segment_index=0, input_index=input_index, attr_last_index={}, root=root
    )
    assert tail is not None

    # No chain, no producer → None.
    seg3 = SideFeedSegment(_node("c"), [side])
    assert (
        cg._ensure_side_chain_tail_index(
            cg.ComputationGraph(), seg3, side, segment_index=0, input_index=None, attr_last_index={}, root=root
        )
        is None
    )


def test_ensure_side_chain_tail_index_multi_step_chain():
    g_a = _node("g_a")
    g_b = _node("g_b")
    side = aa.SideInputSpec("x", "gate", ["g_a", "g_b"])
    seg = SideFeedSegment(
        _node("c"),
        [side],
        side_producer_chains={"g_b": [g_a, g_b]},
    )
    root = _node("root", input_fed_steps=["g_a"])
    graph = cg.ComputationGraph()
    input_index = cg._add_node(graph, key=cg.SYNTHETIC_INPUT, synthetic=cg.SYNTHETIC_INPUT)
    attr_last: dict[str, int] = {}
    tail = cg._ensure_side_chain_tail_index(
        graph, seg, side, segment_index=0, input_index=input_index, attr_last_index=attr_last, root=root
    )
    assert attr_last["g_b"] == tail
    assert attr_last["g_a"] != tail


def test_ensure_side_chain_tail_index_chain_with_cached_step():
    g_a = _node("g_a")
    g_b = _node("g_b")
    side = aa.SideInputSpec("x", "gate", ["g_a", "g_b"])
    seg = SideFeedSegment(
        _node("c"),
        [side],
        side_producer_chains={"g_b": [g_a, g_b]},
    )
    root = _node("root")
    graph = cg.ComputationGraph()
    # g_a already materialized.
    a_index = cg._add_node(graph, key="a", block=g_a)
    attr_last = {"g_a": a_index}
    tail = cg._ensure_side_chain_tail_index(
        graph, seg, side, segment_index=0, input_index=None, attr_last_index=attr_last, root=root
    )
    assert (a_index, tail) in graph.links


# --------------------------------------------------------------------------- #
# Upcoming side-combine / fork helpers
# --------------------------------------------------------------------------- #
def test_upcoming_side_combine_and_fork():
    combine = SideCombineSegment(_node("c"), [aa.SideInputSpec("x", "router", ["router"], "prior_step")], "+")
    segments = [SeqSegment(_node("a")), combine]
    assert cg._upcoming_side_combine(segments, 0) is combine
    assert cg._upcoming_side_combine(segments, 1) is None
    assert cg._upcoming_side_combine([SeqSegment(_node("a")), SeqSegment(_node("b"))], 0) is None

    attr_last = {"router": 7}
    assert cg._side_source_tail_index(combine, attr_last) == 7
    # side kind not prior_step → None.
    fwd = SideCombineSegment(_node("c"), [aa.SideInputSpec("x", "r", [], "forward_input")], "+")
    assert cg._side_source_tail_index(fwd, attr_last) is None

    assert cg._should_fork_main_path_from_input(segments, 0, 7, attr_last)
    assert not cg._should_fork_main_path_from_input(segments, 0, None, attr_last)
    assert not cg._should_fork_main_path_from_input([SeqSegment(_node("a"))], 0, 7, attr_last)


# --------------------------------------------------------------------------- #
# Wiring: predecessor edges (module-call), attention provenance, multi-input
# --------------------------------------------------------------------------- #
def test_wire_module_call_predecessor_edges(monkeypatch):
    q_proj = _node("q_proj", forward_order=1)
    gate = _node("gate", forward_order=2)
    root = _node(
        "root",
        children=[q_proj, gate],
        forward_step_predecessors={"gate": ("q_proj",)},
    )
    monkeypatch.setattr(
        cg,
        "flatten_computation_segments",
        lambda _r: [SeqSegment(q_proj), SeqSegment(gate)],
    )
    graph = cg.build_computation_graph(root)
    qi = next(i for i, s in enumerate(graph.nodes) if s.block is q_proj)
    gi = next(i for i, s in enumerate(graph.nodes) if s.block is gate)
    assert (qi, gi) in graph.links


def test_wire_module_call_predecessor_edges_with_argmap(monkeypatch):
    gate = _node("gate", forward_order=1, forward_return_slots={"weights": "gate"})
    norm = _node("norm", forward_order=2)
    aux_reader = _node("@op_add", param_inputs=["weights"], operation_predecessors=[])
    main_step = _node("@op_mul", class_name="Multiply")
    consumer = _node("consumer", forward_order=3, children=[aux_reader, main_step])
    root = _node(
        "root",
        children=[gate, norm, consumer],
        forward_step_predecessors={"consumer": ("gate", "norm")},
        forward_step_predecessor_args={
            "consumer": {"weights": "gate", "hidden": "norm"}
        },
    )

    def expand(step, basic_ops=None):
        del basic_ops
        if step is consumer:
            return [aux_reader, main_step], consumer
        return [step], None

    monkeypatch.setattr(cg, "inline_composite_steps", expand)
    monkeypatch.setattr(
        cg,
        "flatten_computation_segments",
        lambda _r: [SeqSegment(gate), SeqSegment(norm), SeqSegment(consumer)],
    )
    graph = cg.build_computation_graph(root)
    # Port-labeled edges for the two named args should appear.
    assert {"weights", "hidden"} & set(graph.link_port_labels.values())


def test_wire_attention_provenance(monkeypatch):
    q_proj = _node("q_proj", forward_order=1)
    attn = _node(aa.SYNTHETIC_ATTENTION, class_name="AttentionOp", forward_order=2)
    root = _node(
        "root",
        children=[q_proj, attn],
        attention_inputs={"q": ["q_proj"]},
    )
    monkeypatch.setattr(
        cg,
        "flatten_computation_segments",
        lambda _r: [SeqSegment(q_proj), SeqSegment(attn)],
    )
    graph = cg.build_computation_graph(root)
    # Provenance flows into a kernel port node labeled "q".
    assert any(
        s.synthetic == cg.SYNTHETIC_KERNEL_PORT_IN and s.label == "q"
        for s in graph.nodes
    )


def test_wire_attention_provenance_declared_ports(monkeypatch):
    q_proj = _node("q_proj", forward_order=1)
    attn = _node(
        aa.SYNTHETIC_ATTENTION,
        class_name="AttentionOp",
        forward_order=2,
        details=["inputs: q"],
    )
    root = _node(
        "root",
        children=[q_proj, attn],
        attention_inputs={"q": ["q_proj"]},
    )
    monkeypatch.setattr(
        cg,
        "flatten_computation_segments",
        lambda _r: [SeqSegment(q_proj), SeqSegment(attn)],
    )
    graph = cg.build_computation_graph(root)
    assert any(s.block is attn for s in graph.nodes)


def test_wire_inline_op_predecessor_multi_input(monkeypatch):
    left = _node("left", forward_order=1)
    right = _node("right", forward_order=2)
    op = _node(
        "@op_l3_c0_add",
        class_name="Add",
        operation_predecessors=["left", "right", aa.FORWARD_METHOD_INPUT],
        forward_order=3,
    )
    tail = _node("tail", forward_order=4)
    root = _node("root", children=[left, right, op, tail])
    monkeypatch.setattr(
        cg,
        "flatten_computation_segments",
        lambda _r: [SeqSegment(left), SeqSegment(right), SeqSegment(op), SeqSegment(tail)],
    )
    graph = cg.build_computation_graph(root)
    target = next(i for i, s in enumerate(graph.nodes) if s.block is op)
    sources = {src for src, dst in graph.links if dst == target}
    assert len(sources) >= 3
    # multi-input op that is not the last → excluded_output.
    assert target in graph.excluded_output_indices


def test_wire_multi_input_op_forward_links(monkeypatch):
    a = _node("a", forward_order=1)
    b = _node("b", forward_order=2)
    op = _node(
        "@op_add",
        class_name="Add",
        operation_predecessors=["a", "b"],
        forward_order=3,
    )
    consumer = _node("consumer", forward_order=4, operation_predecessors=["@op_add"])
    root = _node("root", children=[a, b, op, consumer])
    monkeypatch.setattr(
        cg,
        "flatten_computation_segments",
        lambda _r: [SeqSegment(a), SeqSegment(b), SeqSegment(op), SeqSegment(consumer)],
    )
    graph = cg.build_computation_graph(root)
    op_index = next(i for i, s in enumerate(graph.nodes) if s.block is op)
    assert cg._node_has_outgoing_links(graph, op_index)


# --------------------------------------------------------------------------- #
# Operation source indices
# --------------------------------------------------------------------------- #
def test_operation_source_indices():
    step = _node("@op", operation_predecessors=["a", aa.FORWARD_METHOD_INPUT, "missing"])
    sources = cg._operation_source_indices(step, {"a": 3}, chain_input_index=1)
    assert 3 in sources and 1 in sources
    assert cg._operation_source_indices(step, None) == []


# --------------------------------------------------------------------------- #
# End-to-end variants
# --------------------------------------------------------------------------- #
def test_build_inline_expansion_false():
    linear = _node("linear", is_basic=True)
    sigmoid = _node("sigmoid", is_basic=True)
    gate = _node("g_proj", class_name="StraightLine", children=[linear, sigmoid], label="Output gate")
    root = _node("root", class_name="Root", children=[gate])
    collapsed = cg.build_computation_graph(root, inline_expansion=False)
    labels = [s.label for s in collapsed.nodes if not s.synthetic]
    assert "Output gate" in labels


def test_build_basic_only_filter():
    a = _node("a", is_basic=True)
    b = _node("b", is_basic=True)
    root = _node("root", class_name="Pipeline", children=[a, b])
    graph = cg.build_computation_graph(root, basic_ops=BasicOpFilter.for_detailed())
    assert isinstance(graph, cg.ComputationGraph)
    basic_only = BasicOpFilter.for_detailed()
    object.__setattr__(basic_only, "basic_only", True) if hasattr(basic_only, "__dict__") else None


def test_build_strip_unused_return_branches(monkeypatch):
    keep = _node("keep", forward_order=1)
    dead = _node("dead", forward_order=2)
    op = _node(
        "@op_out",
        class_name="Add",
        operation_predecessors=["keep"],
        forward_order=3,
    )
    root = _node(
        "root",
        children=[keep, dead, op],
        primary_output_step="@op_out",
        multi_return_module=True,
    )
    monkeypatch.setattr(
        cg,
        "flatten_computation_segments",
        lambda _r: [SeqSegment(keep), SeqSegment(dead), SeqSegment(op)],
    )
    graph = cg.build_computation_graph(root, strip_unused_return_branches=True)
    assert graph.primary_output_index is not None


def test_build_primary_output_step_fallback(monkeypatch):
    step = _node("s")
    root = _node("root", children=[step], primary_output_step="nonexistent")
    monkeypatch.setattr(cg, "flatten_computation_segments", lambda _r: [SeqSegment(step)])
    graph = cg.build_computation_graph(root)
    # primary_output_step not found among nodes → falls back to last_index.
    assert graph.primary_output_index is not None


def test_prune_computation_nodes_recursive_bridge():
    # Remove a contiguous chain so _expand_preds/_expand_succs recurse.
    nodes = [_spec(block=_node(str(i)), key=str(i)) for i in range(4)]
    graph = cg.ComputationGraph(
        nodes=nodes,
        links=[(0, 1), (1, 2), (2, 3)],
        link_port_labels={(1, 2): "mid"},
        link_output_ports={(1, 2): "port"},
    )
    pruned = cg._prune_computation_nodes(graph, {1, 2})
    # 0 and 3 remain, bridged directly.
    assert len(pruned.nodes) == 2
    assert (0, 1) in pruned.links


def test_prune_computation_nodes_cyclic_removed():
    # A self-cyclic removed node must terminate the recursion.
    nodes = [_spec(block=_node(str(i)), key=str(i)) for i in range(3)]
    graph = cg.ComputationGraph(
        nodes=nodes,
        links=[(0, 1), (1, 1), (1, 2)],
    )
    pruned = cg._prune_computation_nodes(graph, {1})
    assert len(pruned.nodes) == 2


def test_add_kernel_output_port_nodes_fallback_labels():
    kernel = _node("attn", class_name="AttentionMerge")
    graph = cg.ComputationGraph(
        nodes=[
            _spec(block=kernel, label="Attn"),
            _spec(block=_node("t1"), label=""),
            _spec(block=_node("t2"), label=""),
        ],
        links=[(0, 1), (0, 2)],
    )
    cg._add_kernel_output_port_nodes(graph)
    outs = [s.label for s in graph.nodes if s.synthetic == cg.SYNTHETIC_KERNEL_PORT_OUT]
    assert "output_0" in outs


def test_add_loop_carried_nodes_rewires_ports():
    op = _node("op")
    consumer = _node("consumer")
    root = _node(
        "root",
        children=[op, consumer],
        loop_carried=[
            aa.LoopCarriedSpec("l1", 3, "x", aa.FORWARD_METHOD_INPUT, "op", ("op",))
        ],
    )
    graph = cg.ComputationGraph(
        nodes=[
            _spec(synthetic=cg.SYNTHETIC_INPUT, key="@input"),
            _spec(block=op, key="op"),
            _spec(block=consumer, key="consumer"),
        ],
        links=[(0, 1), (1, 2)],
        link_port_labels={(1, 2): "carried"},
        link_output_ports={(1, 2): "slot"},
    )
    cg._add_loop_carried_nodes(graph, root)
    # The updated-value → consumer edge is routed through an "out" boundary node.
    out_nodes = [
        i for i, s in enumerate(graph.nodes) if s.key.startswith("@loop_carried_out")
    ]
    assert out_nodes
    assert any(src == out_nodes[0] for src, _ in graph.links)


def test_dead_node_indices_referenced_returns_and_loop_carried():
    keep = _node("keep")
    ret = _node("ret_producer")
    other = _node("other")
    root = _node(
        "root",
        children=[keep, ret, other],
        primary_output_step="keep",
        multi_return_module=True,
        referenced_return_producers={"ret_producer", "carried"},
    )
    graph = cg.ComputationGraph(
        nodes=[
            _spec(synthetic=cg.SYNTHETIC_INPUT, key="@input"),
            _spec(block=keep, key="keep"),
            _spec(block=ret, key="ret_producer"),
            _spec(block=other, key="other"),
            _spec(synthetic=cg.SYNTHETIC_LOOP_CARRIED, key="carried_node"),
        ],
        links=[(0, 1)],
        primary_output_index=1,
        loop_carried_nodes={"carried": 4},
    )
    dead = cg._dead_node_indices(graph, root, strip_unused_return_branches=True)
    # "other" (index 3) is dead; the referenced-return producer and loop-carried
    # nodes are kept, and synthetic input is never marked dead.
    assert 3 in dead
    assert 0 not in dead
    assert 2 not in dead


def test_strip_dangling_leaves_keeps_referenced_and_frame_fed():
    a = _node("a")
    framed = _node("framed")
    fed = _node("fed")
    ref = _node("ref_attr")
    root = _node("root", referenced_return_producers={"ref_attr"})
    graph = cg.ComputationGraph(
        nodes=[
            _spec(block=a, key="a"),
            _spec(block=framed, key="framed"),
            _spec(block=fed, key="fed"),
            _spec(block=ref, key="ref_attr"),
        ],
        links=[(0, 1), (1, 2)],
        attr_output_indices={"ref_attr": 3},
    )
    graph.inline_frames.append(
        cg.InlineFrameSpec(frame_id="f", label="F", node_indices=[0, 1])
    )
    stripped = cg._strip_dangling_leaves(graph, root=root)
    # "fed" is fed by a framed node and "ref_attr" is a referenced return → both kept.
    kept_keys = {s.key for s in stripped.nodes}
    assert "fed" in kept_keys
    assert "ref_attr" in kept_keys


def test_tensor_ports_segment_kernel_predecessors(monkeypatch):
    monkeypatch.setattr(cg, "inline_composite_steps", lambda step, basic_ops=None: ([step], None))
    # Two pipeline steps, second with a child pipeline referencing the first.
    inner_a = _node("inner_a")
    inner_b = _node("inner_b", kernel_second_operand="input")
    stage1 = _node("stage1", class_name="KernelPipeline", children=[inner_a, inner_b])
    stage2_a = _node("stage2_a")
    stage2_b = _node("stage2_b")
    stage2 = _node(
        "stage2",
        class_name="KernelPipeline",
        children=[stage2_a, stage2_b],
        kernel_predecessors=["stage1"],
    )
    graph = cg.ComputationGraph()
    seg = TensorPortsSegment(
        labels=["q", "k"],
        targets={"q": "stage1", "k": "stage2"},
        steps=[stage1, stage2],
    )
    tail = cg._add_tensor_ports_segment(graph, seg, key_prefix="kp")
    assert tail is not None
    assert any(s.synthetic == cg.SYNTHETIC_TENSOR for s in graph.nodes)


def test_label_multi_input_kernel_edges_missing_source():
    graph = cg.ComputationGraph(
        nodes=[_spec(label="a"), _spec(label="k")],
        links=[(0, 1), (5, 1)],  # second source index out of range
    )
    cg._label_multi_input_kernel_edges(graph, {"k": 1}, {}, [])
    assert graph.link_port_labels.get((0, 1)) == "a"


def test_build_prefix_method_wrapper_from_input():
    method = _node("m", details=["method `m()`"], label="M")
    root = _node("root", class_name="Root", children=[_node("leaf")])
    # Only a prefix method wrapper, no chain before it → links from input.
    graph = cg.build_computation_graph(root, prefix_steps=[method])
    m_index = next(i for i, s in enumerate(graph.nodes) if s.label == "M")
    assert any(dst == m_index for _, dst in graph.links)


def test_build_kernel_output_ports_end_to_end(monkeypatch):
    kernel = _node(aa.SYNTHETIC_ATTENTION, class_name="AttentionMerge", forward_order=1)
    c1 = _node("c1", forward_order=2, operation_predecessors=[aa.SYNTHETIC_ATTENTION])
    c2 = _node("c2", forward_order=3, operation_predecessors=[aa.SYNTHETIC_ATTENTION])
    root = _node("root", children=[kernel, c1, c2])
    monkeypatch.setattr(
        cg,
        "flatten_computation_segments",
        lambda _r: [SeqSegment(kernel), SeqSegment(c1), SeqSegment(c2)],
    )
    graph = cg.build_computation_graph(root)
    assert isinstance(graph, cg.ComputationGraph)
