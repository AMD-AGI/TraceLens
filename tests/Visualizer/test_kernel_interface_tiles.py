###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""A fused kernel reading several tensors must name each one it reads.

A kernel is drawn as ONE opaque tile precisely because its insides are not
shown, so its interface is the only thing a reader has: which tensor arrives
where, and what comes back. Four labelled arrows and one anonymous one is a
picture that cannot be read.

Measured on Kimi when this was written, each of them an operand wired after
``computation_graph`` had already assigned ports:

* ``chunk_kda_fwd_intra`` was handed five tensors and declared four ports, so
  ``v`` arrived as a bare edge off the tile shared with the rest of the
  pipeline;
* ``flash_attn_varlen_func`` declares ``attention_mask`` among its ``inputs:``
  and then read it off a plain boundary;
* the three ``ShortConvolution`` kernels took ``cu_seqlens`` straight from
  another box's output mirror.

Nothing was asking, because every check that looks at a kernel asked about its
ARITY -- how many operands -- and these all had the right number. What they did
not have was a name on each.

The port pass and the check are deliberately two things: the pass draws the
interface, the check states the invariant, and the check is what keeps the pass
from being quietly lost (these tiles have been lost before).
"""

from __future__ import annotations

from TraceLens.Visualizer.model_explorer_export.merge import (
    _declared_kernel_input_names,
    _port_unnamed_kernel_operands,
)
from TraceLens.Visualizer.model_explorer_export.type_check import (
    _kernel_interface_tile_warnings,
)


def _node(
    node_id,
    label,
    *,
    synthetic=None,
    operation=None,
    details=None,
    constant=False,
    namespace="box",
    sources=(),
):
    attrs = []
    if synthetic:
        attrs.append({"key": "synthetic", "value": synthetic})
    if operation:
        attrs.append({"key": "operation", "value": operation})
    if details:
        attrs.append({"key": "details", "value": details})
    if constant:
        attrs.append({"key": "constant", "value": "true"})
    attrs.append({"key": "output_shape", "value": "[B, S] bfloat16"})
    return {
        "id": node_id,
        "label": label,
        "namespace": namespace,
        "attrs": attrs,
        "incomingEdges": [{"sourceNodeId": s} for s in sources],
    }


def _kernel(sources, details="kernel: k"):
    return _node("kern", "k", operation="gpu_kernel", details=details, sources=sources)


def _port(node_id, label, sources=()):
    return _node(node_id, label, synthetic="@kernel_port_in", sources=sources)


class TestTheCheck:
    def test_an_operand_with_no_port_of_its_own(self) -> None:
        nodes = [
            _node("mask", "attention_mask", synthetic="@input"),
            _port("p:q", "query"),
            _kernel(["p:q", "mask"]),
        ]
        warnings = _kernel_interface_tile_warnings(nodes)
        assert len(warnings) == 1
        assert "attention_mask" in warnings[0]

    def test_every_operand_on_a_port(self) -> None:
        nodes = [_port("p:q", "query"), _port("p:k", "key"), _kernel(["p:q", "p:k"])]
        assert _kernel_interface_tile_warnings(nodes) == []

    def test_a_single_operand_needs_no_port(self) -> None:
        """One unnamed edge into a box is readable; several are not."""
        nodes = [_node("x", "Linear"), _kernel(["x"])]
        assert _kernel_interface_tile_warnings(nodes) == []

    def test_a_learned_weight_is_not_part_of_the_interface(self) -> None:
        nodes = [
            _node("w", "weight", constant=True),
            _port("p:q", "query"),
            _kernel(["p:q", "w"]),
        ]
        assert _kernel_interface_tile_warnings(nodes) == []

    def test_several_readers_with_no_output_tile(self) -> None:
        nodes = [
            _port("p:q", "query"),
            _kernel(["p:q"]),
            _node("a", "Slice", sources=["kern"]),
            _node("b", "Slice", sources=["kern"]),
        ]
        warnings = _kernel_interface_tile_warnings(nodes)
        assert len(warnings) == 1
        assert "publishes to 2 readers" in warnings[0]

    def test_several_readers_each_on_an_output_tile(self) -> None:
        nodes = [
            _port("p:q", "query"),
            _kernel(["p:q"]),
            _node("o1", "out1", synthetic="@kernel_port_out", sources=["kern"]),
            _node("o2", "out2", synthetic="@output", sources=["kern"]),
        ]
        assert _kernel_interface_tile_warnings(nodes) == []

    def test_an_ordinary_op_is_not_a_kernel(self) -> None:
        nodes = [
            _node("a", "A"),
            _node("b", "B"),
            _node("add", "Add", operation="torch_functional", sources=["a", "b"]),
        ]
        assert _kernel_interface_tile_warnings(nodes) == []

    def test_an_empty_graph(self) -> None:
        assert _kernel_interface_tile_warnings([]) == []


class TestThePassThatDrawsThem:
    def test_the_unported_operand_gets_a_port(self) -> None:
        nodes = [
            _node("mask", "attention_mask", synthetic="@input", namespace="other"),
            _port("p:q", "query"),
            _kernel(["p:q", "mask"], details="kernel: k; inputs: query,attention_mask"),
        ]
        _port_unnamed_kernel_operands(nodes)
        assert _kernel_interface_tile_warnings(nodes) == []
        added = [n for n in nodes if n["id"].endswith(":@kernel_in:attention_mask")]
        assert len(added) == 1
        assert added[0]["label"] == "attention_mask"

    def test_the_port_takes_the_kernels_own_remaining_name(self) -> None:
        nodes = [
            _node("src", "Rearrange", namespace="other"),
            _port("p:q", "query"),
            _kernel(["p:q", "src"], details="kernel: k; inputs: query,value"),
        ]
        _port_unnamed_kernel_operands(nodes)
        added = [n for n in nodes if ":@kernel_in:" in n["id"] and n["id"] != "p:q"]
        assert [n["label"] for n in added] == ["value"]

    def test_the_producers_name_when_the_kernel_declares_none(self) -> None:
        nodes = [
            _node("src", "cu_seqlens", namespace="other"),
            _port("p:x", "x"),
            _kernel(["p:x", "src"]),
        ]
        _port_unnamed_kernel_operands(nodes)
        added = [n for n in nodes if ":@kernel_in:" in n["id"] and n["id"] != "p:x"]
        assert [n["label"] for n in added] == ["cu_seqlens"]

    def test_one_tensor_read_by_several_kernels_is_one_port(self) -> None:
        """The convention ``_share_one_tile_per_kernel_operand`` already draws."""
        nodes = [
            _node("src", "cu_seqlens", namespace="other"),
            _port("p:a", "a"),
            _port("p:b", "b"),
            _node("k1", "k", operation="gpu_kernel", sources=["p:a", "src"]),
            _node("k2", "k", operation="gpu_kernel", sources=["p:b", "src"]),
        ]
        _port_unnamed_kernel_operands(nodes)
        added = [
            n
            for n in nodes
            if ":@kernel_in:" in n["id"] and n["id"] not in {"p:a", "p:b"}
        ]
        assert len(added) == 1
        assert _kernel_interface_tile_warnings(nodes) == []

    def test_a_tile_the_port_duplicates_is_folded_away(self) -> None:
        """Two tiles of one name in one box; the port keeps the name."""
        nodes = [
            _node("up", "Rearrange", namespace="box"),
            _node("v", "v", synthetic="@tensor", namespace="box", sources=["up"]),
            _port("p:q", "q"),
            _kernel(["p:q", "v"]),
        ]
        _port_unnamed_kernel_operands(nodes)
        assert [n["id"] for n in nodes if n["id"] == "v"] == []
        port = next(n for n in nodes if n["id"].endswith(":@kernel_in:v"))
        assert port["label"] == "v"
        assert port["incomingEdges"][0]["sourceNodeId"] == "up"

    def test_a_tile_several_consumers_share_is_kept(self) -> None:
        nodes = [
            _node("up", "Rearrange", namespace="box"),
            _node("v", "v", synthetic="@tensor", namespace="box", sources=["up"]),
            _node("other", "Slice", namespace="box", sources=["v"]),
            _port("p:q", "q"),
            _kernel(["p:q", "v"]),
        ]
        _port_unnamed_kernel_operands(nodes)
        assert any(n["id"] == "v" for n in nodes)

    def test_a_kernel_already_fully_ported_is_untouched(self) -> None:
        nodes = [_port("p:q", "query"), _port("p:k", "key"), _kernel(["p:q", "p:k"])]
        before = len(nodes)
        _port_unnamed_kernel_operands(nodes)
        assert len(nodes) == before


class TestTheNamesAKernelDeclares:
    def test_an_inputs_list(self) -> None:
        node = _node("k", "k", details="kernel: f; inputs: query,key,value")
        assert _declared_kernel_input_names(node) == ["query", "key", "value"]

    def test_operand_pairs(self) -> None:
        node = _node("k", "k", details="kernel: f; operand: stage_a=q, stage_b=gk")
        assert _declared_kernel_input_names(node) == ["q", "gk"]

    def test_a_kernel_that_declares_nothing(self) -> None:
        assert _declared_kernel_input_names(_node("k", "k", details="kernel: f")) == []
        assert _declared_kernel_input_names(_node("k", "k")) == []
