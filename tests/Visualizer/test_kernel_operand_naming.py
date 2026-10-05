###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""A kernel operand is named by the kernel, and drawn once.

``chunk_kda_fwd_intra(q=q, k=k, v=v, gk=g, beta=beta)`` says what it calls each
operand. Keeping only the SET of operand names threw that away, so a port was
named after whatever produced its value -- which for a decomposed Triton stage
is a glyph for the arithmetic it does, unprintable in the viewer.
"""

from __future__ import annotations

import ast

from TraceLens.ModelUtils.kernel_pipeline import _operand_parameter_bindings
from TraceLens.Visualizer.model_explorer_export.merge import (
    _share_one_tile_per_kernel_operand,
)


def _call(source: str) -> ast.Call:
    node = ast.parse(source).body[0]
    assert isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
    return node.value


class TestOperandBindings:
    def test_keywords_bind_each_operand_to_its_parameter(self) -> None:
        call = _call("chunk_kda_fwd_intra(q=q, k=k, v=v, gk=g, beta=beta)")
        bindings = _operand_parameter_bindings(
            call,
            {"q": "l2norm_q", "k": "l2norm_k", "g": "cumsum", "beta": "sigmoid"},
            {},
            set(),
        )
        assert dict(bindings) == {
            "l2norm_q": "q",
            "l2norm_k": "k",
            "cumsum": "gk",
            "sigmoid": "beta",
        }

    def test_the_parameter_name_wins_over_the_variable_name(self) -> None:
        """``gk=g`` means the kernel calls it ``gk``, whatever the caller called it."""
        call = _call("kernel(gk=g)")
        assert dict(_operand_parameter_bindings(call, {"g": "cumsum"}, {}, set())) == {
            "cumsum": "gk"
        }

    def test_a_positional_argument_is_not_guessed_at(self) -> None:
        """Without the callee's signature in hand, a wrong binding beats none."""
        call = _call("kernel(q, k)")
        assert _operand_parameter_bindings(call, {"q": "a", "k": "b"}, {}, set()) == ()

    def test_an_unresolvable_operand_is_skipped(self) -> None:
        call = _call("kernel(q=q, scale=1.0)")
        assert dict(_operand_parameter_bindings(call, {"q": "a"}, {}, set())) == {
            "a": "q"
        }


def _node(node_id, label, namespace, synthetic=None, sources=()):
    node = {
        "id": node_id,
        "label": label,
        "namespace": namespace,
        "incomingEdges": [
            {"sourceNodeId": s, "sourceNodeOutputId": "0"} for s in sources
        ],
    }
    if synthetic:
        node["attrs"] = [{"key": "synthetic", "value": synthetic}]
    return node


class TestOneTilePerOperand:
    def test_one_tensor_read_by_two_kernels_is_one_node(self) -> None:
        nodes = [
            _node("pipe/cumsum", "CumSum", "pipe"),
            _node(
                "pipe/@kernel_in:1:gk", "gk", "pipe", "@kernel_port_in", ["pipe/cumsum"]
            ),
            _node(
                "pipe/@kernel_in:2:gk", "gk", "pipe", "@kernel_port_in", ["pipe/cumsum"]
            ),
            _node("pipe/intra", "intra", "pipe", None, ["pipe/@kernel_in:1:gk"]),
            _node("pipe/delta", "delta", "pipe", None, ["pipe/@kernel_in:2:gk"]),
        ]

        _share_one_tile_per_kernel_operand(nodes)

        ports = [n for n in nodes if "@kernel_in" in n["id"]]
        assert len(ports) == 1, [n["id"] for n in ports]
        kept = ports[0]["id"]
        for consumer in ("pipe/intra", "pipe/delta"):
            node = next(n for n in nodes if n["id"] == consumer)
            assert node["incomingEdges"][0]["sourceNodeId"] == kept

    def test_two_slots_of_one_kernel_stay_two_operands(self) -> None:
        """A kernel binding one tensor twice really does take two operands."""
        nodes = [
            _node("pipe/src", "CumSum", "pipe"),
            _node("pipe/@kernel_in:1:a", "x", "pipe", "@kernel_port_in", ["pipe/src"]),
            _node("pipe/@kernel_in:2:b", "x", "pipe", "@kernel_port_in", ["pipe/src"]),
            _node(
                "pipe/one",
                "one",
                "pipe",
                None,
                ["pipe/@kernel_in:1:a", "pipe/@kernel_in:2:b"],
            ),
        ]

        _share_one_tile_per_kernel_operand(nodes)

        assert len([n for n in nodes if "@kernel_in" in n["id"]]) == 2

    def test_a_port_repeating_the_tile_feeding_it_is_dropped(self) -> None:
        """``v`` docking into a second ``v`` says nothing the first did not."""
        nodes = [
            _node("pipe/v", "v", "pipe", "@tensor"),
            _node("pipe/@kernel_in:1:v", "v", "pipe", "@kernel_port_in", ["pipe/v"]),
            _node("pipe/intra", "intra", "pipe", None, ["pipe/@kernel_in:1:v"]),
        ]

        _share_one_tile_per_kernel_operand(nodes)

        assert [n["id"] for n in nodes] == ["pipe/v", "pipe/intra"]
        consumer = next(n for n in nodes if n["id"] == "pipe/intra")
        assert consumer["incomingEdges"][0]["sourceNodeId"] == "pipe/v"

    def test_a_port_naming_something_else_is_kept(self) -> None:
        nodes = [
            _node("pipe/v", "v", "pipe", "@tensor"),
            _node("pipe/@kernel_in:1:k", "k", "pipe", "@kernel_port_in", ["pipe/v"]),
            _node("pipe/intra", "intra", "pipe", None, ["pipe/@kernel_in:1:k"]),
        ]

        _share_one_tile_per_kernel_operand(nodes)

        assert len(nodes) == 3
