###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""A kernel operand fed from a forward parameter names, and reaches, that parameter.

``attention_interface(self, query_states, key_states, value_states, attention_mask)``
hands the kernel a tensor that no op in this forward produced -- it arrived as a
parameter. Two things went wrong with such an operand:

* it was recorded only when written as a KEYWORD argument, so Kimi's positional
  ``attention_mask`` was dropped and the port it should have named went nameless;
* the per-parameter boundary token (``@method_input:attention_mask``) was
  flattened onto the module's single primary ``@input``, so the port reported
  ``hidden_states`` -- the module's activation -- as the mask's producer, at the
  activation's shape.

Together those rendered sdpa as reading a ``hidden_states [B, S, 7168]`` operand
it never takes.
"""

from __future__ import annotations

import ast

from TraceLens.ModelUtils.ast_analyze import _capture_attention_inputs


def _stmt(source: str) -> ast.stmt:
    return ast.parse(source).body[0]


class TestBoundaryOperandIsRecorded:
    """How the call WRITES an argument says nothing about what the tensor is."""

    def test_a_positional_forward_parameter_is_an_operand(self) -> None:
        captured: dict[str, list[str]] = {}
        _capture_attention_inputs(
            _stmt("out, _ = attention_interface(self, q, k, v, attention_mask)"),
            {"q": ["@op_q"], "k": ["@op_k"], "v": ["@op_v"]},
            captured,
            forward_input_names={"attention_mask", "hidden_states"},
        )
        assert "attention_mask" in captured, captured
        # An empty chain is the marker for "fed from an enclosing scope": no op in
        # THIS forward produced it. That is what the boundary wiring reads.
        assert captured["attention_mask"] == []

    def test_the_same_parameter_by_keyword_is_recorded_identically(self) -> None:
        positional: dict[str, list[str]] = {}
        keyword: dict[str, list[str]] = {}
        names = {"attention_mask"}
        _capture_attention_inputs(
            _stmt("out = attention_interface(self, q, attention_mask)"),
            {"q": ["@op_q"]},
            positional,
            forward_input_names=names,
        )
        _capture_attention_inputs(
            _stmt("out = attention_interface(self, q, attention_mask=attention_mask)"),
            {"q": ["@op_q"]},
            keyword,
            forward_input_names=names,
        )
        assert positional == keyword

    def test_a_local_that_is_not_a_forward_parameter_is_still_skipped(self) -> None:
        """Only a declared parameter is a boundary; an unknown name is not invented."""
        captured: dict[str, list[str]] = {}
        _capture_attention_inputs(
            _stmt("out = attention_interface(self, q, scratch)"),
            {"q": ["@op_q"]},
            captured,
            forward_input_names={"attention_mask"},
        )
        assert "scratch" not in captured


def _node_attr(node: dict, key: str) -> str | None:
    for attr in node.get("attrs", []) or []:
        if attr.get("key") == key:
            return attr.get("value")
    return None


def _ports_of_attention_kernels(nodes: list[dict]) -> list[tuple[str, dict]]:
    """Each attention kernel's operand ports, as ``(kernel id, port node)``."""
    by_id = {node["id"]: node for node in nodes}
    found: list[tuple[str, dict]] = []
    for node in nodes:
        if "@attention" not in node.get("id", ""):
            continue
        for edge in node.get("incomingEdges", []) or []:
            port = by_id.get(str(edge.get("sourceNodeId")))
            if port is not None and _node_attr(port, "synthetic") == "@kernel_port_in":
                found.append((node["id"], port))
    return found


class TestBoundaryOperandIsWired:
    def test_no_attention_kernel_reads_a_port_named_for_the_activation(
        self, kimi_nodes
    ) -> None:
        """``sdpa`` has no ``hidden_states`` parameter, so no port may claim one."""
        offenders = [
            kernel
            for kernel, port in _ports_of_attention_kernels(kimi_nodes)
            if (port.get("label") or "").strip().lower() == "hidden_states"
        ]
        assert offenders == [], offenders

    def test_the_mask_port_is_sourced_from_the_mask_boundary(self, kimi_nodes) -> None:
        by_id = {node["id"]: node for node in kimi_nodes}
        masks = [
            port
            for _kernel, port in _ports_of_attention_kernels(kimi_nodes)
            if (port.get("label") or "").strip().lower() == "attention_mask"
        ]
        assert masks, "no attention_mask operand port found"
        for port in masks:
            sources = [
                by_id.get(str(edge.get("sourceNodeId")))
                for edge in port.get("incomingEdges", []) or []
            ]
            labels = {
                (source.get("label") or "").strip().lower()
                for source in sources
                if source is not None
            }
            # The mask's producer is the mask, not the block's activation.
            assert "hidden_states" not in labels, (port["id"], labels)
            assert labels == {"attention_mask"}, (port["id"], labels)
