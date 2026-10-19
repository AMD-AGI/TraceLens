###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""An op that reads its own buffer reads THAT, not what the module was handed.

A rotary embedding opens on::

    inv_freq_expanded = self.inv_freq[None, :, None].expand(...).to(..., device=x.device)

The first statement reads the buffer; ``x`` supplies nothing but a device. But
the module's chain input is docked on whichever step comes FIRST, so this
unsqueeze was handed two operands -- the buffer it reads and the hidden state it
does not. Shape rules read the first operand, and the chain input was arriving
first, so the unsqueeze reported ``[B, S, hidden]`` and every op built from it
(the second insert, the expand, the cast, the ``@`` and the ``cat`` after it)
inherited that rank.

The buffer edge goes first now, and the chain edge is gone: ``x`` is not an
operand of this unsqueeze, which is what the operand-arity check was saying all
along ("takes at most 1 tensor operand, but 2 are wired"). An op is wired the
operands it declares, and this one declares the buffer.

With the fabricated edge gone the whole ``inv_freq`` layout chain is a pure
function of a registered buffer, so the constant tag propagates along it and the
render filter drops it the way it drops every weight -- the chain stays in the
JSON, and the ``@`` that reads it keeps its other operand. That is the same
bargain every ``nn.Linear`` already makes; it was only drawn before because the
fabricated edge made it look like it read an activation. Nothing is orphaned:
I2/I3 are clean on the render-filtered graph.
"""

from __future__ import annotations

import pytest

from TraceLens.ModelUtils.loader import load_model_spec
from TraceLens.ModelUtils.shape_inference import ShapeInferencer
from TraceLens.Visualizer.model_explorer_export.merge import build_merged_model_graph

from tests.model_pins import pin_for

MODEL = "MiniMaxAI/MiniMax-M3"


@pytest.fixture(scope="module")
def nodes() -> list[dict]:
    pytest.importorskip("huggingface_hub")
    pin = pin_for(MODEL)
    spec = load_model_spec(MODEL, detailed=True, revision=pin.revision if pin else None)
    graph = build_merged_model_graph(spec, shape_inferencer=ShapeInferencer(spec))
    return graph["nodes"]


def _attr(node: dict, key: str) -> str | None:
    for item in node.get("attrs", []) or []:
        if item.get("key") == key:
            return str(item.get("value"))
    return None


def _rotary_first_unsqueeze(nodes: list[dict]) -> dict:
    found = [
        node
        for node in nodes
        if "rotary_emb" in str(node.get("id", ""))
        and str(node.get("label")) == "Unsqueeze"
        and "_unsqueeze:" in str(node.get("id", ""))
    ]
    assert found, "expected the rotary embedding's first axis insert"
    return found[0]


class TestTheBufferIsReadFirst:
    def test_its_first_operand_is_the_buffer(self, nodes: list[dict]) -> None:
        by_id = {str(n["id"]): n for n in nodes}
        node = _rotary_first_unsqueeze(nodes)
        edges = node.get("incomingEdges", []) or []
        assert edges, node["id"]
        first = by_id.get(str(edges[0]["sourceNodeId"]))
        assert first is not None, edges[0]
        assert _attr(first, "constant") == "true", (
            first.get("id"),
            _attr(first, "constant"),
        )

    def test_it_reports_the_buffer_s_rank_not_the_hidden_state_s(
        self, nodes: list[dict]
    ) -> None:
        shape = _attr(_rotary_first_unsqueeze(nodes), "output_shape") or ""
        # ``inv_freq`` is 1-D, so inserting one axis gives rank 2. The hidden
        # state is rank 3, and inheriting it made this rank 4.
        assert shape.count(",") == 1, shape

    def test_the_buffer_is_its_only_operand(self, nodes: list[dict]) -> None:
        """``x`` supplies a device, not a tensor, so it is not wired here."""
        by_id = {str(n["id"]): n for n in nodes}
        node = _rotary_first_unsqueeze(nodes)
        sources = [
            by_id.get(str(edge["sourceNodeId"]))
            for edge in node.get("incomingEdges", []) or []
        ]
        assert len(sources) == 1, [s and s.get("id") for s in sources]
        assert _attr(sources[0], "constant") == "true"

    def test_dropping_the_constants_orphans_nothing(self, nodes: list[dict]) -> None:
        """The chain goes with the buffer rather than being left sourceless."""
        from TraceLens.Visualizer.model_explorer_export.type_check import (
            integrity_check_graph_nodes,
        )
        from TraceLens.Visualizer.model_explorer_export.viewer_page import (
            _graph_without_constants,
        )

        filtered = _graph_without_constants({"nodes": nodes})["nodes"]
        assert integrity_check_graph_nodes(filtered, label="render-filtered") == []
        assert not [
            node
            for node in filtered
            if "rotary_emb" in str(node.get("id", ""))
            and _attr(node, "constant") == "true"
        ]
