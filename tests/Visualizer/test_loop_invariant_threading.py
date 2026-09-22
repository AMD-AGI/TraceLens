###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Regression tests for loop-invariant decoder-input threading (Task A).

A decoder layer receives tensors handed to every iteration by keyword
(``layer(hidden_states, position_embeddings=..., attention_mask=...)``). The
collapsed repeat group surfaces each as a namespaced ``@input:<param>`` tile deep
in the body; unlike the primary spine input (threaded by the loop-carried
boundary), nothing sources them, so they float. ``merge._thread_loop_invariant_inputs``
reconnects each to its legitimate model-level producer, resolved *structurally* by
the input's own name -- a top-level forward parameter, a value assigned from a
``self.<submodule>(...)`` call, or the primary spine input -- never by a hardcoded
class/parameter name. These tests assert the boundaries end up sourced and that
the tightened I2 no-source check is clean across the built + render-filtered graph.
"""

from __future__ import annotations

import pytest

from TraceLens.Visualizer.model_explorer_export.merge import build_merged_model_graph
from TraceLens.Visualizer.model_explorer_export.type_check import (
    integrity_check_graph_nodes,
)
from TraceLens.ModelUtils.loader import load_model_spec
from TraceLens.ModelUtils.shape_inference import ShapeInferencer


def _build_nodes(model_id: str):
    spec = load_model_spec(model_id, detailed=True)
    graph = build_merged_model_graph(spec, shape_inferencer=ShapeInferencer(spec))
    return graph, {n["id"]: n for n in graph["nodes"]}


def _floating_namespaced_inputs(nodes) -> list[str]:
    """Namespaced ``@input`` / ``@input:<param>`` boundaries with no incoming edge.

    A *top-level* model input (empty namespace, no ``/`` in the id) is a legitimate
    sourceless entry point and is excluded; any namespaced one that floats is the
    defect this pass exists to prevent.
    """
    out = []
    for node in nodes:
        nid = node["id"]
        if "/@input" not in nid:
            continue
        if node.get("incomingEdges"):
            continue
        out.append(nid)
    return out


def test_deepseek_v4_attention_invariant_inputs_are_sourced():
    """The two ``DeepseekV4Attention`` boundaries Task A targets gain a real edge.

    ``position_embeddings`` resolves through a materialized model-scope rotary
    producer; ``attention_mask`` docks onto its top-level model-input parameter
    boundary. Both must have an incoming edge, and no namespaced ``@input`` may float.
    """
    pytest.importorskip("huggingface_hub")
    graph, by_id = _build_nodes("deepseek-ai/DeepSeek-V4-Flash")

    for boundary in (
        "decoder/self_attn/@input:position_embeddings",
        "decoder/self_attn/@input:attention_mask",
    ):
        node = by_id.get(boundary)
        assert node is not None, f"missing boundary {boundary}"
        assert node.get("incomingEdges"), f"{boundary} still floats"

    assert _floating_namespaced_inputs(graph["nodes"]) == []


def test_minimax_m3_variant_loop_invariant_inputs_are_sourced():
    """The general (non-DeepSeek) resolution path is exercised by MiniMax-M3.

    MiniMax's variant decoder loop leaves ``forward_step_predecessor_args`` empty,
    so threading must resolve structurally: ``position_embeddings`` from the
    ``self.rotary_emb(...)`` submodule producer, ``position_ids`` (the rotary
    submodule's own input) from its top-level parameter boundary, and the nested
    indexer's bare ``@input`` mirrored from the enclosing attention boundary. No
    namespaced ``@input`` may float afterwards.
    """
    pytest.importorskip("huggingface_hub")
    graph, _ = _build_nodes("MiniMaxAI/MiniMax-M3")
    assert _floating_namespaced_inputs(graph["nodes"]) == []


@pytest.mark.parametrize(
    "model_id",
    ["deepseek-ai/DeepSeek-V4-Flash", "MiniMaxAI/MiniMax-M3"],
)
def test_loop_invariant_threading_i2_clean(model_id):
    """I2 no-source is clean on the built + render-filtered graph after threading."""
    pytest.importorskip("huggingface_hub")
    from TraceLens.Visualizer.model_explorer_export.viewer_page import (
        _graph_without_constants,
    )

    graph, _ = _build_nodes(model_id)

    built = [w for w in integrity_check_graph_nodes(graph["nodes"], label="built") if "I2" in w]
    assert built == [], built

    rendered = _graph_without_constants(graph)
    filtered = [
        w
        for w in integrity_check_graph_nodes(rendered["nodes"], label="render-filtered")
        if "I2" in w
    ]
    assert filtered == [], filtered
