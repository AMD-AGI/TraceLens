###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Regression tests for DeepSeek-V4-Flash compressor/indexer structural integrity.

Expanding ``compressor`` from an opaque leaf into its real class
(``DeepseekV4CSACompressor``) surfaced a batch of I1 dead-node / I2 no-source
integrity warnings. Two distinct wiring bugs were fixed at the extraction/graph-
build source (never by pruning a node or suppressing a warning):

1. A fully inline-flattened, multi-return composite (``cos, sin =
   self.rotary_emb(...)``) collapsed its identity onto a single physical node (the
   last-built internal producer) in ``attr_last_index``. A sibling step reading a
   *specific* return ordinal then fabricated a port tag onto that one node instead
   of docking onto the return slot's own producer, permanently orphaning the other
   slot's real op (``computation_graph._track_attr_index`` / ``_multi_return_slot_key``
   / ``_operation_source_indices``).
2. ``DeepseekV4Indexer.forward`` mutates a host-allocated buffer via slice
   assignment (``new_kv[:, :, ratio:] = chunk_kv[..., self.head_dim:]``). The
   assignment target is a ``Subscript``, not a ``Name``, so the generic
   name-keyed ``_bind`` path silently dropped it: the RHS operand's producer
   (``chunk_kv``'s View, ``chunk_gate``'s Add) was resolved by ``expression()``
   but never wired to anything, and the mutated buffer's own ``var_producer``
   entry was never rebound, so later reads of it kept resolving to the original
   allocation instead of the assignment chain
   (``ast_analyze._ForwardOperationExtractor.statements``, the ``ast.Assign``
   Subscript-target branch, mirroring the pre-existing ``x[idx].copy_(y)``
   in-place-method handler).

These tests rebuild the real model graph and assert the fix holds end to end,
not just at the unit level covered in
``tests/Modeling/test_ast_graph_coverage.py::test_subscript_target_assignment_consumes_rhs_and_rebinds_root``.
"""

from __future__ import annotations

import pytest

from TraceLens.ModelUtils.loader import load_model_spec
from TraceLens.ModelUtils.shape_inference import ShapeInferencer
from TraceLens.Visualizer.model_explorer_export.merge import build_merged_model_graph
from TraceLens.Visualizer.model_explorer_export.type_check import (
    integrity_check_graph_nodes,
)
from TraceLens.Visualizer.model_explorer_export.viewer_page import (
    _graph_without_constants,
)


def _build_graph():
    spec = load_model_spec("deepseek-ai/DeepSeek-V4-Flash", detailed=True)
    return build_merged_model_graph(spec, shape_inferencer=ShapeInferencer(spec))


def test_deepseek_v4_flash_integrity_clean_built_and_render_filtered():
    """No I1 dead-node / I2 no-source warnings survive on either graph."""
    pytest.importorskip("huggingface_hub")
    graph = _build_graph()

    built = integrity_check_graph_nodes(graph["nodes"], label="built")
    assert built == [], built

    filtered_graph = _graph_without_constants(graph)
    filtered = integrity_check_graph_nodes(
        filtered_graph["nodes"], label="render-filtered"
    )
    assert filtered == [], filtered


def test_deepseek_v4_indexer_chunk_kv_and_gate_are_consumed():
    """``chunk_kv``'s View and ``chunk_gate``'s Add feed the ``new_kv``/``new_gate``
    slice-assignment chain instead of dead-ending.

    Regression guard for the Subscript-target-assignment wiring gap: assert each
    node has at least one consumer (mirrors the ``check-dead-nodes`` skill), keyed
    structurally (by label under the indexer namespace), not by a hardcoded line
    number, so the assertion survives incidental line-number churn upstream.
    """
    pytest.importorskip("huggingface_hub")
    graph = _build_graph()
    nodes = graph["nodes"]
    consumed = {
        edge["sourceNodeId"] for node in nodes for edge in node.get("incomingEdges", [])
    }

    indexer_view_and_add = [
        node
        for node in nodes
        if ":indexer:" in node["id"] and node.get("label") in {"View", "Add"}
    ]
    assert indexer_view_and_add, "expected indexer View/Add nodes to still exist"
    dead = [node["id"] for node in indexer_view_and_add if node["id"] not in consumed]
    assert not dead, f"indexer View/Add nodes with no consumer: {dead}"
