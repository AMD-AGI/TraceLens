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


def test_expanded_rope_single_tensor_ops_read_one_operand():
    """A ``repeat_interleave``/``unsqueeze`` inside an expanded ``apply_rotary_pos_emb``
    frame reads exactly its one real tensor input.

    Regression guard for the Task-G composite-expansion wiring: a positional
    free-function inlined inside an expanded submodule composite
    (``compressor``/``indexer``'s ``apply_rotary_pos_emb``) reassigns its params
    (``cos = cos.repeat_interleave(...).unsqueeze(...)``) and is fed by a real
    multi-return producer (``cos, sin = self.rotary_emb(...)``) rather than a
    boundary alias. The producer-arg wiring over-attached side operands (the
    other slot, the ``x`` primary, the enclosing module's ``hidden_states``) onto
    the frame's single-tensor first ops. Assert structurally (by label under an
    ``apply_rotary_pos_emb`` namespace) that each such op has exactly one incoming
    tensor edge, mirroring how the same op wires at the top-level Attention call
    site. Keyed by op label, not by line number / class name.
    """
    pytest.importorskip("huggingface_hub")
    graph = _build_graph()
    nodes = graph["nodes"]

    offenders: list[tuple[str, int]] = []
    for node in nodes:
        if "apply_rotary_pos_emb" not in node["id"]:
            continue
        if node.get("label") not in {"Repeat interleave", "Unsqueeze"}:
            continue
        incoming = node.get("incomingEdges", []) or []
        if len(incoming) != 1:
            offenders.append((node["id"], len(incoming)))
    assert not offenders, (
        "expanded-rope single-tensor ops must read exactly one operand; "
        f"over-attached: {offenders}"
    )


def test_repeated_rope_instances_source_own_return_slot():
    """Two calls of the same multi-return submodule in one forward
    (``rotary_emb`` at distinct source lines inside the indexer) each feed their
    own call site.

    Regression guard for the ``_resolve_return_slot_source`` collision: both
    inline-expanded ``rotary_emb`` instances share identical internal op
    attr_names, so a flat ``attr_last_index`` lookup by slot name collapsed both
    onto whichever instance built last -- the first indexer rope frame then read
    the *second* frame's ``cos``. Assert the first ``Repeat interleave`` of each
    distinct indexer ``apply_rotary_pos_emb`` frame sources from a *different*
    producer chain (per-instance disambiguation), keyed structurally.
    """
    pytest.importorskip("huggingface_hub")
    graph = _build_graph()
    nodes = graph["nodes"]
    by_id = {node["id"]: node for node in nodes}

    def _tensor_source(node):
        # Resolve one hop through the frame's @input tile to the real producer.
        incoming = node.get("incomingEdges", []) or []
        assert len(incoming) == 1, node["id"]
        src_id = incoming[0]["sourceNodeId"]
        tile = by_id.get(src_id)
        if tile is not None and "/@input" in src_id:
            tin = tile.get("incomingEdges", []) or []
            if tin:
                return tin[0]["sourceNodeId"]
        return src_id

    # First repeat_interleave (cos slot) of each distinct indexer rope frame.
    first_ri: dict[str, str] = {}
    for node in nodes:
        nid = node["id"]
        if ":indexer:" not in nid or "apply_rotary_pos_emb" not in nid:
            continue
        if node.get("label") != "Repeat interleave":
            continue
        frame = nid.rsplit(":@op", 1)[0]
        first_ri.setdefault(frame, nid)

    assert len(first_ri) >= 2, f"expected >=2 indexer rope frames, got {first_ri}"
    sources = {frame: _tensor_source(by_id[op]) for frame, op in first_ri.items()}
    assert len(set(sources.values())) == len(sources), (
        "distinct rope frames must source cos from their own rotary_emb instance, "
        f"not collapse onto one: {sources}"
    )
