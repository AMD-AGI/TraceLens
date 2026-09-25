###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Regression tests for Kimi-K3 ``index_first_axis`` expansion.

``index_first_axis(x, indices): return x[indices]`` used to render as an opaque
``Index first axis`` function tile because the bare advanced-index subscript --
whose index operand is the free function's own tensor *parameter* -- captured as
zero ops. It is now captured as a dedicated ``Index select`` op and inlined at the
call site, keeping the exact base/index wiring the opaque leaf already had.
"""

from __future__ import annotations

import pytest

from TraceLens.ModelUtils.loader import load_model_spec
from TraceLens.ModelUtils.shape_inference import ShapeInferencer
from TraceLens.Visualizer.model_explorer_export.merge import build_merged_model_graph


def _kimi_nodes():
    pytest.importorskip("huggingface_hub")
    spec = load_model_spec("moonshotai/Kimi-K3", detailed=True)
    graph = build_merged_model_graph(spec, shape_inferencer=ShapeInferencer(spec))
    return graph["nodes"]


def test_kimi_index_first_axis_is_no_longer_an_opaque_leaf():
    nodes = _kimi_nodes()
    opaque = [n for n in nodes if n.get("label") == "Index first axis"]
    assert not opaque, [n["id"] for n in opaque]


def test_kimi_index_select_reads_base_and_indices_and_feeds_unsqueeze():
    nodes = _kimi_nodes()
    by_id = {n["id"]: n for n in nodes}
    selects = [n for n in nodes if n.get("label") == "Index select"]
    assert selects, "expected at least one Index select node"

    node = selects[0]
    # Two real tensor operands: the rearranged base and the gathered indices. The
    # ``indices`` operand now flows from the *expanded* ``get_unpad_data`` body
    # (its ``torch.nonzero(...).flatten()`` int64 ``[nnz]`` result) rather than the
    # former opaque ``Get unpad data`` tile.
    edges = {
        edge["metadata"].get("port_label"): edge
        for edge in (node.get("incomingEdges") or [])
    }
    assert by_id.get(edges["x"]["sourceNodeId"], {}).get("label") == "Rearrange"
    indices_src = by_id.get(edges["indices"]["sourceNodeId"], {})
    assert "get_unpad_data" in indices_src.get("id", "")

    # The ``.unsqueeze(0)`` chained on the call result reconnects to the op (no
    # orphaned consumer left behind by the inlining).
    consumers = [
        m.get("label")
        for m in nodes
        if any(
            edge.get("sourceNodeId") == node["id"]
            for edge in (m.get("incomingEdges") or [])
        )
    ]
    assert "Unsqueeze" in consumers
