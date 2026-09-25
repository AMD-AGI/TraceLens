###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Regression tests for per-op host-ness (``get_unpad_data`` expansion).

``get_unpad_data`` ends in a scalar ``lens.max().item()``, so the whole helper was
tagged ``device: cpu`` and collapsed into one opaque tile -- hiding its
``torch.nonzero`` / ``torch.max`` device work. Host-ness is now a *per-op*
property: the helper expands into its real ops and none is tagged ``device: cpu``
because the only host crossing (``.item()``) collapses to a Python scalar that is
never traced as a visible op. The device gather/nonzero work is shown as the
device ops it is.
"""

from __future__ import annotations

import pytest

from TraceLens.ModelUtils.loader import load_model_spec
from TraceLens.ModelUtils.shape_inference import ShapeInferencer
from TraceLens.Visualizer.model_explorer_export.merge import build_merged_model_graph


def _nodes(model_id: str):
    pytest.importorskip("huggingface_hub")
    spec = load_model_spec(model_id, detailed=True)
    graph = build_merged_model_graph(spec, shape_inferencer=ShapeInferencer(spec))
    return graph["nodes"]


def _is_cpu(node) -> bool:
    return any(
        a.get("key") == "device" and a.get("value") == "cpu"
        for a in node.get("attrs", [])
    )


def test_kimi_get_unpad_data_is_no_longer_an_opaque_leaf():
    nodes = _nodes("moonshotai/Kimi-K3")
    opaque = [n for n in nodes if n.get("label") == "Get unpad data"]
    assert not opaque, [n["id"] for n in opaque]


def test_kimi_get_unpad_data_ops_are_device_not_cpu():
    nodes = _nodes("moonshotai/Kimi-K3")
    unpad = [n for n in nodes if "get_unpad_data" in n.get("id", "")]
    labels = {n.get("label") for n in unpad}
    # The device work is visible...
    assert "Nonzero" in labels
    assert "Max" in labels
    # ...and none of the expanded ops is tagged device: cpu (the only host crossing
    # is the untraced ``.item()`` scalar read).
    cpu_ops = [n.get("label") for n in unpad if _is_cpu(n)]
    assert cpu_ops == [], cpu_ops


def test_kimi_index_select_shape_leads_with_nnz():
    """With ``get_unpad_data`` expanded, ``indices`` carries an int64 ``[nnz]``
    shape, so the ``Index select`` output leads with the symbolic ``nnz`` gathered
    row count instead of the base-passthrough batch dim."""
    nodes = _nodes("moonshotai/Kimi-K3")
    selects = [n for n in nodes if n.get("label") == "Index select"]
    assert selects
    node = selects[0]
    meta = node.get("outputsMetadata") or []
    shape = next(
        (a.get("value") for a in meta[0].get("attrs", []) if a.get("key") == "shape"),
        "",
    )
    assert shape.startswith("[nnz"), shape


def test_no_device_cpu_nodes_remain_after_per_op_hostness():
    """No previously-collapsed host helper leaves a ``device: cpu`` tile behind:
    per-op host-ness expands them all into their device ops (no traced
    materialisation op exists in any of the four models to carry a cpu tag)."""
    for model_id in (
        "moonshotai/Kimi-K3",
        "zai-org/GLM-5.3-Flash",
        "MiniMaxAI/MiniMax-M3",
    ):
        nodes = _nodes(model_id)
        cpu = [n.get("label") for n in nodes if _is_cpu(n)]
        assert cpu == [], (model_id, cpu)
