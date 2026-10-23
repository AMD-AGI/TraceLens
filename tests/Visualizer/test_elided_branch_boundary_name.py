###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""A boundary keeps a branch's name only while the branch is drawn.

``create_causal_mask`` is handed ``position_ids``. That tensor is built by one
of two arms, so a phi joins them, and the mask's input tile is named after the
phi. When only one arm remains the phi is folded away and the tile reads
``Expand`` directly. The name ``Mutually exclusive branch`` then describes a
node that is not in the picture. The tile takes the variable the branch
rebound, which is the tensor the reader can see.
"""

from __future__ import annotations

from TraceLens.ModelUtils.ast_analyze import PHI_CLASS, PHI_LABEL
from TraceLens.Visualizer.model_explorer_export.merge import (
    _elide_noop_merges_and_casts,
)


def _edge(source: str) -> dict[str, str]:
    return {
        "sourceNodeId": source,
        "sourceNodeOutputId": "0",
        "targetNodeInputId": "0",
    }


def test_an_input_named_for_a_removed_branch_takes_the_variable() -> None:
    nodes = [
        {"id": "expand", "label": "Expand", "namespace": "", "attrs": []},
        {
            "id": "phi",
            "label": PHI_LABEL,
            "namespace": "",
            "attrs": [
                {"key": "class_name", "value": PHI_CLASS},
                {
                    "key": "details",
                    "value": "merge: position_ids is None; merges: position_ids",
                },
            ],
            "incomingEdges": [_edge("expand")],
        },
        {
            "id": "mask/@input:Mutually exclusive branch",
            "label": PHI_LABEL,
            "namespace": "create_causal_mask",
            "attrs": [{"key": "synthetic", "value": "@input"}],
            "incomingEdges": [_edge("phi")],
        },
    ]
    _elide_noop_merges_and_casts(nodes)
    assert [node["id"] for node in nodes] == [
        "expand",
        "mask/@input:Mutually exclusive branch",
    ]
    tile = nodes[1]
    assert tile["label"] == "position_ids"
    assert tile["incomingEdges"][0]["sourceNodeId"] == "expand"


def test_a_sibling_boundary_names_the_tensor_when_the_branch_recorded_none() -> None:
    """The same Expand already enters rotary as ``position_ids``."""
    nodes = [
        {"id": "expand", "label": "Expand", "namespace": "", "attrs": []},
        {
            "id": "phi",
            "label": PHI_LABEL,
            "namespace": "",
            "attrs": [
                {"key": "class_name", "value": PHI_CLASS},
                {"key": "details", "value": "merge: position_ids is None"},
            ],
            "incomingEdges": [_edge("expand")],
        },
        {
            "id": "rotary/@input:position_ids",
            "label": "position_ids",
            "namespace": "rotary_emb",
            "attrs": [{"key": "synthetic", "value": "@input"}],
            "incomingEdges": [_edge("expand")],
        },
        {
            "id": "mask/@input:Mutually exclusive branch",
            "label": PHI_LABEL,
            "namespace": "create_causal_mask",
            "attrs": [{"key": "synthetic", "value": "@input"}],
            "incomingEdges": [_edge("phi")],
        },
    ]
    _elide_noop_merges_and_casts(nodes)
    mask = next(node for node in nodes if node["id"].startswith("mask/"))
    assert mask["label"] == "position_ids"
    assert mask["incomingEdges"][0]["sourceNodeId"] == "expand"


def test_a_branch_that_is_still_drawn_keeps_its_name() -> None:
    nodes = [
        {"id": "add", "label": "Add", "namespace": "create_causal_mask", "attrs": []},
        {
            "id": "inner_expand",
            "label": "Expand",
            "namespace": "create_causal_mask",
            "attrs": [],
        },
        {
            "id": "phi",
            "label": PHI_LABEL,
            "namespace": "create_causal_mask",
            "attrs": [
                {"key": "class_name", "value": PHI_CLASS},
                {"key": "details", "value": "merge: not use_vmap; merges: mask"},
            ],
            "incomingEdges": [_edge("add"), _edge("inner_expand")],
        },
        {
            "id": "create_causal_mask/@output:Mutually exclusive branch",
            "label": PHI_LABEL,
            "namespace": "create_causal_mask",
            "attrs": [{"key": "synthetic", "value": "@output"}],
            "incomingEdges": [_edge("phi")],
        },
    ]
    _elide_noop_merges_and_casts(nodes)
    assert any(node["id"] == "phi" for node in nodes)
    output = next(
        node
        for node in nodes
        if str(node["id"]).endswith("@output:Mutually exclusive branch")
    )
    assert output["label"] == PHI_LABEL
