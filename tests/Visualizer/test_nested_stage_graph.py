###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Per-stage labels and leading downsampling nodes, without a checkpoint.

The live ConvNeXt / ResNet graphs are covered separately and skip offline. This
pins the emission itself: each stage container is labeled with its index, and a
mixed-class downsample list is drawn element-by-element ahead of that stage's
repeated body rather than as N× its first class.
"""

from __future__ import annotations

from TraceLens.ModelUtils.basic_ops import BasicOpFilter
from TraceLens.ModelUtils.blocks import RepeatedGroup
from TraceLens.ModelUtils.extract import ArchitectureSpec
from TraceLens.Visualizer.model_explorer_export.merge import (
    _append_nested_repeated_group,
)


def _convnext_stages() -> RepeatedGroup:
    # Children are in discovery order (layers first). Emission must still lead
    # with the downsample pipeline.
    return RepeatedGroup(
        path="convnext.encoder.stages",
        count=4,
        element_class="ConvNextStage",
        label="",
        children=[
            RepeatedGroup(
                path="convnext.encoder.stages.*.layers",
                count=3,
                element_class="ConvNextLayer",
                label="",
                per_index_counts=[3, 3, 27, 3],
            ),
            RepeatedGroup(
                path="convnext.encoder.stages.*.downsampling_layer",
                count=2,
                element_class="ConvNextLayerNorm",
                label="",
                element_classes=("ConvNextLayerNorm", "Conv2d"),
                per_index_counts=[0, 2, 2, 2],
            ),
        ],
    )


def test_stages_are_labeled_and_downsampling_convs_lead_them():
    nodes: list[dict] = []
    attributes: dict[str, dict[str, str]] = {}
    _append_nested_repeated_group(
        nodes,
        spec=ArchitectureSpec(name="convnext", model_type="convnext"),
        group=_convnext_stages(),
        namespace_prefix="encoder",
        id_prefix="encoder",
        basic_ops=BasicOpFilter.for_roofline(),
        previous_exits=["@input"],
        group_node_configs=[],
        group_node_attributes=attributes,
        shape_inferencer=None,
        inline_expansion=True,
    )

    labels = {attrs["label"] for attrs in attributes.values()}
    assert labels == {
        "stage 0 · 3× ConvNextLayer",
        "stage 1 · 3× ConvNextLayer",
        "stage 2 · 27× ConvNextLayer",
        "stage 3 · 3× ConvNextLayer",
    }
    # Stage 0 has an empty downsample list: no leading conv.
    assert not any("s0_" in node["id"] and "Conv2d" in node["id"] for node in nodes)

    def _id(fragment: str) -> str:
        return next(node["id"] for node in nodes if fragment in node["id"])

    conv = _id("s1_downsampling_layer_1_Conv2d")
    norm = _id("s1_downsampling_layer_0_ConvNextLayerNorm")
    # The conv is the ModuleList's second element, so its meta path ends in .1.
    conv_node = next(node for node in nodes if node["id"] == conv)
    meta = {
        item["key"]: item["value"]
        for item in conv_node.get("attrs") or []
        if item.get("key") in {"meta_path", "class_name"}
    }
    assert meta["class_name"] == "Conv2d"
    assert meta["meta_path"].endswith("stages.1.downsampling_layer.1")
    assert conv_node["label"] == "stage 1 · Conv2d"

    # Norm then conv lead the stage, and the conv feeds the stage's N× body.
    body_input = _id("s1_3x_ConvNextLayer/@input")
    sources = [
        edge["sourceNodeId"]
        for edge in next(node for node in nodes if node["id"] == body_input)[
            "incomingEdges"
        ]
    ]
    assert sources == [conv]
    norm_node = next(node for node in nodes if node["id"] == norm)
    assert [edge["sourceNodeId"] for edge in conv_node["incomingEdges"]] == [norm]
    assert norm_node["label"] == "stage 1 · ConvNextLayerNorm"
