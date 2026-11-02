###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""ResNet-50 is the second non-uniform model: stage depths live in Sequentials.

``depths=[3,4,6,3]`` cannot be one uniform stack. Unlike ConvNeXt, the repeated
body is an ``nn.Sequential``, not a ``ModuleList``. Skips when the checkpoint or
torch is unavailable (offline CI).
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from TraceLens.Visualizer.model_explorer_export.ast_build import (  # noqa: E402
    build_model_explorer_payload,
)
from TraceLens.Visualizer.model_explorer_export.roofline_cli import (  # noqa: E402
    build_parser,
    load_architecture_spec_from_args,
)

CHECKPOINT = "microsoft/resnet-50"


@pytest.fixture(scope="module")
def resnet_export():
    try:
        args = build_parser().parse_args([CHECKPOINT])
        spec = load_architecture_spec_from_args(CHECKPOINT, args)
        payload = build_model_explorer_payload(
            spec, include_shapes=True, meta_shapes_checkpoint=CHECKPOINT
        )
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"ResNet graph unavailable (offline / no torch): {exc}")
    if not spec.nested_module_groups:
        pytest.skip("meta module tree unavailable (instantiation failed)")
    graph = payload["graphCollections"][0]["graphs"][0]
    return payload, graph


def test_resnet50_stages_render_with_per_stage_labels(resnet_export):
    payload, graph = resnet_export
    labels = {
        attrs.get("label")
        for attrs in graph["groupNodeAttributes"].values()
        if isinstance(attrs, dict)
    }
    for stage, count in enumerate([3, 4, 6, 3]):
        assert f"stage {stage} · {count}× ResNetBottleNeckLayer" in labels
        assert any(
            f"s{stage}_{count}x_ResNetBottleNeckLayer" in (node.get("namespace") or "")
            for node in graph["nodes"]
        )
    body = payload["tracelensViewer"]["factSheet"]["body"]
    assert "Stages: 4× ResNetStage" in body
    assert "stage 2: 6× ResNetBottleNeckLayer" in body
    # The stem convolution leads the encoder (embeddings), distinct from the stages.
    convs = []
    for node in graph["nodes"]:
        for item in node.get("attrs") or []:
            if item.get("key") == "op_type" and item.get("value") == "Conv2d":
                convs.append(node["id"])
                break
    assert any("embed" in node_id for node_id in convs)
