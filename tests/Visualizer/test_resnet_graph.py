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
    # The stem convolution leads the encoder, and each bottleneck's three convs
    # are nodes — not one opaque Sequential. The stem's shape is spatial.
    convs = []
    for node in graph["nodes"]:
        attrs = {
            item.get("key"): item.get("value")
            for item in node.get("attrs") or []
            if isinstance(item, dict)
        }
        if attrs.get("op_type") == "Conv2d":
            convs.append((node["id"], attrs))
    assert any("embed" in node_id for node_id, _attrs in convs)
    assert any(
        "layer" in node_id and "convolution" in node_id for node_id, _attrs in convs
    )
    stem = next(attrs for node_id, attrs in convs if "embed" in node_id)
    assert stem["output_shape"].startswith("[B, 64, 112, 112]")
    assert "conv_weight=" in (stem.get("details") or "")

    # The stage depth multiplies the bottleneck body. The projection shortcut
    # exists on the first block only, so it is counted once.
    def _repeat(node_id: str) -> str | None:
        node = next(node for node in graph["nodes"] if node["id"] == node_id)
        for item in node.get("attrs") or []:
            if item.get("key") == "repeat_count":
                return item.get("value")
        return None

    body = next(
        node_id
        for node_id, _attrs in convs
        if "s2_6x_" in node_id and "shortcut" not in node_id
    )
    shortcut = next(
        node_id
        for node_id, _attrs in convs
        if "s0_" in node_id and "shortcut" in node_id
    )
    assert _repeat(body) == "6"
    assert _repeat(shortcut) == "1"
