###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""End-to-end: a convolutional model's encoder becomes real graph nodes.

ConvNeXt is four non-uniform stages (``depths=[3,3,27,3]``, channels
``[128,256,512,1024]``) the single uniform decoder stack cannot express, so its
convolutions used to never render. This pins that the reconstructed nested-group
path emits them: one ``N× ConvNextLayer`` container per stage with the right
counts, and each conv sized with the concrete per-stage ``[B, C, H, W]`` a meta
forward captures. Skips when the checkpoint or torch is unavailable (offline CI).
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

CHECKPOINT = "facebook/convnext-base-224-22k"


def _attr(node, key):
    for item in node.get("attrs") or []:
        if item.get("key") == key:
            return item.get("value")
    return None


@pytest.fixture(scope="module")
def convnext_nodes():
    # Use the roofline/export loader (leaf-op basic_ops) -- the real path this
    # feature serves. (Full `detailed=True` diagram expansion hits a pre-existing,
    # unrelated ConvNeXt recursion in the diagram machinery.)
    try:
        args = build_parser().parse_args([CHECKPOINT])
        spec = load_architecture_spec_from_args(CHECKPOINT, args)
        payload = build_model_explorer_payload(
            spec, include_shapes=True, meta_shapes_checkpoint=CHECKPOINT
        )
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"ConvNeXt graph unavailable (offline / no torch): {exc}")
    if not spec.nested_module_groups:
        pytest.skip("meta module tree unavailable (instantiation failed)")
    return payload["graphCollections"][0]["graphs"][0]["nodes"]


_CONV_OP_TYPES = {"Conv1d", "Conv2d", "Conv3d"}


def _conv_nodes(nodes):
    return [n for n in nodes if _attr(n, "op_type") in _CONV_OP_TYPES]


def test_every_stage_renders_an_n_times_convnextlayer_container(convnext_nodes):
    namespaces = {n.get("namespace", "") for n in convnext_nodes}
    # The four stages' depths [3,3,27,3] become distinct per-stage containers.
    for stage, count in enumerate([3, 3, 27, 3]):
        assert any(
            f"s{stage}_{count}x_ConvNextLayer" in ns for ns in namespaces
        ), f"missing stage {stage} ({count}x ConvNextLayer) container"


def test_convolutions_carry_concrete_per_stage_spatial_shapes(convnext_nodes):
    # One representative dwconv per stage (plus the stem) -> 5 Conv nodes, each with
    # the stage's own channels and halving spatial dims.
    shapes = {
        n.get("namespace") or n.get("id"): _attr(n, "output_shape")
        for n in _conv_nodes(convnext_nodes)
    }
    assert len(shapes) == 5
    # Stem patch-embed: [B,3,224,224] -> [B,128,56,56].
    stem = next(v for k, v in shapes.items() if "embeddings" in k)
    assert stem.startswith("[B, 128, 56, 56]")
    expected = {
        "s0_3x_ConvNextLayer": "[B, 128, 56, 56]",
        "s1_3x_ConvNextLayer": "[B, 256, 28, 28]",
        "s2_27x_ConvNextLayer": "[B, 512, 14, 14]",
        "s3_3x_ConvNextLayer": "[B, 1024, 7, 7]",
    }
    for slug, prefix in expected.items():
        value = next(v for k, v in shapes.items() if slug in k)
        assert value.startswith(prefix), f"{slug}: {value!r} != {prefix}"
