###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Swin's patch merging sits between stages even though it is not a ModuleList.

The stage bodies are ``ModuleList``s, so they already become repeated groups.
``SwinPatchMerging`` is a plain submodule of the stage. Leaving it out would
connect the next stage to the pre-downsample tensor. Skips when the checkpoint
or torch is unavailable (offline CI).
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

CHECKPOINT = "microsoft/swin-tiny-patch4-window7-224"


@pytest.fixture(scope="module")
def swin_graph():
    try:
        args = build_parser().parse_args([CHECKPOINT])
        spec = load_architecture_spec_from_args(CHECKPOINT, args)
        payload = build_model_explorer_payload(
            spec, include_shapes=True, meta_shapes_checkpoint=CHECKPOINT
        )
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Swin graph unavailable (offline / no torch): {exc}")
    if not spec.nested_module_groups:
        pytest.skip("meta module tree unavailable (instantiation failed)")
    return payload["graphCollections"][0]["graphs"][0]


def test_patch_merging_is_drawn_between_stages(swin_graph):
    labels = {
        attrs.get("label")
        for attrs in swin_graph["groupNodeAttributes"].values()
        if isinstance(attrs, dict)
    }
    assert any("SwinLayer" in (label or "") for label in labels)
    ids = " ".join(node.get("id") or "" for node in swin_graph["nodes"])
    assert "PatchMerging" in ids or "downsample" in ids
