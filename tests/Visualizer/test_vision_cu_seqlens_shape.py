###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""GLM's packed-attention ``cu_seqlens`` counts images, not patches.

``get_vision_cu_seqlens`` documents its return as ``(num_segments + 1,) int32``.
The export reported ``[Pv + 1] bfloat16``: one segment boundary per image patch,
in the activation's floating dtype, for a tensor of integer offsets.

The cause was upstream of the frame. ``grid_thw`` is a model input the caller
hands the tower untouched (``self.visual(pixel_values, grid_thw=image_grid_thw)``)
so nothing computes it, and with no boundary of its own it docked onto the
tower's activation ``@input`` -- the flat image patches ``[Pv, 1176]``. Every
node downstream then inherited that geometry perfectly consistently, which is why
no integrity or type check ever fired on it.
"""

from __future__ import annotations

import pytest

_MODEL = "zai-org/GLM-5.3-Flash"


def _attr(node: dict, key: str) -> str:
    for item in node.get("attrs") or []:
        if item.get("key") == key:
            return str(item.get("value"))
    return ""


def _shape(node: dict) -> str:
    return _attr(node, "output_shape")


@pytest.fixture(scope="module")
def glm_nodes(model_graph_nodes):
    return model_graph_nodes(_MODEL)


def _labelled(nodes: list[dict], label: str) -> list[dict]:
    return [node for node in nodes if (node.get("label") or "") == label]


class TestTheGridIsItsOwnInput:
    def test_the_grid_has_a_model_input_boundary(self, glm_nodes) -> None:
        boundary = [
            node for node in glm_nodes if str(node["id"]) == "@vision_input:grid_thw"
        ]
        assert boundary, "the grid the tower is handed has no boundary of its own"

    def test_the_grid_does_not_read_the_image_patches(self, glm_nodes) -> None:
        """The defect itself: the grid docking onto the tower's activation."""
        for node in _labelled(glm_nodes, "grid_thw"):
            sources = {
                str(edge.get("sourceNodeId"))
                for edge in node.get("incomingEdges") or []
            }
            assert (
                "visual/@input" not in sources
            ), f"{node['id']} reads the image patches as its grid"

    def test_the_grid_is_rows_of_coordinates(self, glm_nodes) -> None:
        """``(num_images_or_videos, 3)`` per the source docstring."""
        for node in _labelled(glm_nodes, "grid_thw"):
            assert _shape(node) == "[Img, 3] int64", (node["id"], _shape(node))


class TestCuSeqlens:
    def test_it_is_one_offset_per_segment_boundary(self, glm_nodes) -> None:
        """``(num_segments + 1,) int32``, per the helper's own docstring.

        The dtype is the ``cumsum(dim=0, dtype=dtype)`` the source asks for,
        where ``dtype`` resolves through ``torch.jit.is_tracing()`` -- False in
        the forward this export documents.
        """
        tiles = _labelled(glm_nodes, "cu_seqlens")
        assert tiles, "no cu_seqlens tile in the graph"
        for node in tiles:
            assert _shape(node) == "[Img + 1] int32", (node["id"], _shape(node))

    def test_no_cast_reports_a_device_as_its_dtype(self, glm_nodes) -> None:
        """``x.to(inputs_embeds.device)`` moves a tensor; it casts nothing.

        Recorded as a target dtype it made a pure device move look like a dtype
        change, which then survived cast elision as a Cast that changes nothing.
        """
        offenders = [
            (str(node["id"]), _attr(node, "details"))
            for node in glm_nodes
            if "dtype: " in _attr(node, "details")
            and ".device" in _attr(node, "details").split("dtype: ", 1)[1]
        ]
        assert not offenders, offenders

    def test_no_node_in_the_chain_counts_patches(self, glm_nodes) -> None:
        """Pv is the patch axis; the grid chain must never report it."""
        chain = [
            node
            for node in glm_nodes
            if "get_vision_cu_seqlens" in str(node["id"])
            and not _attr(node, "constant")
        ]
        assert chain, "the cu_seqlens frame is missing"
        offenders = [
            (str(node["id"]), _shape(node)) for node in chain if "Pv" in _shape(node)
        ]
        assert not offenders, offenders
