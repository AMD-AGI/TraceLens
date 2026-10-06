###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""A method expansion is shaped by what its caller hands it.

MiniMax builds its sparse attention mask with

    block_indices = self.indexer(...)
    ... self.indexer.build_block_mask(block_indices, attention_mask, ...)

and ``build_block_mask`` is inferred as its own section, with no caller context.
Its ``@input`` therefore fell back to the activation default, so the body read
``block_indices`` as the decoder activation ``[B, S, 6144] bfloat16`` where the
tensor is ``[B, index_n_heads, 4] int64`` -- and the comparison, masked fill and
floor-divide that size the mask all inherited it.

The tensor it is handed is the output of the step that produced it, named by the
expansion's first recorded predecessor, and sibling sections are annotated in
order -- so by the time the expansion is inferred, that output is known.

Fixing only the boundary is not enough and was explicitly rejected: it leaves the
two ends of one edge disagreeing. The seed has to reach the inferencer so the
whole body re-reads.
"""

from __future__ import annotations

import re

import pytest

_MODEL = "MiniMaxAI/MiniMax-M3"


def _attr(node: dict, key: str) -> str:
    for item in node.get("attrs") or []:
        if item.get("key") == key:
            return str(item.get("value"))
    return ""


@pytest.fixture(scope="module")
def mask_builder(model_graph_nodes):
    """Nodes of one ``build_block_mask`` expansion."""
    nodes = model_graph_nodes(_MODEL)
    ids = [str(n["id"]) for n in nodes if "indexer@l468" in str(n["id"])]
    if not ids:
        pytest.skip("no build_block_mask expansion in this graph")
    prefix = sorted({re.match(r"(.*indexer@l468[^/]*)", i).group(1) for i in ids})[0]
    return [n for n in nodes if str(n["id"]).startswith(prefix)]


def _labelled(nodes: list[dict], label: str) -> list[dict]:
    return [n for n in nodes if (n.get("label") or "") == label]


class TestTheExpansionReadsWhatItWasHanded:
    def test_the_boundary_is_the_block_index_tensor(self, mask_builder) -> None:
        tiles = _labelled(mask_builder, "block_indices")
        assert tiles, "no block_indices boundary"
        for tile in tiles:
            assert _attr(tile, "output_shape") == "[B, index_n_heads, 4] int64", (
                tile["id"],
                _attr(tile, "output_shape"),
            )

    def test_the_boundary_and_its_mirror_agree(self, mask_builder) -> None:
        """Two halves of one crossing; they carried different shapes."""
        shapes = {
            _attr(tile, "output_shape")
            for tile in _labelled(mask_builder, "block_indices")
        }
        assert len(shapes) == 1, shapes

    def test_nothing_in_the_body_reads_the_decoder_activation(
        self, mask_builder
    ) -> None:
        """``[B, S, 6144]`` is the hidden state, not a block-index tensor."""
        offenders = [
            (str(n["id"]), _attr(n, "output_shape"))
            for n in mask_builder
            if "6144" in _attr(n, "output_shape")
        ]
        assert not offenders, offenders

    def test_the_mask_it_builds_is_four_dimensional(self, mask_builder) -> None:
        """ "We build the full 4D attention mask", per the method's docstring."""
        outputs = [n for n in mask_builder if _attr(n, "synthetic") == "@output"]
        assert outputs
        for node in outputs:
            shape = _attr(node, "output_shape")
            assert shape.count(",") == 3, (node["id"], shape)
