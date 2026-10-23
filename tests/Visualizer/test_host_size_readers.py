###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Everything sized by a tensor shows that it reads it, not just the first.

GLM's vision attention splits all three projections by one set of lengths::

    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    splits = [torch.split(t, lengths.tolist(), dim=2) for t in (q, k, v)]

The analyzer records all three identically -- same predecessors, same
``host_params: cu_seqlens``. But a module's parameter is wired to ONE entry point
inside it (``_build_module_param_entries`` keeps the first consumer), which is
right for an activation: a rebound local reaches its later readers through the op
that rebound it. A size dependency is not rebound and does not travel along the
tensor edge, so the k and v splits were drawn with no sizing input at all --
split, apparently, by nothing.
"""

from __future__ import annotations

import pytest

_MODEL = "zai-org/GLM-5.3-Flash"


def _attr(node: dict, key: str) -> str:
    for item in node.get("attrs") or []:
        if item.get("key") == key:
            return str(item.get("value"))
    return ""


@pytest.fixture(scope="module")
def glm_nodes(model_graph_nodes):
    return model_graph_nodes(_MODEL)


def _vision_splits(nodes: list[dict]) -> list[dict]:
    return [
        node
        for node in nodes
        if (node.get("label") or "") == "Split"
        and "blocks:attn" in str(node["id"])
        and "cu_seqlens" in _attr(node, "details")
    ]


class TestEverySizedOpShowsItsSize:
    def test_all_three_projections_are_split(self, glm_nodes) -> None:
        """q, k and v -- the fan-out is over a literal tuple of all three."""
        assert len(_vision_splits(glm_nodes)) == 3, [
            n["id"] for n in _vision_splits(glm_nodes)
        ]

    def test_each_one_reads_the_lengths_it_is_split_by(self, glm_nodes) -> None:
        by_id = {str(node["id"]): node for node in glm_nodes}
        for split in _vision_splits(glm_nodes):
            sources = [
                str(edge.get("sourceNodeId"))
                for edge in split.get("incomingEdges") or []
            ]
            sized_by = [
                src
                for src in sources
                if "cu_seqlens" in str((by_id.get(src) or {}).get("label") or "")
                or "cu_seqlens" in src
            ]
            assert sized_by, (split["id"], sources)

    def test_the_size_is_not_drawn_as_a_tensor_operand(self, glm_nodes) -> None:
        """It is a size, so it types as a Scalar -- a split takes one tensor."""
        for split in _vision_splits(glm_nodes):
            types = _attr(split, "input_types")
            assert types.count("bfloat16") == 1, (split["id"], types)
            assert "Scalar" in types, (split["id"], types)
