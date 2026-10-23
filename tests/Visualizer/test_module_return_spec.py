###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""A module\'s boundary reports what that module returns.

``DeepseekV4Indexer`` is given hidden states and returns int64 top-k picks::

    return torch.where(invalid, torch.full_like(top_k_indices, -1), top_k_indices)

Its ``@output`` was resolved correctly in class context and then OVERWRITTEN by
the root-less subgraph recursion, which has no class to resolve against and
reported the scorer\'s float32 scores in its place.

What this does NOT yet fix: the module drawn as a subgraph in its PARENT graph is
sized before its own body is walked, so the parent still sees a passthrough of
the input and the compressor downstream still compares and scatters with the
hidden width. Carrying the return outward needs the scorer\'s own boundary to be
right first -- it reports the tensor from before its ``.sum(dim=2)`` -- or the
wrong value simply propagates further.
"""

from __future__ import annotations

import re

import pytest

_MODEL = "deepseek-ai/DeepSeek-V4-Flash"


def _attr(node: dict, key: str) -> str:
    for item in node.get("attrs") or []:
        if item.get("key") == key:
            return str(item.get("value"))
    return ""


@pytest.fixture(scope="module")
def deepseek_nodes(model_graph_nodes):
    return model_graph_nodes(_MODEL)


def _one(nodes, fragment):
    found = [n for n in nodes if fragment in str(n["id"])]
    assert found, fragment
    return found[0]


class TestTheIndexerReturnsItsPicks:
    def test_its_output_is_the_int64_picks(self, deepseek_nodes) -> None:
        """int64, and as wide as the picks -- never the hidden width it was
        handed. The leading axes differ between indexer instances, so they are
        not what this pins."""
        # The indexer's OWN boundary -- the segment before ``/@output`` is the
        # indexer call itself, not one of the frames nested inside it.
        outputs = [
            n
            for n in deepseek_nodes
            if re.search(r"indexer:\d+/@output$", str(n["id"]))
        ]
        assert outputs, "no indexer output boundary"
        for node in outputs:
            shape = _attr(node, "output_shape")
            assert shape.endswith("6] int64"), (node["id"], shape)
