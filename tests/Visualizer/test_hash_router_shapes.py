###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The hash router reads its routing table at the table's own shape and dtype.

``DeepseekV4HashRouter`` selects experts from a frozen lookup table::

    self.tid2eid = nn.Buffer(torch.zeros(config.vocab_size, self.top_k, dtype=torch.long))
    indices = self.tid2eid[input_ids.reshape(-1)].long()
    weights = scores.gather(1, indices)

The table rendered as ``[1]`` in the module's working precision, so the lookup
reading it lost both its trailing axis and its integer-ness, and every operand
feeding the final gather was wrong -- while that gather still reached
``[B*S, top_k]``, by a fallback that happens to agree. See
``tests/Modeling/test_buffer_and_index_shapes.py`` for the pieces.
"""

from __future__ import annotations

import pytest

_MODEL = "deepseek-ai/DeepSeek-V4-Flash"


@pytest.fixture(scope="module")
def router_nodes(model_graph_nodes):
    nodes = model_graph_nodes(_MODEL)
    return [n for n in nodes if "HashRouter" in str(n.get("namespace") or "")]


def _shape(nodes, label):
    for node in nodes:
        if (node.get("label") or "") == label:
            for item in node.get("attrs") or []:
                if item.get("key") == "output_shape":
                    return str(item.get("value"))
    return None


class TestTheRouterReadsItsTable:
    def test_the_table_keeps_its_shape_and_dtype(self, router_nodes) -> None:
        assert _shape(router_nodes, "tid2eid") == "[129280, 6] int64"

    def test_the_lookup_keeps_the_tables_trailing_axis(self, router_nodes) -> None:
        """``table[ids]`` is ``ids.shape ++ table.shape[1:]``, not ``ids.shape``."""
        assert _shape(router_nodes, "indices") == "[B*S, 6] int64"

    def test_the_selected_weights_are_one_per_expert_per_token(
        self, router_nodes
    ) -> None:
        assert _shape(router_nodes, "weights") == "[B*S, 6] bfloat16"
