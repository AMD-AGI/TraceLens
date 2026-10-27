###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Group-entry buckets follow the order the consumer's edges arrive in."""

from __future__ import annotations

from TraceLens.Visualizer.model_explorer_export.merge import _group_entry_buckets


class TestGroupEntryBucketOrder:
    """Two producers for one consumer must bucket in the consumer's edge order."""

    @staticmethod
    def _entry(node_id, sources):
        return {
            "id": node_id,
            "incomingEdges": [
                {"sourceNodeId": s, "sourceNodeOutputId": "0"} for s in sources
            ],
        }

    def test_buckets_follow_the_order_the_edges_arrive_in(self) -> None:
        node = self._entry("op", ["outside_b", "outside_a"])
        buckets = _group_entry_buckets([node], internal_ids=set())
        assert [sorted(sources)[0][0] for sources, _ in buckets] == [
            "outside_b",
            "outside_a",
        ]

    def test_the_reverse_edge_order_gives_the_reverse_buckets(self) -> None:
        node = self._entry("op", ["outside_a", "outside_b"])
        buckets = _group_entry_buckets([node], internal_ids=set())
        assert [sorted(sources)[0][0] for sources, _ in buckets] == [
            "outside_a",
            "outside_b",
        ]

    def test_one_producer_read_twice_opens_one_bucket(self) -> None:
        node = self._entry("op", ["outside_a", "outside_a"])
        buckets = _group_entry_buckets([node], internal_ids=set())
        assert buckets
