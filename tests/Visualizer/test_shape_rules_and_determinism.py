###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Shape rules that read a call's own declaration, and order that must not drift.

Kimi's unpadded attention reported a rank-5 ``[1, nnz, 3, S, 12288]`` for a
query projection because three ops passed their input straight through: a bare
``flatten()``, an einops ``rearrange`` and an ``F.pad``. Separately, the export
produced different output from one run to the next because set iteration order
-- which Python randomises per process -- decided how tiles were named.
"""

from __future__ import annotations

from TraceLens.ModelUtils.shape_inference import _einops_shape
from TraceLens.Visualizer.model_explorer_export.merge import _group_entry_buckets


class TestEinopsPatterns:
    """The pattern string is the whole specification; apply it exactly."""

    def test_collapsing_leading_axes_keeps_the_ellipsis_tail(self) -> None:
        assert _einops_shape("b s ... -> (b s) ...", ("B", "S", 7168), {}) == (
            "B*S",
            7168,
        )

    def test_splitting_an_axis_uses_the_given_size(self) -> None:
        assert _einops_shape("... (h d) -> ... h d", (1, "nnz", 12288), {"d": 128}) == (
            1,
            "nnz",
            96,
            128,
        )

    def test_merging_trailing_axes_multiplies_them(self) -> None:
        assert _einops_shape("b t h d -> b t (h d)", ("B", "S", 64, 128), {}) == (
            "B",
            "S",
            8192,
        )

    def test_a_split_with_no_size_given_is_declined(self) -> None:
        """Nothing says how to divide the axis, so the tensor passes through."""
        assert _einops_shape("... (h d) -> ... h d", (1, "nnz", 12288), {}) is None

    def test_a_pattern_with_no_arrow_is_declined(self) -> None:
        assert _einops_shape("b s d", ("B", "S", 8), {}) is None

    def test_a_right_side_name_from_nowhere_is_declined(self) -> None:
        assert _einops_shape("b s -> b s k", ("B", "S"), {}) is None

    def test_a_rank_the_pattern_does_not_describe_is_declined(self) -> None:
        assert _einops_shape("b s -> s b", ("B", "S", 8), {}) is None


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
        assert len(buckets) == 1


class TestSliceBoundedByALocalName:
    """Partial RoPE splits on a local, then concatenates the halves back."""

    @staticmethod
    def _bound(raw):
        from TraceLens.ModelUtils.shape_inference import _slice_bound_name

        return _slice_bound_name(raw)

    def test_head_slice_takes_the_named_width(self) -> None:
        assert self._bound("(..., :rotary_dim)") == ("rotary_dim", True)

    def test_tail_slice_takes_the_remainder(self) -> None:
        assert self._bound("(..., rotary_dim:)") == ("rotary_dim", False)

    def test_a_numeric_bound_is_left_to_the_folding_branches(self) -> None:
        assert self._bound("(..., :2051)") is None

    def test_a_full_slice_narrows_nothing(self) -> None:
        assert self._bound("(..., :)") is None

    def test_the_two_halves_sum_back_to_the_whole(self) -> None:
        """``r`` and ``128-(r)`` must cancel, or the concat reports double."""
        from TraceLens.ModelUtils.shape_inference import _sum_dim_sizes

        assert _sum_dim_sizes(["rotary_dim", "128-(rotary_dim)"]) == 128
