###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""A slice bound counted from the far end takes the other side.

Partial RoPE splits a tensor into the part it rotates and the part it leaves
alone. DeepSeek writes that split with negative bounds::

    x_pass, x_rope = x[..., :-rope_dim], x[..., -rope_dim:]
    ...
    return torch.cat([x_pass, rotated], dim=-1)

A bound that is a bare NAME was already narrowed symbolically, so the two
halves sum back to the source width. A NEGATED name was not: it failed the
identifier test and narrowed nothing, so both halves reported the full 512 and
the concat rejoining them answered 1024. That doubled width left DeepSeek's CSA
compressor and disagreed with the other arm of the phi below it -- the
empty-chunk arm, which got its 512 honestly from its own
``new_zeros((batch, 0, self.head_dim))``.

Negation does not need its own arithmetic: `[..., :-n]` is everything but the
last ``n``, which is what `[..., n:]` already computes, and `[..., -n:]` is
exactly those, which is `[..., :n]`. So the sign is stripped and the side is
inverted.
"""

from __future__ import annotations

from TraceLens.ModelUtils.shape_inference import TensorSpec, _slice_bound_name

from tests.ModelUtils.test_index_axes_placement import _make_inferencer, _node


class TestWhichSideTheBoundTakes:
    def test_a_plain_name(self) -> None:
        assert _slice_bound_name("(..., :rotary_dim)") == (
            "rotary_dim",
            True,
            -1,
            False,
        )
        assert _slice_bound_name("(..., rotary_dim:)") == (
            "rotary_dim",
            False,
            -1,
            False,
        )

    def test_a_negated_name_takes_the_other_side(self) -> None:
        # ``:-rope_dim`` is everything BUT the last rope_dim -> width - rope_dim.
        assert _slice_bound_name("(..., :-rope_dim)") == ("rope_dim", False, -1, True)
        # ``-rope_dim:`` is exactly those -> rope_dim.
        assert _slice_bound_name("(..., -rope_dim:)") == ("rope_dim", True, -1, True)

    def test_whitespace_after_the_colon(self) -> None:
        assert _slice_bound_name("(..., : -rope_dim)") == ("rope_dim", False, -1, True)

    def test_a_numeric_bound_is_left_to_the_branches_that_fold_it(self) -> None:
        assert _slice_bound_name("(..., :-1)") is None
        assert _slice_bound_name("(..., -1:)") is None

    def test_an_axis_that_is_not_the_last(self) -> None:
        assert _slice_bound_name("(:, :usable)") == ("usable", True, 1, False)

    def test_two_named_bounds_name_two_axes(self) -> None:
        assert _slice_bound_name("(:, :a, :b)") is None


class TestTheTwoHalvesSumBackToTheWhole:
    def _narrowed(self, slice_detail: str):
        inf = _make_inferencer()
        source = TensorSpec(shape=("batch", 1, "n_windows", 512), dtype="bfloat16")
        node = _node("Slice", details=[f"slice: {slice_detail}"])
        return inf._infer_node_output(node, [source], root=None).shape

    def test_the_part_rope_leaves_alone(self) -> None:
        assert self._narrowed("(..., :-rope_dim)") == (
            "batch",
            1,
            "n_windows",
            "512-(rope_dim)",
        )

    def test_the_part_rope_rotates(self) -> None:
        assert self._narrowed("(..., -rope_dim:)") == (
            "batch",
            1,
            "n_windows",
            "rope_dim",
        )

    def test_neither_half_is_the_whole(self) -> None:
        """The bug this fixes: both halves reporting 512, so the concat said 1024."""
        for detail in ("(..., :-rope_dim)", "(..., -rope_dim:)"):
            assert self._narrowed(detail)[-1] != 512, detail


class TestWhenTheEndIsNotKnown:
    """A bound counted from the END needs a known end.

    GLM cuts its conv tail with ``mixed_qkv[:, :, -seq_len:]``, and
    ``batch_size, seq_len = hidden_states.shape[:2]`` -- so ``seq_len`` IS the
    sequence dim the rest of the graph calls ``S``. Narrowing a SYMBOLIC axis to
    it adds no information and merely renames the axis, which left the
    ``view(batch_size, seq_len, -1, head_dim)`` below unable to cancel its
    ``-1``: three q/k/v views silently stopped reshaping. A CONCRETE axis is
    different -- there the width really is computable.
    """

    def _narrow(self, source_axis, detail):
        inf = _make_inferencer()
        source = TensorSpec(shape=("B", 3072, source_axis), dtype="bfloat16")
        node = _node("Slice", details=[f"slice: {detail}"])
        return inf._infer_node_output(node, [source], root=None).shape

    def test_a_symbolic_axis_keeps_its_own_name(self) -> None:
        assert self._narrow("S", "(:, :, -seq_len:)") == ("B", 3072, "S")

    def test_a_concrete_axis_is_genuinely_narrowed(self) -> None:
        assert self._narrow(512, "(:, :, -rope_dim:)") == ("B", 3072, "rope_dim")

    def test_a_bound_counted_from_the_front_is_unaffected(self) -> None:
        """``kv[:, :usable]`` narrows a symbolic axis and must keep doing so."""
        inf = _make_inferencer()
        source = TensorSpec(shape=("B", "S", 1024), dtype="bfloat16")
        node = _node("Slice", details=["slice: (:, :usable)"])
        assert inf._infer_node_output(node, [source], root=None).shape == (
            "B",
            "usable",
            1024,
        )
