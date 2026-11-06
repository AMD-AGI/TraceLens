###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""A consumer of one split slot is handed that slot, not the undivided tensor.

``q_pass, q_rot = q.split([128, 64], dim=-1)`` followed by
``torch.cat((q_pass, q_rot), dim=-1)`` must come back to 192. It came back 384:
the per-ordinal slices were published only AFTER the whole graph had been
inferred, so while the ``cat`` was being inferred the only spec available for
either operand was the undivided ``[..., 192]`` parent -- and the concat rule
dutifully summed it twice.

The matching risk is a split whose recorded sizes do not actually divide the
parent (GLM's indexer has a 3-way split of ``[B, S, 257]`` whose sizes read
1/0/0). Those slices are still good enough to LABEL an output port, but feeding
a zero-width operand into a shape rule silently drops an axis for the rest of
the chain -- so they must not be computed with.
"""

from __future__ import annotations

from TraceLens.ModelUtils.shape_inference import TensorSpec, _slices_tile


def _spec(*shape) -> TensorSpec:
    return TensorSpec(shape=tuple(shape), dtype="bfloat16")


class TestSlicesTile:
    def test_sizes_that_add_up_to_the_parent_are_usable(self) -> None:
        whole = _spec("B", 96, "S", 192)
        slices = [_spec("B", 96, "S", 128), _spec("B", 96, "S", 64)]
        assert _slices_tile(whole, slices, removes_axis=False)

    def test_sizes_that_do_not_add_up_are_rejected(self) -> None:
        """GLM's indexer: a [B, S, 257] split whose recorded sizes read 1/0/0."""
        whole = _spec("B", "S", 257)
        slices = [_spec("B", "S", 1), _spec("B", "S", 0), _spec("B", "S", 0)]
        assert not _slices_tile(whole, slices, removes_axis=False)

    def test_a_zero_width_slice_is_never_usable(self) -> None:
        whole = _spec("B", "S", 8)
        slices = [_spec("B", "S", 8), _spec("B", "S", 0)]
        assert not _slices_tile(whole, slices, removes_axis=False)

    def test_a_symbolic_width_is_accepted_when_otherwise_consistent(self) -> None:
        """A width that cannot be summed is normal; one that contradicts is not."""
        whole = _spec("B", "S", "D")
        slices = [_spec("B", "S", "d0"), _spec("B", "S", "d1")]
        assert _slices_tile(whole, slices, removes_axis=False)

    def test_slices_that_differ_on_two_axes_are_rejected(self) -> None:
        """A split divides exactly one axis; anything else was not understood."""
        whole = _spec("B", 8, 192)
        slices = [_spec("B", 4, 128), _spec("B", 4, 64)]
        assert not _slices_tile(whole, slices, removes_axis=False)

    def test_slices_identical_to_the_parent_carry_nothing(self) -> None:
        whole = _spec("B", "S", 192)
        slices = [_spec("B", "S", 192), _spec("B", "S", 192)]
        assert not _slices_tile(whole, slices, removes_axis=False)

    def test_unbind_slices_drop_the_axis(self) -> None:
        """``qkv.unbind(0)`` on [3, Pv, 16, 64] gives three [Pv, 16, 64]."""
        whole = _spec(3, "Pv", 16, 64)
        slices = [_spec("Pv", 16, 64)] * 3
        assert _slices_tile(whole, slices, removes_axis=True)

    def test_unbind_slices_that_keep_the_axis_are_rejected(self) -> None:
        whole = _spec(3, "Pv", 16, 64)
        slices = [_spec(3, "Pv", 16, 64)] * 3
        assert not _slices_tile(whole, slices, removes_axis=True)
