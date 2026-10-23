###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""A write into a tensor returns THAT tensor -- the one the call names first.

GLM's experts accumulate across their loop::

    final = torch.zeros_like(hidden_states)     # [B*S, 4096] bfloat16
    for expert_idx in hit:
        top_k_pos, token_idx = torch.where(mask[expert_idx])
        ...
        final.index_add_(0, token_idx, current.to(final.dtype))

``index_add_`` hands back ``final``; the index addresses into it and the source
supplies values, but neither decides what comes out. The rule saying so was
right, and ``final`` was the call's first operand and the graph's first edge.

It still returned ``[nnz, 1] int64`` -- the shape of ``token_idx``. The operand
list has every operand whose shape did not resolve DROPPED from it rather than
held open, so with ``final`` not yet resolved the later operands moved up and
``inputs[0]`` was the index. The rule then faithfully returned the index.

So the operand is taken by the position the CALL passes it at, and the rule
declines when that operand is genuinely one of the dropped ones rather than
answering with a different one.

Which ops are in this class is torch's answer too. The schema marks the written
argument -- ``index_add_`` is ``(Tensor(a!) self, ...)`` -- and the out-of-place
spelling beside it (``masked_fill``, ``copy``) returns a copy of the receiver
with the write applied, which has the receiver's shape just the same. Having an
in-place sibling is not sufficient on its own: ``add`` has ``add_`` and ``add``
broadcasts, so its result can be wider than its receiver. Torch settles that as
well, by being handed a second operand bigger than the first.
"""

from __future__ import annotations

from TraceLens.ModelUtils.shape_inference import (
    ShapeInferencer,
    TensorSpec,
    op_writes_into_its_first_operand,
)


class TestWhichOpsWriteIntoTheirReceiver:
    def test_the_in_place_spellings(self) -> None:
        for name in ("index_add_", "scatter_add_", "masked_fill_", "copy_", "scatter_"):
            assert op_writes_into_its_first_operand(name), name

    def test_the_out_of_place_spellings_beside_them(self) -> None:
        for name in ("masked_fill", "copy", "scatter", "index_add"):
            assert op_writes_into_its_first_operand(name), name

    def test_a_display_label_is_looked_up_not_matched(self) -> None:
        """``index add`` is the same op written with a space."""
        for label in ("index add", "masked fill", "masked scatter", "scatter add"):
            assert op_writes_into_its_first_operand(label), label

    def test_an_elementwise_op_with_an_in_place_sibling_is_excluded(self) -> None:
        """``add`` has ``add_``, but ``add`` broadcasts past its receiver."""
        for name in ("add", "sub", "mul", "div", "pow", "maximum"):
            assert not op_writes_into_its_first_operand(name), name

    def test_ops_that_are_not_writes_at_all(self) -> None:
        for name in ("where", "cat", "linear", "matmul", "softmax", "view", "sum"):
            assert not op_writes_into_its_first_operand(name), name

    def test_nothing_is_claimed_for_an_unknown_name(self) -> None:
        assert not op_writes_into_its_first_operand("some_custom_kernel")
        assert not op_writes_into_its_first_operand("")


def _inferencer(positions: list[int]) -> ShapeInferencer:
    inferencer = ShapeInferencer.__new__(ShapeInferencer)
    inferencer._last_input_positions = positions
    return inferencer


ACCUMULATOR = TensorSpec(shape=("B*S", 4096), dtype="bfloat16")
INDEX = TensorSpec(shape=("nnz", 1), dtype="int64")
SOURCE = TensorSpec(shape=(1,), dtype="bfloat16")


class TestTheOperandComesFromTheCallsPosition:
    def test_every_operand_resolved(self) -> None:
        inputs = [ACCUMULATOR, INDEX, SOURCE]
        assert _inferencer([0, 1, 2])._operand_at(inputs, 0) is ACCUMULATOR

    def test_a_dropped_receiver_is_not_replaced_by_the_index(self) -> None:
        """The defect: operands move up, and operand 0 becomes the index."""
        inputs = [INDEX, SOURCE]
        assert _inferencer([1, 2])._operand_at(inputs, 0) is None

    def test_a_dropped_middle_operand_does_not_shift_the_receiver(self) -> None:
        inputs = [ACCUMULATOR, SOURCE]
        assert _inferencer([0, 2])._operand_at(inputs, 0) is ACCUMULATOR

    def test_a_later_position_is_reachable_too(self) -> None:
        inputs = [ACCUMULATOR, INDEX, SOURCE]
        assert _inferencer([0, 1, 2])._operand_at(inputs, 2) is SOURCE


class TestWhereItDeclines:
    def test_inputs_the_caller_replaced_wholesale(self) -> None:
        """A seeded/descriptor override leaves the positions describing nothing."""
        assert _inferencer([0, 1, 2])._operand_at([ACCUMULATOR], 0) is None

    def test_no_operands_at_all(self) -> None:
        assert _inferencer([])._operand_at([], 0) is None
