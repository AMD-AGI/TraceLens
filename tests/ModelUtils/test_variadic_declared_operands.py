###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""A variadic op counts the operands it DECLARES, not a ceiling.

An ``@op_l`` id in this file is a name this test gives a node. It is not a
lookup into an upstream modeling file.


``_needs_another_tensor_operand`` decides whether an op still has an operand
slot its edges have not filled, by comparing its predecessors against the
operand ceiling its signature allows. ``torch.cat`` and ``torch.stack`` have no
ceiling -- they take a list -- so the test answered "no" for every one of them,
and "it already reads something" is no evidence at all that every operand it
names is wired.

DeepSeek extends its attention mask with::

    attention_mask = torch.cat([attention_mask, block_bias.to(attention_mask.dtype)], dim=-1)

The cast of ``block_bias`` is the cat's one predecessor and ``attention_mask``
is a forward parameter, so the parameter never got its edge. A ``Concat`` left
with a single incoming edge is a forbidden single-input concat, so the elision
pass folded the whole operation away -- and the phi below it, having no cat to
read, took raw ``block_bias``. A deleted operation, presenting downstream as a
shape disagreement between two phi arms.
"""

from __future__ import annotations

from TraceLens.ModelUtils.block_tree import BlockNode
from TraceLens.ModelUtils.computation_graph import _needs_another_tensor_operand


def _op(raw_op: str, *, predecessors=(), param_inputs=()) -> BlockNode:
    return BlockNode(
        attr_name="@op_l1_c0_x",
        class_name=raw_op.title(),
        role="other",
        label=raw_op.title(),
        details=[f"raw_op: {raw_op}"],
        operation_predecessors=list(predecessors),
        param_inputs=list(param_inputs),
    )


class TestAVariadicOp:
    def test_a_cat_that_names_a_parameter_it_does_not_read_yet(self) -> None:
        node = _op("cat", predecessors=["cast"], param_inputs=["attention_mask"])
        assert _needs_another_tensor_operand(node) is True

    def test_a_cat_whose_operands_are_all_predecessors(self) -> None:
        """Nothing is declared beyond the edges, so nothing is missing."""
        node = _op("cat", predecessors=["a", "b"])
        assert _needs_another_tensor_operand(node) is False

    def test_a_stack_is_variadic_the_same_way(self) -> None:
        node = _op("stack", predecessors=["a"], param_inputs=["cos"])
        assert _needs_another_tensor_operand(node) is True


class TestABoundedOpIsUnchanged:
    def test_a_binary_op_short_of_its_ceiling(self) -> None:
        """``arange(n) > position_ids`` takes two and has one producer."""
        node = _op("greater", predecessors=["arange"], param_inputs=["position_ids"])
        assert _needs_another_tensor_operand(node) is True

    def test_a_binary_op_already_at_its_ceiling(self) -> None:
        node = _op("greater", predecessors=["a", "b"], param_inputs=["x"])
        assert _needs_another_tensor_operand(node) is False

    def test_an_op_with_no_resolvable_signature(self) -> None:
        node = _op("not_a_real_torch_op", predecessors=["a"], param_inputs=["x"])
        assert _needs_another_tensor_operand(node) is False
