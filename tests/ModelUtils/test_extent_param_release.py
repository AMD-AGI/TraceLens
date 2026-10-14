###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""A sizing read must not take the boundary an operand read needs.

A forward parameter reaches its consumer through ONE boundary -- the op that
names it as its boundary input -- so two ops reading the same parameter are in
competition, and the first one wins. That is harmless while both really take the
tensor, and wrong when one of them only reads its SIZE.

GLM's ``append_visible_tail`` sizes ``torch.full((batch_size,), ...)`` off
``token_visible`` two lines before ``token_visible.long().sum(-1)`` reads it. The
sum lost the parameter, fell back to chaining onto its neighbour, and closed a
cycle between the ops on either side of it.

So an extent read yields the boundary when another op takes the same tensor as an
operand -- and keeps it when nothing else does, which is the whole point of
recovering it (``torch.arange(k_len)`` where ``k_len = idx_q.shape[-1]`` has no
operand at all and would otherwise draw rootless).
"""

from __future__ import annotations

from TraceLens.ModelUtils.ast_analyze import (
    EXTENT_PARAMS_DETAIL,
    ForwardOperation,
    _extent_param_names,
    _release_extent_only_params,
)


def _op(attr_name, label, params=(), sized=()):
    details = (f"{EXTENT_PARAMS_DETAIL}{','.join(sized)}",) if sized else ()
    return ForwardOperation(
        attr_name=attr_name,
        label=label,
        class_name="Op",
        param_inputs=tuple(params),
        details=details,
    )


class TestWhoKeepsTheParameter:
    def test_a_sizing_read_yields_to_an_operand_read(self) -> None:
        ops = [
            _op("full", "Full", params=("token_visible",), sized=("token_visible",)),
            _op("sum", "Sum", params=("token_visible",)),
        ]
        _release_extent_only_params(ops)
        assert ops[0].param_inputs == ()
        assert ops[1].param_inputs == ("token_visible",)

    def test_it_keeps_a_parameter_nothing_else_takes(self) -> None:
        """The generator case the recovery exists for -- no operand read at all."""
        ops = [_op("arange", "Arange", params=("idx_q",), sized=("idx_q",))]
        _release_extent_only_params(ops)
        assert ops[0].param_inputs == ("idx_q",)

    def test_order_does_not_matter(self) -> None:
        """The operand read wins whether it comes first or last."""
        ops = [
            _op("sum", "Sum", params=("token_visible",)),
            _op("full", "Full", params=("token_visible",), sized=("token_visible",)),
        ]
        _release_extent_only_params(ops)
        assert ops[0].param_inputs == ("token_visible",)
        assert ops[1].param_inputs == ()

    def test_only_the_sized_parameter_is_released(self) -> None:
        """An op that both sizes off one tensor and takes another keeps the operand."""
        ops = [
            _op("full", "Full", params=("lengths", "x"), sized=("lengths",)),
            _op("sum", "Sum", params=("lengths",)),
        ]
        _release_extent_only_params(ops)
        assert ops[0].param_inputs == ("x",)

    def test_two_sizing_reads_of_one_parameter_both_keep_it(self) -> None:
        """Neither is an operand read, so neither is starving the other."""
        ops = [
            _op("a", "Arange", params=("lens",), sized=("lens",)),
            _op("b", "Full", params=("lens",), sized=("lens",)),
        ]
        _release_extent_only_params(ops)
        assert ops[0].param_inputs == ("lens",)
        assert ops[1].param_inputs == ("lens",)


class TestWhatIsLeftAlone:
    def test_an_op_with_no_mark(self) -> None:
        ops = [_op("sum", "Sum", params=("x",))]
        _release_extent_only_params(ops)
        assert ops[0].param_inputs == ("x",)

    def test_no_operations_at_all(self) -> None:
        ops: list[ForwardOperation] = []
        _release_extent_only_params(ops)
        assert ops == []


class TestReadingTheMark:
    def test_several_names_on_one_detail(self) -> None:
        assert _extent_param_names((f"{EXTENT_PARAMS_DETAIL}a, b ,c",)) == {
            "a",
            "b",
            "c",
        }

    def test_unrelated_details_say_nothing(self) -> None:
        assert _extent_param_names(("raw_op: full", "size0: batch_size")) == set()

    def test_an_empty_mark(self) -> None:
        assert _extent_param_names((EXTENT_PARAMS_DETAIL,)) == set()
