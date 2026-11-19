###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""An argument is read by the name its op's own schema gives it.

Counting positions by hand works only while every op puts its arguments in the
same order, and they do not:

* ``x.norm(2, dim=-1)`` passes the ORDER first, so reading argument 0 as the axis
  recorded ``dim: 2`` -- an axis that call never named.
* ``x.var(-1, False)`` passes ``unbiased`` where ``sum`` takes ``keepdim``, so a
  rule that reads argument 1 as ``keepdim`` would collapse an axis on a call that
  asked for nothing of the sort.
* ``x.sum(1, True)`` passes ``keepdim`` POSITIONALLY, which the keyword-only
  reading dropped entirely.
* ``F.pad(x, (1, 0))`` is a free function whose namespace is not ``torch``, so the
  "is it a method?" heuristic took it for one and read the padding from the wrong
  argument.

The schema answers all four, and ``torch`` is the authority on it.
"""

from __future__ import annotations

import ast

import pytest

from TraceLens.ModelUtils.ast_analyze import (
    _calls_through_namespace,
    _schema_arguments,
    _schema_positional_names,
)


def _bound(expression: str, op: str) -> dict[str, str]:
    call = ast.parse(expression, mode="eval").body
    assert isinstance(call, ast.Call)
    return {
        name: ast.unparse(value) for name, value in _schema_arguments(op, call).items()
    }


class TestSchemaPositionalNames:
    def test_a_reduction_names_dim_then_keepdim(self) -> None:
        assert _schema_positional_names("sum") == ("self", "dim", "keepdim")

    def test_norm_names_the_order_first(self) -> None:
        assert _schema_positional_names("norm") == ("self", "p", "dim", "keepdim")

    def test_var_names_unbiased_between_them(self) -> None:
        assert _schema_positional_names("var") == (
            "self",
            "dim",
            "unbiased",
            "keepdim",
        )

    def test_cat_takes_tensors_not_self(self) -> None:
        assert _schema_positional_names("cat") == ("tensors", "dim")

    def test_an_op_with_no_schema_answers_empty(self) -> None:
        assert _schema_positional_names("not_an_aten_op") == ()


class TestCallsThroughNamespace:
    @pytest.mark.parametrize(
        "expression", ["torch.cat([x, y], 1)", "F.pad(x, (1, 0))", "cat(x, 1)"]
    )
    def test_a_free_function_passes_the_tensor_positionally(
        self, expression: str
    ) -> None:
        call = ast.parse(expression, mode="eval").body
        assert isinstance(call, ast.Call)
        assert _calls_through_namespace(call.func) is True

    def test_a_method_takes_its_tensor_as_the_receiver(self) -> None:
        call = ast.parse("x.sum(1)", mode="eval").body
        assert isinstance(call, ast.Call)
        assert _calls_through_namespace(call.func) is False


class TestArgumentBinding:
    def test_keepdim_passed_positionally(self) -> None:
        assert _bound("x.sum(1, True)", "sum") == {"dim": "1", "keepdim": "True"}

    def test_keepdim_passed_by_keyword(self) -> None:
        assert _bound("x.sum(dim=1, keepdim=True)", "sum") == {
            "dim": "1",
            "keepdim": "True",
        }

    def test_the_mixed_spelling(self) -> None:
        assert _bound("x.sum(1, keepdim=True)", "sum") == {
            "dim": "1",
            "keepdim": "True",
        }

    def test_norms_first_argument_is_the_order_not_the_axis(self) -> None:
        bound = _bound("x.norm(2, dim=-1)", "norm")
        assert bound["p"] == "2"
        assert bound["dim"] == "-1"

    def test_vars_second_argument_is_unbiased_not_keepdim(self) -> None:
        bound = _bound("x.var(-1, False)", "var")
        assert bound["dim"] == "-1"
        assert "keepdim" not in bound
        assert bound["unbiased"] == "False"

    def test_var_still_reaches_keepdim_in_its_own_place(self) -> None:
        assert _bound("x.var(-1, False, True)", "var")["keepdim"] == "True"

    def test_a_free_function_shifts_every_argument_along(self) -> None:
        assert _bound("torch.transpose(x, 1, 2)", "transpose") == {
            "self": "x",
            "dim0": "1",
            "dim1": "2",
        }

    def test_the_same_call_as_a_method(self) -> None:
        assert _bound("x.transpose(1, 2)", "transpose") == {"dim0": "1", "dim1": "2"}

    def test_transpose_by_keyword(self) -> None:
        assert _bound("x.transpose(dim0=1, dim1=2)", "transpose") == {
            "dim0": "1",
            "dim1": "2",
        }

    def test_pad_is_a_free_function_despite_its_namespace(self) -> None:
        assert _bound("F.pad(x, (1, 0))", "pad")["pad"] == "(1, 0)"

    def test_pad_by_keyword(self) -> None:
        assert _bound("F.pad(x, pad=(1, 0))", "pad")["pad"] == "(1, 0)"

    def test_topk_names_its_axis(self) -> None:
        assert _bound("x.topk(2, dim=1)", "topk") == {"k": "2", "dim": "1"}

    def test_topk_as_a_free_function(self) -> None:
        bound = _bound("torch.topk(x, 2, 1)", "topk")
        assert bound["k"] == "2"
        assert bound["dim"] == "1"
