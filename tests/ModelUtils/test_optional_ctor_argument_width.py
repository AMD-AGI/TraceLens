###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""A width the caller may override, and the ``None`` default that says so.

MiniMax builds one MLP class two ways::

    class MiniMaxM3VLDenseMLP(nn.Module):
        def __init__(self, config, intermediate_size: int | None = None):
            inter = intermediate_size if intermediate_size is not None else config.dense_intermediate_size
            self.gate_up_proj = nn.Linear(config.hidden_size, 2 * inter, bias=False)
            self.down_proj = nn.Linear(inter, config.hidden_size, bias=False)

-- once as the dense layers' own ``mlp`` (12288) and once as the sparse block's
``shared_experts`` (3072). The conditional was not read at all, so ``inter`` had
no value, neither ``nn.Linear`` registered a width, and both projections fell
back to passing their input width through. ``down_proj`` then reported
``[B, S, 3072]`` where the model produces ``[B, S, 6144]``, and the reshape that
closes the block was handed a tensor with the wrong number of elements.

Which branch holds is decided by the call site, and the table of constructor
bindings already records that: a parameter the caller pinned is bound, one left
to its ``None`` default is not. So this reads as a fact about the two passes
rather than as a guess about the configuration -- and only for the idiom where
the guarded name IS the value it guards. Any other conditional asks a question
about the model's configuration that this scope cannot answer.
"""

from __future__ import annotations

import ast

import pytest

from TraceLens.ModelUtils.shape_inference import (
    ShapeContext,
    _optional_argument_guard,
    _resolve_dim_expr,
)


def _expr(source: str) -> ast.AST:
    return ast.parse(source, mode="eval").body


def _resolve(source: str, *, config=None, local_vars=None):
    return _resolve_dim_expr(
        _expr(source),
        config=config or {},
        local_vars=local_vars or {},
        context=ShapeContext(),
    )


_GUARD = "intermediate_size if intermediate_size is not None else config.inter"


class TestTheCallSiteDecides:
    def test_a_pinned_argument_wins(self) -> None:
        assert (
            _resolve(
                _GUARD, config={"inter": 12288}, local_vars={"intermediate_size": 3072}
            )
            == 3072
        )

    def test_an_argument_left_at_its_default_falls_back_to_config(self) -> None:
        assert _resolve(_GUARD, config={"inter": 12288}) == 12288

    def test_the_test_written_the_other_way_round(self) -> None:
        source = "config.inter if intermediate_size is None else intermediate_size"
        assert _resolve(source, config={"inter": 12288}) == 12288
        assert (
            _resolve(
                source, config={"inter": 12288}, local_vars={"intermediate_size": 4}
            )
            == 4
        )

    def test_the_chosen_branch_is_still_an_expression(self) -> None:
        """The fallback is evaluated, not copied: ``2 * inter`` is arithmetic."""
        assert (
            _resolve(
                "head_dim if head_dim is not None else config.h // 4", config={"h": 256}
            )
            == 64
        )


class TestWhatItWillNotDecide:
    def test_a_conditional_on_the_configuration(self) -> None:
        """Nothing here knows whether the model sets ``use_bias``."""
        assert (
            _resolve(
                "a if config.use_bias else b",
                config={"use_bias": 1},
                local_vars={"a": 8, "b": 4},
            )
            is None
        )

    def test_a_none_test_on_a_name_that_is_not_the_branch(self) -> None:
        """``x is not None`` guarding somebody ELSE's width decides nothing."""
        assert (
            _resolve("a if x is not None else b", local_vars={"x": 1, "a": 8, "b": 4})
            is None
        )

    def test_a_comparison_that_is_not_against_none(self) -> None:
        assert (
            _resolve("x if x is not Ellipsis else y", local_vars={"x": 8, "y": 4})
            is None
        )

    def test_the_branch_is_answered_by_the_ordinary_rules(self) -> None:
        """A config key with no number stated is still a named extent."""
        assert _resolve(_GUARD, config={}) == "inter"

    def test_a_fallback_that_names_nothing_in_scope(self) -> None:
        assert _resolve("x if x is not None else y") is None


class TestTheGuardItself:
    @pytest.mark.parametrize(
        "source",
        [
            "x if x is not None else d",
            "d if x is None else x",
        ],
    )
    def test_the_idiom_names_its_argument(self, source: str) -> None:
        assert _optional_argument_guard(_expr(source)) == "x"

    @pytest.mark.parametrize(
        "source",
        [
            "a if x is not None else b",
            "x if x else d",
            "x if x == 0 else d",
            "x if x is not None and y else d",
        ],
    )
    def test_anything_else_is_not_this_idiom(self, source: str) -> None:
        assert _optional_argument_guard(_expr(source)) is None
