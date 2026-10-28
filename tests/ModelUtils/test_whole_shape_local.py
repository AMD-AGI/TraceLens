###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""A local bound to a tensor's whole shape, and what is not one.

``orig_shape = hidden_states.shape`` followed by ``.view(*orig_shape)`` is how a
block that flattens its batch axes puts them back. Unrecognised, the star
rendered as the bare name and the view reported whatever it was handed -- which,
for GLM's experts, is the flattened ``[B*S, H]`` the call exists to undo.

``x.size()`` spells the same thing as a call, so it reads the same. A slice of a
shape is a different local and belongs to the sibling that handles those; so
does anything that is not a shape read at all.
"""

from __future__ import annotations

import ast

from TraceLens.ModelUtils.ast_analyze import _whole_shape_token


def _expr(source: str) -> ast.AST:
    return ast.parse(source, mode="eval").body


class TestWhatItRecognises:
    def test_an_attribute_shape_read(self) -> None:
        assert _whole_shape_token(_expr("hidden_states.shape")) == (
            "hidden_states.shape[:]"
        )

    def test_the_size_call_spelling_of_the_same_thing(self) -> None:
        assert _whole_shape_token(_expr("x.size()")) == "x.shape[:]"


class TestWhatItDeclines:
    def test_a_sized_call_with_an_axis(self) -> None:
        """``x.size(-1)`` is one axis, not the whole shape."""
        assert _whole_shape_token(_expr("x.size(-1)")) is None

    def test_a_slice_of_a_shape(self) -> None:
        """That is a leading-axes local, which the sibling reads."""
        assert _whole_shape_token(_expr("x.shape[:-1]")) is None

    def test_an_indexed_axis(self) -> None:
        assert _whole_shape_token(_expr("x.shape[0]")) is None

    def test_an_attribute_that_is_not_a_shape(self) -> None:
        assert _whole_shape_token(_expr("x.dtype")) is None

    def test_a_shape_read_on_self(self) -> None:
        """``self.shape`` is a module attribute, not an operand's shape."""
        assert _whole_shape_token(_expr("self.shape")) is None

    def test_something_that_is_not_a_read_at_all(self) -> None:
        for source in ("f(x)", "a + b", "[1, 2]", "x"):
            assert _whole_shape_token(_expr(source)) is None, source

    def test_a_size_call_on_something_unnameable(self) -> None:
        assert _whole_shape_token(_expr("f(x).size()")) is None
