###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""A starred term in a reshape is not always the first one.

Kimi's own ``pad_input`` ends::

    out = hidden_states.new_zeros((batch_size * seq_len, *hidden_states.shape[1:]))
    out[indices] = hidden_states
    return out.view(batch_size, seq_len, *hidden_states.shape[1:])

-- two axes it names, then "and whatever feature axes came in". The resolver
recognised a star only at position 0, so this returned ``None`` and the view
rendered as its own source: a reshape reported as doing nothing, which this
module's own comment calls the one thing a reshape is never written to do.

Its two leading axes are ``pad_input``'s own host PARAMETERS, which no config
states. They are resolvable all the same, because the source's leading axis is
literally ``batch_size * seq_len`` -- the view splits one axis into the two
factors that axis is already written in terms of, so the elements agree by
construction. A name the source does NOT mention stays refused: that is the
case where the axis would have to be inferred from leading dims that may
themselves be a fallback.
"""

from __future__ import annotations

from TraceLens.ModelUtils.shape_inference import TensorSpec, _resolve_view_shape


def _spec(*shape):
    return TensorSpec(shape=tuple(shape), dtype="bfloat16")


class TestAStarAfterNamedAxes:
    def test_kimi_pad_input(self) -> None:
        out = _resolve_view_shape(
            "batch_size, seq_len, *hidden_states.shape[1:]",
            _spec("batch_size * seq_len", 7168),
            {},
        )
        assert out == ("batch_size", "seq_len", 7168)

    def test_the_star_may_carry_several_axes(self) -> None:
        out = _resolve_view_shape(
            "batch_size, *x.shape[1:]", _spec("batch_size * seq_len", 8, 128), {}
        )
        assert out == ("batch_size", 8, 128)

    def test_a_star_with_no_subscript_takes_every_axis(self) -> None:
        out = _resolve_view_shape("batch_size, *x.shape", _spec("batch_size", 4), {})
        assert out == ("batch_size", "batch_size", 4)


class TestWhichBareNamesAreAccepted:
    def test_a_name_the_source_is_written_in_terms_of(self) -> None:
        out = _resolve_view_shape(
            "batch_size, seq_len", _spec("batch_size * seq_len"), {}
        )
        assert out == ("batch_size", "seq_len")

    def test_a_name_the_source_never_mentions_stays_refused(self) -> None:
        """Inferring it would mean trusting leading dims that may be a fallback."""
        assert _resolve_view_shape("nope_name", _spec("B", "S", 8), {}) is None

    def test_a_star_prefix_plus_an_unknown_name_stays_refused(self) -> None:
        assert (
            _resolve_view_shape("*x.shape[:-1], nope", _spec("B", "S", 8, 16), {})
            is None
        )


class TestWhatIsUnchanged:
    def test_a_leading_star_with_a_trailing_minus_one(self) -> None:
        """The ``-1`` spans the whole trailing block (8*16), as it already did."""
        out = _resolve_view_shape("*x.shape[:-1], -1", _spec("B", "S", 8, 16), {})
        assert out == ("B", "S", 128)

    def test_an_empty_detail(self) -> None:
        assert _resolve_view_shape("", _spec("B", "S"), {}) is None
