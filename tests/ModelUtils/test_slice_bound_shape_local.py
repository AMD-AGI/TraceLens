###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""A slice bounded by a shape local names the axis that local measured.

DeepSeek's CSA compressor builds its block bias one column wider than it needs
and cuts the spare off on the way out::

    compressed_len = compressed_kv.shape[2]
    block_bias = compressed_kv.new_full((batch, 1, seq_len, compressed_len + 1), -inf)
    ...
    return compressed_kv, block_bias[..., :compressed_len]

``compressed_len`` IS ``compressed_kv.shape[2]``, which everything else calls
``n_windows``. Reported under the local's own name, the mask that concatenates
this bias read ``S + compressed_len`` while the ``F.pad`` that extends the same
mask on the other branch read ``S + n_windows`` -- one width, two spellings,
and a phi that could only call them different.

The extractor records what the bound was read from; the slice resolves it when
that tensor is in scope and keeps the local's own name when it is not.
"""

from __future__ import annotations

import ast

from TraceLens.ModelUtils.shape_inference import _shape_local_reads, _slice_bound_name


class TestWhatTheExtractorRecordsForABound:
    def test_the_detail_round_trips(self) -> None:
        details = [
            "slice: (..., :compressed_len)",
            "shape_locals: compressed_len=compressed_kv.shape[2]",
        ]
        assert _shape_local_reads(details) == {
            "compressed_len": "compressed_kv.shape[2]"
        }

    def test_the_bound_is_still_read_as_a_head_take(self) -> None:
        bound = _slice_bound_name("(..., :compressed_len)")
        assert bound is not None
        name, takes_head, axis, from_end = bound
        assert (name, takes_head, from_end) == ("compressed_len", True, False)
        assert axis == -1

    def test_a_numeric_bound_records_no_local(self) -> None:
        assert _shape_local_reads(["slice: (..., :8)"]) == {}


class TestTheNamesASubscriptMentions:
    """What the extractor walks for: every Name in the subscript, not just a
    bare bound, since a bound can be an expression (``:compressed_len + 1``)."""

    @staticmethod
    def _names(source: str) -> set[str]:
        subscript = ast.parse(source, mode="eval").body
        return {
            node.id for node in ast.walk(subscript.slice) if isinstance(node, ast.Name)
        }

    def test_a_bare_bound(self) -> None:
        assert self._names("x[..., :compressed_len]") == {"compressed_len"}

    def test_an_expression_bound(self) -> None:
        assert self._names("x[..., :compressed_len + 1]") == {"compressed_len"}

    def test_several_axes(self) -> None:
        assert self._names("x[:usable, :, :width]") == {"usable", "width"}

    def test_no_names_at_all(self) -> None:
        assert self._names("x[..., :8]") == set()
