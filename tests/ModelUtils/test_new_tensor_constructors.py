###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""``x.new_empty(*sizes)`` builds a tensor, and it is sized by its arguments.

GLM's ``expand_kv`` allocates its key buffer and then fills it::

    key_states = kv_nope.new_empty(*kv_nope.shape[:-1], qk_nope + qk_rope)
    key_states[..., :qk_nope].copy_(k_nope)
    key_states[..., qk_nope:].copy_(k_rot)

With no node for the allocation, ``key_states`` had no producer at all, so the
first ``copy_`` spine-fell onto the preceding op -- the undivided ``kv_nope`` --
and the key reached sdpa at the latent width (512) instead of the head width
(256), against a query of 256.

Two things are needed. The ``new_*`` family has to BE an op, like ``torch.zeros``
already is; and its shape comes from its own sizes, which can be arithmetic over
config scalars (``qk_nope_head_dim + qk_rope_head_dim``) and can name the
receiver's leading axes (``*x.shape[:-1]``) rather than being one scalar each.
"""

from __future__ import annotations

import textwrap

from TraceLens.ModelUtils.ast_analyze import analyze_source
from TraceLens.ModelUtils.shape_inference import (
    TensorSpec,
    _constructed_shape,
    _starred_shape_axes,
)


def _spec(*shape) -> TensorSpec:
    return TensorSpec(shape=tuple(shape), dtype="bfloat16")


class TestStarredShapeSlice:
    def test_all_but_the_last_axis(self) -> None:
        assert _starred_shape_axes("*kv.shape[:-1]", _spec("B", 64, "S", 512)) == [
            "B",
            64,
            "S",
        ]

    def test_a_leading_count(self) -> None:
        assert _starred_shape_axes("*x.shape[:2]", _spec("B", 64, "S", 512)) == [
            "B",
            64,
        ]

    def test_a_single_index(self) -> None:
        assert _starred_shape_axes("*x.shape[0]", _spec("B", 64)) == ["B"]

    def test_a_non_shape_token_is_not_this(self) -> None:
        assert _starred_shape_axes("*sizes", _spec("B", 64)) is None
        assert _starred_shape_axes("head_dim", _spec("B", 64)) is None

    def test_no_template_means_no_answer(self) -> None:
        assert _starred_shape_axes("*x.shape[:-1]", None) is None


class TestConstructedShape:
    def test_the_glm_key_buffer(self) -> None:
        """The case that was reporting 512 where the source says 256."""
        details = [
            "size0: *kv_nope.shape[:-1]",
            "size1: self.qk_nope_head_dim + self.qk_rope_head_dim",
        ]
        dims = {"qk_nope_head_dim": 256, "qk_rope_head_dim": 0}
        shape = _constructed_shape(details, dims, _spec("B", 64, "S", 512))
        assert shape == ("B", 64, "S", 256), shape

    def test_arithmetic_over_config_scalars_folds(self) -> None:
        shape = _constructed_shape(
            ["size0: 4", "size1: self.head_dim * 2"], {"head_dim": 64}, None
        )
        assert shape == (4, 128), shape

    def test_an_unresolvable_size_stays_symbolic(self) -> None:
        """Better a name the reader recognises than a folded guess."""
        shape = _constructed_shape(["size0: 4", "size1: self.mystery"], {}, None)
        assert shape == (4, "mystery"), shape

    def test_plain_constructors_are_unaffected(self) -> None:
        assert _constructed_shape(["size0: B", "size1: S"], {"B": 2, "S": 8}) == (2, 8)


_SIZES = textwrap.dedent("""
    class M(nn.Module):
        def forward(self, block_indices, key_length):
            batch, n_idx_heads, q_len, _ = block_indices.shape
            num_key_blocks = -(-key_length // 8)
            bias = block_indices.new_full(
                (batch, n_idx_heads, q_len, num_key_blocks + 1), float("-inf")
            )
            buffer = block_indices.new_empty(*block_indices.shape[:-1], 256)
            return bias, buffer
    """)


def _sizes_of(label: str) -> list[str]:
    analysis = analyze_source(_SIZES, config={"hidden_size": 8})
    operation = next(
        op
        for op in analysis.class_registry["M"].forward_operations.values()
        if (op.label or "") == label
    )
    return [d for d in (operation.details or []) if d.startswith("size")]


class TestWhatCountsAsASize:
    def test_a_fill_value_is_not_an_axis(self) -> None:
        """``new_full(size, value)`` takes ONE size argument, then a value.

        Counting ``float("-inf")`` as a fifth axis gave MiniMax's sparse block
        mask -- and the twelve ops downstream of it -- a rank the model's own
        docstring ("the full 4D attention mask") says it does not have.
        """
        assert _sizes_of("New full") == [
            "size0: batch",
            "size1: n_idx_heads",
            "size2: q_len",
            "size3: num_key_blocks + 1",
        ]

    def test_the_varargs_family_still_reads_every_argument(self) -> None:
        """``new_empty(*sizes)`` has no fill value: all of it is the shape."""
        assert _sizes_of("New empty") == [
            "size0: *block_indices.shape[:-1]",
            "size1: 256",
        ]
