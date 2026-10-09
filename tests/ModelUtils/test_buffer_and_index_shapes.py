###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""A routing table keeps its shape, its dtype, and its trailing axis.

``DeepseekV4HashRouter`` selects experts from a frozen lookup table::

    self.tid2eid = nn.Buffer(torch.zeros(config.vocab_size, self.top_k, dtype=torch.long))
    indices = self.tid2eid[input_ids.reshape(-1)].long()
    weights = scores.gather(1, indices)

Three things went wrong, each hiding the next:

* buffer capture only understood 1-D ``torch.arange`` buffers, so a 2-D
  ``torch.zeros`` table was unknown and rendered ``[1]`` in the module's working
  precision;
* a subscript advanced index was given ``torch.gather``'s rule, which returns the
  INDEX's shape -- right for ``gather(dim, index)``, wrong for ``table[ids]``,
  which keeps the table's trailing axes;
* and once the table was correctly int64, "the base is the non-integer operand"
  found no base at all, because a routing table is int64 exactly like the ids
  that index it.

The final ``scores.gather(1, indices)`` reached ``[B*S, top_k]`` throughout --
by a fallback that happens to agree, while every operand feeding it was wrong.
"""

from __future__ import annotations

import ast

from TraceLens.ModelUtils.shape_inference import (
    ShapeContext,
    _constructor_buffer_spec,
    _literal_torch_dtype,
)


def _call(source: str) -> ast.Call:
    return ast.parse(source, mode="eval").body


class TestLiteralTorchDtype:
    def test_long_is_int64(self) -> None:
        assert _literal_torch_dtype(_call("torch.zeros(1)").func.value) is None
        assert (
            _literal_torch_dtype(ast.parse("torch.long", mode="eval").body) == "int64"
        )

    def test_bool_and_int(self) -> None:
        assert _literal_torch_dtype(ast.parse("torch.bool", mode="eval").body) == "bool"
        assert (
            _literal_torch_dtype(ast.parse("torch.int32", mode="eval").body) == "int32"
        )

    def test_an_unnamed_dtype_answers_nothing(self) -> None:
        assert _literal_torch_dtype(ast.parse("x.dtype", mode="eval").body) is None


class TestConstructorBufferSpec:
    def _spec(self, source: str, **self_dims):
        return _constructor_buffer_spec(
            _call(source),
            config={"vocab_size": 129280},
            context=ShapeContext(),
            dim_locals={},
            self_dims=self_dims,
        )

    def test_the_routing_table(self) -> None:
        spec = self._spec(
            "torch.zeros(config.vocab_size, self.top_k, dtype=torch.long)", top_k=6
        )
        assert spec is not None
        assert spec.shape == (129280, 6), spec.shape
        assert spec.dtype == "int64", spec.dtype

    def test_a_buffer_sized_by_an_attribute_the_constructor_set(self) -> None:
        """``self.top_k`` is not a local, so the ordinary resolver cannot see it."""
        assert self._spec("torch.zeros(self.top_k)", top_k=6).shape == (6,)

    def test_no_dtype_stated_means_no_dtype_claimed(self) -> None:
        spec = self._spec("torch.zeros(config.vocab_size)")
        assert spec is not None and spec.dtype is None

    def test_an_unknown_config_size_stays_symbolic(self) -> None:
        """A name the reader recognises, never a guessed number or a ``?``."""
        spec = self._spec("torch.zeros(config.mystery, 4)")
        assert spec is not None and spec.shape[1] == 4, spec

    def test_a_non_constructor_is_not_this(self) -> None:
        assert self._spec("torch.arange(16)") is None
