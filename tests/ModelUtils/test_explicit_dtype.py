###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""An op given an explicit ``dtype=`` produces it, whatever flowed in.

``get_vision_cu_seqlens`` ends with::

    dtype = grid_thw.dtype if torch.jit.is_tracing() else torch.int32
    return F.pad(seqlens.cumsum(dim=0, dtype=dtype), (1, 0), value=0)

and documents its result as ``(num_segments + 1,) int32``. The export reported
int64 -- whatever flowed in -- because only tensor CONSTRUCTORS read a ``dtype:``
detail, and the one here is spelled through a local bound to a conditional.

Two things that look similar but are not:

* ``torch.jit.is_tracing()`` is False in the forward we document. That is a fact
  about what is being modelled, not a guess about what a caller passes -- the
  distinction that makes this sound where inferring a parameter's value from its
  default was not (see ``_EAGER_FALSE_PREDICATES``).
* ``x.to(inputs_embeds.device)`` names a DEVICE. Recording it as the target dtype
  made a pure device move report a dtype change and survive cast elision.
"""

from __future__ import annotations

import ast
import textwrap

from TraceLens.ModelUtils.ast_analyze import (
    _literal_dtype_token,
    _names_a_device,
    analyze_source,
)


def _expr(source: str) -> ast.expr:
    return ast.parse(source, mode="eval").body


class TestLiteralDtypeToken:
    def test_a_plain_torch_dtype(self) -> None:
        assert _literal_dtype_token(_expr("torch.int32")) == "torch.int32"

    def test_the_tracing_ternary_takes_the_eager_arm(self) -> None:
        assert (
            _literal_dtype_token(
                _expr("grid.dtype if torch.jit.is_tracing() else torch.int32")
            )
            == "torch.int32"
        )

    def test_a_scripting_guard_reads_the_same_way(self) -> None:
        assert (
            _literal_dtype_token(
                _expr("x.dtype if torch.jit.is_scripting() else torch.bool")
            )
            == "torch.bool"
        )

    def test_another_tensors_dtype_names_none(self) -> None:
        """``x.dtype`` points at a dtype; it does not name one."""
        assert _literal_dtype_token(_expr("x.dtype")) is None

    def test_an_unknown_condition_is_not_resolved(self) -> None:
        assert _literal_dtype_token(_expr("a if flag else torch.int32")) is None


class TestNamesADevice:
    def test_a_device_attribute(self) -> None:
        assert _names_a_device(_expr("inputs_embeds.device"))

    def test_a_device_constructor(self) -> None:
        assert _names_a_device(_expr('torch.device("cuda")'))

    def test_a_device_string(self) -> None:
        assert _names_a_device(_expr('"cuda"'))

    def test_a_dtype_is_not_a_device(self) -> None:
        assert not _names_a_device(_expr("torch.int32"))
        assert not _names_a_device(_expr("x.dtype"))


def _details(source: str, label_fragment: str) -> tuple[str, ...]:
    analysis = analyze_source(textwrap.dedent(source), config={"hidden_size": 8})
    for op in analysis.class_registry["M"].forward_operations.values():
        if label_fragment in (op.label or "").lower():
            return tuple(op.details or ())
    raise AssertionError(f"no {label_fragment!r} op found")


class TestWhatTheOpRecords:
    def test_a_cumsum_records_its_declared_dtype(self) -> None:
        details = _details(
            """
            class M(nn.Module):
                def forward(self, grid_thw):
                    dtype = grid_thw.dtype if torch.jit.is_tracing() else torch.int32
                    seqlens = grid_thw[:, 1] * grid_thw[:, 2]
                    return seqlens.cumsum(dim=0, dtype=dtype)
            """,
            "sum",
        )
        assert "dtype: torch.int32" in details, details
