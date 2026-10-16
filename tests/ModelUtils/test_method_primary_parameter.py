###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""A method's primary TENSOR parameter is a tensor.

The primary parameter is whatever the expansion docks its input boundary on, and
it used to be "the first parameter after ``self``" with nothing asked about what
that parameter holds. DeepSeek's KV cache names its entry first::

    def store_compression_weights(self, name: str, kv, gate):
        first_window_position = self.entry_count[name] * self.compress_rate
        buffered_kv, buffered_gate = self.buffer_kv[name], self.buffer_gate[name]
        usable = (kv.shape[1] // self.compress_rate) * self.compress_rate
        return kv[:, :usable], gate[:, :usable], first_window_position

so ``name`` -- a ``str`` -- became the primary, the real tensors ``kv`` and
``gate`` were demoted to secondary params, and ``kv[:, :usable]`` had no base to
dock on. It reported ``[B, usable, 6]`` where the model returns
``[B, usable, 256]``; the ``6`` is ``num_experts_per_tok``, left there by a MoE
fallback that fires when an index op has no integer index operand.

The same annotation that settles it is the one already trusted for ``q_length:
int``: the model states the type itself, and a bare ``str`` is unambiguous in a
way an unannotated parameter is not. A string-keyed lookup like
``self.entry_count[name]`` is then host bookkeeping rather than a tensor index,
so it stops being emitted as one.
"""

from __future__ import annotations

import ast
import textwrap

from TraceLens.ModelUtils.ast_analyze import (
    _host_scalar_param_names,
    _primary_forward_input_name,
)

_SOURCE = textwrap.dedent("""
    class Cache:
        def store_compression_weights(self, name: str, kv, gate):
            first_window_position = self.entry_count[name] * self.compress_rate
            usable = (kv.shape[1] // self.compress_rate) * self.compress_rate
            return kv[:, :usable], gate[:, :usable], first_window_position

        def plain(self, hidden_states, scale: float):
            return hidden_states * scale

        def every_parameter_is_a_scalar(self, start: int, stop: int):
            return stop - start
""")


def _method(name: str) -> ast.FunctionDef:
    tree = ast.parse(_SOURCE)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    return next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name
    )


class TestWhichParameterLeads:
    def test_a_str_first_parameter_is_skipped(self) -> None:
        """``name: str`` names the cache entry; ``kv`` carries the data."""
        assert _primary_forward_input_name(_method("store_compression_weights")) == "kv"

    def test_an_unannotated_first_parameter_still_leads(self) -> None:
        assert _primary_forward_input_name(_method("plain")) == "hidden_states"

    def test_a_method_with_no_tensor_parameter_at_all(self) -> None:
        """Nothing to dock on is reported as nothing, not as a scalar."""
        assert (
            _primary_forward_input_name(_method("every_parameter_is_a_scalar")) is None
        )


class TestWhatCountsAsAHostValue:
    def test_str_joins_the_numbers(self) -> None:
        names = _host_scalar_param_names(_method("store_compression_weights"))
        assert "name" in names
        assert "kv" not in names and "gate" not in names

    def test_the_numeric_annotations_still_count(self) -> None:
        assert _host_scalar_param_names(_method("plain")) == {"scale"}
        assert _host_scalar_param_names(_method("every_parameter_is_a_scalar")) == {
            "start",
            "stop",
        }
