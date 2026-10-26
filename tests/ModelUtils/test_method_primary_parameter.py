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

vLLM/SGLang additionally break the "first non-scalar parameter" convention
outright: ``forward(self, positions, hidden_states)`` puts the position ids
before the residual stream (HuggingFace's own convention always leads with
``hidden_states``). Both are real tensors, so neither the ``str`` annotation nor
the host-scalar check demotes ``positions`` -- yet it must still lose to
``hidden_states``, which is the one actually built from a REAL submodule
(``self.self_attn(positions=positions, hidden_states=hidden_states)``) while
``positions`` only ever rides along as a bare argument, never itself read,
sliced, or operated on. DeepSeek's indexer scorer has the same *shape*
(``forward(self, q, compressed_kv, hidden_states)``, with ``hidden_states``
alone feeding a submodule) but the OPPOSITE answer: ``q.float()`` is applied
directly to ``q``, so it is not a pure passthrough and must stay primary.
"""

from __future__ import annotations

import ast
import textwrap

from TraceLens.ModelUtils.ast_analyze import (
    _host_scalar_param_names,
    _merely_forwarded_names,
    _names_touched_by_submodule_calls,
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

_POSITION_ORDER_SOURCE = textwrap.dedent("""
    class Decoder:
        def forward(self, positions, hidden_states):
            hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
            return hidden_states

    class Rope:
        def forward(self, positions, q, k):
            q, k = self.rotary_emb(q, k, positions=positions)
            return q, k

    class IndexerScorer:
        def forward(self, q, compressed_kv, hidden_states):
            scores = torch.matmul(q.float(), compressed_kv.transpose(-1, -2))
            weights = self.weights_proj(hidden_states)
            return scores * weights
""")

_DECODER_INIT_ASSIGNMENTS = {"self_attn": "Attention"}
_ROPE_INIT_ASSIGNMENTS = {"rotary_emb": "RotaryEmbedding"}
_INDEXER_INIT_ASSIGNMENTS = {"weights_proj": "Linear"}


def _method(
    name: str, source: str = _SOURCE, class_name: str | None = None
) -> ast.FunctionDef:
    tree = ast.parse(source)
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    cls = next(c for c in classes if class_name is None or c.name == class_name)
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


class TestAPurelyForwardedLeaderYieldsToARealSubmoduleReader:
    """vLLM/SGLang lead with ``positions``, not ``hidden_states``."""

    def test_positions_loses_to_the_submodule_fed_hidden_states(self) -> None:
        func = _method("forward", _POSITION_ORDER_SOURCE, "Decoder")
        assert (
            _primary_forward_input_name(func, _DECODER_INIT_ASSIGNMENTS)
            == "hidden_states"
        )

    def test_without_init_assignments_the_keyword_call_is_not_a_signal(self) -> None:
        """No ``init_assignments`` means no call can be confirmed a real submodule."""
        func = _method("forward", _POSITION_ORDER_SOURCE, "Decoder")
        assert _primary_forward_input_name(func) == "positions"

    def test_positions_itself_is_never_operated_on(self) -> None:
        """``positions`` is only ever a bare argument; ``hidden_states`` is
        reassigned, so it is not a pure passthrough either -- but it still
        wins because it is the one touching the real submodule."""
        func = _method("forward", _POSITION_ORDER_SOURCE, "Decoder")
        assert "positions" in _merely_forwarded_names(func)
        assert "hidden_states" not in _merely_forwarded_names(func)

    def test_hidden_states_is_recorded_as_touching_a_real_submodule(self) -> None:
        func = _method("forward", _POSITION_ORDER_SOURCE, "Decoder")
        touches = _names_touched_by_submodule_calls(func, _DECODER_INIT_ASSIGNMENTS)
        assert "hidden_states" in touches
        # Both arguments are passed by KEYWORD, so ``positions`` must not also
        # "qualify" and win back the tie-break on signature order alone.
        assert "positions" not in touches


class TestATupleReturningSubmoduleAlsoFollowsTheConvention:
    def test_positions_loses_to_q_which_is_reassigned_by_the_submodule(self) -> None:
        func = _method("forward", _POSITION_ORDER_SOURCE, "Rope")
        assert _primary_forward_input_name(func, _ROPE_INIT_ASSIGNMENTS) == "q"


class TestALeaderOperatedOnDirectlyIsNeverDemoted:
    """DeepSeek's indexer scorer: same shape as the convention fix, opposite answer."""

    def test_q_stays_primary_even_though_hidden_states_feeds_a_submodule(self) -> None:
        func = _method("forward", _POSITION_ORDER_SOURCE, "IndexerScorer")
        assert _primary_forward_input_name(func, _INDEXER_INIT_ASSIGNMENTS) == "q"

    def test_q_is_not_merely_forwarded(self) -> None:
        """``q.float()`` is a real use, not a bare pass-through argument."""
        func = _method("forward", _POSITION_ORDER_SOURCE, "IndexerScorer")
        assert "q" not in _merely_forwarded_names(func)
