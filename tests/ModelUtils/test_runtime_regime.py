###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""A forward is not one program, so the drawing has to say which run it is of.

These branches are decided by nothing in the config -- only by the call:

    if attention_mask is not None:                      # is this batch padded?
        hidden_states = index_first_axis(
            rearrange(hidden_states, "b s ... -> (b s) ..."), indices).unsqueeze(0)

    if use_precomputed_states and seq_len == 1:         # decoding one token?
        mixed_qkv = causal_conv1d_update(...)

    if cache_layer is None:                             # anything cached yet?
        ...

Both arms are real and they disagree -- Kimi's packed arm carries
``[1, nnz, H]`` where its rectangular arm carries ``[B, S, H]`` -- so drawing
both leaves every shape below true of only one of them. The run is stated
instead, and decided AT the branch, so the arm not taken is never walked and
nothing downstream carries a value from a run this graph is not of.
"""

from __future__ import annotations

import ast

import pytest

from TraceLens.ModelUtils.ast_analyze import (
    RUNTIME_REGIME_KEY,
    RuntimeRegime,
    _ForwardOperationExtractor,
    _merges_leading_axes,
    _regime_facts,
)
from TraceLens.ModelUtils.extract import (
    _with_config_overrides,
    parse_config_overrides,
)

FORWARD = """
def forward(self, hidden_states, attention_mask=None, past_key_values: Cache | None = None):
    batch_size, seq_len, _ = hidden_states.shape
    cache_layer = past_key_values.layers[self.layer_idx] if past_key_values is not None else None
    if attention_mask is not None:
        indices, cu_seqlens, _ = get_unpad_data(attention_mask)
        hidden_states = index_first_axis(
            rearrange(hidden_states, "b s ... -> (b s) ..."), indices).unsqueeze(0)
    if self.training:
        hidden_states = self.dropout(hidden_states)
    if seq_len == 1:
        out = self.recurrent(hidden_states)
    if cache_layer is None:
        out = self.plain(hidden_states)
    return out
"""


def _func(source: str = FORWARD) -> ast.FunctionDef:
    return ast.parse(source).body[0]


def _extractor(regime: RuntimeRegime, func: ast.FunctionDef | None = None):
    func = func or _func()
    extractor = _ForwardOperationExtractor(
        self_values={},
        all_tensor_ops=False,
        config={RUNTIME_REGIME_KEY: regime},
    )
    extractor._function_root = func
    extractor._regime_facts = _regime_facts(func)
    return extractor


def _test_of(func: ast.FunctionDef, index: int) -> ast.expr:
    return [node for node in ast.walk(func) if isinstance(node, ast.If)][index].test


class TestWhatTheForwardItselfSays:
    def test_the_cache_is_the_thing_the_model_declares_as_one(self) -> None:
        # By its own annotation, not by a list of blessed parameter names.
        assert "past_key_values" in _regime_facts(_func()).cache_names

    def test_a_local_taken_out_of_the_cache_is_cache_too(self) -> None:
        assert "cache_layer" in _regime_facts(_func()).cache_names

    def test_the_sequence_axis_local_is_known(self) -> None:
        assert "seq_len" in _regime_facts(_func()).sequence_names

    def test_the_packing_branch_is_found_by_its_own_pattern(self) -> None:
        assert _regime_facts(_func()).layout_tests == {"attention_mask is not None"}

    def test_a_pattern_that_packs(self) -> None:
        assert _merges_leading_axes("b s ... -> (b s) ...")

    def test_a_pattern_that_merely_reorders_does_not(self) -> None:
        assert not _merges_leading_axes("b t h d -> b t (h d)")

    def test_a_pattern_that_splits_does_not(self) -> None:
        assert not _merges_leading_axes("... (h d) -> ... h d")


class TestEachChoicePicksExactlyOneArm:
    @pytest.mark.parametrize(
        "layout, taken", [("rectangular", False), ("packed", True)]
    )
    def test_the_layout_decides_the_packing_branch(self, layout, taken) -> None:
        func = _func()
        extractor = _extractor(RuntimeRegime(sequence_layout=layout), func)
        assert extractor._regime_outcome(_test_of(func, 0)) is taken

    @pytest.mark.parametrize("mode, taken", [("inference", False), ("training", True)])
    def test_the_mode_decides_self_training(self, mode, taken) -> None:
        func = _func()
        extractor = _extractor(RuntimeRegime(mode=mode), func)
        assert extractor._regime_outcome(_test_of(func, 1)) is taken

    @pytest.mark.parametrize("phase, taken", [("prefill", False), ("decode", True)])
    def test_the_phase_decides_a_single_token(self, phase, taken) -> None:
        func = _func()
        extractor = _extractor(RuntimeRegime(phase=phase), func)
        assert extractor._regime_outcome(_test_of(func, 2)) is taken

    @pytest.mark.parametrize("cache, absent", [("cold", True), ("warm", False)])
    def test_the_cache_decides_what_is_stored(self, cache, absent) -> None:
        func = _func()
        extractor = _extractor(RuntimeRegime(cache=cache), func)
        # `cache_layer is None` -- true exactly when no cache was handed in.
        assert extractor._regime_outcome(_test_of(func, 3)) is absent

    def test_decoding_always_reads_a_cache(self) -> None:
        assert RuntimeRegime(phase="decode", cache="cold").has_cache


class TestATestIsDecidedWhereverTheLanguageWouldDecideIt:
    """Short-circuit, so one known part can settle a test with unknown ones.

    GPT-2 guards its cache update with ``(past_key_values is not None and not
    is_cross_attention) or (past_key_values is not None and is_cross_attention
    and not is_updated)``. With no cache BOTH groups are False whatever
    ``is_cross_attention`` says -- demanding every part be known left it
    undecided and drew a phi over a branch that cannot run.
    """

    GPT2 = (
        "(past_key_values is not None and not is_cross_attention) or "
        "(past_key_values is not None and is_cross_attention and not is_updated)"
    )

    def _outcome(self, expression: str, regime: RuntimeRegime):
        func = _func(
            "def forward(self, hidden_states, past_key_values: Cache | None = None):\n"
            f"    if {expression}:\n        out = self.f(hidden_states)\n    return out\n"
        )
        return _extractor(regime, func)._regime_outcome(_test_of(func, 0))

    def test_a_false_part_settles_an_and(self) -> None:
        assert self._outcome(self.GPT2, RuntimeRegime(cache="cold")) is False

    def test_an_unknown_part_still_blocks_a_true_and(self) -> None:
        assert (
            self._outcome(
                "past_key_values is not None and not is_cross_attention",
                RuntimeRegime(cache="warm"),
            )
            is None
        )

    def test_a_true_part_settles_an_or(self) -> None:
        assert (
            self._outcome(
                "past_key_values is not None or something_unknowable",
                RuntimeRegime(cache="warm"),
            )
            is True
        )

    def test_a_negation_inverts_what_is_known(self) -> None:
        assert (
            self._outcome(
                "not past_key_values is not None", RuntimeRegime(cache="cold")
            )
            is True
        )

    def test_a_test_about_nothing_in_the_regime_stays_open(self) -> None:
        assert self._outcome("output_attentions", RuntimeRegime()) is None


class TestTheOverridesAreTheLastWord:
    def test_values_are_read_as_the_config_would(self) -> None:
        assert parse_config_overrides(
            ["num_hidden_layers=4", "tie=true", "name=bfloat16", "dims=[1, 2]"]
        ) == {"num_hidden_layers": 4, "tie": True, "name": "bfloat16", "dims": [1, 2]}

    def test_left_to_right_keeps_the_last(self, caplog) -> None:
        with caplog.at_level("WARNING"):
            assert parse_config_overrides(["a=1", "a=2"]) == {"a": 2}
        assert "overridden more than once" in caplog.text

    def test_repeating_the_same_value_is_not_worth_a_warning(self, caplog) -> None:
        with caplog.at_level("WARNING"):
            parse_config_overrides(["a=1", "a=1"])
        assert "overridden more than once" not in caplog.text

    def test_a_missing_equals_is_refused(self) -> None:
        with pytest.raises(ValueError):
            parse_config_overrides(["just_a_key"])

    def test_a_dotted_key_reaches_a_nested_config(self) -> None:
        out = _with_config_overrides(
            {"vision_config": {"hidden_size": 768}, "hidden_size": 4096},
            {"vision_config.hidden_size": 1024},
        )
        assert out == {"vision_config": {"hidden_size": 1024}, "hidden_size": 4096}

    def test_a_plain_key_is_written_over(self) -> None:
        assert _with_config_overrides({"a": 1}, {"a": 2}) == {"a": 2}

    def test_nothing_to_override_leaves_the_config_alone(self) -> None:
        config = {"a": 1}
        assert _with_config_overrides(config, {}) is config


KIMI_MODE = """
def forward(self, hidden_states, cache_params: KimiLinearCache | None = None):
    batch_size, q_len, _ = hidden_states.shape
    use_cache = cache_params is not None
    mode = 'fused_recurrent' if use_cache and q_len == 1 else self.mode
    if mode == 'chunk':
        o = chunk_kda(q, k, v)
    else:
        o = fused_recurrent_kda(q, k, v)
    return o
"""


class TestAConstructorConstantIsReadToo:
    """Kimi picks its KERNEL through three hops, all of them knowable.

        self.mode = "chunk"                     # __init__, a constant
        use_cache = cache_params is not None    # the cache question
        mode = 'fused_recurrent' if use_cache and q_len == 1 else self.mode
        if mode == 'chunk': chunk_kda(...) else: fused_recurrent_kda(...)

    Reading only the comparison left the choice of kernel undecided, so a
    decode graph drew the chunk kernels it does not run.
    """

    @staticmethod
    def _decide(phase: str):
        func = _func(KIMI_MODE)
        extractor = _ForwardOperationExtractor(
            self_values={"mode": "chunk"},
            all_tensor_ops=False,
            config={RUNTIME_REGIME_KEY: RuntimeRegime(phase=phase)},
        )
        extractor._function_root = func
        extractor._regime_facts = _regime_facts(func)
        return extractor._regime_outcome(_test_of(func, 0))

    def test_prefill_runs_the_chunk_kernel(self) -> None:
        assert self._decide("prefill") is True

    def test_decode_runs_the_recurrent_one(self) -> None:
        # decode: a cache is read and q_len is 1, so the conditional settles on
        # 'fused_recurrent' and `mode == 'chunk'` is false.
        assert self._decide("decode") is False


class TestTheArmsTheRunDoesNotTake:
    """The kernel-naming pass must not read an arm that cannot run.

    A forward is read more than once -- the extractor walks it for operations
    and takes the arm the run takes, while a separate parse reads the same
    function for its calls, its attention inputs and the KERNEL it runs. Only
    the first consulted the run, so that parse reported the FIRST kernel-shaped
    call in source order: Kimi was labelled ``chunk_kda`` even when decoding,
    and GLM drew its prefill convolution beside the kernel a DECODE step runs.

    Spans, not a pruned tree: every other pass keeps reading the function as
    the model wrote it. Pruning the tree instead changed what the dataflow
    passes saw, orphaned a node, and the orphan was then wired onto the
    attention as a bogus ``query/key/value`` operand carrying the pre-split
    tensor.
    """

    @staticmethod
    def _spans(phase: str):
        from TraceLens.ModelUtils.ast_analyze import _regime_skipped_spans

        return _regime_skipped_spans(
            _func(KIMI_MODE),
            config={RUNTIME_REGIME_KEY: RuntimeRegime(phase=phase)},
            self_values={"mode": "chunk"},
        )

    @staticmethod
    def _lines(source: str, spans) -> set[str]:
        from TraceLens.ModelUtils.ast_analyze import _within_skipped

        return {
            ast.unparse(node).split("(")[0]
            for node in ast.walk(_func(source))
            if isinstance(node, ast.Call) and _within_skipped(node, spans)
        }

    def test_prefill_skips_the_recurrent_call(self) -> None:
        assert "fused_recurrent_kda" in self._lines(KIMI_MODE, self._spans("prefill"))

    def test_prefill_keeps_the_chunk_call(self) -> None:
        assert "chunk_kda" not in self._lines(KIMI_MODE, self._spans("prefill"))

    def test_decode_skips_the_chunk_call(self) -> None:
        assert "chunk_kda" in self._lines(KIMI_MODE, self._spans("decode"))

    def test_decode_keeps_the_recurrent_call(self) -> None:
        assert "fused_recurrent_kda" not in self._lines(
            KIMI_MODE, self._spans("decode")
        )

    def test_an_undecided_branch_skips_nothing(self) -> None:
        from TraceLens.ModelUtils.ast_analyze import _regime_skipped_spans

        source = """
def forward(self, x):
    if self.whatever_nothing_settles:
        y = f(x)
    else:
        y = g(x)
    return y
"""
        spans = _regime_skipped_spans(
            _func(source),
            config={RUNTIME_REGIME_KEY: RuntimeRegime()},
            self_values={},
        )
        assert spans == ()
