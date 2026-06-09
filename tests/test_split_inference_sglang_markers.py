###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Tests for sglang ``step[...]`` annotation parsing in the inference trace splitter.

Regression coverage for issue #723: the splitter previously only recognized
vLLM-style ``execute_..._context_..._generation_...`` iteration markers and
crashed on sglang traces whose per-step ``user_annotation`` markers are named
``step[DECODE bs=N]`` / ``step[EXTEND bs=N toks=M]``.
"""

import pytest

from TraceLens.TraceUtils.split_inference_trace_annotation import (
    SGLANG_STEP_PATTERN,
    _parse_sglang_step_name,
    find_phase_from_window,
    get_iter_details_from_name,
)


# ---------------------------------------------------------------------------
# Marker matching (find_events_by_pattern uses pattern.match, anchored at start)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "step[DECODE bs=64]",
        "step[EXTEND bs=4 toks=512]",
        "step[ EXTEND bs=4 toks=512 ]",
        "step[MIXED bs=8 toks=100]",
        "step[IDLE bs=0]",
    ],
)
def test_sglang_step_pattern_matches_step_markers(name):
    assert any(p.match(name) for p in SGLANG_STEP_PATTERN), name


@pytest.mark.parametrize(
    "name",
    [
        "execute_context_5(10)_generation_3(8)",
        "ProfilerStep#3",
        "vllm/v1/worker/gpu_model_runner.py(123): _dummy_run",
        "nn.Module: FusedMoE",
    ],
)
def test_sglang_step_pattern_ignores_non_sglang_names(name):
    assert not any(p.match(name) for p in SGLANG_STEP_PATTERN), name


# ---------------------------------------------------------------------------
# _parse_sglang_step_name grammar
# ---------------------------------------------------------------------------


def test_parse_decode_step():
    details = _parse_sglang_step_name("step[DECODE bs=64]")
    assert details == {
        "batch_size": 64,
        "num_requests": 64,
        "context_requests": 0,
        "context_sum": 0,
        "generation_requests": 64,
        "generation_sum": 64,
    }


def test_parse_extend_step_with_tokens():
    details = _parse_sglang_step_name("step[EXTEND bs=4 toks=512]")
    assert details == {
        "batch_size": 512,
        "num_requests": 4,
        "context_requests": 4,
        "context_sum": 512,
        "generation_requests": 0,
        "generation_sum": 0,
    }


def test_parse_extend_step_without_tokens_falls_back_to_bs():
    details = _parse_sglang_step_name("step[EXTEND bs=4]")
    assert details["context_requests"] == 4
    assert details["context_sum"] == 4
    assert details["generation_requests"] == 0


def test_parse_mixed_step_carries_both_phases():
    details = _parse_sglang_step_name("step[MIXED bs=8 toks=100]")
    assert details["context_requests"] == 8
    assert details["context_sum"] == 100
    assert details["generation_requests"] == 8
    assert details["generation_sum"] == 8


def test_parse_idle_step_is_empty():
    details = _parse_sglang_step_name("step[IDLE bs=0]")
    assert details["num_requests"] == 0
    assert details["batch_size"] == 0


def test_parse_is_case_insensitive_for_mode():
    assert _parse_sglang_step_name("step[decode bs=2]")["generation_requests"] == 2


@pytest.mark.parametrize(
    "name",
    [
        "execute_context_5(10)_generation_3(8)",
        "ProfilerStep#3",
        "completely unrelated name",
    ],
)
def test_parse_returns_none_for_non_sglang(name):
    assert _parse_sglang_step_name(name) is None


# ---------------------------------------------------------------------------
# get_iter_details_from_name dispatches sglang first, vLLM fallback unchanged
# ---------------------------------------------------------------------------


def test_get_iter_details_uses_sglang_branch():
    assert get_iter_details_from_name("step[DECODE bs=64]") == _parse_sglang_step_name(
        "step[DECODE bs=64]"
    )


def test_get_iter_details_vllm_grammar_still_parses():
    """vLLM names must still flow through the legacy parser (no regression)."""
    details = get_iter_details_from_name("execute_context_5(10)_generation_3(8)")
    assert details == {
        "batch_size": 18,
        "num_requests": 8,
        "context_requests": 5,
        "context_sum": 10,
        "generation_requests": 3,
        "generation_sum": 8,
    }


# ---------------------------------------------------------------------------
# Phase classification over a window of sglang steps
# ---------------------------------------------------------------------------


def test_find_phase_from_window_classifies_sglang_steps():
    names = (
        ["step[DECODE bs=64]"] * 3
        + ["step[EXTEND bs=4 toks=512]"] * 2
        + ["step[MIXED bs=8 toks=100]"]
    )
    iter_details = [get_iter_details_from_name(n) for n in names]
    phase = find_phase_from_window(iter_details)
    assert phase["num_decode"] == 3
    assert phase["num_prefill"] == 2
    assert phase["num_prefilldecode"] == 1
