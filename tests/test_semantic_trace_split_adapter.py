###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for semantic_analyses/trace_split_adapter pure helpers.

Covers the deterministic phase/metadata mapping used to convert TraceUtils
annotation-split output into the region_metadata shape expected by
extract_trace_data. The subprocess driver ``split_vllm_trace`` is excluded
from coverage (it shells out to TraceLens.TraceUtils.split_inference_trace_
annotation and is exercised only in the end-to-end path).
"""

import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SEM = os.path.join(REPO_ROOT, "TraceLens", "Agent", "Analysis", "semantic_analyses")
sys.path.insert(0, SEM)

import trace_split_adapter as tsa  # noqa: E402


# --------------------------------------------------------------------------- #
# get_steady_state_key
# --------------------------------------------------------------------------- #
def test_steady_state_key_prefill_only():
    meta = {"context_requests": 4, "generation_requests": 0, "context_sum": 128}
    assert tsa.get_steady_state_key(meta) == "prefill_only_128"


def test_steady_state_key_decode_only():
    meta = {"context_requests": 0, "generation_requests": 8, "generation_sum": 64}
    assert tsa.get_steady_state_key(meta) == "decode_only_64"


def test_steady_state_key_prefill_decode():
    meta = {
        "context_requests": 2,
        "generation_requests": 3,
        "context_sum": 100,
        "generation_sum": 50,
    }
    assert tsa.get_steady_state_key(meta) == "prefill_decode_100_50"


def test_steady_state_key_fallback_uses_batch():
    # ctx == 0 and gen == 0 -> batch/batch fallback branch
    meta = {"context_requests": 0, "generation_requests": 0, "batch_size": 16}
    assert tsa.get_steady_state_key(meta) == "prefill_decode_16_16"


def test_steady_state_key_defaults_when_empty():
    assert tsa.get_steady_state_key({}) == "prefill_decode_0_0"


# --------------------------------------------------------------------------- #
# _phase_to_region_meta
# --------------------------------------------------------------------------- #
def test_phase_to_region_meta_prefill_only():
    phase = {
        "num_prefill": 5,
        "num_prefilldecode": 0,
        "num_decode": 0,
        "avg_bs": 32,
        "avg_conc": 5,
    }
    out = tsa._phase_to_region_meta(phase)
    assert out == {
        "context_requests": 5,
        "generation_requests": 0,
        "context_sum": 32,
        "generation_sum": 0,
        "batch_size": 32,
        "num_requests": 5,
    }


def test_phase_to_region_meta_decode_only():
    phase = {
        "num_prefill": 0,
        "num_prefilldecode": 0,
        "num_decode": 7,
        "avg_bs": 8,
        "avg_conc": 7,
    }
    out = tsa._phase_to_region_meta(phase)
    assert out == {
        "context_requests": 0,
        "generation_requests": 7,
        "context_sum": 0,
        "generation_sum": 8,
        "batch_size": 8,
        "num_requests": 7,
    }


def test_phase_to_region_meta_combined():
    phase = {
        "num_prefill": 2,
        "num_prefilldecode": 3,
        "num_decode": 4,
        "avg_bs": 10,
        "avg_conc": 9,
    }
    out = tsa._phase_to_region_meta(phase)
    assert out["context_requests"] == 5  # num_prefill + num_prefilldecode
    assert out["generation_requests"] == 7  # num_decode + num_prefilldecode
    assert out["context_sum"] == 10
    assert out["generation_sum"] == 10
    assert out["batch_size"] == 10
    assert out["num_requests"] == 9


def test_phase_to_region_meta_defaults_empty():
    # empty phase -> falls through to combined branch with all zeros
    out = tsa._phase_to_region_meta({})
    assert out["context_requests"] == 0
    assert out["generation_requests"] == 0
    assert out["batch_size"] == 0


# --------------------------------------------------------------------------- #
# _is_single_iteration
# --------------------------------------------------------------------------- #
def test_is_single_iteration_true():
    assert tsa._is_single_iteration({"num_prefill": 1}) is True
    assert tsa._is_single_iteration({}) is True  # total 0


def test_is_single_iteration_false():
    assert tsa._is_single_iteration({"num_prefill": 1, "num_decode": 2}) is False


# --------------------------------------------------------------------------- #
# _iter_type_key
# --------------------------------------------------------------------------- #
def test_iter_type_key_prefill():
    assert tsa._iter_type_key({"num_prefill": 1, "avg_bs": 4}) == "prefill_4"


def test_iter_type_key_prefilldecode():
    assert (
        tsa._iter_type_key({"num_prefilldecode": 1, "avg_bs": 2}) == "prefilldecode_2"
    )


def test_iter_type_key_decode():
    assert tsa._iter_type_key({"num_decode": 1, "avg_bs": 8}) == "decode_8"


def test_iter_type_key_empty():
    assert tsa._iter_type_key({"avg_bs": 3}) == "empty_3"


# --------------------------------------------------------------------------- #
# _load_trace (thin wrapper over _helpers.load_json)
# --------------------------------------------------------------------------- #
def test_load_trace_reads_plain_json(tmp_path):
    p = tmp_path / "trace.json"
    p.write_text(json.dumps({"traceEvents": [{"name": "a"}]}))
    out = tsa._load_trace(str(p))
    assert out == {"traceEvents": [{"name": "a"}]}
