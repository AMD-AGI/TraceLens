###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for semantic_analyses/annotation_metadata.py."""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SEM = os.path.join(REPO_ROOT, "TraceLens", "Agent", "Analysis", "semantic_analyses")
sys.path.insert(0, SEM)

import annotation_metadata


# ---------------------------------------------------------------------------
# parse_filename_metadata
# ---------------------------------------------------------------------------
def test_parse_filename_all_fields_with_path():
    path = "/data/traces/mi355_tp2_isl1024_osl8_conc4_opt.pt.trace.json.gz"
    result = annotation_metadata.parse_filename_metadata(path)
    assert result["isl"] == 1024
    assert result["osl"] == 8
    assert result["conc"] == 4
    assert result["tp"] == 2
    assert result["num_tokens_prefill"] == 1024 * 4
    assert result["num_tokens_decode"] == 4


def test_parse_filename_all_fields_no_slash():
    result = annotation_metadata.parse_filename_metadata("tp1_isl16_osl2_conc3.json")
    assert result["isl"] == 16
    assert result["osl"] == 2
    assert result["conc"] == 3
    assert result["tp"] == 1
    assert result["num_tokens_prefill"] == 48
    assert result["num_tokens_decode"] == 3


def test_parse_filename_no_fields():
    assert annotation_metadata.parse_filename_metadata("randomfile.txt") == {}


def test_parse_filename_conc_only_no_isl():
    result = annotation_metadata.parse_filename_metadata("run_conc5.json")
    assert result["conc"] == 5
    assert result["num_tokens_decode"] == 5
    assert "num_tokens_prefill" not in result


def test_parse_filename_isl_only_no_conc():
    result = annotation_metadata.parse_filename_metadata("run_isl99.json")
    assert result["isl"] == 99
    assert "num_tokens_prefill" not in result
    assert "num_tokens_decode" not in result


# ---------------------------------------------------------------------------
# parse_trace_input_dims
# ---------------------------------------------------------------------------
def test_parse_trace_input_dims_empty():
    assert annotation_metadata.parse_trace_input_dims([]) == {}


def test_parse_trace_input_dims_full():
    events = [
        # non-kernel event -> skipped
        {"cat": "cpu_op", "name": "unified_attention"},
        # kernel but not attention -> skipped
        {"cat": "kernel", "name": "gemm_kernel"},
        # attention kernel, q/k len == 3
        {
            "cat": "kernel",
            "name": "unified_attention",
            "args": {"Input Dims": [[5, 1, 1], [7, 1, 1]]},
        },
        # attention kernel, q/k len > 3 (uses [-3])
        {
            "cat": "kernel",
            "name": "flash_attention_fwd",
            "args": {"Input Dims": [[2, 3, 4, 5], [6, 7, 8, 9]]},
        },
        # attention kernel, dims too short -> skipped
        {
            "cat": "kernel",
            "name": "unified_attention",
            "args": {"Input Dims": [[1, 1]]},
        },
        # attention kernel, q/k not list-like -> skips nq/nkv appends
        {
            "cat": "kernel",
            "name": "unified_attention",
            "args": {"Input Dims": [5, 6]},
        },
        # attention kernel, q/k list but len < 3 -> skipped
        {
            "cat": "kernel",
            "name": "unified_attention",
            "args": {"Input Dims": [[1, 2], [3, 4]]},
        },
        # attention kernel, dims is a non-subscriptable set -> triggers except
        {
            "cat": "kernel",
            "name": "unified_attention",
            "args": {"Input Dims": {1, 2, 3}},
        },
    ]
    result = annotation_metadata.parse_trace_input_dims(events)
    # nq_values = [5, 3] -> avg = 4, count = 2
    assert result["trace_avg_nq"] == 4
    assert result["trace_attention_kernel_count"] == 2
    # nkv_values = [7, 7] -> avg = 7
    assert result["trace_avg_nkv"] == 7


def test_parse_trace_input_dims_no_dims_key():
    events = [{"cat": "kernel", "name": "unified_attention", "args": {}}]
    assert annotation_metadata.parse_trace_input_dims(events) == {}


# ---------------------------------------------------------------------------
# run_sanity_checks
# ---------------------------------------------------------------------------
def test_run_sanity_checks_all_warnings():
    warnings = annotation_metadata.run_sanity_checks(
        annotation_meta={"batch_size": 100, "context_sum": 100},
        filename_meta={"isl": 10, "conc": 100},
        trace_meta={"trace_avg_nq": 200},
        user_meta={"num_tokens": 500},
    )
    assert len(warnings) == 3
    assert any("Batch size mismatch" in w for w in warnings)
    assert any("num_tokens mismatch" in w for w in warnings)
    assert any("Context sum mismatch" in w for w in warnings)


def test_run_sanity_checks_no_warnings_matching():
    warnings = annotation_metadata.run_sanity_checks(
        annotation_meta={"batch_size": 1000, "context_sum": 100},
        filename_meta={"isl": 10, "conc": 100},
        trace_meta={"trace_avg_nq": 100},
        user_meta={"num_tokens": 1000},
    )
    assert warnings == []


def test_run_sanity_checks_empty_all():
    assert annotation_metadata.run_sanity_checks({}, {}, {}, {}) == []


def test_run_sanity_checks_filename_missing_conc():
    # isl present but conc missing -> batch_file stays None, no batch warning
    warnings = annotation_metadata.run_sanity_checks(
        annotation_meta={"batch_size": 100},
        filename_meta={"isl": 10},
        trace_meta={},
        user_meta={},
    )
    assert warnings == []


# ---------------------------------------------------------------------------
# merge_metadata
# ---------------------------------------------------------------------------
def test_merge_metadata_all_none():
    merged = annotation_metadata.merge_metadata()
    assert merged["annotation"] == {}
    assert merged["filename"] == {}
    assert merged["trace"] == {}
    assert merged["user"] == {}
    assert merged["num_tokens"] is None
    assert merged["context_length"] is None
    assert merged["batch_size"] is None
    assert merged["context_sum"] is None
    assert merged["generation_sum"] is None
    assert merged["_warnings"] == []


def test_merge_metadata_user_priority():
    merged = annotation_metadata.merge_metadata(
        user_meta={"num_tokens": 42, "context_length": 77},
    )
    assert merged["num_tokens"] == 42
    assert merged["context_length"] == 77


def test_merge_metadata_annotation_and_filename_fallbacks():
    merged = annotation_metadata.merge_metadata(
        annotation_meta={"context_sum": 256, "generation_sum": 8},
        filename_meta={"num_tokens_prefill": 512, "isl": 128},
    )
    # num_tokens: no user/annotation batch_size -> prefill
    assert merged["num_tokens"] == 512
    # context_length: annotation context_sum wins
    assert merged["context_length"] == 256
    assert merged["batch_size"] == 512
    assert merged["context_sum"] == 256
    assert merged["generation_sum"] == 8


def test_merge_metadata_context_from_num_tokens_fallback():
    # no context sources except num_tokens (from decode)
    merged = annotation_metadata.merge_metadata(
        filename_meta={"num_tokens_decode": 4},
    )
    assert merged["num_tokens"] == 4
    assert merged["context_length"] == 4


def test_merge_metadata_emits_warning():
    merged = annotation_metadata.merge_metadata(
        annotation_meta={"batch_size": 100},
        user_meta={"num_tokens": 500},
    )
    assert merged["_warnings"]
    assert any("num_tokens mismatch" in w for w in merged["_warnings"])


# ---------------------------------------------------------------------------
# gather_metadata
# ---------------------------------------------------------------------------
def test_gather_metadata_with_user_values():
    merged = annotation_metadata.gather_metadata(
        trace_path="/x/tp1_isl1024_conc4.json",
        events=None,
        annotation_meta={"batch_size": 4096},
        num_tokens=4096,
        context_length=1024,
    )
    assert merged["num_tokens"] == 4096
    assert merged["context_length"] == 1024
    assert merged["filename"]["isl"] == 1024
    assert merged["trace"] == {}


def test_gather_metadata_no_user_values():
    merged = annotation_metadata.gather_metadata(
        trace_path="plain.json",
        events=None,
        annotation_meta=None,
        num_tokens=None,
        context_length=None,
    )
    assert merged["user"] == {}
    assert merged["num_tokens"] is None
