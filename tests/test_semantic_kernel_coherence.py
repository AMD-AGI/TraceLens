###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for semantic_analyses/kernel_coherence.py deterministic core."""

import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SEM = os.path.join(REPO_ROOT, "TraceLens", "Agent", "Analysis", "semantic_analyses")
sys.path.insert(0, SEM)

import kernel_coherence  # noqa: E402
import kernel_runlength  # noqa: E402

# ---------------------------------------------------------------------------
# _load / _dims_repr
# ---------------------------------------------------------------------------


def test_load_reads_json(tmp_path):
    p = tmp_path / "doc.json"
    p.write_text(json.dumps({"labeled_kernels": []}))
    assert kernel_coherence._load(str(p)) == {"labeled_kernels": []}


def test_dims_repr_empty_and_compact_and_truncate():
    assert kernel_coherence._dims_repr(None) == ""
    assert kernel_coherence._dims_repr([[1, 2]]) == "[[1,2]]"
    long = kernel_coherence._dims_repr([[i] for i in range(200)], limit=10)
    assert long.endswith("...")
    assert len(long) == 13


# ---------------------------------------------------------------------------
# _run_to_kernel_indices
# ---------------------------------------------------------------------------


def test_run_to_kernel_indices():
    mapping = kernel_coherence._run_to_kernel_indices([0, 0, 1, 2, 2])
    assert dict(mapping) == {0: [0, 1], 1: [2], 2: [3, 4]}


# ---------------------------------------------------------------------------
# _symbol_evidence
# ---------------------------------------------------------------------------


def _kernels_for_evidence():
    return [
        {"name": "gemm", "dur": 10.0, "perf_category": "GEMM", "input_dims": [[1, 2]]},
        {"name": "gemm", "dur": 5.0, "perf_category": "GEMM"},
        {
            "name": "elem",
            "dur": 3.0,
            "perf_category": "Elementwise",
            "input_dims": [[9]],
        },
    ]


def test_symbol_evidence_ranks_and_aggregates():
    kernels = _kernels_for_evidence()
    cats, dims, top = kernel_coherence._symbol_evidence(kernels, [0, 1, 2], 5)
    assert cats == ["Elementwise", "GEMM"]
    assert dims == "[[1,2]]"  # first non-empty input_dims
    assert top[0] == {"kernel_name": "gemm", "total_us": 15.0, "kernel_count": 2}
    assert top[1] == {"kernel_name": "elem", "total_us": 3.0, "kernel_count": 1}


def test_symbol_evidence_top_kernels_limit():
    kernels = _kernels_for_evidence()
    _, _, top = kernel_coherence._symbol_evidence(kernels, [0, 1, 2], 1)
    assert [t["kernel_name"] for t in top] == ["gemm"]


# ---------------------------------------------------------------------------
# _collect_contexts
# ---------------------------------------------------------------------------


def test_collect_contexts_unique_neighbor_contexts():
    seq = ["A", "X", "B", "A", "X", "B", "A", "X", "C"]
    kernels = [
        {"name": f"k{i}", "semantic_block": s, "dur": float(i + 1)}
        for i, s in enumerate(seq)
    ]
    condensed = kernel_runlength.collapse_consecutive(seq)
    run_per_kernel = kernel_runlength.run_index_per_kernel(seq)
    shared = {"A", "B", "C"}
    problematic = {"X"}

    detail = kernel_coherence._collect_contexts(
        "A", kernels, condensed, run_per_kernel, shared, problematic, 1, 5
    )
    assert list(detail) == ["X"]
    contexts = detail["X"]["contexts"]
    # (A,B) appears twice but is deduped; (A,C) is distinct -> 2 contexts
    assert detail["X"]["context_count"] == 2
    assert contexts[0]["id"] == "A:0"
    assert contexts[0]["left_window"] == ["A"]
    assert contexts[0]["right_window"] == ["B"]
    assert contexts[1]["id"] == "A:1"
    assert contexts[1]["right_window"] == ["C"]
    assert contexts[0]["first_pass_block"] == "X"


# ---------------------------------------------------------------------------
# _context_lookup
# ---------------------------------------------------------------------------


def test_context_lookup_builds_key_map():
    catalog = [
        {
            "id": "A:0",
            "workload": "A",
            "first_pass_block": "X",
            "left_window": ["A"],
            "right_window": ["B"],
        },
        {
            "id": "A:1",
            "workload": "A",
            "first_pass_block": "X",
            "left_window": None,
            "right_window": ["C"],
        },
    ]
    lookup = kernel_coherence._context_lookup(catalog)
    assert lookup[("A", "X", ("A",), ("B",))] == "A:0"
    # left_window None coerced to empty tuple
    assert lookup[("A", "X", (), ("C",))] == "A:1"


# ---------------------------------------------------------------------------
# _final_blocks
# ---------------------------------------------------------------------------


def _final_kernels():
    return [
        {"name": "a", "semantic_block": "S1", "dur": 1.0, "index": 100},
        {"name": "b", "semantic_block": "P", "dur": 5.0, "index": 101},
        {"name": "c", "semantic_block": "S2", "dur": 2.0},  # no index -> uses i
    ]


def test_final_blocks_context_rename():
    kernels = _final_kernels()
    lookup = {("A", "P", ("S1",), ("S2",)): "A:0"}
    finals, audit = kernel_coherence._final_blocks(
        "A", kernels, {"S1", "S2"}, {"P"}, 1, lookup, {"A:0": "QKV"}, {}
    )
    assert finals == ["S1", "QKV", "S2"]
    assert audit[1]["context_id"] == "A:0"
    assert audit[1]["first_pass_block"] == "P"
    assert audit[1]["final_block"] == "QKV"
    assert audit[1]["kernel_index"] == 101
    # kernel without "index" falls back to positional index
    assert audit[2]["kernel_index"] == 2


def test_final_blocks_fallback_remap():
    kernels = _final_kernels()
    lookup = {("A", "P", ("S1",), ("S2",)): "A:0"}
    finals, _ = kernel_coherence._final_blocks(
        "A", kernels, {"S1", "S2"}, {"P"}, 1, lookup, {}, {"P": "FB"}
    )
    # cid found but not in renames -> fallback used
    assert finals == ["S1", "FB", "S2"]


def test_final_blocks_no_decision_keeps_first_pass():
    kernels = _final_kernels()
    finals, audit = kernel_coherence._final_blocks(
        "A", kernels, {"S1", "S2"}, {"P"}, 1, {}, {}, {}
    )
    assert finals == ["S1", "P", "S2"]
    assert audit[1]["context_id"] == ""


def test_final_blocks_missing_context_uses_fallback():
    kernels = _final_kernels()
    # empty lookup -> cid "" -> falls through to fallback
    finals, audit = kernel_coherence._final_blocks(
        "A", kernels, {"S1", "S2"}, {"P"}, 1, {}, {}, {"P": "FB"}
    )
    assert finals == ["S1", "FB", "S2"]
    assert audit[1]["context_id"] == ""


# ---------------------------------------------------------------------------
# _residual_one_sided
# ---------------------------------------------------------------------------


def test_residual_one_sided():
    labels_a = {
        "labeled_kernels": [
            {"semantic_block": "A"},
            {"semantic_block": "A"},
            {"semantic_block": "X"},
        ]
    }
    labels_b = {
        "labeled_kernels": [
            {"semantic_block": "A"},
            {"semantic_block": "Y"},
        ]
    }
    res_a, res_b = kernel_coherence._residual_one_sided(labels_a, labels_b)
    assert res_a == ["X"]
    assert res_b == ["Y"]
