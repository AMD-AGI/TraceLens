###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for semantic_analyses/build_semantic_labels.py.

Uses in-memory dict fixtures (no file I/O) to exercise the deterministic
labeling logic: positional block numbering, the layer-cycle detection and
the region (pre / body / post / secondary) helpers.
"""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SEM = os.path.join(REPO_ROOT, "TraceLens", "Agent", "Analysis", "semantic_analyses")
sys.path.insert(0, SEM)

import build_semantic_labels


# --------------------------------------------------------------------------- #
# _build_cycle_names
# --------------------------------------------------------------------------- #
def test_build_cycle_names_nonpositive_period():
    rle = [("GEMM", 1, [0], ["T"])]
    assert build_semantic_labels._build_cycle_names(rle, 0, {}) == []


def test_build_cycle_names_uses_global_counter():
    rle = [
        ("GEMM", 1, [0], ["T"]),
        ("Normalization", 1, [1], ["T"]),
        ("SDPA", 1, [2], ["T"]),
    ]
    counter = {"GEMM": 1}  # pretend one GEMM was already numbered
    names = build_semantic_labels._build_cycle_names(rle, 3, counter)
    assert names == ["GEMM_1", "Normalization_0", "SDPA_0"]
    assert counter == {"GEMM": 2, "Normalization": 1, "SDPA": 1}


# --------------------------------------------------------------------------- #
# _build_region_block_names
# --------------------------------------------------------------------------- #
def test_build_region_block_names_empty():
    assert build_semantic_labels._build_region_block_names(set(), {}, {}) == {}


def test_build_region_block_names_groups_consecutive():
    cls_by_idx = {
        0: {"perf_category": "GEMM"},
        1: {"perf_category": "GEMM"},
        2: {"perf_category": "Normalization"},
        # index 3 intentionally absent -> defaults to "Others"
    }
    counter = {}
    result = build_semantic_labels._build_region_block_names(
        {0, 1, 2, 3}, cls_by_idx, counter
    )
    assert result == {
        0: "GEMM_0",
        1: "GEMM_0",
        2: "Normalization_0",
        3: "Others_0",
    }
    assert counter == {"GEMM": 1, "Normalization": 1, "Others": 1}


# --------------------------------------------------------------------------- #
# build_labels (integration over the pure labeling path)
# --------------------------------------------------------------------------- #
def _classified(cats):
    return {
        "classified_kernels": [
            {"index": i, "perf_category": cat, "kernel_type": "T"}
            for i, cat in enumerate(cats)
        ]
    }


def _extracted(n, total_time=123.456, graph_mode=True):
    return {
        "source_file": "trace.json",
        "metadata": {
            "total_kernel_time_us": total_time,
            "is_graph_mode": graph_mode,
        },
        "kernels": [
            {"name": f"k{i}", "dur": float(i + 1), "gpu_op_uid": f"raw{i}"}
            for i in range(n)
        ],
    }


def test_build_labels_positional_labels_and_layers():
    # 0,1 = preamble ; 2..13 = body (G,N,S repeated x4) ; 14,15 = epilogue ; 16 = secondary
    cats = (
        ["GEMM", "GEMM"]
        + ["GEMM", "Normalization", "SDPA"] * 4
        + ["Normalization", "Normalization"]
        + ["GEMM"]
    )
    extracted = _extracted(len(cats))
    classified = _classified(cats)
    pattern = {
        "preamble_indices": [0, 1],
        "epilogue_indices": [14, 15],
        "secondary_stream_indices": [16],
    }
    result = build_semantic_labels.build_labels(extracted, classified, pattern)

    info = result["model_info"]
    assert info["period"] == 3
    assert info["num_layers"] == 4
    assert info["graph_mode"] is True
    assert result["total_kernel_time_us"] == 123.46
    assert result["source_file"] == "trace.json"

    kernels = result["labeled_kernels"]

    # Preamble region: GEMM_0 (numbered before the body cycle).
    assert kernels[0]["region"] == "pre"
    assert kernels[0]["layer"] is None
    assert kernels[0]["semantic_block"] == "GEMM_0"

    # Body cycle: GEMM in the body is GEMM_1 (global counter continues).
    assert kernels[2]["region"] == "body"
    assert kernels[2]["layer"] == 0
    assert kernels[2]["semantic_block"] == "GEMM_1"
    assert kernels[3]["semantic_block"] == "Normalization_0"
    assert kernels[4]["semantic_block"] == "SDPA_0"

    # Positional labels repeat across layers.
    assert kernels[5]["semantic_block"] == "GEMM_1"
    assert kernels[5]["layer"] == 1
    assert kernels[13]["semantic_block"] == "SDPA_0"
    assert kernels[13]["layer"] == 3

    # Epilogue + secondary reuse the global counter (unique numbering).
    assert kernels[14]["region"] == "post"
    assert kernels[14]["semantic_block"] == "Normalization_1"
    assert kernels[16]["region"] == "secondary"
    assert kernels[16]["semantic_block"] == "GEMM_2"

    # Enrichment fields are always empty (no trace-tree build); gpu_op_uid
    # comes straight from the raw-index UID stamped by extract_trace_data.
    assert kernels[2]["nn_module"] == ""
    assert kernels[2]["cpu_op"] == ""
    assert kernels[2]["input_dims"] == []
    assert kernels[2]["gpu_op_uid"] == "raw2"
    assert kernels[4]["nn_module"] == ""
    assert kernels[4]["gpu_op_uid"] == "raw4"
    assert kernels[3]["nn_module"] == ""
    assert kernels[3]["cpu_op"] == ""
    assert kernels[3]["gpu_op_uid"] == "raw3"


def test_build_labels_no_body_period_zero():
    cats = ["GEMM", "GEMM", "Normalization"]
    extracted = {"kernels": [{"name": f"k{i}", "dur": 1.0} for i in range(3)]}
    classified = _classified(cats)
    pattern = {
        "preamble_indices": [0, 1, 2],
        "epilogue_indices": [],
        "secondary_stream_indices": [],
    }

    result = build_semantic_labels.build_labels(extracted, classified, pattern)

    info = result["model_info"]
    assert info["period"] == 0
    assert info["num_layers"] == 0
    assert info["graph_mode"] is False
    assert result["total_kernel_time_us"] == 0
    assert result["source_file"] == ""

    kernels = result["labeled_kernels"]
    assert [k["semantic_block"] for k in kernels] == [
        "GEMM_0",
        "GEMM_0",
        "Normalization_0",
    ]
    assert all(k["region"] == "pre" for k in kernels)
    assert all(k["layer"] is None for k in kernels)
    # No tree context and no raw uid -> gpu_op_uid is None.
    assert kernels[0]["gpu_op_uid"] is None
    assert kernels[0]["nn_module"] == ""
