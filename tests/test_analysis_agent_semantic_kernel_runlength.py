###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for semantic_analyses/kernel_runlength.py."""

import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SEM = os.path.join(REPO_ROOT, "TraceLens", "Agent", "Analysis", "semantic_analyses")
sys.path.insert(0, SEM)

import kernel_runlength


# ---------------------------------------------------------------------------
# load_sequence
# ---------------------------------------------------------------------------
def test_load_sequence(tmp_path):
    p = tmp_path / "semantic_labels.json"
    payload = {
        "labeled_kernels": [
            {"semantic_block": "A"},
            {"semantic_block": "B"},
            {},  # missing key -> ""
        ]
    }
    p.write_text(json.dumps(payload))
    assert kernel_runlength.load_sequence(str(p)) == ["A", "B", ""]


def test_load_sequence_no_kernels(tmp_path):
    p = tmp_path / "empty_labels.json"
    p.write_text(json.dumps({}))
    assert kernel_runlength.load_sequence(str(p)) == []


# ---------------------------------------------------------------------------
# collapse_consecutive
# ---------------------------------------------------------------------------
def test_collapse_consecutive_empty():
    assert kernel_runlength.collapse_consecutive([]) == []


def test_collapse_consecutive_basic():
    assert kernel_runlength.collapse_consecutive(["A", "A", "B", "D", "D"]) == [
        "A",
        "B",
        "D",
    ]


def test_collapse_consecutive_no_dups():
    assert kernel_runlength.collapse_consecutive(["A", "B", "C"]) == ["A", "B", "C"]


# ---------------------------------------------------------------------------
# run_index_per_kernel
# ---------------------------------------------------------------------------
def test_run_index_per_kernel_empty():
    assert kernel_runlength.run_index_per_kernel([]) == []


def test_run_index_per_kernel_basic():
    # A A B D D -> runs: A(0) A(0) B(1) D(2) D(2)
    assert kernel_runlength.run_index_per_kernel(["A", "A", "B", "D", "D"]) == [
        0,
        0,
        1,
        2,
        2,
    ]


# ---------------------------------------------------------------------------
# shared_neighbor_windows_skip_non_shared
# ---------------------------------------------------------------------------
def test_shared_neighbor_windows_basic():
    #            0    1    2    3    4    5    6
    condensed = ["S1", "x", "S2", "C", "S3", "y", "S4"]
    shared = {"S1", "S2", "S3", "S4"}
    left, right = kernel_runlength.shared_neighbor_windows_skip_non_shared(
        condensed, center_j=3, shared=shared, radius=2
    )
    # left: walk from idx 2 -> S2 (shared), idx 1 x (skip), idx 0 S1 (shared)
    assert left == ["S2", "S1"]
    # right: idx 4 S3 (shared), idx 5 y (skip), idx 6 S4 (shared)
    assert right == ["S3", "S4"]


def test_shared_neighbor_windows_radius_limit():
    condensed = ["S1", "S2", "C", "S3", "S4"]
    shared = {"S1", "S2", "S3", "S4"}
    left, right = kernel_runlength.shared_neighbor_windows_skip_non_shared(
        condensed, center_j=2, shared=shared, radius=1
    )
    assert left == ["S2"]
    assert right == ["S3"]


def test_shared_neighbor_windows_boundaries():
    condensed = ["C", "S1"]
    shared = {"S1"}
    left, right = kernel_runlength.shared_neighbor_windows_skip_non_shared(
        condensed, center_j=0, shared=shared, radius=3
    )
    assert left == []
    assert right == ["S1"]
