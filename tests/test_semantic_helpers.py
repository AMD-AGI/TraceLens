###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for semantic_analyses/_helpers.py."""

import gzip
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SEM = os.path.join(REPO_ROOT, "TraceLens", "Agent", "Analysis", "semantic_analyses")
sys.path.insert(0, SEM)

from _helpers import build_rle, detect_period, load_json, load_labels


# ---------------------------------------------------------------------------
# build_rle
# ---------------------------------------------------------------------------
def test_build_rle_empty():
    assert build_rle([], {}) == []


def test_build_rle_single_group():
    cls = {
        0: {"perf_category": "Compute", "kernel_type": "gemm"},
        1: {"perf_category": "Compute", "kernel_type": "gemm"},
    }
    groups = build_rle([0, 1], cls)
    assert groups == [("Compute", 2, [0, 1], ["gemm", "gemm"])]


def test_build_rle_multiple_groups_with_defaults():
    cls = {
        0: {"perf_category": "Compute", "kernel_type": "gemm"},
        1: {"perf_category": "Compute", "kernel_type": "gemm"},
        2: {"perf_category": "Memory", "kernel_type": "copy"},
        # idx 3 missing -> defaults ("Others", "Unknown")
    }
    groups = build_rle([0, 1, 2, 3], cls)
    assert groups == [
        ("Compute", 2, [0, 1], ["gemm", "gemm"]),
        ("Memory", 1, [2], ["copy"]),
        ("Others", 1, [3], ["Unknown"]),
    ]


# ---------------------------------------------------------------------------
# detect_period
# ---------------------------------------------------------------------------
def test_detect_period_short_returns_length():
    groups = [("A",), ("B",), ("C",), ("D",), ("E",)]  # n = 5 < 6
    assert detect_period(groups) == 5


def test_detect_period_repeating():
    # ABC repeated 4 times -> n = 12, period = 3
    cats = ["A", "B", "C"] * 4
    groups = [(c,) for c in cats]
    assert detect_period(groups) == 3


def test_detect_period_no_period():
    cats = ["A", "B", "C", "D", "E", "F"]  # n = 6, no repetition
    groups = [(c,) for c in cats]
    assert detect_period(groups) == 6


def test_detect_period_empty():
    assert detect_period([]) == 0


# ---------------------------------------------------------------------------
# load_json / load_labels
# ---------------------------------------------------------------------------
def test_load_json_plain(tmp_path):
    p = tmp_path / "data.json"
    payload = {"a": 1, "b": [2, 3]}
    p.write_text(json.dumps(payload))
    assert load_json(str(p)) == payload


def test_load_json_gzip(tmp_path):
    p = tmp_path / "data.json.gz"
    payload = {"x": "y"}
    with gzip.open(str(p), "wt") as f:
        json.dump(payload, f)
    assert load_json(str(p)) == payload


def test_load_labels(tmp_path):
    p = tmp_path / "semantic_labels.json"
    payload = {"labeled_kernels": [{"semantic_block": "attn"}]}
    p.write_text(json.dumps(payload))
    assert load_labels(str(p)) == payload
