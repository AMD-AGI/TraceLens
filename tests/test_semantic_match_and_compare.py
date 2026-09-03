###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for semantic_analyses/match_and_compare.py.

Uses in-memory dict fixtures (no file I/O) to exercise the pure
aggregation / comparison / assertion helpers.
"""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SEM = os.path.join(REPO_ROOT, "TraceLens", "Agent", "Analysis", "semantic_analyses")
sys.path.insert(0, SEM)

from match_and_compare import aggregate, build_comparison, run_assertions


def _labeled(spec):
    """Build a list of labeled-kernel dicts from compact tuples.

    Each tuple is (semantic_block, name, dur, perf_category, nn_module).
    """
    return [
        {
            "name": name,
            "dur": dur,
            "semantic_block": block,
            "perf_category": pc,
            "nn_module": nm,
        }
        for (block, name, dur, pc, nm) in spec
    ]


# Standard, self-consistent A/B fixture used by several tests.
LABELED_A = _labeled(
    [
        ("GEMM_0", "aa", 10.0, "GEMM", "Blk"),
        ("GEMM_0", "bb", 30.0, "GEMM", "Blk"),
        ("GEMM_0", "aa", 10.0, "GEMM", "Blk"),  # duplicate name -> deduped in set
        ("Norm_0", "cc", 20.0, "Normalization", "Blk"),
        ("Extra_0", "dd", 10.0, "Others", ""),  # block only present in A
    ]
)
TOTAL_A = 80.0

LABELED_B = _labeled(
    [
        ("GEMM_0", "dd", 20.0, "GEMM", "Blk"),
        ("Norm_0", "ee", 10.0, "Normalization", "Blk"),
        ("Norm_0", "ff", 10.0, "Normalization", "Blk"),
        ("SDPA_0", "gg", 5.0, "SDPA", "Blk"),  # block only present in B
    ]
)
TOTAL_B = 45.0


def _agg_pair():
    return aggregate(LABELED_A), aggregate(LABELED_B)


def _rows(**kwargs):
    agg_a, agg_b = _agg_pair()
    return build_comparison(agg_a, agg_b, TOTAL_A, TOTAL_B, "MI355", "B200", **kwargs)


# --------------------------------------------------------------------------- #
# aggregate
# --------------------------------------------------------------------------- #
def test_aggregate_groups_dedups_names_and_counts():
    agg = aggregate(LABELED_A)
    assert list(agg.keys()) == ["GEMM_0", "Norm_0", "Extra_0"]
    gemm = agg["GEMM_0"]
    assert gemm["count"] == 3
    assert gemm["durs"] == [10.0, 30.0, 10.0]
    assert gemm["names"] == {"aa", "bb"}  # duplicate "aa" collapsed


def test_aggregate_carries_perf_category_and_nn_module():
    agg = aggregate(LABELED_A)
    assert agg["GEMM_0"]["perf_category"] == "GEMM"
    assert agg["GEMM_0"]["nn_module"] == "Blk"


def test_aggregate_empty_input():
    assert aggregate([]) == {}


# --------------------------------------------------------------------------- #
# build_comparison
# --------------------------------------------------------------------------- #
def test_build_comparison_rows_ratios_and_order():
    rows = _rows()
    blocks = [r["semantic_block"] for r in rows]
    assert blocks == ["GEMM_0", "Norm_0", "Extra_0", "SDPA_0"]

    by_block = {r["semantic_block"]: r for r in rows}

    gemm = by_block["GEMM_0"]
    assert gemm["MI355_kernel_count"] == 3
    assert gemm["MI355_total_us"] == 50.0
    assert gemm["MI355_avg_us"] == round(50.0 / 3, 2)
    assert gemm["B200_total_us"] == 20.0
    assert gemm["MI355_vs_B200_ratio"] == 2.5
    assert gemm["MI355_kernel_names"] == "aa | bb"
    assert gemm["algorithm_order"] == 1

    # Block only in A -> B side empty, ratio "inf", zero avg on B.
    extra = by_block["Extra_0"]
    assert extra["B200_kernel_count"] == 0
    assert extra["B200_avg_us"] == 0
    assert extra["MI355_vs_B200_ratio"] == "inf"

    # Block only in B -> A side empty.
    sdpa = by_block["SDPA_0"]
    assert sdpa["MI355_kernel_count"] == 0
    assert sdpa["MI355_avg_us"] == 0
    assert sdpa["MI355_vs_B200_ratio"] == 0.0


def test_build_comparison_region_field_present():
    rows = _rows(region="prefill_only_3072")
    assert rows[0]["region"] == "prefill_only_3072"
    assert list(rows[0].keys())[0] == "region"


def test_build_comparison_gpu_timeline_fields():
    rows = _rows(
        gpu_timeline_a={"busy_time_us": 2000.0, "idle_pct": 10.0},
        gpu_timeline_b={"busy_time_us": 4000.0, "idle_pct": 25.0},
    )
    r = rows[0]
    assert r["MI355_busy_ms"] == 2.0
    assert r["MI355_idle_pct"] == 10.0
    assert r["B200_busy_ms"] == 4.0
    assert r["B200_idle_pct"] == 25.0


def test_build_comparison_no_gpu_timeline_when_one_missing():
    rows = _rows(gpu_timeline_a={"busy_time_us": 2000.0, "idle_pct": 10.0})
    assert "MI355_busy_ms" not in rows[0]


def test_build_comparison_empty_inputs():
    assert build_comparison({}, {}, 0, 0, "MI355", "B200") == []


def test_build_comparison_perf_category_and_module_fallback():
    labeled = _labeled([("X_0", "x", 5.0, None, None)])
    agg = aggregate(labeled)
    rows = build_comparison(agg, agg, 5.0, 5.0, "A", "B")
    assert rows[0]["perf_category"] == "Others"
    assert rows[0]["nn_module"] == ""


def test_build_comparison_zero_totals_give_zero_pct():
    agg_a, agg_b = _agg_pair()
    rows = build_comparison(agg_a, agg_b, 0, 0, "MI355", "B200")
    assert all(r["MI355_pct"] == 0 for r in rows)
    assert all(r["B200_pct"] == 0 for r in rows)


# --------------------------------------------------------------------------- #
# run_assertions
# --------------------------------------------------------------------------- #
def test_run_assertions_pass_on_consistent_data():
    rows = _rows()
    errors = run_assertions(
        rows, LABELED_A, LABELED_B, TOTAL_A, TOTAL_B, "MI355", "B200"
    )
    assert errors == []


def test_run_assertions_count_mismatch_a():
    rows = _rows()
    padded_a = LABELED_A + [{"name": "z", "dur": 0.0, "semantic_block": "GEMM_0"}]
    errors = run_assertions(
        rows, padded_a, LABELED_B, TOTAL_A, TOTAL_B, "MI355", "B200"
    )
    assert any("A6.1" in e and "MI355" in e for e in errors)
    assert not any("A6.2" in e for e in errors)


def test_run_assertions_count_mismatch_b():
    rows = _rows()
    padded_b = LABELED_B + [{"name": "z", "dur": 0.0, "semantic_block": "SDPA_0"}]
    errors = run_assertions(
        rows, LABELED_A, padded_b, TOTAL_A, TOTAL_B, "MI355", "B200"
    )
    assert any("A6.2" in e and "B200" in e for e in errors)


def test_run_assertions_time_mismatch_both_sides():
    rows = _rows()
    errors = run_assertions(
        rows, LABELED_A, LABELED_B, TOTAL_A + 5.0, TOTAL_B + 5.0, "MI355", "B200"
    )
    assert any("A6.3" in e and "MI355" in e for e in errors)
    assert any("A6.3" in e and "B200" in e for e in errors)


def test_run_assertions_pct_mismatch_both_sides():
    # Feed doubled totals into build_comparison so percentages sum to ~50%.
    agg_a, agg_b = _agg_pair()
    rows = build_comparison(agg_a, agg_b, TOTAL_A * 2, TOTAL_B * 2, "MI355", "B200")
    errors = run_assertions(
        rows, LABELED_A, LABELED_B, TOTAL_A * 2, TOTAL_B * 2, "MI355", "B200"
    )
    assert any("A7.2" in e and "MI355" in e for e in errors)
    assert any("A7.2" in e and "B200" in e for e in errors)


def test_run_assertions_ratio_mismatch():
    rows = _rows()
    # Corrupt the stored ratio of the GEMM_0 row (expected 2.5).
    rows[0]["MI355_vs_B200_ratio"] = 9.999
    errors = run_assertions(
        rows, LABELED_A, LABELED_B, TOTAL_A, TOTAL_B, "MI355", "B200"
    )
    assert any("A7.5" in e for e in errors)
