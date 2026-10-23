###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for semantic_analyses/pattern_finder.py."""

import json
import os
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SEM = os.path.join(REPO_ROOT, "TraceLens", "Agent", "Analysis", "semantic_analyses")
sys.path.insert(0, SEM)

import pattern_finder

PATTERN_NAMES = ["k0", "k1", "k2", "k3", "k4", "k5"]


def _make_gpu_operations(names, stream=0, start_ts=0.0):
    ops = []
    ts = start_ts
    for name in names:
        ops.append(
            pattern_finder.GPUOperation(name=name, ts=ts, dur=1.0, stream=stream)
        )
        ts += 1.0
    return ops


def _make_kernels(names, stream_id=0, start_ts=0.0, with_stream=True):
    kernels = []
    ts = start_ts
    for name in names:
        k = {"name": name, "ts": ts, "dur": 1.0}
        if with_stream:
            k["stream_id"] = stream_id
        kernels.append(k)
        ts += 1.0
    return kernels


# ---------------------------------------------------------------------------
# GPUOperation.end_ts
# ---------------------------------------------------------------------------
def test_gpu_operation_end_ts():
    op = pattern_finder.GPUOperation(name="a", ts=10.0, dur=2.5)
    assert op.end_ts == 12.5


def test_operation_fields():
    op = pattern_finder.Operation(name="a", ts=1.0, dur=3.0)
    assert (op.name, op.ts, op.dur) == ("a", 1.0, 3.0)


# ---------------------------------------------------------------------------
# Patterns.get_pattern_name
# ---------------------------------------------------------------------------
def test_get_pattern_name_labels():
    patterns = pattern_finder.Patterns(patterns=[[0] for _ in range(28)])
    assert patterns.get_pattern_name(0) == "A"
    assert patterns.get_pattern_name(25) == "Z"
    assert patterns.get_pattern_name(26) == "A1"
    assert patterns.get_pattern_name(27) == "B1"


# ---------------------------------------------------------------------------
# LoopStructures construction
# ---------------------------------------------------------------------------
def test_loop_structures_initial_state():
    ops = _make_gpu_operations(["a", "b", "c", "a"])
    ls = pattern_finder.LoopStructures(ops)
    assert ls.operation_name_to_idx == {"a": 0, "b": 1, "c": 2}
    assert ls.pattern_operation_names == {0: "a", 1: "b", 2: "c"}
    assert ls.patterns == []
    assert ls.get_pattern_coverage() == 0.0


def test_get_operation_frequencies_uncovered():
    ops = _make_gpu_operations(["a", "b", "c", "a"])
    ls = pattern_finder.LoopStructures(ops)
    freqs = pattern_finder.get_operation_frequencies(ls)
    assert sum(freqs.values()) == len(ops)
    assert freqs[ls.operation_name_to_idx["a"]] == 2


# ---------------------------------------------------------------------------
# find_kernel_loops_single_stream
# ---------------------------------------------------------------------------
def test_find_kernel_loops_single_stream():
    ops = _make_gpu_operations(PATTERN_NAMES * 30, stream=0)
    ls = pattern_finder.find_kernel_loops_single_stream(
        ops, min_loop_count=20, min_pattern_length=5
    )
    assert len(ls.patterns) >= 1
    assert ls.get_pattern_coverage() > 0.9


# ---------------------------------------------------------------------------
# find_repeating_pattern
# ---------------------------------------------------------------------------
def test_find_repeating_pattern_end_to_end():
    names = ["pre0", "pre1", "pre2"] + PATTERN_NAMES * 30 + ["epi0", "epi1"]
    kernels = _make_kernels(names, stream_id=0)
    result = pattern_finder.find_repeating_pattern({"kernels": kernels})

    assert result["total_kernels"] == 185
    assert result["coverage"] > 0.9
    assert len(result["patterns"]) >= 1
    assert len(result["patterns"][0]) == 6
    assert result["preamble_indices"][:3] == [0, 1, 2]
    assert result["epilogue_indices"] == [183, 184]


def test_find_repeating_pattern_no_pattern():
    names = [f"unique_{i}" for i in range(10)]
    kernels = _make_kernels(names, stream_id=0)
    result = pattern_finder.find_repeating_pattern({"kernels": kernels})

    assert result["coverage"] == 0.0
    assert result["patterns"] == []
    assert len(result["preamble_indices"]) == 10
    assert result["epilogue_indices"] == []


def test_find_repeating_pattern_multi_stream():
    primary = _make_kernels(PATTERN_NAMES * 30, stream_id=0)
    secondary = _make_kernels(
        ["s0", "s1", "s2", "s3", "s4"], stream_id=1, start_ts=1000.0
    )
    result = pattern_finder.find_repeating_pattern({"kernels": primary + secondary})

    assert result["primary_stream_id"] == 0
    assert len(result["secondary_stream_indices"]) == 5
    assert len(result["patterns"]) >= 1


# ---------------------------------------------------------------------------
# main (CLI)
# ---------------------------------------------------------------------------
def test_main_writes_output_file(tmp_path, monkeypatch):
    extracted = {"kernels": _make_kernels(PATTERN_NAMES * 30, stream_id=0)}
    in_path = tmp_path / "extracted.json"
    in_path.write_text(json.dumps(extracted))
    out_path = tmp_path / "patterns.json"
    monkeypatch.setattr(
        sys, "argv", ["pattern_finder", str(in_path), "-o", str(out_path)]
    )
    pattern_finder.main()
    result = json.loads(out_path.read_text())
    assert result["total_kernels"] == 180
    assert len(result["patterns"]) >= 1


def test_main_prints_to_stdout(tmp_path, monkeypatch, capsys):
    extracted = {"kernels": _make_kernels(PATTERN_NAMES * 30, stream_id=0)}
    in_path = tmp_path / "extracted.json"
    in_path.write_text(json.dumps(extracted))
    monkeypatch.setattr(sys, "argv", ["pattern_finder", str(in_path)])
    pattern_finder.main()
    result = json.loads(capsys.readouterr().out)
    assert result["total_kernels"] == 180


def test_main_reports_multi_stream(tmp_path, monkeypatch, capsys):
    primary = _make_kernels(PATTERN_NAMES * 30, stream_id=0)
    secondary = _make_kernels(["s0", "s1", "s2"], stream_id=1, start_ts=1000.0)
    extracted = {"kernels": primary + secondary}
    in_path = tmp_path / "extracted.json"
    in_path.write_text(json.dumps(extracted))
    monkeypatch.setattr(sys, "argv", ["pattern_finder", str(in_path)])
    pattern_finder.main()
    assert "Multi-stream" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# long-pattern splitting
# ---------------------------------------------------------------------------
def test_find_kernel_loops_splits_long_pattern():
    names = [f"p{i}" for i in range(50)]
    ops = _make_gpu_operations(names * 25, stream=0)
    ls = pattern_finder.find_kernel_loops_single_stream(
        ops, min_loop_count=20, min_pattern_length=10, split_pattern_length_limit=20
    )
    assert len(ls.patterns) >= 1
    assert ls.get_pattern_coverage() > 0.5


# ---------------------------------------------------------------------------
# Patterns mapping derivation
# ---------------------------------------------------------------------------
def test_patterns_derive_name_to_idx_from_names():
    patterns = pattern_finder.Patterns(pattern_operation_names={0: "a", 1: "b"})
    assert patterns.operation_name_to_idx == {"a": 0, "b": 1}


def test_patterns_derive_names_from_name_to_idx():
    patterns = pattern_finder.Patterns(operation_name_to_idx={"a": 0, "b": 1})
    assert patterns.pattern_operation_names == {0: "a", 1: "b"}


def test_get_pattern_name_out_of_range_returns_placeholder():
    patterns = pattern_finder.Patterns(patterns=[[0]])
    assert patterns.get_pattern_name(-1) == "?"
    assert patterns.get_pattern_name(5) == "?"


# ---------------------------------------------------------------------------
# Patterns._validate_pattern_indices_mapped
# ---------------------------------------------------------------------------
def test_validate_fills_missing_name_from_name_to_idx():
    patterns = pattern_finder.Patterns(
        patterns=[[0]],
        pattern_operation_names={0: "a"},
        operation_name_to_idx={"a": 0, "b": 1},
    )
    patterns._validate_pattern_indices_mapped()
    assert patterns.pattern_operation_names[1] == "b"


def test_validate_raises_for_unmapped_index():
    patterns = pattern_finder.Patterns(
        patterns=[[0, 1]],
        pattern_operation_names={0: "a"},
        operation_name_to_idx={"a": 0},
    )
    with pytest.raises(ValueError, match="Missing mappings for indices"):
        patterns._validate_pattern_indices_mapped()


# ---------------------------------------------------------------------------
# LoopStructures construction edge cases
# ---------------------------------------------------------------------------
def test_loop_structures_accepts_supplied_patterns_obj():
    ops = _make_gpu_operations(["a", "b"])
    shared = pattern_finder.Patterns(pattern_operation_names={0: "a", 1: "b"})
    ls = pattern_finder.LoopStructures(ops, _patterns_obj=shared)
    assert ls._patterns_obj is shared


def test_loop_structures_registers_names_absent_from_patterns_obj():
    ops = _make_gpu_operations(["a", "b"])
    partial = pattern_finder.Patterns(pattern_operation_names={0: "a"})
    ls = pattern_finder.LoopStructures(ops, _patterns_obj=partial)
    assert ls.operation_name_to_idx["b"] == 1
    assert ls.pattern_operation_names[1] == "b"


def test_get_covered_indices_accepts_single_int():
    ops = _make_gpu_operations(["a", "b", "a", "b"])
    ls = pattern_finder.LoopStructures(ops)
    ls.patterns.append([0, 1])
    ls.sequences.append([[0, 1], [2, 3]])
    assert ls.get_covered_indices(0) == {0, 1, 2, 3}


def test_pattern_coverage_is_zero_without_operations():
    ls = pattern_finder.LoopStructures([])
    assert ls.get_pattern_coverage() == 0.0


# ---------------------------------------------------------------------------
# extend_pattern
# ---------------------------------------------------------------------------
def test_extend_pattern_grows_backward_and_merges_occurrences():
    ops = _make_gpu_operations(["a", "a", "a", "a"])
    ls = pattern_finder.LoopStructures(ops)
    ls.patterns.append([0])
    ls.sequences.append([[1], [2], [3]])
    best_agree = pattern_finder.extend_pattern(ls, 0, 2, 2)
    assert ls.patterns[0] == [0, 0]
    assert ls.sequences[0] == [[0, 1], [2, 3]]
    assert best_agree == 1


def test_extend_pattern_grows_forward_skipping_overlap():
    ops = _make_gpu_operations(["a", "a", "a", "a"])
    ls = pattern_finder.LoopStructures(ops)
    ls.patterns.append([0])
    ls.sequences.append([[0], [1], [2]])
    pattern_finder.extend_pattern(ls, 0, 3, 2)
    assert ls.patterns[0] == [0, 0]
    assert ls.sequences[0] == [[0, 1], [2, 3]]


def test_extend_pattern_stops_when_neighbors_disagree():
    ops = _make_gpu_operations(["a", "x", "b", "x", "c", "x"])
    ls = pattern_finder.LoopStructures(ops)
    xi = ls.operation_name_to_idx["x"]
    ls.patterns.append([xi])
    ls.sequences.append([[1], [3], [5]])
    best_agree = pattern_finder.extend_pattern(ls, 0, 2, 2)
    assert ls.patterns[0] == [xi]
    assert best_agree == 1


def test_extend_pattern_snippets_stops_when_not_all_agree():
    ops = _make_gpu_operations(["a", "b", "a", "b", "a", "b"])
    ls = pattern_finder.LoopStructures(ops)
    ls.patterns.append([0, 1])
    ls.sequences.append([[0, 1], [2, 3], [4, 5]])
    best_agree = pattern_finder.extend_pattern(ls, 0, 2, 2, snippets=True)
    assert ls.patterns[0] == [0, 1]
    assert best_agree == 2


# ---------------------------------------------------------------------------
# split_long_pattern
# ---------------------------------------------------------------------------
def test_split_long_pattern_decomposes_internal_repeat():
    ops = _make_gpu_operations(["a", "b"] * 5)
    ls = pattern_finder.LoopStructures(ops)
    ls.patterns.append([0, 1, 0, 1, 0, 1])
    ls.sequences.append([[0, 1]])
    assert pattern_finder.split_long_pattern(ls, 0, 2, 4) is True
    assert ls.patterns == [[0, 1]]
    assert ls.sequences == [[[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]]


def test_split_long_pattern_short_pattern_not_split():
    ops = _make_gpu_operations(["a", "b"] * 5)
    ls = pattern_finder.LoopStructures(ops)
    ls.patterns.append([0, 1])
    ls.sequences.append([[0, 1]])
    assert pattern_finder.split_long_pattern(ls, 0, 2, 4) is False


def test_split_long_pattern_without_sequences_not_split():
    ops = _make_gpu_operations(["a", "b"] * 5)
    ls = pattern_finder.LoopStructures(ops)
    ls.patterns.append([0, 1, 0, 1, 0, 1])
    ls.sequences.append([])
    assert pattern_finder.split_long_pattern(ls, 0, 2, 4) is False


def test_split_long_pattern_without_internal_repeat_not_split():
    ops = _make_gpu_operations(["a", "b", "c", "d", "e", "f"])
    ls = pattern_finder.LoopStructures(ops)
    ls.patterns.append([0, 1, 2, 3, 4, 5])
    ls.sequences.append([[0, 1, 2, 3, 4, 5]])
    assert pattern_finder.split_long_pattern(ls, 0, 2, 4) is False


def test_split_long_pattern_non_consecutive_subpattern_not_split():
    ops = _make_gpu_operations(["a", "b", "z", "a", "b"])
    ls = pattern_finder.LoopStructures(ops)
    zi = ls.operation_name_to_idx["z"]
    ls.patterns.append([0, 1, zi, 0, 1])
    ls.sequences.append([[0, 1, 2, 3, 4]])
    assert pattern_finder.split_long_pattern(ls, 0, 2, 4) is False


def test_split_long_pattern_reuses_existing_pattern():
    ops = _make_gpu_operations(["a", "b"] * 5)
    ls = pattern_finder.LoopStructures(ops)
    ls.patterns.append([0, 1])
    ls.sequences.append([[0, 1]])
    ls.patterns.append([0, 1, 0, 1, 0, 1])
    ls.sequences.append([[0, 1]])
    assert pattern_finder.split_long_pattern(ls, 1, 2, 4) is True
    assert ls.patterns == [[0, 1]]
    assert ls.sequences == [[[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]]


# ---------------------------------------------------------------------------
# rotate_pattern
# ---------------------------------------------------------------------------
def test_rotate_pattern_returns_false_when_no_shift():
    ops = _make_gpu_operations(["a", "b", "c", "a", "b", "c"])
    ls = pattern_finder.LoopStructures(ops)
    ls.patterns.append([0, 1, 2])
    ls.sequences.append([[0, 1, 2], [3, 4, 5]])
    assert pattern_finder.rotate_pattern(ls, 0) is False
    assert ls.sequences[0] == [[0, 1, 2], [3, 4, 5]]


def test_rotate_pattern_recovers_earlier_occurrence():
    ops = _make_gpu_operations(["a", "b", "a", "a", "b", "a", "c"])
    ls = pattern_finder.LoopStructures(ops)
    ls.patterns.append([0, 1, 0])
    ls.sequences.append([[3, 4, 5]])
    assert pattern_finder.rotate_pattern(ls, 0) is True
    assert ls.patterns[0] == [0, 1, 0]
    assert ls.sequences[0] == [[0, 1, 2], [3, 4, 5]]


def test_rotate_pattern_skips_adjacent_occurrence():
    ops = _make_gpu_operations(["a", "b", "a", "a", "b", "a"])
    ls = pattern_finder.LoopStructures(ops)
    ls.patterns.append([0, 1, 0])
    ls.sequences.append([[0, 1, 2], [3, 4, 5]])
    assert pattern_finder.rotate_pattern(ls, 0) is True
    assert ls.sequences[0] == [[0, 1, 2], [3, 4, 5]]


# ---------------------------------------------------------------------------
# find_pattern_occurrences
# ---------------------------------------------------------------------------
def test_find_pattern_occurrences_skips_overlapping_candidates():
    ops = _make_gpu_operations(["a", "a", "a", "a"])
    ls = pattern_finder.LoopStructures(ops)
    ls.patterns.append([0, 0])
    ls.sequences.append([])
    assert pattern_finder.find_pattern_occurrences(ls, 0) == [[0, 1], [2, 3]]


# ---------------------------------------------------------------------------
# get_new_pattern_to_start
# ---------------------------------------------------------------------------
def test_get_new_pattern_defaults_failed_ops_to_empty():
    ops = _make_gpu_operations(["a", "a", "b", "b"])
    ls = pattern_finder.LoopStructures(ops)
    result = pattern_finder.get_new_pattern_to_start(ls, ls.operations, 2)
    assert result == ([0], [[0], [1]])


# ---------------------------------------------------------------------------
# find_kernel_loops_single_stream edge cases
# ---------------------------------------------------------------------------
def test_find_kernel_loops_rejects_mismatched_operations():
    ls = pattern_finder.LoopStructures(_make_gpu_operations(["a", "b", "c"]))
    with pytest.raises(ValueError, match="same operations list"):
        pattern_finder.find_kernel_loops_single_stream(
            _make_gpu_operations(["x", "y", "z"]), loop_structures=ls
        )


def test_find_kernel_loops_high_min_loop_count():
    ops = _make_gpu_operations(["a", "b", "c", "d", "e"] * 30)
    ls = pattern_finder.find_kernel_loops_single_stream(
        ops, min_loop_count=25, min_pattern_length=3
    )
    assert len(ls.patterns) == 1
    assert ls.get_pattern_coverage() == 1.0


def test_find_kernel_loops_single_name_covers_everything():
    ops = _make_gpu_operations(["a"] * 25)
    ls = pattern_finder.find_kernel_loops_single_stream(
        ops, min_loop_count=20, min_pattern_length=1
    )
    assert ls.patterns == [[0]]
    assert ls.get_pattern_coverage() == 1.0


def test_find_kernel_loops_discards_pattern_below_min_length():
    names = []
    for i in range(25):
        names += ["a", f"u{i}"]
    ops = _make_gpu_operations(names)
    ls = pattern_finder.find_kernel_loops_single_stream(
        ops, min_loop_count=20, min_pattern_length=5
    )
    assert ls.patterns == []
    assert ls.get_pattern_coverage() == 0.0


def test_find_kernel_loops_splits_long_pattern_with_leftover():
    ops = _make_gpu_operations(list("abcabcdef") * 30 + ["z0", "z1", "z2", "z3"])
    ls = pattern_finder.find_kernel_loops_single_stream(
        ops, min_loop_count=20, min_pattern_length=3, split_pattern_length_limit=5
    )
    assert sorted(len(p) for p in ls.patterns) == [3, 3]
    assert ls.get_pattern_coverage() > 0.95


def test_find_kernel_loops_keeps_long_pattern_with_short_subblock():
    """A long pattern whose internal sub-block is below min_pattern_length is
    kept whole across detection passes rather than being split."""
    ops = _make_gpu_operations(list("abcabcdef") * 8 + ["z0", "z1"])
    ls = pattern_finder.find_kernel_loops_single_stream(
        ops, min_loop_count=5, min_pattern_length=4, split_pattern_length_limit=4
    )
    assert len(ls.patterns) == 1
    assert [ls.pattern_operation_names[i] for i in ls.patterns[0]] == [
        "a",
        "b",
        "c",
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
    ]
    assert 0.9 < ls.get_pattern_coverage() < 1.0
