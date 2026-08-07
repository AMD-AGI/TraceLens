###############################################################################
# Copyright (c) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for TraceLens.TraceUtils modules."""

import json
import os
import zipfile

import pytest

from TraceLens.TraceUtils import split_inference_trace_annotation as split
from TraceLens.TraceUtils.annotation_utils import (
    PHASE_DECODE_ONLY,
    PHASE_PREFILLDECODE,
    average_detail,
    find_events_by_patterns,
    find_iteration_roots_by_priority,
    find_phase_from_window,
    ITERATION_BACKUP_PATTERNS,
    ITERATION_PATTERNS,
)
from TraceLens.TraceUtils.match_inference_trace_blocks import (
    Block,
    _avg_block_distance,
    _block_steps,
    _block_summary_dict,
    base_name_from_path,
    compute_output_path,
    find_blocks,
    select_best_per_phase,
    window_blocks,
)

VLLM_PREFILLDECODE = (
    "execute_0_context_3(sq128sk256sqsq1sqsk1)_generation_2(sq1sk300sqsq1sqsk1)"
)
VLLM_DECODE_ONLY = (
    "execute_0_context_0(sq0sk0sqsq0sqsk0)_generation_2(sq1sk300sqsq1sqsk1)"
)
VLLM_DECODE_ONLY_ALT = (
    "execute_0_context_0(sq0sk0sqsq0sqsk0)_generation_2(sq2sk400sqsq2sqsk2)"
)


def _root(name: str, ts: int = 0) -> dict:
    return {
        "name": name,
        "cat": "user_annotation",
        "ph": "X",
        "ts": ts,
        "dur": 1,
        "pid": 1,
        "tid": 1,
        "args": {},
    }


def _decode_block(num_steps: int, g_sq_base: int = 1) -> Block:
    roots = []
    details = []
    for i in range(num_steps):
        g_sq = g_sq_base + i
        name = (
            "execute_0_context_0(sq0sk0sqsq0sqsk0)_"
            f"generation_2(sq{g_sq}sk300sqsq{g_sq}sqsk1)"
        )
        roots.append(_root(name, ts=i))
        details.append(
            {
                "name": name,
                "context_requests": 0,
                "generation_requests": 2,
                "g_sq": g_sq,
                "g_sk": 300,
                "has_sqsk": True,
            }
        )
    return Block(
        phase=PHASE_DECODE_ONLY,
        start_idx=0,
        end_idx=num_steps,
        roots=roots,
        details=details,
    )


# ---------------------------------------------------------------------------
# split_inference_trace_annotation helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "range_str,max_len,expected",
    [
        ("all", 10, (0, 10)),
        ("3", 10, (3, 4)),
        ("2:5", 10, (2, 5)),
        ("8:20", 10, (8, 10)),
    ],
)
def test_parse_range(range_str, max_len, expected):
    assert split.parse_range(range_str, max_len) == expected


def test_get_filename_returns_path_for_json():
    assert split.get_filename("/tmp/trace.json.gz") == "/tmp/trace.json.gz"


def test_get_filename_reads_json_from_zip(tmp_path):
    trace = {"traceEvents": [{"name": "x"}], "schemaVersion": 1}
    zip_path = tmp_path / "trace.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("nested/trace.json", json.dumps(trace))

    assert split.get_filename(str(zip_path)) == "nested/trace.json"


def test_preprocess_trace_builds_correlation_maps():
    events = [
        {"ts": 1, "ph": "X", "cat": "kernel", "args": {"correlation": 42}},
        {"ts": 2, "ph": "s", "id": 7},
        {"ph": "M", "name": "process_name"},
    ]
    gpu_map, flow_map, meta = split.preprocess_trace(events)
    assert gpu_map[42][0]["cat"] == "kernel"
    assert flow_map[7][0]["ph"] == "s"
    assert meta[0]["name"] == "process_name"


def test_compute_reference_pd_ratio_uses_largest_region(capsys):
    iter_details = [{"context_requests": 1} for _ in range(4)] + [
        {"context_requests": 0} for _ in range(6)
    ]
    regions = [(0, 4), (4, 10)]
    largest, avg_ratio, largest_ratio = split.compute_reference_pd_ratio(
        regions, iter_details
    )
    assert largest == (4, 10)
    assert avg_ratio == pytest.approx(0.4)
    assert largest_ratio == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# annotation_utils helpers (TraceUtils)
# ---------------------------------------------------------------------------


def test_find_events_by_patterns_filters_and_sorts():
    events = [
        _root("other", ts=2),
        _root(VLLM_PREFILLDECODE, ts=1),
        _root(VLLM_DECODE_ONLY, ts=3),
    ]
    matches = find_events_by_patterns(events, ITERATION_PATTERNS)
    assert [e["ts"] for e in matches] == [1, 3]


def test_find_iteration_roots_by_priority_prefers_primary_patterns():
    events = [
        _root(VLLM_PREFILLDECODE, ts=1),
        _root("execute_context_3(100)_generation_2(50)", ts=2),
    ]
    roots = find_iteration_roots_by_priority(events)
    assert len(roots) == 1
    assert "sq128" in roots[0]["name"]

    backup_only = [_root("execute_context_3(100)_generation_2(50)", ts=1)]
    roots = find_iteration_roots_by_priority(
        backup_only, pattern_tiers=[ITERATION_PATTERNS, ITERATION_BACKUP_PATTERNS]
    )
    assert len(roots) == 1
    assert roots[0]["name"].startswith("execute_context")


def test_average_detail_and_find_phase_from_window():
    details = [
        {
            "batch_size": 10,
            "num_requests": 2,
            "context_requests": 1,
            "generation_requests": 1,
        },
        {
            "batch_size": 20,
            "num_requests": 4,
            "context_requests": 0,
            "generation_requests": 2,
        },
    ]
    assert average_detail(details, "batch_size") == 15.0
    phase = find_phase_from_window(details)
    assert phase["num_prefilldecode"] == 1
    assert phase["num_decode"] == 1
    assert phase["avg_bs"] == 15


# ---------------------------------------------------------------------------
# match_inference_trace_blocks
# ---------------------------------------------------------------------------


def test_block_properties_and_has_full_sqsk():
    block = _decode_block(3)
    assert block.num_steps == 3
    assert block.truncated is False
    assert block.has_full_sqsk() is True
    assert block.avg("g_sq") == pytest.approx(2.0)


def test_window_blocks_splits_and_drops_tail():
    block = _decode_block(5)
    windows = window_blocks([block], max_steps=2)
    assert len(windows) == 2
    assert all(w.num_steps == 2 for w in windows)
    assert windows[0].start_idx == 0
    assert windows[1].start_idx == 2
    assert windows[0].truncated is True
    assert windows[0].original_num_steps == 5


def test_window_blocks_passthrough_when_disabled():
    block = _decode_block(3)
    assert window_blocks([block], max_steps=0) == [block]
    assert window_blocks([block], max_steps=-1) == [block]


def test_find_blocks_groups_contiguous_phases():
    roots = [
        _root(VLLM_DECODE_ONLY, ts=0),
        _root(VLLM_DECODE_ONLY_ALT, ts=1),
        _root(VLLM_PREFILLDECODE, ts=2),
    ]
    blocks = find_blocks(roots)
    assert len(blocks) == 2
    assert blocks[0].phase == PHASE_DECODE_ONLY
    assert blocks[0].num_steps == 2
    assert blocks[1].phase == PHASE_PREFILLDECODE
    assert blocks[1].num_steps == 1


def test_find_blocks_skips_short_decode_only_runs():
    roots = [_root(VLLM_DECODE_ONLY, ts=0)]
    assert find_blocks(roots) == []


def test_avg_block_distance_for_decode_only():
    a = _decode_block(2, g_sq_base=10)
    b = _decode_block(2, g_sq_base=11)
    assert _avg_block_distance(a, b) == pytest.approx((1.0, 0.0))


def test_avg_block_distance_returns_none_for_mismatched_blocks():
    a = _decode_block(2)
    b = _decode_block(3)
    assert _avg_block_distance(a, b) is None


def test_select_best_per_phase_picks_closest_pair():
    a_blocks = [_decode_block(2, g_sq_base=10), _decode_block(2, g_sq_base=50)]
    b_blocks = [_decode_block(2, g_sq_base=11), _decode_block(2, g_sq_base=100)]
    matches, notes = select_best_per_phase(
        a_blocks, b_blocks, phases=[PHASE_DECODE_ONLY]
    )
    assert notes == []
    assert len(matches) == 1
    assert matches[0]["a_block_index"] == 0
    assert matches[0]["b_block_index"] == 0
    assert matches[0]["distance"] == [1.0, 0.0]


def test_select_best_per_phase_records_note_when_no_match():
    a_blocks = [_decode_block(2)]
    b_blocks = [_decode_block(3)]
    matches, notes = select_best_per_phase(
        a_blocks, b_blocks, phases=[PHASE_DECODE_ONLY]
    )
    assert matches == []
    assert len(notes) == 1
    assert notes[0]["phase"] == PHASE_DECODE_ONLY


def test_base_name_from_path_strips_known_suffixes():
    assert base_name_from_path("/data/run.pt.trace") == "run"
    assert base_name_from_path("/data/run.json.gz") == "run"


def test_block_steps_and_summary_dict():
    block = _decode_block(2, g_sq_base=5)
    steps = _block_steps(block)
    assert len(steps) == 2
    assert steps[0]["g_sq"] == 5
    summary = _block_summary_dict(block)
    assert summary["num_steps"] == 2
    assert summary["truncated"] is False
    assert "avg_g_sq" in summary


def test_compute_output_path_for_multi_step_block(tmp_path):
    block = _decode_block(2)
    path = compute_output_path(
        str(tmp_path), PHASE_DECODE_ONLY, "label", "traceA", block
    )
    assert path.endswith(".json.gz")
    assert PHASE_DECODE_ONLY in path
    assert "label" in path


def test_compute_output_path_for_single_step_block(tmp_path):
    block = Block(
        phase=PHASE_PREFILLDECODE,
        start_idx=0,
        end_idx=1,
        roots=[_root(VLLM_PREFILLDECODE)],
        details=[{"name": VLLM_PREFILLDECODE, "batch_size": 129, "num_requests": 5}],
    )
    path = compute_output_path(
        str(tmp_path), PHASE_PREFILLDECODE, "label", "traceB", block
    )
    assert "execute_0_context_3" in os.path.basename(path)
