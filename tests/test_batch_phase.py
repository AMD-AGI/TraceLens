###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for TraceLens.TraceUtils.split_trace.batch_phase.

Covers the shape-based batch-size inference and the decode/prefill phase
classification used for LLM-inference traces without parseable annotations.
"""

from TraceLens.TraceUtils.split_trace.batch_phase import (
    _GAP_SPLIT_MIN_RATIO,
    classify_phases_from_batch_sizes,
    infer_batch_sizes_from_shapes,
)


# --------------------------------------------------------------------------- #
# classify_phases_from_batch_sizes
# --------------------------------------------------------------------------- #
def test_classify_all_none_returns_all_decode():
    assert classify_phases_from_batch_sizes([None, None, None]) == [
        "decode",
        "decode",
        "decode",
    ]


def test_classify_empty_input():
    assert classify_phases_from_batch_sizes([]) == []


def test_classify_none_entries_stay_decode_with_threshold():
    # None batch sizes are always 'decode' regardless of threshold.
    labels = classify_phases_from_batch_sizes([None, 256, None], max_num_seq=16)
    assert labels == ["decode", "prefill_bearing", "decode"]


def test_classify_max_num_seq_threshold_and_boundary():
    # b <= threshold -> decode; b > threshold -> prefill_bearing.
    labels = classify_phases_from_batch_sizes([4, 16, 17, 256], max_num_seq=16)
    assert labels == ["decode", "decode", "prefill_bearing", "prefill_bearing"]


def test_classify_auto_gap_split_picks_largest_ratio():
    # No max_num_seq: threshold is the low side of the largest multiplicative
    # gap. unique=[4, 8, 256]; ratios 2.0 and 32.0 -> split at 8.
    labels = classify_phases_from_batch_sizes([4, 8, 256, 8])
    assert labels == ["decode", "decode", "prefill_bearing", "decode"]


def test_classify_auto_gap_below_ratio_all_decode():
    # All adjacent ratios < _GAP_SPLIT_MIN_RATIO -> no threshold -> all decode.
    assert _GAP_SPLIT_MIN_RATIO == 2.0
    labels = classify_phases_from_batch_sizes([10, 12, 14])
    assert labels == ["decode", "decode", "decode"]


def test_classify_auto_split_at_exact_min_ratio():
    # A ratio exactly at the threshold (2.0) still splits (>=).
    labels = classify_phases_from_batch_sizes([8, 16])
    assert labels == ["decode", "prefill_bearing"]


def test_classify_drops_two_smallest_ramp_noise():
    # unique=[1, 2, 40, 800]; the two smallest (ramp-up) are dropped because a
    # remaining value (40) is <= 64, so the threshold is computed from
    # [40, 800] -> 40. Without the drop the largest early gap would put the
    # threshold at 2 and misclassify the 40 iteration.
    labels = classify_phases_from_batch_sizes([1, 2, 40, 800])
    assert labels == ["decode", "decode", "decode", "prefill_bearing"]


def test_classify_no_drop_when_remaining_all_large():
    # unique=[1, 2, 200, 300]; candidate=[200, 300] has nothing <= 64, so the
    # two smallest are NOT dropped and the huge 1->2->200 gap governs.
    labels = classify_phases_from_batch_sizes([1, 2, 200, 300])
    # largest ratio is 200/2=100 at index 1 -> threshold=2.
    assert labels == ["decode", "decode", "prefill_bearing", "prefill_bearing"]


def test_classify_ignores_nonpositive_batch_sizes():
    # Zero/negative values are dropped from the unique set used for the gap
    # heuristic but still labelled decode (b <= threshold).
    labels = classify_phases_from_batch_sizes([0, 4, 8, 256])
    assert labels[0] == "decode"
    assert labels[-1] == "prefill_bearing"


# --------------------------------------------------------------------------- #
# infer_batch_sizes_from_shapes
# --------------------------------------------------------------------------- #
def _cpu(ts, dur, first_dim, name="aten::mm"):
    return {
        "ts": ts,
        "dur": dur,
        "cat": "cpu_op",
        "name": name,
        "args": {"Input Dims": [[first_dim, 1, 1]]},
    }


def _index(events):
    """Build the (cpu_events, cpu_starts) index infer_* expects (ts-sorted)."""
    ordered = sorted(events, key=lambda e: e["ts"])
    return ordered, [e["ts"] for e in ordered]


def test_infer_most_common_first_dim_in_window():
    root = {"ts": 100, "dur": 100, "pid": 0, "tid": 0}
    events = [
        _cpu(100, 5, 8),
        _cpu(110, 5, 8),
        _cpu(120, 5, 4),  # 8 is more common than 4
        _cpu(250, 5, 999),  # outside [100, 200) -> excluded
        _cpu(105, 200, 777),  # encloses window (dur > win_dur) -> excluded
    ]
    assert infer_batch_sizes_from_shapes([root], _index(events)) == [8]


def test_infer_returns_none_when_no_shapes_in_window():
    root = {"ts": 0, "dur": 50, "pid": 0, "tid": 0}
    events = [_cpu(500, 5, 8)]  # far outside the window
    assert infer_batch_sizes_from_shapes([root], _index(events)) == [None]


def test_infer_one_value_per_root():
    roots = [
        {"ts": 0, "dur": 100, "pid": 0, "tid": 0},
        {"ts": 200, "dur": 100, "pid": 0, "tid": 0},
    ]
    events = [_cpu(10, 5, 8), _cpu(210, 5, 64)]
    assert infer_batch_sizes_from_shapes(roots, _index(events)) == [8, 64]


def test_infer_excludes_memory_view_ops():
    # A view op with a huge first dim must not dominate the batch estimate.
    root = {"ts": 0, "dur": 100, "pid": 0, "tid": 0}
    events = [
        _cpu(10, 5, 8),
        _cpu(20, 5, 8),
        _cpu(30, 5, 99999, name="aten::view"),  # excluded as a memory/view op
    ]
    assert infer_batch_sizes_from_shapes([root], _index(events)) == [8]


def test_infer_uses_root_tiles_window_over_root_span():
    # The root spans everything, but the tile restricts the window to the first
    # cluster, so the later (larger) cluster is not counted.
    root = {"ts": 0, "dur": 1000, "pid": 0, "tid": 0}
    events = [
        _cpu(10, 5, 8),
        _cpu(20, 5, 8),
        _cpu(30, 5, 8),  # cluster A
        _cpu(500, 5, 64),
        _cpu(510, 5, 64),
        _cpu(520, 5, 64),
        _cpu(530, 5, 64),
        _cpu(540, 5, 64),  # cluster B (larger)
    ]
    idx = _index(events)

    # Without a tile the whole span is scanned and cluster B (5x) wins.
    assert infer_batch_sizes_from_shapes([root], idx) == [64]

    # With a tile pinning the window to [0, 100) only cluster A is seen.
    tiles = {(0, 0, 0): (0, 100)}
    assert infer_batch_sizes_from_shapes([root], idx, root_tiles=tiles) == [8]
