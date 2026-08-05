###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the inference-trace block matcher.

Two synthetic traces of 128 annotation roots each are run through the full
pipeline once at import: root discovery -> ``find_blocks`` -> ``window_blocks``
-> ``select_best_per_phase``. The tests then assert against those results.

The traces are laid out so exactly one decode_only window and exactly one
prefilldecode window are identical *step for step* between A and B, which makes
the winning pair and its zero distance unique and hand-checkable. Distance is a
mean of per-step absolute differences, so an all-zero distance requires
element-wise equality, not merely equal averages.

Trace A                            Trace B
  [0:48]    decode, g_sk 4000..      [0:20]    decode, g_sk 9000
  [48]      idle (neither phase)     [20]      idle
  [49:81]   prefilldecode            [21:37]   decode, g_sk 5600..7100
  [81]      idle                     [37]      idle
  [82]      decode (lone step)       [38:54]   prefilldecode, c_req 2
  [83]      idle                     [54]      idle
  [84:128]  decode, g_sk 20000       [55:95]   prefilldecode, c_req 3
                                     [95]      idle
                                     [96:128]  decode, g_sk 100

With the default 16-step windowing, A's window [16:32) is step-for-step equal
to B's [21:37), and A's [49:65) is step-for-step equal to B's [38:54). Those
are the two expected winners.
"""

import json
import os
import sys

import pandas as pd
import pytest

from TraceLens.TraceUtils import match_inference_trace_blocks as match
from TraceLens.TraceUtils.annotation_utils import ITERATION_PATTERNS
from TraceLens.TraceUtils.split_inference_trace_annotation import (
    extract_and_save,
    preprocess_trace,
)

DECODE_ONLY = match.PHASE_DECODE_ONLY
PREFILLDECODE = match.PHASE_PREFILLDECODE
PHASES = [DECODE_ONLY, PREFILLDECODE]
NUM_STEPS = 16  # the tool's default --num-steps


def step(c_req=0, c_sq=0, c_sk=0, g_req=8, g_sk=4096):
    """One vLLM detailed annotation, which is the format carrying sq/sk.

    Defaults describe a decode of 8 requests; pass ``c_req``/``c_sq``/``c_sk``
    for a step that also prefills, or ``g_req=0`` for an idle step. ``g_sq``
    tracks ``g_req`` (no speculative decoding) and ``c_sq`` is the context
    token count, so ``batch_size`` parses as ``c_sq + g_req``.
    """
    return (
        f"execute_{c_sq + g_req}"
        f"_context_{c_req}(sq{c_sq}sk{c_sk}sqsq{c_sq * c_sq}sqsk{c_sq * c_sk})"
        f"_generation_{g_req}(sq{g_req}sk{g_sk}sqsq{g_req * g_req}sqsk{g_req * g_sk})"
    )


# No requests of either kind, so classify_phase() returns None and the step
# terminates the block it follows.
IDLE = step(g_req=0, g_sk=0)

# Native annotations, which carry no sq/sk and are therefore never eligible.
NATIVE_PREFILLDECODE = "execute_context_2(100)_generation_8(8)"
NATIVE_DECODE = "execute_context_0(0)_generation_8(8)"

TRACE_A_NAMES = (
    [step(g_sk=4000 + 100 * i) for i in range(48)]
    + [IDLE]
    + [step(c_req=2, c_sq=1000 + 10 * i, c_sk=2000 + 10 * i) for i in range(32)]
    + [IDLE]
    + [step(g_sk=4000)]  # lone decode step: dropped by the 2-step minimum
    + [IDLE]
    + [step(g_req=4, g_sk=20000)] * 44
)

TRACE_B_NAMES = (
    [step(g_sk=9000)] * 20
    + [IDLE]
    # Identical, step for step, to A's second decode window [16:32).
    + [step(g_sk=4000 + 100 * (16 + i)) for i in range(16)]
    + [IDLE]
    # Identical, step for step, to A's first prefilldecode window [49:65).
    + [step(c_req=2, c_sq=1000 + 10 * i, c_sk=2000 + 10 * i) for i in range(16)]
    + [IDLE]
    + [step(c_req=3, c_sq=1500, c_sk=2500)] * 40
    + [IDLE]
    + [step(g_req=2, g_sk=100)] * 32
)


def make_trace(root_names, pid=1, cpu_tid=10, gpu_tid=99):
    """Build a trace with one annotation root per name.

    Each root has 3 cpu_ops (each carrying ``args.correlation``) inside its time
    window and 2 kernels linked to the first two correlations. Kernels sit
    outside the root window on purpose: extraction attributes them via the
    correlation map, not by time.
    """
    events = []
    for i, name in enumerate(root_names):
        base = 1_000 + i * 1_000  # spaced so per-root windows never overlap
        events.append(
            {
                "name": name,
                "cat": "user_annotation",
                "ph": "X",
                "ts": base,
                "dur": 100,
                "tid": cpu_tid,
                "pid": pid,
                "args": {},
            }
        )
        corrs = [1000 + 3 * i + j for j in range(3)]
        for j, c in enumerate(corrs):
            events.append(
                {
                    "name": f"cpu_op_{i}_{j}",
                    "cat": "cpu_op",
                    "ph": "X",
                    "ts": base + 1 + j * 5,
                    "dur": 3,
                    "tid": cpu_tid,
                    "pid": pid,
                    "args": {"correlation": c},
                }
            )
        for j, c in enumerate(corrs[:2]):
            events.append(
                {
                    "name": f"kernel_{i}_{j}",
                    "cat": "kernel",
                    "ph": "X",
                    "ts": base + 200 + j * 5,
                    "dur": 8,
                    "tid": gpu_tid,
                    "pid": pid,
                    "args": {"correlation": c},
                }
            )
    return {"traceEvents": events, "schemaVersion": 1}


# The pipeline, run once. Discovery mirrors load_trace(); windowing and
# selection mirror main(). Nothing below mutates these.
A_TRACE, B_TRACE = make_trace(TRACE_A_NAMES), make_trace(TRACE_B_NAMES)
A_ROOTS = match.find_iteration_roots_by_priority(A_TRACE["traceEvents"])
B_ROOTS = match.find_iteration_roots_by_priority(B_TRACE["traceEvents"])
A_BLOCKS, B_BLOCKS = match.find_blocks(A_ROOTS), match.find_blocks(B_ROOTS)
A_WINDOWS = match.window_blocks(A_BLOCKS, NUM_STEPS)
B_WINDOWS = match.window_blocks(B_BLOCKS, NUM_STEPS)
MATCHES, NOTES = match.select_best_per_phase(A_WINDOWS, B_WINDOWS, PHASES)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def test_fixtures_are_128_detailed_annotation_roots():
    for names, roots in ((TRACE_A_NAMES, A_ROOTS), (TRACE_B_NAMES, B_ROOTS)):
        assert len(names) == 128
        assert len(roots) == 128
        # Discovery picks the detailed tier, so every root carries sq/sk.
        assert all(ITERATION_PATTERNS[0].match(r["name"]) for r in roots)


def test_base_name_from_path():
    assert match.base_name_from_path("/x/y/run1.pt.trace.json.gz") == "run1"
    assert match.base_name_from_path("/x/y/run2.json") == "run2"
    assert match.base_name_from_path("run3.zip") == "run3"


# --------------------------------------------------------------------------- #
# Block discovery and windowing
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "blocks,expected",
    [
        pytest.param(
            A_BLOCKS,
            [
                (DECODE_ONLY, 0, 48),
                (PREFILLDECODE, 49, 81),
                # No block at 82: that lone decode step is below the minimum.
                (DECODE_ONLY, 84, 128),
            ],
            id="A",
        ),
        pytest.param(
            B_BLOCKS,
            [
                (DECODE_ONLY, 0, 20),
                (DECODE_ONLY, 21, 37),
                (PREFILLDECODE, 38, 54),
                (PREFILLDECODE, 55, 95),
                (DECODE_ONLY, 96, 128),
            ],
            id="B",
        ),
    ],
)
def test_find_blocks_splits_on_every_phase_change(blocks, expected):
    assert [(b.phase, b.start_idx, b.end_idx) for b in blocks] == expected


def test_find_blocks_applies_the_per_phase_minimum():
    """prefilldecode needs 1 step, decode_only needs 2."""
    assert match.find_blocks([]) == []

    lone_prefilldecode = [{"name": IDLE}, {"name": step(c_req=2)}, {"name": IDLE}]
    assert [(b.phase, b.num_steps) for b in match.find_blocks(lone_prefilldecode)] == [
        (PREFILLDECODE, 1)
    ]

    lone_decode = [{"name": IDLE}, {"name": step()}, {"name": IDLE}]
    assert match.find_blocks(lone_decode) == []


@pytest.mark.parametrize(
    "windows,expected",
    [
        pytest.param(
            A_WINDOWS,
            [
                (DECODE_ONLY, 0, 16, 48, True),
                (DECODE_ONLY, 16, 32, 48, True),
                (DECODE_ONLY, 32, 48, 48, True),
                (PREFILLDECODE, 49, 65, 32, True),
                (PREFILLDECODE, 65, 81, 32, True),
                # 44 steps -> two 16-step windows; the 12-step tail is dropped.
                (DECODE_ONLY, 84, 100, 44, True),
                (DECODE_ONLY, 100, 116, 44, True),
            ],
            id="A",
        ),
        pytest.param(
            B_WINDOWS,
            [
                (DECODE_ONLY, 0, 16, 20, True),
                # A 16-step block is kept whole, so it is not truncated.
                (DECODE_ONLY, 21, 37, 16, False),
                (PREFILLDECODE, 38, 54, 16, False),
                (PREFILLDECODE, 55, 71, 40, True),
                (PREFILLDECODE, 71, 87, 40, True),
                (DECODE_ONLY, 96, 112, 32, True),
                (DECODE_ONLY, 112, 128, 32, True),
            ],
            id="B",
        ),
    ],
)
def test_window_blocks_emits_full_windows_and_drops_the_tail(windows, expected):
    assert [
        (w.phase, w.start_idx, w.end_idx, w.original_num_steps, w.truncated)
        for w in windows
    ] == expected


def test_window_blocks_disabled_passes_blocks_through():
    assert match.window_blocks(A_BLOCKS, 0) is A_BLOCKS
    assert match.window_blocks(A_BLOCKS, None) is A_BLOCKS


def test_block_avg():
    # g_sk over steps 0..15 is 4000, 4100, ... 5500.
    assert A_WINDOWS[0].avg("g_sk") == 4750.0
    assert A_WINDOWS[0].avg("g_sq") == 8.0
    assert A_WINDOWS[0].avg("context_requests") == 0.0
    assert A_WINDOWS[0].has_full_sqsk()


# --------------------------------------------------------------------------- #
# Distance and selection
# --------------------------------------------------------------------------- #


def test_avg_block_distance_is_mean_absolute_per_step_difference():
    # A[0]: g_sk 4000..5500 (mean 4750) vs B[0]: g_sk 9000 flat.
    assert match._avg_block_distance(A_WINDOWS[0], B_WINDOWS[0]) == (0.0, 4250.0)
    # A[1] is element-wise identical to B[1], and A[3] to B[2].
    assert match._avg_block_distance(A_WINDOWS[1], B_WINDOWS[1]) == (0.0, 0.0)
    assert match._avg_block_distance(A_WINDOWS[3], B_WINDOWS[2]) == (0, 0, 0, 0)
    # B[3] holds c_req 3 and c_sq 1500 flat, so the c_sq gap shrinks from 500
    # to 350 across the window and averages 425.
    assert match._avg_block_distance(A_WINDOWS[3], B_WINDOWS[3]) == (1.0, 0, 425.0, 0)


def test_avg_block_distance_returns_none_when_not_comparable():
    assert match._avg_block_distance(A_WINDOWS[0], B_WINDOWS[2]) is None  # phase
    assert match._avg_block_distance(A_WINDOWS[0], B_BLOCKS[0]) is None  # 16 vs 20
    empty = match.Block(phase=DECODE_ONLY, start_idx=0, end_idx=0)
    assert match._avg_block_distance(empty, empty) is None


def test_select_best_per_phase_picks_the_identical_windows():
    assert NOTES == []
    assert [
        (m["phase"], m["a_block_index"], m["b_block_index"], m["num_steps"])
        for m in MATCHES
    ] == [(DECODE_ONLY, 1, 1, 16), (PREFILLDECODE, 3, 2, 16)]
    assert [m["distance"] for m in MATCHES] == [[0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]


def test_select_best_per_phase_honours_the_requested_phase_list():
    matches, notes = match.select_best_per_phase(A_WINDOWS, B_WINDOWS, [DECODE_ONLY])
    assert [m["phase"] for m in matches] == [DECODE_ONLY]
    assert notes == []


def test_select_best_per_phase_skips_blocks_without_sqsk():
    """Native annotations carry no sq/sk, so no block is eligible."""
    names = [NATIVE_PREFILLDECODE] * 4 + [NATIVE_DECODE] * 4
    names += [NATIVE_PREFILLDECODE] * 4
    blocks = match.find_blocks([{"name": n} for n in names])
    assert [b.phase for b in blocks] == [PREFILLDECODE, DECODE_ONLY, PREFILLDECODE]
    assert not any(b.has_full_sqsk() for b in blocks)

    matches, notes = match.select_best_per_phase(blocks, blocks, PHASES)
    assert matches == []
    assert [n["phase"] for n in notes] == PHASES
    assert all("A blocks of phase: 0" in n["reason"] for n in notes)


def test_select_best_per_phase_notes_missing_same_size_candidate():
    """Strict size matching: A's 16-step windows cannot pair with a 20-step B."""
    b_wrong_size = [b for b in B_BLOCKS if b.num_steps == 20]
    matches, notes = match.select_best_per_phase(A_WINDOWS, b_wrong_size, [DECODE_ONLY])
    assert matches == []
    assert "with same-size B candidate: 0" in notes[0]["reason"]


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def test_write_reports(tmp_path):
    match.write_reports(MATCHES, NOTES, str(tmp_path), "trace_a", "trace_b")

    report = json.loads((tmp_path / "match_report.json").read_text())
    assert [e["match_id"] for e in report] == [0, 1]
    assert json.loads((tmp_path / "match_notes.json").read_text()) == []

    decode = report[0]
    assert decode["phase"] == DECODE_ONLY
    assert decode["num_steps"] == 16
    assert decode["distance"] == [0.0, 0.0]
    assert decode["TraceB"]["trace"] == "trace_b"
    assert decode["TraceB"]["block_index"] == 1

    a_side = decode["TraceA"]
    assert a_side["trace"] == "trace_a"
    assert a_side["block_index"] == 1
    assert a_side["output_path"] is None  # only set by main()
    assert (a_side["start"], a_side["end"]) == (16, 32)
    assert a_side["num_steps"] == 16
    assert a_side["original_num_steps"] == 48
    assert a_side["truncated"] is True
    assert a_side["avg_g_sq"] == 8.0
    assert a_side["avg_g_sk"] == 6350.0
    assert a_side["avg_context_requests"] == 0.0

    # Steps are numbered by absolute position in the source trace.
    steps = a_side["steps"]
    assert [s["step"] for s in steps] == list(range(16, 32))
    assert steps[0]["num_generation"] == 8
    assert steps[0]["num_context"] == 0
    assert [steps[0]["g_sk"], steps[-1]["g_sk"]] == [5600, 7100]

    prefilldecode = report[1]
    assert prefilldecode["phase"] == PREFILLDECODE
    assert prefilldecode["TraceA"]["avg_context_requests"] == 2.0
    assert prefilldecode["TraceA"]["avg_c_sq"] == 1075.0
    assert prefilldecode["TraceB"]["avg_c_sq"] == 1075.0

    df = pd.read_csv(tmp_path / "match_report.csv")
    assert len(df) == 2
    assert list(df["phase"]) == PHASES
    assert list(df["distance"]) == ["0.0|0.0", "0.0|0.0|0.0|0.0"]
    assert list(df["TraceA_start"]) == [16, 49]
    assert list(df["TraceB_start"]) == [21, 38]
    assert list(df["TraceA_original_num_steps"]) == [48, 32]
    assert list(df["TraceB_truncated"]) == [False, False]
    assert list(df["TraceA_avg_g_sk"]) == [6350.0, 4096.0]


def test_write_reports_records_notes(tmp_path):
    """A phase with no candidate at all becomes a note, not a match."""
    decode_only_b = [w for w in B_WINDOWS if w.phase == DECODE_ONLY]
    matches, notes = match.select_best_per_phase(A_WINDOWS, decode_only_b, PHASES)
    match.write_reports(matches, notes, str(tmp_path), "trace_a", "trace_b")

    written = json.loads((tmp_path / "match_notes.json").read_text())
    assert [n["phase"] for n in written] == [PREFILLDECODE]
    report = json.loads((tmp_path / "match_report.json").read_text())
    assert [e["phase"] for e in report] == [DECODE_ONLY]


def test_compute_output_path(tmp_path):
    decode_path = match.compute_output_path(
        str(tmp_path), DECODE_ONLY, "lbl_A", "trace_a", MATCHES[0]["a_block"]
    )
    assert decode_path == os.path.join(
        str(tmp_path),
        "decode_only",
        "lbl_A_prefilldecode_0_decode_16_bs8_conc8_trace_a.json.gz",
    )

    prefilldecode_path = match.compute_output_path(
        str(tmp_path), PREFILLDECODE, "lbl_B", "trace_b", MATCHES[1]["b_block"]
    )
    assert prefilldecode_path == os.path.join(
        str(tmp_path),
        "prefilldecode",
        "lbl_B_prefilldecode_16_decode_0_bs1083_conc10_trace_b.json.gz",
    )


def test_compute_output_path_of_single_step_block_uses_the_step_name(tmp_path):
    block = match.find_blocks([{"name": step(c_req=2, c_sq=64, c_sk=64)}])[0]
    path = match.compute_output_path(
        str(tmp_path), PREFILLDECODE, "lbl", "trace_a", block
    )
    assert os.path.basename(path) == (
        "lbl_execute_72_context_2_sq64sk64sqsq4096sqsk4096"
        "_generation_8_sq8sk4096sqsq64sqsk32768_trace_a.json.gz"
    )


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #


def test_extraction_of_a_matched_block(tmp_path):
    """End-to-end extraction of one matched block, as ``main()`` performs it.

    Also pins a known bug: ``compute_output_path`` (used to report a path under
    ``--no-extract``) disagrees with the name ``extract_and_save`` actually
    writes -- the latter has a ``prefill_{n}_`` prefix.
    """
    events = A_TRACE["traceEvents"]
    gpu_map, flow_map, meta = preprocess_trace(events)
    block = A_WINDOWS[1]
    label = "decode_only_best_A1_B1_A"

    summary = extract_and_save(
        [block.roots],
        events,
        A_TRACE,
        str(tmp_path),
        "trace_a",
        "annotation_iteration",
        0,
        1,
        gpu_map,
        flow_map,
        meta,
        output_label=label,
    )
    assert len(summary) == 1
    written = summary[0]["output_path"]
    assert os.path.exists(written)
    assert os.path.basename(written) == (
        f"{label}_prefill_0_prefilldecode_0_decode_16_bs8_conc8_trace_a.json.gz"
    )
    assert summary[0]["num_gpu_events"] == 32  # 16 roots x 2 kernels
    assert summary[0]["phase"] == {
        "num_prefill": 0,
        "num_prefilldecode": 0,
        "num_decode": 16,
        "avg_bs": 8,
        "avg_conc": 8,
    }

    predicted = match.compute_output_path(
        str(tmp_path), DECODE_ONLY, label, "trace_a", block
    )
    assert os.path.basename(predicted) != os.path.basename(written)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def trace_paths(tmp_path_factory):
    """A and B on disk, suffixed the way profiler output is."""
    directory = tmp_path_factory.mktemp("traces")
    paths = []
    for base, trace in (("run_a", A_TRACE), ("run_b", B_TRACE)):
        path = directory / f"{base}.pt.trace.json"
        path.write_text(json.dumps(trace))
        paths.append(str(path))
    return tuple(paths)


def run_main(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["match_inference_trace_blocks.py", *argv])
    match.main()


def test_load_trace(trace_paths):
    loaded = match.load_trace(trace_paths[0])

    assert loaded["trace_json"] == A_TRACE
    assert loaded["events"] == A_TRACE["traceEvents"]
    assert [r["name"] for r in loaded["iteration_roots"]] == TRACE_A_NAMES
    # 128 roots x 2 kernels, each kernel with its own correlation id.
    assert len(loaded["gpu_corr_map"]) == 256
    # The fixture has no flow events, and every event carries a timestamp.
    assert loaded["flow_corr_map"] == {}
    assert loaded["meta_events"] == []


def test_main_matches_and_extracts(tmp_path, trace_paths, monkeypatch):
    """The default invocation, end to end: same two winners as the in-process
    pipeline, plus one extracted trace file per side."""
    out_dir = tmp_path / "out"  # not pre-created: main() must make it
    run_main(monkeypatch, *trace_paths, "-o", str(out_dir))

    report = json.loads((out_dir / "match_report.json").read_text())
    assert [
        (e["phase"], e["TraceA"]["block_index"], e["TraceB"]["block_index"])
        for e in report
    ] == [(DECODE_ONLY, 1, 1), (PREFILLDECODE, 3, 2)]
    assert [e["distance"] for e in report] == [[0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    assert [e["TraceA"]["trace"] for e in report] == ["run_a", "run_a"]
    assert [e["TraceB"]["trace"] for e in report] == ["run_b", "run_b"]
    assert json.loads((out_dir / "match_notes.json").read_text()) == []
    assert len(pd.read_csv(out_dir / "match_report.csv")) == 2

    # output_path is the file extract_and_save actually wrote, so it exists.
    decode, prefilldecode = report
    assert os.path.basename(decode["TraceA"]["output_path"]) == (
        "decode_only_best_A1_B1_A_prefill_0_prefilldecode_0"
        "_decode_16_bs8_conc8_run_a.json.gz"
    )
    assert os.path.basename(prefilldecode["TraceB"]["output_path"]) == (
        "prefilldecode_best_A3_B2_B_prefill_0_prefilldecode_16"
        "_decode_0_bs1083_conc10_run_b.json.gz"
    )
    for entry in report:
        for side in ("TraceA", "TraceB"):
            written = entry[side]["output_path"]
            assert os.path.dirname(written) == str(out_dir / entry["phase"])
            assert os.path.exists(written)


def test_main_no_extract_reports_paths_without_writing_them(
    tmp_path, trace_paths, monkeypatch
):
    """--no-extract still reports a path per side, but writes no trace file.

    This is the CLI-level face of the naming bug pinned by
    ``test_extraction_of_a_matched_block``: the reported path is the
    ``compute_output_path`` prediction, which omits the ``prefill_{n}_``
    prefix and so does not name the file extraction would have produced.
    """
    out_dir = tmp_path / "out"
    run_main(monkeypatch, *trace_paths, "-o", str(out_dir), "--no-extract")

    report = json.loads((out_dir / "match_report.json").read_text())
    assert len(report) == 2
    for entry in report:
        for side in ("TraceA", "TraceB"):
            predicted = entry[side]["output_path"]
            assert os.path.dirname(predicted) == str(out_dir / entry["phase"])
            assert not os.path.exists(predicted)
    assert list(out_dir.rglob("*.json.gz")) == []


def test_main_without_windowing_finds_no_same_size_pair(
    tmp_path, trace_paths, monkeypatch
):
    """--num-steps 0 disables windowing, leaving whole blocks to match by size.

    None of A's blocks (48, 32, 44 steps) shares a size with any of B's
    (20, 16, 16, 40, 32), so both phases fall through to notes.
    """
    out_dir = tmp_path / "out"
    run_main(
        monkeypatch,
        *trace_paths,
        "-o",
        str(out_dir),
        "--no-extract",
        "--num-steps",
        "0",
    )

    assert json.loads((out_dir / "match_report.json").read_text()) == []
    notes = json.loads((out_dir / "match_notes.json").read_text())
    assert [n["phase"] for n in notes] == PHASES
    assert all("with same-size B candidate: 0" in n["reason"] for n in notes)
    # No rows, so write_reports skips the CSV entirely.
    assert not (out_dir / "match_report.csv").exists()


def test_main_reports_a_phase_with_no_eligible_pair(tmp_path, trace_paths, monkeypatch):
    """prefill_only never occurs in the fixtures, so it yields a note only."""
    out_dir = tmp_path / "out"
    run_main(monkeypatch, *trace_paths, "-o", str(out_dir), "--phases", "prefill_only")

    assert json.loads((out_dir / "match_report.json").read_text()) == []
    notes = json.loads((out_dir / "match_notes.json").read_text())
    assert [n["phase"] for n in notes] == [match.PHASE_PREFILL_ONLY]
    assert list(out_dir.rglob("*.json.gz")) == []


def test_main_rejects_an_unknown_phase(tmp_path, trace_paths, monkeypatch, capsys):
    out_dir = tmp_path / "out"
    with pytest.raises(SystemExit) as excinfo:
        run_main(monkeypatch, *trace_paths, "-o", str(out_dir), "--phases", "decode")

    assert excinfo.value.code == 2
    assert "Unknown phase(s): ['decode']" in capsys.readouterr().err
    assert not out_dir.exists()  # rejected before any output is created
