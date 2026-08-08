###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""CPU-only coverage for split_inference_trace_annotation CLI and edge cases."""

from __future__ import annotations

import gzip
import json
import os
import sys
import zipfile

import pytest

from TraceLens.TraceUtils import split_inference_trace_annotation as split

VLLM_PRIMARY = (
    "execute_{i}_context_3(sq128sk256sqsq1sqsk1)_generation_2(sq1sk300sqsq1sqsk1)"
)
SGLANG_DECODE = "step[DECODE bs={i}]"
SGLANG_EXTEND = "step[EXTEND bs=2 toks={t}]"
VLLM_BACKUP = "execute_context_3({i})_generation_2(50)"


def _make_trace(root_names, pid=1, cpu_tid=10, gpu_tid=99):
    events = []
    corr = 1000
    for i, name in enumerate(root_names):
        base = 1_000 + i * 1_000
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
        corrs = [corr + 3 * i + j for j in range(3)]
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


def _mixed_phase_roots(count=40):
    """Alternating prefill-decode and decode-only annotation roots."""
    names = []
    for i in range(count):
        if i % 3 == 0:
            names.append(SGLANG_EXTEND.format(t=800))
        else:
            names.append(SGLANG_DECODE.format(i=20))
    return names


def test_get_filename_zip_raises_when_no_json(tmp_path):
    zip_path = tmp_path / "empty.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("readme.txt", "no json here")
    with pytest.raises(ValueError, match="No .json file found"):
        split.get_filename(str(zip_path))


def test_find_iteration_roots_backup_fallback(capsys):
    names = [SGLANG_DECODE.format(i=20) for _ in range(8)]
    trace = _make_trace(names)
    roots = split.find_iteration_roots(trace["traceEvents"])
    captured = capsys.readouterr().out
    assert "falling back to backup patterns" in captured
    assert roots is not None
    assert len(roots) == 8


def test_find_iteration_roots_generic_fallback(capsys):
    events = [
        {
            "ph": "X",
            "cat": "cpu_op",
            "name": "training_loop",
            "pid": 1,
            "tid": 1,
            "ts": 0,
            "dur": 5000,
            "args": {"Sequence number": 0},
        }
    ]
    corr = 400
    for iteration in range(2):
        base_ts = 100 + iteration * 2000
        op = {
            "ph": "X",
            "cat": "cpu_op",
            "name": "iter_fwd",
            "pid": 1,
            "tid": 1,
            "ts": base_ts,
            "dur": 300,
            "args": {"Sequence number": iteration, "correlation": corr},
        }
        events.append(op)
        events.extend(
            [
                {
                    "ph": "X",
                    "cat": "kernel",
                    "name": "k",
                    "pid": 0,
                    "tid": 7,
                    "ts": base_ts + 50,
                    "dur": 20,
                    "args": {"correlation": corr, "stream": 7},
                },
                {"ph": "s", "id": corr, "pid": 0, "tid": 7, "ts": base_ts + 50, "cat": "ac2g", "name": "ac2g"},
                {"ph": "f", "id": corr, "pid": 0, "tid": 7, "ts": base_ts + 70, "cat": "ac2g", "name": "ac2g", "bp": "e"},
            ]
        )
        corr += 1

    split.find_iteration_roots(events)
    captured = capsys.readouterr().out
    assert "trying generic call-tree traversal" in captured


def test_find_steady_state_window_mixed_mode_with_conc_osl_r():
    names = _mixed_phase_roots(48)
    trace = _make_trace(names)
    roots = split.find_iteration_roots(trace["traceEvents"])
    regions, _ = split.identify_steady_state_regions(
        split.iteration_details(roots), num_steps=16
    )
    window = split.find_steady_state_window(
        roots,
        num_steps=4,
        steady_state_regions=regions,
        mode="mixed",
        CONC=20,
        OSL=100.0,
        R=1.5,
    )
    assert len(window) >= 1


def test_find_steady_state_window_mixed_no_pd_candidates(capsys):
    names = [SGLANG_DECODE.format(i=20) for _ in range(24)]
    trace = _make_trace(names)
    roots = split.find_iteration_roots(trace["traceEvents"])
    window = split.find_steady_state_window(
        roots,
        num_steps=8,
        steady_state_regions=[(0, len(roots))],
        mode="mixed",
    )
    captured = capsys.readouterr().out
    assert "falling back to the full candidate set" in captured or len(window) >= 1


def test_find_steady_state_window_decode_only_no_pure_run():
    names = [SGLANG_EXTEND.format(t=800) for _ in range(16)]
    trace = _make_trace(names)
    roots = split.find_iteration_roots(trace["traceEvents"])
    window = split.find_steady_state_window(
        roots,
        num_steps=8,
        steady_state_regions=[(0, len(roots))],
        mode="decode_only",
    )
    assert window == []


def test_find_steady_state_window_max_prefilldecode():
    names = []
    for i in range(24):
        if i < 8:
            names.append(SGLANG_EXTEND.format(t=800))
        else:
            names.append(SGLANG_DECODE.format(i=20))
    trace = _make_trace(names)
    roots = split.find_iteration_roots(trace["traceEvents"])
    window = split.find_steady_state_window(
        roots,
        num_steps=8,
        steady_state_regions=[(0, len(roots))],
        mode="max_prefilldecode",
    )
    assert len(window) >= 1


def test_find_steady_state_window_max_prefilldecode_empty():
    names = [SGLANG_DECODE.format(i=20) for _ in range(16)]
    trace = _make_trace(names)
    roots = split.find_iteration_roots(trace["traceEvents"])
    window = split.find_steady_state_window(
        roots,
        num_steps=8,
        steady_state_regions=[(0, len(roots))],
        mode="max_prefilldecode",
    )
    assert window == []


def test_find_steady_state_window_invalid_mode():
    names = [VLLM_PRIMARY.format(i=i) for i in range(8)]
    trace = _make_trace(names)
    roots = split.find_iteration_roots(trace["traceEvents"])
    with pytest.raises(ValueError, match="Unknown mode"):
        split.find_steady_state_window(
            roots, num_steps=4, steady_state_regions=[(0, 8)], mode="invalid"
        )


def test_find_steady_state_window_conc_mismatch_warning(capsys):
    names = [SGLANG_DECODE.format(i=5) for _ in range(16)]
    trace = _make_trace(names)
    roots = split.find_iteration_roots(trace["traceEvents"])
    split.find_steady_state_window(
        roots,
        num_steps=8,
        steady_state_regions=[(0, len(roots))],
        mode="mixed",
        CONC=999,
    )
    assert "expected peak concurrency" in capsys.readouterr().out


def test_extract_and_save_empty_roots(tmp_path):
    trace = _make_trace([VLLM_PRIMARY.format(i=0)])
    events = trace["traceEvents"]
    gpu_map, flow_map, meta = split.preprocess_trace(events)
    summary = split.extract_and_save(
        [[]],
        events,
        trace,
        str(tmp_path),
        "trace",
        "annotation_iteration",
        0,
        1,
        gpu_map,
        flow_map,
        meta,
    )
    assert summary == []


def test_extract_phases_and_save_decode_branch(tmp_path):
    names = []
    for i in range(8):
        if i % 2 == 0:
            names.append(SGLANG_EXTEND.format(t=800))
        else:
            names.append(SGLANG_DECODE.format(i=20))
    trace = _make_trace(names)
    events = trace["traceEvents"]
    gpu_map, flow_map, meta = split.preprocess_trace(events)
    roots = split.find_iteration_roots(events)
    summary = split.extract_phases_and_save(
        [roots],
        events,
        trace,
        str(tmp_path),
        "trace",
        "annotation_iteration",
        0,
        1,
        gpu_map,
        flow_map,
        meta,
    )
    assert len(summary) >= 2
    labels = {os.path.basename(item["output_path"]) for item in summary}
    assert any("prefilldecode_" in name for name in labels)
    assert any("decode_" in name for name in labels)


def test_extract_phases_and_save_skips_non_annotation_prefix(tmp_path):
    trace = _make_trace([VLLM_PRIMARY.format(i=0)])
    events = trace["traceEvents"]
    gpu_map, flow_map, meta = split.preprocess_trace(events)
    roots = split.find_iteration_roots(events)
    summary = split.extract_phases_and_save(
        [[r] for r in roots],
        events,
        trace,
        str(tmp_path),
        "trace",
        "run_iteration",
        0,
        1,
        gpu_map,
        flow_map,
        meta,
    )
    assert summary == []


def test_identify_steady_state_regions_end_and_middle():
    details = [{"num_requests": 2} for _ in range(6)]
    details += [{"num_requests": 20} for _ in range(20)]
    details += [{"num_requests": 2} for _ in range(6)]
    regions, global_max = split.identify_steady_state_regions(details, num_steps=16)
    assert global_max == 20
    assert len(regions) >= 1


def test_compute_reference_pd_ratio_median_fallback(capsys):
    iter_details = []
    for i in range(20):
        iter_details.append(
            {
                "num_requests": 20,
                "context_requests": 3 if i < 10 else 0,
                "generation_requests": 2 if i >= 10 else 20,
            }
        )
    regions = [(0, 20)]
    _, ref_ratio, _ = split.compute_reference_pd_ratio(regions, iter_details)
    assert 0.0 <= ref_ratio <= 1.0


def test_main_store_single_iteration(tmp_path):
    names = [VLLM_PRIMARY.format(i=i) for i in range(6)]
    trace = _make_trace(names)
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(json.dumps(trace))
    out_dir = tmp_path / "out"

    old_argv = sys.argv
    sys.argv = [
        "split_inference_trace_annotation",
        str(trace_path),
        "-o",
        str(out_dir),
        "--store-single-iteration",
        "--iterations",
        "0:3",
    ]
    try:
        split.main()
    finally:
        sys.argv = old_argv

    assert out_dir.exists()
    assert (out_dir / "execution_details.json").exists()
    assert (out_dir / "execution_details.csv").exists()


def test_main_find_steady_state(tmp_path):
    names = _mixed_phase_roots(40)
    trace = _make_trace(names)
    trace_path = tmp_path / "trace.json.gz"
    with gzip.open(trace_path, "wt", encoding="utf-8") as f:
        json.dump(trace, f)
    out_dir = tmp_path / "steady"

    old_argv = sys.argv
    sys.argv = [
        "split_inference_trace_annotation",
        str(trace_path),
        "-o",
        str(out_dir),
        "--find-steady-state",
        "--num-steps",
        "8",
    ]
    try:
        split.main()
    finally:
        sys.argv = old_argv

    assert (out_dir / "execution_details.json").exists()
    with open(out_dir / "execution_details.json") as f:
        details = json.load(f)
    labels = {os.path.basename(d["output_path"]) for d in details}
    assert any("mixed_steady_state" in name for name in labels)
    assert any("decode_only_steady_state" in name for name in labels)


def test_main_divide_phases(tmp_path):
    names = _mixed_phase_roots(24)
    trace = _make_trace(names)
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(json.dumps(trace))
    out_dir = tmp_path / "phases"

    old_argv = sys.argv
    sys.argv = [
        "split_inference_trace_annotation",
        str(trace_path),
        "-o",
        str(out_dir),
        "--divide-phases",
        "--num-steps",
        "12",
    ]
    try:
        split.main()
    finally:
        sys.argv = old_argv

    assert os.path.isdir(out_dir / "prefilldecodemix") or os.path.isdir(out_dir / "decode_only")


def test_main_explicit_iteration_range(tmp_path):
    names = [VLLM_BACKUP.format(i=100 + i) for i in range(12)]
    trace = _make_trace(names)
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(json.dumps(trace))
    out_dir = tmp_path / "range"

    old_argv = sys.argv
    sys.argv = [
        "split_inference_trace_annotation",
        str(trace_path),
        "-o",
        str(out_dir),
        "--iterations",
        "2:6",
    ]
    try:
        split.main()
    finally:
        sys.argv = old_argv

    assert out_dir.exists()


def test_main_dummy_runs_and_steady_state_message(tmp_path, capsys):
    dummy_name = "vllm/v1/worker/gpu_model_runner.py(99): _dummy_run"
    trace = {
        "traceEvents": [
            {
                "name": dummy_name,
                "cat": "cpu_op",
                "ph": "X",
                "ts": 0,
                "dur": 100,
                "tid": 1,
                "pid": 1,
                "args": {},
            }
        ],
        "schemaVersion": 1,
    }
    trace_path = tmp_path / "dummy_trace.json"
    trace_path.write_text(json.dumps(trace))
    out_dir = tmp_path / "dummy"

    old_argv = sys.argv
    sys.argv = [
        "split_inference_trace_annotation",
        str(trace_path),
        "-o",
        str(out_dir),
        "--dummy",
        "0:1",
        "--find-steady-state",
    ]
    try:
        split.main()
    finally:
        sys.argv = old_argv

    assert "finding steady state without annotations not supported" in capsys.readouterr().out
