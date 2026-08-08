###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the inference trace splitting tool.

Covers:
- Annotation pattern selection with primary/backup fallback
  (``find_iteration_roots``) and the priority of the primary 1211 pattern.
- Per-iteration splitting / time-window isolation (``extract_iteration``).
- Annotation-name parsing of the names these fixtures generate.
- Steady-state window finding (``identify_steady_state_regions`` and
  ``find_steady_state_window``).

No trace files are written; everything operates on in-memory dicts.
"""

import gzip
import json
import os
import zipfile
from typing import Dict, List

from TraceLens.TraceUtils import split_inference_trace_annotation as split
from TraceLens.TraceUtils.annotation_utils import IterationAnnotation

# --------------------------------------------------------------------------- #
# Dummy-trace builder
# --------------------------------------------------------------------------- #

VLLM_PRIMARY_ANNOTATION = (
    "execute_{i}_context_3(sq128sk256sqsq1sqsk1)_generation_2(sq1sk300sqsq1sqsk1)"
)
SGLANG_DECODE_ANNOTATION = "step[DECODE bs={i}]"
SGLANG_EXTEND_ANNOTATION = "step[EXTEND bs=2 toks={t}]"
VLLM_BACKUP_ANNOTATION = "execute_context_3({i})_generation_2(50)"


def make_trace(root_names, pid=1, cpu_tid=10, gpu_tid=99):
    """Build a trace with one annotation root per name.

    Each root has 3 cpu_ops (each carrying ``args.correlation``) inside its time
    window and 2 kernels linked to the first two correlations. Kernels are placed
    outside the root window on purpose: ``extract_iteration`` attributes them via
    the correlation map, not by time.
    """
    events = []
    corr = 1000
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
        corrs = [corr + 3 * i + j for j in range(3)]
        for j, c in enumerate(corrs):  # 3 cpu_ops inside the root window
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
        for j, c in enumerate(corrs[:2]):  # 2 kernels linked by correlation
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


# --------------------------------------------------------------------------- #
# Scenario 1: 1211 only -- selection, parsing, and splitting/window isolation
# --------------------------------------------------------------------------- #


def test_1211_only_selection_and_splitting():
    names = [VLLM_PRIMARY_ANNOTATION.format(i=i) for i in range(16)]
    trace = make_trace(names)
    events = trace["traceEvents"]
    gpu_map, flow_map, meta = split.preprocess_trace(events)

    roots = split.find_iteration_roots(events)
    assert roots is not None
    assert len(roots) == 16

    # Single-root window: verify the splitting logic isolates exactly one root.
    k = 7
    out, batch_list, num_gpu_events, gpu_dur, gpu_busy = split.extract_iteration(
        [roots[k]], events, trace, gpu_map, flow_map, meta
    )
    out_names = {e["name"] for e in out["traceEvents"]}
    assert num_gpu_events == 2
    assert gpu_busy == 16  # 2 kernels x dur 8
    assert gpu_dur > 0
    assert f"kernel_{k}_0" in out_names and f"kernel_{k}_1" in out_names
    assert {f"cpu_op_{k}_0", f"cpu_op_{k}_1", f"cpu_op_{k}_2"} <= out_names
    # No events from neighboring roots leak into this window.
    for j in range(16):
        if j == k:
            continue
        assert f"kernel_{j}_0" not in out_names
        assert f"cpu_op_{j}_0" not in out_names

    # Multi-root window: all 16 roots together.
    _, _, num_gpu_events_all, _, _ = split.extract_iteration(
        roots, events, trace, gpu_map, flow_map, meta
    )
    assert num_gpu_events_all == 32  # 16 x 2

    details = IterationAnnotation(names[0]).iter_details()
    assert details == {
        "batch_size": 129,
        "num_requests": 5,
        "context_requests": 3,
        "context_sum": 128,
        "generation_requests": 2,
        "generation_sum": 1,
    }


# --------------------------------------------------------------------------- #
# Scenario 2: 1211 + 1219 both present -- primary (1211) is prioritized
# --------------------------------------------------------------------------- #


def test_1211_prioritized_over_1219():
    names = [VLLM_PRIMARY_ANNOTATION.format(i=i) for i in range(16)]
    names += [SGLANG_DECODE_ANNOTATION.format(i=20) for _ in range(16)]
    trace = make_trace(names)

    roots = split.find_iteration_roots(trace["traceEvents"])
    assert roots is not None
    assert len(roots) == 16  # only the 1211 roots
    primary = split.ITERATION_PATTERNS[0]
    for r in roots:
        assert primary.match(r["name"])
        assert not r["name"].startswith("step[")


# --------------------------------------------------------------------------- #
# Scenario 3: 1219 only -- backup fallback is used
# --------------------------------------------------------------------------- #


def test_1219_only_uses_backup():
    names = []
    for i in range(16):
        if i % 2 == 0:
            names.append(SGLANG_DECODE_ANNOTATION.format(i=20))
        else:
            names.append(SGLANG_EXTEND_ANNOTATION.format(t=800))
    trace = make_trace(names)

    roots = split.find_iteration_roots(trace["traceEvents"])
    assert roots is not None
    assert len(roots) == 16

    decode = IterationAnnotation("step[DECODE bs=20]").iter_details()
    assert decode["generation_requests"] == 20
    assert decode["context_requests"] == 0

    extend = IterationAnnotation("step[EXTEND bs=2 toks=800]").iter_details()
    assert extend["context_requests"] == 2
    assert extend["generation_requests"] == 0


# --------------------------------------------------------------------------- #
# Scenario 4: 1213 only -- backup fallback is used
# --------------------------------------------------------------------------- #


def test_1213_only_uses_backup():
    names = [VLLM_BACKUP_ANNOTATION.format(i=100 + i) for i in range(16)]
    trace = make_trace(names)

    roots = split.find_iteration_roots(trace["traceEvents"])
    assert roots is not None
    assert len(roots) == 16

    details = IterationAnnotation(
        "execute_context_3(100)_generation_2(50)"
    ).iter_details()
    assert details["context_requests"] == 3
    assert details["context_sum"] == 100
    assert details["generation_requests"] == 2
    assert details["generation_sum"] == 50


# --------------------------------------------------------------------------- #
# Steady-state window finding
# --------------------------------------------------------------------------- #


def _details(num_requests, context_requests=0):
    return {"num_requests": num_requests, "context_requests": context_requests}


def test_identify_steady_state_regions_clear_region():
    iter_details = [_details(2) for _ in range(4)] + [_details(20) for _ in range(30)]
    regions, global_max = split.identify_steady_state_regions(
        iter_details, num_steps=32
    )
    assert global_max == 20
    assert regions == [(4, 33)]


def test_identify_steady_state_regions_fallback():
    iter_details = [_details(20 if i % 2 == 0 else 2) for i in range(10)]
    regions, global_max = split.identify_steady_state_regions(
        iter_details, num_steps=12
    )
    assert global_max == 20
    assert len(regions) == 1
    assert regions == [(4, 6)]


def test_find_steady_state_window_returns_contiguous_slice():
    roots = [
        {
            "name": SGLANG_DECODE_ANNOTATION.format(i=20),
            "cat": "user_annotation",
            "ts": i,
            "dur": 1,
        }
        for i in range(32)
    ]
    window = split.find_steady_state_window(
        roots, num_steps=8, steady_state_regions=[(0, 32)], mode="decode_only"
    )
    assert len(window) == 8
    # The window is a contiguous slice of the original roots.
    start = roots.index(window[0])
    assert window == roots[start : start + 8]


def test_parse_range_variants():
    assert split.parse_range("all", 16) == (0, 16)
    assert split.parse_range("5", 16) == (5, 6)
    assert split.parse_range("10:20", 16) == (10, 16)


def test_preprocess_trace_collects_flow_and_gpu_maps():
    events = [
        {"ph": "M", "name": "process_name", "pid": 1},
        {"ph": "s", "id": 7, "ts": 0, "pid": 1, "tid": 1},
        {"ph": "f", "id": 7, "ts": 10, "pid": 1, "tid": 1},
        {
            "name": "kernel_a",
            "cat": "kernel",
            "ph": "X",
            "ts": 5,
            "dur": 3,
            "pid": 1,
            "tid": 2,
            "args": {"correlation": 7},
        },
    ]
    gpu_map, flow_map, meta = split.preprocess_trace(events)
    assert 7 in gpu_map and gpu_map[7][0]["name"] == "kernel_a"
    assert 7 in flow_map and len(flow_map[7]) == 2
    assert len(meta) == 1


def test_extract_iteration_empty_roots():
    trace = make_trace([VLLM_PRIMARY_ANNOTATION.format(i=0)])
    events = trace["traceEvents"]
    gpu_map, flow_map, meta = split.preprocess_trace(events)
    out, batch_list, num_gpu, gpu_dur, gpu_busy = split.extract_iteration(
        [], events, trace, gpu_map, flow_map, meta
    )
    assert out["traceEvents"] == trace["traceEvents"]
    assert batch_list == []
    assert num_gpu == 0
    assert gpu_dur == 0
    assert gpu_busy == 0


def test_extract_and_save_writes_gzip(tmp_path):
    names = [VLLM_PRIMARY_ANNOTATION.format(i=i) for i in range(4)]
    trace = make_trace(names)
    events = trace["traceEvents"]
    gpu_map, flow_map, meta = split.preprocess_trace(events)
    roots = split.find_iteration_roots(events)
    grouped = [[r] for r in roots]
    summary = split.extract_and_save(
        grouped,
        events,
        trace,
        str(tmp_path),
        "trace",
        "annotation_iteration",
        0,
        2,
        gpu_map,
        flow_map,
        meta,
    )
    assert len(summary) == 2
    assert os.path.exists(summary[0]["output_path"])
    with gzip.open(summary[0]["output_path"], "rt", encoding="utf-8") as f:
        loaded = json.load(f)
    assert len(loaded["traceEvents"]) > 0


def test_extract_phases_and_save(tmp_path):
    names = [VLLM_PRIMARY_ANNOTATION.format(i=i) for i in range(8)]
    trace = make_trace(names)
    events = trace["traceEvents"]
    gpu_map, flow_map, meta = split.preprocess_trace(events)
    roots = split.find_iteration_roots(events)
    summary = split.extract_phases_and_save(
        [[r] for r in roots],
        events,
        trace,
        str(tmp_path),
        "trace",
        "annotation_iteration",
        0,
        8,
        gpu_map,
        flow_map,
        meta,
    )
    assert len(summary) >= 1
    assert all(os.path.exists(item["output_path"]) for item in summary)


def test_compute_reference_pd_ratio():
    iter_details = [_details(20 if i % 2 else 2) for i in range(20)]
    regions = [(0, 20)]
    (start, end), avg_ratio, largest_ratio = split.compute_reference_pd_ratio(
        regions, iter_details
    )
    assert (start, end) == (0, 20)
    assert 0.0 <= avg_ratio <= 1.0
    assert 0.0 <= largest_ratio <= 1.0


def test_find_steady_state_window_decode_only_mode():
    roots = [
        {
            "name": SGLANG_DECODE_ANNOTATION.format(i=20),
            "cat": "user_annotation",
            "ts": i,
            "dur": 1,
        }
        for i in range(32)
    ]
    window = split.find_steady_state_window(
        roots,
        num_steps=8,
        steady_state_regions=[(0, 32)],
        mode="decode_only",
    )
    assert len(window) == 8


def test_divide_phases_and_save(tmp_path):
    names = [VLLM_PRIMARY_ANNOTATION.format(i=i) for i in range(12)]
    trace = make_trace(names)
    events = trace["traceEvents"]
    gpu_map, flow_map, meta = split.preprocess_trace(events)
    roots = split.find_iteration_roots(events)
    summary = split.divide_phases_and_save(
        roots,
        events,
        trace,
        str(tmp_path),
        "trace",
        gpu_map,
        flow_map,
        meta,
        steady_state_regions=[(0, len(roots))],
    )
    assert len(summary) >= 1
    assert any("prefilldecodemix" in item["output_path"] for item in summary)


def test_get_filename_json_and_zip(tmp_path):
    trace = make_trace([VLLM_PRIMARY_ANNOTATION.format(i=0)])
    json_path = tmp_path / "trace.json"
    json_path.write_text(json.dumps(trace))
    assert split.get_filename(str(json_path)) == str(json_path)

    zip_path = tmp_path / "trace.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("inner/trace.json", json.dumps(trace))
    assert split.get_filename(str(zip_path)) == "inner/trace.json"


def test_find_iteration_roots_generic_fallback():
    from TraceLens.Trace2Tree.inference_iteration_roots import (
        find_iteration_roots_generic,
    )

    events: List[Dict] = []
    events.append(
        {
            "ph": "X",
            "cat": "cpu_op",
            "name": "training_loop",
            "pid": 1,
            "tid": 1,
            "ts": 0,
            "dur": 7000,
            "args": {"Sequence number": 0},
        }
    )
    corr = 300
    for iteration in range(3):
        base_ts = 100 + iteration * 2000
        for step_name, offset in [("iter_fwd", 0), ("iter_bwd", 400)]:
            op = {
                "ph": "X",
                "cat": "cpu_op",
                "name": step_name,
                "pid": 1,
                "tid": 1,
                "ts": base_ts + offset,
                "dur": 300,
                "args": {"Sequence number": iteration, "correlation": corr},
            }
            events.append(op)
            events.extend(
                [
                    {
                        "ph": "X",
                        "cat": "cuda_runtime",
                        "name": "hipLaunchKernel",
                        "pid": 1,
                        "tid": 1,
                        "ts": base_ts + offset + 10,
                        "dur": 5,
                        "args": {"correlation": corr},
                    },
                    {
                        "ph": "X",
                        "cat": "kernel",
                        "name": f"{step_name}_kernel",
                        "pid": 0,
                        "tid": 7,
                        "ts": base_ts + offset + 50,
                        "dur": 20,
                        "args": {"correlation": corr, "stream": 7},
                    },
                    {
                        "ph": "s",
                        "id": corr,
                        "pid": 0,
                        "tid": 7,
                        "ts": base_ts + offset + 50,
                        "cat": "ac2g",
                        "name": "ac2g",
                    },
                    {
                        "ph": "f",
                        "id": corr,
                        "pid": 0,
                        "tid": 7,
                        "ts": base_ts + offset + 70,
                        "cat": "ac2g",
                        "name": "ac2g",
                        "bp": "e",
                    },
                ]
            )
            corr += 1

    roots = find_iteration_roots_generic(events)
    assert roots is not None
    assert len(roots) >= 1
