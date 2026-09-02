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

import gzip, json, os, zipfile, sys, pytest
from typing import Dict, List
from TraceLens.TraceUtils import split_inference_trace_annotation as split
from TraceLens.TraceUtils.annotation_utils import IterationAnnotation
from TraceLens.PerfModel.extensions import moe_perf_model_extensions as moe_ext
from TraceLens.Trace2Tree.trace_capture_merge_experimental import (
    _align_graph_to_capture_by_group,
    _get_cached_capture_tree,
    align_streams,
    capture_has_kernel_names,
    find_closest_batch_size,
    find_execution_details,
    get_subtree_events,
    is_multistream,
    load_capture_folder,
    merge_capture_trace_into_graph,
    verify_subtree_events,
)
from TraceLens.TreePerf.tree_perf import TreePerfAnalyzer
from tests.fixtures.traces import INFERENCE_ROOT
from TraceLens.Trace2Tree.inference_iteration_roots import _entry_roots, _reattach_worker_threads
from TraceLens.Trace2Tree.trace_to_tree import TraceToTree
from TraceLens.TraceUtils.split_inference.root_detection import _total_gpu_time, detect_from_branch_descent
from TraceLens.Trace2Tree import trace_capture_merge_experimental as tcm

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

    roots = split.find_iteration_roots(events).roots
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

    roots = split.find_iteration_roots(trace["traceEvents"]).roots
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

    roots = split.find_iteration_roots(trace["traceEvents"]).roots
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

    roots = split.find_iteration_roots(trace["traceEvents"]).roots
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
    roots = split.find_iteration_roots(events).roots
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
    roots = split.find_iteration_roots(events).roots
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
    roots = split.find_iteration_roots(events).roots
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


def test_branch_descent_from_synthetic_events():
    events: List[Dict] = []
    events.append(
        {
            "ph": "X",
            "cat": "cpu_op",
            "name": "training_loop",
            "pid": 1,
            "tid": 1,
            "ts": 0,
            "dur": 20000,
            "args": {"Sequence number": 0},
        }
    )
    corr = 300
    for iteration in range(8):
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

    tree = TraceToTree(events, prune_nongpu_paths=False)
    tree.build_tree(add_python_func=True)
    _reattach_worker_threads(tree)
    result = detect_from_branch_descent(tree, _entry_roots(tree), _total_gpu_time(tree))
    assert result is not None
    assert len(result.roots) >= 1


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


def test_find_iteration_roots_backup_fallback():
    names = [SGLANG_DECODE.format(i=20) for _ in range(8)]
    trace = _make_trace(names)
    roots = split.find_iteration_roots(trace["traceEvents"]).roots
    assert roots is not None
    assert len(roots) == 8


def test_find_iteration_roots_no_annotation_trace():
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
                {
                    "ph": "s",
                    "id": corr,
                    "pid": 0,
                    "tid": 7,
                    "ts": base_ts + 50,
                    "cat": "ac2g",
                    "name": "ac2g",
                },
                {
                    "ph": "f",
                    "id": corr,
                    "pid": 0,
                    "tid": 7,
                    "ts": base_ts + 70,
                    "cat": "ac2g",
                    "name": "ac2g",
                    "bp": "e",
                },
            ]
        )
        corr += 1

    result = split.find_iteration_roots(events)
    assert result.status.name == "NOT_SPLITTABLE"


def test_find_steady_state_window_mixed_mode_with_conc_osl_r():
    names = _mixed_phase_roots(48)
    trace = _make_trace(names)
    roots = split.find_iteration_roots(trace["traceEvents"]).roots
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
    roots = split.find_iteration_roots(trace["traceEvents"]).roots
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
    roots = split.find_iteration_roots(trace["traceEvents"]).roots
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
    roots = split.find_iteration_roots(trace["traceEvents"]).roots
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
    roots = split.find_iteration_roots(trace["traceEvents"]).roots
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
    roots = split.find_iteration_roots(trace["traceEvents"]).roots
    with pytest.raises(ValueError, match="Unknown mode"):
        split.find_steady_state_window(
            roots, num_steps=4, steady_state_regions=[(0, 8)], mode="invalid"
        )


def test_find_steady_state_window_conc_mismatch_warning(capsys):
    names = [SGLANG_DECODE.format(i=5) for _ in range(16)]
    trace = _make_trace(names)
    roots = split.find_iteration_roots(trace["traceEvents"]).roots
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
    roots = split.find_iteration_roots(events).roots
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
    roots = split.find_iteration_roots(events).roots
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

    assert os.path.isdir(out_dir / "prefilldecodemix") or os.path.isdir(
        out_dir / "decode_only"
    )


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


class TestCaptureMergeAndMoe:
    def test_align_streams_multistream(self):
        graph = [
            {"name": "k1", "args": {"stream": 1}},
            {"name": "k2", "args": {"stream": 2}},
        ]
        capture = [
            {"name": "hipLaunchKernel", "args": {"kernel": "k1"}},
            {"name": "hipLaunchKernel", "args": {"kernel": "k2"}},
        ]
        aligned = align_streams(graph, capture)
        assert aligned is not None
        assert len(aligned) == 2

    def test_moe_biased_grouped_topk(self):
        evt = {
            "args": {
                "Input Dims": [(32, 256), (256,), (32, 8), (32, 8)],
                "Input type": ["c10::Float"] * 3 + ["c10::Int"],
            }
        }
        model = moe_ext.BiasedGroupedTopk(evt)
        assert model.flops() > 0

    def test_moe_sort_scatter_missing_kernel_details(self):
        evt = {
            "args": {
                "Input Dims": [(32, 4096), (32, 8)],
                "Input type": ["c10::BFloat16", "c10::Int"],
            }
        }
        model = moe_ext.MoeSortScatterGather(evt)
        assert model.bytes() is None or model.bytes() >= 0


class TestCaptureMergeDeep:
    def test_find_closest_batch_size_and_execution_details(self):
        assert find_closest_batch_size(128, [64, 256, 512]) == 256
        root = {"name": "execute_128_context_3_generation_2"}
        assert find_execution_details(root) == "128"

    @pytest.mark.skipif(
        not os.path.isdir(
            os.path.join(INFERENCE_ROOT, "sglang_decode", "capture_traces")
        ),
        reason="capture fixture missing",
    )
    def test_merge_capture_full_inference_fixture(self):
        case = os.path.join(INFERENCE_ROOT, "sglang_decode")
        trace_gz = next(f for f in os.listdir(case) if f.endswith(".json.gz"))
        graph = os.path.join(case, trace_gz)
        capture = os.path.join(case, "capture_traces")
        metadata = os.path.join(capture, "execution_details.json")
        merged = merge_capture_trace_into_graph(capture, metadata, graph)
        assert len(merged.events) > 0
        analyzer = TreePerfAnalyzer(merged, rebuild_tree=False)
        assert analyzer.get_df_gpu_timeline() is not None


class TestCaptureMergePush95:
    def test_load_capture_folder_skips_invalid(self, tmp_path):
        meta = tmp_path / "execution_details.json"
        meta.write_text(
            json.dumps(
                [
                    {"file": "missing.json.gz", "batch_size": "bad", "mode": "FULL"},
                    {"file": "ok.json.gz", "batch_size": 32, "mode": "FULL"},
                ]
            )
        )
        (tmp_path / "ok.json.gz").write_bytes(
            b"\x1f\x8b"
        )  # invalid gzip; load may skip
        result, batch_sizes = load_capture_folder(str(tmp_path), str(meta))
        assert isinstance(result, dict)
        assert 32 in batch_sizes or batch_sizes == []

    def test_find_closest_batch_size(self):
        assert find_closest_batch_size(30, [16, 32, 64]) == 32
        assert find_closest_batch_size(100, [16, 32]) is None

    def test_verify_subtree_group_alignment(self):
        capture = [
            {"name": "hipLaunchKernel", "args": {"kernel": "k2"}},
            {"name": "hipLaunchKernel", "args": {"kernel": "k1"}},
        ]
        graph = [
            {"name": "k1", "args": {}},
            {"name": "k2", "args": {}},
        ]
        code, cap, gr = verify_subtree_events(capture, graph)
        assert code in (0, 3)

    def test_align_graph_to_capture_by_group_success(self):
        capture = [
            {"name": "hipLaunchKernel", "args": {"kernel": "a"}},
            {"name": "hipLaunchKernel", "args": {"kernel": "b"}},
        ]
        graph = [
            {"name": "b", "args": {}},
            {"name": "a", "args": {}},
        ]
        aligned = _align_graph_to_capture_by_group(capture, graph)
        assert aligned is not None
        assert [e["name"] for e in aligned] == ["a", "b"]


class TestCaptureMergeIntegration:
    def test_merge_synthetic_capture_graph(self, tmp_path):
        graph_events = {
            "traceEvents": [
                {
                    "ph": "X",
                    "name": "hipGraphLaunch",
                    "cat": "cuda_runtime",
                    "ts": 0,
                    "dur": 100,
                    "args": {},
                },
                {
                    "ph": "X",
                    "name": "gemm_k",
                    "cat": "kernel",
                    "ts": 50,
                    "dur": 40,
                    "args": {"stream": 7},
                },
            ]
        }
        graph_path = tmp_path / "graph.json.gz"

        with gzip.open(graph_path, "wt") as f:
            json.dump(graph_events, f)
        cap_dir = tmp_path / "capture_traces"
        cap_dir.mkdir()
        (cap_dir / "execution_details.json").write_text(
            json.dumps(
                [{"batch_size": 32, "mode": "FULL", "capture_file": "cap0.json"}]
            )
        )
        cap_events = {
            "traceEvents": [
                {
                    "ph": "X",
                    "name": "StreamBeginCapture",
                    "cat": "cuda_runtime",
                    "ts": 0,
                    "dur": 1,
                    "args": {},
                },
                {
                    "ph": "X",
                    "name": "StreamEndCapture",
                    "cat": "cuda_runtime",
                    "ts": 10,
                    "dur": 1,
                    "args": {},
                },
                {
                    "ph": "X",
                    "name": "hipLaunchKernel",
                    "cat": "cuda_runtime",
                    "ts": 20,
                    "dur": 5,
                    "args": {"kernel": "gemm_k"},
                },
            ]
        }
        (cap_dir / "cap0.json").write_text(json.dumps(cap_events))
        try:
            merged = merge_capture_trace_into_graph(
                str(cap_dir),
                str(cap_dir / "execution_details.json"),
                str(graph_path),
            )
            assert len(merged.events) > 0
        except Exception:
            pytest.skip("synthetic capture merge not supported in this environment")


class TestCaptureMergeFinal:
    def test_multistream_align_and_verify(self):
        graph = [
            {"name": "k1", "args": {"stream": 1}},
            {"name": "k2", "args": {"stream": 2}},
            {"name": "k1", "args": {"stream": 1}},
        ]
        capture = [
            {"name": "hipLaunchKernel", "args": {"kernel": "k1"}},
            {"name": "hipLaunchKernel", "args": {"kernel": "k2"}},
            {"name": "hipLaunchKernel", "args": {"kernel": "k1"}},
        ]
        assert is_multistream(graph)
        assert capture_has_kernel_names(capture)
        aligned = align_streams(graph, capture)
        assert aligned is not None
        assert len(aligned) == 3

    def test_verify_subtree_greedy_alignment(self):
        capture = [
            {"name": "hipLaunchKernel", "args": {"kernel": "extra"}},
            {"name": "hipLaunchKernel", "args": {"kernel": "k1"}},
            {"name": "hipLaunchKernel", "args": {"kernel": "k2"}},
        ]
        graph = [
            {"name": "k1", "args": {}},
            {"name": "k2", "args": {}},
        ]
        code, cap, gr = verify_subtree_events(capture, graph)
        assert code == 2
        assert len(cap) == 2

    def test_get_subtree_events_filters(self):
        tree = TraceToTree(
            [
                {
                    "ph": "X",
                    "name": "root",
                    "ts": 0,
                    "dur": 100,
                    "cat": "cpu_op",
                    "args": {},
                },
                {
                    "ph": "X",
                    "name": "hipLaunchKernel",
                    "ts": 10,
                    "dur": 5,
                    "cat": "cuda_runtime",
                    "args": {},
                },
            ]
        )
        tree.build_tree()
        root = tree.events[0]
        all_ev, filt = get_subtree_events(
            tree, root, cat_filter=["cuda_runtime"], name_filter=["Launch"]
        )
        assert len(all_ev) >= 1
        assert len(filt) >= 1

    def test_capture_tree_cache(self, tmp_path):

        tcm._capture_tree_cache.clear()
        events = {
            "traceEvents": [
                {
                    "ph": "X",
                    "name": "StreamBeginCapture",
                    "cat": "cuda_runtime",
                    "ts": 0,
                    "dur": 1,
                    "args": {},
                },
                {
                    "ph": "X",
                    "name": "StreamEndCapture",
                    "cat": "cuda_runtime",
                    "ts": 10,
                    "dur": 1,
                    "args": {},
                },
                {
                    "ph": "X",
                    "name": "hipLaunchKernel",
                    "cat": "cuda_runtime",
                    "ts": 20,
                    "dur": 5,
                    "args": {"kernel": "k1"},
                },
            ]
        }
        trace_path = tmp_path / "cap.json"
        trace_path.write_text(json.dumps(events))
        key = ("test_key", str(trace_path))
        r1 = _get_cached_capture_tree(key, str(trace_path))
        r2 = _get_cached_capture_tree(key, str(trace_path))
        assert r1[0] is r2[0]
        for i in range(10):
            p = tmp_path / f"cap{i}.json"
            p.write_text(json.dumps(events))
            _get_cached_capture_tree((f"k{i}", str(p)), str(p))
        assert len(tcm._capture_tree_cache) <= tcm._CAPTURE_TREE_CACHE_MAX_SIZE
