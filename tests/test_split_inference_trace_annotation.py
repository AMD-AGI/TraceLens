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
- Annotation-name parsing (``get_iter_details_from_name``).
- Steady-state window finding (``identify_steady_state_regions`` and
  ``find_steady_state_window``).

No trace files are written; everything operates on in-memory dicts.
"""

from TraceLens.TraceUtils import split_inference_trace_annotation as split

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

    details = split.get_iter_details_from_name(names[0])
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
    primary = split.ANNOTATION_PATTERN[0]
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

    decode = split.get_iter_details_from_name("step[DECODE bs=20]")
    assert decode["generation_requests"] == 20
    assert decode["context_requests"] == 0

    extend = split.get_iter_details_from_name("step[EXTEND bs=2 toks=800]")
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

    details = split.get_iter_details_from_name(
        "execute_context_3(100)_generation_2(50)"
    )
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
