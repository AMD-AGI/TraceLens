###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import copy
import gzip
import json
import os

import pytest

from TraceLens.PerfModel import perf_model
from TraceLens.PerfModel.torch_op_mapping import resolve_perf_model_class
from TraceLens.Trace2Tree.trace_sglang_capture_link import (
    DEFAULT_CAPTURE_OFFSET,
    _augment_capture_args,
    build_graph_replay_kernel_index,
    detect_launch_offset,
    enrich_synthetic_ops_from_sglang_capture,
    find_decode_batch_size_for_graph_launch,
    index_decode_annotations,
    is_usable_capture_input_dims,
    load_sglang_capture_batch_sizes,
)
from TraceLens.Trace2Tree.trace_to_tree import TraceToTree
from TraceLens.TreePerf.tree_perf import TreePerfAnalyzer

GEMM_KERNEL = "_ZN2ck59kernel_gemm_xdl_cshuffle_v3_multi_d_blockscale_b_preshuffleINS_58GridwiseGemm"
PREAMBLE_KERNELS = ["preamble_0", "preamble_1"]


def _mk_event(cat, name, ts, dur=10, pid=1, tid=1, args=None, uid=None):
    event = {
        "ph": "X",
        "cat": cat,
        "name": name,
        "pid": pid,
        "tid": tid,
        "ts": ts,
        "dur": dur,
        "args": args or {},
    }
    if uid is not None:
        event["id"] = uid
    return event


def _mk_ac2g(corr, pid=0, tid=7, ts=1000, phase="s"):
    return {
        "ph": phase,
        "cat": "ac2g",
        "name": "ac2g",
        "pid": pid,
        "tid": tid,
        "ts": ts,
        "id": corr,
    }


@pytest.fixture
def capture_folder(tmp_path):
    """Minimal capture trace with two preamble launches then a GEMM launch."""
    folder = tmp_path / "capture_traces"
    folder.mkdir()

    capture_events = []
    ts = 1000
    for idx, kernel in enumerate(PREAMBLE_KERNELS):
        capture_events.append(
            _mk_event(
                "cuda_runtime",
                "hipLaunchKernel",
                ts,
                args={"kernel": kernel, "correlation": 100 + idx},
            )
        )
        ts += 20

    gemm_corr = 200
    capture_events.append(
        _mk_event(
            "cuda_runtime",
            "hipLaunchKernel",
            ts,
            args={"kernel": GEMM_KERNEL, "correlation": gemm_corr},
        )
    )
    capture_events.append(
        _mk_event(
            "cpu_op",
            "aiter::gemm_a8w8_blockscale_bpreshuffle_",
            ts - 5,
            args={
                "correlation": gemm_corr,
                "Input Dims": [[16, 3072], [2048, 3072], [16, 24]],
                "Input type": ["c10::BFloat16", "c10::BFloat16"],
                "Input Strides": [[3072, 1], [1, 3072]],
            },
        )
    )

    payload = {"traceEvents": capture_events}
    with gzip.open(folder / "bs_16_rank0.json.gz", "wt") as handle:
        json.dump(payload, handle)
    return str(folder)


def _build_replay_trace():
    corr = 850
    events = [
        _mk_event("user_annotation", "step[DECODE bs=16]", 900, dur=50),
        _mk_event(
            "cuda_runtime",
            "hipGraphLaunch",
            950,
            args={"correlation": corr},
        ),
    ]
    kernel_ts = 1000
    for kernel in [GEMM_KERNEL]:
        events.append(
            _mk_event(
                "kernel",
                kernel,
                kernel_ts,
                pid=0,
                tid=7,
                args={"correlation": corr, "stream": 7},
            )
        )
        kernel_ts += 30
    events.extend(
        [
            _mk_ac2g(corr, ts=1000, phase="s"),
            _mk_ac2g(corr, ts=1000, phase="f"),
        ]
    )
    return events


def test_detect_launch_offset_finds_plus_two():
    capture = PREAMBLE_KERNELS + [GEMM_KERNEL, "tail"]
    replay = [GEMM_KERNEL, "tail"]
    assert detect_launch_offset(capture, replay) == DEFAULT_CAPTURE_OFFSET


def test_find_decode_batch_size_for_graph_launch():
    timestamps, batch_sizes = index_decode_annotations(
        [_mk_event("user_annotation", "step[DECODE bs=16]", 900)]
    )
    assert find_decode_batch_size_for_graph_launch(950, timestamps, batch_sizes) == 16
    assert (
        find_decode_batch_size_for_graph_launch(1_000_100, timestamps, batch_sizes)
        is None
    )


def test_load_sglang_capture_batch_sizes(capture_folder):
    assert load_sglang_capture_batch_sizes(capture_folder) == [16]


def test_enrich_synthetic_ops_from_capture(capture_folder):
    tree = TraceToTree(_build_replay_trace())
    tree.build_tree(add_python_func=False)

    decode_ts, decode_bs = index_decode_annotations(tree.events)
    kernel_index = build_graph_replay_kernel_index(tree, decode_ts, decode_bs)
    kernel_uid = next(iter(kernel_index.keys()))

    synthetic = {
        "name": f"hipGraphLaunch->{GEMM_KERNEL} (Synthetic Op)",
        "args": {},
        "gpu_events": [kernel_uid],
    }

    enriched = enrich_synthetic_ops_from_sglang_capture(
        [synthetic], tree, capture_folder
    )
    assert enriched == 1
    assert synthetic["args"]["Input Dims"] == [[16, 3072], [2048, 3072], [16, 24]]
    assert resolve_perf_model_class(synthetic) is perf_model.aten_mm


def test_collect_unified_perf_events_enriches_graph_synthetics(capture_folder):
    tree = TraceToTree(_build_replay_trace())
    analyzer = TreePerfAnalyzer(tree, capture_folder=capture_folder)
    events = analyzer.collect_unified_perf_events()

    synthetics = [
        event
        for event in events
        if " (Synthetic Op)" in event.get("name", "")
        and "hipGraphLaunch" in event.get("name", "")
    ]
    assert len(synthetics) == 1
    assert synthetics[0]["args"].get("Input Dims") == [
        [16, 3072],
        [2048, 3072],
        [16, 24],
    ]


def test_is_usable_capture_input_dims():
    assert is_usable_capture_input_dims([[16, 3072], [2048, 3072]])
    assert not is_usable_capture_input_dims([[4], [4], [4], [], [], [], [], [], []])
    assert not is_usable_capture_input_dims([[], []])
    assert not is_usable_capture_input_dims([])


def test_enrich_overwrites_placeholder_and_isolates_sibling_args(capture_folder):
    """Each synthetic gets its own args; placeholder dims must not block enrichment."""
    tree = TraceToTree(_build_replay_trace())
    tree.build_tree(add_python_func=False)

    decode_ts, decode_bs = index_decode_annotations(tree.events)
    kernel_index = build_graph_replay_kernel_index(tree, decode_ts, decode_bs)
    kernel_uid = next(iter(kernel_index.keys()))

    placeholder = [[4], [4], [4], [], [], [], [], [], []]
    shared_args = {"Input Dims": placeholder, "cid": 1}
    synthetics = [
        {
            "name": f"hipGraphLaunch->{GEMM_KERNEL} (Synthetic Op)",
            "args": shared_args,
            "gpu_events": [kernel_uid],
        },
        {
            "name": f"hipGraphLaunch->other_kernel (Synthetic Op)",
            "args": shared_args,
            "gpu_events": [999999],
        },
    ]

    enriched = enrich_synthetic_ops_from_sglang_capture(
        synthetics, tree, capture_folder
    )
    assert enriched == 1
    assert synthetics[0]["args"]["Input Dims"] == [
        [16, 3072],
        [2048, 3072],
        [16, 24],
    ]
    assert synthetics[1]["args"]["Input Dims"] is placeholder
    assert synthetics[0]["args"] is not synthetics[1]["args"]


MOE_KERNEL = "_ZN2ck15kernel_moe_gemmINS_25GridwiseMoeGemmBlockScale"
QUANT_KERNEL = "_ZN5aiter37dynamic_per_group_scaled_quant_kernelIDhDB8_Li32EEEvPT0_PfPKT_PKfiliibPKii"


def test_augment_truncated_moe_gemm_from_nearby_launch():
    table = [
        {"kernel": "quant", "args": {"Input Dims": [[4, 3072], [], []]}},
        {
            "kernel": MOE_KERNEL,
            "args": {"Input Dims": [[256, 768, 3072], [], [], []]},
        },
        {
            "kernel": MOE_KERNEL,
            "args": {
                "Input Dims": [
                    [4, 8, 384],
                    [256, 768, 3072],
                    [256, 3072, 384],
                ],
                "Input type": ["c10::Float8_e4m3fn"] * 3,
            },
        },
    ]
    augmented = _augment_capture_args(
        MOE_KERNEL, table[1]["args"], table, capture_index=1
    )
    assert is_usable_capture_input_dims(augmented["Input Dims"])
    assert augmented["Input Dims"][0] == [4, 3072]
    assert augmented["Input Dims"][1] == [256, 768, 3072]


def test_augment_rmsnorm_serial_from_nearby_2d_shape():
    table = [
        {"kernel": QUANT_KERNEL, "args": {"Input Dims": [[4, 3072], [], [4, 24]]}},
        {"kernel": "rmsnorm_sumsq_kernel_serial", "args": {"Input Dims": [[], [], []]}},
    ]
    augmented = _augment_capture_args(
        "rmsnorm_sumsq_kernel_serial", table[1]["args"], table, capture_index=1
    )
    assert augmented["Input Dims"][0] == [4, 3072]


def test_augment_quant_kernel_resolves_perf_model():
    from TraceLens.PerfModel.extensions import perf_model_extensions

    event = {
        "name": f"hipGraphLaunch->{QUANT_KERNEL} (Synthetic Op)",
        "args": {
            "Input Dims": [[4, 3072], [96, 128], [4, 24]],
            "Input type": [
                "c10::Float8_e4m3fn",
                "c10::Half",
                "float",
            ],
        },
        "kernel_details": [{"name": QUANT_KERNEL}],
    }
    assert (
        resolve_perf_model_class(event)
        is perf_model_extensions.dynamic_per_group_scaled_quant_kernel
    )
    pm = perf_model_extensions.dynamic_per_group_scaled_quant_kernel(event=event)
    assert pm.flops() > 0
    assert pm.bytes() > 0
