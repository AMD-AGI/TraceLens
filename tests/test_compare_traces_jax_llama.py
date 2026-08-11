###############################################################################
# Copyright (c) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for TraceLens.Reporting.compare_traces_jax_llama."""

import gzip
import json

import pytest

from TraceLens.Reporting.compare_traces_jax_llama import (
    Event,
    Stats,
    Summary,
    classify_stage_base,
    compute_stage_table,
    emit_report,
    extract_block,
    extract_gpu_events,
    fmt_ms,
    fmt_us,
    get_path,
    infer_params,
    is_loop_multiply_fusion,
    mk_stats,
    parse_range,
    percentile,
    pid_map,
    summarize_one,
    token_start_times,
    top_stats_by_key,
)


def test_parse_range_valid():
    assert parse_range("0:3") == (0, 3)
    assert parse_range(" 1 : 8 ") == (1, 8)


def test_parse_range_invalid():
    with pytest.raises(ValueError, match="Bad range"):
        parse_range("bad")
    with pytest.raises(ValueError, match="end < start"):
        parse_range("5:2")


def test_percentile_and_mk_stats():
    assert percentile([], 50) == 0.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5
    assert percentile([1.0, 2.0, 3.0], 0) == 1.0
    assert percentile([1.0, 2.0, 3.0], 100) == 3.0

    empty = mk_stats([])
    assert empty == Stats(0, 0.0, 0.0, 0.0, 0.0)
    stats = mk_stats([10.0, 20.0, 30.0])
    assert stats.count == 3
    assert stats.total_us == 60.0
    assert stats.avg_us == 20.0


def test_fmt_helpers():
    assert fmt_us(1234.567) == "1,234.57"
    assert fmt_ms(5000.0) == "5.00"


def test_classify_stage_base_and_fusion():
    norm = Event(1, 1, 0, 10, "ln", {"name": "/Transformer/block_0/norm_attn/x"})
    assert classify_stage_base(norm) == "norm_attn"

    q_add = Event(1, 1, 0, 5, "k", {"name": "/Transformer/block_0/attn/q/add"})
    assert classify_stage_base(q_add) == "q_add"

    post_gsu = Event(1, 1, 0, 5, "PostGSU4_kernel", {})
    assert classify_stage_base(post_gsu) == "post_gsu"

    fusion = Event(
        1,
        1,
        0,
        5,
        "loop_multiply_fusion",
        {"hlo_op": "loop_multiply_fusion"},
    )
    assert is_loop_multiply_fusion(fusion)


def test_extract_block_and_infer_params():
    assert extract_block("/Transformer/block_12/norm_attn/") == 12
    assert extract_block("no block here") is None

    evs = [
        Event(
            1,
            1,
            0,
            10,
            "ln_fwd_tuned_kernel<Kernel_traits<float, 4096u,",
            {},
        ),
        Event(1, 1, 0, 5, "flash_fprop_hd128", {}),
        Event(1, 1, 0, 5, "PostGSU2_kernel", {}),
    ]
    d_model, head_dim, gsu = infer_params(evs)
    assert d_model == 4096
    assert head_dim == 128
    assert gsu == 2


def _make_synthetic_trace_events(pid=1, tid=10, hint="te_layernorm_forward"):
    events = [
        {
            "ph": "M",
            "name": "process_name",
            "pid": pid,
            "args": {"name": "/device:GPU:0"},
        },
    ]
    base_ts = 1000.0
    for token_idx in range(2):
        token_start = base_ts + token_idx * 5000.0
        for block in range(2):
            layer_start = token_start + block * 500.0
            path_prefix = f"/Transformer/block_{block}"
            stage_specs = [
                (f"{path_prefix}/norm_attn/{hint}", "ln_fwd_tuned_kernel", 20.0),
                (f"{path_prefix}/attn/q/add", "add_kernel", 5.0),
                (f"{path_prefix}/attn/q/dot_general", "gemm_kernel", 30.0),
                (f"{path_prefix}/attn/out/dot_general", "out_gemm", 25.0),
                (f"{path_prefix}/norm_mlp/te_layernorm", "ln_mlp", 15.0),
                (f"{path_prefix}/mlp/in/dot_general", "mlp_in", 40.0),
                (f"{path_prefix}/mlp/out/dot_general", "mlp_out", 35.0),
            ]
            ts = layer_start
            for path, name, dur in stage_specs:
                events.append(
                    {
                        "ph": "X",
                        "pid": pid,
                        "tid": tid,
                        "ts": ts,
                        "dur": dur,
                        "name": name,
                        "args": {"name": path},
                    }
                )
                ts += dur
            events.append(
                {
                    "ph": "X",
                    "pid": pid,
                    "tid": tid,
                    "ts": ts,
                    "dur": 10.0,
                    "name": "loop_multiply_fusion",
                    "args": {
                        "name": "/Transformer/swiglu",
                        "hlo_op": "loop_multiply_fusion",
                    },
                }
            )
    return events


def _write_gz_trace(tmp_path, hint="te_layernorm_forward"):
    path = tmp_path / "trace.json.gz"
    payload = {"traceEvents": _make_synthetic_trace_events(hint=hint)}
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(payload, f)
    return path


def test_pid_map_and_extract_gpu_events(tmp_path):
    path = _write_gz_trace(tmp_path)
    with gzip.open(path, "rt", encoding="utf-8") as f:
        trace = json.load(f)
    mp = pid_map(trace)
    assert "/device:GPU:0" in mp.values()
    events = extract_gpu_events(trace, gpu_index=0)
    assert len(events) > 0
    assert all(isinstance(e, Event) for e in events)


def test_token_start_times_and_compute_stage_table(tmp_path):
    path = _write_gz_trace(tmp_path)
    with gzip.open(path, "rt", encoding="utf-8") as f:
        trace = json.load(f)
    gpu_events = extract_gpu_events(trace, gpu_index=0)
    stream = sorted(gpu_events, key=lambda e: e.ts)
    starts = token_start_times(stream, block0_norm_hint="te_layernorm_forward")
    assert len(starts) == 2

    stage_avg, stage_share, per_layer, per_token, notes = compute_stage_table(
        stream=stream,
        token_starts=starts,
        token_range=(0, 1),
        layer_range=(0, 1),
    )
    assert per_layer > 0
    assert "norm_attn" in stage_avg
    assert abs(sum(stage_share.values()) - 1.0) < 1e-6
    assert isinstance(notes, list)


def test_top_stats_by_key():
    events = [
        Event(1, 1, 0, 10, "kernel_a", {}),
        Event(1, 1, 0, 20, "kernel_a", {}),
        Event(1, 1, 0, 5, "kernel_b", {}),
    ]
    top = top_stats_by_key(events, key_fn=lambda e: e.name, top_n=2)
    assert top[0][0] == "kernel_a"
    assert top[0][1].total_us == 30.0


def test_emit_report_from_summaries():
    stage_avg = {
        s: float(i)
        for i, s in enumerate(
            [
                "norm_attn",
                "q_add",
                "q_gemm",
                "k_add",
                "k_gemm",
                "v_add",
                "v_gemm",
                "attn_core",
                "out_gemm",
                "norm_mlp",
                "mlp_in_gemm",
                "swiglu_elementwise",
                "mlp_out_gemm",
                "post_gsu",
                "other",
            ]
        )
    }
    stage_share = {k: v / sum(stage_avg.values()) for k, v in stage_avg.items()}
    base = Summary(
        label="ROCm",
        d_model=4096,
        head_dim=128,
        gsu=2,
        gpu_index=0,
        main_tid=10,
        token_range=(0, 1),
        layer_range=(0, 1),
        per_layer_us=100.0,
        per_token_us=200.0,
        stage_avg_us=stage_avg,
        stage_share=stage_share,
        top_kernels=[("k1", mk_stats([10.0]))],
        top_ops=[("/path/op", mk_stats([5.0]))],
        notes=["note"],
    )
    cuda = Summary(
        label="CUDA",
        d_model=4096,
        head_dim=128,
        gsu=0,
        gpu_index=0,
        main_tid=10,
        token_range=(0, 1),
        layer_range=(0, 1),
        per_layer_us=80.0,
        per_token_us=160.0,
        stage_avg_us={k: v * 0.8 for k, v in stage_avg.items()},
        stage_share=stage_share,
        top_kernels=[("k2", mk_stats([8.0]))],
        top_ops=[("/path/op2", mk_stats([4.0]))],
        notes=[],
    )
    report = emit_report(base, cuda)
    assert "Trace Comparison" in report
    assert "norm_attn" in report
    assert "ROCm" in report and "CUDA" in report


def test_summarize_one_integration(tmp_path):
    rocm_path = _write_gz_trace(tmp_path, hint="te_layernorm_forward")
    cuda_dir = tmp_path / "cuda"
    cuda_dir.mkdir()
    cuda_path = _write_gz_trace(cuda_dir, hint="te_norm_forward_ffi")

    rocm = summarize_one(
        "ROCm",
        str(rocm_path),
        gpu_index=0,
        tokens=(0, 1),
        layers=(0, 1),
        block0_norm_hint="te_layernorm_forward",
        top_kernels_n=3,
        top_ops_n=3,
    )
    cuda = summarize_one(
        "CUDA",
        str(cuda_path),
        gpu_index=0,
        tokens=(0, 1),
        layers=(0, 1),
        block0_norm_hint="te_norm_forward_ffi",
        top_kernels_n=3,
        top_ops_n=3,
    )
    assert rocm.per_layer_us > 0
    assert cuda.per_layer_us > 0
    assert len(rocm.top_kernels) > 0
    assert len(cuda.top_ops) > 0


def test_get_path():
    ev = Event(1, 1, 0, 1, "k", {"name": "/some/path"})
    assert get_path(ev) == "/some/path"
    assert get_path(Event(1, 1, 0, 1, "k", {})) == ""
