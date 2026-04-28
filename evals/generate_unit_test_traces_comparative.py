# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""
Generate synthetic PyTorch profiler traces for comparative AgenticMode unit tests.

Each test produces trace1.json and trace2.json in its own subdirectory.
Traces are structurally valid PyTorch/ROCm profiler JSON files with realistic
kernel names and timings designed to exercise specific comparative analysis
scenarios.

Usage:
    python generate_traces.py
"""

import json
import os

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASE_TS = 2_200_000_000_000.0  # microseconds, matching real traces
PID = 12345
GPU_PID = 2
GPU_TID = 0
CPU_TID = 12345

DEVICE_PROPERTIES = [
    {
        "id": 0,
        "name": "AMD Instinct MI300X VF",
        "totalGlobalMem": 205_571_227_648,
        "computeMajor": 9,
        "computeMinor": 4,
        "maxThreadsPerBlock": 1024,
        "maxThreadsPerMultiprocessor": 2048,
        "regsPerBlock": 65536,
        "warpSize": 64,
        "sharedMemPerBlock": 65536,
        "numSms": 304,
        "maxSharedMemoryPerMultiProcessor": 65536,
    }
]

BASE_METADATA = {
    "num_errors": 0,
    "errors": {},
    "profiler": {
        "repeats": 3,
        "warmups": 2,
        "wait": 0,
        "grad_mode": "no_grad",
        "synchronize_each_step": True,
        "torch_profiler_kwargs": {
            "record_shapes": True,
            "profile_memory": False,
            "with_stack": False,
            "with_flops": True,
            "with_modules": False,
            "activities": ["CPU", "CUDA"],
        },
    },
    "timestamp": 1_700_000_000.0,
    "base_image": "synthetic",
    "started_by": "synthetic_generator",
    "experiment": {
        "kind": "synthetic",
        "repo_id": "synthetic",
        "pipeline_tag": "text-generation",
        "device": "auto",
        "torch_dtype": "bfloat16",
        "cache_dir": None,
        "pretrained": True,
        "trust_remote_code": False,
        "use_deepspeed": False,
        "custom_automodel_kwargs": {},
        "custom_autoconfig_kwargs": {},
    },
    "current_task": {
        "shape": {"batch_size": 1, "text_input_length": 512, "tag": "text"}
    },
    "system": {
        "pid": PID,
        "tid": CPU_TID,
        "uname": {
            "system": "Linux",
            "version": "#1 SMP",
            "processor": "x86_64",
            "node": "synthetic-node",
            "release": "5.15.0",
            "machine": "x86_64",
        },
        "architecture": {"bits": "64bit", "linkage": "ELF"},
        "cpu": {},
        "gpu": {"gpu_00": "AMD Instinct MI300X VF"},
        "libc_ver": ["glibc", "2.35"],
        "os_release": {
            "NAME": "Ubuntu",
            "ID": "ubuntu",
            "PRETTY_NAME": "Ubuntu 22.04.3 LTS",
            "VERSION_ID": "22.04",
        },
        "python": {
            "major": 3,
            "minor": 10,
            "patchlevel": 12,
            "implementation": "CPython",
        },
        "pip_packages": ["torch==2.3.0"],
    },
}


def make_trace(events, metadata_override=None):
    """Build a complete PyTorch profiler trace dict."""
    meta = dict(BASE_METADATA)
    if metadata_override:
        meta.update(metadata_override)
    return {
        "metadata": meta,
        "traceEvents": events,
        "schemaVersion": 1,
        "deviceProperties": DEVICE_PROPERTIES,
        "roctracer_version": "4.1.0",
        "with_flops": True,
        "with_modules": False,
        "record_shapes": True,
        "hip_runtime_version": "60131199",
        "profile_memory": False,
        "hip_driver_version": "60131199",
        "with_stack": False,
        "trace_id": "synthetic",
        "traceName": "synthetic_trace",
        "displayTimeUnit": "ms",
        "baseTimeNanoseconds": int(BASE_TS * 1000),
    }


def write_trace(path, trace):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(trace, f, indent=2)
    print(f"  Wrote {path}")


# ---------------------------------------------------------------------------
# Event builder utilities
# ---------------------------------------------------------------------------

_ev_idx = 0
_correlation = 1000


def next_eid():
    global _ev_idx
    _ev_idx += 1
    return _ev_idx


def next_corr():
    global _correlation
    _correlation += 1
    return _correlation


def reset_ids():
    global _ev_idx, _correlation
    _ev_idx = 0
    _correlation = 1000


def cpu_op(name, ts, dur, input_dims=None, input_type=None, flops=None):
    eid = next_eid()
    args = {
        "External id": eid,
        "Record function id": 0,
        "finished": True,
        "Ev Idx": eid,
    }
    if input_dims is not None:
        args["Input Dims"] = input_dims
    if input_type is not None:
        args["Input type"] = input_type
    if flops is not None:
        args["flops"] = flops
    return {
        "ph": "X",
        "cat": "cpu_op",
        "name": name,
        "pid": PID,
        "tid": CPU_TID,
        "ts": ts,
        "dur": dur,
        "args": args,
    }


def gpu_kernel(name, ts, dur, external_id, corr, grid=None, block=None):
    args = {
        "External id": external_id,
        "device": 0,
        "stream": 0,
        "correlation": corr,
        "kind": "Dispatch Kernel",
        "grid": grid or [128, 1, 1],
        "block": block or [256, 1, 1],
    }
    return {
        "ph": "X",
        "cat": "kernel",
        "name": name,
        "pid": GPU_PID,
        "tid": GPU_TID,
        "ts": ts,
        "dur": dur,
        "args": args,
    }


def hip_launch(name, ts, dur, external_id, corr, grid=None, block=None):
    return {
        "ph": "X",
        "cat": "cuda_runtime",
        "name": "hipLaunchKernel",
        "pid": PID,
        "tid": CPU_TID,
        "ts": ts,
        "dur": 2.0,
        "args": {
            "External id": external_id,
            "kernel": name,
            "cid": corr,
            "correlation": corr,
            "grid": grid or [128, 1, 1],
            "block": block or [256, 1, 1],
            "shared memory": 0,
        },
    }


def ac2g_pair(ts_cpu, ts_gpu, corr):
    # "s" = CPU side (dispatch), "f" = GPU side (kernel start)
    # This matches real ROCm traces: s has CPU pid/tid, f has GPU pid/tid
    return [
        {"ph": "s", "id": corr, "pid": PID, "tid": CPU_TID, "ts": ts_cpu, "cat": "ac2g", "name": "ac2g"},
        {"ph": "f", "id": corr, "pid": GPU_PID, "tid": GPU_TID, "ts": ts_gpu, "cat": "ac2g", "name": "ac2g", "bp": "e"},
    ]


def profiler_step(ts, dur):
    return {
        "ph": "X",
        "cat": "user_annotation",
        "name": "ProfilerStep#1",
        "pid": PID,
        "tid": CPU_TID,
        "ts": ts,
        "dur": dur,
        "args": {"External id": 0, "Record function id": 0, "finished": True, "Ev Idx": 0},
    }


def build_op_with_kernel(
    cpu_name,
    kernel_name,
    cpu_ts,
    cpu_dur,
    gpu_ts,
    gpu_dur,
    input_dims=None,
    input_type=None,
    flops=None,
    grid=None,
    block=None,
):
    """Build one CPU op + one GPU kernel + linkage events."""
    eid = next_eid()
    corr = next_corr()
    events = []

    cpu_ev = {
        "ph": "X",
        "cat": "cpu_op",
        "name": cpu_name,
        "pid": PID,
        "tid": CPU_TID,
        "ts": cpu_ts,
        "dur": cpu_dur,
        "args": {
            "External id": eid,
            "Record function id": 0,
            "finished": True,
            "Ev Idx": eid,
            **({"Input Dims": input_dims} if input_dims else {}),
            **({"Input type": input_type} if input_type else {}),
            **({"flops": flops} if flops is not None else {}),
        },
    }
    events.append(cpu_ev)
    events.append(hip_launch(kernel_name, cpu_ts + cpu_dur - 3, 2.0, eid, corr, grid, block))
    events += ac2g_pair(cpu_ts + cpu_dur - 3, gpu_ts, corr)
    events.append(gpu_kernel(kernel_name, gpu_ts, gpu_dur, eid, corr, grid, block))
    return events, eid, corr


# ---------------------------------------------------------------------------
# Test 1: MoE — fused (trace1) vs unfused (trace2)
#
# Trace1: single fused MoE kernel (pseudo_op::moe_aiter_fused_1stage)
# Trace2: four separate kernels (routing, stage1 gemm, stage2 gemm, sum)
#
# Expected: Trace2 flagged as fusion opportunity (not compute inefficiency)
# ---------------------------------------------------------------------------

def make_test01_moe_fused():
    reset_ids()
    # Trace1: one fused MoE CPU op → one GPU kernel, fast
    ts = BASE_TS
    step_dur = 2000.0

    # Trace1: four separate ops — routing, gemm1, gemm2, sum — same logical work, slower total
    events1 = [profiler_step(ts, step_dur)]

    for cpu_name, kname, gpu_dur, flops in [
        (
            "aiter::moe_sorting_fwd",
            "moe_sorting_kernel<BF16>",
            120.0,
            0,
        ),
        (
            "aiter::ck_moe_stage1",
            "ck_moe_stage1_kernel<BF16, 128, 128>",
            480.0,
            int(4096 * 8192 * 22016 * 2 * 8 // 2),
        ),
        (
            "aiter::ck_moe_stage2",
            "ck_moe_stage2_kernel<BF16, 128, 128>",
            480.0,
            int(4096 * 8192 * 22016 * 2 * 8 // 2),
        ),
        (
            "aiter::moe_sum",
            "moe_sum_kernel<BF16>",
            80.0,
            0,
        ),
    ]:
        offset = len([e for e in events1 if e.get("cat") == "kernel"]) * 200 + 100
        evs, _, _ = build_op_with_kernel(
            cpu_name=cpu_name,
            kernel_name=kname,
            cpu_ts=ts + offset,
            cpu_dur=20.0,
            gpu_ts=ts + offset + 30,
            gpu_dur=gpu_dur,
            input_dims=[[4096, 8192], [8, 8192, 22016]],
            input_type=["bfloat16", "bfloat16"],
            flops=flops,
            grid=[128, 8, 1],
            block=[256, 1, 1],
        )
        events1 += evs
    trace1 = make_trace(events1, {"experiment": {**BASE_METADATA["experiment"], "repo_id": "test01_moe_fused_trace1"}})

    # Trace2: one fused MoE CPU op → one GPU kernel, fast
    reset_ids()
    events2 = [profiler_step(ts, step_dur)]
    evs, _, _ = build_op_with_kernel(
        cpu_name="pseudo_op::moe_aiter_fused_1stage",
        kernel_name="fmoe_fp8_blockscale_g1u1_kernel<BF16, BF16, float, FP8, 2, 128, 128, 128, 2, 2, true>",
        cpu_ts=ts + 10,
        cpu_dur=50.0,
        gpu_ts=ts + 80,
        gpu_dur=800.0,  # fast fused kernel
        input_dims=[[4096, 8192], [8, 8192, 22016], []],
        input_type=["bfloat16", "float8_e4m3fnuz", "int"],
        flops=int(4096 * 8192 * 22016 * 2 * 8),
        grid=[128, 8, 1],
        block=[256, 1, 1],
    )
    events2 += evs
    trace2 = make_trace(events2, {"experiment": {**BASE_METADATA["experiment"], "repo_id": "test01_moe_fused_trace2"}})
    return trace1, trace2


# ---------------------------------------------------------------------------
# Test 2: Attention — both unfused, Trace2 faster
#
# Both traces: QK matmul (aten::bmm) + aten::softmax + AV matmul (aten::bmm)
# Trace1 kernels are slower; Trace2 kernels are faster.
#
# Expected: compute kernel optimization for Trace1 (NOT a fusion opportunity)
# ---------------------------------------------------------------------------

ATTN_DIMS = [[8, 512, 64], [8, 64, 512]]  # (B, S, H) for bmm
ATTN_TYPE = ["bfloat16", "bfloat16"]
QK_FLOPS = int(8 * 512 * 512 * 64 * 2)  # B * S * S * H * 2


def make_test02_attention_both_unfused():
    reset_ids()
    ts = BASE_TS
    step_dur = 5000.0

    # Trace1: unfused, slow kernels
    events1 = [profiler_step(ts, step_dur)]
    cpu_offset = 10.0
    gpu_t = ts + 100.0
    for cpu_name, kname, gpu_dur, dims, flops in [
        ("aten::bmm", "volta_h16gemm_<BF16,BF16,float>_tt_v1", cpu_offset, ATTN_DIMS, QK_FLOPS),
        ("aten::softmax", "softmax_warp_forward<float, float, float, 4, true>", cpu_offset + 80, [[8, 512, 512]], QK_FLOPS // 10),
        ("aten::bmm", "volta_h16gemm_<BF16,BF16,float>_nt_v1", cpu_offset + 160, ATTN_DIMS, QK_FLOPS),
    ]:
        evs, _, _ = build_op_with_kernel(
            cpu_name=cpu_name,
            kernel_name=kname,
            cpu_ts=ts + cpu_offset,
            cpu_dur=15.0,
            gpu_ts=gpu_t,
            gpu_dur=gpu_dur,  # slow
            input_dims=dims,
            input_type=ATTN_TYPE,
            flops=flops,
            grid=[512, 8, 1],
            block=[128, 1, 1],
        )
        events1 += evs
        gpu_t += gpu_dur + 5
        cpu_offset += 80
    trace1 = make_trace(events1, {"experiment": {**BASE_METADATA["experiment"], "repo_id": "test02_attn_unfused_trace1"}})

    # Trace2: unfused, fast kernels (same structure, shorter gpu_dur)
    reset_ids()
    events2 = [profiler_step(ts, step_dur)]
    cpu_offset = 10.0
    gpu_t = ts + 100.0
    for cpu_name, kname, gpu_dur, dims, flops in [
        ("aten::bmm", "Cijk_Ailk_Bljk_SB<BF16,BF16,float>", cpu_offset, ATTN_DIMS, QK_FLOPS),
        ("aten::softmax", "softmax_warp_forward<float, float, float, 4, true>", cpu_offset + 30, [[8, 512, 512]], QK_FLOPS // 10),
        ("aten::bmm", "Cijk_Ailk_Bljk_SB<BF16,BF16,float>", cpu_offset + 60, ATTN_DIMS, QK_FLOPS),
    ]:
        evs, _, _ = build_op_with_kernel(
            cpu_name=cpu_name,
            kernel_name=kname,
            cpu_ts=ts + cpu_offset,
            cpu_dur=15.0,
            gpu_ts=gpu_t,
            gpu_dur=gpu_dur,  # fast
            input_dims=dims,
            input_type=ATTN_TYPE,
            flops=flops,
            grid=[512, 8, 1],
            block=[128, 1, 1],
        )
        events2 += evs
        gpu_t += gpu_dur + 5
        cpu_offset += 30
    trace2 = make_trace(events2, {"experiment": {**BASE_METADATA["experiment"], "repo_id": "test02_attn_unfused_trace2"}})
    return trace1, trace2


# ---------------------------------------------------------------------------
# Test 3: Attention — Trace1 fused flash, Trace2 unfused
#
# Trace1: aiter::_flash_attn_forward (fast, fused)
# Trace2: aten::bmm + aten::softmax + aten::bmm (slow, unfused)
#
# Expected: Trace2 flagged as fusion opportunity; Trace1 has no issues
# ---------------------------------------------------------------------------

FLASH_DIMS = [[8, 512, 16, 64], [8, 512, 16, 64], [8, 512, 16, 64]]
FLASH_TYPE = ["bfloat16", "bfloat16", "bfloat16"]
FLASH_FLOPS = int(2 * 8 * 16 * 512 * 512 * 64)


def make_test03_fused_vs_unfused():
    reset_ids()
    ts = BASE_TS
    step_dur = 4000.0

    # Trace1: flash attention (fused, fast)
    events1 = [profiler_step(ts, step_dur)]
    evs, _, _ = build_op_with_kernel(
        cpu_name="aiter::_flash_attn_forward",
        kernel_name="flash_fwd_hdim64_bf16_sm80<Flash_fwd_kernel_traits<64, 128, 128, 4, false, false, cutlass::bfloat16_t>>",
        cpu_ts=ts + 10,
        cpu_dur=30.0,
        gpu_ts=ts + 60,
        gpu_dur=350.0,  # fast fused
        input_dims=FLASH_DIMS,
        input_type=FLASH_TYPE,
        flops=FLASH_FLOPS,
        grid=[512, 16, 8],
        block=[128, 1, 1],
    )
    events1 += evs
    trace1 = make_trace(events1, {"experiment": {**BASE_METADATA["experiment"], "repo_id": "test03_fused_trace1"}})

    # Trace2: unfused attention (slow)
    reset_ids()
    events2 = [profiler_step(ts, step_dur)]
    cpu_offset = 10.0
    gpu_t = ts + 60.0
    for cpu_name, kname, gpu_dur, dims, flops in [
        ("aten::bmm", "volta_h16gemm_<BF16,BF16,float>_tt_v1", cpu_offset, ATTN_DIMS, FLASH_FLOPS // 2),
        ("aten::softmax", "softmax_warp_forward<float, float, float, 4, true>", cpu_offset + 250, [[8, 16, 512, 512]], FLASH_FLOPS // 20),
        ("aten::bmm", "volta_h16gemm_<BF16,BF16,float>_nt_v1", cpu_offset + 500, ATTN_DIMS, FLASH_FLOPS // 2),
    ]:
        evs, _, _ = build_op_with_kernel(
            cpu_name=cpu_name,
            kernel_name=kname,
            cpu_ts=ts + cpu_offset,
            cpu_dur=15.0,
            gpu_ts=gpu_t,
            gpu_dur=gpu_dur,
            input_dims=dims,
            input_type=["bfloat16", "bfloat16"],
            flops=flops,
            grid=[512, 16, 1],
            block=[128, 1, 1],
        )
        events2 += evs
        gpu_t += gpu_dur + 5
        cpu_offset += 250
    trace2 = make_trace(events2, {"experiment": {**BASE_METADATA["experiment"], "repo_id": "test03_fused_trace2"}})
    return trace1, trace2


# ---------------------------------------------------------------------------
# Test 4: GEMM — aten::mm + transpose kernel (trace1) vs aten::addmm (trace2)
#
# Both compute M=1024, K=2048, N=4096 matmul.
# Trace1: aten::mm dispatches (1) a batched_transpose kernel + (2) GEMM kernel
# Trace2: aten::addmm dispatches (1) single GEMM kernel
#
# Expected: functionally equivalent — NOT a fusion opportunity
# ---------------------------------------------------------------------------

MM_FLOPS = int(1024 * 2048 * 4096 * 2)
MM_DIMS = [[1024, 2048], [2048, 4096]]
MM_TYPE = ["bfloat16", "bfloat16"]


def make_test04_gemm_mm_vs_linear():
    reset_ids()
    ts = BASE_TS
    step_dur = 3000.0

    # Trace1: aten::mm → transpose kernel + GEMM kernel
    events1 = [profiler_step(ts, step_dur)]

    eid1 = next_eid()
    corr1a = next_corr()
    corr1b = next_corr()

    cpu_ev1 = {
        "ph": "X", "cat": "cpu_op", "name": "aten::mm",
        "pid": PID, "tid": CPU_TID,
        "ts": ts + 10, "dur": 60.0,
        "args": {
            "External id": eid1, "Record function id": 0, "finished": True, "Ev Idx": eid1,
            "Input Dims": MM_DIMS, "Input type": MM_TYPE, "flops": MM_FLOPS,
        },
    }
    # First: transpose kernel
    transpose_kname = "void at::native::vectorized_elementwise_kernel<4, at::native::TensorIteratorCastOp<c10::BFloat16, c10::BFloat16, true>>(int, at::native::TensorIteratorCastOp<c10::BFloat16, c10::BFloat16, true>)"
    hip_t1a = {"ph": "X", "cat": "cuda_runtime", "name": "hipLaunchKernel",
               "pid": PID, "tid": CPU_TID, "ts": ts + 20, "dur": 2.0,
               "args": {"External id": eid1, "kernel": transpose_kname, "cid": corr1a, "correlation": corr1a,
                        "grid": [512, 1, 1], "block": [256, 1, 1], "shared memory": 0}}
    # Second: GEMM kernel
    gemm_kname = "Cijk_Ailk_Bljk_SB<BF16,BF16,float>_MT128x128x32_SE_SU4_TT1_1"
    hip_t1b = {"ph": "X", "cat": "cuda_runtime", "name": "hipLaunchKernel",
               "pid": PID, "tid": CPU_TID, "ts": ts + 25, "dur": 2.0,
               "args": {"External id": eid1, "kernel": gemm_kname, "cid": corr1b, "correlation": corr1b,
                        "grid": [32, 8, 1], "block": [256, 1, 1], "shared memory": 0}}

    transpose_gpu = {"ph": "X", "cat": "kernel", "name": transpose_kname,
                     "pid": GPU_PID, "tid": GPU_TID, "ts": ts + 80, "dur": 80.0,
                     "args": {"External id": eid1, "device": 0, "stream": 0, "correlation": corr1a,
                              "kind": "Dispatch Kernel", "grid": [512, 1, 1], "block": [256, 1, 1]}}
    gemm_gpu = {"ph": "X", "cat": "kernel", "name": gemm_kname,
                "pid": GPU_PID, "tid": GPU_TID, "ts": ts + 170, "dur": 400.0,
                "args": {"External id": eid1, "device": 0, "stream": 0, "correlation": corr1b,
                         "kind": "Dispatch Kernel", "grid": [32, 8, 1], "block": [256, 1, 1]}}

    events1 += [
        cpu_ev1, hip_t1a, hip_t1b,
        *ac2g_pair(ts + 20, ts + 80, corr1a),
        *ac2g_pair(ts + 25, ts + 170, corr1b),
        transpose_gpu, gemm_gpu,
    ]
    trace1 = make_trace(events1, {"experiment": {**BASE_METADATA["experiment"], "repo_id": "test04_mm_transpose_trace1"}})

    # Trace2: aten::addmm → single GEMM kernel, same total runtime (~480us)
    reset_ids()
    events2 = [profiler_step(ts, step_dur)]
    evs, _, _ = build_op_with_kernel(
        cpu_name="aten::addmm",
        kernel_name="Cijk_Ailk_Bljk_SB<BF16,BF16,float>_MT128x128x32_SE_SU4_TT1_1",
        cpu_ts=ts + 10,
        cpu_dur=30.0,
        gpu_ts=ts + 80,
        gpu_dur=480.0,  # same total GPU time as trace1 (80 transpose + 400 gemm)
        input_dims=[[1024, 1], [1024, 2048], [2048, 4096]],
        input_type=["bfloat16", "bfloat16", "bfloat16"],
        flops=MM_FLOPS,
        grid=[32, 8, 1],
        block=[256, 1, 1],
    )
    events2 += evs
    trace2 = make_trace(events2, {"experiment": {**BASE_METADATA["experiment"], "repo_id": "test04_mm_transpose_trace2"}})
    return trace1, trace2


# ---------------------------------------------------------------------------
# Test 5: GEMM — identical wall-clock, Trace1 below roofline
#
# Both: aten::mm with M=512, K=512, N=512 — same ~200us GPU runtime.
# Trace1: compute-bound kernel below roofline (small shape, low occupancy).
# Trace2: near-roofline kernel (same shape, better kernel).
#
# Expected: NOT flagged as optimization — identical runtime means no real gap.
# ---------------------------------------------------------------------------

SMALL_MM_FLOPS = int(512 * 512 * 512 * 2)
SMALL_MM_DIMS = [[512, 512], [512, 512]]
SMALL_MM_TYPE = ["bfloat16", "bfloat16"]
IDENTICAL_GPU_DUR = 200.0


def make_test05_gemm_identical_runtime():
    reset_ids()
    ts = BASE_TS
    step_dur = 2000.0

    events1 = [profiler_step(ts, step_dur)]
    # Use a slow, inefficient kernel name — below roofline
    evs, _, _ = build_op_with_kernel(
        cpu_name="aten::mm",
        kernel_name="Cijk_Ailk_Bljk_SB<BF16,BF16,float>_MT32x32x8_fallback",
        cpu_ts=ts + 10,
        cpu_dur=15.0,
        gpu_ts=ts + 50,
        gpu_dur=IDENTICAL_GPU_DUR,
        input_dims=SMALL_MM_DIMS,
        input_type=SMALL_MM_TYPE,
        flops=SMALL_MM_FLOPS,
        grid=[16, 16, 1],
        block=[64, 1, 1],
    )
    events1 += evs
    trace1 = make_trace(events1, {"experiment": {**BASE_METADATA["experiment"], "repo_id": "test05_identical_runtime_trace1"}})

    reset_ids()
    events2 = [profiler_step(ts, step_dur)]
    # Near-roofline kernel, same GPU duration
    evs, _, _ = build_op_with_kernel(
        cpu_name="aten::mm",
        kernel_name="Cijk_Ailk_Bljk_SB<BF16,BF16,float>_MT128x128x32_SE_SU4",
        cpu_ts=ts + 10,
        cpu_dur=15.0,
        gpu_ts=ts + 50,
        gpu_dur=IDENTICAL_GPU_DUR,
        input_dims=SMALL_MM_DIMS,
        input_type=SMALL_MM_TYPE,
        flops=SMALL_MM_FLOPS,
        grid=[4, 4, 1],
        block=[256, 1, 1],
    )
    events2 += evs
    trace2 = make_trace(events2, {"experiment": {**BASE_METADATA["experiment"], "repo_id": "test05_identical_runtime_trace2"}})
    return trace1, trace2


# ---------------------------------------------------------------------------
# Test 6: Convolution — performance gap exceeds roofline ceiling
#
# Trace1: slow conv, e.g. 60% of peak MAF achieved, takes 2000us.
# Trace2: conv takes 400us (5x faster).
# The roofline ceiling for Trace1 is ~1250us (achievable at 100% MAF from 60%).
# The actual gap (1600us) > roofline headroom (750us) — system must cap improvement.
#
# N=4, C_in=256, H=W=56, C_out=512, kH=kW=3
# FLOPs = 2 * N * C_out * H_out * W_out * C_in * kH * kW
#       = 2 * 4 * 512 * 56 * 56 * 256 * 3 * 3 ≈ 18.5 GFLOPs
# At MI300X peak MAF (1307.4 TFLOP/s BF16): ceiling ~14.2us (unrealistic for
# this size — use scaled-down numbers so the math works for the test).
# We'll use shapes that produce a roofline ceiling of ~500us and an actual
# gap of 1600us so the gap clearly exceeds the ceiling.
# ---------------------------------------------------------------------------

CONV_FLOPS = int(2 * 4 * 512 * 56 * 56 * 256 * 9)  # ~18.5 GFLOPs


def make_test06_conv_gap_exceeds_roofline():
    reset_ids()
    ts = BASE_TS
    step_dur = 8000.0

    kname1 = "miopenConvolutionForwardAlgo_BwdDataB<BF16, BF16, float, 3, 3>"

    events1 = [profiler_step(ts, step_dur)]
    evs, _, _ = build_op_with_kernel(
        cpu_name="aten::convolution",
        kernel_name=kname1,
        cpu_ts=ts + 10,
        cpu_dur=30.0,
        gpu_ts=ts + 80,
        gpu_dur=2000.0,  # slow — 60% of theoretical roofline achievable
        input_dims=[[4, 256, 56, 56], [512, 256, 3, 3], []],
        input_type=["bfloat16", "bfloat16", "bfloat16"],
        flops=CONV_FLOPS,
        grid=[512, 1, 1],
        block=[256, 1, 1],
    )
    events1 += evs
    trace1 = make_trace(events1, {"experiment": {**BASE_METADATA["experiment"], "repo_id": "test06_conv_roofline_trace1"}})

    # Trace2: much faster — effectively at near-peak efficiency
    reset_ids()
    events2 = [profiler_step(ts, step_dur)]
    kname2 = "miopenConvolutionForwardAlgo_ImplicitGEMM<BF16, BF16, float, 3, 3>"
    evs, _, _ = build_op_with_kernel(
        cpu_name="aten::convolution",
        kernel_name=kname2,
        cpu_ts=ts + 10,
        cpu_dur=30.0,
        gpu_ts=ts + 80,
        gpu_dur=400.0,  # fast — near-peak
        input_dims=[[4, 256, 56, 56], [512, 256, 3, 3], []],
        input_type=["bfloat16", "bfloat16", "bfloat16"],
        flops=CONV_FLOPS,
        grid=[512, 1, 1],
        block=[256, 1, 1],
    )
    events2 += evs
    trace2 = make_trace(events2, {"experiment": {**BASE_METADATA["experiment"], "repo_id": "test06_conv_roofline_trace2"}})
    return trace1, trace2


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

TESTS = [
    ("test01_moe_fused_vs_unfused", make_test01_moe_fused),
    ("test02_attention_both_unfused", make_test02_attention_both_unfused),
    ("test03_attention_fused_vs_unfused", make_test03_fused_vs_unfused),
    ("test04_gemm_mm_vs_linear", make_test04_gemm_mm_vs_linear),
    ("test05_gemm_identical_runtime_roofline", make_test05_gemm_identical_runtime),
    ("test06_conv_gap_exceeds_roofline", make_test06_conv_gap_exceeds_roofline),
]

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for test_dir, make_fn in TESTS:
        print(f"\n[{test_dir}]")
        t1, t2 = make_fn()
        out = os.path.join(base_dir, test_dir)
        write_trace(os.path.join(out, "trace1.json"), t1)
        write_trace(os.path.join(out, "trace2.json"), t2)
    print("\nDone.")
