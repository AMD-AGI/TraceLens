###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Shared test helpers migrated from test_reporting_coverage.py."""

from __future__ import annotations
import gzip
import json
from pathlib import Path
import pandas as pd
import pytest

pytestmark = pytest.mark.filterwarnings(
    "ignore:Source column .* not found.*:UserWarning",
    "ignore:There are hipgraph launches.*:UserWarning",
    "ignore:Found .* events with failed performance metric.*:UserWarning",
    "ignore:Input list of events is empty.*:UserWarning",
    "ignore:dict_cat2names_extension is deprecated.*:UserWarning",
)
KERNEL_TRACE_CSV = """\
"Kind","Agent_Id","Queue_Id","Stream_Id","Thread_Id","Dispatch_Id","Kernel_Id","Kernel_Name","Correlation_Id","Start_Timestamp","End_Timestamp","LDS_Block_Size","Scratch_Size","VGPR_Count","Accum_VGPR_Count","SGPR_Count","Workgroup_Size_X","Workgroup_Size_Y","Workgroup_Size_Z","Grid_Size_X","Grid_Size_Y","Grid_Size_Z"
"KERNEL_DISPATCH","Agent 2",1,0,70,1,33,"__amd_rocclr_fillBufferAligned",119662,172352210005122,172352210008687,0,0,12,4,48,256,1,1,256,1,1
"KERNEL_DISPATCH","Agent 2",1,0,70,2,16,"kernel_step_1_c532_0_kernel_6_range_for",119670,172352210061004,172352210062686,0,0,4,4,16,1,1,1,1,1,1
"KERNEL_DISPATCH","Agent 2",1,0,70,3,31,"func_broad_phase_c402_0_kernel_3_range_for",119696,172352210143326,172352210149335,0,0,16,0,32,512,1,1,512,1,1
"""


def _mk_event(cat, name, ts, dur, pid, tid, args=None):
    return {
        "ph": "X",
        "cat": cat,
        "name": name,
        "pid": pid,
        "tid": tid,
        "ts": ts,
        "dur": dur,
        "args": args or {},
    }


def _mk_ac2g(corr_id, pid, tid, ts, phase):
    evt = {
        "ph": phase,
        "id": corr_id,
        "pid": pid,
        "tid": tid,
        "ts": ts,
        "cat": "ac2g",
        "name": "ac2g",
    }
    if phase == "f":
        evt["bp"] = "e"
    return evt


def _build_synthetic_trace(kernel_specs):
    events = []
    ts = 1000
    corr_id = 100
    cpu_pid, cpu_tid = 100, 100
    gpu_pid, gpu_tid = 0, 7

    for cpu_op_name, kernel_name, kernel_dur in kernel_specs:
        cpu_op_ts = ts
        cpu_op_dur = 100
        events.append(
            _mk_event(
                "cpu_op",
                cpu_op_name,
                ts=cpu_op_ts,
                dur=cpu_op_dur,
                pid=cpu_pid,
                tid=cpu_tid,
                args={"Input Dims": [[32, 64]], "Input type": ["float"]},
            )
        )
        events.append(
            _mk_event(
                "cuda_runtime",
                "hipLaunchKernel",
                ts=cpu_op_ts + 10,
                dur=5,
                pid=cpu_pid,
                tid=cpu_tid,
                args={"correlation": corr_id},
            )
        )
        kernel_ts = cpu_op_ts + 50
        events.append(
            _mk_event(
                "kernel",
                kernel_name,
                ts=kernel_ts,
                dur=kernel_dur,
                pid=gpu_pid,
                tid=gpu_tid,
                args={"correlation": corr_id, "stream": 7},
            )
        )
        events.append(_mk_ac2g(corr_id, gpu_pid, gpu_tid, kernel_ts, "s"))
        events.append(_mk_ac2g(corr_id, gpu_pid, gpu_tid, kernel_ts, "f"))
        ts += cpu_op_dur + 200
        corr_id += 1

    return {"traceEvents": events}


def _write_trace(tmp_path: Path, specs, name="trace.json") -> str:
    path = tmp_path / name
    path.write_text(json.dumps(_build_synthetic_trace(specs)))
    return str(path)


def _create_genesis_capture(tmp_path: Path) -> Path:
    capture = tmp_path / "capture"
    kernel_trace = capture / "kernel_trace"
    kernel_trace.mkdir(parents=True)
    (kernel_trace / "kernel_kernel_trace.csv").write_text(KERNEL_TRACE_CSV)
    (capture / "run.log").write_text("wall_time=4.00s\n")
    return capture


def _minimal_pftrace_events():
    return [
        {
            "ph": "X",
            "cat": "gpu_activity",
            "name": "xla_fusion_42",
            "pid": 0,
            "tid": 7,
            "ts": 1000,
            "dur": 50000,
            "args": {"agent": "gpu_0", "begin_ns": 1000000, "delta_ns": 50000000},
        },
        {
            "ph": "X",
            "cat": "hip_api",
            "name": "hipLaunchKernelGGL",
            "pid": 100,
            "tid": 1,
            "ts": 900,
            "dur": 20,
            "args": {"stream_ID": 0},
        },
    ]


def _rich_pftrace_events():
    events = list(_minimal_pftrace_events())
    events.extend(
        [
            {
                "ph": "X",
                "cat": "gpu_activity",
                "name": "ncclAllReduce_ring",
                "pid": 0,
                "tid": 7,
                "ts": 2000,
                "dur": 100000,
                "args": {
                    "agent": "gpu_0",
                    "begin_ns": 2_000_000_000,
                    "delta_ns": 100_000_000,
                },
            },
            {
                "ph": "X",
                "cat": "gpu_activity",
                "name": "Cijk_A_B_gemm",
                "pid": 0,
                "tid": 8,
                "ts": 3000,
                "dur": 80000,
                "args": {
                    "agent": "gpu_0",
                    "begin_ns": 3_000_000_000,
                    "delta_ns": 80_000_000,
                    "grid_size": 256,
                    "workgroup_size": 256,
                    "VGPR_Count": 32,
                    "stream_ID": 1,
                    "queue": 2,
                },
            },
            {
                "ph": "X",
                "cat": "gpu_activity",
                "name": "FmhaBwd_kernel_func_v3",
                "pid": 0,
                "tid": 7,
                "ts": 4000,
                "dur": 60000,
                "args": {
                    "agent": "gpu_0",
                    "begin_ns": 4_000_000_000,
                    "delta_ns": 60_000_000,
                },
            },
            {
                "ph": "X",
                "cat": "hip_api",
                "name": "hipMemcpyAsync",
                "pid": 100,
                "tid": 2,
                "ts": 850,
                "dur": 5000,
                "args": {
                    "stream_ID": 1,
                    "operation": 42,
                    "begin_ns": 850_000,
                    "delta_ns": 5_000_000,
                },
            },
        ]
    )
    return events


def _full_pftrace_events():
    """Events exercising every pftrace classify branch and analyzer option."""
    return [
        {
            "ph": "X",
            "cat": "gpu_activity",
            "name": "ncclAllReduce",
            "pid": 0,
            "tid": 7,
            "ts": 1000,
            "dur": 50000,
            "args": {
                "agent": "gpu_0",
                "begin_ns": 1_000_000_000,
                "delta_ns": 50_000_000,
            },
        },
        {
            "ph": "X",
            "cat": "gpu_activity",
            "name": "Cijk_AB",
            "pid": 0,
            "tid": 7,
            "ts": 2000,
            "dur": 40000,
            "args": {
                "agent": "gpu_0",
                "begin_ns": 2_000_000_000,
                "delta_ns": 40_000_000,
                "grid_size": 128,
                "workgroup_size": 64,
                "VGPR_Count": 16,
            },
        },
        {
            "ph": "X",
            "cat": "gpu_activity",
            "name": "FmhaBwd_kernel_func",
            "pid": 0,
            "tid": 7,
            "ts": 3000,
            "dur": 30000,
            "args": {
                "agent": "gpu_0",
                "begin_ns": 3_000_000_000,
                "delta_ns": 30_000_000,
            },
        },
        {
            "ph": "X",
            "cat": "gpu_activity",
            "name": "FmhaFwd_main",
            "pid": 0,
            "tid": 7,
            "ts": 3500,
            "dur": 25000,
            "args": {
                "agent": "gpu_0",
                "begin_ns": 3_500_000_000,
                "delta_ns": 25_000_000,
            },
        },
        {
            "ph": "X",
            "cat": "gpu_activity",
            "name": "memcpyHtoD",
            "pid": 0,
            "tid": 7,
            "ts": 3600,
            "dur": 20000,
            "args": {
                "agent": "gpu_0",
                "begin_ns": 3_600_000_000,
                "delta_ns": 20_000_000,
            },
        },
        {
            "ph": "X",
            "cat": "gpu_activity",
            "name": "transformer_engine_linear",
            "pid": 0,
            "tid": 7,
            "ts": 3700,
            "dur": 15000,
            "args": {
                "agent": "gpu_0",
                "begin_ns": 3_700_000_000,
                "delta_ns": 15_000_000,
            },
        },
        {
            "ph": "X",
            "cat": "gpu_activity",
            "name": "aiter::fmha_fwd_kernel",
            "pid": 0,
            "tid": 7,
            "ts": 3800,
            "dur": 12000,
            "args": {
                "agent": "gpu_0",
                "begin_ns": 3_800_000_000,
                "delta_ns": 12_000_000,
            },
        },
        {
            "ph": "X",
            "cat": "gpu_activity",
            "name": "aiter::fmha_bwd_kernel",
            "pid": 0,
            "tid": 7,
            "ts": 3900,
            "dur": 11000,
            "args": {
                "agent": "gpu_0",
                "begin_ns": 3_900_000_000,
                "delta_ns": 11_000_000,
            },
        },
        {
            "ph": "X",
            "cat": "gpu_activity",
            "name": "fillBuffer_kernel",
            "pid": 0,
            "tid": 7,
            "ts": 4000,
            "dur": 8000,
            "args": {
                "agent": "gpu_0",
                "begin_ns": 4_000_000_000,
                "delta_ns": 8_000_000,
            },
        },
        {
            "ph": "X",
            "cat": "gpu_activity",
            "name": "xla_generic_fusion",
            "pid": 0,
            "tid": 7,
            "ts": 4100,
            "dur": 7000,
            "args": {
                "agent": "gpu_0",
                "begin_ns": 4_100_000_000,
                "delta_ns": 7_000_000,
            },
        },
        {
            "ph": "X",
            "cat": "hip_api",
            "name": "hipLaunchKernel",
            "pid": 100,
            "tid": 1,
            "ts": 900,
            "dur": 5000,
            "args": {
                "stream_ID": 0,
                "operation": 1,
                "begin_ns": 900_000,
                "delta_ns": 5_000_000,
            },
        },
    ]


class _MockShortKernelAnalyzer:
    def __init__(self, gpu_only=False, kernels=None, total_time_ms=1.0):
        self.gpu_only = gpu_only
        self.total_time_ms = total_time_ms
        self._kernels = (
            kernels
            if kernels is not None
            else pd.DataFrame(
                {
                    "Kernel duration (µs)": [5.0, 8.0, 50.0],
                    "Kernel name": ["k_short_a", "k_short_b", "k_long"],
                    "Parent cpu_op": ["aten::mm"] * 3,
                    "Input dims": ["[[32, 64]]"] * 3,
                    "Input strides": [""] * 3,
                    "Concrete Inputs": [""] * 3,
                }
            )
        )

    def get_df_kernels(self):
        return self._kernels


def _make_trace(rank, n_collectives):
    events = []
    base_ts = 1_000_000 + rank * 50
    for i in range(n_collectives):
        ts = base_ts + i * 1000 + rank * 5
        events.append(
            {
                "ph": "X",
                "cat": "kernel",
                "name": "void rcclGenericKernel<1, false>(ncclDevKernelArgsStorage<4096ul>)",
                "pid": rank,
                "tid": 3,
                "ts": ts,
                "dur": 50,
                "args": {
                    "External id": 100 + i,
                    "device": rank,
                    "stream": 3,
                    "correlation": 50 + i,
                },
            }
        )
    return {"traceEvents": events}


# --- from test_coverage_95_final ---
def _write_gz_trace(tmp_path, events, name="trace.json.gz"):
    path = tmp_path / name
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump({"traceEvents": events}, f)
    return str(path)


def _jax_llama_trace_events(block0_hint: str = "te_layernorm_forward"):
    """Minimal JAX LLaMA-style Chrome trace for compare_traces_jax_llama helpers."""
    base_path = (
        "jit(main)/jit(call)/jit(layer)/Transformer/block_{block}/norm_attn/"
        + block0_hint
    )
    events = [
        {
            "ph": "M",
            "name": "process_name",
            "pid": 1,
            "args": {"name": "/device:GPU:0"},
        },
        {
            "ph": "M",
            "name": "thread_name",
            "pid": 1,
            "tid": 10,
            "args": {"name": "Stream"},
        },
    ]
    ts = 1000.0
    for tok in range(2):
        for block in range(2):
            p = base_path.format(block=block).replace("block_0", f"block_{block}")
            if block == 0 and tok == 0:
                p = base_path.format(block=0)
            events.append(
                {
                    "ph": "X",
                    "pid": 1,
                    "tid": 10,
                    "ts": ts,
                    "dur": 50,
                    "name": "ln_fwd_tuned_kernel<Kernel_traits<float, 4096u, 64>",
                    "args": {"name": p},
                }
            )
            ts += 60
            events.append(
                {
                    "ph": "X",
                    "pid": 1,
                    "tid": 10,
                    "ts": ts,
                    "dur": 80,
                    "name": "Cijk_gemm",
                    "args": {"name": p.replace("norm_attn", "attn/q/dot_general")},
                }
            )
            ts += 90
            events.append(
                {
                    "ph": "X",
                    "pid": 1,
                    "tid": 10,
                    "ts": ts,
                    "dur": 120,
                    "name": "te_fused_attn_forward",
                    "args": {"name": p.replace("norm_attn", "attn/out/dot_general")},
                }
            )
            ts += 130
            events.append(
                {
                    "ph": "X",
                    "pid": 1,
                    "tid": 10,
                    "ts": ts,
                    "dur": 70,
                    "name": "loop_multiply_fusion",
                    "args": {
                        "name": "jit(main)/Transformer/mlp/in/dot_general",
                        "hlo_op": "loop_multiply_fusion",
                    },
                }
            )
            ts += 80
            events.append(
                {
                    "ph": "X",
                    "pid": 1,
                    "tid": 10,
                    "ts": ts,
                    "dur": 60,
                    "name": "Cijk_gemm",
                    "args": {"name": p.replace("norm_attn", "mlp/out/dot_general")},
                }
            )
            ts += 100
    return events
