###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

#!/usr/bin/env python3
"""Generate synthetic single-kernel PyTorch traces for TraceLens eval unit tests.

Each trace contains exactly one CPU op → one GPU kernel, following the Chrome
Trace Event format that TraceLens expects. Kernel durations are deliberately
set to be sub-optimal so that the analysis pipeline produces impact_estimates,
which in turn enables plot generation (perf_improvement.png).

Impact estimate formula: savings_ms = time_ms * (1 - efficiency_pct / 100)
  - Requires efficiency < 100% (not anomalous) and savings >= 0.1 ms
  - "high" confidence when time_ms > 5 and efficiency_pct < 70

Only operations with TraceLens built-in FLOP formulas produce roofline metrics:
  - GEMM: aten::mm, aten::bmm, aten::addmm  (FLOPs = 2*M*N*K)
  - Conv:  aten::convolution                  (FLOPs from conv params)
  - SDPA:  aten::_scaled_dot_product_*        (FLOPs ≈ 4*B*H*N^2*d)

B300 peaks (public estimates): matrix_bf16 ≈ 3000 TFLOPS (dense Tensor Core),
matrix_fp32 ≈ 80 TFLOPS (CUDA cores), HBM3e BW ≈ 8.0 TB/s.
"""

import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UNIT_TESTS_DIR = os.path.join(BASE_DIR, "unit_tests_nv")


def compute_strides(dims):
    """Compute contiguous tensor strides from shape."""
    strides = [1] * len(dims)
    for i in range(len(dims) - 2, -1, -1):
        strides[i] = strides[i + 1] * dims[i + 1]
    return strides


def build_trace(
    op_name,
    input_dims,
    input_types,
    concrete_inputs,
    kernel_name,
    kernel_dur_us,
    cpu_dur_us=None,
    grid=None,
    block=None,
):
    """Build a single-kernel Chrome Trace Event JSON."""
    if cpu_dur_us is None:
        cpu_dur_us = kernel_dur_us + 200
    if grid is None:
        grid = [256, 1, 1]
    if block is None:
        block = [256, 1, 1]

    input_strides = [compute_strides(d) for d in input_dims]

    return {
        "traceEvents": [
            {"ph": "M", "pid": 0, "tid": 0, "name": "process_name",
             "args": {"name": "GPU 0"}},
            {"ph": "M", "pid": 470, "tid": 0, "name": "process_name",
             "args": {"name": "python3"}},
            {"ph": "M", "pid": 470, "tid": 711, "name": "thread_name",
             "args": {"name": "MainThread"}},
            {"ph": "M", "pid": 0, "tid": 7, "name": "thread_name",
             "args": {"name": "stream 7"}},
            {
                "ph": "X", "cat": "cpu_op", "name": op_name,
                "pid": 470, "tid": 711, "ts": 1000000.0, "dur": cpu_dur_us,
                "args": {
                    "Input Dims": input_dims,
                    "Input type": input_types,
                    "Input Strides": input_strides,
                    "Concrete Inputs": concrete_inputs,
                    "External id": 100,
                },
            },
            {
                "ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel",
                "pid": 470, "tid": 711, "ts": 1000010.0, "dur": 5.0,
                "args": {"External id": 100},
            },
            {
                "ph": "X", "cat": "kernel", "name": kernel_name,
                "pid": 0, "tid": 7, "ts": 1000050.0, "dur": kernel_dur_us,
                "args": {
                    "device": 0, "context": 1, "stream": 7,
                    "grid": grid, "block": block,
                    "External id": 100,
                },
            },
            {"ph": "s", "id": 100, "pid": 0, "tid": 7, "ts": 1000050.0,
             "cat": "ac2g", "name": "ac2g"},
            {"ph": "f", "id": 100, "pid": 0, "tid": 7, "ts": 1000050.0,
             "cat": "ac2g", "name": "ac2g", "bp": "e"},
        ]
    }


# ─── Test case definitions ───────────────────────────────────────────────────
# Format: (id, category_dir, op_name, input_dims, input_types, concrete_inputs,
#           kernel_name, kernel_dur_us, grid, block)

CASES = [
    # ── GEMM (BF16, peak ≈ 3000 TFLOPS) ────────────────────
    # gemm_01: Large square. FLOPs=2*8192^3=1100 GFLOPS. @8ms→137 TFLOPS→4.6% eff
    (
        "gemm_01_large_square_bf16", "gemm",
        "aten::mm",
        [[8192, 8192], [8192, 8192]],
        ["BFloat16", "BFloat16"],
        ["", ""],
        "sm100_xmma_gemm_bf16f16_bf16f16_bf16f32_tn_n_tilesize256x128x64_stage4_warpsize2x2x1",
        8000, [512, 1, 1], [256, 1, 1],
    ),
    # gemm_02: Tall-skinny. FLOPs=2*16384*1024*16384=550 GFLOPS.
    #   @6ms→92 TFLOPS→3.1% eff (memory-bound on B300 at this shape)
    (
        "gemm_02_tall_skinny_bf16", "gemm",
        "aten::mm",
        [[16384, 1024], [1024, 16384]],
        ["BFloat16", "BFloat16"],
        ["", ""],
        "sm100_xmma_gemm_bf16f16_bf16f16_bf16f32_nt_n_tilesize128x256x64_stage3_warpsize2x2x1",
        6000, [1024, 1, 1], [256, 1, 1],
    ),

    # ── SDPA / Attention (BF16, peak ≈ 3000 TFLOPS) ──────────────────────────
    # attn_04: Moderate seq len, sub-optimal flash. B=4,H=32,N=4096,d=128
    #   FLOPs≈4*4*32*4096^2*128=1100 GFLOPS. @7ms→157 TFLOPS→5.2% eff
    (
        "attn_04_slow_flash_short_seq", "attention",
        "aten::_scaled_dot_product_flash_attention",
        [[4, 32, 4096, 128], [4, 32, 4096, 128], [4, 32, 4096, 128]],
        ["BFloat16", "BFloat16", "BFloat16"],
        ["", "", "", "0.0", "False", "False", ""],
        "void pytorch_flash::flash_fwd_kernel_bf16<sm100, headdim128>",
        7000, [2048, 32, 4], [128, 1, 1],
    ),
    # attn_05: GQA pattern (H_Q=32, H_KV=8). B=4,N=4096,d=128
    #   FLOPs≈4*4*32*4096^2*128=1100 GFLOPS (uses H_Q for compute).
    #   @8ms→137 TFLOPS→4.6% eff
    (
        "attn_05_slow_flash_gqa", "attention",
        "aten::_scaled_dot_product_flash_attention",
        [[4, 32, 4096, 128], [4, 8, 4096, 128], [4, 8, 4096, 128]],
        ["BFloat16", "BFloat16", "BFloat16"],
        ["", "", "", "0.0", "False", "False", ""],
        "void pytorch_flash::flash_fwd_splitkv_kernel_bf16<sm100, headdim128>",
        8000, [2048, 32, 4], [128, 1, 1],
    ),

    # ── Convolution (FP32, peak ≈ 80 TFLOPS) ──────────────────────
    # conv_08: Large batch conv2d. [16,64,256,256]x[128,64,3,3] stride=1 pad=1
    #   FLOPs ≈ 155 GFLOPs. @6ms→26 TFLOPS→32% eff
    (
        "conv_08_large_batch_small_kernel", "conv",
        "aten::convolution",
        [[16, 64, 256, 256], [128, 64, 3, 3]],
        ["Float", "Float"],
        ["", "", "", "(1,1)", "(1,1)", "(1,1)", "False", "(0,0)", "1"],
        "sm100_xmma_fprop_implicit_gemm_f32f32_f32f32_f32_nhwc_tilesize256x128x16_stage3",
        6000, [2048, 1, 1], [256, 1, 1],
    ),
    # conv_09: Grouped conv2d. [8,128,128,128]x[128,32,5,5] groups=4 stride=2 pad=2
    #   Sub-optimal efficiency due to group fragmentation.
    (
        "conv_09_grouped_conv", "conv",
        "aten::convolution",
        [[8, 128, 128, 128], [128, 32, 5, 5]],
        ["Float", "Float"],
        ["", "", "", "(2,2)", "(2,2)", "(1,1)", "False", "(0,0)", "4"],
        "sm100_xmma_fprop_implicit_gemm_grouped_f32f32_f32f32_f32_nhwc_tilesize128x128x16",
        7000, [1024, 1, 1], [256, 1, 1],
    ),
]


def create_test_case(case_id, category, trace_data):
    """Create the directory structure and write the trace JSON."""
    case_dir_name = f"{case_id}_analysis_output"
    category_dir = os.path.join(UNIT_TESTS_DIR, category)
    case_dir = os.path.join(category_dir, case_dir_name)
    ref_dir = os.path.join(case_dir, "analysis_output_ref")

    os.makedirs(case_dir, exist_ok=True)
    os.makedirs(ref_dir, exist_ok=True)

    trace_path = os.path.join(case_dir, f"{case_id}.json")
    with open(trace_path, "w") as f:
        json.dump(trace_data, f, indent=2)

    return {
        "id": case_id,
        "sub_category": category,
        "trace_path": f"evals_comparative/unit_tests_nv/{category}/{case_dir_name}/{case_id}.json",
        "reference_dir": f"evals_comparative/unit_tests_nv/{category}/{case_dir_name}/analysis_output_ref",
        "platform": "B300",
    }


def main():
    csv_rows = []

    for case_tuple in CASES:
        (case_id, category, op_name, input_dims, input_types,
         concrete_inputs, kernel_name, kernel_dur_us, grid, block) = case_tuple

        trace = build_trace(
            op_name=op_name,
            input_dims=input_dims,
            input_types=input_types,
            concrete_inputs=concrete_inputs,
            kernel_name=kernel_name,
            kernel_dur_us=kernel_dur_us,
            grid=grid,
            block=block,
        )

        row = create_test_case(case_id, category, trace)
        csv_rows.append(row)
        print(f"  Created: {case_id} ({category})")

    print("\n--- CSV entries to add to unit_test_traces.csv ---")
    for r in csv_rows:
        print(f"{r['id']},{r['sub_category']},{r['trace_path']},{r['reference_dir']},{r['platform']}")

    print(f"\nGenerated {len(csv_rows)} test cases.")


if __name__ == "__main__":
    main()
