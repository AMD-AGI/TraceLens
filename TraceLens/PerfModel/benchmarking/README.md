<!--
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# GPU Microbenchmarking Suite

This suite measures a GPU's **performance baseline driven by benchmarks** — matrix (MFMA / tensor-core) TFLOPS across dtypes, vector (SIMD) TFLOPS, and HBM bandwidth — and writes it as a GPU-arch JSON in the exact shape TraceLens uses for roofline analysis (see [`Agent/Analysis/utils/arch/`](../../Agent/Analysis/utils/arch/) and [`examples/gpu_arch_example.md`](../../../examples/gpu_arch_example.md)).

Run it once per platform to produce a `<platform>.json` arch file. Roofline-based analysis (including the [TraceLens Agent](../../Agent/Analysis/README.md)) compares each measured kernel against these values to estimate optimization headroom, so an accurate, hardware-specific baseline directly improves the quality of the analysis.

> **These are Benchmark driven achievable FLOPs, not theoretical peak.** Values reflect what the current software stack (PyTorch, ROCm/HIP, Triton, hipBLASLt, aiter/CK) can actually reach on an **idle** device.

## Contents

| File | Purpose |
|------|---------|
| `microbench.py` | Main suite: matrix/vector TFLOPS + HBM bandwidth; writes the arch JSON. |
| `microbench_rocprof.py` | Validation: cross-checks measured GEMM TFLOPS against `rocprofv3` MFMA hardware counters (AMD only). |
| `fp4fp6_helpers.py` | Triton `dot_scaled` + aiter/CK block-scaled MXFP4 / MXFP6 / INT8 GEMM helpers. |
| `microbench_utils.py` | Device resolution and the pre-flight GPU-idle check (`amdsmi` / `nvidia-smi`). |

## Methodology

- Timing via Triton's `do_bench` with L2 cache clearing, `warmup=30`, `rep=200`, median milliseconds.
- Normal-distributed inputs; **best** shape reported per dtype (peak across the GEMM shape list).
- A **pre-flight idle check** aborts if the target GPU is busy (other processes, high utilization, or resident memory), so the baseline reflects an uncontended device. Override with `--allow-busy`.
- Matrix TFLOPS and INT8 use `2·M·N·K` FLOPs per GEMM. FP8 runs through `torch._scaled_mm` (dtype auto-selected per stack); INT8 through `torch._int_mm` and, when available, aiter CK `gemm_a8w8` (max is kept). MXFP4/MXFP6 use Triton `tl.dot_scaled` and, on gfx950, aiter CK `gemm_a4w4`.
- Vector TFLOPS use a compute-bound Triton FMA dependency chain (not PyTorch elementwise) to saturate the SIMD units.
- HBM bandwidth is measured via device-to-device copy (read = `2·bytes`) and fill (write).

## Prerequisites

- A working PyTorch install with GPU support (ROCm/HIP on AMD, CUDA on NVIDIA).
- [Triton](https://triton-lang.org/) — required for vector TFLOPS and MXFP4/MXFP6; the suite degrades gracefully (those metrics report `0.0`) if it is missing.
- Optional: `amdsmi` (AMD) or `nvidia-smi` (NVIDIA) for the idle check; `aiter` for CK GEMM paths; `rocprofv3` for `microbench_rocprof.py`.

Run from the parent directory of the `TraceLens` package so the module path resolves:

```bash
python -m TraceLens.PerfModel.benchmarking.microbench --help
```

## Quick Start

```bash
# Default run on logical device 0; writes gpu_microbench_results.json
python -m TraceLens.PerfModel.benchmarking.microbench --device 0

# Write directly into the arch directory the analysis agent searches
python -m TraceLens.PerfModel.benchmarking.microbench --device 0 \
    --output TraceLens/Agent/Analysis/utils/arch/MI355X.json

# Pin to a specific physical GPU
HIP_VISIBLE_DEVICES=2 python -m TraceLens.PerfModel.benchmarking.microbench \
    --device 0 --output runs/card2.json

# Faster smoke test (lower warmup/rep)
python -m TraceLens.PerfModel.benchmarking.microbench --device 0 --warmup 5 --rep 20
```

The output filename is used as the arch `name` heuristically (memory tier → `MI300X` / `MI355X`), so name the file after your platform.

### Key options

| Flag | Description |
|------|-------------|
| `--device <int>` | Logical torch device index (default `0`). |
| `--output <path>` | Output JSON path (parent dirs auto-created; default `gpu_microbench_results.json`). |
| `--warmup` / `--rep` | Override `do_bench` warmup / timing iterations (default `30` / `200`). |
| `--skip-vector` / `--skip-bandwidth` | Skip the vector-TFLOPS or HBM-bandwidth sections. |
| `--allow-busy` | Skip the pre-flight idle check and run anyway. |
| `--idle-util-threshold <pct>` | Max GPU utilization considered idle (default `5`). |
| `--shape-sweep` | Sweep large + tile-304 GEMM shapes and multi-GB HBM sizes; writes a comparison JSON/CSV. |

## Output

`microbench.py` writes a JSON object with exactly the keys TraceLens expects:

```json
{
    "name": "GPU_NAME",
    "mem_bw_gbps": 0,
    "memory_gb": 0,
    "max_achievable_tflops": {
        "matrix_fp16": 0, "matrix_bf16": 0, "matrix_fp32": 0, "matrix_fp64": 0,
        "matrix_fp8": 0, "matrix_fp4": 0, "matrix_fp6": 0, "matrix_int8": 0,
        "vector_fp16": 0, "vector_bf16": 0, "vector_fp32": 0, "vector_fp64": 0
    }
}
```

Drop this file into an arch directory (the bundled [`Agent/Analysis/utils/arch/`](../../Agent/Analysis/utils/arch/)) as `<platform>.json`, then pass `--gpu_arch_json_path <platform>.json` to the perf-report CLIs. See [`examples/gpu_arch_example.md`](../../../examples/gpu_arch_example.md) for the field reference.

## Validating against hardware counters (AMD)

`microbench_rocprof.py` reruns each single-shape GEMM under `rocprofv3` and compares measured TFLOPS against the MFMA MOPS counters (`SQ_INSTS_VALU_MFMA_MOPS_*`), reporting a `rocprof / calculated` FLOPs ratio and `rocprof / measured` time ratio per metric — a sanity check that the `2·M·N·K` model matches what the hardware actually issued.

```bash
python -m TraceLens.PerfModel.benchmarking.microbench_rocprof --device 0 \
    --output results/rocprof_compare.json
```
