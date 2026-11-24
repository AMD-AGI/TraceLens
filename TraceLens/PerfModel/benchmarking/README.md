<!--
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# GPU Microbenchmarking Suite

This suite measures a GPU's **performance baseline driven by benchmarks** — matrix (MFMA / tensor-core) TFLOPS across dtypes, vector (SIMD) TFLOPS, and memory bandwidth — and writes it as a GPU-arch JSON in the exact shape TraceLens uses for roofline analysis (see [`Agent/Analysis/utils/arch/`](../../Agent/Analysis/utils/arch/) and [`examples/gpu_arch_example.md`](../../../examples/gpu_arch_example.md)).

Run it once per platform to produce a `<platform>.json` arch file. Roofline-based analysis (including the [TraceLens Agent](../../Agent/Analysis/README.md)) compares each measured kernel against these values to estimate optimization headroom, so an accurate, hardware-specific baseline directly improves the quality of the analysis.

> **These are Benchmark driven achievable FLOPs, not theoretical peak.** Values reflect what the current software stack (PyTorch, ROCm/HIP, Triton, hipBLASLt, aiter/CK) can actually reach on an **idle** device.

## Contents

| File | Purpose |
|------|---------|
| `microbench.py` | Main suite: matrix/vector TFLOPS + memory bandwidth; writes the arch JSON. |
| `microbench_rdna.py` | The measurement protocol `microbench.py` runs on RDNA targets (gfx11, gfx12); see [RDNA targets](#rdna-targets-gfx11-gfx12). |
| `microbench_rocprof.py` | Validation: cross-checks measured GEMM TFLOPS against `rocprofv3` MFMA hardware counters (AMD only); on RDNA targets the shader clock, dual issue and read bytes instead. |
| `fp4fp6_helpers.py` | Triton `dot_scaled` + aiter/CK block-scaled MXFP4 / MXFP6 / INT8 GEMM helpers. |
| `microbench_utils.py` | Device resolution, the pre-flight GPU-idle check (`amdsmi` / `nvidia-smi`), and for RDNA targets the key rule, window floor, bandwidth plateau, sysfs telemetry, platform record and counter checks. |

## Methodology

- Timing via Triton's `do_bench` with L2 cache clearing, `warmup=30` ms, `rep=200` ms, mean milliseconds per call. RDNA targets use the protocol in [RDNA targets](#rdna-targets-gfx11-gfx12) instead.
- Normal-distributed inputs; **best** shape reported per dtype (peak across the GEMM shape list).
- A **pre-flight idle check** aborts if the target GPU is busy (other processes, high utilization, or resident memory), so the baseline reflects an uncontended device. Override with `--allow-busy`.
- Matrix TFLOPS and INT8 use `2·M·N·K` FLOPs per GEMM. FP8 runs through `torch._scaled_mm` (dtype auto-selected per stack); INT8 through `torch._int_mm` and, when available, aiter CK `gemm_a8w8` (max is kept). MXFP4/MXFP6 use Triton `tl.dot_scaled` and, on gfx950, aiter CK `gemm_a4w4`. On AMD targets without native MX matrix instructions (any target not in `NATIVE_MX_GFX_ARCHS` in `microbench_utils.py`), these formats are converted and multiplied on the 16-bit matrix cores, so `matrix_fp4` and `matrix_fp6` report the 16-bit matrix peak instead of the converted kernel's measurement.
- Vector TFLOPS use a compute-bound Triton FMA dependency chain (not PyTorch elementwise) to saturate the SIMD units.
- Memory bandwidth is measured via device-to-device copy (read = `2·bytes`) and fill (write).

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

The output filename is used as the arch `name` heuristically (memory tier → `MI300X` / `MI355X`), so name the file after your platform. Radeon PRO boards are named after the model that follows `PRO`, so both Radeon PRO W7900 boards (`AMD Radeon Pro W7900` and `AMD Radeon Pro W7900 Dual Slot`) are named `W7900`.

### Key options

| Flag | Description |
|------|-------------|
| `--device <int>` | Logical torch device index (default `0`). |
| `--output <path>` | Output JSON path (parent dirs auto-created; default `gpu_microbench_results.json`). |
| `--warmup` / `--rep` | Override the `do_bench` warmup / timing window in milliseconds (default `30` / `200`). On RDNA targets they can only lengthen `--warm-s` / `--window-ms`; see [RDNA targets](#rdna-targets-gfx11-gfx12). |
| `--warm-s` | RDNA targets: seconds of back-to-back calls before each section (default `3`). |
| `--window-ms` | RDNA targets: timing window per config in milliseconds, reported as the median (default `1000`). |
| `--skip-burst` / `--skip-back-to-back` | RDNA targets: skip the burst pass or the back-to-back run. |
| `--skip-vector` / `--skip-bandwidth` | Skip the vector-TFLOPS or memory-bandwidth sections. |
| `--allow-busy` | Skip the pre-flight idle check and run anyway. |
| `--idle-util-threshold <pct>` | Max GPU utilization considered idle (default `5`). |
| `--shape-sweep` | Sweep large + tile-304 GEMM shapes and multi-GB memory bandwidth sizes; writes a comparison JSON/CSV. On RDNA targets the tile-304 shapes are re-tiled to the board's CU count (`cu_tiled_sweep_shapes()`), and the suite's timing is used, not the RDNA protocol. |

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

All numeric values are floats rounded to one decimal place (for example `0.4`
TFLOPS for a slow fp64 unit), not integers.

Drop this file into an arch directory (the bundled [`Agent/Analysis/utils/arch/`](../../Agent/Analysis/utils/arch/)) as `<platform>.json`, then pass `--gpu_arch_json_path <platform>.json` to the perf-report CLIs. See [`examples/gpu_arch_example.md`](../../../examples/gpu_arch_example.md) for the field reference.

On RDNA targets the object also carries `gfx_arch` after `name`, and three fields after `source`, which the readers of the arch JSON ignore:

- `max_achievable_tflops_source`: per key, `measured` with the winning config, `derived` with the key it is taken from and why, or `skipped`.
- `measurement`: the protocol settings (`protocol`, with any `--warmup` / `--rep` override in `cli_overrides`), and per section every timed config with the clocks of its window, plus the burst and back-to-back results of its winners.
- `platform`: GPU name, `gfx_arch`, torch's `multi_processor_count` (work-group processors on RDNA, half the CU count), whether it is an APU, PCI IDs and address, VBIOS, power cap (current and default, and where it came from), the highest window power (`power_held_w`, also where no cap is exposed), performance level, the shader, memory and fabric DPM tables, and the kernel, amdgpu, ROCm, HIP, PyTorch and Triton versions. `sclk_max_mhz` is the top DPM level; gfx12 boards run past it.

## RDNA targets (gfx11, gfx12)

On an RDNA target (`rdna_family()` in `microbench_utils.py`) a default run (not `--shape-sweep`) measures every section under one protocol (`microbench_rdna.py`), the same on every board, so that keys are comparable across boards and runs. Other targets run the sections above unchanged.

- **Timing.** Each section starts with `--warm-s` (3 s) of back-to-back calls of its first config. Each config runs once before its window, so a Triton compile is not timed, and gets 1 s of back-to-back calls first if its setup took over 0.25 s. It is then timed over a `--window-ms` (1000 ms) `do_bench` window and reported as the median ms per call. A config whose call takes over 0.5 s (`RDNA_SERIAL_CALL_S`) is timed one call at a time over the window instead, at least 3 calls, and its row records `synced_calls`: `do_bench` queues about 7 calls before it syncs, and seconds-long calls queued back to back starve a desktop on the same GPU past the driver's 10 s gfx ring timeout, which resets the GPU. The shader, memory and fabric clocks, power and power cap are sampled from sysfs (hwmon) every 0.1 s, and the medians over each window are recorded next to its rate. The sysfs clocks are SMU averages, so a window with fewer than 3 samples, such as a burst window, records only the sample count.
- **Window floor.** `--warmup` and `--rep` can lengthen the warm-up and the window but not shorten them. A caller that passes shorter values, such as Hyperloom's `--warmup 20 --rep 50`, still gets 3 s and 1000 ms; the override is printed and recorded in `measurement.protocol.cli_overrides`. A `--window-ms` under 1000 runs with a warning that its keys are not comparable across boards.
- **Matrix keys** are the best library GEMM: FP16, BF16, FP32 and FP64 over hipBLASLt and rocBLAS (`torch.backends.cuda.preferred_blas_library`) in the NN and NT layouts, less `RDNA_BLAS_SKIPS` (on gfx1151 FP32 and FP64 run through rocBLAS only: hipBLASLt has no tuned FP32 kernel there and takes seconds per call, and its FP64 GEMM runs rocBLAS's kernel), INT8 through `torch._int_mm`, and FP8 through `torch._scaled_mm` on gfx12. The shapes are `GEMM_SHAPES`, plus for the WMMA formats the deep-K shapes `RDNA_DEEP_K_SHAPES` (the fastest found on gfx1151 and gfx1201), and for FP64 one 4096³ GEMM. A Triton WMMA loop on register tiles (BF16, FP16) is recorded next to them with its clock; it is not a key.
- **Vector keys** (FP16, FP32, FP64) are each the best of the FMA chain at `VECTOR_CONFIGS_RDNA` and of a `[ILP, BLOCK]` accumulator tile (`acc = fma(x, y, acc)`) at `VECTOR_TILE_CONFIGS_RDNA`. The tile compiles to VOPD dual-issue FMAs (`v_dual_fmac_f32`) on gfx11 and gfx12; the chain's `fma(acc, x, y)` compiles to `v_fma_f32`, which has no VOPD form. The tile does the chain's FLOPs on 1/8 of its elements, each running 8 chains, so its loads and stores do not compete with the FMAs for power. The share of the winner's FMAs issued as VOPD pairs is recorded.
- **Bandwidth.** `mem_bw_gbps` is the higher of two read-only curves, a Triton read kernel and `torch.sum`, over 512 MiB to 8 GiB (capped at half the free memory). Each call follows a 512 MiB read-only sweep that evicts the caches without leaving dirty lines to write back. A curve's rate is the median from the first size at which every step to the next size changes the rate by less than 2 %; if no plateau forms, its largest rate, and the rule says so. Copy and fill (64 MB to 2 GB) are recorded as diagnostics.
- **Side measurements**, recorded next to the keys but not published. Burst: after the sections, each section's winner runs again for 100 ms after 5 s with no GPU work, the rate a short job sees while the clocks are still rising. Back to back: the FP16 and BF16 GEMM winners run for 30 s each, timed in 0.5 s chunks of wall time, sized from a first 50 ms chunk; the median chunk after 10 s, by which time the board has settled at its power limit, is recorded with its clocks and power, next to the median of the first 10 s and the slowest and fastest chunk after.
- **Keys without a path** are derived from the path they run on, and `max_achievable_tflops_source` says so: FP8 (gfx11), FP4 and FP6 take `matrix_fp16`, and `vector_bf16` takes `vector_fp32`. FP32 and FP64 GEMMs keep the GEMM rate even where the vector rate is higher.
- **Exit codes.** `2`: the GPU is busy (see `--allow-busy`). `3`: a key the target has a path for, or the read bandwidth, measured zero or failed; no JSON is written, so a caller falls back to its own values instead of reading a 0.
- **Idle check.** The median of 20 `amdsmi` readings 0.1 s apart, taken in a fresh interpreter on the device selected by its PCI address. VRAM use over 10 % also counts as busy, except on an APU. If the device cannot be read the check fails instead of being skipped; `--allow-busy` overrides it.

A run takes about 6 minutes on an R9700 (gfx1201).

## Validating against hardware counters (AMD)

`microbench_rocprof.py` reruns each single-shape GEMM under `rocprofv3` and compares measured TFLOPS against the MFMA MOPS counters (`SQ_INSTS_VALU_MFMA_MOPS_*`), reporting a `rocprof / calculated` FLOPs ratio and `rocprof / measured` time ratio per metric — a sanity check that the `2·M·N·K` model matches what the hardware actually issued.

```bash
python -m TraceLens.PerfModel.benchmarking.microbench_rocprof --device 0 \
    --output results/rocprof_compare.json
```

RDNA exposes no WMMA instruction counters, so on an RDNA target the same command runs an RDNA mode instead. Per case (`matrix_fp16`, `matrix_bf16`, `vector_fp32`, `vector_fp16` and `read_bw`, or those of `--metrics`) it times a plain run as the protocol does, then runs `rocprofv3` with at most 2 counters per pass. Each value is the median of the last 5 dispatches of the case's kernel:

- **Shader clock**: `GRBM_GUI_ACTIVE`, else `SQ_BUSY_CYCLES` divided by its instance count, else the sysfs clock of the plain run. Instance counts are the product of the dimensions `rocprofv3 -L` lists (32 WGPs on gfx1201). A counter clock must be within 25 % of the plain run's sysfs clock, which catches a wrong instance count; the sclk DPM table bounds nothing, since gfx12 boards run past its top level.
- **Dual issue** (vector cases): `SQ_INSTS_DUAL_VALU_WAVE32` over `SQ_INSTS_VALU`, within 0 to 1.
- **Read bytes** (`read_bw`): the bytes of the `GL2C_EA_RDREQ_*B` requests the board lists, over the bytes read, within 0.5 to 2.

A counter that is not listed, reads 0, fails or gives an implausible value is rejected with its reason (`clock_rejected`, `<name>_rejected`), never reported. On gfx1201 `rocprofv3 -L` lists no `SQ_INSTS_DUAL_VALU_WAVE32` and every GL2C and TCP counter reads 0, so only the clock is checked; on gfx1100 `GRBM_GUI_ACTIVE` reads 0 and the clock comes from `SQ_BUSY_CYCLES`. `rocprofv3` gets the environment Python sees (`os.environ.copy()`): a process with HIP loaded adds `ROCPROFILER_REGISTER_LIBRARY` to its C environment, and a `rocprofv3` that inherits that exits at once.
