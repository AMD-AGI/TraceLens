<!--
Copyright (c) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->


# TraceLens compatibility matrix
```{meta}
:description: TraceLens compatibility matrix: supported Python versions, operating systems, trace formats, and validated GPU accelerators for roofline analysis.
:keywords: TraceLens, compatibility, Python, ROCm, AMD Instinct, MI300X, PyTorch profiler, JAX, rocprofv3, pftrace, GPU, Linux
```

This topic lists the hardware and software configurations for TraceLens. Only
configurations that have been verified and tested should appear in the
validated tables below.

## Hardware requirements (roofline analysis)

Roofline classification requires a GPU architecture specification that provides
the peak FLOPS and peak bandwidth for the target accelerator (for example,
`mi300.json`). Architecture specifications are bundled under
`TraceLens/Agent/Analysis/utils/arch/`; supply your own JSON with
`--gpu_arch_json_path` for accelerators not listed.

| AMD Instinct GPU | Notes |
|-------------|-------|
| MI300X | Used for the roofline knee point (for example, FP16 ≈ 1300 TFLOPS / 5.3 TB/s ≈ 245 FLOPs/byte). |

## ROCm and CUDA compatibility

TraceLens analyzes trace files rather than talking to the GPU or runtime
directly, so it isn't tied to a specific ROCm version. `rocprofv3`-based capture
(JSON and pftrace) only requires a ROCm installation that provides `rocprofv3`
and, for `.pftrace` conversion, `traceconv`; any ROCm version that emits
well-formed traces should work. If a particular ROCm build produces malformed
traces (for example, missing required fields), TraceLens reports the problem and
stops rather than silently producing misleading results.

The PyTorch report workflow works on any `torch.profiler` trace regardless of
the GPU backend, and is well tested on both AMD ROCm and NVIDIA CUDA traces.

## Software requirements

TraceLens requires the following software.

| Component | Requirement | Notes |
|-----------|-------------|-------|
| Python | 3.6 or later (3.10 recommended) | `python_requires>=3.6`; the 3.10 toolchain is used for documentation and CI builds. |
| Operating system | Linux (OS-independent core) | Report generation is not OS-specific; ROCm-based trace capture requires Linux. |
| TraceLens package | 0.1.0 | Installed from [github.com/AMD-AGI/TraceLens](https://github.com/AMD-AGI/TraceLens). |

### Python dependencies

These are installed automatically with the package:

| Dependency | Purpose |
|------------|---------|
| `pandas` | Tabular analysis and report data frames. |
| `openpyxl` | Excel (`.xlsx`) report output. |
| `tqdm` | Progress bars for long-running analyses. |
| `orjson` | Fast JSON parsing of large traces. |
| `tabulate` | Text/Markdown table rendering. |
| `matplotlib` | Roofline and other plots. |
| `xprof==2.20.1` | JAX XPlane parsing (HLO sidecar generation; supports JAX 0.8+). |
| `protobuf>=6.31.1,<7.0.0` | Required by `xprof`'s `grpcio-status` dependency. |
| `backports.strenum` | `StrEnum` backport for Python < 3.11. |
| `office365-rest-python-client`, `msal` | Optional SharePoint/365 integrations. |
| `traceconv` | Optional; required only for `.pftrace` input. Resolved from `PATH` or downloaded automatically if not provided with `--traceconv`. |

## Supported trace formats

TraceLens supports the following trace formats.

| Format | Producing tool |
|--------|----------------|
| PyTorch Chrome trace (`.json`, `.json.gz`, `.zip`) | `torch.profiler` |
| JAX XPlane protobuf (`.pb`) | JAX profiler / `xprof` | 
| rocprofv3 JSON (`*_results.json`) | AMD ROCm rocprofiler-sdk | 
| rocprofv3 pftrace / Perfetto-style | `rocprofv3 --output-format pftrace` | 


