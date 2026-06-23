<!--
Copyright (c) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.
-->

# Compatibility matrix

This page lists the hardware and software configurations for TraceLens. Only
configurations that have been verified and tested should appear in the
validated tables below.

```{note}
TraceLens analyzes trace files and does not execute GPU kernels itself, so its
core report generation runs on any host with a supported Python version. GPU
and ROCm requirements apply to the *profiling tools* that produce the traces
(for example, `rocprofv3`) and to the roofline specifications used for a given
accelerator.
```

## Software requirements

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
| `backports.strenum`, `StrEnum` | `StrEnum` backport for Python < 3.11. |
| `office365-rest-python-client`, `msal` | Optional SharePoint/365 integrations. |
| `traceconv` | Optional; required only for `.pftrace` input. Resolved from `PATH` or downloaded automatically if not provided with `--traceconv`. |

The optional `comparative` extra (`pip install .[comparative]`) pulls in
`slodels` for LLM-assisted comparative analysis and requires a custom package
index.

## Supported trace formats

| Format | Producing tool | Validated |
|--------|----------------|-----------|
| PyTorch Chrome trace (`.json`, `.json.gz`, `.zip`) | `torch.profiler` | Yes |
| JAX XPlane protobuf (`.pb`) | JAX profiler / `xprof` | Yes |
| rocprofv3 JSON (`*_results.json`) | AMD ROCm rocprofiler-sdk | Yes |
| rocprofv3 pftrace / Perfetto-style | `rocprofv3 --output-format pftrace` | Yes |

## Accelerators (roofline analysis)

Roofline classification requires a GPU architecture specification that provides
the peak FLOPS and peak bandwidth for the target accelerator (for example,
`mi300.json`). Architecture specifications are bundled under
`TraceLens/Agent/Analysis/utils/arch/`; supply your own JSON with
`--gpu_arch_json_path` for accelerators not listed.

| Accelerator | Notes |
|-------------|-------|
| AMD Instinct MI300X | Used for the roofline knee point (for example, FP16 ≈ 1300 TFLOPS / 5.3 TB/s ≈ 245 FLOPs/byte). |

```{note}
Add validated accelerator and ROCm combinations to these tables as they are
tested for each release.
```

## ROCm compatibility

`rocprofv3`-based capture (JSON and pftrace) requires a ROCm installation that
provides `rocprofv3` and, for `.pftrace` conversion, `traceconv`. Record the
specific ROCm versions validated for each TraceLens release here.

| TraceLens version | Validated ROCm versions |
|-------------------|-------------------------|
| 0.1.0 | To be confirmed per release |
