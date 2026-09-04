<!--
Copyright (c) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# Estimate simulated kernel times with Origami in TraceLens
```{meta}
:description: Learn how to enable Origami in TraceLens to estimate simulated GEMM and SDPA kernel times from a GPU architecture description.
:keywords: TraceLens, Origami, rocm-origami, GEMM, SDPA, performance model, simulation, gpu_arch_json_path, ROCm, roofline
```

TraceLens can estimate simulated GEMM and SDPA kernel times using *Origami*
(ROCm's performance modeling library) when you provide a GPU architecture
description and explicitly opt in. This is optional: `pip install TraceLens`
doesn't install Origami.

## What Origami does in TraceLens

Origami integrates into TraceLens in the following ways.

- GEMM and SDPA perf models call Origami's Python bindings to predict a duration in microseconds for forward (and SDPA backward where applicable).
- Results show up in perf reports under columns such as `Origami Time (µs)`, `Origami TFLOPS/s`, `Origami TB/s`, and `Pct Origami` (relative to measured kernel busy time), when simulation uses Origami.
- Roofline metrics from `--gpu_arch_json_path` are separate; they don't require Origami. Origami adds *simulated* timing on top when enabled.

## Installation

Origami consists of a Python package and requires a compatible ROCm system environment; install both components before enabling Origami in TraceLens.

### Python package

Install the published package (PyPI name `rocm-origami`):

```bash
pip install rocm-origami
```

Note that installing Origami requires that ROCm is
installed on the system, because Origami can use the HIP library to detect the
GPU currently installed in the system and use it as an architectural model.
TraceLens doesn't currently use this functionality, but it can't be
removed from Origami.

### System environment

Origami's wheels and bindings expect a ROCm runtime on the machine (a GPU isn't
required for pure Python simulation in many cases, but library loading may
depend on your setup). The project's continuous integration (CI) uses an AMD
ROCm container and installs `rocm-origami` alongside TraceLens (see
[`.github/workflows/unit-tests.yml`](https://github.com/AMD-AGI/TraceLens/blob/main/.github/workflows/unit-tests.yml)).

If `import origami` fails after `pip install`, check:

- Python version and wheel compatibility for `rocm-origami`.
- `LD_LIBRARY_PATH` and ROCm install paths required by the Origami wheel you use.

## When TraceLens uses Origami

Simulation is implemented in `TraceLens/PerfModel/perf_model.py`
(`GEMM.get_simulation_time_func`):

- If `enable_origami` is true and the perf model has the needed architecture and parameters, TraceLens uses Origami.
- If `enable_origami` is false, TraceLens doesn't call Origami; simulated times are omitted for that path.

So you need both:

- A valid GPU architecture (see below), and
- `enable_origami=True` (CLI flag or Python API).

## GPU architecture JSON

Pass the same JSON file you use for roofline analysis with
`--gpu_arch_json_path`. It must include fields Origami expects (for example GPU
name, frequency, memory bandwidth, and CU count), as consumed by
`OrigamiHelper.get_hardware` in `TraceLens/PerfModel/origami_helper.py`.

See [Generate a PyTorch performance report](./generate-perf-report-pytorch.md)
for the general format of the architecture JSON and its use in roofline
analysis.

## Command-line usage

Pass `--enable-origami` alongside `--gpu_arch_json_path` to any TraceLens report command to activate Origami simulation.

### PyTorch perf report

Run the following command to generate a PyTorch performance report with Origami simulation enabled.

```bash
TraceLens_generate_perf_report_pytorch \
  --profile_json_path path/to/profile.json.gz \
  --gpu_arch_json_path path/to/gpu_arch.json \
  --enable-origami \
  --output_csvs_dir ./out_csvs
```

Or:

```bash
python -m TraceLens.Reporting.generate_perf_report_pytorch \
  --profile_json_path path/to/profile.json.gz \
  --gpu_arch_json_path path/to/gpu_arch.json \
  --enable-origami \
  --output_csvs_dir ./out_csvs
```

### vLLM-oriented PyTorch report

Same pattern; the entry point mirrors the PyTorch script (`--enable-origami`).

### JAX perf report

Run the following command to generate a JAX performance report with Origami simulation enabled.

```bash
TraceLens_generate_perf_report_jax \
  --profile_path path/to/trace.xplane.pb \
  --gpu_arch_json_path path/to/gpu_arch.json \
  --enable-origami \
  --output_csvs_dir ./out_csvs
```

### Standalone GEMM/SDPA simulator helper

Use `TraceLens/PerfModel/run_perf_model.py` to run the GEMM or SDPA simulator directly from the command line.

```bash
python -m TraceLens.PerfModel.run_perf_model --op gemm ... --enable_origami
```

## Python API

When building a `TreePerfAnalyzer` (or `JaxTreePerfAnalyzer`) in code, pass:

```python
TreePerfAnalyzer.from_file(
    profile_filepath="profile.json.gz",
    arch=gpu_arch_dict,          # or load JSON
    enable_origami=True,
)
```

Reporting helpers such as `generate_perf_report_pytorch` accept
`enable_origami=True` and forward it to the analyzer.

## Troubleshooting

Use the following table to diagnose common problems when enabling Origami in TraceLens.

| Symptom | What to check |
|---------|---------------|
| No Origami columns in CSVs | Confirm `--enable-origami` (or API `enable_origami=True`) and `--gpu_arch_json_path`. |
| Message on stderr about `origami` import | Install `rocm-origami` and fix ROCm/library paths. |
| Unsupported dtype warning | The Origami path supports a fixed set of dtypes (for example fp16, bf16, fp32, fp64, fp8); others skip simulation. |

## Related topics

- [Generate a PyTorch performance report](./generate-perf-report-pytorch.md)
- [Generate a JAX performance report](./generate-perf-report-jax.md)
- [Generate a PyTorch inference performance report](./generate-perf-report-pytorch-inference.md)
- [API reference](../reference/api-reference.md)
