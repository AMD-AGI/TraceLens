<!--
Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# Origami with TraceLens

TraceLens can estimate **simulated GEMM and SDPA kernel times** using **Origami** (ROCm’s performance modeling library) when you provide a GPU architecture description and explicitly opt in. This is **optional**: `pip install TraceLens` does not install Origami.

---

## What Origami does in TraceLens

- **GEMM** and **SDPA** perf models call Origami’s Python bindings to predict a duration in microseconds for forward (and SDPA backward where applicable).
- Results show up in perf reports under columns such as **Origami Time (µs)**, **Origami TFLOPS/s**, **Origami TB/s**, and **Pct Origami** (relative to measured kernel busy time), when simulation uses Origami (see [Performance Report Column Definitions](perf_report_columns.md)).
- **Roofline** metrics from `--gpu_arch_json_path` are separate; they do not require Origami. Origami adds *simulated* timing on top when enabled.

---

## Installation

### Python package

Install the published package (PyPI name **`rocm-origami`**):

```bash
pip install rocm-origami
```

Note that installing Origami currently (as of April 2026) requries that ROCm is installed on the system because Origami can use the HIP library to detect the GPU currently installed in the system and use it as an architectural model. TraceLens does not currently use this functionality but it currently cannot be removed from Origami.

### System environment

Origami’s wheels/bindings expect a **ROCm** runtime on the machine (GPU not required for pure Python simulation in many cases, but library loading may depend on your setup). The project’s CI uses an AMD ROCm container and installs `rocm-origami` alongside TraceLens (see `.github/workflows/regression-tests.yml`).

If `import origami` fails after `pip install`, check:

- Python version and wheel compatibility for `rocm-origami`.
- `LD_LIBRARY_PATH` / ROCm install paths required by the Origami wheel you use.

---

## When TraceLens uses Origami vs other backends

Simulation is implemented in `TraceLens/PerfModel/perf_model.py` (`GEMM.get_simulation_time_func`):

1. **Otherwise**, if **`enable_origami` is true** and the perf model has the needed architecture and parameters, TraceLens uses **Origami**.

2. If **`enable_origami` is false**, TraceLens does **not** call Origami; simulated times are omitted for that path.

So you need **both**:

- A valid **GPU architecture** (see below), and  
- **`enable_origami=True`** (CLI flag or Python API).

---

## GPU architecture JSON

Pass the same JSON file you use for roofline analysis with **`--gpu_arch_json_path`**. It must include fields Origami expects (for example GPU name, frequency, memory bandwidth, CU count), as consumed by `OrigamiHelper.get_hardware` in `TraceLens/PerfModel/origami_helper.py`.

See [GPU Architecture Example](../examples/gpu_arch_example.md) and [Generate Performance Report](generate_perf_report.md) for the general format.

---

## Command-line usage

### PyTorch perf report

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

```bash
TraceLens_generate_perf_report_jax \
  --profile_path path/to/trace.xplane.pb \
  --gpu_arch_json_path path/to/gpu_arch.json \
  --enable-origami \
  --output_csvs_dir ./out_csvs
```

### Standalone GEMM/SDPA simulator helper

`TraceLens/PerfModel/run_perf_model.py`:

```bash
python -m TraceLens.PerfModel.run_perf_model --op gemm ... --enable_origami
```

---

## Python API

When building a **`TreePerfAnalyzer`** (or **`JaxTreePerfAnalyzer`**) in code, pass:

```python
TreePerfAnalyzer.from_file(
    profile_filepath="profile.json.gz",
    arch=gpu_arch_dict,          # or load JSON
    enable_origami=True,
)
```

Reporting helpers such as **`generate_perf_report_pytorch`** accept **`enable_origami=True`** and forward it to the analyzer.

---

## Troubleshooting

| Symptom | What to check |
|--------|----------------|
| No Origami columns in CSVs | Confirm **`--enable-origami`** (or API **`enable_origami=True`**) **and** **`--gpu_arch_json_path`**. |
| Message on stderr about `origami` import | Install **`rocm-origami`** and fix ROCm/library paths. |
| Unsupported dtype warning | Origami path supports a fixed set of dtypes (e.g. fp16, bf16, fp32, fp64, fp8); others skip simulation. |

---

## Related documentation

- [Generate Performance Report](generate_perf_report.md) — CLI overview and roofline (`--gpu_arch_json_path`).
- [Performance Report Column Definitions](perf_report_columns.md) — sheet and column meanings.
