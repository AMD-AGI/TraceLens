<!--
Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# Generate Performance Report

This Python script (`TraceLens/Reporting/generate_perf_report_pytorch.py`) processes a PyTorch JSON profile trace and outputs an Excel workbook or CSVs with relevant information.

Similarly (`TraceLens/Reporting/generate_perf_report_jax.py`) processes JAX XPLANE.PB profile trace.

---

## 🚀 Quick Start

Run the script with a profile JSON to generate an Excel report:

```bash
python generate_perf_report_pytorch.py --profile_json_path path/to/profile.json 
```

Alternatively you can directly call the entry point with the same command line args,
 without the need to copy the script out of the repo. 

```bash
TraceLens_generate_perf_report_pytorch --profile_json_path path/to/profile.json 
```

Similarly for JAX profile trace:

```bash
python generate_perf_report_jax.py --profile_path path/to/profile.xplane.pb 
```

---

## 📋 Excel Workbook Sheets

| Sheet Name                  | Description                                                                                                      |
|----------------------------|------------------------------------------------------------------------------------------------------------------|
| `gpu_timeline`             | End-to-end GPU activity summary, including compute, memory copies, communication, and idle time.                 |
| `ops_summary_by_category`  | Summary of compute time grouped by operation category (e.g., GEMM, SDPA_fwd, elementwise).                       |
| `ops_summary`              | Summary of compute time at the individual operation level; each row corresponds to a unique operation name.      |
| `ops_unique_args`          | Detailed operation-level summary; each row corresponds to a unique (operation name, argument) combination.       |
| `unified_perf_summary`     | Unified perf metrics for all ops with perf models OR leaf ops that launch GPU kernels. Includes GFLOPS, TFLOPS/s, Data Moved, FLOPS/Byte, TB/s metrics aggregated by unique args. |
| `short_kernels_histogram`  | Histogram showing the distribution of kernel durations below the short-duration threshold.                   |
| `short_kernels_all_details`| Detailed list of short-duration kernels, including count, total/mean time, runtime percentage, and parent op.   |
| Roofline Sheets            | Roofline analysis for each operation category (GEMM, CONV, etc.), including TFLOPs, TB/s, and FLOPs/byte metrics. |

Note: JAX outputs do not include `short_kernels_histogram` or `short_kernels_all_details`, these are for PyTorch only.

### 🎯 GPU Architecture for Roofline Analysis

When `--gpu_arch_json_path` is provided, the report includes additional roofline metrics:

- **Compute Spec**: Combined compute type and precision (e.g., `matrix_bf16`, `vector_fp32`)
- **Roofline Time (µs)**: Theoretical minimum time based on GPU peak capabilities
- **Pct Roofline**: Percentage of roofline achieved (higher is better)

The GPU arch file specifies Max Achievable FLOPS (MAF) for different compute types and precisions. See [GPU Architecture Example](../examples/gpu_arch_example.md) for format details and [AMD MAF measurements](https://rocm.blogs.amd.com/software-tools-optimization/measuring-max-achievable-flops-part2/README.html#amd-maf-results) for reference values.

📖 **For detailed column definitions and usage guide**, see [Performance Report Column Definitions](perf_report_columns.md).

---

## ⚙️ Optional Arguments  

The script supports several optional arguments to customize the output report. By default, it generates an Excel file (`.xlsx`) in the trace directory. If `--output_csvs_dir` is specified, individual CSV files are written instead.

| Argument                          | Default           | Description                                                                 |
|-----------------------------------|-------------------|-----------------------------------------------------------------------------|
| `--gpu_arch_json_path PATH`       | `None`            | Path to GPU architecture JSON file for roofline analysis. See [GPU Architecture Example](../examples/gpu_arch_example.md) for format details. |
| `--topk_ops N`                    | `None`            | Limit the number of rows in the unique-args launcher table.               |
| `--topk_short_kernels N`          | `None`            | Limit the number of rows in the short-kernel table.                          |
| `--topk_roofline_ops N`           | `None`            | Limit the number of rows in the roofline sheet.                             |
| `--extension_file`                | `None`            | Path to extension python file |
| `--include_unlinked_kernels`      | `False`           | Include all kernels in the gpu timeline analysis -  including kernels not linked to host call stack. By default these unlinked kernels are excluded in the analysis.|
| `--micro_idle_thresh_us X`        | `None`            | Threshold (in microseconds) to classify idle intervals as micro idle in GPU timeline analysis. If None, all idle times are included in one category. |
| `--short_kernel_study`            | `False`           | Include short-kernel analysis in the report.                                 |
| `--short_kernel_threshold_us X`   | `10`              | Threshold (in microseconds) to classify a kernel as "short".             |
| `--short_kernel_histogram_bins B` | `100`             | Number of bins to use for the short-kernel duration histogram.              |
| `--detect_recompute`              | `False`           | Detect activation recomputation (checkpointing) and add an `is_recompute` column to `ops_summary`, `ops_unique_args`, and `unified_perf_summary`. See [Activation Recompute Detection](#-activation-recompute-detection) below. |
| `--output_xlsx_path PATH`         | `<auto-inferred>` | Path to save the Excel report. Auto-inferred if not provided.              |
| `--output_csvs_dir DIR`           | `None`            | If set, saves each sheet as a CSV file in the specified directory.         |

Note: currently JAX supports only two optional arguments `--output_xlsx_path` and `--output_csvs_dir`.

### 📦 Output Behavior

- If `--output_csvs_dir` is set, all output sheets are saved as individual CSV files in that directory.
- Otherwise, the script saves a single Excel file:
  - If `--output_xlsx_path` is not provided, it is inferred from the input JSON trace name (e.g., `profile.json` → `profile_perf_report.xlsx`).
- The `openpyxl` package is required only for the case when we write Excel files; it will be auto-installed if missing.

#### 🧪 Example Usage to write CSVs


```bash
python generate_perf_report_pytorch.py \
  --profile_json_path traces/profile.json \
  --output_csvs_dir output_csvs/ \
  --topk_ops 50 \
```

## 🔍 Activation Recompute Detection

When training with activation checkpointing (`torch.utils.checkpoint`), some forward-pass ops are recomputed during the backward pass to save memory. The `--detect_recompute` flag identifies these ops and adds an `is_recompute` column to the report, so you can see exactly how much GPU time and compute is spent on recomputation.

### How it works

TraceLens walks the CPU call-stack tree and finds `python_function` events from `torch/utils/checkpoint.py` corresponding to `recompute_fn`. All ops in those subtrees are marked `is_recompute=True`. This requires `python_function` events in the trace, which TraceLens enables automatically when this flag is set.

### Usage

```bash
TraceLens_generate_perf_report_pytorch \
  --profile_json_path path/to/trace.json \
  --detect_recompute
```

### What changes in the report

The following sheets gain an `is_recompute` column that splits rows into recompute vs non-recompute:

| Sheet | Effect |
|-------|--------|
| `ops_summary` | Same op name appears in separate rows for `is_recompute=True` and `False` |
| `ops_unique_args` | Unique (op, shape, is_recompute) combinations, each with their own time/count |
| `unified_perf_summary` | Full perf metrics split by recompute status |

This makes it straightforward to answer questions like:
- What percentage of GPU time is recomputation?
- Which layers are being recomputed and at what cost?
- Is the recompute overhead acceptable given the memory savings?

### Python API

```python
from TraceLens.TreePerf import TreePerfAnalyzer

analyzer = TreePerfAnalyzer.from_file("trace.json", detect_recompute=True)
df = analyzer.get_df_kernel_launchers(include_kernel_details=True)
print(df["is_recompute"].value_counts())
```

> **Note**: When `--detect_recompute` is not set (the default), there is zero overhead — no extra columns, no tree changes, and no `python_function` parsing.

---

## 🧩 Extensions: Custom Hooks for Tree and PerfModel

The `--extension_file` argument allows users to inject custom logic into the performance report generation pipeline. This is useful for experimenting with:

- Tree post-processing (e.g., injecting pseudo ops)
- Custom performance models for new op types
- Additional operation category definitions

### 🔧 How to Use

Pass a Python file path via `--extension_file`. The file can define one or more of the following optional symbols:

| Symbol Name                  | Type      | Description                                                                 |
|-----------------------------|-----------|-----------------------------------------------------------------------------|
| `tree_postprocess_extension`| `Callable`| Called with `perf_analyzer.tree`. Use to modify the tree structure post-construction. |
| `perf_model_extension`      | `dict`    | A mapping from op name (str) to a custom performance model class. These will override or extend existing models. |
| `dict_cat2names_extension`  | `dict`    | Mapping from new category names to lists of op names, merged into the built-in op categories. |

#### 📄 Example Extension File for MegatronLM in the examples dir

### ✅ Example Usage

```bash
python generate_perf_report_pytorch.py \
  --profile_json_path traces/profile.json \
  --extension_file my_extension.py
```