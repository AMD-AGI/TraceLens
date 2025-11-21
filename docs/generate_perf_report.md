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
| `ops_all`                  | Detailed operation-level summary; each row corresponds to a unique (operation name, argument) combination.       |
| `ops_summary_prefill`      | **[Inference Phase]** Summary of Prefill phase operations (generated when `--inference_phase_analysis` enabled).  |
| `ops_summary_decode`       | **[Inference Phase]** Summary of Decode phase operations (generated when `--inference_phase_analysis` enabled).   |
| `kernel_summary_prefill`   | **[Inference Phase]** Detailed Prefill phase kernel breakdown (when `--enable_kernel_summary` and `--inference_phase_analysis` enabled). |
| `kernel_summary_decode`    | **[Inference Phase]** Detailed Decode phase kernel breakdown (when `--enable_kernel_summary` and `--inference_phase_analysis` enabled). |
| `short_kernels_histogram`  | Histogram showing the distribution of kernel durations below the short-duration threshold.                   |
| `short_kernels_all_details`| Detailed list of short-duration kernels, including count, total/mean time, runtime percentage, and parent op.   |
| Roofline Sheets            | Roofline analysis for each operation category, including TFLOPs, TB/s, and FLOPs/byte metrics.                |

Note: JAX outputs do not include `short_kernels_histogram` or `short_kernels_all_details`, these are for PyTorch only.

---

## ⚙️ Optional Arguments  

The script supports several optional arguments to customize the output report. By default, it generates an Excel file (`.xlsx`) in the trace directory. If `--output_csvs_dir` is specified, individual CSV files are written instead.

| Argument                          | Default           | Description                                                                 |
|-----------------------------------|-------------------|-----------------------------------------------------------------------------|
| `--topk_ops N`                    | `None`            | Limit the number of rows in the unique-args launcher table.               |
| `--topk_short_kernels N`          | `None`            | Limit the number of rows in the short-kernel table.                          |
| `--topk_roofline_ops N`           | `None`            | Limit the number of rows in the roofline sheet.                             |
| `--extension_file`                | `None`            | Path to extension python file |
| `--include_unlinked_kernels`      | `False`           | Include all kernels in the gpu timeline analysis -  including kernels not linked to host call stack. By default these unlinked kernels are excluded in the analysis.|
| `--micro_idle_thresh_us X`        | `None`            | Threshold (in microseconds) to classify idle intervals as micro idle in GPU timeline analysis. If None, all idle times are included in one category. |
| `--short_kernel_study`            | `False`           | Include short-kernel analysis in the report.                                 |
| `--short_kernel_threshold_us X`   | `10`              | Threshold (in microseconds) to classify a kernel as "short".             |
| `--short_kernel_histogram_bins B` | `100`             | Number of bins to use for the short-kernel duration histogram.              |
| `--inference_phase_analysis`      | `False`           | Enable inference phase analysis to categorize operations into Prefill and Decode phases. |
| `--decode_threshold N`            | `10`              | Threshold for GEMM M parameter to distinguish Prefill vs Decode (M > threshold = Prefill). |
| `--phase_detection_method`        | `hybrid` | Method for phase detection: `hybrid`, `kernel_names`, `operation_frequency`, `framework_apis`, `attention_patterns`, `gemm_params`. |
| `--output_xlsx_path PATH`         | `<auto-inferred>` | Path to save the Excel report. Auto-inferred if not provided.              |
| `--output_csvs_dir DIR`           | `None`            | If set, saves each sheet as a CSV file in the specified directory.         |

Note: currently JAX supports only two optional arguments `--output_xlsx_path` and `--output_csvs_dir`.

### � Inference Phase Analysis (New Feature)

The `--inference_phase_analysis` flag enables automatic detection and separation of **Prefill** and **Decode** phases in LLM inference workloads. This is particularly useful for analyzing traces from inference engines like vLLM, SGLang, TGI, and TensorRT-LLM.

#### How It Works

**Phase Detection Methods** (select with `--phase_detection_method`):

1. **`kernel_names` (RECOMMENDED)**: Reads phase information directly from kernel names
   - **Most Reliable**: Detects explicit prefill/decode keywords in kernel names
   - **Sequence Length Encoding**: Recognizes `sl1` (decode), `sl512` (prefill), etc.
   - **Universal**: Works across all inference frameworks and model architectures

2. **`operation_frequency`**: Statistical analysis of operation repetition
   - **Prefill**: Operations called few times (once per layer)
   - **Decode**: Operations called many times (once per token)
   - **Framework-Agnostic**: Works without understanding kernel internals

3. **`attention_patterns`**: Analyzes attention operation characteristics
   - **Prefill**: Large sequence lengths, `prefill_attention` operations
   - **Decode**: KV cache operations (`paged_attention`, `reshape_and_cache`)

4. **`framework_apis`**: Uses framework-specific operation patterns
   - **vLLM**: `paged_attention_*`, `reshape_and_cache_*` 
   - **SGLang**: `radix_attention_*`, `prefix_cache_*`
   - **TGI**: `continuous_batch_*`, `decode_kernel_*`

5. **`gemm_params`**: Original GEMM M parameter analysis
   - **Prefill**: `M > decode_threshold` (typically sequence length > 1)
   - **Decode**: `M ≤ decode_threshold` (typically sequence length = 1)

6. **`hybrid`**: Combines multiple methods in priority order (kernel_names → framework_apis → attention_patterns → frequency)

#### Generated Sheets

When enabled, the following additional sheets are created:
- `ops_summary_prefill` - Operations summary for Prefill phase
- `ops_summary_decode` - Operations summary for Decode phase
- `kernel_summary_prefill` - Detailed kernel breakdown for Prefill phase (requires `--enable_kernel_summary`)
- `kernel_summary_decode` - Detailed kernel breakdown for Decode phase (requires `--enable_kernel_summary`)

#### Example Usage

```bash
# Basic usage (uses hybrid method by default - RECOMMENDED)
python generate_perf_report_pytorch.py \
  --profile_json_path vllm_trace.json \
  --inference_phase_analysis

# With detailed kernel analysis
python generate_perf_report_pytorch.py \
  --profile_json_path sglang_trace.json \
  --inference_phase_analysis \
  --enable_kernel_summary

# Try different detection methods
python generate_perf_report_pytorch.py \
  --profile_json_path trace.json \
  --inference_phase_analysis \
  --phase_detection_method kernel_names

# Statistical frequency-based (fast, no kernel name parsing)
python generate_perf_report_pytorch.py \
  --profile_json_path trace.json \
  --inference_phase_analysis \
  --phase_detection_method operation_frequency
```

#### Tuning the Threshold

The `--decode_threshold` parameter (default: 10) can be adjusted based on your workload:
- **Smaller models** or **shorter sequences**: Use lower threshold (e.g., 5)
- **Larger models** or **longer sequences**: Use higher threshold (e.g., 50)
- **Mixed workloads**: The default value of 10 works well for most LLM inference patterns

### �📦 Output Behavior

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