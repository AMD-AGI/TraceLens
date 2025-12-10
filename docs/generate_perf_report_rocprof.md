<!--
Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# Generate Performance Report from rocprofv3

This Python script (`TraceLens/Reporting/generate_perf_report_rocprof.py`) processes a rocprofv3 JSON profile trace and outputs an Excel workbook or CSVs with relevant performance information.

---

## 🚀 Quick Start

Run the script with a rocprofv3 results JSON to generate an Excel report:

```bash
TraceLens_generate_perf_report_rocprof --profile_json_path path/to/908_results.json
```

Or use the Python module directly:

```bash
python -m TraceLens.Reporting.generate_perf_report_rocprof --profile_json_path path/to/908_results.json
```

---

## 📋 Excel Workbook Sheets

| Sheet Name                      | Description                                                                                           |
|---------------------------------|-------------------------------------------------------------------------------------------------------|
| `gpu_timeline`                  | End-to-end GPU activity summary, including kernel execution, memory operations, and idle time.       |
| `kernel_summary`                | Summary of kernel execution time at the individual kernel level; each row is a unique kernel name.   |
| `kernel_summary_by_category`    | Summary of kernel time grouped by category (e.g., GEMM, Elementwise, Attention).                    |
| `kernel_details` (optional)     | Detailed kernel information including grid/block dimensions for each dispatch.                        |
| `short_kernels_summary` (opt)   | Summary of kernels with duration below the short-duration threshold.                                  |
| `short_kernel_histogram` (opt)  | Histogram showing the distribution of kernel durations below the short-duration threshold.            |

---

## 🎯 Command-Line Options

### Required Arguments

- `--profile_json_path`: Path to the rocprofv3 `*_results.json` file

### Output Options

- `--output_xlsx_path`: Path to the output Excel file (default: auto-generated from input filename)
- `--output_csvs_dir`: Directory to save output as CSV files instead of Excel

### Analysis Options

- `--disable_kernel_summary`: Disable kernel summary sheets (enabled by default)
- `--kernel_details`: Include detailed kernel information with grid/block dimensions
- `--short_kernel_study`: Include short kernel analysis in the report
- `--short_kernel_threshold_us`: Threshold in microseconds for "short" kernels (default: 10)
- `--short_kernel_histogram_bins`: Number of bins for short-kernel histogram (default: 100)
- `--topk_kernels`: Limit kernel details to top K kernels by time

---

## 💡 Examples

### Basic Report Generation

Generate a default Excel report:

```bash
TraceLens_generate_perf_report_rocprof \
    --profile_json_path run_20251209_070939/smci350-zts-gtu-c6-25/908_results.json
```

This creates `908_perf_report.xlsx` in the same directory.

### Short Kernel Analysis

Include analysis of short-duration kernels:

```bash
TraceLens_generate_perf_report_rocprof \
    --profile_json_path 908_results.json \
    --short_kernel_study \
    --short_kernel_threshold_us 20
```

### Detailed Kernel Information

Include grid/block dimensions for top 100 kernels:

```bash
TraceLens_generate_perf_report_rocprof \
    --profile_json_path 908_results.json \
    --kernel_details \
    --topk_kernels 100
```

### Generate CSV Files

Output as CSV files instead of Excel:

```bash
TraceLens_generate_perf_report_rocprof \
    --profile_json_path 908_results.json \
    --output_csvs_dir ./rocprof_analysis
```

### Custom Output Path

Specify a custom output filename:

```bash
TraceLens_generate_perf_report_rocprof \
    --profile_json_path 908_results.json \
    --output_xlsx_path my_custom_report.xlsx \
    --kernel_details \
    --short_kernel_study
```

---

## 📊 Understanding the Output

### GPU Timeline

Shows the breakdown of GPU activity:
- **total_time**: Total profiling duration
- **kernel**: Time spent executing kernels
- **memory**: Time spent on memory operations
- **idle**: Time when GPU was idle

### Kernel Summary

Aggregated statistics for each unique kernel:
- **Count**: Number of times the kernel was dispatched
- **Total Kernel Time (µs)**: Sum of all dispatch durations
- **Mean/Median/Std/Min/Max**: Statistical distribution of durations
- **Percentage (%)**: Percentage of total kernel time
- **Cumulative Percentage (%)**: Running total percentage

### Kernel Categories

Kernels are automatically categorized based on name patterns:
- **GEMM**: Matrix multiplication kernels
- **Elementwise**: Element-wise operations
- **Reduction**: Reduction operations
- **Convolution**: Convolution kernels
- **Normalization**: Batch norm, layer norm, etc.
- **Attention**: Flash attention and related kernels
- **Memory**: Memory copy operations
- **Other**: Uncategorized kernels

---

## 🔍 rocprofv3 Format Details

rocprofv3 uses the `rocprofiler-sdk-tool` format, which is different from PyTorch's Chrome Trace Event format:

### Input File Structure

```json
{
  "rocprofiler-sdk-tool": [{
    "metadata": {"pid": ..., "init_time": ..., "fini_time": ...},
    "agents": [...],
    "kernel_symbols": [...],
    "buffer_records": {
      "kernel_dispatch": [...],
      "memory_copy": [...],
      "hip_api": [...],
      ...
    }
  }]
}
```

### Key Data Sources

- **kernel_dispatch**: Kernel execution records with timestamps and dispatch info
- **kernel_symbols**: Kernel names and metadata indexed by `kernel_id`
- **memory_copy**: Memory transfer operations
- **hip_api**, **hsa_api**: API call traces (when available)

---

## 🆚 Comparison with PyTorch Profiler

| Feature | rocprofv3 | PyTorch Profiler |
|---------|-----------|------------------|
| Format | rocprofiler-sdk JSON | Chrome Trace Event |
| Kernel Names | Direct from ROCm | Via PyTorch ops |
| Grid/Block Dims | ✅ Always available | ✅ Available |
| CPU Operations | ❌ Limited | ✅ Full trace |
| Memory Ops | ✅ Basic support | ✅ Full support |
| API Calls | ✅ HIP/HSA | ✅ CUDA runtime |
| Tree View | ❌ N/A | ✅ CPU call stack |

---

## ⚠️ Limitations

1. **No CPU Call Stack**: rocprofv3 focuses on GPU operations; CPU-side profiling is limited
2. **No Operator Hierarchy**: Unlike PyTorch profiler, there's no tree of CPU operations
3. **Basic Categorization**: Kernel categories are inferred from names, not semantic information
4. **Limited Metadata**: Some PyTorch-specific metadata (input shapes, args) is not available

---

## 🐛 Troubleshooting

### "Not a valid rocprofv3 file"

Ensure your file is a `*_results.json` file from rocprofv3, not a different format.

### "No kernel events found"

The trace may not have captured any GPU activity. Check that:
- GPU operations were actually executed during profiling
- rocprofv3 was configured to capture kernel dispatches

### openpyxl not installed

Install openpyxl for Excel output:

```bash
pip install openpyxl
```

Or use CSV output instead:

```bash
TraceLens_generate_perf_report_rocprof \
    --profile_json_path trace.json \
    --output_csvs_dir ./output
```

---

## 📚 See Also

- [PyTorch Performance Report](generate_perf_report.md)
- [JAX Performance Report](generate_perf_report_jax.md)
- [ROCm rocprofiler-sdk Documentation](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/)

