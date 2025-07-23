# Customized Performance Report Generator

This customized Python script (`generate_perf_report_with_fusion_and_shortkernels.py`) processes a PyTorch JSON profile trace and outputs an Excel workbook with detailed tables. 

---

## 🚀 Quick Start

Run the script with a profile JSON to generate an Excel report:

```bash
python generate_perf_report_with_fusion_and_shortkernels.py \
  --profile_json_path path/to/profile.json \
  --output_xlsx_path output_report.xlsx
```

---

## 📋 Excel Workbook Sheets

| Sheet Name                   | Description                                                                                                     |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------- |
| `gpu_timeline`               | End-to-end GPU activity summary (compute, memcpy, communication, idle times).                                   |
| `ops_summary`                | Aggregate metrics per operation name(counts, total duration, mean, std, min, max).                                  |
| `ops_all` (or `ops_topk<N>`) | Same data as above but split by unique argument signature.                                                   |
| `short_kernels_histogram`    | Histogram showing the distribution of kernels below the short-duration threshold.                               |
| `short_kernels_all_details`  | Detailed list of short-duration kernels (counts, total, mean, percentage of runtime, launching host operation). |
| `fusion_candidates_details`  | Potential operation fusion pairs with estimated time savings (e.g., BatchNorm → ReLU).                          |
| Roofline Sheets              | Roofline analysis per op family:  TFLOPs, TB/s, FLOPs/byte metrics.                                    |

---

## ⚙️ Optional Arguments

Customize the output by adding these optional arguments:

| Argument                          | Default                                            |
| --------------------------------- | -------------------------------------------------- |
| `--topk_ops N`                    | `None` (all rows)                                  |
| `--short_kernel_threshold_us X`   | `10` µs                                            |
| `--short_kernel_histogram_bins B` | `100`                                              |
| `--topk_short_kernels N`          | `None` (all rows)                                  |
| `--fusion_op_names op1,op2,...`   | `aten::miopen_convolution,aten::miopen_batch_norm` |
| `--topk_roofline_ops N`           | `None` (all rows)                                  |

Example with optional flags:

```bash
python generate_perf_report_with_fusion_and_shortkernels.py \
  --profile_json_path path/to/profile.json \
  --output_xlsx_path output_report.xlsx \
  --topk_ops 100 \
  --topk_short_kernels 50 \
  --fusion_op_names aten::miopen_convolution,aten::miopen_batch_norm
```