<!--
Copyright (c) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# TraceIndex catalog schema
```{meta}
:description: SQL table and column names for TraceIndex catalogs, with CSV source mapping. Column semantics live in the performance report column reference.
:keywords: TraceIndex, TraceDB, SQLite, catalog schema, unified_perf_rows, gemm_perf, sdpa_perf, conv_perf, op_kernels
```

This topic is the **SQL naming and mapping index** for TraceIndex catalogs built with
[Index a corpus of traces](../how-to/trace-index.md).

**Column meanings** (units, roofline fields, shape params, kernel details) are documented
in [Performance report columns](./perf-report-columns.md). That topic covers the CSV/Excel
sheets; this topic lists where those fields land in SQLite and which columns are
catalog-only.

## Imported CSV sheets → SQL tables

| CSV file (report dir) | SQL table(s) | Semantics |
|---|---|---|
| `unified_perf_summary.csv` | `unified_perf_rows` | [unified / ops sheets](./perf-report-columns.md) |
| `ops_summary_by_category.csv` | `op_category_rows` | [ops_summary_by_category](./perf-report-columns.md) |
| `gpu_timeline.csv` | `gpu_timeline_rows` | [gpu_timeline](./perf-report-columns.md) |

Derived during import (not separate CSV files): `op_kernels` (from `kernel_details_summary`),
`gemm_perf` / `sdpa_perf` / `conv_perf` (typed shapes from `perf_params`),
`trace_summary`, `trace_search` (FTS).

## Unit conventions in SQL

- Kernel/op times in relational tables: **microseconds** (`*_us`) unless the column ends in `_ms`.
- `gpu_timeline_rows.time_ms` stays in **milliseconds** (matches the CSV).
- `*_json` columns hold parsed structures; use `json_extract` when no typed satellite column exists.

## traces

One row per indexed trace file (or synthetic row for a report-only import).

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Primary key; referenced as `trace_id` elsewhere |
| `root` | TEXT | Optional root directory passed to `append` / `build` |
| `path` | TEXT | Absolute or normalized path to the trace (unique) |
| `rel_path` | TEXT | Path relative to `root`, when set |
| `name` | TEXT | Basename of the trace file |
| `size_bytes` | INTEGER | File size when known |
| `md5` | TEXT | Content hash when computed |
| `format` | TEXT | Trace format hint (for example `chrome`, `pftrace`) |
| `rank` | INTEGER | Rank parsed from filename when present |
| `top_dir` | TEXT | Top-level directory under `root` |
| `parent_rel` | TEXT | Parent directory relative to `root` |
| `should_enrich` | INTEGER | 1 if shape enrichment is expected |
| `skip_reason` | TEXT | Why enrichment was skipped, if any |
| `created_at` | TEXT | ISO timestamp |
| `updated_at` | TEXT | ISO timestamp |

## report_imports

Import audit trail for each CSV report directory loaded into a trace.

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Primary key |
| `trace_id` | INTEGER | FK → `traces.id` |
| `report_dir` | TEXT | Path to the TraceLens CSV report directory |
| `imported_at` | TEXT | ISO timestamp |
| `sheets_json` | TEXT | JSON list of sheet names imported |

## unified_perf_rows

Rows from `unified_perf_summary.csv`, normalized for SQL. Kernel time columns are
always stored in **microseconds** even when the CSV used milliseconds.

| Column | Type | CSV source (typical) | SQL name |
|--------|------|----------------------|----------|
| `id` | INTEGER | — | Primary key |
| `trace_id` | INTEGER | — | FK → `traces.id` |
| `source_row` | INTEGER | — | 1-based row index in the CSV |
| `name` | TEXT | `name` | Operation name |
| `op_category` | TEXT | `op category` | Category label (GEMM, SDPA_fwd, …) |
| `operation_count` | INTEGER | `operation_count` | Invocation count |
| `kernel_time_sum_us` | REAL | `Kernel Time (us)_sum` | Total kernel time (µs) |
| `kernel_time_mean_us` | REAL | `Kernel Time (us)_mean` | Mean kernel time (µs) |
| `kernel_time_median_us` | REAL | `Kernel Time (us)_median` | Median kernel time (µs) |
| `kernel_time_std_us` | REAL | `Kernel Time (us)_std` | Std dev (µs) |
| `kernel_time_min_us` | REAL | `Kernel Time (us)_min` | Min (µs) |
| `kernel_time_max_us` | REAL | `Kernel Time (us)_max` | Max (µs) |
| `op_duration_us` | REAL | `op duration (us)` | Op wall time (µs) |
| `tflops_mean` | REAL | `TFLOPS_mean` | Mean achieved TFLOPS |
| `tflops_median` | REAL | `TFLOPS_median` | Median TFLOPS |
| `tbs_mean` | REAL | `TB/s_mean` | Mean memory throughput |
| `tbs_median` | REAL | `TB/s_median` | Median memory throughput |
| `gflops` | REAL | `GFLOPS` | Reported GFLOPS |
| `data_moved_mb` | REAL | `data moved (MB)` | Data moved |
| `flops_per_byte` | REAL | `FLOPs/Byte` | Arithmetic intensity |
| `compute_spec` | TEXT | `compute spec` | Roofline / spec label |
| `has_perf_model` | INTEGER | `has perf model` | 1 if perf model attached |
| `overlap_pct` | REAL | `overlap %` | Overlap with other streams |
| `perf_params_json` | TEXT | `perf_params` | Parsed shape / params JSON |
| `kernel_details_json` | TEXT | `kernel_details_summary` | Parsed kernel list JSON |
| `raw_row_json` | TEXT | — | Full original CSV row as JSON |

## op_kernels

One row per kernel exploded from `kernel_details_summary` on a unified row.

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Primary key |
| `trace_id` | INTEGER | FK → `traces.id` |
| `unified_row_id` | INTEGER | FK → `unified_perf_rows.id` |
| `kernel_name` | TEXT | HIP kernel symbol |
| `parent_op_name` | TEXT | Parent op from unified row |
| `op_category` | TEXT | Category copied from unified row |
| `stream` | INTEGER | ROCm stream id |
| `count` | INTEGER | Launch count |
| `total_duration_us` | REAL | Total time (µs) |
| `mean_duration_us` | REAL | Mean time (µs) |
| `median_duration_us` | REAL | Median time (µs) |
| `min_duration_us` | REAL | Min time (µs) |
| `max_duration_us` | REAL | Max time (µs) |
| `library` | TEXT | Heuristic label: Tensile, Triton, CK, RCCL/NCCL, … |
| `is_tensile` | INTEGER | 1 if name matches Tensile / Cijk pattern |
| `is_transpose` | INTEGER | 1 if transpose / permute in name |
| `is_layout_conversion` | INTEGER | 1 if layout / cast / copy pattern |
| `details_json` | TEXT | Per-kernel JSON from import |

## gemm_perf

Typed GEMM shapes parsed from `perf_params` on GEMM-category unified rows.

| Column | Type | Description |
|--------|------|-------------|
| `unified_row_id` | INTEGER | PK, FK → `unified_perf_rows.id` |
| `trace_id` | INTEGER | FK → `traces.id` |
| `m`, `n`, `k` | INTEGER | GEMM dimensions |
| `batch` | INTEGER | Batch when present |
| `dtype` | TEXT | Data type string |
| `transpose` | TEXT | Transpose flags |
| `tflops_mean`, `tflops_median` | REAL | Copied from unified row |

## sdpa_perf

Typed attention shapes from SDPA-category unified rows.

| Column | Type | Description |
|--------|------|-------------|
| `unified_row_id` | INTEGER | PK, FK → `unified_perf_rows.id` |
| `trace_id` | INTEGER | FK → `traces.id` |
| `batch` | INTEGER | Batch size |
| `heads` | INTEGER | Number of heads |
| `seq_q` | INTEGER | Query sequence length |
| `seq_kv` | INTEGER | Key/value sequence length |
| `head_dim` | INTEGER | Head dimension |
| `dtype` | TEXT | Data type |
| `causal` | INTEGER | 1 if causal mask |
| `tflops_mean`, `tflops_median` | REAL | Copied from unified row |

## conv_perf

Typed convolution shapes from CONV-category unified rows.

| Column | Type | Description |
|--------|------|-------------|
| `unified_row_id` | INTEGER | PK, FK → `unified_perf_rows.id` |
| `trace_id` | INTEGER | FK → `traces.id` |
| `conv_nd` | TEXT | `"2d"` or `"3d"` |
| `input_shape_json` | TEXT | Input tensor shape JSON |
| `filter_shape_json` | TEXT | Filter shape JSON |
| `input_channels` | INTEGER | Input channels |
| `output_channels` | INTEGER | Output channels |
| `groups` | INTEGER | Group count |
| `kernel_h`, `kernel_w` | INTEGER | Spatial kernel size |
| `is_depthwise` | INTEGER | 1 if depthwise conv |
| `is_transposed_conv` | INTEGER | 1 if transposed conv |

## op_category_rows

Rows from `ops_summary_by_category.csv`.

| Column | Type | CSV source | Description |
|--------|------|------------|-------------|
| `id` | INTEGER | — | Primary key |
| `trace_id` | INTEGER | — | FK → `traces.id` |
| `category` | TEXT | `category` | Op category name |
| `operation_count` | INTEGER | `operation_count` | Count |
| `kernel_time_sum_us` | REAL | kernel time column | Sum in µs |
| `percent` | REAL | `percent` | Fraction of total time |
| `raw_row_json` | TEXT | — | Full CSV row |

## gpu_timeline_rows

Rows from `gpu_timeline.csv`.

| Column | Type | CSV source | Description |
|--------|------|------------|-------------|
| `id` | INTEGER | — | Primary key |
| `trace_id` | INTEGER | — | FK → `traces.id` |
| `type` | TEXT | `type` | Timeline bucket (computation_time, idle_time, …) |
| `time_ms` | REAL | `time ms` | Duration in **milliseconds** |
| `percent` | REAL | `percent` | Percent of wall time |
| `raw_row_json` | TEXT | — | Full CSV row |

## trace_summary

Per-trace rollups computed at import time.

| Column | Type | Description |
|--------|------|-------------|
| `trace_id` | INTEGER | PK, FK → `traces.id` |
| `total_duration_us` | REAL | Total trace duration (µs) |
| `top_categories_json` | TEXT | JSON list of dominant categories |
| `max_gemm_tflops` | REAL | Peak GEMM TFLOPS in this trace |
| `max_sdpa_tflops` | REAL | Peak SDPA TFLOPS in this trace |
| `imported_at` | TEXT | ISO timestamp |

## trace_search (FTS5)

Full-text search virtual table. Query via the `search` CLI subcommand or
`MATCH` syntax; columns are not meant for ad-hoc SELECT listing.

| Column | Type | Description |
|--------|------|-------------|
| `trace_id` | INTEGER | FK → `traces.id` (unindexed in FTS) |
| `kind` | TEXT | Hit kind: trace, op, kernel, category, timeline |
| `text` | TEXT | Searchable text |

## Related topics

- [Index a corpus of traces](../how-to/trace-index.md)
- [Performance report columns](./perf-report-columns.md) — CSV / Excel report sheets
- [API reference](./api-reference.md)
