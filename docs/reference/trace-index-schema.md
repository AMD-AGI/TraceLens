<!--
Copyright (c) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# TraceIndex catalog schema
```{meta}
:description: Column reference for every TraceIndex SQLite catalog table, including traces, unified performance rows, kernel and shape satellites, and full-text search.
:keywords: TraceLens, TraceIndex, SQLite, catalog schema, unified_perf_rows, op_kernels, gemm_perf, sdpa_perf, conv_perf, GPU, full-text search
```

This topic lists the columns in each TraceIndex catalog table. The tables are
the shared query surface for notebooks, SQL, and later storage backends. SQLite
is the first backend; column names and types come from the `CREATE TABLE`
statements in the TraceIndex store.

Per-trace fact tables (`report_imports`, `unified_perf_rows`,
`op_category_rows`, `gpu_timeline_rows`, `trace_summary`, and
`trace_search_FTS5`) point at `traces` with `trace_id`. Kernel rows and GEMM /
SDPA / convolution satellites hang off `unified_perf_rows` with
`unified_row_id`. Join those satellites to `traces` through the unified parent:

```sql
JOIN unified_perf_rows u ON u.id = satellite.unified_row_id
JOIN traces t ON t.id = u.trace_id
```

See [Index a corpus of traces](../how-to/trace-index.md) for the workflow and
example queries.

## The traces table

One row per indexed trace file.

The following table lists the columns in `traces`.

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER | Primary key. |
| `root` | TEXT | Catalog root used to compute `rel_path`. |
| `path` | TEXT | Unique filesystem path of the trace. |
| `rel_path` | TEXT | Path relative to `root`. |
| `name` | TEXT | Trace filename. |
| `size_bytes` | INTEGER | File size in bytes. |
| `md5` | TEXT | Optional content hash. |
| `format` | TEXT | Detected profiler format from the filename and a content prefix. |
| `rank` | INTEGER | Rank parsed from the path when present. |
| `top_dir` | TEXT | First component of `rel_path`. |
| `parent_rel` | TEXT | Parent directory of the trace, relative to `root`. |
| `should_enrich` | INTEGER | `1` when the file is a supported trace that isn't skipped; default `1`. |
| `skip_reason` | TEXT | Why a path was skipped, or `NULL`. |
| `created_at` | TEXT | UTC timestamp when the row was inserted. |
| `updated_at` | TEXT | UTC timestamp of the last upsert. |

## The report_imports table

Import history for TraceLens CSV report directories.

The following table lists the columns in `report_imports`.

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER | Primary key. |
| `trace_id` | INTEGER | Foreign key to `traces.id`. `ON DELETE CASCADE`. |
| `report_dir` | TEXT | Directory of the imported CSV report. |
| `imported_at` | TEXT | UTC timestamp of this import. |
| `sheets_json` | TEXT | JSON array of sheet names that had rows. |

## The unified_perf_rows table

Rows imported from `unified_perf_summary.csv`. `perf_params_json` and
`kernel_details_json` store parsed JSON, not the Python `repr` from the CSV.

The following table lists the columns in `unified_perf_rows`.

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER | Primary key. |
| `trace_id` | INTEGER | Foreign key to `traces.id`. `ON DELETE CASCADE`. |
| `source_row` | INTEGER | Zero-based row index in the CSV. |
| `name` | TEXT | Operation name. |
| `op_category` | TEXT | Operation category (for example `GEMM`, `SDPA_fwd`). |
| `operation_count` | INTEGER | Number of operation occurrences. |
| `kernel_time_sum_us` | REAL | Sum of kernel time in microseconds. |
| `kernel_time_mean_us` | REAL | Mean kernel time in microseconds. |
| `kernel_time_median_us` | REAL | Median kernel time in microseconds. |
| `kernel_time_std_us` | REAL | Standard deviation of kernel time in microseconds. |
| `kernel_time_min_us` | REAL | Minimum kernel time in microseconds. |
| `kernel_time_max_us` | REAL | Maximum kernel time in microseconds. |
| `op_duration_us` | REAL | Host operation duration in microseconds. |
| `tflops_mean` | REAL | Mean TFLOPS/s from the report. |
| `tflops_median` | REAL | Median TFLOPS/s from the report. |
| `tbs_mean` | REAL | Mean TB/s from the report. |
| `tbs_median` | REAL | Median TB/s from the report. |
| `gflops` | REAL | GFLOPS from the report. |
| `data_moved_mb` | REAL | Data moved in megabytes. |
| `flops_per_byte` | REAL | Arithmetic intensity. |
| `compute_spec` | TEXT | Compute spec string from the report. |
| `has_perf_model` | INTEGER | `1` when the row has a perf model, else `0`. |
| `overlap_pct` | REAL | Overlap percentage from the report. |
| `perf_params_json` | TEXT | Parsed `perf_params` as JSON. |
| `kernel_details_json` | TEXT | Parsed `kernel_details_summary` as JSON. |
| `raw_row_json` | TEXT | Full source CSV row as JSON. |

## The op_kernels table

One row per kernel exploded from `kernel_details_summary` on a unified row.
This table has no `trace_id`; join `unified_perf_rows` to reach `traces`.

The following table lists the columns in `op_kernels`.

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER | Primary key. |
| `unified_row_id` | INTEGER | Foreign key to `unified_perf_rows.id`. `NOT NULL`. `ON DELETE CASCADE`. |
| `kernel_name` | TEXT | Kernel name. `NOT NULL`. |
| `parent_op_name` | TEXT | Name of the parent unified row. |
| `op_category` | TEXT | Category of the parent unified row. |
| `stream` | INTEGER | GPU stream id from kernel details. |
| `count` | INTEGER | Kernel launch count. |
| `total_duration_us` | REAL | Total duration in microseconds. |
| `mean_duration_us` | REAL | Mean duration in microseconds. |
| `median_duration_us` | REAL | Median duration in microseconds. |
| `min_duration_us` | REAL | Minimum duration in microseconds. |
| `max_duration_us` | REAL | Maximum duration in microseconds. |
| `library` | TEXT | Library inferred from the kernel name (`Tensile`, `Triton`, `CK`, `RCCL/NCCL`), or `NULL`. |
| `is_tensile` | INTEGER | `1` when the name looks like Tensile; default `0`. |
| `is_transpose` | INTEGER | `1` when the name looks like a transpose or permute; default `0`. |
| `is_layout_conversion` | INTEGER | `1` when the name looks like a layout conversion; default `0`. |
| `details_json` | TEXT | Source kernel-detail dict as JSON. |

## The gemm_perf table

GEMM shapes parsed from `perf_params` on a unified row. Primary key is
`unified_row_id` (at most one GEMM satellite per unified row). Join
`unified_perf_rows` to reach `traces`.

The following table lists the columns in `gemm_perf`.

| Column | Type | Description |
|---|---|---|
| `unified_row_id` | INTEGER | Primary key and foreign key to `unified_perf_rows.id`. `ON DELETE CASCADE`. |
| `m` | INTEGER | `M` from `perf_params`. |
| `n` | INTEGER | `N` from `perf_params`. |
| `k` | INTEGER | `K` from `perf_params`. |
| `batch` | INTEGER | `B` from `perf_params`. |
| `dtype` | TEXT | `dtype_A_B` from `perf_params`, stringified. |
| `transpose` | TEXT | `transpose` from `perf_params`, stringified. |
| `tflops_mean` | REAL | Mean TFLOPS/s copied from the unified row. |
| `tflops_median` | REAL | Median TFLOPS/s copied from the unified row. |

## The sdpa_perf table

Attention shapes parsed from `perf_params` on a unified row. Primary key is
`unified_row_id`. Join `unified_perf_rows` to reach `traces`.

The following table lists the columns in `sdpa_perf`.

| Column | Type | Description |
|---|---|---|
| `unified_row_id` | INTEGER | Primary key and foreign key to `unified_perf_rows.id`. `ON DELETE CASCADE`. |
| `batch` | INTEGER | `B` from `perf_params`. |
| `heads` | INTEGER | `H_Q` from `perf_params`. |
| `seq_q` | INTEGER | `N_Q` from `perf_params`. |
| `seq_kv` | INTEGER | `N_KV` from `perf_params`. |
| `head_dim` | INTEGER | `d_h_qk` from `perf_params`, or `d_h_v` if `d_h_qk` is missing. |
| `dtype` | TEXT | `dtype_A_B` from `perf_params`, stringified. |
| `causal` | INTEGER | `1` / `0` from `perf_params.causal`, or `NULL`. |
| `tflops_mean` | REAL | Mean TFLOPS/s copied from the unified row. |
| `tflops_median` | REAL | Median TFLOPS/s copied from the unified row. |

## The conv_perf table

Convolution shapes parsed from `perf_params` on a unified row. Primary key is
`unified_row_id`. Join `unified_perf_rows` to reach `traces`.

The following table lists the columns in `conv_perf`.

| Column | Type | Description |
|---|---|---|
| `unified_row_id` | INTEGER | Primary key and foreign key to `unified_perf_rows.id`. `ON DELETE CASCADE`. |
| `conv_nd` | TEXT | `convNd` from `perf_params`. |
| `input_shape_json` | TEXT | `input_shape` as JSON. |
| `filter_shape_json` | TEXT | `filter_shape` as JSON. |
| `input_channels` | INTEGER | Channel count from `input_shape[1]`. |
| `output_channels` | INTEGER | Output channels from `filter_shape[0]`. |
| `groups` | INTEGER | `groups` from `perf_params`, or `1` when missing. |
| `kernel_h` | INTEGER | Kernel height from `filter_shape[-2]`. |
| `kernel_w` | INTEGER | Kernel width from `filter_shape[-1]`. |
| `is_depthwise` | INTEGER | `1` when groups equal input channels, the filter is one channel per group, and output channels are a multiple of input channels. |
| `is_transposed_conv` | INTEGER | `1` / `0` from `perf_params.transposed_conv`, or `NULL`. |

## The op_category_rows table

Rows imported from `ops_summary_by_category.csv`.

The following table lists the columns in `op_category_rows`.

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER | Primary key. |
| `trace_id` | INTEGER | Foreign key to `traces.id`. `ON DELETE CASCADE`. |
| `category` | TEXT | Operation category. `NOT NULL`. |
| `operation_count` | INTEGER | Number of operations in the category. |
| `kernel_time_sum_us` | REAL | Sum of kernel time in microseconds. |
| `percent` | REAL | Share of total time from the report. |
| `raw_row_json` | TEXT | Full source CSV row as JSON. |

## The gpu_timeline_rows table

Rows imported from `gpu_timeline.csv`.

The following table lists the columns in `gpu_timeline_rows`.

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER | Primary key. |
| `trace_id` | INTEGER | Foreign key to `traces.id`. `ON DELETE CASCADE`. |
| `type` | TEXT | Timeline metric name (for example `total_time`, `computation_time`). `NOT NULL`. |
| `time_ms` | REAL | Duration in milliseconds. |
| `percent` | REAL | Share of total GPU time. |
| `raw_row_json` | TEXT | Full source CSV row as JSON. |

## The trace_summary table

Per-trace summary metrics derived during import. One row per trace.

The following table lists the columns in `trace_summary`.

| Column | Type | Description |
|---|---|---|
| `trace_id` | INTEGER | Primary key and foreign key to `traces.id`. `ON DELETE CASCADE`. |
| `total_duration_us` | REAL | `total_time` from `gpu_timeline`, converted to microseconds. |
| `top_categories_json` | TEXT | JSON array of the top five categories by kernel time. |
| `max_gemm_tflops` | REAL | Maximum GEMM TFLOPS/s seen on unified rows. |
| `max_sdpa_tflops` | REAL | Maximum SDPA TFLOPS/s seen on unified rows. |
| `imported_at` | TEXT | UTC timestamp of this import. |

## The trace_search_FTS5 table

Full-text search virtual table (`fts5`, `tokenize='unicode61'`). `trace_id` and
`kind` are stored but not tokenized.

The following table lists the columns in `trace_search_FTS5`.

| Column | Type | Description |
|---|---|---|
| `trace_id` | UNINDEXED | Trace id of the indexed document. Join to `traces.id`. |
| `kind` | UNINDEXED | Document kind: `trace`, `op`, `kernel`, `category`, or `timeline`. |
| `text` | TEXT | Searchable text for that kind. |

## Related topics

- [Index a corpus of traces](../how-to/trace-index.md)
- [Performance report columns](./perf-report-columns.md)
- [API reference](./api-reference.md)
- [Generate a PyTorch performance report](../how-to/generate-perf-report-pytorch.md)
