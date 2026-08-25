<!--
Copyright (c) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->


# Index a corpus of traces in TraceLens
```{meta}
:description: Learn how to catalog profiler traces and TraceLens CSV reports into a searchable index. SQLite is the first backend; the table schema is the shared query surface.
:keywords: TraceLens, TraceIndex, corpus search, catalog schema, SQLite, unified_perf_summary, kernel summary, full-text search, performance report
```

This topic shows how to build a queryable catalog of profiler traces and
TraceLens CSV reports so you can search a corpus without reopening every raw
file. TraceIndex stores summaries and paths back to the source traces; it
doesn't replace the traces themselves.

The tables below are the catalog schema — the shared query surface. Notebooks,
SQL, and later storage backends should use these table and column names so
catalogs stay interchangeable across users. SQLite is the first backend, not a
prototype: it ships with Python, writes one file, and needs no extra service.
Other backends can implement the same schema later.

## Before you begin

- TraceLens installed (see [Install TraceLens](../install/install.md)).
- Profiler traces, and optionally existing TraceLens CSV report directories
  (for example from
  [Generate a PyTorch performance report](./generate-perf-report-pytorch.md)).

## Append a trace

Append one trace to the catalog. Pass `--report-dir` when you already have a
CSV report. This is the usual path for inference, rocprof, and pftrace
reports you generated separately:

```bash
TraceLens_trace_index --db trace_index.sqlite append \
  --trace-path /path/to/traces/rank0_trace.json.gz \
  --report-dir ./rank0_perf_report_csvs
```

If you omit `--report-dir`, TraceIndex generates a training PyTorch CSV report
and then imports it:

```bash
TraceLens_trace_index --db trace_index.sqlite append \
  --trace-path /path/to/traces/rank0_trace.json.gz
```

## Build a catalog from a list of traces

`--db` creates the SQLite file if it doesn't exist. `build` walks a list of
trace paths, generates a training PyTorch report for each, and appends it.
Use a text file (one path per line; `#` starts a comment) and/or repeat
`--trace-path`:

```bash
TraceLens_trace_index --db trace_index.sqlite build \
  --traces-file traces.txt \
  --report-root ./trace_index_reports
```

A failed trace is recorded and the rest of the list still runs. For inference,
rocprof, or pftrace, generate the CSV reports first, then `append` each trace
with `--report-dir`.

## Search and query

Full-text search (FTS) over indexed ops, kernels, categories, and timeline
labels:

```bash
TraceLens_trace_index --backend sqlite --db trace_index.sqlite search attention
TraceLens_trace_index --backend sqlite --db trace_index.sqlite search Cijk
```

Run a single read-only SQL statement:

```bash
TraceLens_trace_index --backend sqlite --db trace_index.sqlite sqlite-sql \
  "SELECT op_category, COUNT(*) AS rows FROM unified_perf_rows GROUP BY op_category"
```

## What the catalog stores

The following tables are the catalog schema. There are seven relational tables
plus a full-text search (FTS) virtual table. SQLite holds this comfortably for
typical TraceLens corpora (hundreds of traces, hundreds of thousands of kernel
rows). The practical limit is one writer at a time, not row count.

| Table | Contents |
|---|---|
| `traces` | One row per indexed trace |
| `report_imports` | Import history for TraceLens CSV report directories |
| `unified_perf_rows` | Rows from `unified_perf_summary.csv` |
| `kernel_summary` | Rows from `kernel_summary.csv`, including Tensile and layout flags |
| `op_category_rows` | Rows from `ops_summary_by_category.csv` |
| `gpu_timeline_rows` | Rows from `gpu_timeline.csv` |
| `trace_summary` | Per-trace summary metrics derived during import |
| `trace_search_FTS5` | Full-text search over traces, ops, kernels, categories, and timeline labels |

## Serve read-only SQL

For notebook or browser workflows, serve the SQLite catalog over HTTP:

```bash
TraceLens_trace_index --backend sqlite --db trace_index.sqlite serve --host 127.0.0.1 --port 8765
```

The server exposes:

- `GET /health`
- `GET /tables`
- `POST /query` with `{"sql": "SELECT ...", "params": [], "limit": 500}`

```{note}
Only a single `SELECT`, `WITH`, or `PRAGMA` statement is accepted. The server
is read-only but isn't authenticated, so bind it to loopback unless you put it
behind your own access control.
```

## Python API

```python
from pathlib import Path

from TraceLens.TraceIndex import append_trace, build_traces, search_index

db = Path("trace_index.sqlite")
append_trace(
    db,
    Path("rank0_trace.json.gz"),
    report_dir=Path("rank0_perf_report_csvs"),
)
build_traces(db, [Path("a.json.gz"), Path("b.json.gz")])
rows = search_index(db, "Cijk", limit=20)
```

## Related topics

- [Install TraceLens](../install/install.md)
- [Generate a PyTorch performance report](./generate-perf-report-pytorch.md)
- [Generate a PyTorch inference performance report](./generate-perf-report-pytorch-inference.md)
- [Analyze traces with the TraceLens SDK](./sdk-analysis.md)
- [Performance report columns](../reference/perf-report-columns.md)
- [API reference](../reference/api-reference.md)
