<!--
Copyright (c) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->


# Index a corpus of traces in TraceLens
```{meta}
:description: Learn how to catalog profiler traces and TraceLens CSV reports into a searchable SQLite index without reopening every raw trace.
:keywords: TraceLens, TraceIndex, corpus search, SQLite, unified_perf_summary, kernel summary, full-text search, performance report
```

This topic shows how to build a queryable catalog of profiler traces and
TraceLens CSV reports so you can search a corpus without reopening every raw
file. TraceIndex stores summaries and paths back to the source traces; it
doesn't replace the traces themselves.

The catalog is a SQLite file. It needs no extra service and works for local or
single-team workflows.

## Before you begin

- TraceLens installed (see [Install TraceLens](../install/install.md)).
- A directory of profiler traces, or an existing TraceLens CSV report directory
  (for example from
  [Generate a PyTorch performance report](./generate-perf-report-pytorch.md)).

## Catalog traces

Scan a directory for trace-like files and record them in the catalog:

```bash
TraceLens_trace_index --backend sqlite --db trace_index.sqlite scan --root /path/to/traces
```

This pass peeks at file headers. It doesn't parse whole traces or run
TraceLens analysis.

## Import a report

If you already have a TraceLens CSV report directory, import it. This is the
usual ingest path, including for inference reports you generated separately:

```bash
TraceLens_trace_index --backend sqlite --db trace_index.sqlite import-report \
  --trace-path /path/to/traces/rank0_trace.json.gz \
  --report-dir ./rank0_perf_report_csvs
```

`--trace-path` is optional. If you omit it, the report directory is cataloged
as its own row.

To generate a training PyTorch CSV report and import it in one step:

```bash
TraceLens_trace_index --backend sqlite --db trace_index.sqlite build \
  --trace-path /path/to/traces/rank0_trace.json.gz \
  --report-dir ./trace_index_reports/rank0
```

`build` calls the training PyTorch report generator. For inference, rocprof, or
pftrace reports, generate the CSV directory with the matching report command,
then use `import-report`.

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

The following table lists the SQLite tables filled on import.

| Table | Contents |
|---|---|
| `traces` | One row per trace-like file or imported report directory |
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

from TraceLens.TraceIndex import import_report_dir, scan_traces, search_index

db = Path("trace_index.sqlite")
scan_traces(db, Path("/path/to/traces"))
import_report_dir(db, Path("rank0_perf_report_csvs"), trace_path=Path("rank0_trace.json.gz"))
rows = search_index(db, "Cijk", limit=20)
```

## Related topics

- [Install TraceLens](../install/install.md)
- [Generate a PyTorch performance report](./generate-perf-report-pytorch.md)
- [Generate a PyTorch inference performance report](./generate-perf-report-pytorch-inference.md)
- [Analyze traces with the TraceLens SDK](./sdk-analysis.md)
- [Performance report columns](../reference/perf-report-columns.md)
- [API reference](../reference/api-reference.md)
