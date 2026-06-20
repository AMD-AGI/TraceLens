<!--
Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# TraceIndex

TraceIndex builds a queryable catalog of trace files and imports key TraceLens
CSV report tables so a corpus can be searched without reopening every raw
trace. It is useful when you have many profiler captures and want to answer
questions like:

- Which traces contain GEMM, SDPA, convolution, collective, or short-kernel heavy workloads?
- Which traces contain a specific backend kernel name?
- Which trace should I open next in TraceLens or Perfetto?

TraceIndex does not replace raw traces. It stores searchable summaries and paths
back to the source traces. The first backend is SQLite because it needs no
service to run and works well for local or single-team workflows. The scanner,
importer, and storage interface are separated so other backends can be added by
implementing the same store contract.

## Quick Start

Catalog trace-like files under a directory:

```bash
TraceLens_trace_index --backend sqlite --db trace_index.sqlite scan --root /path/to/traces
```

Generate a TraceLens report for one PyTorch trace and import it into the index:

```bash
TraceLens_trace_index --backend sqlite --db trace_index.sqlite build \
  --trace-path /path/to/traces/rank0_trace.json.gz \
  --report-dir ./trace_index_reports/rank0
```

If you already have a TraceLens CSV report directory, import it directly:

```bash
TraceLens_trace_index --backend sqlite --db trace_index.sqlite import-report \
  --trace-path /path/to/traces/rank0_trace.json.gz \
  --report-dir ./rank0_perf_report_csvs
```

Search the full-text index:

```bash
TraceLens_trace_index --backend sqlite --db trace_index.sqlite search attention
TraceLens_trace_index --backend sqlite --db trace_index.sqlite search Cijk
```

Run a read-only SQLite query:

```bash
TraceLens_trace_index --backend sqlite --db trace_index.sqlite sqlite-sql \
  "SELECT op_category, COUNT(*) AS rows FROM unified_perf_rows GROUP BY op_category"
```

## Imported Tables

TraceIndex imports the stable report tables that are most useful for corpus
search:

| Table | Contents |
|---|---|
| `traces` | One row per trace-like file or imported report directory |
| `report_imports` | Import history for TraceLens CSV report directories |
| `unified_perf_rows` | Rows from `unified_perf_summary.csv` |
| `kernel_summary` | Rows from `kernel_summary.csv`, including basic kernel flags |
| `op_category_rows` | Rows from `ops_summary_by_category.csv` |
| `gpu_timeline_rows` | Rows from `gpu_timeline.csv` |
| `trace_summary` | Per-trace summary metrics derived during import |
| `trace_search_FTS5` | Full-text search over traces, ops, kernels, categories, and timeline labels |

## Backend Model

TraceIndex separates workflow code from persistence:

| Module | Responsibility |
|---|---|
| `models.py` | Backend-neutral trace, report, and search-result objects |
| `store.py` | `TraceIndexStore` interface for persistence/search backends |
| `scanner.py` | File discovery and trace metadata extraction |
| `importer.py` | TraceLens CSV report loading and import workflow |
| `sqlite_store.py` | SQLite schema, inserts, full-text search, and read-only SQL |
| `cli.py` | User-facing commands that call scanner/importer/store APIs |

To add another backend, implement `TraceIndexStore` and wire it into the CLI
backend selector. Backend-neutral commands such as `scan`, `build`,
`import-report`, and `search` should not need to change.

## SQLite Query Server

For notebook or browser workflows, serve read-only SQL access locally:

```bash
TraceLens_trace_index --backend sqlite --db trace_index.sqlite serve --host 127.0.0.1 --port 8765
```

The server exposes:

- `GET /health`
- `GET /tables`
- `POST /query` with `{"sql": "SELECT ...", "params": [], "limit": 500}`

Only single SQLite `SELECT`, `WITH`, or `PRAGMA` statements are accepted. The
server is read-only but does not implement authentication, so bind it to
loopback unless you put it behind your own access control.

## Python API

```python
from pathlib import Path

from TraceLens.TraceIndex import import_report_dir, scan_traces, search_index

db = Path("trace_index.sqlite")
scan_traces(db, Path("/path/to/traces"))
import_report_dir(db, Path("rank0_perf_report_csvs"), trace_path=Path("rank0_trace.json.gz"))
rows = search_index(db, "Cijk", limit=20)
```
