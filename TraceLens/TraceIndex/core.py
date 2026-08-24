###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""SQLite-backed index for searchable TraceLens trace corpora."""

import csv
import gzip
import hashlib
import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


TRACE_NAME_RE = re.compile(r"trace|profile|pytorch_profile|rocprof", re.IGNORECASE)
RANK_RE = re.compile(r"(?:^|[^A-Za-z])rank[-_]?(\d+)(?:[^0-9]|$)", re.IGNORECASE)

SKIP_PARTS_EXACT = {
    ".git",
    "__pycache__",
    "node_modules",
    "_perf_report_csvs",
    "perf_report_csvs",
    "gap_analysis",
    "capture_traces",
    "graph_capture",
}
SKIP_PARTS_CONTAINS = (
    "_perf_report_csvs",
    "perf_report",
    "gap_analysis",
    "capture_traces",
    "graph_capture",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_path(path: Path) -> str:
    return str(path).replace("\\", "/")


def rel_to(path: Path, root: Path) -> str:
    try:
        return normalize_path(path.relative_to(root))
    except ValueError:
        return normalize_path(path)


def iter_files(root: Path) -> Iterable[Path]:
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as entries:
                dirs = []
                for entry in entries:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            dirs.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            yield Path(entry.path)
                    except OSError:
                        continue
                stack.extend(reversed(dirs))
        except OSError:
            continue


def is_json_gz(path: Path) -> bool:
    return path.name.lower().endswith(".json.gz")


def is_candidate(path: Path) -> bool:
    name = path.name.lower()
    suffix = path.suffix.lower()
    if is_json_gz(path):
        return True
    if suffix in {".json", ".pftrace", ".rpd"}:
        return True
    if name.endswith(".xplane.pb"):
        return True
    if ".pt.trace" in name or ".trace." in name:
        return True
    return bool(TRACE_NAME_RE.search(path.name))


def classify_skip(path: Path, root: Path) -> Optional[str]:
    try:
        rel_parts = [part.lower() for part in path.relative_to(root).parts[:-1]]
    except ValueError:
        rel_parts = []
    for part in rel_parts:
        if part in SKIP_PARTS_EXACT:
            return part
        for token in SKIP_PARTS_CONTAINS:
            if token in part:
                return token
    name = path.name.lower()
    if name.endswith((".xlsx", ".csv", ".log", ".jsonl", ".md", ".txt")):
        return "derived_or_log"
    return None


def read_prefix(path: Path, max_bytes: int) -> str:
    try:
        if is_json_gz(path):
            with gzip.open(path, "rb") as f:
                data = f.read(max_bytes)
        else:
            with path.open("rb") as f:
                data = f.read(max_bytes)
    except (OSError, EOFError, gzip.BadGzipFile):
        return ""
    return data.decode("utf-8", errors="ignore")


def content_md5(path: Path, chunk_size: int = 8 * 1024 * 1024) -> Optional[str]:
    hasher = hashlib.md5()
    try:
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                hasher.update(chunk)
    except OSError:
        return None
    return hasher.hexdigest()


def detect_format(path: Path, prefix: str) -> str:
    name = path.name.lower()
    suffix = path.suffix.lower()
    if suffix == ".pftrace":
        return "pftrace"
    if suffix == ".rpd":
        return "rocprof_rpd"
    if name.endswith(".xplane.pb"):
        return "xplane_pb"
    if '"traceEvents"' in prefix:
        return "kineto_chrome_json_gz" if is_json_gz(path) else "kineto_chrome_json"
    if '"rocprofiler-sdk-tool"' in prefix:
        return "rocprofv3_json_gz" if is_json_gz(path) else "rocprofv3_json"
    if is_json_gz(path):
        return "json_gz_unknown"
    if suffix == ".json":
        return "json_unknown"
    return "trace_named_unknown"


def extract_rank(path: Path) -> Optional[int]:
    match = RANK_RE.search(normalize_path(path))
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=60)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=60000")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS traces (
            id INTEGER PRIMARY KEY,
            root TEXT,
            path TEXT NOT NULL UNIQUE,
            rel_path TEXT,
            name TEXT,
            size_bytes INTEGER,
            md5 TEXT,
            format TEXT,
            rank INTEGER,
            top_dir TEXT,
            parent_rel TEXT,
            should_enrich INTEGER NOT NULL DEFAULT 1,
            skip_reason TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_trace_index_traces_md5 ON traces(md5);
        CREATE INDEX IF NOT EXISTS idx_trace_index_traces_top_dir ON traces(top_dir);
        CREATE INDEX IF NOT EXISTS idx_trace_index_traces_format ON traces(format);
        CREATE INDEX IF NOT EXISTS idx_trace_index_traces_should_enrich ON traces(should_enrich);

        CREATE TABLE IF NOT EXISTS report_imports (
            id INTEGER PRIMARY KEY,
            trace_id INTEGER NOT NULL REFERENCES traces(id) ON DELETE CASCADE,
            report_dir TEXT NOT NULL,
            imported_at TEXT NOT NULL,
            sheets_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS unified_perf_rows (
            id INTEGER PRIMARY KEY,
            trace_id INTEGER NOT NULL REFERENCES traces(id) ON DELETE CASCADE,
            source_row INTEGER NOT NULL,
            name TEXT,
            op_category TEXT,
            operation_count INTEGER,
            kernel_time_sum_us REAL,
            kernel_time_mean_us REAL,
            kernel_time_median_us REAL,
            kernel_time_std_us REAL,
            kernel_time_min_us REAL,
            kernel_time_max_us REAL,
            op_duration_us REAL,
            tflops_mean REAL,
            tflops_median REAL,
            tbs_mean REAL,
            tbs_median REAL,
            gflops REAL,
            data_moved_mb REAL,
            flops_per_byte REAL,
            compute_spec TEXT,
            has_perf_model INTEGER,
            overlap_pct REAL,
            perf_params_json TEXT,
            kernel_details_json TEXT,
            raw_row_json TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_trace_index_unified_trace ON unified_perf_rows(trace_id);
        CREATE INDEX IF NOT EXISTS idx_trace_index_unified_category ON unified_perf_rows(op_category);
        CREATE INDEX IF NOT EXISTS idx_trace_index_unified_name ON unified_perf_rows(name);

        CREATE TABLE IF NOT EXISTS kernel_summary (
            id INTEGER PRIMARY KEY,
            trace_id INTEGER NOT NULL REFERENCES traces(id) ON DELETE CASCADE,
            kernel_name TEXT NOT NULL,
            parent_op_name TEXT,
            op_category TEXT,
            stream INTEGER,
            count INTEGER,
            total_duration_us REAL,
            mean_duration_us REAL,
            median_duration_us REAL,
            min_duration_us REAL,
            max_duration_us REAL,
            is_tensile INTEGER NOT NULL DEFAULT 0,
            is_transpose INTEGER NOT NULL DEFAULT 0,
            is_layout_conversion INTEGER NOT NULL DEFAULT 0,
            raw_row_json TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_trace_index_kernel_trace ON kernel_summary(trace_id);
        CREATE INDEX IF NOT EXISTS idx_trace_index_kernel_name ON kernel_summary(kernel_name);
        CREATE INDEX IF NOT EXISTS idx_trace_index_kernel_tensile ON kernel_summary(is_tensile);

        CREATE TABLE IF NOT EXISTS op_category_rows (
            id INTEGER PRIMARY KEY,
            trace_id INTEGER NOT NULL REFERENCES traces(id) ON DELETE CASCADE,
            category TEXT NOT NULL,
            operation_count INTEGER,
            kernel_time_sum_us REAL,
            percent REAL,
            raw_row_json TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_trace_index_category_trace ON op_category_rows(trace_id);

        CREATE TABLE IF NOT EXISTS gpu_timeline_rows (
            id INTEGER PRIMARY KEY,
            trace_id INTEGER NOT NULL REFERENCES traces(id) ON DELETE CASCADE,
            type TEXT NOT NULL,
            time_ms REAL,
            percent REAL,
            raw_row_json TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_trace_index_timeline_trace ON gpu_timeline_rows(trace_id);
        CREATE INDEX IF NOT EXISTS idx_trace_index_timeline_type ON gpu_timeline_rows(type);

        CREATE TABLE IF NOT EXISTS trace_summary (
            trace_id INTEGER PRIMARY KEY REFERENCES traces(id) ON DELETE CASCADE,
            total_duration_us REAL,
            top_categories_json TEXT,
            max_gemm_tflops REAL,
            max_sdpa_tflops REAL,
            imported_at TEXT NOT NULL
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS trace_search_FTS5 USING fts5(
            trace_id UNINDEXED,
            kind UNINDEXED,
            text,
            tokenize='unicode61'
        );
        """
    )


def add_trace(
    conn: sqlite3.Connection,
    trace_path: Path,
    root: Optional[Path] = None,
    peek_bytes: int = 2 * 1024 * 1024,
    compute_md5: bool = False,
) -> int:
    trace_path = trace_path.resolve()
    root = root.resolve() if root is not None else trace_path.parent.resolve()
    prefix = read_prefix(trace_path, peek_bytes)
    skip_reason = classify_skip(trace_path, root)
    should_enrich = int(skip_reason is None and detect_format(trace_path, prefix) in {
        "kineto_chrome_json",
        "kineto_chrome_json_gz",
        "rocprofv3_json",
        "rocprofv3_json_gz",
        "pftrace",
    })
    rel_path = rel_to(trace_path, root)
    rel_parts = Path(rel_path).parts
    now = utc_now()
    conn.execute(
        """
        INSERT INTO traces(
            root, path, rel_path, name, size_bytes, md5, format, rank, top_dir,
            parent_rel, should_enrich, skip_reason, created_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(path) DO UPDATE SET
            root = excluded.root,
            rel_path = excluded.rel_path,
            name = excluded.name,
            size_bytes = excluded.size_bytes,
            md5 = COALESCE(excluded.md5, traces.md5),
            format = excluded.format,
            rank = excluded.rank,
            top_dir = excluded.top_dir,
            parent_rel = excluded.parent_rel,
            should_enrich = excluded.should_enrich,
            skip_reason = excluded.skip_reason,
            updated_at = excluded.updated_at
        """,
        (
            normalize_path(root),
            normalize_path(trace_path),
            rel_path,
            trace_path.name,
            trace_path.stat().st_size if trace_path.exists() else None,
            content_md5(trace_path) if compute_md5 else None,
            detect_format(trace_path, prefix),
            extract_rank(trace_path),
            rel_parts[0] if rel_parts else "",
            normalize_path(Path(*rel_parts[:-1])) if len(rel_parts) > 1 else "",
            should_enrich,
            skip_reason,
            now,
            now,
        ),
    )
    row = conn.execute("SELECT id FROM traces WHERE path = ?", (normalize_path(trace_path),)).fetchone()
    return int(row["id"])


def scan_traces(
    db_path: Path,
    root: Path,
    peek_mb: int = 2,
    compute_md5: bool = False,
) -> int:
    conn = connect(db_path)
    init_db(conn)
    root = root.resolve()
    count = 0
    for path in iter_files(root):
        if not is_candidate(path):
            continue
        add_trace(
            conn,
            path,
            root=root,
            peek_bytes=peek_mb * 1024 * 1024,
            compute_md5=compute_md5,
        )
        count += 1
    conn.commit()
    conn.close()
    return count


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def first_value(row: Dict[str, Any], names: Sequence[str], default: Any = None) -> Any:
    lower_map = {key.lower(): key for key in row.keys()}
    for name in names:
        key = lower_map.get(name.lower())
        if key is None:
            continue
        value = row.get(key)
        if value not in (None, "", "nan", "NaN"):
            return value
    return default


def as_text(value: Any) -> Optional[str]:
    if value in (None, "", "nan", "NaN"):
        return None
    return str(value)


def as_float(value: Any) -> Optional[float]:
    if value in (None, "", "nan", "NaN"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_int(value: Any) -> Optional[int]:
    number = as_float(value)
    return int(number) if number is not None else None


def as_bool_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if value in (None, "", "nan", "NaN"):
        return 0
    text = str(value).strip().lower()
    return int(text in {"1", "true", "yes", "y"})


def search_text(*parts: Any) -> str:
    return " ".join(str(part) for part in parts if part not in (None, "", "nan", "NaN"))


def clear_trace_payload(conn: sqlite3.Connection, trace_id: int) -> None:
    for table in (
        "report_imports",
        "unified_perf_rows",
        "kernel_summary",
        "op_category_rows",
        "gpu_timeline_rows",
        "trace_summary",
    ):
        conn.execute(f"DELETE FROM {table} WHERE trace_id = ?", (trace_id,))
    conn.execute("DELETE FROM trace_search_FTS5 WHERE trace_id = ?", (trace_id,))


def insert_search(conn: sqlite3.Connection, trace_id: int, kind: str, parts: Iterable[Any]) -> None:
    text = search_text(*parts)
    if text:
        conn.execute(
            "INSERT INTO trace_search_FTS5(trace_id, kind, text) VALUES (?, ?, ?)",
            (trace_id, kind, text),
        )


def import_unified_rows(
    conn: sqlite3.Connection,
    trace_id: int,
    rows: Sequence[Dict[str, str]],
) -> Dict[str, Optional[float]]:
    max_gemm_tflops = None
    max_sdpa_tflops = None
    for source_row, row in enumerate(rows):
        name = as_text(first_value(row, ["name", "Name", "op_name"]))
        op_category = as_text(first_value(row, ["op category", "op_category", "category", "Categories"]))
        tflops_mean = as_float(first_value(row, ["TFLOPS/s_mean", "tflops_mean", "TFLOPS_mean"]))
        tflops_median = as_float(first_value(row, ["TFLOPS/s_median", "tflops_median", "TFLOPS_median"]))
        tflops_for_summary = tflops_mean if tflops_mean is not None else tflops_median
        if op_category and "gemm" in op_category.lower() and tflops_for_summary is not None:
            max_gemm_tflops = max(max_gemm_tflops or tflops_for_summary, tflops_for_summary)
        if op_category and "sdpa" in op_category.lower() and tflops_for_summary is not None:
            max_sdpa_tflops = max(max_sdpa_tflops or tflops_for_summary, tflops_for_summary)
        conn.execute(
            """
            INSERT INTO unified_perf_rows(
                trace_id, source_row, name, op_category, operation_count,
                kernel_time_sum_us, kernel_time_mean_us, kernel_time_median_us,
                kernel_time_std_us, kernel_time_min_us, kernel_time_max_us,
                op_duration_us, tflops_mean, tflops_median, tbs_mean, tbs_median,
                gflops, data_moved_mb, flops_per_byte, compute_spec,
                has_perf_model, overlap_pct, perf_params_json, kernel_details_json,
                raw_row_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trace_id,
                source_row,
                name,
                op_category,
                as_int(first_value(row, ["operation_count", "Count", "count"])),
                as_float(first_value(row, ["Kernel Time (us)_sum", "Kernel Time (µs)_sum", "total_direct_kernel_time_sum", "total_subtree_kernel_time_sum"])),
                as_float(first_value(row, ["Kernel Time (us)_mean", "Kernel Time (µs)_mean", "total_direct_kernel_time_mean", "total_subtree_kernel_time_mean"])),
                as_float(first_value(row, ["Kernel Time (us)_median", "Kernel Time (µs)_median", "total_direct_kernel_time_median", "total_subtree_kernel_time_median"])),
                as_float(first_value(row, ["Kernel Time (us)_std", "Kernel Time (µs)_std"])),
                as_float(first_value(row, ["Kernel Time (us)_min", "Kernel Time (µs)_min"])),
                as_float(first_value(row, ["Kernel Time (us)_max", "Kernel Time (µs)_max"])),
                as_float(first_value(row, ["op_duration_us", "CPU duration (us)", "CPU duration (µs)"])),
                tflops_mean,
                tflops_median,
                as_float(first_value(row, ["TB/s_mean", "tbs_mean"])),
                as_float(first_value(row, ["TB/s_median", "tbs_median"])),
                as_float(first_value(row, ["GFLOPS", "gflops"])),
                as_float(first_value(row, ["Data Moved (MB)", "data_moved_mb"])),
                as_float(first_value(row, ["FLOPs/Byte", "flops_per_byte"])),
                as_text(first_value(row, ["Compute Spec", "compute_spec"])),
                as_bool_int(first_value(row, ["has_perf_model", "Has Perf Model"])),
                as_float(first_value(row, ["overlap_pct", "Overlap (%)"])),
                as_text(first_value(row, ["perf_params", "Perf Params"])),
                as_text(first_value(row, ["kernel_details_summary", "trunc_kernel_details"])),
                json.dumps(row, sort_keys=True),
            ),
        )
        insert_search(conn, trace_id, "op", [name, op_category, first_value(row, ["kernel_details_summary", "trunc_kernel_details"])])
    return {"max_gemm_tflops": max_gemm_tflops, "max_sdpa_tflops": max_sdpa_tflops}


def import_kernel_summary_rows(
    conn: sqlite3.Connection,
    trace_id: int,
    rows: Sequence[Dict[str, str]],
) -> None:
    for row in rows:
        kernel_name = as_text(first_value(row, ["Kernel name", "kernel_name", "name"]))
        if not kernel_name:
            continue
        parent_op_name = as_text(first_value(row, ["Parent cpu_op", "parent_op_name", "Launcher"]))
        op_category = as_text(first_value(row, ["Parent op category", "op_category", "category"]))
        conn.execute(
            """
            INSERT INTO kernel_summary(
                trace_id, kernel_name, parent_op_name, op_category, stream, count,
                total_duration_us, mean_duration_us, median_duration_us,
                min_duration_us, max_duration_us, is_tensile, is_transpose,
                is_layout_conversion, raw_row_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trace_id,
                kernel_name,
                parent_op_name,
                op_category,
                as_int(first_value(row, ["stream", "Stream"])),
                as_int(first_value(row, ["Kernel duration (us)_count", "Kernel duration (µs)_count", "count"])),
                as_float(first_value(row, ["Kernel duration (us)_sum", "Kernel duration (µs)_sum", "total_us"])),
                as_float(first_value(row, ["Kernel duration (us)_mean", "Kernel duration (µs)_mean", "mean_us"])),
                as_float(first_value(row, ["Kernel duration (us)_median", "Kernel duration (µs)_median", "median_us"])),
                as_float(first_value(row, ["Kernel duration (us)_min", "Kernel duration (µs)_min", "min_us"])),
                as_float(first_value(row, ["Kernel duration (us)_max", "Kernel duration (µs)_max", "max_us"])),
                int("cijk" in kernel_name.lower() or "tensile" in kernel_name.lower()),
                int("transpose" in kernel_name.lower()),
                int("layout" in kernel_name.lower() or "permute" in kernel_name.lower()),
                json.dumps(row, sort_keys=True),
            ),
        )
        insert_search(conn, trace_id, "kernel", [kernel_name, parent_op_name, op_category])


def import_category_rows(
    conn: sqlite3.Connection,
    trace_id: int,
    rows: Sequence[Dict[str, str]],
) -> str:
    top_categories = []
    for row in rows:
        category = as_text(first_value(row, ["op category", "category", "Categories", "name"]))
        if not category:
            continue
        kernel_time = as_float(first_value(row, ["Kernel Time (us)_sum", "Kernel Time (µs)_sum", "total_direct_kernel_time_sum", "total_subtree_kernel_time_sum"]))
        conn.execute(
            """
            INSERT INTO op_category_rows(
                trace_id, category, operation_count, kernel_time_sum_us, percent, raw_row_json
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                trace_id,
                category,
                as_int(first_value(row, ["operation_count", "Count", "count"])),
                kernel_time,
                as_float(first_value(row, ["Percentage (%)", "percent", "Percent of total time (%)"])),
                json.dumps(row, sort_keys=True),
            ),
        )
        top_categories.append({"category": category, "kernel_time_sum_us": kernel_time or 0.0})
        insert_search(conn, trace_id, "category", [category])
    top_categories.sort(key=lambda item: item["kernel_time_sum_us"], reverse=True)
    return json.dumps(top_categories[:5], sort_keys=True)


def import_gpu_timeline_rows(
    conn: sqlite3.Connection,
    trace_id: int,
    rows: Sequence[Dict[str, str]],
) -> Optional[float]:
    total_duration_us = None
    for row in rows:
        metric_type = as_text(first_value(row, ["type", "metric"]))
        if not metric_type:
            continue
        time_ms = as_float(first_value(row, ["time ms", "time_ms"]))
        if metric_type == "total_time" and time_ms is not None:
            total_duration_us = time_ms * 1000.0
        conn.execute(
            """
            INSERT INTO gpu_timeline_rows(trace_id, type, time_ms, percent, raw_row_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                trace_id,
                metric_type,
                time_ms,
                as_float(first_value(row, ["percent", "Percentage (%)"])),
                json.dumps(row, sort_keys=True),
            ),
        )
        insert_search(conn, trace_id, "timeline", [metric_type])
    return total_duration_us


def ensure_trace_row(
    conn: sqlite3.Connection,
    trace_path: Optional[Path],
    report_dir: Path,
    root: Optional[Path],
) -> int:
    if trace_path is not None:
        return add_trace(conn, trace_path, root=root)
    synthetic_path = normalize_path(report_dir.resolve())
    now = utc_now()
    conn.execute(
        """
        INSERT INTO traces(path, rel_path, name, format, should_enrich, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(path) DO UPDATE SET updated_at = excluded.updated_at
        """,
        (synthetic_path, report_dir.name, report_dir.name, "tracelens_report_dir", 1, now, now),
    )
    row = conn.execute("SELECT id FROM traces WHERE path = ?", (synthetic_path,)).fetchone()
    return int(row["id"])


def import_report_dir(
    db_path: Path,
    report_dir: Path,
    trace_path: Optional[Path] = None,
    root: Optional[Path] = None,
) -> int:
    report_dir = report_dir.resolve()
    conn = connect(db_path)
    init_db(conn)
    trace_id = ensure_trace_row(conn, trace_path, report_dir, root)
    clear_trace_payload(conn, trace_id)

    sheet_rows = {
        "unified_perf_summary": read_csv_rows(report_dir / "unified_perf_summary.csv"),
        "kernel_summary": read_csv_rows(report_dir / "kernel_summary.csv"),
        "ops_summary_by_category": read_csv_rows(report_dir / "ops_summary_by_category.csv"),
        "gpu_timeline": read_csv_rows(report_dir / "gpu_timeline.csv"),
    }
    unified_summary = import_unified_rows(conn, trace_id, sheet_rows["unified_perf_summary"])
    import_kernel_summary_rows(conn, trace_id, sheet_rows["kernel_summary"])
    top_categories_json = import_category_rows(conn, trace_id, sheet_rows["ops_summary_by_category"])
    total_duration_us = import_gpu_timeline_rows(conn, trace_id, sheet_rows["gpu_timeline"])

    conn.execute(
        """
        INSERT INTO trace_summary(
            trace_id, total_duration_us, top_categories_json, max_gemm_tflops,
            max_sdpa_tflops, imported_at
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            trace_id,
            total_duration_us,
            top_categories_json,
            unified_summary["max_gemm_tflops"],
            unified_summary["max_sdpa_tflops"],
            utc_now(),
        ),
    )
    conn.execute(
        """
        INSERT INTO report_imports(trace_id, report_dir, imported_at, sheets_json)
        VALUES (?, ?, ?, ?)
        """,
        (
            trace_id,
            normalize_path(report_dir),
            utc_now(),
            json.dumps([name for name, rows in sheet_rows.items() if rows], sort_keys=True),
        ),
    )
    insert_search(conn, trace_id, "trace", [trace_path or report_dir, report_dir.name])
    conn.commit()
    conn.close()
    return trace_id


def generate_report_and_import(
    db_path: Path,
    trace_path: Path,
    report_dir: Optional[Path] = None,
    root: Optional[Path] = None,
    force: bool = False,
    enable_pseudo_ops: bool = False,
) -> int:
    if report_dir is None:
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", trace_path.name)
        report_dir = db_path.resolve().parent / "trace_index_reports" / safe_name
    report_dir = report_dir.resolve()
    unified_csv = report_dir / "unified_perf_summary.csv"
    if force or not unified_csv.exists():
        report_dir.mkdir(parents=True, exist_ok=True)
        from TraceLens.Reporting.generate_perf_report_pytorch import (  # noqa: PLC0415
            generate_perf_report_pytorch,
        )

        generate_perf_report_pytorch(
            profile_json_path=str(trace_path),
            output_xlsx_path=None,
            output_csvs_dir=str(report_dir),
            kernel_summary=True,
            include_first_occurrence_time=True,
            enable_pseudo_ops=enable_pseudo_ops,
        )
    return import_report_dir(db_path, report_dir, trace_path=trace_path, root=root)


def search_index(db_path: Path, terms: str, limit: int = 50) -> List[Dict[str, Any]]:
    conn = connect(db_path)
    init_db(conn)
    rows = conn.execute(
        """
        SELECT t.id AS trace_id, t.rel_path, s.kind,
               snippet(trace_search_FTS5, 2, '[', ']', '...', 12) AS hit
        FROM trace_search_FTS5 s
        JOIN traces t ON t.id = s.trace_id
        WHERE trace_search_FTS5 MATCH ?
        LIMIT ?
        """,
        (terms, limit),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def is_read_only_sql(sql: str) -> bool:
    stripped = sql.strip().lower()
    if not stripped:
        return False
    if ";" in stripped.rstrip(";"):
        return False
    return stripped.startswith(("select", "with", "pragma"))


def execute_read_query(
    db_path: Path,
    sql: str,
    params: Optional[Sequence[Any]] = None,
    limit: int = 500,
) -> List[Dict[str, Any]]:
    if not is_read_only_sql(sql):
        raise ValueError("only a single read-only SELECT/WITH/PRAGMA statement is allowed")
    conn = connect(db_path)
    init_db(conn)
    conn.execute("PRAGMA query_only=ON")
    rows = conn.execute(sql, params or ()).fetchmany(limit)
    conn.close()
    return [dict(row) for row in rows]
