###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""SQLite TraceIndex backend."""

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from TraceLens.TraceIndex.models import SearchHit, TraceRecord, TraceReport
from TraceLens.TraceIndex.store import TraceIndexStore
from TraceLens.TraceIndex.utils import (
    as_bool_int,
    as_duration_us,
    as_float,
    as_int,
    as_optional_bool_int,
    as_text,
    first_value,
    parse_repr,
    search_text,
    to_json,
    utc_now,
)


def is_read_only_sql(sql: str) -> bool:
    stripped = sql.strip().lower()
    if not stripped:
        return False
    if ";" in stripped.rstrip(";"):
        return False
    return stripped.startswith(("select", "with", "pragma"))


def matches_op_category(value: Optional[str], family: str) -> bool:
    category = (value or "").strip().upper()
    return category == family or category.startswith(family + "_")


class SQLiteTraceIndexStore(TraceIndexStore):
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn = self._connect(db_path)

    def _connect(self, db_path: Path) -> sqlite3.Connection:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path), timeout=60)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=60000")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def init_schema(self) -> None:
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS traces (
                id INTEGER PRIMARY KEY,
                tracelens_id TEXT UNIQUE,
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
                gpu_total_ms REAL,
                gpu_compute_pct REAL,
                gpu_idle_pct REAL,
                gpu_exposed_comm_pct REAL,
                gpu_exposed_memcpy_pct REAL,
                gpu_total_comm_pct REAL,
                gpu_total_memcpy_pct REAL,
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
                excel_path TEXT,
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
                gpu_kernel_pct REAL,
                perf_params_json TEXT,
                kernel_details_json TEXT,
                raw_row_json TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_trace_index_unified_trace ON unified_perf_rows(trace_id);
            CREATE INDEX IF NOT EXISTS idx_trace_index_unified_category ON unified_perf_rows(op_category);
            CREATE INDEX IF NOT EXISTS idx_trace_index_unified_name ON unified_perf_rows(name);

            CREATE TABLE IF NOT EXISTS op_kernels (
                id INTEGER PRIMARY KEY,
                unified_row_id INTEGER NOT NULL REFERENCES unified_perf_rows(id) ON DELETE CASCADE,
                name TEXT NOT NULL,
                stream INTEGER,
                count INTEGER,
                total_duration_us REAL,
                mean_duration_us REAL,
                median_duration_us REAL,
                min_duration_us REAL,
                max_duration_us REAL
            );

            CREATE INDEX IF NOT EXISTS idx_trace_index_op_kernels_unified ON op_kernels(unified_row_id);
            CREATE INDEX IF NOT EXISTS idx_trace_index_op_kernels_name ON op_kernels(name);

            CREATE TABLE IF NOT EXISTS gemm_perf (
                unified_row_id INTEGER PRIMARY KEY REFERENCES unified_perf_rows(id) ON DELETE CASCADE,
                "M" INTEGER,
                "N" INTEGER,
                "K" INTEGER,
                "B" INTEGER,
                bias INTEGER,
                stride_A TEXT,
                stride_B TEXT,
                dtype_A_B TEXT,
                transpose TEXT
            );

            CREATE TABLE IF NOT EXISTS sdpa_perf (
                unified_row_id INTEGER PRIMARY KEY REFERENCES unified_perf_rows(id) ON DELETE CASCADE,
                "B" INTEGER,
                N_Q INTEGER,
                H_Q INTEGER,
                N_KV INTEGER,
                H_KV INTEGER,
                d_h_qk INTEGER,
                d_h_v INTEGER,
                q_stride TEXT,
                k_stride TEXT,
                v_stride TEXT,
                dropout REAL,
                causal INTEGER,
                flash_impl INTEGER,
                dtype_A_B TEXT
            );

            CREATE TABLE IF NOT EXISTS conv_perf (
                unified_row_id INTEGER PRIMARY KEY REFERENCES unified_perf_rows(id) ON DELETE CASCADE,
                "convNd" TEXT,
                input_shape TEXT,
                filter_shape TEXT,
                dtype_input_weight TEXT,
                input_stride TEXT,
                weight_stride TEXT,
                bias INTEGER,
                stride TEXT,
                padding TEXT,
                dilation TEXT,
                transposed_conv INTEGER,
                output_padding TEXT,
                groups INTEGER
            );

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

            CREATE VIRTUAL TABLE IF NOT EXISTS trace_search_FTS5 USING fts5(
                trace_id UNINDEXED,
                kind UNINDEXED,
                text,
                tokenize='unicode61'
            );
            """)
        self.conn.commit()

    def upsert_trace(self, trace: TraceRecord) -> int:
        now = utc_now()
        by_path = self.conn.execute(
            "SELECT id FROM traces WHERE path = ?", (trace.path,)
        ).fetchone()
        by_tracelens_id = (
            self.conn.execute(
                "SELECT id FROM traces WHERE tracelens_id = ?",
                (trace.tracelens_id,),
            ).fetchone()
            if trace.tracelens_id
            else None
        )
        if (
            by_path is not None
            and by_tracelens_id is not None
            and by_path["id"] != by_tracelens_id["id"]
        ):
            raise ValueError(
                "trace_path and tracelens_id identify different catalog rows"
            )
        existing = by_tracelens_id or by_path
        values = (
            trace.tracelens_id,
            trace.root,
            trace.path,
            trace.rel_path,
            trace.name,
            trace.size_bytes,
            trace.md5,
            trace.format,
            trace.rank,
            trace.top_dir,
            trace.parent_rel,
            int(trace.should_enrich),
            trace.skip_reason,
        )
        if existing is None:
            cursor = self.conn.execute(
                """
                INSERT INTO traces(
                    tracelens_id, root, path, rel_path, name, size_bytes, md5,
                    format, rank, top_dir, parent_rel, should_enrich, skip_reason,
                    created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                values + (now, now),
            )
            trace_id = int(cursor.lastrowid)
        else:
            trace_id = int(existing["id"])
            self.conn.execute(
                """
                UPDATE traces SET
                    tracelens_id = COALESCE(?, tracelens_id),
                    root = ?,
                    path = ?,
                    rel_path = ?,
                    name = ?,
                    size_bytes = ?,
                    md5 = COALESCE(?, md5),
                    format = ?,
                    rank = ?,
                    top_dir = ?,
                    parent_rel = ?,
                    should_enrich = ?,
                    skip_reason = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                values + (now, trace_id),
            )
        self.conn.commit()
        return trace_id

    def import_report(self, trace_id: int, report: TraceReport) -> None:
        self._clear_trace_payload(trace_id)
        self._import_unified_rows(
            trace_id, report.sheets.get("unified_perf_summary", [])
        )
        self._import_category_rows(
            trace_id, report.sheets.get("ops_summary_by_category", [])
        )
        self._import_gpu_mix(trace_id, report.sheets.get("gpu_timeline", []))
        self.conn.execute(
            """
            INSERT INTO report_imports(
                trace_id, report_dir, excel_path, imported_at, sheets_json
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                trace_id,
                report.report_dir,
                report.excel_path,
                utc_now(),
                json.dumps(
                    [name for name, rows in report.sheets.items() if rows],
                    sort_keys=True,
                ),
            ),
        )
        self._insert_search(trace_id, "trace", [report.report_dir])
        self.conn.commit()

    def search(self, terms: str, limit: int = 50) -> List[SearchHit]:
        rows = self.conn.execute(
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
        return [
            SearchHit(
                trace_id=int(row["trace_id"]),
                rel_path=row["rel_path"],
                kind=row["kind"],
                hit=row["hit"],
            )
            for row in rows
        ]

    def execute_read_query(
        self,
        sql: str,
        params: Optional[Sequence[Any]] = None,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        """Run one caller-supplied read-only SELECT/WITH/PRAGMA against the catalog."""
        if not is_read_only_sql(sql):
            raise ValueError(
                "only a single read-only SELECT/WITH/PRAGMA statement is allowed"
            )
        self.conn.execute("PRAGMA query_only=ON")
        # Intentional dynamic SQL: rejected above unless read-only; query_only is set.
        # codeql[py/sql-injection]
        rows = self.conn.execute(sql, params or ()).fetchmany(limit)
        return [dict(row) for row in rows]

    def close(self) -> None:
        self.conn.close()

    def _clear_trace_payload(self, trace_id: int) -> None:
        satellite_sql = (
            "DELETE FROM %s WHERE unified_row_id IN "
            "(SELECT id FROM unified_perf_rows WHERE trace_id = ?)"
        )
        for table in ("op_kernels", "gemm_perf", "sdpa_perf", "conv_perf"):
            self.conn.execute(satellite_sql % table, (trace_id,))
        for table in (
            "report_imports",
            "op_category_rows",
            "unified_perf_rows",
        ):
            self.conn.execute("DELETE FROM %s WHERE trace_id = ?" % table, (trace_id,))
        self.conn.execute(
            """
            UPDATE traces SET
                gpu_total_ms = NULL,
                gpu_compute_pct = NULL,
                gpu_idle_pct = NULL,
                gpu_exposed_comm_pct = NULL,
                gpu_exposed_memcpy_pct = NULL,
                gpu_total_comm_pct = NULL,
                gpu_total_memcpy_pct = NULL
            WHERE id = ?
            """,
            (trace_id,),
        )
        self.conn.execute(
            "DELETE FROM trace_search_FTS5 WHERE trace_id = ?", (trace_id,)
        )

    def _insert_search(self, trace_id: int, kind: str, parts: Iterable[Any]) -> None:
        text = search_text(*parts)
        if text:
            self.conn.execute(
                "INSERT INTO trace_search_FTS5(trace_id, kind, text) VALUES (?, ?, ?)",
                (trace_id, kind, text),
            )

    def _import_unified_rows(
        self,
        trace_id: int,
        rows: Sequence[Dict[str, str]],
    ) -> None:
        for source_row, row in enumerate(rows):
            name = as_text(first_value(row, ["name", "Name", "op_name"]))
            op_category = as_text(
                first_value(
                    row, ["op category", "op_category", "category", "Categories"]
                )
            )
            tflops_mean = as_float(
                first_value(row, ["TFLOPS/s_mean", "tflops_mean", "TFLOPS_mean"])
            )
            tflops_median = as_float(
                first_value(row, ["TFLOPS/s_median", "tflops_median", "TFLOPS_median"])
            )
            params = parse_repr(first_value(row, ["perf_params", "Perf Params"]))
            kernel_details = parse_repr(
                first_value(row, ["kernel_details_summary", "trunc_kernel_details"])
            )
            cursor = self.conn.execute(
                """
                INSERT INTO unified_perf_rows(
                    trace_id, source_row, name, op_category, operation_count,
                    kernel_time_sum_us, kernel_time_mean_us, kernel_time_median_us,
                    kernel_time_std_us, kernel_time_min_us, kernel_time_max_us,
                    op_duration_us, tflops_mean, tflops_median, tbs_mean, tbs_median,
                    gflops, data_moved_mb, flops_per_byte, compute_spec,
                    has_perf_model, overlap_pct, gpu_kernel_pct, perf_params_json,
                    kernel_details_json, raw_row_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trace_id,
                    source_row,
                    name,
                    op_category,
                    as_int(first_value(row, ["operation_count", "Count", "count"])),
                    as_duration_us(
                        row,
                        [
                            "Kernel Time (us)_sum",
                            "Kernel Time (µs)_sum",
                            "total_direct_kernel_time_sum",
                            "total_subtree_kernel_time_sum",
                        ],
                        [
                            "total_direct_kernel_time_ms",
                            "total_subtree_kernel_time_ms",
                        ],
                    ),
                    as_float(
                        first_value(
                            row,
                            [
                                "Kernel Time (us)_mean",
                                "Kernel Time (µs)_mean",
                                "total_direct_kernel_time_mean",
                                "total_subtree_kernel_time_mean",
                            ],
                        )
                    ),
                    as_float(
                        first_value(
                            row,
                            [
                                "Kernel Time (us)_median",
                                "Kernel Time (µs)_median",
                                "total_direct_kernel_time_median",
                                "total_subtree_kernel_time_median",
                            ],
                        )
                    ),
                    as_float(
                        first_value(
                            row, ["Kernel Time (us)_std", "Kernel Time (µs)_std"]
                        )
                    ),
                    as_float(
                        first_value(
                            row, ["Kernel Time (us)_min", "Kernel Time (µs)_min"]
                        )
                    ),
                    as_float(
                        first_value(
                            row, ["Kernel Time (us)_max", "Kernel Time (µs)_max"]
                        )
                    ),
                    as_float(
                        first_value(
                            row,
                            [
                                "op_duration_us",
                                "CPU duration (us)",
                                "CPU duration (µs)",
                            ],
                        )
                    ),
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
                    as_float(
                        first_value(
                            row,
                            ["Percentage (%)", "gpu_kernel_pct", "percent"],
                        )
                    ),
                    to_json(params),
                    to_json(kernel_details),
                    to_json(dict(row)),
                ),
            )
            unified_row_id = int(cursor.lastrowid)
            self._import_kernels_from_details(trace_id, unified_row_id, kernel_details)
            self._maybe_insert_gemm(unified_row_id, op_category, params)
            self._maybe_insert_sdpa(unified_row_id, op_category, params)
            self._maybe_insert_conv(unified_row_id, op_category, params)
            self._insert_search(
                trace_id,
                "op",
                [
                    name,
                    op_category,
                    first_value(
                        row, ["kernel_details_summary", "trunc_kernel_details"]
                    ),
                ],
            )

    def _import_kernels_from_details(
        self,
        trace_id: int,
        unified_row_id: int,
        kernel_details: Any,
    ) -> None:
        if not isinstance(kernel_details, list):
            return
        for detail in kernel_details:
            if not isinstance(detail, dict):
                continue
            name = as_text(detail.get("name") or detail.get("Kernel name"))
            if not name:
                continue
            self.conn.execute(
                """
                INSERT INTO op_kernels(
                    unified_row_id, name, stream, count,
                    total_duration_us, mean_duration_us,
                    median_duration_us, min_duration_us, max_duration_us
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    unified_row_id,
                    name,
                    as_int(detail.get("stream")),
                    as_int(detail.get("count")),
                    as_float(detail.get("total_duration_us")),
                    as_float(detail.get("mean_duration_us")),
                    as_float(detail.get("median_duration_us")),
                    as_float(detail.get("min_duration_us")),
                    as_float(detail.get("max_duration_us")),
                ),
            )
            self._insert_search(trace_id, "kernel", [name])

    def _maybe_insert_gemm(
        self,
        unified_row_id: int,
        op_category: Optional[str],
        params: Any,
    ) -> None:
        if not matches_op_category(op_category, "GEMM"):
            return
        params = params if isinstance(params, dict) else {}
        self.conn.execute(
            """
            INSERT OR REPLACE INTO gemm_perf(
                unified_row_id, "M", "N", "K", "B", bias,
                stride_A, stride_B, dtype_A_B, transpose
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                unified_row_id,
                as_int(params.get("M")),
                as_int(params.get("N")),
                as_int(params.get("K")),
                as_int(params.get("B")),
                as_optional_bool_int(params.get("bias")),
                to_json(params.get("stride_A")),
                to_json(params.get("stride_B")),
                to_json(params.get("dtype_A_B")),
                to_json(params.get("transpose")),
            ),
        )

    def _maybe_insert_sdpa(
        self,
        unified_row_id: int,
        op_category: Optional[str],
        params: Any,
    ) -> None:
        if not matches_op_category(op_category, "SDPA"):
            return
        params = params if isinstance(params, dict) else {}
        self.conn.execute(
            """
            INSERT OR REPLACE INTO sdpa_perf(
                unified_row_id, "B", N_Q, H_Q, N_KV, H_KV, d_h_qk, d_h_v,
                q_stride, k_stride, v_stride, dropout, causal, flash_impl, dtype_A_B
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                unified_row_id,
                as_int(params.get("B")),
                as_int(params.get("N_Q")),
                as_int(params.get("H_Q")),
                as_int(params.get("N_KV")),
                as_int(params.get("H_KV")),
                as_int(params.get("d_h_qk")),
                as_int(params.get("d_h_v")),
                to_json(params.get("q_stride")),
                to_json(params.get("k_stride")),
                to_json(params.get("v_stride")),
                as_float(params.get("dropout")),
                as_optional_bool_int(params.get("causal")),
                as_optional_bool_int(params.get("flash_impl")),
                to_json(params.get("dtype_A_B")),
            ),
        )

    def _maybe_insert_conv(
        self,
        unified_row_id: int,
        op_category: Optional[str],
        params: Any,
    ) -> None:
        if not matches_op_category(op_category, "CONV"):
            return
        params = params if isinstance(params, dict) else {}
        self.conn.execute(
            """
            INSERT OR REPLACE INTO conv_perf(
                unified_row_id, "convNd", input_shape, filter_shape,
                dtype_input_weight, input_stride, weight_stride, bias,
                stride, padding, dilation, transposed_conv, output_padding, groups
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                unified_row_id,
                as_text(params.get("convNd")),
                to_json(params.get("input_shape")),
                to_json(params.get("filter_shape")),
                to_json(params.get("dtype_input_weight")),
                to_json(params.get("input_stride")),
                to_json(params.get("weight_stride")),
                as_optional_bool_int(params.get("bias")),
                to_json(params.get("stride")),
                to_json(params.get("padding")),
                to_json(params.get("dilation")),
                as_optional_bool_int(params.get("transposed_conv")),
                to_json(params.get("output_padding")),
                as_int(params.get("groups")),
            ),
        )

    def _import_category_rows(
        self,
        trace_id: int,
        rows: Sequence[Dict[str, str]],
    ) -> None:
        for row in rows:
            category = as_text(
                first_value(row, ["op category", "category", "Categories", "name"])
            )
            if not category:
                continue
            kernel_time = as_duration_us(
                row,
                [
                    "Kernel Time (us)_sum",
                    "Kernel Time (µs)_sum",
                    "total_direct_kernel_time_sum",
                    "total_subtree_kernel_time_sum",
                ],
                [
                    "total_direct_kernel_time_ms",
                    "total_subtree_kernel_time_ms",
                ],
            )
            self.conn.execute(
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
                    as_float(
                        first_value(
                            row,
                            ["Percentage (%)", "percent", "Percent of total time (%)"],
                        )
                    ),
                    json.dumps(row, sort_keys=True),
                ),
            )
            self._insert_search(trace_id, "category", [category])

    def _import_gpu_mix(
        self,
        trace_id: int,
        rows: Sequence[Dict[str, str]],
    ) -> None:
        values: Dict[str, Optional[float]] = {
            "gpu_total_ms": None,
            "gpu_compute_pct": None,
            "gpu_idle_pct": None,
            "gpu_exposed_comm_pct": None,
            "gpu_exposed_memcpy_pct": None,
            "gpu_total_comm_pct": None,
            "gpu_total_memcpy_pct": None,
        }
        metric_columns = {
            "total_time": "gpu_total_ms",
            "computation_time": "gpu_compute_pct",
            "idle_time": "gpu_idle_pct",
            "exposed_comm_time": "gpu_exposed_comm_pct",
            "exposed_memcpy_time": "gpu_exposed_memcpy_pct",
            "total_comm_time": "gpu_total_comm_pct",
            "total_memcpy_time": "gpu_total_memcpy_pct",
        }
        for row in rows:
            metric_type = as_text(first_value(row, ["type", "Type", "metric"]))
            column = metric_columns.get(metric_type or "")
            if not column:
                continue
            source_fields = (
                ["time ms", "time_ms", "time"]
                if column == "gpu_total_ms"
                else ["percent", "Percentage (%)"]
            )
            values[column] = as_float(first_value(row, source_fields))
        self.conn.execute(
            """
            UPDATE traces SET
                gpu_total_ms = ?,
                gpu_compute_pct = ?,
                gpu_idle_pct = ?,
                gpu_exposed_comm_pct = ?,
                gpu_exposed_memcpy_pct = ?,
                gpu_total_comm_pct = ?,
                gpu_total_memcpy_pct = ?
            WHERE id = ?
            """,
            (
                values["gpu_total_ms"],
                values["gpu_compute_pct"],
                values["gpu_idle_pct"],
                values["gpu_exposed_comm_pct"],
                values["gpu_exposed_memcpy_pct"],
                values["gpu_total_comm_pct"],
                values["gpu_total_memcpy_pct"],
                trace_id,
            ),
        )
