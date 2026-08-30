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
    kernel_flags,
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

            CREATE TABLE IF NOT EXISTS op_kernels (
                id INTEGER PRIMARY KEY,
                trace_id INTEGER NOT NULL REFERENCES traces(id) ON DELETE CASCADE,
                unified_row_id INTEGER REFERENCES unified_perf_rows(id) ON DELETE CASCADE,
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
                library TEXT,
                is_tensile INTEGER NOT NULL DEFAULT 0,
                is_transpose INTEGER NOT NULL DEFAULT 0,
                is_layout_conversion INTEGER NOT NULL DEFAULT 0,
                details_json TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_trace_index_op_kernels_trace ON op_kernels(trace_id);
            CREATE INDEX IF NOT EXISTS idx_trace_index_op_kernels_unified ON op_kernels(unified_row_id);
            CREATE INDEX IF NOT EXISTS idx_trace_index_op_kernels_name ON op_kernels(kernel_name);
            CREATE INDEX IF NOT EXISTS idx_trace_index_op_kernels_tensile ON op_kernels(is_tensile);

            CREATE TABLE IF NOT EXISTS gemm_perf (
                unified_row_id INTEGER PRIMARY KEY REFERENCES unified_perf_rows(id) ON DELETE CASCADE,
                trace_id INTEGER NOT NULL REFERENCES traces(id) ON DELETE CASCADE,
                m INTEGER,
                n INTEGER,
                k INTEGER,
                batch INTEGER,
                dtype TEXT,
                transpose TEXT,
                tflops_mean REAL,
                tflops_median REAL
            );

            CREATE INDEX IF NOT EXISTS idx_trace_index_gemm_trace ON gemm_perf(trace_id);
            CREATE INDEX IF NOT EXISTS idx_trace_index_gemm_tflops ON gemm_perf(tflops_mean);

            CREATE TABLE IF NOT EXISTS sdpa_perf (
                unified_row_id INTEGER PRIMARY KEY REFERENCES unified_perf_rows(id) ON DELETE CASCADE,
                trace_id INTEGER NOT NULL REFERENCES traces(id) ON DELETE CASCADE,
                batch INTEGER,
                heads INTEGER,
                seq_q INTEGER,
                seq_kv INTEGER,
                head_dim INTEGER,
                dtype TEXT,
                causal INTEGER,
                tflops_mean REAL,
                tflops_median REAL
            );

            CREATE INDEX IF NOT EXISTS idx_trace_index_sdpa_trace ON sdpa_perf(trace_id);

            CREATE TABLE IF NOT EXISTS conv_perf (
                unified_row_id INTEGER PRIMARY KEY REFERENCES unified_perf_rows(id) ON DELETE CASCADE,
                trace_id INTEGER NOT NULL REFERENCES traces(id) ON DELETE CASCADE,
                conv_nd TEXT,
                input_shape_json TEXT,
                filter_shape_json TEXT,
                input_channels INTEGER,
                output_channels INTEGER,
                groups INTEGER,
                kernel_h INTEGER,
                kernel_w INTEGER,
                is_depthwise INTEGER,
                is_transposed_conv INTEGER
            );

            CREATE INDEX IF NOT EXISTS idx_trace_index_conv_trace ON conv_perf(trace_id);
            CREATE INDEX IF NOT EXISTS idx_trace_index_conv_depthwise ON conv_perf(is_depthwise);

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
            """)
        self.conn.commit()

    def upsert_trace(self, trace: TraceRecord) -> int:
        now = utc_now()
        self.conn.execute(
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
                now,
                now,
            ),
        )
        row = self.conn.execute(
            "SELECT id FROM traces WHERE path = ?", (trace.path,)
        ).fetchone()
        self.conn.commit()
        return int(row["id"])

    def import_report(self, trace_id: int, report: TraceReport) -> None:
        self._clear_trace_payload(trace_id)
        unified_summary = self._import_unified_rows(
            trace_id, report.sheets.get("unified_perf_summary", [])
        )
        top_categories_json = self._import_category_rows(
            trace_id, report.sheets.get("ops_summary_by_category", [])
        )
        total_duration_us = self._import_gpu_timeline_rows(
            trace_id, report.sheets.get("gpu_timeline", [])
        )

        self.conn.execute(
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
        self.conn.execute(
            """
            INSERT INTO report_imports(trace_id, report_dir, imported_at, sheets_json)
            VALUES (?, ?, ?, ?)
            """,
            (
                trace_id,
                report.report_dir,
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
        rows = self.conn.execute(sql, params or ()).fetchmany(limit)  # codeql[py/sql-injection]
        return [dict(row) for row in rows]

    def close(self) -> None:
        self.conn.close()

    def _clear_trace_payload(self, trace_id: int) -> None:
        for table in (
            "report_imports",
            "gemm_perf",
            "sdpa_perf",
            "conv_perf",
            "op_kernels",
            "op_category_rows",
            "gpu_timeline_rows",
            "trace_summary",
            "unified_perf_rows",
        ):
            self.conn.execute("DELETE FROM %s WHERE trace_id = ?" % table, (trace_id,))
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
    ) -> Dict[str, Optional[float]]:
        max_gemm_tflops = None
        max_sdpa_tflops = None
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
            tflops_for_summary = (
                tflops_mean if tflops_mean is not None else tflops_median
            )
            if (
                op_category
                and "gemm" in op_category.lower()
                and tflops_for_summary is not None
            ):
                max_gemm_tflops = max(
                    max_gemm_tflops or tflops_for_summary, tflops_for_summary
                )
            if (
                op_category
                and "sdpa" in op_category.lower()
                and tflops_for_summary is not None
            ):
                max_sdpa_tflops = max(
                    max_sdpa_tflops or tflops_for_summary, tflops_for_summary
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
                    to_json(params),
                    to_json(kernel_details),
                    to_json(dict(row)),
                ),
            )
            unified_row_id = int(cursor.lastrowid)
            self._import_kernels_from_details(
                trace_id, unified_row_id, name, op_category, kernel_details
            )
            self._maybe_insert_gemm(
                trace_id, unified_row_id, params, tflops_mean, tflops_median
            )
            self._maybe_insert_sdpa(
                trace_id, unified_row_id, params, tflops_mean, tflops_median
            )
            self._maybe_insert_conv(trace_id, unified_row_id, params)
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
        return {"max_gemm_tflops": max_gemm_tflops, "max_sdpa_tflops": max_sdpa_tflops}

    def _import_kernels_from_details(
        self,
        trace_id: int,
        unified_row_id: int,
        parent_op_name: Optional[str],
        op_category: Optional[str],
        kernel_details: Any,
    ) -> None:
        if not isinstance(kernel_details, list):
            return
        for detail in kernel_details:
            if not isinstance(detail, dict):
                continue
            kernel_name = as_text(detail.get("name") or detail.get("Kernel name"))
            if not kernel_name:
                continue
            library, is_tensile, is_transpose, is_layout = kernel_flags(kernel_name)
            self.conn.execute(
                """
                INSERT INTO op_kernels(
                    trace_id, unified_row_id, kernel_name, parent_op_name, op_category,
                    stream, count, total_duration_us, mean_duration_us,
                    median_duration_us, min_duration_us, max_duration_us, library,
                    is_tensile, is_transpose, is_layout_conversion, details_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trace_id,
                    unified_row_id,
                    kernel_name,
                    parent_op_name,
                    op_category,
                    as_int(detail.get("stream")),
                    as_int(detail.get("count")),
                    as_float(detail.get("total_duration_us")),
                    as_float(detail.get("mean_duration_us")),
                    as_float(detail.get("median_duration_us")),
                    as_float(detail.get("min_duration_us")),
                    as_float(detail.get("max_duration_us")),
                    library,
                    is_tensile,
                    is_transpose,
                    is_layout,
                    to_json(detail),
                ),
            )
            self._insert_search(
                trace_id,
                "kernel",
                [kernel_name, library, parent_op_name, op_category],
            )

    def _maybe_insert_gemm(
        self,
        trace_id: int,
        unified_row_id: int,
        params: Any,
        tflops_mean: Optional[float],
        tflops_median: Optional[float],
    ) -> None:
        if not isinstance(params, dict):
            return
        if not {"M", "N", "K"}.intersection(params):
            return
        self.conn.execute(
            """
            INSERT OR REPLACE INTO gemm_perf(
                unified_row_id, trace_id, m, n, k, batch, dtype, transpose,
                tflops_mean, tflops_median
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                unified_row_id,
                trace_id,
                as_int(params.get("M")),
                as_int(params.get("N")),
                as_int(params.get("K")),
                as_int(params.get("B")),
                (
                    str(params.get("dtype_A_B"))
                    if params.get("dtype_A_B") is not None
                    else None
                ),
                (
                    str(params.get("transpose"))
                    if params.get("transpose") is not None
                    else None
                ),
                tflops_mean,
                tflops_median,
            ),
        )

    def _maybe_insert_sdpa(
        self,
        trace_id: int,
        unified_row_id: int,
        params: Any,
        tflops_mean: Optional[float],
        tflops_median: Optional[float],
    ) -> None:
        if not isinstance(params, dict):
            return
        if not {"N_Q", "N_KV", "d_h_qk", "d_h_v"}.intersection(params):
            return
        self.conn.execute(
            """
            INSERT OR REPLACE INTO sdpa_perf(
                unified_row_id, trace_id, batch, heads, seq_q, seq_kv,
                head_dim, dtype, causal, tflops_mean, tflops_median
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                unified_row_id,
                trace_id,
                as_int(params.get("B")),
                as_int(params.get("H_Q")),
                as_int(params.get("N_Q")),
                as_int(params.get("N_KV")),
                as_int(params.get("d_h_qk") or params.get("d_h_v")),
                (
                    str(params.get("dtype_A_B"))
                    if params.get("dtype_A_B") is not None
                    else None
                ),
                as_optional_bool_int(params.get("causal")),
                tflops_mean,
                tflops_median,
            ),
        )

    def _maybe_insert_conv(
        self,
        trace_id: int,
        unified_row_id: int,
        params: Any,
    ) -> None:
        if not isinstance(params, dict):
            return
        if "convNd" not in params and "filter_shape" not in params:
            return
        input_shape = params.get("input_shape")
        filter_shape = params.get("filter_shape")
        groups = as_int(params.get("groups")) or 1
        input_channels = None
        output_channels = None
        kernel_h = None
        kernel_w = None
        if isinstance(input_shape, (list, tuple)) and len(input_shape) >= 2:
            input_channels = as_int(input_shape[1])
        if isinstance(filter_shape, (list, tuple)) and len(filter_shape) >= 4:
            output_channels = as_int(filter_shape[0])
            kernel_h = as_int(filter_shape[-2])
            kernel_w = as_int(filter_shape[-1])
        filter_c_per_group = None
        if isinstance(filter_shape, (list, tuple)) and len(filter_shape) >= 2:
            filter_c_per_group = as_int(filter_shape[1])
        is_depthwise = int(
            bool(input_channels)
            and groups == input_channels
            and filter_c_per_group == 1
            and bool(output_channels)
            and output_channels % input_channels == 0
        )
        self.conn.execute(
            """
            INSERT OR REPLACE INTO conv_perf(
                unified_row_id, trace_id, conv_nd, input_shape_json, filter_shape_json,
                input_channels, output_channels, groups, kernel_h, kernel_w,
                is_depthwise, is_transposed_conv
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                unified_row_id,
                trace_id,
                params.get("convNd"),
                to_json(input_shape),
                to_json(filter_shape),
                input_channels,
                output_channels,
                groups,
                kernel_h,
                kernel_w,
                is_depthwise,
                as_optional_bool_int(params.get("transposed_conv")),
            ),
        )

    def _import_category_rows(
        self,
        trace_id: int,
        rows: Sequence[Dict[str, str]],
    ) -> str:
        top_categories = []
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
            top_categories.append(
                {"category": category, "kernel_time_sum_us": kernel_time or 0.0}
            )
            self._insert_search(trace_id, "category", [category])
        top_categories.sort(key=lambda item: item["kernel_time_sum_us"], reverse=True)
        return json.dumps(top_categories[:5], sort_keys=True)

    def _import_gpu_timeline_rows(
        self,
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
            self.conn.execute(
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
            self._insert_search(trace_id, "timeline", [metric_type])
        return total_duration_us
