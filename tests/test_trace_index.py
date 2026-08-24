###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import csv
import json
from pathlib import Path

import pytest

from TraceLens.TraceIndex.core import (
    execute_read_query,
    import_report_dir,
    scan_traces,
    search_index,
)
from TraceLens.TraceIndex.importer import (
    import_report_dir as import_report_dir_with_store,
)
from TraceLens.TraceIndex.scanner import scan_traces as scan_traces_with_store
from TraceLens.TraceIndex.sqlite_store import SQLiteTraceIndexStore

FIXTURES = Path(__file__).resolve().parent / "traces"
TRAINING_REPORT_DIR = (
    FIXTURES / "mi300" / "Qwen_Qwen1.5-0.5B-Chat__1016005_perf_report_csvs"
)
INFERENCE_REPORT_DIR = FIXTURES / "inference" / "sglang_decode" / "perf_csvs"


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_trace_index_scan_import_and_search(tmp_path):
    db_path = tmp_path / "trace_index.sqlite"
    trace_root = tmp_path / "traces"
    trace_path = trace_root / "model_a" / "rank0_trace.json"
    trace_path.parent.mkdir(parents=True)
    trace_path.write_text(json.dumps({"traceEvents": []}), encoding="utf-8")

    assert scan_traces(db_path, trace_root) == 1

    report_dir = tmp_path / "reports" / "trace_1"
    write_csv(
        report_dir / "unified_perf_summary.csv",
        [
            {
                "name": "aten::mm",
                "op category": "GEMM",
                "operation_count": "2",
                "Kernel Time (us)_sum": "123.5",
                "TFLOPS/s_mean": "98.1",
                "kernel_details_summary": "Cijk_test_kernel",
            }
        ],
    )
    write_csv(
        report_dir / "kernel_summary.csv",
        [
            {
                "Kernel name": "Cijk_test_kernel",
                "Parent cpu_op": "aten::mm",
                "Parent op category": "GEMM",
                "Kernel duration (us)_count": "2",
                "Kernel duration (us)_sum": "123.5",
            }
        ],
    )
    write_csv(
        report_dir / "ops_summary_by_category.csv",
        [
            {
                "op category": "GEMM",
                "operation_count": "2",
                "Kernel Time (us)_sum": "123.5",
                "Percentage (%)": "80.0",
            }
        ],
    )
    write_csv(
        report_dir / "gpu_timeline.csv",
        [
            {"type": "total_time", "time ms": "1.0", "percent": "100.0"},
            {"type": "computation_time", "time ms": "0.8", "percent": "80.0"},
        ],
    )

    trace_id = import_report_dir(
        db_path, report_dir, trace_path=trace_path, root=trace_root
    )
    assert trace_id == 1

    rows = execute_read_query(
        db_path,
        "SELECT name, op_category, kernel_time_sum_us FROM unified_perf_rows",
    )
    assert rows == [
        {"name": "aten::mm", "op_category": "GEMM", "kernel_time_sum_us": 123.5}
    ]

    search_rows = search_index(db_path, "Cijk", limit=10)
    assert search_rows
    assert search_rows[0]["trace_id"] == trace_id


def test_trace_index_rejects_write_sql(tmp_path):
    db_path = tmp_path / "trace_index.sqlite"
    with pytest.raises(ValueError):
        execute_read_query(db_path, "DELETE FROM traces")


def test_trace_index_store_boundary_supports_scan_import_and_search(tmp_path):
    db_path = tmp_path / "trace_index.sqlite"
    trace_root = tmp_path / "traces"
    trace_path = trace_root / "rank0_trace.json"
    trace_path.parent.mkdir(parents=True)
    trace_path.write_text(json.dumps({"traceEvents": []}), encoding="utf-8")

    report_dir = tmp_path / "reports" / "trace_1"
    write_csv(
        report_dir / "unified_perf_summary.csv",
        [
            {
                "name": "aten::scaled_dot_product_attention",
                "op category": "SDPA_fwd",
                "operation_count": "1",
                "Kernel Time (us)_sum": "10.0",
            }
        ],
    )

    store = SQLiteTraceIndexStore(db_path)
    try:
        assert scan_traces_with_store(store, trace_root) == 1
        trace_id = import_report_dir_with_store(
            store,
            report_dir,
            trace_path=trace_path,
            root=trace_root,
        )
        assert trace_id == 1
        hits = store.search("scaled", limit=10)
        assert hits[0].kind == "op"
    finally:
        store.close()


@pytest.mark.skipif(
    not TRAINING_REPORT_DIR.exists(),
    reason="checked-in Qwen training report CSVs are missing",
)
def test_import_real_training_report_maps_kernel_stream_and_times(tmp_path):
    db_path = tmp_path / "trace_index.sqlite"
    trace_id = import_report_dir(db_path, TRAINING_REPORT_DIR)

    unified = execute_read_query(
        db_path,
        "SELECT name, kernel_time_sum_us FROM unified_perf_rows WHERE name = 'aten::mm'",
    )
    assert unified
    assert unified[0]["kernel_time_sum_us"] > 0

    kernels = execute_read_query(
        db_path,
        "SELECT kernel_name, stream, total_duration_us FROM kernel_summary "
        "WHERE kernel_name LIKE 'Cijk%' LIMIT 1",
    )
    assert kernels
    assert kernels[0]["stream"] == 0
    assert kernels[0]["total_duration_us"] > 0

    categories = execute_read_query(
        db_path,
        "SELECT category, kernel_time_sum_us FROM op_category_rows "
        "WHERE kernel_time_sum_us IS NOT NULL ORDER BY kernel_time_sum_us DESC",
    )
    assert categories
    assert categories[0]["kernel_time_sum_us"] > 0
    assert trace_id == 1


@pytest.mark.skipif(
    not INFERENCE_REPORT_DIR.exists(),
    reason="checked-in inference report CSVs are missing",
)
def test_import_real_inference_report_converts_category_kernel_time_ms(tmp_path):
    db_path = tmp_path / "trace_index.sqlite"
    import_report_dir(db_path, INFERENCE_REPORT_DIR)
    rows = execute_read_query(
        db_path,
        "SELECT category, kernel_time_sum_us FROM op_category_rows "
        "WHERE category = 'GEMM'",
    )
    assert rows
    assert rows[0]["kernel_time_sum_us"] > 1000
