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
    append_trace,
    execute_read_query,
    search_index,
)
from TraceLens.TraceIndex.cli import main as trace_index_main
from TraceLens.TraceIndex.importer import (
    build_traces as build_traces_with_store,
    import_report_dir as import_report_dir_with_store,
)
from TraceLens.TraceIndex.sqlite_store import SQLiteTraceIndexStore
from TraceLens.TraceIndex.utils import (
    collect_trace_paths,
    parse_repr,
    read_traces_file,
    to_json,
)

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


def test_parse_repr_strips_numpy_scalars_and_to_json():
    """parse_repr turns a CSV Python repr (with np scalars) into plain data that
    to_json can serialize."""
    parsed = parse_repr("[{'name': 'k', 'total_duration_us': np.float64(1.5)}]")
    assert parsed[0]["name"] == "k"
    assert parsed[0]["total_duration_us"] == 1.5
    encoded = to_json(parse_repr("{'M': 128, 'bias': False}"))
    assert encoded is not None
    assert json.loads(encoded) == {"M": 128, "bias": False}


def test_trace_index_append_from_report_and_search(tmp_path):
    """Appending a report explodes kernels into op_kernels and fills the
    gemm/sdpa/conv satellites, and the parsed perf_params are queryable via
    json_extract and FTS."""
    db_path = tmp_path / "trace_index.sqlite"
    trace_root = tmp_path / "traces"
    trace_path = trace_root / "model_a" / "rank0_trace.json"
    trace_path.parent.mkdir(parents=True)
    trace_path.write_text(json.dumps({"traceEvents": []}), encoding="utf-8")

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
                "perf_params": (
                    "{'M': 128, 'N': 64, 'K': 32, 'B': 1, 'bias': False, "
                    "'dtype_A_B': ('c10::BFloat16', 'c10::BFloat16'), "
                    "'transpose': (True, False)}"
                ),
                "kernel_details_summary": (
                    "[{'name': 'Cijk_test_kernel', 'stream': 0, 'count': 2, "
                    "'total_duration_us': 123.5, 'mean_duration_us': 61.75, "
                    "'median_duration_us': 61.75, 'min_duration_us': 60.0, "
                    "'max_duration_us': 63.5}]"
                ),
            },
            {
                "name": "aten::scaled_dot_product_attention",
                "op category": "SDPA_fwd",
                "operation_count": "1",
                "Kernel Time (us)_sum": "10.0",
                "TFLOPS/s_mean": "40.0",
                "perf_params": (
                    "{'B': 2, 'H_Q': 8, 'N_Q': 128, 'N_KV': 128, "
                    "'d_h_qk': 64, 'd_h_v': 64, 'causal': True, "
                    "'dtype_A_B': ('c10::BFloat16', 'c10::BFloat16')}"
                ),
            },
            {
                "name": "aten::convolution",
                "op category": "CONV",
                "operation_count": "1",
                "Kernel Time (us)_sum": "20.0",
                "perf_params": (
                    "{'convNd': 'conv2d', 'input_shape': (2, 32, 56, 56), "
                    "'filter_shape': (32, 1, 3, 3), 'groups': 32, "
                    "'transposed_conv': False}"
                ),
            },
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

    trace_id = append_trace(db_path, trace_path, report_dir=report_dir, root=trace_root)
    assert trace_id == 1

    rows = execute_read_query(
        db_path,
        "SELECT name, op_category, kernel_time_sum_us FROM unified_perf_rows "
        "ORDER BY source_row",
    )
    assert rows[0] == {
        "name": "aten::mm",
        "op_category": "GEMM",
        "kernel_time_sum_us": 123.5,
    }

    gemm = execute_read_query(db_path, "SELECT m, n, k, batch FROM gemm_perf")
    assert gemm == [{"m": 128, "n": 64, "k": 32, "batch": 1}]

    params = execute_read_query(
        db_path,
        "SELECT json_extract(perf_params_json, '$.M') AS m "
        "FROM unified_perf_rows WHERE name = 'aten::mm'",
    )
    assert params[0]["m"] == 128

    kernels = execute_read_query(
        db_path,
        "SELECT kernel_name, unified_row_id, library, stream, parent_op_name "
        "FROM op_kernels",
    )
    assert kernels[0]["kernel_name"] == "Cijk_test_kernel"
    assert kernels[0]["unified_row_id"] is not None
    assert kernels[0]["library"] == "Tensile"
    assert kernels[0]["stream"] == 0
    assert kernels[0]["parent_op_name"] == "aten::mm"

    sdpa = execute_read_query(
        db_path, "SELECT seq_q, seq_kv, head_dim, causal FROM sdpa_perf"
    )
    assert sdpa == [{"seq_q": 128, "seq_kv": 128, "head_dim": 64, "causal": 1}]

    conv = execute_read_query(
        db_path, "SELECT groups, is_depthwise, is_transposed_conv FROM conv_perf"
    )
    assert conv == [{"groups": 32, "is_depthwise": 1, "is_transposed_conv": 0}]

    search_rows = search_index(db_path, "Cijk", limit=10)
    assert search_rows
    assert search_rows[0]["trace_id"] == trace_id


def test_trace_index_rejects_write_sql(tmp_path):
    """The read-only query path refuses non-SELECT statements."""
    db_path = tmp_path / "trace_index.sqlite"
    with pytest.raises(ValueError):
        execute_read_query(db_path, "DELETE FROM traces")


def test_trace_index_store_boundary_supports_append_and_search(tmp_path):
    """Driving SQLiteTraceIndexStore directly (the storage boundary) imports a
    report and returns an op-kind FTS hit."""
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
    """On the checked-in Qwen training report, op_kernels/gemm/sdpa are populated
    from real perf_params and kernel_details with correct shapes and stream."""
    db_path = tmp_path / "trace_index.sqlite"
    trace_path = tmp_path / "qwen_trace.json"
    trace_path.write_text(json.dumps({"traceEvents": []}), encoding="utf-8")
    trace_id = append_trace(db_path, trace_path, report_dir=TRAINING_REPORT_DIR)

    unified = execute_read_query(
        db_path,
        "SELECT name, kernel_time_sum_us FROM unified_perf_rows WHERE name = 'aten::mm'",
    )
    assert unified
    assert unified[0]["kernel_time_sum_us"] > 0

    kernels = execute_read_query(
        db_path,
        "SELECT k.kernel_name, k.stream, k.total_duration_us, k.unified_row_id, "
        "k.library FROM op_kernels k "
        "JOIN unified_perf_rows u ON u.id = k.unified_row_id "
        "WHERE k.kernel_name LIKE 'Cijk%' LIMIT 1",
    )
    assert kernels
    assert kernels[0]["stream"] == 0
    assert kernels[0]["total_duration_us"] > 0
    assert kernels[0]["unified_row_id"] is not None
    assert kernels[0]["library"] == "Tensile"

    gemm = execute_read_query(
        db_path,
        "SELECT g.m, g.n, g.k FROM gemm_perf g "
        "JOIN unified_perf_rows u ON u.id = g.unified_row_id "
        "WHERE u.name = 'aten::mm' AND g.n = 2816 AND g.k = 1024 LIMIT 1",
    )
    assert gemm
    assert gemm[0]["m"] == 8944
    assert gemm[0]["n"] == 2816
    assert gemm[0]["k"] == 1024

    params = execute_read_query(
        db_path,
        "SELECT json_extract(perf_params_json, '$.M') AS m "
        "FROM unified_perf_rows WHERE name = 'aten::mm' LIMIT 1",
    )
    assert params[0]["m"] == 8944

    sdpa = execute_read_query(
        db_path,
        "SELECT seq_q, seq_kv, head_dim FROM sdpa_perf LIMIT 1",
    )
    assert sdpa
    assert sdpa[0]["seq_q"] is not None

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
    """On the checked-in inference report, category kernel time in ms is converted
    to microseconds during import."""
    db_path = tmp_path / "trace_index.sqlite"
    trace_path = tmp_path / "decode_trace.json"
    trace_path.write_text(json.dumps({"traceEvents": []}), encoding="utf-8")
    append_trace(db_path, trace_path, report_dir=INFERENCE_REPORT_DIR)
    rows = execute_read_query(
        db_path,
        "SELECT category, kernel_time_sum_us FROM op_category_rows "
        "WHERE category = 'GEMM'",
    )
    assert rows
    assert rows[0]["kernel_time_sum_us"] > 1000


def test_read_traces_file_skips_comments_and_blanks(tmp_path):
    """A traces-file drops blank/comment lines and collect_trace_paths merges and
    de-duplicates paths."""
    traces_file = tmp_path / "traces.txt"
    traces_file.write_text(
        "# header\n" "\n" " /data/a.json.gz \n" "# skip me\n" "C:/traces/b.json\n",
        encoding="utf-8",
    )
    paths = read_traces_file(traces_file)
    assert paths == [Path("/data/a.json.gz"), Path("C:/traces/b.json")]
    combined = collect_trace_paths(
        traces_file, [Path("/data/a.json.gz"), Path("c.json")]
    )
    assert combined == [
        Path("/data/a.json.gz"),
        Path("C:/traces/b.json"),
        Path("c.json"),
    ]


def test_cli_append_from_existing_report(tmp_path, capsys):
    """The CLI append command imports an existing report dir without regenerating
    it and reports the new trace_id."""
    db_path = tmp_path / "trace_index.sqlite"
    trace_path = tmp_path / "rank0_trace.json"
    trace_path.write_text(json.dumps({"traceEvents": []}), encoding="utf-8")
    report_dir = tmp_path / "report"
    write_csv(
        report_dir / "unified_perf_summary.csv",
        [
            {
                "name": "aten::mm",
                "op category": "GEMM",
                "operation_count": "1",
                "Kernel Time (us)_sum": "10.0",
            }
        ],
    )

    rc = trace_index_main(
        [
            "--db",
            str(db_path),
            "append",
            "--trace-path",
            str(trace_path),
            "--report-dir",
            str(report_dir),
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["trace_id"] == 1
    assert payload["generated_report"] is False
    rows = execute_read_query(db_path, "SELECT name FROM unified_perf_rows")
    assert rows[0]["name"] == "aten::mm"


def test_build_traces_continues_after_failure(tmp_path):
    """A batch build records per-trace failures and keeps processing the rest of
    the list."""
    db_path = tmp_path / "trace_index.sqlite"
    store = SQLiteTraceIndexStore(db_path)
    try:
        result = build_traces_with_store(
            store,
            [tmp_path / "missing_a.json", tmp_path / "missing_b.json"],
            report_root=tmp_path / "reports",
        )
    finally:
        store.close()
    assert result["imported"] == []
    assert len(result["failed"]) == 2
    assert "missing_a.json" in result["failed"][0]["trace_path"]
