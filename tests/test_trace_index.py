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
from TraceLens.TraceIndex.cli import build_parser, main as trace_index_main
from TraceLens.TraceIndex.importer import (
    build_traces as build_traces_with_store,
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


def write_stub_trace(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"traceEvents": []}), encoding="utf-8")
    return path


def write_mini_report(report_dir):
    write_csv(
        report_dir / "unified_perf_summary.csv",
        [
            {
                "name": "aten::mm",
                "op category": "GEMM",
                "operation_count": "1",
                "Kernel Time (us)_sum": "10.0",
            },
            {
                "name": "aten::add",
                "op category": "elementwise",
                "operation_count": "1",
                "Kernel Time (us)_sum": "1.0",
            },
        ],
    )
    return report_dir


def table_column_names(db_path, table):
    rows = execute_read_query(db_path, "PRAGMA table_info(%s)" % table)
    return {row["name"] for row in rows}


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
    trace_path = write_stub_trace(trace_root / "model_a" / "rank0_trace.json")

    report_dir = tmp_path / "reports" / "trace_1"
    write_csv(
        report_dir / "unified_perf_summary.csv",
        [
            {
                "name": "aten::mm",
                "op category": "GEMM",
                "operation_count": "2",
                "Kernel Time (us)_sum": "123.5",
                "Percentage (%)": "80.0",
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
        "SELECT name, op_category, kernel_time_sum_us, gpu_kernel_pct "
        "FROM unified_perf_rows "
        "ORDER BY source_row",
    )
    assert rows[0] == {
        "name": "aten::mm",
        "op_category": "GEMM",
        "kernel_time_sum_us": 123.5,
        "gpu_kernel_pct": 80.0,
    }

    gemm = execute_read_query(db_path, 'SELECT "M", "N", "K", "B" FROM gemm_perf')
    assert gemm == [{"M": 128, "N": 64, "K": 32, "B": 1}]

    params = execute_read_query(
        db_path,
        "SELECT json_extract(perf_params_json, '$.M') AS m "
        "FROM unified_perf_rows WHERE name = 'aten::mm'",
    )
    assert params[0]["m"] == 128

    kernels = execute_read_query(
        db_path,
        "SELECT name, unified_row_id, stream FROM op_kernels",
    )
    assert kernels[0]["name"] == "Cijk_test_kernel"
    assert kernels[0]["unified_row_id"] is not None
    assert kernels[0]["stream"] == 0
    kernel_cols = table_column_names(db_path, "op_kernels")
    assert "library" not in kernel_cols
    assert "parent_op_name" not in kernel_cols
    assert "kernel_name" not in kernel_cols

    sdpa = execute_read_query(
        db_path, "SELECT N_Q, N_KV, d_h_qk, causal FROM sdpa_perf"
    )
    assert sdpa == [{"N_Q": 128, "N_KV": 128, "d_h_qk": 64, "causal": 1}]

    conv = execute_read_query(
        db_path, "SELECT groups, transposed_conv, filter_shape FROM conv_perf"
    )
    assert conv[0]["groups"] == 32
    assert conv[0]["transposed_conv"] == 0
    assert json.loads(conv[0]["filter_shape"]) == [32, 1, 3, 3]

    gpu_mix = execute_read_query(
        db_path,
        "SELECT gpu_total_ms, gpu_compute_pct, gpu_idle_pct FROM traces",
    )
    assert gpu_mix == [
        {"gpu_total_ms": 1.0, "gpu_compute_pct": 80.0, "gpu_idle_pct": None}
    ]
    tables = {
        row["name"]
        for row in execute_read_query(
            db_path,
            "SELECT name FROM sqlite_master WHERE type = 'table'",
        )
    }
    assert "gpu_timeline_rows" not in tables
    assert "trace_summary" not in tables

    for table in ("op_kernels", "gemm_perf", "sdpa_perf", "conv_perf"):
        cols = table_column_names(db_path, table)
        assert "trace_id" not in cols
        assert "unified_row_id" in cols

    joined = execute_read_query(
        db_path,
        "SELECT t.id AS trace_id FROM op_kernels k "
        "JOIN unified_perf_rows u ON u.id = k.unified_row_id "
        "JOIN traces t ON t.id = u.trace_id LIMIT 1",
    )
    assert joined[0]["trace_id"] == trace_id

    search_rows = search_index(db_path, "Cijk", limit=10)
    assert search_rows
    assert search_rows[0]["trace_id"] == trace_id


def test_import_handoff_uses_runner_tracelens_id_and_artifact_paths(tmp_path):
    db_path = tmp_path / "trace_index.sqlite"
    trace_path = write_stub_trace(tmp_path / "runner" / "trace.json")
    report_path = write_mini_report(tmp_path / "runner" / "report")
    excel_path = tmp_path / "runner" / "report.xlsx"
    handoff_path = tmp_path / "handoff.jsonl"
    handoff_path.write_text(
        json.dumps(
            {
                "tracelens_id": "hyperloom-session-1-profile",
                "trace_path": str(trace_path),
                "report_path": str(report_path),
                "excel_path": str(excel_path),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = trace_index_main(
        [
            "--db",
            str(db_path),
            "import-handoff",
            "--handoff",
            str(handoff_path),
        ]
    )
    assert result == 0
    assert (
        trace_index_main(
            [
                "--db",
                str(db_path),
                "import-handoff",
                "--handoff",
                str(handoff_path),
            ]
        )
        == 0
    )
    rows = execute_read_query(
        db_path,
        "SELECT t.tracelens_id, t.path, r.report_dir, r.excel_path "
        "FROM traces t JOIN report_imports r ON r.trace_id = t.id",
    )
    assert rows == [
        {
            "tracelens_id": "hyperloom-session-1-profile",
            "path": str(trace_path).replace("\\", "/"),
            "report_dir": str(report_path).replace("\\", "/"),
            "excel_path": str(excel_path).replace("\\", "/"),
        }
    ]


def test_satellites_are_routed_by_op_category(tmp_path):
    db_path = tmp_path / "trace_index.sqlite"
    trace_path = write_stub_trace(tmp_path / "trace.json")
    report_path = tmp_path / "report"
    write_csv(
        report_path / "unified_perf_summary.csv",
        [
            {
                "name": "gemm_without_params",
                "op category": "GEMM",
                "perf_params": "",
            },
            {
                "name": "sdpa_without_params",
                "op category": "SDPA_fwd",
                "perf_params": "",
            },
            {
                "name": "conv_without_params",
                "op category": "CONV",
                "perf_params": "",
            },
            {
                "name": "mhc_with_mnk",
                "op category": "mHC",
                "perf_params": "{'M': 128, 'N': 24, 'K': 28672}",
            },
        ],
    )

    append_trace(db_path, trace_path, report_dir=report_path)

    assert execute_read_query(
        db_path,
        "SELECT u.name FROM gemm_perf g "
        "JOIN unified_perf_rows u ON u.id = g.unified_row_id",
    ) == [{"name": "gemm_without_params"}]
    assert execute_read_query(
        db_path,
        "SELECT u.name FROM sdpa_perf s "
        "JOIN unified_perf_rows u ON u.id = s.unified_row_id",
    ) == [{"name": "sdpa_without_params"}]
    assert execute_read_query(
        db_path,
        "SELECT u.name FROM conv_perf c "
        "JOIN unified_perf_rows u ON u.id = c.unified_row_id",
    ) == [{"name": "conv_without_params"}]


def test_trace_index_rejects_write_sql(tmp_path):
    """The read-only query path refuses non-SELECT statements."""
    db_path = tmp_path / "trace_index.sqlite"
    with pytest.raises(ValueError):
        execute_read_query(db_path, "DELETE FROM traces")


@pytest.mark.skipif(
    not TRAINING_REPORT_DIR.exists(),
    reason="checked-in Qwen training report CSVs are missing",
)
def test_import_real_training_report_maps_kernel_stream_and_times(tmp_path):
    """On the checked-in Qwen training report, op_kernels/gemm/sdpa are populated
    from real perf_params and kernel_details with correct shapes and stream."""
    db_path = tmp_path / "trace_index.sqlite"
    trace_path = write_stub_trace(tmp_path / "qwen_trace.json")
    trace_id = append_trace(db_path, trace_path, report_dir=TRAINING_REPORT_DIR)

    unified = execute_read_query(
        db_path,
        "SELECT name, kernel_time_sum_us FROM unified_perf_rows WHERE name = 'aten::mm'",
    )
    assert unified
    assert unified[0]["kernel_time_sum_us"] > 0

    kernels = execute_read_query(
        db_path,
        "SELECT k.name, k.stream, k.total_duration_us, k.unified_row_id "
        "FROM op_kernels k "
        "JOIN unified_perf_rows u ON u.id = k.unified_row_id "
        "WHERE k.name LIKE 'Cijk%' LIMIT 1",
    )
    assert kernels
    assert kernels[0]["stream"] == 0
    assert kernels[0]["total_duration_us"] > 0
    assert kernels[0]["unified_row_id"] is not None

    gemm = execute_read_query(
        db_path,
        'SELECT g."M", g."N", g."K" FROM gemm_perf g '
        "JOIN unified_perf_rows u ON u.id = g.unified_row_id "
        'WHERE u.name = \'aten::mm\' AND g."N" = 2816 AND g."K" = 1024 LIMIT 1',
    )
    assert gemm
    assert gemm[0]["M"] == 8944
    assert gemm[0]["N"] == 2816
    assert gemm[0]["K"] == 1024

    params = execute_read_query(
        db_path,
        "SELECT json_extract(perf_params_json, '$.M') AS m "
        "FROM unified_perf_rows WHERE name = 'aten::mm' LIMIT 1",
    )
    assert params[0]["m"] == 8944

    sdpa = execute_read_query(
        db_path,
        "SELECT N_Q, N_KV, d_h_qk FROM sdpa_perf LIMIT 1",
    )
    assert sdpa
    assert sdpa[0]["N_Q"] is not None

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
    trace_path = write_stub_trace(tmp_path / "decode_trace.json")
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


def test_collect_trace_paths_from_dir_skips_report_csvs(tmp_path):
    """Walking --trace-dir keeps trace-like files and skips report CSV dirs."""
    stub = write_stub_trace(tmp_path / "rank0_trace.json")
    report_dir = tmp_path / "perf_report_csvs"
    report_dir.mkdir()
    (report_dir / "unified_perf_summary.csv").write_text("name\nx\n", encoding="utf-8")
    write_stub_trace(report_dir / "nested.json")
    found = collect_trace_paths(trace_dirs=[tmp_path])
    assert found == [stub.resolve()]
    extra = Path("explicit.json")
    combined = collect_trace_paths(
        traces_file=None,
        trace_paths=[extra],
        trace_dirs=[tmp_path],
    )
    assert combined == [extra, stub.resolve()]


def test_cli_build_accepts_trace_dir():
    """build argparse accepts --trace-dir and combines it with --trace-path."""
    args = build_parser().parse_args(
        ["build", "--trace-dir", "tests/traces", "--trace-path", "a.json"]
    )
    assert args.trace_dirs == [Path("tests/traces")]
    assert args.trace_path == [Path("a.json")]


def test_cli_append_from_existing_report(tmp_path, capsys):
    """The CLI append command imports an existing report dir without regenerating
    it and reports the new trace_id."""
    db_path = tmp_path / "trace_index.sqlite"
    trace_path = write_stub_trace(tmp_path / "rank0_trace.json")
    report_dir = write_mini_report(tmp_path / "report")

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
    rows = execute_read_query(
        db_path, "SELECT name FROM unified_perf_rows ORDER BY source_row"
    )
    assert [row["name"] for row in rows] == ["aten::mm", "aten::add"]


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
