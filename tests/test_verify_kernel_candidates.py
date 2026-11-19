###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Contract tests for kernel-candidate verification against a trace and CSVs."""

import gzip
import json
from pathlib import Path

from TraceLens.Agent.Analysis.triage.checks import _events_of, _load_trace_json
from TraceLens.Agent.Analysis.triage.verify_kernel_candidates import (
    check_candidate_kernels,
    check_csv_quality,
    check_gpu_time,
    check_kernel_counts,
    check_ranked_kernels,
    check_top_rows,
    index_candidates,
    kernel_detail_totals,
    load_unified_rows,
    main,
    scan_trace_kernels,
)

_KERNEL_DETAILS_A = (
    "[{'name': 'kern_a', 'stream': 1, 'count': 2, "
    "'total_duration_us': np.float64(100.0)}]"
)
_KERNEL_DETAILS_B = (
    "[{'name': 'kern_b', 'stream': 1, 'count': 1, "
    "'total_duration_us': np.float64(50.0)}]"
)


def _write_analysis(root: Path) -> None:
    trace = {
        "traceEvents": [
            {"ph": "X", "cat": "kernel", "name": "kern_a", "dur": 70},
            {"ph": "X", "cat": "kernel", "name": "kern_a", "dur": 30},
            {"ph": "X", "cat": "cpu_op", "name": "kern_a", "dur": 999},
            {"ph": "X", "cat": "kernel", "name": "kern_b", "dur": 50},
            {"ph": "X", "cat": "gpu_memcpy", "name": "Memcpy DtoD", "dur": 1},
        ]
    }
    trace_path = root / "mixed_steady_state_case.json.gz"
    with gzip.open(trace_path, "wt", encoding="utf-8") as handle:
        json.dump(trace, handle)
    csv_dir = root / "perf_report_csvs"
    csv_dir.mkdir()
    (csv_dir / "gpu_timeline.csv").write_text(
        "type,time ms,percent\n"
        "computation_time,0.12,80\n"
        "busy_time,0.14,93\n"
        "total_time,0.15,100\n",
        encoding="utf-8",
    )
    (csv_dir / "unified_perf_summary.csv").write_text(
        "name,Input Dims,Kernel Time (µs)_sum,has_perf_model,kernel_details_summary,Percentage (%)\n"
        f'op_a,"((4, 8), ())",100,True,"{_KERNEL_DETAILS_A}",66.7\n'
        f'op_b,,50,False,"{_KERNEL_DETAILS_B}",33.3\n',
        encoding="utf-8",
    )
    (root / "category_data").mkdir()
    (root / "category_data" / "category_manifest.json").write_text(
        json.dumps({"trace_path": str(trace_path)}),
        encoding="utf-8",
    )
    (root / "kernel_candidates.json").write_text(
        json.dumps(
            {
                "hot_kernels": [
                    {
                        "name": "op_a",
                        "device_kernel_name": "kern_a",
                        "device_kernel_names": ["kern_a"],
                        "gpu_pct": 66.0,
                        "shapes": ["(4, 8) bf16"],
                        "input_shapes": [{"shape": "(4, 8) bf16", "call_num": 2}],
                    },
                    {
                        "name": "op_b",
                        "device_kernel_name": "kern_b",
                        "gpu_pct": 30.0,
                        "shapes": [],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )


def _events(root: Path) -> list:
    path = root / "mixed_steady_state_case.json.gz"
    return _events_of(_load_trace_json(str(path)))


def test_scan_counts_kernels_and_memcpy_not_cpu_ops(tmp_path):
    _write_analysis(tmp_path)
    kernels = scan_trace_kernels(_events(tmp_path))
    assert set(kernels) == {"kern_a", "kern_b", "Memcpy DtoD"}
    assert kernels["kern_a"]["count"] == 2
    assert kernels["kern_a"]["dur_us"] == 100


def test_checks_cover_time_counts_rank_shapes_and_models(tmp_path):
    _write_analysis(tmp_path)
    rows, time_col = load_unified_rows(
        tmp_path / "perf_report_csvs" / "unified_perf_summary.csv"
    )
    timeline = {"busy_time": 0.14, "computation_time": 0.12}
    trace_kernels = scan_trace_kernels(_events(tmp_path))
    csv_kernels = kernel_detail_totals(rows)
    payload = json.loads(
        (tmp_path / "kernel_candidates.json").read_text(encoding="utf-8")
    )
    by_op, by_device = index_candidates(payload)

    gpu = check_gpu_time(rows, time_col, timeline)
    assert gpu["status"] == "pass"
    assert gpu["kernel_time_us"] == 150

    counts = check_kernel_counts(trace_kernels, csv_kernels)
    assert counts["status"] == "fail"
    assert counts["trace_kernel_events"] == 4
    assert counts["mismatch_count"] == 1

    ranked = check_ranked_kernels(
        trace_kernels,
        csv_kernels,
        by_device,
        busy_us=140.0,
        candidates_present=True,
    )
    assert ranked["missing_from_csv"] == 0
    assert ranked["missing_from_candidates"] == 0
    assert ranked["kernels"][0]["name"] == "kern_a"

    top = check_top_rows(rows, time_col, by_op, candidates_present=True)
    by_name = {item["name"]: item for item in top["rows"]}
    assert by_name["op_a"]["missing_shapes"] == []
    assert by_name["op_b"]["input_dims_missing"] is True

    payload["hot_kernels"][0]["source_file"] = "kern_a.py"
    payload["hot_kernels"].append({"name": "op_c", "device_kernel_name": "kern_c"})
    candidates = check_candidate_kernels(payload, trace_kernels)
    assert candidates["status"] == "fail"
    assert candidates["absent_from_trace"] == ["op_c"]
    assert candidates["source_unresolved"] == ["op_b", "op_c"]

    quality = check_csv_quality(rows, time_col)
    assert quality["missing_shapes"]["rows"] == 1
    assert quality["missing_shapes"]["fraction_of_kernel_time"] == 50 / 150
    assert quality["missing_perf_model"]["rows"] == 1


def test_cli_fails_when_candidates_are_absent(tmp_path):
    _write_analysis(tmp_path)
    (tmp_path / "kernel_candidates.json").unlink()
    out = tmp_path / "report.json"
    rc = main(["--analysis-dir", str(tmp_path), "--json-out", str(out)])
    assert rc == 1
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["kernel_candidates"] is None
    assert "top_rows" in report["failing_checks"]
    assert "ranked_kernels" in report["failing_checks"]
    assert report["trace"].endswith("mixed_steady_state_case.json.gz")


def test_prefers_mixed_split_next_to_a_recorded_decode_trace(tmp_path):
    _write_analysis(tmp_path)
    mixed = tmp_path / "mixed_steady_state_case.json.gz"
    decode = tmp_path / "decode_only_steady_state_case.json.gz"
    decode.write_bytes(mixed.read_bytes())
    (tmp_path / "category_data" / "category_manifest.json").write_text(
        json.dumps({"trace_path": str(decode)}),
        encoding="utf-8",
    )
    out = tmp_path / "report.json"
    main(["--analysis-dir", str(tmp_path), "--json-out", str(out)])
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["trace"].endswith("mixed_steady_state_case.json.gz")
    assert report["recorded_trace"].endswith("decode_only_steady_state_case.json.gz")
