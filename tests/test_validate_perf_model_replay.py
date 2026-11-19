###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import csv
import subprocess
import sys

import pytest

from TraceLens.Agent.PerfModel.validate_perf_model.harnesses import (
    _CSV_ONLY_OPS,
)
from TraceLens.Agent.PerfModel.validate_perf_model.validator import (
    _enrich_results,
    parse_counter_csv,
    parse_kernel_runtime_ns,
)

# Warmup, L2 flush and marker, then a measured call that launches two kernels.
_DISPATCHES = [
    (1, "gemm_a"),
    (2, "gemm_b"),
    (3, "void at::native::FillFunctor<unsigned char>"),
    (4, "void at::native::bitwise_not_kernel_cuda"),
    (5, "gemm_a"),
    (6, "void at::native::copy_kernel"),
    (7, "gemm_b"),
]


def _write(path, header, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def test_counters_sum_over_the_measured_call(tmp_path):
    rows = [
        (d, k, name, d * 10)
        for d, k in _DISPATCHES
        for name in ("TOTAL_16_OPS", "WRITE_SIZE")
    ]
    path = _write(
        tmp_path / "pmc.csv",
        ["Dispatch_Id", "Kernel_Name", "Counter_Name", "Counter_Value"],
        rows,
    )
    assert parse_counter_csv(path) == {"TOTAL_16_OPS": 120.0, "WRITE_SIZE": 120.0}
    assert parse_counter_csv(path, "gemm_b") == {
        "TOTAL_16_OPS": 70.0,
        "WRITE_SIZE": 70.0,
    }


def test_runtime_sums_the_measured_call(tmp_path):
    path = _write(
        tmp_path / "trace.csv",
        ["Dispatch_Id", "Kernel_Name", "Start_Timestamp", "End_Timestamp"],
        [(d, k, d * 100, d * 100 + d) for d, k in _DISPATCHES],
    )
    assert parse_kernel_runtime_ns(path) == (12, 6.0, 2)


def test_trace_without_marker_fails(tmp_path):
    path = _write(
        tmp_path / "pmc.csv",
        ["Dispatch_Id", "Kernel_Name", "Counter_Name", "Counter_Value"],
        [(1, "gemm_a", "TOTAL_16_OPS", 1)],
    )
    with pytest.raises(RuntimeError, match="marker"):
        parse_counter_csv(path)


def test_validator_imports_without_torch():
    code = (
        "import sys\n"
        "import TraceLens.Agent.PerfModel.validate_perf_model.validator\n"
        "assert 'torch' not in sys.modules, 'validator parent imported torch'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_every_csv_only_op_has_a_call_spec():
    pytest.importorskip("torch")
    from TraceLens.Agent.PerfModel.validate_perf_model.replay.generic import (
        OP_CALL_SPEC,
    )

    missing = [key for key, *_ in _CSV_ONLY_OPS if key not in OP_CALL_SPEC]
    assert not missing, f"no OP_CALL_SPEC entry in replay/generic.py for {missing}"


def _row(op, **fields):
    return {
        "op": op,
        "run_status": "OK",
        "flops_ratio": 1.0,
        "bytes_ratio": 1.0,
        "predicted_flops": 100.0,
        "predicted_bytes": 10.0,
        **fields,
    }


def test_precision_mismatch_blocks_only_mfma_ops():
    gemm, norm, absent = _enrich_results(
        [
            _row("gemm_a8w8_blockscale", precision="MISMATCH"),
            _row("rms_norm", precision="MISMATCH"),
            _row(
                "gemm_a8w8_blockscale",
                precision="MISMATCH",
                precision_reason="fp4 counter is not available on gfx942",
            ),
        ]
    )
    assert gemm["precision_check"] == "MISMATCH"
    assert gemm["match_status"] == "precision-mismatch"
    assert norm["match_status"] == "match"
    assert absent["match_status"] == "match"


def test_report_mismatch_fails_the_row():
    same, differs = _enrich_results(
        [
            _row("rms_norm", csv_reported_flops=100.0, csv_reported_bytes=10.0),
            _row("rms_norm", csv_reported_flops=150.0, csv_reported_bytes=10.0),
        ]
    )
    assert same["run_status"] == "OK"
    assert differs["run_status"].startswith("FAIL: prediction FLOPs")
    assert differs["match_status"] == ""


def test_kernel_match_compares_kernel_families():
    rows = _enrich_results(
        [
            _row(
                "rms_norm",
                trace_kernels=["_k_BLOCK_SIZE_M_32"],
                kernel_name="_k_BLOCK_SIZE_M_64; void other<int>(x)",
            ),
            _row("rms_norm", trace_kernels=["_k"], kernel_name="void other<int>(x)"),
            _row("rms_norm", kernel_name="_k"),
        ]
    )
    assert [r["kernel_match"] for r in rows] == ["match", "mismatch", ""]


def test_arch_is_required():
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "TraceLens.Agent.PerfModel.validate_perf_model.validator",
            "--op",
            "rms_norm",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0 and "--arch is required" in proc.stderr
