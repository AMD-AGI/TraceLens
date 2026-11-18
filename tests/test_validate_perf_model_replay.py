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
