###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""CLI main() coverage for TraceLens.Reporting entry points."""

from __future__ import annotations

import importlib
import json
import os
import sys

import pytest

from tests.test_reporting_coverage import (
    KERNEL_TRACE_CSV,
    _build_synthetic_trace,
    _create_genesis_capture,
    _minimal_pftrace_events,
    _write_trace,
)


def test_pytorch_report_main(tmp_path):
    trace = _write_trace(
        tmp_path,
        [("aten::mm", "gemm_kernel", 100), ("aten::add", "add_kernel", 15)],
    )
    out_dir = tmp_path / "py_csvs"
    xlsx = tmp_path / "py.xlsx"
    mod = importlib.import_module("TraceLens.Reporting.generate_perf_report_pytorch")

    old_argv = sys.argv
    sys.argv = [
        "generate_perf_report_pytorch",
        "--profile_json_path",
        trace,
        "--output_csvs_dir",
        str(out_dir),
        "--output_xlsx_path",
        str(xlsx),
        "--enable_kernel_summary",
        "--short_kernel_study",
        "--disable_coll_analysis",
        "--group_by_num_kernels",
    ]
    try:
        mod.main()
    finally:
        sys.argv = old_argv
    assert xlsx.exists()


def test_inference_report_main(tmp_path):
    trace = _write_trace(tmp_path, [("aten::mm", "gemm_kernel", 80)])
    out_dir = tmp_path / "inf_csvs"
    import TraceLens.Reporting.generate_perf_report_pytorch_inference as mod

    old_argv = sys.argv
    sys.argv = [
        "generate_perf_report_pytorch_inference",
        "--profile_json_path",
        trace,
        "--output_csvs_dir",
        str(out_dir),
        "--disable_coll_analysis",
        "--enable_kernel_summary",
    ]
    try:
        mod.main()
    finally:
        sys.argv = old_argv
    assert (out_dir / "gpu_timeline.csv").exists()


def test_collective_report_main_trace_glob(tmp_path):
    for rank in (0, 1):
        events = {
            "traceEvents": [
                {
                    "ph": "X",
                    "cat": "kernel",
                    "name": "ncclKernel_AllReduce",
                    "pid": rank,
                    "tid": 3,
                    "ts": 1000,
                    "dur": 40,
                    "args": {"External id": rank, "Collective name": "allreduce", "stream": 3},
                }
            ]
        }
        (tmp_path / f"trace_rank_{rank}.json").write_text(json.dumps(events))
    out = tmp_path / "coll.xlsx"
    import TraceLens.Reporting.generate_multi_rank_collective_report_pytorch as mod

    old_argv = sys.argv
    sys.argv = [
        "generate_multi_rank_collective_report_pytorch",
        "--trace_glob",
        str(tmp_path / "trace_rank_*.json"),
        "--world_size",
        "2",
        "--output_xlsx_path",
        str(out),
    ]
    try:
        mod.main()
    finally:
        sys.argv = old_argv
    assert out.exists()


def test_pftrace_hip_activity_main(tmp_path):
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(json.dumps({"traceEvents": _minimal_pftrace_events()}))
    md = tmp_path / "pf.md"
    import TraceLens.Reporting.generate_perf_report_pftrace_hip_activity as mod

    old_argv = sys.argv
    sys.argv = [
        "generate_perf_report_pftrace_hip_activity",
        "--trace_path",
        str(trace_path),
        "--output_md_path",
        str(md),
        "--write_md",
    ]
    try:
        mod.main()
    finally:
        sys.argv = old_argv
    assert md.exists()



@pytest.mark.skipif(
    not os.path.exists(
        os.path.join(
            os.path.dirname(__file__),
            "traces/mi300/jax_conv_minimal_legacy/chi-mi300x-013.ord.vultr.cpe.ice.amd.com.xplane.pb",
        )
    ),
    reason="JAX trace fixture missing",
)
def test_jax_report_main(tmp_path):
    trace = os.path.join(
        os.path.dirname(__file__),
        "traces/mi300/jax_conv_minimal_legacy/chi-mi300x-013.ord.vultr.cpe.ice.amd.com.xplane.pb",
    )
    out = tmp_path / "jax.xlsx"
    import TraceLens.Reporting.generate_perf_report_jax as mod

    old_argv = sys.argv
    sys.argv = [
        "generate_perf_report_jax",
        "--profile_path",
        trace,
        "--output_xlsx_path",
        str(out),
    ]
    try:
        mod.main()
    finally:
        sys.argv = old_argv
    assert out.exists()
