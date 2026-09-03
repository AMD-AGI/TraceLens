###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Integration test for the semantic->report handoff via precomputed diff_stats.

The semantic comparison path emits a TraceDiff-schema ``diff_stats.csv`` and
feeds it to the perf-report generators through ``precomputed_diff_stats_csv``.
When set (and no ``comparison_json_path``), the generator skips the internal
TraceDiff and loads the CSV as-is, then enriches the report and emits a
``diff_stats`` sheet. These tests exercise that branch for both the training
and inference report variants.
"""

import json
import os

import pytest

from tests.fixtures.reporting import _build_synthetic_trace
from TraceLens.Reporting.generate_perf_report_pytorch import (
    generate_perf_report_pytorch,
)
from TraceLens.Reporting.generate_perf_report_pytorch_inference import (
    generate_perf_report_pytorch as generate_inference_report,
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_DIFF_STATS_CSV = os.path.join(
    REPO_ROOT, "tests", "traces", "tracediff_test", "diff_stats.csv"
)

pytestmark = pytest.mark.filterwarnings(
    "ignore:Source column .* not found.*:UserWarning",
    "ignore:There are hipgraph launches.*:UserWarning",
    "ignore:Found .* events with failed performance metric.*:UserWarning",
    "ignore:Input list of events is empty.*:UserWarning",
)


def _write_trace(tmp_path, specs):
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(_build_synthetic_trace(specs)))
    return str(path)


def test_precomputed_diff_stats_csv_is_used_pytorch(tmp_path):
    trace = _write_trace(
        tmp_path,
        [("aten::mm", "gemm_kernel", 100), ("aten::relu", "relu_kernel", 20)],
    )
    result = generate_perf_report_pytorch(
        profile_json_path=trace,
        output_csvs_dir=str(tmp_path / "csvs"),
        collective_analysis=False,
        precomputed_diff_stats_csv=_DIFF_STATS_CSV,
    )
    # The elif branch loaded the CSV and emitted the diff_stats sheet without
    # running the internal TraceDiff (no comparison_json_path was supplied).
    assert "diff_stats" in result
    assert not result["diff_stats"].empty


def test_precomputed_diff_stats_csv_is_used_inference(tmp_path):
    trace = _write_trace(tmp_path, [("aten::mm", "gemm_kernel", 100)])
    result = generate_inference_report(
        profile_json_path=trace,
        output_csvs_dir=str(tmp_path / "csvs"),
        collective_analysis=False,
        precomputed_diff_stats_csv=_DIFF_STATS_CSV,
    )
    assert "diff_stats" in result
    assert not result["diff_stats"].empty
