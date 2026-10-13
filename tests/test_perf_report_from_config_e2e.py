###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""End-to-end: a YAML run writes the same CSVs as calling the report directly."""

import os

import pandas as pd
import pytest

from TraceLens.Reporting.generate_perf_report_from_config import run_config
from TraceLens.Reporting.generate_perf_report_pytorch import (
    generate_perf_report_pytorch as training_report,
)
from TraceLens.Reporting.generate_perf_report_pytorch_inference import (
    generate_perf_report_pytorch as inference_report,
)
from TraceLens.Trace2Tree.trace_capture_merge_experimental import (
    merge_capture_trace_into_graph,
)
from tests.test_reporting_utils import _write_trace

TRAINING_TRACE = os.path.join(
    "tests", "traces", "mi300", "Falconsai_nsfw_image_detection__1016002.json.gz"
)
INFERENCE_DIR = os.path.join("tests", "traces", "inference", "sglang_decode")
INFERENCE_TRACE = os.path.join(INFERENCE_DIR, "sglang_Qwen3-8B_decode.json.gz")
INFERENCE_CAPTURE = os.path.join(INFERENCE_DIR, "capture_traces")

TRAINING_FLAGS = {
    "enable_pseudo_ops": True,
    "group_by_num_kernels": True,
    "include_call_stack": True,
}
INFERENCE_FLAGS = {
    **TRAINING_FLAGS,
    "group_by_parent_module": True,
    "collective_analysis": False,
}


def _assert_same_csvs(expected_dir, actual_dir):
    expected = sorted(os.listdir(expected_dir))
    assert expected
    assert sorted(os.listdir(actual_dir)) == expected
    for name in expected:
        pd.testing.assert_frame_equal(
            pd.read_csv(os.path.join(actual_dir, name)),
            pd.read_csv(os.path.join(expected_dir, name)),
            check_exact=False,
            obj=name,
        )


def _skip_if_missing(*paths):
    for path in paths:
        if not os.path.exists(path):
            pytest.skip(f"fixture not found: {path}")


def test_training_yaml_matches_direct_call(tmp_path):
    _skip_if_missing(TRAINING_TRACE)
    direct = tmp_path / "direct"
    training_report(
        profile_json_path=TRAINING_TRACE,
        output_csvs_dir=str(direct),
        gpu_arch_platform="MI300X",
        **TRAINING_FLAGS,
    )
    run_config(
        {
            "analysis_mode": "default",
            "output_dir": str(tmp_path / "yaml"),
            "jobs": [{"trace_path": TRAINING_TRACE, "platform": "MI300X"}],
        }
    )
    _assert_same_csvs(str(direct), str(tmp_path / "yaml" / "perf_report_csvs"))
    assert (tmp_path / "yaml" / "perf_report.xlsx").is_file()
    assert (tmp_path / "yaml" / "run_config.resolved.yaml").is_file()


def test_training_compare_to_matches_comparison_json_path(tmp_path):
    new = _write_trace(tmp_path, [("aten::mm", "gemm_kernel", 100)], "new.json")
    old = _write_trace(tmp_path, [("aten::mm", "gemm_kernel", 120)], "old.json")
    direct = tmp_path / "direct"
    training_report(
        profile_json_path=new,
        output_csvs_dir=str(direct),
        gpu_arch_platform="MI300X",
        comparison_json_path=old,
        **TRAINING_FLAGS,
    )
    run_config(
        {
            "analysis_mode": "default",
            "output_dir": str(tmp_path / "yaml"),
            "jobs": [
                {
                    "trace_path": new,
                    "platform": "MI300X",
                    "compare_to": {"trace_path": old},
                }
            ],
        }
    )
    assert "diff_stats.csv" in os.listdir(direct)
    _assert_same_csvs(str(direct), str(tmp_path / "yaml" / "perf_report_csvs"))


def test_inference_capture_yaml_matches_direct_call(tmp_path):
    metadata = os.path.join(INFERENCE_CAPTURE, "execution_details.json")
    # Without execution_details.json the base code would write one into the fixture.
    _skip_if_missing(INFERENCE_TRACE, metadata)
    direct = tmp_path / "direct"
    inference_report(
        profile_json_path=INFERENCE_TRACE,
        augmented_tree=merge_capture_trace_into_graph(
            INFERENCE_CAPTURE, metadata, INFERENCE_TRACE
        ),
        output_csvs_dir=str(direct),
        gpu_arch_platform="MI300X",
        **INFERENCE_FLAGS,
    )
    run_config(
        {
            "analysis_mode": "inference",
            "output_dir": str(tmp_path / "yaml"),
            "jobs": [
                {
                    "trace_path": INFERENCE_TRACE,
                    "platform": "MI300X",
                    "capture_folder": INFERENCE_CAPTURE,
                }
            ],
        }
    )
    _assert_same_csvs(str(direct), str(tmp_path / "yaml" / "perf_report_csvs"))
