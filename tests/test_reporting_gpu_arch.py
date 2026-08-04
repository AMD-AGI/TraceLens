###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for GPU arch resolution in Reporting scripts."""

import json
import os
import tempfile

import pytest

from TraceLens.Agent.Analysis.utils.arch_utils import load_arch
from TraceLens.Reporting.reporting_utils import resolve_gpu_arch
from TraceLens.Reporting.generate_perf_report_pytorch import (
    generate_perf_report_pytorch,
)

TRACE_PATH = os.path.join(
    "tests",
    "traces",
    "mi300",
    "Falconsai_nsfw_image_detection__1016002.json.gz",
)

VALID_BOUND_VALUES = {"COMPUTE_BOUND", "MEMORY_BOUND"}


def test_resolve_gpu_arch_returns_none_by_default():
    assert resolve_gpu_arch() is None


def test_resolve_gpu_arch_from_json_path():
    arch = {"name": "test-gpu", "mem_bw_gbps": 1}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(arch, f)
        path = f.name
    try:
        assert resolve_gpu_arch(gpu_arch_json_path=path) == arch
    finally:
        os.unlink(path)


def test_resolve_gpu_arch_from_platform():
    loaded = resolve_gpu_arch(gpu_arch_platform="MI300X")
    assert loaded["name"] == "MI300X"
    assert "max_achievable_tflops" in loaded


def test_resolve_gpu_arch_platform_matches_load_arch():
    assert resolve_gpu_arch(gpu_arch_platform="MI300X") == load_arch("MI300X")


def test_resolve_gpu_arch_direct_dict():
    arch = {"name": "inline"}
    assert resolve_gpu_arch(gpu_arch=arch) is arch


def test_resolve_gpu_arch_rejects_multiple_sources():
    with pytest.raises(ValueError, match="At most one"):
        resolve_gpu_arch(
            gpu_arch_json_path="/tmp/a.json",
            gpu_arch_platform="MI300X",
        )
    with pytest.raises(ValueError, match="At most one"):
        resolve_gpu_arch(gpu_arch={"name": "x"}, gpu_arch_platform="MI300X")


def test_resolve_gpu_arch_unknown_platform():
    with pytest.raises(KeyError):
        resolve_gpu_arch(gpu_arch_platform="NotARealPlatformXYZ")


def _find_col(df, prefix):
    for col in df.columns:
        if col.startswith(prefix):
            return col
    return None


@pytest.fixture(scope="module")
def perf_report_from_platform(tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("gpu_arch_platform")
    csv_dir = str(output_dir / "perf_report_csvs")
    generate_perf_report_pytorch(
        profile_json_path=TRACE_PATH,
        output_xlsx_path=None,
        output_csvs_dir=csv_dir,
        gpu_arch_platform="MI300X",
        enable_origami=True,
    )
    return csv_dir


def test_roofline_bound_with_gpu_arch_platform(perf_report_from_platform):
    import pandas as pd

    df = pd.read_csv(
        os.path.join(perf_report_from_platform, "unified_perf_summary.csv")
    )
    bound_col = _find_col(df, "Roofline Bound")
    assert bound_col is not None
    bound_vals = set(df[bound_col].dropna().unique())
    assert bound_vals <= VALID_BOUND_VALUES, f"Unexpected values: {bound_vals}"
