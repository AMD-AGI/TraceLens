###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for GPU arch resolution in Reporting scripts."""

import json, logging, os, tempfile, pytest, pandas as pd
from TraceLens.Agent.Analysis.utils.arch_utils import list_platforms, load_arch
from TraceLens.Reporting.reporting_utils import resolve_gpu_arch
from TraceLens.Reporting.generate_perf_report_pytorch import (
    generate_perf_report_pytorch,
)
from tests.fixtures.reporting import _build_synthetic_trace

TRACE_PATH = os.path.join(
    "tests",
    "traces",
    "mi300",
    "Falconsai_nsfw_image_detection__1016002.json.gz",
)

VALID_BOUND_VALUES = {"COMPUTE_BOUND", "MEMORY_BOUND"}


def test_resolve_gpu_arch_returns_none_by_default():
    assert resolve_gpu_arch() is None


NO_SPEC_WARNING = "No GPU architecture spec given"


def test_resolve_gpu_arch_warns_when_no_spec_given(caplog):
    with caplog.at_level(logging.WARNING, logger="TraceLens.Reporting.reporting_utils"):
        assert resolve_gpu_arch() is None
    assert NO_SPEC_WARNING in caplog.text
    assert "no default spec is assumed" in caplog.text


def test_resolve_gpu_arch_with_spec_does_not_warn(caplog):
    with caplog.at_level(logging.WARNING, logger="TraceLens.Reporting.reporting_utils"):
        resolve_gpu_arch(gpu_arch_platform="MI300X")
        resolve_gpu_arch(gpu_arch={"name": "inline"})
    assert NO_SPEC_WARNING not in caplog.text


W7900_KEYS = {"matrix_fp16", "matrix_int8", "vector_fp32"}


def test_resolve_w7900_platform():
    loaded = resolve_gpu_arch(gpu_arch_platform="W7900")
    assert "W7900" in list_platforms()
    assert loaded["name"] == "W7900"
    assert loaded["gfx_arch"] == "gfx1100"
    assert loaded["memory_gb"] == 48
    assert loaded["mem_bw_gbps"] > 0
    maf = loaded["max_achievable_tflops"]
    assert set(maf) == W7900_KEYS
    assert all(value > 0 for value in maf.values())


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
        enable_origami_gemm=True,
    )
    return csv_dir


def test_roofline_bound_with_gpu_arch_platform(perf_report_from_platform):

    df = pd.read_csv(
        os.path.join(perf_report_from_platform, "unified_perf_summary.csv")
    )
    bound_col = _find_col(df, "Roofline Bound")
    assert bound_col is not None
    bound_vals = set(df[bound_col].dropna().unique())
    assert bound_vals <= VALID_BOUND_VALUES, f"Unexpected values: {bound_vals}"


GEMM_DIM = 4096


def _matmul_trace(path, dtype):
    """A single aten::mm of *dtype*, large enough to be compute bound on a W7900."""
    kernel = {"c10::Half": "Cijk_f16_gemm.kd", "c10::BFloat16": "Cijk_bf16_gemm.kd"}
    trace = _build_synthetic_trace([("aten::mm", kernel[dtype], 4000.0)])
    cpu_op = next(e for e in trace["traceEvents"] if e["cat"] == "cpu_op")
    cpu_op["args"].update(
        {
            "Input Dims": [[GEMM_DIM, GEMM_DIM], [GEMM_DIM, GEMM_DIM]],
            "Input Strides": [[GEMM_DIM, 1], [GEMM_DIM, 1]],
            "Input type": [dtype, dtype],
        }
    )
    path.write_text(json.dumps(trace), encoding="utf-8")
    return str(path)


def _unified_summary(tmp_path, dtype="c10::Half", **arch_kwargs):
    csv_dir = tmp_path / "csvs"
    generate_perf_report_pytorch(
        profile_json_path=_matmul_trace(tmp_path / "mm.json", dtype),
        output_xlsx_path=None,
        output_csvs_dir=str(csv_dir),
        **arch_kwargs,
    )
    return pd.read_csv(csv_dir / "unified_perf_summary.csv")


def test_w7900_report_uses_the_bundled_spec(tmp_path):
    df = _unified_summary(tmp_path, gpu_arch_platform="W7900")
    row = df[df["name"] == "aten::mm"].iloc[0]
    roofline_us = row[_find_col(df, "Roofline Time")]

    def compute_us(platform):
        peak = load_arch(platform)["max_achievable_tflops"]["matrix_fp16"]
        return 2 * GEMM_DIM**3 / (peak * 1e12) * 1e6

    assert row[_find_col(df, "Roofline Bound")] == "COMPUTE_BOUND"
    assert roofline_us == pytest.approx(compute_us("W7900"))
    for other in ("MI300X", "MI325X"):
        assert roofline_us != pytest.approx(compute_us(other))


def test_w7900_bf16_gemm_has_no_roofline(tmp_path):
    # AMD publishes no BF16 matrix figure for the W7900, so W7900.json has no
    # matrix_bf16 ceiling.
    df = _unified_summary(tmp_path, "c10::BFloat16", gpu_arch_platform="W7900")
    row = df[df["name"] == "aten::mm"].iloc[0]
    assert row[_find_col(df, "Compute Spec")] == "matrix_bf16"
    assert row[_find_col(df, "GFLOPS")] > 0
    roofline_col = _find_col(df, "Roofline Time")
    assert roofline_col is None or pd.isna(row[roofline_col])


def test_report_without_spec_warns_and_borrows_none(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="TraceLens.Reporting.reporting_utils"):
        df = _unified_summary(tmp_path)
    assert _find_col(df, "Compute Spec") is not None
    assert _find_col(df, "Roofline") is None
    assert NO_SPEC_WARNING in caplog.text


def test_report_with_spec_has_roofline_and_no_warning(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="TraceLens.Reporting.reporting_utils"):
        df = _unified_summary(tmp_path, gpu_arch_platform="MI300X")
    assert _find_col(df, "Roofline Time") is not None
    assert NO_SPEC_WARNING not in caplog.text
