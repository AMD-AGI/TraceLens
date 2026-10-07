###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Pin real Origami results on a bundled trace.

The values below come from rocm-origami ORIGAMI_VERSION (the version CI installs).
They guard the Origami path through the perf report against refactors; a new
Origami release needs the values regenerated together with the CI pin.
"""

import importlib.metadata
import importlib.util

import pytest

from TraceLens.Reporting.generate_perf_report_pytorch import (
    generate_perf_report_pytorch,
)

ORIGAMI_VERSION = "0.0.3"


def _installed_origami_version():
    if importlib.util.find_spec("origami") is None:
        return None
    try:
        return importlib.metadata.version("rocm-origami")
    except importlib.metadata.PackageNotFoundError:
        return None


pytestmark = pytest.mark.skipif(
    _installed_origami_version() != ORIGAMI_VERSION,
    reason=f"needs rocm-origami=={ORIGAMI_VERSION}",
)

TRACE = "tests/traces/mi300/gaunernst_bert-small-uncased__1016001.json.gz"

ARCH = {
    "name": "MI300X",
    "freq_mhz": 2100,
    "mem_bw_gbps": 5300,
    "l1_bw_gbps": 100,
    "num_cus": 304,
    "gemm_units_per_cu": 4,
    "max_achievable_tflops": {"matrix_bf16": 708, "vector_bf16": 163},
}

# (M, N, K) -> Origami Time (µs) for the bf16 aten::addmm GEMMs in the trace.
EXPECTED_GEMM_TIME_US = {
    (141, 512, 512): 2.2984960394779197,
    (141, 2048, 512): 3.92727577185768,
    (141, 512, 2048): 7.640015353858991,
    (141, 30522, 512): 24.797284449747668,
}


@pytest.fixture(scope="module")
def origami_report(tmp_path_factory):
    return generate_perf_report_pytorch(
        profile_json_path=TRACE,
        output_csvs_dir=str(tmp_path_factory.mktemp("origami_reference")),
        gpu_arch=ARCH,
        enable_origami=True,
        collective_analysis=False,
    )


def test_origami_gemm_times_match_reference(origami_report):
    df = origami_report["GEMM"]
    got = {
        tuple(int(row[f"param: {dim}"]) for dim in "MNK"): row[
            "Origami Time (µs)_first"
        ]
        for _, row in df.iterrows()
    }
    assert got == pytest.approx(EXPECTED_GEMM_TIME_US, rel=1e-9)


def test_origami_gemm_rates_follow_time(origami_report):
    df = origami_report["GEMM"]
    for _, row in df.iterrows():
        time_us = row["Origami Time (µs)_first"]
        assert row["Origami TFLOPS/s_first"] == pytest.approx(
            row["GFLOPS_first"] / time_us * 1e3, rel=1e-9
        )
