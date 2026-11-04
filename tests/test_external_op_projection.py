###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Op projections: built-in Origami and an extension-registered external model."""

import importlib
from dataclasses import replace
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from TraceLens.PerfModel import origami_helper
from TraceLens.PerfModel.perf_model import aten_mm
from TraceLens.PerfModel.op_projections import (
    OpWork,
    op_work,
    origami_gemm_op_projection,
)
from TraceLens.Reporting.generate_perf_report_pytorch import (
    generate_perf_report_pytorch,
)
from TraceLens.TreePerf.tree_perf import TreePerfAnalyzer

REPO = Path(__file__).resolve().parents[1]
STUB = REPO / "examples" / "external_op_projection_stub.py"
TRACE = (
    REPO
    / "tests"
    / "traces"
    / "mi300"
    / "gaunernst_bert-small-uncased__1016001.json.gz"
)
ARCH = {"name": "mi300x", "freq_mhz": 2100, "mem_bw_gbps": 5300}


def _mm():
    return aten_mm(
        {
            "name": "aten::mm",
            "args": {
                "Input Dims": [[128, 64], [64, 32]],
                "Input type": ["c10::BFloat16", "c10::BFloat16"],
                "Input Strides": [[64, 1], [1, 32]],
            },
        }
    )


def test_gemm_work_has_the_category_and_a_copy_of_the_params():
    gemm = _mm()
    work = op_work(gemm)
    assert work.category == "GEMM"
    assert (work.params["M"], work.params["N"], work.params["K"]) == (128, 32, 64)
    work.params["M"] = -1
    assert gemm.param_details["M"] == 128


def _gemm_work(category="GEMM"):
    return OpWork(
        category=category,
        params={"M": 1, "N": 1, "K": 1, "B": 1},
        gflops=2e-9,
        bytes_moved=None,
        compute_spec=None,
    )


@pytest.mark.parametrize(
    "report_module",
    [
        "TraceLens.Reporting.generate_perf_report_pytorch",
        "TraceLens.Reporting.generate_perf_report_pytorch_inference",
    ],
)
def test_extension_file_registers_external_model(report_module):
    registered = []
    analyzer = SimpleNamespace(set_external_op_projection=registered.append)
    importlib.import_module(report_module).apply_extension(analyzer, str(STUB))
    assert len(registered) == 1
    output = registered[0](_gemm_work(), None)
    assert output["time_us"] == pytest.approx(2e-8)
    assert output["Transpose"] == "unknown"
    assert registered[0](_gemm_work("SDPA_fwd"), None) is None


def test_origami_model_passes_gemm_shape_to_origami():
    work = op_work(_mm())
    with patch.object(origami_helper, "gemm_time_us", return_value=7.0) as sim:
        assert origami_gemm_op_projection(work, ARCH) == {"time_us": 7.0}
        attention = replace(work, category="SDPA_fwd")
        assert origami_gemm_op_projection(attention, ARCH) is None
        assert origami_gemm_op_projection(work, None) is None
    sim.assert_called_once_with(ARCH, 128, 32, 64, 1, "bf16")


def test_set_external_op_projection():
    analyzer = SimpleNamespace(op_projections={})
    analyzer.register_op_projection = partial(
        TreePerfAnalyzer.register_op_projection, analyzer
    )
    TreePerfAnalyzer.set_external_op_projection(analyzer, len)
    assert list(analyzer.op_projections) == ["External"]
    assert analyzer.op_projections["External"] is len
    TreePerfAnalyzer.set_external_op_projection(analyzer, None)
    assert analyzer.op_projections == {}
    with pytest.raises(TypeError):
        TreePerfAnalyzer.set_external_op_projection(analyzer, "not a function")


def test_report_has_origami_and_external_columns_for_gemms(tmp_path):
    with patch.object(origami_helper, "gemm_time_us", return_value=1.0):
        dfs = generate_perf_report_pytorch(
            profile_json_path=str(TRACE),
            output_csvs_dir=str(tmp_path / "csvs"),
            extension_file=str(STUB),
            gpu_arch=ARCH,
            enable_origami_gemm=True,
            collective_analysis=False,
        )
    summary = dfs["unified_perf_summary"]
    gemms = summary[summary["op category"] == "GEMM"]
    assert not gemms.empty
    assert (gemms["Origami Time (µs)_first"] == 1.0).all()

    external = summary["External Time (µs)_first"]
    assert (summary.loc[external.notna(), "op category"] == "GEMM").all()
    p = gemms.iloc[0]["perf_params"]
    expected = 2 * p["M"] * p["N"] * p["K"] * p["B"] / 1e8
    assert gemms.iloc[0]["External Time (µs)_first"] == pytest.approx(expected)
    assert "External Time (µs)_first" in dfs["GEMM"].columns
    assert "External Transpose_first" in dfs["GEMM"].columns


def test_report_without_models_has_no_simulated_columns(tmp_path):
    dfs = generate_perf_report_pytorch(
        profile_json_path=str(TRACE),
        output_csvs_dir=str(tmp_path / "csvs"),
        gpu_arch=ARCH,
        collective_analysis=False,
    )
    columns = dfs["unified_perf_summary"].columns
    assert not [c for c in columns if "Origami" in c or "External" in c]
