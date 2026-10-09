###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Op models: built-in Origami and an extension-registered external model."""

import importlib
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from TraceLens.PerfModel import origami_helper
from TraceLens.PerfModel.perf_model import aten_mm
from TraceLens.PerfModel.op_models import (
    external_op_model,
    op_work,
    origami_gemm_model,
)
from TraceLens.Reporting.generate_perf_report_pytorch import (
    generate_perf_report_pytorch,
)
from TraceLens.TreePerf.tree_perf import TreePerfAnalyzer

REPO = Path(__file__).resolve().parents[1]
STUB = REPO / "examples" / "external_op_model_stub.py"
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


def test_model_gets_category_a_copy_of_params_and_arch():
    seen = {}

    def model(category, params, arch):
        seen.update(category=category, params=params, arch=arch)
        params["M"] = -1
        return 12

    gemm = _mm()
    assert external_op_model(model)(op_work(gemm), ARCH) == 12.0
    assert seen["category"] == "GEMM"
    assert (seen["params"]["N"], seen["params"]["K"]) == (32, 64)
    assert seen["arch"] is ARCH
    assert gemm.param_details["M"] == 128


def test_model_returning_none_gives_no_prediction():
    def model(category, params, arch):
        return None

    assert external_op_model(model)(op_work(_mm()), ARCH) is None


@pytest.mark.parametrize(
    "report_module",
    [
        "TraceLens.Reporting.generate_perf_report_pytorch",
        "TraceLens.Reporting.generate_perf_report_pytorch_inference",
    ],
)
def test_extension_file_registers_external_model(report_module):
    registered = []
    analyzer = SimpleNamespace(set_external_op_model=registered.append)
    importlib.import_module(report_module).apply_extension(analyzer, str(STUB))
    assert len(registered) == 1
    output = registered[0]("GEMM", {"M": 1, "N": 1, "K": 1, "B": 1}, None)
    assert output["time_us"] == pytest.approx(2e-8)
    assert output["Transpose"] == "unknown"


def test_origami_model_passes_gemm_shape_to_origami():
    gemm = _mm()
    with patch.object(origami_helper, "gemm_time_us", return_value=7.0) as sim:
        assert origami_gemm_model("GEMM", gemm.param_details, ARCH) == 7.0
        assert origami_gemm_model("SDPA_fwd", gemm.param_details, ARCH) is None
        assert origami_gemm_model("GEMM", gemm.param_details, None) is None
    sim.assert_called_once_with(ARCH, 128, 32, 64, 1, "bf16")


def test_set_external_op_model():
    analyzer = SimpleNamespace(op_models={})
    analyzer.register_op_model = partial(TreePerfAnalyzer.register_op_model, analyzer)
    TreePerfAnalyzer.set_external_op_model(analyzer, len)
    assert list(analyzer.op_models) == ["External"]
    assert analyzer.op_models["External"].external_model is len
    TreePerfAnalyzer.set_external_op_model(analyzer, None)
    assert analyzer.op_models == {}
    with pytest.raises(TypeError):
        TreePerfAnalyzer.set_external_op_model(analyzer, "not a function")


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
