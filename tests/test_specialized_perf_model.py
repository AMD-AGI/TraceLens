###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Time models: built-in Origami and an extension-registered specialized model."""

import importlib
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from TraceLens.PerfModel import perf_model
from TraceLens.PerfModel.perf_model import aten_mm
from TraceLens.PerfModel.time_models import (
    builtin_origami_model,
    origami_perf_model,
    predict_time,
)
from TraceLens.Reporting.generate_perf_report_pytorch import (
    generate_perf_report_pytorch,
)
from TraceLens.TreePerf.tree_perf import TreePerfAnalyzer

REPO = Path(__file__).resolve().parents[1]
STUB = REPO / "examples" / "specialized_perf_model_stub.py"
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
    assert predict_time(model, gemm, ARCH) == 12.0
    assert seen["category"] == "GEMM"
    assert (seen["params"]["N"], seen["params"]["K"]) == (32, 64)
    assert seen["arch"] is ARCH
    assert gemm.param_details["M"] == 128


def test_predict_time_returns_none_without_a_prediction():
    def model(category, params, arch):
        return None

    assert predict_time(model, _mm(), ARCH) is None
    assert predict_time(len, SimpleNamespace(category="GEMM"), ARCH) is None
    assert predict_time(len, SimpleNamespace(param_details={}), ARCH) is None


@pytest.mark.parametrize(
    "report_module",
    [
        "TraceLens.Reporting.generate_perf_report_pytorch",
        "TraceLens.Reporting.generate_perf_report_pytorch_inference",
    ],
)
def test_extension_file_registers_specialized_model(report_module):
    registered = []
    analyzer = SimpleNamespace(set_specialized_perf_model=registered.append)
    importlib.import_module(report_module).apply_extension(analyzer, str(STUB))
    assert len(registered) == 1
    assert registered[0](
        "GEMM", {"M": 1, "N": 1, "K": 1, "B": 1}, None
    ) == pytest.approx(2e-8)


def test_origami_model_passes_gemm_shape_to_the_simulator():
    gemm = _mm()
    with patch.object(
        perf_model.GEMM, "get_simulation_time_func", return_value=(7.0, "cmd")
    ) as sim:
        assert origami_perf_model("GEMM", gemm.param_details, ARCH) == 7.0
        assert origami_perf_model("SDPA_fwd", gemm.param_details, ARCH) is None
        assert origami_perf_model("GEMM", gemm.param_details, None) is None
    sim.assert_called_once_with(ARCH, 128, 32, 64, 1, "bf16", None, enable_origami=True)


def test_origami_is_registered_only_when_enabled(monkeypatch):
    monkeypatch.delenv("GEMM_SIMULATOR_PATH", raising=False)
    assert builtin_origami_model(False) is None
    assert builtin_origami_model(True) is not None
    monkeypatch.setenv("GEMM_SIMULATOR_PATH", "sim.py")
    assert builtin_origami_model(False) is not None


def test_set_specialized_perf_model():
    analyzer = SimpleNamespace(time_estimators={})
    analyzer.register_time_model = partial(
        TreePerfAnalyzer.register_time_model, analyzer
    )
    TreePerfAnalyzer.set_specialized_perf_model(analyzer, len)
    assert list(analyzer.time_estimators) == ["Specialized"]
    assert analyzer.time_estimators["Specialized"].time_model is len
    TreePerfAnalyzer.set_specialized_perf_model(analyzer, None)
    assert analyzer.time_estimators == {}
    with pytest.raises(TypeError):
        TreePerfAnalyzer.set_specialized_perf_model(analyzer, "not a function")


def test_report_has_origami_and_specialized_columns_for_gemms(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMM_SIMULATOR_PATH", raising=False)
    with patch.object(
        perf_model.GEMM, "get_simulation_time_func", return_value=(1.0, "cmd")
    ):
        dfs = generate_perf_report_pytorch(
            profile_json_path=str(TRACE),
            output_csvs_dir=str(tmp_path / "csvs"),
            extension_file=str(STUB),
            gpu_arch=ARCH,
            enable_origami=True,
            collective_analysis=False,
        )
    summary = dfs["unified_perf_summary"]
    gemms = summary[summary["op category"] == "GEMM"]
    assert not gemms.empty
    assert (gemms["Origami Time (µs)_first"] == 1.0).all()

    specialized = summary["Specialized Time (µs)_first"]
    assert (summary.loc[specialized.notna(), "op category"] == "GEMM").all()
    p = gemms.iloc[0]["perf_params"]
    expected = 2 * p["M"] * p["N"] * p["K"] * p["B"] / 1e8
    assert gemms.iloc[0]["Specialized Time (µs)_first"] == pytest.approx(expected)
    assert "Specialized Time (µs)_first" in dfs["GEMM"].columns


def test_report_without_models_has_no_simulated_columns(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMM_SIMULATOR_PATH", raising=False)
    dfs = generate_perf_report_pytorch(
        profile_json_path=str(TRACE),
        output_csvs_dir=str(tmp_path / "csvs"),
        gpu_arch=ARCH,
        collective_analysis=False,
    )
    columns = dfs["unified_perf_summary"].columns
    assert not [c for c in columns if "Origami" in c or "Specialized" in c]
