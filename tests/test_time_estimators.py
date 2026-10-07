###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Time estimators: OpWork, the built-in estimators, registration, columns."""

import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from TraceLens.PerfModel import perf_model
from TraceLens.PerfModel.time_models import (
    OpWork,
    TimeEstimate,
    add_time_estimate_columns,
    default_time_estimators,
    external_time_model,
    origami_estimator,
    roofline_estimator,
    time_estimate_group_columns,
    time_estimate_labels,
)
from TraceLens.Reporting.generate_perf_report_pytorch import (
    apply_extension,
    generate_perf_report_pytorch,
)

REPO = Path(__file__).resolve().parents[1]
TRACE = REPO / "tests/traces/mi300/gaunernst_bert-small-uncased__1016001.json.gz"
ARCH = {
    "name": "mi300x",
    "freq_mhz": 2100,
    "mem_bw_gbps": 5000,
    "max_achievable_tflops": {"matrix_bf16": 500},
}


def _work(**kw):
    defaults = dict(
        category="GEMM",
        params={"M": 4, "N": 8, "K": 16, "B": 1},
        gflops=1.0,
        bytes_moved=1e6,
        compute_spec="matrix_bf16",
    )
    defaults.update(kw)
    return OpWork(**defaults)


class TestRoofline:
    def test_compute_bound(self):
        est = roofline_estimator(_work(gflops=1.0, bytes_moved=1e3), ARCH)
        assert est.time_us == pytest.approx(1e9 / 500e12 * 1e6)
        assert est.extra_columns == {"Roofline Bound": "COMPUTE_BOUND"}

    def test_memory_bound(self):
        est = roofline_estimator(_work(gflops=1e-6, bytes_moved=1e9), ARCH)
        assert est.time_us == pytest.approx(1e9 / 5000e9 * 1e6)
        assert est.extra_columns == {"Roofline Bound": "MEMORY_BOUND"}

    @pytest.mark.parametrize(
        "work, arch",
        [
            (_work(), None),
            (_work(compute_spec=None), ARCH),
            (_work(compute_spec="vector_fp32"), ARCH),
            (_work(bytes_moved=None), ARCH),
            (_work(gflops=0), ARCH),
            (_work(), {"mem_bw_gbps": 5000}),
            (_work(), {"max_achievable_tflops": {"matrix_bf16": 500}}),
        ],
    )
    def test_skips_without_inputs(self, work, arch):
        assert roofline_estimator(work, arch) is None


class TestExternalModel:
    def test_gets_category_a_copy_of_params_and_arch(self):
        seen = {}

        def model(category, params, arch):
            seen.update(category=category, params=params, arch=arch)
            params["M"] = -1
            return 3

        work = _work()
        assert external_time_model(model)(work, ARCH) == 3.0
        assert seen["category"] == "GEMM"
        assert seen["arch"] is ARCH
        assert seen["params"] is not work.params
        assert work.params["M"] == 4

    def test_skips_backward_and_unknown_category(self):
        model = external_time_model(lambda category, params, arch: 1.0)
        assert model(_work(bwd=True), ARCH) is None
        assert model(_work(category=None), ARCH) is None
        assert model(_work(params=None), ARCH) is None


class TestOrigamiEstimator:
    def test_uses_the_class_simulation_for_forward_and_backward(self):
        pm = SimpleNamespace(
            get_simulation_time=lambda: 2.0,
            get_simulation_time_bwd=lambda: 5.0,
            category="SDPA_fwd",
            param_details={},
        )
        estimate = origami_estimator()
        assert estimate(_work(perf_model=pm), ARCH) == 2.0
        assert estimate(_work(perf_model=pm, bwd=True), ARCH) == 5.0

    def test_gemm_params_only_when_enabled(self, monkeypatch):
        monkeypatch.delenv("GEMM_SIMULATOR_PATH", raising=False)
        pm = SimpleNamespace(
            category="GEMM",
            param_details={
                "M": 4,
                "N": 8,
                "K": 16,
                "B": 1,
                "dtype_A_B": ("c10::BFloat16",),
            },
        )
        with patch.object(
            perf_model.GEMM, "get_simulation_time_func", return_value=(7.0, "cmd")
        ):
            assert origami_estimator(False)(_work(perf_model=pm), ARCH) is None
            assert origami_estimator(True)(_work(perf_model=pm), ARCH) == 7.0
            assert origami_estimator(True)(_work(perf_model=pm, bwd=True), ARCH) is None

    def test_default_estimators_order(self):
        assert list(default_time_estimators()) == ["Roofline", "Origami"]


class TestColumns:
    def test_estimate_with_extra_columns(self):
        metrics = {}
        add_time_estimate_columns(
            metrics,
            "Roofline",
            TimeEstimate(2.0, {"Roofline Bound": "X"}),
            4.0,
            2e6,
            8.0,
        )
        assert list(metrics) == [
            "Roofline Time (µs)",
            "Roofline TFLOPS/s",
            "Roofline TB/s",
            "Roofline Bound",
            "Pct Roofline",
        ]
        assert metrics["Roofline TFLOPS/s"] == pytest.approx(2000.0)
        assert metrics["Pct Roofline"] == 25.0

    @pytest.mark.parametrize("estimate", [None, 0, 0.0])
    def test_missing_estimate_writes_nothing(self, estimate):
        metrics = {}
        add_time_estimate_columns(metrics, "X", estimate, 1.0, 1.0, 1.0)
        assert metrics == {}

    def test_zero_measured_time_gives_nan_pct(self):
        metrics = {}
        add_time_estimate_columns(metrics, "X", 1.0, 1.0, None, 0)
        assert math.isnan(metrics["Pct X"])
        assert math.isnan(metrics["X TB/s"])

    def test_labels_put_builtins_first(self):
        columns = []
        for label in ("Mine", "Specialized", "Origami", "Roofline"):
            columns += [f"{label} Time (µs)", f"Pct {label}"]
        columns += ["Non-Data-Mov Kernel Time (µs)", "Pct Nothing"]
        assert time_estimate_labels(columns) == [
            "Roofline",
            "Origami",
            "Specialized",
            "Mine",
        ]

    def test_group_columns_respect_longer_labels(self):
        columns = [
            "GEMM Time (µs)",
            "GEMM TB/s",
            "GEMM Sim Time (µs)",
            "GEMM Sim TB/s",
            "Pct GEMM",
            "Pct GEMM Sim",
        ]
        labels = time_estimate_labels(columns)
        assert time_estimate_group_columns(columns, "GEMM", labels) == [
            "GEMM Time (µs)",
            "GEMM TB/s",
            "Pct GEMM",
        ]
        assert time_estimate_group_columns(columns, "GEMM Sim", labels) == [
            "GEMM Sim Time (µs)",
            "GEMM Sim TB/s",
            "Pct GEMM Sim",
        ]


def test_extension_time_models_dict(tmp_path):
    ext = tmp_path / "ext.py"
    ext.write_text("time_models = {'A': lambda c, p, a: 1.0, 'B': None}\n")
    registered = []
    analyzer = SimpleNamespace(
        register_time_model=lambda label, model: registered.append((label, model))
    )
    apply_extension(analyzer, str(ext))
    assert [label for label, _ in registered] == ["A", "B"]

    ext.write_text("time_models = [1]\n")
    with pytest.raises(TypeError):
        apply_extension(analyzer, str(ext))


def test_report_shows_every_registered_model(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMM_SIMULATOR_PATH", raising=False)
    ext = tmp_path / "ext.py"
    ext.write_text(
        "def gemm_only(category, params, arch):\n"
        "    return 3.0 if category == 'GEMM' else None\n"
        "time_models = {'ModelA': gemm_only, 'ModelB': lambda c, p, a: 4.0}\n"
    )
    dfs = generate_perf_report_pytorch(
        profile_json_path=str(TRACE),
        output_csvs_dir=str(tmp_path / "csvs"),
        extension_file=str(ext),
        gpu_arch=ARCH,
        collective_analysis=False,
    )
    gemm = dfs["GEMM"]
    assert (gemm["ModelA Time (µs)_first"] == 3.0).all()
    assert (gemm["ModelB Time (µs)_first"] == 4.0).all()
    assert "Pct ModelA_mean" in gemm.columns
    summary = dfs["unified_perf_summary"]
    gemms = summary[summary["op category"] == "GEMM"]
    assert (gemms["ModelA Time (µs)_first"] == 3.0).all()
    others = summary[summary["ModelB Time (µs)_first"].notna()]
    assert set(others["op category"]) > {"GEMM"}
    assert "ModelA Time (µs)_first" not in dfs["SDPA_fwd"].columns
    assert (dfs["SDPA_fwd"]["ModelB Time (µs)_first"] == 4.0).all()
