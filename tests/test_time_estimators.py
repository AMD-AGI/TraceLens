###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Time estimators: OpWork, the built-in estimators, registration, columns."""

import math
import warnings
from dataclasses import replace
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from TraceLens.PerfModel import perf_model
from TraceLens.PerfModel.time_models import (
    OpWork,
    TimeEstimate,
    add_time_estimate_columns,
    add_time_estimates,
    class_simulation_time,
    default_time_estimators,
    external_time_model,
    gemm_simulator_estimator,
    op_work,
    origami_estimator,
    roofline_estimator,
    time_estimate_group_columns,
    time_estimate_labels,
)
from TraceLens.Reporting.generate_perf_report_pytorch import (
    apply_extension,
    generate_perf_report_pytorch,
)
from TraceLens.TreePerf.tree_perf import TreePerfAnalyzer, kernel_filter_labels

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

    def test_backward_gets_the_backward_category(self):
        model = external_time_model(lambda category, params, arch: len(category))
        assert model(_work(category="SDPA_bwd", bwd=True), ARCH) == 8.0
        assert model(_work(category=None, bwd=True), ARCH) is None

    def test_skips_unknown_category(self):
        model = external_time_model(lambda category, params, arch: 1.0)
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
        work = _work(perf_model=pm, params=pm.param_details)
        with patch.object(
            perf_model.GEMM, "get_simulation_time_func", return_value=(7.0, "cmd")
        ) as sim:
            assert origami_estimator(False)(work, ARCH) is None
            assert origami_estimator(True)(work, ARCH) == 7.0
            assert origami_estimator(True)(replace(work, bwd=True), ARCH) is None
        assert sim.call_args.kwargs["backend"] == "origami"

    def test_default_estimators_order(self, monkeypatch):
        monkeypatch.delenv("GEMM_SIMULATOR_PATH", raising=False)
        assert list(default_time_estimators()) == ["Roofline", "Origami"]
        monkeypatch.setenv("GEMM_SIMULATOR_PATH", "sim.py")
        assert list(default_time_estimators()) == [
            "Roofline",
            "Origami",
            "GEMM Simulator",
        ]


class TestGemmSimulatorEstimator:
    def test_class_simulation_gets_the_backend(self):
        class WithBackend:
            def get_simulation_time(self, backend=None):
                return {"origami": 1.0, "simulator": 2.0}[backend]

        class Legacy:
            def get_simulation_time(self):
                return 3.0

        assert class_simulation_time(WithBackend(), False, "simulator") == 2.0
        assert class_simulation_time(WithBackend(), False, "origami") == 1.0
        assert class_simulation_time(Legacy(), False, "origami") == 3.0
        assert class_simulation_time(Legacy(), False, "simulator") is None
        assert class_simulation_time(object(), False, "origami") is None

    def test_gemm_params_use_the_simulator(self):
        params = {**_work().params, "dtype_A_B": ("c10::BFloat16",)}
        pm = SimpleNamespace(category="GEMM", param_details=params)
        work = _work(perf_model=pm, params=params)
        with patch.object(
            perf_model.GEMM, "get_simulation_time_func", return_value=(7.0, "cmd")
        ) as sim:
            assert gemm_simulator_estimator()(work, ARCH) == 7.0
            assert gemm_simulator_estimator()(replace(work, bwd=True), ARCH) is None
        assert sim.call_args.kwargs["backend"] == "simulator"

    def test_unknown_backend(self):
        with pytest.raises(ValueError, match="backend"):
            perf_model.GEMM.get_simulation_time_func(
                ARCH, 4, 8, 16, 1, "bf16", backend="other"
            )

    def test_origami_backend_ignores_the_simulator_path(self, monkeypatch):
        monkeypatch.setenv("GEMM_SIMULATOR_PATH", "/nonexistent/sim.py")
        assert perf_model.GEMM.get_simulation_time_func(
            ARCH, 4, 8, 16, 1, "bf16", backend="origami"
        ) == (None, None)

    def test_sdpa_passes_the_backend_to_its_tile_gemms(self):
        backends = []

        def fake_gemm(*args, backend=None, **kwargs):
            backends.append(backend)
            return 1.0, "cmd"

        arch = {**ARCH, "num_cus": 304}
        with patch.object(
            perf_model.GEMM, "get_simulation_time_func", side_effect=fake_gemm
        ), patch.object(perf_model.Softmax, "get_time", return_value=0.0):
            args = (arch, "bf16", None, "c10::BFloat16", 1024, 1, 8, 128, 128, 64)
            perf_model.SDPA.get_simulation_time_func(*args, backend="simulator")
            perf_model.SDPA.get_simulation_time_bwd_func(*args, backend="simulator")
        assert backends == ["simulator"] * 4


class _Attention(perf_model.SDPA):
    @staticmethod
    def get_param_details(event):
        dims = dict(B=1, N_Q=128, H_Q=8, N_KV=128, H_KV=8, d_h_qk=64, d_h_v=64)
        return {**dims, "causal": False, "dtype_A_B": ("c10::BFloat16",)}


class TestSimulationWarning:
    @pytest.fixture(autouse=True)
    def _fresh(self, monkeypatch):
        monkeypatch.delenv("GEMM_SIMULATOR_PATH", raising=False)
        monkeypatch.setattr(perf_model.SDPA, "_simulation_warnings", set())

    def test_warns_once_when_a_requested_simulation_fails(self):
        model = _Attention({}, arch={"name": "mi300x"}, enable_origami=True)
        with pytest.warns(RuntimeWarning, match="_Attention: no simulated time"):
            assert model.get_simulation_time(backend="origami") is None
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert model.get_simulation_time(backend="origami") is None
        with pytest.warns(RuntimeWarning, match="no simulated backward time"):
            assert model.get_simulation_time_bwd(backend="origami") is None

    def test_silent_when_no_simulation_was_asked_for(self):
        model = _Attention({}, arch={"name": "mi300x"})
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert model.get_simulation_time(backend="origami") is None


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
        for label in ("Mine", "External", "Origami", "Roofline"):
            columns += [f"{label} Time (µs)", f"Pct {label}"]
        columns += ["Non-Data-Mov Kernel Time (µs)", "Pct Nothing"]
        assert time_estimate_labels(columns) == [
            "Roofline",
            "Origami",
            "External",
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


class _FakeAttention:
    category = "SDPA_fwd"
    bwd_category = "SDPA_bwd"
    param_details = {"N_Q": 128}

    def flops(self):
        return 2e9

    def bytes(self):
        return 1e6

    def flops_bwd(self):
        return 5e9

    def bytes_bwd(self):
        return 2e6


@pytest.mark.parametrize(
    "bwd, category, gflops", [(False, "SDPA_fwd", 2.0), (True, "SDPA_bwd", 5.0)]
)
def test_external_model_sees_the_op_direction(bwd, category, gflops):
    seen = []

    def model(category, params, arch):
        seen.append((category, params))
        return 10.0

    analyzer = SimpleNamespace(arch=ARCH, time_estimators={})
    TreePerfAnalyzer.register_time_model(analyzer, "Mine", model)
    metrics = {}
    work = op_work(_FakeAttention(), bwd=bwd)
    add_time_estimates(metrics, analyzer.time_estimators, work, ARCH, 20.0)
    assert seen == [(category, {"N_Q": 128})]
    assert metrics["Mine Time (µs)"] == 10.0
    assert metrics["Mine TFLOPS/s"] == pytest.approx(gflops / 10.0 * 1e3)
    assert metrics["Pct Mine"] == 50.0


def test_register_kernel_filter():
    analyzer = SimpleNamespace(kernel_filters={})
    register = partial(TreePerfAnalyzer.register_kernel_filter, analyzer)
    register("NDM", TreePerfAnalyzer.non_data_mov_filter)
    assert analyzer.kernel_filters == {"NDM": TreePerfAnalyzer.non_data_mov_filter}
    with pytest.raises(TypeError):
        register("Bad", 1)
    register("NDM", None)
    assert analyzer.kernel_filters == {}


def test_kernel_filter_labels():
    columns = ["Kernel Time (µs)", "A Kernel Time (µs)", "A TFLOPS/s", "B Time (µs)"]
    assert kernel_filter_labels(columns) == ["A"]


def test_report_shows_every_kernel_filter(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMM_SIMULATOR_PATH", raising=False)
    ext = tmp_path / "ext.py"
    ext.write_text(
        "kernel_filters = {'All': lambda k: True, 'Empty': lambda k: False}\n"
    )
    dfs = generate_perf_report_pytorch(
        profile_json_path=str(TRACE),
        output_csvs_dir=str(tmp_path / "csvs"),
        extension_file=str(ext),
        gpu_arch=ARCH,
        collective_analysis=False,
    )
    gemm = dfs["GEMM"]
    assert gemm["All Kernel Time (µs)_sum"].tolist() == pytest.approx(
        gemm["Kernel Time (µs)_sum"].tolist()
    )
    assert gemm["All TFLOPS/s_mean"].tolist() == pytest.approx(
        gemm["TFLOPS/s_mean"].tolist()
    )
    assert (gemm["Empty Kernel Time (µs)_sum"] == 0).all()
    summary = dfs["unified_perf_summary"]
    gemms = summary[summary["op category"] == "GEMM"]
    assert gemms["All Kernel Time (µs)_sum"].tolist() == pytest.approx(
        gemms["Kernel Time (µs)_sum"].tolist()
    )


@pytest.mark.parametrize("enable_origami", [False, True])
def test_report_has_gemm_simulator_columns_next_to_origami(
    tmp_path, monkeypatch, enable_origami
):
    monkeypatch.setenv("GEMM_SIMULATOR_PATH", "sim.py")

    def fake_gemm(*args, backend=None, **kwargs):
        return {"simulator": 3.0, "origami": 5.0}[backend], "cmd"

    with patch.object(
        perf_model.GEMM, "get_simulation_time_func", side_effect=fake_gemm
    ):
        dfs = generate_perf_report_pytorch(
            profile_json_path=str(TRACE),
            output_csvs_dir=str(tmp_path / "csvs"),
            gpu_arch=ARCH,
            enable_origami=enable_origami,
            collective_analysis=False,
        )
    for df in (
        dfs["GEMM"],
        dfs["unified_perf_summary"].query("`op category` == 'GEMM'"),
    ):
        assert (df["GEMM Simulator Time (µs)_first"] == 3.0).all()
        assert "Pct GEMM Simulator_mean" in df.columns
        if enable_origami:
            assert (df["Origami Time (µs)_first"] == 5.0).all()
        else:
            assert (
                df.get("Origami Time (µs)_first", pd.Series(dtype=float)).isna().all()
            )


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
