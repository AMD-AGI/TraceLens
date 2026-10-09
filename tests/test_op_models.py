###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Op models: OpWork, the built-in op models, registration, columns."""

import importlib.util
import math
import warnings
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from TraceLens.PerfModel import perf_model
from TraceLens.PerfModel.op_models import (
    OpWork,
    add_op_model_columns,
    add_op_model_outputs,
    default_op_models,
    external_op_model,
    op_model_group_columns,
    op_model_labels,
    op_work,
    origami_gemm_model,
    roofline_op_model,
    sdpa_tile_origami_op_model,
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
        output = roofline_op_model(_work(gflops=1.0, bytes_moved=1e3), ARCH)
        assert output["time_us"] == pytest.approx(1e9 / 500e12 * 1e6)
        assert output["Bound"] == "COMPUTE_BOUND"

    def test_memory_bound(self):
        output = roofline_op_model(_work(gflops=1e-6, bytes_moved=1e9), ARCH)
        assert output["time_us"] == pytest.approx(1e9 / 5000e9 * 1e6)
        assert output["Bound"] == "MEMORY_BOUND"

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
        assert roofline_op_model(work, arch) is None


class TestExternalModel:
    def test_gets_category_a_copy_of_params_and_arch(self):
        seen = {}

        def model(category, params, arch):
            seen.update(category=category, params=params, arch=arch)
            params["M"] = -1
            return 3

        work = _work()
        assert external_op_model(model)(work, ARCH) == 3.0
        assert seen["category"] == "GEMM"
        assert seen["arch"] is ARCH
        assert seen["params"] is not work.params
        assert work.params["M"] == 4

    def test_backward_gets_the_backward_category(self):
        model = external_op_model(lambda category, params, arch: len(category))
        assert model(_work(category="SDPA_bwd", bwd=True), ARCH) == 8.0
        assert model(_work(category=None, bwd=True), ARCH) is None

    def test_skips_unknown_category(self):
        model = external_op_model(lambda category, params, arch: 1.0)
        assert model(_work(category=None), ARCH) is None
        assert model(_work(params=None), ARCH) is None

    def test_dict_output_passes_through(self):
        output = {"time_us": 2.0, "Config": "256x128"}
        model = external_op_model(lambda category, params, arch: output)
        assert model(_work(), ARCH) is output


class TestOpModelErrors:
    @pytest.fixture(autouse=True)
    def _fresh(self, monkeypatch):
        from TraceLens.PerfModel import op_models

        monkeypatch.setattr(op_models, "_op_model_warnings", set())

    def test_failing_model_only_loses_its_own_columns(self):
        def broken(category, params, arch):
            raise ValueError("unsupported dtype")

        models = {
            "Roofline": roofline_op_model,
            "Broken": external_op_model(broken),
            "Fine": external_op_model(lambda category, params, arch: 4.0),
        }
        metrics = {}
        with pytest.warns(RuntimeWarning, match="'Broken' failed on a GEMM op"):
            add_op_model_outputs(metrics, models, _work(), ARCH, 8.0)
        assert "Roofline Time (µs)" in metrics
        assert metrics["Fine Time (µs)"] == 4.0
        assert not [col for col in metrics if "Broken" in col]

    def test_warns_once_per_model_category_and_error(self):
        models = {"Broken": lambda work, arch: 1 / 0}
        with pytest.warns(RuntimeWarning):
            add_op_model_outputs({}, models, _work(), ARCH, 8.0)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            add_op_model_outputs({}, models, _work(), ARCH, 8.0)
        with pytest.warns(RuntimeWarning, match="SDPA_fwd"):
            add_op_model_outputs({}, models, _work(category="SDPA_fwd"), ARCH, 8.0)


class TestOrigamiGemm:
    def test_forward_gemms_only(self):
        params = {**_work().params, "dtype_A_B": ("c10::BFloat16",)}
        origami = external_op_model(origami_gemm_model)
        with patch.object(
            perf_model.GEMM, "get_simulation_time_func", return_value=(7.0, "cmd")
        ) as sim:
            assert origami(_work(params=params), ARCH) == 7.0
            assert origami(_work(params=params, category=None), ARCH) is None
            assert origami(_work(category="SDPA_fwd", params={}), ARCH) is None
            assert origami_gemm_model("GEMM", params, None) is None
        assert sim.call_args.kwargs["enable_origami"] is True

    def test_without_origami_enabled_gives_no_time(self):
        assert perf_model.GEMM.get_simulation_time_func(ARCH, 4, 8, 16, 1, "bf16") == (
            None,
            None,
        )

    def test_default_op_models_order(self):
        assert list(default_op_models()) == ["Roofline"]
        assert list(default_op_models(enable_origami_gemm=True)) == [
            "Roofline",
            "Origami",
        ]
        assert list(default_op_models(enable_origami_sdpa_tile=True)) == [
            "Roofline",
            "SDPA Tile Origami",
        ]
        assert list(default_op_models(True, True)) == [
            "Roofline",
            "Origami",
            "SDPA Tile Origami",
        ]


class TestSdpaTile:
    def test_uses_the_class_simulation_for_forward_and_backward(self):
        pm = SimpleNamespace(
            get_simulation_time=lambda: 2.0, get_simulation_time_bwd=lambda: 5.0
        )
        assert sdpa_tile_origami_op_model(_work(perf_model=pm), ARCH) == 2.0
        assert sdpa_tile_origami_op_model(_work(perf_model=pm, bwd=True), ARCH) == 5.0
        assert sdpa_tile_origami_op_model(_work(perf_model=object()), ARCH) is None

    def test_tile_gemms_use_origami_by_default(self):
        calls = []

        def fake_gemm(*args, **kwargs):
            calls.append(kwargs)
            return 1.0, "cmd"

        arch = {**ARCH, "num_cus": 304}
        with patch.object(
            perf_model.GEMM, "get_simulation_time_func", side_effect=fake_gemm
        ), patch.object(perf_model.Softmax, "get_time", return_value=0.0):
            args = (arch, "bf16", None, "c10::BFloat16", 1024, 1, 8, 128, 128, 64)
            perf_model.SDPA.get_simulation_time_func(*args, enable_origami=True)
            perf_model.SDPA.get_simulation_time_bwd_func(*args, enable_origami=True)
        assert len(calls) == 4
        assert all(c["enable_origami"] and c["num_cus"] == 1 for c in calls)

    def test_tile_gemms_can_use_another_gemm_model(self):
        shapes = []

        def gemm_time(arch, M, N, K, B, dtype, force_to_l1, num_cus):
            shapes.append((M, N, K, B, num_cus))
            return 2.0

        arch = {**ARCH, "num_cus": 304}
        with patch.object(
            perf_model.GEMM, "get_simulation_time_func"
        ) as origami, patch.object(perf_model.Softmax, "get_time", return_value=0.0):
            args = (arch, "bf16", None, "c10::BFloat16", 1024, 1, 8, 256, 256, 64)
            fwd = perf_model.SDPA.get_simulation_time_func(*args, gemm_time=gemm_time)
        origami.assert_not_called()
        assert shapes == [(128, 256, 64, 1, 1), (128, 64, 256, 1, 1)]
        assert fwd > 4.0


class _Attention(perf_model.SDPA):
    @staticmethod
    def get_param_details(event):
        dims = dict(B=1, N_Q=128, H_Q=8, N_KV=128, H_KV=8, d_h_qk=64, d_h_v=64)
        return {**dims, "causal": False, "dtype_A_B": ("c10::BFloat16",)}


class TestSimulationWarning:
    @pytest.fixture(autouse=True)
    def _fresh(self, monkeypatch):
        monkeypatch.setattr(perf_model.SDPA, "_simulation_warnings", set())

    def test_warns_once_when_a_requested_simulation_fails(self):
        model = _Attention({}, arch={"name": "mi300x"}, enable_origami=True)
        with pytest.warns(RuntimeWarning, match="_Attention: no simulated time"):
            assert model.get_simulation_time() is None
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert model.get_simulation_time() is None
        with pytest.warns(RuntimeWarning, match="no simulated backward time"):
            assert model.get_simulation_time_bwd() is None

    def test_a_gemm_time_function_counts_as_asked_for(self):
        model = _Attention({}, arch={"name": "mi300x"})
        with pytest.warns(RuntimeWarning, match="_Attention: no simulated time"):
            assert model.get_simulation_time(gemm_time=lambda *a, **k: 1.0) is None

    def test_silent_when_no_simulation_was_asked_for(self):
        model = _Attention({}, arch={"name": "mi300x"})
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert model.get_simulation_time() is None


class TestColumns:
    def test_dict_output_with_extra_columns(self):
        metrics = {}
        add_op_model_columns(
            metrics,
            "Roofline",
            {"time_us": 2.0, "Bound": "X"},
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

    def test_dict_output_without_a_time_keeps_its_columns(self):
        metrics = {}
        add_op_model_columns(
            metrics, "Tiles", {"Tile": "256x128", "Waves": 3}, 4.0, 2e6, 8.0
        )
        assert metrics["Tiles Tile"] == "256x128"
        assert metrics["Tiles Waves"] == 3
        assert math.isnan(metrics["Tiles Time (µs)"])
        assert math.isnan(metrics["Pct Tiles"])
        assert "Tiles TFLOPS/s" not in metrics

    def test_only_scalar_values_are_written(self):
        metrics = {}
        add_op_model_columns(
            metrics,
            "X",
            {"time_us": 1.0, "Name": "a", "N": 2, "Ok": True, "List": [1], "D": {}},
            1.0,
            1.0,
            1.0,
        )
        assert {"X Name", "X N", "X Ok"} <= set(metrics)
        assert "X List" not in metrics and "X D" not in metrics

    @pytest.mark.parametrize("output", [None, 0, 0.0, {}, {"time_us": None}])
    def test_missing_output_writes_nothing(self, output):
        metrics = {}
        add_op_model_columns(metrics, "X", output, 1.0, 1.0, 1.0)
        assert metrics == {}

    def test_zero_measured_time_gives_nan_pct(self):
        metrics = {}
        add_op_model_columns(metrics, "X", 1.0, 1.0, None, 0)
        assert math.isnan(metrics["Pct X"])
        assert math.isnan(metrics["X TB/s"])

    def test_labels_put_builtins_first(self):
        columns = []
        for label in ("Mine", "External", "SDPA Tile Origami", "Origami", "Roofline"):
            columns += [f"{label} Time (µs)", f"Pct {label}"]
        columns += ["Non-Data-Mov Kernel Time (µs)", "Pct Nothing"]
        assert op_model_labels(columns) == [
            "Roofline",
            "Origami",
            "SDPA Tile Origami",
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
        labels = op_model_labels(columns)
        assert op_model_group_columns(columns, "GEMM", labels) == [
            "GEMM Time (µs)",
            "GEMM TB/s",
            "Pct GEMM",
        ]
        assert op_model_group_columns(columns, "GEMM Sim", labels) == [
            "GEMM Sim Time (µs)",
            "GEMM Sim TB/s",
            "Pct GEMM Sim",
        ]


def test_extension_op_models_dict(tmp_path):
    ext = tmp_path / "ext.py"
    ext.write_text("op_models = {'A': lambda c, p, a: 1.0, 'B': None}\n")
    registered = []
    analyzer = SimpleNamespace(
        register_op_model=lambda label, model: registered.append((label, model))
    )
    apply_extension(analyzer, str(ext))
    assert [label for label, _ in registered] == ["A", "B"]

    ext.write_text("op_models = [1]\n")
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

    analyzer = SimpleNamespace(arch=ARCH, op_models={})
    TreePerfAnalyzer.register_op_model(analyzer, "Mine", model)
    metrics = {}
    work = op_work(_FakeAttention(), bwd=bwd)
    add_op_model_outputs(metrics, analyzer.op_models, work, ARCH, 20.0)
    assert seen == [(category, {"N_Q": 128})]
    assert metrics["Mine Time (µs)"] == 10.0
    assert metrics["Mine TFLOPS/s"] == pytest.approx(gflops / 10.0 * 1e3)
    assert metrics["Pct Mine"] == 50.0


def _example_filter():
    spec = importlib.util.spec_from_file_location(
        "kernel_filter_example", REPO / "examples/kernel_filter_example.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.non_data_mov_filter


def test_example_kernel_filter():
    keep = _example_filter()
    assert keep({"name": "Cijk_Alik_Bljk_gemm"})
    assert not keep({"name": "void at::native::direct_copy_kernel_cuda<...>"})
    assert not keep({"name": "batched_transpose_32x32"})


def test_register_kernel_filter():
    keep = _example_filter()
    analyzer = SimpleNamespace(kernel_filters={})
    register = partial(TreePerfAnalyzer.register_kernel_filter, analyzer)
    register("NDM", keep)
    assert analyzer.kernel_filters == {"NDM": keep}
    with pytest.raises(TypeError):
        register("Bad", 1)
    register("NDM", None)
    assert analyzer.kernel_filters == {}


def test_kernel_filter_labels():
    columns = ["Kernel Time (µs)", "A Kernel Time (µs)", "A TFLOPS/s", "B Time (µs)"]
    assert kernel_filter_labels(columns) == ["A"]


def test_report_shows_every_kernel_filter(tmp_path):
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


@pytest.mark.parametrize("enable_origami_gemm", [False, True])
def test_report_has_origami_columns_only_with_the_flag(tmp_path, enable_origami_gemm):
    with patch.object(
        perf_model.GEMM, "get_simulation_time_func", return_value=(5.0, "cmd")
    ):
        dfs = generate_perf_report_pytorch(
            profile_json_path=str(TRACE),
            output_csvs_dir=str(tmp_path / "csvs"),
            gpu_arch=ARCH,
            enable_origami_gemm=enable_origami_gemm,
            collective_analysis=False,
        )
    for df in (
        dfs["GEMM"],
        dfs["unified_perf_summary"].query("`op category` == 'GEMM'"),
    ):
        if enable_origami_gemm:
            assert (df["Origami Time (µs)_first"] == 5.0).all()
            assert "Pct Origami_mean" in df.columns
        else:
            assert (
                df.get("Origami Time (µs)_first", pd.Series(dtype=float)).isna().all()
            )
    assert "Origami Time (µs)_first" not in dfs["SDPA_fwd"].columns


def test_report_has_sdpa_tile_columns_only_with_the_flag(tmp_path):
    arch = {**ARCH, "num_cus": 304}

    def report(out, **flags):
        with patch.object(
            perf_model.GEMM, "get_simulation_time_func", return_value=(3.0, "cmd")
        ), patch.object(perf_model.Softmax, "get_time", return_value=0.0):
            return generate_perf_report_pytorch(
                profile_json_path=str(TRACE),
                output_csvs_dir=str(tmp_path / out),
                gpu_arch=arch,
                collective_analysis=False,
                **flags,
            )

    label = "SDPA Tile Origami Time (µs)_first"
    gemm_only = report("gemm", enable_origami_gemm=True)
    assert label not in gemm_only["SDPA_fwd"].columns

    tiled = report("tiled", enable_origami_sdpa_tile=True)
    sdpa = tiled["SDPA_fwd"]
    assert (sdpa[label] > 0).all()
    assert "Pct SDPA Tile Origami_mean" in sdpa.columns
    assert "Origami Time (µs)_first" not in sdpa.columns
    assert label not in tiled["GEMM"].columns
    assert "Origami Time (µs)_first" not in tiled["GEMM"].columns


def test_report_keeps_ops_when_a_model_fails_and_shows_extra_columns(tmp_path):
    ext = tmp_path / "ext.py"
    ext.write_text(
        "def configured(category, params, arch):\n"
        "    if category != 'GEMM':\n"
        "        return None\n"
        "    return {'time_us': 3.0, 'Config': f\"{params['M']}x{params['N']}\"}\n"
        "def tiles_only(category, params, arch):\n"
        "    return {'Tile': '64x64'} if category == 'GEMM' else None\n"
        "def broken(category, params, arch):\n"
        "    raise ValueError('unsupported')\n"
        "op_models = {'Configured': configured, 'Tiles': tiles_only, "
        "'Broken': broken}\n"
    )
    with pytest.warns(RuntimeWarning, match="'Broken' failed"):
        dfs = generate_perf_report_pytorch(
            profile_json_path=str(TRACE),
            output_csvs_dir=str(tmp_path / "csvs"),
            extension_file=str(ext),
            gpu_arch=ARCH,
            collective_analysis=False,
        )
    for df in (
        dfs["GEMM"],
        dfs["unified_perf_summary"].query("`op category` == 'GEMM'"),
    ):
        assert df["Roofline Time (µs)_first"].notna().all()
        assert (df["Configured Time (µs)_first"] == 3.0).all()
        assert (df["Tiles Tile_first"] == "64x64").all()
        assert df["Tiles Time (µs)_first"].isna().all()
        columns = list(df.columns)
        config = columns.index("Configured Config_first")
        assert columns.index("Configured TB/s_first") < config
        assert config < columns.index("Pct Configured_mean")
        assert not [col for col in columns if "Broken" in col]
    gemm = dfs["GEMM"].iloc[0]
    assert gemm["Configured Config_first"] == (
        f"{int(gemm['param: M'])}x{int(gemm['param: N'])}"
    )


def test_report_shows_every_registered_model(tmp_path):
    ext = tmp_path / "ext.py"
    ext.write_text(
        "def gemm_only(category, params, arch):\n"
        "    return 3.0 if category == 'GEMM' else None\n"
        "op_models = {'ModelA': gemm_only, 'ModelB': lambda c, p, a: 4.0}\n"
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
