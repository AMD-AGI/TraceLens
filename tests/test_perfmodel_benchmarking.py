###############################################################################
# Copyright (c) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for TraceLens/PerfModel/benchmarking.

CPU-only helpers run in CI without torch. Torch/CUDA benchmarks are imported
lazily and skipped unless a GPU runtime is available.
"""

from __future__ import annotations

import importlib.util, json, pytest, statistics, time, types
from pathlib import Path
from unittest.mock import MagicMock, patch
from TraceLens.PerfModel.benchmarking.microbench_utils import (
    MeasurementError,
    SysfsTelemetry,
    _int_metric,
    bandwidth_plateau,
    check_gpu_idle,
    check_rdna_gpu_idle,
    checked_counter_value,
    counter_clock_mhz,
    cu_tiled_sweep_shapes,
    gfx_target,
    mx_peak_without_native_support,
    parse_counter_list,
    parse_rdna_counters,
    pci_bdf,
    platform_record,
    power_held_w,
    rdna_family,
    rdna_published_keys,
    resolve_physical_device,
    resolve_rdna_window,
    run_python_probe,
    VECTOR_FMA_BLOCK,
    VECTOR_FMA_BLOCK_DEFAULT,
    vector_fma_block,
    vopd_share,
)

# These modules import torch; without it they stay None and the tests that use them skip in _require_torch.
try:
    from TraceLens.PerfModel.benchmarking import (
        fp4fp6_helpers,
        microbench,
        microbench_rdna,
        microbench_rocprof,
    )
except ImportError:
    fp4fp6_helpers = microbench = microbench_rdna = microbench_rocprof = None

HAS_TORCH = importlib.util.find_spec("torch") is not None


def _require_torch():
    return pytest.importorskip("torch")


def _require_cuda_gpu():
    torch = _require_torch()
    if not torch.cuda.is_available():
        pytest.skip("Requires CUDA/HIP with at least one visible GPU")
    return torch


def _import_microbench():
    _require_torch()
    return microbench


def _import_microbench_rdna():
    _require_torch()
    return microbench_rdna


def _import_microbench_rocprof():
    _require_torch()
    return microbench_rocprof


def _import_fp4fp6_helpers():
    _require_torch()
    return fp4fp6_helpers


class TestMicrobenchUtils:
    @pytest.mark.parametrize(
        "logical,env,expected_phys,expected_src_part",
        [
            (0, {}, 0, "identity"),
            (1, {"HIP_VISIBLE_DEVICES": "2,3"}, 3, "HIP_VISIBLE_DEVICES"),
            (0, {"CUDA_VISIBLE_DEVICES": "5"}, 5, "CUDA_VISIBLE_DEVICES"),
            (2, {"HIP_VISIBLE_DEVICES": "0,1"}, 2, "out of range"),
            (0, {"HIP_VISIBLE_DEVICES": "a,b"}, 0, "non-numeric"),
        ],
    )
    def test_resolve_physical_device(
        self, logical, env, expected_phys, expected_src_part, monkeypatch
    ):
        for key in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
            monkeypatch.delenv(key, raising=False)
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        phys, src = resolve_physical_device(logical)
        assert phys == expected_phys
        assert expected_src_part in src

    @pytest.mark.parametrize(
        "value,expected",
        [
            (7, 7),
            (7.9, 7),
            ("N/A", 0),
            (None, 0),
            (True, 0),
        ],
    )
    def test_int_metric(self, value, expected):
        assert _int_metric(value, default=0) == expected

    def test_check_gpu_idle_when_no_tools(self):
        with patch(
            "TraceLens.PerfModel.benchmarking.microbench_utils.shutil.which",
            return_value=None,
        ), patch.dict(
            "sys.modules",
            {"amdsmi": None},
        ):
            idle, msg = check_gpu_idle(0)
        assert idle is True
        assert "skipping idle check" in msg

    def test_check_gpu_idle_via_nvidia_smi(self):
        smi_output = "2, 512, 8192\n"

        def fake_run(cmd, **kwargs):
            assert "nvidia-smi" in cmd[0]
            return type(
                "R",
                (),
                {"stdout": smi_output, "returncode": 0},
            )()

        with patch(
            "TraceLens.PerfModel.benchmarking.microbench_utils.shutil.which",
            return_value="/usr/bin/nvidia-smi",
        ), patch.dict("sys.modules", {"amdsmi": None}), patch(
            "TraceLens.PerfModel.benchmarking.microbench_utils.subprocess.run",
            side_effect=fake_run,
        ):
            idle, msg = check_gpu_idle(0)
        assert idle is True
        assert "idle" in msg.lower()


class TestMxPeakWithoutNativeSupport:
    MATRIX = {"matrix_fp16": 100.0, "matrix_bf16": 120.0, "matrix_fp4": 10.0}

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("gfx942:sramecc+:xnack-", "gfx942"),
            ("gfx1201", "gfx1201"),
            ("", ""),
            (None, ""),
        ],
    )
    def test_gfx_target(self, name, expected):
        assert gfx_target(name) == expected

    @pytest.mark.parametrize("gfx", ["gfx1201", "gfx1151", "gfx942"])
    def test_non_native_targets_use_16bit_peak(self, gfx):
        assert mx_peak_without_native_support(gfx, self.MATRIX) == 120.0

    @pytest.mark.parametrize("gfx", ["gfx950", ""])
    def test_native_or_non_amd_targets_keep_measurement(self, gfx):
        assert mx_peak_without_native_support(gfx, self.MATRIX) is None


class TestRunPythonProbe:
    def test_last_line_is_the_result(self):
        code = (
            "import json, sys; print('noise'); print(json.dumps({'a': sys.argv[1:]}))"
        )
        assert run_python_probe(code, 1, "x", timeout_s=60) == {"a": ["1", "x"]}

    @pytest.mark.parametrize(
        "code,expected",
        [
            ("raise SystemExit('boom')", "boom"),
            ("print('not json')", "unparseable probe output: not json"),
            ("pass", "probe exited with 0"),
        ],
    )
    def test_failure_is_an_error(self, code, expected):
        assert run_python_probe(code, timeout_s=60) == {"error": expected}

    def test_timeout_is_an_error(self):
        assert "timed out" in run_python_probe("while True: pass", timeout_s=1)["error"]


# An amdsmi module for the idle probe's fresh interpreter: one GPU at
# 0000:c3:00.0, idle, whose process list needs more rights than the caller has.
_FAKE_AMDSMI = """
class AmdSmiException(Exception):
    pass

def amdsmi_init():
    pass

def amdsmi_shut_down():
    pass

def amdsmi_get_processor_handle_from_bdf(bdf):
    if bdf != "0000:c3:00.0":
        raise AmdSmiException(f"no GPU at {bdf}")
    return bdf

def amdsmi_get_gpu_activity(handle):
    return {"gfx_activity": 0}

def amdsmi_get_gpu_vram_usage(handle):
    return {"vram_used": 26, "vram_total": 32624}

def amdsmi_get_gpu_process_list(handle):
    raise AmdSmiException("process list needs root")
"""


class TestRdnaIdleCheck:
    # A 25% duty load: idle one reading in four.
    BUSY_25PCT = [0, 60, 60, 60] * 5

    @staticmethod
    def _probe(readings, *, used=26, total=32624, processes=1):
        return {
            "gfx_activity": readings,
            "vram": {"vram_used": used, "vram_total": total},
            "processes": processes,
        }

    def _check(self, probe, **kwargs):
        with patch(
            "TraceLens.PerfModel.benchmarking.microbench_utils.run_python_probe",
            return_value=probe,
        ):
            return check_rdna_gpu_idle("0000:c3:00.0", **kwargs)

    def test_median_of_readings_sees_a_duty_cycle_load(self):
        idle, msg = self._check(self._probe(self.BUSY_25PCT))
        assert idle is False
        assert "median of 20 readings: busy" in msg

    def test_median_ignores_a_single_spike(self):
        idle, msg = self._check(self._probe([0] * 19 + [100]))
        assert idle is True
        assert "max 100%" in msg

    def test_vram_counts_on_a_discrete_board_only(self):
        probe = self._probe([0] * 20, used=20000)
        assert self._check(probe)[0] is False
        assert self._check(probe, mem_threshold_pct=None)[0] is True

    def test_processes_are_reported_not_counted(self):
        idle, msg = self._check(self._probe([0] * 20, processes=3))
        assert idle is True
        assert "3 process(es) on the GPU" in msg

    @pytest.mark.parametrize(
        "probe,expected",
        [
            ({"error": "No module named 'amdsmi'"}, "No module named 'amdsmi'"),
            ({"gfx_activity": ["N/A"] * 20}, "gfx_activity reads"),
            ({"gfx_activity": []}, "gfx_activity reads"),
        ],
    )
    def test_a_check_that_cannot_decide_is_not_idle(self, probe, expected):
        idle, msg = self._check(probe)
        assert idle is False
        assert expected in msg
        assert "idle state unknown" in msg

    def test_probe_runs_amdsmi_in_a_fresh_interpreter(self, tmp_path, monkeypatch):
        (tmp_path / "amdsmi.py").write_text(_FAKE_AMDSMI)
        monkeypatch.setenv("PYTHONPATH", str(tmp_path))
        idle, msg = check_rdna_gpu_idle("0000:c3:00.0", samples=3, interval_s=0)
        assert idle is True
        assert "process list unavailable (process list needs root)" in msg
        idle, msg = check_rdna_gpu_idle("0000:43:00.0", samples=3, interval_s=0)
        assert idle is False
        assert "no GPU at 0000:43:00.0" in msg


class TestCuTiledSweepShapes:
    def test_304_cus_give_the_tile304_shapes(self):
        assert cu_tiled_sweep_shapes(304) == [
            (4864, 4096, 8256),
            (4864, 8192, 4160),
            (8192, 4864, 6878),
        ]

    # Strix Halo (gfx1151), R9700 (gfx1201), W7900 (gfx1100).
    @pytest.mark.parametrize("n_cu,dim", [(40, 5120), (64, 5120), (96, 4608)])
    def test_output_tiles_split_evenly_over_the_cus(self, n_cu, dim):
        shapes = cu_tiled_sweep_shapes(n_cu)
        assert shapes[0][0] == dim
        for M, N, _ in shapes:
            assert (M // 256) * (N // 256) % n_cu == 0


class TestRdnaPublishedKeys:
    RESULTS = {
        "matrix_fp16": 81.7,
        "matrix_bf16": 84.5,
        "matrix_fp32": 2.6,
        "matrix_fp64": 1.0,
        "matrix_int8": 74.5,
        "vector_fp16": 48.6,
        "vector_fp32": 34.8,
        "vector_fp64": 1.1,
    }

    @pytest.mark.parametrize(
        "gfx,expected",
        [
            ("gfx1100", "gfx11"),
            ("gfx1151", "gfx11"),
            ("gfx1201", "gfx12"),
            ("gfx1030", None),
            ("gfx942", None),
            ("gfx950", None),
            ("", None),
        ],
    )
    def test_rdna_family(self, gfx, expected):
        assert rdna_family(gfx) == expected

    @pytest.mark.parametrize("gfx", ["gfx942", "gfx950", "gfx1030", ""])
    def test_non_rdna_targets_unchanged(self, gfx):
        assert rdna_published_keys(gfx, self.RESULTS) is None

    def test_rdna3_derives_only_the_formats_without_a_path(self):
        values, sources = rdna_published_keys("gfx1100", self.RESULTS)
        derived = {k for k, s in sources.items() if s["source"] == "derived"}
        assert derived == {"matrix_fp8", "matrix_fp4", "matrix_fp6", "vector_bf16"}
        assert (
            values["matrix_fp8"] == values["matrix_fp4"] == values["matrix_fp6"] == 81.7
        )
        assert sources["matrix_fp8"]["from"] == "matrix_fp16"
        assert "no FP8 path" in sources["matrix_fp8"]["reason"]
        assert values["vector_bf16"] == 34.8
        assert sources["vector_bf16"]["from"] == "vector_fp32"

    def test_fp32_gemm_keeps_the_gemm_rate(self):
        # Strix Halo: FP32 GEMMs run at 3.1 while the vector FMA rate is 20.8.
        results = {**self.RESULTS, "matrix_fp32": 3.1, "vector_fp32": 20.8}
        values, sources = rdna_published_keys("gfx1151", results)
        assert values["matrix_fp32"] == 3.1
        assert sources["matrix_fp32"]["source"] == "measured"

    def test_rdna4_measures_fp8(self):
        values, sources = rdna_published_keys(
            "gfx1201", {**self.RESULTS, "matrix_fp8": 150.0}
        )
        assert values["matrix_fp8"] == 150.0
        assert sources["matrix_fp8"]["source"] == "measured"
        assert sources["matrix_fp4"]["source"] == "derived"

    @pytest.mark.parametrize(
        "gfx,override,missing",
        [
            ("gfx1201", {"matrix_fp8": 0.0}, "matrix_fp8"),
            ("gfx1100", {"vector_fp64": 0.0}, "vector_fp64"),
            ("gfx1151", {"matrix_bf16": None}, "matrix_bf16"),
        ],
    )
    def test_zero_or_failed_native_key_raises(self, gfx, override, missing):
        with pytest.raises(MeasurementError, match=missing):
            rdna_published_keys(gfx, {**self.RESULTS, **override})

    def test_skipped_section_is_marked(self):
        results = {k: v for k, v in self.RESULTS.items() if not k.startswith("vector_")}
        skipped = {"vector_fp16", "vector_fp32", "vector_fp64"}
        values, sources = rdna_published_keys("gfx1100", results, skipped=skipped)
        assert values["vector_fp32"] == values["vector_bf16"] == 0.0
        assert sources["vector_fp32"] == sources["vector_bf16"] == {"source": "skipped"}

    def test_configs_go_into_the_provenance(self):
        config = {"config": {"library": "hipblaslt", "layout": "nt"}}
        _, sources = rdna_published_keys(
            "gfx1100", self.RESULTS, {"matrix_bf16": config}
        )
        assert sources["matrix_bf16"] == {"source": "measured", **config}


class TestBandwidthPlateau:
    SIZES = [64, 128, 256, 512, 1024, 2048, 4096, 8192]

    def test_plateau_starts_where_every_later_step_is_under_tol(self):
        gbps = [500.0, 650.0, 700.0, 720.0, 726.0, 728.0, 728.5, 728.2]
        start, value = bandwidth_plateau(self.SIZES, gbps)
        assert start == 512
        assert value == statistics.median(gbps[3:])

    def test_no_plateau_while_the_curve_moves(self):
        assert (
            bandwidth_plateau(self.SIZES, [100.0 * (i + 1) for i in range(8)]) is None
        )

    def test_a_dip_restarts_the_plateau(self):
        gbps = [700.0, 702.0, 650.0, 700.0, 701.0, 701.5, 701.0, 700.8]
        assert bandwidth_plateau(self.SIZES, gbps)[0] == 512


class TestVopdShare:
    def test_counts_both_halves_of_a_pair(self):
        asm = "\n".join(
            [
                "\tv_dual_fmac_f32 v5, v1, v2 :: v_dual_fmac_f32 v6, v1, v2",
                "\tv_fmac_f32_e32 v7, v1, v2",
                "\tv_fma_f32 v8, v8, v1, v2 ; comment v_dual_fmac_f32",
                "\tv_dual_mul_f32 v9, v1, v2 :: v_dual_fmac_f32 v10, v3, v4",
                "\ts_delay_alu instid0(VALU_DEP_1)",
            ]
        )
        assert vopd_share(asm) == pytest.approx(3 / 5)

    def test_no_fma(self):
        assert vopd_share("\ts_endpgm\n") is None


def _fake_card(root: Path, bdf: str = "0000:43:00.0", mclk: bool = True) -> Path:
    dev = root / "bus" / "pci" / "devices" / bdf
    hw = dev / "hwmon" / "hwmon3"
    hw.mkdir(parents=True)
    hwmon = {
        "freq1_input": "2100000000",
        "freq1_label": "sclk",
        "power1_average": "241000000",
        "power1_cap": "241000000",
        "power1_cap_default": "241000000",
    }
    if mclk:
        hwmon.update({"freq2_input": "1249000000", "freq2_label": "mclk"})
    for name, value in hwmon.items():
        (hw / name).write_text(value + "\n")
    device = {
        "vendor": "0x1002",
        "device": "0x7448",
        "revision": "0x00",
        "subsystem_vendor": "0x1002",
        "subsystem_device": "0x0e0d",
        "vbios_version": "113-D7070100-100",
        "power_dpm_force_performance_level": "auto",
        "pp_dpm_sclk": "S: 19Mhz\n0: 500Mhz\n1: 1000Mhz *\n2: 2495Mhz\n",
        "pp_dpm_mclk": "0: 96Mhz\n1: 456Mhz\n2: 1249Mhz *\n",
        "pp_dpm_fclk": "0: 800Mhz\n1: 1400Mhz *\n",
    }
    for name, value in device.items():
        (dev / name).write_text(value + "\n")
    module = root / "module" / "amdgpu"
    module.mkdir(parents=True)
    (module / "version").write_text("6.19.0\n")
    return dev


class TestSysfsTelemetry:
    def test_read_scales_to_mhz_and_watts(self, tmp_path):
        _fake_card(tmp_path)
        sample = SysfsTelemetry("0000:43:00.0", root=str(tmp_path)).read()
        assert sample["sclk_mhz"] == 2100.0
        assert sample["mclk_mhz"] == 1249.0
        assert sample["fclk_mhz"] == 1400.0
        assert sample["power_w"] == sample["power_cap_w"] == 241.0

    def test_apu_mclk_is_the_current_dpm_level(self, tmp_path):
        dev = _fake_card(tmp_path, mclk=False)
        (dev / "pp_dpm_mclk").write_text("0: 400Mhz\n1: 800Mhz *\n2: 1000Mhz\n")
        telemetry = SysfsTelemetry("0000:43:00.0", root=str(tmp_path))
        assert telemetry.read()["mclk_mhz"] == 800.0
        (dev / "pp_dpm_mclk").unlink()
        assert telemetry.read()["mclk_mhz"] is None

    def test_missing_card(self, tmp_path):
        telemetry = SysfsTelemetry("0000:99:00.0", root=str(tmp_path))
        assert telemetry.available is False
        assert telemetry.start().summary(0, time.time() + 1) == {"samples": 0}

    def test_short_window_records_no_clock(self, tmp_path):
        _fake_card(tmp_path)
        telemetry = SysfsTelemetry("0000:43:00.0", root=str(tmp_path))
        telemetry._samples = [telemetry.read(), telemetry.read()]
        assert telemetry.summary(0, time.time() + 1) == {"samples": 2}

    def test_summary_is_the_median_of_the_window(self, tmp_path):
        dev = _fake_card(tmp_path)
        telemetry = SysfsTelemetry("0000:43:00.0", period_s=0.01, root=str(tmp_path))
        t0 = time.time()
        telemetry.start()
        time.sleep(0.1)
        (dev / "hwmon" / "hwmon3" / "freq1_input").write_text("1900000000\n")
        time.sleep(0.3)
        telemetry.stop()
        summary = telemetry.summary(t0, time.time())
        assert summary["samples"] > 10
        assert summary["sclk_mhz"] == 1900.0
        assert summary["mclk_mhz"] == 1249.0


class TestPlatformRecord:
    def test_reads_board_identity_and_configuration(self, tmp_path):
        _fake_card(tmp_path)
        with patch(
            "TraceLens.PerfModel.benchmarking.microbench_utils.rocm_version",
            return_value="10.0.0",
        ):
            record = platform_record("0000:43:00.0", root=str(tmp_path))
        assert record["pci_device_id"] == "0x7448"
        assert record["pci_subsystem_device_id"] == "0x0e0d"
        assert record["vbios_version"] == "113-D7070100-100"
        assert record["power_cap_w"] == record["power_cap_default_w"] == 241.0
        assert record["power_cap_source"] == "hwmon power1_cap"
        assert record["perf_level"] == "auto"
        assert (record["sclk_max_mhz"], record["mclk_max_mhz"]) == (2495, 1249)
        assert record["dpm_levels_mhz"] == {
            "sclk": [500, 1000, 2495],
            "mclk": [96, 456, 1249],
            "fclk": [800, 1400],
        }
        assert record["amdgpu_version"] == "6.19.0"
        assert record["rocm_version"] == "10.0.0"
        assert record["kernel"]

    def test_missing_fields_are_none(self, tmp_path):
        record = platform_record("0000:99:00.0", root=str(tmp_path))
        assert record["vbios_version"] is None
        assert record["power_cap_w"] is None
        assert record["power_cap_source"] == "not exposed: no hwmon power1_cap"
        assert record["sclk_max_mhz"] is None
        assert record["dpm_levels_mhz"] == {"sclk": None, "mclk": None, "fclk": None}


class TestVectorFmaBlock:
    @pytest.mark.parametrize("gfx", sorted(VECTOR_FMA_BLOCK))
    def test_vector_fma_block_listed_target(self, gfx):
        assert vector_fma_block(gfx) == VECTOR_FMA_BLOCK[gfx]

    @pytest.mark.parametrize("gfx", ["gfx9999", ""])
    def test_vector_fma_block_falls_back_to_default(self, gfx):
        assert vector_fma_block(gfx) == VECTOR_FMA_BLOCK_DEFAULT


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestMicrobenchHelpers:
    def test_gemm_flops(self):
        mb = _import_microbench()
        assert mb._gemm_flops(4, 8, 16) == 2 * 4 * 8 * 16

    @pytest.mark.parametrize(
        "gpu_name,mem_gb,expected",
        [
            ("AMD Instinct MI300X", 192.0, "MI300X"),
            ("AMD Instinct MI210", 64.0, "MI210"),
            # Trailing "Graphics" is a marketing suffix, not the model.
            ("AMD Radeon 8060S Graphics", 64.0, "Radeon_8060S"),
            ("AMD Radeon 890M Graphics", 32.0, "Radeon_890M"),
            ("AMD Radeon PRO W7900 Dual Slot", 45.0, "W7900"),
            ("AMD Radeon PRO W7900", 45.0, "W7900"),
            ("AMD Radeon AI PRO R9700", 31.9, "R9700"),
            ("Generic GPU", 280.0, "MI355X"),
            ("Some Card", 64.0, "Card"),
            ("", 0.0, "GPU"),
            # Workstation boards are named after the model that follows "PRO".
            # Both W7900 boards are one platform; only the cooler differs.
            ("AMD Radeon Pro W7900", 45.0, "W7900"),
            ("AMD Radeon PRO W7900 Dual Slot", 45.0, "W7900"),
            ("AMD Radeon PRO  W7900  Dual Slot ", 45.0, "W7900"),
            ("AMD Radeon Pro W7800", 32.0, "W7800"),
            ("AMD Radeon Pro W7800 48GB", 48.0, "W7800"),
            ("AMD Radeon AI PRO R9700", 32.0, "R9700"),
            # RX boards and a bare gfx target keep the last-word fallback.
            ("AMD Radeon RX 7900 XTX", 24.0, "XTX"),
            ("AMD Radeon RX 7900 XT", 20.0, "XT"),
            ("AMD Radeon RX 7900 GRE", 16.0, "GRE"),
            ("gfx1100", 45.0, "gfx1100"),
        ],
    )
    def test_arch_product_name(self, gpu_name, mem_gb, expected):
        mb = _import_microbench()
        assert mb._arch_product_name(gpu_name, mem_gb) == expected

    def test_build_measured_arch_json_shape(self):
        mb = _import_microbench()
        payload = mb._build_measured_arch_json(
            gpu_name="MI300X",
            mem_gb=192.0,
            read_bw_gbps=5200.0,
            matrix_results={"matrix_fp16": 123.4},
            vector_results={"vector_fp16": 45.6},
        )
        assert payload["name"] == "MI300X"
        assert payload["max_achievable_tflops"]["matrix_fp16"] == 123.4
        assert payload["max_achievable_tflops"]["vector_fp16"] == 45.6
        assert set(mb.ARCH_TLOPS_KEYS) <= set(payload["max_achievable_tflops"])

    def test_build_measured_arch_json_keeps_sub_one_tflops(self):
        mb = _import_microbench()
        payload = mb._build_measured_arch_json(
            gpu_name="R9600D",
            mem_gb=47.6,
            read_bw_gbps=640.4,
            matrix_results={"matrix_fp64": 0.4, "matrix_bf16": 69.1},
            vector_results={"vector_fp64": 0.4},
        )
        maf = payload["max_achievable_tflops"]
        assert maf["matrix_fp64"] == 0.4
        assert maf["vector_fp64"] == 0.4
        assert maf["matrix_bf16"] == 69.1
        assert payload["mem_bw_gbps"] == 640.4
        assert payload["memory_gb"] == 47.6
        assert maf["vector_fp16"] == 0.0

    @pytest.mark.parametrize(
        "gfx,recorded", [("gfx1100", True), ("gfx942", False), (None, False)]
    )
    def test_build_measured_arch_json_records_rdna_gfx_arch(self, gfx, recorded):
        mb = _import_microbench()
        payload = mb._build_measured_arch_json(
            gpu_name="AMD Radeon PRO W7900 Dual Slot",
            mem_gb=45.0,
            read_bw_gbps=717.8,
            matrix_results={"matrix_bf16": 66.0},
            vector_results={},
            gfx_arch=gfx,
        )
        assert ("gfx_arch" in payload) == recorded
        assert list(payload)[0] == "name"

    @pytest.mark.parametrize("gfx", [None, "gfx942", "gfx950"])
    def test_non_rdna_arch_json_has_only_the_template_keys(self, gfx):
        mb = _import_microbench()
        payload = mb._build_measured_arch_json(
            "AMD Instinct MI300X", 192.0, 5200.0, {}, {}, gfx_arch=gfx
        )
        assert list(payload) == [
            "name",
            "mem_bw_gbps",
            "memory_gb",
            "max_achievable_tflops",
            "source",
        ]

    def test_rdna_extras_follow_the_template_keys(self):
        mb = _import_microbench()
        extras = {"max_achievable_tflops_source": {}, "measurement": {}, "platform": {}}
        payload = mb._build_measured_arch_json(
            "AMD Radeon PRO W7900",
            45.0,
            728.0,
            {},
            {},
            gfx_arch="gfx1100",
            extras=extras,
        )
        assert list(payload)[-4:] == ["source", *extras]

    def test_gemm_shapes_non_empty(self):
        mb = _import_microbench()
        assert len(mb.GEMM_SHAPES) >= 1


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestRdnaProtocolHelpers:
    @pytest.mark.parametrize("gfx", ["gfx1100", "gfx1151", "gfx1201"])
    def test_vector_configs_include_the_default_launch(self, gfx):
        rd = _import_microbench_rdna()
        assert (vector_fma_block(gfx), 8) in rd.VECTOR_CONFIGS_RDNA
        assert all(block % (32 * warps) == 0 for block, warps in rd.VECTOR_CONFIGS_RDNA)

    def test_chain_launch_sets_every_constexpr_of_the_suite_kernel(self, monkeypatch):
        mb = _import_microbench()
        rd = _import_microbench_rdna()
        if mb.triton is None:
            pytest.skip("triton not installed")
        suite = mb._rdna_suite_parts()
        launched = {}

        class _Recorder:
            def __getitem__(self, grid):
                return lambda *args, **kwargs: launched.update(kwargs)

        monkeypatch.setattr(rd, "VECTOR_N_ELEMENTS", 1024)
        rd._vector_call(
            "chain", 256, 4, mb.torch.float32, "cpu", 64, _Recorder(), suite.chain_args
        )()
        constexprs = {p.name for p in suite.chain_kernel.params if p.is_constexpr}
        assert set(launched) - {"num_warps"} == constexprs

    def test_tile_configs_split_the_chain_evenly(self):
        rd = _import_microbench_rdna()
        assert rd.VECTOR_N_ELEMENTS % rd.VECTOR_TILE_ILP == 0
        assert 1024 % rd.VECTOR_TILE_UNROLL == 0
        assert all(
            block % (32 * warps) == 0 for block, warps in rd.VECTOR_TILE_CONFIGS_RDNA
        )

    def test_bandwidth_sizes_reach_past_every_cache(self):
        rd = _import_microbench_rdna()
        # 4x the largest RDNA last-level cache (96 MB) at the smallest size.
        assert rd.BW_SIZES_RDNA == sorted(rd.BW_SIZES_RDNA)
        assert rd.BW_SIZES_RDNA[0] >= 4 * 96 * 1024**2
        assert rd.BW_SIZES_RDNA[-1] == 8 * 1024**3
        assert rd.BW_FLUSH_BYTES >= 4 * 96 * 1024**2
        assert [kernel for kernel, _ in rd.READ_KERNELS] == ["triton", "torch_sum"]

    def test_deep_k_shapes_are_wmma_only(self):
        rd = _import_microbench_rdna()
        mb = _import_microbench()
        suite = rd.SuiteParts(mb.GEMM_SHAPES, None, lambda *a: None)
        plans = rd._matrix_plans(0, "gfx1100", suite)
        deep = [list(s) for s in rd.RDNA_DEEP_K_SHAPES]
        for key in ("matrix_fp16", "matrix_bf16", "matrix_int8"):
            assert all(s in [c.config["shape"] for c in plans[key]] for s in deep)
        assert not any(c.config["shape"] in deep for c in plans["matrix_fp32"])
        assert [c.config["shape"] for c in plans["matrix_fp64"]] == [
            list(s) for s in rd.RDNA_FP64_SHAPES
        ] * len(rd.RDNA_BLAS_LIBRARIES) * len(rd.RDNA_GEMM_LAYOUTS)
        assert "matrix_fp8" not in plans
        assert {c.config["layout"] for c in plans["matrix_int8"]} == {"nn", "nt"}

    def test_fp8_is_planned_on_gfx12_only(self):
        rd = _import_microbench_rdna()
        mb = _import_microbench()
        fp8 = mb.torch.float8_e4m3fn
        suite = rd.SuiteParts(mb.GEMM_SHAPES, None, lambda *a: fp8)
        assert len(rd._matrix_plans(0, "gfx1201", suite)["matrix_fp8"]) == len(
            mb.GEMM_SHAPES
        ) + len(rd.RDNA_DEEP_K_SHAPES)
        no_fp8 = rd.SuiteParts(mb.GEMM_SHAPES, None, lambda *a: None)
        assert rd._matrix_plans(0, "gfx1201", no_fp8)["matrix_fp8"] == []

    @pytest.mark.parametrize("gfx", ["gfx1100", "gfx1151", "gfx1201"])
    def test_gfx1151_runs_fp32_and_fp64_through_rocblas(self, gfx):
        rd = _import_microbench_rdna()
        mb = _import_microbench()
        suite = rd.SuiteParts(mb.GEMM_SHAPES, None, lambda *a: None)
        plans = rd._matrix_plans(0, gfx, suite)
        both = {lib for lib, _ in rd.RDNA_BLAS_LIBRARIES}
        wide = {"rocblas"} if gfx == "gfx1151" else both
        for key, libs in (
            ("matrix_fp16", both),
            ("matrix_bf16", both),
            ("matrix_fp32", wide),
            ("matrix_fp64", wide),
        ):
            assert {c.config["library"] for c in plans[key]} == libs
        if gfx == "gfx1151":
            assert {c.blas for c in plans["matrix_fp32"]} == {"cublas"}
            assert len(plans["matrix_fp32"]) == len(mb.GEMM_SHAPES) * len(
                rd.RDNA_GEMM_LAYOUTS
            )

    @pytest.mark.parametrize("call_s", [0.8, 0.01])
    def test_slow_calls_are_timed_one_at_a_time(self, call_s):
        # Past RDNA_SERIAL_CALL_S no call is queued behind another; below it,
        # do_bench as before.
        rd = _import_microbench_rdna()
        state = types.SimpleNamespace(wall=0.0, gpu=0.0, queued=0, most_queued=0)

        def call():
            state.wall += call_s
            state.gpu += call_s
            state.queued += 1
            state.most_queued = max(state.most_queued, state.queued)

        def sync():
            state.queued = 0

        class FakeEvent:
            def __init__(self, enable_timing=False):
                self.at = 0.0

            def record(self):
                self.at = state.gpu

            def synchronize(self):
                sync()

            def elapsed_time(self, other):
                return (other.at - self.at) * 1e3

        fake_time = types.SimpleNamespace(
            perf_counter=lambda: state.wall, time=lambda: state.wall
        )
        proto = rd._RdnaProtocol(
            SysfsTelemetry("0000:99:00.0", root="/nonexistent"),
            warm_s=3.0,
            window_ms=1000,
        )
        bench = MagicMock(return_value=call_s * 1e3)
        with patch.object(rd, "time", fake_time), patch.object(
            rd.torch.cuda, "Event", FakeEvent
        ), patch.object(rd.torch.cuda, "synchronize", sync), patch.object(
            rd, "do_bench", bench
        ):
            ms, clock = proto.time(call)
        assert ms == pytest.approx(call_s * 1e3)
        if call_s > rd.RDNA_SERIAL_CALL_S:
            bench.assert_not_called()
            assert clock == {"synced_calls": rd.RDNA_SERIAL_MIN_CALLS}
            assert state.most_queued == 1
        else:
            bench.assert_called_once()
            assert clock == {}

    def test_config_label(self):
        rd = _import_microbench_rdna()
        label = rd._config_label(
            {"library": "hipblaslt", "layout": "nt", "shape": [8192, 8192, 8192]}
        )
        assert label == "hipblaslt nt 8192x8192x8192"
        assert rd._config_label({"kernel": "tile", "block": 256}) == "tile block=256"

    @pytest.mark.parametrize("integrated,threshold", [(0, 10.0), (1, None)])
    def test_idle_check_skips_vram_on_an_apu(self, integrated, threshold):
        rd = _import_microbench_rdna()
        props = types.SimpleNamespace(
            pci_domain_id=0, pci_bus_id=0xC3, pci_device_id=0, is_integrated=integrated
        )
        with patch.object(
            rd, "check_rdna_gpu_idle", return_value=(True, "idle")
        ) as check:
            assert rd.check_idle(props, 5) == (True, "idle")
        check.assert_called_once_with(
            "0000:c3:00.0", util_threshold=5, mem_threshold_pct=threshold
        )

    @pytest.mark.parametrize("skip", [False, True])
    def test_protocol_record(self, skip):
        rd = _import_microbench_rdna()
        proto = rd._RdnaProtocol(
            SysfsTelemetry("0000:99:00.0", root="/nonexistent"),
            warm_s=3.0,
            window_ms=1000,
        )
        record = rd._protocol_record(
            proto, skip_burst=skip, skip_back_to_back=skip, cli_overrides=["--rep 50"]
        )
        assert (record["warm_s"], record["window_ms"]) == (3.0, 1000)
        assert record["statistic"] == "median"
        assert record["cli_overrides"] == ["--rep 50"]
        assert (record["burst"] is None) is skip
        assert (record["back_to_back"] is None) is skip
        assert record["telemetry"]["source"] is None

    def test_back_to_back_chunks_follow_wall_time(self):
        # A launch-bound GEMM: 1 ms of wall time per call, 0.25 ms of it on the GPU.
        rd = _import_microbench_rdna()
        clock = types.SimpleNamespace(wall=0.0, gpu=0.0)

        def call():
            clock.wall += 1e-3
            clock.gpu += 0.25e-3

        class FakeEvent:
            def __init__(self, enable_timing=False):
                self.at = 0.0

            def record(self):
                self.at = clock.gpu

            def synchronize(self):
                pass

            def elapsed_time(self, other):
                return (other.at - self.at) * 1e3

        fake_time = types.SimpleNamespace(
            perf_counter=lambda: clock.wall, time=lambda: clock.wall
        )
        work = 2 * 512**3
        cand = rd._Candidate({"layout": "nt"}, lambda: call, work)
        winner = rd._Winner(
            "matrix", "matrix_fp16", cand, "tflops", work / 0.25e-3 / 1e12
        )
        proto = rd._RdnaProtocol(
            SysfsTelemetry("0000:99:00.0", root="/nonexistent"),
            warm_s=3.0,
            window_ms=1000,
        )
        with patch.object(rd, "time", fake_time), patch.object(
            rd.torch.cuda, "Event", FakeEvent
        ):
            held = rd._back_to_back(winner, proto, seconds=2.0, settle_s=0.75)
        # Sized from the rate, a 0.5 s chunk would take 2 s and leave none before 0.75 s.
        assert held["calls_per_chunk"] == 500
        assert held["chunks"] == 5
        assert held["first_0.75_s_tflops"] == held["tflops"] == round(winner.rate, 1)


class TestRdnaWindowAndRecord:
    def test_hyperloom_settings_are_floored_and_logged(self, capsys):
        warm_s, window_ms, notes = resolve_rdna_window(3.0, 1000, 20, 50)
        assert (warm_s, window_ms) == (3.0, 1000)
        assert notes == [
            "--warmup 20 ms: 3 s warm-up per section",
            "--rep 50 ms: 1000 ms window per config",
        ]
        assert capsys.readouterr().out.count("RDNA protocol floor:") == 2

    def test_longer_settings_lengthen_the_window(self, capsys):
        assert resolve_rdna_window(3.0, 1000, 5000, 2000) == (5.0, 2000, [])
        assert resolve_rdna_window(3.0, 1000, None, None) == (3.0, 1000, [])
        assert capsys.readouterr().out == ""

    def test_a_short_window_is_flagged(self, capsys):
        assert resolve_rdna_window(3.0, 200, None, None) == (3.0, 200, [])
        assert "not comparable across boards" in capsys.readouterr().out

    def test_pci_bdf(self):
        props = types.SimpleNamespace(pci_domain_id=0, pci_bus_id=0xC3, pci_device_id=0)
        assert pci_bdf(props) == "0000:c3:00.0"

    def test_power_held_is_the_highest_window_median(self):
        record = {
            "matrix": {"matrix_fp16": {"power_w": 290.0, "rows": [{"power_w": 299.5}]}},
            "bandwidth": {"read": {"triton": {"clock": {"power_w": 180.0}}}},
            "protocol": {"window_ms": 1000},
        }
        assert power_held_w(record) == 299.5
        assert power_held_w({"protocol": {}}) is None


def _fake_protocol_result():
    sources = {
        "matrix_fp16": {"source": "measured"},
        "matrix_fp8": {
            "source": "derived",
            "from": "matrix_fp16",
            "reason": "gfx1100 WMMA has no FP8 path",
        },
    }
    extras = {
        "max_achievable_tflops_source": sources,
        "measurement": {},
        "platform": {},
    }
    return (
        {"matrix_fp16": 120.0, "matrix_fp8": 120.0},
        {"vector_fp32": 44.0},
        564.0,
        extras,
    )


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestMainDispatch:
    """Every target but gfx11 / gfx12 runs main's suite exactly as on main
    (a2a789a4): same constants, idle check, sections and publish rule. RDNA
    targets run :func:`microbench_rdna.run_rdna_protocol`."""

    MI = ["gfx942:sramecc+:xnack-", "gfx950", "gfx90a"]
    RDNA = ["gfx1100", "gfx1151", "gfx1201"]
    MAIN_GEMM_SHAPES = [
        (8192, 8192, 8192),
        (16384, 8192, 1280),
        (16384, 1024, 8192),
        (16384, 8192, 7168),
        (16384, 3584, 8192),
    ]
    MAIN_BW_SIZES = [
        64 * 1024**2,
        128 * 1024**2,
        256 * 1024**2,
        512 * 1024**2,
        1024**3,
        2 * 1024**3,
    ]
    MAIN_TILE304 = [(4864, 4096, 8256), (4864, 8192, 4160), (8192, 4864, 6878)]

    def _main(self, monkeypatch, tmp_path, gcn, *argv, idle=(True, "idle"), **rdna):
        mb = _import_microbench()
        monkeypatch.setattr(mb, "WARMUP", mb.WARMUP)
        monkeypatch.setattr(mb, "REP", mb.REP)
        props = types.SimpleNamespace(
            gcnArchName=gcn,
            total_memory=32 * 1024**3,
            multi_processor_count=64,
            is_integrated=0,
            pci_domain_id=0,
            pci_bus_id=0xC3,
            pci_device_id=0,
        )
        cuda = mb.torch.cuda
        monkeypatch.setattr(cuda, "set_device", lambda _d: None)
        monkeypatch.setattr(cuda, "get_device_name", lambda _d: "AMD Test GPU")
        monkeypatch.setattr(cuda, "get_device_properties", lambda _d: props)
        fakes = {
            "check_gpu_idle": MagicMock(return_value=idle),
            "bench_matrix_tflops": MagicMock(return_value={"matrix_fp16": 100.0}),
            "bench_vector_tflops": MagicMock(return_value={"vector_fp32": 10.0}),
            "bench_mem_bandwidth": MagicMock(
                return_value={"read_bw_gbps": 500.0, "write_bw_gbps": 400.0}
            ),
            "run_shape_sweep": MagicMock(),
        }
        for name, fake in fakes.items():
            monkeypatch.setattr(mb, name, fake)
        rd = mb.microbench_rdna
        fakes["check_idle"] = MagicMock(return_value=idle)
        fakes["run_rdna_protocol"] = MagicMock(
            return_value=_fake_protocol_result(), **rdna
        )
        monkeypatch.setattr(rd, "check_idle", fakes["check_idle"])
        monkeypatch.setattr(rd, "run_rdna_protocol", fakes["run_rdna_protocol"])
        out = tmp_path / "arch.json"
        monkeypatch.setattr("sys.argv", ["microbench", "--output", str(out), *argv])
        mb.main()
        return mb, fakes, out

    def test_constants_match_main(self):
        mb = _import_microbench()
        assert mb.GEMM_SHAPES == self.MAIN_GEMM_SHAPES
        assert mb.BW_SIZES == self.MAIN_BW_SIZES
        assert mb.GEMM_SHAPES_SWEEP_TILE304 == self.MAIN_TILE304
        assert (mb.WARMUP, mb.REP) == (30, 200)

    @pytest.mark.parametrize("gcn", MI)
    def test_mi_runs_main_suite(self, gcn, monkeypatch, tmp_path):
        mb, fakes, out = self._main(
            monkeypatch, tmp_path, gcn, "--warmup", "20", "--rep", "50"
        )
        fakes["check_gpu_idle"].assert_called_once_with(0, util_threshold=5)
        fakes["bench_matrix_tflops"].assert_called_once_with(0)
        fakes["bench_vector_tflops"].assert_called_once_with(
            0, repeat=1024, mem_every=0
        )
        fakes["bench_mem_bandwidth"].assert_called_once_with(0)
        assert not fakes["check_idle"].called
        assert not fakes["run_rdna_protocol"].called
        assert (mb.WARMUP, mb.REP) == (20, 50)
        payload = json.loads(out.read_text())
        assert list(payload) == [
            "name",
            "mem_bw_gbps",
            "memory_gb",
            "max_achievable_tflops",
            "source",
        ]
        assert payload["mem_bw_gbps"] == 500.0

    @pytest.mark.parametrize("gcn", MI + RDNA)
    def test_busy_gpu_exits_unless_allowed(self, gcn, monkeypatch, tmp_path):
        busy = (False, "busy (util=90%)")
        with pytest.raises(SystemExit) as exit_info:
            self._main(monkeypatch, tmp_path, gcn, idle=busy)
        assert exit_info.value.code == 2
        _, _, out = self._main(monkeypatch, tmp_path, gcn, "--allow-busy", idle=busy)
        assert out.exists()

    @pytest.mark.parametrize("gcn", RDNA)
    def test_rdna_runs_the_protocol(self, gcn, monkeypatch, tmp_path, capsys):
        _, fakes, out = self._main(
            monkeypatch, tmp_path, gcn, "--warmup", "20", "--rep", "50"
        )
        assert not fakes["check_gpu_idle"].called
        assert not fakes["bench_matrix_tflops"].called
        fakes["check_idle"].assert_called_once()
        kwargs = fakes["run_rdna_protocol"].call_args.kwargs
        assert (kwargs["warm_s"], kwargs["window_ms"]) == (3.0, 1000)
        assert len(kwargs["cli_overrides"]) == 2
        assert not (kwargs["skip_burst"] or kwargs["skip_back_to_back"])
        payload = json.loads(out.read_text())
        assert payload["gfx_arch"] == gcn
        assert list(payload)[-4:] == [
            "source",
            "max_achievable_tflops_source",
            "measurement",
            "platform",
        ]
        assert payload["mem_bw_gbps"] == 564.0
        assert payload["max_achievable_tflops"]["matrix_fp8"] == 120.0
        assert "[matrix_fp8] = matrix_fp16" in capsys.readouterr().out

    def test_rdna_skip_flags_reach_the_protocol(self, monkeypatch, tmp_path):
        _, fakes, _ = self._main(
            monkeypatch, tmp_path, "gfx1201", "--skip-burst", "--skip-back-to-back"
        )
        kwargs = fakes["run_rdna_protocol"].call_args.kwargs
        assert kwargs["skip_burst"] and kwargs["skip_back_to_back"]

    def test_failed_native_key_exits_3_without_json(self, monkeypatch, tmp_path):
        error = MeasurementError("gfx1201: matrix_fp8 measured zero or failed")
        with pytest.raises(SystemExit) as exit_info:
            self._main(monkeypatch, tmp_path, "gfx1201", side_effect=error)
        assert exit_info.value.code == 3
        assert not (tmp_path / "arch.json").exists()

    def test_shape_sweep_keeps_the_suite_path_on_rdna(self, monkeypatch, tmp_path):
        _, fakes, _ = self._main(monkeypatch, tmp_path, "gfx1201", "--shape-sweep")
        fakes["check_gpu_idle"].assert_called_once_with(0, util_threshold=5)
        assert not fakes["run_rdna_protocol"].called
        # 64 work-group processors are 128 CUs.
        kwargs = fakes["run_shape_sweep"].call_args.kwargs
        assert kwargs["tile_shapes"] == cu_tiled_sweep_shapes(128)

    @pytest.mark.parametrize("gcn", MI)
    def test_shape_sweep_on_mi_keeps_the_tile304_shapes(
        self, monkeypatch, tmp_path, gcn
    ):
        mb, fakes, _ = self._main(monkeypatch, tmp_path, gcn, "--shape-sweep")
        assert fakes["run_shape_sweep"].call_args.kwargs["tile_shapes"] is None
        assert mb.GEMM_SHAPES_SWEEP_TILE304 == self.MAIN_TILE304


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestMicrobenchRocprofParsing:
    def test_norm_and_find_col(self):
        rp = _import_microbench_rocprof()
        assert rp._norm("Kernel Name") == "kernel_name"
        headers = ["Kernel Name", "Start Timestamp", "End Timestamp", "Counter"]
        headers_norm = [rp._norm(h) for h in headers]
        assert rp._find_col(headers_norm, ("kernel_name",)) == 0
        assert rp._find_col(headers_norm, ("start_timestamp",)) == 1

    def test_parse_shapes(self):
        rp = _import_microbench_rocprof()
        mb = _import_microbench()
        assert rp._parse_shapes("128,256,512;64,64,64") == [
            (128, 256, 512),
            (64, 64, 64),
        ]
        assert rp._parse_shapes(None) == list(mb.GEMM_SHAPES)

    @pytest.mark.parametrize(
        "metric,k,ok",
        [
            ("matrix_fp16", 8192, True),
            ("matrix_fp4", 32, True),
            ("matrix_fp4", 33, False),
            ("matrix_fp6", 31, False),
        ],
    )
    def test_shape_ok_for_metric(self, metric, k, ok):
        rp = _import_microbench_rocprof()
        passed, _ = rp._shape_ok_for_metric(metric, 64, 64, k)
        assert passed is ok

    def test_parse_rocprof_csv_last_gemm_row(self, tmp_path: Path):
        rp = _import_microbench_rocprof()
        csv_path = tmp_path / "pmc.csv"
        csv_path.write_text(
            "Kernel Name,Start Timestamp,End Timestamp,SQ_INSTS_VALU_MFMA_MOPS_F16\n"
            "copy_kernel,0,1000000,0\n"
            "Cijk_Alik_Bljk_GEMM,1000000,3000000,4096\n"
        )
        parsed = rp._parse_rocprof_csv(csv_path, "SQ_INSTS_VALU_MFMA_MOPS_F16")
        assert parsed["rocprof_kernel_name"] == "Cijk_Alik_Bljk_GEMM"
        assert parsed["rocprof_kernel_ms"] == pytest.approx(2.0)
        assert parsed["rocprof_mops"] == 4096
        assert parsed["rocprof_num_gemm_kernels"] == 1

    def test_find_pmc_csv(self, tmp_path: Path):
        rp = _import_microbench_rocprof()
        nested = tmp_path / "run"
        nested.mkdir()
        csv_path = nested / "123_counter_collection.csv"
        csv_path.write_text("a,b\n1,2\n")
        assert rp._find_pmc_csv(tmp_path) == csv_path

    def test_counter_map_covers_matrix_metrics(self):
        rp = _import_microbench_rocprof()
        mb = _import_microbench()
        matrix_keys = [k for k in mb.ARCH_TLOPS_KEYS if k.startswith("matrix_")]
        assert set(matrix_keys) <= set(rp.COUNTER_MAP)


# ``rocprofv3 -L`` on gfx1201, three of its 145 counters.
_ROCPROF_LIST = """GPU:0
Name:gfx1201
Counter_Name        :\tGL2C_EA_RDREQ_32B
Description         :\tNumber of 32-byte GL2C/EA read requests
Block               :\tGL2C
Dimensions          :\tDIMENSION_INSTANCE[0:31]


Counter_Name        :\tGRBM_GUI_ACTIVE
Description         :\tThe GUI is Active
Dimensions          :\tDIMENSION_INSTANCE[0:0]


Counter_Name        :\tSQ_BUSY_CYCLES
Description         :\tNumber of clock cycles there are active waves in a shader engine
Dimensions          :\tDIMENSION_WGP[0:3] DIMENSION_SHADER_ARRAY[0:1] DIMENSION_SHADER_ENGINE[0:3]
"""

_RDNA_CSV_HEADER = (
    "Dispatch_Id,Kernel_Name,Counter_Name,Counter_Value,"
    "Start_Timestamp,End_Timestamp\n"
)


class TestRdnaCounters:
    def test_counter_list_gives_the_instances(self):
        assert parse_counter_list(_ROCPROF_LIST) == {
            "GL2C_EA_RDREQ_32B": 32,
            "GRBM_GUI_ACTIVE": 1,
            "SQ_BUSY_CYCLES": 32,
        }

    def test_counters_are_medians_of_the_last_matching_dispatches(self, tmp_path):
        rows = ["1,fill_kernel,GRBM_GUI_ACTIVE,5,0,1000"]
        for d in range(2, 10):
            cycles = 2_500_000 if d > 3 else 9_000_000
            rows.append(f"{d},Cijk_Alik_Bljk,GRBM_GUI_ACTIVE,{cycles},0,1000000")
            rows.append(f"{d},Cijk_Alik_Bljk,SQ_WAVES,{d},0,1000000")
        csv_path = tmp_path / "run_counter_collection.csv"
        csv_path.write_text(_RDNA_CSV_HEADER + "\n".join(rows) + "\n")
        parsed = parse_rdna_counters(csv_path, ("Cijk_",))
        assert parsed["matching_dispatches"] == 8
        assert parsed["kernel_ms"] == 1.0
        assert parsed["counters"] == {"GRBM_GUI_ACTIVE": 2_500_000, "SQ_WAVES": 7}
        with pytest.raises(RuntimeError, match="no dispatches"):
            parse_rdna_counters(csv_path, ("_read_kernel",))

    @pytest.mark.parametrize(
        "counter,value,instances,expected",
        [
            # gfx1201, one hipBLASLt FP16 GEMM: 2828276 cycles in 1.1004 ms.
            ("GRBM_GUI_ACTIVE", 2_828_276, 1, 2570.2),
            ("SQ_BUSY_CYCLES", 80_449_253, 32, 2284.7),
        ],
    )
    def test_counter_clock(self, counter, value, instances, expected):
        mhz, why = counter_clock_mhz(counter, value, instances, 1.1004, 2498.0)
        assert mhz == pytest.approx(expected, abs=0.1)
        assert why == ""

    def test_clock_above_the_top_dpm_level_is_accepted(self):
        # gfx1201 lists sclk levels 500 and 2350 MHz; the read kernel ran at
        # 3371 MHz by GRBM_GUI_ACTIVE and 3352 MHz by sysfs.
        mhz, why = counter_clock_mhz("GRBM_GUI_ACTIVE", 3_371_000, 1, 1.0, 3352.0)
        assert mhz == pytest.approx(3371.0) and why == ""

    @pytest.mark.parametrize(
        "value,instances,kernel_ms,expected",
        [
            (0, 1, 1.0, "read 0"),
            (80_449_253, 1, 1.0, "outside 1875-3125 MHz (sysfs 2500 MHz)"),
            # SQ_BUSY_CYCLES divided by 64 CUs instead of 32 WGPs.
            (80_449_253, 64, 1.0, "gives 1257 MHz, outside 1875-3125 MHz"),
            (2_500_000, 1, 0.0, "no kernel time"),
        ],
    )
    def test_zero_or_implausible_clock_is_rejected(
        self, value, instances, kernel_ms, expected
    ):
        mhz, why = counter_clock_mhz("X", value, instances, kernel_ms, 2500.0)
        assert mhz is None
        assert expected in why

    def test_clock_without_a_sysfs_clock_only_rejects_zero(self):
        assert counter_clock_mhz("X", 5e6, 1, 1.0, None) == (5000.0, "")

    @pytest.mark.parametrize(
        "name,counters,value,why",
        [
            (
                "dual_issue_share",
                {"SQ_INSTS_VALU": 1000, "SQ_INSTS_DUAL_VALU_WAVE32": 450},
                0.45,
                "",
            ),
            ("dual_issue_share", {"SQ_INSTS_VALU": 0}, None, "SQ_INSTS_VALU read 0"),
            (
                "dual_issue_share",
                {"SQ_INSTS_VALU": 10, "SQ_INSTS_DUAL_VALU_WAVE32": 30},
                None,
                "outside 0-1",
            ),
            (
                "read_bytes_ratio",
                {"GL2C_EA_RDREQ_64B": 8, "GL2C_EA_RDREQ_128B": 4},
                1.0,
                "",
            ),
            ("read_bytes_ratio", {"GL2C_EA_RDREQ_64B": 0}, None, "read 0"),
            ("read_bytes_ratio", {"GL2C_EA_RDREQ_32B": 1}, None, "outside 0.5-2"),
        ],
    )
    def test_checked_counter_value(self, name, counters, value, why):
        got, reason = checked_counter_value(name, counters, 1024)
        assert got == value
        assert why in reason and bool(reason) is bool(why)


def _rdna_run(rp, tmp_path, listed, children):
    """An RDNA mode run whose children return *children* in turn."""
    run = rp._RdnaRun(
        args=types.SimpleNamespace(),
        shape=(4096, 4096, 4096),
        work_dir=tmp_path,
        listed=listed,
    )
    run.child = MagicMock(side_effect=children)
    return run


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestRocprofRdnaMode:
    LISTED = {"GRBM_GUI_ACTIVE": 1, "SQ_BUSY_CYCLES": 32}

    @staticmethod
    def _parsed(counter, value, kernel_ms=1.0):
        return {"kernel_ms": kernel_ms, "counters": {counter: value}}

    def test_clock_falls_back_to_sq_busy_cycles(self, tmp_path):
        rp = _import_microbench_rocprof()
        # gfx1100: GRBM_GUI_ACTIVE reads 0; SQ_BUSY_CYCLES sums 48 WGPs.
        run = _rdna_run(
            rp,
            tmp_path,
            {**self.LISTED, "SQ_BUSY_CYCLES": 48},
            [
                self._parsed("GRBM_GUI_ACTIVE", 0),
                self._parsed("SQ_BUSY_CYCLES", 48 * 2_300_000),
            ],
        )
        clock = rp._rdna_clock(run, "matrix_fp16", {"sclk_mhz": 2400.0})
        assert clock["clock_mhz"] == 2300.0
        assert clock["clock_source"] == "SQ_BUSY_CYCLES / 48 instance(s)"
        assert clock["clock_vs_sysfs"] == 0.958
        assert clock["clock_rejected"] == {"GRBM_GUI_ACTIVE": "GRBM_GUI_ACTIVE read 0"}

    def test_clock_falls_back_to_sysfs(self, tmp_path):
        rp = _import_microbench_rocprof()
        # Neither clock counter listed.
        run = _rdna_run(rp, tmp_path, {"SQ_WAVES": 1}, [])
        clock = rp._rdna_clock(run, "vector_fp32", {"sclk_mhz": 2410.0})
        assert clock["clock_mhz"] == 2410.0
        assert clock["clock_source"] == "sysfs hwmon, plain run"
        assert set(clock["clock_rejected"]) == set(rp.RDNA_CLOCK_COUNTERS)
        assert not run.child.called

    def test_failed_rocprof_run_is_a_rejection(self, tmp_path):
        rp = _import_microbench_rocprof()
        run = _rdna_run(
            rp,
            tmp_path,
            None,  # rocprofv3 -L failed: every counter is tried
            [{"error": "GRBM_GUI_ACTIVE run failed (rc=1)"}, {"error": "boom"}],
        )
        clock = rp._rdna_clock(run, "read_bw", {})
        assert clock["clock_mhz"] is None and clock["clock_source"] is None
        assert clock["clock_rejected"]["SQ_BUSY_CYCLES"] == "boom"

    def test_dual_issue_needs_both_counters(self, tmp_path):
        rp = _import_microbench_rocprof()
        # gfx1201 lists SQ_INSTS_VALU but no SQ_INSTS_DUAL_VALU_WAVE32.
        run = _rdna_run(rp, tmp_path, {"SQ_INSTS_VALU": 32}, [])
        out = rp._rdna_counter_check(run, "vector_fp32")
        assert out["dual_issue_share"] is None
        assert "SQ_INSTS_DUAL_VALU_WAVE32" in out["dual_issue_share_rejected"]
        assert rp._rdna_counter_check(run, "matrix_fp16") == {}

    def test_read_bytes_sum_the_listed_sizes_over_passes(self, tmp_path):
        rp = _import_microbench_rocprof()
        # gfx1201 lists the 32, 64, 128 and 256 B read requests, not 96 B.
        sizes = ["GL2C_EA_RDREQ_32B", "GL2C_EA_RDREQ_64B"]
        sizes += ["GL2C_EA_RDREQ_128B", "GL2C_EA_RDREQ_256B"]
        reads = {c: 0.0 for c in sizes}
        reads["GL2C_EA_RDREQ_128B"] = rp.RDNA_READ_BYTES / 128
        run = _rdna_run(
            rp,
            tmp_path,
            dict.fromkeys(sizes, 32),
            [
                {"kernel_ms": 2.0, "counters": {c: reads[c] for c in sizes[:2]}},
                {"kernel_ms": 2.0, "counters": {c: reads[c] for c in sizes[2:]}},
            ],
        )
        out = rp._rdna_counter_check(run, "read_bw")
        assert out["read_bytes_ratio"] == 1.0
        groups = [c.args[2] for c in run.child.call_args_list]
        assert groups == [
            ["GL2C_EA_RDREQ_32B", "GL2C_EA_RDREQ_64B"],
            ["GL2C_EA_RDREQ_128B", "GL2C_EA_RDREQ_256B"],
        ]

    def test_counter_probe_passes_a_clean_environment(self):
        rp = _import_microbench_rocprof()
        result = types.SimpleNamespace(stdout=_ROCPROF_LIST)
        with patch.object(rp.subprocess, "run", return_value=result) as run:
            assert rp.list_rocprof_counters("rocprofv3")["SQ_BUSY_CYCLES"] == 32
        assert run.call_args.kwargs["env"] == rp.os.environ
        with patch.object(rp.subprocess, "run", side_effect=OSError("no rocprofv3")):
            assert rp.list_rocprof_counters("rocprofv3") is None

    def test_stderr_reason(self):
        rp = _import_microbench_rocprof()
        stderr = "warning\nTraceback\nRuntimeError: HIP error\nexit\n"
        assert rp.stderr_reason(stderr) == "RuntimeError: HIP error"
        assert rp.stderr_reason("x" * 500) == "x" * 400

    @pytest.mark.parametrize("gfx,rdna", [("gfx1201", True), ("gfx942", False)])
    def test_orchestrator_runs_rdna_mode_on_rdna(self, gfx, rdna):
        rp = _import_microbench_rocprof()
        args = types.SimpleNamespace(device=0, output="/nonexistent/x.json")
        with patch.object(rp, "_device_gfx", return_value=(gfx, None)), patch.object(
            rp, "run_rdna", return_value=0
        ) as run_rdna, patch.object(rp.Path, "mkdir", side_effect=OSError("stop")):
            if rdna:
                assert rp.run_orchestrator(args) == 0
            else:
                with pytest.raises(OSError, match="stop"):
                    rp.run_orchestrator(args)
        assert run_rdna.called is rdna


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestFp4Fp6Helpers:
    def test_triton_available_is_bool(self):
        fp = _import_fp4fp6_helpers()
        assert isinstance(fp.triton_available(), bool)

    def test_mx_block_constant(self):
        fp = _import_fp4fp6_helpers()
        assert fp.MX_BLOCK == 32

    def test_bench_mxfp4_gemm_invalid_k_returns_zero(self):
        fp = _import_fp4fp6_helpers()

        def fake_bench(_fn, *, warmup, rep):
            return 1.0

        assert (
            fp.bench_mxfp4_gemm(64, 64, 33, 0, warmup=1, rep=1, do_bench_fn=fake_bench)
            == 0.0
        )

    def test_mx_tile_sweep_keeps_original_tile_first(self):
        fp = _import_fp4fp6_helpers()
        assert fp.MX_TILE_CONFIGS[0] == (128, 128, 256, 8, None)

    def test_best_tile_keeps_fastest_and_skips_non_dividing(self):
        fp = _import_fp4fp6_helpers()
        ms_by_tile = {
            (128, 128, 256, 8, None): 8.0,
            (128, 128, 128, 8, None): 2.0,
            (128, 128, 64, 8, None): 3.0,
            (64, 128, 64, 4, None): 4.0,
            (128, 128, 64, 4, 1): 1.0,
        }
        tried = []

        def run(*tile):
            tried.append(tile)

        def fake_bench(fn, *, warmup, rep):
            fn()
            return ms_by_tile[tried[-1]]

        m, n, k = 256, 256, 1152  # 1152 is not a multiple of BLOCK_K=256
        best = fp._bench_best_tile(
            m, n, k, run, warmup=1, rep=1, do_bench_fn=fake_bench
        )
        assert (128, 128, 256, 8, None) not in tried
        assert (128, 128, 64, 4, 1) in tried
        assert best == pytest.approx(fp.gemm_tflops(m, n, k, 1.0))

    def test_best_tile_raises_when_every_tile_fails(self):
        fp = _import_fp4fp6_helpers()

        def fake_bench(_fn, *, warmup, rep):
            raise RuntimeError("out of resources")

        with pytest.raises(RuntimeError, match="out of resources"):
            fp._bench_best_tile(
                256, 256, 256, lambda *t: None, warmup=1, rep=1, do_bench_fn=fake_bench
            )


@pytest.mark.gpu
@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestMicrobenchGpuBenchmarks:
    def test_bench_gemm_smoke(self):
        torch = _require_cuda_gpu()
        mb = _import_microbench()
        mb.WARMUP = 1
        mb.REP = 1
        tflops = mb.bench_gemm(64, 64, 64, torch.float16, device=0)
        assert tflops > 0

    @pytest.mark.parametrize("dtype_name", ["float16", "bfloat16", "float32"])
    def test_fma_chain_matches_reference(self, dtype_name):
        torch = _require_cuda_gpu()
        mb = _import_microbench()
        mb.WARMUP = 1
        mb.REP = 1
        dtype = getattr(torch, dtype_name)
        acc_fp32 = dtype != torch.float16
        n, repeat = 4096, 20
        x = torch.full((n,), 1.0, device="cuda", dtype=dtype)
        y = torch.full((n,), 0.25, device="cuda", dtype=dtype)
        z = torch.linspace(-1, 1, n, device="cuda").to(dtype)
        out = torch.empty(n, device="cuda", dtype=torch.float32 if acc_fp32 else dtype)
        mb._triton_fma_chain_ms(
            x, y, z, out, repeat=repeat, block=2048, acc_fp32=acc_fp32
        )
        # x = 1 makes acc grow by y per step, so a dropped FMA shifts it by 0.25.
        ref = z.float() + repeat * 0.25
        torch.testing.assert_close(out.float(), ref, atol=0.05, rtol=0)

    def test_prepare_mxfp4_gemm_on_gpu(self):
        torch = _require_cuda_gpu()
        fp = _import_fp4fp6_helpers()
        dev = torch.device("cuda:0")
        a, b, sa, sb, c = fp.prepare_mxfp4_gemm(64, 64, 64, dev)
        assert a.shape == (64, 32)
        assert b.shape == (64, 32)
        assert sa.shape == (64, 2)
        assert sb.shape == (64, 2)
        assert c.shape == (64, 64)
        assert c.device.type == "cuda"

    def test_microbench_rocprof_single_run_json(self, tmp_path: Path):
        _require_cuda_gpu()
        mb = _import_microbench()
        rp = _import_microbench_rocprof()
        mb.WARMUP = 1
        mb.REP = 1
        out_json = tmp_path / "timing.json"
        M, N, K = 64, 64, 64
        result = rp._run_single("matrix_fp16", M, N, K, device=0)
        out_json.write_text(json.dumps(result))
        loaded = json.loads(out_json.read_text())
        assert loaded["metric"] == "matrix_fp16"
        assert loaded["measured_tflops"] > 0
        assert loaded["flops_per_call"] == mb._gemm_flops(M, N, K)


def _require_rdna_gpu():
    torch = _require_cuda_gpu()
    props = torch.cuda.get_device_properties(0)
    if rdna_family(gfx_target(getattr(props, "gcnArchName", ""))) is None:
        pytest.skip("Requires an RDNA (gfx11 / gfx12) GPU")
    return torch, props


@pytest.mark.gpu
@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestRdnaProtocolGpu:
    @staticmethod
    def _proto(rd, props):
        return rd._RdnaProtocol(
            SysfsTelemetry(pci_bdf(props)), warm_s=0.1, window_ms=20
        )

    @staticmethod
    def _gemm(rd, torch, layout="nt"):
        return rd._Candidate(
            {"layout": layout},
            lambda: rd._gemm_call(512, 512, 512, torch.float16, "cuda:0", layout),
            rd._gemm_flops(512, 512, 512),
        )

    def test_telemetry_reads_the_device(self):
        _, props = _require_rdna_gpu()
        telemetry = SysfsTelemetry(pci_bdf(props))
        if not telemetry.available:
            pytest.skip("sysfs hwmon not visible")
        assert telemetry.read()["sclk_mhz"] > 0

    def test_gemm_call_nt_layout(self):
        torch, _ = _require_rdna_gpu()
        rd = _import_microbench_rdna()
        out = rd._gemm_call(64, 128, 32, torch.bfloat16, "cuda:0", "nt")()
        assert out.shape == (64, 128)
        out = rd._int8_call(64, 128, 32, "cuda:0", "nt")()
        assert out.shape == (64, 128)

    def test_best_of_keeps_the_faster_layout(self):
        torch, props = _require_rdna_gpu()
        rd = _import_microbench_rdna()
        candidates = [self._gemm(rd, torch, layout) for layout in ("nn", "nt")]
        best, rows = self._proto(rd, props).best_of(candidates)
        assert round(best["rate"], 1) == max(r["tflops"] for r in rows)
        assert best["config"]["layout"] in ("nn", "nt")

    def test_accumulator_tile_dual_issues(self):
        torch, _ = _require_rdna_gpu()
        rd = _import_microbench_rdna()
        compiled = rd._vector_call("tile", 256, 4, torch.float32, "cuda:0", 64)()
        torch.cuda.synchronize()
        assert vopd_share(compiled.asm["amdgcn"]) > 0.9

    @pytest.mark.parametrize("kernel", ["triton", "torch_sum"])
    def test_read_kernels_with_clean_flush(self, kernel):
        torch, _ = _require_rdna_gpu()
        rd = _import_microbench_rdna()
        nbytes = 64 * 1024**2
        flush = rd._read_buffer_call(nbytes, 4096, 1024, "cuda:0")
        read = (
            rd._read_buffer_call(nbytes, 4096, 1024, "cuda:0")
            if kernel == "triton"
            else rd._sum_buffer_call(nbytes, "cuda:0")
        )
        ms = rd._time_clean(read, flush, window_ms=20)
        assert nbytes / (ms * 1e-3) / 1e9 > 10

    def test_burst_and_back_to_back(self):
        torch, props = _require_rdna_gpu()
        rd = _import_microbench_rdna()
        proto = self._proto(rd, props)
        best, _ = proto.best_of([self._gemm(rd, torch)])
        winner = rd._Winner(
            "matrix", "matrix_fp16", best["candidate"], "tflops", best["rate"]
        )
        [(_, burst)] = rd._burst_pass([winner], proto, idle_s=0.1, window_ms=20)
        assert burst["tflops"] > 0
        held = rd._back_to_back(winner, proto, seconds=2.0, settle_s=0.75)
        assert held["tflops"] > 0 and held["first_0.75_s_tflops"] > 0
        assert held["min_tflops"] <= held["tflops"] <= held["max_tflops"]
        assert held["chunks"] >= 2
