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

import importlib.util, json, pytest
from pathlib import Path
from unittest.mock import patch
from TraceLens.PerfModel.benchmarking.microbench_utils import (
    _int_metric,
    check_gpu_idle,
    resolve_physical_device,
)

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

    return mb


def _import_microbench_rocprof():
    _require_torch()

    return rp


def _import_fp4fp6_helpers():
    _require_torch()

    return fp


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


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestMicrobenchHelpers:
    def test_gemm_flops(self):
        mb = _import_microbench()
        assert mb._gemm_flops(4, 8, 16) == 2 * 4 * 8 * 16

    @pytest.mark.parametrize(
        "gpu_name,mem_gb,expected",
        [
            ("AMD Instinct MI300X", 192.0, "MI300X"),
            ("Generic GPU", 280.0, "MI355X"),
            ("Some Card", 64.0, "Card"),
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
        assert payload["max_achievable_tflops"]["matrix_fp16"] == 123
        assert payload["max_achievable_tflops"]["vector_fp16"] == 46
        assert set(mb.ARCH_TLOPS_KEYS) <= set(payload["max_achievable_tflops"])

    def test_gemm_shapes_non_empty(self):
        mb = _import_microbench()
        assert len(mb.GEMM_SHAPES) >= 1


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
