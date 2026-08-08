###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Phase-8: analysis_utils, perf_model edge cases, tree_perf flags."""

from __future__ import annotations

import gzip
import json
import os

import pandas as pd
import pytest

from TraceLens.Agent.Analysis.category_analyses import analysis_utils as au
from TraceLens.PerfModel import perf_model
from TraceLens.Reporting.generate_perf_report_pytorch_inference import classify_graph_capture_trace
from TraceLens.TreePerf.tree_perf import TreePerfAnalyzer

from tests.test_conv_backward_bytes import _conv_bias_fwd_event
from tests.test_reporting_coverage import _mk_event

RESNET_CKPT = os.path.join(
    os.path.dirname(__file__), "traces/mi300/resnet_act_checkpoint.json.gz"
)


class TestAnalysisUtilsPhase8:
    def test_validate_efficiency_branches(self):
        assert au.validate_efficiency(50, 0, "TFLOPS")["is_anomaly"]
        assert au.validate_efficiency(None, 100, "TFLOPS")["value"] is None
        assert au.validate_efficiency(120, 100, "TFLOPS")["is_anomaly"]
        assert au.validate_efficiency(105, 100, "TFLOPS")["warning"] is not None
        assert au.validate_efficiency(80, 100, "TFLOPS")["value"] == 80.0

    def test_calculate_time_metrics_no_kernel_time(self):
        ops = pd.DataFrame({"name": ["aten::mm"], "operation_count": [3]})
        summary = au.calculate_time_metrics(ops, {"gpu_utilization": {"total_time_ms": 10}})
        assert summary["total_time_ms"] == 0

    def test_calculate_efficiency_with_validation(self):
        out = au.calculate_efficiency_with_validation(50.0, 0.5, 100.0, 5300.0)
        assert "compute_efficiency_pct" in out

    def test_build_operation_metrics(self, tmp_path):
        cat_dir = tmp_path / "category_data"
        cat_dir.mkdir()
        manifest = {
            "platform": "MI300X",
            "gpu_utilization": {"total_time_ms": 100.0},
        }
        (cat_dir / "gemm_metrics.json").write_text("{}")
        ops = pd.DataFrame({
            "name": ["aten::mm"],
            "Kernel Time (µs)_sum": [50000.0],
            "TFLOPS/s_mean": [10.0],
            "TB/s_mean": [0.5],
            "FLOPS/Byte": [1.0],
            "Roofline Bound": ["COMPUTE_BOUND"],
            "Compute Spec": ["matrix_fp16"],
            "kernel_details_summary": ["[{'name': 'Cijk_a'}]"],
            "call_stack_full": ["['aten::mm']"],
        })
        metrics = au.build_operation_metrics(
            ops,
            {"gpu_utilization": {"total_time_ms": 100.0}, "peak_hbm_bw_tbs": 5.3, "max_achievable_tflops": {"matrix_fp16": 100.0}},
            {},
            comparison_scope="standalone",
        )
        assert isinstance(metrics, list)


class TestPerfModelPhase8:
    def test_tex_ts_te_gemm_no_strides(self):
        input_dims = [()] * 19
        input_dims[0] = [128, 64]
        input_dims[5] = [256, 64]
        input_dims[10] = [128, 256]
        event = {
            "args": {
                "Input Dims": input_dims,
                "Input type": ["c10::Float8_e4m3fn"] * 19,
                "Concrete Inputs": [""] * 4 + ["1"] + [""] * 4 + ["1"] + [""] * 4 + [""],
            }
        }
        model = perf_model.tex_ts_te_gemm_ts(event)
        assert model.flops() > 0
        with pytest.raises(NotImplementedError):
            model.flops_bwd()

    def test_grouped_gemm_variable_k(self):
        event = {
            "args": {
                "Input Dims": [
                    [(4, 16), (8, 16)],
                    [(16, 32), (16, 64)],
                ],
                "Input type": ["c10::BFloat16", "c10::BFloat16"],
            }
        }
        g = perf_model.primus_turbo_grouped_gemm_variable_k(event)
        assert g.flops() > 0
        assert g.bytes() > 0

    def test_vllm_unified_attention_with_output(self):
        event = {
            "annotation": "(prefill_128_64_8_0_0_0_0)",
            "args": {
                "Input Dims": [
                    [128, 8, 64],
                    [128, 8, 64],
                    [128, 8, 64],
                    [128, 8, 64],
                ],
                "Input type": ["c10::BFloat16"] * 4,
            },
        }
        model = perf_model.vllm_unified_attention_with_output(event)
        assert model.flops() > 0

    def test_conv_bytes_bwd_none(self):
        assert perf_model.CONV.bytes_bwd_func(
            (2, 3, 8, 8), (4, 3, 3, 3), (2, 4, 6, 6), True, None
        ) is None

    def test_conv_bias_relu_forward(self):
        perf_model.ConvBiasReLU_.fwd_pass_cache.clear()
        fwd = perf_model.ConvBiasReLU_(_conv_bias_fwd_event())
        assert fwd.flops() > 0
        assert fwd.bytes() > 0


@pytest.mark.skipif(not os.path.isfile(RESNET_CKPT), reason="resnet trace missing")
class TestTreePerfPhase8:
    def test_resnet_detect_recompute_nccl_bwd(self):
        analyzer = TreePerfAnalyzer.from_file(
            RESNET_CKPT,
            rebuild_tree=True,
            enable_pseudo_ops=True,
            add_python_func=True,
            detect_recompute=True,
            include_unlinked_kernels=True,
        )
        unified = analyzer.build_df_unified_perf_table(include_nccl=True)
        assert isinstance(unified, pd.DataFrame)
        bwd = [e for e in analyzer.tree.events if "backward" in e.get("name", "")]
        if bwd:
            analyzer.build_df_bwd_perf_metrics(events=bwd[:5])
        launchers = analyzer.get_df_kernel_launchers(
            include_args=True,
            include_kernel_details=True,
            include_call_stack=True,
        )
        if not launchers.empty:
            TreePerfAnalyzer.get_df_kernel_launchers_unique_args(launchers, include_pct=True)


class TestInferenceZipPhase8:
    def test_classify_graph_capture_json_gz(self, tmp_path):
        capture_dir = tmp_path / "cap"
        capture_dir.mkdir()
        events = {"traceEvents": [
            _mk_event(
                "cpu_op",
                "vllm/v1/worker/gpu_model_runner.py(1): _dummy_run",
                1000,
                50,
                1,
                1,
                {},
            ),
            _mk_event("cuda_runtime", "cudaStreamBeginCapture", 1100, 10, 1, 1, {}),
            _mk_event(
                "cpu_op",
                "aten::mm",
                1200,
                20,
                1,
                1,
                {"Input Dims": [[4, 8], [8, 16]]},
            ),
        ]}
        gz_path = capture_dir / "graph_capture_rank_0.json.gz"
        with gzip.open(gz_path, "wt", encoding="utf-8") as f:
            json.dump(events, f)
        classify_graph_capture_trace(str(capture_dir))
        details = json.loads((capture_dir / "execution_details.json").read_text())
        assert details[0]["batch_size"] == 4


class TestPftraceAndArchPhase8:
    def test_pftrace_utils_branches(self, tmp_path):
        from TraceLens.Reporting import pftrace_utils

        preferred = tmp_path / "traceconv"
        preferred.write_text("#!/bin/sh\necho ok\n")
        preferred.chmod(0o755)
        got = pftrace_utils.acquire_traceconv(preferred, tmp_path / "out")
        assert got == preferred.resolve()
        p = tmp_path / "t.json"
        p.write_text("{}")
        assert pftrace_utils.ensure_trace_json(str(p)) == str(p.resolve())

    def test_arch_utils_tl_extension(self, tmp_path, monkeypatch):
        import sys
        import types
        from TraceLens.Agent.Analysis.utils import arch_utils

        pkg_root = tmp_path / "fake_pkg"
        ext_arch = pkg_root / "Agent" / "Analysis" / "utils" / "arch"
        ext_arch.mkdir(parents=True)
        (ext_arch / "CUSTOM.json").write_text('{"mem_bw_gbps": 100}')
        init_py = pkg_root / "__init__.py"
        init_py.write_text("")
        pkg = types.ModuleType("fake_tl_ext")
        pkg.__file__ = str(init_py)
        monkeypatch.setitem(sys.modules, "fake_tl_ext", pkg)
        monkeypatch.setenv("TL_EXTENSION", "fake_tl_ext")
        assert "CUSTOM" in arch_utils._collect_arch_jsons()
