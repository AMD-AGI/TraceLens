###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Phase-6 targeted coverage push toward 95% CPU-only line coverage."""

from __future__ import annotations

import json
import os
import sys
from unittest.mock import patch

import pandas as pd
import pytest

from TraceLens.Agent.Analysis.utils.orchestrator_prepare import (
    _extract_standalone_fusion_candidates,
)
from TraceLens.PerfModel import perf_model
from TraceLens.PerfModel.extensions import moe_perf_model_extensions as moe_ext
from TraceLens.Reporting.generate_perf_report_pytorch import (
    generate_perf_report_pytorch,
)
from TraceLens.Reporting.generate_perf_report_pytorch_inference import (
    classify_graph_capture_trace,
)
from TraceLens.Reporting.rocprof_analysis import RocprofAnalyzer, _categorize_kernel
from TraceLens.TreePerf.tree_perf import TreePerfAnalyzer
from TraceLens.util import RocprofParser

from tests.test_agent_coverage import (
    _StubAnalyzer,
    _StubTree,
    _kernel_event,
    _write_minimal_orchestrator_csvs,
)
from tests.test_conv_backward_bytes import _conv_bias_bwd_event, _conv_bias_fwd_event
from tests.test_flash_attention_backward import _bwd_event as _flash_bwd_event
from tests.test_perfmodel_coverage import _ARCH, _gemm_event, _moe_unfused_event
from tests.test_reporting_coverage import (
    _mk_event,
    _write_trace,
)
from tests.test_treeperf_coverage import _build_analyzer, _make_gpu_event, _mk_ac2g

ROCprof_FILE = os.path.join(os.path.dirname(__file__), "rocprof/908_results.json.gz")
NORM_TRACE = os.path.join(
    os.path.dirname(__file__),
    "traces/perf_model/normalization/normalization_layer_test.json.gz",
)
INFERENCE_ROOT = os.path.join(os.path.dirname(__file__), "traces/inference")


class TestOrchestratorPhase6:
    def _run_main(self, tmp_path, monkeypatch, fusion_extract_raises=False):
        from TraceLens.Agent.Analysis.utils import orchestrator_prepare as op

        out = str(tmp_path)
        _write_minimal_orchestrator_csvs(out, comparative=False)
        csv_dir = os.path.join(out, "perf_report_csvs")
        pd.DataFrame(
            {
                "name": ["aten::mm", "aten::add"],
                "op category": ["GEMM", ""],
                "Kernel Time (µs)_sum": [1000.0, 500.0],
                "total_duration_us": [60000.0, 1000.0],
                "kernel_details_summary": [
                    "[{'name': 'Cijk_a'}]",
                    "[{'name': 'ew_add'}]",
                ],
                "Data Moved (MB)": [10.0, 1.0],
                "perf_params": ["{}", "{}"],
                "Input Dims": ["[[2,3]]", "[[2,3]]"],
            }
        ).to_csv(os.path.join(csv_dir, "unified_perf_summary.csv"), index=False)
        pd.DataFrame({"name": ["aten::mm"], "op category": ["GEMM"]}).to_csv(
            os.path.join(csv_dir, "ops_summary.csv"), index=False
        )

        events = [
            {
                "name": "gemm_kernel",
                "dur": 100,
                "ts": 1000,
                "UID": 0,
                "_category": "kernel",
                "cat": "kernel",
                "args": {"stream": 0},
            },
            {
                "name": "MemcpyHtoD",
                "dur": 20,
                "ts": 1100,
                "UID": 1,
                "_category": "gpu_memcpy",
                "cat": "gpu_memcpy",
                "args": {"bytes": 4096, "stream": 1},
            },
            {
                "name": "MemcpyDtoH",
                "dur": 25,
                "ts": 1120,
                "UID": 2,
                "_category": "gpu_memcpy",
                "cat": "gpu_memcpy",
                "args": {"bytes": 8192, "stream": 1},
            },
            {
                "name": "MemcpyDtoD",
                "dur": 15,
                "ts": 1150,
                "UID": 3,
                "_category": "gpu_memcpy",
                "cat": "gpu_memcpy",
                "args": {"bytes": 2048, "stream": 1},
            },
            {
                "name": "MemcpyCustom",
                "dur": 10,
                "ts": 1170,
                "UID": 4,
                "_category": "gpu_memcpy",
                "cat": "gpu_memcpy",
                "args": {"bytes": 1024, "stream": 1},
            },
            {
                "name": "ncclKernel_AllReduce",
                "dur": 40,
                "ts": 1200,
                "UID": 5,
                "_category": "kernel",
                "cat": "kernel",
                "args": {"stream": 2},
            },
        ]
        tree = _StubTree(events, {i: e for i, e in enumerate(events)})
        analyzer = _StubAnalyzer(tree)

        class _FakeTreePerfAnalyzer:
            @classmethod
            def from_file(cls, *args, **kwargs):
                return analyzer

        monkeypatch.setattr(op, "TreePerfAnalyzer", _FakeTreePerfAnalyzer)
        if fusion_extract_raises:
            monkeypatch.setattr(
                op,
                "_extract_standalone_fusion_candidates",
                lambda *a, **k: (_ for _ in ()).throw(RuntimeError("fusion fail")),
            )
        else:
            k1 = _kernel_event(10, "Cijk_a", dur=500)
            k2 = _kernel_event(11, "vectorized_elementwise_kernel add", dur=300)
            module = {
                "name": "nn.Module: MLP_0",
                "_category": "aten",
                "gpu_events": [10, 11],
                "args": {"Input Dims": "[[2,3]]"},
            }
            tree2 = _StubTree([module], {10: k1, 11: k2})
            monkeypatch.setattr(
                op,
                "_extract_standalone_fusion_candidates",
                lambda a, t, d: _extract_standalone_fusion_candidates(
                    _StubAnalyzer(tree2), tree2, d
                ),
            )

        old_argv = sys.argv
        sys.argv = [
            "orchestrator_prepare",
            "--trace-path",
            "/fake/trace.json",
            "--platform",
            "MI300X",
            "--output-dir",
            out,
        ]
        try:
            op.main()
        finally:
            sys.argv = old_argv
        return out

    def test_main_no_time_column_sync_and_memcpy_dirs(self, tmp_path, monkeypatch):
        out = self._run_main(tmp_path, monkeypatch)
        mk = json.loads(
            open(os.path.join(out, "category_data", "multi_kernel_data.json")).read()
        )
        dirs = mk["memcpy_summary"]["by_direction"]
        assert "H2D" in dirs and "D2H" in dirs and "D2D" in dirs and "other" in dirs
        assert mk["nccl_summary"]["total_count"] >= 1
        manifest = json.loads(
            open(os.path.join(out, "category_data", "category_manifest.json")).read()
        )
        cats = {c["name"]: c for c in manifest["categories"]}
        assert "gemm" in cats
        assert cats["gemm"].get("has_sync_bottleneck") is True

    def test_main_fusion_extraction_failure_writes_empty(self, tmp_path, monkeypatch):
        out = self._run_main(tmp_path, monkeypatch, fusion_extract_raises=True)
        fusion = json.loads(
            open(os.path.join(out, "category_data", "fusion_candidates.json")).read()
        )
        assert fusion == []

    def test_standalone_enrichment_without_data_mb(self, tmp_path):
        k1 = _kernel_event(10, "Cijk_gemm_a", dur=500)
        k2 = _kernel_event(11, "vectorized_elementwise_kernel add", dur=300)
        module = {
            "name": "nn.Module: Attn_0",
            "_category": "aten",
            "gpu_events": [10, 11],
            "args": {"Input Dims": "[[2,3]]"},
        }
        tree = _StubTree([module], {10: k1, 11: k2})
        analyzer = _StubAnalyzer(tree)
        csv_dir = tmp_path / "csv"
        csv_dir.mkdir()
        pd.DataFrame(
            {
                "kernel_details_summary": [
                    "[{'name': 'Cijk_gemm_a'}]",
                    "[{'name': 'vectorized_elementwise_kernel add'}]",
                ],
                "op category": ["GEMM", "elementwise"],
                "Data Moved (MB)": [10.0, 4.0],
                "perf_params": ["{'M':2,'N':4,'K':3}", "{'shape_in1': [4, 4]}"],
                "Input Dims": ["[[2,3]]", "[[4,4]]"],
            }
        ).to_csv(csv_dir / "unified_perf_summary.csv", index=False)
        cands = _extract_standalone_fusion_candidates(analyzer, tree, str(csv_dir))
        assert isinstance(cands, list)


class TestRocprofPhase6:
    @pytest.mark.parametrize(
        "name,expected",
        [
            ("Cijk_gemm", "GEMM"),
            ("vectorized_elementwise_kernel", "Elementwise"),
            ("reduce_kernel_sum", "Reduction"),
            ("conv2d_fwd", "Convolution"),
            ("layer_norm_kernel", "Normalization"),
            ("flash_attn_fwd", "Attention"),
            ("MemcpyDtoD", "Memory"),
            ("ncclBroadcast", "COMM"),
            ("unknown_xyz", "Other"),
        ],
    )
    def test_categorize_kernel_branches(self, name, expected):
        assert _categorize_kernel(name) == expected

    @pytest.mark.skipif(
        not os.path.isfile(ROCprof_FILE), reason="rocprof fixture missing"
    )
    def test_rocprof_analyzer_all_dataframes(self):
        data = RocprofParser.load_rocprof_data(ROCprof_FILE)
        kernels = RocprofParser.extract_kernel_events(data)
        memory = RocprofParser.extract_memory_events(data)
        api = RocprofParser.extract_api_events(data)
        metadata = RocprofParser.get_metadata(data)
        analyser = RocprofAnalyzer(kernels, memory, api, metadata)
        assert not analyser.get_df_gpu_timeline().empty
        assert not analyser.get_df_kernel_summary().empty
        assert isinstance(analyser.get_df_kernel_details(), pd.DataFrame)
        assert isinstance(analyser.get_df_short_kernels(5), pd.DataFrame)
        assert isinstance(analyser.get_df_short_kernel_histogram(), pd.DataFrame)
        assert isinstance(analyser.get_df_kernel_summary_by_category(), pd.DataFrame)


class TestPerfModelPhase6:
    def test_conv_bias_bwd_empty_dims(self):
        perf_model.ConvBias_.fwd_pass_cache.clear()
        evt = {
            "args": {
                "Input Dims": [],
                "Input type": ["c10::BFloat16"],
                "Sequence number": 42,
            }
        }
        details = perf_model.ConvBias_Backward.get_param_details(evt)
        assert details["input_shape"] is None

    def test_conv_bias_relu_bwd_cache_path(self):
        perf_model.ConvBiasReLU_.fwd_pass_cache.clear()
        fwd = perf_model.ConvBiasReLU_(_conv_bias_fwd_event())
        assert fwd.flops() > 0
        bwd = perf_model.ConvBiasReLU_Backward(_conv_bias_bwd_event())
        assert bwd.flops_bwd() > 0

    def test_conv1d_bytes_func_none(self):
        assert (
            perf_model.CONV.bytes_func((2, 4, 32), (8, 4, 3), (2, 8, 30), False, None)
            is None
        )

    def test_tev2_pseudo_gemm_and_grouped_gemm(self):
        event = _gemm_event("tev2::pseudo_gemm", (4, 8), (8, 16))
        model = perf_model.tev2_pseudo_gemm(event)
        assert model.flops() > 0

        gg = {
            "args": {
                "Input Dims": [[12, 16], [4, 16, 32]],
                "Input type": ["c10::BFloat16", "c10::BFloat16"],
            }
        }
        g = perf_model.primus_turbo_grouped_gemm(gg)
        assert g.bytes() > 0
        with pytest.raises(NotImplementedError):
            g.flops_bwd()

    def test_sdpa_simulation_qkt_none_returns_none(self):
        event = {
            "args": {
                "Input Dims": [[2, 128, 8, 64], [2, 128, 8, 64], [2, 128, 8, 64]],
                "Input type": ["c10::BFloat16"] * 3,
                "Concrete Inputs": ["", "", "", "0.0", "False"],
            }
        }
        model = perf_model.aten__scaled_dot_product_flash_attention(event, arch=_ARCH)
        with patch.object(
            perf_model.GEMM, "get_simulation_time_func", return_value=(None, None)
        ):
            assert model.get_simulation_time() is None

    def test_flash_attention_backward_flops(self):
        model = perf_model.flash_attention_backward(_flash_bwd_event())
        assert model.flops_bwd() > 0
        assert model.bytes() > 0


class TestMoeExtensionsPhase6:
    MOE_FUSED = {
        "args": {
            "Input Dims": [
                [32, 4096],
                [8, 28672, 512],
                [8, 4096, 7168],
                [32, 2],
            ],
            "Input type": [
                "c10::BFloat16",
                "c10::Float8_e4m3fn",
                "c10::Float8_e4m3fn",
                "c10::Float32",
            ],
        }
    }

    def test_fused_moe_bytes(self):
        model = moe_ext.moe_aiter_fused_1stage(self.MOE_FUSED)
        assert model.bytes() > 0

    def test_unfused_moe_bytes_none_bpe(self):
        assert (
            moe_ext.UnfusedMoE_Up.bytes_func(8, 4096, 14336, 8, 2, True, None, 2, 2)
            is None
        )
        up = moe_ext.moe_triton_unfused_up(
            _moe_unfused_event(kernel_name="moe_fp8_up_kernel")
        )
        assert up.bytes() > 0

    def test_moe_auxiliary_classes(self):
        blockscale = {
            "args": {
                "Input Dims": [
                    [128, 256],
                    [128, 256],
                    [512, 28672, 256],
                    [512, 256, 7168],
                ],
                "Input type": [
                    "c10::BFloat16",
                    "c10::Float8_e4m3fn",
                    "c10::Float8_e4m3fn",
                    "c10::Float8_e4m3fn",
                ],
                "Concrete Inputs": [""] * 8 + ["2"],
            }
        }
        assert moe_ext.moe_aiter_fused_blockscale(blockscale).bytes() > 0
        topk_evt = {
            "args": {
                "Input Dims": [(32, 256), (256,), (32, 8), (32, 8)],
                "Input type": ["c10::Float"] * 3 + ["c10::Int"],
            }
        }
        b = moe_ext.BiasedGroupedTopk(topk_evt).bytes()
        assert b is None or b >= 0


class TestTreePerfPhase6:
    def test_inconsistent_kernel_list_length(self):
        with pytest.warns(UserWarning, match="Inconsistent kernel list length"):
            stats = TreePerfAnalyzer._summarize_kernel_stats(
                [
                    [{"name": "a", "dur": 10}],
                    [{"name": "a", "dur": 20}, {"name": "b", "dur": 30}],
                ],
                agg_metrics=["mean"],
            )
        assert isinstance(stats, list)

    def test_bwd_linked_not_implemented_fallback(self):
        corr_fwd, corr_bwd = 900, 901
        events = [
            _make_gpu_event(
                "fwd",
                1000,
                100,
                "cpu_op",
                "aten::unknown_custom_op",
                pid=100,
                args={"Input Dims": [[4, 4]]},
            ),
            _make_gpu_event(
                "rt1",
                1010,
                5,
                "cuda_runtime",
                "hipLaunchKernel",
                pid=100,
                args={"correlation": corr_fwd},
            ),
            _make_gpu_event(
                "k1",
                1050,
                50,
                "kernel",
                "custom_kernel",
                pid=0,
                tid=7,
                args={"correlation": corr_fwd, "stream": 7},
            ),
            _mk_ac2g(corr_fwd, 0, 7, 1050, "s"),
            _mk_ac2g(corr_fwd, 0, 7, 1100, "f"),
            _make_gpu_event(
                "bwd",
                2000,
                100,
                "cpu_op",
                "aten::unknown_custom_op_backward",
                pid=100,
                args={"Input Dims": [[4, 4]]},
            ),
            _make_gpu_event(
                "rt2",
                2010,
                5,
                "cuda_runtime",
                "hipLaunchKernel",
                pid=100,
                args={"correlation": corr_bwd},
            ),
            _make_gpu_event(
                "k2",
                2050,
                60,
                "kernel",
                "custom_bwd_kernel",
                pid=0,
                tid=7,
                args={"correlation": corr_bwd, "stream": 7},
            ),
            _mk_ac2g(corr_bwd, 0, 7, 2050, "s"),
            _mk_ac2g(corr_bwd, 0, 7, 2110, "f"),
        ]
        analyzer = _build_analyzer(events)
        fwd = next(
            e
            for e in analyzer.tree.events
            if "custom_op" in e["name"] and "backward" not in e["name"]
        )
        bwd = next(e for e in analyzer.tree.events if "backward" in e["name"])
        fwd["bwd_events"] = [bwd["UID"]]
        bwd["fwd_event"] = fwd["UID"]
        df = analyzer.build_df_unified_perf_table(events=[fwd, bwd])
        assert isinstance(df, pd.DataFrame)


class TestReportingPhase6:
    @pytest.mark.skipif(not os.path.isfile(NORM_TRACE), reason="norm trace missing")
    def test_pytorch_report_all_bwd_overlap_flags(self, tmp_path):
        generate_perf_report_pytorch(
            profile_json_path=NORM_TRACE,
            output_csvs_dir=str(tmp_path / "csv"),
            output_xlsx_path=str(tmp_path / "out.xlsx"),
            kernel_summary=True,
            short_kernel_study=True,
            include_overlap_info=True,
            group_by_num_kernels=True,
            topk_ops=10,
            topk_roofline_ops=5,
            include_unlinked_kernels=True,
            include_call_stack=True,
            enable_pseudo_ops=True,
        )
        assert (tmp_path / "csv" / "gpu_timeline.csv").exists()

    def test_classify_graph_capture_zip_and_dummy_run(self, tmp_path):
        capture_dir = tmp_path / "capture"
        capture_dir.mkdir()
        events = {
            "traceEvents": [
                _mk_event(
                    "cpu_op",
                    "vllm/v1/worker/gpu_model_runner.py(100): _dummy_run",
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
                    {"Input Dims": [[8, 64], [64, 128]]},
                ),
            ]
        }
        json_path = capture_dir / "trace.json"
        json_path.write_text(json.dumps(events))
        classify_graph_capture_trace(str(capture_dir))
        details = json.loads((capture_dir / "execution_details.json").read_text())
        assert details[0]["batch_size"] == 8

    def test_inference_extension_and_sanity(self, tmp_path):
        from TraceLens.Reporting.generate_perf_report_pytorch_inference import (
            generate_perf_report_pytorch as gen_inf,
            perf_report_sanity_check,
        )

        trace = _write_trace(
            tmp_path,
            [
                ("aten::mm", "gemm_kernel", 100),
                ("aten::native_layer_norm", "layer_norm_kernel", 30),
            ],
        )
        gen_inf(
            profile_json_path=trace,
            output_csvs_dir=str(tmp_path / "out"),
            output_xlsx_path=str(tmp_path / "r.xlsx"),
            collective_analysis=True,
            kernel_summary=True,
            short_kernel_study=True,
            include_overlap_info=True,
            group_by_parent_module=True,
            group_by_num_kernels=True,
            topk_ops=5,
            include_unlinked_kernels=True,
            micro_idle_thresh_us=0,
        )
        analyzer = TreePerfAnalyzer.from_file(trace)
        sanity = perf_report_sanity_check(
            analyzer.tree.events,
            pd.read_csv(str(tmp_path / "out" / "gpu_timeline.csv")),
            analyzer.get_df_kernel_launchers(),
            analyzer.build_df_unified_perf_table(),
        )
        assert isinstance(sanity, dict)

    @pytest.mark.skipif(
        not os.path.isdir(os.path.join(INFERENCE_ROOT, "vllm_decode_full")),
        reason="inference fixture missing",
    )
    def test_inference_real_fixture_report(self, tmp_path):
        from TraceLens.Reporting.generate_perf_report_pytorch_inference import (
            generate_perf_report_pytorch as gen_inf,
        )

        case = os.path.join(INFERENCE_ROOT, "vllm_decode_full")
        trace = next(
            os.path.join(case, f) for f in os.listdir(case) if f.endswith(".json.gz")
        )
        gen_inf(
            profile_json_path=trace,
            output_csvs_dir=str(tmp_path / "inf"),
            output_xlsx_path=str(tmp_path / "inf.xlsx"),
            collective_analysis=False,
            kernel_summary=True,
        )
        assert (tmp_path / "inf" / "gpu_timeline.csv").exists()


class TestKernelFusionMainPhase6:
    def test_kernel_fusion_main_end_to_end(self, tmp_path):
        from TraceLens.Agent.Analysis.category_analyses import (
            kernel_fusion_analysis as kfa,
        )

        out = str(tmp_path)
        _write_minimal_orchestrator_csvs(out, comparative=False)
        cat_dir = os.path.join(out, "category_data")
        os.makedirs(cat_dir, exist_ok=True)
        fusion_path = os.path.join(cat_dir, "fusion_candidates.json")
        with open(fusion_path, "w") as f:
            json.dump(
                [
                    {
                        "module_name": "nn.Module: MLP_0",
                        "base_name": "MLP",
                        "kernels": [
                            {"name": "Cijk_a", "type": "GEMM", "dur_us": 500},
                            {"name": "ew_add", "type": "elementwise", "dur_us": 300},
                        ],
                        "total_kernel_time_us": 800,
                        "instance_count": 1,
                    }
                ],
                f,
            )
        old_argv = sys.argv
        sys.argv = ["kernel_fusion_analysis", "--output-dir", out]
        try:
            kfa.main()
        finally:
            sys.argv = old_argv
        cat_dir = os.path.join(out, "category_data")
        assert os.path.isfile(os.path.join(cat_dir, "kernel_fusion_metrics.json"))
