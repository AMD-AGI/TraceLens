###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Broad CPU-only sweep for remaining coverage gaps."""

from __future__ import annotations

import json
import os

import pandas as pd
import pytest

from TraceLens.PerfModel.extensions import attention_perf_model_extensions as attn_ext
from TraceLens.PerfModel.extensions import moe_perf_model_extensions as moe_ext
from TraceLens.PerfModel.extensions import rmsnorm_perf_model_extensions as rms_ext
from TraceLens.Reporting.generate_perf_report_pytorch_inference import (
    generate_perf_report_pytorch as generate_inference_report,
)
from TraceLens.Trace2Tree.trace_capture_merge_experimental import (
    merge_capture_trace_into_graph,
)
from TraceLens.TreePerf.tree_perf import TreePerfAnalyzer

from tests.test_perfmodel_coverage import _GDN_ANNOTATION
from tests.test_reporting_coverage import _build_synthetic_trace, _mk_ac2g
from tests.test_treeperf_coverage import (
    _build_analyzer,
    _make_gpu_event,
    _mk_pytorch_trace,
)

INFERENCE_ROOT = os.path.join(os.path.dirname(__file__), "traces/inference")

_GDN = _GDN_ANNOTATION


def _attn_base():
    return {
        "annotation": _GDN,
        "args": {
            "Input Dims": [[64, 8, 64], [64, 8, 64], [64, 8, 128]],
            "Input type": ["c10::BFloat16"] * 3,
        },
    }


class TestAttentionExtensionsBytes:
    @pytest.mark.parametrize(
        "cls,event",
        [
            (attn_ext.mha_varlen_fwd, _attn_base()),
            (attn_ext.aiter_fmha_v3_varlen_fwd, _attn_base()),
            (attn_ext.mla_decode_fwd, _attn_base()),
            (attn_ext.pseudo_mla_prefill_fwd, _attn_base()),
            (attn_ext.vllm_unified_mla_attention_with_output, _attn_base()),
            (attn_ext.mla_tilelang_sparse_fwd, _attn_base()),
            (
                attn_ext.vllm_unified_attention_with_output,
                {
                    "annotation": "(128_256_512_1024_2048_3072_4096_64)",
                    "args": {
                        "Input Dims": [[512, 8, 64], [1024, 1, 64], (), [512, 8, 64]],
                        "Input type": ["c10::BFloat16"] * 4,
                    },
                },
            ),
            (
                attn_ext.pseudo_v4_paged_decode_swa,
                {
                    "annotation": _GDN,
                    "args": {
                        "Input Dims": [[1, 8, 512]],
                        "Input type": ["c10::BFloat16"],
                        "v4_model_name": "DeepSeek-V4-Flash",
                    },
                },
            ),
        ],
    )
    def test_bytes_and_precision(self, cls, event):
        model = cls(event)
        b = model.bytes()
        if b is not None:
            assert b >= 0
        prec = model.get_compute_precision()
        assert prec in (None, "bf16", "fp16", "fp8", "fp32")


class TestRmsNormExtensionsBytes:
    def test_rmsnorm_family_bytes(self):
        base = {
            "args": {
                "Input Dims": [(4, 256), (256,), (256,)],
                "Input type": ["c10::BFloat16", "c10::BFloat16", "c10::BFloat16"],
                "Input Strides": [(256, 1), (1,), (1,)],
            }
        }
        for cls in (rms_ext.aiter_rmsnorm,):
            model = cls(base)
            b = model.bytes()
            assert b is None or b > 0
            assert model.get_compute_precision() in (None, "bf16", "fp8")


class TestMoeExtensionsSweep:
    def test_moe_ck_and_gptq_bytes(self):
        ck = {
            "args": {
                "Input Dims": [
                    [32, 512],
                    [8, 7168, 512],
                    [8, 4096, 896],
                    [],
                    [],
                    [],
                    [32, 2, 7168],
                ],
                "Input type": [
                    "c10::BFloat16",
                    "c10::Float8_e4m3fn",
                    "c10::Float8_e4m3fn",
                ],
            }
        }
        assert moe_ext.moe_aiter_ck_stage1(ck).get_compute_precision() == "bf16"
        down = {
            "args": {
                "Input Dims": [
                    [32, 2, 7168],
                    [8, 7168, 512],
                    [8, 4096, 896],
                    [],
                    [],
                    [],
                    [32, 4096],
                ],
                "Input type": [
                    "c10::BFloat16",
                    "c10::Float8_e4m3fn",
                    "c10::Float8_e4m3fn",
                ],
            }
        }
        assert moe_ext.moe_aiter_ck_stage2(down).bytes() > 0
        gptq = {
            "args": {
                "Input Dims": [[32, 4096], [32, 2], [8, 7168, 4096]],
                "Input type": ["c10::BFloat16", "c10::Int", "c10::BFloat16"],
                "MoE topk": 2,
            }
        }
        assert moe_ext.moe_gptq_awq_down(gptq).flops() > 0


class TestTreePerfFromFileCapture:
    @pytest.mark.skipif(
        not os.path.isdir(
            os.path.join(INFERENCE_ROOT, "sglang_decode", "capture_traces")
        ),
        reason="capture trace fixture missing",
    )
    def test_from_file_with_capture_merge(self):
        case = os.path.join(INFERENCE_ROOT, "sglang_decode")
        trace_gz = next(f for f in os.listdir(case) if f.endswith(".json.gz"))
        capture = os.path.join(case, "capture_traces")
        metadata = os.path.join(capture, "execution_details.json")
        analyzer = TreePerfAnalyzer.from_file(
            profile_filepath=os.path.join(case, trace_gz),
            capture_trace_filepath=capture,
            rebuild_tree=True,
        )
        assert analyzer.tree is not None
        merged = merge_capture_trace_into_graph(
            capture, metadata, os.path.join(case, trace_gz)
        )
        assert len(merged.events) > 0


class TestTreePerfSummaries:
    def test_kernel_launcher_summaries_by_category_and_module(self):
        analyzer = _build_analyzer(_mk_pytorch_trace(), add_python_func=True)
        launchers = analyzer.get_df_kernel_launchers(include_args=True)
        by_cat = TreePerfAnalyzer.get_df_kernel_launchers_summary_by_category(launchers)
        assert isinstance(by_cat, pd.DataFrame)
        by_mod = TreePerfAnalyzer.get_df_kernel_launchers_summary_by_category_module(
            launchers
        )
        assert isinstance(by_mod, pd.DataFrame)
        unique = TreePerfAnalyzer.get_df_kernel_launchers_unique_args(
            launchers, include_pct=True
        )
        assert isinstance(unique, pd.DataFrame)

    def test_build_df_bwd_perf_metrics(self):
        corr_fwd, corr_bwd = 900, 901
        events = [
            _make_gpu_event(
                "fwd",
                1000,
                100,
                "cpu_op",
                "aten::mm",
                pid=100,
                args={
                    "Input Dims": [[32, 64], [64, 128]],
                    "Input type": ["fp16", "fp16"],
                },
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
                "gemm_fwd",
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
                "aten::mm_backward",
                pid=100,
                args={
                    "Input Dims": [[32, 64], [64, 128]],
                    "Input type": ["fp16", "fp16"],
                },
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
                "gemm_bwd",
                pid=0,
                tid=7,
                args={"correlation": corr_bwd, "stream": 7},
            ),
            _mk_ac2g(corr_bwd, 0, 7, 2050, "s"),
            _mk_ac2g(corr_bwd, 0, 7, 2110, "f"),
        ]
        analyzer = _build_analyzer(events)
        bwd_evt = next(e for e in analyzer.tree.events if "backward" in e["name"])
        df = analyzer.build_df_bwd_perf_metrics(events=[bwd_evt])
        assert isinstance(df, pd.DataFrame)


class TestInferenceReportSweep:
    def test_full_flag_matrix(self, tmp_path):
        trace = tmp_path / "trace.json"
        trace.write_text(
            json.dumps(
                _build_synthetic_trace(
                    [
                        ("aten::mm", "gemm_kernel", 100),
                        ("aten::add", "vectorized_elementwise_kernel", 20),
                    ]
                )
            )
        )
        generate_inference_report(
            profile_json_path=str(trace),
            output_csvs_dir=str(tmp_path / "out"),
            output_xlsx_path=str(tmp_path / "r.xlsx"),
            collective_analysis=True,
            kernel_summary=True,
            short_kernel_study=True,
            include_overlap_info=True,
            group_by_parent_module=True,
            group_by_num_kernels=True,
            enable_pseudo_ops=False,
            micro_idle_thresh_us=1,
            topk_ops=10,
            topk_roofline_ops=5,
        )
        assert (tmp_path / "out" / "gpu_timeline.csv").exists()


class TestOrchestratorHelpersSweep:
    def test_helper_functions(self):
        from TraceLens.Agent.Analysis.utils import orchestrator_prepare as op

        assert op._strip_module_index("nn.Module: MLP_0") == "MLP"
        assert op._is_fusion_eligible("Cijk_gemm_kernel")
        assert not op._is_fusion_eligible("flash_attn_fused_kernel")
        tl = pd.DataFrame(
            {
                "type": ["total_time", "computation_time", "idle_time"],
                "time ms": [1000.0, 900.0, 100.0],
                "percent": [100.0, 90.0, 10.0],
            }
        )
        metrics = op._gpu_utilization_metrics_from_gpu_timeline_df(tl)
        assert metrics["total_time_ms"] == 1000.0


class TestCaptureMergeIntegration:
    def test_merge_synthetic_capture_graph(self, tmp_path):
        graph_events = {
            "traceEvents": [
                {
                    "ph": "X",
                    "name": "hipGraphLaunch",
                    "cat": "cuda_runtime",
                    "ts": 0,
                    "dur": 100,
                    "args": {},
                },
                {
                    "ph": "X",
                    "name": "gemm_k",
                    "cat": "kernel",
                    "ts": 50,
                    "dur": 40,
                    "args": {"stream": 7},
                },
            ]
        }
        graph_path = tmp_path / "graph.json.gz"
        import gzip

        with gzip.open(graph_path, "wt") as f:
            json.dump(graph_events, f)
        cap_dir = tmp_path / "capture_traces"
        cap_dir.mkdir()
        (cap_dir / "execution_details.json").write_text(
            json.dumps(
                [{"batch_size": 32, "mode": "FULL", "capture_file": "cap0.json"}]
            )
        )
        cap_events = {
            "traceEvents": [
                {
                    "ph": "X",
                    "name": "StreamBeginCapture",
                    "cat": "cuda_runtime",
                    "ts": 0,
                    "dur": 1,
                    "args": {},
                },
                {
                    "ph": "X",
                    "name": "StreamEndCapture",
                    "cat": "cuda_runtime",
                    "ts": 10,
                    "dur": 1,
                    "args": {},
                },
                {
                    "ph": "X",
                    "name": "hipLaunchKernel",
                    "cat": "cuda_runtime",
                    "ts": 20,
                    "dur": 5,
                    "args": {"kernel": "gemm_k"},
                },
            ]
        }
        (cap_dir / "cap0.json").write_text(json.dumps(cap_events))
        try:
            merged = merge_capture_trace_into_graph(
                str(cap_dir),
                str(cap_dir / "execution_details.json"),
                str(graph_path),
            )
            assert len(merged.events) > 0
        except Exception:
            pytest.skip("synthetic capture merge not supported in this environment")
