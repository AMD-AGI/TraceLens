###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Additional CPU-only tests to raise TraceLens unit coverage toward 95%."""

from __future__ import annotations

import gzip
import json
import os
import sys

import pandas as pd
import pytest

from TraceLens.PerfModel import perf_model
from TraceLens.PerfModel.extensions import perf_model_extensions as ext
from TraceLens.PerfModel.extensions.moe_perf_model_extensions import (
    BiasedGroupedTopk,
    MoeSortScatterGather,
    moe_aiter_fused_1stage,
    moe_aiter_unfused_down,
    moe_aiter_unfused_up,
    moe_flydsl_stage1,
    moe_flydsl_stage2,
    moe_gptq_awq_down,
    moe_gptq_awq_up,
    moe_triton_invoke_grouped_gemm,
)
from TraceLens.Reporting.generate_multi_rank_collective_report_pytorch import (
    generate_collective_report,
)
from TraceLens.Reporting.generate_perf_report_pytorch import (
    generate_perf_report_pytorch,
)
from TraceLens.Reporting.generate_perf_report_pytorch_inference import (
    generate_perf_report_pytorch as generate_inference_report,
)
from TraceLens.Reporting.generate_perf_report_pftrace_hip_activity import (
    generate_perf_report_pftrace_hip_activity,
)
from TraceLens.Reporting.generate_perf_report_pftrace_hip_api import (
    generate_perf_report_pftrace_hip_api,
)
from TraceLens.Reporting.generate_perf_report_rocprof import (
    generate_perf_report_rocprof,
)
from TraceLens.Reporting.generate_perf_report_jax import generate_perf_report_jax
from TraceLens.TreePerf.tree_perf import TreePerfAnalyzer

from tests.test_conv_backward_bytes import (
    _conv_bias_bwd_event,
    _conv_bias_fwd_event,
    _conv_bias_relu_bwd_event,
    _conv_bias_relu_fwd_event,
)
from tests.test_dit_fused_ln_modulate import (
    _fused_ln_bwd_event,
    _fused_ln_fwd_event,
)
from tests.test_evoformer_attention_ops import _event as _evoformer_event
from tests.test_reporting_coverage import _build_synthetic_trace, _minimal_pftrace_events
from tests.test_treeperf import GPU_ONLY_TRACE, _build_analyzer, _mk_pytorch_trace


def _write_trace(tmp_path, specs, name="trace.json"):
    path = tmp_path / name
    path.write_text(json.dumps(_build_synthetic_trace(specs)))
    return str(path)


class TestPerfModelConvAndNormBoost:
    @pytest.mark.parametrize(
        "cls,fwd_factory,bwd_cls,bwd_factory",
        [
            (perf_model.ConvBias_, _conv_bias_fwd_event, perf_model.ConvBias_Backward, _conv_bias_bwd_event),
            (
                perf_model.ConvBiasReLU_,
                _conv_bias_relu_fwd_event,
                perf_model.ConvBiasReLU_Backward,
                _conv_bias_relu_bwd_event,
            ),
        ],
    )
    def test_conv_bias_family(self, cls, fwd_factory, bwd_cls, bwd_factory):
        fwd = cls(fwd_factory())
        assert fwd.flops() > 0
        assert fwd.bytes() > 0
        bwd = bwd_cls(bwd_factory())
        assert bwd.flops_bwd() > 0
        assert bwd.bytes_bwd() > 0

    def test_fused_ln_modulate(self):
        fwd = perf_model.FusedLnModulate(_fused_ln_fwd_event())
        assert fwd.flops() > 0
        assert fwd.bytes() > 0

    def test_evoformer_attention(self):
        evo = perf_model.evoformer_attention(_evoformer_event())
        assert evo.flops() > 0
        assert evo.bytes() > 0

    def test_reduce_and_grouped_gemm(self):
        reduce_evt = {
            "name": "aten::mean",
            "args": {
                "Input Dims": [(4, 256)],
                "Input type": ["c10::BFloat16"],
                "Concrete Inputs": ["", "[1]", "True"],
            },
        }
        model = perf_model.aten_reduce(reduce_evt)
        assert model.flops() > 0
        gg_event = {
            "args": {
                "Input Dims": [[4, 128], [8, 256, 128], [8, 256], [8], [8], [8], [8], [8], [4, 4]],
                "Input type": ["c10::BFloat16", "c10::Float8_e4m3fn", "c10::Float", "c10::Int"]
                + ["c10::Int"] * 5,
            }
        }
        g = perf_model.primus_turbo_grouped_gemm(gg_event)
        assert g.flops() > 0


class TestMoeExtensionsBoost:
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

    def test_moe_aiter_fused_1stage_full(self):
        model = moe_aiter_fused_1stage(self.MOE_FUSED)
        assert model.flops() > 0
        assert model.bytes() > 0
        assert model.get_maf_type() == "matrix"
        with pytest.raises(NotImplementedError):
            model.flops_bwd()

    def test_moe_unfused_with_output_dtype(self):
        up_event = {
            "args": {
                "Input Dims": [[32, 4096], [8, 14336, 512], [32, 2, 7168]],
                "Input type": [
                    "c10::BFloat16",
                    "c10::Float8_e4m3fn",
                    "c10::BFloat16",
                ],
            }
        }
        down_event = {
            "args": {
                "Input Dims": [
                    [32, 2, 7168],
                    [8, 4096, 896],
                    [32, 4096],
                    [],
                    [],
                    [],
                ],
                "Input type": [
                    "c10::BFloat16",
                    "c10::Float8_e4m3fn",
                    "c10::BFloat16",
                ],
            }
        }
        up = moe_aiter_unfused_up(up_event)
        down = moe_aiter_unfused_down(down_event)
        assert up.bytes() > 0
        assert down.flops() > 0

    def test_moe_flydsl_gptq_grouped(self):
        fly = {
            "args": {
                "Input Dims": [
                    [32, 4096],
                    [8, 14336, 4096],
                    [8, 4096, 7168],
                    [32, 2],
                ],
                "Input type": [
                    "c10::BFloat16",
                    "c10::Float8_e4m3fn",
                    "c10::Float8_e4m3fn",
                ],
            }
        }
        assert moe_flydsl_stage1(fly).flops() > 0
        assert moe_flydsl_stage2(fly).bytes() > 0
        gptq = {
            "args": {
                "Input Dims": [[32, 4096], [32, 2], [8, 7168, 4096]],
                "Input type": ["c10::BFloat16", "c10::Int", "c10::BFloat16"],
                "MoE topk": 2,
            }
        }
        assert moe_gptq_awq_up(gptq).flops() > 0
        assert moe_gptq_awq_down(gptq).bytes() > 0
        grouped = {
            "args": {
                "Input Dims": [
                    [64, 2048],
                    [128, 1536, 2048],
                    (),
                    [512, 1536],
                    (),
                    (),
                    (),
                    (),
                    [64, 4],
                ],
                "Input type": ["c10::BFloat16", "c10::Float8_e4m3fn", "", "c10::BFloat16"],
            }
        }
        assert moe_triton_invoke_grouped_gemm(grouped).flops() > 0

    def test_moe_aux_models(self):
        topk = BiasedGroupedTopk(
            {
                "args": {
                    "Input Dims": [(32, 256), (256,), (32, 8), (32, 8)],
                    "Input type": ["c10::Float", "c10::Float", "c10::Float", "c10::Int"],
                }
            }
        )
        assert topk.flops() > 0


class TestReportingCliBoost:
    def test_inference_report_all_sheets(self, tmp_path):
        trace = _write_trace(
            tmp_path,
            [
                ("aten::mm", "gemm_kernel", 100),
                ("aten::add", "add_kernel", 15),
                ("aten::relu", "relu_kernel", 10),
            ],
        )
        result = generate_inference_report(
            profile_json_path=trace,
            output_csvs_dir=str(tmp_path / "inf_csvs"),
            output_xlsx_path=str(tmp_path / "inf.xlsx"),
            collective_analysis=False,
            kernel_summary=True,
            short_kernel_study=True,
            include_overlap_info=True,
            group_by_parent_module=True,
            group_by_num_kernels=True,
            micro_idle_thresh_us=5,
        )
        assert "ops_summary" in result or "gpu_timeline" in result

    def test_collective_report_multiprocessing(self, tmp_path):
        for rank in range(2):
            events = {
                "traceEvents": [
                    {
                        "ph": "X",
                        "cat": "kernel",
                        "name": "ncclKernel_AllReduce",
                        "pid": rank,
                        "tid": 3,
                        "ts": 1000 + rank,
                        "dur": 50,
                        "args": {
                            "External id": 10 + rank,
                            "Collective name": "allreduce",
                            "stream": 3,
                        },
                    }
                ]
            }
            (tmp_path / f"rank{rank}_trace.json").write_text(json.dumps(events))
        dfs = generate_collective_report(
            trace_dir=str(tmp_path),
            world_size=2,
            output_csvs_dir=str(tmp_path / "mp_out"),
            use_multiprocessing=True,
            max_workers=2,
            strict_world_size_check=False,
            all2allv_heatmap=False,
        )
        assert "nccl_summary_implicit_sync" in dfs

    @pytest.mark.skipif(
        not os.path.exists(
            os.path.join(os.path.dirname(__file__), "traces/mi210/gpu_only_trace/gpu_only_trace.json.gz")
        ),
        reason="gpu_only trace fixture missing",
    )
    def test_treeperf_gpu_only_extended(self):
        analyzer = TreePerfAnalyzer.from_file(GPU_ONLY_TRACE, rebuild_tree=True)
        launchers = analyzer.get_df_kernel_launchers(include_args=True)
        assert not launchers.empty
        summary = TreePerfAnalyzer.get_df_kernel_launchers_summary(launchers)
        assert not summary.empty
        unique = TreePerfAnalyzer.get_df_kernel_launchers_unique_args(
            launchers, include_pct=True
        )
        assert not unique.empty
        unified = analyzer.build_df_unified_perf_table(include_nccl=False)
        assert isinstance(unified, pd.DataFrame)


class TestPerfModelExtensionsBoost:
    def test_jax_conv_metadata_path(self):
        conv_event = {
            "args": {
                "Input Dims": [[2, 3, 8, 8], [4, 3, 3, 3]],
                "Output Dims": [[2, 4, 6, 6]],
                "Filter Shape": [4, 3, 3, 3],
                "Input type": ["bf16", "bf16"],
                "Concrete Inputs": ["", "", "(1,1)", "(0,0)", "(1,1)", "False", "(0,0)", "1"],
            }
        }
        model = perf_model.jax_conv(conv_event)
        assert model.flops_bwd() > 0
