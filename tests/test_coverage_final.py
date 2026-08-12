###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Final CPU-only coverage push toward 95% line coverage."""

from __future__ import annotations

import gzip
import json
import os
import sys
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from TraceLens.Agent.Analysis.utils.orchestrator_prepare import (
    _extract_comparative_fusion_candidates,
    _extract_standalone_fusion_candidates,
)
from TraceLens.PerfModel import perf_model
from TraceLens.PerfModel.extensions import moe_perf_model_extensions as moe_ext
from TraceLens.PerfModel.extensions import perf_model_extensions as pext
from TraceLens.Reporting.generate_perf_report_pytorch import (
    generate_perf_report_pytorch,
)
from TraceLens.Reporting.generate_perf_report_pytorch_inference import (
    add_truncated_kernel_details as add_truncated_inference,
    generate_perf_report_pytorch as generate_inference_report,
    perf_report_sanity_check,
)
from TraceLens.Trace2Tree.trace_capture_merge_experimental import (
    _get_cached_capture_tree,
    align_streams,
    capture_has_kernel_names,
    get_subtree_events,
    is_multistream,
    verify_subtree_events,
)
from TraceLens.Trace2Tree.trace_to_tree import TraceToTree
from TraceLens.TreePerf.tree_perf import JaxTreePerfAnalyzer, TreePerfAnalyzer

from tests.test_agent_coverage import (
    _StubAnalyzer,
    _StubTree,
    _kernel_event,
    _write_minimal_orchestrator_csvs,
)
from tests.test_conv_backward_bytes import (
    _conv_bias_bwd_event,
    _conv_bias_fwd_event,
    _conv_bias_relu_bwd_event,
    _conv_bias_relu_fwd_event,
)
from tests.test_perfmodel_coverage import _ARCH, _gemm_event
from tests.test_reporting_coverage import _build_synthetic_trace, _mk_ac2g, _mk_event
from tests.test_treeperf_coverage import (
    _build_analyzer,
    _make_gpu_event,
    _mk_pytorch_trace,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore:get_df_kernel_launchers_summary_by_shape is deprecated.*:UserWarning",
    "ignore:Source column .* not found.*:UserWarning",
    "ignore:There are hipgraph launches.*:UserWarning",
    "ignore:Found .* events with failed performance metric.*:UserWarning",
    "ignore:Inconsistent kernel list length found.*:UserWarning",
    "ignore:Input DataFrame is empty.*:UserWarning",
    "ignore:Kernel name missing in capture event args.*:UserWarning",
)


def _write_trace(tmp_path, specs, name="trace.json"):
    path = tmp_path / name
    path.write_text(json.dumps(_build_synthetic_trace(specs)))
    return str(path)


def _write_trace_gz(tmp_path, specs, name="trace.json.gz"):
    path = tmp_path / name
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(_build_synthetic_trace(specs), f)
    return str(path)


# ---------------------------------------------------------------------------
# PerfModel gaps
# ---------------------------------------------------------------------------


class TestPerfModelFinalCoverage:
    @pytest.mark.parametrize(
        "cls,fwd_factory,bwd_cls,bwd_factory",
        [
            (
                perf_model.aten_mm,
                lambda: _gemm_event("aten::mm", (4, 8), (8, 16)),
                None,
                None,
            ),
            (perf_model.aten_addmm, None, None, None),
            (perf_model.aten_bmm, None, None, None),
            (perf_model.aten_baddbmm, None, None, None),
        ],
    )
    def test_gemm_backward_not_implemented(
        self, cls, fwd_factory, bwd_cls, bwd_factory
    ):
        if cls is perf_model.aten_addmm:
            event = {
                "args": {
                    "Input Dims": [(4, 16), (4, 8), (8, 16)],
                    "Input type": ["c10::BFloat16"] * 3,
                    "Input Strides": [(16, 1), (8, 1), (16, 1)],
                }
            }
        elif cls is perf_model.aten_bmm:
            event = {
                "args": {
                    "Input Dims": [[2, 4, 8], [2, 8, 16]],
                    "Input type": ["c10::BFloat16", "c10::BFloat16"],
                    "Input Strides": [(32, 8, 1), (128, 16, 1)],
                }
            }
        elif cls is perf_model.aten_baddbmm:
            event = {
                "args": {
                    "Input Dims": [[2, 4, 16], [2, 4, 8], [2, 8, 16]],
                    "Input type": ["c10::BFloat16"] * 3,
                    "Input Strides": [(64, 16, 1), (32, 8, 1), (128, 16, 1)],
                }
            }
        else:
            event = fwd_factory()
        model = cls(event)
        with pytest.raises(NotImplementedError):
            model.flops_bwd()
        with pytest.raises(NotImplementedError):
            model.bytes_bwd()

    @pytest.mark.parametrize(
        "cls,event",
        [
            (
                perf_model.aten_addmm,
                {
                    "args": {
                        "Input Dims": [(4, 16), (4, 8), (8, 16)],
                        "Input type": ["c10::BFloat16", "c10::Half", "c10::BFloat16"],
                    }
                },
            ),
            (
                perf_model.aten_bmm,
                {
                    "args": {
                        "Input Dims": [[2, 4, 8], [2, 8, 16]],
                        "Input type": ["c10::BFloat16", "c10::Half"],
                    }
                },
            ),
            (
                perf_model.aten_baddbmm,
                {
                    "args": {
                        "Input Dims": [[2, 4, 16], [2, 4, 8], [2, 8, 16]],
                        "Input type": ["c10::BFloat16", "c10::Half", "c10::BFloat16"],
                    }
                },
            ),
        ],
    )
    def test_batched_gemm_mixed_dtype_warns(self, cls, event):
        model = cls(event)
        with pytest.warns(UserWarning):
            model.bytes()

    def test_gemm_simulator_with_python_path(self, monkeypatch, tmp_path):
        sim = tmp_path / "run_gemm.py"
        sim.write_text("# stub\n")
        monkeypatch.setenv("GEMM_SIMULATOR_PATH", str(sim))
        perf_model.GEMM.cache_gemm_results.clear()
        try:
            with patch("TraceLens.PerfModel.perf_model.subprocess.run") as run:
                run.return_value = MagicMock(stdout="Time=5.5\n", stderr="")
                t, cmd = perf_model.GEMM.get_simulation_time_func(
                    _ARCH,
                    4,
                    8,
                    16,
                    None,
                    "bf16",
                    python_path="/usr/bin/python3",
                    num_cus=64,
                )
            assert t == 5.5
            assert "/usr/bin/python3" in cmd
        finally:
            perf_model.GEMM.cache_gemm_results.clear()

    def test_aten_reduce_edge_cases(self):
        empty = perf_model.aten_reduce(
            {"name": "aten::sum", "args": {"Input Dims": [None]}}
        )
        assert empty.param_details["num_input_elems"] == 0

        mean_evt = {
            "name": "aten::mean",
            "args": {
                "Input Dims": [(4, 256)],
                "Input type": ["c10::BFloat16"],
                "Output type": ["c10::BFloat16"],
                "Concrete Inputs": ["", "[1]", "True"],
            },
        }
        m = perf_model.aten_reduce(mean_evt)
        assert m.flops() > 0
        assert m.bytes() > 0

        cumsum_evt = {
            "name": "aten::cumsum",
            "args": {
                "Input Dims": [(8, 32)],
                "Input type": ["c10::Float"],
                "Concrete Inputs": ["", "[0]", "False"],
            },
        }
        c = perf_model.aten_reduce(cumsum_evt)
        assert c.param_details["num_output_elems"] == 8 * 32

    def test_conv_bias_backward_with_sequence_cache(self):
        fwd = perf_model.ConvBias_(_conv_bias_fwd_event())
        assert fwd.flops() > 0
        bwd_evt = _conv_bias_bwd_event()
        bwd_evt["args"]["Sequence number"] = bwd_evt["args"].get("Sequence number", 42)
        perf_model.ConvBias_.fwd_pass_cache[42] = fwd.param_details
        bwd = perf_model.ConvBias_Backward(bwd_evt)
        assert bwd.flops_bwd() > 0
        assert bwd.bytes_bwd() > 0
        perf_model.ConvBias_.fwd_pass_cache.pop(42, None)

    def test_conv_bias_relu_backward_paths(self):
        fwd = perf_model.ConvBiasReLU_(_conv_bias_relu_fwd_event())
        bwd_evt = _conv_bias_relu_bwd_event()
        seq = bwd_evt["args"].get("Sequence number", 43)
        perf_model.ConvBiasReLU_.fwd_pass_cache[seq] = fwd.param_details
        bwd = perf_model.ConvBiasReLU_Backward(bwd_evt)
        assert bwd.flops_bwd() > 0
        assert bwd.bytes_bwd() > 0
        perf_model.ConvBiasReLU_.fwd_pass_cache.pop(seq, None)

    def test_aten_scaled_mm_mixed_output_bpe(self):
        event = {
            "args": {
                "Input Dims": [[4, 8], [8, 16], [4, 16]],
                "Input type": [
                    "c10::Float8_e4m3fn",
                    "c10::Float8_e4m3fn",
                    "c10::Float8_e4m3fn",
                ],
            }
        }
        model = perf_model.aten_scaled_mm(event)
        assert model.bytes() > 0

    def test_primus_grouped_gemm_variable_k(self):
        event = {
            "name": "primus_turbo::grouped_gemm_variable_k_impl",
            "args": {
                "Input Dims": [[24576, 1408], [24576, 2048]],
                "Input type": ["c10::BFloat16", "c10::BFloat16"],
            },
        }
        g = perf_model.primus_turbo_grouped_gemm_variable_k(event)
        assert g.flops() > 0
        assert g.bytes() > 0


# ---------------------------------------------------------------------------
# Extension modules — bytes / precision paths
# ---------------------------------------------------------------------------


class TestMoeExtensionsFinal:
    MOE_BLOCKSCALE = {
        "args": {
            "Input Dims": [
                [32, 4096],
                [32, 4096],
                [8, 14336, 4096],
                [8, 4096, 7168],
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

    CK_STAGE1 = {
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

    CK_STAGE2 = {
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

    @pytest.mark.parametrize(
        "factory,event",
        [
            (moe_ext.moe_aiter_fused_blockscale, "MOE_BLOCKSCALE"),
            (moe_ext.moe_aiter_ck_stage1, "CK_STAGE1"),
            (moe_ext.moe_aiter_ck_stage2, "CK_STAGE2"),
        ],
    )
    def test_moe_bytes_and_precision(self, factory, event):
        event_obj = getattr(self, event) if isinstance(event, str) else event
        if event == "MOE_STD":
            event_obj = {
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
        elif event == "GROUPED":
            event_obj = {
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
                    "Input type": [
                        "c10::BFloat16",
                        "c10::Float8_e4m3fn",
                        "",
                        "c10::BFloat16",
                    ],
                }
            }
        elif event == "GPTQ":
            event_obj = {
                "args": {
                    "Input Dims": [[32, 4096], [32, 2], [8, 7168, 4096]],
                    "Input type": ["c10::BFloat16", "c10::Int", "c10::BFloat16"],
                    "MoE topk": 2,
                }
            }
        model = factory(event_obj)
        b = model.bytes()
        assert b is None or b > 0
        prec = model.get_compute_precision()
        assert prec in (None, "bf16", "fp8", "fp4", "fp16", "fp32")
        if hasattr(model, "flops_bwd"):
            with pytest.raises(NotImplementedError):
                model.flops_bwd()

    def test_moe_triton_unfused_and_sglang(self):
        from tests.test_perfmodel_coverage import _moe_unfused_event

        up = moe_ext.moe_triton_unfused_up(
            _moe_unfused_event(kernel_name="moe_mxfp4_up_kernel")
        )
        down = moe_ext.moe_triton_unfused_down(
            _moe_unfused_event(kernel_name="moe_fp8_down_kernel")
        )
        assert up.bytes() > 0
        assert down.bytes() > 0
        sgl = moe_ext.sglang_fused_append_shared_experts(
            {
                "args": {
                    "Input Dims": [(32, 4096), (32, 4096), (32, 4096)],
                    "Input type": ["c10::BFloat16"] * 3,
                }
            }
        )
        assert sgl.bytes() > 0

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
        assert moe_ext.moe_flydsl_stage1(fly).bytes() > 0
        assert moe_ext.moe_flydsl_stage2(fly).bytes() > 0
        gptq = {
            "args": {
                "Input Dims": [[32, 4096], [32, 2], [8, 7168, 4096]],
                "Input type": ["c10::BFloat16", "c10::Int", "c10::BFloat16"],
                "MoE topk": 2,
            }
        }
        assert moe_ext.moe_gptq_awq_up(gptq).bytes() > 0
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
                "Input type": [
                    "c10::BFloat16",
                    "c10::Float8_e4m3fn",
                    "",
                    "c10::BFloat16",
                ],
            }
        }
        assert moe_ext.moe_triton_invoke_grouped_gemm(grouped).bytes() > 0

    def test_biased_topk_and_sort_scatter_precision(self):
        topk = moe_ext.BiasedGroupedTopk(
            {
                "args": {
                    "Input Dims": [(32, 256), (256,), (32, 8), (32, 8)],
                    "Input type": ["c10::Float"] * 3 + ["c10::Int"],
                }
            }
        )
        assert topk.flops() > 0
        sort = moe_ext.MoeSortScatterGather(
            {
                "args": {
                    "Input Dims": [(32, 4096), (32, 2), (32, 4096)],
                    "Input type": ["c10::BFloat16", "c10::Int", "c10::BFloat16"],
                }
            }
        )
        assert sort.bytes() > 0


class TestPerfExtensionsFinal:
    def test_mhc_and_sampling_bytes(self):
        fused = pext.mhc_fused_post_pre_gemm_sqrsum(
            {
                "args": {
                    "Input Dims": [
                        (2, 4, 8),
                        (2, 4),
                        (4, 2, 128),
                        (4, 128),
                        (4, 2, 128),
                        (4, 2, 1),
                        (4, 2, 2),
                        (8, 256),
                    ],
                    "Input type": ["float"] * 8,
                }
            }
        )
        assert fused.bytes() > 0
        assert fused.get_maf_type() == "matrix"

        topk = pext.topk_softplus(
            {
                "args": {
                    "Input Dims": [(4, 2), (4, 2), (4, 8)],
                    "Input type": ["c10::Float", "c10::Int", "c10::BFloat16"],
                }
            }
        )
        assert topk.bytes() > 0

        sample = pext.mixed_sample_outer_exponential(
            {
                "args": {
                    "Input Dims": [(), (4, 32000), (4, 32000)],
                    "Input type": ["Scalar", "float", "float"],
                }
            }
        )
        assert sample.flops() > 0
        assert sample.get_maf_type() == "vector"

    def test_fused_qk_rope_and_batched_gemm_bytes(self):
        rope = pext.fused_qk_rope_concat_and_cache_mla(
            {
                "args": {
                    "Input Dims": [
                        (2, 8, 512),
                        (2, 8, 64),
                        (2, 1, 512),
                        (2, 1, 64),
                        (128, 1, 1, 576),
                    ],
                    "Input type": ["c10::BFloat16"] * 4 + ["c10::Float8_e4m3fn"],
                }
            }
        )
        assert rope.bytes() > 0

        fp4 = pext.batched_gemm_a16wfp4(
            {
                "args": {
                    "Input Dims": [[2, 4, 128], [2, 256, 64], [2, 256, 4]],
                    "Input type": ["c10::BFloat16", "unsigned char", "c10::Float"],
                }
            }
        )
        assert fp4.bytes() > 0

        post = pext.mhc_post(
            {
                "args": {
                    "Input Dims": [(4, 2, 128), (4, 128)],
                    "Input type": ["c10::BFloat16", "c10::BFloat16"],
                }
            }
        )
        assert post.bytes() > 0

        pre = pext.mhc_pre_gemm_sqrsum(
            {
                "args": {
                    "Input Dims": [(2, 4, 8), (2, 4), (4, 2, 128), (8, 256)],
                    "Input type": ["float", "float", "c10::BFloat16", "float"],
                }
            }
        )
        assert pre.bytes() > 0

        rope2 = pext.aiter_rope_cached_positions_2c_fwd_impl(
            {
                "args": {
                    "Input Dims": [
                        (2, 128, 8, 64),
                        (2, 128, 1, 64),
                        (2, 128, 8, 64),
                        (2, 128, 1, 64),
                        (2048, 1, 1, 64),
                        (2048, 1, 1, 64),
                        (2, 128),
                    ],
                    "Input type": ["c10::BFloat16"] * 7,
                }
            }
        )
        assert rope2.bytes() > 0


# ---------------------------------------------------------------------------
# TreePerf gaps
# ---------------------------------------------------------------------------


class TestTreePerfFinalCoverage:
    def _nn_module_trace(self):
        corr1, corr2 = 200, 201
        return [
            _make_gpu_event(
                "py_root", 0, 500, "python_function", "nn.Module: Block_0", pid=100
            ),
            _make_gpu_event(
                "py_child",
                10,
                400,
                "python_function",
                "nn.Module: Block_0.linear",
                pid=100,
            ),
            _make_gpu_event(
                "cpu1",
                20,
                80,
                "cpu_op",
                "aten::mm",
                pid=100,
                args={
                    "Input Dims": [[32, 64], [64, 128]],
                    "Input type": ["c10::BFloat16", "c10::BFloat16"],
                    "Input Strides": [[128, 1], [128, 1]],
                },
            ),
            _make_gpu_event(
                "rt1",
                25,
                5,
                "cuda_runtime",
                "hipLaunchKernel",
                pid=100,
                args={"correlation": corr1},
            ),
            _make_gpu_event(
                "k1",
                50,
                40,
                "kernel",
                "Cijk_gemm",
                pid=0,
                tid=7,
                args={"correlation": corr1, "stream": 7},
            ),
            _mk_ac2g(corr1, 0, 7, 50, "s"),
            _mk_ac2g(corr1, 0, 7, 90, "f"),
            _make_gpu_event(
                "cpu2",
                120,
                80,
                "cpu_op",
                "aten::add",
                pid=100,
                args={
                    "Input Dims": [[32, 128], [32, 128]],
                    "Input type": ["c10::BFloat16"] * 2,
                },
            ),
            _make_gpu_event(
                "rt2",
                125,
                5,
                "cuda_runtime",
                "hipLaunchKernel",
                pid=100,
                args={"correlation": corr2},
            ),
            _make_gpu_event(
                "k2",
                150,
                20,
                "kernel",
                "vectorized_elementwise_kernel",
                pid=0,
                tid=7,
                args={"correlation": corr2, "stream": 7},
            ),
            _mk_ac2g(corr2, 0, 7, 150, "s"),
            _mk_ac2g(corr2, 0, 7, 170, "f"),
        ]

    def test_kernel_launchers_extended_columns(self):
        analyzer = _build_analyzer(self._nn_module_trace(), add_python_func=True)
        df = analyzer.get_df_kernel_launchers(
            include_args=True,
            include_kernel_details=True,
            include_call_stack=True,
            id_cols=True,
            include_first_occurrence_time=True,
        )
        assert not df.empty
        assert "parent_module" in df.columns
        assert "call_stack" in df.columns

    def test_kernel_launchers_summary_by_shape(self):
        analyzer = _build_analyzer(_mk_pytorch_trace())
        launchers = analyzer.get_df_kernel_launchers(include_args=True)
        summary = TreePerfAnalyzer.get_df_kernel_launchers_summary_by_shape(
            launchers, "aten::mm"
        )
        assert not summary.empty
        assert "Total Kernel Time (µs)" in summary.columns

    def test_summarize_kernel_stats_and_unified_table(self):
        stats = TreePerfAnalyzer._summarize_kernel_stats(
            [
                [{"name": "a", "dur": 10}, {"name": "b", "dur": 20}],
                [{"name": "a", "dur": 12}, {"name": "b", "dur": 18}],
            ],
            agg_metrics=["mean", "median", "max", "min", "std"],
        )
        assert len(stats) == 2
        with pytest.warns(UserWarning):
            TreePerfAnalyzer._summarize_kernel_stats([[{"name": "a", "dur": 1}], []])

        analyzer = _build_analyzer(_mk_pytorch_trace())
        unified = analyzer.build_df_unified_perf_table()
        summarized = TreePerfAnalyzer.summarize_df_unified_perf_table(
            unified, include_pct=True, tree=analyzer.tree
        )
        assert isinstance(summarized, pd.DataFrame)

    def test_build_df_perf_metrics_unknown_op(self):
        events = _mk_pytorch_trace()
        events[0]["name"] = "aten::unknown_custom_op"
        analyzer = _build_analyzer(events)
        df = analyzer.build_df_perf_metrics(events=[analyzer.tree.events[0]])
        assert isinstance(df, pd.DataFrame)

    def test_jax_parse_gemm_metadata_and_operands(self):
        gemm = {
            "gpu_kernel_op_cat": "GEMM",
            "metadata": {
                "custom_call_target": "cublasLt_matmul",
                "operands": ["bf16[4,8]{1,0}", "bf16[8,16]{0,1}"],
                "output": "bf16[4,16]{1,0}",
                "backend_config": 'foo={"gemm_backend_config":{"beta":0}}',
                "computation": "gemm",
            },
        }
        meta = JaxTreePerfAnalyzer.parse_gemm_metadata(gemm)
        assert meta["Beta"] == 0
        assert len(meta["Input Dims"]) == 2
        dims, _, _ = JaxTreePerfAnalyzer.parse_operands(gemm)
        assert dims == ((4, 8), (8, 16))

    def test_summarize_df_perf_metrics_origami_cols(self):
        analyzer = _build_analyzer(_mk_pytorch_trace())
        df_raw = analyzer.build_df_perf_metrics(
            events=[e for e in analyzer.tree.events if e.get("cat") == "cpu_op"]
        )
        df_raw["Origami Time (µs)"] = [10.0]
        df_raw["Origami TFLOPS/s"] = [1.0]
        df_raw["Origami TB/s"] = [0.5]
        df_raw["Pct Origami"] = [50.0]
        df_raw["Non-Data-Mov TFLOPS/s"] = [0.8]
        df_raw["Non-Data-Mov Kernel Time (µs)"] = [5.0]
        summary = analyzer.summarize_df_perf_metrics(
            df_raw, agg_metrics=["mean", "std"]
        )
        assert isinstance(summary, pd.DataFrame)

    def test_collect_unified_perf_events_with_python_stack(self):
        analyzer = _build_analyzer(self._nn_module_trace(), add_python_func=True)
        events = analyzer.collect_unified_perf_events()
        assert isinstance(events, list)

    def test_build_df_bwd_linked_metrics(self):
        corr_fwd, corr_bwd = 800, 801
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
        fwd = next(e for e in analyzer.tree.events if e["name"] == "aten::mm")
        bwd = next(e for e in analyzer.tree.events if e["name"] == "aten::mm_backward")
        fwd["bwd_events"] = [bwd["UID"]]
        bwd["fwd_event"] = fwd["UID"]
        df = analyzer.build_df_unified_perf_table(events=[fwd, bwd])
        assert len(df) >= 1

    def test_build_nn_module_latency_tree_cpu(self):
        analyzer = _build_analyzer(self._nn_module_trace(), add_python_func=True)
        root = next(
            e for e in analyzer.tree.events if e["name"] == "nn.Module: Block_0"
        )
        analyzer.build_nn_module_latency_tree(root)
        assert "GPU Time" in root
        assert root["GPU Time"] > 0


# ---------------------------------------------------------------------------
# Orchestrator prepare — deeper branch coverage
# ---------------------------------------------------------------------------


class TestOrchestratorPrepareFinal:
    def test_standalone_sibling_sequence_and_duplicate_base(self, tmp_path):
        k1 = _kernel_event(10, "Cijk_gemm_a", dur=500)
        k2 = _kernel_event(11, "vectorized_elementwise_kernel add", dur=300)
        k3 = _kernel_event(12, "Cijk_gemm_b", dur=400)
        parent = {
            "name": "aten::linear",
            "_category": "aten",
            "gpu_events": [12],
            "parent": None,
            "args": {"Input Dims": "[[2,3]]"},
        }
        mod1 = {
            "name": "nn.Module: MLP_0",
            "_category": "aten",
            "gpu_events": [10, 11],
            "args": {"Input Dims": "[[2,3]]"},
        }
        mod2 = {
            "name": "nn.Module: MLP_1",
            "_category": "aten",
            "gpu_events": [10, 11],
        }
        child_op = {
            "name": "aten::add",
            "_category": "aten",
            "gpu_events": [12],
            "parent": 99,
        }
        uid_map = {10: k1, 11: k2, 12: k3, 99: parent}
        tree = _StubTree(
            [mod1, mod2, parent, child_op], uid_map, parent_map={id(child_op): parent}
        )
        unified = [
            {"name": "aten::mm", "gpu_events": [10], "parent": 99},
            {"name": "aten::relu", "gpu_events": [11], "parent": 99},
        ]
        analyzer = _StubAnalyzer(tree, unified_events=unified)

        csv_dir = tmp_path / "csv"
        csv_dir.mkdir()
        pd.DataFrame(
            {
                "kernel_details_summary": [
                    "[{'name': 'Cijk_gemm_a'}]",
                    "[{'name': 'vectorized_elementwise_kernel add'}]",
                    "[{'name': 'Cijk_gemm_b'}]",
                ],
                "op category": ["GEMM", "elementwise", "GEMM"],
                "Data Moved (MB)": [10.0, 4.0, 8.0],
                "perf_params": ["{}", "{}", "{}"],
                "Input Dims": ["[[2,3]]", "[[4,4]]", "[[2,3]]"],
            }
        ).to_csv(csv_dir / "unified_perf_summary.csv", index=False)

        cands = _extract_standalone_fusion_candidates(analyzer, tree, str(csv_dir))
        assert isinstance(cands, list)

    def test_comparative_duplicate_base_accumulation(self, tmp_path, capsys):
        csv_dir = tmp_path / "trace1_csvs"
        csv_dir.mkdir()
        pd.DataFrame(
            {
                "name": ["Cijk_A", "Cijk_B"],
                "source": ["trace1", "trace1"],
                "lowest_common_ancestor_id": [100, 100],
                "kernel_time": [5000.0, 3000.0],
                "gpu_op_uid": [10, 11],
            }
        ).to_csv(csv_dir / "diff_stats.csv", index=False)

        uid_map = {
            10: {
                "name": "Cijk_A",
                "dur": 5000,
                "_category": "kernel",
                "gpu_events": [],
            },
            11: {
                "name": "Cijk_B",
                "dur": 3000,
                "_category": "kernel",
                "gpu_events": [],
            },
        }
        mod_a = {
            "name": "nn.Module: Attn_0",
            "_category": "aten",
            "gpu_events": [10, 11],
        }
        mod_b = {
            "name": "nn.Module: Attn_1",
            "_category": "aten",
            "gpu_events": [10, 11],
        }
        tree = _StubTree([mod_a, mod_b], uid_map)
        analyzer = _StubAnalyzer(tree)
        cands = _extract_comparative_fusion_candidates(str(csv_dir), analyzer, tree)
        assert isinstance(cands, list)


# ---------------------------------------------------------------------------
# Reporting — inference / pytorch paths
# ---------------------------------------------------------------------------


class TestReportingFinalCoverage:
    def test_inference_graphlaunch_warning(self, tmp_path):
        specs = [("aten::mm", "gemm_kernel", 100)]
        trace = _write_trace(tmp_path, specs)
        data = json.loads(open(trace).read())
        data["traceEvents"].append(
            _mk_event(
                "cuda_runtime",
                "hipGraphLaunch",
                2000,
                10,
                100,
                100,
                {"correlation": 999},
            )
        )
        path = tmp_path / "graph_trace.json"
        path.write_text(json.dumps(data))
        with pytest.warns(UserWarning, match="hipgraph launches"):
            generate_inference_report(
                profile_json_path=str(path),
                output_csvs_dir=str(tmp_path / "out"),
                output_xlsx_path=str(tmp_path / "r.xlsx"),
                collective_analysis=False,
            )

    def test_inference_bwd_and_overlap_sheets(self, tmp_path):
        corr_fwd, corr_bwd = 300, 301
        events = [
            _mk_event(
                "cpu_op",
                "aten::mm",
                1000,
                100,
                100,
                100,
                {
                    "Input Dims": [[32, 64], [64, 128]],
                    "Input type": ["float", "float"],
                },
            ),
            _mk_event(
                "cuda_runtime",
                "hipLaunchKernel",
                1010,
                5,
                100,
                100,
                {"correlation": corr_fwd},
            ),
            _mk_event(
                "kernel",
                "gemm_fwd",
                1050,
                80,
                0,
                7,
                {"correlation": corr_fwd, "stream": 7},
            ),
            _mk_ac2g(corr_fwd, 0, 7, 1050, "s"),
            _mk_ac2g(corr_fwd, 0, 7, 1130, "f"),
            _mk_event(
                "cpu_op",
                "aten::mm_backward",
                2000,
                100,
                100,
                100,
                {
                    "Input Dims": [[32, 64], [64, 128]],
                    "Input type": ["float", "float"],
                },
            ),
            _mk_event(
                "cuda_runtime",
                "hipLaunchKernel",
                2010,
                5,
                100,
                100,
                {"correlation": corr_bwd},
            ),
            _mk_event(
                "kernel",
                "gemm_bwd",
                2050,
                60,
                0,
                7,
                {"correlation": corr_bwd, "stream": 7},
            ),
            _mk_ac2g(corr_bwd, 0, 7, 2050, "s"),
            _mk_ac2g(corr_bwd, 0, 7, 2110, "f"),
        ]
        path = tmp_path / "bwd_trace.json"
        path.write_text(json.dumps({"traceEvents": events}))
        result = generate_inference_report(
            profile_json_path=str(path),
            output_csvs_dir=str(tmp_path / "bwd_out"),
            output_xlsx_path=str(tmp_path / "bwd.xlsx"),
            include_overlap_info=True,
            kernel_summary=True,
            short_kernel_study=True,
            group_by_parent_module=False,
        )
        assert "gpu_timeline" in result

    def test_sanity_check_kernel_details_summary_column(self):
        events = [{"name": "k_a", "cat": "kernel"}]
        tl = pd.DataFrame({"type": ["computation_time"], "time ms": [0.1]})
        kl = pd.DataFrame(
            {
                "total_direct_kernel_time": [100.0],
                "kernel_details_summary": [[{"name": "k_a", "count": 1}]],
            }
        )
        up = pd.DataFrame(
            {"Kernel Time (µs)": [100.0], "kernel_details_summary": [[{"name": "k_a"}]]}
        )
        result = perf_report_sanity_check(events, tl, kl, up)
        assert result["kl_count_pass"]

    def test_pytorch_report_with_comparison(self, tmp_path):
        trace1 = _write_trace(tmp_path, [("aten::mm", "gemm_kernel", 100)], "t1.json")
        trace2 = _write_trace(tmp_path, [("aten::mm", "gemm_kernel", 120)], "t2.json")
        generate_perf_report_pytorch(
            profile_json_path=trace1,
            comparison_json_path=trace2,
            output_csvs_dir=str(tmp_path / "cmp_out"),
            output_xlsx_path=str(tmp_path / "cmp.xlsx"),
            kernel_summary=True,
            short_kernel_study=True,
        )
        assert (tmp_path / "cmp_out" / "gpu_timeline.csv").exists()

    def test_add_truncated_kernel_details_inference(self):
        df = pd.DataFrame({"kernel_details": [[{"name": "x" * 200, "dur": 1}]]})
        out = add_truncated_inference(df, "kernel_details")
        assert "trunc_kernel_details" in out.columns


# ---------------------------------------------------------------------------
# Capture merge experimental
# ---------------------------------------------------------------------------


class TestCaptureMergeFinal:
    def test_multistream_align_and_verify(self):
        graph = [
            {"name": "k1", "args": {"stream": 1}},
            {"name": "k2", "args": {"stream": 2}},
            {"name": "k1", "args": {"stream": 1}},
        ]
        capture = [
            {"name": "hipLaunchKernel", "args": {"kernel": "k1"}},
            {"name": "hipLaunchKernel", "args": {"kernel": "k2"}},
            {"name": "hipLaunchKernel", "args": {"kernel": "k1"}},
        ]
        assert is_multistream(graph)
        assert capture_has_kernel_names(capture)
        aligned = align_streams(graph, capture)
        assert aligned is not None
        assert len(aligned) == 3

    def test_verify_subtree_greedy_alignment(self):
        capture = [
            {"name": "hipLaunchKernel", "args": {"kernel": "extra"}},
            {"name": "hipLaunchKernel", "args": {"kernel": "k1"}},
            {"name": "hipLaunchKernel", "args": {"kernel": "k2"}},
        ]
        graph = [
            {"name": "k1", "args": {}},
            {"name": "k2", "args": {}},
        ]
        code, cap, gr = verify_subtree_events(capture, graph)
        assert code == 2
        assert len(cap) == 2

    def test_get_subtree_events_filters(self):
        tree = TraceToTree(
            [
                {
                    "ph": "X",
                    "name": "root",
                    "ts": 0,
                    "dur": 100,
                    "cat": "cpu_op",
                    "args": {},
                },
                {
                    "ph": "X",
                    "name": "hipLaunchKernel",
                    "ts": 10,
                    "dur": 5,
                    "cat": "cuda_runtime",
                    "args": {},
                },
            ]
        )
        tree.build_tree()
        root = tree.events[0]
        all_ev, filt = get_subtree_events(
            tree, root, cat_filter=["cuda_runtime"], name_filter=["Launch"]
        )
        assert len(all_ev) >= 1
        assert len(filt) >= 1

    def test_capture_tree_cache(self, tmp_path):
        from TraceLens.Trace2Tree import trace_capture_merge_experimental as tcm

        tcm._capture_tree_cache.clear()
        events = {
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
                    "args": {"kernel": "k1"},
                },
            ]
        }
        trace_path = tmp_path / "cap.json"
        trace_path.write_text(json.dumps(events))
        key = ("test_key", str(trace_path))
        r1 = _get_cached_capture_tree(key, str(trace_path))
        r2 = _get_cached_capture_tree(key, str(trace_path))
        assert r1[0] is r2[0]
        for i in range(10):
            p = tmp_path / f"cap{i}.json"
            p.write_text(json.dumps(events))
            _get_cached_capture_tree((f"k{i}", str(p)), str(p))
        assert len(tcm._capture_tree_cache) <= tcm._CAPTURE_TREE_CACHE_MAX_SIZE


# ---------------------------------------------------------------------------
# Additional perf-model / reporting / orchestrator depth
# ---------------------------------------------------------------------------


class TestPerfModelDeepCoverage2:
    def test_aten_conv_bwd_with_bias_grad(self):
        event = {
            "args": {
                "Input Dims": [
                    [2, 4, 6, 6],
                    [2, 3, 8, 8],
                    [4, 3, 3, 3],
                    [4],
                ],
                "Input type": ["c10::BFloat16"] * 4,
                "Concrete Inputs": [
                    "",
                    "",
                    "",
                    "",
                    "(1,1)",
                    "(0,0)",
                    "(1,1)",
                    "False",
                    "(0,0)",
                    "1",
                    "[True, True, False]",
                ],
            }
        }
        model = perf_model.aten_conv_bwd(event)
        assert model.flops_bwd() > 0

    def test_extract_sdpa_configs(self):
        cfg = perf_model.extract_sdpa_cfg(
            q_shape=[2, 8, 128, 64],
            k_shape=[2, 8, 128, 64],
            v_shape=[2, 8, 128, 64],
            bhnd_idx=(0, 1, 2, 3),
        )
        assert cfg["B"] == 2
        vcfg = perf_model.extract_sdpa_varlen_cfg(
            q_shape=[8, 128, 64],
            k_shape=[8, 128, 64],
            v_shape=[8, 128, 64],
            hnd_idx=(0, 1, 2),
        )
        assert vcfg["B"] == 1

    def test_grouped_gemm_zipped_and_impl_formats(self):
        zipped = {
            "name": "primus_turbo::grouped_gemm",
            "args": {
                "Input Dims": [[[4, 8], [5, 8]], [[8, 16], [8, 16]]],
                "Input type": ["c10::BFloat16", "c10::BFloat16"],
            },
        }
        g = perf_model.primus_turbo_grouped_gemm(zipped)
        assert g.flops() > 0
        assert g.get_compute_precision() == "bf16"

    def test_gemm_simulator_default_batch_and_invalid_path(self, monkeypatch, tmp_path):
        monkeypatch.setenv("GEMM_SIMULATOR_PATH", str(tmp_path / "missing.py"))
        with pytest.raises(ValueError, match="does not exist"):
            perf_model.GEMM.get_simulation_time_func(_ARCH, 4, 8, 16, 1, "bf16")

    def test_fused_rope_and_cross_entropy_precision(self):
        rope = perf_model.fused_rope_fwd(
            {
                "args": {
                    "Input Dims": [[128, 2, 8, 64], [128, 1, 1, 64]],
                    "Input type": ["c10::BFloat16", "c10::BFloat16"],
                }
            }
        )
        assert rope.get_compute_precision() == "bf16"
        ce = perf_model.cross_entropy_fwd(
            {
                "args": {
                    "Input Dims": [[4, 1, 32000], [4, 1]],
                    "Input type": ["c10::BFloat16", "long int"],
                }
            }
        )
        assert ce.get_compute_precision() is not None


class TestOrchestratorMainExtended:
    def test_orchestrator_disable_pseudo_ops(self, tmp_path, monkeypatch):
        from TraceLens.Agent.Analysis.utils import orchestrator_prepare as op

        out = str(tmp_path)
        _write_minimal_orchestrator_csvs(out, comparative=False)
        k1 = _kernel_event(10, "Cijk_a")
        k2 = _kernel_event(11, "ew_add")
        module = {
            "name": "nn.Module: MLP_0",
            "_category": "aten",
            "gpu_events": [10, 11],
        }
        tree = _StubTree([module], {10: k1, 11: k2})
        analyzer = _StubAnalyzer(tree)

        class _FakeTreePerfAnalyzer:
            @classmethod
            def from_file(cls, *args, **kwargs):
                return analyzer

        monkeypatch.setattr(op, "TreePerfAnalyzer", _FakeTreePerfAnalyzer)
        monkeypatch.setattr(
            op, "_extract_standalone_fusion_candidates", lambda *a, **k: []
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
            "--disable_pseudo_ops",
        ]
        try:
            op.main()
        finally:
            sys.argv = old_argv
        manifest = json.loads(
            open(os.path.join(out, "category_data", "category_manifest.json")).read()
        )
        assert manifest["comparison_scope"] == "standalone"


class TestReportingExtended:
    def test_inference_report_topk_and_roofline(self, tmp_path):
        trace = _write_trace(
            tmp_path,
            [
                ("aten::mm", "gemm_kernel", 100),
                ("aten::add", "add_kernel", 15),
            ],
        )
        result = generate_inference_report(
            profile_json_path=trace,
            output_csvs_dir=str(tmp_path / "topk_out"),
            output_xlsx_path=str(tmp_path / "topk.xlsx"),
            topk_ops=5,
            topk_roofline_ops=3,
            topk_short_kernels=2,
            short_kernel_threshold_us=50,
            include_unlinked_kernels=True,
            include_call_stack=True,
        )
        assert "gpu_timeline" in result

    def test_pytorch_report_extension_and_arch(self, tmp_path):
        trace = _write_trace(tmp_path, [("aten::mm", "gemm_kernel", 100)])
        ext = tmp_path / "ext.py"
        ext.write_text(
            "def apply_extension(analyzer, path):\n"
            "    analyzer.tree.events[0]['ext'] = True\n"
        )
        generate_perf_report_pytorch(
            profile_json_path=trace,
            output_csvs_dir=str(tmp_path / "ext_out"),
            output_xlsx_path=str(tmp_path / "ext.xlsx"),
            extension_file=str(ext),
            gpu_arch={
                "name": "mi300x",
                "freq_mhz": 2200,
                "num_cus": 304,
                "gemm_units_per_cu": 4,
                "mem_bw_gbps": 5300,
                "l1_bw_gbps": 100,
            },
            include_call_stack=True,
        )
        assert (tmp_path / "ext_out" / "gpu_timeline.csv").exists()
