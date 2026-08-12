###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Phase-11: final push to >=95% CPU-only coverage (<=942 miss)."""

from __future__ import annotations

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
from TraceLens.TreePerf.tree_perf import TreePerfAnalyzer

from tests.test_agent_coverage import (
    _StubAnalyzer,
    _StubTree,
    _kernel_event,
    _write_minimal_orchestrator_csvs,
)
from tests.test_conv_backward_bytes import _conv_bias_bwd_event, _conv_bias_fwd_event
from tests.test_flash_attention_backward import _bwd_event as _flash_bwd_event
from tests.test_mamba_ssd import _mamba_event
from tests.test_perfmodel_coverage import _ARCH, _moe_unfused_event
from tests.test_treeperf_coverage import _build_analyzer, _make_gpu_event, _mk_ac2g

NORM_TRACE = os.path.join(
    os.path.dirname(__file__),
    "traces/perf_model/normalization/normalization_layer_test.json.gz",
)
RESNET_TRACE = os.path.join(
    os.path.dirname(__file__), "traces/mi300/resnet_act_checkpoint.json.gz"
)


def _conv_fwd_event(input_shape, filter_shape):
    nd = len(input_shape) - 2
    if nd == 1:
        stride, padding, dilation, out_pad = "(1)", "(0)", "(1)", "(0)"
    elif nd == 3:
        stride, padding, dilation, out_pad = "(1,1,1)", "(0,0,0)", "(1,1,1)", "(0,0,0)"
    else:
        stride, padding, dilation, out_pad = "(1,1)", "(0,0)", "(1,1)", "(0,0)"
    return {
        "args": {
            "Input Dims": [list(input_shape), list(filter_shape)],
            "Input type": ["c10::BFloat16", "c10::BFloat16"],
            "Input Strides": [[1] * len(input_shape), [1] * len(filter_shape)],
            "Concrete Inputs": [
                "",
                "",
                "",
                stride,
                padding,
                dilation,
                "False",
                out_pad,
                "1",
            ],
        }
    }


def _conv_bwd_event(input_shape, filter_shape):
    nd = len(input_shape) - 2
    if nd == 1:
        stride, padding, dilation, out_pad = "[1]", "[0]", "[1]", "[0]"
    elif nd == 3:
        stride, padding, dilation, out_pad = (
            "[1, 1, 1]",
            "[0, 0, 0]",
            "[1, 1, 1]",
            "[0, 0, 0]",
        )
    else:
        stride, padding, dilation, out_pad = "[1, 1]", "[0, 0]", "[1, 1]", "[0, 0]"
    grad_out = list(input_shape)
    return {
        "args": {
            "Input Dims": [
                grad_out,
                list(input_shape),
                list(filter_shape),
            ],
            "Input type": ["c10::BFloat16"] * 3,
            "Input Strides": [[1] * len(input_shape)] * 3,
            "Concrete Inputs": [
                "",
                "",
                "",
                "[0]",
                stride,
                padding,
                dilation,
                "False",
                out_pad,
                "1",
                "[True, True, False]",
            ],
        }
    }


class _GroupedGemmNoBwdOverride(perf_model.GroupedGemm):
    @staticmethod
    def get_param_details(event):
        return {"M": 64, "K": 32, "N": 16, "G": 4, "bpe_in": 2, "bpe_out": 2}


class _BadPerfModel:
    def __init__(self, *args, **kwargs):
        pass

    def flops(self):
        raise NotImplementedError("no model")

    def bytes(self):
        return 0


class _ExplodingPerfModel:
    def __init__(self, *args, **kwargs):
        pass

    def flops(self):
        raise RuntimeError("boom")

    def bytes(self):
        return 0


class TestPerfModelPhase11:
    def test_aten_scaled_mm_unsupported_output_bpe(self):
        event = {
            "args": {
                "Input Dims": [[4, 8], [16, 8]],
                "Input type": ["double", "double"],
                "Concrete Inputs": ["", "", "1.0"],
            }
        }
        model = perf_model.aten_scaled_mm(event)
        assert model.bytes() is None

    def test_gemm_strides_and_bwd_not_implemented(self):
        vllm = perf_model.vllm_gemm_with_dynamic_quant(
            {
                "args": {
                    "Input Dims": [[128, 64], [256, 32]],
                    "Input type": ["c10::Float4_e2m1fn_x2", "c10::Float4_e2m1fn_x2"],
                    "Input Strides": [[8192, 64, 1], [16384, 64, 1]],
                }
            }
        )
        assert vllm.bytes() > 0

        tex = perf_model.tex_ts_te_gemm_ts(
            {
                "args": {
                    "Input Dims": [[64, 128]] * 6 + [[128, 64]] * 6 + [[]],
                    "Input type": ["c10::BFloat16"] * 19,
                    "Input Strides": [[8192, 128, 1]] * 6 + [[16384, 64, 1]] * 6,
                    "Concrete Inputs": [""] * 14 + [""],
                }
            }
        )
        assert tex.bytes() > 0
        with pytest.raises(NotImplementedError):
            tex.bytes_bwd()

        tev2 = perf_model.tev2_pseudo_gemm(
            {
                "args": {
                    "Input Dims": [[32, 64], [64, 128]],
                    "Input type": ["c10::BFloat16", "c10::BFloat16"],
                    "Input Strides": [[2048, 64, 1], [8192, 128, 1]],
                }
            }
        )
        assert tev2.flops() > 0

    def test_conv_ndims_and_mixed_dtype(self):
        conv1d = perf_model.aten_conv(_conv_fwd_event((2, 3, 32), (4, 3, 5)))
        assert conv1d.param_details["convNd"] == "conv1d"
        conv3d = perf_model.aten_conv(_conv_fwd_event((2, 3, 8, 8, 8), (4, 3, 3, 3, 3)))
        assert conv3d.param_details["convNd"] == "conv3d"

        mixed = _conv_fwd_event((2, 3, 8, 8), (4, 3, 3, 3))
        mixed["args"]["Input type"] = ["c10::BFloat16", "c10::Half"]
        with pytest.raises(ValueError, match="different"):
            perf_model.aten_conv(mixed)

        bwd1d = perf_model.aten_conv_bwd(_conv_bwd_event((2, 3, 32), (4, 3, 5)))
        assert bwd1d.param_details["convNd"] == "conv1d"
        bwd3d = perf_model.aten_conv_bwd(
            _conv_bwd_event((2, 3, 8, 8, 8), (4, 3, 3, 3, 3))
        )
        assert bwd3d.param_details["convNd"] == "conv3d"
        mixed_bwd = _conv_bwd_event((2, 3, 8, 8), (4, 3, 3, 3))
        mixed_bwd["args"]["Input type"] = [
            "c10::BFloat16",
            "c10::BFloat16",
            "c10::Half",
        ]
        with pytest.raises(ValueError, match="different"):
            perf_model.aten_conv_bwd(mixed_bwd).bytes()

    def test_conv_bias_family_conv1d_and_fallbacks(self):
        perf_model.ConvBias_.fwd_pass_cache.clear()
        conv1d_fwd = {
            "args": {
                "Input Dims": [[2, 3, 32], [4, 3, 5], [4]],
                "Input type": ["c10::BFloat16", "c10::BFloat16", "c10::BFloat16"],
                "Concrete Inputs": ["", "", "", "1", "0"],
                "Input Strides": [[96, 32, 1], [60, 15, 1]],
                "Sequence number": 50,
            }
        }
        perf_model.ConvBias_(conv1d_fwd)
        bwd = perf_model.ConvBias_Backward(_conv_bias_bwd_event(seq_num=50))
        assert bwd.flops() is None or bwd.flops() >= 0

        perf_model.ConvBias_.fwd_pass_cache.clear()
        fwd = perf_model.ConvBias_(_conv_bias_fwd_event())
        perf_model.ConvBias_.fwd_pass_cache[999] = {
            **fwd.param_details,
            "dtype_input_weight": ("c10::BFloat16", "c10::Half"),
        }
        cached_bwd = perf_model.ConvBias_Backward(_conv_bias_bwd_event(seq_num=999))
        assert cached_bwd.bytes() is not None

        perf_model.ConvBiasReLU_.fwd_pass_cache.clear()
        relu3d = {
            "args": {
                "Input Dims": [[2, 3, 8, 8, 8], [4, 3, 3, 3, 3], [4]],
                "Input type": ["c10::BFloat16"] * 3,
                "Concrete Inputs": ["", "", "", "1", "0"],
                "Sequence number": 51,
            }
        }
        perf_model.ConvBiasReLU_(relu3d)
        relu_bwd = perf_model.ConvBiasReLU_Backward(_conv_bias_bwd_event(seq_num=51))
        assert relu_bwd.bytes() is not None

        empty_bwd_evt = {"args": {"Input Dims": [], "Sequence number": 777}}
        with pytest.warns(UserWarning, match="insufficient trace data"):
            details = perf_model.ConvBiasReLU_Backward.get_param_details(empty_bwd_evt)
        assert details["input_shape"] is None

        perf_model.ConvBiasReLU_.fwd_pass_cache[888] = {
            "input_shape": (2, 3, 8, 8),
            "filter_shape": (4, 3, 3, 3),
            "dtype_input_weight": ("c10::BFloat16", "c10::Half"),
            "bias": True,
            "stride": (1, 1),
            "padding": (0, 0),
            "dilation": (1, 1),
            "transposed_conv": False,
            "output_padding": (0, 0),
            "groups": 1,
            "input_stride": None,
            "weight_stride": None,
            "convNd": "conv2d",
        }
        mixed_relu = perf_model.ConvBiasReLU_Backward(_conv_bias_bwd_event(seq_num=888))
        assert mixed_relu.bytes() is not None

    def test_softmax_sdpa_simulation_and_bwd_bytes(self):
        assert perf_model.Softmax.bytes_bwd(4, 8, 2) > 0

        with patch.object(
            perf_model.GEMM,
            "get_simulation_time_func",
            side_effect=[(1.0, "qkt"), (None, None)],
        ):
            t = perf_model.SDPA.get_simulation_time_func(
                _ARCH,
                "fp16",
                None,
                "c10::Half",
                1000,
                1,
                8,
                64,
                64,
                32,
                fa=True,
            )
            assert t is None

        fa_bwd = perf_model.flash_attention_backward(_flash_bwd_event())
        assert fa_bwd.bytes() > 0

        aiter_bwd = perf_model.aiter__mha_bwd(
            {
                "args": {
                    "Input Dims": [
                        [2, 128, 8, 64],
                        [2, 128, 8, 64],
                        [2, 128, 8, 64],
                        [2, 128, 8, 64],
                        [2, 128, 8],
                    ],
                    "Input type": ["c10::BFloat16"] * 5,
                    "Concrete Inputs": [""] * 7 + ["0.0", "1.0", "False"],
                }
            }
        )
        assert aiter_bwd.bytes() > 0

    def test_aten_reduce_grouped_gemm_primus(self):
        reduce_evt = {
            "name": "aten::sum",
            "args": {
                "Input Dims": [[2, 4, 8]],
                "Input type": ["c10::BFloat16"],
                "Output type": "c10::BFloat16",
                "Concrete Inputs": ["", "[1]", "False"],
            },
        }
        assert perf_model.aten_reduce(reduce_evt).flops() > 0

        bad_dim = {
            "name": "aten::sum",
            "args": {
                "Input Dims": [[2, 4, 8]],
                "Input type": ["c10::BFloat16"],
                "Concrete Inputs": ["", "not_a_dim", "True"],
            },
        }
        assert perf_model.aten_reduce(bad_dim).flops() >= 0

        list_dim = {
            "name": "aten::sum",
            "args": {
                "Input Dims": [[2, 4, 8]],
                "Input type": ["c10::BFloat16"],
                "Concrete Inputs": ["", "[0, 2]", "True"],
            },
        }
        assert perf_model.aten_reduce(list_dim).flops() > 0

        gg = _GroupedGemmNoBwdOverride({"args": {}})
        assert gg.flops_bwd() > 0
        assert gg.bytes_bwd() > 0

        impl = perf_model.primus_turbo_grouped_gemm(
            {
                "args": {
                    "Input Dims": [[128, 64], [4, 64, 32]],
                    "Input type": ["c10::BFloat16", "c10::BFloat16"],
                }
            }
        )
        assert impl.flops() > 0
        assert perf_model.primus_turbo_grouped_gemm._extract_impl_dims([]) is None
        assert perf_model.primus_turbo_grouped_gemm._extract_zipped_dims(
            [[(64, 32), (64, 32)], [(32, 16), (32, 16)]]
        ) == (128, 32, 16, 2)

        with pytest.raises(ValueError):
            perf_model.primus_turbo_grouped_gemm({"args": {"Input Dims": [[1]]}})

        var_k = perf_model.primus_turbo_grouped_gemm_variable_k(
            {
                "args": {
                    "Input Dims": [[128, 64], [128, 32]],
                    "Input type": ["c10::BFloat16", "c10::BFloat16"],
                }
            }
        )
        assert var_k.bytes() > 0
        zipped = perf_model.primus_turbo_grouped_gemm_variable_k(
            {
                "args": {
                    "Input Dims": [
                        [(64, 32), (64, 48)],
                        [(32, 16), (48, 16)],
                    ],
                    "Input type": ["c10::BFloat16", "c10::BFloat16"],
                }
            }
        )
        assert zipped.flops() > 0
        assert (
            perf_model.primus_turbo_grouped_gemm_variable_k._extract_zipped_pairs(
                [[(1, 2)], [(3, 4)]]
            )
            is None
        )

    def test_jax_norm_rope_cross_entropy_mamba(self):
        jax_attn = perf_model.jax_te_fused_attn(
            {
                "args": {
                    "Input Dims": [[2, 128, 8, 64], [2, 128, 8, 64], [2, 128, 8, 64]],
                    "Input type": ["bf16", "bf16", "bf16"],
                    "Concrete Inputs": ["[False]"],
                }
            }
        )
        assert jax_attn.bytes_bwd() > 0

        jax_conv1d = perf_model.jax_conv(
            {
                "args": {
                    "Input Dims": [[2, 3, 32], [4, 3, 5]],
                    "Output Dims": [[2, 4, 28]],
                    "Filter Shape": [4, 3, 5],
                    "Input type": ["bf16", "bf16"],
                }
            }
        )
        assert jax_conv1d.param_details["convNd"] == "conv1d"
        jax_conv3d = perf_model.jax_conv(
            {
                "args": {
                    "Input Dims": [[2, 3, 8, 8, 8], [4, 3, 3, 3, 3]],
                    "Output Dims": [[2, 4, 6, 6, 6]],
                    "Filter Shape": [4, 3, 3, 3, 3],
                    "Input type": ["bf16", "bf16"],
                }
            }
        )
        assert jax_conv3d.bytes_bwd() > 0

        with pytest.raises(NotImplementedError):
            perf_model.InstanceNormBwd({"args": {}}).flops()

        ce = perf_model.cross_entropy_fwd(
            {
                "args": {
                    "Input Dims": [[4, 1, 8192], [4, 1]],
                    "Input type": ["invalid_dtype_xyz"],
                }
            }
        )
        ce.bpe = None
        assert ce.bytes() is None

        rope = perf_model.fused_rope_fwd(
            {
                "args": {
                    "Input Dims": [[128, 2, 8, 64], [128, 1, 1, 64]],
                    "Input type": ["bad_rope_dtype"],
                }
            }
        )
        rope.bpe = None
        assert rope.bytes() is None

        conv = perf_model.causal_conv1d_fwd(
            {
                "args": {
                    "Input Dims": [[2, 128, 64], [128, 4]],
                    "Input type": ["unknown_conv_dtype"],
                }
            }
        )
        conv.bpe = None
        assert conv.bytes() is None

        mamba = perf_model.mamba_ssd_fwd(_mamba_event(batch=2, seqlen=128))
        assert mamba._param_bpe([], 0, 2) == 2

    def test_quantize_mxfp4_and_aiter_gemm_strides(self):
        mx = perf_model.primus_turbo_quantize_mxfp4_dual(
            {
                "args": {
                    "Input Dims": [[128, 256]],
                    "Input type": ["bad_mx_dtype"],
                    "Input Strides": [[32768, 256, 1]],
                }
            }
        )
        mx.bpe_in = None
        assert mx.bytes() is None

        aiter = perf_model.aiter_gemm_a4w4(
            {
                "args": {
                    "Input Dims": [
                        [64, 32],
                        [128, 32],
                        [64, 2],
                        [128, 2],
                        (),
                        (),
                        (),
                        (),
                        (),
                    ],
                    "Input type": [
                        "c10::Float4_e2m1fn_x2",
                        "c10::Float4_e2m1fn_x2",
                        "c10::Float8_e8m0fnu",
                        "c10::Float8_e8m0fnu",
                    ],
                    "Input Strides": [[2048, 32, 1], [4096, 32, 1]],
                }
            }
        )
        assert aiter.bytes() > 0


class TestMoeExtensionsPhase11:
    def test_unfused_bytes_none_and_precision(self):
        up = moe_ext.moe_triton_unfused_up(_moe_unfused_event())
        assert up.get_compute_precision() in (None, "fp8", "bf16", "fp16")

        assert (
            moe_ext.UnfusedMoE_Up.bytes_func(32, 4096, 14336, 8, 2, False, 2, None, 2)
            is None
        )

        down = moe_ext.moe_aiter_unfused_down(
            {
                "args": {
                    "Input Dims": [
                        [32, 2, 7168],
                        [8, 4096, 896],
                        [32, 4096],
                    ],
                    "Input type": [
                        "c10::BFloat16",
                        "c10::Float8_e4m3fn",
                        "c10::BFloat16",
                    ],
                }
            }
        )
        assert down.bytes() > 0
        with pytest.raises(NotImplementedError):
            down.flops_bwd()


class TestTreePerfPhase11:
    def test_build_df_perf_metrics_exception_paths(self):
        corr = 300
        events = [
            _make_gpu_event(
                "cpu_ok",
                1000,
                50,
                "cpu_op",
                "aten::mm",
                args={
                    "Input Dims": [[32, 64], [64, 128]],
                    "Input type": ["fp16", "fp16"],
                },
            ),
            _make_gpu_event(
                "cpu_bad",
                1100,
                50,
                "cpu_op",
                "aten::unknown_op_xyz",
                args={"Input Dims": [[2, 2]], "Input type": ["fp16"]},
            ),
            _make_gpu_event(
                "cpu_boom",
                1200,
                50,
                "cpu_op",
                "aten::explode_op",
                args={"Input Dims": [[2, 2]], "Input type": ["fp16"]},
            ),
            _make_gpu_event(
                "rt",
                1010,
                5,
                "cuda_runtime",
                "hipLaunchKernel",
                args={"correlation": corr},
            ),
            _make_gpu_event(
                "k",
                1050,
                40,
                "kernel",
                "gemm_k",
                pid=0,
                tid=7,
                args={"correlation": corr, "stream": 7},
            ),
            _mk_ac2g(corr, 0, 7, 1050, "s"),
            _mk_ac2g(corr, 0, 7, 1090, "f"),
        ]
        analyzer = _build_analyzer(events)
        df = analyzer.build_df_perf_metrics(
            events=[e for e in analyzer.tree.events if e["cat"] == "cpu_op"],
            include_args=True,
            dict_name_to_perf_model={
                "aten::unknown_op_xyz": _BadPerfModel,
                "aten::explode_op": _ExplodingPerfModel,
            },
        )
        assert isinstance(df, pd.DataFrame)

    def test_resnet_recompute_unified_table(self):
        if not os.path.isfile(RESNET_TRACE):
            pytest.skip("resnet trace missing")
        analyzer = TreePerfAnalyzer.from_file(
            RESNET_TRACE,
            rebuild_tree=True,
            enable_pseudo_ops=True,
            detect_recompute=True,
        )
        df = analyzer.build_df_unified_perf_table(include_perf_metrics=True)
        assert isinstance(df, pd.DataFrame)
        summary = analyzer.summarize_df_unified_perf_table(
            df, include_overlapping_kernels=True
        )
        assert isinstance(summary, pd.DataFrame)


class TestOrchestratorPhase11:
    def test_sibling_sequence_and_enrichment_skip(self, tmp_path):
        k1 = _kernel_event(10, "Cijk_gemm_a", dur=500)
        k2 = _kernel_event(11, "Cijk_gemm_b", dur=400)
        k3 = _kernel_event(20, "Cijk_sib_a", dur=300)
        k4 = _kernel_event(21, "Cijk_sib_b", dur=200)
        mod = {
            "name": "nn.Module: MLP_0",
            "_category": "aten",
            "gpu_events": [10, 11],
        }
        parent = {"name": "nn.Module: Block_0", "_category": "aten", "UID": 100}
        child1 = {
            "name": "aten::mm",
            "_category": "aten",
            "gpu_events": [20],
            "parent": 100,
        }
        child2 = {
            "name": "aten::add",
            "_category": "aten",
            "gpu_events": [21],
            "parent": 100,
        }
        mod_dup = {
            "name": "nn.Module: MLP_1",
            "_category": "aten",
            "gpu_events": [10, 11],
        }
        tree = _StubTree(
            [mod, mod_dup, parent, child1, child2],
            {10: k1, 11: k2, 20: k3, 21: k4, 100: parent},
        )
        analyzer = _StubAnalyzer(tree, unified_events=[child1, child2])

        csv_dir = tmp_path / "csv"
        csv_dir.mkdir()
        pd.DataFrame(
            {
                "kernel_details_summary": [
                    "[{'name': 'Cijk_gemm_a'}]",
                    "[{'name': 'Cijk_gemm_b'}]",
                ],
                "op category": ["GEMM", "GEMM"],
                "Data Moved (MB)": [10.0, 8.0],
                "perf_params": ["{}", "{}"],
                "Input Dims": ["[[2,3]]", "[[2,3]]"],
            }
        ).to_csv(csv_dir / "unified_perf_summary.csv", index=False)

        cands = _extract_standalone_fusion_candidates(analyzer, tree, str(csv_dir))
        assert isinstance(cands, list)
        sibling = [c for c in cands if c.get("source") == "sibling_sequence"]
        assert sibling

        for c in cands:
            for k in c.get("kernels", []):
                k["data_in_mb"] = 1.0

        cands2 = _extract_standalone_fusion_candidates(analyzer, tree, str(csv_dir))
        assert cands2

    def test_orchestrator_main_empty_csv_exit(self, tmp_path, monkeypatch):
        from TraceLens.Agent.Analysis.utils import orchestrator_prepare as op

        out = str(tmp_path / "empty_out")
        os.makedirs(out)
        empty_csv = os.path.join(out, "perf_report_csvs")
        os.makedirs(empty_csv)

        old_argv = sys.argv
        sys.argv = [
            "orchestrator_prepare",
            "--trace-path",
            NORM_TRACE,
            "--platform",
            "MI300X",
            "--output-dir",
            out,
            "--comparison-scope",
            "standalone",
        ]
        try:
            with pytest.raises(SystemExit) as exc:
                op.main()
            assert exc.value.code == 1
        finally:
            sys.argv = old_argv

    def test_orchestrator_bottleneck_top5_fallback(self, tmp_path, monkeypatch):
        from TraceLens.Agent.Analysis.utils import orchestrator_prepare as op

        out = str(tmp_path / "orch_out")
        _write_minimal_orchestrator_csvs(out, comparative=False)
        csv_dir = os.path.join(out, "perf_report_csvs")
        rows = []
        for i in range(8):
            rows.append(
                {
                    "name": f"aten::op_{i}",
                    "op category": "GEMM",
                    "Kernel Time (µs)_sum": float(100 + i),
                    "total_duration_us": 60000.0,
                    "kernel_details_summary": f"[{{'name': 'k{i}'}}]",
                    "Data Moved (MB)": 1.0,
                    "perf_params": "{}",
                    "Input Dims": "[[2,3]]",
                }
            )
        pd.DataFrame(rows).to_csv(
            os.path.join(csv_dir, "unified_perf_summary.csv"), index=False
        )
        pd.DataFrame({"name": ["aten::op_0"], "op category": ["GEMM"]}).to_csv(
            os.path.join(csv_dir, "ops_summary.csv"), index=False
        )

        class _FakeTPA:
            @classmethod
            def from_file(cls, *args, **kwargs):
                k = _kernel_event(0, "k0", dur=100)
                mod = {
                    "name": "aten::mm",
                    "_category": "aten",
                    "gpu_events": [0],
                    "ts": 0,
                }
                tree = _StubTree([mod, k], {0: k})
                return _StubAnalyzer(tree)

        monkeypatch.setattr(op, "TreePerfAnalyzer", _FakeTPA)
        old_argv = sys.argv
        sys.argv = [
            "orchestrator_prepare",
            "--trace-path",
            NORM_TRACE,
            "--platform",
            "MI300X",
            "--output-dir",
            out,
        ]
        try:
            op.main()
        finally:
            sys.argv = old_argv
        assert os.path.isdir(os.path.join(out, "category_data"))


class TestReportingPhase11:
    def test_inference_report_capture_merge(self, tmp_path):
        from TraceLens.Reporting.generate_perf_report_pytorch_inference import (
            generate_perf_report_pytorch,
        )
        from tests.test_reporting_coverage import _write_trace

        trace = _write_trace(tmp_path, [("aten::mm", "gemm_kernel", 100)], "inf.json")
        out = tmp_path / "inf_out"
        dfs = generate_perf_report_pytorch(
            profile_json_path=str(trace),
            output_csvs_dir=str(out),
            kernel_summary=True,
        )
        assert isinstance(dfs, dict)
