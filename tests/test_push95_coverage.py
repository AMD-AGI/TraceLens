###############################################################################
# Copyright (c) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""CPU-only coverage push toward 95% line coverage on TraceLens."""

from __future__ import annotations

import importlib
import json
import os
import sys
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from TraceLens.Agent.Analysis.utils.orchestrator_prepare import (
    _extract_attention_core,
    _extract_comparative_fusion_candidates,
    _extract_standalone_fusion_candidates,
    _is_gemm_norm_only,
)
from TraceLens.PerfModel import perf_model
from TraceLens.PerfModel.extensions import moe_perf_model_extensions as moe_ext
from TraceLens.PerfModel.extensions import perf_model_extensions as pext
from TraceLens.Reporting.generate_perf_report_pytorch import (
    generate_perf_report_pytorch,
)
from TraceLens.Reporting.generate_perf_report_pytorch_inference import (
    generate_perf_report_pytorch as generate_inference_report,
)
from TraceLens.TreePerf.tree_perf import TreePerfAnalyzer

from tests.test_agent_coverage import (
    _StubAnalyzer,
    _StubTree,
    _kernel_event,
    _write_minimal_orchestrator_csvs,
)
from tests.test_mamba_ssd import _mamba_event
from tests.test_perfmodel_coverage import (
    _ARCH,
    _GDN_ANNOTATION,
    _moe_unfused_event,
    _norm_event,
)
from tests.test_reporting_coverage import (
    _build_synthetic_trace,
    _create_genesis_capture,
    _write_trace,
)
from tests.test_treeperf_coverage import (
    _build_analyzer,
    _mk_pytorch_trace,
)

TESTS_DIR = os.path.dirname(__file__)
TRACES_ROOT = os.path.join(TESTS_DIR, "traces")
INFERENCE_ROOT = os.path.join(TRACES_ROOT, "inference")
ROCprof_FILE = os.path.join(TESTS_DIR, "rocprof/908_results.json.gz")
NORM_TRACE = os.path.join(
    TRACES_ROOT, "perf_model/normalization/normalization_layer_test.json.gz"
)

pytestmark = pytest.mark.filterwarnings(
    "ignore:Found .* events with failed performance metric.*:UserWarning",
    "ignore:There are hipgraph launches.*:UserWarning",
    "ignore:Kernel name missing in capture event args.*:UserWarning",
    "ignore:Inconsistent kernel list length found.*:UserWarning",
    "ignore:Source column .* not found.*:UserWarning",
    "ignore:get_df_kernel_launchers_summary_by_shape is deprecated.*:UserWarning",
    "ignore:Input DataFrame is empty.*:UserWarning",
)


def _discover_trace_gz_files():
    cases = []
    for root, _dirs, files in os.walk(TRACES_ROOT):
        for name in sorted(files):
            if not name.endswith(".json.gz"):
                continue
            path = os.path.join(root, name)
            rel = os.path.relpath(path, TESTS_DIR)
            cases.append(pytest.param(path, id=rel.replace(os.sep, "/")))
    return cases


def _discover_inference_cases():
    if not os.path.isdir(INFERENCE_ROOT):
        return []
    cases = []
    for entry in sorted(os.listdir(INFERENCE_ROOT)):
        dirpath = os.path.join(INFERENCE_ROOT, entry)
        if not os.path.isdir(dirpath):
            continue
        gz = [f for f in os.listdir(dirpath) if f.endswith(".json.gz")]
        if not gz:
            continue
        cases.append(pytest.param(dirpath, gz[0], id=entry))
    return cases


# ---------------------------------------------------------------------------
# TreePerf — sweep all json.gz fixtures
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("trace_path", _discover_trace_gz_files())
def test_treeperf_from_file_full_methods(trace_path):
    analyzer = TreePerfAnalyzer.from_file(
        trace_path,
        rebuild_tree=True,
        enable_pseudo_ops=True,
        add_python_func=True,
    )
    assert analyzer.tree is not None
    gpu_only = analyzer.check_gpu_only()
    assert gpu_only in (True, False, None)

    unified = analyzer.build_df_unified_perf_table(include_nccl=False)
    assert isinstance(unified, pd.DataFrame)

    summarized = TreePerfAnalyzer.summarize_df_unified_perf_table(
        unified, include_pct=True, tree=analyzer.tree
    )
    assert isinstance(summarized, pd.DataFrame)

    kernels = analyzer.get_df_kernels(
        launcher_detail=True,
        cpu_op_detail=True,
        nn_module_detail=analyzer.add_python_func,
    )
    assert isinstance(kernels, pd.DataFrame)

    try:
        timeline = analyzer.get_df_gpu_timeline(micro_idle_thresh_us=1)
    except ValueError:
        timeline = pd.DataFrame()
    assert isinstance(timeline, pd.DataFrame)

    launchers = analyzer.get_df_kernel_launchers(include_args=True)
    assert isinstance(launchers, pd.DataFrame)
    if not launchers.empty:
        summary = TreePerfAnalyzer.get_df_kernel_launchers_summary(launchers)
        assert isinstance(summary, pd.DataFrame)


# ---------------------------------------------------------------------------
# PerfModel — subprocess / origami / norm / SDPA / jax / MoE comm
# ---------------------------------------------------------------------------


class TestPerfModelPush95Coverage:
    @pytest.mark.parametrize(
        "missing,kwargs",
        [
            ("M", {"M": None, "N": 8, "K": 16, "B": 1, "dtype": "bf16"}),
            ("N", {"M": 4, "N": None, "K": 16, "B": 1, "dtype": "bf16"}),
            ("K", {"M": 4, "N": 8, "K": None, "B": 1, "dtype": "bf16"}),
            ("dtype", {"M": 4, "N": 8, "K": 16, "B": 1, "dtype": None}),
            ("arch['name']", {"M": 4, "N": 8, "K": 16, "B": 1, "dtype": "bf16"}),
        ],
    )
    def test_gemm_simulator_missing_inputs(
        self, monkeypatch, tmp_path, missing, kwargs
    ):
        sim = tmp_path / "run_gemm.py"
        sim.write_text("# stub\n")
        monkeypatch.setenv("GEMM_SIMULATOR_PATH", str(sim))
        perf_model.GEMM.cache_gemm_results.clear()
        arch = dict(_ARCH) if "arch['name']" not in missing else {"freq_mhz": 2200}
        with pytest.raises(AssertionError, match="Invalid inputs"):
            perf_model.GEMM.get_simulation_time_func(arch, **kwargs)
        perf_model.GEMM.cache_gemm_results.clear()

    def test_gemm_simulator_subprocess_failure(self, monkeypatch, tmp_path):
        sim = tmp_path / "run_gemm.py"
        sim.write_text("# stub\n")
        monkeypatch.setenv("GEMM_SIMULATOR_PATH", str(sim))
        perf_model.GEMM.cache_gemm_results.clear()
        with patch("TraceLens.PerfModel.perf_model.subprocess.run") as run:
            run.return_value = MagicMock(stdout="", stderr="fail")
            with pytest.raises(AssertionError, match="Failed to simulate"):
                perf_model.GEMM.get_simulation_time_func(_ARCH, 4, 8, 16, 1, "bf16")
        perf_model.GEMM.cache_gemm_results.clear()

    def test_gemm_origami_mock_path(self, monkeypatch):
        monkeypatch.delenv("GEMM_SIMULATOR_PATH", raising=False)
        perf_model.GEMM.cache_gemm_results.clear()
        mock_hw = MagicMock()
        mock_helper = MagicMock()
        mock_helper.get_simulation_time.return_value = 7.5
        mock_origami = MagicMock()
        mock_origami.data_type_t.BFloat16 = "bf16_enum"
        with patch.dict(sys.modules, {"origami": mock_origami}):
            with patch(
                "TraceLens.PerfModel.perf_model.OrigamiHelper",
                create=True,
            ) as helper_cls:
                with patch(
                    "TraceLens.PerfModel.origami_helper.OrigamiHelper",
                    helper_cls,
                ):
                    helper_cls.get_hardware.return_value = mock_hw
                    helper_cls.return_value = mock_helper
                    t, cmd = perf_model.GEMM.get_simulation_time_func(
                        _ARCH,
                        4,
                        8,
                        16,
                        1,
                        "bf16",
                        num_cus=64,
                        force_to_l1=True,
                        enable_origami=True,
                    )
        assert t == 7.5
        assert "Origami" in cmd

    def test_gemm_origami_unsupported_dtype(self, monkeypatch):
        monkeypatch.delenv("GEMM_SIMULATOR_PATH", raising=False)
        mock_origami = MagicMock()
        mock_origami.data_type_t = MagicMock()
        with patch.dict(sys.modules, {"origami": mock_origami}):
            with patch("TraceLens.PerfModel.origami_helper.OrigamiHelper"):
                with pytest.warns(RuntimeWarning, match="Unsupported dtype"):
                    t, _ = perf_model.GEMM.get_simulation_time_func(
                        _ARCH, 4, 8, 16, 1, "unknown_dtype", enable_origami=True
                    )
        assert t is None

    def test_sdpa_simulation_via_subprocess_gemm(self, monkeypatch, tmp_path):
        sim = tmp_path / "run_gemm.py"
        sim.write_text("# stub\n")
        monkeypatch.setenv("GEMM_SIMULATOR_PATH", str(sim))
        perf_model.GEMM.cache_gemm_results.clear()
        with patch("TraceLens.PerfModel.perf_model.subprocess.run") as run:
            run.return_value = MagicMock(stdout="Time=2.0\n", stderr="")
            with patch.object(perf_model.Softmax, "get_time", return_value=0.25):
                t = perf_model.SDPA.get_simulation_time_func(
                    _ARCH,
                    "bf16",
                    "/usr/bin/python3",
                    "c10::BFloat16",
                    1024,
                    2,
                    8,
                    128,
                    128,
                    64,
                    fa=True,
                )
        assert t > 0
        perf_model.GEMM.cache_gemm_results.clear()

    @pytest.mark.parametrize(
        "cls,event",
        [
            (
                perf_model.BatchNormBwd,
                {
                    "name": "aten::miopen_batch_norm_backward",
                    "args": {
                        "Input Dims": [
                            (8, 16, 32, 32),
                            (8, 16, 32, 32),
                            (16,),
                            (16,),
                            (16,),
                            (16,),
                            (16,),
                            (),
                        ],
                        "Input type": ["float"] * 7 + ["Scalar"],
                        "Input Strides": [(16384, 1024, 32, 1)] * 2 + [(1,)] * 5 + [()],
                        "Concrete Inputs": ["", "", "", "", "", "", "", "1e-5"],
                    },
                },
            ),
            (
                perf_model.GroupNormBwd,
                {
                    "args": {
                        "Input Dims": [
                            None,
                            (4, 8, 32, 32),
                            (8,),
                            (8,),
                            (8,),
                            (8,),
                            (4, 8, 32, 32),
                            (),
                        ],
                        "Input type": ["c10::BFloat16"] * 7 + ["Scalar"],
                        "Input Strides": [(), (8192, 1024, 32, 1), (1,)] * 2
                        + [(8192, 1024, 32, 1)] * 2
                        + [(), ()],
                        "Concrete Inputs": [
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "8",
                            "8",
                            "[True, True]",
                        ],
                    }
                },
            ),
        ],
    )
    def test_norm_backward_variants(self, cls, event):
        model = cls(event)
        assert model.flops() > 0
        assert model.bytes() > 0

    def test_instance_norm_training_flag(self):
        event = _norm_event((4, 8, 32, 32), 8, training=False)
        event["args"]["Concrete Inputs"][5] = ""
        model = perf_model.InstanceNorm(event)
        assert model.is_training is False

    def test_instance_norm_bwd_raises(self):
        with pytest.raises(NotImplementedError):
            perf_model.InstanceNormBwd.get_param_details({})

    def test_mamba_cross_entropy_moe_comm(self):
        mamba = perf_model.mamba_ssd_fwd(_mamba_event(batch=2, seqlen=128))
        assert mamba.flops() > 0
        assert mamba.bytes() > 0

        ce = perf_model.cross_entropy_fwd(
            {
                "args": {
                    "Input Dims": [[4, 32000], [4]],
                    "Input type": ["c10::BFloat16", "long int"],
                }
            }
        )
        assert ce.flops() > 0
        assert ce.get_compute_precision() is not None

        conv = perf_model.causal_conv1d_fwd(
            {
                "args": {
                    "Input Dims": [[2, 128, 512], [128, 4], [128]],
                    "Input type": ["c10::BFloat16"] * 3,
                }
            }
        )
        assert conv.bytes() > 0

        empty_comm = perf_model.moe_dispatch(
            {"args": {"Input Dims": [[]], "Input type": []}}
        )
        assert empty_comm.bytes() is None
        assert empty_comm.flops_bwd() == 0

    def test_jax_gemm_and_conv(self):
        gemm = perf_model.jax_gemm(
            {
                "args": {
                    "Batch": 2,
                    "M": 4,
                    "N": 8,
                    "K": 16,
                    "Beta": 1,
                    "Type": "bf16",
                }
            }
        )
        assert gemm.flops() > 0
        conv = perf_model.jax_conv(
            {
                "args": {
                    "Input Dims": [[2, 3, 8, 8], [4, 3, 3, 3]],
                    "Output Dims": [[2, 4, 6, 6]],
                    "Filter Shape": [4, 3, 3, 3],
                    "Input type": ["bf16", "bf16"],
                    "Concrete Inputs": [
                        "",
                        "",
                        "(1,1)",
                        "(0,0)",
                        "(1,1)",
                        "False",
                        "(0,0)",
                        "1",
                    ],
                }
            }
        )
        assert conv.flops_bwd() > 0

    def test_hipblaslt_gemm_fp8_fp4(self):
        from tests.test_primus_fp8_gemm_quantize import _fp8_gemm_event
        from tests.test_primus_mxfp4_gemm_quantize import _fp4_gemm_event

        fp8 = perf_model.hipblaslt_gemm_fp8(
            _fp8_gemm_event((128, 64), (256, 64), trans_b=True)
        )
        assert fp8.flops() > 0
        assert fp8.bytes() > 0
        with pytest.raises(NotImplementedError):
            fp8.flops_bwd()

        fp4 = perf_model.hipblaslt_gemm_fp4(
            _fp4_gemm_event((128, 64), (256, 32), trans_b=True)
        )
        assert fp4.flops() > 0
        assert fp4.bytes() > 0


# ---------------------------------------------------------------------------
# MoE extensions — bytes / flops_bwd on remaining classes
# ---------------------------------------------------------------------------


class TestMoeExtensionsPush95Coverage:
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

    MOE_BLOCKSCALE = {
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

    CK1 = {
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
            "Input type": ["c10::BFloat16", "c10::Float8_e4m3fn", "c10::Float8_e4m3fn"],
        }
    }

    CK2 = {
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
            "Input type": ["c10::BFloat16", "c10::Float8_e4m3fn", "c10::Float8_e4m3fn"],
        }
    }

    FLY = {
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

    GPTQ = {
        "args": {
            "Input Dims": [[32, 4096], [32, 2], [8, 7168, 4096]],
            "Input type": ["c10::BFloat16", "c10::Int", "c10::BFloat16"],
            "MoE topk": 2,
        }
    }

    GROUPED = {
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

    @pytest.mark.parametrize(
        "factory,event",
        [
            (moe_ext.moe_aiter_fused_1stage, "MOE_FUSED"),
            (moe_ext.moe_aiter_fused_blockscale, "MOE_BLOCKSCALE"),
            (moe_ext.moe_aiter_ck_stage1, "CK1"),
            (moe_ext.moe_aiter_ck_stage2, "CK2"),
            (moe_ext.moe_flydsl_stage1, "FLY"),
            (moe_ext.moe_flydsl_stage2, "FLY"),
            (moe_ext.moe_gptq_awq_up, "GPTQ"),
            (moe_ext.moe_gptq_awq_down, "GPTQ"),
            (moe_ext.moe_triton_invoke_grouped_gemm, "GROUPED"),
            (moe_ext.moe_triton_unfused_up, None),
            (moe_ext.moe_triton_unfused_down, None),
            (moe_ext.moe_aiter_unfused_up, None),
            (moe_ext.moe_aiter_unfused_down, None),
            (moe_ext.sglang_fused_append_shared_experts, None),
            (moe_ext.BiasedGroupedTopk, None),
            (moe_ext.MoeSortScatterGather, None),
        ],
    )
    def test_moe_bytes_and_bwd_raises(self, factory, event):
        if event is None:
            if factory in (
                moe_ext.moe_triton_unfused_up,
                moe_ext.moe_triton_unfused_down,
            ):
                evt = _moe_unfused_event(kernel_name="moe_fp8_up_kernel")
            elif factory is moe_ext.moe_aiter_unfused_up:
                evt = {
                    "args": {
                        "Input Dims": [
                            [32, 4096],
                            [8, 14336, 512],
                            [32, 2, 7168],
                        ],
                        "Input type": [
                            "c10::BFloat16",
                            "c10::Float8_e4m3fn",
                            "c10::BFloat16",
                        ],
                    }
                }
            elif factory is moe_ext.moe_aiter_unfused_down:
                evt = {
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
            elif factory is moe_ext.sglang_fused_append_shared_experts:
                evt = {
                    "args": {
                        "Input Dims": [(32, 4096), (32, 4096), (32, 4096)],
                        "Input type": ["c10::BFloat16"] * 3,
                    }
                }
            elif factory is moe_ext.BiasedGroupedTopk:
                evt = {
                    "args": {
                        "Input Dims": [(32, 256), (256,), (32, 8), (32, 8)],
                        "Input type": ["c10::Float"] * 3 + ["c10::Int"],
                    }
                }
            else:
                evt = {
                    "args": {
                        "Input Dims": [(32, 4096), (32, 2), (32, 4096)],
                        "Input type": ["c10::BFloat16", "c10::Int", "c10::BFloat16"],
                    }
                }
        else:
            evt = getattr(self, event)

        model = factory(evt)
        b = model.bytes()
        assert b is None or b >= 0
        if hasattr(model, "flops_bwd"):
            with pytest.raises(NotImplementedError):
                model.flops_bwd()
        if hasattr(model, "bytes_bwd"):
            with pytest.raises(NotImplementedError):
                model.bytes_bwd()


# ---------------------------------------------------------------------------
# Orchestrator prepare — fusion edge cases
# ---------------------------------------------------------------------------


class TestOrchestratorPush95Coverage:
    def test_attention_core_narrowing(self):
        perf_lookup = {
            "Cijk_QK": {
                None: {"op_category": "GEMM", "data_in_mb": 1.0, "data_out_mb": 1.0},
            },
            "Cijk_PV": {
                None: {"op_category": "GEMM", "data_in_mb": 2.0, "data_out_mb": 2.0},
            },
            "softmax_kernel": {
                None: {
                    "op_category": "SDPA_fwd",
                    "data_in_mb": 0.5,
                    "data_out_mb": 0.5,
                },
            },
        }
        kernels = [
            {"name": "Cijk_QK", "type": "GEMM", "dur_us": 100},
            {"name": "softmax_kernel", "type": "SDPA", "dur_us": 50},
            {"name": "Cijk_PV", "type": "GEMM", "dur_us": 120},
            {
                "name": "vectorized_elementwise_kernel",
                "type": "Elementwise",
                "dur_us": 10,
            },
        ]
        core = _extract_attention_core(kernels, perf_lookup)
        assert core is not None
        assert len(core) == 3
        assert core[1]["name"] == "softmax_kernel"

    def test_gemm_norm_only_detection(self):
        entry = {
            "kernels": [
                {"name": "Cijk_gemm", "type": "GEMM"},
                {"name": "rmsnorm2d_kernel", "type": "NORM"},
            ]
        }
        assert _is_gemm_norm_only(entry) is True

    def test_standalone_attention_enrichment_and_duplicate_base(self, tmp_path):
        k_qk = _kernel_event(10, "Cijk_QK_gemm", dur=500)
        k_sm = _kernel_event(11, "softmax_warp_forward", dur=200)
        k_pv = _kernel_event(12, "Cijk_PV_gemm", dur=400)
        mod1 = {
            "name": "nn.Module: Attn_0",
            "_category": "aten",
            "gpu_events": [10, 11, 12],
            "args": {"Input Dims": "[[2,8,128,64]]"},
        }
        mod2 = {
            "name": "nn.Module: Attn_1",
            "_category": "aten",
            "gpu_events": [10, 11, 12],
        }
        tree = _StubTree([mod1, mod2], {10: k_qk, 11: k_sm, 12: k_pv})
        analyzer = _StubAnalyzer(tree)

        csv_dir = tmp_path / "csv"
        csv_dir.mkdir()
        pd.DataFrame(
            {
                "kernel_details_summary": [
                    "[{'name': 'Cijk_QK_gemm'}]",
                    "[{'name': 'softmax_warp_forward'}]",
                    "[{'name': 'Cijk_PV_gemm'}]",
                ],
                "op category": ["GEMM", "SDPA_fwd", "GEMM"],
                "Data Moved (MB)": [10.0, 2.0, 8.0],
                "perf_params": ["{'M':2}", "{}", "{'M':2}"],
                "Input Dims": ["[[2,8,128,64]]", "[[2,8,128,64]]", "[[2,8,128,64]]"],
            }
        ).to_csv(csv_dir / "unified_perf_summary.csv", index=False)

        cands = _extract_standalone_fusion_candidates(analyzer, tree, str(csv_dir))
        assert isinstance(cands, list)

    def test_standalone_keyerror_on_missing_uid(self, tmp_path):
        mod = {
            "name": "nn.Module: Broken_0",
            "_category": "aten",
            "gpu_events": [10, 99],
        }
        tree = _StubTree([mod], {10: _kernel_event(10, "Cijk_a")})
        analyzer = _StubAnalyzer(tree)
        csv_dir = tmp_path / "csv2"
        csv_dir.mkdir()
        pd.DataFrame(
            {
                "kernel_details_summary": ["[{'name': 'Cijk_a'}]"],
                "op category": ["GEMM"],
                "Data Moved (MB)": [1.0],
                "perf_params": ["{}"],
                "Input Dims": ["[[1,1]]"],
            }
        ).to_csv(csv_dir / "unified_perf_summary.csv", index=False)
        cands = _extract_standalone_fusion_candidates(analyzer, tree, str(csv_dir))
        assert isinstance(cands, list)

    def test_comparative_duplicate_base_accumulation(self, tmp_path):
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

    def test_orchestrator_main_alt_ops_summary_column(self, tmp_path, monkeypatch):
        from TraceLens.Agent.Analysis.utils import orchestrator_prepare as op

        out = str(tmp_path)
        csv_dir = os.path.join(out, "perf_report_csvs")
        os.makedirs(csv_dir)
        pd.DataFrame(
            {
                "type": ["total_time", "computation_time", "idle_time"],
                "time ms": [1000.0, 900.0, 100.0],
                "percent": [100.0, 90.0, 10.0],
            }
        ).to_csv(os.path.join(csv_dir, "gpu_timeline.csv"), index=False)
        pd.DataFrame(
            {
                "name": ["aten::mm"],
                "Kernel Time (µs)_sum": [800000.0],
                "op category": ["GEMM"],
            }
        ).to_csv(os.path.join(csv_dir, "ops_summary.csv"), index=False)
        pd.DataFrame(
            {
                "name": ["aten::mm"],
                "op category": ["GEMM"],
                "Kernel Time (µs)": [800.0],
            }
        ).to_csv(os.path.join(csv_dir, "unified_perf_summary.csv"), index=False)
        pd.DataFrame({"name": ["aten::mm"], "op category": ["GEMM"]}).to_csv(
            os.path.join(csv_dir, "ops_summary_by_category.csv"), index=False
        )

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
        ]
        try:
            op.main()
        finally:
            sys.argv = old_argv
        manifest = json.loads(
            open(os.path.join(out, "category_data", "category_manifest.json")).read()
        )
        assert manifest["comparison_scope"] == "standalone"


# ---------------------------------------------------------------------------
# Reporting — inference traces, compare CLI, rocprof, genesis, normalization
# ---------------------------------------------------------------------------


class TestReportingPush95Coverage:
    @pytest.mark.parametrize("dirpath,trace_gz", _discover_inference_cases())
    def test_inference_report_with_capture_merge(self, dirpath, trace_gz, tmp_path):
        trace_path = os.path.join(dirpath, trace_gz)
        capture = os.path.join(dirpath, "capture_traces")
        metadata = os.path.join(capture, "execution_details.json")
        kwargs = {
            "profile_json_path": trace_path,
            "output_csvs_dir": str(tmp_path / "csv"),
            "output_xlsx_path": str(tmp_path / "report.xlsx"),
            "collective_analysis": False,
            "enable_pseudo_ops": True,
            "kernel_summary": True,
            "short_kernel_study": True,
            "group_by_parent_module": True,
            "include_overlap_info": True,
        }
        if os.path.isdir(capture) and os.path.isfile(metadata):
            from TraceLens.Trace2Tree.trace_capture_merge_experimental import (
                merge_capture_trace_into_graph,
            )

            merged = merge_capture_trace_into_graph(capture, metadata, trace_path)
            kwargs["augmented_tree"] = merged
        result = generate_inference_report(**kwargs)
        assert "gpu_timeline" in result

    @pytest.mark.skipif(
        not os.path.isfile(NORM_TRACE), reason="normalization trace missing"
    )
    def test_normalization_trace_pytorch_report(self, tmp_path):
        generate_perf_report_pytorch(
            profile_json_path=NORM_TRACE,
            output_csvs_dir=str(tmp_path / "norm_csv"),
            output_xlsx_path=str(tmp_path / "norm.xlsx"),
            collective_analysis=False,
            kernel_summary=True,
            short_kernel_study=True,
        )
        assert (tmp_path / "norm_csv" / "gpu_timeline.csv").exists()

    def test_compare_perf_reports_cli(self, tmp_path):
        r1 = _write_trace(tmp_path, [("aten::mm", "gemm_kernel", 100)], "r1.json")
        r2 = _write_trace(tmp_path, [("aten::mm", "gemm_kernel", 120)], "r2.json")
        out1 = tmp_path / "rep1"
        out2 = tmp_path / "rep2"
        generate_perf_report_pytorch(
            profile_json_path=r1,
            output_csvs_dir=str(out1),
            output_xlsx_path=str(tmp_path / "r1.xlsx"),
            collective_analysis=False,
        )
        generate_perf_report_pytorch(
            profile_json_path=r2,
            output_csvs_dir=str(out2),
            output_xlsx_path=str(tmp_path / "r2.xlsx"),
            collective_analysis=False,
        )
        cmp_xlsx = tmp_path / "comparison.xlsx"
        mod = importlib.import_module(
            "TraceLens.Reporting.compare_perf_reports_pytorch"
        )
        old_argv = sys.argv
        sys.argv = [
            "compare_perf_reports_pytorch",
            str(out1),
            str(out2),
            "-o",
            str(cmp_xlsx),
            "--names",
            "baseline",
            "candidate",
            "--sheets",
            "gpu_timeline",
            "ops_summary",
        ]
        try:
            mod.main()
        finally:
            sys.argv = old_argv
        assert cmp_xlsx.exists()

    @pytest.mark.skipif(
        not os.path.isfile(ROCprof_FILE), reason="rocprof fixture missing"
    )
    def test_rocprof_report_cli(self, tmp_path):
        out = tmp_path / "roc.xlsx"
        mod = importlib.import_module(
            "TraceLens.Reporting.generate_perf_report_rocprof"
        )
        old_argv = sys.argv
        sys.argv = [
            "generate_perf_report_rocprof",
            "--profile_json_path",
            ROCprof_FILE,
            "--output_xlsx_path",
            str(out),
            "--short_kernel_study",
        ]
        try:
            mod.main()
        finally:
            sys.argv = old_argv
        assert out.exists()

    def test_genesis_report_cli(self, tmp_path):
        capture = _create_genesis_capture(tmp_path)
        out = tmp_path / "gen_out"
        mod = importlib.import_module(
            "TraceLens.Reporting.generate_perf_report_genesis"
        )
        old_argv = sys.argv
        sys.argv = [
            "generate_perf_report_genesis",
            "--capture-dir",
            str(capture),
            "--output-dir",
            str(out),
        ]
        try:
            mod.main()
        finally:
            sys.argv = old_argv
        assert (out / "genesis_perf_report.xlsx").exists()

    def test_perf_extensions_rope_and_quant(self):
        rope = pext.fused_qk_rope_concat_and_cache_mla(
            {
                "args": {
                    "Input Dims": [
                        (2, 8, 512),
                        (2, 8, 64),
                        (2, 1, 512),
                        (2, 1, 64),
                        (128, 1, 1, 576),
                        (2, 128),
                    ],
                    "Input type": ["c10::BFloat16"] * 5 + ["c10::Float8_e4m3fn"],
                }
            }
        )
        assert rope.flops() > 0

        silu = {
            "args": {
                "Input Dims": [(4, 512), (4, 512)],
                "Input type": ["c10::BFloat16", "c10::BFloat16"],
                "Input Strides": [(512, 1), (512, 1)],
            }
        }
        assert pext.aiter_silu_and_mul(silu).bytes() > 0


class TestTreePerfSyntheticPush95:
    def test_bwd_perf_and_launcher_summaries(self):
        analyzer = _build_analyzer(_mk_pytorch_trace())
        launchers = analyzer.get_df_kernel_launchers(
            include_args=True, include_kernel_details=True
        )
        assert isinstance(launchers, pd.DataFrame)
        if not launchers.empty:
            by_cat = TreePerfAnalyzer.get_df_kernel_launchers_summary_by_category(
                launchers
            )
            assert isinstance(by_cat, pd.DataFrame)
            by_mod = (
                TreePerfAnalyzer.get_df_kernel_launchers_summary_by_category_module(
                    launchers
                )
            )
            assert isinstance(by_mod, pd.DataFrame)


# ---------------------------------------------------------------------------
# Phase 2 — deeper coverage on orchestrator, reporting, analysis helpers
# ---------------------------------------------------------------------------


class TestPush95Phase2:
    def test_normalization_trace_treeperf_full(self):
        if not os.path.isfile(NORM_TRACE):
            pytest.skip("normalization trace missing")
        analyzer = TreePerfAnalyzer.from_file(
            NORM_TRACE,
            rebuild_tree=True,
            enable_pseudo_ops=True,
            add_python_func=True,
        )
        unified = analyzer.build_df_unified_perf_table(include_nccl=False)
        assert not unified.empty
        ops = analyzer.build_df_perf_metrics(
            events=[e for e in analyzer.tree.events if e.get("cat") == "cpu_op"]
        )
        assert isinstance(ops, pd.DataFrame)

    def test_orchestrator_main_multi_kernel_memcpy_nccl(self, tmp_path, monkeypatch):
        from TraceLens.Agent.Analysis.utils import orchestrator_prepare as op

        out = str(tmp_path)
        _write_minimal_orchestrator_csvs(out, comparative=False)
        gpu_events = [
            {
                "name": "gemm_kernel",
                "dur": 100,
                "ts": 1000,
                "_category": "kernel",
                "cat": "kernel",
                "args": {"stream": 0},
                "gpu_events": [],
            },
            {
                "name": "MemcpyHtoD",
                "dur": 20,
                "ts": 1100,
                "_category": "kernel",
                "cat": "kernel",
                "args": {"bytes": 4096, "stream": 1},
                "gpu_events": [],
            },
            {
                "name": "MemcpyDtoD",
                "dur": 15,
                "ts": 1120,
                "_category": "kernel",
                "cat": "kernel",
                "args": {"bytes": 2048, "stream": 1},
                "gpu_events": [],
            },
            {
                "name": "ncclKernel_AllReduce",
                "dur": 40,
                "ts": 1200,
                "_category": "kernel",
                "cat": "kernel",
                "args": {"stream": 2},
                "gpu_events": [],
            },
        ]
        tree = _StubTree(gpu_events, {i: e for i, e in enumerate(gpu_events)})
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
        ]
        try:
            op.main()
        finally:
            sys.argv = old_argv

        mk = json.loads(
            open(os.path.join(out, "category_data", "multi_kernel_data.json")).read()
        )
        assert "memcpy_summary" in mk and "nccl_summary" in mk

    def test_gemm_origami_import_error_path(self, monkeypatch):
        monkeypatch.delenv("GEMM_SIMULATOR_PATH", raising=False)
        perf_model.GEMM._origami_import_error_printed = False
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "origami":
                raise ImportError("no origami")
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", fake_import):
            t, _ = perf_model.GEMM.get_simulation_time_func(
                _ARCH, 4, 8, 16, 1, "bf16", enable_origami=True
            )
        assert t is None

    def test_orchestrator_comparative_main(self, tmp_path, monkeypatch):
        from TraceLens.Agent.Analysis.utils import orchestrator_prepare as op

        out = str(tmp_path)
        _write_minimal_orchestrator_csvs(out, comparative=True)
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
            op, "_extract_comparative_fusion_candidates", lambda *a, **k: []
        )
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
            "--comparison-scope",
            "comparative",
        ]
        try:
            op.main()
        finally:
            sys.argv = old_argv
        assert os.path.isfile(
            os.path.join(out, "metadata", "trace2_gpu_utilization.json")
        )

    def test_analysis_utils_and_kernel_fusion(self, tmp_path):
        from TraceLens.Agent.Analysis.category_analyses import analysis_utils as au
        from TraceLens.Agent.Analysis.category_analyses import (
            kernel_fusion_analysis as kfa,
        )

        row = pd.Series(
            {
                "FLOPS/Byte": 0.5,
                "TFLOPS/s_mean": 10.0,
                "TB/s_mean": 0.5,
                "Roofline Bound": "MEMORY_BOUND",
                "Compute Spec": "vector_fp32",
            }
        )
        eff = au.calculate_efficiency(
            row, peak_maf_or_maf_dict={"vector_fp32": 100.0}, peak_hbm_bw=5300
        )
        assert eff["bound_type"] == "memory"

        fusion_dir = tmp_path / "category_data"
        fusion_dir.mkdir()
        (fusion_dir / "kernel_fusion_metrics.json").write_text(
            json.dumps({"high_confidence_kernel_map": {"gemm_a": "fused_a"}})
        )
        assert au._load_fusion_map(str(tmp_path))["gemm_a"] == "fused_a"

        ops = [{"kernel_names": ["a", "b"], "base_name": "Block", "instance_count": 2}]
        assert len(kfa._filter_and_dedup(ops)) == 1

    def test_reporting_pftrace_and_collective(self, tmp_path):
        from TraceLens.Reporting.generate_multi_rank_collective_report_pytorch import (
            generate_collective_report,
        )
        from TraceLens.Reporting.generate_perf_report_pftrace_hip_activity import (
            generate_perf_report_pftrace_hip_activity,
        )
        from tests.test_reporting_coverage import _minimal_pftrace_events

        trace_path = tmp_path / "pf.json"
        trace_path.write_text(json.dumps({"traceEvents": _minimal_pftrace_events()}))
        generate_perf_report_pftrace_hip_activity(
            trace_path=str(trace_path),
            output_csvs_dir=str(tmp_path / "pf_csv"),
            merge_kernels=True,
            kernel_summary_baseline="compute",
        )
        assert (tmp_path / "pf_csv" / "category_summary.csv").exists()

        for rank in (0, 1):
            (tmp_path / f"rank{rank}_trace.json").write_text(
                json.dumps(
                    {
                        "traceEvents": [
                            {
                                "ph": "X",
                                "cat": "kernel",
                                "name": "ncclKernel_AllReduce",
                                "pid": rank,
                                "tid": 3,
                                "ts": 1000 + rank,
                                "dur": 40,
                                "args": {
                                    "External id": 10 + rank,
                                    "Collective name": "allreduce",
                                    "stream": 3,
                                    "collective_id": rank,
                                },
                            }
                        ]
                    }
                )
            )
        dfs = generate_collective_report(
            trace_dir=str(tmp_path),
            world_size=2,
            output_csvs_dir=str(tmp_path / "coll"),
            use_multiprocessing=False,
            strict_world_size_check=False,
            all2allv_heatmap=False,
        )
        assert isinstance(dfs, dict)

    def test_inference_report_all_flags(self, tmp_path):
        trace = tmp_path / "trace.json"
        trace.write_text(
            json.dumps(
                _build_synthetic_trace(
                    [
                        ("aten::mm", "gemm_kernel", 100),
                        ("aten::add", "vectorized_elementwise_kernel", 20),
                        (
                            "aten::_scaled_dot_product_flash_attention",
                            "flash_fwd_kernel",
                            80,
                        ),
                    ]
                )
            )
        )
        result = generate_inference_report(
            profile_json_path=str(trace),
            output_csvs_dir=str(tmp_path / "inf"),
            output_xlsx_path=str(tmp_path / "inf.xlsx"),
            collective_analysis=True,
            kernel_summary=True,
            short_kernel_study=True,
            include_overlap_info=True,
            group_by_parent_module=True,
            group_by_num_kernels=True,
            topk_ops=5,
            topk_roofline_ops=3,
            topk_short_kernels=2,
            include_unlinked_kernels=True,
            include_call_stack=True,
            micro_idle_thresh_us=1,
        )
        assert "gpu_timeline" in result

    def test_attention_extensions_remaining(self):
        from TraceLens.PerfModel.extensions import (
            attention_perf_model_extensions as aext,
        )

        base = {
            "annotation": _GDN_ANNOTATION,
            "args": {
                "Input Dims": [[64, 8, 64], [64, 8, 64], [64, 8, 128]],
                "Input type": ["c10::BFloat16"] * 3,
            },
        }
        for cls in (
            aext.pa_decode_gluon,
            aext.pa_sparse_prefill_opus_fwd,
            aext.pseudo_v4_paged_decode_hca,
            aext.pseudo_v4_paged_decode_csa,
        ):
            model = cls(base)
            assert model.bytes() is None or model.bytes() >= 0


class TestPush95Phase3:
    def test_tracediff_perf_summary_branches(self):
        from TraceLens.Reporting import tracediff_comparison_extension as tde

        assert tde.tracediff_perf_summary_from_diff_stats(pd.DataFrame()).empty
        diff = pd.DataFrame(
            {
                "source": ["trace1", "trace2"],
                "lowest_common_ancestor_id": [1, 1],
                "lowest_common_ancestor_name": ["aten::mm", "aten::mm"],
                "cpu_op_name": ["aten::mm", "aten::add"],
                "busy_time": [100.0, 80.0],
                "name": ["k1", "k2"],
                "gpu_op_uid": [10, 20],
                "nn_module_stack": ["[]", "[]"],
                "nn_module_parent": ["", ""],
                "Input Dims": ["[[2,3]]", "[[2,3]]"],
                "Input type": ["['fp16']", "['fp16']"],
                "Input Strides": ["[]", "[]"],
                "Concrete Inputs": ["", ""],
            }
        )
        summary = tde.tracediff_perf_summary_from_diff_stats(diff)
        assert not summary.empty

        multi = pd.DataFrame(
            {
                "source": ["trace1", "trace1"],
                "lowest_common_ancestor_id": [2, 2],
                "lowest_common_ancestor_name": ["block", "block"],
                "cpu_op_name": ["aten::mm", "aten::relu"],
                "busy_time": [50.0, 30.0],
                "name": ["k1", "k2"],
                "gpu_op_uid": [1, 2],
                "nn_module_stack": ["[]", "[]"],
                "nn_module_parent": ["", ""],
                "Input Dims": ["[[2,3]]", "[[2,3]]"],
                "Input type": ["['fp16']", "['fp16']"],
                "Input Strides": ["[]", "[]"],
                "Concrete Inputs": ["", ""],
            }
        )
        assert (
            " | " in tde.tracediff_perf_summary_from_diff_stats(multi).iloc[0]["name"]
        )

    def test_kernel_fusion_impact_pipeline(self, tmp_path):
        from TraceLens.Agent.Analysis.category_analyses import (
            kernel_fusion_analysis as kfa,
        )

        csv_dir = tmp_path / "perf_report_csvs"
        csv_dir.mkdir()
        pd.DataFrame(
            {
                "kernel_details_summary": [
                    "[{'name': 'Cijk_a'}]",
                    "[{'name': 'ew_add'}]",
                ],
                "op category": ["GEMM", "elementwise"],
                "Data Moved (MB)": [10.0, 2.0],
                "perf_params": ["{'M':2}", "{}"],
                "Input Dims": ["[[2,3]]", "[[4,4]]"],
            }
        ).to_csv(csv_dir / "unified_perf_summary.csv", index=False)

        cat_dir = tmp_path / "category_data"
        cat_dir.mkdir()
        (cat_dir / "category_manifest.json").write_text(
            json.dumps(
                {
                    "platform": "MI300X",
                    "gpu_utilization": {"total_time_ms": 1000.0},
                }
            )
        )
        (cat_dir / "fusion_candidates.json").write_text(
            json.dumps(
                [
                    {
                        "module_name": "nn.Module: Block",
                        "kernels": [
                            {
                                "name": "Cijk_a",
                                "type": "GEMM",
                                "dur_us": 100,
                                "data_in_mb": 10.0,
                            },
                            {
                                "name": "ew_add",
                                "type": "Elementwise Add",
                                "dur_us": 20,
                                "data_in_mb": 2.0,
                            },
                        ],
                        "instance_count": 1,
                    }
                ]
            )
        )
        (cat_dir / "arch_config.json").write_text(
            json.dumps(
                {
                    "peak_hbm_bw_tbs": 5.3,
                    "max_achievable_tflops": {
                        "matrix_bf16": 1000.0,
                        "vector_fp32": 100.0,
                    },
                }
            )
        )

        candidates, manifest, csv_path = kfa.load_fusion_data(str(tmp_path))
        lookup = kfa.build_kernel_perf_lookup(csv_path)
        estimates = kfa.compute_fusion_impact_estimates(
            candidates,
            lookup,
            peak_bw_tbs=5.3,
            peak_maf_tflops={"matrix_bf16": 1000.0, "vector_fp32": 100.0},
            baseline_ms=1000.0,
            is_comparative=False,
        )
        assert isinstance(estimates, list)

        mod = importlib.import_module(
            "TraceLens.Agent.Analysis.category_analyses.kernel_fusion_analysis"
        )
        old_argv = sys.argv
        sys.argv = [
            "kernel_fusion_analysis",
            "--output-dir",
            str(tmp_path),
            "--comparison-scope",
            "standalone",
        ]
        try:
            mod.main()
        finally:
            sys.argv = old_argv
        assert (cat_dir / "kernel_fusion_metrics.json").exists()

    def test_pftrace_cli_mains(self, tmp_path):
        from tests.test_pftrace_memory_copy_report import _make_memory_copy_events
        from tests.test_reporting_coverage import _minimal_pftrace_events

        hip_api = importlib.import_module(
            "TraceLens.Reporting.generate_perf_report_pftrace_hip_api"
        )
        trace_path = tmp_path / "hip_api.json"
        trace_path.write_text(json.dumps({"traceEvents": _minimal_pftrace_events()}))
        old_argv = sys.argv
        sys.argv = [
            "generate_perf_report_pftrace_hip_api",
            "--trace_path",
            str(trace_path),
            "--output_csvs_dir",
            str(tmp_path / "hip_api_csv"),
        ]
        try:
            hip_api.main()
        finally:
            sys.argv = old_argv
        assert (tmp_path / "hip_api_csv" / "api_kernel_summary.csv").exists()

        mem_copy = importlib.import_module(
            "TraceLens.Reporting.generate_perf_report_pftrace_memory_copy"
        )
        mem_path = tmp_path / "mem.json"
        mem_path.write_text(json.dumps({"traceEvents": _make_memory_copy_events()}))
        sys.argv = [
            "generate_perf_report_pftrace_memory_copy",
            "--trace_path",
            str(mem_path),
            "--output_csvs_dir",
            str(tmp_path / "mem_csv"),
        ]
        try:
            mem_copy.main()
        finally:
            sys.argv = old_argv
        assert any(f.endswith(".csv") for f in os.listdir(tmp_path / "mem_csv"))

    def test_llama_fsdp_traces_treeperf(self):
        fsdp_dir = os.path.join(TRACES_ROOT, "mi300/llama_70b_fsdp")
        if not os.path.isdir(fsdp_dir):
            pytest.skip("fsdp traces missing")
        trace = os.path.join(fsdp_dir, "rank0_trace_no_pyfn.json.gz")
        if not os.path.isfile(trace):
            pytest.skip("rank0 trace missing")
        analyzer = TreePerfAnalyzer.from_file(
            trace, rebuild_tree=True, enable_pseudo_ops=True, add_python_func=False
        )
        unified = analyzer.build_df_unified_perf_table(include_nccl=True)
        assert isinstance(unified, pd.DataFrame)

    def test_perf_model_tex_gemm_and_reduce_edges(self):
        input_dims = [()] * 19
        input_dims[0] = [128, 64]
        input_dims[5] = [256, 64]
        input_dims[10] = [128, 256]
        tex = perf_model.tex_ts_te_gemm_ts(
            {
                "args": {
                    "Input Dims": input_dims,
                    "Input type": ["c10::Float8_e4m3fn"] * 19,
                    "Concrete Inputs": [""] * 4
                    + ["1"]
                    + [""] * 4
                    + ["1"]
                    + [""] * 4
                    + ["bias"],
                }
            }
        )
        assert tex.flops() > 0
        with pytest.raises(NotImplementedError):
            tex.flops_bwd()

        mean_evt = {
            "name": "aten::mean",
            "args": {
                "Input Dims": [(4, 256)],
                "Input type": ["c10::BFloat16"],
                "Output type": ["c10::BFloat16"],
                "Concrete Inputs": ["", "[1]", "True"],
            },
        }
        assert perf_model.aten_reduce(mean_evt).flops() > 0


class TestPush95Phase4:
    def test_orchestrator_main_real_fusion_extraction(self, tmp_path, monkeypatch):
        from TraceLens.Agent.Analysis.utils import orchestrator_prepare as op

        out = str(tmp_path)
        csv_dir = os.path.join(out, "perf_report_csvs")
        os.makedirs(csv_dir)
        pd.DataFrame(
            {
                "type": ["total_time", "computation_time", "idle_time"],
                "time ms": [1000.0, 900.0, 100.0],
                "percent": [100.0, 90.0, 10.0],
            }
        ).to_csv(os.path.join(csv_dir, "gpu_timeline.csv"), index=False)
        pd.DataFrame(
            {
                "name": ["aten::mm"],
                "total_direct_kernel_time_ms": [800.0],
                "op category": ["GEMM"],
            }
        ).to_csv(os.path.join(csv_dir, "ops_summary.csv"), index=False)
        pd.DataFrame(
            {
                "name": ["aten::mm"],
                "op category": ["GEMM"],
                "Kernel Time (µs)_sum": [800000.0],
                "kernel_details_summary": ["[{'name': 'Cijk_a'}]"],
                "Data Moved (MB)": [10.0],
                "perf_params": ["{}"],
                "Input Dims": ["[[2,3]]"],
            }
        ).to_csv(os.path.join(csv_dir, "unified_perf_summary.csv"), index=False)
        pd.DataFrame({"name": ["aten::mm"], "op category": ["GEMM"]}).to_csv(
            os.path.join(csv_dir, "ops_summary_by_category.csv"), index=False
        )

        k1 = _kernel_event(10, "Cijk_a", dur=500)
        k2 = _kernel_event(11, "vectorized_elementwise_kernel add", dur=300)
        module = {
            "name": "nn.Module: MLP_0",
            "_category": "aten",
            "gpu_events": [10, 11],
            "args": {"Input Dims": "[[2,3]]"},
        }
        tree = _StubTree([module], {10: k1, 11: k2})
        analyzer = _StubAnalyzer(tree)

        class _FakeTreePerfAnalyzer:
            @classmethod
            def from_file(cls, *args, **kwargs):
                return analyzer

        monkeypatch.setattr(op, "TreePerfAnalyzer", _FakeTreePerfAnalyzer)

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
        fusion_path = os.path.join(out, "category_data", "fusion_candidates.json")
        assert os.path.isfile(fusion_path)
        assert isinstance(json.loads(open(fusion_path).read()), list)

    def test_orchestrator_comparative_fusion_via_main(self, tmp_path, monkeypatch):
        from TraceLens.Agent.Analysis.utils import orchestrator_prepare as op

        out = str(tmp_path)
        _write_minimal_orchestrator_csvs(out, comparative=True)
        t1_csv = os.path.join(out, "perf_report_trace1_csvs")
        pd.DataFrame(
            {
                "name": ["Cijk_A", "ew_add"],
                "source": ["trace1", "trace1"],
                "lowest_common_ancestor_id": [100, 100],
                "kernel_time": [5000.0, 3000.0],
                "gpu_op_uid": [10, 11],
            }
        ).to_csv(os.path.join(t1_csv, "diff_stats.csv"), index=False)

        k1 = _kernel_event(10, "Cijk_A", dur=500)
        k2 = _kernel_event(11, "ew_add", dur=300)
        module = {
            "name": "nn.Module: Attn_0",
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

        old_argv = sys.argv
        sys.argv = [
            "orchestrator_prepare",
            "--trace-path",
            "/fake/trace.json",
            "--platform",
            "MI300X",
            "--output-dir",
            out,
            "--comparison-scope",
            "comparative",
        ]
        try:
            op.main()
        finally:
            sys.argv = old_argv
        fusion = json.loads(
            open(os.path.join(out, "category_data", "fusion_candidates.json")).read()
        )
        assert isinstance(fusion, list)

    @pytest.mark.parametrize(
        "rank",
        list(range(8)),
    )
    def test_llama_fsdp_all_ranks(self, rank):
        trace = os.path.join(
            TRACES_ROOT, "mi300/llama_70b_fsdp", f"rank{rank}_trace_no_pyfn.json.gz"
        )
        if not os.path.isfile(trace):
            pytest.skip("fsdp trace missing")
        analyzer = TreePerfAnalyzer.from_file(
            trace, rebuild_tree=True, enable_pseudo_ops=True
        )
        launchers = analyzer.get_df_kernel_launchers(include_args=True)
        assert isinstance(launchers, pd.DataFrame)

    def test_inference_comparison_report(self, tmp_path):
        trace1 = _write_trace(tmp_path, [("aten::mm", "gemm_kernel", 100)], "t1.json")
        trace2 = _write_trace(tmp_path, [("aten::mm", "gemm_kernel", 120)], "t2.json")
        result = generate_inference_report(
            profile_json_path=trace1,
            comparison_json_path=trace2,
            output_csvs_dir=str(tmp_path / "cmp"),
            output_xlsx_path=str(tmp_path / "cmp.xlsx"),
            collective_analysis=True,
            include_overlap_info=True,
            kernel_summary=True,
            short_kernel_study=True,
            group_by_parent_module=True,
        )
        assert "gpu_timeline" in result
        assert (tmp_path / "cmp" / "gpu_timeline.csv").exists()

    @pytest.mark.parametrize(
        "trace_name",
        [
            "traces/mi300/resnet_act_checkpoint.json.gz",
            "traces/mi300/Qwen_Qwen1.5-0.5B-Chat__1016005.json.gz",
            "traces/torch_compile_triton/trace.json.gz",
        ],
    )
    def test_key_traces_extended_dfs(self, trace_name):
        path = os.path.join(TESTS_DIR, trace_name)
        if not os.path.isfile(path):
            pytest.skip("trace missing")
        analyzer = TreePerfAnalyzer.from_file(
            path, rebuild_tree=True, enable_pseudo_ops=True, add_python_func=True
        )
        perf = analyzer.build_df_perf_metrics(
            events=[e for e in analyzer.tree.events if e.get("cat") == "cpu_op"][:20]
        )
        assert isinstance(perf, pd.DataFrame)
        if analyzer.add_python_func:
            nn = [
                e
                for e in analyzer.tree.events
                if str(e.get("name", "")).startswith("nn.Module")
            ]
            if nn:
                analyzer.build_nn_module_latency_tree(nn[0])


# ---------------------------------------------------------------------------
# Phase 5 — orchestrator sync/memcpy (see also test_coverage_95_phase6.py)
# ---------------------------------------------------------------------------


class TestPush95Phase5:
    def test_orchestrator_sync_bottleneck_via_phase6(self, tmp_path, monkeypatch):
        from tests.test_coverage_95_phase6 import TestOrchestratorPhase6

        TestOrchestratorPhase6().test_main_no_time_column_sync_and_memcpy_dirs(
            tmp_path, monkeypatch
        )

    def test_rocprof_categorize_branches(self):
        from TraceLens.Reporting.rocprof_analysis import _categorize_kernel

        assert _categorize_kernel("conv2d_fwd") == "Convolution"
        assert _categorize_kernel("layer_norm") == "Normalization"
        assert _categorize_kernel("flash_attn") == "Attention"
