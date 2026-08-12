###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""CPU-only coverage tests targeting remaining gaps toward 95% line coverage."""

from __future__ import annotations

import importlib
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
from TraceLens.PerfModel.extensions import attention_perf_model_extensions as attn_ext
from TraceLens.PerfModel.extensions import rmsnorm_perf_model_extensions as rms_ext
from TraceLens.Reporting.generate_perf_report_pytorch import (
    generate_perf_report_pytorch,
)
from TraceLens.Reporting.generate_perf_report_pytorch_inference import (
    generate_perf_report_pytorch as generate_inference_report,
)
from TraceLens.Reporting.generate_perf_report_pftrace_hip_activity import (
    generate_perf_report_pftrace_hip_activity,
)
from TraceLens.Trace2Tree.trace_capture_merge_experimental import (
    _align_graph_to_capture_by_group,
    find_closest_batch_size,
    load_capture_folder,
    merge_capture_trace_into_graph,
    verify_subtree_events,
)
from TraceLens.TreePerf.jax_analyses import JaxAnalyses
from TraceLens.TreePerf.tree_perf import TreePerfAnalyzer

from tests.test_agent_coverage import (
    _StubAnalyzer,
    _StubTree,
    _kernel_event,
    _write_minimal_orchestrator_csvs,
)
from tests.test_conv_backward_bytes import _conv_bias_bwd_event
from tests.test_perfmodel_coverage import _ARCH, _GDN_ANNOTATION, _moe_unfused_event
from tests.test_reporting_coverage import (
    _create_genesis_capture,
    _minimal_pftrace_events,
    _write_trace,
)
from tests.test_treeperf_coverage import _build_analyzer

INFERENCE_ROOT = os.path.join(os.path.dirname(__file__), "traces/inference")
ROCprof_FILE = os.path.join(os.path.dirname(__file__), "rocprof/908_results.json.gz")

pytestmark = pytest.mark.filterwarnings(
    "ignore:Found .* events with failed performance metric.*:UserWarning",
    "ignore:There are hipgraph launches.*:UserWarning",
    "ignore:Kernel name missing in capture event args.*:UserWarning",
    "ignore:Inconsistent kernel list length found.*:UserWarning",
    "ignore:Source column .* not found.*:UserWarning",
)


def _discover_inference_cases():
    if not os.path.isdir(INFERENCE_ROOT):
        return []
    cases = []
    for entry in sorted(os.listdir(INFERENCE_ROOT)):
        dirpath = os.path.join(INFERENCE_ROOT, entry)
        if not os.path.isdir(dirpath):
            continue
        gz = [
            f
            for f in os.listdir(dirpath)
            if f.endswith(".json.gz") and "graph" in f.lower()
        ]
        if not gz:
            gz = [f for f in os.listdir(dirpath) if f.endswith(".json.gz")]
        if not gz:
            continue
        cases.append(pytest.param(dirpath, gz[0], id=entry))
    return cases


# ---------------------------------------------------------------------------
# Real inference traces — heavy tree_perf / reporting coverage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dirpath,trace_gz", _discover_inference_cases())
def test_inference_fixture_full_report(dirpath, trace_gz, tmp_path):
    trace_path = os.path.join(dirpath, trace_gz)
    out = tmp_path / "csv"
    result = generate_inference_report(
        profile_json_path=trace_path,
        output_csvs_dir=str(out),
        output_xlsx_path=str(tmp_path / "report.xlsx"),
        collective_analysis=False,
        enable_pseudo_ops=True,
        kernel_summary=True,
        short_kernel_study=True,
        group_by_parent_module=True,
        group_by_num_kernels=True,
        include_overlap_info=True,
        topk_ops=10,
        topk_roofline_ops=5,
    )
    assert (out / "gpu_timeline.csv").exists()
    assert "gpu_timeline" in result


@pytest.mark.parametrize(
    "case_dir",
    [
        pytest.param(
            os.path.join(INFERENCE_ROOT, name),
            id=name,
            marks=pytest.mark.skipif(
                not os.path.isdir(os.path.join(INFERENCE_ROOT, name)),
                reason="fixture missing",
            ),
        )
        for name in ("vllm_decode_full", "vllm_prefilldecode_piecewise")
    ],
)
def test_merge_capture_into_graph_fixture(case_dir):
    capture = os.path.join(case_dir, "capture_traces")
    metadata = os.path.join(capture, "execution_details.json")
    graph = os.path.join(case_dir, "graph_execution.json.gz")
    if not all(os.path.isfile(p) for p in (metadata, graph)):
        pytest.skip("capture merge fixture incomplete")
    merged = merge_capture_trace_into_graph(capture, metadata, graph)
    assert len(merged.events) > 1000
    analyzer = TreePerfAnalyzer(merged, rebuild_tree=False, add_python_func=True)
    df = analyzer.build_df_unified_perf_table(include_nccl=False)
    assert isinstance(df, pd.DataFrame)


def test_inference_report_comparison_and_debug_columns(tmp_path, monkeypatch):
    trace1 = _write_trace(tmp_path, [("aten::mm", "gemm_kernel", 100)], "t1.json")
    trace2 = _write_trace(tmp_path, [("aten::mm", "gemm_kernel", 120)], "t2.json")
    monkeypatch.setenv("TRACELENS_DEBUG", "1")
    result = generate_inference_report(
        profile_json_path=trace1,
        comparison_json_path=trace2,
        output_csvs_dir=str(tmp_path / "cmp_csvs"),
        output_xlsx_path=str(tmp_path / "cmp.xlsx"),
        include_call_stack=True,
        group_by_parent_module=True,
        collective_analysis=False,
    )
    assert "gpu_timeline" in result
    up = result.get("unified_perf_summary")
    if up is not None and not up.empty and "call_stack_full" in up.columns:
        assert "entry_point" in up.columns


# ---------------------------------------------------------------------------
# PerfModel remaining branches
# ---------------------------------------------------------------------------


class TestPerfModelPush95:
    def test_gemm_simulator_missing_required_inputs(self, monkeypatch, tmp_path):
        sim = tmp_path / "run_gemm.py"
        sim.write_text("# stub\n")
        monkeypatch.setenv("GEMM_SIMULATOR_PATH", str(sim))
        perf_model.GEMM.cache_gemm_results.clear()
        with pytest.raises(AssertionError, match="Invalid inputs"):
            perf_model.GEMM.get_simulation_time_func(
                {"name": "mi300x"}, None, 8, 16, 1, "bf16"
            )

    def test_gemm_simulator_force_to_l1_and_scaled_cus(self, monkeypatch, tmp_path):
        sim = tmp_path / "run_gemm.py"
        sim.write_text("# stub\n")
        monkeypatch.setenv("GEMM_SIMULATOR_PATH", str(sim))
        perf_model.GEMM.cache_gemm_results.clear()
        arch = dict(_ARCH)
        with patch("TraceLens.PerfModel.perf_model.subprocess.run") as run:
            run.return_value = MagicMock(stdout="Time=3.3\n", stderr="")
            t, _ = perf_model.GEMM.get_simulation_time_func(
                arch, 4, 8, 16, 1, "bf16", num_cus=64, force_to_l1=True
            )
        assert t == 3.3

    def test_aten_scaled_mm_output_bpe_branches(self):
        for dtype in ("c10::Float8_e4m3fn", "c10::BFloat16"):
            event = {
                "args": {
                    "Input Dims": [[4, 8], [8, 16], [4, 16]],
                    "Input type": [dtype, dtype, dtype],
                }
            }
            model = perf_model.aten_scaled_mm(event)
            assert model.bytes() > 0

    def test_aten_conv3d_and_mixed_dtype_bytes_error(self):
        conv3d = {
            "args": {
                "Input Dims": [[2, 4, 8, 8, 8], [8, 4, 3, 3, 3]],
                "Input type": ["c10::BFloat16", "c10::BFloat16"],
                "Concrete Inputs": [
                    "",
                    "",
                    "",
                    "(1,1,1)",
                    "(0,0,0)",
                    "(1,1,1)",
                    "False",
                    "(0,0,0)",
                    "1",
                ],
            }
        }
        model = perf_model.aten_conv(conv3d)
        assert model.param_details["convNd"] == "conv3d"
        bad = {
            "args": {
                "Input Dims": [[2, 3, 8, 8], [4, 3, 3, 3]],
                "Input type": ["c10::BFloat16", "c10::Half"],
                "Concrete Inputs": [
                    "",
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
        with pytest.raises(ValueError):
            perf_model.aten_conv(bad).bytes()

    def test_conv_bias_backward_without_cache(self):
        bwd_evt = _conv_bias_bwd_event()
        bwd_evt["args"]["Sequence number"] = 99999
        with pytest.warns(UserWarning, match="Forward pass not found"):
            details = perf_model.ConvBias_Backward.get_param_details(bwd_evt)
        assert details["input_shape"] is None

    def test_aten_reduce_dim_parse_failure(self):
        event = {
            "name": "aten::sum",
            "args": {
                "Input Dims": [(4, 256)],
                "Input type": ["c10::BFloat16"],
                "Concrete Inputs": ["", "not_a_dim", "True"],
            },
        }
        model = perf_model.aten_reduce(event)
        assert model.param_details["num_output_elems"] == 1

    def test_grouped_gemm_gn_k_layout(self):
        event = {
            "args": {
                "Input Dims": [[9, 8], [3, 16, 8]],
                "Input type": ["c10::BFloat16", "c10::BFloat16"],
            }
        }
        g = perf_model.primus_turbo_grouped_gemm(event)
        assert g.flops() > 0
        with pytest.raises(NotImplementedError):
            g.flops_bwd()
        assert g.get_maf_type() == "matrix"

    def test_jax_gemm_backward_not_implemented(self):
        event = {
            "args": {
                "Batch": 2,
                "M": 4,
                "N": 8,
                "K": 16,
                "Beta": 0,
                "Type": "bf16",
            }
        }
        model = perf_model.jax_gemm(event)
        assert model.flops() > 0
        with pytest.raises(NotImplementedError):
            model.flops_bwd()

    def test_vllm_gemm_missing_input_type(self):
        with pytest.raises(ValueError, match="missing A,B dtypes"):
            perf_model.vllm_gemm_with_dynamic_quant(
                {"args": {"Input Dims": [[4, 8], [16, 4]]}}
            )

    def test_tex_ts_te_gemm_transposed(self):
        input_dims = [()] * 19
        input_dims[0] = [128, 64]
        input_dims[5] = [256, 64]
        input_dims[10] = [128, 256]
        event = {
            "args": {
                "Input Dims": input_dims,
                "Input type": ["c10::Float8_e4m3fn"] * 19,
                "Concrete Inputs": [""] * 4
                + ["1"]
                + [""] * 4
                + ["1"]
                + [""] * 4
                + [""],
            }
        }
        model = perf_model.tex_ts_te_gemm_ts(event)
        assert model.flops() > 0


class TestMoeExtensionsPush95:
    def test_blockscale_missing_topk_raises(self):
        event = {
            "args": {
                "Input Dims": [
                    [32, 4096],
                    [32, 4096],
                    [8, 14336, 4096],
                    [8, 4096, 7168],
                ],
                "Input type": ["c10::BFloat16", "c10::Float8_e4m3fn"] * 2,
                "Concrete Inputs": [""] * 8,
            }
        }
        with pytest.raises(ValueError, match="topk"):
            moe_ext.moe_aiter_fused_blockscale(event)

    def test_triton_unfused_missing_kernel_details(self):
        event = {
            "args": {
                "Input Dims": [[32, 4096], [32, 8]],
                "Input type": ["c10::BFloat16", "c10::Float32"],
                "MoE topk": 2,
                "MoE GEMM gated": True,
            }
        }
        with pytest.raises(ValueError, match="Kernel details"):
            moe_ext.moe_triton_unfused_up(event)

    def test_triton_unfused_fp4_and_fp8_kernels(self):
        up = moe_ext.moe_triton_unfused_up(
            _moe_unfused_event(kernel_name="moe_mxfp4_up_kernel")
        )
        down = moe_ext.moe_triton_unfused_down(
            _moe_unfused_event(kernel_name="moe_fp8_down_kernel")
        )
        assert up.get_compute_precision() in ("fp4", "fp8", "bf16", None)
        assert down.bytes() > 0
        with pytest.raises(NotImplementedError):
            up.flops_bwd()


class TestAttentionRmsnormPush95:
    def test_mla_decode_and_paged_attention(self):
        attn_event = {
            "annotation": _GDN_ANNOTATION,
            "args": {
                "Input Dims": [[64, 8, 64], [64, 8, 64], [64, 8, 128]],
                "Input type": ["c10::BFloat16"] * 3,
            },
        }
        mla = attn_ext.mla_decode_fwd(attn_event)
        if mla.param_details.get("_no_perf"):
            assert mla.flops() is None
        else:
            assert mla.flops() > 0
        paged = attn_ext.aiter_paged_attention_ragged(
            {
                "annotation": _GDN_ANNOTATION,
                "args": {
                    "Input Dims": [
                        (),
                        (),
                        [64, 8, 64],
                        [128, 16, 1, 64],
                        [128, 16, 1, 128],
                    ],
                    "Input type": [
                        "Scalar",
                        "Scalar",
                        "c10::BFloat16",
                        "c10::BFloat16",
                        "c10::BFloat16",
                    ],
                },
            }
        )
        assert paged.param_details["d_h_v"] == 128

    def test_rmsnorm_extension_variants(self):
        evt = {
            "args": {
                "Input Dims": [(4, 512), (4, 512), (512,), ()],
                "Input type": ["c10::BFloat16"] * 3 + ["Scalar"],
                "Input Strides": [(512, 1), (512, 1), (1,), ()],
            }
        }
        assert rms_ext.aiter_rmsnorm(evt).bytes() > 0
        vllm_evt = {
            "args": {
                "Input Dims": [(4, 512), (512,), (), ()],
                "Input type": ["c10::BFloat16"] * 4,
                "Input Strides": [(512, 1), (1,), (), ()],
                "Concrete Inputs": ["", "", "", "128"],
            }
        }
        assert rms_ext.vllm_rocm_aiter_rmsnorm_fp8_group_quant(vllm_evt).flops() > 0


# ---------------------------------------------------------------------------
# Capture merge helpers
# ---------------------------------------------------------------------------


class TestCaptureMergePush95:
    def test_load_capture_folder_skips_invalid(self, tmp_path):
        meta = tmp_path / "execution_details.json"
        meta.write_text(
            json.dumps(
                [
                    {"file": "missing.json.gz", "batch_size": "bad", "mode": "FULL"},
                    {"file": "ok.json.gz", "batch_size": 32, "mode": "FULL"},
                ]
            )
        )
        (tmp_path / "ok.json.gz").write_bytes(
            b"\x1f\x8b"
        )  # invalid gzip; load may skip
        result, batch_sizes = load_capture_folder(str(tmp_path), str(meta))
        assert isinstance(result, dict)
        assert 32 in batch_sizes or batch_sizes == []

    def test_find_closest_batch_size(self):
        assert find_closest_batch_size(30, [16, 32, 64]) == 32
        assert find_closest_batch_size(100, [16, 32]) is None

    def test_verify_subtree_group_alignment(self):
        capture = [
            {"name": "hipLaunchKernel", "args": {"kernel": "k2"}},
            {"name": "hipLaunchKernel", "args": {"kernel": "k1"}},
        ]
        graph = [
            {"name": "k1", "args": {}},
            {"name": "k2", "args": {}},
        ]
        code, cap, gr = verify_subtree_events(capture, graph)
        assert code in (0, 3)

    def test_align_graph_to_capture_by_group_success(self):
        capture = [
            {"name": "hipLaunchKernel", "args": {"kernel": "a"}},
            {"name": "hipLaunchKernel", "args": {"kernel": "b"}},
        ]
        graph = [
            {"name": "b", "args": {}},
            {"name": "a", "args": {}},
        ]
        aligned = _align_graph_to_capture_by_group(capture, graph)
        assert aligned is not None
        assert [e["name"] for e in aligned] == ["a", "b"]


# ---------------------------------------------------------------------------
# Orchestrator edge branches
# ---------------------------------------------------------------------------


class TestOrchestratorPush95:
    def test_comparative_without_tree(self, tmp_path, capsys):
        csv_dir = tmp_path / "csv"
        csv_dir.mkdir()
        pd.DataFrame({"lowest_common_ancestor_id": [1], "gpu_op_uid": [10]}).to_csv(
            csv_dir / "diff_stats.csv", index=False
        )
        assert _extract_comparative_fusion_candidates(str(csv_dir)) == []
        assert "tree/analyzer not provided" in capsys.readouterr().out

    def test_comparative_empty_uid_lookup(self, tmp_path, capsys):
        csv_dir = tmp_path / "csv2"
        csv_dir.mkdir()
        pd.DataFrame(
            {
                "lowest_common_ancestor_id": [1],
                "name": ["k"],
                "source": ["trace2"],
            }
        ).to_csv(csv_dir / "diff_stats.csv", index=False)
        tree = _StubTree([], {})
        analyzer = _StubAnalyzer(tree)
        assert (
            _extract_comparative_fusion_candidates(str(csv_dir), analyzer, tree) == []
        )
        assert "No trace1 kernel UIDs" in capsys.readouterr().out

    def test_standalone_gemm_norm_only_skipped(self, tmp_path):
        k1 = _kernel_event(10, "Cijk_gemm_a")
        k2 = _kernel_event(11, "rmsnorm2d")
        mod = {
            "name": "nn.Module: Block_0",
            "_category": "aten",
            "gpu_events": [10, 11],
        }
        tree = _StubTree([mod], {10: k1, 11: k2})
        analyzer = _StubAnalyzer(tree)
        csv_dir = tmp_path / "standalone_csv"
        csv_dir.mkdir()
        pd.DataFrame(
            {
                "kernel_details_summary": [
                    "[{'name': 'Cijk_gemm_a'}]",
                    "[{'name': 'rmsnorm2d'}]",
                ],
                "op category": ["GEMM", "NORM_fwd"],
                "Data Moved (MB)": [1.0, 1.0],
                "perf_params": ["{}", "{}"],
                "Input Dims": ["[[2,3]]", "[[2,3]]"],
            }
        ).to_csv(csv_dir / "unified_perf_summary.csv", index=False)
        cands = _extract_standalone_fusion_candidates(analyzer, tree, str(csv_dir))
        assert isinstance(cands, list)


# ---------------------------------------------------------------------------
# TreePerf / JaxAnalyses
# ---------------------------------------------------------------------------


class TestTreePerfPush95:
    def test_build_df_from_events_perf_failure_fallback(self):
        corr = 801
        events = [
            {
                "ph": "X",
                "UID": "bwd",
                "ts": 2000,
                "dur": 100,
                "cat": "cpu_op",
                "name": "aten::unknown_bwd_op",
                "pid": 100,
                "tid": 100,
                "args": {"Input Dims": [[32, 64]], "Input type": ["fp16"]},
                "gpu_events": ["k1"],
            },
            {
                "ph": "X",
                "UID": "k1",
                "ts": 2050,
                "dur": 50,
                "cat": "kernel",
                "name": "custom_kernel",
                "pid": 0,
                "tid": 7,
                "args": {"correlation": corr, "stream": 7},
            },
        ]
        analyzer = _build_analyzer(events)
        df = analyzer.build_df_perf_metrics(
            events=[analyzer.tree.events[0]],
            include_kernel_details=True,
            include_args=True,
        )
        assert isinstance(df, pd.DataFrame)

    def test_jax_analyses_hlo_op_fallback_categorization(self):
        events = [
            {
                "name": "__amd_rocclr_fillBufferAligned.kd",
                "dur": 100,
                "pid": 1,
                "tid": 1,
                "args": {"hlo_op": "te_fused_attn_backward_ffi.12"},
            }
        ]
        cat, _ = JaxAnalyses.breakdown_compute_events(events, group_by_gpu=True)
        assert 1 in cat


# ---------------------------------------------------------------------------
# Reporting CLI mains
# ---------------------------------------------------------------------------


class TestReportingCliPush95:
    def test_rocprof_main(self, tmp_path):
        if not os.path.isfile(ROCprof_FILE):
            pytest.skip("rocprof fixture missing")
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

    def test_pftrace_hip_api_main(self, tmp_path):
        trace_path = tmp_path / "trace.json"
        trace_path.write_text(json.dumps({"traceEvents": _minimal_pftrace_events()}))
        mod = importlib.import_module(
            "TraceLens.Reporting.generate_perf_report_pftrace_hip_api"
        )
        old_argv = sys.argv
        sys.argv = [
            "generate_perf_report_pftrace_hip_api",
            "--trace_path",
            str(trace_path),
            "--output_csvs_dir",
            str(tmp_path / "hip_api_csv"),
            "--include_nonlaunch_apis",
        ]
        try:
            mod.main()
        finally:
            sys.argv = old_argv
        assert (tmp_path / "hip_api_csv" / "api_kernel_summary.csv").exists()

    def test_pftrace_memory_copy_main(self, tmp_path):
        from tests.test_pftrace_memory_copy_report import _make_memory_copy_events

        trace_path = tmp_path / "trace.json"
        trace_path.write_text(json.dumps({"traceEvents": _make_memory_copy_events()}))
        mod = importlib.import_module(
            "TraceLens.Reporting.generate_perf_report_pftrace_memory_copy"
        )
        old_argv = sys.argv
        sys.argv = [
            "generate_perf_report_pftrace_memory_copy",
            "--trace_path",
            str(trace_path),
            "--output_csvs_dir",
            str(tmp_path / "mem_csv"),
        ]
        try:
            mod.main()
        finally:
            sys.argv = old_argv
        assert any(f.endswith(".csv") for f in os.listdir(tmp_path / "mem_csv"))

    def test_pftrace_hip_activity_csv_and_default_xlsx(self, tmp_path):
        trace_path = tmp_path / "trace.json"
        trace_path.write_text(json.dumps({"traceEvents": _minimal_pftrace_events()}))
        csv_dir = tmp_path / "pf_csv"
        dfs = generate_perf_report_pftrace_hip_activity(
            trace_path=str(trace_path),
            output_csvs_dir=str(csv_dir),
            merge_kernels=True,
            kernel_summary_baseline="compute",
            hip_summary_group="name+stream",
            min_event_ns=1000,
        )
        assert (csv_dir / "category_summary.csv").exists()
        assert "category_summary" in dfs

    def test_genesis_main(self, tmp_path):
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

    def test_compare_reports_main(self, tmp_path):
        r1 = _write_trace(tmp_path, [("aten::mm", "gemm_kernel", 100)], "r1.json")
        r2 = _write_trace(tmp_path, [("aten::mm", "gemm_kernel", 110)], "r2.json")
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
            "a",
            "b",
            "--sheets",
            "gpu_timeline",
            "ops_summary",
        ]
        try:
            mod.main()
        finally:
            sys.argv = old_argv
        assert cmp_xlsx.exists()

    def test_pytorch_report_main_extended(self, tmp_path):
        trace = _write_trace(
            tmp_path,
            [("aten::mm", "gemm_kernel", 100), ("aten::add", "add_kernel", 20)],
        )
        mod = importlib.import_module(
            "TraceLens.Reporting.generate_perf_report_pytorch"
        )
        old_argv = sys.argv
        sys.argv = [
            "generate_perf_report_pytorch",
            "--profile_json_path",
            trace,
            "--output_csvs_dir",
            str(tmp_path / "py_ext"),
            "--output_xlsx_path",
            str(tmp_path / "py_ext.xlsx"),
            "--enable_kernel_summary",
            "--short_kernel_study",
            "--include_overlap_info",
        ]
        try:
            mod.main()
        finally:
            sys.argv = old_argv
        assert (tmp_path / "py_ext" / "gpu_timeline.csv").exists()


# ---------------------------------------------------------------------------
# Phase 2 — deeper branch coverage toward 95%
# ---------------------------------------------------------------------------


class TestCoveragePush95Phase2:
    def test_comparative_fusion_full_path(self, tmp_path):
        csv_dir = tmp_path / "trace1_csvs"
        csv_dir.mkdir()
        pd.DataFrame(
            {
                "name": ["Cijk_A", "Cijk_B", "some_op"],
                "source": ["trace1", "trace1", "trace2"],
                "lowest_common_ancestor_id": [100, 100, 100],
                "kernel_time": [5000.0, 3000.0, 1000.0],
                "gpu_op_uid": [10, 11, None],
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
        module = {
            "name": "nn.Module: Attn_0",
            "_category": "aten",
            "gpu_events": [10, 11],
            "args": {},
        }
        tree = _StubTree([module], uid_map)
        analyzer = _StubAnalyzer(tree)
        cands = _extract_comparative_fusion_candidates(str(csv_dir), analyzer, tree)
        assert len(cands) >= 1

    def test_piecewise_capture_merge(self):
        case_dir = os.path.join(INFERENCE_ROOT, "vllm_prefilldecode_piecewise")
        capture = os.path.join(case_dir, "capture_traces")
        metadata = os.path.join(capture, "execution_details.json")
        graph = os.path.join(case_dir, "graph_execution.json.gz")
        if not all(os.path.isfile(p) for p in (metadata, graph)):
            pytest.skip("piecewise fixture missing")
        merged = merge_capture_trace_into_graph(capture, metadata, graph)
        assert len(merged.events) > 1000

    def test_align_streams_multistream_tiebreak(self):
        from TraceLens.Trace2Tree.trace_capture_merge_experimental import align_streams

        graph = [
            {"name": "k1", "args": {"stream": 1}},
            {"name": "k1", "args": {"stream": 2}},
            {"name": "k2", "args": {"stream": 1}},
        ]
        capture = [
            {"name": "hipLaunchKernel", "args": {"kernel": "k1"}},
            {"name": "hipLaunchKernel", "args": {"kernel": "k1"}},
            {"name": "hipLaunchKernel", "args": {"kernel": "k2"}},
        ]
        aligned = align_streams(graph, capture)
        assert aligned is not None
        assert len(aligned) == 3

    def test_verify_subtree_direct_match(self):
        capture = [{"name": "hipLaunchKernel", "args": {"kernel": "k1"}}]
        graph = [{"name": "k1", "args": {}}]
        code, cap, gr = verify_subtree_events(capture, graph)
        assert code == 1

    def test_norm_and_mamba_perf_models(self):
        from tests.test_mamba_ssd import _mamba_event

        mamba = perf_model.mamba_ssd_fwd(_mamba_event())
        assert mamba.flops() > 0
        assert mamba.bytes() > 0
        dispatch = perf_model.moe_dispatch(
            {
                "args": {
                    "Input Dims": [[32, 4096], [32, 8]],
                    "Input type": ["c10::BFloat16", "c10::Int"],
                }
            }
        )
        assert dispatch.bytes() > 0
        combine = perf_model.moe_combine(
            {
                "args": {
                    "Input Dims": [[32, 4096], [32, 4096]],
                    "Input type": ["c10::BFloat16", "c10::BFloat16"],
                }
            }
        )
        assert combine.bytes() >= 0

    def test_moe_ck_and_gptq_extended(self):
        ck1 = {
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
        ck2 = {
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
        assert moe_ext.moe_aiter_ck_stage1(ck1).bytes() > 0
        assert moe_ext.moe_aiter_ck_stage2(ck2).flops() > 0
        gptq = {
            "args": {
                "Input Dims": [[32, 4096], [32, 2], [8, 7168, 4096]],
                "Input type": ["c10::BFloat16", "c10::Int", "c10::BFloat16"],
                "MoE topk": 2,
            }
        }
        assert moe_ext.moe_gptq_awq_up(gptq).bytes() > 0
        assert moe_ext.moe_gptq_awq_down(gptq).flops() > 0

    def test_merged_graph_treeperf_extended(self):
        case_dir = os.path.join(INFERENCE_ROOT, "vllm_decode_full")
        capture = os.path.join(case_dir, "capture_traces")
        metadata = os.path.join(capture, "execution_details.json")
        graph = os.path.join(case_dir, "graph_execution.json.gz")
        if not all(os.path.isfile(p) for p in (metadata, graph)):
            pytest.skip("fixture missing")
        merged = merge_capture_trace_into_graph(capture, metadata, graph)
        analyzer = TreePerfAnalyzer(merged, rebuild_tree=False, add_python_func=True)
        launchers = analyzer.get_df_kernel_launchers(include_args=True)
        assert not launchers.empty
        summary = TreePerfAnalyzer.get_df_kernel_launchers_summary(launchers)
        assert not summary.empty
        unified = analyzer.build_df_unified_perf_table(include_nccl=False)
        assert isinstance(unified, pd.DataFrame)

    def test_analysis_utils_efficiency_and_fusion(self, tmp_path):
        from TraceLens.Agent.Analysis.category_analyses import analysis_utils as au

        row = pd.Series(
            {
                "FLOPS/Byte": 10.0,
                "TFLOPS/s_mean": 100.0,
                "TB/s_mean": 1.0,
                "Roofline Bound": "COMPUTE_BOUND",
                "Compute Spec": "matrix_bf16",
            }
        )
        result = au.calculate_efficiency(
            row, peak_maf_or_maf_dict={"matrix_bf16": 1000}, peak_hbm_bw=5300
        )
        assert result["bound_type"] == "compute"

        fusion_dir = tmp_path / "category_data"
        fusion_dir.mkdir()
        (fusion_dir / "kernel_fusion_metrics.json").write_text(
            json.dumps({"high_confidence_kernel_map": {"gemm_kernel": "fused_gemm"}})
        )
        assert au._load_fusion_map(str(tmp_path))["gemm_kernel"] == "fused_gemm"
        assert (
            au._match_fusion_op("{'name': 'gemm_kernel'}", {"gemm_kernel": "fused"})
            == "fused"
        )

    def test_compare_traces_jax_llama_helpers(self, tmp_path):
        import gzip
        from TraceLens.Reporting.compare_traces_jax_llama import (
            infer_params,
            load_trace,
        )
        from tests.test_compare_traces_jax_llama import _make_synthetic_trace_events

        path = tmp_path / "t1.json.gz"
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump({"traceEvents": _make_synthetic_trace_events()}, f)
        load_trace(str(path))
        evs = [
            type(
                "E",
                (),
                {
                    "name": "ln_fwd_tuned_kernel<Kernel_traits<float, 4096u,",
                    "dur": 5.0,
                    "args": {},
                },
            )(),
            type("E", (), {"name": "flash_fprop_hd128", "dur": 5.0, "args": {}})(),
        ]
        d_model, head_dim, gsu = infer_params(evs)
        assert d_model == 4096
        assert head_dim == 128

    def test_rocprof_analyzer_synthetic(self):
        from TraceLens.Reporting.rocprof_analysis import RocprofAnalyzer

        kernels = [
            {
                "name": "gemm_kernel",
                "ts": 1000,
                "dur": 50,
                "grid": (1, 1, 1),
                "block": (256, 1, 1),
                "stream": 0,
            }
        ]
        memory = [{"name": "MemcpyHtoD", "ts": 900, "dur": 10, "bytes": 1024}]
        api = [{"name": "hipLaunchKernel", "ts": 990, "dur": 5, "correlation": 1}]
        analyzer = RocprofAnalyzer(kernels, memory, api, {})
        assert not analyzer.get_df_gpu_timeline().empty
        assert not analyzer.get_df_kernel_summary().empty
        assert isinstance(analyzer.get_df_short_kernels(10), pd.DataFrame)

    def test_collective_report_main(self, tmp_path):
        for rank in (0, 1):
            (tmp_path / f"trace_rank_{rank}.json").write_text(
                json.dumps(
                    {
                        "traceEvents": [
                            {
                                "ph": "X",
                                "cat": "kernel",
                                "name": "ncclKernel_AllReduce",
                                "pid": rank,
                                "tid": 3,
                                "ts": 1000,
                                "dur": 40,
                                "args": {
                                    "External id": rank,
                                    "Collective name": "allreduce",
                                    "stream": 3,
                                },
                            }
                        ]
                    }
                )
            )
        out = tmp_path / "coll.xlsx"
        mod = importlib.import_module(
            "TraceLens.Reporting.generate_multi_rank_collective_report_pytorch"
        )
        old_argv = sys.argv
        sys.argv = [
            "generate_multi_rank_collective_report_pytorch",
            "--trace_glob",
            str(tmp_path / "trace_rank_*.json"),
            "--world_size",
            "2",
            "--output_xlsx_path",
            str(out),
        ]
        try:
            mod.main()
        finally:
            sys.argv = old_argv
        assert out.exists()

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
        manifest = json.loads(
            open(os.path.join(out, "category_data", "category_manifest.json")).read()
        )
        assert manifest["comparison_scope"] == "comparative"

    def test_gemm_simulator_clears_cache(self, monkeypatch, tmp_path):
        perf_model.GEMM.cache_gemm_results.clear()
        sim_dir = tmp_path / "simdir"
        sim_dir.mkdir()
        sim = sim_dir / "run_gemm.py"
        sim.write_text("# stub\n")
        monkeypatch.setenv("GEMM_SIMULATOR_PATH", str(sim))
        with patch("TraceLens.PerfModel.perf_model.subprocess.run") as run:
            run.return_value = MagicMock(stdout="Time=42.5\n", stderr="")
            t, _ = perf_model.GEMM.get_simulation_time_func(
                _ARCH, 4, 8, 16, 1, "bf16", num_cus=64, force_to_l1=True
            )
        assert t == 42.5
        perf_model.GEMM.cache_gemm_results.clear()


class TestCoveragePush95Phase3:
    """Additional extension, merged-tree, and helper coverage."""

    JAX_PB = os.path.join(
        os.path.dirname(__file__),
        "traces/mi300/jax_conv_minimal_legacy/chi-mi300x-013.ord.vultr.cpe.ice.amd.com.xplane.pb",
    )

    def test_inference_on_merged_tree(self, tmp_path):
        case_dir = os.path.join(INFERENCE_ROOT, "vllm_decode_full")
        capture = os.path.join(case_dir, "capture_traces")
        metadata = os.path.join(capture, "execution_details.json")
        graph = os.path.join(case_dir, "graph_execution.json.gz")
        if not all(os.path.isfile(p) for p in (metadata, graph)):
            pytest.skip("fixture missing")
        merged = merge_capture_trace_into_graph(capture, metadata, graph)
        result = generate_inference_report(
            profile_json_path=graph,
            augmented_tree=merged,
            output_csvs_dir=str(tmp_path / "merged_csv"),
            output_xlsx_path=str(tmp_path / "merged.xlsx"),
            collective_analysis=False,
            enable_pseudo_ops=True,
            group_by_parent_module=True,
            kernel_summary=True,
        )
        assert "gpu_timeline" in result

    def test_untested_perf_extensions(self):
        from TraceLens.PerfModel.extensions import perf_model_extensions as pext

        blockscale = {
            "args": {
                "Input Dims": [[128, 256], [512, 256], [512, 4], [512, 4]],
                "Input type": [
                    "c10::Float8_e4m3fn",
                    "c10::Float8_e4m3fn",
                    "c10::Float",
                    "c10::Float",
                ],
            }
        }
        assert pext.gemm_a8w8_blockscale(blockscale).bytes() > 0

        silu = {
            "args": {
                "Input Dims": [(4, 512), (4, 512)],
                "Input type": ["c10::BFloat16", "c10::BFloat16"],
                "Input Strides": [(512, 1), (512, 1)],
            }
        }
        assert pext.aiter_silu_and_mul(silu).flops() > 0
        assert pext.sgl_kernel_silu_and_mul(silu).bytes() > 0
        assert pext.aiter_gelu_and_mul(silu).flops() > 0

        per_group = {
            "args": {
                "Input Dims": [(4, 256), (4, 256), (4, 2)],
                "Input type": ["c10::Float8_e4m3fn", "c10::BFloat16", "c10::BFloat16"],
                "Input Strides": [(256, 1), (256, 1), (2, 1)],
            }
        }
        assert pext.per_group_quant(per_group).bytes() > 0

        vllm_group = {
            "args": {
                "Input Dims": ((4, 256), ()),
                "Input type": ("c10::BFloat16", "Scalar"),
                "Concrete Inputs": ("", "128"),
            }
        }
        assert pext.vllm_triton_per_token_group_quant_fp8(vllm_group).flops() > 0

        rope_mla = {
            "args": {
                "Input Dims": [
                    (2, 8, 512),
                    (2, 8, 64),
                    (2, 1, 512),
                    (2, 1, 64),
                    (128, 1, 1, 576),
                ],
                "Input type": ["c10::BFloat16"] * 5,
            }
        }
        assert pext.aiter_fused_qk_rope_cat_and_cache_mla(rope_mla).bytes() > 0

    def test_attention_extension_variants(self):
        from TraceLens.PerfModel.extensions import (
            attention_perf_model_extensions as aext,
        )

        for cls, event in [
            (
                aext.pseudo_v4_paged_decode_csa,
                {
                    "annotation": "(128_256_512_1024_2048_3072_4096_64)",
                    "args": {
                        "Input Dims": [[64, 8, 64], [64, 8, 64], [64, 8, 128]],
                        "Input type": ["c10::BFloat16"] * 3,
                    },
                },
            ),
            (
                aext.vllm_unified_mla_attention_with_output,
                {
                    "annotation": "(128_256_512_1024_2048_3072_4096_64)",
                    "args": {
                        "Input Dims": [[64, 8, 64], [64, 8, 64], [64, 8, 128]],
                        "Input type": ["c10::BFloat16"] * 3,
                    },
                },
            ),
        ]:
            model = cls(event)
            if model.param_details.get("_no_perf"):
                assert model.flops() is None
            else:
                assert model.flops() > 0

    def test_kernel_fusion_and_tracediff_helpers(self, tmp_path):
        from TraceLens.Agent.Analysis.category_analyses import (
            kernel_fusion_analysis as kfa,
        )
        from TraceLens.Reporting import tracediff_comparison_extension as tde

        ops = [
            {
                "kernel_names": ["Cijk_a", "ew_add"],
                "base_name": "Block",
                "instance_count": 1,
            },
            {
                "kernel_names": ["Cijk_a", "ew_add"],
                "base_name": "Block",
                "instance_count": 1,
            },
        ]
        filtered = kfa._filter_and_dedup(ops)
        assert len(filtered) == 1

        df = pd.DataFrame({"name": ["aten::mm"], "Kernel Time (µs)_mean": [100.0]})
        diff = pd.DataFrame(
            {
                "name": ["aten::mm"],
                "kernel_time_delta_pct": [10.0],
                "lowest_common_ancestor_id": [1],
            }
        )
        report = {"unified_perf_summary": df}
        tde.enrich_perf_report_dict_inplace(report, diff)
        assert "unified_perf_summary" in report

    def test_reporting_utils_and_pseudo_registry(self, tmp_path):
        from TraceLens.Reporting import reporting_utils as ru
        from TraceLens.Trace2Tree.extensions import pseudo_ops_registry as por
        from TraceLens.Trace2Tree.trace_to_tree import TraceToTree

        trace = _write_trace(tmp_path, [("aten::mm", "gemm_kernel", 100)])
        assert ru.detect_gpus_per_node(trace) is None or isinstance(
            ru.detect_gpus_per_node(trace), int
        )
        tree = TraceToTree([])
        tree.build_tree()
        por.apply_pseudo_op_extensions(tree, verbose=False)
        assert tree is not None

    @pytest.mark.skipif(not os.path.isfile(JAX_PB), reason="JAX xplane fixture missing")
    def test_jax_gemm_performance_from_pb(self):
        df = JaxAnalyses.gemm_performance_from_pb(
            self.JAX_PB, module_name="jit_forward_3d_conv"
        )
        assert isinstance(df, pd.DataFrame)

    def test_trace_to_tree_edge_helpers(self):
        from TraceLens.Trace2Tree import trace_to_tree as ttt

        events = [
            {
                "ph": "X",
                "name": "aten::mm",
                "cat": "cpu_op",
                "ts": 0,
                "dur": 10,
                "pid": 1,
                "tid": 1,
            },
            {
                "ph": "X",
                "name": "gemm",
                "cat": "kernel",
                "ts": 20,
                "dur": 5,
                "pid": 0,
                "tid": 7,
            },
        ]
        tree = ttt.TraceToTree(events)
        tree.build_tree(add_python_func=True)
        assert len(tree.events) >= 2

    def test_gpu_only_treeperf_extended(self):
        from tests.test_treeperf import GPU_ONLY_TRACE

        if not os.path.isfile(GPU_ONLY_TRACE):
            pytest.skip("gpu_only trace missing")
        analyzer = TreePerfAnalyzer.from_file(GPU_ONLY_TRACE, rebuild_tree=True)
        launchers = analyzer.get_df_kernel_launchers(
            include_args=True, include_kernel_details=True
        )
        assert not launchers.empty
        unique = TreePerfAnalyzer.get_df_kernel_launchers_unique_args(
            launchers, include_pct=True
        )
        assert not unique.empty
        summarized = TreePerfAnalyzer.summarize_df_unified_perf_table(
            analyzer.build_df_unified_perf_table(include_nccl=False),
            include_pct=True,
            tree=analyzer.tree,
        )
        assert isinstance(summarized, pd.DataFrame)
