###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Phase-9: perf_model SDPA/conv edges, reporting CLI mains, trace_diff merge."""

from __future__ import annotations

import importlib
import json
import os
import sys

import pandas as pd
import pytest

from TraceLens.PerfModel import perf_model
from TraceLens.TraceDiff.trace_diff import TraceDiff

from tests.test_conv_backward_bytes import _conv_bias_bwd_event, _conv_bias_fwd_event
from tests.test_flash_attention_backward import _bwd_event as _flash_bwd_event
from tests.test_perfmodel_coverage import _ARCH
from tests.test_reporting_coverage import _minimal_pftrace_events, _write_trace
from tests.test_tracediff import TraceDiff as _TD  # noqa: F401 — ensure module loaded
from tests.test_tracediff import _add_gpu_chain, _build_tree, _mk_event

ROCprof_FILE = os.path.join(os.path.dirname(__file__), "rocprof/908_results.json.gz")


class TestPerfModelPhase9:
    def test_extract_sdpa_cfg_errors(self):
        with pytest.raises(ValueError, match="Batch sizes"):
            perf_model.extract_sdpa_cfg([2, 8, 64, 32], [1, 8, 64, 32], [2, 8, 64, 32], (0, 1, 2, 3))
        with pytest.raises(ValueError, match="Head sizes"):
            perf_model.extract_sdpa_cfg([2, 8, 64, 32], [2, 4, 64, 32], [2, 8, 64, 32], (0, 1, 2, 3))
        with pytest.raises(ValueError, match="Length sizes"):
            perf_model.extract_sdpa_cfg([2, 8, 64, 32], [2, 8, 32, 32], [2, 8, 64, 32], (0, 1, 2, 3))
        with pytest.raises(ValueError, match="Head dimensions"):
            perf_model.extract_sdpa_cfg([2, 8, 64, 32], [2, 8, 64, 16], [2, 8, 64, 32], (0, 1, 2, 3))

    def test_extract_sdpa_varlen_cfg_errors(self):
        with pytest.raises(ValueError, match="Head sizes"):
            perf_model.extract_sdpa_varlen_cfg([8, 64, 32], [4, 64, 32], [8, 64, 32], (0, 1, 2))

    def test_sdpa_causal_mismatch_raises(self):
        with pytest.raises(ValueError, match="causal=True"):
            perf_model.SDPA.flops_bwd_func(1, 64, 8, 32, 8, 64, 64, True, True)

    def test_sdpa_varlen_multi_seq_flops(self):
        event = {
            "args": {
                "Input Dims": [
                    [2, 64, 4, 32],
                    [2, 64, 4, 32],
                    [2, 64, 4, 32],
                ],
                "Input type": ["c10::BFloat16"] * 3,
                "Concrete Inputs": ["", "", "", "0.0", "True", "True", "2", "2", "64", "64"],
            }
        }
        model = perf_model.aten__scaled_dot_product_flash_attention(event)
        assert model.flops() > 0
        assert model.bytes() > 0

    def test_cudnn_and_efficient_attention(self):
        base = {
            "args": {
                "Input Dims": [[2, 128, 8, 64], [2, 128, 8, 64], [2, 128, 8, 64]],
                "Input type": ["c10::BFloat16"] * 3,
                "Concrete Inputs": ["", "", "", "0.0", "0.0", "False", "False", ""],
            }
        }
        cudnn = perf_model.aten__scaled_dot_product_cudnn_attention(base)
        assert cudnn.flops() > 0
        efficient = perf_model.aten__scaled_dot_product_efficient_attention(base)
        assert efficient.flops() > 0

    def test_conv_bias_bwd_cached_mixed_dtype(self):
        perf_model.ConvBias_.fwd_pass_cache.clear()
        fwd_evt = _conv_bias_fwd_event()
        perf_model.ConvBias_(fwd_evt)
        bwd_evt = _conv_bias_bwd_event()
        bwd_evt["args"]["Input type"] = ["c10::BFloat16", "c10::Half"]
        bwd = perf_model.ConvBias_Backward(bwd_evt)
        assert bwd.bytes() is None or bwd.bytes() >= 0

    def test_conv_bias_relu_bwd_cached(self):
        perf_model.ConvBiasReLU_.fwd_pass_cache.clear()
        perf_model.ConvBiasReLU_(_conv_bias_fwd_event())
        bwd = perf_model.ConvBiasReLU_Backward(_conv_bias_bwd_event())
        assert bwd.flops_bwd() > 0
        assert bwd.bytes_bwd() > 0

    def test_flash_attention_backward_simulation_none(self):
        from unittest.mock import patch

        model = perf_model.flash_attention_backward(_flash_bwd_event())
        with patch.object(perf_model.GEMM, "get_simulation_time_func", return_value=(None, None)):
            assert model.get_simulation_time() is None


class TestTraceDiffPhase9:
    def test_single_side_gpu_child_collapse_merge(self):
        events1 = [
            _mk_event("cpu_op", "aten::outer", ts=0, dur=100, pid=1, tid=1),
            _mk_event("cpu_op", "aten::inner", ts=10, dur=100, pid=1, tid=1),
        ]
        _add_gpu_chain(events1, events1[1], 100, "gemm_v1", ts_launch=20, ts_kernel=60)
        events2 = [
            _mk_event("cpu_op", "aten::outer", ts=0, dur=100, pid=1, tid=1),
            _mk_event("cpu_op", "aten::inner_a", ts=10, dur=50, pid=1, tid=1),
            _mk_event("cpu_op", "aten::inner_b", ts=60, dur=50, pid=1, tid=1),
        ]
        _add_gpu_chain(events2, events2[1], 200, "gemm_v2a", ts_launch=20, ts_kernel=40)
        _add_gpu_chain(events2, events2[2], 201, "gemm_v2b", ts_launch=70, ts_kernel=40)
        td = TraceDiff(_build_tree(events1), _build_tree(events2))
        td.generate_tracediff_report()
        stats = td.get_diff_stats_df()
        assert stats is None or isinstance(stats, pd.DataFrame)

    def test_trace_only_branch_diff_stats(self):
        events1 = [_mk_event("cpu_op", "aten::mm", ts=0, dur=100, pid=1, tid=1)]
        _add_gpu_chain(events1, events1[0], 100, "gemm_only_t1", ts_launch=10, ts_kernel=50)
        events2 = [_mk_event("cpu_op", "aten::add", ts=0, dur=100, pid=1, tid=1)]
        td = TraceDiff(_build_tree(events1), _build_tree(events2))
        td.generate_tracediff_report()
        stats = td.get_diff_stats_df()
        assert stats is None or isinstance(stats, pd.DataFrame)


class TestReportingCliPhase9:
    def test_generate_perf_report_pytorch_main(self, tmp_path):
        mod = importlib.import_module("TraceLens.Reporting.generate_perf_report_pytorch")
        trace = _write_trace(tmp_path, [("aten::mm", "gemm_kernel", 100)])
        old_argv = sys.argv
        sys.argv = [
            "generate_perf_report_pytorch",
            "--profile_json_path",
            trace,
            "--output_csvs_dir",
            str(tmp_path / "csv"),
            "--output_xlsx_path",
            str(tmp_path / "out.xlsx"),
            "--enable_kernel_summary",
            "--disable_coll_analysis",
        ]
        try:
            mod.main()
        finally:
            sys.argv = old_argv
        assert (tmp_path / "csv" / "gpu_timeline.csv").exists()

    def test_generate_perf_report_inference_main(self, tmp_path):
        mod = importlib.import_module(
            "TraceLens.Reporting.generate_perf_report_pytorch_inference"
        )
        trace = _write_trace(tmp_path, [("aten::mm", "gemm_kernel", 100)], "inf.json")
        old_argv = sys.argv
        sys.argv = [
            "generate_perf_report_pytorch_inference",
            "--profile_json_path",
            trace,
            "--output_csvs_dir",
            str(tmp_path / "csv"),
            "--output_xlsx_path",
            str(tmp_path / "out.xlsx"),
        ]
        try:
            mod.main()
        finally:
            sys.argv = old_argv
        assert (tmp_path / "csv" / "gpu_timeline.csv").exists()

    @pytest.mark.skipif(not os.path.isfile(ROCprof_FILE), reason="rocprof fixture missing")
    def test_generate_multi_rank_collective_main(self, tmp_path):
        mod = importlib.import_module(
            "TraceLens.Reporting.generate_multi_rank_collective_report_pytorch"
        )
        for rank in (0, 1):
            (tmp_path / f"rank{rank}_trace.json").write_text(json.dumps({
                "traceEvents": [{
                    "ph": "X", "cat": "kernel", "name": "ncclKernel_AllReduce",
                    "pid": rank, "tid": 3, "ts": 1000, "dur": 40,
                    "args": {"External id": 10, "stream": 3},
                }]
            }))
        old_argv = sys.argv
        sys.argv = [
            "generate_multi_rank_collective_report_pytorch",
            "--trace_dir",
            str(tmp_path),
            "--world_size",
            "2",
            "--output_csvs_dir",
            str(tmp_path / "coll"),
        ]
        try:
            mod.main()
        finally:
            sys.argv = old_argv
        assert os.path.isdir(tmp_path / "coll")

    def test_pftrace_hip_api_main(self, tmp_path):
        mod = importlib.import_module("TraceLens.Reporting.generate_perf_report_pftrace_hip_api")
        trace_path = tmp_path / "pf.json"
        trace_path.write_text(json.dumps({"traceEvents": _minimal_pftrace_events()}))
        old_argv = sys.argv
        sys.argv = [
            "generate_perf_report_pftrace_hip_api",
            "--trace_path",
            str(trace_path),
            "--output_csvs_dir",
            str(tmp_path / "csv"),
        ]
        try:
            mod.main()
        except SystemExit:
            pass
        finally:
            sys.argv = old_argv


# ---------------------------------------------------------------------------
# Additional coverage: pftrace, arch_utils, run_category_analysis, MoE edges
# ---------------------------------------------------------------------------

import shutil
import urllib.request
from copy import deepcopy
from pathlib import Path
from typing import Dict, List
from unittest.mock import patch

from TraceLens.Agent.Analysis.category_analyses import analysis_utils as au
from TraceLens.Agent.Analysis.utils import arch_utils
from TraceLens.Reporting import pftrace_utils
from TraceLens.Reporting.generate_perf_report_pftrace_hip_activity import (
    _write_markdown_report,
    generate_perf_report_pftrace_hip_activity,
)
from TraceLens.Reporting.pftrace_hip_activity_analysis import (
    Event,
    HIPEvent,
    PftraceHipActivityAnalyzer,
    build_event_lists,
    build_hip_summary_df,
    build_kernel_summary_df_for_config,
    classify,
    extract_time_ns,
    rccl_overlap_two_pointer,
)
from TraceLens.Trace2Tree.extensions.moe_aiter_pseudo_ops import (
    _create_pseudo_op_moe_fused_aiter,
    _has_cpu_op_descendant,
    create_pseudo_ops_moe_fused_aiter,
    is_aiter_fused_moe_kernel,
)
from TraceLens.Trace2Tree.extensions.moe_flydsl_pseudo_ops import (
    FUSED_MOE_PARENT,
    create_pseudo_ops_moe_flydsl,
)
from TraceLens.Trace2Tree.extensions.moe_gptq_awq_pseudo_ops import (
    _create_pseudo_op_moe_gptq_awq,
    _extract_topk_from_outplace,
    create_pseudo_ops_moe_gptq_awq,
)
from TraceLens.Trace2Tree.trace_to_tree import TraceToTree
from TraceLens.TreePerf.tree_perf import TreePerfAnalyzer

from tests.test_treeperf_coverage import _build_analyzer, _make_gpu_event, _mk_ac2g


def _setup_gemm_output_dir(tmp_path):
    out = tmp_path / "analysis_output"
    (out / "category_data").mkdir(parents=True)
    (out / "metadata").mkdir()
    df = pd.DataFrame(
        {
            "name": ["aten::mm", "aten::addmm"],
            "count": [12, 3],
            "Kernel Time (µs)_sum": [100_000.0, 50_000.0],
            "Kernel Time (µs)_mean": [8000.0, 16000.0],
            "Kernel Time (µs)_std": [500.0, 100.0],
            "TFLOPS/s_mean": [400.0, 350.0],
            "TB/s_mean": [0.5, 0.4],
            "FLOPS/Byte": [2000.0, 1800.0],
            "Roofline Bound": ["COMPUTE_BOUND", "COMPUTE_BOUND"],
            "Compute Spec": ["matrix_bf16", "matrix_bf16"],
            "kernel_details_summary": ["[{'name': 'Cijk_a'}]", "[{'name': 'Cijk_b'}]"],
            "call_stack_full": ["['aten::mm', 'Linear']", "['aten::addmm']"],
            "Input Dims": ["[[32, 64], [64, 128]]", "[[32, 64], [64, 128], [32, 128]]"],
            "Input type": ["['fp16', 'fp16']", "['fp16', 'fp16', 'fp16']"],
        }
    )
    df.to_csv(out / "category_data" / "gemm_ops.csv", index=False)
    meta = {
        "platform": "MI300X",
        "peak_hbm_bw_tbs": 5.3,
        "max_achievable_tflops": {"matrix_bf16": 708},
        "gpu_utilization": {"total_time_ms": 1000.0},
        "output_dir": str(out),
    }
    (out / "metadata" / "gemm_metadata.json").write_text(json.dumps(meta))
    return str(out)


def _rich_pftrace_events():
    events = list(_minimal_pftrace_events())
    events.extend(
        [
            {
                "ph": "X", "cat": "gpu_activity", "name": "ncclAllReduce_ring",
                "pid": 0, "tid": 7, "ts": 2000, "dur": 100000,
                "args": {"agent": "gpu_0", "begin_ns": 2_000_000_000, "delta_ns": 100_000_000},
            },
            {
                "ph": "X", "cat": "gpu_activity", "name": "Cijk_A_B_gemm",
                "pid": 0, "tid": 8, "ts": 3000, "dur": 80000,
                "args": {
                    "agent": "gpu_0", "begin_ns": 3_000_000_000, "delta_ns": 80_000_000,
                    "grid_size": 256, "workgroup_size": 256, "VGPR_Count": 32,
                    "stream_ID": 1, "queue": 2,
                },
            },
            {
                "ph": "X", "cat": "gpu_activity", "name": "FmhaBwd_kernel_func_v3",
                "pid": 0, "tid": 7, "ts": 4000, "dur": 60000,
                "args": {"agent": "gpu_0", "begin_ns": 4_000_000_000, "delta_ns": 60_000_000},
            },
            {
                "ph": "X", "cat": "hip_api", "name": "hipMemcpyAsync",
                "pid": 100, "tid": 2, "ts": 850, "dur": 5000,
                "args": {"stream_ID": 1, "operation": 42, "begin_ns": 850_000, "delta_ns": 5_000_000},
            },
        ]
    )
    return events


def _build_moe_tree(events: List[Dict], add_python_func: bool = False) -> TraceToTree:
    tree = TraceToTree(deepcopy(events), prune_nongpu_paths=False)
    tree.build_tree(add_python_func=add_python_func)
    return tree


class TestAnalysisUtilsRunCategoryPhase9:
    def test_run_category_analysis_success(self, tmp_path):
        out = _setup_gemm_output_dir(tmp_path)
        au.run_category_analysis("gemm", out, {}, lambda ops_df, _m: {"n": len(ops_df)})
        metrics = json.loads(Path(out, "category_data", "gemm_metrics.json").read_text())
        assert metrics["status"] == "OK"

    def test_run_category_analysis_no_data(self, tmp_path):
        out = tmp_path / "empty"
        (out / "category_data").mkdir(parents=True)
        au.run_category_analysis(
            "gemm", str(out), {}, lambda _o, _m: {},
            no_data_check_fn=lambda _o, c, _s: {"category": c, "status": "NO_DATA"},
        )
        assert json.loads((out / "category_data" / "gemm_metrics.json").read_text())["status"] == "NO_DATA"

    def test_run_category_analysis_missing_csv_exits(self, tmp_path):
        out = tmp_path / "missing"
        (out / "category_data").mkdir(parents=True)
        with pytest.raises(SystemExit):
            au.run_category_analysis("gemm", str(out), {}, lambda _o, _m: {})


class TestPftraceExtendedPhase9:
    def test_pftrace_utils_branches(self, tmp_path, monkeypatch):
        on_path = tmp_path / "traceconv"
        on_path.write_text("#!/bin/sh\necho ok\n")
        on_path.chmod(0o755)
        with patch.object(shutil, "which", return_value=str(on_path)):
            assert pftrace_utils.acquire_traceconv(tmp_path / "missing", tmp_path).exists()

        def fail_run(cmd, cwd=None):
            raise RuntimeError("curl failed")

        def fake_urlretrieve(_url, target):
            Path(target).write_bytes(b"#!/bin/sh\necho ok\n")
            Path(target).chmod(0o755)

        with patch.object(shutil, "which", return_value=None):
            with patch.object(pftrace_utils, "run", side_effect=fail_run):
                with patch.object(urllib.request, "urlretrieve", side_effect=fake_urlretrieve):
                    assert pftrace_utils.acquire_traceconv(None, tmp_path / "dl").exists()

        pf = tmp_path / "t.pftrace"
        pf.write_bytes(b"fake")
        conv = tmp_path / "tc"
        conv.write_text("#!/bin/sh\necho ok\n")
        conv.chmod(0o755)

        def mock_run(cmd, cwd=None):
            Path(cmd[-1]).write_text(json.dumps({"traceEvents": _minimal_pftrace_events()}))

        monkeypatch.setattr(pftrace_utils, "run", mock_run)
        assert pftrace_utils.ensure_trace_json(str(pf), str(conv)).endswith(".json")

    def test_pftrace_analyzer_and_report(self, tmp_path):
        assert extract_time_ns({"ts": 100, "dur": 50, "args": {}}) == (100_000, 50_000)
        assert classify("Cijk_x") == "gemm"
        compute = [Event(gpu=0, name="xla_k", ts_ns=0, dur_ns=100), Event(gpu=0, name="xla_k", ts_ns=50, dur_ns=100)]
        ov, _ = rccl_overlap_two_pointer(compute, [Event(gpu=0, name="nccl", ts_ns=60, dur_ns=80)])
        assert ov == 120

        events = _rich_pftrace_events()
        analyser = PftraceHipActivityAnalyzer(
            events, merge_kernels=True, kernel_summary_include_rccl=True,
            kernel_summary_baseline="compute", hip_summary_group="name+stream+op",
        )
        assert analyser.used_fav3
        assert not analyser.get_df_category_summary().empty

        trace_path = tmp_path / "pf.json"
        trace_path.write_text(json.dumps({"traceEvents": events}))
        generate_perf_report_pftrace_hip_activity(
            trace_path=str(trace_path), output_csvs_dir=str(tmp_path / "csv"), merge_kernels=True,
        )
        _write_markdown_report(
            tmp_path / "md.md", pd.DataFrame(), [], False, ["gpu_0"], None, None,
        )


class TestArchAndMoePhase9:
    def test_arch_utils(self, monkeypatch):
        assert arch_utils.list_platforms()
        monkeypatch.setenv("TL_EXTENSION", "not_a_real_package_xyz")
        assert isinstance(arch_utils._collect_arch_jsons(), dict)

    def test_moe_pseudo_op_edges(self):
        assert not is_aiter_fused_moe_kernel({"cat": "kernel", "name": "aiter::quant_fmoe"})
        tree = _build_moe_tree([])
        _create_pseudo_op_moe_fused_aiter(tree, {"name": "wrong", "UID": 0})
        assert _extract_topk_from_outplace({"UID": 1, "args": {}}) == 8

        events = []
        moe = _mk_event("cpu_op", "vllm::outplace_fused_experts", 100, 200, 1, 1, {
            "Input Dims": [[128, 4096], [8, 4096, 512], [8, 4096, 512], [128, 6], [128, 6]],
            "Sequence number": 2,
        })
        events.append(moe)
        _add_gpu_chain(events, moe, 20, "fused_moe_kernel_gptq_awq_up", 110, 150)
        _add_gpu_chain(events, moe, 21, "fused_moe_kernel_gptq_awq_down", 160, 190)
        tree2 = _build_moe_tree(events)
        create_pseudo_ops_moe_gptq_awq(tree2)
        assert any(e.get("args", {}).get("Pseudo op") for e in tree2.events)

        fly_events = [
            _mk_event("cpu_op", FUSED_MOE_PARENT, 0, 500, 1, 1, {"Sequence number": 9}),
            _mk_event("python_function", "flydsl.py(10): flydsl_moe_stage1", 50, 100, 1, 1, {}),
        ]
        create_pseudo_ops_moe_flydsl(_build_moe_tree(fly_events, add_python_func=True))


class TestTreePerfExtendedPhase9:
    def test_launcher_summaries(self):
        corr1, corr2 = 100, 101
        events = [
            _make_gpu_event("cpu1", 1000, 100, "cpu_op", "aten::mm",
                            args={"Input Dims": [[32, 64], [64, 128]], "Input type": ["fp16", "fp16"]}),
            _make_gpu_event("rt1", 1010, 5, "cuda_runtime", "hipLaunchKernel", args={"correlation": corr1}),
            _make_gpu_event("k1", 1050, 50, "kernel", "gemm_a", pid=0, tid=7, args={"correlation": corr1, "stream": 7}),
            _mk_ac2g(corr1, 0, 7, 1050, "s"), _mk_ac2g(corr1, 0, 7, 1100, "f"),
            _make_gpu_event("rt2", 1060, 5, "cuda_runtime", "hipLaunchKernel", args={"correlation": corr2}),
            _make_gpu_event("k2", 1065, 40, "kernel", "gemm_b", pid=0, tid=7, args={"correlation": corr2, "stream": 7}),
            _mk_ac2g(corr2, 0, 7, 1065, "s"), _mk_ac2g(corr2, 0, 7, 1105, "f"),
        ]
        analyzer = _build_analyzer(events, add_python_func=True)
        launchers = analyzer.get_df_kernel_launchers(include_args=True)
        TreePerfAnalyzer.get_df_kernel_launchers_summary(launchers)
        TreePerfAnalyzer.get_df_kernel_launchers_summary_module(launchers)
        unified = analyzer.build_df_unified_perf_table(include_nccl=False, include_perf_metrics=True)
        assert isinstance(unified, pd.DataFrame)
        compute, _, _, used_fav3, _ = build_event_lists(_rich_pftrace_events(), True, -999, 999)
        assert used_fav3 and any(len(g) > 0 for g in compute)
        cfg = build_kernel_summary_df_for_config(
            [Event(gpu=0, name="Cijk_test", ts_ns=0, dur_ns=1_000_000, grid_size=256, workgroup_size=256)],
            2_000_000, False,
        )
        assert not cfg.empty
        assert not build_hip_summary_df(
            [HIPEvent(name="hipLaunch", ts_ns=0, dur_ns=1_000_000, pid=1, tid=1)],
            group="name+stream+op",
        ).empty
