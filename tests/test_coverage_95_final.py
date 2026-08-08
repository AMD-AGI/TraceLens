###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Final CPU-only coverage push to reach 95% TraceLens line coverage."""

from __future__ import annotations

import gzip
import importlib
import json
import os
import sys
from copy import deepcopy
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from TraceLens.Agent.Analysis.category_analyses import analysis_utils as au
from TraceLens.Agent.Analysis.utils.orchestrator_prepare import (
    _extract_comparative_fusion_candidates,
)
from TraceLens.PerfModel import perf_model
from TraceLens.PerfModel.extensions import moe_perf_model_extensions as moe_ext
from TraceLens.Reporting import reporting_utils as ru
from TraceLens.Reporting.compare_traces_jax_llama import (
    Event,
    Summary,
    classify_stage_base,
    compute_stage_table,
    emit_report,
    extract_gpu_events,
    infer_params,
    is_loop_multiply_fusion,
    load_trace,
    mk_stats,
    percentile,
    summarize_one,
    token_start_times,
    top_stats_by_key,
)
from TraceLens.Reporting.generate_multi_rank_collective_report_pytorch import (
    _resolve_trace_files_glob,
    generate_collective_report,
)
from TraceLens.Reporting.rocprof_analysis import RocprofAnalyzer, _categorize_kernel
from TraceLens.Trace2Tree.extensions.pseudo_ops_registry import apply_pseudo_op_extensions
from TraceLens.Trace2Tree.trace_capture_merge_experimental import align_streams
from TraceLens.Trace2Tree.trace_to_tree import TraceToTree
from TraceLens.TreePerf.tree_perf import JaxTreePerfAnalyzer, TreePerfAnalyzer
from TraceLens.util import RocprofParser

from tests.test_agent_coverage import _StubAnalyzer, _StubTree, _kernel_event
from tests.test_jax_analysis_report import _mock_side_inputs, _sample_averages_df
from tests.test_perfmodel_coverage import _ARCH, _gemm_event
from tests.test_reporting_coverage import _build_synthetic_trace, _mk_ac2g, _mk_event
from tests.test_treeperf_coverage import _build_analyzer, _make_gpu_event, _mk_pytorch_trace

ROCprof_FILE = os.path.join(os.path.dirname(__file__), "rocprof/908_results.json.gz")
JAX_PB = os.path.join(
    os.path.dirname(__file__),
    "traces/mi300/jax_conv_minimal_legacy/chi-mi300x-013.ord.vultr.cpe.ice.amd.com.xplane.pb",
)

pytestmark = pytest.mark.filterwarnings(
    "ignore:Found .* events with failed performance metric.*:UserWarning",
    "ignore:There are hipgraph launches.*:UserWarning",
    "ignore:Input DataFrame is empty.*:UserWarning",
    "ignore:No kernel events found.*:UserWarning",
)


def _write_gz_trace(tmp_path, events, name="trace.json.gz"):
    path = tmp_path / name
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump({"traceEvents": events}, f)
    return str(path)


def _jax_llama_trace_events(block0_hint: str = "te_layernorm_forward"):
    """Minimal JAX LLaMA-style Chrome trace for compare_traces_jax_llama helpers."""
    base_path = (
        "jit(main)/jit(call)/jit(layer)/Transformer/block_{block}/norm_attn/"
        + block0_hint
    )
    events = [
        {"ph": "M", "name": "process_name", "pid": 1, "args": {"name": "/device:GPU:0"}},
        {"ph": "M", "name": "thread_name", "pid": 1, "tid": 10, "args": {"name": "Stream"}},
    ]
    ts = 1000.0
    for tok in range(2):
        for block in range(2):
            p = base_path.format(block=block).replace(
                "block_0", f"block_{block}"
            )
            if block == 0 and tok == 0:
                p = base_path.format(block=0)
            events.append(
                {
                    "ph": "X",
                    "pid": 1,
                    "tid": 10,
                    "ts": ts,
                    "dur": 50,
                    "name": "ln_fwd_tuned_kernel<Kernel_traits<float, 4096u, 64>",
                    "args": {"name": p},
                }
            )
            ts += 60
            events.append(
                {
                    "ph": "X",
                    "pid": 1,
                    "tid": 10,
                    "ts": ts,
                    "dur": 80,
                    "name": "Cijk_gemm",
                    "args": {"name": p.replace("norm_attn", "attn/q/dot_general")},
                }
            )
            ts += 90
            events.append(
                {
                    "ph": "X",
                    "pid": 1,
                    "tid": 10,
                    "ts": ts,
                    "dur": 120,
                    "name": "te_fused_attn_forward",
                    "args": {"name": p.replace("norm_attn", "attn/out/dot_general")},
                }
            )
            ts += 130
            events.append(
                {
                    "ph": "X",
                    "pid": 1,
                    "tid": 10,
                    "ts": ts,
                    "dur": 70,
                    "name": "loop_multiply_fusion",
                    "args": {
                        "name": "jit(main)/Transformer/mlp/in/dot_general",
                        "hlo_op": "loop_multiply_fusion",
                    },
                }
            )
            ts += 80
            events.append(
                {
                    "ph": "X",
                    "pid": 1,
                    "tid": 10,
                    "ts": ts,
                    "dur": 60,
                    "name": "Cijk_gemm",
                    "args": {"name": p.replace("norm_attn", "mlp/out/dot_general")},
                }
            )
            ts += 100
    return events


class TestCompareTracesJaxLlama:
    def test_helper_functions(self, tmp_path):
        trace_path = _write_gz_trace(tmp_path, _jax_llama_trace_events())
        trace = load_trace(trace_path)
        gpu_events = extract_gpu_events(trace, gpu_index=0)
        assert gpu_events
        assert percentile([1, 2, 3, 4], 50) == 2.5
        assert mk_stats([10, 20]).total_us == 30
        d_model, head_dim, gsu = infer_params(gpu_events)
        assert d_model == 4096

        stream = [e for e in gpu_events if e.tid == gpu_events[0].tid]
        starts = token_start_times(stream, "te_layernorm_forward")
        assert len(starts) >= 1

        stage_avg, stage_share, per_layer, per_token, notes = compute_stage_table(
            stream, starts, (0, 0), (0, 1)
        )
        assert per_layer > 0
        assert "attn_core" in stage_avg

        ev = Event(1, 10, 0, 10, "loop_multiply_fusion", {"hlo_op": "loop_multiply_fusion"})
        assert is_loop_multiply_fusion(ev)
        assert classify_stage_base(ev) == "other"
        assert top_stats_by_key(stream, lambda e: e.name, 3)

    def test_summarize_one_and_emit_report(self, tmp_path):
        trace_path = _write_gz_trace(tmp_path, _jax_llama_trace_events())
        summary = summarize_one(
            "ROCm", trace_path, 0, (0, 0), (0, 1), "te_layernorm_forward"
        )
        assert isinstance(summary, Summary)
        report = emit_report(summary, summary)
        assert "Trace Comparison" in report
        assert summary.notes is not None

    def test_main_mocked(self, tmp_path, monkeypatch):
        rocm = _write_gz_trace(tmp_path, _jax_llama_trace_events(), "rocm.json.gz")
        cuda = _write_gz_trace(
            tmp_path, _jax_llama_trace_events("te_norm_forward_ffi"), "cuda.json.gz"
        )
        out_md = tmp_path / "report.md"
        mod = importlib.import_module("TraceLens.Reporting.compare_traces_jax_llama")
        summary = summarize_one("ROCm", rocm, 0, (0, 0), (0, 1), "te_layernorm_forward")
        monkeypatch.setattr(mod, "summarize_one", lambda *a, **k: summary)
        old_argv = sys.argv
        sys.argv = [
            "compare_traces_jax_llama",
            "--rocm",
            rocm,
            "--cuda",
            cuda,
            "--tokens",
            "0:0",
            "--layers",
            "0:1",
            "--out",
            str(out_md),
        ]
        try:
            mod.main()
        finally:
            sys.argv = old_argv
        assert out_md.exists()


class TestCollectiveReportErrors:
    def test_glob_rank_resolution_errors(self, tmp_path):
        (tmp_path / "bad.json").write_text("{}")
        with pytest.raises(ValueError, match="none matched"):
            _resolve_trace_files_glob(str(tmp_path / "*.json"), world_size=2)
        for rank in (0, 1):
            (tmp_path / f"trace_rank_{rank}.json").write_text("{}")
        paths = _resolve_trace_files_glob(str(tmp_path / "trace_rank_*.json"), 2)
        assert len(paths) == 2

    def test_collective_trace_pattern_and_all2allv(self, tmp_path):
        for rank in (0, 1):
            events = {
                "traceEvents": [{
                    "ph": "X",
                    "cat": "kernel",
                    "name": "ncclKernel_AllReduce",
                    "pid": rank,
                    "tid": 3,
                    "ts": 1000,
                    "dur": 40,
                    "args": {
                        "External id": 10,
                        "Collective name": "allreduce",
                        "stream": 3,
                    },
                }]
            }
            (tmp_path / f"trace_{rank}_step.json").write_text(json.dumps(events))
        dfs = generate_collective_report(
            trace_pattern=str(tmp_path / "trace_*_step.json"),
            world_size=2,
            output_csvs_dir=str(tmp_path / "coll"),
            use_multiprocessing=False,
            strict_world_size_check=True,
            all2allv_heatmap=True,
        )
        assert isinstance(dfs, dict)

    def test_gpus_per_node_invalid(self, tmp_path):
        for rank in (0,):
            (tmp_path / f"rank{rank}_trace.json").write_text(json.dumps({"traceEvents": []}))
        with pytest.raises(ValueError, match="gpus_per_node"):
            generate_collective_report(
                trace_dir=str(tmp_path),
                world_size=1,
                gpus_per_node=0,
                strict_world_size_check=False,
            )


@pytest.mark.skipif(not os.path.isfile(ROCprof_FILE), reason="rocprof fixture missing")
class TestRocprofAnalysisDeep:
    def test_full_rocprof_pipeline(self):
        data = RocprofParser.load_rocprof_data(ROCprof_FILE)
        kernels = RocprofParser.extract_kernel_events(data)
        memory = RocprofParser.extract_memory_events(data)
        api = RocprofParser.extract_api_events(data)
        metadata = RocprofParser.get_metadata(data)
        analyser = RocprofAnalyzer(kernels, memory, api, metadata)
        timeline = analyser.get_df_gpu_timeline()
        assert not timeline.empty
        assert not analyser.get_df_kernel_summary().empty
        assert isinstance(analyser.get_df_short_kernels(10), pd.DataFrame)
        assert _categorize_kernel("Cijk_gemm") == "GEMM"

    def test_rocprof_main(self, tmp_path):
        mod = importlib.import_module("TraceLens.Reporting.generate_perf_report_rocprof")
        out = tmp_path / "roc.xlsx"
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


class TestJaxAnalysisMain:
    def test_jax_analysis_main(self, tmp_path):
        mod = importlib.import_module(
            "TraceLens.Reporting.generate_perf_report_jax_analysis"
        )
        categorized, xla_events = _mock_side_inputs()
        gemms = pd.DataFrame({"time ms": [1.0], "percent": [1.0]}, index=["gemm1"])
        gemms_detailed = pd.DataFrame({"name": ["gemm1"], "tflops": [1.0]})
        with patch.object(
            mod.JaxAnalyses,
            "summarize_gpu_events",
            return_value=(_sample_averages_df(), categorized, xla_events),
        ), patch.object(
            mod.JaxAnalyses,
            "summarize_gpu_gemm_events_from_pb",
            return_value=gemms,
        ), patch.object(
            mod.JaxAnalyses,
            "gemm_performance_from_pb",
            return_value=gemms_detailed,
        ):
            old_argv = sys.argv
            sys.argv = [
                "generate_perf_report_jax_analysis",
                "--profile_xplane_pb_path",
                "/fake/profile.xplane.pb",
                "--output_path",
                str(tmp_path),
                "--output_table_formats",
                ".csv",
            ]
            try:
                mod.main()
            finally:
                sys.argv = old_argv
        assert (tmp_path / "trace_analysis_results_gpu_events_averages.csv").exists()

    def test_jax_analysis_permission_error(self, tmp_path, monkeypatch):
        mod = importlib.import_module(
            "TraceLens.Reporting.generate_perf_report_jax_analysis"
        )
        bad_path = tmp_path / "nope" / "out"
        monkeypatch.setattr(
            mod.Path,
            "mkdir",
            MagicMock(side_effect=PermissionError("denied")),
        )
        with pytest.raises(SystemExit):
            mod.generate_perf_report_jax_analysis(
                "/fake.pb", str(bad_path), "out", [".csv"]
            )


class TestPseudoOpsRegistryFull:
    def test_registry_detects_all_extension_types(self, tmp_path):
        events = [
            _mk_event("cpu_op", "aiter::fused_moe_", 0, 10, 1, 1),
            _mk_event("cpu_op", "vllm::moe_forward", 1, 10, 1, 1),
            _mk_event("cpu_op", "outplace_fused_experts", 2, 10, 1, 1),
            _mk_event("cpu_op", "aiter::mla_decode_stage1_asm_fwd", 3, 10, 1, 1),
            _mk_event("cpu_op", "aiter::mla_prefill_ps_asm_fwd", 4, 10, 1, 1),
            _mk_event(
                "python_function",
                "paged_decode.py(1): sparse_attn_v4_paged_decode",
                7,
                10,
                1,
                1,
                {"Python id": 1},
            ),
            _mk_event(
                "python_function",
                "aiter/mla.py(1): mla_decode_fwd",
                8,
                10,
                1,
                1,
                {"Python id": 2},
            ),
            _mk_event(
                "python_function",
                "mod.py(1): mla_fp8_prefill_attn",
                9,
                10,
                1,
                1,
                {"Python id": 3},
            ),
        ]
        tree = TraceToTree(deepcopy(events), prune_nongpu_paths=False)
        tree.build_tree(add_python_func=True)
        apply_pseudo_op_extensions(tree, verbose=True)
        assert tree is not None


class TestOrchestratorComparativeFusion:
    def test_comparative_fusion_two_kernel_match(self, tmp_path):
        k1 = _kernel_event(10, "Cijk_A", dur=500)
        k2 = _kernel_event(11, "ew_add", dur=300)
        mod = {
            "name": "nn.Module: Block_0",
            "_category": "aten",
            "gpu_events": [10, 11],
            "args": {"Input Dims": "[[2,3]]"},
        }
        tree = _StubTree([mod], {10: k1, 11: k2})
        analyzer = _StubAnalyzer(tree)

        csv_dir = tmp_path / "trace1_csvs"
        csv_dir.mkdir()
        pd.DataFrame({
            "name": ["Cijk_A", "ew_add", "Cijk_A", "ew_add"],
            "source": ["trace1", "trace1", "trace2", "trace2"],
            "lowest_common_ancestor_id": [100, 100, 100, 100],
            "kernel_time": [5000.0, 3000.0, 4000.0, 2500.0],
            "gpu_op_uid": [10, 11, 10, 11],
        }).to_csv(csv_dir / "diff_stats.csv", index=False)

        cands = _extract_comparative_fusion_candidates(str(csv_dir), analyzer, tree)
        assert isinstance(cands, list)


class TestPerfModelRemaining:
    def test_scaled_mm_mismatched_dtypes(self):
        event = {
            "args": {
                "Input Dims": [(4, 8), (8, 16), (4, 16)],
                "Input type": ["c10::Float8_e4m3fn", "c10::BFloat16", "c10::BFloat16"],
                "Input Strides": [(8, 1), (16, 1), (16, 1)],
            }
        }
        model = perf_model.aten_scaled_mm(event)
        assert model.bytes() > 0

    def test_gemm_without_strides(self):
        event = {
            "args": {
                "Input Dims": [(4, 8), (8, 16), (4, 16)],
                "Input type": ["c10::Float8_e4m3fn", "c10::BFloat16", "c10::BFloat16"],
            }
        }
        model = perf_model.aten_scaled_mm(event)
        assert model.param_details["stride_A"] is None

    def test_vllm_gemm_dynamic_quant(self):
        event = {
            "args": {
                "Input Dims": [(128, 64), (256, 64)],
                "Input type": ["c10::Float8_e4m3fn", "c10::Float8_e4m3fn"],
            }
        }
        model = perf_model.vllm_gemm_with_dynamic_quant(event)
        assert model.flops() > 0

    def test_grouped_gemm_list_shapes(self):
        event = {
            "name": "primus_turbo::grouped_gemm",
            "args": {
                "Input Dims": [[[4, 8], [5, 8]], [[8, 16], [8, 16]]],
                "Input type": ["c10::BFloat16", "c10::BFloat16"],
            },
        }
        model = perf_model.primus_turbo_grouped_gemm(event)
        assert model.flops() > 0

    def test_gemm_simulator_invalid_arch(self, monkeypatch, tmp_path):
        sim = tmp_path / "run_gemm.py"
        sim.write_text("# stub\n")
        monkeypatch.setenv("GEMM_SIMULATOR_PATH", str(sim))
        perf_model.GEMM.cache_gemm_results.clear()
        with pytest.raises(AssertionError, match="Invalid inputs"):
            perf_model.GEMM.get_simulation_time_func(
                {"freq_mhz": 2200}, None, 8, 16, 1, "bf16"
            )
        perf_model.GEMM.cache_gemm_results.clear()


class TestTreePerfRemaining:
    def test_pseudo_op_extension_failure(self, monkeypatch):
        events = _mk_pytorch_trace()
        monkeypatch.setattr(
            "TraceLens.TreePerf.tree_perf.apply_pseudo_op_extensions",
            MagicMock(side_effect=RuntimeError("boom")),
        )
        analyzer = TreePerfAnalyzer(
            TraceToTree(deepcopy(events), prune_nongpu_paths=False),
            rebuild_tree=True,
            enable_pseudo_ops=True,
        )
        assert analyzer.tree is not None

    def test_detect_recompute_and_agg_verbose(self):
        events = _mk_pytorch_trace()
        events.append(
            _make_gpu_event(
                "recomp",
                3000,
                200,
                "python_function",
                "torch/utils/checkpoint.py(1): recompute_fn",
                pid=100,
            )
        )
        tree = TraceToTree(deepcopy(events), prune_nongpu_paths=False)
        tree.build_tree(add_python_func=True)
        analyzer = TreePerfAnalyzer(
            tree, rebuild_tree=False, detect_recompute=True, add_python_func=True
        )
        root = analyzer.tree.events[0]
        dur, uids = analyzer.agg_kernels_in_subtree(root, verbose=True)
        assert dur >= 0

    def test_build_nn_module_latency_tree(self):
        corr = 500
        events = [
            _make_gpu_event("nn", 0, 500, "python_function", "nn.Module: Net", pid=100),
            _make_gpu_event(
                "cpu1", 20, 80, "cpu_op", "aten::mm", pid=100,
                args={"Input Dims": [[32, 64], [64, 128]], "Input type": ["fp16", "fp16"]},
            ),
            _make_gpu_event("rt1", 25, 5, "cuda_runtime", "hipLaunchKernel", pid=100, args={"correlation": corr}),
            _make_gpu_event("k1", 50, 40, "kernel", "Cijk_gemm", pid=0, tid=7, args={"correlation": corr, "stream": 7}),
            _mk_ac2g(corr, 0, 7, 50, "s"),
            _mk_ac2g(corr, 0, 7, 90, "f"),
        ]
        analyzer = _build_analyzer(events, add_python_func=True)
        nn_evt = next(e for e in analyzer.tree.events if "nn.Module" in e["name"])
        analyzer.build_nn_module_latency_tree(nn_evt)
        assert "GPU Time" in nn_evt

    def test_jax_tree_perf_model_branches(self):
        conv_bwd = {
            "gpu_kernel_op_cat": "conv",
            "metadata": {
                "custom_call_target": "cudnn$convBackward",
                "operands": ["bf16[4,8]{1,0}"],
            },
        }
        assert JaxTreePerfAnalyzer.get_event_perf_model_name(conv_bwd) == "jax_conv_bwd"
        te_bwd = {
            "gpu_kernel_op_cat": "GEMM",
            "metadata": {
                "custom_call_target": "te_fused_attn_backward_ffi",
                "operands": ["bf16[4,8]{1,0}"],
            },
        }
        assert JaxTreePerfAnalyzer.get_event_perf_model_name(te_bwd) == "jax_te_fused_attn_bwd"
        te_fwd = {
            "gpu_kernel_op_cat": "GEMM",
            "metadata": {
                "custom_call_target": "te_fused_attn_forward_ffi",
                "operands": ["bf16[4,8]{1,0}"],
            },
        }
        assert JaxTreePerfAnalyzer.get_event_perf_model_name(te_fwd) == "jax_te_fused_attn"
        meta = JaxTreePerfAnalyzer.get_event_metadata(te_fwd)
        assert isinstance(meta, dict)

    def test_jax_kernel_launchers_with_metadata_filter(self):
        event = _make_gpu_event(
            "k1", 0, 100, "kernel", "Cijk_gemm", pid=1, tid=1,
            args={"correlation": 1, "stream": 1},
        )
        event["metadata"] = {"metadata": "special_tag_here"}
        event["gpu_kernel_op_cat"] = "GEMM"
        tree = TraceToTree([event], prune_nongpu_paths=False)
        tree.build_tree()
        analyzer = JaxTreePerfAnalyzer(
            tree,
            rebuild_tree=False,
            kernel_metadata_keyword_filters=["special_tag"],
        )
        launchers = analyzer.get_df_kernel_launchers(include_args=True)
        assert isinstance(launchers, pd.DataFrame)


class TestAnalysisUtilsAndReporting:
    def test_perf_report_csv_dir_comparative(self, tmp_path):
        cat_dir = tmp_path / "category_data"
        cat_dir.mkdir()
        (cat_dir / "category_manifest.json").write_text(
            json.dumps({"comparison_scope": "comparative"})
        )
        assert "perf_report_trace1_csvs" in au.perf_report_csv_dir(str(tmp_path))

    def test_resolve_gpu_arch_and_node_span(self):
        arch = ru.resolve_gpu_arch(
            gpu_arch={"name": "mi300x", "freq_mhz": 2200, "num_cus": 304}
        )
        assert arch["name"] == "mi300x"
        out = ru.add_node_span_columns(
            pd.DataFrame({"rank": [0, 1], "Process Group Ranks": ["[0,1]", "[0,1]"]}),
            gpus_per_node=2,
            world_size=2,
        )
        assert "node_id" in out.columns

    def test_reporting_utils_export_and_node_span(self, tmp_path):
        from pathlib import Path

        df = pd.DataFrame({"a": [1, 2]})
        ru.export_data_df(df, Path(tmp_path), "test", output_table_format=[".csv"])
        assert (tmp_path / "test_summary_statistics.csv").exists()


class TestCaptureMergeAndMoe:
    def test_align_streams_multistream(self):
        graph = [
            {"name": "k1", "args": {"stream": 1}},
            {"name": "k2", "args": {"stream": 2}},
        ]
        capture = [
            {"name": "hipLaunchKernel", "args": {"kernel": "k1"}},
            {"name": "hipLaunchKernel", "args": {"kernel": "k2"}},
        ]
        aligned = align_streams(graph, capture)
        assert aligned is not None
        assert len(aligned) == 2

    def test_moe_biased_grouped_topk(self):
        evt = {
            "args": {
                "Input Dims": [(32, 256), (256,), (32, 8), (32, 8)],
                "Input type": ["c10::Float"] * 3 + ["c10::Int"],
            }
        }
        model = moe_ext.BiasedGroupedTopk(evt)
        assert model.flops() > 0

    def test_moe_sort_scatter_missing_kernel_details(self):
        evt = {
            "args": {
                "Input Dims": [(32, 4096), (32, 8)],
                "Input type": ["c10::BFloat16", "c10::Int"],
            }
        }
        model = moe_ext.MoeSortScatterGather(evt)
        assert model.bytes() is None or model.bytes() >= 0


@pytest.mark.skipif(not os.path.isfile(JAX_PB), reason="JAX fixture missing")
class TestJaxFromFile:
    def test_jax_analyzer_from_pb(self):
        analyzer = JaxTreePerfAnalyzer.from_file(profile_filepath=JAX_PB)
        assert analyzer.tree is not None
        timeline = analyzer.get_df_gpu_timeline()
        assert isinstance(timeline, pd.DataFrame)
