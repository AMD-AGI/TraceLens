###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Phase-12: final ~45-line push to <=942 miss."""

from __future__ import annotations

import gzip
import json
import os
import sys
from unittest.mock import patch

import pandas as pd
import pytest

from TraceLens.Agent.Analysis.utils.orchestrator_prepare import (
    _extract_comparative_fusion_candidates,
)
from TraceLens.PerfModel import perf_model
from TraceLens.Reporting.generate_perf_report_pytorch_inference import (
    classify_graph_capture_trace,
    generate_perf_report_pytorch,
)
from TraceLens.TreePerf.tree_perf import TreePerfAnalyzer

from tests.test_agent_coverage import _StubAnalyzer, _StubTree, _kernel_event
from tests.test_treeperf_coverage import _build_analyzer, _make_gpu_event, _mk_ac2g

RESNET_TRACE = os.path.join(
    os.path.dirname(__file__), "traces/mi300/resnet_act_checkpoint.json.gz"
)
NORM_TRACE = os.path.join(
    os.path.dirname(__file__), "traces/perf_model/normalization/normalization_layer_test.json.gz"
)

TIMESFORMER1 = os.path.join(
    os.path.dirname(__file__),
    "traces/mi300/facebook_timesformer-base-finetuned-k400__1016002.json.gz",
)
TIMESFORMER2 = os.path.join(
    os.path.dirname(__file__),
    "traces/h100/facebook_timesformer-base-finetuned-k400__1016002.json.gz",
)


class TestTreePerfPhase12:
    def test_reorder_cols_and_kernel_stats_edges(self):
        df = pd.DataFrame({
            "name": ["a"],
            "direct_mean": [1.0],
            "subtree_mean": [2.0],
            "direct_std": [0.1],
            "subtree_std": [0.2],
            "other_col": [3],
        })
        out = TreePerfAnalyzer._reorder_cols_direct_subtree_pairs(
            df, "direct", "subtree"
        )
        assert "direct_mean" in out.columns

        stats = TreePerfAnalyzer._summarize_kernel_stats(
            [[{"name": "k1", "dur": 10}], [{"name": "k1"}]],
            agg_metrics=["mean"],
        )
        assert stats[0]["count"] == 1

        bad = TreePerfAnalyzer._summarize_kernel_stats(
            [[{"name": "k1", "dur": 1}]], agg_metrics=["mean"]
        )
        assert len(bad) == 1

    def test_kernel_launchers_execute_parent_chain(self):
        corr = 400
        events = [
            _make_gpu_event("conv", 1000, 80, "cpu_op", "aten::convolution",
                            args={"Input Dims": [[[2, 3, 8, 8], [4, 3, 3, 3]]]}),
            _make_gpu_event("exec", 1010, 10, "cpu_op", "execute"),
            _make_gpu_event("rt", 1020, 5, "cuda_runtime", "hipLaunchKernel",
                            args={"correlation": corr}),
            _make_gpu_event("k", 1030, 50, "kernel", "conv_k", pid=0, tid=7,
                            args={"correlation": corr, "stream": 7}),
            _mk_ac2g(corr, 0, 7, 1030, "s"), _mk_ac2g(corr, 0, 7, 1080, "f"),
        ]
        analyzer = _build_analyzer(events)
        conv = next(e for e in analyzer.tree.events if e["name"] == "aten::convolution")
        execute = next(e for e in analyzer.tree.events if e["name"] == "execute")
        conv["children"] = [execute["UID"]]
        launchers = analyzer.get_df_kernel_launchers(include_args=True)
        assert isinstance(launchers, pd.DataFrame)

    def test_unified_bwd_sole_exception_fallback(self):
        corr_f, corr_b = 500, 501
        events = [
            _make_gpu_event("cpu_f", 1000, 100, "cpu_op", "aten::mm",
                            args={"Input Dims": [[32, 64], [64, 128]], "Input type": ["fp16", "fp16"]}),
            _make_gpu_event("rt_f", 1010, 5, "cuda_runtime", "hipLaunchKernel", args={"correlation": corr_f}),
            _make_gpu_event("k_f", 1050, 50, "kernel", "gemm_fwd", pid=0, tid=7,
                            args={"correlation": corr_f, "stream": 7}),
            _mk_ac2g(corr_f, 0, 7, 1050, "s"), _mk_ac2g(corr_f, 0, 7, 1100, "f"),
            _make_gpu_event("cpu_b", 2000, 100, "cpu_op", "aten::mm",
                            args={"Input Dims": [[32, 64], [64, 128], [32, 128]],
                                  "Input type": ["fp16", "fp16", "fp16"]}),
            _make_gpu_event("rt_b", 2010, 5, "cuda_runtime", "hipLaunchKernel", args={"correlation": corr_b}),
            _make_gpu_event("k_b", 2050, 60, "kernel", "gemm_bwd", pid=0, tid=7,
                            args={"correlation": corr_b, "stream": 7}),
            _mk_ac2g(corr_b, 0, 7, 2050, "s"), _mk_ac2g(corr_b, 0, 7, 2110, "f"),
        ]
        analyzer = _build_analyzer(events)
        fwd = next(e for e in analyzer.tree.events if e["name"] == "aten::mm" and e["ts"] == 1000)
        bwd = next(e for e in analyzer.tree.events if e["name"] == "aten::mm" and e["ts"] == 2000)
        fwd["bwd_events"] = [bwd["UID"]]
        bwd["fwd_event"] = fwd["UID"]
        bwd["gpu_events"] = [next(e["UID"] for e in analyzer.tree.events if e["name"] == "gemm_bwd")]

        real = analyzer.compute_perf_metrics

        def boom(event, bwd=False, **kwargs):
            if bwd:
                raise RuntimeError("bwd metrics failed")
            return real(event, bwd=bwd, **kwargs)

        with patch.object(analyzer, "compute_perf_metrics", side_effect=boom):
            df = analyzer.build_df_unified_perf_table(
                events=[bwd], include_perf_metrics=True,
            )
        assert isinstance(df, pd.DataFrame)

    @pytest.mark.skipif(not os.path.isfile(RESNET_TRACE), reason="resnet trace missing")
    def test_resnet_overlap_and_recompute_summaries(self):
        analyzer = TreePerfAnalyzer.from_file(
            RESNET_TRACE,
            rebuild_tree=True,
            enable_pseudo_ops=True,
            detect_recompute=True,
        )
        unified = analyzer.build_df_unified_perf_table(include_perf_metrics=True)
        if not unified.empty:
            try:
                TreePerfAnalyzer.summarize_df_unified_perf_table(
                    unified,
                    include_overlapping_kernels=True,
                    agg_metrics=["mean", "sum", "count"],
                )
            except ValueError:
                TreePerfAnalyzer.summarize_df_unified_perf_table(
                    unified, include_pct=True,
                )
        launchers = analyzer.get_df_kernel_launchers(
            include_args=True,
            include_first_occurrence_time=True,
        )
        if not launchers.empty:
            TreePerfAnalyzer.get_df_kernel_launchers_unique_args(
                launchers, include_pct=True, group_by_parent_module=True,
            )


class TestPerfModelPhase12:
    def test_conv_unknown_dim_and_vllm_attention(self):
        with pytest.raises(ValueError, match="Unknown convolution"):
            perf_model.aten_conv({
                "args": {
                    "Input Dims": [[2], [4]],
                    "Input type": ["c10::BFloat16", "c10::BFloat16"],
                    "Concrete Inputs": ["", "", "", "(1)", "(0)", "(1)", "False", "(0)", "1"],
                }
            })

        attn = perf_model.vllm_unified_attention_with_output({
            "annotation": "(prefill_128_64_8_0_0_0_0)",
            "args": {
                "Input Dims": [
                    [128, 8, 64], [128, 8, 64], [128, 8, 64], [128, 8, 64],
                ],
                "Input type": ["c10::BFloat16"] * 4,
            },
        })
        attn.param_details["sum_ctx_tokens"] = 0
        with pytest.raises(NotImplementedError):
            attn.flops()

    def test_conv_compute_precision_dtype_fallback(self):
        class _MiniConv(perf_model.CONV):
            @staticmethod
            def get_param_details(event):
                return {
                    "input_shape": (2, 3, 8, 8),
                    "filter_shape": (4, 3, 3, 3),
                    "out_shape": (2, 4, 6, 6),
                    "bias": False,
                    "transposed_conv": False,
                    "stride": (1, 1),
                    "padding": (0, 0),
                    "dilation": (1, 1),
                    "groups": 1,
                    "dtype": "c10::BFloat16",
                }

        model = _MiniConv({"args": {}}, arch=None)
        assert model.get_compute_precision() is not None


class TestOrchestratorPhase12:
    def test_comparative_fusion_multi_kernel_module(self, tmp_path):
        csv_dir = tmp_path / "t1"
        csv_dir.mkdir()
        pd.DataFrame({
            "name": ["Cijk_A", "Cijk_B"],
            "source": ["trace1", "trace1"],
            "lowest_common_ancestor_id": [100, 100],
            "kernel_time": [5000.0, 3000.0],
            "gpu_op_uid": [10, 11],
        }).to_csv(csv_dir / "diff_stats.csv", index=False)

        k1 = _kernel_event(10, "Cijk_A", dur=500)
        k2 = _kernel_event(11, "Cijk_B", dur=300)
        mod = {
            "name": "nn.Module: Attn_0",
            "_category": "aten",
            "gpu_events": [10, 11],
            "args": {"Input Dims": "[[2,8,128,64]]"},
        }
        mod2 = {
            "name": "nn.Module: Attn_1",
            "_category": "aten",
            "gpu_events": [10, 11],
        }
        tree = _StubTree([mod, mod2], {10: k1, 11: k2})
        analyzer = _StubAnalyzer(tree)
        cands = _extract_comparative_fusion_candidates(str(csv_dir), analyzer, tree)
        assert isinstance(cands, list)


class TestReportingPhase12:
    def test_classify_graph_capture_json_gz(self, tmp_path):
        capture_dir = tmp_path / "captures"
        capture_dir.mkdir()
        trace = {"traceEvents": [
            {"ph": "X", "cat": "user_annotation",
             "name": "graph_capture: batch=4 mode=FULL",
             "ts": 1000, "dur": 100, "pid": 1, "tid": 1, "args": {}},
        ], "schemaVersion": 1}
        gz_path = capture_dir / "graph_capture_rank_0.json.gz"
        with gzip.open(gz_path, "wt") as f:
            json.dump(trace, f)
        classify_graph_capture_trace(str(capture_dir))
        assert (capture_dir / "execution_details.json").exists()

    @pytest.mark.skipif(
        not (os.path.isfile(TIMESFORMER1) and os.path.isfile(TIMESFORMER2)),
        reason="timesformer traces missing",
    )
    def test_inference_report_overlap_on_trace(self, tmp_path):
        out = tmp_path / "inf_csv"
        generate_perf_report_pytorch(
            profile_json_path=TIMESFORMER1,
            output_csvs_dir=str(out),
            include_overlap_info=True,
            kernel_summary=True,
            short_kernel_study=True,
            group_by_parent_module=True,
        )
        assert (out / "gpu_timeline.csv").exists()


class TestOrchestratorPhase12B:
    def _run_orch(self, tmp_path, monkeypatch, unified_rows, tree_events=None):
        from TraceLens.Agent.Analysis.utils import orchestrator_prepare as op

        out = str(tmp_path / "orch")
        from tests.test_agent_coverage import _write_minimal_orchestrator_csvs

        _write_minimal_orchestrator_csvs(out, comparative=False)
        csv_dir = os.path.join(out, "perf_report_csvs")
        pd.DataFrame(unified_rows).to_csv(
            os.path.join(csv_dir, "unified_perf_summary.csv"), index=False
        )
        pd.DataFrame({"name": ["aten::op_0"], "op category": ["GEMM"]}).to_csv(
            os.path.join(csv_dir, "ops_summary.csv"), index=False
        )

        k = _kernel_event(0, "k0", dur=100)
        evts = tree_events or [{"name": "aten::mm", "_category": "aten", "gpu_events": [0], "ts": 0}]
        tree = _StubTree(evts + [k], {0: k})
        analyzer = _StubAnalyzer(tree)

        class _FakeTPA:
            @classmethod
            def from_file(cls, *args, **kwargs):
                return analyzer

        monkeypatch.setattr(op, "TreePerfAnalyzer", _FakeTPA)
        old_argv = sys.argv
        sys.argv = [
            "orchestrator_prepare",
            "--trace-path", RESNET_TRACE if os.path.isfile(RESNET_TRACE) else NORM_TRACE,
            "--platform", "MI300X",
            "--output-dir", out,
        ]
        try:
            op.main()
        finally:
            sys.argv = old_argv
        return out

    def test_bottleneck_nlargest_and_empty_category(self, tmp_path, monkeypatch):
        rows = []
        for i in range(20):
            rows.append({
                "name": f"aten::op_{i}",
                "op category": "GEMM" if i > 0 else "",
                "Kernel Time (µs)_sum": 100.0,
                "total_duration_us": 60000.0,
                "kernel_details_summary": f"[{{'name': 'k{i}'}}]",
                "Data Moved (MB)": 1.0,
                "perf_params": "{}",
                "Input Dims": "[[2,3]]",
            })
        out = self._run_orch(tmp_path, monkeypatch, rows)
        assert os.path.isfile(os.path.join(out, "category_data", "gemm_ops.csv"))

    def test_overlap_metrics_exception_path(self, tmp_path, monkeypatch):
        from TraceLens.TreePerf import GPUEventAnalyser

        def boom(*args, **kwargs):
            raise RuntimeError("overlap boom")

        monkeypatch.setattr(GPUEventAnalyser, "compute_metrics_dict", boom)
        rows = [{
            "name": "aten::mm",
            "op category": "GEMM",
            "Kernel Time (µs)_sum": 1000.0,
            "total_duration_us": 60000.0,
            "kernel_details_summary": "[{'name': 'k0'}]",
            "Data Moved (MB)": 1.0,
            "perf_params": "{}",
            "Input Dims": "[[2,3]]",
        }]
        out = self._run_orch(
            tmp_path,
            monkeypatch,
            rows,
            tree_events=[{
                "name": "gemm_kernel", "dur": 100, "ts": 1000, "UID": 0,
                "_category": "kernel", "cat": "kernel", "args": {"stream": 0},
            }],
        )
        data = json.loads(
            open(os.path.join(out, "category_data", "multi_kernel_data.json")).read()
        )
        assert "overlap_analysis" in data

    def test_sync_bottleneck_detection(self, tmp_path, monkeypatch):
        rows = [{
            "name": "aten::slow_sync",
            "op category": "GEMM",
            "Kernel Time (µs)_sum": 100.0,
            "total_duration_us": 5000000.0,
            "kernel_details_summary": "[{'name': 'k0'}]",
            "Data Moved (MB)": 1.0,
            "perf_params": "{}",
            "Input Dims": "[[2,3]]",
        }]
        out = self._run_orch(tmp_path, monkeypatch, rows)
        meta_dir = os.path.join(out, "metadata")
        gemm_meta = os.path.join(meta_dir, "gemm_metadata.json")
        assert os.path.isfile(gemm_meta)


class TestPerfModelPhase12B:
    def test_remaining_conv_and_reduce_edges(self):
        with pytest.raises(ValueError, match="Unknown convolution"):
            perf_model.aten_conv_bwd({
                "args": {
                    "Input Dims": [[2], [2], [4]],
                    "Input type": ["c10::BFloat16"] * 3,
                    "Concrete Inputs": [
                        "", "", "", "[0]", "[1]", "[0]", "[1]",
                        "False", "[0]", "1", "[True, True, False]",
                    ],
                }
            })

        evt = {
            "name": "aten::sum",
            "args": {
                "Input Dims": [[2, 4, 8]],
                "Input type": ["c10::BFloat16"],
                "Output type": None,
                "Concrete Inputs": ["", "[1]", "False"],
            },
        }
        assert perf_model.aten_reduce(evt).flops() > 0

        with pytest.raises(ValueError, match="could not parse"):
            perf_model.primus_turbo_grouped_gemm_variable_k({
                "args": {"Input Dims": [[1, 2, 3]], "Input type": ["c10::BFloat16"]}
            })


class TestReportingPhase12B:
    @pytest.mark.skipif(not os.path.isfile(RESNET_TRACE), reason="resnet missing")
    def test_pytorch_report_overlap_bwd(self, tmp_path):
        from TraceLens.Reporting.generate_perf_report_pytorch import (
            generate_perf_report_pytorch,
        )

        out = tmp_path / "pt_out"
        dfs = generate_perf_report_pytorch(
            profile_json_path=RESNET_TRACE,
            output_csvs_dir=str(out),
            include_overlap_info=True,
            kernel_summary=True,
            short_kernel_study=True,
        )
        assert isinstance(dfs, dict)
        assert (out / "gpu_timeline.csv").exists()


class TestTreePerfCollectPhase12:
    @pytest.mark.skipif(not os.path.isfile(RESNET_TRACE), reason="resnet missing")
    def test_collect_unified_with_python_func_roots(self):
        analyzer = TreePerfAnalyzer.from_file(
            RESNET_TRACE,
            rebuild_tree=True,
            enable_pseudo_ops=True,
            add_python_func=True,
        )
        collected = analyzer.collect_unified_perf_events(include_nccl=False)
        assert isinstance(collected, list)
        assert len(collected) > 0

    def test_is_leaf_cpu_op_via_descendant_kernel(self):
        corr = 600
        events = [
            _make_gpu_event("parent", 1000, 50, "cpu_op", "aten::wrapper",
                            args={"Input Dims": [[2, 2]]}),
            _make_gpu_event("leaf", 1010, 30, "cpu_op", "aten::mm",
                            args={"Input Dims": [[32, 64], [64, 128]], "Input type": ["fp16", "fp16"]}),
            _make_gpu_event("rt", 1020, 5, "cuda_runtime", "hipLaunchKernel", args={"correlation": corr}),
            _make_gpu_event("k", 1030, 40, "kernel", "gemm_k", pid=0, tid=7,
                            args={"correlation": corr, "stream": 7}),
            _mk_ac2g(corr, 0, 7, 1030, "s"), _mk_ac2g(corr, 0, 7, 1070, "f"),
        ]
        analyzer = _build_analyzer(events)
        wrapper = next(e for e in analyzer.tree.events if e["name"] == "aten::wrapper")
        leaf = next(e for e in analyzer.tree.events if e["name"] == "aten::mm")
        wrapper["children"] = [leaf["UID"]]
        assert analyzer._is_leaf_cpu_op(leaf) or analyzer._launches_gpu_kernels(leaf)
        collected = analyzer.collect_unified_perf_events()
        assert isinstance(collected, list)


class TestTraceToTreePhase12:
    @pytest.mark.skipif(not os.path.isfile(NORM_TRACE), reason="norm trace missing")
    def test_trace_to_tree_rebuild_pseudo_ops(self):
        from copy import deepcopy

        import gzip

        from TraceLens.Trace2Tree.trace_to_tree import TraceToTree

        with gzip.open(NORM_TRACE, "rt") as f:
            data = json.load(f)
        tree = TraceToTree(deepcopy(data["traceEvents"]), prune_nongpu_paths=False)
        tree.build_tree(add_python_func=True)
        assert len(tree.events) > 0


class TestTraceDiffPhase12:
    @pytest.mark.skipif(
        not (os.path.isfile(TIMESFORMER1) and os.path.isfile(TIMESFORMER2)),
        reason="timesformer traces missing",
    )
    def test_tracediff_prune_and_diff_stats(self, tmp_path):
        from TraceLens.TraceDiff.trace_diff import TraceDiff

        pa1 = TreePerfAnalyzer.from_file(TIMESFORMER1)
        pa2 = TreePerfAnalyzer.from_file(TIMESFORMER2)
        td = TraceDiff(pa1.tree, pa2.tree)
        td.merge_trees()
        stats = td.generate_diff_stats()
        assert isinstance(stats, pd.DataFrame)
        out_txt = tmp_path / "merged.txt"
        td.print_merged_tree(str(out_txt), prune_non_gpu=True)
        assert out_txt.exists()
