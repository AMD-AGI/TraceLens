###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Phase-5 targeted coverage for remaining gaps toward 95%."""

from __future__ import annotations

import gzip
import json
import os
import sys
from copy import deepcopy
from unittest.mock import patch

import pandas as pd
import pytest

from TraceLens.PerfModel import perf_model
from TraceLens.Reporting.compare_traces_jax_llama import (
    Event,
    compute_stage_table,
    extract_gpu_events,
    load_trace,
    summarize_one,
)
from TraceLens.Reporting.pftrace_hip_activity_analysis import PftraceHipActivityAnalyzer
from TraceLens.Trace2Tree.trace_capture_merge_experimental import (
    find_closest_batch_size,
    find_execution_details,
    merge_capture_trace_into_graph,
)
from TraceLens.TreePerf.tree_perf import TreePerfAnalyzer

from tests.test_conv_backward_bytes import _conv_bias_bwd_event, _conv_bias_fwd_event
from tests.test_perfmodel_coverage import _norm_event
from tests.test_reporting_coverage import _minimal_pftrace_events, _write_trace
from tests.test_treeperf_coverage import _build_analyzer, _make_gpu_event, _mk_ac2g, _mk_pytorch_trace

INFERENCE_ROOT = os.path.join(os.path.dirname(__file__), "traces/inference")


class TestCompareTracesEdgeCases:
    def test_extract_gpu_events_partial_pid_match(self, tmp_path):
        events = [
            {"ph": "M", "name": "process_name", "pid": 1, "args": {"name": "prefix/device:GPU:0/suffix"}},
            {
                "ph": "X", "pid": 1, "tid": 10, "ts": 100, "dur": 50,
                "name": "k", "args": {},
            },
        ]
        path = tmp_path / "t.json.gz"
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump({"traceEvents": events}, f)
        trace = load_trace(str(path))
        evs = extract_gpu_events(trace, gpu_index=0)
        assert len(evs) == 1

    def test_compute_stage_table_incomplete_blocks_note(self):
        stream = [
            Event(1, 10, 1000, 50, "k", {"name": "/Transformer/block_0/norm_attn/x"}),
            Event(1, 10, 2000, 50, "k", {"name": "/Transformer/block_0/norm_attn/y"}),
        ]
        starts = [1000.0, 2000.0]
        with pytest.raises(RuntimeError, match="No complete token"):
            compute_stage_table(stream, starts, (0, 0), (0, 2))


class TestPftraceHipActivityDeep:
    def test_analyser_all_methods(self, tmp_path):
        events = _minimal_pftrace_events()
        trace_path = tmp_path / "pf.json"
        trace_path.write_text(json.dumps({"traceEvents": events}))
        analyser = PftraceHipActivityAnalyzer(events)
        summary = analyser.get_df_category_summary()
        assert isinstance(summary, pd.DataFrame)
        kernels = analyser.get_df_kernel_summary()
        assert isinstance(kernels, pd.DataFrame)
        hip = analyser.get_df_hip_summary()
        assert isinstance(hip, pd.DataFrame)


class TestPerfModelNormAndConvDeep:
    def test_batch_norm_bwd_full(self):
        event = {
            "name": "aten::miopen_batch_norm_backward",
            "args": {
                "Input Dims": [
                    (8, 16, 32, 32), (8, 16, 32, 32), (16,), (16,),
                    (16,), (16,), (16,), (),
                ],
                "Input type": ["float"] * 7 + ["Scalar"],
                "Input Strides": [(16384, 1024, 32, 1)] * 2 + [(1,)] * 5 + [()],
                "Concrete Inputs": ["", "", "", "", "", "", "", "1e-5"],
            },
        }
        model = perf_model.BatchNormBwd(event)
        assert model.flops() > 0
        assert model.bytes() > 0

    def test_group_norm_bwd(self):
        event = {
            "args": {
                "Input Dims": [
                    None, (4, 8, 32, 32), (8,), (8,), (8,), (8,),
                    (4, 8, 32, 32), (),
                ],
                "Input type": ["c10::BFloat16"] * 7 + ["Scalar"],
                "Input Strides": [(), (8192, 1024, 32, 1), (1,)] * 2 + [(8192, 1024, 32, 1)] * 2 + [(), ()],
                "Concrete Inputs": ["", "", "", "", "", "", "", "8", "8", "[True, True]"],
            }
        }
        model = perf_model.GroupNormBwd(event)
        assert model.flops() > 0

    def test_conv_bias_bwd_with_forward_cache(self):
        fwd = perf_model.ConvBias_(_conv_bias_fwd_event())
        assert fwd.flops() > 0
        bwd = perf_model.ConvBias_Backward(_conv_bias_bwd_event())
        assert bwd.flops_bwd() > 0


class TestTreePerfDeepPaths:
    def test_kernel_launchers_all_options(self):
        corr1, corr2 = 300, 301
        events = [
            _make_gpu_event("py", 0, 500, "python_function", "nn.Module: Block", pid=100),
            _make_gpu_event(
                "cpu1", 20, 80, "cpu_op", "aten::mm", pid=100,
                args={
                    "Input Dims": [[32, 64], [64, 128]],
                    "Input type": ["c10::BFloat16", "c10::BFloat16"],
                    "Input Strides": [[128, 1], [128, 1]],
                },
            ),
            _make_gpu_event("rt1", 25, 5, "cuda_runtime", "hipLaunchKernel", pid=100, args={"correlation": corr1}),
            _make_gpu_event("k1", 50, 40, "kernel", "Cijk_gemm", pid=0, tid=7, args={"correlation": corr1, "stream": 7}),
            _mk_ac2g(corr1, 0, 7, 50, "s"), _mk_ac2g(corr1, 0, 7, 90, "f"),
            _make_gpu_event("cpu2", 120, 80, "cpu_op", "aten::add", pid=100,
                            args={"Input Dims": [[32, 128], [32, 128]], "Input type": ["c10::BFloat16"] * 2}),
            _make_gpu_event("rt2", 125, 5, "cuda_runtime", "hipLaunchKernel", pid=100, args={"correlation": corr2}),
            _make_gpu_event("k2", 150, 20, "kernel", "vectorized_elementwise_kernel", pid=0, tid=7,
                            args={"correlation": corr2, "stream": 7}),
            _mk_ac2g(corr2, 0, 7, 150, "s"), _mk_ac2g(corr2, 0, 7, 170, "f"),
        ]
        analyzer = _build_analyzer(events, add_python_func=True)
        df = analyzer.get_df_kernel_launchers(
            include_args=True,
            include_kernel_details=True,
            include_call_stack=True,
            id_cols=True,
            include_first_occurrence_time=True,
        )
        assert not df.empty

    def test_build_df_perf_metrics_cpu_ops(self):
        analyzer = _build_analyzer(_mk_pytorch_trace())
        perf = analyzer.build_df_perf_metrics(
            events=[e for e in analyzer.tree.events if e.get("cat") == "cpu_op"],
        )
        assert isinstance(perf, pd.DataFrame)

    def test_summarize_kernel_stats_all_metrics(self):
        stats = TreePerfAnalyzer._summarize_kernel_stats(
            [[{"name": "a", "dur": 10}, {"name": "b", "dur": 20}]],
            agg_metrics=["mean", "median", "max", "min", "std", "sum", "count"],
        )
        assert len(stats) == 2


class TestCaptureMergeDeep:
    def test_find_closest_batch_size_and_execution_details(self):
        assert find_closest_batch_size(128, [64, 256, 512]) == 256
        root = {"name": "execute_128_context_3_generation_2"}
        assert find_execution_details(root) == "128"

    @pytest.mark.skipif(
        not os.path.isdir(os.path.join(INFERENCE_ROOT, "sglang_decode", "capture_traces")),
        reason="capture fixture missing",
    )
    def test_merge_capture_full_inference_fixture(self):
        case = os.path.join(INFERENCE_ROOT, "sglang_decode")
        trace_gz = next(f for f in os.listdir(case) if f.endswith(".json.gz"))
        graph = os.path.join(case, trace_gz)
        capture = os.path.join(case, "capture_traces")
        metadata = os.path.join(capture, "execution_details.json")
        merged = merge_capture_trace_into_graph(capture, metadata, graph)
        assert len(merged.events) > 0
        analyzer = TreePerfAnalyzer(merged, rebuild_tree=False)
        assert analyzer.get_df_gpu_timeline() is not None


class TestReportingInferenceSheets:
    def test_inference_all_report_variants(self, tmp_path):
        from TraceLens.Reporting.generate_perf_report_pytorch_inference import (
            generate_perf_report_pytorch as gen_inf,
        )

        trace = _write_trace(tmp_path, [
            ("aten::mm", "gemm_kernel", 100),
            ("aten::add", "vectorized_elementwise_kernel", 20),
            ("aten::native_layer_norm", "layer_norm_kernel", 30),
        ])
        gen_inf(
            profile_json_path=trace,
            output_csvs_dir=str(tmp_path / "out"),
            output_xlsx_path=str(tmp_path / "r.xlsx"),
            collective_analysis=True,
            enable_pseudo_ops=True,
            kernel_summary=True,
            short_kernel_study=True,
            include_overlap_info=True,
            group_by_parent_module=True,
            group_by_num_kernels=True,
            topk_ops=10,
            topk_roofline_ops=5,
            include_unlinked_kernels=True,
            include_call_stack=True,
        )
        assert (tmp_path / "out" / "gpu_timeline.csv").exists()
