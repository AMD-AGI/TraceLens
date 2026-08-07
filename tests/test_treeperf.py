###############################################################################
# Copyright (c) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for TraceLens/TreePerf/ covering tree_perf.py, jax_analyses.py,
and gpu_event_analyser.py.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from copy import deepcopy
from types import SimpleNamespace

import pandas as pd
import pytest

from TraceLens.Trace2Tree.trace_to_tree import TraceToTree
from TraceLens.TreePerf import (
    GPUEventAnalyser,
    JaxAnalyses,
    JaxGPUEventAnalyser,
    JaxTreePerfAnalyzer,
    PytorchGPUEventAnalyser,
    TreePerfAnalyzer,
)
from TraceLens.TreePerf.tree_perf import (
    get_compute_spec,
    get_max_achievable_tflops,
    normalize_dtype_to_precision,
)
from TraceLens.util import TraceEventUtils

JAX_CONV_TRACE = os.path.join(
    os.path.dirname(__file__),
    "traces/mi300/jax_conv_minimal_legacy/chi-mi300x-013.ord.vultr.cpe.ice.amd.com.xplane.pb",
)
GPU_ONLY_TRACE = os.path.join(
    os.path.dirname(__file__),
    "traces/mi210/gpu_only_trace/gpu_only_trace.json.gz",
)

pytestmark_jax = pytest.mark.xdist_group("jax_traces")


def _require_cuda_torch():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("Requires CUDA/HIP")
    return torch


def _make_gpu_event(
    uid, ts, dur, cat="kernel", name="kernel", pid=100, tid=100, args=None
):
    event = {
        "ph": "X",
        "UID": uid,
        "ts": ts,
        "dur": dur,
        "cat": cat,
        "name": name or "kernel",
        "pid": pid,
        "tid": tid,
    }
    if args is not None:
        event["args"] = args
    return event


def _make_jax_gpu_event(
    pid, tid, ts, dur, name="gemm", thread_name="Stream #0", args=None
):
    event = {
        "ph": "X",
        "pid": pid,
        "tid": tid,
        "ts": ts,
        "dur": dur,
        "name": name,
        "process": {"process_name": f"/device:GPU:{pid - 1}"},
        "thread": {"thread_name": thread_name},
    }
    if args is not None:
        event["args"] = args
    return event


def _mk_ac2g(corr_id, pid, tid, ts, phase):
    evt = {
        "ph": phase,
        "id": corr_id,
        "pid": pid,
        "tid": tid,
        "ts": ts,
        "cat": "ac2g",
        "name": "ac2g",
    }
    if phase == "f":
        evt["bp"] = "e"
    return evt


def _mk_pytorch_trace():
    corr = 100
    return [
        _make_gpu_event(
            "cpu",
            1000,
            100,
            "cpu_op",
            "aten::mm",
            pid=100,
            args={"Input Dims": [[32, 64], [64, 128]]},
        ),
        _make_gpu_event(
            "rt",
            1010,
            5,
            "cuda_runtime",
            "hipLaunchKernel",
            pid=100,
            args={"correlation": corr},
        ),
        _make_gpu_event(
            "kern",
            1050,
            50,
            "kernel",
            "gemm_kernel",
            pid=0,
            tid=7,
            args={"correlation": corr, "stream": 7},
        ),
        _mk_ac2g(corr, 0, 7, 1050, "s"),
        _mk_ac2g(corr, 0, 7, 1100, "f"),
    ]


def _build_analyzer(events, **kwargs):
    tree = TraceToTree(deepcopy(events))
    tree.build_tree(add_python_func=kwargs.pop("add_python_func", False))
    return TreePerfAnalyzer(tree, add_python_func=False, rebuild_tree=False, **kwargs)


class TestTreePerfModuleHelpers:
    @pytest.mark.parametrize(
        "dtype_str,expected",
        [
            (None, None),
            ("unknown", None),
            ("c10::half", "fp16"),
            ("bfloat16", "bf16"),
            ("float", "fp32"),
            ("fp8", "fp8"),
        ],
    )
    def test_normalize_dtype_to_precision(self, dtype_str, expected):
        assert normalize_dtype_to_precision(dtype_str) == expected

    def test_get_compute_spec_and_max_tflops(self):
        perf_model = SimpleNamespace(
            get_maf_type=lambda: "matrix",
            get_compute_precision=lambda: "fp16",
        )
        assert get_compute_spec(perf_model) == "matrix_fp16"
        arch = {"max_achievable_tflops": {"matrix_fp16": 100.0}}
        assert get_max_achievable_tflops(perf_model, arch) == 100.0
        assert get_max_achievable_tflops(perf_model, None) is None


class TestGPUEventAnalyserStatic:
    def test_merge_intervals_empty(self):
        assert GPUEventAnalyser.merge_intervals([]) == []

    def test_merge_intervals_overlapping(self):
        intervals = [(0, 10), (5, 15), (20, 30)]
        assert GPUEventAnalyser.merge_intervals(intervals) == [(0, 15), (20, 30)]

    def test_compute_metrics_dict_basic(self):
        events = {
            GPUEventAnalyser.all_gpu_key: [
                _make_gpu_event(1, 0, 100, "kernel"),
                _make_gpu_event(2, 200, 50, "gpu_memcpy"),
            ],
            GPUEventAnalyser.computation_key: [_make_gpu_event(1, 0, 100, "kernel")],
            GPUEventAnalyser.communication_key: [],
            GPUEventAnalyser.memcpy_key: [_make_gpu_event(2, 200, 50, "gpu_memcpy")],
        }
        for bucket in events.values():
            for event in bucket:
                event["t_end"] = event["ts"] + event["dur"]

        metrics = GPUEventAnalyser.compute_metrics_dict(events)
        assert metrics["computation_time"] == pytest.approx(100.0)
        assert metrics["total_memcpy_time"] == pytest.approx(50.0)
        assert metrics["total_time"] == pytest.approx(250.0)
        assert metrics["idle_time"] == pytest.approx(100.0)

    def test_compute_metrics_dict_micro_idle(self):
        events = {
            GPUEventAnalyser.all_gpu_key: [
                _make_gpu_event(1, 0, 50, "kernel"),
                _make_gpu_event(2, 100, 50, "kernel"),
            ],
            GPUEventAnalyser.computation_key: [
                _make_gpu_event(1, 0, 50, "kernel"),
                _make_gpu_event(2, 100, 50, "kernel"),
            ],
            GPUEventAnalyser.communication_key: [],
            GPUEventAnalyser.memcpy_key: [],
        }
        for bucket in events.values():
            for event in bucket:
                event["t_end"] = event["ts"] + event["dur"]

        metrics = GPUEventAnalyser.compute_metrics_dict(events, micro_idle_thresh_us=30)
        assert metrics["micro_idle_time"] == pytest.approx(0.0)
        assert metrics["macro_idle_time"] == pytest.approx(50.0)

    def test_verify_dict_gpu_event_lists_rejects_bad_keys(self):
        with pytest.raises(ValueError, match="Expected keys"):
            GPUEventAnalyser.verify_dict_gpu_event_lists({"bad_key": []})

    def test_verify_dict_gpu_event_lists_rejects_missing_ts(self):
        bad = {key: [{"dur": 1}] for key in GPUEventAnalyser.gpu_event_keys}
        with pytest.raises(ValueError, match="does not have 'ts' or 't_end'"):
            GPUEventAnalyser.verify_dict_gpu_event_lists(bad)

    def test_verify_dict_gpu_event_lists_rejects_empty_gpu(self):
        empty = {key: [] for key in GPUEventAnalyser.gpu_event_keys}
        with pytest.raises(ValueError, match="No GPU events found"):
            GPUEventAnalyser.verify_dict_gpu_event_lists(empty)

    def test_get_breakdown_df_from_dict(self):
        metrics = {
            "computation_time": 100.0,
            "idle_time": 50.0,
            "total_time": 150.0,
        }
        df = GPUEventAnalyser.get_breakdown_df_from_dict(metrics)
        assert set(df["type"]) == {"computation_time", "idle_time", "total_time"}

    def test_pytorch_analyser_is_subclass(self):
        assert issubclass(PytorchGPUEventAnalyser, GPUEventAnalyser)


class TestJaxGPUEventAnalyser:
    def test_detects_gpu_pids_from_process_name(self):
        events = [
            _make_jax_gpu_event(1, 1, 0, 10),
            {"ph": "X", "pid": 99, "tid": 1, "ts": 0, "dur": 1, "name": "host"},
        ]
        analyser = JaxGPUEventAnalyser(events)
        assert analyser.gpu_pids == [1]

    def test_categorizes_memcpy_comm_and_compute(self):
        events = [
            _make_jax_gpu_event(1, 1, 0, 10, name="memcpyHtoD"),
            _make_jax_gpu_event(1, 1, 20, 10, name="ncclAllReduce"),
            _make_jax_gpu_event(1, 1, 40, 10, name="Cijk_gemm"),
        ]
        analyser = JaxGPUEventAnalyser(events)
        buckets = analyser.get_gpu_event_lists(gpu_pid=1)
        assert len(buckets[GPUEventAnalyser.memcpy_key]) == 1
        assert len(buckets[GPUEventAnalyser.communication_key]) == 1
        assert len(buckets[GPUEventAnalyser.computation_key]) == 1

    def test_default_gpu_event_filter_stream_thread(self):
        stream_event = {"thread": {"thread_name": "Stream #1"}, "tid": 1}
        host_event = {"thread": {"thread_name": "Host"}, "tid": 1}
        legacy_event = {"tid": 1}
        assert JaxAnalyses.default_gpu_event_filter(stream_event) is True
        assert JaxAnalyses.default_gpu_event_filter(host_event) is False
        assert JaxAnalyses.default_gpu_event_filter(legacy_event) is True

    def test_compute_metrics_and_average(self):
        events = [
            _make_jax_gpu_event(1, 1, 0, 100, name="kernel_a"),
            _make_jax_gpu_event(2, 1, 0, 200, name="kernel_b"),
        ]
        analyser = JaxGPUEventAnalyser(events)
        assert analyser.compute_metrics(gpu_pid=1)["computation_time"] == pytest.approx(
            100.0
        )
        assert analyser.get_average_metrics()["computation_time"] == pytest.approx(
            150.0
        )
        assert set(analyser.get_breakdown_df()["gpu_pid"]) == {1, 2}

    def test_multigpu_breakdown_helpers(self):
        events = [
            _make_jax_gpu_event(1, 1, 0, 100),
            _make_jax_gpu_event(2, 1, 0, 200),
        ]
        analyser = JaxGPUEventAnalyser(events)
        frames = analyser.get_breakdown_df_multigpu()
        assert len(frames) == 2
        verify_df = analyser.get_average_df_verify_with_jax_analyses()
        assert "busy_time" in verify_df["type"].values

    def test_cpu_bucket_and_missing_gpu_pid(self):
        events = [
            _make_jax_gpu_event(1, 1, 0, 10),
            {"ph": "X", "pid": 99, "tid": 1, "ts": 0, "dur": 5, "name": "host"},
            {"ph": "X", "pid": 1, "tid": 1, "name": "metadata_only"},
        ]
        analyser = JaxGPUEventAnalyser(events)
        all_events = analyser.get_gpu_event_lists()
        assert GPUEventAnalyser.all_cpu_key in all_events[99]
        assert analyser.get_gpu_event_lists(gpu_pid=999) == {}


class TestJaxAnalyses:
    def test_is_gpu_stream_pid_and_tid(self):
        assert JaxAnalyses.is_gpu_stream_pid(1) is True
        assert JaxAnalyses.is_gpu_stream_pid(101) is False
        assert JaxAnalyses.is_gpu_stream_tid(1) is True
        assert JaxAnalyses.is_gpu_stream_tid(100) is False
        assert JaxAnalyses.is_gpu_stream_pid({"pid": 8}) is True

    def test_breakdown_compute_events_categorizes_gemm(self):
        events = [
            {
                TraceEventUtils.TraceKeys.PID: 1,
                TraceEventUtils.TraceKeys.Name: "Cijk_Alik_Bljk",
                TraceEventUtils.TraceKeys.Duration: 100.0,
            },
            {
                TraceEventUtils.TraceKeys.PID: 1,
                TraceEventUtils.TraceKeys.Name: "unknown_kernel",
                TraceEventUtils.TraceKeys.Duration: 50.0,
            },
        ]
        categorized, uncategorized = JaxAnalyses.breakdown_compute_events(
            events, group_by_gpu=False, group_by_name=True
        )
        assert categorized["GEMM"] == [1, 100.0]
        assert "unknown_kernel" in uncategorized

    def test_breakdown_compute_events_group_by_gpu(self):
        events = [
            {
                TraceEventUtils.TraceKeys.PID: 1,
                TraceEventUtils.TraceKeys.Name: "Cijk_gemm",
                TraceEventUtils.TraceKeys.Duration: 10.0,
            },
            {
                TraceEventUtils.TraceKeys.PID: 2,
                TraceEventUtils.TraceKeys.Name: "Cijk_gemm",
                TraceEventUtils.TraceKeys.Duration: 20.0,
            },
        ]
        categorized, _ = JaxAnalyses.breakdown_compute_events(
            events, group_by_gpu=True, group_by_name=False
        )
        assert categorized[1]["GEMM"] == [1, 10.0]
        assert categorized[2]["GEMM"] == [1, 20.0]

    def test_breakdown_compute_events_hlo_op_fallback(self):
        events = [
            {
                TraceEventUtils.TraceKeys.PID: 1,
                TraceEventUtils.TraceKeys.Name: "__amd_rocclr_fillBufferAligned.kd",
                TraceEventUtils.TraceKeys.Duration: 25.0,
                TraceEventUtils.TraceKeys.Args: {
                    TraceEventUtils.JaxKernelEventArgs.hlo_op: "te_fused_attn_forward_ffi.1"
                },
            }
        ]
        categorized, _ = JaxAnalyses.breakdown_compute_events(
            events, group_by_gpu=False
        )
        assert categorized["TE"] == [1, 25.0]

    def test_create_breakdown_df(self):
        events = {"GEMM": [2, 200.0], "Conv": [1, 100.0]}
        df = JaxAnalyses.create_breakdown_df(events, total_time=300.0, num_gpus=2)
        assert df.loc["GEMM", "count"] == 2
        assert df.loc["GEMM", "time ms per gpu"] == pytest.approx(0.1)

    def test_get_just_gpu_events(self):
        all_events = {
            1: {GPUEventAnalyser.computation_key: [{"name": "k1"}]},
            2: {GPUEventAnalyser.computation_key: []},
        }
        assert list(JaxAnalyses.get_just_gpu_events(all_events).keys()) == [1]

    def test_process_communication_events_from_xla_dump(self):
        xla_lines = [
            "HloModule jit_train\n",
            "  value: all-reduce-start.3 size=4096: allreduce.3[shape]\n",
            "  value: reduce-scatter.1 size=8192: scatter.1[shape]\n",
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.writelines(xla_lines)
            xla_path = f.name
        try:
            result = JaxAnalyses.process_communication_events_from_xla_dump(xla_path)
            assert result["all-reduce-start"] == [["3", "4096", "allreduce.3"]]
            assert result["reduce-scatter"] == [["1", "8192", "scatter.1"]]
        finally:
            os.unlink(xla_path)

    def test_process_communication_events_from_profile(self):
        events = [
            _make_jax_gpu_event(1, 1, 0, 100, name="Cijk_gemm"),
            _make_jax_gpu_event(
                2,
                1,
                10,
                1000,
                name="ncclAllGather",
                args={"hlo_op": "all-reduce-start.3"},
            ),
            _make_jax_gpu_event(2, 1, 0, 100, name="Cijk_gemm"),
            _make_jax_gpu_event(
                2,
                1,
                10,
                2000,
                name="ncclAllGather",
                args={"hlo_op": "all-reduce-start.3"},
            ),
        ]
        analyser = JaxGPUEventAnalyser(events)
        messages = {"all-reduce-start": [["3", "4096", "allreduce.3"]]}
        processed = JaxAnalyses.process_communication_events_from_profile(
            analyser, messages
        )
        assert "all-reduce" in processed
        assert processed["all-reduce"]["all-reduce-start.3"][1] == 4096

    def test_summarize_communication_data(self):
        comm_data = {
            "all-reduce": {
                "all-reduce-start.1": [100.0, 4096, 1, 40.96],
                "all-reduce-start.2": [200.0, 8192, 1, 40.96],
            }
        }
        summary = JaxAnalyses.summarize_communication_data(comm_data)
        df, bw_data, count_data, time_by_size, range_data = summary["all-reduce"]
        assert not df.empty
        assert not bw_data.empty
        assert not count_data.empty
        assert not time_by_size.empty
        assert not range_data.empty

    def test_get_perf_model_and_gemm_metrics(self):
        gemm_event = {
            TraceEventUtils.TraceKeys.Name: "Cijk_Alik_Bljk",
            TraceEventUtils.TraceKeys.Duration: 100.0,
        }
        assert JaxAnalyses.get_perf_model(gemm_event) is JaxAnalyses.JaxGemm
        op_params = {
            "M": 128,
            "N": 256,
            "K": 64,
            "Beta": 0,
            "Type": "f16",
            "Batch": 2,
        }
        metrics = JaxAnalyses.gemm_perf_metrics(gemm_event, op_params)
        assert metrics["GFLOPS"] > 0
        assert metrics["param: M"] == 128
        with pytest.raises(NotImplementedError):
            JaxAnalyses.JaxGemm(
                {
                    TraceEventUtils.JaxKernelEventArgs.hlo_op: op_params,
                    "kernel_names": ["x"],
                }
            ).flops_bwd()

    def test_create_gpu_summary(self):
        events = [
            _make_jax_gpu_event(1, 1, 0, 100, name="Cijk_gemm"),
            _make_jax_gpu_event(2, 1, 0, 200, name="Cijk_gemm"),
        ]
        analyser = JaxGPUEventAnalyser(events)
        breakdown_df, categorized_df, uncategorized_df = JaxAnalyses.create_gpu_summary(
            analyser
        )
        assert "type" in breakdown_df.columns
        assert "GEMM" in categorized_df.index
        assert isinstance(uncategorized_df, pd.DataFrame)


class TestTreePerfAnalyzer:
    def test_check_gpu_only(self):
        gpu_only_events = [_make_gpu_event(1, 0, 10, "kernel")]
        cpu_events = gpu_only_events + [
            {"ph": "X", "cat": "cpu_op", "name": "aten::mm", "dur": 1}
        ]
        assert (
            TreePerfAnalyzer(
                TraceToTree(deepcopy(gpu_only_events)), rebuild_tree=False
            ).check_gpu_only()
            is True
        )
        assert (
            TreePerfAnalyzer(
                TraceToTree(deepcopy(cpu_events)), rebuild_tree=False
            ).check_gpu_only()
            is False
        )

    def test_agg_kernels_in_subtree(self):
        analyzer = _build_analyzer(_mk_pytorch_trace())
        cpu_op = next(e for e in analyzer.tree.events if e["cat"] == "cpu_op")
        total_dur, kernel_uids = analyzer.agg_kernels_in_subtree(cpu_op)
        assert total_dur == 50
        assert len(kernel_uids) == 1

    def test_loop_and_aggregate_kernels(self):
        analyzer = _build_analyzer(_mk_pytorch_trace())
        cpu_ops = [e for e in analyzer.tree.events if e["cat"] == "cpu_op"]
        total, kernel_uids = analyzer.loop_and_aggregate_kernels(cpu_ops)
        assert total == 50
        assert len(kernel_uids) == 1

    def test_non_data_mov_filter(self):
        kernel = _make_gpu_event(1, 0, 10, "kernel", "aten::mm")
        data_mov = _make_gpu_event(
            2, 0, 10, "kernel", "at::native::direct_copy_kernel_cuda"
        )
        assert TreePerfAnalyzer.non_data_mov_filter(kernel) is True
        assert TreePerfAnalyzer.non_data_mov_filter(data_mov) is False

    def test_get_df_gpu_timeline_from_synthetic_trace(self):
        analyzer = _build_analyzer(_mk_pytorch_trace())
        df = analyzer.get_df_gpu_timeline()
        assert "busy_time" in df["type"].values

    def test_get_kernel_details_and_df_kernels(self):
        analyzer = _build_analyzer(_mk_pytorch_trace())
        kernel = next(e for e in analyzer.tree.events if e["cat"] == "kernel")
        details = analyzer.get_kernel_details(
            kernel, launcher_detail=True, cpu_op_detail=True
        )
        assert details["Kernel name"] == "gemm_kernel"
        assert details["Parent cpu_op"] in {"aten::matmul", "aten::mm"}
        df = analyzer.get_df_kernels(launcher_detail=True, cpu_op_detail=True)
        assert len(df) == 1

    def test_get_df_gpu_timeline_detect_recompute(self):
        events = [
            _make_gpu_event(1, 0, 100, "kernel"),
            _make_gpu_event(2, 200, 100, "kernel"),
        ]
        for event in events:
            event["tree"] = True
        tree = TraceToTree(events)
        analyzer = TreePerfAnalyzer(tree, rebuild_tree=False, detect_recompute=True)
        analyzer.tree.events[0]["is_recompute"] = True
        df = analyzer.get_df_gpu_timeline()
        assert "is_recompute" in df.columns

    def test_compute_fwd_bwd_perf_metrics_wrappers(self, monkeypatch):
        analyzer = _build_analyzer(_mk_pytorch_trace())
        calls = []

        def fake_compute(event, bwd=False, non_data_mov=False, perf_model_class=None):
            calls.append(bwd)
            return {"bwd": bwd}

        monkeypatch.setattr(analyzer, "compute_perf_metrics", fake_compute)
        assert analyzer.compute_fwd_perf_metrics({}) == {"bwd": False}
        assert analyzer.compute_bwd_perf_metrics({}) == {"bwd": True}
        assert calls == [False, True]

    @pytest.mark.skipif(
        not os.path.exists(GPU_ONLY_TRACE),
        reason=f"Trace not found: {GPU_ONLY_TRACE}",
    )
    def test_from_file_gpu_only_trace(self):
        analyzer = TreePerfAnalyzer.from_file(GPU_ONLY_TRACE, rebuild_tree=True)
        assert analyzer.check_gpu_only() is True
        assert not analyzer.get_df_gpu_timeline().empty


class TestJaxTreePerfAnalyzer:
    @pytest.fixture(scope="module")
    def jax_analyzer(self):
        if not os.path.exists(JAX_CONV_TRACE):
            pytest.skip(f"Trace not found: {JAX_CONV_TRACE}")
        return JaxTreePerfAnalyzer.from_file(profile_filepath=JAX_CONV_TRACE)

    def test_get_event_perf_model_name_branches(self):
        conv = {
            "gpu_kernel_op_cat": "Conv",
            "metadata": {
                "custom_call_target": "cudnn$convForward",
                "operands": ["bf16[1,16,32,60,104]{4,3,2,1,0}"],
            },
        }
        assert JaxTreePerfAnalyzer.get_event_perf_model_name(conv) == "jax_conv"
        assert (
            JaxTreePerfAnalyzer.get_event_perf_model_name({"gpu_kernel_op_cat": "GEMM"})
            == "rest"
        )

    def test_parse_operands_and_metadata(self):
        event = {
            "metadata": {
                "operands": ["bf16[8,768]{1,0}", "bf16[8,384]{1,0}"],
            }
        }
        dims, types, _ = JaxTreePerfAnalyzer.parse_operands(event)
        assert dims == ((8, 768), (8, 384))
        meta = JaxTreePerfAnalyzer.parse_metadata(event)
        assert meta["Input Dims"] == dims

    def test_parse_conv_te_and_gemm_metadata(self):
        conv_event = {
            "gpu_kernel_op_cat": "Conv",
            "metadata": {
                "custom_call_target": "cudnn$convForward",
                "operands": [
                    "bf16[1,16,32,60,104]{4,3,2,1,0}",
                    "bf16[5120,16,1,2,2]{4,3,2,1,0}",
                ],
                "output": "bf16[1,5120,34,31,53]{4,3,2,1,0}",
            },
        }
        assert JaxTreePerfAnalyzer.parse_conv_metadata(conv_event)["Filter Shape"] == (
            1,
            2,
            2,
        )

        te_event = {
            "metadata": {
                "operands": [
                    "bf16[1,8,4,128]{3,2,1,0}",
                    "bf16[1,8,4,128]{3,2,1,0}",
                    "bf16[1,8,4,128]{3,2,1,0}",
                    "bf16[0]{0}",
                ]
            }
        }
        te_meta = JaxTreePerfAnalyzer.parse_te_fused_attn_metadata(te_event)
        assert len(te_meta["Input Dims"]) == 3

        gemm_event = {
            "gpu_kernel_op_cat": "GEMM",
            "metadata": {
                "custom_call_target": "cublasLt_matmul",
                "operands": ["bf16[4,8]{1,0}", "bf16[8,16]{0,1}"],
                "output": "bf16[4,16]{1,0}",
                "backend_config": 'foo={"gemm_backend_config":{"beta":0}}',
                "computation": "gemm",
            },
        }
        gemm_meta = JaxTreePerfAnalyzer.parse_gemm_metadata(gemm_event)
        assert gemm_meta["Beta"] == 0
        full_meta = JaxTreePerfAnalyzer.get_event_metadata(conv_event)
        assert full_meta["Filter Shape"] == (1, 2, 2)

    def test_empty_kernel_launchers_and_summaries(self):
        analyzer = JaxTreePerfAnalyzer(TraceToTree([]), rebuild_tree=False)
        launchers = analyzer.get_kernel_launchers()
        assert isinstance(launchers, pd.DataFrame)
        assert launchers.empty
        assert JaxTreePerfAnalyzer.get_df_kernel_launchers_summary(pd.DataFrame()).empty

    def test_get_df_xla_perf_and_launch_latency(self):
        analyzer = JaxTreePerfAnalyzer(TraceToTree([]), rebuild_tree=False)
        df = pd.DataFrame(
            [{"kernel_details": [{"operands": ["bf16[2,4]{1,0}", "bf16[4,8]{0,1}"]}]}]
        )
        out = analyzer.get_df_xla_perf(df)
        assert out.loc[0, "total_input_bytes"] == 2 * 4 * 2 + 4 * 8 * 2

        parent = {"UID": 1, "ts": 0}
        child = {"UID": 2, "parent": 1, "ts": 25}
        analyzer.tree.events_by_uid = {1: parent, 2: child}
        assert analyzer.get_GPU_kernel_launch_latency(child) == 25
        assert math.isnan(analyzer.get_GPU_kernel_launch_latency({"UID": 3}))

    @pytestmark_jax
    def test_jax_gpu_timeline_and_averages(self, jax_analyzer):
        df_one = jax_analyzer.get_df_gpu_timeline(gpu_pid=1)
        busy = df_one.set_index("type")["time ms"]["busy_time"]
        assert busy > 0
        df_avg = jax_analyzer.get_df_gpu_events_averages()
        assert df_avg.set_index("type")["time ms"]["busy_time"] > 0


@pytest.mark.gpu
def test_tree_perf_analyzer_live_gpu_profile(tmp_path):
    _require_cuda_torch()
    import torch

    device = "cuda"
    model = torch.nn.Linear(32, 16, bias=False).to(device=device, dtype=torch.float16)
    x = torch.randn(4, 32, device=device, dtype=torch.float16)
    trace_path = tmp_path / "linear_trace.json"

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        on_trace_ready=lambda p: p.export_chrome_trace(str(trace_path)),
    ) as prof:
        for _ in range(3):
            model(x).sum().backward()
            prof.step()

    analyzer = TreePerfAnalyzer.from_file(str(trace_path), rebuild_tree=True)
    assert analyzer.check_gpu_only() is False
    assert "busy_time" in analyzer.get_df_gpu_timeline()["type"].values
    assert len(analyzer.get_kernel_launchers()) >= 1


@pytest.mark.gpu
def test_gpu_event_analyser_compute_metrics_on_live_trace(tmp_path):
    _require_cuda_torch()
    import torch

    x = torch.randn(8, 8, device="cuda")
    trace_path = tmp_path / "gemm_trace.json"

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        on_trace_ready=lambda p: p.export_chrome_trace(str(trace_path)),
    ) as prof:
        for _ in range(2):
            torch.mm(x, x)
            prof.step()

    with open(trace_path, encoding="utf-8") as f:
        events = json.load(f)["traceEvents"]

    metrics = GPUEventAnalyser(events).compute_metrics()
    assert metrics["computation_time"] > 0
    assert metrics["total_time"] > 0


@pytest.mark.gpu
def test_build_nn_module_latency_tree():
    _require_cuda_torch()
    corr = 100
    events = [
        _make_gpu_event("py1", 0, 500, "python_function", "nn.Module: Net", pid=100),
        _make_gpu_event(
            "py2", 10, 400, "python_function", "nn.Module: Net.sub", pid=100
        ),
        _make_gpu_event(
            "cpu", 20, 50, "cpu_op", "aten::mm", pid=100, args={"correlation": corr}
        ),
        _make_gpu_event(
            "rt",
            25,
            5,
            "cuda_runtime",
            "hipLaunchKernel",
            pid=100,
            args={"correlation": corr},
        ),
        _make_gpu_event(
            "kern",
            100,
            80,
            "kernel",
            "gemm",
            pid=0,
            tid=7,
            args={"correlation": corr, "stream": 7},
        ),
        _mk_ac2g(corr, pid=0, tid=7, ts=100, phase="s"),
        _mk_ac2g(corr, pid=0, tid=7, ts=100, phase="f"),
    ]
    tree = TraceToTree(events)
    tree.build_tree(add_python_func=True)
    analyzer = TreePerfAnalyzer(tree, add_python_func=True, rebuild_tree=False)
    root = next(e for e in tree.events if e["name"] == "nn.Module: Net")
    analyzer.build_nn_module_latency_tree(root)
    assert root["GPU Time"] == 80
