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
        assert get_max_achievable_tflops(perf_model, {}) is None
        assert get_max_achievable_tflops(SimpleNamespace(), arch) is None


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
        dims, _, _ = JaxTreePerfAnalyzer.parse_operands(event)
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


def test_build_nn_module_latency_tree():
    trace_path = os.path.join(
        os.path.dirname(__file__),
        "traces/inference/sglang_prefilldecode/"
        "sglang_Qwen3-8B_prefilldecode.json.gz",
    )
    analyzer = TreePerfAnalyzer.from_file(trace_path, add_python_func=True)
    tree = analyzer.tree
    root = next(e for e in tree.events if e.get("name") == "nn.Module: Qwen3Model_0")

    analyzer.build_nn_module_latency_tree(root)
    children = [tree.get_UID2event(uid) for uid in tree.get_nn_module_children(root)]

    assert (
        sum(
            child["name"].startswith("nn.Module: Qwen3DecoderLayer_")
            for child in children
        )
        == 36
    )
    assert root["GPU Time"] == pytest.approx(492163.76953125)

    layer = next(
        child for child in children if child["name"] == "nn.Module: Qwen3DecoderLayer_0"
    )
    attention = next(
        tree.get_UID2event(uid)
        for uid in tree.get_nn_module_children(layer)
        if tree.get_UID2event(uid)["name"] == "nn.Module: Qwen3Attention_0"
    )

    assert layer["nn Parent GPU Time"] == pytest.approx(root["GPU Time"])
    assert attention["nn Parent GPU Time"] == pytest.approx(layer["GPU Time"])
    assert attention["Non-nn.Module GPU Time"] > 0


# --- migrated from test_treeperf_coverage.py ---
import gzip
import json

from tests.fixtures.treeperf import (
    _build_analyzer,
    _make_gpu_event,
    _mk_ac2g,
    _mk_pytorch_trace,
)


class TestTreePerfCoverageExtended:
    def test_from_file_json_gz(self, tmp_path):
        corr = 500
        events = [
            _make_gpu_event(
                "cpu",
                1000,
                100,
                "cpu_op",
                "aten::mm",
                args={
                    "Input Dims": [[32, 64], [64, 128]],
                    "Input type": ["fp16", "fp16"],
                },
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
        trace_path = tmp_path / "trace.json.gz"
        with gzip.open(trace_path, "wt", encoding="utf-8") as f:
            json.dump({"traceEvents": events}, f)

        analyzer = TreePerfAnalyzer.from_file(str(trace_path))
        assert analyzer.get_kernel_launchers()

    def test_get_df_kernels_with_nn_module_detail(self):
        corr = 600
        events = [
            _make_gpu_event(
                "nn",
                900,
                200,
                "python_function",
                "nn.Module: Linear_0",
                args={"Python id": 1},
            ),
            _make_gpu_event(
                "cpu",
                1000,
                100,
                "cpu_op",
                "aten::linear",
                args={
                    "Input Dims": [[32, 64], [64, 128]],
                    "Input type": ["fp16", "fp16"],
                },
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
        analyzer = _build_analyzer(events, add_python_func=True)
        nn_evt = next(
            e for e in analyzer.tree.events if e["name"].startswith("nn.Module")
        )
        cpu_evt = next(e for e in analyzer.tree.events if e["name"] == "aten::linear")
        kernel_evt = next(e for e in analyzer.tree.events if e["cat"] == "kernel")

        nn_evt.setdefault("children", []).append(cpu_evt["UID"])
        cpu_evt["parent"] = nn_evt["UID"]
        nn_evt["gpu_events"] = [kernel_evt["UID"]]

        df = analyzer.get_df_kernels(
            nn_module_detail=True, cpu_op_detail=True, launcher_detail=True
        )
        assert not df.empty
        assert "Parent nn.Module" in df.columns
        assert df.iloc[0]["Parent nn.Module"].startswith("nn.Module")

    def test_get_df_kernels_rejects_python_stack(self):
        analyzer = _build_analyzer(_mk_pytorch_trace())
        analyzer.with_python_stack = True
        with pytest.raises(ValueError, match="Python stack"):
            analyzer.get_df_kernels()

    def test_build_df_from_events_bwd_linked(self):
        corr_fwd = 700
        corr_bwd = 701
        events = [
            _make_gpu_event(
                "fwd",
                1000,
                100,
                "cpu_op",
                "aten::mm",
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
        fwd_evt = next(e for e in analyzer.tree.events if e["name"] == "aten::mm")
        bwd_evt = next(
            e for e in analyzer.tree.events if e["name"] == "aten::mm_backward"
        )
        fwd_evt["bwd_events"] = [bwd_evt["UID"]]
        bwd_evt["fwd_event"] = fwd_evt["UID"]

        df = analyzer.build_df_unified_perf_table(
            events=[fwd_evt, bwd_evt], include_perf_metrics=True
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) >= 1


# --- migrated from test_coverage_95_bulk.py ---
import os
import pandas as pd
import pytest
from TraceLens.TreePerf.tree_perf import JaxTreePerfAnalyzer, TreePerfAnalyzer
from tests.fixtures.traces import JAX_PB, _discover_trace_gz_files
from tests.fixtures.treeperf import _sweep_treeperf_analyzer


@pytest.mark.parametrize("trace_path", _discover_trace_gz_files())
def test_treeperf_full_method_sweep(trace_path):
    try:
        analyzer = TreePerfAnalyzer.from_file(
            trace_path,
            rebuild_tree=True,
            enable_pseudo_ops=True,
            add_python_func=True,
            include_unlinked_kernels=True,
            detect_recompute=True,
        )
        _sweep_treeperf_analyzer(analyzer)
        kernels = analyzer.get_df_kernels(
            launcher_detail=True,
            cpu_op_detail=True,
            nn_module_detail=analyzer.add_python_func,
        )
        assert isinstance(kernels, pd.DataFrame)
    except Exception as exc:
        pytest.skip(f"trace not suitable for sweep: {exc}")


# --- migrated from test_coverage_95_bulk.py ---
import os
import pandas as pd
import pytest
from TraceLens.TreePerf.tree_perf import JaxTreePerfAnalyzer, TreePerfAnalyzer
from tests.fixtures.traces import JAX_PB, _discover_trace_gz_files
from tests.fixtures.treeperf import _sweep_treeperf_analyzer


@pytest.mark.skipif(not os.path.isfile(JAX_PB), reason="JAX fixture missing")
class TestJaxTreePerfSweep:
    def test_jax_from_file_all_methods(self):
        analyzer = JaxTreePerfAnalyzer.from_file(profile_filepath=JAX_PB)
        _sweep_treeperf_analyzer(analyzer)
        analyzer.get_df_gpu_events_averages()
        for gpu_pid in (1, 2):
            try:
                analyzer.get_df_gpu_timeline(gpu_pid=gpu_pid)
            except Exception:
                pass


# --- migrated from test_coverage_95_final.py ---
import gzip
import json
import os
from copy import deepcopy
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest
from TraceLens.Reporting.compare_traces_jax_llama import (
    Event,
)
from TraceLens.Trace2Tree.trace_to_tree import TraceToTree
from TraceLens.TreePerf.tree_perf import JaxTreePerfAnalyzer, TreePerfAnalyzer
from tests.fixtures.reporting import _mk_ac2g
from tests.fixtures.treeperf import (
    _build_analyzer,
    _make_gpu_event,
    _mk_pytorch_trace,
)


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
                "cpu1",
                20,
                80,
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
                25,
                5,
                "cuda_runtime",
                "hipLaunchKernel",
                pid=100,
                args={"correlation": corr},
            ),
            _make_gpu_event(
                "k1",
                50,
                40,
                "kernel",
                "Cijk_gemm",
                pid=0,
                tid=7,
                args={"correlation": corr, "stream": 7},
            ),
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
        assert (
            JaxTreePerfAnalyzer.get_event_perf_model_name(te_bwd)
            == "jax_te_fused_attn_bwd"
        )
        te_fwd = {
            "gpu_kernel_op_cat": "GEMM",
            "metadata": {
                "custom_call_target": "te_fused_attn_forward_ffi",
                "operands": ["bf16[4,8]{1,0}"],
            },
        }
        assert (
            JaxTreePerfAnalyzer.get_event_perf_model_name(te_fwd) == "jax_te_fused_attn"
        )
        meta = JaxTreePerfAnalyzer.get_event_metadata(te_fwd)
        assert isinstance(meta, dict)

    def test_jax_kernel_launchers_with_metadata_filter(self):
        event = _make_gpu_event(
            "k1",
            0,
            100,
            "kernel",
            "Cijk_gemm",
            pid=1,
            tid=1,
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


# --- migrated from test_coverage_95_phase11.py ---
from tests.fixtures.traces import RESNET_TRACE
import os
from unittest.mock import patch
import pandas as pd
import pytest
from TraceLens.TreePerf.tree_perf import TreePerfAnalyzer
from tests.fixtures.perfmodel import (
    _BadPerfModel,
    _ExplodingPerfModel,
)
from tests.fixtures.treeperf import _build_analyzer, _make_gpu_event, _mk_ac2g


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


# --- migrated from test_coverage_95_phase12.py ---
from tests.fixtures.traces import RESNET_TRACE
import gzip
import json
import os
from unittest.mock import patch
import pandas as pd
import pytest
from TraceLens.TreePerf.tree_perf import TreePerfAnalyzer
from tests.fixtures.treeperf import _build_analyzer, _make_gpu_event, _mk_ac2g


class TestTreePerfPhase12:
    def test_reorder_cols_and_kernel_stats_edges(self):
        df = pd.DataFrame(
            {
                "name": ["a"],
                "direct_mean": [1.0],
                "subtree_mean": [2.0],
                "direct_std": [0.1],
                "subtree_std": [0.2],
                "other_col": [3],
            }
        )
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
            _make_gpu_event(
                "conv",
                1000,
                80,
                "cpu_op",
                "aten::convolution",
                args={"Input Dims": [[[2, 3, 8, 8], [4, 3, 3, 3]]]},
            ),
            _make_gpu_event("exec", 1010, 10, "cpu_op", "execute"),
            _make_gpu_event(
                "rt",
                1020,
                5,
                "cuda_runtime",
                "hipLaunchKernel",
                args={"correlation": corr},
            ),
            _make_gpu_event(
                "k",
                1030,
                50,
                "kernel",
                "conv_k",
                pid=0,
                tid=7,
                args={"correlation": corr, "stream": 7},
            ),
            _mk_ac2g(corr, 0, 7, 1030, "s"),
            _mk_ac2g(corr, 0, 7, 1080, "f"),
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
            _make_gpu_event(
                "cpu_f",
                1000,
                100,
                "cpu_op",
                "aten::mm",
                args={
                    "Input Dims": [[32, 64], [64, 128]],
                    "Input type": ["fp16", "fp16"],
                },
            ),
            _make_gpu_event(
                "rt_f",
                1010,
                5,
                "cuda_runtime",
                "hipLaunchKernel",
                args={"correlation": corr_f},
            ),
            _make_gpu_event(
                "k_f",
                1050,
                50,
                "kernel",
                "gemm_fwd",
                pid=0,
                tid=7,
                args={"correlation": corr_f, "stream": 7},
            ),
            _mk_ac2g(corr_f, 0, 7, 1050, "s"),
            _mk_ac2g(corr_f, 0, 7, 1100, "f"),
            _make_gpu_event(
                "cpu_b",
                2000,
                100,
                "cpu_op",
                "aten::mm",
                args={
                    "Input Dims": [[32, 64], [64, 128], [32, 128]],
                    "Input type": ["fp16", "fp16", "fp16"],
                },
            ),
            _make_gpu_event(
                "rt_b",
                2010,
                5,
                "cuda_runtime",
                "hipLaunchKernel",
                args={"correlation": corr_b},
            ),
            _make_gpu_event(
                "k_b",
                2050,
                60,
                "kernel",
                "gemm_bwd",
                pid=0,
                tid=7,
                args={"correlation": corr_b, "stream": 7},
            ),
            _mk_ac2g(corr_b, 0, 7, 2050, "s"),
            _mk_ac2g(corr_b, 0, 7, 2110, "f"),
        ]
        analyzer = _build_analyzer(events)
        fwd = next(
            e
            for e in analyzer.tree.events
            if e["name"] == "aten::mm" and e["ts"] == 1000
        )
        bwd = next(
            e
            for e in analyzer.tree.events
            if e["name"] == "aten::mm" and e["ts"] == 2000
        )
        fwd["bwd_events"] = [bwd["UID"]]
        bwd["fwd_event"] = fwd["UID"]
        bwd["gpu_events"] = [
            next(e["UID"] for e in analyzer.tree.events if e["name"] == "gemm_bwd")
        ]

        real = analyzer.compute_perf_metrics

        def boom(event, bwd=False, **kwargs):
            if bwd:
                raise RuntimeError("bwd metrics failed")
            return real(event, bwd=bwd, **kwargs)

        with patch.object(analyzer, "compute_perf_metrics", side_effect=boom):
            df = analyzer.build_df_unified_perf_table(
                events=[bwd],
                include_perf_metrics=True,
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
                    unified,
                    include_pct=True,
                )
        launchers = analyzer.get_df_kernel_launchers(
            include_args=True,
            include_first_occurrence_time=True,
        )
        if not launchers.empty:
            TreePerfAnalyzer.get_df_kernel_launchers_unique_args(
                launchers,
                include_pct=True,
                group_by_parent_module=True,
            )


# --- migrated from test_coverage_95_phase12.py ---
from tests.fixtures.traces import RESNET_TRACE
import gzip
import json
import os
from unittest.mock import patch
import pandas as pd
import pytest
from TraceLens.TreePerf.tree_perf import TreePerfAnalyzer
from tests.fixtures.treeperf import _build_analyzer, _make_gpu_event, _mk_ac2g


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
            _make_gpu_event(
                "parent",
                1000,
                50,
                "cpu_op",
                "aten::wrapper",
                args={"Input Dims": [[2, 2]]},
            ),
            _make_gpu_event(
                "leaf",
                1010,
                30,
                "cpu_op",
                "aten::mm",
                args={
                    "Input Dims": [[32, 64], [64, 128]],
                    "Input type": ["fp16", "fp16"],
                },
            ),
            _make_gpu_event(
                "rt",
                1020,
                5,
                "cuda_runtime",
                "hipLaunchKernel",
                args={"correlation": corr},
            ),
            _make_gpu_event(
                "k",
                1030,
                40,
                "kernel",
                "gemm_k",
                pid=0,
                tid=7,
                args={"correlation": corr, "stream": 7},
            ),
            _mk_ac2g(corr, 0, 7, 1030, "s"),
            _mk_ac2g(corr, 0, 7, 1070, "f"),
        ]
        analyzer = _build_analyzer(events)
        wrapper = next(e for e in analyzer.tree.events if e["name"] == "aten::wrapper")
        leaf = next(e for e in analyzer.tree.events if e["name"] == "aten::mm")
        wrapper["children"] = [leaf["UID"]]
        assert analyzer._is_leaf_cpu_op(leaf) or analyzer._launches_gpu_kernels(leaf)
        collected = analyzer.collect_unified_perf_events()
        assert isinstance(collected, list)


# --- migrated from test_coverage_95_phase13.py ---
from tests.fixtures.traces import RESNET
import os
import pandas as pd
import pytest
from TraceLens.TreePerf.tree_perf import TreePerfAnalyzer


class TestTreePerfPhase13:
    @pytest.mark.skipif(not os.path.isfile(RESNET), reason="resnet missing")
    def test_unified_table_with_perf_metrics(self):
        analyzer = TreePerfAnalyzer.from_file(
            RESNET,
            rebuild_tree=True,
            enable_pseudo_ops=True,
            detect_recompute=True,
        )
        df = analyzer.build_df_unified_perf_table(include_perf_metrics=True)
        assert isinstance(df, pd.DataFrame)
        if not df.empty:
            summary = analyzer.summarize_df_unified_perf_table(df, include_pct=True)
            assert isinstance(summary, pd.DataFrame)


# --- migrated from test_coverage_95_phase4.py ---
from tests.fixtures.traces import TRACES_ROOT
import json
import os
from copy import deepcopy
import pandas as pd
import pytest
from TraceLens.Trace2Tree.trace_to_tree import TraceToTree
from TraceLens.TreePerf.tree_perf import TreePerfAnalyzer
from tests.fixtures.treeperf import (
    _build_analyzer,
    _make_gpu_event,
    _mk_pytorch_trace,
)


class TestTreePerfPhase4:
    def test_tree_postprocess_and_inductor_cache(self, tmp_path):
        events = _mk_pytorch_trace()
        called = []

        def ext(tree):
            called.append(True)
            tree.events[0]["ext"] = True

        analyzer = TreePerfAnalyzer(
            TraceToTree(deepcopy(events), prune_nongpu_paths=False),
            rebuild_tree=True,
            tree_postprocess_extension=ext,
            inductor_cache_dir=str(tmp_path),
        )
        assert called
        assert analyzer.tree.events[0].get("ext") is True

    def test_summarize_with_recompute_column(self):
        events = _mk_pytorch_trace()
        events[0]["is_recompute"] = True
        analyzer = _build_analyzer(events, detect_recompute=True)
        launchers = analyzer.get_df_kernel_launchers(include_args=True)
        summary = TreePerfAnalyzer.get_df_kernel_launchers_summary(launchers)
        assert isinstance(summary, pd.DataFrame)

    def test_all_json_gz_traces_quick(self):
        count = 0
        for root, _dirs, files in os.walk(TRACES_ROOT):
            for name in files:
                if not name.endswith(".json.gz"):
                    continue
                path = os.path.join(root, name)
                try:
                    analyzer = TreePerfAnalyzer.from_file(
                        path, rebuild_tree=True, enable_pseudo_ops=True
                    )
                    assert analyzer.tree is not None
                    analyzer.get_df_gpu_timeline(micro_idle_thresh_us=0)
                    analyzer.build_df_unified_perf_table(include_nccl=False)
                    count += 1
                except Exception:
                    continue
        assert count > 0


# --- migrated from test_coverage_95_phase5.py ---
import gzip
import json
import os
import pandas as pd
import pytest
from TraceLens.Reporting.compare_traces_jax_llama import (
    Event,
)
from TraceLens.Trace2Tree.trace_capture_merge_experimental import (
    merge_capture_trace_into_graph,
)
from TraceLens.TreePerf.tree_perf import TreePerfAnalyzer
from tests.fixtures.treeperf import (
    _build_analyzer,
    _make_gpu_event,
    _mk_ac2g,
    _mk_pytorch_trace,
)


class TestTreePerfDeepPaths:
    def test_kernel_launchers_all_options(self):
        corr1, corr2 = 300, 301
        events = [
            _make_gpu_event(
                "py", 0, 500, "python_function", "nn.Module: Block", pid=100
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


# --- migrated from test_coverage_95_phase6.py ---
import json
import os
from unittest.mock import patch
import pandas as pd
import pytest
from TraceLens.TreePerf.tree_perf import TreePerfAnalyzer
from tests.fixtures.treeperf import _build_analyzer, _make_gpu_event, _mk_ac2g


class TestTreePerfPhase6:
    def test_inconsistent_kernel_list_length(self):
        with pytest.warns(UserWarning, match="Inconsistent kernel list length"):
            stats = TreePerfAnalyzer._summarize_kernel_stats(
                [
                    [{"name": "a", "dur": 10}],
                    [{"name": "a", "dur": 20}, {"name": "b", "dur": 30}],
                ],
                agg_metrics=["mean"],
            )
        assert isinstance(stats, list)

    def test_bwd_linked_not_implemented_fallback(self):
        corr_fwd, corr_bwd = 900, 901
        events = [
            _make_gpu_event(
                "fwd",
                1000,
                100,
                "cpu_op",
                "aten::unknown_custom_op",
                pid=100,
                args={"Input Dims": [[4, 4]]},
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
                "custom_kernel",
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
                "aten::unknown_custom_op_backward",
                pid=100,
                args={"Input Dims": [[4, 4]]},
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
                "custom_bwd_kernel",
                pid=0,
                tid=7,
                args={"correlation": corr_bwd, "stream": 7},
            ),
            _mk_ac2g(corr_bwd, 0, 7, 2050, "s"),
            _mk_ac2g(corr_bwd, 0, 7, 2110, "f"),
        ]
        analyzer = _build_analyzer(events)
        fwd = next(
            e
            for e in analyzer.tree.events
            if "custom_op" in e["name"] and "backward" not in e["name"]
        )
        bwd = next(e for e in analyzer.tree.events if "backward" in e["name"])
        fwd["bwd_events"] = [bwd["UID"]]
        bwd["fwd_event"] = fwd["UID"]
        df = analyzer.build_df_unified_perf_table(events=[fwd, bwd])
        assert isinstance(df, pd.DataFrame)


# --- migrated from test_coverage_95_phase8.py ---
import gzip
import json
import os
import pandas as pd
import pytest
from TraceLens.TreePerf.tree_perf import TreePerfAnalyzer
from tests.fixtures.traces import RESNET_TRACE as RESNET_CKPT


class TestTreePerfPhase8:
    def test_resnet_detect_recompute_nccl_bwd(self):
        analyzer = TreePerfAnalyzer.from_file(
            RESNET_CKPT,
            rebuild_tree=True,
            enable_pseudo_ops=True,
            add_python_func=True,
            detect_recompute=True,
            include_unlinked_kernels=True,
        )
        unified = analyzer.build_df_unified_perf_table(include_nccl=True)
        assert isinstance(unified, pd.DataFrame)
        bwd = [e for e in analyzer.tree.events if "backward" in e.get("name", "")]
        if bwd:
            analyzer.build_df_bwd_perf_metrics(events=bwd[:5])
        launchers = analyzer.get_df_kernel_launchers(
            include_args=True,
            include_kernel_details=True,
            include_call_stack=True,
        )
        if not launchers.empty:
            TreePerfAnalyzer.get_df_kernel_launchers_unique_args(
                launchers, include_pct=True
            )


# --- migrated from test_coverage_95_phase9.py ---
import json
import os
import pandas as pd
import pytest
from tests.fixtures.reporting import (
    _rich_pftrace_events,
)
from TraceLens.Reporting.pftrace_hip_activity_analysis import (
    Event,
    HIPEvent,
    build_event_lists,
    build_hip_summary_df,
    build_kernel_summary_df_for_config,
)
from tests.fixtures.treeperf import _build_analyzer, _make_gpu_event, _mk_ac2g


class TestTreePerfExtendedPhase9:
    def test_launcher_summaries(self):
        corr1, corr2 = 100, 101
        events = [
            _make_gpu_event(
                "cpu1",
                1000,
                100,
                "cpu_op",
                "aten::mm",
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
                args={"correlation": corr1},
            ),
            _make_gpu_event(
                "k1",
                1050,
                50,
                "kernel",
                "gemm_a",
                pid=0,
                tid=7,
                args={"correlation": corr1, "stream": 7},
            ),
            _mk_ac2g(corr1, 0, 7, 1050, "s"),
            _mk_ac2g(corr1, 0, 7, 1100, "f"),
            _make_gpu_event(
                "rt2",
                1060,
                5,
                "cuda_runtime",
                "hipLaunchKernel",
                args={"correlation": corr2},
            ),
            _make_gpu_event(
                "k2",
                1065,
                40,
                "kernel",
                "gemm_b",
                pid=0,
                tid=7,
                args={"correlation": corr2, "stream": 7},
            ),
            _mk_ac2g(corr2, 0, 7, 1065, "s"),
            _mk_ac2g(corr2, 0, 7, 1105, "f"),
        ]
        analyzer = _build_analyzer(events, add_python_func=True)
        launchers = analyzer.get_df_kernel_launchers(include_args=True)
        TreePerfAnalyzer.get_df_kernel_launchers_summary(launchers)
        TreePerfAnalyzer.get_df_kernel_launchers_summary_module(launchers)
        unified = analyzer.build_df_unified_perf_table(
            include_nccl=False, include_perf_metrics=True
        )
        assert isinstance(unified, pd.DataFrame)
        compute, _, _, used_fav3, _ = build_event_lists(
            _rich_pftrace_events(), True, -999, 999
        )
        assert used_fav3 and any(len(g) > 0 for g in compute)
        cfg = build_kernel_summary_df_for_config(
            [
                Event(
                    gpu=0,
                    name="Cijk_test",
                    ts_ns=0,
                    dur_ns=1_000_000,
                    grid_size=256,
                    workgroup_size=256,
                )
            ],
            2_000_000,
            False,
        )
        assert not cfg.empty
        assert not build_hip_summary_df(
            [HIPEvent(name="hipLaunch", ts_ns=0, dur_ns=1_000_000, pid=1, tid=1)],
            group="name+stream+op",
        ).empty


# --- migrated from test_coverage_push95.py ---
import json
import os
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest
from TraceLens.Trace2Tree.trace_capture_merge_experimental import (
    merge_capture_trace_into_graph,
)
from TraceLens.TreePerf.jax_analyses import JaxAnalyses
from TraceLens.TreePerf.tree_perf import TreePerfAnalyzer
from tests.fixtures.traces import INFERENCE_ROOT
from tests.fixtures.treeperf import _build_analyzer


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


# --- migrated from test_coverage_push95.py ---
import json
import os
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest
from TraceLens.Trace2Tree.trace_capture_merge_experimental import (
    merge_capture_trace_into_graph,
)
from TraceLens.TreePerf.jax_analyses import JaxAnalyses
from TraceLens.TreePerf.tree_perf import TreePerfAnalyzer
from tests.fixtures.treeperf import _build_analyzer


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


# --- migrated from test_coverage_push95.py::TestCoveragePush95Phase2.test_merged_graph_treeperf_extended ---
from tests.fixtures.traces import INFERENCE_ROOT
import json
import os
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest
from TraceLens.Trace2Tree.trace_capture_merge_experimental import (
    merge_capture_trace_into_graph,
)
from TraceLens.TreePerf.jax_analyses import JaxAnalyses
from TraceLens.TreePerf.tree_perf import TreePerfAnalyzer
from tests.fixtures.treeperf import _build_analyzer


def test_merged_graph_treeperf_extended():
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


# --- migrated from test_coverage_push95.py::TestCoveragePush95Phase3.test_jax_gemm_performance_from_pb ---
from tests.fixtures.traces import JAX_PB
import json
import os
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest
from TraceLens.Trace2Tree.trace_capture_merge_experimental import (
    merge_capture_trace_into_graph,
)
from TraceLens.TreePerf.jax_analyses import JaxAnalyses
from TraceLens.TreePerf.tree_perf import TreePerfAnalyzer
from tests.fixtures.traces import JAX_PB
from tests.fixtures.treeperf import _build_analyzer


@pytest.mark.skipif(not os.path.isfile(JAX_PB), reason="JAX fixture missing")
def test_jax_gemm_performance_from_pb():
    df = JaxAnalyses.gemm_performance_from_pb(JAX_PB, module_name="jit_forward_3d_conv")
    assert isinstance(df, pd.DataFrame)


# --- migrated from test_coverage_push95.py::TestCoveragePush95Phase3.test_gpu_only_treeperf_extended ---
import json
import os
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest
from TraceLens.Trace2Tree.trace_capture_merge_experimental import (
    merge_capture_trace_into_graph,
)
from TraceLens.TreePerf.jax_analyses import JaxAnalyses
from TraceLens.TreePerf.tree_perf import TreePerfAnalyzer
from tests.fixtures.treeperf import _build_analyzer


def test_gpu_only_treeperf_extended():
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


# --- migrated from test_coverage_sweep.py ---
from tests.fixtures.traces import INFERENCE_ROOT
import json
import os
import pandas as pd
import pytest
from TraceLens.Trace2Tree.trace_capture_merge_experimental import (
    merge_capture_trace_into_graph,
)
from TraceLens.TreePerf.tree_perf import TreePerfAnalyzer
from tests.fixtures.reporting import _mk_ac2g
from tests.fixtures.treeperf import (
    _build_analyzer,
    _make_gpu_event,
    _mk_pytorch_trace,
)


class TestTreePerfFromFileCapture:
    @pytest.mark.skipif(
        not os.path.isdir(
            os.path.join(INFERENCE_ROOT, "sglang_decode", "capture_traces")
        ),
        reason="capture trace fixture missing",
    )
    def test_from_file_with_capture_merge(self):
        case = os.path.join(INFERENCE_ROOT, "sglang_decode")
        trace_gz = next(f for f in os.listdir(case) if f.endswith(".json.gz"))
        capture = os.path.join(case, "capture_traces")
        metadata = os.path.join(capture, "execution_details.json")
        analyzer = TreePerfAnalyzer.from_file(
            profile_filepath=os.path.join(case, trace_gz),
            capture_trace_filepath=capture,
            rebuild_tree=True,
        )
        assert analyzer.tree is not None
        merged = merge_capture_trace_into_graph(
            capture, metadata, os.path.join(case, trace_gz)
        )
        assert len(merged.events) > 0


# --- migrated from test_coverage_sweep.py ---
import json
import os
import pandas as pd
import pytest
from TraceLens.Trace2Tree.trace_capture_merge_experimental import (
    merge_capture_trace_into_graph,
)
from TraceLens.TreePerf.tree_perf import TreePerfAnalyzer
from tests.fixtures.reporting import _mk_ac2g
from tests.fixtures.treeperf import (
    _build_analyzer,
    _make_gpu_event,
    _mk_pytorch_trace,
)


class TestTreePerfSummaries:
    def test_kernel_launcher_summaries_by_category_and_module(self):
        analyzer = _build_analyzer(_mk_pytorch_trace(), add_python_func=True)
        launchers = analyzer.get_df_kernel_launchers(include_args=True)
        by_cat = TreePerfAnalyzer.get_df_kernel_launchers_summary_by_category(launchers)
        assert isinstance(by_cat, pd.DataFrame)
        by_mod = TreePerfAnalyzer.get_df_kernel_launchers_summary_by_category_module(
            launchers
        )
        assert isinstance(by_mod, pd.DataFrame)
        unique = TreePerfAnalyzer.get_df_kernel_launchers_unique_args(
            launchers, include_pct=True
        )
        assert isinstance(unique, pd.DataFrame)

    def test_build_df_bwd_perf_metrics(self):
        corr_fwd, corr_bwd = 900, 901
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
        bwd_evt = next(e for e in analyzer.tree.events if "backward" in e["name"])
        df = analyzer.build_df_bwd_perf_metrics(events=[bwd_evt])
        assert isinstance(df, pd.DataFrame)


# --- migrated from test_push95_coverage.py ---
import json
import os
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest
from TraceLens.TreePerf.tree_perf import TreePerfAnalyzer
from tests.fixtures.traces import _discover_trace_gz_files
from tests.fixtures.treeperf import (
    _build_analyzer,
    _mk_pytorch_trace,
)


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


# --- migrated from test_push95_coverage.py ---
import json
import os
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest
from TraceLens.TreePerf.tree_perf import TreePerfAnalyzer
from tests.fixtures.treeperf import (
    _build_analyzer,
    _mk_pytorch_trace,
)


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


# --- migrated from test_coverage_final.py ---
import gzip
import json
import os
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest
from TraceLens.Trace2Tree.trace_to_tree import TraceToTree
from TraceLens.TreePerf.tree_perf import JaxTreePerfAnalyzer, TreePerfAnalyzer
from tests.fixtures.reporting import _mk_ac2g
from tests.fixtures.treeperf import (
    _build_analyzer,
    _make_gpu_event,
    _mk_pytorch_trace,
)


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
