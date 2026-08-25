###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""CPU-only unit tests targeting uncovered paths in TraceLens.Reporting modules."""

from __future__ import annotations

import gzip
import json
import os
import textwrap
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from TraceLens.Reporting.generate_multi_rank_collective_report_pytorch import (
    find_trace_files,
    generate_collective_report,
)
from TraceLens.Reporting.generate_perf_report_genesis import (
    _cleanup_work_dir,
    generate_perf_report_genesis,
)
from TraceLens.Reporting.generate_perf_report_pftrace_hip_activity import (
    _write_markdown_report,
    generate_perf_report_pftrace_hip_activity,
)
from TraceLens.Reporting.generate_perf_report_pytorch import (
    _find_entry_point,
    _is_wrapper_frame,
    add_truncated_kernel_details,
    apply_extension as apply_extension_pytorch,
    generate_perf_report_pytorch,
    get_dfs_short_kernels as get_dfs_short_kernels_pytorch,
)
from TraceLens.Reporting.generate_perf_report_pytorch_inference import (
    add_truncated_kernel_details as add_truncated_kernel_details_inference,
    apply_extension as apply_extension_inference,
    classify_graph_capture_trace,
    generate_perf_report_pytorch as generate_inference_report,
    get_dfs_short_kernels as get_dfs_short_kernels_inference,
    perf_report_sanity_check,
)
from TraceLens.Reporting.pftrace_hip_activity_analysis import (
    Event,
    HIPEvent,
    build_hip_summary_df,
    build_kernel_summary_df_for_name,
)
from TraceLens.Reporting.pftrace_utils import ensure_trace_json

pytestmark = pytest.mark.filterwarnings(
    "ignore:Source column .* not found.*:UserWarning",
    "ignore:There are hipgraph launches.*:UserWarning",
    "ignore:Found .* events with failed performance metric.*:UserWarning",
    "ignore:Input list of events is empty.*:UserWarning",
    "ignore:dict_cat2names_extension is deprecated.*:UserWarning",
)

KERNEL_TRACE_CSV = """\
"Kind","Agent_Id","Queue_Id","Stream_Id","Thread_Id","Dispatch_Id","Kernel_Id","Kernel_Name","Correlation_Id","Start_Timestamp","End_Timestamp","LDS_Block_Size","Scratch_Size","VGPR_Count","Accum_VGPR_Count","SGPR_Count","Workgroup_Size_X","Workgroup_Size_Y","Workgroup_Size_Z","Grid_Size_X","Grid_Size_Y","Grid_Size_Z"
"KERNEL_DISPATCH","Agent 2",1,0,70,1,33,"__amd_rocclr_fillBufferAligned",119662,172352210005122,172352210008687,0,0,12,4,48,256,1,1,256,1,1
"KERNEL_DISPATCH","Agent 2",1,0,70,2,16,"kernel_step_1_c532_0_kernel_6_range_for",119670,172352210061004,172352210062686,0,0,4,4,16,1,1,1,1,1,1
"KERNEL_DISPATCH","Agent 2",1,0,70,3,31,"func_broad_phase_c402_0_kernel_3_range_for",119696,172352210143326,172352210149335,0,0,16,0,32,512,1,1,512,1,1
"""


def _mk_event(cat, name, ts, dur, pid, tid, args=None):
    return {
        "ph": "X",
        "cat": cat,
        "name": name,
        "pid": pid,
        "tid": tid,
        "ts": ts,
        "dur": dur,
        "args": args or {},
    }


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


def _build_synthetic_trace(kernel_specs):
    events = []
    ts = 1000
    corr_id = 100
    cpu_pid, cpu_tid = 100, 100
    gpu_pid, gpu_tid = 0, 7

    for cpu_op_name, kernel_name, kernel_dur in kernel_specs:
        cpu_op_ts = ts
        cpu_op_dur = 100
        events.append(
            _mk_event(
                "cpu_op",
                cpu_op_name,
                ts=cpu_op_ts,
                dur=cpu_op_dur,
                pid=cpu_pid,
                tid=cpu_tid,
                args={"Input Dims": [[32, 64]], "Input type": ["float"]},
            )
        )
        events.append(
            _mk_event(
                "cuda_runtime",
                "hipLaunchKernel",
                ts=cpu_op_ts + 10,
                dur=5,
                pid=cpu_pid,
                tid=cpu_tid,
                args={"correlation": corr_id},
            )
        )
        kernel_ts = cpu_op_ts + 50
        events.append(
            _mk_event(
                "kernel",
                kernel_name,
                ts=kernel_ts,
                dur=kernel_dur,
                pid=gpu_pid,
                tid=gpu_tid,
                args={"correlation": corr_id, "stream": 7},
            )
        )
        events.append(_mk_ac2g(corr_id, gpu_pid, gpu_tid, kernel_ts, "s"))
        events.append(_mk_ac2g(corr_id, gpu_pid, gpu_tid, kernel_ts, "f"))
        ts += cpu_op_dur + 200
        corr_id += 1

    return {"traceEvents": events}


def _write_trace(tmp_path: Path, specs, name="trace.json") -> str:
    path = tmp_path / name
    path.write_text(json.dumps(_build_synthetic_trace(specs)))
    return str(path)


def _create_genesis_capture(tmp_path: Path) -> Path:
    capture = tmp_path / "capture"
    kernel_trace = capture / "kernel_trace"
    kernel_trace.mkdir(parents=True)
    (kernel_trace / "kernel_kernel_trace.csv").write_text(KERNEL_TRACE_CSV)
    (capture / "run.log").write_text("wall_time=4.00s\n")
    return capture


def _minimal_pftrace_events():
    return [
        {
            "ph": "X",
            "cat": "gpu_activity",
            "name": "xla_fusion_42",
            "pid": 0,
            "tid": 7,
            "ts": 1000,
            "dur": 50000,
            "args": {"agent": "gpu_0", "begin_ns": 1000000, "delta_ns": 50000000},
        },
        {
            "ph": "X",
            "cat": "hip_api",
            "name": "hipLaunchKernelGGL",
            "pid": 100,
            "tid": 1,
            "ts": 900,
            "dur": 20,
            "args": {"stream_ID": 0},
        },
    ]


class _MockShortKernelAnalyzer:
    def __init__(self, gpu_only=False, kernels=None, total_time_ms=1.0):
        self.gpu_only = gpu_only
        self.total_time_ms = total_time_ms
        self._kernels = kernels if kernels is not None else pd.DataFrame(
            {
                "Kernel duration (µs)": [5.0, 8.0, 50.0],
                "Kernel name": ["k_short_a", "k_short_b", "k_long"],
                "Parent cpu_op": ["aten::mm"] * 3,
                "Input dims": ["[[32, 64]]"] * 3,
                "Input strides": [""] * 3,
                "Concrete Inputs": [""] * 3,
            }
        )

    def get_df_kernels(self):
        return self._kernels


# ---------------------------------------------------------------------------
# perf_report_sanity_check (inference)
# ---------------------------------------------------------------------------


def test_sanity_check_include_nccl_busy_time():
    events = [{"name": "ncclAllReduce", "cat": "kernel"}]
    tl = pd.DataFrame({"type": ["busy_time"], "time ms": [0.05]})
    kl = pd.DataFrame({"total_direct_kernel_time_sum": [60.0]})
    up = pd.DataFrame({"Kernel Time (µs)_sum": [60.0]})
    result = perf_report_sanity_check(events, tl, kl, up, include_nccl=True)
    assert result["kl_time_pass"]
    assert result["total_gpu_events"] == 1


def test_sanity_check_time_mismatch():
    events = [{"name": "k", "cat": "kernel"}]
    tl = pd.DataFrame({"type": ["computation_time"], "time ms": [10.0]})
    kl = pd.DataFrame({"total_direct_kernel_time_sum": [1.0]})
    up = pd.DataFrame({"Kernel Time (µs)_sum": [1.0]})
    result = perf_report_sanity_check(events, tl, kl, up)
    assert not result["kl_time_pass"]
    assert not result["up_time_pass"]


def test_sanity_check_kernel_details_column():
    events = [{"name": "k_a", "cat": "kernel"}]
    tl = pd.DataFrame({"type": ["computation_time"], "time ms": [0.1]})
    kl = pd.DataFrame(
        {
            "total_direct_kernel_time": [100.0],
            "kernel_details": [[{"name": "k_a", "count": 1}]],
        }
    )
    up = pd.DataFrame(
        {
            "Kernel Time (µs)": [100.0],
            "kernel_details": [[{"name": "k_a", "count": 1}]],
        }
    )
    result = perf_report_sanity_check(events, tl, kl, up)
    assert result["kl_count_pass"]
    assert result["up_count_pass"]


def test_sanity_check_missing_kernel_details_column(capsys):
    events = [{"name": "k", "cat": "kernel"}]
    tl = pd.DataFrame({"type": ["computation_time"], "time ms": [0.1]})
    kl = pd.DataFrame({"total_direct_kernel_time_sum": [100.0]})
    up = pd.DataFrame({"Kernel Time (µs)_sum": [100.0]})
    result = perf_report_sanity_check(events, tl, kl, up)
    assert not result["kl_count_pass"]
    assert "WARNING" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# classify_graph_capture_trace
# ---------------------------------------------------------------------------


def test_classify_skips_when_execution_details_exist(tmp_path, capsys):
    (tmp_path / "execution_details.json").write_text("[]")
    classify_graph_capture_trace(str(tmp_path))
    assert "Skipping classification" in capsys.readouterr().out


def test_classify_from_capture_annotation(tmp_path):
    events = [
        {
            "name": "vllm/v1/worker/gpu_model_runner.py(10): _dummy_run",
            "ts": 1,
        },
        {"name": "capture_32_FULL", "cat": "user_annotation", "ts": 2},
    ]
    (tmp_path / "graph.json").write_text(json.dumps({"traceEvents": events}))
    classify_graph_capture_trace(str(tmp_path))
    details = json.loads((tmp_path / "execution_details.json").read_text())
    assert details[0]["batch_size"] == 32
    assert details[0]["mode"] == "FULL"


def test_classify_inferred_from_stream_captures(tmp_path):
    events = [
        {
            "name": "vllm/v1/worker/gpu_model_runner.py(10): _dummy_run",
            "ts": 1,
        },
        {"cat": "cuda_runtime", "name": "cudaStreamBeginCapture", "ts": 2},
        {"cat": "cuda_runtime", "name": "cudaStreamBeginCapture", "ts": 3},
        {
            "cat": "cpu_op",
            "name": "aten::mm",
            "args": {"Input Dims": [[64, 128], [32, 64]]},
        },
    ]
    (tmp_path / "capture.json.gz").write_bytes(
        gzip.compress(json.dumps({"traceEvents": events}).encode())
    )
    classify_graph_capture_trace(str(tmp_path))
    details = json.loads((tmp_path / "execution_details.json").read_text())
    assert details[0]["mode"] == "PIECEWISE"
    assert details[0]["batch_size"] == 64


def test_classify_json_gz_roundtrip(tmp_path):
    events = [
        {"name": "vllm/v1/worker/gpu_model_runner.py(1): _dummy_run", "ts": 0},
        {"cat": "cuda_runtime", "name": "cudaStreamBeginCapture", "ts": 1},
        {"cat": "cpu_op", "name": "x", "args": {"Input Dims": [[8, 16]]}},
    ]
    gz_path = tmp_path / "capture.json.gz"
    gz_path.write_bytes(gzip.compress(json.dumps({"traceEvents": events}).encode()))
    classify_graph_capture_trace(str(tmp_path))
    details = json.loads((tmp_path / "execution_details.json").read_text())
    assert details[0]["batch_size"] == 8
    assert details[0]["mode"] == "FULL"


def test_classify_no_trace_files_exits(tmp_path):
    with pytest.raises(SystemExit) as exc:
        classify_graph_capture_trace(str(tmp_path))
    assert exc.value.code == 0


# ---------------------------------------------------------------------------
# get_dfs_short_kernels
# ---------------------------------------------------------------------------


def test_inference_get_dfs_short_kernels_with_data():
    analyzer = _MockShortKernelAnalyzer()
    hist, grouped = get_dfs_short_kernels_inference(analyzer, topk=1)
    assert not hist.empty
    assert len(grouped) == 1
    assert "Short Kernel duration (µs) sum" in grouped.columns


def test_inference_get_dfs_short_kernels_empty():
    empty = pd.DataFrame(columns=["Kernel duration (µs)", "Kernel name"])
    analyzer = _MockShortKernelAnalyzer(kernels=empty)
    hist, grouped = get_dfs_short_kernels_inference(analyzer)
    assert hist.empty
    assert grouped.empty


def test_inference_get_dfs_short_kernels_gpu_only():
    kernels = pd.DataFrame(
        {"Kernel duration (µs)": [3.0], "Kernel name": ["k"]},
    )
    analyzer = _MockShortKernelAnalyzer(gpu_only=True, kernels=kernels)
    hist, grouped = get_dfs_short_kernels_inference(analyzer)
    assert not hist.empty
    assert grouped.iloc[0]["Kernel name"] == "k"


def test_pytorch_get_dfs_short_kernels_with_data():
    analyzer = _MockShortKernelAnalyzer()
    hist, grouped = get_dfs_short_kernels_pytorch(analyzer, topk=2)
    assert len(grouped) == 2


# ---------------------------------------------------------------------------
# apply_extension
# ---------------------------------------------------------------------------


def test_inference_apply_extension_op_category(tmp_path):
    ext_path = tmp_path / "ext.py"
    ext_path.write_text(
        textwrap.dedent(
            """
            def tree_postprocess_extension(tree):
                tree.events[0]["ext_applied"] = True

            op_category_extension = {"custom::op": "Other"}
            """
        )
    )
    tree = SimpleNamespace(events=[{}], label_non_gpu_paths=lambda: None)
    analyzer = SimpleNamespace(
        tree=tree,
        op_to_perf_model_class_map={},
    )
    apply_extension_inference(analyzer, str(ext_path))
    assert analyzer.tree.events[0]["ext_applied"]


def test_inference_apply_extension_invalid_perf_model(tmp_path):
    ext_path = tmp_path / "bad_ext.py"
    ext_path.write_text("perf_model_extension = {'aten::mm': 'not_a_class'}")
    analyzer = SimpleNamespace(
        tree=SimpleNamespace(events=[], label_non_gpu_paths=lambda: None),
        op_to_perf_model_class_map={},
    )
    with pytest.raises(ValueError, match="category attribute"):
        apply_extension_inference(analyzer, str(ext_path))


def test_pytorch_apply_extension_valid_perf_model(tmp_path):
    ext_path = tmp_path / "ext.py"
    ext_path.write_text(
        textwrap.dedent(
            """
            class DummyGemm:
                category = "GEMM"

            perf_model_extension = {"aten::mm": DummyGemm}
            """
        )
    )
    analyzer = SimpleNamespace(
        tree=SimpleNamespace(events=[], label_non_gpu_paths=lambda: None),
        op_to_perf_model_class_map={},
    )
    apply_extension_pytorch(analyzer, str(ext_path))
    assert "aten::mm" in analyzer.op_to_perf_model_class_map


# ---------------------------------------------------------------------------
# trunc / wrapper helpers
# ---------------------------------------------------------------------------


def test_inference_add_truncated_kernel_details_missing_column():
    df = pd.DataFrame({"other": [1]})
    out = add_truncated_kernel_details_inference(df, source_col="missing")
    assert "trunc_missing" not in out.columns


def test_pytorch_is_wrapper_frame():
    assert _is_wrapper_frame("torch/nn/modules/module.py(5): _call_impl")
    assert _is_wrapper_frame("torch/_ops.py(10): wrapper_custom")
    assert not _is_wrapper_frame("user_code.py(10): forward")


def test_find_entry_point_stripped_suffix():
    stack = str(["user.py(1): addmm_triton", "aten::addmm_triton"])
    result = _find_entry_point(stack, "aten::addmm_triton_340")
    assert result["traversal"] == "outward"
    assert "user.py" in result["entry_point"]


# ---------------------------------------------------------------------------
# generate_perf_report_pytorch / inference with synthetic traces
# ---------------------------------------------------------------------------


def test_pytorch_report_synthetic_minimal(tmp_path):
    trace = _write_trace(
        tmp_path,
        [("aten::mm", "gemm_kernel", 100), ("aten::relu", "relu_kernel", 20)],
    )
    out = str(tmp_path / "csvs")
    result = generate_perf_report_pytorch(
        profile_json_path=trace,
        output_csvs_dir=out,
        collective_analysis=False,
        kernel_summary=True,
        short_kernel_study=True,
    )
    assert "gpu_timeline" in result
    assert "kernel_summary" in result
    assert os.path.isfile(os.path.join(out, "gpu_timeline.csv"))


def test_inference_report_synthetic_with_flags(tmp_path):
    trace = _write_trace(tmp_path, [("aten::mm", "gemm_kernel", 100)])
    out = str(tmp_path / "csvs")
    xlsx = str(tmp_path / "report.xlsx")
    result = generate_inference_report(
        profile_json_path=trace,
        output_csvs_dir=out,
        output_xlsx_path=xlsx,
        collective_analysis=False,
        kernel_summary=True,
        short_kernel_study=True,
        group_by_parent_module=True,
        group_by_num_kernels=True,
    )
    assert os.path.isfile(xlsx)
    assert "short_kernel_histogram" in result
    assert "short_kernels_summary" in result


def test_inference_report_with_extension_additional_dfs(tmp_path):
    trace = _write_trace(tmp_path, [("aten::mm", "gemm_kernel", 100)])
    ext_path = tmp_path / "extra.py"
    ext_path.write_text(
        textwrap.dedent(
            """
            import pandas as pd

            def get_additional_dataframes_extension(tree):
                return {"custom_extra": pd.DataFrame({"value": [42]})}
            """
        )
    )
    result = generate_inference_report(
        profile_json_path=trace,
        output_csvs_dir=str(tmp_path / "csvs"),
        collective_analysis=False,
        extension_file=str(ext_path),
    )
    assert "custom_extra" in result


def test_pytorch_report_include_overlap_and_call_stack(tmp_path):
    trace = _write_trace(
        tmp_path,
        [
            ("aten::mm", "gemm_kernel", 100),
            ("aten::add", "add_kernel", 15),
        ],
    )
    result = generate_perf_report_pytorch(
        profile_json_path=trace,
        output_csvs_dir=str(tmp_path / "csvs"),
        collective_analysis=False,
        include_overlap_info=True,
        include_call_stack=True,
        group_by_num_kernels=True,
    )
    assert "unified_perf_summary" in result
    assert "call_stack_full" in result["unified_perf_summary"].columns


# ---------------------------------------------------------------------------
# multi-rank collective report
# ---------------------------------------------------------------------------


def test_find_trace_files_empty_dir(tmp_path, capsys):
    assert find_trace_files(str(tmp_path)) == []
    assert "No trace files found" in capsys.readouterr().out


def test_collective_report_trace_dir_synthetic(tmp_path):
    for rank in range(2):
        trace = _make_trace(rank, 3)
        (tmp_path / f"rank{rank}_trace.json").write_text(json.dumps(trace))
    out = str(tmp_path / "nccl_out")
    dfs = generate_collective_report(
        trace_dir=str(tmp_path),
        world_size=2,
        output_csvs_dir=out,
        detailed_analysis=False,
        gpus_per_node=2,
        strict_world_size_check=False,
    )
    assert "nccl_summary_implicit_sync" in dfs
    assert os.path.isfile(os.path.join(out, "nccl_summary_implicit_sync.csv"))


def test_collective_report_trace_pattern_non_strict(tmp_path):
    for rank in (0, 2):
        trace = _make_trace(rank, 2)
        (tmp_path / f"trace_rank_{rank}.json").write_text(json.dumps(trace))
    pattern = str(tmp_path / "trace_rank_*.json")
    dfs = generate_collective_report(
        trace_pattern=pattern,
        world_size=4,
        output_csvs_dir=str(tmp_path / "out"),
        strict_world_size_check=False,
        detailed_analysis=True,
    )
    assert "nccl_long" in dfs


def test_collective_report_xlsx_output(tmp_path):
    trace = _make_trace(0, 2)
    (tmp_path / "rank0_trace.json").write_text(json.dumps(trace))
    xlsx = str(tmp_path / "report.xlsx")
    generate_collective_report(
        trace_dir=str(tmp_path),
        world_size=1,
        output_xlsx_path=xlsx,
        strict_world_size_check=False,
    )
    assert os.path.isfile(xlsx)


def _make_trace(rank, n_collectives):
    events = []
    base_ts = 1_000_000 + rank * 50
    for i in range(n_collectives):
        ts = base_ts + i * 1000 + rank * 5
        events.append(
            {
                "ph": "X",
                "cat": "kernel",
                "name": "void rcclGenericKernel<1, false>(ncclDevKernelArgsStorage<4096ul>)",
                "pid": rank,
                "tid": 3,
                "ts": ts,
                "dur": 50,
                "args": {
                    "External id": 100 + i,
                    "device": rank,
                    "stream": 3,
                    "correlation": 50 + i,
                },
            }
        )
    return {"traceEvents": events}


# ---------------------------------------------------------------------------
# genesis report
# ---------------------------------------------------------------------------


def test_cleanup_work_dir(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    (work / "temp.json").write_text("{}")
    _cleanup_work_dir(work)
    assert not work.exists()


def test_generate_perf_report_genesis_integration(tmp_path):
    capture = _create_genesis_capture(tmp_path)
    out = tmp_path / "analysis"
    reports = generate_perf_report_genesis(
        capture_dir=str(capture),
        output_dir=str(out),
        short_kernel_study=False,
        keep_work=True,
    )
    assert "rocprof" in reports
    assert (out / "genesis_perf_report.xlsx").exists()
    assert (out / "genesis_summary.md").exists()
    assert reports["rocprof"]["kernel_summary_by_category"] is not None


@mock.patch(
    "TraceLens.Reporting.generate_perf_report_genesis.generate_perf_report_pftrace_memory_copy"
)
@mock.patch(
    "TraceLens.Reporting.generate_perf_report_genesis.generate_perf_report_pftrace_hip_activity"
)
@mock.patch("TraceLens.Reporting.generate_perf_report_genesis.pftrace_to_json")
def test_generate_perf_report_genesis_with_pftrace(
    mock_pftrace_to_json,
    mock_hip_activity,
    mock_memory_copy,
    tmp_path,
):
    capture = _create_genesis_capture(tmp_path)
    pftrace = capture / "kernel_trace" / "kernel_results.pftrace"
    pftrace.write_bytes(b"\x00")
    mock_pftrace_to_json.return_value = capture / "pf.json"
    mock_hip_activity.return_value = {"hip_summary": pd.DataFrame({"api": ["hipMalloc"]})}
    mock_memory_copy.return_value = {
        "memory_copy_by_copy_bytes": pd.DataFrame({"copy_bytes": [1024], "count": [1]})
    }
    out = tmp_path / "analysis_pf"
    reports = generate_perf_report_genesis(
        capture_dir=str(capture),
        output_dir=str(out),
        short_kernel_study=False,
    )
    assert "pftrace_hip_activity" in reports
    assert "pftrace_memory_copy" in reports
    mock_hip_activity.assert_called_once()


# ---------------------------------------------------------------------------
# pftrace modules
# ---------------------------------------------------------------------------


def test_write_markdown_report(tmp_path):
    df_cat = pd.DataFrame({"GPU ID": [0], "Category": ["xla"], "Time (ms)": [1.0]})
    md_path = tmp_path / "report.md"
    _write_markdown_report(
        md_path,
        df_category=df_cat,
        xla_top=[("xla_fusion_1", 1_000_000, 2, 0.5)],
        used_fav3=False,
        agents=["gpu_0"],
        kernel_df=pd.DataFrame({"Name": ["k1"], "Instances": [1]}),
        hip_df=pd.DataFrame({"Name": ["hipMalloc"], "Instances": [1]}),
    )
    text = md_path.read_text()
    assert "ROCm Perfetto Trace Report" in text
    assert "xla_fusion_1" in text


def test_ensure_trace_json_returns_json_path(tmp_path):
    trace = tmp_path / "trace.json"
    trace.write_text('{"traceEvents": []}')
    assert ensure_trace_json(str(trace)) == str(trace.resolve())


def test_ensure_trace_json_unsupported_format(tmp_path):
    bad = tmp_path / "trace.bin"
    bad.write_bytes(b"\x00")
    with pytest.raises(ValueError, match="Unsupported trace format"):
        ensure_trace_json(str(bad))


def test_build_kernel_summary_df_for_name():
    events = [
        Event(gpu=0, name="gemm_1", dur_ns=1000, ts_ns=0),
        Event(gpu=0, name="gemm_2", dur_ns=2000, ts_ns=0),
    ]
    df = build_kernel_summary_df_for_name(events, baseline_total_ns=3000, merge_names=True)
    assert len(df) == 1
    assert df.iloc[0]["Instances"] == 2


@pytest.mark.parametrize("group", ["name", "name+stream", "name+op", "name+stream+op"])
def test_build_hip_summary_df_groups(group):
    hip_events = [
        HIPEvent(name="hipMalloc", dur_ns=100, ts_ns=0, pid=1, tid=1, stream_id=1, operation=2),
        HIPEvent(name="hipMalloc", dur_ns=200, ts_ns=0, pid=1, tid=1, stream_id=1, operation=2),
    ]
    df = build_hip_summary_df(hip_events, group=group)
    assert not df.empty
    assert df.iloc[0]["Instances"] == 2


def test_pftrace_hip_activity_markdown_output(tmp_path):
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(json.dumps({"traceEvents": _minimal_pftrace_events()}))
    md_path = tmp_path / "out.md"
    generate_perf_report_pftrace_hip_activity(
        trace_path=str(trace_path),
        output_md_path=str(md_path),
        min_event_ns=0,
        kernel_summary=True,
        hip_summary=True,
    )
    assert md_path.exists()
    assert "ROCm Perfetto Trace Report" in md_path.read_text()


def test_pftrace_hip_activity_default_xlsx_path(tmp_path):
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(json.dumps({"traceEvents": _minimal_pftrace_events()}))
    generate_perf_report_pftrace_hip_activity(
        trace_path=str(trace_path),
        min_event_ns=0,
    )
    assert (tmp_path / "trace_pftrace_activity_report.xlsx").exists()
