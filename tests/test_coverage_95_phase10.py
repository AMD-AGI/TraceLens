###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Phase-10: final ~140-line coverage push for >=95% total."""

from __future__ import annotations

import importlib
import json
import sys
import types
from copy import deepcopy
from typing import Dict, List
from unittest.mock import patch

import pandas as pd

from TraceLens.Agent.Analysis.utils import arch_utils
from TraceLens.Reporting.generate_perf_report_pftrace_hip_activity import (
    _write_markdown_report,
    generate_perf_report_pftrace_hip_activity,
)
from TraceLens.Reporting.pftrace_hip_activity_analysis import (
    PftraceHipActivityAnalyzer,
    classify,
)
from TraceLens.Reporting.tracediff_comparison_extension import (
    tracediff_perf_summary_from_diff_stats,
)
from TraceLens.Trace2Tree.extensions.moe_aiter_pseudo_ops import (
    _create_pseudo_op_moe_fused_aiter,
    _has_cpu_op_descendant,
    create_pseudo_ops_moe_fused_aiter,
)
from TraceLens.Trace2Tree.extensions.moe_flydsl_pseudo_ops import (
    FUSED_MOE_PARENT,
    create_pseudo_ops_moe_flydsl,
)
from TraceLens.Trace2Tree.extensions.moe_gptq_awq_pseudo_ops import (
    _create_pseudo_op_moe_gptq_awq,
    create_pseudo_ops_moe_gptq_awq,
)
from TraceLens.Trace2Tree.trace_to_tree import TraceToTree

from tests.test_reporting_coverage import _write_trace
from tests.test_trace2tree import _add_gpu_chain, _mk_event
from tests.test_treeperf_coverage import _build_analyzer, _make_gpu_event, _mk_ac2g


def _build_tree(events: List[Dict], add_python_func: bool = False) -> TraceToTree:
    tree = TraceToTree(deepcopy(events), prune_nongpu_paths=False)
    tree.build_tree(add_python_func=add_python_func)
    return tree


def _full_pftrace_events():
    """Events exercising every pftrace classify branch and analyzer option."""
    return [
        {
            "ph": "X",
            "cat": "gpu_activity",
            "name": "ncclAllReduce",
            "pid": 0,
            "tid": 7,
            "ts": 1000,
            "dur": 50000,
            "args": {
                "agent": "gpu_0",
                "begin_ns": 1_000_000_000,
                "delta_ns": 50_000_000,
            },
        },
        {
            "ph": "X",
            "cat": "gpu_activity",
            "name": "Cijk_AB",
            "pid": 0,
            "tid": 7,
            "ts": 2000,
            "dur": 40000,
            "args": {
                "agent": "gpu_0",
                "begin_ns": 2_000_000_000,
                "delta_ns": 40_000_000,
                "grid_size": 128,
                "workgroup_size": 64,
                "VGPR_Count": 16,
            },
        },
        {
            "ph": "X",
            "cat": "gpu_activity",
            "name": "FmhaBwd_kernel_func",
            "pid": 0,
            "tid": 7,
            "ts": 3000,
            "dur": 30000,
            "args": {
                "agent": "gpu_0",
                "begin_ns": 3_000_000_000,
                "delta_ns": 30_000_000,
            },
        },
        {
            "ph": "X",
            "cat": "gpu_activity",
            "name": "FmhaFwd_main",
            "pid": 0,
            "tid": 7,
            "ts": 3500,
            "dur": 25000,
            "args": {
                "agent": "gpu_0",
                "begin_ns": 3_500_000_000,
                "delta_ns": 25_000_000,
            },
        },
        {
            "ph": "X",
            "cat": "gpu_activity",
            "name": "memcpyHtoD",
            "pid": 0,
            "tid": 7,
            "ts": 3600,
            "dur": 20000,
            "args": {
                "agent": "gpu_0",
                "begin_ns": 3_600_000_000,
                "delta_ns": 20_000_000,
            },
        },
        {
            "ph": "X",
            "cat": "gpu_activity",
            "name": "transformer_engine_linear",
            "pid": 0,
            "tid": 7,
            "ts": 3700,
            "dur": 15000,
            "args": {
                "agent": "gpu_0",
                "begin_ns": 3_700_000_000,
                "delta_ns": 15_000_000,
            },
        },
        {
            "ph": "X",
            "cat": "gpu_activity",
            "name": "aiter::fmha_fwd_kernel",
            "pid": 0,
            "tid": 7,
            "ts": 3800,
            "dur": 12000,
            "args": {
                "agent": "gpu_0",
                "begin_ns": 3_800_000_000,
                "delta_ns": 12_000_000,
            },
        },
        {
            "ph": "X",
            "cat": "gpu_activity",
            "name": "aiter::fmha_bwd_kernel",
            "pid": 0,
            "tid": 7,
            "ts": 3900,
            "dur": 11000,
            "args": {
                "agent": "gpu_0",
                "begin_ns": 3_900_000_000,
                "delta_ns": 11_000_000,
            },
        },
        {
            "ph": "X",
            "cat": "gpu_activity",
            "name": "fillBuffer_kernel",
            "pid": 0,
            "tid": 7,
            "ts": 4000,
            "dur": 8000,
            "args": {
                "agent": "gpu_0",
                "begin_ns": 4_000_000_000,
                "delta_ns": 8_000_000,
            },
        },
        {
            "ph": "X",
            "cat": "gpu_activity",
            "name": "xla_generic_fusion",
            "pid": 0,
            "tid": 7,
            "ts": 4100,
            "dur": 7000,
            "args": {
                "agent": "gpu_0",
                "begin_ns": 4_100_000_000,
                "delta_ns": 7_000_000,
            },
        },
        {
            "ph": "X",
            "cat": "hip_api",
            "name": "hipLaunchKernel",
            "pid": 100,
            "tid": 1,
            "ts": 900,
            "dur": 5000,
            "args": {
                "stream_ID": 0,
                "operation": 1,
                "begin_ns": 900_000,
                "delta_ns": 5_000_000,
            },
        },
    ]


class TestPftraceClassifyAndReportPhase10:
    def test_classify_all_branches(self):
        assert classify("ncclRing") == "rccl"
        assert classify("Cijk_AB") == "gemm"
        assert classify("FmhaBwd_kernel_func") == "ckbwd"
        assert classify("FmhaFwd_x") == "ckfwd"
        assert classify("memcpyDtoH") == "memcpy"
        assert classify("transformer_engine_linear") == "te"
        assert classify("aiter::fmha_fwd_x") == "aiterfwd"
        assert classify("aiter::fmha_bwd_x") == "aiterbwd"
        assert classify("fillBuffer_x") == "fillBuffer"
        assert classify("xla_fusion") == "xla"

    def test_analyzer_config_rccl_and_markdown_fallback(self, tmp_path):
        events = _full_pftrace_events()
        analyser = PftraceHipActivityAnalyzer(
            events,
            merge_kernels=False,
            min_event_ns=1000,
            kernel_summary_include_rccl=True,
            kernel_summary_baseline="total",
            kernel_summary_group="config",
            hip_summary_group="name+op",
        )
        assert not analyser.get_df_kernel_summary().empty
        assert not analyser.get_df_hip_summary().empty

        df = pd.DataFrame({"a": [1]})
        md = tmp_path / "m.md"

        class _NoMarkdownDF(pd.DataFrame):
            @property
            def _constructor(self):
                return _NoMarkdownDF

            def to_markdown(self, *args, **kwargs):
                raise AttributeError("no tabulate")

        _write_markdown_report(
            md,
            df_category=_NoMarkdownDF(df),
            xla_top=[("k", 1_000_000, 1, 1.0)],
            used_fav3=True,
            agents=["gpu_0"],
            kernel_df=_NoMarkdownDF(df),
            hip_df=_NoMarkdownDF(df),
        )
        assert md.read_text()

    def test_pftrace_activity_default_xlsx_and_gz_stem(self, tmp_path):
        events = _full_pftrace_events()
        gz = tmp_path / "trace.json.gz"
        import gzip

        with gzip.open(gz, "wt", encoding="utf-8") as f:
            json.dump({"traceEvents": events}, f)
        out_xlsx = tmp_path / "custom.xlsx"
        generate_perf_report_pftrace_hip_activity(
            trace_path=str(gz),
            output_xlsx_path=str(out_xlsx),
            kernel_summary=True,
            hip_summary=True,
        )
        assert out_xlsx.exists()

        pf = tmp_path / "t.pftrace"
        pf.write_bytes(b"x")
        with patch(
            "TraceLens.Reporting.generate_perf_report_pftrace_hip_activity.ensure_trace_json",
            return_value=str(tmp_path / "converted.json"),
        ):
            (tmp_path / "converted.json").write_text(
                json.dumps({"traceEvents": events})
            )
            generate_perf_report_pftrace_hip_activity(trace_path=str(pf))

    def test_pftrace_memory_copy_main(self, tmp_path):
        mod = importlib.import_module(
            "TraceLens.Reporting.generate_perf_report_pftrace_memory_copy"
        )
        events = [
            _mk_event("gpu_memcpy", "MemcpyHtoD", 1000, 20, 0, 1, {"bytes": 4096}),
            _mk_event("gpu_memcpy", "MemcpyDtoH", 1100, 15, 0, 1, {"bytes": 2048}),
        ]
        trace_path = tmp_path / "pf.json"
        trace_path.write_text(json.dumps({"traceEvents": events}))
        old_argv = sys.argv
        sys.argv = [
            "generate_perf_report_pftrace_memory_copy",
            "--trace_path",
            str(trace_path),
            "--output_csvs_dir",
            str(tmp_path / "csv"),
        ]
        try:
            mod.main()
        finally:
            sys.argv = old_argv
        assert (tmp_path / "csv").is_dir()


class TestMoePseudoOpsFullPhase10:
    def test_aiter_all_warning_paths(self):
        create_pseudo_ops_moe_fused_aiter(
            _build_tree([_mk_event("cpu_op", "other", 0, 1, 1, 1)])
        )

        events = []
        moe = _mk_event(
            "cpu_op",
            "vllm::rocm_aiter_fused_moe",
            100,
            200,
            1,
            1,
            {"Sequence number": 1},
        )
        child = _mk_event("cpu_op", "nested_cpu", 110, 10, 1, 1)
        events.extend([moe, child])
        _add_gpu_chain(events, moe, 10, "aiter::fmoe_kernel", 110, 150)
        tree = _build_tree(events)
        moe_evt = next(
            e for e in tree.events if e["name"] == "vllm::rocm_aiter_fused_moe"
        )
        child_evt = next(e for e in tree.events if e["name"] == "nested_cpu")
        moe_evt.setdefault("children", []).append(child_evt["UID"])
        child_evt["parent"] = moe_evt["UID"]
        assert _has_cpu_op_descendant(tree, moe_evt)
        _create_pseudo_op_moe_fused_aiter(tree, moe_evt)

        no_gpu = _mk_event("cpu_op", "vllm::rocm_aiter_fused_moe", 0, 1, 1, 1, {})
        tree_no_gpu = _build_tree([no_gpu])
        no_gpu_evt = next(
            e for e in tree_no_gpu.events if e["name"] == "vllm::rocm_aiter_fused_moe"
        )
        _create_pseudo_op_moe_fused_aiter(tree_no_gpu, no_gpu_evt)

        bad_kernels = _mk_event(
            "cpu_op", "vllm::rocm_aiter_fused_moe", 100, 200, 1, 1, {}
        )
        evs = [bad_kernels]
        _add_gpu_chain(evs, bad_kernels, 11, "aiter::MoeSorting", 110, 150)
        _add_gpu_chain(evs, bad_kernels, 12, "aiter::quant_fmoe", 160, 190)
        tree2 = _build_tree(evs)
        moe2 = next(
            e for e in tree2.events if e["name"] == "vllm::rocm_aiter_fused_moe"
        )
        _create_pseudo_op_moe_fused_aiter(tree2, moe2)

    def test_gptq_all_warning_paths(self):
        create_pseudo_ops_moe_gptq_awq(_build_tree([]))
        _create_pseudo_op_moe_gptq_awq(_build_tree([]), {"name": "wrong", "UID": 0})

        no_gpu = _mk_event(
            "cpu_op", "vllm::outplace_fused_experts", 0, 1, 1, 1, {"UID": 1}
        )
        tree_no_gpu = _build_tree([no_gpu])
        no_gpu_evt = next(
            e for e in tree_no_gpu.events if e["name"] == "vllm::outplace_fused_experts"
        )
        _create_pseudo_op_moe_gptq_awq(tree_no_gpu, no_gpu_evt)

        moe = _mk_event(
            "cpu_op",
            "vllm::outplace_fused_experts",
            100,
            200,
            1,
            1,
            {
                "Input Dims": [
                    [128, 4096],
                    [8, 4096, 512],
                    [8, 4096, 512],
                    [128, 6],
                    [128, 6],
                ],
            },
        )
        evs = [moe]
        _add_gpu_chain(evs, moe, 20, "other_kernel", 110, 150)
        tree = _build_tree(evs)
        moe_evt = next(
            e for e in tree.events if e["name"] == "vllm::outplace_fused_experts"
        )
        _create_pseudo_op_moe_gptq_awq(tree, moe_evt)

        moe2 = _mk_event(
            "cpu_op",
            "vllm::outplace_fused_experts",
            100,
            200,
            1,
            1,
            {
                "Input Dims": [
                    [128, 4096],
                    [8, 4096, 512],
                    [8, 4096, 512],
                    [128, 6],
                    [128, 6],
                ],
            },
        )
        evs2 = [moe2]
        _add_gpu_chain(evs2, moe2, 21, "fused_moe_kernel_gptq_awq_up", 110, 150)
        tree3 = _build_tree(evs2)
        moe3 = next(
            e for e in tree3.events if e["name"] == "vllm::outplace_fused_experts"
        )
        _create_pseudo_op_moe_gptq_awq(tree3, moe3)

    def test_flydsl_skip_paths(self):
        create_pseudo_ops_moe_flydsl(_build_tree([]))
        create_pseudo_ops_moe_flydsl(
            _build_tree(
                [
                    _mk_event("cpu_op", "not_moe", 0, 1, 1, 1),
                    _mk_event(
                        "python_function", "flydsl.py: flydsl_moe_stage1", 10, 5, 1, 1
                    ),
                ],
                add_python_func=True,
            )
        )
        create_pseudo_ops_moe_flydsl(
            _build_tree(
                [
                    _mk_event("cpu_op", FUSED_MOE_PARENT, 0, 500, 1, 1),
                    _mk_event("cpu_op", "flydsl_moe_stage1", 50, 100, 1, 1),
                ]
            )
        )


class TestArchTracediffTreePerfPhase10:
    def test_arch_utils_missing_ext_dir(self, tmp_path, monkeypatch):
        pkg_root = tmp_path / "pkg_no_arch"
        pkg_root.mkdir()
        init_py = pkg_root / "__init__.py"
        init_py.write_text("")
        pkg = types.ModuleType("pkg_no_arch_ext")
        pkg.__file__ = str(init_py)
        monkeypatch.setitem(sys.modules, "pkg_no_arch_ext", pkg)
        monkeypatch.setenv("TL_EXTENSION", "pkg_no_arch_ext")
        assert isinstance(arch_utils._collect_arch_jsons(), dict)

    def test_tracediff_summary_trace2_only_ops(self):
        diff = pd.DataFrame(
            {
                "source": ["trace2", "trace2"],
                "lowest_common_ancestor_id": [1, 1],
                "lowest_common_ancestor_name": ["aten::mm", "aten::mm"],
                "cpu_op_name": ["aten::add", "aten::mul"],
                "busy_time": [10.0, 20.0],
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
        summary = tracediff_perf_summary_from_diff_stats(diff)
        assert " | " in summary.iloc[0]["name"]

    def test_unified_perf_bwd_linked(self):
        corr_fwd, corr_bwd = 200, 201
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
                args={"correlation": corr_fwd},
            ),
            _make_gpu_event(
                "k_f",
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
                "cpu_b",
                2000,
                100,
                "cpu_op",
                "aten::mm_backward",
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
                args={"correlation": corr_bwd},
            ),
            _make_gpu_event(
                "k_b",
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
        df = analyzer.build_df_unified_perf_table(
            events=[fwd, bwd],
            include_perf_metrics=True,
            include_nccl=False,
        )
        assert isinstance(df, pd.DataFrame)


class TestPytorchReportOverlapPhase10:
    def test_pytorch_report_with_overlap_sheets(self, tmp_path):
        from TraceLens.Reporting.generate_perf_report_pytorch import (
            generate_perf_report_pytorch,
        )

        trace = _write_trace(
            tmp_path,
            [
                ("aten::convolution", "conv_kernel", 80),
                ("aten::convolution_backward", "conv_bwd_kernel", 70),
                ("aten::mm", "gemm_kernel", 100),
            ],
            "conv.json",
        )
        out = tmp_path / "py_out"
        dfs = generate_perf_report_pytorch(
            profile_json_path=str(trace),
            output_csvs_dir=str(out),
            include_overlap_info=True,
            short_kernel_study=True,
            kernel_summary=True,
            collective_analysis=False,
        )
        assert isinstance(dfs, dict)
        assert (out / "gpu_timeline.csv").exists()


class TestSplitAnnotationDummyPhase10:
    def test_main_dummy_store_single_iteration(self, tmp_path):
        from TraceLens.TraceUtils import split_inference_trace_annotation as split

        dummy_name = "vllm/v1/worker/gpu_model_runner.py(99): _dummy_run"
        trace = {"traceEvents": [], "schemaVersion": 1}
        for i in range(5):
            trace["traceEvents"].append(
                {
                    "name": dummy_name,
                    "cat": "user_annotation",
                    "ph": "X",
                    "ts": 1000 + i * 1000,
                    "dur": 100,
                    "tid": 10,
                    "pid": 1,
                    "args": {},
                }
            )
        trace_path = tmp_path / "trace.json"
        trace_path.write_text(json.dumps(trace))
        out_dir = tmp_path / "out"
        old_argv = sys.argv
        sys.argv = [
            "split_inference_trace_annotation",
            str(trace_path),
            "--output-dir",
            str(out_dir),
            "--dummy",
            "1:3",
            "--store-single-iteration",
        ]
        try:
            split.main()
        finally:
            sys.argv = old_argv
        assert (out_dir / "execution_details.json").exists()
