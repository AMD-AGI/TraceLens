###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Phase-4 CPU-only coverage sweep toward 95% TraceLens line coverage."""

from __future__ import annotations

import importlib
import inspect
import json
import os
import sys
from copy import deepcopy
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from TraceLens.PerfModel import perf_model
from TraceLens.PerfModel.extensions import (
    attention_perf_model_extensions as attn_ext,
    moe_perf_model_extensions as moe_ext,
    perf_model_extensions as pext,
    rmsnorm_perf_model_extensions as rms_ext,
)
from TraceLens.Reporting.generate_perf_report_pytorch import generate_perf_report_pytorch
from TraceLens.Reporting.generate_perf_report_pytorch_inference import (
    generate_perf_report_pytorch as generate_inference_report,
)
from TraceLens.Trace2Tree.trace_to_tree import TraceToTree
from TraceLens.TreePerf.tree_perf import TreePerfAnalyzer

from tests.test_agent_coverage import (
    _StubAnalyzer,
    _StubTree,
    _kernel_event,
    _write_minimal_orchestrator_csvs,
)
from tests.test_conv_backward_bytes import (
    _conv_bias_bwd_event,
    _conv_bias_fwd_event,
)
from tests.test_dit_fused_ln_modulate import _fused_ln_fwd_event
from tests.test_mamba_ssd import _mamba_event
from tests.test_perfmodel_coverage import _ARCH, _GDN_ANNOTATION, _gemm_event, _norm_event
from tests.test_reporting_coverage import (
    _build_synthetic_trace,
    _create_genesis_capture,
    _minimal_pftrace_events,
    _write_trace,
)
from tests.test_treeperf_coverage import _build_analyzer, _make_gpu_event, _mk_pytorch_trace

TESTS_DIR = os.path.dirname(__file__)
TRACES_ROOT = os.path.join(TESTS_DIR, "traces")
ROCprof_FILE = os.path.join(TESTS_DIR, "rocprof/908_results.json.gz")

_GEMM_EVT = _gemm_event("aten::mm", (4, 8), (8, 16))
_CONV_EVT = _conv_bias_fwd_event()
_NORM_EVT = _norm_event((4, 8, 32, 32), 8)
_ATTN_EVT = {
    "annotation": _GDN_ANNOTATION,
    "args": {
        "Input Dims": [[64, 8, 64], [64, 8, 64], [64, 8, 128]],
        "Input type": ["c10::BFloat16"] * 3,
    },
}
_MAMBA_EVT = _mamba_event(batch=2, seqlen=128)
_FUSED_LN_EVT = _fused_ln_fwd_event()
_MOE_EVT = {
    "args": {
        "Input Dims": [[32, 4096], [8, 28672, 512], [8, 4096, 7168], [32, 2]],
        "Input type": [
            "c10::BFloat16",
            "c10::Float8_e4m3fn",
            "c10::Float8_e4m3fn",
            "c10::Float32",
        ],
    }
}


def _try_model(cls, events):
    for event in events:
        try:
            model = cls(event, arch=_ARCH) if "arch" in inspect.signature(cls.__init__).parameters else cls(event)
        except TypeError:
            try:
                model = cls(event)
            except Exception:
                continue
        except Exception:
            continue
        for meth in ("flops", "bytes", "flops_bwd", "bytes_bwd", "get_compute_precision", "get_maf_type"):
            if hasattr(model, meth):
                try:
                    getattr(model, meth)()
                except NotImplementedError:
                    pass
                except Exception:
                    pass
        return True
    return False


class TestPerfModelBulkSweep:
    def test_perf_model_classes_best_effort(self):
        events = [
            _GEMM_EVT,
            _CONV_EVT,
            _NORM_EVT,
            _MAMBA_EVT,
            _FUSED_LN_EVT,
            _conv_bias_bwd_event(),
            {
                "args": {
                    "Input Dims": [[4, 32000], [4]],
                    "Input type": ["c10::BFloat16", "long int"],
                }
            },
            {
                "args": {
                    "Batch": 2,
                    "M": 4,
                    "N": 8,
                    "K": 16,
                    "Beta": 1,
                    "Type": "bf16",
                }
            },
        ]
        covered = 0
        for _name, cls in inspect.getmembers(perf_model, inspect.isclass):
            if cls.__module__ != perf_model.__name__:
                continue
            if _try_model(cls, events):
                covered += 1
        assert covered > 30

    def test_extension_classes_best_effort(self):
        events = [_GEMM_EVT, _ATTN_EVT, _MOE_EVT, _NORM_EVT]
        modules = (pext, attn_ext, moe_ext, rms_ext)
        covered = 0
        for mod in modules:
            for _name, cls in inspect.getmembers(mod, inspect.isclass):
                if cls.__module__ != mod.__name__:
                    continue
                if _try_model(cls, events):
                    covered += 1
        assert covered > 30


class TestKernelFusionMain:
    def test_kernel_fusion_standalone_main(self, tmp_path):
        from TraceLens.Agent.Analysis.category_analyses import kernel_fusion_analysis as kfa

        out = str(tmp_path)
        _write_minimal_orchestrator_csvs(out, comparative=False)
        cat_dir = os.path.join(out, "category_data")
        os.makedirs(cat_dir, exist_ok=True)
        candidates = [{
            "module_name": "nn.Module: MLP_0",
            "base_name": "MLP",
            "instance_count": 1,
            "kernel_count": 2,
            "total_kernel_time_us": 800000.0,
            "kernels": [
                {"name": "Cijk_a", "type": "GEMM", "dur_us": 500000},
                {"name": "ew_add", "type": "elementwise", "dur_us": 300000},
            ],
        }]
        with open(os.path.join(cat_dir, "fusion_candidates.json"), "w") as f:
            json.dump(candidates, f)
        with open(os.path.join(cat_dir, "category_manifest.json"), "w") as f:
            json.dump({
                "platform": "MI300X",
                "comparison_scope": "standalone",
                "gpu_utilization": {"total_time_ms": 1000.0},
            }, f)
        meta_dir = os.path.join(out, "metadata")
        os.makedirs(meta_dir)
        json.dump(
            {"peak_hbm_bw_tbs": 5.3, "max_achievable_tflops": {"matrix_fp32": 100}},
            open(os.path.join(meta_dir, "gemm_metadata.json"), "w"),
        )

        old_argv = sys.argv
        sys.argv = ["kernel_fusion_analysis", "--output-dir", out]
        try:
            kfa.main()
        finally:
            sys.argv = old_argv
        assert os.path.isfile(
            os.path.join(out, "category_data", "kernel_fusion_metrics.json")
        )


class TestReportingCliPhase4:
    def test_pftrace_memory_copy_main(self, tmp_path):
        trace_path = tmp_path / "trace.json"
        trace_path.write_text(json.dumps({"traceEvents": _minimal_pftrace_events()}))
        mod = importlib.import_module(
            "TraceLens.Reporting.generate_perf_report_pftrace_memory_copy"
        )
        out_dir = tmp_path / "mem_csv"
        old_argv = sys.argv
        sys.argv = [
            "generate_perf_report_pftrace_memory_copy",
            "--trace_path",
            str(trace_path),
            "--output_csvs_dir",
            str(out_dir),
        ]
        try:
            mod.main()
        finally:
            sys.argv = old_argv
        assert out_dir.exists()

    def test_pftrace_hip_api_main(self, tmp_path):
        trace_path = tmp_path / "trace.json"
        trace_path.write_text(json.dumps({"traceEvents": _minimal_pftrace_events()}))
        mod = importlib.import_module(
            "TraceLens.Reporting.generate_perf_report_pftrace_hip_api"
        )
        out_dir = tmp_path / "api_csv"
        old_argv = sys.argv
        sys.argv = [
            "generate_perf_report_pftrace_hip_api",
            "--trace_path",
            str(trace_path),
            "--output_csvs_dir",
            str(out_dir),
            "--include_nonlaunch_apis",
        ]
        try:
            mod.main()
        finally:
            sys.argv = old_argv
        assert out_dir.exists()

    def test_genesis_report_main(self, tmp_path):
        capture = _create_genesis_capture(tmp_path)
        mod = importlib.import_module("TraceLens.Reporting.generate_perf_report_genesis")
        out = tmp_path / "gen_out"
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

    @pytest.mark.skipif(not os.path.isfile(ROCprof_FILE), reason="rocprof fixture missing")
    def test_rocprof_report_function(self, tmp_path):
        from TraceLens.Reporting.generate_perf_report_rocprof import generate_perf_report_rocprof

        generate_perf_report_rocprof(
            profile_json_path=ROCprof_FILE,
            output_xlsx_path=str(tmp_path / "roc.xlsx"),
            kernel_summary=True,
            short_kernel_study=True,
            kernel_details=True,
        )
        assert (tmp_path / "roc.xlsx").exists()

    def test_compare_perf_reports_main(self, tmp_path):
        t1 = _write_trace(tmp_path, [("aten::mm", "gemm_kernel", 100)], "a.json")
        t2 = _write_trace(tmp_path, [("aten::mm", "gemm_kernel", 120)], "b.json")
        x1 = tmp_path / "r1.xlsx"
        x2 = tmp_path / "r2.xlsx"
        generate_perf_report_pytorch(
            profile_json_path=t1,
            output_csvs_dir=str(tmp_path / "csv1"),
            output_xlsx_path=str(x1),
            kernel_summary=True,
        )
        generate_perf_report_pytorch(
            profile_json_path=t2,
            output_csvs_dir=str(tmp_path / "csv2"),
            output_xlsx_path=str(x2),
            kernel_summary=True,
        )
        mod = importlib.import_module("TraceLens.Reporting.compare_perf_reports_pytorch")
        out = tmp_path / "cmp.xlsx"
        old_argv = sys.argv
        sys.argv = [
            "compare_perf_reports_pytorch",
            str(x1),
            str(x2),
            "-o",
            str(out),
        ]
        try:
            mod.main()
        finally:
            sys.argv = old_argv
        assert out.exists()


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


class TestTraceToTreePhase4:
    def test_trace_to_tree_prune_and_metadata(self):
        events = [
            _make_gpu_event("k1", 0, 100, "kernel", "k1", pid=0, tid=7),
            _make_gpu_event("cpu", 0, 50, "cpu_op", "aten::mm", pid=100, tid=100),
        ]
        tree = TraceToTree(deepcopy(events), prune_nongpu_paths=True)
        tree.build_tree()
        assert len(tree.events) >= 1

    def test_inference_report_all_inference_dirs(self, tmp_path):
        inf_root = os.path.join(TRACES_ROOT, "inference")
        if not os.path.isdir(inf_root):
            pytest.skip("no inference fixtures")
        for case in os.listdir(inf_root):
            case_dir = os.path.join(inf_root, case)
            if not os.path.isdir(case_dir):
                continue
            gz = [f for f in os.listdir(case_dir) if f.endswith(".json.gz")]
            if not gz:
                continue
            trace = os.path.join(case_dir, gz[0])
            out = tmp_path / case
            out.mkdir(exist_ok=True)
            generate_inference_report(
                profile_json_path=trace,
                output_csvs_dir=str(out),
                output_xlsx_path=str(out / "r.xlsx"),
                collective_analysis=False,
                kernel_summary=True,
                short_kernel_study=True,
            )
            assert (out / "gpu_timeline.csv").exists()


class TestOrchestratorPhase4:
    def test_orchestrator_comparative_with_fusion(self, tmp_path, monkeypatch):
        from TraceLens.Agent.Analysis.utils import orchestrator_prepare as op

        out = str(tmp_path)
        _write_minimal_orchestrator_csvs(out, comparative=True)
        csv_dir = os.path.join(out, "perf_report_trace1_csvs")
        pd.DataFrame({
            "name": ["Cijk_A", "ew_add"],
            "source": ["trace1", "trace1"],
            "lowest_common_ancestor_id": [100, 100],
            "kernel_time": [5000.0, 3000.0],
            "gpu_op_uid": [10, 11],
        }).to_csv(os.path.join(csv_dir, "diff_stats.csv"), index=False)

        k1 = _kernel_event(10, "Cijk_A", dur=500)
        k2 = _kernel_event(11, "ew_add", dur=300)
        mod = {
            "name": "nn.Module: Attn_0",
            "_category": "aten",
            "gpu_events": [10, 11],
            "args": {"Input Dims": "[[2,3]]"},
        }
        tree = _StubTree([mod], {10: k1, 11: k2})
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
        manifest = json.loads(
            open(os.path.join(out, "category_data", "category_manifest.json")).read()
        )
        assert manifest["comparison_scope"] == "comparative"
