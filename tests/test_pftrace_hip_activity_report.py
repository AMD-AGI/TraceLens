###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for pftrace hip activity report (category, kernel, HIP API summaries)."""

import json, os, tempfile, pandas as pd, pytest, importlib, sys, shutil, urllib.request, gzip, types
from TraceLens.Reporting.generate_perf_report_pftrace_hip_activity import (
    _write_markdown_report,
    generate_perf_report_pftrace_hip_activity,
)
from TraceLens.Reporting.pftrace_hip_activity_analysis import (
    Event,
    PftraceHipActivityAnalyzer,
    build_event_lists,
    build_hip_api_events,
    classify,
    discover_gpus,
    extract_time_ns,
    rccl_overlap_two_pointer,
)
from unittest.mock import patch
from tests.fixtures.reporting import (
    _full_pftrace_events,
    _minimal_pftrace_events,
    _mk_event,
    _rich_pftrace_events,
    _write_trace,
)
from pathlib import Path
from TraceLens.Reporting import pftrace_utils
from tests.test_trace2tree import _mk_event
from TraceLens.Agent.Analysis.utils import arch_utils


def _minimal_trace_events_with_agent():
    """Trace with one GPU (agent), one kernel event, one hip_api event."""
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


class TestPftraceHipActivityAnalysis:
    def test_extract_time_ns_from_args(self):
        e = {"args": {"begin_ns": 1000, "delta_ns": 500}}
        ts, dur = extract_time_ns(e)
        assert ts == 1000
        assert dur == 500

    def test_extract_time_ns_from_ts_dur(self):
        e = {"ts": 1, "dur": 2}
        ts, dur = extract_time_ns(e)
        assert ts == 1000
        assert dur == 2000

    def test_discover_gpus(self):
        events = _minimal_trace_events_with_agent()
        agent_to_idx, agents = discover_gpus(events)
        assert "gpu_0" in agent_to_idx
        assert agents == ["gpu_0"]

    def test_classify(self):
        assert classify("ncclAllReduce") == "rccl"
        assert classify("Cijk_gemm") == "gemm"
        assert classify("xla_fusion_1") == "xla"

    def test_build_event_lists(self):
        events = _minimal_trace_events_with_agent()
        compute, rccl, xla_agg, used_fav3, agents = build_event_lists(
            events, merge_kernels=False, min_tid=-(10**9), max_tid=10**9
        )
        assert len(agents) == 1
        assert len(compute[0]) == 1
        assert compute[0][0].name == "xla_fusion_42"
        assert "xla_fusion_42" in xla_agg or "xla_fusion_" in str(xla_agg)

    def test_build_hip_api_events(self):
        events = _minimal_trace_events_with_agent()
        hip = build_hip_api_events(events, min_tid=-(10**9), max_tid=10**9)
        assert len(hip) == 1
        assert hip[0].name == "hipLaunchKernelGGL"


class TestPftraceHipActivityAnalyzer:
    def test_analyzer_returns_dataframes(self):
        events = _minimal_trace_events_with_agent()
        analyzer = PftraceHipActivityAnalyzer(events, min_event_ns=0)
        df_cat = analyzer.get_df_category_summary()
        assert isinstance(df_cat, pd.DataFrame)
        assert not df_cat.empty
        assert "GPU ID" in df_cat.columns
        assert "Category" in df_cat.columns
        df_xla = analyzer.get_df_xla_top(top_n=10)
        assert isinstance(df_xla, pd.DataFrame)
        df_kern = analyzer.get_df_kernel_summary()
        assert isinstance(df_kern, pd.DataFrame)
        df_hip = analyzer.get_df_hip_summary()
        assert isinstance(df_hip, pd.DataFrame)
        assert len(df_hip) == 1


class TestGeneratePerfReportPftraceHipActivity:
    def test_generate_excel(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_path = os.path.join(tmpdir, "trace.json")
            with open(trace_path, "w") as f:
                json.dump({"traceEvents": _minimal_trace_events_with_agent()}, f)
            out_xlsx = os.path.join(tmpdir, "report.xlsx")
            dfs = generate_perf_report_pftrace_hip_activity(
                trace_path=trace_path,
                output_xlsx_path=out_xlsx,
                min_event_ns=0,
            )
            assert os.path.exists(out_xlsx)
            assert "category_summary" in dfs
            assert "xla_top" in dfs
            assert "kernel_summary" in dfs
            assert "hip_summary" in dfs

    def test_generate_csv_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_path = os.path.join(tmpdir, "trace.json")
            with open(trace_path, "w") as f:
                json.dump({"traceEvents": _minimal_trace_events_with_agent()}, f)
            generate_perf_report_pftrace_hip_activity(
                trace_path=trace_path,
                output_csvs_dir=tmpdir,
                min_event_ns=0,
            )
            assert "category_summary.csv" in os.listdir(tmpdir)
            assert "hip_summary.csv" in os.listdir(tmpdir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


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


class TestPftraceAndArchPhase8:
    def test_pftrace_utils_branches(self, tmp_path):

        preferred = tmp_path / "traceconv"
        preferred.write_text("#!/bin/sh\necho ok\n")
        preferred.chmod(0o755)
        got = pftrace_utils.acquire_traceconv(preferred, tmp_path / "out")
        assert got == preferred.resolve()
        p = tmp_path / "t.json"
        p.write_text("{}")
        assert pftrace_utils.ensure_trace_json(str(p)) == str(p.resolve())

    def test_arch_utils_tl_extension(self, tmp_path, monkeypatch):

        pkg_root = tmp_path / "fake_pkg"
        ext_arch = pkg_root / "Agent" / "Analysis" / "utils" / "arch"
        ext_arch.mkdir(parents=True)
        (ext_arch / "CUSTOM.json").write_text('{"mem_bw_gbps": 100}')
        init_py = pkg_root / "__init__.py"
        init_py.write_text("")
        pkg = types.ModuleType("fake_tl_ext")
        pkg.__file__ = str(init_py)
        monkeypatch.setitem(sys.modules, "fake_tl_ext", pkg)
        monkeypatch.setenv("TL_EXTENSION", "fake_tl_ext")
        assert "CUSTOM" in arch_utils._collect_arch_jsons()


class TestPftraceExtendedPhase9:
    def test_pftrace_utils_branches(self, tmp_path, monkeypatch):
        on_path = tmp_path / "traceconv"
        on_path.write_text("#!/bin/sh\necho ok\n")
        on_path.chmod(0o755)
        with patch.object(shutil, "which", return_value=str(on_path)):
            assert pftrace_utils.acquire_traceconv(
                tmp_path / "missing", tmp_path
            ).exists()

        def fail_run(cmd, cwd=None):
            raise RuntimeError("curl failed")

        def fake_urlretrieve(_url, target):
            Path(target).write_bytes(b"#!/bin/sh\necho ok\n")
            Path(target).chmod(0o755)

        with patch.object(shutil, "which", return_value=None):
            with patch.object(pftrace_utils, "run", side_effect=fail_run):
                with patch.object(
                    urllib.request, "urlretrieve", side_effect=fake_urlretrieve
                ):
                    assert pftrace_utils.acquire_traceconv(
                        None, tmp_path / "dl"
                    ).exists()

        pf = tmp_path / "t.pftrace"
        pf.write_bytes(b"fake")
        conv = tmp_path / "tc"
        conv.write_text("#!/bin/sh\necho ok\n")
        conv.chmod(0o755)

        def mock_run(cmd, cwd=None):
            Path(cmd[-1]).write_text(
                json.dumps({"traceEvents": _minimal_pftrace_events()})
            )

        monkeypatch.setattr(pftrace_utils, "run", mock_run)
        assert pftrace_utils.ensure_trace_json(str(pf), str(conv)).endswith(".json")

    def test_pftrace_analyzer_and_report(self, tmp_path):
        assert extract_time_ns({"ts": 100, "dur": 50, "args": {}}) == (100_000, 50_000)
        assert classify("Cijk_x") == "gemm"
        compute = [
            Event(gpu=0, name="xla_k", ts_ns=0, dur_ns=100),
            Event(gpu=0, name="xla_k", ts_ns=50, dur_ns=100),
        ]
        ov, _ = rccl_overlap_two_pointer(
            compute, [Event(gpu=0, name="nccl", ts_ns=60, dur_ns=80)]
        )
        assert ov == 120

        events = _rich_pftrace_events()
        analyser = PftraceHipActivityAnalyzer(
            events,
            merge_kernels=True,
            kernel_summary_include_rccl=True,
            kernel_summary_baseline="compute",
            hip_summary_group="name+stream+op",
        )
        assert analyser.used_fav3
        assert not analyser.get_df_category_summary().empty

        trace_path = tmp_path / "pf.json"
        trace_path.write_text(json.dumps({"traceEvents": events}))
        generate_perf_report_pftrace_hip_activity(
            trace_path=str(trace_path),
            output_csvs_dir=str(tmp_path / "csv"),
            merge_kernels=True,
        )
        _write_markdown_report(
            tmp_path / "md.md",
            pd.DataFrame(),
            [],
            False,
            ["gpu_0"],
            None,
            None,
        )


def test_inference_report_main(tmp_path):
    trace = _write_trace(tmp_path, [("aten::mm", "gemm_kernel", 80)])
    out_dir = tmp_path / "inf_csvs"
    import TraceLens.Reporting.generate_perf_report_pytorch_inference as mod

    old_argv = sys.argv
    sys.argv = [
        "generate_perf_report_pytorch_inference",
        "--profile_json_path",
        trace,
        "--output_csvs_dir",
        str(out_dir),
        "--disable_coll_analysis",
        "--enable_kernel_summary",
    ]
    try:
        mod.main()
    finally:
        sys.argv = old_argv
    assert (out_dir / "gpu_timeline.csv").exists()


def test_pftrace_hip_activity_main(tmp_path):
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(json.dumps({"traceEvents": _minimal_pftrace_events()}))
    md = tmp_path / "pf.md"
    import TraceLens.Reporting.generate_perf_report_pftrace_hip_activity as mod

    old_argv = sys.argv
    sys.argv = [
        "generate_perf_report_pftrace_hip_activity",
        "--trace_path",
        str(trace_path),
        "--output_md_path",
        str(md),
        "--write_md",
    ]
    try:
        mod.main()
    finally:
        sys.argv = old_argv
    assert md.exists()
