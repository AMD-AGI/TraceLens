###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import warnings

import pytest

from TraceLens import trace_check
from TraceLens.trace_check import Status
from TraceLens.Trace2Tree.trace_to_tree import TraceToTree


def _kernel(ts, dur=10, corr=None, pid=0, tid=1):
    e = {
        "cat": "kernel",
        "name": "gemm",
        "ph": "X",
        "ts": ts,
        "dur": dur,
        "pid": pid,
        "tid": tid,
    }
    if corr is not None:
        e["args"] = {"correlation": corr}
    return e


def _cpu_op(ts, dur=5):
    return {"cat": "cpu_op", "name": "aten::mm", "ph": "X", "ts": ts, "dur": dur}


def _python_func(ts, dur=5):
    return {
        "cat": "python_function",
        "name": "forward",
        "ph": "X",
        "ts": ts,
        "dur": dur,
    }


def _graph_launch(ts, dur=5):
    return {
        "cat": "cuda_runtime",
        "name": "hipGraphLaunch",
        "ph": "X",
        "ts": ts,
        "dur": dur,
    }


def _runtime(ts, corr, pid=0, tid=0, dur=5):
    return {
        "cat": "cuda_runtime",
        "name": "cudaLaunchKernel",
        "ph": "X",
        "ts": ts,
        "dur": dur,
        "pid": pid,
        "tid": tid,
        "args": {"correlation": corr},
    }


def _ac2g(corr, ph, pid, tid):
    return {"cat": "ac2g", "id": corr, "ph": ph, "pid": pid, "tid": tid}


class TestKernelPresence:
    def test_empty_events_reports_error(self):
        report = trace_check.run_trace_checks([], framework="pytorch")
        assert any(
            "No GPU kernel events" in f.message and f.status == Status.ERROR
            for f in report.findings
        )

    def test_has_kernels_passes(self):
        events = [_kernel(i * 100) for i in range(5)]
        report = trace_check.run_trace_checks(events, framework="pytorch")
        assert not any("No GPU kernel events" in f.message for f in report.findings)

    def test_skipped_for_jax(self):
        report = trace_check.run_trace_checks([], framework="jax")
        assert not any("No GPU kernel events" in f.message for f in report.findings)


class TestTraceFileSize:
    def test_normal_size_passes(self, tmp_path):
        p = tmp_path / "trace.json"
        p.write_text('{"traceEvents": []}' + " " * 200_000)
        report = trace_check.run_trace_checks([], filepath=str(p))
        assert not any("Trace file is" in f.message for f in report.findings)

    def test_small_file_warns(self, tmp_path):
        p = tmp_path / "trace.json"
        p.write_text('{"traceEvents": []}')
        report = trace_check.run_trace_checks([], filepath=str(p))
        assert any("Trace file is" in f.message for f in report.findings)

    def test_missing_file_no_finding(self, tmp_path):
        report = trace_check.run_trace_checks([], filepath=str(tmp_path / "nope.json"))
        assert not any("Trace file is" in f.message for f in report.findings)


class TestCpuOpShapes:
    def test_shapes_present_passes(self):
        events = [_kernel(100)] + [
            {
                "cat": "cpu_op",
                "name": "aten::mm",
                "ph": "X",
                "ts": i * 100,
                "dur": 5,
                "args": {"Input Dims": [[32, 64]]},
            }
            for i in range(20)
        ]
        report = trace_check.run_trace_checks(events, framework="pytorch")
        assert not any("input shapes" in f.message for f in report.findings)

    def test_no_shapes_warns(self):
        events = [_kernel(100)] + [_cpu_op(i * 100) for i in range(20)]
        report = trace_check.run_trace_checks(events, framework="pytorch")
        assert any(
            "input shapes" in f.message and f.status == Status.WARN
            for f in report.findings
        )

    def test_no_cpu_ops_no_finding(self):
        events = [_kernel(i * 100) for i in range(20)]
        report = trace_check.run_trace_checks(events, framework="pytorch")
        assert not any("input shapes" in f.message for f in report.findings)

    def test_skipped_for_jax(self):
        events = [_kernel(100)] + [_cpu_op(i * 100) for i in range(20)]
        report = trace_check.run_trace_checks(events, framework="jax")
        assert not any("input shapes" in f.message for f in report.findings)


def _build_tree(events):
    tree = TraceToTree(events)
    tree.build_tree(add_python_func=False)
    return tree


class TestKernelDropDetection:
    def test_all_launches_matched_no_drops(self):
        """Every runtime launch has a matching GPU kernel via correlation ID."""
        events = [
            _cpu_op(0, dur=1000),
            _runtime(100, corr=1),
            _ac2g(1, "s", pid=0, tid=0),
            _ac2g(1, "f", pid=0, tid=1),
            _kernel(110, corr=1),
            _runtime(200, corr=2),
            _ac2g(2, "s", pid=0, tid=0),
            _ac2g(2, "f", pid=0, tid=1),
            _kernel(210, corr=2),
        ]
        report = trace_check.run_trace_checks(
            pre_report=False, tree=_build_tree(events)
        )
        assert not any(
            "no corresponding GPU kernel" in f.message for f in report.findings
        )

    def test_missing_kernel_triggers_warning(self):
        """A runtime launch with no matching GPU kernel is a dropped kernel."""
        events = [
            _cpu_op(0, dur=1000),
            _runtime(100, corr=1),
            _ac2g(1, "s", pid=0, tid=0),
            _ac2g(1, "f", pid=0, tid=1),
            _kernel(110, corr=1),
            _runtime(200, corr=2),
            _ac2g(2, "s", pid=0, tid=0),
            _ac2g(2, "f", pid=0, tid=1),
            # no kernel with corr=2 — dropped
        ]
        report = trace_check.run_trace_checks(
            pre_report=False, tree=_build_tree(events)
        )
        assert any(
            "no corresponding GPU kernel" in f.message and f.status == Status.WARN
            for f in report.findings
        )

    def test_no_runtime_events_no_finding(self):
        """Traces without runtime events (e.g. rocprof) skip the check."""
        events = [_kernel(i * 100) for i in range(20)]
        report = trace_check.run_trace_checks(
            pre_report=False, tree=_build_tree(events)
        )
        assert not any(
            "no corresponding GPU kernel" in f.message for f in report.findings
        )

    def test_message_includes_count(self):
        """The finding message should report how many launches had no kernel."""
        events = [
            _cpu_op(0, dur=1000),
            _runtime(100, corr=1),
            _runtime(200, corr=2),
            _runtime(300, corr=3),
        ]
        report = trace_check.run_trace_checks(
            pre_report=False, tree=_build_tree(events)
        )
        finding = next(
            f for f in report.findings if "no corresponding GPU kernel" in f.message
        )
        assert "3" in finding.message


class TestCallStackMissing:
    def test_python_function_events_pass(self):
        events = [_kernel(i * 100) for i in range(20)] + [_python_func(50)]
        report = trace_check.run_trace_checks(events)
        assert not any("call stack data" in f.message for f in report.findings)

    def test_no_python_func_warns(self):
        events = [_kernel(i * 100) for i in range(20)]
        report = trace_check.run_trace_checks(events)
        assert any(
            "call stack data" in f.message and f.status == Status.WARN
            for f in report.findings
        )


class TestGraphMode:
    def test_eager_mode_no_finding(self):
        events = [_kernel(i * 100) for i in range(20)]
        report = trace_check.run_trace_checks(events)
        assert not any("graph launch" in f.message for f in report.findings)

    def test_graph_launch_without_capture_warns(self):
        events = [_kernel(i * 100) for i in range(20)] + [_graph_launch(500)]
        report = trace_check.run_trace_checks(events, capture_trace_filepath=None)
        assert any(
            "graph launch" in f.message and f.status == Status.WARN
            for f in report.findings
        )

    def test_graph_launch_with_capture_ok(self):
        events = [_kernel(i * 100) for i in range(20)] + [_graph_launch(500)]
        report = trace_check.run_trace_checks(
            events, capture_trace_filepath="/some/path"
        )
        assert not any("graph launch" in f.message for f in report.findings)


class TestBypass:
    def test_env_var_skips_quality_check(self, monkeypatch, tmp_path):
        """TRACELENS_SKIP_QUALITY_CHECK=1 prevents run_trace_checks from being called."""
        import json

        from TraceLens.util import DataLoader

        trace_file = tmp_path / "trace.json"
        trace_file.write_text(json.dumps({"traceEvents": []}))
        monkeypatch.setenv("TRACELENS_SKIP_QUALITY_CHECK", "1")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            DataLoader.load_trace_events(str(trace_file))
            health_warnings = [x for x in w if "trace_check" in str(x.message)]
            assert len(health_warnings) == 0

    def test_env_var_unset_runs_checks(self, monkeypatch, tmp_path):
        import json

        from TraceLens.util import DataLoader

        trace_file = tmp_path / "trace.json"
        trace_file.write_text(json.dumps({"traceEvents": []}))
        monkeypatch.delenv("TRACELENS_SKIP_QUALITY_CHECK", raising=False)
        with pytest.raises(ValueError, match="No GPU kernel events"):
            DataLoader.load_trace_events(str(trace_file), framework="pytorch")


class TestLogFindings:
    def test_warn_emits_user_warning(self):
        report = trace_check.TraceCheckReport(
            findings=[trace_check.TraceCheck(Status.WARN, "test message")]
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            report.log_findings()
            assert len(w) == 1
            assert "test message" in str(w[0].message)
            assert w[0].category is UserWarning

    def test_error_raises_value_error(self):
        report = trace_check.TraceCheckReport(
            findings=[trace_check.TraceCheck(Status.ERROR, "error message")]
        )
        with pytest.raises(ValueError, match="error message"):
            report.log_findings()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class TestHelpers:
    def test_coefficient_of_variation_empty(self):
        assert trace_check.coefficient_of_variation([]) == 0.0

    def test_coefficient_of_variation_uniform(self):
        assert trace_check.coefficient_of_variation([5, 5, 5]) == 0.0

    def test_coefficient_of_variation_varied(self):
        cv = trace_check.coefficient_of_variation([1, 2, 3, 4, 5])
        assert 0.4 < cv < 0.6

    def test_busy_idle_from_events_empty(self):
        m = trace_check.busy_idle_from_events([])
        assert m["busy_time"] == 0.0
        assert m["idle_pct"] == 0.0

    def test_busy_idle_from_events_no_idle(self):
        events = [{"ts": 0, "dur": 100}, {"ts": 100, "dur": 100}]
        m = trace_check.busy_idle_from_events(events)
        assert m["idle_pct"] == 0.0

    def test_busy_idle_from_events_with_gap(self):
        events = [{"ts": 0, "dur": 10}, {"ts": 100, "dur": 10}]
        m = trace_check.busy_idle_from_events(events)
        assert m["idle_pct"] > 0


# ---------------------------------------------------------------------------
# Detailed pre-report checks
# ---------------------------------------------------------------------------
class TestDetailedPreReport:
    def test_detailed_checks_skipped_by_default(self):
        events = [_kernel(i * 100) for i in range(50)]
        report = trace_check.run_trace_checks(events, framework="pytorch")
        assert not any("CV>=" in f.message for f in report.findings)
        assert not any("GPU idle" in f.message for f in report.findings)

    def test_detailed_checks_run_when_enabled(self, monkeypatch):
        monkeypatch.setenv("TRACELENS_DETAILED_QUALITY_CHECKS", "1")
        events = [_kernel(0, dur=10), _kernel(1000000, dur=10)]
        report = trace_check.run_trace_checks(events, framework="pytorch")
        # With only 2 kernels and a huge gap, gpu_busy_idle should fire
        assert any("GPU idle" in f.message for f in report.findings)

    def test_cpu_op_shapes_detailed_low_coverage(self, monkeypatch):
        monkeypatch.setenv("TRACELENS_DETAILED_QUALITY_CHECKS", "1")
        events = [_kernel(100)] + [_cpu_op(i * 100) for i in range(50)]
        report = trace_check.run_trace_checks(events, framework="pytorch")
        assert any("cpu_ops have shapes" in f.message for f in report.findings)

    def test_cpu_op_shapes_detailed_high_coverage(self, monkeypatch):
        monkeypatch.setenv("TRACELENS_DETAILED_QUALITY_CHECKS", "1")
        events = [_kernel(100)] + [
            {
                "cat": "cpu_op",
                "name": "aten::mm",
                "ph": "X",
                "ts": i * 100,
                "dur": 5,
                "args": {"Input Dims": [[32, 64]]},
            }
            for i in range(50)
        ]
        report = trace_check.run_trace_checks(events, framework="pytorch")
        assert not any("cpu_ops have shapes" in f.message for f in report.findings)

    def test_cpu_gpu_ratio_normal(self, monkeypatch):
        monkeypatch.setenv("TRACELENS_DETAILED_QUALITY_CHECKS", "1")
        events = [_kernel(i * 100) for i in range(20)] + [
            _cpu_op(i * 100) for i in range(20)
        ]
        report = trace_check.run_trace_checks(events, framework="pytorch")
        assert not any("CPU:GPU" in f.message for f in report.findings)

    def test_cpu_gpu_ratio_suspicious(self, monkeypatch):
        monkeypatch.setenv("TRACELENS_DETAILED_QUALITY_CHECKS", "1")
        events = [_kernel(100)] + [_cpu_op(i) for i in range(1000)]
        report = trace_check.run_trace_checks(events, framework="pytorch")
        assert any("CPU:GPU" in f.message for f in report.findings)

    def test_runtime_variability_stable(self, monkeypatch):
        monkeypatch.setenv("TRACELENS_DETAILED_QUALITY_CHECKS", "1")
        events = [
            {
                "cat": "kernel",
                "name": "gemm",
                "ph": "X",
                "ts": i * 100,
                "dur": 10,
                "args": {"grid": "[1,1,1]", "block": "[256,1,1]"},
            }
            for i in range(50)
        ]
        report = trace_check.run_trace_checks(events, framework="pytorch")
        assert not any("CV>=" in f.message for f in report.findings)

    def test_jax_metadata_richness_skipped_for_pytorch(self, monkeypatch):
        monkeypatch.setenv("TRACELENS_DETAILED_QUALITY_CHECKS", "1")
        events = [_kernel(i * 100) for i in range(20)]
        report = trace_check.run_trace_checks(events, framework="pytorch")
        assert not any("HLO metadata" in f.message for f in report.findings)

    def test_jax_metadata_richness_sparse(self, monkeypatch):
        monkeypatch.setenv("TRACELENS_DETAILED_QUALITY_CHECKS", "1")
        events = [
            {"cat": "kernel", "ph": "X", "ts": i, "dur": 1, "name": "k", "args": {}}
            for i in range(20)
        ]
        report = trace_check.run_trace_checks(events, framework="jax")
        assert any("HLO metadata" in f.message for f in report.findings)


# ---------------------------------------------------------------------------
# Post-report checks
# ---------------------------------------------------------------------------
class TestPostReportChecks:
    def _make_dfs(self, idle_pct=5.0, num_ops=10, shape_coverage=1.0):
        import pandas as pd

        timeline = pd.DataFrame(
            {
                "type": ["computation_time", "idle_time"],
                "percent": [100 - idle_pct, idle_pct],
            }
        )
        ops_summary = pd.DataFrame(
            {"name": [f"op_{i}" for i in range(num_ops)], "Count": [10] * num_ops}
        )
        shaped = int(num_ops * shape_coverage)
        dims = [[[32, 64]]] * shaped + [None] * (num_ops - shaped)
        ops_unique_args = pd.DataFrame(
            {
                "name": [f"op_{i}" for i in range(num_ops)],
                "Input Dims": dims,
                "total_subtree_kernel_time_sum": [100.0] * num_ops,
                "total_direct_kernel_time_sum": [90.0] * num_ops,
            }
        )
        return {
            "gpu_timeline": timeline,
            "ops_summary": ops_summary,
            "ops_unique_args": ops_unique_args,
        }

    def test_post_report_skipped_without_env(self):
        report = trace_check.run_trace_checks(pre_report=False, dfs={})
        # Only kernels_dropped_correlation runs (basic), not detailed checks
        assert not any("GPU idle" in f.message for f in report.findings)

    def test_idle_timeline_high_idle(self, monkeypatch):
        monkeypatch.setenv("TRACELENS_DETAILED_QUALITY_CHECKS", "1")
        dfs = self._make_dfs(idle_pct=70.0)
        report = trace_check.run_trace_checks(
            pre_report=False, dfs=dfs, framework="pytorch"
        )
        assert any("GPU idle" in f.message for f in report.findings)

    def test_idle_timeline_low_idle(self, monkeypatch):
        monkeypatch.setenv("TRACELENS_DETAILED_QUALITY_CHECKS", "1")
        dfs = self._make_dfs(idle_pct=5.0)
        report = trace_check.run_trace_checks(
            pre_report=False, dfs=dfs, framework="pytorch"
        )
        assert not any("GPU idle" in f.message for f in report.findings)

    def test_ops_have_shapes_good_coverage(self, monkeypatch):
        monkeypatch.setenv("TRACELENS_DETAILED_QUALITY_CHECKS", "1")
        dfs = self._make_dfs(shape_coverage=1.0)
        report = trace_check.run_trace_checks(
            pre_report=False, dfs=dfs, framework="pytorch"
        )
        assert not any("ops have shapes" in f.message for f in report.findings)

    def test_ops_have_shapes_poor_coverage(self, monkeypatch):
        monkeypatch.setenv("TRACELENS_DETAILED_QUALITY_CHECKS", "1")
        dfs = self._make_dfs(shape_coverage=0.1)
        report = trace_check.run_trace_checks(
            pre_report=False, dfs=dfs, framework="pytorch"
        )
        assert any("ops have shapes" in f.message for f in report.findings)

    def test_high_idle_ops_detected(self, monkeypatch):
        import pandas as pd

        monkeypatch.setenv("TRACELENS_DETAILED_QUALITY_CHECKS", "1")
        dfs = {
            "gpu_timeline": pd.DataFrame({"type": ["idle_time"], "percent": [5.0]}),
            "ops_summary": pd.DataFrame({"name": ["op"], "Count": [1]}),
            "ops_unique_args": pd.DataFrame(
                {
                    "name": ["op"],
                    "Input Dims": [[[1]]],
                    "total_subtree_kernel_time_sum": [100.0],
                    "total_direct_kernel_time_sum": [10.0],
                }
            ),
        }
        report = trace_check.run_trace_checks(
            pre_report=False, dfs=dfs, framework="pytorch"
        )
        assert any("non-kernel time" in f.message for f in report.findings)

    def test_attention_kernel_count_skipped_for_pytorch(self, monkeypatch):
        monkeypatch.setenv("TRACELENS_DETAILED_QUALITY_CHECKS", "1")
        dfs = self._make_dfs()
        report = trace_check.run_trace_checks(
            pre_report=False, dfs=dfs, framework="pytorch"
        )
        assert not any("Attention launches" in f.message for f in report.findings)

    def test_attention_kernel_count_mismatch(self, monkeypatch):
        import pandas as pd

        monkeypatch.setenv("TRACELENS_DETAILED_QUALITY_CHECKS", "1")
        dfs = self._make_dfs()
        dfs["gpu_timeline"] = pd.DataFrame(
            {"type": ["idle_time"], "percent": [5.0], "gpu_pid": [0]}
        )
        dfs["op_fa_fwd"] = pd.DataFrame({"name": ["fmha_fwd"], "name_count": [10]})
        cfg = trace_check.AttnCountConfig(num_layers=4, num_steps=2, num_gpus=1)
        report = trace_check.run_trace_checks(
            pre_report=False, dfs=dfs, framework="jax", attn_cfg=cfg
        )
        assert any("Attention launches" in f.message for f in report.findings)

    def test_attention_kernel_count_disabled_without_config(self, monkeypatch):
        import pandas as pd

        monkeypatch.setenv("TRACELENS_DETAILED_QUALITY_CHECKS", "1")
        dfs = self._make_dfs()
        dfs["op_fa_fwd"] = pd.DataFrame({"name": ["fmha_fwd"], "name_count": [10]})
        report = trace_check.run_trace_checks(
            pre_report=False, dfs=dfs, framework="jax"
        )
        assert not any("Attention launches" in f.message for f in report.findings)

    def test_idle_timeline_warn_threshold(self, monkeypatch):
        monkeypatch.setenv("TRACELENS_DETAILED_QUALITY_CHECKS", "1")
        dfs = self._make_dfs(idle_pct=35.0)
        report = trace_check.run_trace_checks(
            pre_report=False, dfs=dfs, framework="pytorch"
        )
        assert any("GPU idle" in f.message for f in report.findings)

    def test_ops_have_shapes_missing_column(self, monkeypatch):
        import pandas as pd

        monkeypatch.setenv("TRACELENS_DETAILED_QUALITY_CHECKS", "1")
        dfs = self._make_dfs()
        dfs["ops_unique_args"] = pd.DataFrame({"name": ["op"]})
        report = trace_check.run_trace_checks(
            pre_report=False, dfs=dfs, framework="pytorch"
        )
        assert not any("ops have shapes" in f.message for f in report.findings)

    def test_high_idle_ops_none_detected(self, monkeypatch):
        monkeypatch.setenv("TRACELENS_DETAILED_QUALITY_CHECKS", "1")
        dfs = self._make_dfs()
        report = trace_check.run_trace_checks(
            pre_report=False, dfs=dfs, framework="pytorch"
        )
        assert not any("non-kernel time" in f.message for f in report.findings)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
class TestEdgeCases:
    def test_coefficient_of_variation_zero_mean(self):
        assert trace_check.coefficient_of_variation([0, 0, 0]) == 0.0

    def test_attn_count_config_enabled(self):
        cfg = trace_check.AttnCountConfig(num_layers=4, num_steps=2)
        assert cfg.enabled
        cfg2 = trace_check.AttnCountConfig()
        assert not cfg2.enabled

    def test_runtime_variability_high_cv(self, monkeypatch):
        monkeypatch.setenv("TRACELENS_DETAILED_QUALITY_CHECKS", "1")
        events = []
        for i in range(20):
            dur = 10 if i % 2 == 0 else 1000
            events.append(
                {
                    "cat": "kernel",
                    "name": "unstable_kernel",
                    "ph": "X",
                    "ts": i * 2000,
                    "dur": dur,
                    "args": {"grid": "[1,1,1]", "block": "[256,1,1]"},
                }
            )
        report = trace_check.run_trace_checks(events, framework="pytorch")
        assert any("CV>=" in f.message for f in report.findings)

    def test_gpu_busy_idle_warn_threshold(self, monkeypatch):
        monkeypatch.setenv("TRACELENS_DETAILED_QUALITY_CHECKS", "1")
        events = [_kernel(0, dur=10), _kernel(200, dur=10)]
        report = trace_check.run_trace_checks(events, framework="pytorch")
        idle_findings = [f for f in report.findings if "GPU idle" in f.message]
        assert len(idle_findings) > 0

    def test_jax_metadata_good_coverage(self, monkeypatch):
        monkeypatch.setenv("TRACELENS_DETAILED_QUALITY_CHECKS", "1")
        events = [
            {
                "cat": "kernel",
                "ph": "X",
                "ts": i,
                "dur": 1,
                "name": "k",
                "args": {"hlo_op": "conv", "hlo_module": "m", "tf_op": "t"},
            }
            for i in range(20)
        ]
        report = trace_check.run_trace_checks(events, framework="jax")
        assert not any("HLO metadata" in f.message for f in report.findings)
