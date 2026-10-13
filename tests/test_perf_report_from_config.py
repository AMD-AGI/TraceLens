###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Schema and dispatch tests for the YAML perf-report wrapper.

These tests never open a trace. Report functions and capture merge are fakes.
"""

import json
import os

import pytest
import yaml

from TraceLens.Reporting.generate_perf_report_from_config import (
    ConfigError,
    load_run_config,
    main,
    plan_reports,
    resolve_config,
    run_config,
)


def _touch(tmp_path, *parts):
    path = os.path.join(str(tmp_path), *parts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("{}\n")
    return path


def _mkdir(tmp_path, *parts):
    path = os.path.join(str(tmp_path), *parts)
    os.makedirs(path, exist_ok=True)
    return path


def _config(tmp_path, mode="default", jobs=None, **extra):
    cfg = {
        "analysis_mode": mode,
        "output_dir": os.path.join(str(tmp_path), "out"),
        "jobs": jobs
        or [{"trace_path": _touch(tmp_path, "rank0.json.gz"), "platform": "MI300X"}],
    }
    cfg.update(extra)
    return cfg


class _Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return kwargs


def _run(cfg, mode="default"):
    runner = _Recorder()
    merged = []

    def merge(folder, trace_path):
        merged.append((folder, trace_path))
        return f"TREE:{os.path.basename(folder)}"

    run_config(cfg, runners={mode: runner}, merge_capture=merge)
    return runner.calls, merged


def test_single_job_writes_agent_layout(tmp_path):
    plan = plan_reports(_config(tmp_path))[0]
    out = os.path.join(str(tmp_path), "out")
    assert plan.output_xlsx_path == os.path.join(out, "perf_report.xlsx")
    assert plan.output_csvs_dir == os.path.join(out, "perf_report_csvs")


def test_single_job_name_is_optional_and_used_when_set(tmp_path):
    trace = _touch(tmp_path, "t.json")
    cfg = _config(
        tmp_path, jobs=[{"name": "llama", "trace_path": trace, "platform": "MI300X"}]
    )
    plan = plan_reports(cfg)[0]
    assert os.path.basename(plan.output_xlsx_path) == "perf_report_llama.xlsx"
    assert os.path.basename(plan.output_csvs_dir) == "perf_report_llama_csvs"


def test_default_recipe_flags_follow_function_signature(tmp_path):
    kwargs = plan_reports(_config(tmp_path))[0].kwargs
    assert kwargs["enable_pseudo_ops"] is True
    assert kwargs["group_by_num_kernels"] is True
    assert kwargs["include_call_stack"] is True
    assert "group_by_parent_module" not in kwargs


def test_inference_recipe_includes_group_by_parent_module(tmp_path):
    kwargs = plan_reports(_config(tmp_path, mode="inference"))[0].kwargs
    assert kwargs["group_by_parent_module"] is True


def test_inference_matches_cli_collective_analysis_off(tmp_path):
    kwargs = plan_reports(_config(tmp_path, mode="inference"))[0].kwargs
    assert kwargs["collective_analysis"] is False
    kwargs = plan_reports(
        _config(tmp_path, mode="inference", collective_analysis=True)
    )[0].kwargs
    assert kwargs["collective_analysis"] is True


def test_default_mode_leaves_collective_analysis_to_function(tmp_path):
    assert "collective_analysis" not in plan_reports(_config(tmp_path))[0].kwargs


def test_resolved_config_is_written_and_reruns_identically(tmp_path):
    cap = _mkdir(tmp_path, "cap")
    cfg = _config(
        tmp_path,
        mode="inference",
        include_call_stack=False,
        jobs=[
            {
                "trace_path": _touch(tmp_path, "rank0.json.gz"),
                "platform": "MI300X",
                "capture_folder": cap,
            }
        ],
    )
    _run(cfg, mode="inference")
    path = os.path.join(str(tmp_path), "out", "run_config.resolved.yaml")
    resolved = load_run_config(path)
    assert resolved == resolve_config(cfg)
    assert resolved["include_call_stack"] is False
    assert resolved["collective_analysis"] is False
    assert resolved["topk_ops"] is None
    assert resolved["jobs"][0]["capture_folder"] == cap
    first = plan_reports(cfg)[0]
    again = plan_reports(resolved)[0]
    assert again.output_xlsx_path == first.output_xlsx_path
    assert {k: again.kwargs[k] for k in first.kwargs} == first.kwargs


def test_resolved_config_written_before_first_job(tmp_path):
    cfg = _config(tmp_path)
    seen = []

    def runner(**kwargs):
        seen.append(
            os.path.isfile(
                os.path.join(str(tmp_path), "out", "run_config.resolved.yaml")
            )
        )
        raise RuntimeError("bad trace")

    with pytest.raises(RuntimeError, match="bad trace"):
        run_config(cfg, runners={"default": runner})
    assert seen == [True]


def test_first_failure_stops_later_jobs(tmp_path):
    trace = _touch(tmp_path, "t.json")
    cfg = _config(
        tmp_path,
        jobs=[
            {"name": "a", "trace_path": trace, "platform": "MI300X"},
            {"name": "b", "trace_path": trace, "platform": "MI300X"},
        ],
    )
    names = []

    def runner(**kwargs):
        names.append(os.path.basename(kwargs["output_xlsx_path"]))
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        run_config(cfg, runners={"default": runner})
    assert names == ["perf_report_a.xlsx"]


def test_yaml_overrides_recipe_default(tmp_path):
    kwargs = plan_reports(_config(tmp_path, include_call_stack=False, topk_ops=50))[
        0
    ].kwargs
    assert kwargs["include_call_stack"] is False
    assert kwargs["topk_ops"] == 50


def test_unknown_top_level_key(tmp_path):
    with pytest.raises(ConfigError, match="unknown keys"):
        plan_reports(_config(tmp_path, not_a_report_flag=True))


def test_inference_only_flag_rejected_on_default(tmp_path):
    with pytest.raises(ConfigError, match="group_by_parent_module"):
        plan_reports(_config(tmp_path, group_by_parent_module=True))


@pytest.mark.parametrize(
    "key", ["comparison_json_path", "gpu_arch_json_path", "precomputed_diff_stats"]
)
def test_job_level_keys_rejected_at_top_level(tmp_path, key):
    with pytest.raises(ConfigError, match="belong on a job"):
        plan_reports(_config(tmp_path, **{key: "x"}))


@pytest.mark.parametrize("mode", ["spec_dec", "pd_disaggregation", "", None])
def test_only_two_modes(tmp_path, mode):
    with pytest.raises(ConfigError, match="analysis_mode"):
        plan_reports(_config(tmp_path, mode=mode))


def test_multiple_jobs_require_names(tmp_path):
    prefill = _touch(tmp_path, "prefill", "rank0.json.gz")
    decode = _touch(tmp_path, "decode", "rank0.json.gz")
    cfg = _config(
        tmp_path,
        mode="inference",
        jobs=[
            {"name": "prefill_rank0", "trace_path": prefill, "platform": "MI300X"},
            {"trace_path": decode, "platform": "MI300X"},
        ],
    )
    with pytest.raises(ConfigError, match=r"jobs\[1\]\.name is required"):
        plan_reports(cfg)


def test_prefill_and_decode_are_two_inference_jobs(tmp_path):
    prefill = _touch(tmp_path, "prefill", "rank0.json.gz")
    decode = _touch(tmp_path, "decode", "rank0.json.gz")
    prefill_cap = _mkdir(tmp_path, "prefill", "capture_traces")
    decode_cap = _mkdir(tmp_path, "decode", "capture_traces")
    cfg = _config(
        tmp_path,
        mode="inference",
        jobs=[
            {
                "name": "prefill_rank0",
                "trace_path": prefill,
                "platform": "MI300X",
                "capture_folder": prefill_cap,
            },
            {
                "name": "decode_rank0",
                "trace_path": decode,
                "platform": "MI300X",
                "capture_folder": decode_cap,
            },
        ],
    )
    calls, merged = _run(cfg, mode="inference")
    assert [os.path.basename(c["output_xlsx_path"]) for c in calls] == [
        "perf_report_prefill_rank0.xlsx",
        "perf_report_decode_rank0.xlsx",
    ]
    assert merged == [(prefill_cap, prefill), (decode_cap, decode)]
    assert all(c["augmented_tree"] == "TREE:capture_traces" for c in calls)


def test_duplicate_names(tmp_path):
    trace = _touch(tmp_path, "t.json")
    job = {"name": "a", "trace_path": trace, "platform": "MI300X"}
    with pytest.raises(ConfigError, match="duplicate job name"):
        plan_reports(_config(tmp_path, jobs=[job, dict(job)]))


@pytest.mark.parametrize("name", ["", "has space", "a/b", 7])
def test_bad_names(tmp_path, name):
    trace = _touch(tmp_path, "t.json")
    cfg = _config(
        tmp_path, jobs=[{"name": name, "trace_path": trace, "platform": "MI300X"}]
    )
    with pytest.raises(ConfigError, match="name"):
        plan_reports(cfg)


def test_platform_is_passed_by_name(tmp_path):
    calls, _ = _run(_config(tmp_path, kernel_summary=True))
    assert calls[0]["gpu_arch_platform"] == "MI300X"
    assert "gpu_arch_json_path" not in calls[0]
    assert calls[0]["kernel_summary"] is True
    assert "augmented_tree" not in calls[0]


def test_arch_json_path_per_job(tmp_path):
    arch = os.path.join(str(tmp_path), "MI355X.json")
    with open(arch, "w", encoding="utf-8") as handle:
        json.dump({"name": "MI355X"}, handle)
    a = _touch(tmp_path, "a.json")
    b = _touch(tmp_path, "b.json")
    cfg = _config(
        tmp_path,
        jobs=[
            {"name": "mi300x", "trace_path": a, "platform": "MI300X"},
            {"name": "mi355x", "trace_path": b, "gpu_arch_json_path": arch},
        ],
    )
    calls, _ = _run(cfg)
    assert calls[0]["gpu_arch_platform"] == "MI300X"
    assert calls[1]["gpu_arch_json_path"] == os.path.abspath(arch)
    assert "gpu_arch_platform" not in calls[1]


@pytest.mark.parametrize(
    "arch", [{}, {"platform": "MI300X", "gpu_arch_json_path": "x"}]
)
def test_exactly_one_arch_source(tmp_path, arch):
    job = {"trace_path": _touch(tmp_path, "t.json"), **arch}
    with pytest.raises(ConfigError, match="exactly one"):
        plan_reports(_config(tmp_path, jobs=[job]))


def test_unknown_platform(tmp_path):
    job = {"trace_path": _touch(tmp_path, "t.json"), "platform": "NotAPlatform"}
    with pytest.raises(ConfigError, match="NotAPlatform"):
        plan_reports(_config(tmp_path, jobs=[job]))


def test_missing_trace(tmp_path):
    job = {
        "trace_path": os.path.join(str(tmp_path), "missing.json"),
        "platform": "MI300X",
    }
    with pytest.raises(ConfigError, match="not found"):
        plan_reports(_config(tmp_path, jobs=[job]))


def test_later_job_error_runs_nothing(tmp_path):
    good = _touch(tmp_path, "good.json")
    cfg = _config(
        tmp_path,
        jobs=[
            {"name": "a", "trace_path": good, "platform": "MI300X"},
            {"name": "b", "trace_path": good, "platform": "NotAPlatform"},
        ],
    )
    runner = _Recorder()
    with pytest.raises(ConfigError):
        run_config(cfg, runners={"default": runner})
    assert runner.calls == []


def test_capture_folder_only_in_inference(tmp_path):
    job = {
        "trace_path": _touch(tmp_path, "t.json"),
        "platform": "MI300X",
        "capture_folder": _mkdir(tmp_path, "capture"),
    }
    with pytest.raises(ConfigError, match="capture_folder"):
        plan_reports(_config(tmp_path, jobs=[job]))


def test_inference_without_capture_is_eager(tmp_path):
    calls, merged = _run(_config(tmp_path, mode="inference"), mode="inference")
    assert merged == []
    assert "augmented_tree" not in calls[0]


def test_compare_to_matches_comparison_json_path(tmp_path):
    new = _touch(tmp_path, "new.json.gz")
    old = _touch(tmp_path, "old.json.gz")
    cfg = _config(
        tmp_path,
        jobs=[
            {"trace_path": new, "platform": "MI300X", "compare_to": {"trace_path": old}}
        ],
    )
    calls, _ = _run(cfg)
    assert len(calls) == 1
    assert calls[0]["profile_json_path"] == new
    assert calls[0]["comparison_json_path"] == old
    assert "comparison_augmented_tree" not in calls[0]


def test_compare_to_capture_folder_in_inference(tmp_path):
    new = _touch(tmp_path, "new.json.gz")
    old = _touch(tmp_path, "old.json.gz")
    new_cap = _mkdir(tmp_path, "new_cap")
    old_cap = _mkdir(tmp_path, "old_cap")
    cfg = _config(
        tmp_path,
        mode="inference",
        jobs=[
            {
                "trace_path": new,
                "platform": "MI300X",
                "capture_folder": new_cap,
                "compare_to": {"trace_path": old, "capture_folder": old_cap},
            }
        ],
    )
    calls, merged = _run(cfg, mode="inference")
    assert merged == [(new_cap, new), (old_cap, old)]
    assert calls[0]["augmented_tree"] == "TREE:new_cap"
    assert calls[0]["comparison_augmented_tree"] == "TREE:old_cap"


def test_compare_to_capture_folder_rejected_on_default(tmp_path):
    job = {
        "trace_path": _touch(tmp_path, "new.json"),
        "platform": "MI300X",
        "compare_to": {
            "trace_path": _touch(tmp_path, "old.json"),
            "capture_folder": _mkdir(tmp_path, "old_cap"),
        },
    }
    with pytest.raises(ConfigError, match="compare_to.capture_folder"):
        plan_reports(_config(tmp_path, jobs=[job]))


@pytest.mark.parametrize(
    "compare", ["old.json", {}, {"trace_path": "missing.json"}, {"platform": "MI300X"}]
)
def test_bad_compare_to(tmp_path, compare):
    job = {
        "trace_path": _touch(tmp_path, "new.json"),
        "platform": "MI300X",
        "compare_to": compare,
    }
    with pytest.raises(ConfigError, match="compare_to"):
        plan_reports(_config(tmp_path, jobs=[job]))


def test_precomputed_diff_stats_passthrough(tmp_path):
    stats = _touch(tmp_path, "diff_stats.csv")
    job = {
        "trace_path": _touch(tmp_path, "t.json"),
        "platform": "MI300X",
        "precomputed_diff_stats": stats,
    }
    calls, _ = _run(_config(tmp_path, jobs=[job]))
    assert calls[0]["precomputed_diff_stats"] == stats
    assert "comparison_json_path" not in calls[0]


def test_compare_to_and_precomputed_diff_stats_exclusive(tmp_path):
    job = {
        "trace_path": _touch(tmp_path, "new.json"),
        "platform": "MI300X",
        "compare_to": {"trace_path": _touch(tmp_path, "old.json")},
        "precomputed_diff_stats": _touch(tmp_path, "diff_stats.csv"),
    }
    with pytest.raises(ConfigError, match="both compare_to and precomputed_diff_stats"):
        plan_reports(_config(tmp_path, jobs=[job]))


def test_report_flag_on_job_rejected(tmp_path):
    job = {
        "trace_path": _touch(tmp_path, "t.json"),
        "platform": "MI300X",
        "topk_ops": 10,
    }
    with pytest.raises(ConfigError, match="top level"):
        plan_reports(_config(tmp_path, jobs=[job]))


def test_empty_jobs(tmp_path):
    with pytest.raises(ConfigError, match="jobs"):
        plan_reports(_config(tmp_path, jobs=[]) | {"jobs": []})


def test_load_run_config_roundtrip(tmp_path):
    path = os.path.join(str(tmp_path), "run.yaml")
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(_config(tmp_path, kernel_summary=True), handle)
    assert plan_reports(load_run_config(path))[0].kwargs["kernel_summary"] is True


def test_main_exits_on_config_error(tmp_path):
    path = os.path.join(str(tmp_path), "bad.yaml")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("analysis_mode: default\n")
    with pytest.raises(SystemExit):
        main(["--config", path])
