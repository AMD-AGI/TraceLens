###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Schema and dispatch tests for the YAML perf-report wrapper.

These tests never open a trace. Report functions and capture merge are fakes.
"""

import os

import pytest
import yaml

from TraceLens.Reporting.generate_perf_report_from_config import (
    ConfigError,
    load_run_config,
    main,
    plan_reports,
    run_config,
)


def _touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("{}\n")
    return path


def _config(tmp_path, jobs=None, **extra):
    trace = _touch(os.path.join(str(tmp_path), "traces", "rank0.json.gz"))
    cfg = {
        "analysis_mode": "default",
        "output_dir": os.path.join(str(tmp_path), "out"),
        "jobs": jobs
        or [
            {"trace_path": trace, "platform": "MI300X"},
        ],
    }
    cfg.update(extra)
    return cfg


def test_default_plan_applies_recipe_defaults_the_function_accepts(tmp_path):
    plans = plan_reports(_config(tmp_path))
    assert len(plans) == 1
    plan = plans[0]
    assert plan.analysis_mode == "default"
    assert plan.platform == "MI300X"
    assert plan.kwargs["enable_pseudo_ops"] is True
    assert plan.kwargs["group_by_num_kernels"] is True
    assert plan.kwargs["include_call_stack"] is True
    assert "group_by_parent_module" not in plan.kwargs
    assert plan.output_xlsx_path.endswith(os.path.join("out", "perf_report.xlsx"))
    assert plan.output_csvs_dir.endswith(os.path.join("out", "csvs"))
    assert plan.capture_folder is None
    assert plan.spec_decode is None


def test_yaml_overrides_recipe_default(tmp_path):
    plans = plan_reports(_config(tmp_path, include_call_stack=False, topk_ops=50))
    assert plans[0].kwargs["include_call_stack"] is False
    assert plans[0].kwargs["topk_ops"] == 50
    assert plans[0].kwargs["enable_pseudo_ops"] is True


def test_unknown_key_rejected_before_run(tmp_path):
    cfg = _config(tmp_path, not_a_report_flag=True)
    with pytest.raises(ConfigError, match="unknown keys"):
        plan_reports(cfg)


def test_inference_only_flag_rejected_on_default(tmp_path):
    cfg = _config(tmp_path, group_by_parent_module=True)
    with pytest.raises(ConfigError, match="group_by_parent_module"):
        plan_reports(cfg)


def test_comparison_kwargs_rejected(tmp_path):
    cfg = _config(tmp_path, comparison_json_path="other.json")
    with pytest.raises(ConfigError, match="compare step"):
        plan_reports(cfg)


def test_unknown_platform_rejected(tmp_path):
    trace = _touch(os.path.join(str(tmp_path), "t.json"))
    cfg = _config(
        tmp_path,
        jobs=[{"trace_path": trace, "platform": "NotAPlatform"}],
    )
    with pytest.raises(ConfigError, match="NotAPlatform"):
        plan_reports(cfg)


def test_missing_trace_rejected(tmp_path):
    cfg = _config(
        tmp_path,
        jobs=[
            {
                "trace_path": os.path.join(str(tmp_path), "missing.json"),
                "platform": "MI300X",
            }
        ],
    )
    with pytest.raises(ConfigError, match="not found"):
        plan_reports(cfg)


def test_second_job_error_does_not_call_runner(tmp_path):
    good = _touch(os.path.join(str(tmp_path), "good.json"))
    cfg = _config(
        tmp_path,
        jobs=[
            {"trace_path": good, "platform": "MI300X"},
            {"trace_path": good, "platform": "NotAPlatform"},
        ],
    )
    calls = []

    def runner(**kwargs):
        calls.append(kwargs)

    with pytest.raises(ConfigError, match="NotAPlatform"):
        run_config(cfg, runners={"default": runner})
    assert calls == []


def test_run_passes_platform_and_outputs(tmp_path):
    cfg = _config(tmp_path, kernel_summary=True)
    calls = []

    def runner(**kwargs):
        calls.append(kwargs)
        return {"ok": True}

    results = run_config(cfg, runners={"default": runner})
    assert results == [{"ok": True}]
    assert calls[0]["gpu_arch_platform"] == "MI300X"
    assert calls[0]["kernel_summary"] is True
    assert calls[0]["enable_pseudo_ops"] is True
    assert calls[0]["output_xlsx_path"].endswith("perf_report.xlsx")
    assert os.path.isdir(calls[0]["output_csvs_dir"])
    assert "augmented_tree" not in calls[0]


def test_inference_capture_is_merged_after_validation(tmp_path):
    trace = _touch(os.path.join(str(tmp_path), "rank0.json.gz"))
    capture = os.path.join(str(tmp_path), "capture_traces")
    os.makedirs(capture)
    cfg = {
        "analysis_mode": "inference_graph_capture",
        "output_dir": os.path.join(str(tmp_path), "out"),
        "jobs": [
            {
                "trace_path": trace,
                "platform": "MI300X",
                "capture_folder": capture,
            }
        ],
    }
    calls = []
    merged = []

    def runner(**kwargs):
        calls.append(kwargs)

    def merge(folder, trace_path):
        merged.append((folder, trace_path))
        return "TREE"

    run_config(
        cfg,
        runners={"inference_graph_capture": runner},
        merge_capture=merge,
    )
    assert merged[0][0] == os.path.abspath(capture)
    assert merged[0][1].endswith("rank0.json.gz")
    assert calls[0]["augmented_tree"] == "TREE"
    assert calls[0]["group_by_parent_module"] is True


def test_capture_folder_rejected_for_default(tmp_path):
    trace = _touch(os.path.join(str(tmp_path), "t.json"))
    capture = os.path.join(str(tmp_path), "capture")
    os.makedirs(capture)
    cfg = _config(
        tmp_path,
        jobs=[
            {
                "trace_path": trace,
                "platform": "MI300X",
                "capture_folder": capture,
            }
        ],
    )
    with pytest.raises(ConfigError, match="capture_folder"):
        plan_reports(cfg)


def test_spec_dec_validates_and_does_not_forward_spec_decode(tmp_path):
    trace = _touch(os.path.join(str(tmp_path), "rank0.trace.json.gz"))
    cfg = {
        "analysis_mode": "spec_dec",
        "output_dir": os.path.join(str(tmp_path), "out"),
        "spec_decode": {"method": "eagle", "num_spec_tokens": 3},
        "jobs": [{"trace_path": trace, "platform": "MI300X"}],
    }
    calls = []

    def runner(**kwargs):
        calls.append(kwargs)

    plans = plan_reports(cfg)
    assert plans[0].spec_decode == {"method": "eagle", "num_spec_tokens": 3}
    run_config(cfg, runners={"spec_dec": runner}, merge_capture=lambda *_: None)
    assert "spec_decode" not in calls[0]
    assert "method" not in calls[0]
    assert "num_spec_tokens" not in calls[0]


def test_spec_decode_rejected_on_other_modes(tmp_path):
    cfg = _config(tmp_path, spec_decode={"method": "eagle", "num_spec_tokens": 3})
    with pytest.raises(ConfigError, match="only valid"):
        plan_reports(cfg)


@pytest.mark.parametrize(
    "spec",
    [
        None,
        {},
        {"method": "draft", "num_spec_tokens": 3},
        {"method": "eagle3", "num_spec_tokens": 0},
        {"method": "mtp", "num_spec_tokens": True},
        {"method": "eagle", "num_spec_tokens": 2, "extra": 1},
    ],
)
def test_bad_spec_decode(tmp_path, spec):
    trace = _touch(os.path.join(str(tmp_path), "t.json.gz"))
    cfg = {
        "analysis_mode": "spec_dec",
        "output_dir": os.path.join(str(tmp_path), "out"),
        "jobs": [{"trace_path": trace, "platform": "MI300X"}],
    }
    if spec is not None:
        cfg["spec_decode"] = spec
    with pytest.raises(ConfigError):
        plan_reports(cfg)


def test_pd_names_one_report_per_role_rank(tmp_path):
    prefill = _touch(os.path.join(str(tmp_path), "prefill", "rank0.json.gz"))
    decode = _touch(os.path.join(str(tmp_path), "decode", "rank0.json.gz"))
    cfg = {
        "analysis_mode": "pd_disaggregation",
        "output_dir": os.path.join(str(tmp_path), "out"),
        "jobs": [
            {
                "trace_path": prefill,
                "platform": "MI300X",
                "role": "prefill",
                "rank": 0,
            },
            {
                "trace_path": decode,
                "platform": "MI300X",
                "role": "decode",
                "rank": 0,
            },
        ],
    }
    plans = plan_reports(cfg)
    names = [os.path.basename(plan.output_xlsx_path) for plan in plans]
    assert names == [
        "perf_report_prefill_rank0.xlsx",
        "perf_report_decode_rank0.xlsx",
    ]
    assert plans[0].output_csvs_dir.endswith("perf_report_prefill_rank0_csvs")
    assert plans[1].rank == 0
    assert plans[0].role == "prefill"


def test_pd_requires_role_and_rank(tmp_path):
    trace = _touch(os.path.join(str(tmp_path), "t.json"))
    cfg = {
        "analysis_mode": "pd_disaggregation",
        "output_dir": os.path.join(str(tmp_path), "out"),
        "jobs": [{"trace_path": trace, "platform": "MI300X"}],
    }
    with pytest.raises(ConfigError, match="role"):
        plan_reports(cfg)


def test_role_rejected_outside_pd(tmp_path):
    trace = _touch(os.path.join(str(tmp_path), "t.json"))
    cfg = _config(
        tmp_path,
        jobs=[
            {"trace_path": trace, "platform": "MI300X", "role": "prefill", "rank": 0}
        ],
    )
    with pytest.raises(ConfigError, match="pd_disaggregation"):
        plan_reports(cfg)


def test_duplicate_pd_role_rank(tmp_path):
    trace = _touch(os.path.join(str(tmp_path), "t.json"))
    job = {"trace_path": trace, "platform": "MI300X", "role": "prefill", "rank": 0}
    cfg = {
        "analysis_mode": "pd_disaggregation",
        "output_dir": os.path.join(str(tmp_path), "out"),
        "jobs": [job, dict(job)],
    }
    with pytest.raises(ConfigError, match="duplicate PD"):
        plan_reports(cfg)


def test_multiple_default_jobs_get_distinct_dirs(tmp_path):
    first = _touch(os.path.join(str(tmp_path), "a.json.gz"))
    second = _touch(os.path.join(str(tmp_path), "nested", "a.json.gz"))
    cfg = _config(
        tmp_path,
        jobs=[
            {"trace_path": first, "platform": "MI300X"},
            {"trace_path": second, "platform": "MI300X"},
        ],
    )
    plans = plan_reports(cfg)
    assert plans[0].output_xlsx_path.endswith(os.path.join("a", "perf_report.xlsx"))
    assert plans[1].output_xlsx_path.endswith(os.path.join("a_2", "perf_report.xlsx"))


def test_job_level_report_flag_rejected(tmp_path):
    trace = _touch(os.path.join(str(tmp_path), "t.json"))
    cfg = _config(
        tmp_path,
        jobs=[{"trace_path": trace, "platform": "MI300X", "topk_ops": 10}],
    )
    with pytest.raises(ConfigError, match="top level"):
        plan_reports(cfg)


def test_load_run_config_roundtrip(tmp_path):
    cfg = _config(tmp_path, kernel_summary=True)
    path = os.path.join(str(tmp_path), "run.yaml")
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle)
    loaded = load_run_config(path)
    assert plan_reports(loaded)[0].kwargs["kernel_summary"] is True


def test_main_reports_config_error(tmp_path):
    path = os.path.join(str(tmp_path), "bad.yaml")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("analysis_mode: default\n")
    with pytest.raises(SystemExit):
        main(["--config", path])


def test_empty_jobs_rejected(tmp_path):
    with pytest.raises(ConfigError, match="jobs"):
        plan_reports(
            {
                "analysis_mode": "default",
                "output_dir": str(tmp_path),
                "jobs": [],
            }
        )
