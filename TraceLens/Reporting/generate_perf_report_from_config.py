###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Run one TraceLens perf report per job from a YAML config.

    TraceLens_generate_perf_report_from_config --config run.yaml

The config is validated in full before any trace is opened. ``analysis_mode``
selects the report function:

* ``default`` — ``generate_perf_report_pytorch`` (training and plain PyTorch)
* ``inference`` — ``generate_perf_report_pytorch_inference``. Set a job's
  ``capture_folder`` for graph capture traces. Speculative-decoding drafter
  graphs live in the same capture folder. Prefill and decode are separate jobs.

Top-level keys other than ``analysis_mode``, ``output_dir``, and ``jobs`` must
be parameter names of the selected report function. They apply to every job.
A flag the YAML doesn't set gets the value the matching CLI passes when the
flag isn't given. For ``inference`` that turns ``collective_analysis`` off.

A job's ``compare_to`` behaves like ``comparison_json_path``: it adds diff
columns and a ``diff_stats`` sheet to that job's report and writes nothing for
the compared trace.

One job writes ``perf_report.xlsx`` and ``perf_report_csvs/``. With more than
one job, every job needs a ``name`` and writes ``perf_report_<name>.xlsx`` and
``perf_report_<name>_csvs/``. Before the first job runs, the config is written
back as ``run_config.resolved.yaml`` with every flag explicit and absolute
paths. Jobs run in order, and the first failure stops the run.
"""

import argparse
import inspect
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional

import yaml

from TraceLens.Agent.Analysis.utils.arch_utils import list_platforms

_CONFIG_KEYS = frozenset({"analysis_mode", "output_dir", "jobs"})
_JOB_KEYS = frozenset(
    {
        "name",
        "trace_path",
        "platform",
        "gpu_arch_json_path",
        "capture_folder",
        "compare_to",
        "precomputed_diff_stats",
    }
)
_COMPARE_KEYS = frozenset({"trace_path", "capture_folder"})
_MODES = ("default", "inference")
# Where a CLI passes a different value than its function's default. The
# inference CLI never enables collective analysis.
_CLI_DEFAULTS = {
    "default": {},
    "inference": {"collective_analysis": False},
}
RESOLVED_CONFIG_NAME = "run_config.resolved.yaml"
_WRAPPER_KWARGS = frozenset(
    {
        "profile_json_path",
        "output_xlsx_path",
        "output_csvs_dir",
        "gpu_arch_json_path",
        "gpu_arch_platform",
        "gpu_arch",
        "augmented_tree",
        "comparison_json_path",
        "comparison_augmented_tree",
        "precomputed_diff_stats",
    }
)
_NAME_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-."
)


class ConfigError(ValueError):
    """The run YAML is invalid. No trace has been loaded."""


@dataclass
class PlannedJob:
    """One validated report call. ``kwargs`` holds the shared report flags."""

    analysis_mode: str
    name: Optional[str]
    trace_path: str
    platform: Optional[str]
    gpu_arch_json_path: Optional[str]
    capture_folder: Optional[str]
    compare_trace_path: Optional[str]
    compare_capture_folder: Optional[str]
    precomputed_diff_stats: Optional[str]
    output_xlsx_path: str
    output_csvs_dir: str
    kwargs: Dict[str, Any] = field(default_factory=dict)


def load_run_config(path: str) -> dict:
    """Read a YAML file and require a mapping at the top level."""
    if not isinstance(path, str) or not path:
        raise ConfigError("--config path must be a non-empty string.")
    if not os.path.isfile(path):
        raise ConfigError(f"config file not found: {path}")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError("config must be a YAML mapping.")
    return data


def plan_reports(config: Mapping[str, Any]) -> List[PlannedJob]:
    """Validate ``config`` and return one plan per job. Does not open traces."""
    if not isinstance(config, Mapping):
        raise ConfigError("config must be a mapping.")

    mode = config.get("analysis_mode")
    if mode not in _MODES:
        raise ConfigError(
            "analysis_mode is required and must be one of: " + ", ".join(_MODES) + "."
        )

    output_dir = config.get("output_dir")
    if not isinstance(output_dir, str) or not output_dir.strip():
        raise ConfigError("output_dir is required and must be a non-empty string.")
    output_dir = os.path.abspath(output_dir)

    jobs = config.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ConfigError("jobs is required and must be a non-empty list.")

    accepted = _accepted_params(_report_targets()[mode])
    kwargs = _report_kwargs(config, accepted, _CLI_DEFAULTS[mode])

    multi = len(jobs) > 1
    names = set()
    planned: List[PlannedJob] = []
    for index, job in enumerate(jobs):
        parsed = _validate_job(job, index, mode, require_name=multi)
        name = parsed["name"]
        if name is not None:
            if name in names:
                raise ConfigError(f"duplicate job name {name!r}.")
            names.add(name)
        stem = "perf_report" if name is None else f"perf_report_{name}"
        planned.append(
            PlannedJob(
                analysis_mode=mode,
                output_xlsx_path=os.path.join(output_dir, stem + ".xlsx"),
                output_csvs_dir=os.path.join(output_dir, stem + "_csvs"),
                kwargs=dict(kwargs),
                **parsed,
            )
        )
    return planned


def run_config(
    config: Mapping[str, Any],
    *,
    runners: Optional[Mapping[str, Callable]] = None,
    merge_capture: Optional[Callable[[str, str], Any]] = None,
) -> List[Any]:
    """Validate, then run one report per job.

    ``runners`` maps ``analysis_mode`` to a callable. The default map is the
    real report functions. ``merge_capture(capture_folder, trace_path)`` builds
    the inference augmented tree. Tests pass substitutes for both.
    """
    plans = plan_reports(config)
    selected = dict(_report_targets() if runners is None else runners)
    merge = _merge_capture if merge_capture is None else merge_capture
    output_dir = os.path.dirname(plans[0].output_xlsx_path)
    os.makedirs(output_dir, exist_ok=True)
    with open(
        os.path.join(output_dir, RESOLVED_CONFIG_NAME), "w", encoding="utf-8"
    ) as handle:
        yaml.safe_dump(_resolved_from_plans(plans), handle, sort_keys=False)
    results = []
    for index, plan in enumerate(plans, start=1):
        print(
            f"[{index}/{len(plans)}] {plan.name or 'report'}: "
            f"{plan.trace_path} -> {plan.output_xlsx_path}"
        )
        if plan.analysis_mode not in selected:
            raise ConfigError(
                f"no report function registered for analysis_mode {plan.analysis_mode!r}."
            )
        call = dict(plan.kwargs)
        call["profile_json_path"] = plan.trace_path
        call["output_xlsx_path"] = plan.output_xlsx_path
        call["output_csvs_dir"] = plan.output_csvs_dir
        if plan.platform is not None:
            call["gpu_arch_platform"] = plan.platform
        else:
            call["gpu_arch_json_path"] = plan.gpu_arch_json_path
        if plan.capture_folder is not None:
            call["augmented_tree"] = merge(plan.capture_folder, plan.trace_path)
        if plan.compare_trace_path is not None:
            call["comparison_json_path"] = plan.compare_trace_path
        if plan.compare_capture_folder is not None:
            call["comparison_augmented_tree"] = merge(
                plan.compare_capture_folder, plan.compare_trace_path
            )
        if plan.precomputed_diff_stats is not None:
            call["precomputed_diff_stats"] = plan.precomputed_diff_stats
        results.append(selected[plan.analysis_mode](**call))
    return results


def resolve_config(config: Mapping[str, Any]) -> dict:
    """Return ``config`` as a runnable YAML dict with every report flag explicit.

    Paths are absolute. This is what ``run_config`` writes to
    ``<output_dir>/run_config.resolved.yaml`` before the first job runs.
    """
    return _resolved_from_plans(plan_reports(config))


def _resolved_from_plans(plans: List[PlannedJob]) -> dict:
    mode = plans[0].analysis_mode
    params = inspect.signature(_report_targets()[mode]).parameters
    flags = {
        name: plans[0].kwargs.get(name, params[name].default)
        for name in _accepted_params(_report_targets()[mode])
        if name not in _WRAPPER_KWARGS
    }
    jobs = []
    for plan in plans:
        job = {}
        if plan.name is not None:
            job["name"] = plan.name
        job["trace_path"] = plan.trace_path
        if plan.platform is not None:
            job["platform"] = plan.platform
        else:
            job["gpu_arch_json_path"] = plan.gpu_arch_json_path
        if plan.capture_folder is not None:
            job["capture_folder"] = plan.capture_folder
        if plan.compare_trace_path is not None:
            job["compare_to"] = {"trace_path": plan.compare_trace_path}
            if plan.compare_capture_folder is not None:
                job["compare_to"]["capture_folder"] = plan.compare_capture_folder
        if plan.precomputed_diff_stats is not None:
            job["precomputed_diff_stats"] = plan.precomputed_diff_stats
        jobs.append(job)
    return {
        "analysis_mode": mode,
        "output_dir": os.path.dirname(plans[0].output_xlsx_path),
        **flags,
        "jobs": jobs,
    }


def run_config_file(path: str, **kwargs) -> List[Any]:
    """Load ``path`` and run it."""
    return run_config(load_run_config(path), **kwargs)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate TraceLens perf reports from a YAML run config."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the run YAML (analysis_mode, output_dir, jobs).",
    )
    args = parser.parse_args(argv)
    try:
        run_config_file(args.config)
    except ConfigError as exc:
        parser.error(str(exc))


def _report_targets() -> Dict[str, Callable]:
    from TraceLens.Reporting.generate_perf_report_pytorch import (
        generate_perf_report_pytorch as training_report,
    )
    from TraceLens.Reporting.generate_perf_report_pytorch_inference import (
        generate_perf_report_pytorch as inference_report,
    )

    return {"default": training_report, "inference": inference_report}


def _accepted_params(fn: Callable) -> set:
    params = inspect.signature(fn).parameters
    return {
        name
        for name, param in params.items()
        if param.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }


def _report_kwargs(
    config: Mapping[str, Any], accepted: set, cli_defaults: Mapping[str, Any]
) -> Dict[str, Any]:
    wrapper = sorted(set(config) & _WRAPPER_KWARGS)
    unknown = sorted(set(config) - _CONFIG_KEYS - _WRAPPER_KWARGS - accepted)
    if wrapper or unknown:
        details = []
        if wrapper:
            details.append(
                "keys that belong on a job or are set by the wrapper: "
                + ", ".join(wrapper)
            )
        if unknown:
            details.append(
                "unknown keys (not config fields or report parameters): "
                + ", ".join(unknown)
            )
        raise ConfigError("; ".join(details) + ".")

    kwargs = {key: config[key] for key in config if key in accepted}
    for key, value in cli_defaults.items():
        if key in accepted and key not in kwargs:
            kwargs[key] = value
    return kwargs


def _validate_job(job: Any, index: int, mode: str, *, require_name: bool) -> dict:
    where = f"jobs[{index}]"
    if not isinstance(job, dict):
        raise ConfigError(f"{where} must be a mapping.")
    extra = sorted(set(job) - _JOB_KEYS)
    if extra:
        raise ConfigError(
            f"{where} has unknown keys: " + ", ".join(extra) + ". "
            "Report flags belong at the top level."
        )

    name = job.get("name")
    if name is None:
        if require_name:
            raise ConfigError(
                f"{where}.name is required when there is more than one job."
            )
    elif (
        not isinstance(name, str)
        or not name
        or any(ch not in _NAME_CHARS for ch in name)
    ):
        raise ConfigError(
            f"{where}.name must contain only letters, digits, '_', '-' or '.'."
        )

    trace_path = _existing_file(job.get("trace_path"), f"{where}.trace_path")

    platform = job.get("platform")
    arch_path = job.get("gpu_arch_json_path")
    if (platform is None) == (arch_path is None):
        raise ConfigError(
            f"{where} needs exactly one of platform or gpu_arch_json_path."
        )
    if platform is not None:
        known = list_platforms()
        if platform not in known:
            raise ConfigError(
                f"{where}.platform {platform!r} is not a known arch "
                f"({', '.join(known)}). Use gpu_arch_json_path or TL_EXTENSION."
            )
    else:
        arch_path = _existing_file(arch_path, f"{where}.gpu_arch_json_path")

    capture_folder = _optional_dir(job.get("capture_folder"), f"{where}.capture_folder")
    if capture_folder is not None and mode != "inference":
        raise ConfigError(
            f"{where}.capture_folder is only valid for analysis_mode inference."
        )

    compare_trace = None
    compare_capture = None
    compare = job.get("compare_to")
    if compare is not None:
        if not isinstance(compare, dict):
            raise ConfigError(f"{where}.compare_to must be a mapping.")
        extra = sorted(set(compare) - _COMPARE_KEYS)
        if extra:
            raise ConfigError(
                f"{where}.compare_to has unknown keys: " + ", ".join(extra) + "."
            )
        compare_trace = _existing_file(
            compare.get("trace_path"), f"{where}.compare_to.trace_path"
        )
        compare_capture = _optional_dir(
            compare.get("capture_folder"), f"{where}.compare_to.capture_folder"
        )
        if compare_capture is not None and mode != "inference":
            raise ConfigError(
                f"{where}.compare_to.capture_folder is only valid for analysis_mode inference."
            )

    diff_stats = job.get("precomputed_diff_stats")
    if diff_stats is not None:
        if compare is not None:
            raise ConfigError(
                f"{where} cannot set both compare_to and precomputed_diff_stats."
            )
        diff_stats = _existing_file(diff_stats, f"{where}.precomputed_diff_stats")

    return {
        "name": name,
        "trace_path": trace_path,
        "platform": platform,
        "gpu_arch_json_path": arch_path,
        "capture_folder": capture_folder,
        "compare_trace_path": compare_trace,
        "compare_capture_folder": compare_capture,
        "precomputed_diff_stats": diff_stats,
    }


def _existing_file(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{where} is required and must be a string.")
    if not os.path.isfile(value):
        raise ConfigError(f"{where} not found: {value}")
    return os.path.abspath(value)


def _optional_dir(value: Any, where: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{where} must be a non-empty string.")
    if not os.path.isdir(value):
        raise ConfigError(f"{where} not found: {value}")
    return os.path.abspath(value)


def _merge_capture(capture_folder: str, trace_path: str):
    from TraceLens.Reporting.generate_perf_report_pytorch_inference import (
        classify_graph_capture_trace,
    )
    from TraceLens.Trace2Tree.trace_capture_merge_experimental import (
        merge_capture_trace_into_graph,
    )

    metadata_json_path = os.path.join(capture_folder, "execution_details.json")
    classify_graph_capture_trace(capture_folder)
    return merge_capture_trace_into_graph(
        capture_folder, metadata_json_path, trace_path
    )


if __name__ == "__main__":
    main()
