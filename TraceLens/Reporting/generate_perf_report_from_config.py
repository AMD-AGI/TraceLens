###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Run one TraceLens perf report per job from a YAML config.

    python -m TraceLens.Reporting.generate_perf_report_from_config --config run.yaml

The config is validated in full before any trace is opened. ``analysis_mode``
selects the report function:

* ``default`` — ``generate_perf_report_pytorch``
* ``inference_graph_capture``, ``spec_dec``, ``pd_disaggregation`` —
  ``generate_perf_report_pytorch_inference``

``spec_dec`` and ``pd_disaggregation`` do not have dedicated report functions
yet. Both run the inference report. ``spec_decode`` (method, num_spec_tokens)
is checked and kept on the plan, and is not forwarded as a kwarg.
``pd_disaggregation`` names each workbook ``perf_report_<role>_rank<r>``.

Top-level keys other than ``analysis_mode``, ``output_dir``, ``jobs``, and
``spec_decode`` must be parameter names of the selected report function.
They apply to every job. These recipe flags default on when that function
accepts them, unless the YAML sets them:

* ``enable_pseudo_ops``
* ``group_by_num_kernels``
* ``include_call_stack``
* ``group_by_parent_module``

Comparison stays outside this wrapper: pass one trace per job, then run the
compare step separately.
"""

import argparse
import inspect
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional

import yaml

from TraceLens.Agent.Analysis.utils.arch_utils import list_platforms

_CONFIG_KEYS = frozenset({"analysis_mode", "output_dir", "jobs", "spec_decode"})
_JOB_KEYS = frozenset({"trace_path", "platform", "capture_folder", "role", "rank"})
_MODES = (
    "default",
    "inference_graph_capture",
    "spec_dec",
    "pd_disaggregation",
)
_INFERENCE_MODES = frozenset(
    {"inference_graph_capture", "spec_dec", "pd_disaggregation"}
)
_SPEC_METHODS = frozenset({"eagle", "eagle3", "mtp"})
_RECIPE_DEFAULTS = {
    "enable_pseudo_ops": True,
    "group_by_num_kernels": True,
    "include_call_stack": True,
    "group_by_parent_module": True,
}
# Derived by the wrapper, or owned by the separate compare step.
_REJECTED_KWARGS = frozenset(
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
_ROLE_OK = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")


class ConfigError(ValueError):
    """The run YAML is invalid. No trace has been loaded."""


@dataclass
class PlannedJob:
    """One validated report call. ``kwargs`` does not include path or arch fields."""

    analysis_mode: str
    trace_path: str
    platform: str
    capture_folder: Optional[str]
    role: Optional[str]
    rank: Optional[int]
    output_xlsx_path: str
    output_csvs_dir: str
    kwargs: Dict[str, Any] = field(default_factory=dict)
    spec_decode: Optional[Dict[str, Any]] = None


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

    jobs = config.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ConfigError("jobs is required and must be a non-empty list.")

    spec_decode = _validate_spec_decode(config.get("spec_decode"), mode)
    report_fn = _report_targets()[mode]
    accepted = _accepted_params(report_fn)
    kwargs = _report_kwargs(config, accepted)

    parsed_jobs: List[dict] = []
    seen_pd = set()
    stems: List[str] = []
    for index, job in enumerate(jobs):
        parsed = _validate_job(job, index, mode)
        if mode == "pd_disaggregation":
            key = (parsed["role"], parsed["rank"])
            if key in seen_pd:
                raise ConfigError(
                    f"duplicate PD job role={parsed['role']!r} rank={parsed['rank']}."
                )
            seen_pd.add(key)
        else:
            stems.append(_unique_stem(_trace_stem(parsed["trace_path"]), stems))
        parsed_jobs.append(parsed)

    output_dir = os.path.abspath(output_dir)
    planned: List[PlannedJob] = []
    for index, parsed in enumerate(parsed_jobs):
        xlsx_path, csvs_dir = _output_paths(
            output_dir,
            mode,
            parsed,
            stems[index] if stems else None,
            len(parsed_jobs),
        )
        planned.append(
            PlannedJob(
                analysis_mode=mode,
                trace_path=parsed["trace_path"],
                platform=parsed["platform"],
                capture_folder=parsed["capture_folder"],
                role=parsed["role"],
                rank=parsed["rank"],
                output_xlsx_path=xlsx_path,
                output_csvs_dir=csvs_dir,
                kwargs=dict(kwargs),
                spec_decode=dict(spec_decode) if spec_decode else None,
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
    results = []
    for plan in plans:
        if plan.analysis_mode not in selected:
            raise ConfigError(
                f"no report function registered for analysis_mode {plan.analysis_mode!r}."
            )
        fn = selected[plan.analysis_mode]
        call = dict(plan.kwargs)
        call["profile_json_path"] = plan.trace_path
        call["output_xlsx_path"] = plan.output_xlsx_path
        call["output_csvs_dir"] = plan.output_csvs_dir
        call["gpu_arch_platform"] = plan.platform
        if plan.capture_folder is not None:
            call["augmented_tree"] = merge(plan.capture_folder, plan.trace_path)
        os.makedirs(plan.output_csvs_dir, exist_ok=True)
        os.makedirs(os.path.dirname(plan.output_xlsx_path), exist_ok=True)
        results.append(fn(**call))
    return results


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

    return {
        "default": training_report,
        "inference_graph_capture": inference_report,
        "spec_dec": inference_report,
        "pd_disaggregation": inference_report,
    }


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


def _report_kwargs(config: Mapping[str, Any], accepted: set) -> Dict[str, Any]:
    unknown = sorted(set(config) - _CONFIG_KEYS - accepted)
    rejected = sorted(set(config) & _REJECTED_KWARGS)
    if unknown or rejected:
        details = []
        if unknown:
            details.append(
                "unknown keys (not config fields or report parameters): "
                + ", ".join(unknown)
            )
        if rejected:
            details.append(
                "keys set by the wrapper or the separate compare step: "
                + ", ".join(rejected)
            )
        raise ConfigError("; ".join(details) + ".")

    kwargs = {key: config[key] for key in config if key in accepted}
    for key, value in _RECIPE_DEFAULTS.items():
        if key in accepted and key not in kwargs:
            kwargs[key] = value
    return kwargs


def _validate_spec_decode(spec: Any, mode: str) -> Optional[dict]:
    if mode != "spec_dec":
        if spec is not None:
            raise ConfigError(
                "spec_decode is only valid when analysis_mode is spec_dec."
            )
        return None
    if not isinstance(spec, dict):
        raise ConfigError(
            "spec_dec requires spec_decode with method and num_spec_tokens."
        )
    extra = sorted(set(spec) - {"method", "num_spec_tokens"})
    if extra:
        raise ConfigError("unknown spec_decode keys: " + ", ".join(extra) + ".")
    method = spec.get("method")
    if method not in _SPEC_METHODS:
        raise ConfigError(
            "spec_decode.method must be one of: "
            + ", ".join(sorted(_SPEC_METHODS))
            + "."
        )
    tokens = spec.get("num_spec_tokens")
    if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 1:
        raise ConfigError("spec_decode.num_spec_tokens must be an integer >= 1.")
    return {"method": method, "num_spec_tokens": tokens}


def _validate_job(job: Any, index: int, mode: str) -> dict:
    where = f"jobs[{index}]"
    if not isinstance(job, dict):
        raise ConfigError(f"{where} must be a mapping.")
    extra = sorted(set(job) - _JOB_KEYS)
    if extra:
        raise ConfigError(
            f"{where} has unknown keys: " + ", ".join(extra) + ". "
            "Report flags belong at the top level."
        )

    trace_path = job.get("trace_path")
    if not isinstance(trace_path, str) or not trace_path.strip():
        raise ConfigError(f"{where}.trace_path is required and must be a string.")
    if not os.path.isfile(trace_path):
        raise ConfigError(f"{where}.trace_path not found: {trace_path}")

    platform = job.get("platform")
    if not isinstance(platform, str) or not platform.strip():
        raise ConfigError(f"{where}.platform is required and must be a string.")
    known = list_platforms()
    if platform not in known:
        raise ConfigError(
            f"{where}.platform {platform!r} is not a known arch "
            f"({', '.join(known)})."
        )

    capture = job.get("capture_folder", None)
    if capture is None:
        capture_folder = None
    elif not isinstance(capture, str) or not capture.strip():
        raise ConfigError(f"{where}.capture_folder must be a non-empty string.")
    elif not os.path.isdir(capture):
        raise ConfigError(f"{where}.capture_folder not found: {capture}")
    else:
        capture_folder = capture

    if mode == "default" and capture_folder is not None:
        raise ConfigError(f"{where}.capture_folder is only valid for inference modes.")

    role = job.get("role", None)
    rank = job.get("rank", None)
    if mode == "pd_disaggregation":
        if (
            not isinstance(role, str)
            or not role
            or any(ch not in _ROLE_OK for ch in role)
        ):
            raise ConfigError(
                f"{where}.role is required for pd_disaggregation "
                "and must contain only letters, digits, '_' or '-'."
            )
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
            raise ConfigError(
                f"{where}.rank is required for pd_disaggregation "
                "and must be an integer >= 0."
            )
    elif role is not None or rank is not None:
        raise ConfigError(
            f"{where}.role and rank are only valid for analysis_mode pd_disaggregation."
        )

    return {
        "trace_path": os.path.abspath(trace_path),
        "platform": platform,
        "capture_folder": (
            os.path.abspath(capture_folder) if capture_folder is not None else None
        ),
        "role": role if mode == "pd_disaggregation" else None,
        "rank": rank if mode == "pd_disaggregation" else None,
    }


def _output_paths(output_dir, mode, parsed, stem, n_jobs):
    if mode == "pd_disaggregation":
        name = f"perf_report_{parsed['role']}_rank{parsed['rank']}"
        return (
            os.path.join(output_dir, name + ".xlsx"),
            os.path.join(output_dir, name + "_csvs"),
        )
    if n_jobs == 1:
        return (
            os.path.join(output_dir, "perf_report.xlsx"),
            os.path.join(output_dir, "csvs"),
        )
    job_dir = os.path.join(output_dir, stem)
    return (
        os.path.join(job_dir, "perf_report.xlsx"),
        os.path.join(job_dir, "csvs"),
    )


def _trace_stem(path: str) -> str:
    name = os.path.basename(path)
    if name.endswith(".json.gz"):
        name = name[: -len(".json.gz")]
    elif name.endswith(".json"):
        name = name[: -len(".json")]
    return name or "trace"


def _unique_stem(base: str, used: List[str]) -> str:
    if base not in used:
        return base
    index = 2
    while f"{base}_{index}" in used:
        index += 1
    return f"{base}_{index}"


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
