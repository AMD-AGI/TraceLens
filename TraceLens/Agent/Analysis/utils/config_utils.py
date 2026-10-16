###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Complete and check the run config the analysis orchestrator hands to the
perf report wrapper.

    python3 -m TraceLens.Agent.Analysis.utils.config_utils build ...
    python3 -m TraceLens.Agent.Analysis.utils.config_utils check --config run.yaml
    python3 -m TraceLens.Agent.Analysis.utils.config_utils set-diff-stats \\
        --config <output_dir>/cache/run_config.yaml --path <output_dir>/semantic/tracediff_output/diff_stats.csv

Every front end (interactive answers, a user YAML, Hyperloom) produces a dict in
the wrapper's schema. ``check_config`` rejects a wrong shape for the scope,
adds the wiring and forces the agent report flags on, then lets the wrapper
validate the result. Every change is printed. The completed config is written to
``<output_dir>/cache/run_config.yaml`` only when all three phases pass. That
file is the wrapper's input only: a re-check of it reports job 1's platform as
platform2.
"""

import argparse
import copy
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import yaml

from TraceLens.Reporting.generate_perf_report_from_config import (
    ConfigError,
    load_run_config,
    plan_reports,
)

# Constants
SCOPES = ("standalone", "comparative", "semantic")
_DEFAULT_FLAGS = {
    "enable_pseudo_ops": True,
    "group_by_num_kernels": True,
    "include_call_stack": True,
}
AGENT_REPORT_FLAGS = {
    "default": _DEFAULT_FLAGS,
    "inference": {**_DEFAULT_FLAGS, "group_by_parent_module": True},
}
_PATH_KEYS = ("trace_path", "capture_folder")
_COMPARISON_KEYS = ("compare_to", "precomputed_diff_stats")
_EXPLICIT_KEYS = ("analysis_mode", "output_dir", "jobs")
_CACHE_CONFIG_PATH = os.path.join("cache", "run_config.yaml")


@dataclass(frozen=True)
class AnalysisConfig:
    run_config: dict  # completed, wrapper schema, written to cache/run_config.yaml
    changes: List[Tuple[str, str]]  # (verb, text); verb is added/set/renamed/removed
    variables: Dict[str, str]  # the orchestrator's variables


def check_config(config: dict, scope: Optional[str] = None) -> AnalysisConfig:
    """Run the three phases on ``config``. Raises ``ConfigError``.

    ``config`` is not modified. ``scope`` is derived when the shape makes it
    unambiguous.
    """
    jobs = config.get("jobs")
    if (
        not isinstance(config.get("analysis_mode"), str)
        or not isinstance(jobs, list)
        or not jobs
        or not all(isinstance(job, dict) for job in jobs)
    ):
        # The wrapper owns the message for a malformed analysis_mode or jobs
        # list, and raises on it.
        plan_reports(config)

    if scope is None:
        if len(jobs) == 1:
            scope = "standalone"
        elif len(jobs) == 2 and any(job.get("compare_to") for job in jobs):
            scope = "comparative"
        else:
            raise ConfigError(
                "scope is not stated and cannot be derived from this config; "
                "pass one of: " + ", ".join(SCOPES) + "."
            )
    elif scope not in SCOPES:
        raise ConfigError(f"unknown scope {scope!r}; use one of: {', '.join(SCOPES)}.")

    paths = {"output_dir": config.get("output_dir")}
    errors = []
    for index, job in enumerate(jobs):
        if job.get("platform") is None:
            errors.append(
                f"jobs[{index}] needs a platform, not gpu_arch_json_path: "
                "the analysis steps need a platform name."
            )
        for key in _PATH_KEYS:
            paths[f"jobs[{index}].{key}"] = job.get(key)
        compare = job.get("compare_to")
        if isinstance(compare, dict):
            for key in _PATH_KEYS:
                paths[f"jobs[{index}].compare_to.{key}"] = compare.get(key)
        paths[f"jobs[{index}].precomputed_diff_stats"] = job.get(
            "precomputed_diff_stats"
        )
    errors += [
        f"{where} must be an absolute path: {value}"
        for where, value in paths.items()
        if isinstance(value, str) and not os.path.isabs(value)
    ]
    errors += _shape_errors(scope, jobs, config.get("analysis_mode"))
    if errors:
        raise ConfigError("\n".join(errors))

    completed = copy.deepcopy(config)
    jobs = completed["jobs"]
    changes = []
    if scope == "standalone":
        name = jobs[0].pop("name", None)
        if name is not None:
            changes.append(("removed", f"jobs[0].name: {name}"))
    else:
        for index, job in enumerate(jobs):
            name = f"trace{index + 1}"
            if job.get("name") is None:
                changes.append(("added", f"jobs[{index}].name: {name}"))
            elif job["name"] != name:
                changes.append(
                    ("renamed", f"jobs[{index}].name: {job['name']} -> {name}")
                )
            job["name"] = name
        if scope == "comparative" and jobs[0].get("compare_to") is None:
            jobs[0]["compare_to"] = {
                key: jobs[1][key] for key in _PATH_KEYS if jobs[1].get(key) is not None
            }
            changes.append(
                ("added", f"jobs[0].compare_to: {json.dumps(jobs[0]['compare_to'])}")
            )
        platform = jobs[0]["platform"]
        platform2 = jobs[1]["platform"]
        if platform2 != platform:
            changes.append(
                (
                    "set",
                    f"jobs[1].platform: {platform}   (was {platform2}, kept as platform2)",
                )
            )
        jobs[1]["platform"] = platform

    for flag, value in AGENT_REPORT_FLAGS.get(
        completed.get("analysis_mode"), {}
    ).items():
        if flag not in completed:
            changes.append(("added", f"{flag}: {json.dumps(value)}"))
        elif completed[flag] != value:
            changes.append(
                (
                    "set",
                    f"{flag}: {json.dumps(value)}   (was {json.dumps(completed[flag])})",
                )
            )
        completed[flag] = value

    plans = plan_reports(completed)

    variables = {
        "comparison_scope": "standalone" if scope == "standalone" else "comparative",
        "trace_path": plans[0].trace_path,
        "platform": plans[0].platform,
        "output_dir": os.path.dirname(plans[0].output_xlsx_path),
    }
    if scope != "standalone":
        variables.update(
            comparison_method="semantic" if scope == "semantic" else "tracediff",
            trace2_path=plans[1].trace_path,
            platform2=platform2,
        )
    return AnalysisConfig(completed, changes, variables)


def add_diff_stats(config: dict, path: str) -> Tuple[dict, Tuple[str, str]]:
    """Return ``config`` with job 1 reading ``path`` as its diff stats, and the
    change. An existing diff stats path on job 1 is replaced. Raises
    ``ConfigError`` unless ``config`` is semantic-shaped, ``path`` is absolute
    and the wrapper accepts the result."""
    updated = copy.deepcopy(config)
    old = updated["jobs"][0].pop("precomputed_diff_stats", None)
    errors = _shape_errors("semantic", updated["jobs"], updated.get("analysis_mode"))
    if not os.path.isabs(path):
        errors.append(f"path must be an absolute path: {path}")
    if errors:
        raise ConfigError("\n".join(errors))
    updated["jobs"][0]["precomputed_diff_stats"] = path
    plan_reports(updated)
    if old in (None, path):
        return updated, ("added", f"jobs[0].precomputed_diff_stats: {path}")
    return updated, ("set", f"jobs[0].precomputed_diff_stats: {path}   (was {old})")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Complete and check the analysis agent's run config."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser(
        "build", help="Build the config from arguments, then check it."
    )
    build.add_argument("--scope", choices=SCOPES)
    build.add_argument("--analysis_mode", required=True)
    build.add_argument("--output_dir", required=True)
    build.add_argument(
        "--job",
        action="append",
        default=[],
        help="key=value,... for one job; repeat per job. Dotted keys nest. "
        "Values cannot contain commas; use check with a YAML file for those.",
    )
    build.add_argument(
        "--set",
        action="append",
        default=[],
        help="key=value for a top-level key; dotted keys nest.",
    )

    check = commands.add_parser("check", help="Check a run YAML.")
    check.add_argument("--config", required=True)
    check.add_argument("--scope", choices=SCOPES)

    stats = commands.add_parser(
        "set-diff-stats", help="Point job 1 at a precomputed diff_stats.csv."
    )
    stats.add_argument("--config", required=True)
    stats.add_argument("--path", required=True)

    args = parser.parse_args(argv)
    try:
        if args.command == "set-diff-stats":
            config, change = add_diff_stats(load_run_config(args.config), args.path)
            path, rows = args.config, [change]
        else:
            if args.command == "build":
                config = {
                    "analysis_mode": args.analysis_mode,
                    "output_dir": args.output_dir,
                    **_assign_pairs(args.set, _EXPLICIT_KEYS),
                    "jobs": [_assign_pairs(job.split(",")) for job in args.job],
                }
            else:
                config = load_run_config(args.config)
            result = check_config(config, args.scope)
            config = result.run_config
            path = os.path.join(result.variables["output_dir"], _CACHE_CONFIG_PATH)
            rows = [
                *result.changes,
                *((None, f"{key}={value}") for key, value in result.variables.items()),
                ("wrote", path),
            ]
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
    except (ConfigError, OSError) as exc:
        parser.error(str(exc))

    for verb, text in rows:
        print(text if verb is None else f"{verb:<8}{text}")


def _shape_errors(
    scope: str, jobs: List[dict], analysis_mode: Optional[str]
) -> List[str]:
    errors = []
    checked = list(enumerate(jobs))
    keys = _COMPARISON_KEYS
    if scope == "standalone":
        if len(jobs) != 1:
            errors.append(f"standalone takes exactly 1 job, got {len(jobs)}.")
    elif scope == "semantic":
        if len(jobs) != 2:
            errors.append(f"semantic takes exactly 2 jobs, got {len(jobs)}.")
        if analysis_mode != "inference":
            errors.append("semantic needs analysis_mode inference.")
        keys += ("capture_folder",)
    else:
        if len(jobs) != 2:
            return [f"comparative takes exactly 2 jobs, got {len(jobs)}."]
        checked = [(1, jobs[1])]
        target = {
            key: jobs[1][key] for key in _PATH_KEYS if jobs[1].get(key) is not None
        }
        if jobs[0].get("compare_to") not in (None, target):
            errors.append(
                "comparative: jobs[0].compare_to must point at jobs[1]'s "
                "trace_path and capture_folder."
            )
    return errors + [
        f"{scope}: jobs[{index}] takes no {key}."
        for index, job in checked
        for key in keys
        if job.get(key) is not None
    ]


def _assign_pairs(pairs: Iterable[str], reserved: Tuple[str, ...] = ()) -> dict:
    target = {}
    for pair in pairs:
        key, separator, value = pair.partition("=")
        parts = key.split(".")
        if not separator or "" in parts:
            raise ConfigError(f"expected key=value, got {pair!r}.")
        if parts[0] in reserved:
            raise ConfigError(f"{parts[0]} has its own argument, not --set.")
        *parents, leaf = parts
        node = target
        for parent in parents:
            node = node.setdefault(parent, {})
            if not isinstance(node, dict):
                raise ConfigError(f"{pair!r} nests into {parent}, already a value.")
        try:
            node[leaf] = yaml.safe_load(value)
        except yaml.YAMLError as exc:
            raise ConfigError(f"invalid YAML value in {pair!r}: {exc}") from exc
    return target


if __name__ == "__main__":
    main()
