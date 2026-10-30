###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for the analysis agent's run config check (utils/config_utils).

Traces are empty files: the wrapper's validation only checks that they exist.
"""

import json
import os

import pytest
import yaml

from TraceLens.Agent.Analysis.utils.config_utils import (
    add_diff_stats,
    check_config,
    main,
)
from TraceLens.Reporting.generate_perf_report_from_config import ConfigError

# name -> (scope or None to derive it, user config, expected completed config,
# expected change lines or None).
# ROOT stands for the temp dir holding t1/t2 traces and c1/c2 capture folders.
CASES = {
    "standalone_named": (
        None,
        """
analysis_mode: default
output_dir: ROOT/out
jobs:
  - {name: net_a, trace_path: ROOT/t1.json.gz, platform: MI300X}
""",
        """
analysis_mode: default
output_dir: ROOT/out
enable_pseudo_ops: true
group_by_num_kernels: true
include_call_stack: true
jobs:
  - {trace_path: ROOT/t1.json.gz, platform: MI300X}
""",
        None,
    ),
    "standalone_inference_capture": (
        "standalone",
        """
analysis_mode: inference
output_dir: ROOT/out
enable_pseudo_ops: false
jobs:
  - {trace_path: ROOT/t1.json.gz, platform: MI300X, capture_folder: ROOT/c1}
""",
        """
analysis_mode: inference
output_dir: ROOT/out
enable_pseudo_ops: true
group_by_num_kernels: true
include_call_stack: true
group_by_parent_module: true
jobs:
  - {trace_path: ROOT/t1.json.gz, platform: MI300X, capture_folder: ROOT/c1}
""",
        [
            "set     enable_pseudo_ops: true   (was false)",
            "added   group_by_num_kernels: true",
            "added   include_call_stack: true",
            "added   group_by_parent_module: true",
        ],
    ),
    "comparative": (
        None,
        """
analysis_mode: default
output_dir: ROOT/out
jobs:
  - {trace_path: ROOT/t1.json.gz, platform: MI300X, compare_to: {trace_path: ROOT/t2.json.gz}}
  - {trace_path: ROOT/t2.json.gz, platform: MI325X}
""",
        """
analysis_mode: default
output_dir: ROOT/out
enable_pseudo_ops: true
group_by_num_kernels: true
include_call_stack: true
jobs:
  - name: trace1
    trace_path: ROOT/t1.json.gz
    platform: MI300X
    compare_to: {trace_path: ROOT/t2.json.gz}
  - {name: trace2, trace_path: ROOT/t2.json.gz, platform: MI325X}
""",
        [
            "added   jobs[0].name: trace1",
            "added   jobs[1].name: trace2",
            "added   enable_pseudo_ops: true",
            "added   group_by_num_kernels: true",
            "added   include_call_stack: true",
        ],
    ),
    "comparative_capture": (
        "comparative",
        """
analysis_mode: inference
output_dir: ROOT/out
jobs:
  - {name: base, trace_path: ROOT/t1.json.gz, platform: MI300X, capture_folder: ROOT/c1, compare_to: other}
  - {name: other, trace_path: ROOT/t2.json.gz, platform: OTHER_GPU, capture_folder: ROOT/c2}
""",
        """
analysis_mode: inference
output_dir: ROOT/out
enable_pseudo_ops: true
group_by_num_kernels: true
include_call_stack: true
group_by_parent_module: true
jobs:
  - name: trace1
    trace_path: ROOT/t1.json.gz
    platform: MI300X
    capture_folder: ROOT/c1
    compare_to: {trace_path: ROOT/t2.json.gz, capture_folder: ROOT/c2}
  - {name: trace2, trace_path: ROOT/t2.json.gz, platform: MI300X, capture_folder: ROOT/c2}
""",
        [
            "renamed jobs[0].name: base -> trace1",
            "renamed jobs[1].name: other -> trace2",
            'set     jobs[0].compare_to: {"trace_path": "ROOT/t2.json.gz", "capture_folder": "ROOT/c2"}   (was other)',
            "set     jobs[1].platform: MI300X   (was OTHER_GPU, no arch file; kept as platform2)",
            "added   enable_pseudo_ops: true",
            "added   group_by_num_kernels: true",
            "added   include_call_stack: true",
            "added   group_by_parent_module: true",
        ],
    ),
    "semantic": (
        "semantic",
        """
analysis_mode: inference
output_dir: ROOT/out
jobs:
  - {trace_path: ROOT/t1.json.gz, platform: MI300X}
  - {trace_path: ROOT/t2.json.gz, platform: MI325X}
""",
        """
analysis_mode: inference
output_dir: ROOT/out
enable_pseudo_ops: true
group_by_num_kernels: true
include_call_stack: true
group_by_parent_module: true
jobs:
  - {name: trace1, trace_path: ROOT/t1.json.gz, platform: MI300X}
  - {name: trace2, trace_path: ROOT/t2.json.gz, platform: MI325X}
""",
        None,
    ),
    "multi_trace": (
        "multi_trace",
        """
analysis_mode: inference
output_dir: ROOT/out
aggregation: {policy: combine_by_node}
jobs:
  - {name: prefill, phase: prefill, node: 0, trace_path: ROOT/t1.json.gz, platform: MI300X}
  - {name: decode, phase: decode, node: 0, trace_path: ROOT/t2.json.gz, platform: MI300X, capture_folder: ROOT/c2}
""",
        """
analysis_mode: inference
output_dir: ROOT/out
enable_pseudo_ops: true
group_by_num_kernels: true
include_call_stack: true
group_by_parent_module: true
aggregation: {policy: combine_by_node}
jobs:
  - {name: prefill, phase: prefill, node: 0, trace_path: ROOT/t1.json.gz, platform: MI300X}
  - {name: decode, phase: decode, node: 0, trace_path: ROOT/t2.json.gz, platform: MI300X, capture_folder: ROOT/c2}
""",
        [
            "added   enable_pseudo_ops: true",
            "added   group_by_num_kernels: true",
            "added   include_call_stack: true",
            "added   group_by_parent_module: true",
        ],
    ),
}


@pytest.fixture
def root(tmp_path):
    for name in ("t1.json.gz", "t2.json.gz"):
        (tmp_path / name).touch()
    for name in ("c1", "c2"):
        (tmp_path / name).mkdir()
    return str(tmp_path)


def _fill(text, root):
    return text.replace("ROOT", root)


def _load(text, root):
    return yaml.safe_load(_fill(text, root))


def _scope_args(scope):
    return [] if scope is None else ["--scope", scope]


def _write_run(tmp_path, config):
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(config))
    return str(path)


def _cache(tmp_path):
    return yaml.safe_load((tmp_path / "out" / "cache" / "run_config.yaml").read_text())


@pytest.mark.parametrize("case", CASES)
def test_check_config_completes_each_scope(root, case):
    scope, text, expected, changes = CASES[case]
    config = _load(text, root)
    result = check_config(config, scope)
    assert result.run_config == _load(expected, root)
    if changes is not None:
        assert [f"{verb:<8}{text}" for verb, text in result.changes] == [
            _fill(line, root) for line in changes
        ]


def test_multi_trace_scope_derived_from_aggregation(root):
    result = check_config(_load(CASES["multi_trace"][1], root))
    assert result.variables["focus_job"] == "prefill_0"


def test_multi_trace_repeated_node_rejected(root):
    config = _load(CASES["multi_trace"][1], root)
    config["jobs"][1]["phase"] = "prefill"
    config["jobs"][1]["name"] = "prefill_b"
    with pytest.raises(ConfigError, match="one job per node; repeated: prefill_0"):
        check_config(config)


@pytest.mark.parametrize("case", CASES)
def test_build_matches_yaml(root, tmp_path, capsys, case):
    scope, text, _, _ = CASES[case]
    config = _load(text, root)
    main(["check", "--config", _write_run(tmp_path, config), *_scope_args(scope)])
    from_yaml = capsys.readouterr().out
    from_yaml_file = _cache(tmp_path)

    argv = ["build", *_scope_args(scope)]
    argv += ["--analysis_mode", config["analysis_mode"]]
    argv += ["--output_dir", config["output_dir"]]
    for job in config["jobs"]:
        argv += [
            "--job",
            ",".join(f"{key}={json.dumps(value)}" for key, value in job.items()),
        ]
    for key, value in config.items():
        if key not in ("analysis_mode", "output_dir", "jobs", "aggregation"):
            argv += ["--set", f"{key}={json.dumps(value)}"]
    main(argv)

    assert capsys.readouterr().out == from_yaml
    assert _cache(tmp_path) == from_yaml_file


@pytest.mark.parametrize(
    "scope, text, expected",
    [
        pytest.param(
            None,
            "{analysis_mode: default, output_dir: ROOT/out, jobs: ["
            "{trace_path: ROOT/t1.json.gz, platform: MI300X},"
            "{trace_path: ROOT/t2.json.gz, platform: MI300X}]}",
            ["scope is not stated", "standalone, comparative, semantic"],
            id="ambiguous_scope",
        ),
        pytest.param(
            None,
            "{analysis_mode: default, output_dir: ROOT/out, jobs: []}",
            ["jobs is required"],
            id="empty_jobs",
        ),
        pytest.param(
            "standalone",
            "{analysis_mode: default, output_dir: out, jobs: ["
            "{trace_path: ROOT/t1.json.gz, gpu_arch_json_path: ROOT/arch.json,"
            " compare_to: {trace_path: ROOT/t2.json.gz}},"
            "{trace_path: t2.json.gz}]}",
            [
                "standalone takes exactly 1 job, got 2",
                "jobs[0] needs a platform",
                "jobs[1] needs a platform",
                "output_dir must be an absolute path",
                "jobs[1].trace_path must be an absolute path",
                "jobs[0] takes no compare_to",
            ],
            id="standalone_errors",
        ),
        pytest.param(
            "semantic",
            "{analysis_mode: default, output_dir: ROOT/out, jobs: ["
            "{trace_path: ROOT/t1.json.gz, platform: MI300X, capture_folder: ROOT/c1},"
            "{trace_path: ROOT/t2.json.gz, platform: MI300X}]}",
            [
                "semantic needs analysis_mode inference",
                "jobs[0] takes no capture_folder",
            ],
            id="semantic_errors",
        ),
        pytest.param(
            "multi_trace",
            "{analysis_mode: default, output_dir: ROOT/out,"
            " aggregation: {policy: combine_by_node}, jobs: ["
            "{name: a, phase: x, trace_path: ROOT/t1.json.gz, platform: MI300X},"
            "{name: b, phase: x, trace_path: ROOT/t2.json.gz, platform: MI325X,"
            " compare_to: {trace_path: ROOT/t1.json.gz}}]}",
            [
                "multi_trace: every job needs the same platform",
                "multi_trace: jobs[1] takes no compare_to",
            ],
            id="multi_trace_errors",
        ),
        pytest.param(
            "multi_trace",
            "{analysis_mode: default, output_dir: ROOT/out, include_call_stack: false,"
            " aggregation: {policy: combine}, jobs: ["
            "{name: a, phase: x, trace_path: ROOT/t1.json.gz, platform: MI300X},"
            "{name: b, phase: y, trace_path: ROOT/t2.json.gz, platform: MI300X}]}",
            [
                "aggregation.policy must be combine_by_node",
                "include_call_stack must be true, got false",
            ],
            id="multi_trace_aggregation_and_flags",
        ),
        pytest.param(
            "comparative",
            "{analysis_mode: inference, output_dir: ROOT/out, jobs: ["
            "{trace_path: ROOT/t1.json.gz, platform: MI300X, capture_folder: ROOT/c1,"
            " compare_to: {trace_path: ROOT/t3.json.gz}},"
            "{trace_path: ROOT/t2.json.gz, platform: MI300X,"
            " compare_to: {trace_path: ROOT/t1.json.gz}}]}",
            [
                "must point at jobs[1]",
                "jobs[1] takes no compare_to",
            ],
            id="comparative_errors",
        ),
        pytest.param(
            "comparative",
            "{analysis_mode: default, output_dir: ROOT/out, jobs: ["
            "{trace_path: ROOT/t1.json.gz, platform: MI300X},"
            "{trace_path: ROOT/t2.json.gz, platform: MI300X},"
            "{trace_path: ROOT/t2.json.gz, platform: MI300X}]}",
            ["comparative takes exactly 2 jobs, got 3"],
            id="three_jobs_comparative",
        ),
    ],
)
def test_check_config_rejects_structure(root, scope, text, expected):
    with pytest.raises(ConfigError) as excinfo:
        check_config(_load(text, root), scope)
    for fragment in expected:
        assert fragment in str(excinfo.value)


def test_add_diff_stats(root, tmp_path, monkeypatch, capsys):
    scope, text, _, _ = CASES["semantic"]
    completed = check_config(_load(text, root), scope).run_config
    stats = str(tmp_path / "diff_stats.csv")
    (tmp_path / "diff_stats.csv").touch()
    _write_run(tmp_path, completed)
    monkeypatch.chdir(tmp_path)

    main(["set-diff-stats", "--config", "run.yaml", "--path", stats])

    config = yaml.safe_load((tmp_path / "run.yaml").read_text())
    assert config["jobs"][0]["precomputed_diff_stats"] == stats
    assert capsys.readouterr().out == (
        f"added   jobs[0].precomputed_diff_stats: {stats}\n"
    )
    with pytest.raises(ConfigError):
        add_diff_stats(completed, os.path.join(root, "missing.csv"))
    with pytest.raises(ConfigError, match="absolute"):
        add_diff_stats(completed, "diff_stats.csv")
    standalone = check_config(_load(CASES["standalone_named"][1], root)).run_config
    with pytest.raises(ConfigError, match="semantic takes exactly 2 jobs"):
        add_diff_stats(standalone, stats)


@pytest.mark.parametrize(
    "case, tail",
    [
        pytest.param(
            "standalone_named",
            [
                "comparison_scope=standalone",
                "trace_path=ROOT/t1.json.gz",
                "platform=MI300X",
                "output_dir=ROOT/out",
                "wrote   ROOT/out/cache/run_config.yaml",
            ],
            id="standalone",
        ),
        pytest.param(
            "standalone_inference_capture",
            [
                "comparison_scope=standalone",
                "trace_path=ROOT/t1.json.gz",
                "platform=MI300X",
                "output_dir=ROOT/out",
                "capture_folder=ROOT/c1",
                "wrote   ROOT/out/cache/run_config.yaml",
            ],
            id="standalone_capture",
        ),
        pytest.param(
            "comparative",
            [
                "comparison_scope=comparative",
                "trace_path=ROOT/t1.json.gz",
                "platform=MI300X",
                "output_dir=ROOT/out",
                "comparison_method=tracediff",
                "trace2_path=ROOT/t2.json.gz",
                "platform2=MI325X",
                "wrote   ROOT/out/cache/run_config.yaml",
            ],
            id="comparative",
        ),
        pytest.param(
            "semantic",
            [
                "comparison_scope=comparative",
                "trace_path=ROOT/t1.json.gz",
                "platform=MI300X",
                "output_dir=ROOT/out",
                "comparison_method=semantic",
                "trace2_path=ROOT/t2.json.gz",
                "platform2=MI325X",
                "wrote   ROOT/out/cache/run_config.yaml",
            ],
            id="semantic",
        ),
        pytest.param(
            "multi_trace",
            [
                "comparison_scope=standalone",
                "trace_path=ROOT/t1.json.gz",
                "platform=MI300X",
                "output_dir=ROOT/out",
                "focus_job=prefill_0",
                "wrote   ROOT/out/cache/run_config.yaml",
            ],
            id="multi_trace",
        ),
    ],
)
def test_cli_check_prints_variables(root, tmp_path, capsys, case, tail):
    scope, text, _, _ = CASES[case]

    main(
        [
            "check",
            "--config",
            _write_run(tmp_path, _load(text, root)),
            *_scope_args(scope),
        ]
    )

    assert capsys.readouterr().out.splitlines()[-len(tail) :] == [
        _fill(line, root) for line in tail
    ]


def test_cli_check_failures(root, tmp_path, capsys):
    config = _load(CASES["standalone_named"][1], root)
    argv = ["check", *_scope_args("standalone")]

    config["output_dir"] = root + "/bad_out"
    del config["jobs"][0]["platform"]
    with pytest.raises(SystemExit) as excinfo:
        main([*argv, "--config", _write_run(tmp_path, config)])
    assert excinfo.value.code == 2
    assert not os.path.exists(root + "/bad_out")

    config["output_dir"] = root + "/t1.json.gz/out"
    config["jobs"][0]["platform"] = "MI300X"
    capsys.readouterr()
    with pytest.raises(SystemExit) as excinfo:
        main([*argv, "--config", _write_run(tmp_path, config)])
    assert excinfo.value.code == 2
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    "args, message",
    [
        (["--job", "trace_path"], "expected key=value"),
        (["--job", "=MI300X"], "expected key=value"),
        (["--job", "platform=[MI300X"], "invalid YAML value"),
        (["--set", "a=1", "--set", "a.b=2"], "already a value"),
        (["--set", "output_dir.x=1"], "output_dir has its own argument"),
    ],
)
def test_build_rejects_malformed_pairs(tmp_path, capsys, args, message):
    argv = ["build", "--analysis_mode", "default", "--output_dir", str(tmp_path)]
    with pytest.raises(SystemExit) as excinfo:
        main(argv + args)
    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert message in captured.err
