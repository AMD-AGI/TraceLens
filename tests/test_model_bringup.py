###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import io
import json
import sys
import time
import types
import urllib.error

import pytest
import yaml

from TraceLens.Agent.Analysis.utils.arch_utils import list_platforms
from TraceLens.Agent.ModelBringup.utils import bringup_gates
from TraceLens.Agent.PerfModel.utils import scan_framework_ops, unified_perf_ops
from TraceLens.Agent.PerfModel.utils.unified_perf_ops import (
    load_rows,
    missing_perf_model_share,
    validation_cases,
)
from TraceLens.Agent.PerfModel.validate_perf_model.ratio_bands import (
    match_status,
    ratio_band,
)
from TraceLens.Agent.PerfModel.validate_perf_model.validator import (
    measured_bytes,
)

_HEADER = "name,op category,Input Dims,Input type,Kernel Time (µs)_sum,has_perf_model,kernel_details_summary,GFLOPS\n"
_BAND_HEADER = "run_status,flops_ratio,flops_band,bytes_ratio,bytes_band,match_status\n"
_OK_ROW = "OK,1.0000,match,1.0000,match,match\n"
_MAPPING_MODULE = "TraceLens.PerfModel.torch_op_mapping"
_PERF_CSV = (
    "name,op category,Kernel Time (µs)_sum,has_perf_model,call_stack,kernel_details_summary\n"
    "aiter::gemm_a,other,50,False,vllm/x.py,kernel_a\n"
    "aiter::gemm_a,other,30,False,vllm/x.py aiter/y.py,\n"
    "op_b,GEMM,20,True,,\n"
    ",other,5,False,,\n"
)


def _template_body(name):
    return (bringup_gates.TEMPLATES_DIR / name).read_text().rsplit("-->", 1)[1]


# Cut before Ratio Deviations so text appended to the draft lands in Validation.
_DRAFT = _template_body("perf_model_pr.md").split("## Ratio Deviations")[0]
_ISSUE = _template_body("perf_model_issue.md")


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_HEADER + "".join(rows), encoding="utf-8")
    return path


def _gate(bringup, name):
    return json.loads((bringup / "gates" / f"{name}.json").read_text())


def _accept_op(bringup, op_slug, status="pass", problems=()):
    """Write the op-result gate Step 6 leaves for an op."""
    gates = bringup / "gates"
    gates.mkdir(exist_ok=True)
    (gates / f"op_result_{op_slug}.json").write_text(
        json.dumps({"status": status, "problems": list(problems)})
    )


def _write_gaps(bringup, *op_slugs):
    ops = [{"name": s, "op_slug": s} for s in op_slugs]
    (bringup / "gaps.json").write_text(json.dumps({"ops": ops, "comm_excluded": []}))


def _modeled_op(bringup, csv_rows):
    (bringup / "cache").mkdir()
    (bringup / "cache" / "bringup_config.yaml").write_text(
        "model: org/Model-7B\nplatform: platform_a\ndocker_image: null\n"
        "hf_cache_path: /data\ntracelens_dir: /srv/workdir_a\n"
    )
    op_dir = bringup / "perf_models" / "op_a"
    op_dir.mkdir(parents=True)
    (op_dir / "validation.csv").write_text(_BAND_HEADER + csv_rows)
    (op_dir / "perf_model_pr.md").write_text(_DRAFT)
    result = {
        "op": "op_a",
        "op_slug": "op_a",
        "outcome": "modeled",
        "commit": "abc123",
        "validation_csv": str(op_dir / "validation.csv"),
    }
    (op_dir / "result.json").write_text(json.dumps(result))
    args = ["op-result", "--bringup-dir", str(bringup)]
    return op_dir, args + ["--result", str(op_dir / "result.json")]


@pytest.mark.parametrize(
    "column,below,low,high,above",
    [
        ("flops_ratio", 0.99, 1.0, 1.05, 1.06),
        ("bytes_ratio", 0.99, 1.0, 1.2, 1.21),
    ],
)
def test_ratio_band(column, below, low, high, above):
    assert ratio_band(column, None) == ""
    assert ratio_band(column, low) == ratio_band(column, high) == "match"
    assert ratio_band(column, above) == "deviation"
    assert ratio_band(column, below) == "over-predicted"


def test_match_status():
    assert match_status("FAIL: x", ["match", "match"]) == ""
    assert match_status("MODEL_ONLY", ["", ""]) == ""
    assert match_status("OK", ["match", ""]) == "no prediction"
    assert match_status("OK", ["deviation", "over-predicted"]) == "over-predicted"
    assert match_status("OK", ["match", "deviation"]) == "deviation-suspect"
    assert match_status("OK", ["match", "match"]) == "match"
    assert match_status("OK", ["match", "match"], False) == "precision-mismatch"
    assert match_status("OK", ["over-predicted", "match"], False) == "over-predicted"


def test_measured_bytes():
    counters = {
        "TCC_EA0_RDREQ_sum": 10,
        "TCC_EA0_RDREQ_32B_sum": 2,
        "TCC_EA0_RDREQ_64B_sum": 3,
        "WRITE_SIZE": 4,
    }
    assert measured_bytes(counters) == (2 * 32 + 3 * 64 + 5 * 128, 4 * 1024)


def test_unified_perf_ops_share_and_cases(tmp_path):
    csv = _write_csv(
        tmp_path / "perf_report_csvs" / "unified_perf_summary.csv",
        [
            'op_a,GEMM,"((4, 8),)",bf16,60,True,,2\n',
            'op_a,GEMM,"((4, 8),)",bf16,20,True,,2\n',
            "op_b,other,,,20,False,,0\n",
        ],
    )
    assert missing_perf_model_share(load_rows(csv)[0]) == 0.2
    cases = validation_cases(csv.parent)
    assert [c["trace_name"] for c in cases] == ["op_a"]
    assert cases[0]["input_dims"] == [(4, 8)] and cases[0]["csv_flops"] == 2e9


def test_cases_split_by_concrete_inputs_and_carry_trace_kernels(tmp_path):
    report = tmp_path / "perf_report_csvs"
    report.mkdir()
    header = "name,Input Dims,Input type,Concrete Inputs,Kernel Time (µs)_sum,has_perf_model,GFLOPS,kernel_details_summary\n"
    kernels = "\"[{'name': 'k_a', 'count': 1}, {'name': 'k_b', 'count': 2}]\""
    rows = [
        f'op_a,"((4, 8),)",bf16,"(1e-06,)",1,True,2,{kernels}\n',
        f'op_a,"((4, 8),)",bf16,"(1e-06,)",1,True,2,{kernels}\n',
        f'op_a,"((4, 8),)",bf16,"(1e-05,)",1,True,2,{kernels}\n',
    ]
    (report / "unified_perf_summary.csv").write_text(header + "".join(rows))
    cases = validation_cases(report)
    assert [c["concrete_inputs"] for c in cases] == [[1e-06], [1e-05]]
    assert cases[0]["trace_kernels"] == ["k_a", "k_b"]


def test_gaps_drops_collectives_and_slugs(tmp_path):
    csv = _write_csv(
        tmp_path / "unified_perf_summary.csv",
        [
            "ns::op.x,other,,,50,False,,\n",
            "ns::op x,other,,,30,False,,\n",
            "ns::ar,CustomCollective,,,20,False,,\n",
            "x (Synthetic Op),other,,,10,False,,\n",
        ],
    )
    rc = bringup_gates.main(
        ["gaps", "--bringup-dir", str(tmp_path), "--unified-csv", str(csv)]
    )
    gaps = json.loads((tmp_path / "gaps.json").read_text())
    assert rc == 0
    assert [o["op_slug"] for o in gaps["ops"]] == [
        "ns_op_x",
        "ns_op_x_2",
        "x_synthetic_op",
    ]
    assert [o["categorize_only"] for o in gaps["ops"]] == [False, False, True]
    assert gaps["comm_excluded"] == ["ns::ar"]


def test_report_health_blocks_and_accepts(tmp_path):
    (tmp_path / "triage_details.csv").write_text(
        "DIAG Tag,Category,Failure Mode,Evidence,Remedy,Implied By\n"
        "DIAG:perf_model:3c_CAPTURE_GRAPH_MERGE_FAIL,perf_model,m,e,r,\n"
        "DIAG:trace_quality:2j_RUNTIME_INSTABILITY,trace_quality,m,e,r,\n"
    )
    args = ["report-health", "--phase", "report", "--bringup-dir", str(tmp_path)]
    args += ["--triage-dir", str(tmp_path)]
    assert bringup_gates.main(args) == 1
    assert _gate(tmp_path, "report_health")["other"] == [
        "DIAG:trace_quality:2j_RUNTIME_INSTABILITY"
    ]
    assert bringup_gates.main(args + ["--accept", "CAPTURE_GRAPH_MERGE_FAIL"]) == 0
    empty = tmp_path / "empty"
    empty.mkdir()
    assert bringup_gates.main(args[:-1] + [str(empty)]) == 2
    assert "missing" in _gate(tmp_path, "report_health")["error"]
    (empty / "triage_details.csv").write_text("DIAG Tag,Evidence,Remedy\n")
    assert bringup_gates.main(args[:-1] + [str(empty), "--phase", "profile"]) == 0


def test_coverage_needs_all_rows_ok_and_summary(tmp_path):
    after = _write_csv(tmp_path / "after.csv", ["op_a,GEMM,,,10,True,,\n"])
    op_dir = tmp_path / "perf_models" / "op_a"
    op_dir.mkdir(parents=True)
    (op_dir / "validation.csv").write_text(
        "op,run_status,match_status\nop_a,OK,match\nop_a,MODEL_ONLY,\n"
    )
    (op_dir / "result.json").write_text(
        json.dumps(
            {
                "op": "op_a",
                "op_slug": "op_a",
                "outcome": "modeled",
                "validation_csv": str(op_dir / "validation.csv"),
            }
        )
    )
    _accept_op(tmp_path, "op_a")
    _write_gaps(tmp_path, "op_a")
    args = ["coverage", "--bringup-dir", str(tmp_path)]
    args += ["--before", str(after), "--after", str(after)]
    assert bringup_gates.main(args) == 1
    assert _gate(tmp_path, "coverage")["failing"] == ["op_a"]
    bringup_gates.main(["summary", "--bringup-dir", str(tmp_path)])
    text = (tmp_path / "bringup_summary.md").read_text()
    assert "| prepare | not_run |" in text and "| coverage | fail |" in text
    assert "for ops op_a" in text
    assert _gate(tmp_path, "summary")["stop"] is None
    stop = [
        "summary",
        "--bringup-dir",
        str(tmp_path),
        "--step",
        "4",
        "--code",
        "CODE_A",
        "--reason",
        "reason a",
    ]
    assert bringup_gates.main(stop) == 0
    tag = "[DIAG:bringup:STEP4_CODE_A]"
    assert _gate(tmp_path, "summary")["stop"] == tag
    assert (
        f"**Stopped:** {tag} reason a" in (tmp_path / "bringup_summary.md").read_text()
    )


def test_op_result_gates_outcomes(tmp_path):
    op_dir, args = _modeled_op(tmp_path, _OK_ROW)
    csv = op_dir / "validation.csv"
    path = op_dir / "result.json"
    result = json.loads(path.read_text())
    assert bringup_gates.main(args) == 0

    csv.write_text(_BAND_HEADER + _OK_ROW + "MODEL_ONLY,,,,,\n")
    assert bringup_gates.main(args) == 1

    csv.write_text(
        _BAND_HEADER + "OK,0.9000,over-predicted,1.0000,match,over-predicted\n"
    )
    assert bringup_gates.main(args) == 1

    csv.write_text(
        _BAND_HEADER + "OK,1.3000,deviation,1.0000,match,deviation-suspect\n"
    )
    assert bringup_gates.main(args) == 1

    csv.write_text(_BAND_HEADER + _OK_ROW)
    path.write_text(json.dumps({**result, "commit": None}))
    assert bringup_gates.main(args) == 1

    path.write_text(json.dumps({**result, "outcome": "error", "reason": "cmd failed"}))
    assert bringup_gates.main(args) == 1

    path.write_text(json.dumps(result))
    (op_dir / "perf_model_pr.md").write_text("## Operator\n")
    assert bringup_gates.main(args) == 1

    (op_dir / "perf_model_pr.md").unlink()
    assert bringup_gates.main(args) == 1

    issue = op_dir / "perf_model_issue.md"
    issue.write_text(_ISSUE)
    path.write_text(json.dumps({**result, "outcome": "blocked"}))
    assert bringup_gates.main(args) == 1
    path.write_text(json.dumps({**result, "outcome": "blocked", "reason": "r"}))
    assert bringup_gates.main(args) == 0
    issue.write_text("## Validation\n")
    assert bringup_gates.main(args) == 1
    issue.write_text("## Operator\n")
    assert bringup_gates.main(args) == 1
    issue.write_text(_ISSUE + "\n| Shape | Time (us) |\n|---|---|\n| s | 1 |\n")
    assert bringup_gates.main(args) == 1
    issue.write_text("## Operator\n")
    path.write_text(
        json.dumps(
            {**result, "outcome": "blocked", "reason": "r", "validation_csv": None}
        )
    )
    assert bringup_gates.main(args) == 0

    path.write_text("{")
    assert bringup_gates.main(args) == 1
    assert "invalid JSON" in _gate(tmp_path, "op_result_op_a")["problems"][0]

    path.unlink()
    assert bringup_gates.main(args) == 2


def test_op_result_needs_a_real_validation_table(tmp_path):
    op_dir, args = _modeled_op(tmp_path, _OK_ROW)
    draft = op_dir / "perf_model_pr.md"
    header = bringup_gates._template_headers()[0]
    draft.write_text(f"## Validation\n\nThe header is {header} here.\n")
    assert bringup_gates.main(args) == 1
    table = _DRAFT[_DRAFT.index(header) :]
    draft.write_text(f"## Validation\n\n```\n{table}```\n")
    assert bringup_gates.main(args) == 1
    (op_dir / "validation.csv").unlink()
    draft.write_text(_DRAFT)
    assert bringup_gates.main(args) == 1


@pytest.mark.parametrize("config", [None, "model: [unclosed\n", ""])
def test_op_result_bad_config_is_input_error(tmp_path, config):
    _, args = _modeled_op(tmp_path, _OK_ROW)
    path = tmp_path / "cache" / "bringup_config.yaml"
    if config is None:
        path.unlink()
    else:
        path.write_text(config)
    assert bringup_gates.main(args) == 2


def test_op_result_rejects_leaked_names(tmp_path):
    op_dir, args = _modeled_op(tmp_path, _OK_ROW)
    assert bringup_gates.main(args) == 0
    for leak in ("model 7b", "/srv/workdir_a/x", "gfx950", list_platforms()[0]):
        (op_dir / "perf_model_pr.md").write_text(_DRAFT + leak + "\n")
        assert bringup_gates.main(args) == 1


def test_op_result_requires_explained_deviations(tmp_path):
    op_dir, args = _modeled_op(
        tmp_path, "OK,1.0000,match,2.3204,deviation,deviation-explained\n"
    )
    draft = op_dir / "perf_model_pr.md"
    draft.write_text(_DRAFT + "| 2.3204 |\n")
    assert bringup_gates.main(args) == 1
    draft.write_text(_DRAFT + "\n## Ratio Deviations\n\n- Bytes Ratio 2.3204: a.\n")
    assert bringup_gates.main(args) == 0
    (op_dir / "validation.csv").write_text("run_status,match_status\nOK,match\n")
    assert bringup_gates.main(args) == 1
    assert _gate(tmp_path, "op_result_op_a")["problems"] == [
        "validation CSV has no band columns"
    ]


def test_analysis_result_gates_outcome_and_outputs(tmp_path):
    analysis = tmp_path / "analysis"
    for name in bringup_gates.ANALYSIS_OUTPUTS:
        (analysis / name).parent.mkdir(parents=True, exist_ok=True)
        (analysis / name).write_text("x")
    args = ["analysis-result", "--bringup-dir", str(tmp_path)]
    args += ["--analysis-dir", str(analysis)]
    assert bringup_gates.main(args) == 1

    result = analysis / "result.json"
    result.write_text(json.dumps({"outcome": "stop", "reason": "reason a"}))
    assert bringup_gates.main(args) == 1
    assert "reason a" in _gate(tmp_path, "analysis")["problems"][0]

    result.write_text("[]")
    assert bringup_gates.main(args) == 1
    assert _gate(tmp_path, "analysis")["problems"] == ["result.json is not an object"]

    result.write_text(json.dumps({"outcome": "analyzed"}))
    assert bringup_gates.main(args) == 0
    bringup_gates.main(["summary", "--bringup-dir", str(tmp_path)])
    assert "| analysis | pass |" in (tmp_path / "bringup_summary.md").read_text()

    (analysis / "perf_report.log").write_text("")
    assert bringup_gates.main(args) == 1


def test_op_result_categorized_needs_commit_and_no_validation(tmp_path):
    op_dir, args = _modeled_op(tmp_path, _OK_ROW)
    path = op_dir / "result.json"
    result = {**json.loads(path.read_text()), "outcome": "categorized"}
    result["validation_csv"] = None
    (op_dir / "perf_model_pr.md").write_text("## Operator\n\nOp a.\n\n## Changes\n")
    path.write_text(json.dumps(result))
    assert bringup_gates.main(args) == 0

    path.write_text(json.dumps({**result, "commit": None}))
    assert bringup_gates.main(args) == 1

    path.write_text(json.dumps(result))
    (op_dir / "perf_model_pr.md").write_text(_DRAFT)
    assert bringup_gates.main(args) == 1


@pytest.mark.parametrize("category,code", [("GEMM", 0), ("other", 1)])
def test_coverage_needs_categorized_op_to_have_a_category(tmp_path, category, code):
    after = _write_csv(tmp_path / "after.csv", [f"op_a,{category},,,10,False,,\n"])
    op_dir = tmp_path / "perf_models" / "op_a"
    op_dir.mkdir(parents=True)
    (op_dir / "result.json").write_text(
        json.dumps(
            {
                "op": "op_a",
                "op_slug": "op_a",
                "outcome": "categorized",
                "validation_csv": None,
            }
        )
    )
    _accept_op(tmp_path, "op_a")
    args = ["coverage", "--bringup-dir", str(tmp_path)]
    args += ["--before", str(after), "--after", str(after)]
    assert bringup_gates.main(args) == code


def test_excluded_and_skipped_ops(tmp_path):
    op_dir, args = _modeled_op(tmp_path, _OK_ROW)
    (op_dir / "perf_model_pr.md").write_text("## Operator\n")
    assert bringup_gates.main(args) == 1
    gate = _gate(tmp_path, "op_result_op_a")
    assert gate["status"] == "fail" and gate["problems"]

    after = _write_csv(tmp_path / "after.csv", ["op_a,GEMM,,,10,False,,\n"])
    cov = ["coverage", "--bringup-dir", str(tmp_path)]
    assert (
        bringup_gates.main(cov + ["--before", str(after), "--after", str(after)]) == 0
    )

    _write_gaps(tmp_path, "op_a", "op_b")
    bringup_gates.main(["summary", "--bringup-dir", str(tmp_path)])
    text = (tmp_path / "bringup_summary.md").read_text()
    assert f"| op_a | excluded | {gate['problems'][0]} |" in text
    assert "| op_b | skipped | not launched |" in text
    assert "Publish with" not in text


def test_prepare_links_replay_rank_captures(tmp_path):
    bringup = tmp_path / "bringup"
    profile = tmp_path / "profile"
    args = ["prepare", "--bringup-dir", str(bringup), "--profile-dir", str(profile)]
    args += ["--platform", "platform_a"]
    assert bringup_gates.main(args) == 1
    assert "no torch_trace" in _gate(bringup, "prepare")["error"]

    bench = profile / "bench_a"
    for rel in (
        "capture_traces/cap_rank0.json.gz",
        "capture_traces/cap_rank1.json.gz",
        "graph_capture_profile/cap_all.json",
        "other.json",
    ):
        (bench / "torch_trace" / rel).parent.mkdir(parents=True, exist_ok=True)
        (bench / "torch_trace" / rel).write_text("{}")
    for rank in (0, 1):
        (bench / f"mixed_steady_state_rank{rank}.json.gz").write_text("{}")
    assert bringup_gates.main(args) == 0
    gate = _gate(bringup, "prepare")
    folder = bringup / "capture_rank1"
    assert sorted(p.name for p in folder.iterdir()) == [
        "cap_all.json",
        "cap_rank1.json.gz",
    ]
    assert gate["replay_rank"] == 1 and gate["capture_folder"] == str(folder)
    assert gate["bench_dir"] == str(bench) and gate["platform"] == "platform_a"
    assert gate["mixed_split"].endswith("mixed_steady_state_rank1.json.gz")


def test_cases_writes_json_with_attention_params(tmp_path, capsys):
    _write_csv(
        tmp_path / "unified_perf_summary.csv",
        ['op_a,GEMM,"((4, 8),)",bf16,60,True,,2\n'],
    )
    (tmp_path / "InferenceAttention_fwd.csv").write_text(
        'name,Input Dims_first,param: causal\nop_a,"((4, 8),)",True\n'
    )
    out = tmp_path / "cases.json"
    args = ["cases", "--report-dir", str(tmp_path), "--out", str(out)]
    assert bringup_gates.main(args) == 0
    assert json.loads(out.read_text())[0]["attn_params"] == {"causal": "True"}
    assert "1 case(s)" in capsys.readouterr().out


@pytest.mark.parametrize(
    "extra", [["--step", "4"], ["--reason", "reason a"]], ids=["step", "reason"]
)
def test_summary_flags_go_together(tmp_path, extra):
    with pytest.raises(SystemExit) as exc:
        bringup_gates.main(["summary", "--bringup-dir", str(tmp_path), *extra])
    assert exc.value.code == 2


def _model_config(bringup, framework, envs):
    profile = bringup / "profile"
    profile.mkdir()
    benchmark = {"framework": framework, "model": "org/Model-A", "envs": envs}
    (profile / "benchmark.yaml").write_text(yaml.safe_dump({"benchmark": benchmark}))
    args = [
        "model-config",
        "--bringup-dir",
        str(bringup),
        "--profile-dir",
        str(profile),
    ]
    rc = bringup_gates.main(args)
    return rc, json.loads((bringup / "cache" / "model_config.json").read_text())


def test_model_config_resolves(tmp_path, monkeypatch):
    paper = {"url": "https://example.com/paper.pdf", "title": None}
    monkeypatch.setattr(bringup_gates, "_find_paper", lambda model_id: paper)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    hf_config = {"model_type": "example_moe", "hidden_size": 64, "num_experts": 4}
    configs = {str(checkpoint): (hf_config, "config.json")}
    monkeypatch.setattr(bringup_gates, "_load_checkpoint_config", configs.__getitem__)
    envs = {
        "tp": 8,
        "CONC": 64,
        "EXTRA_SGLANG_ARGS": "--dp 8 --enable-dp-attention",
        "MODEL_PATH": str(checkpoint),
    }
    rc, out = _model_config(tmp_path, "sglang", envs)
    assert rc == 0 and _gate(tmp_path, "model_config")["status"] == "pass"
    assert out["model_id"] == "org/Model-A"
    assert out["launch"] == {
        "TP": 8,
        "EXTRA_SGLANG_ARGS": "--dp 8 --enable-dp-attention",
    }
    assert out["config"] == hf_config
    assert out["paper"] == paper


@pytest.mark.parametrize(
    "benchmark_yaml, message",
    [
        (None, "missing"),
        ("benchmark: [unclosed\n", "malformed"),
        ("benchmark: {}\n", "no benchmark.model"),
    ],
)
def test_model_config_bad_benchmark_yaml(tmp_path, capsys, benchmark_yaml, message):
    profile = tmp_path / "profile"
    profile.mkdir()
    if benchmark_yaml is not None:
        (profile / "benchmark.yaml").write_text(benchmark_yaml)
    args = ["model-config", "--bringup-dir", str(tmp_path)]
    assert bringup_gates.main(args + ["--profile-dir", str(profile)]) == 2
    assert message in capsys.readouterr().err


def test_load_checkpoint_config_defers_to_config_resolve(monkeypatch):
    resolve = types.ModuleType("config_resolve")
    resolve.load_checkpoint_config = lambda checkpoint: ({"hidden_size": 64}, "label")
    monkeypatch.setitem(sys.modules, "TraceLens.ModelUtils", types.ModuleType("m"))
    monkeypatch.setitem(sys.modules, "TraceLens.ModelUtils.config_resolve", resolve)
    assert bringup_gates._load_checkpoint_config("ckpt") == (
        {"hidden_size": 64},
        "label",
    )


def test_model_config_unresolved(tmp_path, monkeypatch):
    def offline(checkpoint):
        raise FileNotFoundError("no cached config")

    monkeypatch.setattr(bringup_gates, "_load_checkpoint_config", offline)
    monkeypatch.setattr(bringup_gates, "_find_paper", lambda model_id: None)
    rc, out = _model_config(tmp_path, "sglang", {"TP": 4})
    assert rc == 1
    assert out["config"] is None
    assert out["launch"] == {"TP": 4}
    assert _gate(tmp_path, "model_config") == {
        "status": "fail",
        "error": "no cached config",
    }


@pytest.mark.parametrize(
    "base_card, expected",
    [
        (
            "[report](https://example.com/fam.pdf) "
            "[paper](https://arxiv.org/pdf/2402.00002.pdf)",
            {
                "url": "https://arxiv.org/abs/2402.00002",
                "title": "Fam Technical Report",
            },
        ),
        (
            "[report](https://example.com/fam.pdf)",
            {"url": "https://example.com/fam.pdf", "title": None},
        ),
    ],
)
def test_find_paper(monkeypatch, base_card, expected):
    hub = "https://huggingface.co/"
    pages = {
        hub
        + "org/Fam-1-Quant/raw/main/README.md": (
            "---\nbase_model: org/Fam-1\n---\nhttps://arxiv.org/abs/2401.00001v2"
        ),
        hub + "api/papers/2401.00001": '{"title": "Other Method"}',
        hub + "org/Fam-1/raw/main/README.md": base_card,
        hub + "api/papers/2402.00002": '{"title": "Fam Technical Report"}',
    }

    def urlopen(url, timeout):
        if url not in pages:
            raise urllib.error.HTTPError(url, 404, "not found", {}, None)
        return io.BytesIO(pages[url].encode())

    monkeypatch.setattr(bringup_gates.urllib.request, "urlopen", urlopen)
    assert bringup_gates._find_paper("org/Fam-1-Quant") == expected


def test_find_paper_times_out(monkeypatch):
    def urlopen(url, timeout):
        time.sleep(0.5)
        return io.BytesIO(b"[paper](https://example.com/fam.pdf)")

    monkeypatch.setattr(bringup_gates.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(bringup_gates, "PAPER_BUDGET_S", 0.05)
    assert bringup_gates._find_paper("org/Fam-1") is None


@pytest.mark.parametrize(
    "error",
    [
        urllib.error.URLError("offline"),
        urllib.error.HTTPError("url", 404, "not found", {}, None),
    ],
    ids=["unreachable", "no_card"],
)
def test_find_paper_none_when_lookup_fails(monkeypatch, error):
    def urlopen(url, timeout):
        raise error

    monkeypatch.setattr(bringup_gates.urllib.request, "urlopen", urlopen)
    assert bringup_gates._find_paper("org/Fam-1") is None


@pytest.fixture
def perf_model_map(monkeypatch):
    mapping = {"aiter::gemm_a": object, "gemm_b": object}
    module = types.ModuleType("torch_op_mapping")
    module.op_to_perf_model_class_map = mapping
    monkeypatch.setitem(sys.modules, _MAPPING_MODULE, module)
    return mapping


@pytest.fixture
def aiter_repo(tmp_path):
    ops = tmp_path / "aiter" / "ops"
    ops.mkdir(parents=True)
    (ops / "gemm.py").write_text(
        '@compile_ops("aiter::gemm_a")\n'
        "def gemm_a(x):\n"
        "    return gemm_a_kernel(x)\n\n"
        'torch.library.custom_op(op_name="aiter::gemm_b")\n'
        'torch.library.custom_op("other::foreign")\n'
    )
    (ops / "test_gemm.py").write_text("")
    (ops / "bench_gemm.py").write_text("")
    (tmp_path / "aiter" / "norm").mkdir()
    (tmp_path / "aiter" / "norm" / "norm.py").write_text(
        'torch.library.define("norm_a", "(Tensor x) -> Tensor")\n'
    )
    (tmp_path / "aiter" / "build").mkdir()
    (tmp_path / "aiter" / "build" / "ghost.py").write_text(
        '@compile_ops("aiter::ghost")\n'
    )
    (tmp_path / "aiter" / "dangling.py").symlink_to(tmp_path / "absent.py")
    return tmp_path


def test_scan_lists_registered_ops(aiter_repo, capsys):
    out = aiter_repo / "out" / "ops.json"
    args = ["--repo-root", str(aiter_repo), "--output-json", str(out)]
    args += ["--vllm", str(aiter_repo / "absent"), "--name-max", "12"]
    assert scan_framework_ops.main(args) == 0
    captured = capsys.readouterr()
    assert "Warning: vllm dir not found" in captured.err
    assert "aiter::ge..." in captured.out
    ops = {op["op_name"]: op for op in json.loads(out.read_text())}
    assert sorted(ops) == ["aiter::gemm_a", "aiter::gemm_b", "aiter::norm_a"]
    assert ops["aiter::gemm_a"]["kernel_hints"] == ["gemm_a_kernel"]
    assert ops["aiter::gemm_a"]["sibling_benchmarks"] == [
        "bench_gemm.py",
        "test_gemm.py",
    ]
    assert ops["aiter::norm_a"]["sibling_benchmarks"] == []

    assert scan_framework_ops.main(args + ["--filter", "gemm"]) == 0
    assert sorted(op["op_name"] for op in json.loads(out.read_text())) == [
        "aiter::gemm_a",
        "aiter::gemm_b",
    ]


def test_scan_tags_and_filters_by_perf_model(aiter_repo, capsys, perf_model_map):
    args = ["--aiter", str(aiter_repo / "aiter"), "--check-mapping"]
    assert scan_framework_ops.main(args) == 0
    out = capsys.readouterr().out
    assert "aiter::gemm_a [HAS MODEL]" in out and "aiter::gemm_b [HAS MODEL]" in out
    assert "aiter::norm_a [NO MODEL]" in out

    assert scan_framework_ops.main(args + ["--no-model-only"]) == 0
    out = capsys.readouterr().out
    assert "HAS MODEL" not in out and "Total ops found: 1" in out


def test_scan_reports_no_ops(tmp_path, capsys):
    assert scan_framework_ops.main(["--repo-root", str(tmp_path)]) == 0
    assert "No ops found" in capsys.readouterr().err


def test_builtin_perf_model_map(monkeypatch, capsys, perf_model_map):
    assert scan_framework_ops.builtin_perf_model_map() is perf_model_map
    monkeypatch.setitem(sys.modules, _MAPPING_MODULE, None)
    assert scan_framework_ops.builtin_perf_model_map() is None
    assert "Mapping check skipped" in capsys.readouterr().err


def test_perf_ops_other_bucket_report(tmp_path, capsys, perf_model_map):
    csv = tmp_path / "summary.csv"
    csv.write_text(_PERF_CSV, encoding="utf-8")
    extension = tmp_path / "extension.py"
    args = [str(csv), "--fraction", "0.9", "--name-max", "8", "--also-global-pareto"]
    args += ["--emit-extension", "--extension-out", str(extension), "--check-mapping"]
    assert unified_perf_ops.main(args) == 0
    out = capsys.readouterr().out
    assert "Rows (other): 3" in out
    assert "2 row(s), covering 80.000 µs (94.12% of other)" in out
    assert "Definition B" in out and "3 row(s), sum 85.000 µs" in out
    assert "name: aiter..." in out and "kernel hint: kernel_a" in out
    assert out.index("[2 rows] vLLM") < out.index("[1 rows] AITER")
    assert "'aiter::gemm_a'," in extension.read_text()
    assert "True   aiter::gemm_a" in out

    assert unified_perf_ops.main([str(csv), "--other-value", "absent"]) == 0
    assert "Definition A: N/A" in capsys.readouterr().out


def test_perf_ops_top_ops_report(tmp_path, capsys):
    csv = tmp_path / "summary.csv"
    csv.write_text(_PERF_CSV, encoding="utf-8")
    args = [str(csv), "--mode", "top-ops", "--threshold", "0.1", "--name-max", "8"]
    assert unified_perf_ops.main(args) == 0
    out = capsys.readouterr().out
    assert "aiter..." in out and "stack: vllm/x.py aiter/y.py" in out
    assert "Total candidate ops: 1" in out

    assert unified_perf_ops.main(args + ["--include-covered"]) == 0
    assert "Total candidate ops: 2" in capsys.readouterr().out

    args[args.index("0.1")] = "0.9"
    assert unified_perf_ops.main(args) == 0
    assert "No ops found above threshold" in capsys.readouterr().out


def test_perf_ops_reject_bad_inputs(tmp_path, capsys):
    assert unified_perf_ops.main([str(tmp_path / "absent.csv")]) == 1
    assert "Not a file" in capsys.readouterr().err
    csv = tmp_path / "summary.csv"
    csv.write_text("name,GFLOPS\nop_a,1\n")
    with pytest.raises(SystemExit, match="no runtime column"):
        load_rows(csv)
    csv.write_text("name,total_duration_us\nop_a,n/a\n")
    assert load_rows(csv)[0][0]["_us"] == 0.0
