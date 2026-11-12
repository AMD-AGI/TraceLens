###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import json

import pytest

from TraceLens.Agent.Analysis.utils.arch_utils import list_platforms
from TraceLens.Agent.ModelBringup.utils import bringup_gates
from TraceLens.Agent.ModelBringup.utils.unified_perf_ops import (
    load_rows,
    missing_perf_model_share,
    validation_cases,
)
from TraceLens.Agent.PerfModel.ratio_bands import ratio_band

_HEADER = "name,op category,Input Dims,Input type,Kernel Time (µs)_sum,has_perf_model,kernel_details_summary,GFLOPS\n"
_BAND_HEADER = "status,flops_ratio,flops_band,bytes_ratio,bytes_band\n"
_OK_ROW = "OK,1.0000,match,1.0000,match\n"
_DRAFT = f"## Operator\n\nOp a, data: None.\n\n## Validation\n\n{bringup_gates.VALIDATION_HEADER}\n"


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_HEADER + "".join(rows), encoding="utf-8")
    return path


def _gate(bringup, name):
    return json.loads((bringup / "gates" / f"{name}.json").read_text())


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
        ("flops_ratio", 0.94, 0.95, 1.05, 1.06),
        ("bytes_ratio", 0.89, 0.9, 1.2, 1.21),
    ],
)
def test_ratio_band(column, below, low, high, above):
    assert ratio_band(column, None) == ""
    assert ratio_band(column, low) == ratio_band(column, high) == "match"
    assert ratio_band(column, below) == ratio_band(column, above) == "deviation"


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
        "DIAG:perf_model:3i_CAPTURE_RANK_MISMATCH,perf_model,m,e,r,\n"
        "DIAG:trace_quality:2j_RUNTIME_INSTABILITY,trace_quality,m,e,r,\n"
    )
    args = ["report-health", "--phase", "report", "--bringup-dir", str(tmp_path)]
    args += ["--triage-dir", str(tmp_path)]
    assert bringup_gates.main(args) == 1
    assert _gate(tmp_path, "report_health")["other"] == [
        "DIAG:trace_quality:2j_RUNTIME_INSTABILITY"
    ]
    assert bringup_gates.main(args + ["--accept", "CAPTURE_RANK_MISMATCH"]) == 0
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
    (op_dir / "validation.csv").write_text("op,status\nop_a,OK\nop_a,MODEL_ONLY\n")
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

    csv.write_text(_BAND_HEADER + _OK_ROW + "MODEL_ONLY,,,,\n")
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
    issue.write_text(_DRAFT)
    path.write_text(json.dumps({**result, "outcome": "blocked"}))
    assert bringup_gates.main(args) == 1
    path.write_text(json.dumps({**result, "outcome": "blocked", "reason": "r"}))
    assert bringup_gates.main(args) == 0
    issue.write_text("## Validation\n")
    assert bringup_gates.main(args) == 1

    path.unlink()
    assert bringup_gates.main(args) == 2


def test_op_result_rejects_leaked_names(tmp_path):
    op_dir, args = _modeled_op(tmp_path, _OK_ROW)
    assert bringup_gates.main(args) == 0
    for leak in ("model 7b", "/srv/workdir_a/x", "gfx950", list_platforms()[0]):
        (op_dir / "perf_model_pr.md").write_text(_DRAFT + leak + "\n")
        assert bringup_gates.main(args) == 1


def test_op_result_requires_explained_deviations(tmp_path):
    op_dir, args = _modeled_op(tmp_path, "OK,1.0000,match,2.3204,deviation\n")
    draft = op_dir / "perf_model_pr.md"
    draft.write_text(_DRAFT + "| 2.3204 |\n")
    assert bringup_gates.main(args) == 1
    draft.write_text(_DRAFT + "\n## Ratio Deviations\n\n- Bytes Ratio 2.3204: a.\n")
    assert bringup_gates.main(args) == 0


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

    result.write_text(json.dumps({"outcome": "analyzed"}))
    assert bringup_gates.main(args) == 0
    bringup_gates.main(["summary", "--bringup-dir", str(tmp_path)])
    assert "| analysis | pass |" in (tmp_path / "bringup_summary.md").read_text()

    (analysis / "perf_report.log").write_text("")
    assert bringup_gates.main(args) == 1
