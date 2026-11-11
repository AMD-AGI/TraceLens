###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import json

from TraceLens.Agent.ModelBringup.utils import bringup_gates
from TraceLens.Agent.ModelBringup.utils.unified_perf_ops import (
    load_rows,
    missing_perf_model_share,
    validation_cases,
)

_HEADER = "name,op category,Input Dims,Input type,Kernel Time (µs)_sum,has_perf_model,kernel_details_summary,GFLOPS\n"


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_HEADER + "".join(rows), encoding="utf-8")
    return path


def _gate(bringup, name):
    return json.loads((bringup / "gates" / f"{name}.json").read_text())


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
        ],
    )
    rc = bringup_gates.main(
        ["gaps", "--bringup-dir", str(tmp_path), "--unified-csv", str(csv)]
    )
    gaps = json.loads((tmp_path / "gaps.json").read_text())
    assert rc == 0
    assert [o["op_slug"] for o in gaps["ops"]] == ["ns_op_x", "ns_op_x_2"]
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
                "outcome": "pr_draft",
                "draft": "pr.md",
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


def test_op_result_gates_pr_draft(tmp_path):
    (tmp_path / "pr.md").write_text("title\n")
    csv = tmp_path / "validation.csv"
    result = {
        "op": "op_a",
        "op_slug": "op_a",
        "outcome": "pr_draft",
        "commit": "abc123",
        "draft": str(tmp_path / "pr.md"),
        "validation_csv": str(csv),
    }
    path = tmp_path / "result.json"
    args = ["op-result", "--result", str(path)]
    assert bringup_gates.main(args) == 2

    csv.write_text("status\nOK\n")
    path.write_text(json.dumps(result))
    assert bringup_gates.main(args) == 0

    csv.write_text("status\nOK\nMODEL_ONLY\n")
    assert bringup_gates.main(args) == 1

    csv.write_text("status\nOK\n")
    path.write_text(json.dumps({**result, "commit": None}))
    assert bringup_gates.main(args) == 1

    path.write_text(json.dumps({**result, "outcome": "error", "reason": "cmd failed"}))
    assert bringup_gates.main(args) == 1

    path.write_text(json.dumps(result))
    (tmp_path / "pr.md").unlink()
    assert bringup_gates.main(args) == 1
