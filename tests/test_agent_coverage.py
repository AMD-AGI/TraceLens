###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Additional CPU-only coverage for TraceLens Agent Analysis utils.

Targets validation_utils, orchestrator_prepare, plot_utils, report_utils, and
classify_kernels CLI paths using tmp_path fixtures and mocks.
"""

import json
import os
import subprocess
import sys

import pandas as pd
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ANALYSIS_DIR = os.path.join(REPO_ROOT, "TraceLens", "Agent", "Analysis")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, ANALYSIS_DIR)

from TraceLens.Agent.Analysis.utils import plot_utils
from TraceLens.Agent.Analysis.utils.classify_kernels import (
    classify_all,
    main as classify_main,
)
from TraceLens.Agent.Analysis.utils.orchestrator_prepare import (
    _extract_comparative_fusion_candidates,
    _extract_standalone_fusion_candidates,
)
from TraceLens.Agent.Analysis.utils.report_utils import (
    load_findings,
    load_manifest,
    load_manifest_categories,
    prepare_model_identification_data,
    generate_priority_data,
)
from TraceLens.Agent.Analysis.utils.validation_utils import (
    MarkerValidator,
    _category_findings_empty,
    _check_coverage,
    _check_time_sanity,
    _extract_detailed_analysis_subsection,
    _load_valid_args,
    _metrics_json_for_findings,
    _scan_args_cells,
    _validate_report_args_column,
    _validate_report_comparison_scope_diffs,
    _validate_report_reasoning_candidates,
    validate_findings_file,
    validate_report,
    validate_subagent_outputs,
)

# ----- Fixtures -----


def _write(path, text):
    with open(path, "w") as f:
        f.write(text)


def _valid_compute_findings(rank=1, row=None):
    row = row or (
        "| aten::mm | M=2,N=3 | path/to/launch | mm_kernel | "
        "1.0 | 10 | 5 | 2000 | 50% | compute |"
    )
    header = (
        "| Operation | Args | Kernel Path | Kernel Name | Time (ms) | %E2E | "
        "Count | FLOPS/Byte | Efficiency | Bound |"
    )
    sep = "|---|---|---|---|---|---|---|---|---|---|"
    return (
        "## Recommendations\n\n"
        f"### P{rank}: Optimize gemm\n"
        "**Insight**\n slow\n**Action**\n tune\n**Impact**\n high\n"
        "<!-- impact-begin kind=p_item low=1.0 mid=2.0 high=3.0 -->\n"
        "<!-- impact-end -->\n\n"
        "## Detailed Analysis\n\n"
        f"<!-- reasoning-candidate tier=compute rank={rank} -->\n"
        "**Data:**\n\n"
        f"{header}\n{sep}\n{row}\n"
        "<!-- impact-begin kind=detail_estimate low=1.0 high=3.0 -->\n"
        "<!-- impact-end -->\n"
    )


def _valid_system_findings():
    return (
        "## Recommendations\n\n"
        "### P1: Fix idle\n"
        "**Insight**\n idle high\n**Action**\n overlap\n**Impact**\n medium\n"
        "<!-- impact-begin kind=p_item low=1.0 mid=2.0 high=3.0 -->\n"
        "<!-- impact-end -->\n\n"
        "## Detailed Analysis\n\n"
        "<!-- reasoning-candidate tier=system rank=1 -->\n"
        "System detail block.\n"
        "<!-- impact-begin kind=detail_estimate low=1.0 high=3.0 -->\n"
        "<!-- impact-end -->\n"
    )


@pytest.fixture
def output_dir_with_manifest(tmp_path):
    out = tmp_path / "analysis_output"
    cat_data = out / "category_data"
    cat_data.mkdir(parents=True)
    manifest = {
        "platform": "MI300X",
        "gpu_utilization": {
            "total_time_ms": 1000.0,
            "computation_time_percent": 80.0,
        },
        "categories": [
            {
                "name": "gemm",
                "display_name": "GEMM",
                "tier": "compute_kernel",
                "gpu_kernel_time_ms": 600.0,
            },
            {
                "name": "sdpa_fwd",
                "display_name": "SDPA Fwd",
                "tier": "compute_kernel",
                "gpu_kernel_time_ms": 200.0,
            },
            {"name": "cpu_idle", "tier": "system", "gpu_kernel_time_ms": 0},
        ],
    }
    (cat_data / "category_manifest.json").write_text(json.dumps(manifest, indent=2))
    return str(out)


@pytest.fixture
def perf_csv_dir(tmp_path):
    csv_dir = tmp_path / "perf_report_csvs"
    csv_dir.mkdir()
    pd.DataFrame(
        {
            "name": ["aten::mm", "aten::add"],
            "Input type": ["['c10::BFloat16']", "['c10::BFloat16']"],
            "Input Dims": ["[[2,3]]", "[[4,4]]"],
            "call_stack_full": [
                "['nn.Module: LlamaMLP', 'myfile.py']",
                "['nn.Module: Attention', 'other.py']",
            ],
        }
    ).to_csv(csv_dir / "unified_perf_summary.csv", index=False)
    return str(csv_dir)


# ----- validation_utils: validate_findings_file -----


def test_validate_findings_file_compute_pass(tmp_path):
    content = _valid_compute_findings()
    fp = tmp_path / "gemm_findings.md"
    _write(str(fp), content)
    (tmp_path / "category_data").mkdir()
    (tmp_path / "category_data" / "gemm_metrics.json").write_text(
        json.dumps(
            {
                "operations": [
                    {
                        "args": "M=2,N=3",
                        "launcher_path": "path/to/launch",
                        "kernel_name_trunc": "mm_kernel",
                    }
                ]
            }
        )
    )
    passed, errors = validate_findings_file(str(fp), "compute")
    assert passed, errors
    assert errors == []


def test_validate_findings_file_missing_file(tmp_path):
    passed, errors = validate_findings_file(str(tmp_path / "nope.md"), "compute")
    assert not passed
    assert any("File not found" in e for e in errors)


def test_validate_findings_file_wrong_header_order(tmp_path):
    content = "## Detailed Analysis\n\n## Recommendations\n"
    fp = tmp_path / "gemm_findings.md"
    _write(str(fp), content)
    passed, errors = validate_findings_file(str(fp), "compute")
    assert not passed
    assert any("Recommendations must appear before" in e for e in errors)


def test_validate_findings_file_relaxed_empty_category(tmp_path):
    content = (
        "## Recommendations\n\n## Detailed Analysis\n"
        "not quantifiable from trace data\n"
    )
    findings_dir = tmp_path / "category_findings"
    findings_dir.mkdir(parents=True)
    fp = findings_dir / "gemm_findings.md"
    _write(str(fp), content)
    (tmp_path / "category_data").mkdir()
    (tmp_path / "category_data" / "gemm_metrics.json").write_text(
        json.dumps({"category_findings": []})
    )
    passed, errors = validate_findings_file(str(fp), "compute")
    assert passed, errors


def test_validate_findings_file_tier_mismatch(tmp_path):
    content = _valid_compute_findings().replace("tier=compute", "tier=system")
    fp = tmp_path / "gemm_findings.md"
    _write(str(fp), content)
    (tmp_path / "category_data").mkdir()
    (tmp_path / "category_data" / "gemm_metrics.json").write_text("{}")
    passed, errors = validate_findings_file(str(fp), "compute")
    assert not passed
    assert any("tier mismatch" in e for e in errors)


def test_validate_findings_file_p_item_count_mismatch(tmp_path):
    content = _valid_compute_findings()
    content += "\n### P2: Extra item\n"
    fp = tmp_path / "gemm_findings.md"
    _write(str(fp), content)
    (tmp_path / "category_data").mkdir()
    (tmp_path / "category_data" / "gemm_metrics.json").write_text("{}")
    passed, errors = validate_findings_file(str(fp), "compute")
    assert not passed
    assert any("P-item count" in e or "headings but" in e for e in errors)


def test_validate_findings_file_system_pass(tmp_path):
    fp = tmp_path / "cpu_idle_findings.md"
    _write(str(fp), _valid_system_findings())
    passed, errors = validate_findings_file(str(fp), "system")
    assert passed, errors


def test_metrics_json_for_findings_and_empty_category(tmp_path):
    fp = tmp_path / "category_findings" / "gemm_findings.md"
    fp.parent.mkdir(parents=True)
    fp.write_text("x")
    mp = _metrics_json_for_findings(str(fp))
    assert mp.endswith("category_data/gemm_metrics.json")
    assert _category_findings_empty(str(fp)) is False
    (tmp_path / "category_data").mkdir()
    (tmp_path / "category_data" / "gemm_metrics.json").write_text(
        json.dumps({"category_findings": []})
    )
    assert _category_findings_empty(str(fp)) is True


def test_scan_args_cells_and_load_valid_args(tmp_path):
    text = "| Operation | Args | Time |\n" "|---|---|---|\n" "| op | M=2,N=3 | 1 |\n"
    cells = list(_scan_args_cells(text))
    assert cells == [(3, "M=2,N=3")]
    (tmp_path / "gemm_metrics.json").write_text(
        json.dumps({"operations": [{"args": "M=2,N=3"}, {"args": None}]})
    )
    valid = _load_valid_args(str(tmp_path / "gemm_metrics.json"), "/nonexistent.json")
    assert valid == {"M=2,N=3"}


def test_validate_findings_file_missing_recommendations_section(tmp_path):
    fp = tmp_path / "gemm_findings.md"
    _write(str(fp), "## Detailed Analysis\n\nbody\n")
    passed, errors = validate_findings_file(str(fp), "compute")
    assert not passed
    assert any("Missing required section" in e for e in errors)


def test_validate_findings_file_comparative_data_table(tmp_path):
    header = (
        "| Operation | Args (T1) | Kernel Path | Kernel Name | Trace 1 Time (ms) | "
        "Trace 2 Time (ms) | Count (T1/T2) | Difference (ms) | FLOPS/Byte (T1) | Bound (T1) |"
    )
    sep = "|---|" * 10 + "---|"
    row = (
        "| aten::mm | M=2,N=3 | path/to/launch | mm_kernel | 1.0 | 0.8 | 5 | -0.2 | "
        "2000 | compute |"
    )
    content = (
        "## Recommendations\n\n"
        "### P1: Optimize gemm\n"
        "**Insight**\n x\n**Action**\n y\n**Impact**\n z\n"
        "<!-- impact-begin kind=p_item low=1.0 mid=2.0 high=3.0 -->\n"
        "<!-- impact-end -->\n\n"
        "## Detailed Analysis\n\n"
        "<!-- reasoning-candidate tier=compute rank=1 -->\n"
        "**Data:**\n\n"
        f"{header}\n{sep}\n{row}\n"
        "<!-- impact-begin kind=detail_estimate low=1.0 high=3.0 -->\n"
        "<!-- impact-end -->\n"
    )
    findings_dir = tmp_path / "category_findings"
    findings_dir.mkdir()
    fp = findings_dir / "gemm_findings.md"
    _write(str(fp), content)
    (tmp_path / "category_data").mkdir()
    (tmp_path / "category_data" / "gemm_metrics.json").write_text(
        json.dumps(
            {
                "operations": [
                    {
                        "args": "M=2,N=3",
                        "launcher_path": "path/to/launch",
                        "kernel_name_trunc": "mm_kernel",
                    }
                ]
            }
        )
    )
    passed, errors = validate_findings_file(str(fp), "compute", "comparative")
    assert passed, errors


def test_validate_compute_data_kernel_name_mismatch(tmp_path):
    row = (
        "| aten::mm | M=2,N=3 | path/to/launch | WRONG_KERNEL | "
        "1.0 | 10 | 5 | 2000 | 50% | compute |"
    )
    content = _valid_compute_findings(row=row)
    findings_dir = tmp_path / "category_findings"
    findings_dir.mkdir()
    fp = findings_dir / "gemm_findings.md"
    _write(str(fp), content)
    (tmp_path / "category_data").mkdir()
    (tmp_path / "category_data" / "gemm_metrics.json").write_text(
        json.dumps(
            {
                "operations": [
                    {
                        "args": "M=2,N=3",
                        "launcher_path": "path/to/launch",
                        "kernel_name_trunc": "mm_kernel",
                    }
                ]
            }
        )
    )
    passed, errors = validate_findings_file(str(fp), "compute")
    assert not passed
    assert any("Kernel Name cell" in e for e in errors)


def test_check_priority_consistency_unreadable_json(tmp_path):
    from TraceLens.Agent.Analysis.utils.validation_utils import (
        _check_priority_consistency,
    )

    (tmp_path / "priority_data.json").write_text("{not json")
    result = _check_priority_consistency(str(tmp_path), {})
    assert result["status"] == "WARN"
    assert any("unreadable" in m for m in result["messages"])


def test_validate_report_args_column_no_metrics_dir(tmp_path):
    content = "| Operation | Args | Time |\n|---|---|---|\n| op | x | 1 |\n"
    assert _validate_report_args_column(content, str(tmp_path)) == []


def test_extract_detailed_analysis_no_section():
    assert _extract_detailed_analysis_subsection("no detailed section", "### X") is None


def test_load_findings_failed_compute(output_dir_with_manifest):
    out = output_dir_with_manifest
    os.makedirs(os.path.join(out, "category_findings"))
    _write(os.path.join(out, "category_findings", "gemm_findings.md"), "Status: ERROR")
    data = load_findings(out)
    assert data["failed_compute"][0]["category"] == "gemm"


def test_generate_and_embed_plot_priority_data_failure(tmp_path, monkeypatch):
    def boom(_output_dir):
        raise RuntimeError("boom")

    monkeypatch.setattr(plot_utils, "generate_priority_data", boom)
    results = plot_utils.generate_and_embed_plot(str(tmp_path), "Title")
    assert results["plot_data"] is False


# ----- validation_utils: subagent batch checks -----


def test_validate_subagent_outputs_summary(output_dir_with_manifest, tmp_path):
    out = output_dir_with_manifest
    os.makedirs(os.path.join(out, "category_findings"), exist_ok=True)
    os.makedirs(os.path.join(out, "system_findings"), exist_ok=True)
    _write(os.path.join(out, "category_findings", "gemm_findings.md"), "ok")
    _write(os.path.join(out, "system_findings", "cpu_idle_findings.md"), "ok")
    generate_priority_data(out)
    results = validate_subagent_outputs(out)
    assert "time_check" in results
    assert "coverage_check" in results
    assert "priority_check" in results


def test_check_time_sanity_pass_and_warn():
    manifest = {
        "gpu_utilization": {"total_time_ms": 1000.0, "computation_time_percent": 80.0},
        "categories": [
            {"tier": "compute_kernel", "gpu_kernel_time_ms": 790.0},
        ],
    }
    assert _check_time_sanity(manifest)["status"] == "PASS"

    no_comp = {"gpu_utilization": {"total_time_ms": 0}, "categories": []}
    assert _check_time_sanity(no_comp)["status"] == "WARN"

    bad_sum = {
        "gpu_utilization": {"total_time_ms": 1000.0, "computation_time_percent": 80.0},
        "categories": [{"tier": "compute_kernel", "gpu_kernel_time_ms": 100.0}],
    }
    assert _check_time_sanity(bad_sum)["status"] == "WARN"


def test_check_coverage_missing_findings(output_dir_with_manifest):
    result = _check_coverage(
        output_dir_with_manifest,
        json.loads(
            open(
                os.path.join(
                    output_dir_with_manifest,
                    "category_data",
                    "category_manifest.json",
                )
            ).read()
        ),
    )
    assert result["status"] == "WARN"
    assert any("Missing compute findings" in m for m in result["messages"])


# ----- validation_utils: validate_report extras -----


def _full_report(extra_kf_impact=""):
    kf = "## Kernel Fusion Opportunities (Experimental)\n\n"
    if extra_kf_impact:
        kf += extra_kf_impact + "\n"
    return f"""# Analysis Report

## Executive Summary

| Metric | Value |
|--------|-------|
| Total Time | 1000 ms |
| Compute % | 99.8% |
| Idle % | 0.2% |
| Exposed Communication % | 0.05% |
| Top Bottleneck Category | gemm |

## Compute Kernel Optimizations

### P1: Optimize gemm
<!-- impact-begin kind=p_item category=gemm low=85.0 mid=100.0 high=115.0 -->
Detail text for gemm optimization goes here with enough content.
<!-- impact-end -->

## {kf.split('## ')[1]}

## System-Level Optimizations

Top Operations:

<!-- impact-begin kind=top_ops -->

| Op | Time |
|----|------|
| gemm | 100 |

<!-- impact-end -->

## Detailed Analysis

### Compute Kernel Insights

#### P1: gemm detail
<!-- reasoning-candidate tier=compute rank=1 -->
More detail here.

### System-Level Insights

#### P1: idle detail
<!-- reasoning-candidate tier=system rank=1 -->
System detail.

## Appendix

Reference material.
"""


def test_validate_report_empty_and_placeholders(tmp_path):
    _write(str(tmp_path / "analysis.md"), "short")
    passed, missing = validate_report(str(tmp_path))
    assert not passed
    assert any("empty or too short" in m for m in missing)

    content = _full_report().replace(
        "Detail text for gemm optimization goes here with enough content.",
        "Unfilled template uses <Library> placeholder here.",
    )
    _write(str(tmp_path / "analysis.md"), content)
    (tmp_path / "priority_data.json").write_text(
        json.dumps({"findings": [], "priorities": []})
    )
    passed, missing = validate_report(str(tmp_path))
    assert any("Unfilled placeholders" in m for m in missing)


def test_validate_report_missing_metrics_row(tmp_path):
    content = _full_report().replace("| Compute % | 99.8% |", "")
    _write(str(tmp_path / "analysis.md"), content)
    (tmp_path / "priority_data.json").write_text(
        json.dumps(
            {
                "findings": [
                    {
                        "category": "gemm",
                        "global_rank": 1,
                        "impact_score": 100.0,
                        "impact_score_low": 85.0,
                        "impact_score_high": 115.0,
                    }
                ],
                "priorities": [{"rank": 1, "category": "gemm"}],
            }
        )
    )
    passed, missing = validate_report(str(tmp_path))
    assert not passed
    assert any("Missing metrics row" in m for m in missing)


def test_validate_report_no_exec_table(tmp_path):
    content = _full_report()
    exec_start = content.find("## Executive Summary")
    exec_end = content.find("## Compute Kernel Optimizations")
    content = (
        content[:exec_start]
        + "## Executive Summary\n\nNo table here.\n\n"
        + content[exec_end:]
    )
    _write(str(tmp_path / "analysis.md"), content)
    (tmp_path / "priority_data.json").write_text(
        json.dumps({"findings": [], "priorities": []})
    )
    passed, missing = validate_report(str(tmp_path))
    assert any("No metrics table" in m for m in missing)


def test_validate_report_kf_section_absent_skips_scope_check(tmp_path):
    content = _full_report().replace(
        "## Kernel Fusion Opportunities (Experimental)\n\n", ""
    )
    errs = _validate_report_comparison_scope_diffs(content, str(tmp_path), "standalone")
    assert errs == []


def test_validate_report_args_column_mismatch(tmp_path):
    (tmp_path / "category_data").mkdir()
    (tmp_path / "category_data" / "gemm_metrics.json").write_text(
        json.dumps({"operations": [{"args": "M=2,N=3"}]})
    )
    content = _full_report()
    content += "\n| Operation | Args | Time |\n|---|---|---|\n| op | WRONG | 1 |\n"
    _write(str(tmp_path / "analysis.md"), content)
    (tmp_path / "priority_data.json").write_text(
        json.dumps({"findings": [], "priorities": []})
    )
    errors = _validate_report_args_column(content, str(tmp_path))
    assert any("Args cell" in e for e in errors)


def test_validate_report_kf_impact_standalone(tmp_path):
    good = "**Impact**: impact_score: 1.5 (perf-model coverage 2/3 kernels)"
    bad = "**Impact**: impact_score: 1.5"
    content = _full_report(extra_kf_impact=good)
    assert (
        _validate_report_comparison_scope_diffs(content, str(tmp_path), "standalone")
        == []
    )
    content_bad = _full_report(extra_kf_impact=bad)
    errs = _validate_report_comparison_scope_diffs(
        content_bad, str(tmp_path), "standalone"
    )
    assert any("standalone mode" in e for e in errs)


def test_validate_report_kf_impact_comparative(tmp_path):
    good = "**Impact**: impact_score: 1.5"
    bad_paren = "**Impact**: impact_score: 1.5 (perf-model coverage 2/3 kernels)"
    content = _full_report(extra_kf_impact=good)
    assert (
        _validate_report_comparison_scope_diffs(content, str(tmp_path), "comparative")
        == []
    )
    content_bad = _full_report(extra_kf_impact=bad_paren)
    errs = _validate_report_comparison_scope_diffs(
        content_bad, str(tmp_path), "comparative"
    )
    assert any("comparative mode" in e for e in errs)


def test_validate_report_reasoning_candidates_r5(tmp_path):
    content = _full_report()
    content = content.replace(
        "<!-- reasoning-candidate tier=compute rank=1 -->",
        "<!-- missing marker -->",
        1,
    )
    errs = _validate_report_reasoning_candidates(content)
    assert any("R5:" in e for e in errs)


def test_extract_detailed_analysis_subsection():
    content = "## Detailed Analysis\n\n### Compute Kernel Insights\n\n#### P1: x\nbody\n### Other\n"
    sub = _extract_detailed_analysis_subsection(content, "### Compute Kernel Insights")
    assert "#### P1: x" in sub
    assert _extract_detailed_analysis_subsection(content, "### Missing") is None


def test_marker_validator_system_findings_requires_p_item(tmp_path):
    fp = tmp_path / "cpu_idle_findings.md"
    _write(str(fp), "## Recommendations\n\n## Detailed Analysis\n")
    errors = MarkerValidator.check_findings_file(str(fp), "system_findings")
    assert any("missing required kind=p_item" in e for e in errors)


def test_marker_validator_p_item_heading_mismatch(tmp_path):
    text = (
        "### P1: One\n### P2: Two\n"
        "<!-- impact-begin kind=p_item low=1 mid=2 high=3 -->\n"
        "<!-- impact-end -->\n"
    )
    fp = tmp_path / "gemm_findings.md"
    _write(str(fp), text)
    errors = MarkerValidator.check_findings_file(str(fp), "category_findings")
    assert any("headings but" in e for e in errors)


def test_marker_validator_detail_estimate_sentinel(tmp_path):
    text = (
        "<!-- reasoning-candidate tier=compute rank=1 -->\n"
        "not quantifiable from trace data\n"
    )
    fp = tmp_path / "gemm_findings.md"
    _write(str(fp), text)
    errors = MarkerValidator._check_detail_estimate_per_candidate(
        text, "gemm_findings.md"
    )
    assert errors == []


# ----- report_utils -----


def test_load_manifest(output_dir_with_manifest):
    manifest = load_manifest(output_dir_with_manifest)
    assert manifest["platform"] == "MI300X"


def test_prepare_model_identification_standalone(tmp_path, perf_csv_dir):
    out = str(tmp_path)
    os.makedirs(os.path.join(out, "perf_report_csvs"), exist_ok=True)
    src = os.path.join(perf_csv_dir, "unified_perf_summary.csv")
    dst = os.path.join(out, "perf_report_csvs", "unified_perf_summary.csv")
    pd.read_csv(src).to_csv(dst, index=False)
    assert prepare_model_identification_data(out, "standalone") is True
    assert os.path.isfile(os.path.join(out, "metadata", "condensed_op_info.csv"))
    nn = open(os.path.join(out, "metadata", "nn_modules.txt")).read()
    assert "LlamaMLP" in nn


def test_prepare_model_identification_comparative(tmp_path, perf_csv_dir):
    out = str(tmp_path)
    t1 = os.path.join(out, "perf_report_trace1_csvs")
    os.makedirs(t1)
    pd.read_csv(os.path.join(perf_csv_dir, "unified_perf_summary.csv")).to_csv(
        os.path.join(t1, "unified_perf_summary.csv"), index=False
    )
    assert prepare_model_identification_data(out, "comparative") is True


def test_prepare_model_identification_missing_csv(tmp_path):
    assert prepare_model_identification_data(str(tmp_path)) is False


def test_prepare_model_identification_bad_columns(tmp_path):
    csv_dir = tmp_path / "perf_report_csvs"
    csv_dir.mkdir()
    pd.DataFrame({"name": ["op"]}).to_csv(
        csv_dir / "unified_perf_summary.csv", index=False
    )
    assert prepare_model_identification_data(str(tmp_path)) is False


def test_load_manifest_categories(capsys, output_dir_with_manifest):
    data = load_manifest_categories(output_dir_with_manifest)
    assert "gemm" in [c["name"] for c in data["compute_categories"]]
    assert "cpu_idle" in [c["name"] for c in data["system_categories"]]
    captured = capsys.readouterr()
    assert "compute_categories" in captured.out


def test_load_findings(capsys, output_dir_with_manifest):
    out = output_dir_with_manifest
    os.makedirs(os.path.join(out, "category_findings"))
    os.makedirs(os.path.join(out, "system_findings"))
    _write(os.path.join(out, "category_findings", "gemm_findings.md"), "ok")
    _write(
        os.path.join(out, "system_findings", "cpu_idle_findings.md"), "Status: ERROR"
    )
    _write(os.path.join(out, "system_findings", "multi_kernel_findings.md"), "ok")
    manifest = json.loads(
        open(os.path.join(out, "category_data", "category_manifest.json")).read()
    )
    manifest["top_operations"] = [{"name": "aten::mm"}]
    open(os.path.join(out, "category_data", "category_manifest.json"), "w").write(
        json.dumps(manifest)
    )
    data = load_findings(out)
    assert "gemm" in data["compute_findings"]
    assert data["failed_system"][0]["category"] == "cpu_idle"
    assert len(data["top_ops"]) == 1


def test_generate_priority_data_unmodeled_and_heuristic(tmp_path):
    cat_data = tmp_path / "category_data"
    cat_data.mkdir()
    manifest = {
        "gpu_utilization": {
            "total_time_ms": 1000.0,
            "computation_time_percent": 100.0,
        },
        "categories": [
            {
                "name": "gemm",
                "display_name": "GEMM",
                "tier": "compute_kernel",
                "gpu_kernel_time_ms": 100.0,
            },
            {
                "name": "norm",
                "display_name": "Norm",
                "tier": "compute_kernel",
                "gpu_kernel_time_ms": 80.0,
            },
        ],
    }
    (cat_data / "category_manifest.json").write_text(json.dumps(manifest))
    (cat_data / "gemm_metrics.json").write_text(
        json.dumps(
            {
                "status": "OK",
                "category": "gemm",
                "impact_estimates": [],
                "category_findings": [
                    {
                        "rank": 1,
                        "impact_score": 10.0,
                        "impact_score_low": 8.0,
                        "impact_score_high": 12.0,
                        "operation_count": 1,
                        "estimate_method": "heuristic",
                    }
                ],
            }
        )
    )
    out_path = generate_priority_data(str(tmp_path))
    with open(out_path) as f:
        data = json.load(f)
    assert data["baseline_ms"] == 1000.0
    assert any(p["source"] == "manifest_fallback" for p in data["priorities"])
    assert len(data["recommendations"]) == 0


def test_generate_priority_data_exception_fallback(tmp_path):
    out_path = generate_priority_data(str(tmp_path))
    with open(out_path) as f:
        data = json.load(f)
    assert data["baseline_ms"] == 0
    assert data["findings"] == []


# ----- plot_utils -----


def test_generate_perf_plot_success(output_dir_with_manifest):
    ok = plot_utils.generate_perf_plot(
        output_dir_with_manifest, "Test Breakdown", write_base64=True
    )
    assert ok is True
    png = os.path.join(output_dir_with_manifest, "perf_improvement.png")
    b64 = os.path.join(output_dir_with_manifest, "perf_improvement_base64.txt")
    assert os.path.isfile(png)
    assert os.path.isfile(b64)


def test_generate_perf_plot_missing_manifest(tmp_path):
    assert plot_utils.generate_perf_plot(str(tmp_path), "Title") is False


def test_generate_perf_plot_invalid_baseline(tmp_path):
    cat_data = tmp_path / "category_data"
    cat_data.mkdir()
    (cat_data / "category_manifest.json").write_text(
        json.dumps({"gpu_utilization": {"total_time_ms": 0}, "categories": []})
    )
    assert plot_utils.generate_perf_plot(str(tmp_path), "Title") is False


def test_generate_perf_plot_rest_only_segment(tmp_path):
    """No compute_kernel tiers -> plot shows only Non-computing rest segment."""
    cat_data = tmp_path / "category_data"
    cat_data.mkdir()
    (cat_data / "category_manifest.json").write_text(
        json.dumps(
            {
                "gpu_utilization": {"total_time_ms": 100.0},
                "categories": [{"name": "cpu_idle", "tier": "system"}],
            }
        )
    )
    assert plot_utils.generate_perf_plot(str(tmp_path), "Title") is True


def test_embed_plot_in_report(tmp_path):
    _write(str(tmp_path / "analysis.md"), "Before {{PERF_PLOT}} After")
    _write(str(tmp_path / "perf_improvement_base64.txt"), "abc123")
    assert plot_utils.embed_plot_in_report(str(tmp_path)) is True
    text = open(tmp_path / "analysis.md").read()
    assert "data:image/png;base64,abc123" in text


def test_embed_plot_missing_report(tmp_path):
    assert plot_utils.embed_plot_in_report(str(tmp_path)) is False


def test_embed_plot_removes_placeholder_without_b64(tmp_path):
    _write(str(tmp_path / "analysis.md"), "Before {{PERF_PLOT}} After")
    assert plot_utils.embed_plot_in_report(str(tmp_path)) is False
    assert "{{PERF_PLOT}}" not in open(tmp_path / "analysis.md").read()


def test_generate_and_embed_plot(output_dir_with_manifest):
    _write(os.path.join(output_dir_with_manifest, "analysis.md"), "{{PERF_PLOT}}")
    open(
        os.path.join(output_dir_with_manifest, "category_data", "gemm_metrics.json"),
        "w",
    ).write(
        json.dumps(
            {
                "status": "OK",
                "category": "gemm",
                "impact_estimates": [],
                "category_findings": [],
            }
        )
    )
    results = plot_utils.generate_and_embed_plot(output_dir_with_manifest, "Title")
    assert results["plot_data"] is True
    assert results["plot"] is True
    assert results["embed"] is True


def test_short_name():
    assert plot_utils._short_name("gemm") == "Gemm"
    assert plot_utils._short_name("verylongcategory") == "Verylon…"


# ----- classify_kernels CLI -----


def test_classify_all_and_main(tmp_path):
    extracted = {
        "source_file": "trace.json",
        "kernels": [
            {"name": "Cijk_Alik_Bljk", "dur": 100},
            {"name": "rmsnorm2d", "dur": 50},
        ],
    }
    inp = tmp_path / "extracted.json"
    out = tmp_path / "classified.json"
    inp.write_text(json.dumps(extracted))
    classified = classify_all(extracted)
    assert classified[0]["kernel_type"] == "GEMM"
    assert classified[1]["kernel_type"] == "RMSNorm"

    old_argv = sys.argv
    sys.argv = ["classify_kernels", str(inp), "-o", str(out)]
    try:
        classify_main()
    finally:
        sys.argv = old_argv
    result = json.loads(out.read_text())
    assert result["total_kernels"] == 2
    assert "type_summary" in result


def test_classify_kernels_main_stdout(tmp_path, capsys):
    extracted = {"kernels": [{"name": "Cijk_x", "dur": 1}]}
    inp = tmp_path / "extracted.json"
    inp.write_text(json.dumps(extracted))
    old_argv = sys.argv
    sys.argv = ["classify_kernels", str(inp)]
    try:
        classify_main()
    finally:
        sys.argv = old_argv
    assert "classified_kernels" in capsys.readouterr().out


# ----- orchestrator_prepare fusion extraction (mocked tree) -----


class _StubTree:
    def __init__(self, events, uid_map, parent_map=None):
        self.events = events
        self._uid_map = uid_map
        self._parent_map = parent_map or {}

    def get_UID2event(self, uid):
        return self._uid_map[uid]

    def get_parent_event(self, ev):
        return self._parent_map.get(id(ev))


class _StubAnalyzer:
    def __init__(self, tree, unified_events=None):
        self.tree = tree
        self._unified = unified_events or []

    def event_to_category(self, ev):
        return ev.get("_category", "aten")

    def collect_unified_perf_events(self):
        return self._unified


def _kernel_event(uid, name, dur=1000):
    return {"name": name, "dur": dur, "_category": "kernel", "gpu_events": []}


def test_extract_standalone_fusion_candidates(tmp_path):
    k1 = _kernel_event(10, "Cijk_gemm_a")
    k2 = _kernel_event(11, "vectorized_elementwise_kernel add")
    module = {
        "name": "nn.Module: MLP_0",
        "_category": "aten",
        "gpu_events": [10, 11],
        "args": {"Input Dims": "[[2,3]]"},
    }
    uid_map = {10: k1, 11: k2}
    tree = _StubTree([module], uid_map)
    analyzer = _StubAnalyzer(tree)

    csv_dir = tmp_path / "perf_csvs"
    csv_dir.mkdir()
    pd.DataFrame(
        {
            "kernel_details_summary": [
                "[{'name': 'Cijk_gemm_a'}]",
                "[{'name': 'vectorized_elementwise_kernel add'}]",
            ],
            "op category": ["GEMM", "elementwise"],
            "Data Moved (MB)": [10.0, 4.0],
            "perf_params": ["{'M':2,'N':4,'K':3}", "{'shape_in1': [4, 4]}"],
            "Input Dims": ["[[2,3]]", "[[4,4]]"],
        }
    ).to_csv(csv_dir / "unified_perf_summary.csv", index=False)

    cands = _extract_standalone_fusion_candidates(analyzer, tree, str(csv_dir))
    assert isinstance(cands, list)
    assert any(c.get("base_name") == "MLP" for c in cands)


def test_extract_standalone_skips_fused_and_gemm_norm(tmp_path):
    k1 = _kernel_event(10, "Cijk_gemm")
    k2 = _kernel_event(11, "rmsnorm_kernel")
    module = {
        "name": "nn.Module: Block_0",
        "_category": "aten",
        "gpu_events": [10, 11],
    }
    tree = _StubTree([module], {10: k1, 11: k2})
    analyzer = _StubAnalyzer(tree)
    csv_dir = tmp_path / "csv"
    csv_dir.mkdir()
    pd.DataFrame(
        {
            "kernel_details_summary": ["[{'name': 'Cijk_gemm'}]"],
            "op category": ["GEMM"],
            "Data Moved (MB)": [1.0],
            "perf_params": ["{}"],
            "Input Dims": ["[[1,1]]"],
        }
    ).to_csv(csv_dir / "unified_perf_summary.csv", index=False)
    cands = _extract_standalone_fusion_candidates(analyzer, tree, str(csv_dir))
    assert cands == []


def test_extract_comparative_fusion_candidates(tmp_path):
    csv_dir = tmp_path / "trace1_csvs"
    csv_dir.mkdir()
    pd.DataFrame(
        {
            "name": ["Cijk_A", "Cijk_B", "some_op"],
            "source": ["trace1", "trace1", "trace2"],
            "lowest_common_ancestor_id": [100, 100, 100],
            "kernel_time": [5000.0, 3000.0, 1000.0],
            "gpu_op_uid": [10, 11, None],
        }
    ).to_csv(csv_dir / "diff_stats.csv", index=False)

    uid_map = {
        10: {"name": "Cijk_A", "dur": 5000, "_category": "kernel", "gpu_events": []},
        11: {"name": "Cijk_B", "dur": 3000, "_category": "kernel", "gpu_events": []},
    }
    module = {
        "name": "nn.Module: Attn_0",
        "_category": "aten",
        "gpu_events": [10, 11],
        "args": {},
    }
    tree = _StubTree([module], uid_map)
    analyzer = _StubAnalyzer(tree)

    cands = _extract_comparative_fusion_candidates(str(csv_dir), analyzer, tree)
    assert len(cands) >= 1
    assert cands[0]["kernel_count_trace1"] >= 2


def test_extract_comparative_missing_inputs(tmp_path):
    assert _extract_comparative_fusion_candidates(str(tmp_path)) == []
    csv_dir = tmp_path / "csv"
    csv_dir.mkdir()
    pd.DataFrame({"name": []}).to_csv(csv_dir / "diff_stats.csv", index=False)
    assert _extract_comparative_fusion_candidates(str(csv_dir)) == []


def _write_minimal_orchestrator_csvs(base, comparative=False):
    t1 = os.path.join(
        base, "perf_report_trace1_csvs" if comparative else "perf_report_csvs"
    )
    os.makedirs(t1)
    pd.DataFrame(
        {
            "type": ["total_time", "computation_time", "idle_time"],
            "time ms": [1000.0, 900.0, 100.0],
            "percent": [100.0, 90.0, 10.0],
        }
    ).to_csv(os.path.join(t1, "gpu_timeline.csv"), index=False)
    pd.DataFrame(
        {
            "name": ["aten::mm"],
            "total_direct_kernel_time_ms": [800.0],
            "op category": ["GEMM"],
        }
    ).to_csv(os.path.join(t1, "ops_summary.csv"), index=False)
    pd.DataFrame(
        {
            "name": ["aten::mm"],
            "op category": ["GEMM"],
            "Kernel Time (µs)_sum": [800_000.0],
            "total_duration_us": [900_000.0],
        }
    ).to_csv(os.path.join(t1, "unified_perf_summary.csv"), index=False)
    if comparative:
        t2 = os.path.join(base, "perf_report_trace2_csvs")
        os.makedirs(t2)
        pd.DataFrame(
            {
                "type": ["total_time", "computation_time", "idle_time"],
                "time ms": [900.0, 810.0, 90.0],
                "percent": [100.0, 90.0, 10.0],
            }
        ).to_csv(os.path.join(t2, "gpu_timeline.csv"), index=False)
        pd.DataFrame(
            {
                "op category": ["GEMM"],
                "Kernel Time (µs)_sum": [700_000.0],
                "operation_count": [1],
            }
        ).to_csv(os.path.join(t2, "unified_perf_summary.csv"), index=False)


def test_orchestrator_main_missing_csv_exits(tmp_path):
    script = os.path.join(ANALYSIS_DIR, "utils", "orchestrator_prepare.py")
    env = os.environ.copy()
    env["PYTHONPATH"] = REPO_ROOT
    result = subprocess.run(
        [
            sys.executable,
            script,
            "--trace-path",
            "/fake/trace.json",
            "--platform",
            "MI300X",
            "--output-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=env,
    )
    assert result.returncode == 1
    assert "STEP1_FAIL" in result.stderr + result.stdout


def test_orchestrator_main_comparative_mocked(tmp_path, monkeypatch):
    from TraceLens.Agent.Analysis.utils import orchestrator_prepare as op

    out = str(tmp_path)
    _write_minimal_orchestrator_csvs(out, comparative=True)

    k1 = _kernel_event(10, "Cijk_a")
    k2 = _kernel_event(11, "ew_add")
    module = {"name": "nn.Module: MLP_0", "_category": "aten", "gpu_events": [10, 11]}
    tree = _StubTree([module], {10: k1, 11: k2})
    analyzer = _StubAnalyzer(tree)

    class _FakeTreePerfAnalyzer:
        @classmethod
        def from_file(cls, *args, **kwargs):
            return analyzer

    monkeypatch.setattr(op, "TreePerfAnalyzer", _FakeTreePerfAnalyzer)
    monkeypatch.setattr(
        op,
        "_extract_comparative_fusion_candidates",
        lambda *a, **k: [{"module_name": "m", "total_kernel_time_us_trace1": 100}],
    )

    old_argv = sys.argv
    sys.argv = [
        "orchestrator_prepare",
        "--trace-path",
        "/fake/trace.json",
        "--platform",
        "MI300X",
        "--output-dir",
        out,
        "--comparison-scope",
        "comparative",
    ]
    try:
        op.main()
    finally:
        sys.argv = old_argv

    manifest = json.loads(
        open(os.path.join(out, "category_data", "category_manifest.json")).read()
    )
    assert manifest["comparison_scope"] == "comparative"
    assert "trace2_gpu_utilization" in manifest
    assert os.path.isfile(os.path.join(out, "metadata", "trace2_gpu_utilization.json"))


def test_orchestrator_main_standalone_mocked(tmp_path, monkeypatch):
    from TraceLens.Agent.Analysis.utils import orchestrator_prepare as op

    out = str(tmp_path)
    _write_minimal_orchestrator_csvs(out, comparative=False)

    k1 = _kernel_event(10, "Cijk_a")
    k2 = _kernel_event(11, "ew_add")
    module = {"name": "nn.Module: MLP_0", "_category": "aten", "gpu_events": [10, 11]}
    tree = _StubTree([module], {10: k1, 11: k2})
    analyzer = _StubAnalyzer(tree)

    class _FakeTreePerfAnalyzer:
        @classmethod
        def from_file(cls, *args, **kwargs):
            return analyzer

    monkeypatch.setattr(op, "TreePerfAnalyzer", _FakeTreePerfAnalyzer)
    monkeypatch.setattr(
        op,
        "_extract_standalone_fusion_candidates",
        lambda *a, **k: [
            {
                "module_name": "nn.Module: MLP_0",
                "base_name": "MLP",
                "total_kernel_time_us": 5000,
                "kernels": [{"name": "a"}, {"name": "b"}],
            }
        ],
    )

    old_argv = sys.argv
    sys.argv = [
        "orchestrator_prepare",
        "--trace-path",
        "/fake/trace.json",
        "--platform",
        "MI300X",
        "--output-dir",
        out,
    ]
    try:
        op.main()
    finally:
        sys.argv = old_argv

    manifest = json.loads(
        open(os.path.join(out, "category_data", "category_manifest.json")).read()
    )
    assert manifest["comparison_scope"] == "standalone"
    names = [c["name"] for c in manifest["categories"]]
    assert "gemm" in names
    assert "cpu_idle" in names
    assert "kernel_fusion" in names
