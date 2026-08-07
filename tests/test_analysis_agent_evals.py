###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the agent_evals/Analysis eval harness.

The eval harness grades agent-generated reports; these tests grade the graders.
All target functions are pure (dict/df/csv in, score/tuple out) and exercised
with small synthetic fixtures. The modules under ``agent_evals/Analysis`` are
not importable via the ``TraceLens`` package path, so each is loaded by file
path with importlib. ``eval_utils`` is placed on ``sys.path`` first so
run_post_processing's ``from aggregate_repeatability import ...`` resolves.
"""

import importlib.util
import json
import os
import sys
from collections import Counter

import pytest
import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_ANALYSIS_DIR = os.path.join(REPO_ROOT, "agent_evals", "Analysis")
_EVAL_UTILS_DIR = os.path.join(_ANALYSIS_DIR, "eval_utils")
_RULES_PATH = os.path.join(_EVAL_UTILS_DIR, "report_section_rules.yaml")

if _EVAL_UTILS_DIR not in sys.path:
    sys.path.insert(0, _EVAL_UTILS_DIR)


def _load_module(name, rel_path):
    """Load a module from a file path under agent_evals/Analysis."""
    path = os.path.join(_ANALYSIS_DIR, rel_path)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


post = _load_module(
    "run_post_processing_under_test",
    os.path.join("skills", "eval-post-processing", "run_post_processing.py"),
)
quality = _load_module(
    "quality_scripted_evals_under_test",
    os.path.join("eval_utils", "quality_scripted_evals.py"),
)
workflow = _load_module(
    "workflow_scripted_evals_under_test",
    os.path.join("eval_utils", "workflow_scripted_evals.py"),
)
aggregate = _load_module(
    "aggregate_repeatability_under_test",
    os.path.join("eval_utils", "aggregate_repeatability.py"),
)


@pytest.fixture(scope="module")
def rules():
    with open(_RULES_PATH) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# run_post_processing.classify_row (real report_section_rules.yaml)
# ---------------------------------------------------------------------------


def test_classify_row_kernel_fusion_by_eval_index(rules):
    base, standalone, cause, fix = post.classify_row(
        {"eval_index": "workflow_eval_13"}, rules
    )
    assert base == "Kernel Fusion"
    assert standalone == "Kernel Fusion Opportunities"


def test_classify_row_reasoning_and_compute_by_eval_index(rules):
    base, standalone, _, _ = post.classify_row({"eval_index": "quality_eval_2"}, rules)
    assert base == "Reasoning"
    # quality_eval_(2|3) routes standalone section to Compute Kernel Optimizations.
    assert standalone == "Compute Kernel Optimizations"


def test_classify_row_exec_summary_by_issue_summary(rules):
    base, standalone, _, _ = post.classify_row(
        {"eval_index": "", "issue_summary": "Executive Summary mismatch"}, rules
    )
    # No base_section rule matches an exec-summary issue, so it falls through.
    assert base == "Others"
    assert standalone == "Executive Summary"


def test_classify_row_case_insensitive_issue_summary(rules):
    base, _, _, _ = post.classify_row(
        {"issue_summary": "KERNEL FUSION INSIGHTS"}, rules
    )
    assert base == "Kernel Fusion"


def test_classify_row_fall_through_to_defaults(rules):
    base, standalone, cause, fix = post.classify_row(
        {"eval_index": "zzz_unknown", "issue_summary": "nothing here", "details": ""},
        rules,
    )
    assert base == "Others"
    assert standalone == "Detailed Analysis"
    assert cause == rules["defaults"]["likely_cause"]
    assert fix == rules["defaults"]["suggested_fix"]


def test_classify_row_failure_mode_first_match_wins():
    # Hermetic: earlier rule wins when both regexes match the same haystack,
    # independent of the live yaml's rule ordering.
    synthetic_rules = {
        "base_section_rules": [],
        "standalone_section_rules": [],
        "failure_mode_rules": [
            {
                "match_regex": "alpha",
                "likely_cause": "cause_first",
                "suggested_fix": "fix_first",
            },
            {
                "match_regex": "beta",
                "likely_cause": "cause_second",
                "suggested_fix": "fix_second",
            },
        ],
        "defaults": {"likely_cause": "default_cause", "suggested_fix": "default_fix"},
    }
    _, _, cause, fix = post.classify_row(
        {"details": "alpha and beta both present"}, synthetic_rules
    )
    assert cause == "cause_first"
    assert fix == "fix_first"


def test_classify_row_failure_mode_efficiency(rules):
    _, _, cause, fix = post.classify_row({"issue_summary": "efficiency drift"}, rules)
    assert cause == rules["failure_mode_rules"][1]["likely_cause"]
    assert fix == rules["failure_mode_rules"][1]["suggested_fix"]


# ---------------------------------------------------------------------------
# run_post_processing.normalize_row (legacy CSV column coalescing)
# ---------------------------------------------------------------------------


def test_normalize_row_seven_columns_one_to_one():
    row = ["7", "cat", "summ", "PASS", "det", "cause", "fix"]
    out = post.normalize_row(row)
    assert out == {
        "index": "7",
        "category": "cat",
        "issue_summary": "summ",
        "result": "PASS",
        "details": "det",
        "root_cause": "cause",
        "recommended_fix": "fix",
    }


def test_normalize_row_five_columns_legacy():
    row = ["1", "cat", "summ", "FAIL", "det"]
    out = post.normalize_row(row)
    assert out["index"] == "1"
    assert out["details"] == "det"
    # root_cause/recommended_fix absent in legacy 5-col schema.
    assert out["root_cause"] == ""
    assert out["recommended_fix"] == ""


def test_normalize_row_more_than_seven_columns_coalesces_details():
    # Embedded-comma "details" splits into extra columns; row[4:] is re-joined.
    row = ["1", "cat", "summ", "FAIL", "part_a", "part_b", "part_c", "part_d"]
    out = post.normalize_row(row)
    # First 7 map positionally, then details is overwritten by the join of row[4:].
    assert out["details"] == "part_a,part_b,part_c,part_d"
    assert out["root_cause"] == "part_b"
    assert out["recommended_fix"] == "part_c"


def test_normalize_row_fewer_than_five_columns_pads():
    row = ["1", "cat", "summ"]
    out = post.normalize_row(row)
    assert out["index"] == "1"
    assert out["issue_summary"] == "summ"
    assert out["result"] == ""
    assert out["details"] == ""


# ---------------------------------------------------------------------------
# quality_scripted_evals._check_csv_alignment_dirs
# ---------------------------------------------------------------------------


def _write_csv(path, text):
    with open(path, "w") as f:
        f.write(text)


def test_csv_alignment_missing_ref_dir(tmp_path):
    gen = tmp_path / "gen"
    gen.mkdir()
    result, details = quality._check_csv_alignment_dirs(
        str(gen), str(tmp_path / "nope")
    )
    assert result == "FAIL"
    assert "Reference CSV directory not found" in details


def test_csv_alignment_missing_generated_file(tmp_path):
    gen = tmp_path / "gen"
    ref = tmp_path / "ref"
    gen.mkdir()
    ref.mkdir()
    _write_csv(str(ref / "a.csv"), "x,y\n1,2\n")
    result, details = quality._check_csv_alignment_dirs(str(gen), str(ref))
    assert result == "FAIL"
    assert "a.csv: missing" in details


def test_csv_alignment_within_tolerance_passes(tmp_path):
    gen = tmp_path / "gen"
    ref = tmp_path / "ref"
    gen.mkdir()
    ref.mkdir()
    _write_csv(str(ref / "a.csv"), "x,y\n1,2.0\n")
    # rel diff 0.01, abs diff 0.02 -> under both gates.
    _write_csv(str(gen / "a.csv"), "x,y\n1,2.02\n")
    assert quality._check_csv_alignment_dirs(str(gen), str(ref)) == ("PASS", "")


def test_csv_alignment_large_rel_but_small_abs_passes(tmp_path):
    # Gate requires rel>0.01 AND abs>0.05; huge rel with tiny abs must PASS.
    gen = tmp_path / "gen"
    ref = tmp_path / "ref"
    gen.mkdir()
    ref.mkdir()
    _write_csv(str(ref / "a.csv"), "x,y\n1,0.001\n")
    _write_csv(str(gen / "a.csv"), "x,y\n1,0.04\n")  # abs 0.039 < 0.05
    assert quality._check_csv_alignment_dirs(str(gen), str(ref)) == ("PASS", "")


def test_csv_alignment_large_abs_but_small_rel_passes(tmp_path):
    gen = tmp_path / "gen"
    ref = tmp_path / "ref"
    gen.mkdir()
    ref.mkdir()
    _write_csv(str(ref / "a.csv"), "x,y\n1,1000.0\n")
    _write_csv(str(gen / "a.csv"), "x,y\n1,1005.0\n")  # rel 0.005 < 0.01
    assert quality._check_csv_alignment_dirs(str(gen), str(ref)) == ("PASS", "")


def test_csv_alignment_out_of_tolerance_fails(tmp_path):
    gen = tmp_path / "gen"
    ref = tmp_path / "ref"
    gen.mkdir()
    ref.mkdir()
    _write_csv(str(ref / "a.csv"), "x,y\n1,2.0\n")
    _write_csv(str(gen / "a.csv"), "x,y\n1,50.0\n")  # both gates exceeded
    result, details = quality._check_csv_alignment_dirs(str(gen), str(ref))
    assert result == "FAIL"
    assert "a.csv:y" in details


def test_csv_alignment_row_count_mismatch_fails(tmp_path):
    gen = tmp_path / "gen"
    ref = tmp_path / "ref"
    gen.mkdir()
    ref.mkdir()
    _write_csv(str(ref / "a.csv"), "x,y\n1,2\n3,4\n")
    _write_csv(str(gen / "a.csv"), "x,y\n1,2\n")
    result, details = quality._check_csv_alignment_dirs(str(gen), str(ref))
    assert result == "FAIL"
    assert "row count" in details


def test_csv_alignment_missing_required_column_fails(tmp_path):
    gen = tmp_path / "gen"
    ref = tmp_path / "ref"
    gen.mkdir()
    ref.mkdir()
    _write_csv(str(ref / "a.csv"), "x,required_col\n1,2\n")
    _write_csv(str(gen / "a.csv"), "x\n1\n")
    result, details = quality._check_csv_alignment_dirs(str(gen), str(ref))
    assert result == "FAIL"
    assert "missing required columns" in details


def test_csv_alignment_optional_column_in_ref_absent_from_gen_passes(tmp_path):
    # num_kernels is optional: ref may carry it while gen omits it entirely.
    # The alignment must tolerate this and not raise when re-selecting columns.
    gen = tmp_path / "gen"
    ref = tmp_path / "ref"
    gen.mkdir()
    ref.mkdir()
    _write_csv(str(ref / "a.csv"), "x,num_kernels\n1,5\n")
    _write_csv(str(gen / "a.csv"), "x\n1\n")
    assert quality._check_csv_alignment_dirs(str(gen), str(ref)) == ("PASS", "")


# ---------------------------------------------------------------------------
# workflow_scripted_evals: numeric / string extractors
# ---------------------------------------------------------------------------


def test_extract_numeric_strips_thousands_separator():
    assert workflow._extract_numeric("1,234.5 ms") == 1234.5


def test_extract_numeric_signed_with_spaces():
    assert workflow._extract_numeric("- 3.2") == -3.2
    assert workflow._extract_numeric("+5.0") == 5.0


def test_extract_numeric_plus_minus_symbol_takes_magnitude():
    # The ± glyph is not a regex sign; the first numeric token is returned.
    assert workflow._extract_numeric("±5.5") == 5.5


def test_extract_numeric_no_number_returns_none():
    assert workflow._extract_numeric("n/a") is None


def test_extract_percent():
    assert workflow._extract_percent("82.2%") == 82.2
    assert workflow._extract_percent("50 %") == 50.0
    assert workflow._extract_percent("no percent here") is None


def test_find_table_row_skips_separator_and_header():
    section = (
        "| Metric | Value |\n"
        "|---|---|\n"
        "| Idle Time | 12.5% |\n"
        "| Computation | 80.0% |\n"
    )
    assert workflow._find_table_row(section, ["Idle"]) == "12.5%"
    assert workflow._find_table_row(section, ["Computation"]) == "80.0%"


def test_find_table_row_no_match_returns_none():
    section = "| Metric | Value |\n|---|---|\n| Idle Time | 12.5% |\n"
    assert workflow._find_table_row(section, ["Bandwidth"]) is None


# ---------------------------------------------------------------------------
# workflow_scripted_evals._check_exec_summary_comparative
# ---------------------------------------------------------------------------


def _comparative_row(output_dir, diff_cell):
    exec_section = (
        "| Metric | Trace1 | Trace2 | Difference |\n"
        "|---|---|---|---|\n"
        f"| Computation | 80.0% | 70.0% | {diff_cell} |\n"
    )
    rows = workflow._check_exec_summary_comparative(str(output_dir), exec_section)
    return next(r for r in rows if r["index"] == "workflow_eval_10_compute_pct")


def test_exec_summary_comparative_difference_arithmetic_pass(tmp_path):
    # No gpu_timeline CSVs present, so only structural + arithmetic checks run.
    row = _comparative_row(tmp_path, "10.0%")
    assert row["result"] == "PASS"


def test_exec_summary_comparative_difference_arithmetic_fail(tmp_path):
    row = _comparative_row(tmp_path, "99.0%")  # |80-70|=10, reported 99
    assert row["result"] == "FAIL"
    assert "arithmetic error" in row["details"]


def test_exec_summary_comparative_missing_difference_column_fails(tmp_path):
    exec_section = (
        "| Metric | Trace1 | Trace2 |\n"
        "|---|---|---|\n"
        "| Computation | 80.0% | 70.0% |\n"
    )
    rows = workflow._check_exec_summary_comparative(str(tmp_path), exec_section)
    row = next(r for r in rows if r["index"] == "workflow_eval_10_compute_pct")
    assert row["result"] == "FAIL"
    assert "Difference column missing" in row["details"]


def test_exec_summary_comparative_row_not_found_fails(tmp_path):
    exec_section = (
        "| Metric | Trace1 | Trace2 | Difference |\n"
        "|---|---|---|---|\n"
        "| Computation | 80.0% | 70.0% | 10.0% |\n"
    )
    rows = workflow._check_exec_summary_comparative(str(tmp_path), exec_section)
    # idle_pct row is absent from the table -> row-not-found FAIL.
    idle = next(r for r in rows if r["index"] == "workflow_eval_10_idle_pct")
    assert idle["result"] == "FAIL"
    assert "Row not found" in idle["details"]


# ---------------------------------------------------------------------------
# workflow_scripted_evals._extract_p_items / _check_p_item_fields
# ---------------------------------------------------------------------------


def test_extract_p_items_returns_numbered_blocks():
    section = (
        "### Kernel P1: First\n"
        "**Insight**: something\n**Action**: do it\n**Impact**: big\n"
        "### Kernel P2: Second\n"
        "**Issue**: bad\n"
    )
    items = workflow._extract_p_items(section)
    assert [n for n, _ in items] == [1, 2]
    assert "First" in items[0][1]
    assert "Second" in items[1][1]


def test_check_p_item_fields_complete_compute_item():
    p_text = "**Insight**: x\n**Action**: y\n**Impact**: z\n"
    assert workflow._check_p_item_fields(p_text, is_compute=True) == []


def test_check_p_item_fields_missing_action_and_impact():
    p_text = "**Issue**: bad\n"
    missing = workflow._check_p_item_fields(p_text, is_compute=True)
    assert "**Action**" in missing
    assert "**Impact**" in missing


def test_check_p_item_fields_system_item_no_impact_required():
    p_text = "**Insight**: x\n**Action**: y\n"
    assert workflow._check_p_item_fields(p_text, is_compute=False) == []


# ---------------------------------------------------------------------------
# aggregate_repeatability._classify_stability
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pass_count,total,expected",
    [
        (4, 4, "STABLE_PASS"),  # rate == 1.0
        (3, 4, "FLAKY_PASS"),  # rate > 0.5
        (2, 4, "FLAKY_FAIL"),  # rate == 0.5 (boundary, not > 0.5)
        (1, 4, "FLAKY_FAIL"),  # 0 < rate <= 0.5
        (0, 4, "STABLE_FAIL"),  # rate == 0.0
        (0, 0, "N/A"),  # no runs
    ],
)
def test_classify_stability_boundaries(pass_count, total, expected):
    assert aggregate._classify_stability(pass_count, total) == expected


# ---------------------------------------------------------------------------
# aggregate_repeatability.build_run_level_summary + build_failure_nature_summary
# ---------------------------------------------------------------------------


def _eval_row(trace_id, run_id, issue, result):
    return {
        "trace_id": trace_id,
        "run_id": run_id,
        "issue_summary": issue,
        "result": result,
    }


def test_build_run_level_summary_flags_catastrophic():
    rows = [
        _eval_row("T", 0, "A", "FAIL"),
        _eval_row("T", 0, "B", "FAIL"),
        _eval_row("T", 0, "C", "FAIL"),  # 3/3 fail -> catastrophic
        _eval_row("T", 1, "A", "FAIL"),
        _eval_row("T", 1, "B", "PASS"),
        _eval_row("T", 1, "C", "PASS"),  # 1/3 fail -> not catastrophic
    ]
    run_rows = aggregate.build_run_level_summary(rows)
    by_run = {(r["trace_id"], r["run_id"]): r for r in run_rows}
    assert by_run[("T", 0)]["is_catastrophic"] is True
    assert by_run[("T", 1)]["is_catastrophic"] is False


def test_build_failure_nature_partitions_stable_flaky_catastrophic():
    # T1: non-catastrophic runs; eval A fails in all runs (stable),
    #     eval B fails in one run only (flaky).
    # T2: a fully-failing run (catastrophic).
    rows = [
        _eval_row("T1", 0, "A", "FAIL"),
        _eval_row("T1", 0, "B", "FAIL"),
        _eval_row("T1", 0, "C", "PASS"),
        _eval_row("T1", 0, "D", "PASS"),
        _eval_row("T1", 1, "A", "FAIL"),
        _eval_row("T1", 1, "B", "PASS"),
        _eval_row("T1", 1, "C", "PASS"),
        _eval_row("T1", 1, "D", "PASS"),
        _eval_row("T2", 0, "E", "FAIL"),
        _eval_row("T2", 0, "F", "FAIL"),
        _eval_row("T2", 0, "G", "FAIL"),
    ]
    run_rows = aggregate.build_run_level_summary(rows)
    nature = aggregate.build_failure_nature_summary(rows, run_rows)
    assert nature["stable"] == 2  # A fails in both T1 runs
    assert nature["flaky"] == 1  # B fails in one T1 run
    assert nature["catastrophic_pipeline"] == 3  # all T2 run_0 failures
    assert nature["total"] == 6


def test_build_failure_nature_ignores_missing_rows():
    rows = [
        _eval_row("T", 0, "A", "MISSING"),
        _eval_row("T", 0, "B", "PASS"),
    ]
    run_rows = aggregate.build_run_level_summary(rows)
    nature = aggregate.build_failure_nature_summary(rows, run_rows)
    assert nature == {
        "catastrophic_pipeline": 0,
        "stable": 0,
        "flaky": 0,
        "total": 0,
    }


# ---------------------------------------------------------------------------
# aggregate_repeatability.build_pass_rate_summary / build_stability_summary
# ---------------------------------------------------------------------------


def _pr_row(trace_id, eval_index, result):
    return {"trace_id": trace_id, "eval_index": eval_index, "result": result}


def test_build_pass_rate_summary_mixed_and_missing():
    rows = [
        _pr_row("T", "e1", "PASS"),
        _pr_row("T", "e1", "FAIL"),
        _pr_row("T", "e2", "PASS"),
        _pr_row("T", "e2", "MISSING"),  # skipped entirely
    ]
    summary_rows, cols = aggregate.build_pass_rate_summary(rows)
    assert cols == ["trace_id", "e1", "e2", "overall_pass_rate"]
    row = summary_rows[0]
    assert row["e1"] == "1/2"
    assert row["e2"] == "1/1"
    # 2 passes out of 3 non-missing checks -> 67%.
    assert row["overall_pass_rate"] == "2/3 (67%)"


def test_build_pass_rate_summary_eval_absent_for_one_trace_is_na():
    rows = [
        _pr_row("T1", "e1", "PASS"),
        _pr_row("T2", "e2", "FAIL"),
    ]
    summary_rows, cols = aggregate.build_pass_rate_summary(rows)
    by_trace = {r["trace_id"]: r for r in summary_rows}
    # e2 was never run for T1 -> N/A; e1 never run for T2 -> N/A.
    assert by_trace["T1"]["e2"] == "N/A"
    assert by_trace["T2"]["e1"] == "N/A"


def test_build_stability_summary_classes_and_totals():
    rows = [
        _pr_row("T", "stable_pass", "PASS"),
        _pr_row("T", "stable_pass", "PASS"),
        _pr_row("T", "stable_fail", "FAIL"),
        _pr_row("T", "stable_fail", "FAIL"),
        _pr_row("T", "flaky", "PASS"),
        _pr_row("T", "flaky", "FAIL"),
        _pr_row("T", "gone", "MISSING"),  # ignored
    ]
    stability_rows, cols, class_totals = aggregate.build_stability_summary(rows)
    row = stability_rows[0]
    assert row["stable_pass"] == "STABLE_PASS"
    assert row["stable_fail"] == "STABLE_FAIL"
    assert row["flaky"] == "FLAKY_FAIL"  # 1/2 == 0.5, not > 0.5
    assert "gone" not in cols  # missing-only eval dropped
    assert class_totals["STABLE_PASS"] == 1
    assert class_totals["STABLE_FAIL"] == 1
    assert class_totals["FLAKY_FAIL"] == 1


# ---------------------------------------------------------------------------
# aggregate_repeatability: small pure helpers
# ---------------------------------------------------------------------------


def test_extract_tool_stdout_variants():
    assert aggregate._extract_tool_stdout(None) == ""
    assert aggregate._extract_tool_stdout({"content": None}) == ""
    result = {
        "content": [
            {"type": "text", "text": "hello "},
            {"type": "image", "data": "ignored"},
            {"type": "text", "text": "world"},
        ]
    }
    assert aggregate._extract_tool_stdout(result) == "hello world"


def test_accumulate_usage_aliases_and_noop():
    totals = {"input": 0, "output": 0, "cache_read": 0}
    aggregate._accumulate_usage(None, totals)
    assert totals == {"input": 0, "output": 0, "cache_read": 0}
    aggregate._accumulate_usage(
        {"inputTokens": 10, "output_tokens": 5, "cache_read": 3}, totals
    )
    assert totals == {"input": 10, "output": 5, "cache_read": 3}
    # Alternate aliases accumulate on top.
    aggregate._accumulate_usage(
        {"input": 1, "outputTokens": 2, "cacheReadTokens": 4}, totals
    )
    assert totals == {"input": 11, "output": 7, "cache_read": 7}


def test_detect_steps_from_shell_keywords():
    steps = set()
    aggregate._detect_steps_from_shell(
        "TraceLens_generate_perf_report_pytorch ...", "", steps
    )
    assert "step1_perf_report" in steps
    steps = set()
    aggregate._detect_steps_from_shell("python orchestrator_prepare.py", "", steps)
    assert "step2_5_prepare" in steps
    steps = set()
    aggregate._detect_steps_from_shell("cat foo_findings.md", "", steps)
    assert "step7_subagent_findings" in steps
    steps = set()
    aggregate._detect_steps_from_shell("write analysis.md", "", steps)
    assert "step11_report" in steps


def test_compute_last_step_priority_and_empty():
    assert aggregate._compute_last_step(set()) == "none"
    both = {"step1_perf_report", "step11_report"}
    assert aggregate._compute_last_step(both) == "Step 11: Report"
    assert aggregate._compute_last_step({"step1_perf_report"}) == "Step 1: Perf report"


def test_detect_report_write_from_stdout_and_write_tool():
    # analysis.md in command + big stdout -> returns the stdout content.
    big = "x" * 150
    tc = {
        "shellToolCall": {
            "args": {"command": "cat analysis.md"},
            "result": {"success": {"stdout": big}},
        }
    }
    assert aggregate._detect_report_write(tc) == big
    # writeToolCall path.
    wtc = {
        "writeToolCall": {
            "args": {"path": "out/analysis.md", "content": "## Header\nbody"}
        }
    }
    assert aggregate._detect_report_write(wtc) == "## Header\nbody"
    # No analysis.md anywhere -> None.
    assert aggregate._detect_report_write({"shellToolCall": {"args": {}}}) is None


def test_detect_report_write_from_command_heredoc():
    cmd = "cat > out/analysis.md <<'REPORT_EOF'\n## Title\nbody line\nREPORT_EOF\n"
    tc = {"shellToolCall": {"args": {"command": cmd}}}
    assert aggregate._detect_report_write_from_command(tc) == "## Title\nbody line"


def test_detect_report_write_from_command_generic_delimiter():
    cmd = "tee analysis.md <<CUSTOM\nline one\nline two\nCUSTOM\n"
    tc = {"shellToolCall": {"args": {"command": cmd}}}
    got = aggregate._detect_report_write_from_command(tc)
    assert "line one" in got and "line two" in got


def test_detect_report_write_from_command_returns_none_without_redirect():
    # analysis.md present but no tee/cat> redirect.
    tc = {"shellToolCall": {"args": {"command": "ls analysis.md"}}}
    assert aggregate._detect_report_write_from_command(tc) is None
    # No analysis.md at all.
    tc2 = {"shellToolCall": {"args": {"command": "tee foo.txt <<EOF\nx\nEOF"}}}
    assert aggregate._detect_report_write_from_command(tc2) is None


# ---------------------------------------------------------------------------
# aggregate_repeatability.parse_ndjson_stream
# ---------------------------------------------------------------------------


def test_parse_ndjson_missing_file(tmp_path):
    diag = aggregate.parse_ndjson_stream(str(tmp_path / "nope.ndjson"))
    assert diag["outcome"] == "missing_file"


def test_parse_ndjson_small_unavailable(tmp_path):
    p = tmp_path / "s.ndjson"
    p.write_text("service unavailable")
    assert aggregate.parse_ndjson_stream(str(p))["outcome"] == "agent_cli_unavailable"


def test_parse_ndjson_small_error(tmp_path):
    p = tmp_path / "s.ndjson"
    p.write_text("some error occurred")
    assert aggregate.parse_ndjson_stream(str(p))["outcome"] == "agent_cli_error"


def test_parse_ndjson_small_empty_or_minimal(tmp_path):
    p = tmp_path / "s.ndjson"
    p.write_text("ok done")
    assert aggregate.parse_ndjson_stream(str(p))["outcome"] == "empty_or_minimal"


def _write_ndjson(path, records):
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    # Pad past the 100-byte small-file gate.
    if os.path.getsize(path) < 100:
        with open(path, "a") as f:
            f.write(" " * 120 + "\n")


def test_parse_ndjson_result_success(tmp_path):
    p = tmp_path / "s.ndjson"
    _write_ndjson(
        p,
        [
            {
                "type": "result",
                "is_error": False,
                "duration_ms": 4200,
                "usage": {"inputTokens": 100, "outputTokens": 20, "cacheReadTokens": 5},
            }
        ],
    )
    diag = aggregate.parse_ndjson_stream(str(p))
    assert diag["outcome"] == "success"
    assert diag["duration_ms"] == 4200
    assert diag["input_tokens"] == 100
    assert diag["output_tokens"] == 20
    assert diag["cache_read_tokens"] == 5


def test_parse_ndjson_result_error(tmp_path):
    p = tmp_path / "s.ndjson"
    _write_ndjson(p, [{"type": "result", "is_error": True}])
    assert aggregate.parse_ndjson_stream(str(p))["outcome"] == "error"


def test_parse_ndjson_agent_end_fallback(tmp_path):
    p = tmp_path / "s.ndjson"
    _write_ndjson(p, [{"type": "agent_end", "is_error": False}])
    assert aggregate.parse_ndjson_stream(str(p))["outcome"] == "success"


def test_parse_ndjson_no_result_record(tmp_path):
    p = tmp_path / "s.ndjson"
    _write_ndjson(p, [{"type": "something_else", "foo": "bar"}])
    assert aggregate.parse_ndjson_stream(str(p))["outcome"] == "no_result_record"


def test_parse_ndjson_turn_end_accumulates_tokens_and_duration(tmp_path):
    p = tmp_path / "s.ndjson"
    _write_ndjson(
        p,
        [
            {
                "type": "turn_end",
                "message": {
                    "timestamp": 1000,
                    "usage": {"input_tokens": 30, "output_tokens": 4},
                },
            },
            {
                "type": "turn_end",
                "message": {
                    "timestamp": 3500,
                    "usage": {"input_tokens": 10, "output_tokens": 6},
                },
            },
            {"type": "agent_end", "is_error": False},
        ],
    )
    diag = aggregate.parse_ndjson_stream(str(p))
    assert diag["turns"] == 2
    assert diag["input_tokens"] == 40
    assert diag["output_tokens"] == 10
    # Duration falls back to last_ts - first_ts when completion has none.
    assert diag["duration_ms"] == 2500


def test_parse_ndjson_tool_exec_report_and_steps(tmp_path):
    p = tmp_path / "s.ndjson"
    _write_ndjson(
        p,
        [
            {
                "type": "tool_execution_start",
                "args": {"command": "TraceLens_generate_perf_report_pytorch trace"},
            },
            {
                "type": "tool_execution_end",
                "args": {"command": "cat > analysis.md; echo done"},
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": "## Executive Summary\ntext\n## Appendix\nmore "
                            + "x" * 120,
                        }
                    ]
                },
            },
            {"type": "result", "is_error": False},
        ],
    )
    diag = aggregate.parse_ndjson_stream(str(p))
    assert diag["report_written"] is True
    assert "Executive Summary" in diag["report_headers"]
    assert diag["last_step_reached"] == "Step 11: Report"
    assert diag["tool_calls"] == 1  # only tool_execution_start increments pi_tool_calls


def test_parse_ndjson_invalid_json_unavailable_line(tmp_path):
    p = tmp_path / "s.ndjson"
    with open(p, "w") as f:
        f.write("this is not json but mentions service unavailable\n")
        f.write(" " * 120 + "\n")
    assert aggregate.parse_ndjson_stream(str(p))["outcome"] == "agent_cli_unavailable"


# ---------------------------------------------------------------------------
# workflow_scripted_evals: pure string helpers
# ---------------------------------------------------------------------------


def test_extract_section_returns_body_until_next_header():
    content = "## First\nbody one\nline two\n## Second\nbody two\n"
    assert workflow._extract_section(content, "## First") == "body one\nline two"
    assert workflow._extract_section(content, "## Missing") is None


def test_rebase_path_markers_and_passthrough():
    assert (
        workflow._rebase_path("/orig/run/category_data/x.json", "/new")
        == "/new/category_data/x.json"
    )
    assert (
        workflow._rebase_path("/orig/metadata/model_info.json", "/new")
        == "/new/metadata/model_info.json"
    )
    # No known marker -> returned unchanged.
    assert workflow._rebase_path("/somewhere/else.txt", "/new") == "/somewhere/else.txt"


def test_marker_attrs_from_inner():
    attrs = workflow._marker_attrs_from_inner(
        "kind=p_item category=compute low=1 mid=2 high=3"
    )
    assert attrs == {
        "kind": "p_item",
        "category": "compute",
        "low": "1",
        "mid": "2",
        "high": "3",
    }


def test_find_table_row_all_cells():
    section = (
        "| Metric | T1 | T2 | Diff |\n"
        "|---|---|---|---|\n"
        "| Computation | 80% | 70% | 10% |\n"
    )
    cells = workflow._find_table_row_all_cells(section, ["Computation"])
    assert cells == ["Computation", "80%", "70%", "10%"]
    assert workflow._find_table_row_all_cells(section, ["Nope"]) is None


def test_extract_detailed_analysis_subsection():
    content = (
        "## Detailed Analysis\n"
        "### Compute Kernel Insights\n"
        "compute body\n"
        "### System-Level Insights\n"
        "system body\n"
        "## Appendix\n"
    )
    sub = workflow._extract_detailed_analysis_subsection(
        content, "### Compute Kernel Insights"
    )
    assert "compute body" in sub
    assert "system body" not in sub
    assert (
        workflow._extract_detailed_analysis_subsection(
            "## Other\nx", "### Compute Kernel Insights"
        )
        is None
    )


# ---------------------------------------------------------------------------
# workflow_scripted_evals: analysis.md marker + template checks
# ---------------------------------------------------------------------------


def _write_report(tmp_path, text):
    (tmp_path / "analysis.md").write_text(text)
    return str(tmp_path)


def test_check_report_template_all_headers_and_table(tmp_path):
    text = (
        "## Executive Summary\n"
        "| Metric | Value |\n|---|---|\n| Total | 1 |\n"
        "## Compute Kernel Optimizations\n"
        "## Kernel Fusion Opportunities (Experimental)\n"
        "## System-Level Optimizations\n"
        "## Detailed Analysis\n"
        "## Appendix\n"
    )
    out = _write_report(tmp_path, text)
    rows = workflow._check_report_template(out)
    assert all(r["result"] == "PASS" for r in rows)
    table_row = next(r for r in rows if r["index"] == "workflow_eval_9_metrics_table")
    assert table_row["result"] == "PASS"


def test_check_report_template_missing_header_and_table(tmp_path):
    out = _write_report(tmp_path, "## Executive Summary\nno table here\n")
    rows = workflow._check_report_template(out)
    by_index = {r["index"]: r for r in rows}
    assert by_index["workflow_eval_9_compute"]["result"] == "FAIL"
    assert by_index["workflow_eval_9_metrics_table"]["result"] == "FAIL"
    assert by_index["workflow_eval_9_executive_summary"]["result"] == "PASS"


def test_check_marker_top_ops_complete(tmp_path):
    text = (
        "## Compute Kernel Optimizations\n"
        "<!-- impact-begin kind=top_ops -->\n"
        "<!-- top-ops-row name=foo low=1 high=3 -->\n"
        "<!-- impact-end -->\n"
    )
    out = _write_report(tmp_path, text)
    rows = workflow._check_marker_top_ops(out)
    assert rows[0]["result"] == "PASS"


def test_check_marker_top_ops_missing_marker(tmp_path):
    out = _write_report(tmp_path, "## Compute Kernel Optimizations\nno markers\n")
    rows = workflow._check_marker_top_ops(out)
    assert rows[0]["result"] == "FAIL"
    assert "top_ops" in rows[0]["details"]


def test_check_marker_top_ops_row_missing_attrs(tmp_path):
    text = (
        "<!-- impact-begin kind=top_ops -->\n"
        "<!-- top-ops-row name=foo low=1 -->\n"
        "<!-- impact-end -->\n"
    )
    out = _write_report(tmp_path, text)
    rows = workflow._check_marker_top_ops(out)
    assert rows[0]["result"] == "FAIL"
    assert "high" in rows[0]["details"]


def test_check_marker_top_ops_no_analysis_md(tmp_path):
    rows = workflow._check_marker_top_ops(str(tmp_path))
    assert rows[0]["result"] == "FAIL"
    assert "not found" in rows[0]["details"]


def test_check_marker_p_items_complete(tmp_path):
    text = (
        "## Compute Kernel Optimizations\n"
        "### Kernel P1: First\n"
        "<!-- impact-begin kind=p_item category=compute low=1 mid=2 high=3 -->\n"
        "detail\n"
        "<!-- impact-end -->\n"
    )
    out = _write_report(tmp_path, text)
    rows = workflow._check_marker_p_items(out)
    assert rows[0]["result"] == "PASS"


def test_check_marker_p_items_missing_attrs_and_unpaired(tmp_path):
    text = (
        "## Compute Kernel Optimizations\n"
        "### Kernel P1: First\n"
        "<!-- impact-begin kind=p_item category=compute low=1 -->\n"
        "no end marker\n"
    )
    out = _write_report(tmp_path, text)
    rows = workflow._check_marker_p_items(out)
    assert rows[0]["result"] == "FAIL"
    assert "mid" in rows[0]["details"] and "high" in rows[0]["details"]
    assert "Unpaired" in rows[0]["details"]


def test_check_marker_p_items_section_missing(tmp_path):
    out = _write_report(tmp_path, "## Executive Summary\nx\n")
    rows = workflow._check_marker_p_items(out)
    assert rows[0]["result"] == "FAIL"
    assert "Compute Kernel Optimizations" in rows[0]["details"]


def test_check_marker_detail_estimates_complete(tmp_path):
    text = (
        "## Detailed Analysis\n"
        "### Compute Kernel Insights\n"
        "#### Kernel P1: First\n"
        "<!-- impact-begin kind=detail_estimate low=1 high=5 -->\n"
        "body\n"
        "<!-- impact-end -->\n"
    )
    out = _write_report(tmp_path, text)
    rows = workflow._check_marker_detail_estimates(out)
    assert rows[0]["result"] == "PASS"


def test_check_marker_detail_estimates_sentinel_passes(tmp_path):
    text = (
        "## Detailed Analysis\n"
        "### Compute Kernel Insights\n"
        "#### Kernel P1: First\n"
        "Not quantifiable from trace data.\n"
    )
    out = _write_report(tmp_path, text)
    rows = workflow._check_marker_detail_estimates(out)
    assert rows[0]["result"] == "PASS"


def test_check_marker_detail_estimates_missing_marker_fails(tmp_path):
    text = (
        "## Detailed Analysis\n"
        "### Compute Kernel Insights\n"
        "#### Kernel P1: First\n"
        "just prose, no marker, no sentinel\n"
    )
    out = _write_report(tmp_path, text)
    rows = workflow._check_marker_detail_estimates(out)
    assert rows[0]["result"] == "FAIL"


def test_check_marker_detail_estimates_no_section(tmp_path):
    out = _write_report(tmp_path, "## Executive Summary\nx\n")
    rows = workflow._check_marker_detail_estimates(out)
    assert rows[0]["result"] == "FAIL"
    assert "Detailed Analysis" in rows[0]["details"]


def test_check_marker_reasoning_candidates_match(tmp_path):
    text = (
        "## Detailed Analysis\n"
        "### Compute Kernel Insights\n"
        "<!-- reasoning-candidate tier=compute rank=1 -->\n"
        "#### Kernel P1: First\n"
        "body\n"
    )
    out = _write_report(tmp_path, text)
    rows = workflow._check_marker_reasoning_candidates(out)
    compute = next(r for r in rows if r["index"] == "marker_eval_4_compute")
    assert compute["result"] == "PASS"


def test_check_marker_reasoning_candidates_mismatch(tmp_path):
    text = (
        "## Detailed Analysis\n"
        "### Compute Kernel Insights\n"
        "#### Kernel P1: First\n"
        "#### Kernel P2: Second\n"
    )
    out = _write_report(tmp_path, text)
    rows = workflow._check_marker_reasoning_candidates(out)
    compute = next(r for r in rows if r["index"] == "marker_eval_4_compute")
    assert compute["result"] == "FAIL"
    assert "reasoning-candidate" in compute["details"]


def test_check_marker_reasoning_candidates_no_subsections(tmp_path):
    out = _write_report(tmp_path, "## Executive Summary\nx\n")
    rows = workflow._check_marker_reasoning_candidates(out)
    assert rows[0]["result"] == "PASS"
    assert "No Detailed Analysis" in rows[0]["details"]


# ---------------------------------------------------------------------------
# run_post_processing: pure scalar helpers
# ---------------------------------------------------------------------------


def test_post_classify_stability():
    assert post.classify_stability(4, 4) == "STABLE_PASS"
    assert post.classify_stability(3, 4) == "FLAKY_PASS"
    assert post.classify_stability(1, 4) == "FLAKY_FAIL"
    assert post.classify_stability(0, 4) == "STABLE_FAIL"
    assert post.classify_stability(0, 0) == "N/A"


def test_post_pct_and_fmt_pass_rate():
    assert post.pct(1, 0) == 0.0
    assert post.pct(1, 2) == 50.0
    assert post.fmt_pass_rate(0, 0) == "0/0 (0%)"
    assert post.fmt_pass_rate(1, 2) == "1/2 (50.0%)"


def test_post_safe_name():
    assert post.safe_name("Hello World!!") == "hello_world"
    assert post.safe_name("a" * 100, max_len=10) == "a" * 10


def test_post_pattern_label():
    assert post.pattern_label([], []) == "stable"
    assert post.pattern_label([5], [6]) == "catastrophic"  # 5/6 > 0.5
    assert post.pattern_label([1, 1], [10, 10]) == "stable"  # spread 0
    assert post.pattern_label([0, 3], [10, 10]) == "flaky"  # spread 3, not catastrophic


def test_post_split_unit_e2e():
    trace_meta = {
        "t1": {"sub_category": "full_model", "platform": "x"},
        "t2": {"sub_category": "attention", "platform": "y"},
    }
    unit, e2e = post.split_unit_e2e(trace_meta)
    assert list(e2e.keys()) == ["t1"]
    assert list(unit.keys()) == ["t2"]


# ---------------------------------------------------------------------------
# run_post_processing: summary builders
# ---------------------------------------------------------------------------


def _agg_row(trace_id, run_id, eval_index, result, issue=""):
    return {
        "trace_id": trace_id,
        "run_id": run_id,
        "eval_index": eval_index,
        "result": result,
        "issue_summary": issue,
    }


def test_post_build_pass_rate_summary_unknown_eval_index():
    rows = [
        _agg_row("t1", 0, "", "PASS"),
        _agg_row("t1", 0, "e1", "FAIL"),
    ]
    out = post.build_pass_rate_summary(rows, {"t1"})
    row = out[0]
    assert row["_unknown"] == "1/1"
    assert row["e1"] == "0/1"
    assert row["overall_pass_rate"] == "1/2 (50.0%)"


def test_post_build_stability_summary():
    rows = [
        _agg_row("t1", 0, "e1", "PASS"),
        _agg_row("t1", 1, "e1", "FAIL"),
    ]
    out = post.build_stability_summary(rows, {"t1"})
    assert out[0]["e1"] == "FLAKY_FAIL"


def test_post_build_run_level_summary_catastrophic_string():
    rows = [
        _agg_row("t1", 0, "e1", "FAIL"),
        _agg_row("t1", 0, "e2", "FAIL"),
        _agg_row("t1", 0, "e3", "PASS"),  # 2/3 fail -> catastrophic
    ]
    out = post.build_run_level_summary(rows)
    assert out[0]["is_catastrophic"] == "True"
    assert out[0]["fail"] == 2
    assert out[0]["total"] == 3


def test_post_build_failure_nature():
    run_level = [
        {"trace_id": "t1", "run_id": 0, "is_catastrophic": "True"},
        {"trace_id": "t2", "run_id": 0, "is_catastrophic": "False"},
    ]
    stability = [
        {
            "trace_id": "t1",
            "e1": "STABLE_FAIL",
            "e2": "FLAKY_FAIL",
            "e3": "STABLE_PASS",
        },
    ]
    fn = post.build_failure_nature(run_level, stability, [])
    assert fn["catastrophic_pipeline"] == 1
    assert fn["stable"] == 1
    assert fn["flaky"] == 1
    assert fn["total"] == 2


def test_post_compute_per_split():
    trace_ids = {"t1"}
    agg_rows = [
        _agg_row("t1", 0, "e1", "PASS"),
        _agg_row("t1", 0, "e2", "FAIL", issue="bad thing"),
        _agg_row("t1", 0, "e3", "MISSING"),
    ]
    stream_rows = [
        {"trace_id": "t1", "outcome": "success", "duration_ms": "2000"},
        {"trace_id": "t1", "outcome": "no_result_record", "duration_ms": ""},
    ]
    out = post.compute_per_split(trace_ids, agg_rows, [], stream_rows, {})
    assert out["pass"] == 1
    assert out["fail"] == 1
    assert out["missing"] == 1
    assert out["total"] == 3
    assert out["rate"] == pytest.approx(33.3)
    assert out["top_issues"] == [("bad thing", 1)]
    case = out["per_case"][0]
    assert case["runs"] == "1/2"  # one successful of two stream rows
    assert case["avg_dur"] == 2.0  # only the 2000ms duration counted
    assert out["per_trace_fail_count"] == [("t1", 1)]


def test_post_top_failure_modes():
    classified = [
        {
            "trace_id": "t1",
            "issue_summary": "boom",
            "likely_cause": "c",
            "suggested_fix": "f",
        },
        {
            "trace_id": "t1",
            "issue_summary": "boom",
            "likely_cause": "c",
            "suggested_fix": "f",
        },
        {
            "trace_id": "t2",
            "issue_summary": "other",
            "likely_cause": "c2",
            "suggested_fix": "f2",
        },
    ]
    out = post.top_failure_modes(classified, {"t1"}, n=8)
    assert out == [("boom", 2, "c", "f")]


# ---------------------------------------------------------------------------
# run_post_processing: markdown table builders
# ---------------------------------------------------------------------------


def _metrics(p=1, f=2, m=0, rate=50.0):
    return {"pass": p, "fail": f, "missing": m, "rate": rate}


def test_post_metrics_table():
    out = post.metrics_table(_metrics(), _metrics(p=0), _metrics(p=1))
    assert "| PASS |" in out
    assert "| Pass rate |" in out


def test_post_failure_nature_table():
    fn = {"catastrophic_pipeline": 1, "stable": 2, "flaky": 1, "total": 4}
    out = post.failure_nature_table(fn)
    assert "Catastrophic pipeline" in out
    assert "Stable" in out
    assert "Flaky" in out


def test_post_failure_sections_table():
    out = post.failure_sections_table(Counter({"Compute": 3, "Others": 1}))
    assert "| Compute | 3 |" in out
    assert "| Others | 1 |" in out


def test_post_top_issues_table_empty_and_populated():
    empty = post.top_issues_table([], "Overall")
    assert "(no failures)" in empty
    populated = post.top_issues_table([("boom", 2)], "Unit")
    assert "| boom | 2 |" in populated


def test_post_per_case_table_empty_and_populated():
    assert "No unit test cases" in post.per_case_table([], "Unit")
    per_case = [
        {
            "trace_id": "t1",
            "sub_category": "attention",
            "platform": "x",
            "pass": 1,
            "fail": 2,
            "missing": 0,
            "pass_rate": "1/3 (33%)",
            "runs": "2/2",
            "avg_dur": 5.0,
        }
    ]
    out = post.per_case_table(per_case, "Unit")
    assert "| t1 |" in out
    assert "5.0s" in out


def test_post_failure_modes_table_empty_and_populated():
    assert "No failures" in post.failure_modes_table([], "Unit")
    out = post.failure_modes_table([("boom", 2, "cause", "fix")], "Unit")
    assert "| boom | 2 | cause | fix |" in out


def test_post_top_reproducers_table_empty_and_populated():
    empty = post.top_reproducers_table([("t1", 0)], {}, "traces.csv", "")
    assert "(none)" in empty
    out = post.top_reproducers_table(
        [("t1", 3)], {"t1": {"platform": "x"}}, "traces.csv", "cont"
    )
    assert "| t1 | 3 | x |" in out
    assert 'CONTAINER="cont"' in out


def test_post_catastrophic_table_empty_and_populated():
    assert "No catastrophic runs" in post.catastrophic_table([])
    runs = [{"trace_id": "t1", "run_id": 0, "pass": 1, "fail": 5, "total": 6}]
    out = post.catastrophic_table(runs)
    assert "| t1 | run_0 | 1 | 5 | 6 |" in out


def test_post_per_case_pattern_table():
    per_case_pattern = [
        {
            "trace_id": "t1",
            "runs": [
                {"run_id": "0", "fail": "5", "total": "6", "is_catastrophic": "true"},
                {"run_id": "1", "fail": "0", "total": "6", "is_catastrophic": "false"},
            ],
            "total_fails": 5,
            "label": "flaky",
        }
    ]
    out = post.per_case_pattern_table(per_case_pattern, num_runs=2)
    assert "(**crash**)" in out
    assert "| 5 | flaky |" in out


# ---------------------------------------------------------------------------
# quality_scripted_evals: _normalize_numpy_reprs + _pre_check_gates
# ---------------------------------------------------------------------------


def test_normalize_numpy_reprs():
    assert quality._normalize_numpy_reprs("np.int64(135)") == "135"
    assert quality._normalize_numpy_reprs("np.float32(1.5)") == "1.5"
    assert quality._normalize_numpy_reprs("plain text") == "plain text"
    assert quality._normalize_numpy_reprs("np.int64(1) and np.int64(2)") == "1 and 2"


def test_quality_pre_check_gates_missing_dirs(tmp_path):
    out = tmp_path / "out"
    ref = tmp_path / "ref"
    assert (
        quality._pre_check_gates(str(out), str(ref), "standalone")
        == "generated output directory does not exist"
    )
    out.mkdir()
    assert (
        quality._pre_check_gates(str(out), str(ref), "standalone")
        == "reference directory does not exist"
    )


def test_quality_pre_check_gates_standalone_csv_dirs(tmp_path):
    out = tmp_path / "out"
    ref = tmp_path / "ref"
    out.mkdir()
    ref.mkdir()
    assert "generated perf_report_csvs/" in quality._pre_check_gates(
        str(out), str(ref), "standalone"
    )
    (out / "perf_report_csvs").mkdir()
    assert "reference perf_report_csvs/" in quality._pre_check_gates(
        str(out), str(ref), "standalone"
    )
    (ref / "perf_report_csvs").mkdir()
    assert quality._pre_check_gates(str(out), str(ref), "standalone") is None


def test_quality_pre_check_gates_comparative(tmp_path):
    out = tmp_path / "out"
    ref = tmp_path / "ref"
    out.mkdir()
    ref.mkdir()
    (out / "perf_report_trace1_csvs").mkdir()
    (ref / "perf_report_trace1_csvs").mkdir()
    # trace1 present, trace2 missing on generated side.
    assert "generated perf_report_trace2_csvs/" in quality._pre_check_gates(
        str(out), str(ref), "comparative"
    )
    (out / "perf_report_trace2_csvs").mkdir()
    (ref / "perf_report_trace2_csvs").mkdir()
    assert quality._pre_check_gates(str(out), str(ref), "comparative") is None


# ---------------------------------------------------------------------------
# quality_scripted_evals.run (end-to-end driver)
# ---------------------------------------------------------------------------


def test_quality_run_gate_fail_writes_csv(tmp_path):
    out = tmp_path / "out"  # does not exist -> gate trips
    ref = tmp_path / "ref"
    ref.mkdir()
    results = tmp_path / "results.csv"
    rows = quality.run(str(out), str(ref), str(results), "standalone")
    assert rows[0]["result"] == "FAIL"
    assert "Pre-check gate" in rows[0]["details"]
    assert results.is_file()


def test_quality_run_standalone_pass(tmp_path):
    out = tmp_path / "out"
    ref = tmp_path / "ref"
    (out / "perf_report_csvs").mkdir(parents=True)
    (ref / "perf_report_csvs").mkdir(parents=True)
    _write_csv(str(out / "perf_report_csvs" / "a.csv"), "x,y\n1,2.0\n")
    _write_csv(str(ref / "perf_report_csvs" / "a.csv"), "x,y\n1,2.0\n")
    results = tmp_path / "results.csv"
    rows = quality.run(str(out), str(ref), str(results), "standalone")
    assert rows[0]["index"] == "quality_eval_1"
    assert rows[0]["result"] == "PASS"


def test_quality_run_comparative_two_traces(tmp_path):
    out = tmp_path / "out"
    ref = tmp_path / "ref"
    for tnum in (1, 2):
        (out / f"perf_report_trace{tnum}_csvs").mkdir(parents=True)
        (ref / f"perf_report_trace{tnum}_csvs").mkdir(parents=True)
        _write_csv(str(out / f"perf_report_trace{tnum}_csvs" / "a.csv"), "x,y\n1,2\n")
        _write_csv(str(ref / f"perf_report_trace{tnum}_csvs" / "a.csv"), "x,y\n1,2\n")
    results = tmp_path / "results.csv"
    rows = quality.run(str(out), str(ref), str(results), "comparative")
    assert {r["index"] for r in rows} == {
        "quality_eval_1_trace1",
        "quality_eval_1_trace2",
    }
    assert all(r["result"] == "PASS" for r in rows)


def test_quality_csv_alignment_bool_and_string_diffs(tmp_path):
    gen = tmp_path / "gen"
    ref = tmp_path / "ref"
    gen.mkdir()
    ref.mkdir()
    # bool column differs.
    _write_csv(str(ref / "b.csv"), "flag\nTrue\nFalse\n")
    _write_csv(str(gen / "b.csv"), "flag\nTrue\nTrue\n")
    result, details = quality._check_csv_alignment_dirs(str(gen), str(ref))
    assert result == "FAIL"
    assert "bool value" in details


def test_quality_csv_alignment_string_column_numpy_normalized(tmp_path):
    gen = tmp_path / "gen"
    ref = tmp_path / "ref"
    gen.mkdir()
    ref.mkdir()
    # String column where numpy wrapper is normalized away -> PASS.
    _write_csv(str(ref / "s.csv"), "name\nnp.int64(5)\n")
    _write_csv(str(gen / "s.csv"), "name\n5\n")
    assert quality._check_csv_alignment_dirs(str(gen), str(ref)) == ("PASS", "")


# ---------------------------------------------------------------------------
# workflow_scripted_evals: manifest + file-existence checks
# ---------------------------------------------------------------------------


def _valid_output_dir(tmp_path, comparison_scope="standalone"):
    """Build a fully-populated analysis output_dir that passes most checks."""
    out = tmp_path / "out"
    for d in ("metadata", "category_data", "system_findings", "category_findings"):
        (out / d).mkdir(parents=True)

    manifest = {
        "gpu_utilization": {"idle_time_percent": 5},
        "categories": [
            {
                "name": "gemm",
                "tier": "compute_kernel",
                "metadata_file": "category_data/gemm_metrics.json",
                "tree_data_file": "category_data/gemm_tree.json",
            }
        ],
    }
    (out / "category_data" / "category_manifest.json").write_text(json.dumps(manifest))
    (out / "category_data" / "gemm_metrics.json").write_text(
        json.dumps({"impact_estimates": []})
    )
    (out / "category_data" / "gemm_tree.json").write_text("{}")
    (out / "category_findings" / "gemm_findings.md").write_text("# findings\n")
    (out / "metadata" / "model_info.json").write_text(
        json.dumps(
            {
                "model": "TestNet",
                "architecture": "dense",
                "scale": "7B",
                "precision": "fp16",
            }
        )
    )

    if comparison_scope == "comparative":
        csv_dir = out / "perf_report_trace1_csvs"
    else:
        csv_dir = out / "perf_report_csvs"
    csv_dir.mkdir()
    (csv_dir / "unified_perf_summary.csv").write_text("a,b\n1,2\n")
    (csv_dir / "gpu_timeline.csv").write_text(
        "type,percent\ncomputation_time,80.0\nidle_time,5.0\n"
    )

    report = (
        "## Executive Summary\n"
        "| Metric | Value |\n|---|---|\n"
        "| Total Compute Time | 100 ms |\n"
        "| Computation | 80.0% |\n"
        "| Idle Time | 5.0% |\n"
        "| Exposed Communication | 2.0% |\n"
        "| Top Bottleneck Category | gemm |\n"
        "## Compute Kernel Optimizations\n"
        "<!-- impact-begin kind=top_ops -->\n"
        "<!-- top-ops-row name=foo low=1 high=3 -->\n"
        "<!-- impact-end -->\n"
        "### Kernel P1: Fix gemm\n"
        "**Insight**: slow\n**Action**: tune\n**Impact**: big\n"
        "<!-- impact-begin kind=p_item category=compute low=1 mid=2 high=3 -->\n"
        "detail\n<!-- impact-end -->\n"
        "## Kernel Fusion Opportunities (Experimental)\n"
        "## System-Level Optimizations\n"
        "## Detailed Analysis\n"
        "### Compute Kernel Insights\n"
        "<!-- reasoning-candidate tier=compute rank=1 -->\n"
        "#### Kernel P1: Fix gemm\n"
        "<!-- impact-begin kind=detail_estimate low=1 high=5 -->\n"
        "body\n<!-- impact-end -->\n"
        "## Appendix\n"
        "Model: TestNet dense 7B fp16\n"
    )
    (out / "analysis.md").write_text(report)
    return str(out)


def test_workflow_check_directories_and_metadata(tmp_path):
    out = _valid_output_dir(tmp_path)
    assert workflow._check_directories(out) == ("PASS", "")
    assert workflow._check_metadata_files(out) == ("PASS", "")
    assert workflow._check_tree_data_files(out) == ("PASS", "")
    assert workflow._check_model_info(out) == ("PASS", "")
    assert workflow._check_findings_exist(out) == ("PASS", "")
    assert workflow._check_findings_placement(out) == ("PASS", "")
    assert workflow._check_unified_perf_report(out, "standalone") == ("PASS", "")


def test_workflow_check_directories_missing(tmp_path):
    out = tmp_path / "empty"
    out.mkdir()
    result, details = workflow._check_directories(str(out))
    assert result == "FAIL"
    assert "Missing directories" in details


def test_workflow_check_model_info_missing_keys(tmp_path):
    out = tmp_path / "out"
    (out / "metadata").mkdir(parents=True)
    (out / "metadata" / "model_info.json").write_text(json.dumps({"model": "X"}))
    result, details = workflow._check_model_info(str(out))
    assert result == "FAIL"
    assert "Missing keys" in details


def test_workflow_check_plot_skipped_when_no_estimates(tmp_path):
    out = _valid_output_dir(tmp_path)
    # No png, priority_data.json absent, metrics have empty impact_estimates.
    result, details = workflow._check_plot(out)
    assert result == "PASS"
    assert "correctly skipped" in details


def test_workflow_run_standalone_end_to_end(tmp_path):
    out = _valid_output_dir(tmp_path)
    results = tmp_path / "results.csv"
    rows = workflow.run(out, str(results), "standalone")
    indices = {r["index"] for r in rows}
    assert "workflow_eval_1" in indices
    assert "marker_eval_1" in indices
    assert results.is_file()
    # The eight registry evals should all pass on a valid tree.
    assert next(r for r in rows if r["index"] == "workflow_eval_1")["result"] == "PASS"


def test_workflow_run_gate_fail(tmp_path):
    out = tmp_path / "missing"  # no dir -> pre-check gate trips
    results = tmp_path / "results.csv"
    rows = workflow.run(str(out), str(results), "standalone")
    assert all(r["result"] == "FAIL" for r in rows)
    assert any(r["index"] == "marker_eval_4" for r in rows)


def test_workflow_check_model_id_fields(tmp_path):
    out = _valid_output_dir(tmp_path)
    rows = workflow._check_model_id(out)
    # All four model fields appear in the Appendix -> PASS.
    assert all(r["result"] == "PASS" for r in rows)


def test_workflow_check_issue_template(tmp_path):
    out = _valid_output_dir(tmp_path)
    rows = workflow._check_issue_template(out)
    # Compute P1 has Insight/Action/Impact -> PASS.
    assert any(
        r["index"] == "workflow_eval_11_compute_P1" and r["result"] == "PASS"
        for r in rows
    )


def test_workflow_check_exec_summary_standalone(tmp_path):
    out = _valid_output_dir(tmp_path)
    rows = workflow._check_exec_summary(out, "standalone")
    compute = next(r for r in rows if r["index"] == "workflow_eval_10_compute_pct")
    assert compute["result"] == "PASS"


# ---------------------------------------------------------------------------
# aggregate_repeatability: collection + main driver
# ---------------------------------------------------------------------------


def _make_run_tree(root):
    """Create a repeatability_results tree with one trace, two runs."""
    t1 = root / "t1"
    (t1 / "run_0").mkdir(parents=True)
    (t1 / "run_1").mkdir(parents=True)
    # run_0 has eval_summary.csv, run_1 is missing it.
    with open(t1 / "run_0" / "eval_summary.csv", "w") as f:
        f.write("index,category,issue_summary,result,details\n")
        f.write("e1,Workflow,thing,PASS,ok\n")
        f.write("e2,Workflow,other,FAIL,bad\n")
    _write_ndjson(
        t1 / "run_0" / "analysis_stream.ndjson",
        [{"type": "result", "is_error": False, "duration_ms": 100}],
    )
    return root


def test_aggregate_find_runs_and_summaries(tmp_path):
    root = _make_run_tree(tmp_path)
    runs = aggregate.find_runs(str(root))
    assert [(t, r) for t, r, _ in runs] == [("t1", 0), ("t1", 1)]
    rows = aggregate.aggregate_eval_summaries(runs)
    # run_0 -> 2 rows, run_1 -> 1 MISSING placeholder row.
    results = sorted(r["result"] for r in rows)
    assert results == ["FAIL", "MISSING", "PASS"]


def test_aggregate_stream_diagnostics(tmp_path):
    root = _make_run_tree(tmp_path)
    runs = aggregate.find_runs(str(root))
    rows = aggregate.aggregate_stream_diagnostics(runs)
    by_run = {(r["trace_id"], r["run_id"]): r for r in rows}
    assert by_run[("t1", 0)]["outcome"] == "success"
    assert by_run[("t1", 1)]["outcome"] == "missing_file"


def test_aggregate_write_csv(tmp_path):
    path = tmp_path / "o.csv"
    aggregate.write_csv(
        str(path), [{"a": 1, "b": 2, "extra": 9}], ["a", "b"]
    )  # extra ignored
    text = path.read_text()
    assert "a,b" in text
    assert "1,2" in text


def test_aggregate_detect_steps_read_tool(tmp_path):
    steps = set()
    aggregate._detect_steps(
        {"readToolCall": {"args": {"path": "cat/foo_metrics.json"}}}, steps
    )
    assert "step7_subagent_findings" in steps
    steps = set()
    aggregate._detect_steps(
        {"readToolCall": {"args": {"path": "category_manifest.json"}}}, steps
    )
    assert "step2_5_prepare" in steps


def test_aggregate_main_end_to_end(tmp_path, monkeypatch, capsys):
    root = _make_run_tree(tmp_path / "results")
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    monkeypatch.setattr(aggregate, "RESULTS_ROOT", str(root))
    monkeypatch.setattr(aggregate, "OUTPUT_DIR", str(out_dir))
    aggregate.main()
    # main writes six CSVs into OUTPUT_DIR.
    written = {p.name for p in out_dir.iterdir()}
    assert "aggregated_results.csv" in written
    assert "pass_rate_summary.csv" in written
    assert "stream_diagnostics.csv" in written


# ---------------------------------------------------------------------------
# run_post_processing: collection helpers + main driver
# ---------------------------------------------------------------------------


def _make_post_results_tree(root):
    t1 = root / "t1"
    (t1 / "run_0").mkdir(parents=True)
    with open(t1 / "run_0" / "eval_results.csv", "w") as f:
        f.write(",".join(post.CSV_COLUMNS) + "\n")
        f.write("e1,Workflow,thing,PASS,ok,,\n")
        f.write("e2,Workflow,Kernel Fusion Insights,FAIL,bad detail,,\n")
    with open(t1 / "run_0" / "eval_summary.csv", "w") as f:
        f.write("index,result\ne1,PASS\n")
    _write_ndjson(
        t1 / "run_0" / "analysis_stream.ndjson",
        [{"type": "result", "is_error": False, "duration_ms": 1500}],
    )
    return root


def test_post_reaggregate_from_sources(tmp_path):
    root = _make_post_results_tree(tmp_path)
    rows = post.reaggregate_from_sources(str(root))
    assert len(rows) == 2
    assert {r["result"] for r in rows} == {"PASS", "FAIL"}
    assert rows[0]["trace_id"] == "t1"


def test_post_parse_stream_diagnostics(tmp_path):
    root = _make_post_results_tree(tmp_path)
    rows = post.parse_stream_diagnostics(str(root))
    assert rows[0]["outcome"] == "success"
    assert rows[0]["duration_ms"] == 1500


def test_post_build_trace_meta_and_load_csv(tmp_path):
    csv_path = tmp_path / "traces.csv"
    csv_path.write_text(
        "id,sub_category,platform\nt1,full_model,rocm\nt2,attention,rocm\n"
    )
    meta = post.build_trace_meta(str(csv_path))
    assert meta["t1"]["sub_category"] == "full_model"
    assert post.load_csv(str(csv_path))[0]["id"] == "t1"


def test_post_write_csv(tmp_path):
    path = tmp_path / "o.csv"
    post.write_csv(str(path), [{"a": "1", "b": "2"}], ["a", "b"])
    assert "a,b" in path.read_text()


def test_post_main_end_to_end(tmp_path, monkeypatch, capsys):
    results_root = _make_post_results_tree(tmp_path / "results")
    report_dir = tmp_path / "reports"
    agg_dir = report_dir / "aggregates"
    reproducers_dir = report_dir / "reproducers"
    latest_dir = tmp_path / "latest"
    agg_dir.mkdir(parents=True)
    traces_csv = tmp_path / "traces.csv"
    traces_csv.write_text("id,sub_category,platform\nt1,full_model,rocm\n")

    monkeypatch.setattr(post, "RESULTS_ROOT", str(results_root))
    monkeypatch.setattr(post, "REPORT_DIR", str(report_dir))
    monkeypatch.setattr(post, "AGG_DIR", str(agg_dir))
    monkeypatch.setattr(post, "REPRODUCERS_DIR", str(reproducers_dir))
    monkeypatch.setattr(post, "LATEST_DIR", str(latest_dir))
    monkeypatch.setattr(post, "TEST_TRACES_CSV", str(traces_csv))
    monkeypatch.setattr(post, "REPO_ROOT", str(tmp_path))
    monkeypatch.setattr(post, "RULES_PATH", _RULES_PATH)

    post.main()

    assert (report_dir / "pr_report.md").is_file()
    assert (report_dir / "fix_ticket_report.md").is_file()
    assert (agg_dir / "aggregated_results.csv").is_file()
    assert latest_dir.is_dir()
    # One failing issue -> one reproducer package built.
    assert reproducers_dir.is_dir()
