###############################################################################
# Copyright (c) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the analysis triage toolkit.

These are the safety net for the `feat/port-analysis-triage` refactor (see
`Claude_Docs/Plan/PLAN_triage-code-quality-refactor.md`, Part C). Triage
previously shipped with zero automated coverage. Everything here is hermetic:
tiny json / json.gz / csv fixtures are synthesized under pytest ``tmp_path`` with
no dependence on real trace trees.

Tests are written against the POST-refactor contract:
  * ``_load_trace_json`` is ``functools.lru_cache(maxsize=1)`` and delegates to
    ``DataLoader.load_data`` (strict UTF-8 via orjson) — so a repeat call returns
    the SAME object, and a corrupt file raises (a ``ValueError`` subclass) on
    every call because ``lru_cache`` does not memoize exceptions.
  * ``_events_of`` is a new helper collapsing the ``list`` / ``traceEvents`` idiom.
Tests exercising these behaviors may fail against the pre-refactor code; that is
expected and called out in the delivery report.
"""

import gzip
import json

import pytest

from TraceLens.Agent.Analysis.triage.checks import (
    ALL_CHECKS,
    _load_events,
    _load_gpu_timeline,
    _load_trace_json,
    _parse_float,
    _significant_rows,
    check_capture_graph_merge_fail,
    check_perf_report_command_incorrect,
    check_split_incorrect,
)
from TraceLens.Agent.Analysis.triage.postprocess import collect_findings
from TraceLens.Agent.Analysis.triage.runner import run_triage, write_detail_csv

# Registry size is load-bearing: the plan pins it and the reserved-stub /
# renumber checks below depend on it not drifting silently.
_EXPECTED_CHECK_COUNT = 36


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f)
    return str(path)


def _write_json_gz(path, obj):
    with gzip.open(path, "wt") as f:
        json.dump(obj, f)
    return str(path)


# ---------------------------------------------------------------------------
# _load_trace_json  (guards B1 + B3)
# ---------------------------------------------------------------------------


def test_load_trace_json_plain_json(tmp_path):
    p = _write_json(tmp_path / "trace.json", {"traceEvents": [{"cat": "kernel"}]})
    assert _load_trace_json(p) == {"traceEvents": [{"cat": "kernel"}]}


def test_load_trace_json_gzip(tmp_path):
    p = _write_json_gz(tmp_path / "trace.json.gz", [{"cat": "cpu_op"}])
    assert _load_trace_json(p) == [{"cat": "cpu_op"}]


def test_load_trace_json_lru_cache_returns_same_object(tmp_path):
    # POST-refactor: lru_cache(maxsize=1) keyed on the resolved path collapses a
    # repeat parse of the same file to a single object. Pre-refactor this is a
    # plain function and each call re-parses to a distinct object (test fails).
    p = _write_json(tmp_path / "trace.json", {"traceEvents": []})
    first = _load_trace_json(p)
    second = _load_trace_json(p)
    assert first is second


def test_load_trace_json_corrupt_raises_and_reraises(tmp_path):
    # Truncated / invalid JSON must raise a ValueError subclass (json's own
    # JSONDecodeError, or orjson's which subclasses both json.JSONDecodeError
    # and ValueError). check_corrupt_json depends on this raise.
    p = tmp_path / "corrupt.json"
    p.write_text('{"traceEvents": [')
    with pytest.raises((json.JSONDecodeError, ValueError)):
        _load_trace_json(str(p))
    # lru_cache does not memoize exceptions — the second call re-raises too.
    with pytest.raises((json.JSONDecodeError, ValueError)):
        _load_trace_json(str(p))


# ---------------------------------------------------------------------------
# _load_events  (guards the checks.py:322 `except (OSError, ValueError)` swallow)
# ---------------------------------------------------------------------------


def test_load_events_unknown_type_returns_none(tmp_path):
    # A plain .gz (not .json.gz) is reachable via a user-supplied trace_path;
    # DataLoader raises ValueError("Unknown file type", ...) — _load_events must
    # swallow it to None, not propagate and crash the detection checks.
    p = tmp_path / "x.gz"
    p.write_bytes(b"not a recognized trace")
    assert _load_events(str(p)) is None


def test_load_events_corrupt_json_returns_none(tmp_path):
    # A truncated .json makes _load_trace_json raise a ValueError subclass
    # (contrast test_load_trace_json_corrupt_raises_and_reraises); _load_events
    # is the swallowing wrapper, so it must convert that raise into None.
    p = tmp_path / "corrupt.json"
    p.write_text('{"traceEvents": [')
    with pytest.raises((json.JSONDecodeError, ValueError)):
        _load_trace_json(str(p))
    assert _load_events(str(p)) is None


# ---------------------------------------------------------------------------
# _events_of  (new helper, guards A2)
# ---------------------------------------------------------------------------


def test_events_of():
    # Imported locally: pre-refactor the symbol does not exist yet, so only this
    # test fails rather than breaking module collection for the whole file.
    from TraceLens.Agent.Analysis.triage.checks import _events_of

    assert _events_of([{"cat": "kernel"}]) == [{"cat": "kernel"}]
    assert _events_of({"traceEvents": [{"a": 1}]}) == [{"a": 1}]
    assert _events_of({"no_events_key": True}) == []


# ---------------------------------------------------------------------------
# _parse_float  (guards A1 + A3 + B6)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, None),
        ("", None),
        ("nan", None),
        ("none", None),
        ("null", None),
        ("NaN", None),
        ("garbage", None),
        ("1.5", 1.5),
        (1.5, 1.5),
        (0, 0.0),
    ],
)
def test_parse_float(value, expected):
    assert _parse_float(value) == expected


# ---------------------------------------------------------------------------
# _significant_rows  (guards A1 threshold constants)
# ---------------------------------------------------------------------------


def test_significant_rows_boundary():
    # Significant == Cumulative Percentage (%) <= 90 AND Percentage (%) >= 4.
    def row(name, pct, cum):
        return {
            "name": name,
            "Percentage (%)": pct,
            "Cumulative Percentage (%)": cum,
        }

    rows = [
        row("on_boundary", "4", "90"),  # both inclusive boundaries -> in
        row("cum_over", "10", "90.1"),  # cumulative just over -> out
        row("pct_under", "3.9", "50"),  # percentage just under -> out
        row("well_inside", "50", "10"),  # comfortably significant -> in
        row("unparseable", "garbage", "20"),  # bad cell -> skipped
    ]
    kept = {r["name"] for r in _significant_rows(rows)}
    assert kept == {"on_boundary", "well_inside"}


# ---------------------------------------------------------------------------
# _load_gpu_timeline  (guards A3 reuse of _parse_float)
# ---------------------------------------------------------------------------


def test_load_gpu_timeline_skips_unparseable(tmp_path):
    csv_dir = tmp_path / "perf_report_csvs"
    csv_dir.mkdir()
    (csv_dir / "gpu_timeline.csv").write_text(
        "type,percent\n"
        "idle_time,20.5\n"
        "compute,50\n"
        "broken,notanumber\n"
        ",30\n"  # blank type -> skipped
    )
    timeline = _load_gpu_timeline(str(tmp_path))
    assert timeline == {"idle_time": 20.5, "compute": 50.0}


def test_load_gpu_timeline_missing_file(tmp_path):
    assert _load_gpu_timeline(str(tmp_path)) is None


# ---------------------------------------------------------------------------
# Registry integrity  (guards renumber drift + B5 reserved stubs)
# ---------------------------------------------------------------------------


def test_all_checks_count():
    assert len(ALL_CHECKS) == _EXPECTED_CHECK_COUNT


def test_sublabels_unique_and_nonempty():
    sublabels = [spec.sublabel for spec in ALL_CHECKS]
    assert all(sublabels), "every check must carry a non-empty sublabel"
    assert len(sublabels) == len(set(sublabels)), "sublabels must be unique"


def test_implies_failures_resolve_to_real_sublabels():
    known = {spec.sublabel for spec in ALL_CHECKS}
    for spec in ALL_CHECKS:
        for target in spec.implies_failures:
            assert target in known, (
                f"{spec.sublabel} implies unknown sublabel {target!r} "
                "(renumber drift?)"
            )


def test_reserved_stubs_return_none(tmp_path):
    # 2d / 3a / 3c are intentional reserved-slot stubs; deleting them would force
    # a sublabel renumber. They must stay inert.
    run_dir = str(tmp_path)
    assert check_split_incorrect(run_dir, None) is None
    assert check_perf_report_command_incorrect(run_dir, None) is None
    assert check_capture_graph_merge_fail(run_dir, None) is None


# ---------------------------------------------------------------------------
# End-to-end smoke  (guards everything)
# ---------------------------------------------------------------------------


def test_run_triage_smoke_trace_missing(tmp_path):
    # An empty run dir has no resolvable trace, so 1a TRACE_MISSING must fire and
    # run_triage must complete without raising.
    findings = run_triage(str(tmp_path))
    assert any("1a_TRACE_MISSING" in f.tag for f in findings), (
        "expected a trace-missing finding; tags: " f"{[f.tag for f in findings]}"
    )


# ---------------------------------------------------------------------------
# runner <-> postprocess CSV seam  (guards the six-column contract)
# ---------------------------------------------------------------------------


def test_detail_csv_roundtrip_runner_to_postprocess(tmp_path):
    # write_detail_csv (runner) and collect_findings (postprocess) share an
    # unversioned CSV column contract. An empty run dir yields at least the
    # 1a TRACE_MISSING finding, which is enough to exercise the round-trip:
    # write it, read it back, and confirm the columns line up field-for-field.
    run_dir = str(tmp_path)
    findings = run_triage(run_dir)
    assert findings, "expected at least one finding to round-trip"
    write_detail_csv(findings, run_dir)

    collected, total_runs, unassessed = collect_findings([run_dir])
    assert total_runs == 1
    assert unassessed == 0  # a CSV was written, so this run was assessed

    by_tag = {f.diag_tag: f for f in collected}
    assert by_tag, "collect_findings read no rows back from the CSV"
    for src in findings:
        got = by_tag.get(src.tag)
        assert got is not None, f"tag {src.tag!r} lost across the CSV seam"
        assert got.category == src.category
        assert got.failure_mode == src.failure_mode
        assert got.evidence == src.evidence
        assert got.implied_by == ",".join(src.implied_by)


def test_collect_findings_unassessed_run_not_clean(tmp_path):
    # Two-run mapping: one run wrote a triage CSV (assessed, has findings), the
    # other never did (gracefully skipped). The skipped run must land in the
    # unassessed bucket, not be silently counted as clean.
    assessed = tmp_path / "assessed"
    assessed.mkdir()
    skipped = tmp_path / "skipped"
    skipped.mkdir()

    findings = run_triage(str(assessed))
    write_detail_csv(findings, str(assessed))

    collected, total_runs, unassessed = collect_findings([str(assessed), str(skipped)])
    assert total_runs == 2
    assert unassessed == 1
    runs_with_findings = len({f.run_dir for f in collected})
    assert runs_with_findings == 1
    assert total_runs - runs_with_findings - unassessed == 0  # zero clean runs
