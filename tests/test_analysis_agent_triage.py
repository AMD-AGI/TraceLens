###############################################################################
# Copyright (c) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit + behavioural tests for the analysis triage toolkit (single file).

Consolidates the whole triage test surface: the loader / helper contracts, the
1x-6x detection / reporting / infra / geak checks, the runner orchestration, the
postprocess aggregator, and the runner->postprocess CSV seam. The on-disk
fixture builders that every group shares are inlined at the top (they mirror the
exact directory / file shapes the ``triage`` package reads).

Contracts exercised by the loader group:
  * ``_load_trace_json`` is ``functools.lru_cache(maxsize=1)`` and delegates to
    ``DataLoader.load_data`` (strict UTF-8 via orjson) — so a repeat call returns
    the SAME object, and a corrupt file raises (a ``ValueError`` subclass) on
    every call because ``lru_cache`` does not memoize exceptions.
  * ``_events_of`` collapses the ``list`` / ``traceEvents`` idiom.
  * ``_load_events`` is the swallowing wrapper that converts loader errors to
    ``None`` so detection checks degrade gracefully.
"""

import csv
import gzip
import json
import os
import sys
import tarfile
from pathlib import Path

import pytest

from TraceLens.Agent.Analysis.triage import checks, postprocess, runner
from TraceLens.Agent.Analysis.triage.checks import (
    ALL_CHECKS,
    CheckSpec,
    Finding,
    FindingDraft,
    _apply_path_remaps,
    _events_of,
    _first_load_capture_event_set,
    _load_events,
    _load_gpu_timeline,
    _load_manifest,
    _load_trace_json,
    _load_unified_perf_summary,
    _parse_float,
    _path_remaps,
    _resolve_path,
    _significant_rows,
    _trace_input_dir,
    _violator_evidence,
    check_capture_graph_merge_fail,
    check_capture_missing,
    check_corrupt_json,
    check_disk_full,
    check_docker_missing,
    check_gpu_graph_replay,
    check_high_idle,
    check_hipgraph_launch_in_report,
    check_inference_annotation_missing,
    check_kernel_candidates_missing,
    check_missing_cpu_op_shapes,
    check_missing_dep,
    check_nfs_stale,
    check_no_gpu_kernels,
    check_output_incomplete,
    check_perf_report_command_incorrect,
    check_prefix_fail,
    check_report_too_small,
    check_resource_exhausted,
    check_roofline_pct_missing,
    check_runtime_instability,
    check_shape_profiler_missing,
    check_split_incorrect,
    check_split_low_gpu_kernels,
    check_split_trace_missing,
    check_ssh_fail,
    check_step1_fail,
    check_step2_5_fail,
    check_subagent_budget,
    check_synthetic_op_significant,
    check_tbs_tflops_missing,
    check_tl_not_installed,
    check_trace_missing,
    check_trace_size,
    check_unclassified_op_significant,
    check_zero_pct_ops,
    resolve_capture_folder,
    resolve_main_trace_path,
    resolve_split_trace_dir,
    resolve_trace_path,
    stream_contains,
    stream_find,
    stream_lines,
)
from TraceLens.Agent.Analysis.triage.postprocess import (
    TriageFinding,
    _collectible_paths,
    aggregate,
    build_reproducer_packages,
    collect_findings,
    discover_triage_csvs,
    extract_action_keys,
    extract_model_name,
    load_from_mapping,
    pick_reproducers,
    sanitize_filename,
    write_aggregated_csv,
    write_summary_report,
)
from TraceLens.Agent.Analysis.triage.runner import (
    _auto_detect_log,
    _auto_detect_stream,
    main,
    run_geak_triage,
    run_triage,
    write_detail_csv,
    write_diag_txt,
)

# ---------------------------------------------------------------------------
# Fixture-builder constants — exact column / field contracts the loaders read
# ---------------------------------------------------------------------------

# runner.write_detail_csv writes these; postprocess.collect_findings reads them.
_TRIAGE_CSV_COLUMNS = [
    "DIAG Tag",
    "Category",
    "Failure Mode",
    "Evidence",
    "Remedy",
    "Implied By",
]

# checks._load_gpu_timeline reads "type" and "percent".
_GPU_TIMELINE_FIELDS = ("type", "percent")


@pytest.fixture(autouse=True)
def _reset_triage_caches(monkeypatch):
    """Isolate the two triage-module caches so tests don't leak into each other.

    ``checks._load_trace_json`` is ``lru_cache(maxsize=1)``: rewriting a file at
    the same path and re-reading returns stale content. ``checks._PATH_REMAPS_CACHE``
    memoizes ``TRACELENS_PATH_REMAPS`` once, so a remap set in one test would bleed
    into the next. Clear both before and after each test, and strip the env var so
    remap tests start from a clean environment.
    """
    monkeypatch.delenv("TRACELENS_PATH_REMAPS", raising=False)
    checks._load_trace_json.cache_clear()
    checks._PATH_REMAPS_CACHE = None
    yield
    checks._load_trace_json.cache_clear()
    checks._PATH_REMAPS_CACHE = None


# ---------------------------------------------------------------------------
# Primitive writers
# ---------------------------------------------------------------------------


def write_text(path, s):
    """Write ``s`` to ``path``, creating parent dirs. Returns the path as str."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(s)
    return str(path)


def write_json(path, obj):
    """Write ``obj`` as JSON to ``path``, creating parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f)
    return str(path)


def write_json_gz(path, obj):
    """Write ``obj`` as gzipped JSON to ``path``, creating parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        json.dump(obj, f)
    return str(path)


def write_stream(path, lines):
    """Write ``lines`` as a stream file (one record per line).

    Each element that isn't a ``str`` is JSON-encoded, so ndjson tests can pass
    dict records directly and plain-log tests can pass strings.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for line in lines:
            if not isinstance(line, str):
                line = json.dumps(line)
            f.write(line if line.endswith("\n") else line + "\n")
    return str(path)


# ---------------------------------------------------------------------------
# CSV writers — headers derived from the dict rows the caller supplies
# ---------------------------------------------------------------------------


def _write_dict_rows(path, rows, default_fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    if not fieldnames and default_fields:
        fieldnames = list(default_fields)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return str(path)


def write_perf_csv(path, rows):
    """Write ``perf_report_csvs/unified_perf_summary.csv`` from dict rows.

    Columns are the union of keys across ``rows`` in first-seen order; the
    caller supplies the exact header strings the checks read (e.g. ``name``,
    ``Percentage (%)``, ``Cumulative Percentage (%)``, ``op category``,
    ``Kernel Time (µs)_mean``, ``Kernel Time (µs)_std``, ``TB/s_mean``,
    ``TFLOPS/s_mean``, ``Pct Roofline_mean``).
    """
    return _write_dict_rows(path, rows)


def write_gpu_timeline(path, rows):
    """Write ``perf_report_csvs/gpu_timeline.csv`` from dict rows.

    ``checks._load_gpu_timeline`` reads the ``type`` and ``percent`` columns.
    """
    return _write_dict_rows(path, rows, default_fields=_GPU_TIMELINE_FIELDS)


# ---------------------------------------------------------------------------
# Composite builders
# ---------------------------------------------------------------------------


def make_run_dir(
    tmp_path,
    *,
    cmd_prefix=None,
    manifest=None,
    timeline_rows=None,
    perf_rows=None,
    split_gz=None,
    capture_events=None,
    main_trace_events=None,
    output_dirs=(),
    analysis_md=None,
):
    """Build only the requested pieces of a ``run_dir`` tree; return its ``Path``.

    The ``run_dir`` is ``<tmp_path>/run``; the sibling ``trace_input_manifest.json``
    and ``trace_input/`` (main trace + ``capture_traces/``) land directly under
    ``tmp_path``. Every keyword left as its default is not created.

    Args:
        cmd_prefix: Trace path written into ``cache/cmd_prefix.txt`` as
            ``--profile_json_path <cmd_prefix>``. For malformed / custom content,
            call ``write_text`` on the file directly instead.
        manifest: dict written to ``category_data/category_manifest.json``.
        timeline_rows: dict rows written to ``perf_report_csvs/gpu_timeline.csv``.
        perf_rows: dict rows written to
            ``perf_report_csvs/unified_perf_summary.csv``.
        split_gz: ``{filename: obj}`` written under ``trace_split/`` via
            ``write_json_gz`` (e.g. ``{"mixed_steady_state_0.json.gz": [...]}``).
        capture_events: event list written as
            ``trace_input/capture_traces/capture_0.json.gz`` (wrapped in
            ``{"traceEvents": ...}``). Triggers the trace_input manifest.
        main_trace_events: event list written as
            ``trace_input/model_TP-0_rank0.json.gz`` (wrapped in
            ``{"traceEvents": ...}``). Triggers the trace_input manifest.
        output_dirs: iterable of directory names to create under ``run_dir``
            (e.g. ``("metadata", "category_data", "system_findings",
            "category_findings")``).
        analysis_md: content written to ``analysis.md``.
    """
    tmp_path = Path(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    if cmd_prefix is not None:
        write_text(
            run_dir / "cache" / "cmd_prefix.txt",
            f"--profile_json_path {cmd_prefix}\n",
        )

    if manifest is not None:
        write_json(run_dir / "category_data" / "category_manifest.json", manifest)

    if timeline_rows is not None:
        write_gpu_timeline(
            run_dir / "perf_report_csvs" / "gpu_timeline.csv", timeline_rows
        )

    if perf_rows is not None:
        write_perf_csv(
            run_dir / "perf_report_csvs" / "unified_perf_summary.csv", perf_rows
        )

    if split_gz is not None:
        for fname, obj in split_gz.items():
            write_json_gz(run_dir / "trace_split" / fname, obj)

    if capture_events is not None or main_trace_events is not None:
        trace_input = tmp_path / "trace_input"
        trace_input.mkdir(parents=True, exist_ok=True)
        write_json(
            tmp_path / "trace_input_manifest.json", {"trace_input": str(trace_input)}
        )
        if main_trace_events is not None:
            write_json_gz(
                trace_input / "model_TP-0_rank0.json.gz",
                {"traceEvents": main_trace_events},
            )
        if capture_events is not None:
            write_json_gz(
                trace_input / "capture_traces" / "capture_0.json.gz",
                {"traceEvents": capture_events},
            )

    for d in output_dirs:
        (run_dir / d).mkdir(parents=True, exist_ok=True)

    if analysis_md is not None:
        write_text(run_dir / "analysis.md", analysis_md)

    return run_dir


def make_geak_session(tmp_path, *, phases, with_candidates):
    """Build a Hyperloom session tree for ``check_kernel_candidates_missing``.

    Writes ``session_breakdown.json`` whose ``phase_segments`` carry the given
    ``phases`` (use ``"KERNEL_AGENT"`` to reach the kernel phase). When
    ``with_candidates`` is True, also writes
    ``kernel-agent/runs/<id>/<hash>/kernel_candidates.json`` (the two-glob path
    the check expects). Returns the session dir ``Path``.
    """
    tmp_path = Path(tmp_path)
    session_dir = tmp_path / "session"
    session_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        session_dir / "session_breakdown.json",
        {"phase_segments": [{"phase": p} for p in phases]},
    )
    if with_candidates:
        write_json(
            session_dir
            / "kernel-agent"
            / "runs"
            / "run0"
            / "tlhash"
            / "kernel_candidates.json",
            {"candidates": []},
        )
    return session_dir


def make_triage_csv(run_dir, rows):
    """Write ``<run_dir>/triage_details.csv`` with the exact postprocess columns.

    ``rows`` are dicts keyed by any subset of ``_TRIAGE_CSV_COLUMNS`` (missing
    keys default to ""); the column set and order match
    ``runner.write_detail_csv`` / ``postprocess.collect_findings``.
    """
    path = Path(run_dir) / "triage_details.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_TRIAGE_CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in _TRIAGE_CSV_COLUMNS})
    return str(path)


###############################################################################
# Group 1 — loaders, _parse_float, _significant_rows, registry integrity,
# runner->postprocess CSV seam (originally test_analysis_triage.py)
###############################################################################

# Registry size is load-bearing: the reserved-stub and sublabel checks below
# depend on it not drifting silently.
_EXPECTED_CHECK_COUNT = 36

_PARSE_FLOAT_CASES = [
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
]


# ---------------------------------------------------------------------------
# _load_trace_json
# ---------------------------------------------------------------------------


def test_load_trace_json_plain_json(tmp_path):
    p = write_json(tmp_path / "trace.json", {"traceEvents": [{"cat": "kernel"}]})
    assert _load_trace_json(p) == {"traceEvents": [{"cat": "kernel"}]}


def test_load_trace_json_gzip(tmp_path):
    p = write_json_gz(tmp_path / "trace.json.gz", [{"cat": "cpu_op"}])
    assert _load_trace_json(p) == [{"cat": "cpu_op"}]


def test_load_trace_json_lru_cache_returns_same_object(tmp_path):
    # lru_cache(maxsize=1) keyed on the resolved path collapses a repeat parse of
    # the same file to a single object.
    p = write_json(tmp_path / "trace.json", {"traceEvents": []})
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
# _load_events
# ---------------------------------------------------------------------------


def test_load_events_unknown_type_returns_none(tmp_path):
    # A plain .gz (not .json.gz) is reachable via a user-supplied trace_path;
    # DataLoader raises ValueError("Unknown file type", ...) — _load_events must
    # swallow it to None, not propagate and crash the detection checks.
    p = tmp_path / "x.gz"
    p.write_bytes(b"not a recognized trace")
    assert _load_events(str(p)) is None


def test_load_events_corrupt_json_returns_none(tmp_path):
    # A truncated .json makes _load_trace_json raise a ValueError subclass;
    # _load_events is the swallowing wrapper, so it must convert that to None.
    p = tmp_path / "corrupt.json"
    p.write_text('{"traceEvents": [')
    with pytest.raises((json.JSONDecodeError, ValueError)):
        _load_trace_json(str(p))
    assert _load_events(str(p)) is None


# ---------------------------------------------------------------------------
# _events_of
# ---------------------------------------------------------------------------


def test_events_of():
    assert _events_of([{"cat": "kernel"}]) == [{"cat": "kernel"}]
    assert _events_of({"traceEvents": [{"a": 1}]}) == [{"a": 1}]
    assert _events_of({"no_events_key": True}) == []


# ---------------------------------------------------------------------------
# _parse_float
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value, expected", _PARSE_FLOAT_CASES)
def test_parse_float(value, expected):
    assert _parse_float(value) == expected


# ---------------------------------------------------------------------------
# _significant_rows
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
# _load_gpu_timeline
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
# Registry integrity
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
# End-to-end smoke
# ---------------------------------------------------------------------------


def test_run_triage_smoke_trace_missing(tmp_path):
    # An empty run dir has no resolvable trace, so 1a TRACE_MISSING must fire and
    # run_triage must complete without raising.
    findings = run_triage(str(tmp_path))
    assert any("1a_TRACE_MISSING" in f.tag for f in findings), (
        "expected a trace-missing finding; tags: " f"{[f.tag for f in findings]}"
    )


# ---------------------------------------------------------------------------
# runner <-> postprocess CSV seam
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


###############################################################################
# Group 2 — detection / quality checks (1x, 2x) and resolver / loader / stream
# helpers (originally test_triage_checks_detection.py)
###############################################################################


def _set_remaps(value):
    """Set ``TRACELENS_PATH_REMAPS`` and drop the memo so it recomputes.

    The autouse ``_reset_triage_caches`` fixture clears the cache and the env var
    after the test, so we only need to prime them here.
    """
    os.environ["TRACELENS_PATH_REMAPS"] = value
    checks._PATH_REMAPS_CACHE = None


# ---------------------------------------------------------------------------
# _path_remaps / _apply_path_remaps / _resolve_path
# ---------------------------------------------------------------------------


def test_path_remaps_empty_env():
    assert _path_remaps() == []


def test_path_remaps_parses_and_skips_malformed():
    # "bad" has no '=' (skipped); "=orphan" has an empty old prefix (skipped);
    # "x=" keeps an empty new prefix; whitespace around a pair is stripped.
    _set_remaps(" /old/=/new/ , bad , =orphan , x= ")
    assert _path_remaps() == [("/old/", "/new/"), ("x", "")]


def test_path_remaps_is_memoized():
    _set_remaps("/a/=/b/")
    assert _path_remaps() is _path_remaps()


def test_apply_path_remaps_yields_only_matching_prefix():
    _set_remaps("/old/=/new/")
    assert list(_apply_path_remaps("/old/x/y")) == ["/new/x/y"]
    assert list(_apply_path_remaps("/other/x")) == []
    assert list(_apply_path_remaps("")) == []


def test_resolve_path_original_then_remap_then_none():
    _set_remaps("/old/=/new/")
    # Predicate holds on the original path -> returned unchanged.
    assert _resolve_path("/direct", lambda p: True) == "/direct"
    # Original fails, the remapped candidate satisfies the predicate.
    assert _resolve_path("/old/ok", lambda p: p == "/new/ok") == "/new/ok"
    # Nothing satisfies the predicate -> None; empty path is short-circuited.
    assert _resolve_path("/old/x", lambda p: False) is None
    assert _resolve_path("", lambda p: True) is None


# ---------------------------------------------------------------------------
# resolve_trace_path
# ---------------------------------------------------------------------------


def test_resolve_trace_path_from_cmd_prefix(tmp_path):
    trace = write_json_gz(tmp_path / "trace.json.gz", [{"cat": "kernel"}])
    run_dir = make_run_dir(tmp_path, cmd_prefix=trace)
    assert resolve_trace_path(str(run_dir)) == trace


def test_resolve_trace_path_cmd_prefix_absent_falls_back_to_manifest(tmp_path):
    # cmd_prefix path does not exist -> _resolve_path returns None, so the
    # category_manifest trace_path is consulted (and returned even if absent).
    run_dir = make_run_dir(
        tmp_path,
        cmd_prefix="/does/not/exist.json.gz",
        manifest={"trace_path": "/manifest/trace.json.gz"},
    )
    assert resolve_trace_path(str(run_dir)) == "/manifest/trace.json.gz"


def test_resolve_trace_path_none_when_nothing_wired(tmp_path):
    run_dir = make_run_dir(tmp_path)
    assert resolve_trace_path(str(run_dir)) is None


# ---------------------------------------------------------------------------
# _trace_input_dir / resolve_main_trace_path / resolve_capture_folder
# ---------------------------------------------------------------------------


def test_trace_input_dir_from_sibling_manifest(tmp_path):
    run_dir = make_run_dir(tmp_path, main_trace_events=[{"cat": "kernel"}])
    resolved = _trace_input_dir(str(run_dir))
    assert resolved == str(tmp_path / "trace_input")


def test_trace_input_dir_missing_manifest(tmp_path):
    run_dir = make_run_dir(tmp_path)
    assert _trace_input_dir(str(run_dir)) is None


def test_trace_input_dir_malformed_manifest(tmp_path):
    run_dir = make_run_dir(tmp_path)
    write_text(tmp_path / "trace_input_manifest.json", "{not json")
    assert _trace_input_dir(str(run_dir)) is None


def test_resolve_main_trace_path_from_trace_input(tmp_path):
    run_dir = make_run_dir(tmp_path, main_trace_events=[{"cat": "kernel"}])
    main = resolve_main_trace_path(str(run_dir))
    assert main is not None
    assert main.endswith("model_TP-0_rank0.json.gz")


def test_resolve_main_trace_path_run_scoped_fallback(tmp_path):
    # No trace_input manifest: the recursive fallback globs *rank0* under
    # trace_split within this run only.
    run_dir = make_run_dir(
        tmp_path, split_gz={"model_rank0.json.gz": [{"cat": "kernel"}]}
    )
    main = resolve_main_trace_path(str(run_dir))
    assert main is not None
    assert main.endswith("model_rank0.json.gz")


def test_resolve_main_trace_path_none(tmp_path):
    run_dir = make_run_dir(tmp_path)
    assert resolve_main_trace_path(str(run_dir)) is None


def test_resolve_capture_folder_present(tmp_path):
    run_dir = make_run_dir(tmp_path, capture_events=[{"cat": "cpu_op"}])
    folder = resolve_capture_folder(str(run_dir))
    assert folder == str(tmp_path / "trace_input" / "capture_traces")


def test_resolve_capture_folder_none_without_trace_input(tmp_path):
    run_dir = make_run_dir(tmp_path)
    assert resolve_capture_folder(str(run_dir)) is None


def test_resolve_split_trace_dir(tmp_path):
    run_dir = make_run_dir(tmp_path, split_gz={"mixed_steady_state_0.json.gz": []})
    assert resolve_split_trace_dir(str(run_dir)) == str(run_dir / "trace_split")
    empty = make_run_dir(tmp_path / "other")
    assert resolve_split_trace_dir(str(empty)) is None


# ---------------------------------------------------------------------------
# _load_manifest / _load_unified_perf_summary / _first_load_capture_event_set
# ---------------------------------------------------------------------------


def test_load_manifest_present_and_absent(tmp_path):
    run_dir = make_run_dir(tmp_path, manifest={"categories": [{"name": "gemm"}]})
    assert _load_manifest(str(run_dir)) == {"categories": [{"name": "gemm"}]}
    assert _load_manifest(str(make_run_dir(tmp_path / "bare"))) is None


def test_load_unified_perf_summary_present_and_absent(tmp_path):
    run_dir = make_run_dir(
        tmp_path,
        perf_rows=[{"name": "gemm", "Percentage (%)": "50"}],
    )
    rows = _load_unified_perf_summary(str(run_dir))
    assert rows == [{"name": "gemm", "Percentage (%)": "50"}]
    assert _load_unified_perf_summary(str(make_run_dir(tmp_path / "bare"))) is None


def test_first_load_capture_event_set_skips_corrupt_and_metadata(tmp_path):
    folder = tmp_path / "capture_traces"
    # Metadata files are skipped by name; a corrupt gz is skipped by the
    # except-continue; the good file wins.
    write_json(folder / "execution_details.json", {"ignored": True})
    write_text(folder / "a_capture.json.gz", "not gzip at all")
    write_json_gz(folder / "b_capture.json.gz", {"traceEvents": [{"cat": "cpu_op"}]})
    events = _first_load_capture_event_set(str(folder))
    assert events == [{"cat": "cpu_op"}]


def test_first_load_capture_event_set_none_for_missing_folder(tmp_path):
    assert _first_load_capture_event_set(str(tmp_path / "nope")) is None


# ---------------------------------------------------------------------------
# _violator_evidence
# ---------------------------------------------------------------------------


def test_violator_evidence_with_and_without_names():
    rows = [{"name": "a"}, {"name": "b"}, {"name": "c"}, {"name": "d"}]
    assert _violator_evidence("ops", rows) == "4 ops (e.g. a, b, c)"
    assert _violator_evidence("ops", [{"other": 1}]) == "1 ops"


# ---------------------------------------------------------------------------
# stream helpers
# ---------------------------------------------------------------------------


def test_stream_lines_strips_sdk_prefix_and_handles_missing(tmp_path):
    path = write_stream(tmp_path / "s.log", ["[claude-sdk] hello", "world"])
    lines = [line.rstrip("\n") for line in stream_lines(path)]
    assert lines == ["hello", "world"]
    assert list(stream_lines(None)) == []
    assert list(stream_lines(str(tmp_path / "absent.log"))) == []


def test_stream_contains_is_case_insensitive(tmp_path):
    path = write_stream(tmp_path / "s.log", ["some GPU_GRAPH_REPLAY line"])
    assert stream_contains(path, r"gpu_graph_replay") is True
    assert stream_contains(path, r"nomatch") is False


def test_stream_find_returns_first_truncated_match(tmp_path):
    long_line = "boom " + "x" * 400
    path = write_stream(tmp_path / "s.log", ["clean", long_line])
    found = stream_find(path, r"boom")
    assert found is not None
    assert len(found) == 200
    assert stream_find(path, r"nomatch") is None


# ---------------------------------------------------------------------------
# 1a: check_trace_missing — every branch
# ---------------------------------------------------------------------------


def test_trace_missing_resolved_present_returns_none(tmp_path):
    trace = write_json_gz(tmp_path / "trace.json.gz", [{"cat": "kernel"}])
    run_dir = make_run_dir(tmp_path, cmd_prefix=trace)
    assert check_trace_missing(str(run_dir), None) is None


def test_trace_missing_resolved_but_absent(tmp_path):
    run_dir = make_run_dir(tmp_path, manifest={"trace_path": "/gone/trace.json.gz"})
    finding = check_trace_missing(str(run_dir), None)
    assert isinstance(finding, FindingDraft)
    assert finding.failure_mode == "Trace files missing"
    assert "resolved but does not exist on disk" in finding.evidence
    assert "/gone/trace.json.gz" in finding.evidence


def test_trace_missing_trace_split_present(tmp_path):
    run_dir = make_run_dir(
        tmp_path, split_gz={"mixed_steady_state_0.json.gz": [{"cat": "kernel"}]}
    )
    finding = check_trace_missing(str(run_dir), None)
    assert finding is not None
    assert finding.failure_mode == "Trace files missing (warning: trace_split present)"
    assert "trace_split/ contains" in finding.evidence


def test_trace_missing_main_trace_present(tmp_path):
    run_dir = make_run_dir(tmp_path, main_trace_events=[{"cat": "kernel"}])
    finding = check_trace_missing(str(run_dir), None)
    assert finding is not None
    assert finding.failure_mode == "Trace files missing"
    assert "Main trace exists" in finding.evidence


def test_trace_missing_recursive_scan_empty(tmp_path):
    run_dir = make_run_dir(tmp_path)
    finding = check_trace_missing(str(run_dir), None)
    assert finding is not None
    assert finding.failure_mode == "Trace files missing"
    assert "returned empty / no main trace resolvable" in finding.evidence


def test_trace_missing_nearby_glob_oserror_swallowed(tmp_path, monkeypatch):
    # The trace_split branch calls _nearby_traces, whose glob("*.json.gz") is
    # wrapped in except OSError. Raise only for that pattern so the split glob
    # ("*.gz") still finds the file and the finding still fires.
    run_dir = make_run_dir(
        tmp_path, split_gz={"mixed_steady_state_0.json.gz": [{"cat": "kernel"}]}
    )
    real_glob = checks.glob.glob

    def fake_glob(pattern, *args, **kwargs):
        if pattern.endswith("*.json.gz"):
            raise OSError("nearby scan failed")
        return real_glob(pattern, *args, **kwargs)

    monkeypatch.setattr(checks.glob, "glob", fake_glob)
    finding = check_trace_missing(str(run_dir), None)
    assert finding is not None
    assert finding.failure_mode == "Trace files missing (warning: trace_split present)"
    assert "nearby *.json.gz" not in finding.evidence


def test_trace_missing_appends_nearby_traces(tmp_path):
    # _with_nearby appends any *.json.gz sitting exactly 6 levels above run_dir.
    # Nest run_dir so that root six levels up lands back inside tmp_path, and drop
    # a stray trace there; the trace_split finding must then carry the nearby list.
    nearby_root = tmp_path / "a"
    stray = write_json_gz(nearby_root / "stray.json.gz", [{"cat": "kernel"}])
    run_dir = nearby_root / "b" / "c" / "d" / "e" / "f" / "run"
    write_json_gz(run_dir / "trace_split" / "mixed_steady_state_0.json.gz", [])
    finding = check_trace_missing(str(run_dir), None)
    assert finding is not None
    assert "nearby *.json.gz" in finding.evidence
    assert stray in finding.evidence


def test_trace_missing_recursive_walk_oserror_swallowed(tmp_path, monkeypatch):
    # The final branch calls _recursive_traces, whose os.walk is wrapped in
    # except OSError. Raising it must degrade gracefully to an empty listing.
    run_dir = make_run_dir(tmp_path)

    def fake_walk(*args, **kwargs):
        raise OSError("walk failed")

    monkeypatch.setattr(checks.os, "walk", fake_walk)
    finding = check_trace_missing(str(run_dir), None)
    assert finding is not None
    assert "returned empty / no main trace resolvable" in finding.evidence


# ---------------------------------------------------------------------------
# 1b: check_trace_size
# ---------------------------------------------------------------------------


def test_trace_size_too_small(tmp_path):
    trace = write_text(tmp_path / "tiny.json.gz", "x" * 10)
    run_dir = make_run_dir(tmp_path, cmd_prefix=trace)
    finding = check_trace_size(str(run_dir), None)
    assert finding is not None
    assert finding.failure_mode == "Trace files too small (< 100KB)"


def test_trace_size_normal_returns_none(tmp_path):
    trace = write_text(tmp_path / "ok.json.gz", "x" * 100_000)
    run_dir = make_run_dir(tmp_path, cmd_prefix=trace)
    assert check_trace_size(str(run_dir), None) is None


def test_trace_size_no_trace_returns_none(tmp_path):
    run_dir = make_run_dir(tmp_path)
    assert check_trace_size(str(run_dir), None) is None


# ---------------------------------------------------------------------------
# 1c: check_no_gpu_kernels
# ---------------------------------------------------------------------------


def test_no_gpu_kernels_fires(tmp_path):
    trace = write_json_gz(tmp_path / "t.json.gz", [{"cat": "cpu_op"}])
    run_dir = make_run_dir(tmp_path, cmd_prefix=trace)
    finding = check_no_gpu_kernels(str(run_dir), None)
    assert finding is not None
    assert finding.failure_mode == "No GPU kernel events in trace"


def test_no_gpu_kernels_present_returns_none(tmp_path):
    trace = write_json_gz(tmp_path / "t.json.gz", [{"cat": "kernel"}])
    run_dir = make_run_dir(tmp_path, cmd_prefix=trace)
    assert check_no_gpu_kernels(str(run_dir), None) is None


def test_no_gpu_kernels_no_trace_returns_none(tmp_path):
    run_dir = make_run_dir(tmp_path)
    assert check_no_gpu_kernels(str(run_dir), None) is None


# ---------------------------------------------------------------------------
# 1d: check_capture_missing
# ---------------------------------------------------------------------------


def test_capture_missing_present_returns_none(tmp_path):
    run_dir = make_run_dir(tmp_path, capture_events=[{"cat": "cpu_op"}])
    assert check_capture_missing(str(run_dir), None) is None


def test_capture_missing_empty_capture_folder_fires(tmp_path):
    # trace_input resolves, but capture_traces/ holds no *json.gz.
    trace_input = tmp_path / "trace_input"
    (trace_input / "capture_traces").mkdir(parents=True)
    write_json(
        tmp_path / "trace_input_manifest.json", {"trace_input": str(trace_input)}
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    finding = check_capture_missing(str(run_dir), None)
    assert finding is not None
    assert finding.failure_mode == "Graph capture traces missing"


def test_capture_missing_via_manifest_path(tmp_path):
    run_dir = make_run_dir(
        tmp_path, manifest={"capture_folder_path": "/no/such/capture"}
    )
    finding = check_capture_missing(str(run_dir), None)
    assert finding is not None
    assert finding.failure_mode == "Graph capture traces missing"


def test_capture_missing_no_manifest_returns_none(tmp_path):
    run_dir = make_run_dir(tmp_path)
    assert check_capture_missing(str(run_dir), None) is None


def test_capture_missing_manifest_without_capture_key_returns_none(tmp_path):
    # Manifest present but no capture_folder_path -> falls through to None.
    run_dir = make_run_dir(tmp_path, manifest={"categories": []})
    assert check_capture_missing(str(run_dir), None) is None


# ---------------------------------------------------------------------------
# 2a: check_missing_cpu_op_shapes
# ---------------------------------------------------------------------------


def test_missing_cpu_op_shapes_zero_cpu_ops(tmp_path):
    run_dir = make_run_dir(tmp_path, capture_events=[{"cat": "kernel"}])
    finding = check_missing_cpu_op_shapes(str(run_dir), None)
    assert finding is not None
    assert "zero cpu_op events" in finding.evidence


def test_missing_cpu_op_shapes_too_few_with_shapes(tmp_path):
    run_dir = make_run_dir(
        tmp_path, capture_events=[{"cat": "cpu_op"} for _ in range(5)]
    )
    finding = check_missing_cpu_op_shapes(str(run_dir), None)
    assert finding is not None
    assert "carry 'Input Dims'" in finding.evidence


def test_missing_cpu_op_shapes_enough_returns_none(tmp_path):
    events = [{"cat": "cpu_op", "args": {"Input Dims": [[1]]}} for _ in range(10)]
    run_dir = make_run_dir(tmp_path, capture_events=events)
    assert check_missing_cpu_op_shapes(str(run_dir), None) is None


def test_missing_cpu_op_shapes_no_capture_returns_none(tmp_path):
    run_dir = make_run_dir(tmp_path)
    assert check_missing_cpu_op_shapes(str(run_dir), None) is None


def test_missing_cpu_op_shapes_capture_folder_unloadable_returns_none(tmp_path):
    # capture_traces/ resolves but has no loadable *.json.gz, so
    # _first_load_capture_event_set returns None and the check degrades to None.
    trace_input = tmp_path / "trace_input"
    (trace_input / "capture_traces").mkdir(parents=True)
    write_json(
        tmp_path / "trace_input_manifest.json", {"trace_input": str(trace_input)}
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    assert check_missing_cpu_op_shapes(str(run_dir), None) is None


# ---------------------------------------------------------------------------
# 2b: check_inference_annotation_missing
# ---------------------------------------------------------------------------


def test_inference_annotation_no_main_trace(tmp_path):
    run_dir = make_run_dir(tmp_path)
    finding = check_inference_annotation_missing(str(run_dir), None)
    assert finding is not None
    assert finding.failure_mode == "Main inference trace missing "


def test_inference_annotation_present_returns_none(tmp_path):
    events = [{"cat": "user_annotation", "name": "execute_step"}]
    run_dir = make_run_dir(tmp_path, main_trace_events=events)
    assert check_inference_annotation_missing(str(run_dir), None) is None


def test_inference_annotation_missing_execute_events(tmp_path):
    run_dir = make_run_dir(tmp_path, main_trace_events=[{"cat": "kernel"}])
    finding = check_inference_annotation_missing(str(run_dir), None)
    assert finding is not None
    assert finding.failure_mode == (
        "Main inference trace missing annotations in inference mode"
    )


def test_inference_annotation_corrupt_main_trace(tmp_path):
    # main trace is glob-restricted to *.json*, so a corrupt .json triggers the
    # narrow (OSError, JSONDecodeError) branch, not check_corrupt_json's path.
    trace_input = tmp_path / "trace_input"
    write_text(trace_input / "model_TP-0_rank0.json", "{not valid json")
    write_json(
        tmp_path / "trace_input_manifest.json", {"trace_input": str(trace_input)}
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    finding = check_inference_annotation_missing(str(run_dir), None)
    assert finding is not None
    assert "corrupted/invalid JSON" in finding.failure_mode


# ---------------------------------------------------------------------------
# 2c: check_split_trace_missing
# ---------------------------------------------------------------------------


def test_split_trace_missing_no_dir(tmp_path):
    run_dir = make_run_dir(tmp_path)
    finding = check_split_trace_missing(str(run_dir), None)
    assert finding is not None
    assert "does not exist" in finding.evidence


def test_split_trace_missing_no_steady_state(tmp_path):
    run_dir = make_run_dir(tmp_path, split_gz={"other_0.json.gz": []})
    finding = check_split_trace_missing(str(run_dir), None)
    assert finding is not None
    assert "no mixed_steady_state" in finding.evidence


def test_split_trace_present_returns_none(tmp_path):
    run_dir = make_run_dir(tmp_path, split_gz={"mixed_steady_state_0.json.gz": []})
    assert check_split_trace_missing(str(run_dir), None) is None


# ---------------------------------------------------------------------------
# 2d: check_split_incorrect (reserved stub)
# ---------------------------------------------------------------------------


def test_split_incorrect_is_inert(tmp_path):
    assert check_split_incorrect(str(tmp_path), None) is None


# ---------------------------------------------------------------------------
# 2e: check_shape_profiler_missing
# ---------------------------------------------------------------------------


def test_shape_profiler_missing_fires_from_capture(tmp_path):
    events = [{"cat": "cpu_op", "name": "sglang::forward"}]
    run_dir = make_run_dir(tmp_path, capture_events=events)
    finding = check_shape_profiler_missing(str(run_dir), None)
    assert finding is not None
    assert finding.failure_mode == "Trace was profiled without SGLang patches"


def test_shape_profiler_present_returns_none(tmp_path):
    events = [
        {"cat": "cpu_op", "name": "sglang::forward"},
        {"cat": "python_function", "name": "kernel_shape_profiler_run"},
    ]
    run_dir = make_run_dir(tmp_path, capture_events=events)
    assert check_shape_profiler_missing(str(run_dir), None) is None


def test_shape_profiler_no_sglang_returns_none(tmp_path):
    events = [{"cat": "cpu_op", "name": "aten::mm"}]
    run_dir = make_run_dir(tmp_path, capture_events=events)
    assert check_shape_profiler_missing(str(run_dir), None) is None


def test_shape_profiler_no_source_returns_none(tmp_path):
    # No capture folder and no resolvable trace -> events stays None -> None.
    run_dir = make_run_dir(tmp_path)
    assert check_shape_profiler_missing(str(run_dir), None) is None


def test_shape_profiler_missing_fires_from_trace_fallback(tmp_path):
    # No capture folder: the check falls back to the resolved main trace path.
    trace = write_json_gz(
        tmp_path / "t.json.gz", [{"cat": "cpu_op", "name": "sglang::decode"}]
    )
    run_dir = make_run_dir(tmp_path, cmd_prefix=trace)
    finding = check_shape_profiler_missing(str(run_dir), None)
    assert finding is not None
    assert finding.failure_mode == "Trace was profiled without SGLang patches"
    assert "t.json.gz" in finding.evidence


# ---------------------------------------------------------------------------
# 2f: check_split_low_gpu_kernels
# ---------------------------------------------------------------------------


def test_split_low_gpu_kernels_fires(tmp_path):
    events = [{"cat": "cpu_op"} for _ in range(100)] + [{"cat": "kernel"}]
    trace = write_json_gz(tmp_path / "t.json.gz", events)
    run_dir = make_run_dir(tmp_path, cmd_prefix=trace)
    finding = check_split_low_gpu_kernels(str(run_dir), None)
    assert finding is not None
    assert finding.failure_mode == "No / very few GPU kernel events in split trace"


def test_split_low_gpu_kernels_healthy_returns_none(tmp_path):
    events = [{"cat": "cpu_op"} for _ in range(10)] + [
        {"cat": "kernel"} for _ in range(5)
    ]
    trace = write_json_gz(tmp_path / "t.json.gz", events)
    run_dir = make_run_dir(tmp_path, cmd_prefix=trace)
    assert check_split_low_gpu_kernels(str(run_dir), None) is None


def test_split_low_gpu_kernels_no_cpu_ops_returns_none(tmp_path):
    events = [{"cat": "kernel"} for _ in range(5)]
    trace = write_json_gz(tmp_path / "t.json.gz", events)
    run_dir = make_run_dir(tmp_path, cmd_prefix=trace)
    assert check_split_low_gpu_kernels(str(run_dir), None) is None


def test_split_low_gpu_kernels_scans_siblings(tmp_path):
    # The resolved trace is healthy, but a decode_only sibling in the same dir is
    # kernel-starved. The sibling glob must pick it up and flag it. A corrupt
    # prefilldecode sibling exercises the _load_events None-continue skip.
    healthy = [{"cat": "cpu_op"} for _ in range(10)] + [
        {"cat": "kernel"} for _ in range(5)
    ]
    trace = write_json_gz(tmp_path / "main.json.gz", healthy)
    starved = [{"cat": "cpu_op"} for _ in range(100)] + [{"cat": "kernel"}]
    write_json_gz(tmp_path / "decode_only_0.json.gz", starved)
    write_text(tmp_path / "prefilldecode_0.json.gz", "not gzip")
    run_dir = make_run_dir(tmp_path, cmd_prefix=trace)
    finding = check_split_low_gpu_kernels(str(run_dir), None)
    assert finding is not None
    assert "decode_only_0.json.gz" in finding.evidence


def test_split_low_gpu_kernels_no_trace_returns_none(tmp_path):
    run_dir = make_run_dir(tmp_path)
    assert check_split_low_gpu_kernels(str(run_dir), None) is None


# ---------------------------------------------------------------------------
# 2g: check_high_idle
# ---------------------------------------------------------------------------


def test_high_idle_from_timeline(tmp_path):
    run_dir = make_run_dir(
        tmp_path, timeline_rows=[{"type": "idle_time", "percent": "20"}]
    )
    finding = check_high_idle(str(run_dir), None)
    assert finding is not None
    assert finding.failure_mode == "High idle time (> 15% GPU idle)"
    assert "gpu_timeline.csv" in finding.evidence


def test_high_idle_below_threshold_returns_none(tmp_path):
    run_dir = make_run_dir(
        tmp_path, timeline_rows=[{"type": "idle_time", "percent": "5"}]
    )
    assert check_high_idle(str(run_dir), None) is None


def test_high_idle_from_manifest(tmp_path):
    run_dir = make_run_dir(
        tmp_path, manifest={"gpu_utilization": {"idle_time_percent": 30}}
    )
    finding = check_high_idle(str(run_dir), None)
    assert finding is not None
    assert "category_manifest.json" in finding.evidence


def test_high_idle_no_data_returns_none(tmp_path):
    run_dir = make_run_dir(tmp_path)
    assert check_high_idle(str(run_dir), None) is None


# ---------------------------------------------------------------------------
# 2h: check_gpu_graph_replay
# ---------------------------------------------------------------------------


def test_gpu_graph_replay_fires_from_stream(tmp_path):
    stream = write_stream(
        tmp_path / "s.log", ["... DIAG:trace_quality:GPU_GRAPH_REPLAY ..."]
    )
    finding = check_gpu_graph_replay(str(tmp_path), stream)
    assert finding is not None
    assert finding.failure_mode == "GPU Graph Replay detected in default mode"


def test_gpu_graph_replay_no_match_returns_none(tmp_path):
    stream = write_stream(tmp_path / "s.log", ["nothing interesting here"])
    assert check_gpu_graph_replay(str(tmp_path), stream) is None
    assert check_gpu_graph_replay(str(tmp_path), None) is None


# ---------------------------------------------------------------------------
# 2i: check_corrupt_json
# ---------------------------------------------------------------------------


def test_corrupt_json_fires(tmp_path):
    trace = write_text(tmp_path / "corrupt.json", "{not valid json")
    run_dir = make_run_dir(tmp_path, cmd_prefix=trace)
    finding = check_corrupt_json(str(run_dir), None)
    assert finding is not None
    assert finding.failure_mode == "Trace file is corrupted/invalid JSON"


def test_corrupt_json_valid_returns_none(tmp_path):
    trace = write_json_gz(tmp_path / "t.json.gz", [{"cat": "kernel"}])
    run_dir = make_run_dir(tmp_path, cmd_prefix=trace)
    assert check_corrupt_json(str(run_dir), None) is None


def test_corrupt_json_no_trace_returns_none(tmp_path):
    run_dir = make_run_dir(tmp_path)
    assert check_corrupt_json(str(run_dir), None) is None


# ---------------------------------------------------------------------------
# 2j: check_runtime_instability
# ---------------------------------------------------------------------------


def test_runtime_instability_fires(tmp_path):
    rows = [
        {
            "name": "gemm",
            "Kernel Time (µs)_mean": "100",
            "Kernel Time (µs)_std": "50",
        }
    ]
    run_dir = make_run_dir(tmp_path, perf_rows=rows)
    finding = check_runtime_instability(str(run_dir), None)
    assert finding is not None
    assert finding.failure_mode == "Run-time instability across iterations / per cpu_op"


def test_runtime_instability_skips_rows_without_mean_std(tmp_path):
    # First row has no parseable std (skipped by the continue); the second is
    # genuinely unstable, so only it drives the finding.
    rows = [
        {"name": "skip", "Kernel Time (µs)_mean": "0", "Kernel Time (µs)_std": ""},
        {
            "name": "unstable_op",
            "Kernel Time (µs)_mean": "100",
            "Kernel Time (µs)_std": "50",
        },
    ]
    run_dir = make_run_dir(tmp_path, perf_rows=rows)
    finding = check_runtime_instability(str(run_dir), None)
    assert finding is not None
    assert "1 ops" in finding.evidence
    assert "unstable_op" in finding.evidence


def test_runtime_instability_stable_returns_none(tmp_path):
    rows = [
        {
            "name": "gemm",
            "Kernel Time (µs)_mean": "100",
            "Kernel Time (µs)_std": "10",
        }
    ]
    run_dir = make_run_dir(tmp_path, perf_rows=rows)
    assert check_runtime_instability(str(run_dir), None) is None


def test_runtime_instability_no_summary_returns_none(tmp_path):
    run_dir = make_run_dir(tmp_path)
    assert check_runtime_instability(str(run_dir), None) is None


###############################################################################
# Group 3 — reporting / infra / perf-model / geak checks (3x, 4x, 5x, 6x)
# (originally test_triage_checks_reporting.py)
###############################################################################

# perf_row helper — a "significant, fully-modelled" row (Cumulative <= 90 AND
# Percentage >= 4, with TB/s + TFLOPs + roofline populated) that trips none of
# the perf-model checks. Individual tests override just the cells they need.
_PERF_DEFAULTS = {
    "name": "aten::mm",
    "Percentage (%)": "50",
    "Cumulative Percentage (%)": "50",
    "op category": "gemm",
    "Kernel Time (µs)_mean": "100",
    "Kernel Time (µs)_std": "1",
    "TB/s_mean": "1.5",
    "TFLOPS/s_mean": "10",
    "Pct Roofline_mean": "40",
}

# fn -> CheckSpec: lets a test assert the DIAG tag / category / sublabel the
# runner will attach to a fired FindingDraft without duplicating the registry.
_SPEC_BY_FN = {spec.fn: spec for spec in ALL_CHECKS}


def _row(**over):
    r = dict(_PERF_DEFAULTS)
    r.update(over)
    return r


# ---------------------------------------------------------------------------
# Spec metadata lock — the DIAG tag each covered check contributes
# ---------------------------------------------------------------------------


def test_covered_check_spec_tags():
    expected = {
        check_hipgraph_launch_in_report: "DIAG:perf_model:3b_HIPGRAPH_LAUNCH_IN_REPORT",
        check_synthetic_op_significant: "DIAG:perf_model:3d_SYNTHETIC_OP_SIGNIFICANT",
        check_unclassified_op_significant: (
            "DIAG:perf_model:3e_UNCLASSIFIED_OP_SIGNIFICANT"
        ),
        check_tbs_tflops_missing: "DIAG:perf_model:3f_TBS_TFLOPS_MISSING",
        check_roofline_pct_missing: "DIAG:perf_model:3g_ROOFLINE_PCT_MISSING",
        check_zero_pct_ops: "DIAG:perf_model:3h_ZERO_PCT_OPS",
        check_step1_fail: "DIAG:tracelens_agent_workflow:4a_PERF_REPORT_FAILURE",
        check_step2_5_fail: "DIAG:tracelens_agent_workflow:4b_ORCHESTRATOR_PREP_FAIL",
        check_output_incomplete: (
            "DIAG:tracelens_agent_workflow:4c_OUTPUT_DIRS_MISSING"
        ),
        check_prefix_fail: "DIAG:tracelens_agent_workflow:4d_CMD_PREFIX_INVALID",
        check_report_too_small: (
            "DIAG:tracelens_agent_workflow:4e_ANALYSIS_MD_MISSING_OR_EMPTY"
        ),
        check_subagent_budget: (
            "DIAG:tracelens_agent_workflow:4f_SUBAGENT_FINDINGS_MISSING"
        ),
        check_ssh_fail: "DIAG:infra:5a_SSH_FAIL",
        check_docker_missing: "DIAG:infra:5b_DOCKER_MISSING",
        check_tl_not_installed: "DIAG:infra:5c_TL_NOT_INSTALLED",
        check_disk_full: "DIAG:infra:5d_DISK_FULL",
        check_nfs_stale: "DIAG:infra:5e_NFS_STALE",
        check_missing_dep: "DIAG:infra:5f_MISSING_DEP",
        check_resource_exhausted: "DIAG:infra:5g_CONTEXT_LENGTH_EXCEEDED",
        check_kernel_candidates_missing: (
            "DIAG:geak_interface:6a_KERNEL_CANDIDATES_MISSING"
        ),
    }
    for fn, tag in expected.items():
        assert _SPEC_BY_FN[fn].build_tag() == tag


# ---------------------------------------------------------------------------
# 3x — perf_model checks
# ---------------------------------------------------------------------------


# Every unified_perf_summary-driven perf check must degrade to None when the CSV
# is absent (loader returns None) rather than raising.
@pytest.mark.parametrize(
    "fn",
    [
        check_hipgraph_launch_in_report,
        check_synthetic_op_significant,
        check_unclassified_op_significant,
        check_tbs_tflops_missing,
        check_roofline_pct_missing,
        check_zero_pct_ops,
    ],
)
def test_perf_model_checks_none_without_csv(tmp_path, fn):
    run_dir = make_run_dir(tmp_path)
    assert fn(str(run_dir), None) is None


def test_perf_report_command_incorrect_and_merge_fail_are_inert(tmp_path):
    # 3a / 3c are reserved-slot stubs that must stay inert.
    run_dir = str(make_run_dir(tmp_path))
    assert check_perf_report_command_incorrect(run_dir, None) is None
    assert check_capture_graph_merge_fail(run_dir, None) is None


def test_hipgraph_launch_in_report_fires(tmp_path):
    run_dir = make_run_dir(tmp_path, perf_rows=[_row(name="hipGraphLaunch")])
    finding = check_hipgraph_launch_in_report(str(run_dir), None)
    assert isinstance(finding, FindingDraft)
    assert (
        finding.failure_mode
        == "HipGraph/cuGraph launch events appear in the perf report"
    )
    assert "GraphLaunch" in finding.evidence


def test_hipgraph_launch_in_report_none_without_match(tmp_path):
    run_dir = make_run_dir(tmp_path, perf_rows=[_row(name="aten::mm")])
    assert check_hipgraph_launch_in_report(str(run_dir), None) is None


def test_synthetic_op_significant_fires(tmp_path):
    run_dir = make_run_dir(tmp_path, perf_rows=[_row(name="gemm (Synthetic Op)")])
    finding = check_synthetic_op_significant(str(run_dir), None)
    assert isinstance(finding, FindingDraft)
    assert finding.failure_mode == "Synthetic op appears among significant ops"
    assert "(Synthetic Op)" in finding.evidence


def test_synthetic_op_significant_none_when_not_significant(tmp_path):
    # Name matches but the row is below the significance band -> not flagged.
    run_dir = make_run_dir(
        tmp_path,
        perf_rows=[
            _row(
                name="gemm (Synthetic Op)",
                **{"Percentage (%)": "1", "Cumulative Percentage (%)": "99"},
            )
        ],
    )
    assert check_synthetic_op_significant(str(run_dir), None) is None


def test_unclassified_op_significant_fires(tmp_path):
    run_dir = make_run_dir(tmp_path, perf_rows=[_row(**{"op category": "other"})])
    finding = check_unclassified_op_significant(str(run_dir), None)
    assert isinstance(finding, FindingDraft)
    assert finding.failure_mode == "Unclassified op among significant ops"


def test_unclassified_op_significant_none_when_classified(tmp_path):
    run_dir = make_run_dir(tmp_path, perf_rows=[_row(**{"op category": "gemm"})])
    assert check_unclassified_op_significant(str(run_dir), None) is None


def test_tbs_tflops_missing_fires(tmp_path):
    run_dir = make_run_dir(
        tmp_path,
        perf_rows=[_row(**{"TB/s_mean": "", "TFLOPS/s_mean": ""})],
    )
    finding = check_tbs_tflops_missing(str(run_dir), None)
    assert isinstance(finding, FindingDraft)
    assert finding.failure_mode == "TBs / TFLOPs not recorded for significant ops"


def test_tbs_tflops_missing_none_when_present(tmp_path):
    run_dir = make_run_dir(tmp_path, perf_rows=[_row()])
    assert check_tbs_tflops_missing(str(run_dir), None) is None


def test_tbs_tflops_missing_skips_exempt_category(tmp_path):
    # Collective / custom ops legitimately lack TB/s + TFLOPs and are exempt.
    run_dir = make_run_dir(
        tmp_path,
        perf_rows=[
            _row(
                **{
                    "op category": "collective",
                    "TB/s_mean": "",
                    "TFLOPS/s_mean": "",
                }
            )
        ],
    )
    assert check_tbs_tflops_missing(str(run_dir), None) is None


def test_roofline_pct_missing_fires(tmp_path):
    # Has TB/s but no Pct Roofline_mean -> roofline coverage gap.
    run_dir = make_run_dir(
        tmp_path,
        perf_rows=[_row(**{"Pct Roofline_mean": ""})],
    )
    finding = check_roofline_pct_missing(str(run_dir), None)
    assert isinstance(finding, FindingDraft)
    assert finding.failure_mode == "pct of roofline missing"


def test_roofline_pct_missing_none_when_present(tmp_path):
    run_dir = make_run_dir(tmp_path, perf_rows=[_row()])
    assert check_roofline_pct_missing(str(run_dir), None) is None


def test_zero_pct_ops_fires(tmp_path):
    run_dir = make_run_dir(tmp_path, perf_rows=[_row(**{"Percentage (%)": "0"})])
    finding = check_zero_pct_ops(str(run_dir), None)
    assert isinstance(finding, FindingDraft)
    assert finding.failure_mode == (
        "Op with zero recorded percentage (truncated at < 10us kernel time)"
    )


def test_zero_pct_ops_none_when_nonzero(tmp_path):
    run_dir = make_run_dir(tmp_path, perf_rows=[_row(**{"Percentage (%)": "50"})])
    assert check_zero_pct_ops(str(run_dir), None) is None


# ---------------------------------------------------------------------------
# 4x — tracelens_agent_workflow checks
# ---------------------------------------------------------------------------


def test_step1_fail_perf_dir_missing(tmp_path):
    run_dir = make_run_dir(tmp_path)
    finding = check_step1_fail(str(run_dir), None)
    assert isinstance(finding, FindingDraft)
    assert finding.failure_mode == "TraceLens perf reports are missing"
    assert "perf_report_csvs/ missing or empty" in finding.evidence


def test_step1_fail_required_file_missing(tmp_path):
    # perf_report_csvs/ exists (unified only) but gpu_timeline.csv is absent.
    run_dir = make_run_dir(tmp_path, perf_rows=[_row()])
    finding = check_step1_fail(str(run_dir), None)
    assert isinstance(finding, FindingDraft)
    assert "missing required files: gpu_timeline.csv" in finding.evidence


def test_step1_fail_stream_diag(tmp_path):
    run_dir = make_run_dir(
        tmp_path,
        perf_rows=[_row()],
        timeline_rows=[{"type": "idle_time", "percent": "5"}],
    )
    stream = write_stream(
        tmp_path / "stream.log",
        ["DIAG:tracelens_agent_workflow:PERF_REPORT_FAILURE thrown"],
    )
    finding = check_step1_fail(str(run_dir), stream)
    assert isinstance(finding, FindingDraft)
    assert finding.evidence == "DIAG tag found in agent stream"


def test_step1_fail_none_when_complete(tmp_path):
    run_dir = make_run_dir(
        tmp_path,
        perf_rows=[_row()],
        timeline_rows=[{"type": "idle_time", "percent": "5"}],
    )
    stream = write_stream(tmp_path / "stream.log", ["all good here"])
    assert check_step1_fail(str(run_dir), stream) is None


def test_step2_5_fail_manifest_missing(tmp_path):
    # perf_report_csvs/ present but category_manifest.json absent.
    run_dir = make_run_dir(tmp_path, perf_rows=[_row()])
    finding = check_step2_5_fail(str(run_dir), None)
    assert isinstance(finding, FindingDraft)
    assert finding.failure_mode == "Orchestrator preparation fails (Steps 2-5)"
    assert "category_manifest.json missing" in finding.evidence


def test_step2_5_fail_stream_diag(tmp_path):
    # No perf_report_csvs/, so the manifest branch is skipped; the stream fires.
    run_dir = make_run_dir(tmp_path)
    stream = write_stream(
        tmp_path / "stream.log",
        ["DIAG:tracelens_agent_workflow:ORCHESTRATOR_PREP_FAIL"],
    )
    finding = check_step2_5_fail(str(run_dir), stream)
    assert isinstance(finding, FindingDraft)
    assert finding.evidence == "DIAG tag found in agent stream"


def test_step2_5_fail_none(tmp_path):
    # perf_report_csvs/ and category_manifest.json both present, clean stream.
    run_dir = make_run_dir(tmp_path, perf_rows=[_row()], manifest={"categories": []})
    stream = write_stream(tmp_path / "stream.log", ["nothing to see"])
    assert check_step2_5_fail(str(run_dir), stream) is None


def test_output_incomplete_fires(tmp_path):
    run_dir = make_run_dir(tmp_path)
    finding = check_output_incomplete(str(run_dir), None)
    assert isinstance(finding, FindingDraft)
    assert finding.failure_mode == "Output directory structure incomplete"
    assert finding.evidence == (
        "Missing directories: metadata, category_data, system_findings, "
        "category_findings"
    )


def test_output_incomplete_none_when_all_present(tmp_path):
    run_dir = make_run_dir(
        tmp_path,
        output_dirs=(
            "metadata",
            "category_data",
            "system_findings",
            "category_findings",
        ),
    )
    assert check_output_incomplete(str(run_dir), None) is None


def test_prefix_fail_fires(tmp_path):
    stream = write_stream(
        tmp_path / "stream.log", ["DIAG:tracelens_agent_workflow:CMD_PREFIX_INVALID"]
    )
    finding = check_prefix_fail(str(tmp_path), stream)
    assert isinstance(finding, FindingDraft)
    assert finding.failure_mode == "Command prefix validation fails"
    assert finding.evidence == "DIAG tag found in agent stream"


def test_prefix_fail_none(tmp_path):
    stream = write_stream(tmp_path / "stream.log", ["prefix ok"])
    assert check_prefix_fail(str(tmp_path), stream) is None


def test_report_too_small_fires(tmp_path):
    run_dir = make_run_dir(tmp_path, analysis_md="x")
    finding = check_report_too_small(str(run_dir), None)
    assert isinstance(finding, FindingDraft)
    assert finding.failure_mode == "analysis.md too small (< 100 bytes)"
    assert "analysis.md is 1 bytes" in finding.evidence


def test_report_missing_with_category_findings(tmp_path):
    run_dir = make_run_dir(tmp_path, output_dirs=("category_findings",))
    finding = check_report_too_small(str(run_dir), None)
    assert isinstance(finding, FindingDraft)
    assert finding.failure_mode == "analysis.md missing"
    assert "category_findings/ is present" in finding.evidence


def test_report_too_small_none_when_large(tmp_path):
    run_dir = make_run_dir(tmp_path, analysis_md="x" * 200)
    assert check_report_too_small(str(run_dir), None) is None


def test_report_too_small_none_when_absent_no_findings_dir(tmp_path):
    # No analysis.md and no category_findings/ -> nothing to report.
    run_dir = make_run_dir(tmp_path)
    assert check_report_too_small(str(run_dir), None) is None


def test_subagent_budget_fires(tmp_path):
    run_dir = make_run_dir(tmp_path, manifest={"categories": [{"name": "gemm"}]})
    finding = check_subagent_budget(str(run_dir), None)
    assert isinstance(finding, FindingDraft)
    assert finding.failure_mode == "Subagent exceeds token budget"
    assert "Missing findings for: gemm" in finding.evidence


def test_subagent_budget_none_when_all_written(tmp_path):
    # Exercises every skip/pass branch: cpu_idle below the idle threshold,
    # a compute-kernel category and a system-tier category that both wrote
    # their findings files, plus an unnamed category that is ignored.
    manifest = {
        "gpu_utilization": {"idle_time_percent": 5},
        "categories": [
            {"name": "cpu_idle"},
            {"name": "gemm"},
            {"name": "comm", "tier": "system"},
            {"name": ""},
        ],
    }
    run_dir = make_run_dir(tmp_path, manifest=manifest)
    write_text(run_dir / "category_findings" / "gemm_findings.md", "ok")
    write_text(run_dir / "system_findings" / "comm_findings.md", "ok")
    assert check_subagent_budget(str(run_dir), None) is None


def test_subagent_budget_none_without_manifest(tmp_path):
    run_dir = make_run_dir(tmp_path)
    assert check_subagent_budget(str(run_dir), None) is None


# ---------------------------------------------------------------------------
# 5x — infra checks (stream-driven)
# ---------------------------------------------------------------------------


def test_ssh_fail_fires(tmp_path):
    stream = write_stream(
        tmp_path / "s.log", ["ssh: connect to host node1: Connection refused"]
    )
    finding = check_ssh_fail(str(tmp_path), stream)
    assert isinstance(finding, FindingDraft)
    assert finding.failure_mode == "SSH connection to node fails"
    assert "Connection refused" in finding.evidence


def test_ssh_fail_none(tmp_path):
    stream = write_stream(tmp_path / "s.log", ["ssh connected fine"])
    assert check_ssh_fail(str(tmp_path), stream) is None


def test_docker_missing_fires(tmp_path):
    stream = write_stream(
        tmp_path / "s.log", ["Error: No such container: tracelens_dev"]
    )
    finding = check_docker_missing(str(tmp_path), stream)
    assert isinstance(finding, FindingDraft)
    assert finding.failure_mode == "Docker container not found"
    assert "No such container" in finding.evidence


def test_docker_missing_none(tmp_path):
    stream = write_stream(tmp_path / "s.log", ["container is up"])
    assert check_docker_missing(str(tmp_path), stream) is None


def test_tl_not_installed_fires(tmp_path):
    stream = write_stream(
        tmp_path / "s.log", ["ModuleNotFoundError: No module named 'TraceLens'"]
    )
    finding = check_tl_not_installed(str(tmp_path), stream)
    assert isinstance(finding, FindingDraft)
    assert finding.failure_mode == "TraceLens not installed in remote env"
    assert "TraceLens" in finding.evidence


def test_tl_not_installed_none(tmp_path):
    stream = write_stream(tmp_path / "s.log", ["TraceLens imported ok"])
    assert check_tl_not_installed(str(tmp_path), stream) is None


def test_disk_full_zero_byte_files(tmp_path):
    run_dir = make_run_dir(tmp_path)
    for i in range(4):
        (run_dir / f"empty{i}.txt").touch()
    finding = check_disk_full(str(run_dir), None)
    assert isinstance(finding, FindingDraft)
    assert finding.failure_mode == "Disk space exhausted"
    assert "zero-byte files found" in finding.evidence


def test_disk_full_stream_signal(tmp_path):
    # Clean run dir (no zero-byte files) -> only the stream branch can fire.
    run_dir = make_run_dir(tmp_path)
    stream = write_stream(
        tmp_path / "s.log", ["OSError: [Errno 28] No space left on device"]
    )
    finding = check_disk_full(str(run_dir), stream)
    assert isinstance(finding, FindingDraft)
    assert finding.evidence == "'No space left on device' found in agent stream"


def test_disk_full_none(tmp_path):
    run_dir = make_run_dir(tmp_path)
    stream = write_stream(tmp_path / "s.log", ["disk is fine"])
    assert check_disk_full(str(run_dir), stream) is None


def test_nfs_stale_fires(tmp_path):
    stream = write_stream(
        tmp_path / "s.log", ["OSError: [Errno 116] Stale file handle"]
    )
    finding = check_nfs_stale(str(tmp_path), stream)
    assert isinstance(finding, FindingDraft)
    assert finding.failure_mode == "NFS latency / stale file handles"
    assert "Stale file handle" in finding.evidence


def test_nfs_stale_none(tmp_path):
    stream = write_stream(tmp_path / "s.log", ["nfs mount healthy"])
    assert check_nfs_stale(str(tmp_path), stream) is None


def test_missing_dep_fires(tmp_path):
    stream = write_stream(
        tmp_path / "s.log", ["ModuleNotFoundError: No module named 'numpy'"]
    )
    finding = check_missing_dep(str(tmp_path), stream)
    assert isinstance(finding, FindingDraft)
    assert finding.failure_mode == "Python dependency missing"
    assert "numpy" in finding.evidence


def test_missing_dep_none_for_tracelens(tmp_path):
    # A missing 'TraceLens' import is owned by 5c, not 5f: the negative lookahead
    # must exclude it here.
    stream = write_stream(
        tmp_path / "s.log", ["ModuleNotFoundError: No module named 'TraceLens'"]
    )
    assert check_missing_dep(str(tmp_path), stream) is None


# ---------------------------------------------------------------------------
# 5g — resource-exhausted (three parse paths)
# ---------------------------------------------------------------------------


def test_resource_exhausted_json_error(tmp_path):
    stream = write_stream(
        tmp_path / "s.log",
        [{"type": "error", "message": "RESOURCE_EXHAUSTED: context too long"}],
    )
    finding = check_resource_exhausted(str(tmp_path), stream)
    assert isinstance(finding, FindingDraft)
    assert finding.failure_mode == "Context length exceeded (RESOURCE_EXHAUSTED)"
    assert finding.evidence == "RESOURCE_EXHAUSTED error event in agent stream"


def test_resource_exhausted_json_system(tmp_path):
    stream = write_stream(
        tmp_path / "s.log",
        [{"type": "system", "error": "backend RESOURCE_EXHAUSTED"}],
    )
    finding = check_resource_exhausted(str(tmp_path), stream)
    assert isinstance(finding, FindingDraft)
    assert finding.evidence == "RESOURCE_EXHAUSTED system error in agent stream"


def test_resource_exhausted_plain_regex(tmp_path):
    # A non-JSON log line takes the except branch and matches case-insensitively.
    stream = write_stream(tmp_path / "s.log", ["fatal: Resource_Exhausted on host"])
    finding = check_resource_exhausted(str(tmp_path), stream)
    assert isinstance(finding, FindingDraft)
    assert finding.evidence.startswith("resource_exhausted in stream:")


def test_resource_exhausted_none(tmp_path):
    stream = write_stream(
        tmp_path / "s.log",
        [{"type": "error", "message": "some other error"}, "plain line"],
    )
    assert check_resource_exhausted(str(tmp_path), stream) is None


# ---------------------------------------------------------------------------
# 6x — geak_interface check
# ---------------------------------------------------------------------------


def test_kernel_candidates_present_returns_none(tmp_path):
    session = make_geak_session(tmp_path, phases=["KERNEL_AGENT"], with_candidates=True)
    assert check_kernel_candidates_missing(str(session)) is None


def test_kernel_candidates_missing_fires(tmp_path):
    session = make_geak_session(
        tmp_path, phases=["KERNEL_AGENT"], with_candidates=False
    )
    finding = check_kernel_candidates_missing(str(session))
    assert isinstance(finding, FindingDraft)
    assert finding.failure_mode == "kernel_candidates.json missing"
    assert "Session reached KERNEL phase" in finding.evidence


def test_kernel_candidates_none_when_phase_not_reached(tmp_path):
    # Exact-match gate: only the canonical "KERNEL_AGENT" phase arms the check;
    # a near-miss phase name (and no candidates) must still return None.
    session = make_geak_session(
        tmp_path, phases=["SETUP", "KERNEL"], with_candidates=False
    )
    assert check_kernel_candidates_missing(str(session)) is None


def test_kernel_candidates_none_without_session_breakdown(tmp_path):
    # A directory with no session_breakdown.json (and none 5 levels up).
    assert check_kernel_candidates_missing(str(tmp_path)) is None


def test_kernel_candidates_none_when_session_breakdown_corrupt(tmp_path):
    # A truncated session_breakdown.json is swallowed to None, not raised.
    session = tmp_path / "session"
    write_text(session / "session_breakdown.json", '{"phase_segments": [')
    assert check_kernel_candidates_missing(str(session)) is None


###############################################################################
# Group 4 — postprocess aggregator (originally test_triage_postprocess.py)
#
# ``_finding`` here builds a postprocess ``TriageFinding``; the runner group
# below defines its own ``Finding``-based builder as ``_runner_finding`` to
# avoid a name collision.
###############################################################################


def _triage_finding(
    run_dir="/fake/run_a",
    model_name="model_a",
    diag_tag="1a_TRACE_MISSING",
    category="profiling",
    failure_mode="Trace missing",
    evidence="",
    remedy="Re-run capture",
    implied_by="",
):
    return TriageFinding(
        run_dir=run_dir,
        model_name=model_name,
        diag_tag=diag_tag,
        category=category,
        failure_mode=failure_mode,
        evidence=evidence,
        remedy=remedy,
        implied_by=implied_by,
    )


# ---------------------------------------------------------------------------
# extract_model_name
# ---------------------------------------------------------------------------


def test_extract_model_name_session_dir_plain_z():
    run_dir = "/data/traces/my_model/20250115T093000Z/kernel-agent/runs/run_0"
    assert extract_model_name(run_dir) == "my_model"


def test_extract_model_name_session_dir_hash_suffix_not_fallback():
    # The <timestamp>Z-<hash> session-dir form must still resolve the model
    # segment via the ``[^/]*`` in _MODEL_RE, NOT collapse to the basename
    # fallback (which would be the run leaf, e.g. "run_0").
    run_dir = "/data/traces/big_model/20250115T093000Z-deadbeef/kernel-agent/runs/run_0"
    got = extract_model_name(run_dir)
    assert got == "big_model"
    assert got != "run_0"


def test_extract_model_name_falls_back_to_basename():
    assert extract_model_name("/tmp/some/plain_dir") == "plain_dir"


# ---------------------------------------------------------------------------
# sanitize_filename
# ---------------------------------------------------------------------------


def test_sanitize_filename_lowercases_and_replaces():
    assert sanitize_filename("My Model/v2!!") == "my_model_v2"


def test_sanitize_filename_truncates_and_strips_trailing_underscore():
    # Runs of disallowed chars collapse to one "_"; truncation at max_len can
    # leave a trailing "_" which is then stripped.
    name = "abcdef!!!ghij"
    assert sanitize_filename(name, max_len=7) == "abcdef"


# ---------------------------------------------------------------------------
# extract_action_keys
# ---------------------------------------------------------------------------


def test_extract_action_keys_parses_eg_list():
    ev = "Significant ops unclassified (e.g. aten::mm, aten::bmm, aten::add)"
    assert extract_action_keys(ev) == ["aten::mm", "aten::bmm", "aten::add"]


def test_extract_action_keys_no_match_returns_empty():
    assert extract_action_keys("no parenthetical here") == []


def test_extract_action_keys_drops_blank_items():
    assert extract_action_keys("x (e.g. a, , b,)") == ["a", "b"]


# ---------------------------------------------------------------------------
# discover_triage_csvs
# ---------------------------------------------------------------------------


def test_discover_triage_csvs_finds_and_sorts(tmp_path):
    root = tmp_path / "traces"
    run_b = root / "b_run"
    run_a = root / "nested" / "a_run"
    make_triage_csv(run_a, [])
    make_triage_csv(run_b, [])
    # A dir with no triage_details.csv must be ignored.
    (root / "empty").mkdir(parents=True)

    found = discover_triage_csvs(str(root))
    assert found == sorted([str(run_a), str(run_b)])
    assert str(root / "empty") not in found


def test_discover_triage_csvs_empty_tree(tmp_path):
    assert discover_triage_csvs(str(tmp_path)) == []


# ---------------------------------------------------------------------------
# load_from_mapping
# ---------------------------------------------------------------------------


def test_load_from_mapping_empty_file(tmp_path):
    mapping = write_text(tmp_path / "map.txt", "")
    assert load_from_mapping(mapping) == []


def test_load_from_mapping_reads_first_tab_column(tmp_path):
    mapping = write_text(
        tmp_path / "map.txt",
        "/runs/alpha\t/runs/alpha/triage_diags.txt\n"
        "/runs/beta\textra\n"
        "\n"  # blank line skipped
        "   \t x\n",  # blank first column skipped
    )
    assert load_from_mapping(mapping) == ["/runs/alpha", "/runs/beta"]


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------


def test_aggregate_counts_and_action_keys():
    findings = [
        _triage_finding(
            model_name="m1",
            category="profiling",
            failure_mode="Trace missing",
            diag_tag="1a_TRACE_MISSING",
        ),
        _triage_finding(
            model_name="m2",
            category="profiling",
            failure_mode="Trace missing",
            diag_tag="1a_TRACE_MISSING",
        ),
        _triage_finding(
            model_name="m1",
            category="perf_model",
            failure_mode="Unclassified op among significant ops",
            diag_tag="3b_UNCLASSIFIED_OP",
            evidence="unclassified (e.g. aten::mm, aten::mm, aten::bmm)",
        ),
    ]
    agg = aggregate(findings)

    assert agg["by_failure_mode"]["Trace missing"] == 2
    assert agg["by_diag_tag"]["1a_TRACE_MISSING"] == 2
    assert agg["by_category"]["profiling"]["Trace missing"] == 2
    # models_per_issue dedups models sharing a failure mode.
    assert agg["models_per_issue"]["Trace missing"] == {"m1", "m2"}
    # action_keys only harvested for perf_model findings, counted per op.
    ak = agg["action_keys_per_issue"]["Unclassified op among significant ops"]
    assert ak["aten::mm"] == 2
    assert ak["aten::bmm"] == 1
    # Non-perf_model failure modes never populate action_keys.
    assert "Trace missing" not in agg["action_keys_per_issue"]


# ---------------------------------------------------------------------------
# collect_findings
# ---------------------------------------------------------------------------


def test_collect_findings_unassessed_and_rows(tmp_path):
    # One run with a real CSV (assessed), one with no CSV (unassessed).
    assessed = tmp_path / "assessed"
    make_triage_csv(
        assessed,
        [
            {
                "DIAG Tag": "1a_TRACE_MISSING",
                "Category": "profiling",
                "Failure Mode": "Trace missing",
                "Evidence": "no trace",
                "Remedy": "re-run",
                "Implied By": "",
            }
        ],
    )
    skipped = tmp_path / "skipped"
    skipped.mkdir()

    findings, total_runs, unassessed = collect_findings([str(assessed), str(skipped)])
    assert total_runs == 2
    assert unassessed == 1
    assert len(findings) == 1
    f = findings[0]
    assert f.diag_tag == "1a_TRACE_MISSING"
    assert f.failure_mode == "Trace missing"
    assert f.model_name == "assessed"


def test_collect_findings_swallows_unreadable_csv(tmp_path, monkeypatch):
    # A csv.Error / OSError while reading a present CSV is swallowed so one bad
    # run does not abort the aggregation.
    run_dir = tmp_path / "run"
    make_triage_csv(run_dir, [{"DIAG Tag": "1a_TRACE_MISSING"}])

    def _boom(*_a, **_k):
        raise csv.Error("bad csv")

    monkeypatch.setattr(postprocess.csv, "DictReader", _boom)
    findings, total_runs, unassessed = collect_findings([str(run_dir)])
    assert total_runs == 1
    assert unassessed == 0  # the file existed, so it was assessed
    assert findings == []  # the read blew up and was swallowed


# ---------------------------------------------------------------------------
# pick_reproducers
# ---------------------------------------------------------------------------


def test_pick_reproducers_skips_duplicate_model(tmp_path):
    # Two run dirs whose basename (model) collides: the second must be skipped so
    # reproducers stay one-per-model.
    findings = [
        _triage_finding(run_dir="/a/dup", model_name="dup"),
        _triage_finding(run_dir="/b/dup", model_name="dup"),
    ]
    agg = aggregate(findings)
    selected = pick_reproducers(findings, agg, top_n=5)
    assert [m for _, m, _ in selected] == ["dup"]


def test_pick_reproducers_prefers_coverage_and_distinct_models():
    # alpha covers two top modes + two tags -> higher score than beta.
    findings = [
        _triage_finding(
            run_dir="/fake/alpha",
            failure_mode="Trace missing",
            diag_tag="1a_TRACE_MISSING",
        ),
        _triage_finding(
            run_dir="/fake/alpha",
            failure_mode="Output incomplete",
            diag_tag="4a_OUTPUT_INCOMPLETE",
        ),
        _triage_finding(
            run_dir="/fake/beta",
            failure_mode="Trace missing",
            diag_tag="1a_TRACE_MISSING",
        ),
    ]
    agg = aggregate(findings)

    top1 = pick_reproducers(findings, agg, top_n=1)
    assert len(top1) == 1
    run_dir, model, run_findings = top1[0]
    assert run_dir == "/fake/alpha"
    assert model == "alpha"  # extract_model_name(basename)
    assert len(run_findings) == 2

    top_all = pick_reproducers(findings, agg, top_n=5)
    assert [m for _, m, _ in top_all] == ["alpha", "beta"]


# ---------------------------------------------------------------------------
# write_aggregated_csv
# ---------------------------------------------------------------------------


def test_write_aggregated_csv_roundtrips(tmp_path):
    findings = [
        _triage_finding(
            run_dir="/fake/alpha",
            model_name="alpha",
            diag_tag="1a_TRACE_MISSING",
            category="profiling",
            failure_mode="Trace missing",
            evidence="no trace",
            remedy="re-run",
            implied_by="",
        )
    ]
    path = write_aggregated_csv(findings, str(tmp_path))
    assert path == os.path.join(str(tmp_path), "aggregated_triage.csv")

    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == [
        "Model",
        "Run Dir",
        "DIAG Tag",
        "Category",
        "Failure Mode",
        "Evidence",
        "Remedy",
        "Implied By",
    ]
    assert rows[1] == [
        "alpha",
        "/fake/alpha",
        "1a_TRACE_MISSING",
        "profiling",
        "Trace missing",
        "no trace",
        "re-run",
        "",
    ]


# ---------------------------------------------------------------------------
# write_summary_report — drive every section
# ---------------------------------------------------------------------------


def test_write_summary_report_renders_all_sections(tmp_path):
    findings = [
        # profiling category, two runs -> most common failure mode
        _triage_finding(
            run_dir="/fake/alpha",
            model_name="alpha",
            category="profiling",
            failure_mode="Trace missing",
            diag_tag="1a_TRACE_MISSING",
            remedy="Re-run capture",
        ),
        _triage_finding(
            run_dir="/fake/beta",
            model_name="beta",
            category="profiling",
            failure_mode="Trace missing",
            diag_tag="1a_TRACE_MISSING",
            remedy="Re-run capture",
        ),
        # perf_model finding driving the filtered-ops section: mode is in
        # _OP_ATTENTION_MODES and evidence carries ops matching _LOOKS_LIKE_OP
        # (aten::mm) plus one that does not (notanop, must be filtered out).
        _triage_finding(
            run_dir="/fake/alpha",
            model_name="alpha",
            category="perf_model",
            failure_mode="Unclassified op among significant ops",
            diag_tag="3b_UNCLASSIFIED_OP",
            evidence="significant ops (e.g. aten::mm, notanop)",
            remedy="Add op model",
        ),
    ]
    agg = aggregate(findings)
    reproducers = pick_reproducers(findings, agg, top_n=3)

    path = write_summary_report(
        findings,
        agg,
        total_runs=5,
        reproducers=reproducers,
        report_dir=str(tmp_path),
        unassessed=1,
    )
    assert path == os.path.join(str(tmp_path), "summary_report.md")
    md = open(path).read()

    # Overview accounting: 2 runs with findings, 1 unassessed, 2 clean of 5.
    assert "| Total runs analyzed | 5 |" in md
    assert "| Runs with findings | 2 (40%) |" in md
    assert "| Runs not assessed (no triage output) | 1 (20%) |" in md
    assert "| Clean runs (assessed, no findings) | 2 (40%) |" in md

    # Category table.
    assert "## Issue Breakdown by Category" in md
    assert "| profiling | 2 | 1 | 2 |" in md
    assert "| perf_model | 1 | 1 | 1 |" in md

    # Top failure modes table row.
    assert "## Top Failure Modes" in md
    assert "| 1a_TRACE_MISSING | Trace missing | 2 | 2 | Re-run capture |" in md

    # Filtered-ops section: the _LOOKS_LIKE_OP match is rendered as an ops-table
    # row; the non-matching op is filtered out of that table (it still appears
    # verbatim in the reproducer evidence, so scope the negative to the backtick
    # form the ops table uses).
    assert "## Perf Model — Specific Ops Needing Attention" in md
    assert "| `aten::mm` | 1 |" in md
    assert "`notanop`" not in md

    # Action items grouped by category with per-mode remedy.
    assert "## Action Items" in md
    assert "### 1. Category: `profiling`" in md
    assert "- Remedy: Re-run capture" in md

    # Reproducers section.
    assert "## Reproducers" in md
    assert "### Reproducer 1: `alpha`" in md
    assert "--run-dir '/fake/alpha' --detailed" in md

    # DIAG tag frequency.
    assert "## DIAG Tag Frequency" in md
    assert "| `1a_TRACE_MISSING` | 2 |" in md


def test_write_summary_report_no_filtered_ops_section(tmp_path):
    # perf_model finding whose ops do NOT match _LOOKS_LIKE_OP: the ops section
    # must be omitted entirely (exercises the _LOOKS_LIKE_OP.match skip).
    findings = [
        _triage_finding(
            run_dir="/fake/alpha",
            model_name="alpha",
            category="perf_model",
            failure_mode="Unclassified op among significant ops",
            evidence="ops (e.g. plain_name, another_one)",
        )
    ]
    agg = aggregate(findings)
    path = write_summary_report(findings, agg, 1, [], str(tmp_path))
    md = open(path).read()
    assert "Specific Ops Needing Attention" not in md


def test_write_summary_report_non_attention_mode_skipped(tmp_path):
    # A perf_model finding with matching ops but a failure mode NOT in
    # _OP_ATTENTION_MODES: action keys are aggregated but the mode is skipped, so
    # no ops section renders (exercises the ``mode not in _OP_ATTENTION_MODES``
    # continue).
    findings = [
        _triage_finding(
            run_dir="/fake/alpha",
            model_name="alpha",
            category="perf_model",
            failure_mode="Some other perf mode",
            evidence="ops (e.g. aten::mm, aten::bmm)",
        )
    ]
    agg = aggregate(findings)
    # The op keys were still harvested for this mode...
    assert agg["action_keys_per_issue"]["Some other perf mode"]["aten::mm"] == 1
    path = write_summary_report(findings, agg, 1, [], str(tmp_path))
    md = open(path).read()
    assert "Specific Ops Needing Attention" not in md


def test_write_summary_report_zero_total_runs_no_divide(tmp_path):
    # total_runs=0 must exercise the inline ``... if total_runs else 0`` guard on
    # all three overview percentages instead of raising ZeroDivisionError.
    findings = []
    agg = aggregate(findings)
    path = write_summary_report(findings, agg, 0, [], str(tmp_path), unassessed=0)
    md = open(path).read()
    assert "| Total runs analyzed | 0 |" in md
    assert "| Runs with findings | 0 (0%) |" in md
    assert "| Runs not assessed (no triage output) | 0 (0%) |" in md
    assert "| Clean runs (assessed, no findings) | 0 (0%) |" in md


# ---------------------------------------------------------------------------
# _collectible_paths
# ---------------------------------------------------------------------------


def _build_run_tree(tmp_path):
    """Lay down a run_dir with every artifact _collectible_paths harvests."""
    run_dir = tmp_path / "run"
    make_triage_csv(run_dir, [{"DIAG Tag": "1a_TRACE_MISSING"}])
    write_text(run_dir / "triage_diags.txt", "diag\n")
    write_text(run_dir / "cache" / "cmd_prefix.txt", "--profile_json_path /t\n")
    write_json(run_dir / "category_data" / "category_manifest.json", {"a": 1})
    write_json(tmp_path / "trace_input_manifest.json", {"trace_input": "/t"})
    write_text(run_dir / "perf_report_csvs" / "gpu_timeline.csv", "type,percent\n")
    write_text(run_dir / "perf_report_csvs" / "unified_perf_summary.csv", "name\n")
    write_text(run_dir / "analysis.md", "# analysis\n")
    write_text(run_dir / "trace_split" / "execution_details.csv", "a\n")
    write_json(run_dir / "trace_split" / "execution_details.json", {})
    write_text(run_dir / "trace_split" / "mixed_steady_state_0.json.gz", "gz")
    return run_dir


def test_collectible_paths_harvests_expected_arcnames(tmp_path):
    run_dir = _build_run_tree(tmp_path)
    # Sibling capture_traces dir (dirname(abspath(run_dir))/capture_traces).
    write_text(tmp_path / "capture_traces" / "capture_0.json.gz", "gz")

    items = _collectible_paths(str(run_dir), max_bytes=10 * 1024 * 1024)
    arcnames = {arc for _, arc in items}
    assert arcnames == {
        "triage_details.csv",
        "triage_diags.txt",
        "cache/cmd_prefix.txt",
        "category_data/category_manifest.json",
        "trace_input_manifest.json",
        "perf_report_csvs/gpu_timeline.csv",
        "perf_report_csvs/unified_perf_summary.csv",
        "analysis.md",
        "trace_split/execution_details.csv",
        "trace_split/execution_details.json",
        "trace_split/mixed_steady_state_0.json.gz",
        "capture_traces/capture_0.json.gz",
    }
    # Every returned source path is a real file.
    for src, _ in items:
        assert os.path.isfile(src)


def test_collectible_paths_capture_via_manifest_fallback(tmp_path):
    # No sibling capture_traces dir: the manifest's capture_folder_path is used.
    run_dir = tmp_path / "run"
    make_triage_csv(run_dir, [])
    cap_dir = tmp_path / "external_captures"
    write_text(cap_dir / "capture_0.json.gz", "gz")
    write_json(
        run_dir / "category_data" / "category_manifest.json",
        {"capture_folder_path": str(cap_dir)},
    )

    items = _collectible_paths(str(run_dir), max_bytes=10 * 1024 * 1024)
    arcnames = {arc for _, arc in items}
    assert "capture_traces/capture_0.json.gz" in arcnames


def test_collectible_paths_bad_manifest_swallowed(tmp_path):
    # Malformed manifest JSON: the capture fallback except must swallow it and
    # simply yield no capture entries (still harvesting the CSV).
    run_dir = tmp_path / "run"
    make_triage_csv(run_dir, [])
    write_text(run_dir / "category_data" / "category_manifest.json", "{not valid json")
    items = _collectible_paths(str(run_dir), max_bytes=10 * 1024 * 1024)
    arcnames = {arc for _, arc in items}
    assert "triage_details.csv" in arcnames
    assert not any(a.startswith("capture_traces/") for a in arcnames)


def test_collectible_paths_respects_size_cap(tmp_path):
    run_dir = tmp_path / "run"
    make_triage_csv(run_dir, [{"DIAG Tag": "1a_TRACE_MISSING"}])
    # A 1-byte cap is smaller than the CSV, so nothing is collected.
    items = _collectible_paths(str(run_dir), max_bytes=1)
    assert items == []


# ---------------------------------------------------------------------------
# build_reproducer_packages
# ---------------------------------------------------------------------------


def test_build_reproducer_packages_writes_real_tarball(tmp_path):
    run_dir = _build_run_tree(tmp_path)
    report_dir = tmp_path / "report"
    report_dir.mkdir()
    findings = [_triage_finding(run_dir=str(run_dir))]
    reproducers = [(str(run_dir), "My Model", findings)]

    count = build_reproducer_packages(reproducers, str(report_dir))
    assert count == 1

    tar_path = report_dir / "reproducers" / "my_model.tar.gz"
    assert tar_path.is_file()
    # Uncompressed staging folder is cleaned up.
    assert not (report_dir / "reproducers" / "my_model").exists()

    with tarfile.open(tar_path, "r:gz") as tar:
        members = tar.getnames()
    assert "my_model/README.md" in members
    assert "my_model/triage_details.csv" in members
    assert "my_model/analysis.md" in members


def test_build_reproducer_packages_swallows_copy_error(tmp_path, monkeypatch):
    # A copy2 failure on a collectible artifact is swallowed; the package still
    # builds (README is written directly, not via copy2) and a tarball lands.
    run_dir = _build_run_tree(tmp_path)
    report_dir = tmp_path / "report"
    report_dir.mkdir()

    def _boom(*_a, **_k):
        raise OSError("copy failed")

    monkeypatch.setattr(postprocess.shutil, "copy2", _boom)

    count = build_reproducer_packages(
        [(str(run_dir), "My Model", [_triage_finding(run_dir=str(run_dir))])],
        str(report_dir),
    )
    assert count == 1
    tar_path = report_dir / "reproducers" / "my_model.tar.gz"
    assert tar_path.is_file()
    with tarfile.open(tar_path, "r:gz") as tar:
        members = tar.getnames()
    # README is written directly, so it survives the copy2 failure.
    assert "my_model/README.md" in members
    # The copied artifacts were dropped.
    assert "my_model/triage_details.csv" not in members


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


def test_main_traces_root_with_findings_returns_zero(tmp_path, monkeypatch):
    traces_root = tmp_path / "traces"
    run_dir = traces_root / "run0"
    make_triage_csv(
        run_dir,
        [
            {
                "DIAG Tag": "1a_TRACE_MISSING",
                "Category": "profiling",
                "Failure Mode": "Trace missing",
                "Evidence": "no trace",
                "Remedy": "re-run",
            }
        ],
    )
    report_dir = tmp_path / "report"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "postprocess",
            "--traces-root",
            str(traces_root),
            "--report-dir",
            str(report_dir),
        ],
    )
    rc = postprocess.main()
    assert rc == 0
    assert (report_dir / "aggregated_triage.csv").is_file()
    assert (report_dir / "summary_report.md").is_file()
    tarballs = list((report_dir / "reproducers").glob("*.tar.gz"))
    assert len(tarballs) == 1


def test_main_mapping_empty_returns_one(tmp_path, monkeypatch):
    mapping = write_text(tmp_path / "map.txt", "")
    report_dir = tmp_path / "report"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "postprocess",
            "--mapping",
            mapping,
            "--report-dir",
            str(report_dir),
        ],
    )
    assert postprocess.main() == 1


def test_main_runs_but_no_findings_early_returns_zero(tmp_path, monkeypatch):
    # A run dir with a header-only triage CSV: assessed, but zero findings.
    run_dir = tmp_path / "run0"
    make_triage_csv(run_dir, [])
    mapping = write_text(tmp_path / "map.txt", f"{run_dir}\n")
    report_dir = tmp_path / "report"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "postprocess",
            "--mapping",
            mapping,
            "--report-dir",
            str(report_dir),
        ],
    )
    rc = postprocess.main()
    assert rc == 0
    md = (report_dir / "summary_report.md").read_text()
    assert "All runs passed" in md
    # No aggregated CSV / reproducers on the clean early-return path.
    assert not (report_dir / "aggregated_triage.csv").exists()
    assert not (report_dir / "reproducers").exists()


def test_main_requires_input_mode(monkeypatch):
    # The mutually-exclusive input group is required: argparse exits(2).
    monkeypatch.setattr(sys, "argv", ["postprocess", "--report-dir", "/tmp/x"])
    with pytest.raises(SystemExit) as exc:
        postprocess.main()
    assert exc.value.code == 2


###############################################################################
# Group 5 — runner: check orchestrator + CLI (originally test_triage_runner.py)
#
# ``_runner_finding`` builds a ``checks.Finding`` (distinct from the
# postprocess-group ``TriageFinding`` builder above).
###############################################################################


# ---------------------------------------------------------------------------
# _auto_detect_stream / _auto_detect_log
# ---------------------------------------------------------------------------


def test_auto_detect_stream_finds_ndjson(tmp_path):
    run_dir = make_run_dir(tmp_path)
    stream = write_text(tmp_path / "analysis_stream.ndjson", "{}\n")
    assert _auto_detect_stream(run_dir) == stream


def test_auto_detect_stream_none_when_absent(tmp_path):
    run_dir = make_run_dir(tmp_path)
    assert _auto_detect_stream(run_dir) is None


def test_auto_detect_log_finds_log(tmp_path):
    run_dir = make_run_dir(tmp_path)
    log = write_text(tmp_path / "logs" / "tracelens_analysis" / "a.log", "hello\n")
    assert _auto_detect_log(run_dir) == log


def test_auto_detect_log_none_when_absent(tmp_path):
    run_dir = make_run_dir(tmp_path)
    assert _auto_detect_log(run_dir) is None


# ---------------------------------------------------------------------------
# run_triage
# ---------------------------------------------------------------------------


def test_run_triage_missing_run_dir_exits(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        run_triage(str(tmp_path / "does_not_exist"))
    assert exc.value.code == 1
    assert "OUTPUT_INCOMPLETE" in capsys.readouterr().out


def test_run_triage_warns_on_absent_stream_file(tmp_path, capsys):
    run_dir = make_run_dir(tmp_path)
    run_triage(str(run_dir), stream_file=str(tmp_path / "missing.ndjson"))
    assert "stream file not found" in capsys.readouterr().out


def test_run_triage_auto_detects_stream(tmp_path, capsys):
    run_dir = make_run_dir(tmp_path)
    write_text(tmp_path / "logs" / "tracelens_analysis" / "a.log", "benign\n")
    run_triage(str(run_dir))
    assert "Auto-detected stream file" in capsys.readouterr().out


def test_run_triage_implies_by_annotation(tmp_path):
    # 2c SPLIT_TRACE_MISSING declares implies_failures=["2f"]. With a resolvable
    # trace that has cpu_ops but no kernels (2f SPLIT_LOW_GPU_KERNELS fires) and
    # no trace_split/ (2c fires), the two-pass annotation must stamp the 2f
    # finding as implied by 2c. Both are detailed_only, so run with detailed=True.
    trace = write_json_gz(tmp_path / "trace.json.gz", [{"cat": "cpu_op"}] * 2)
    run_dir = make_run_dir(tmp_path, cmd_prefix=trace)

    findings = run_triage(str(run_dir), detailed=True)
    by_sublabel = {f.sublabel: f for f in findings}
    assert "2c" in by_sublabel, [f.tag for f in findings]
    assert "2f" in by_sublabel, [f.tag for f in findings]
    assert by_sublabel["2f"].implied_by == ["2c"]
    assert "(implied by 2c)" in by_sublabel["2f"].diag_line()


def test_run_triage_swallows_check_exception(tmp_path, capsys, monkeypatch):
    def _boom(run_dir, stream_file):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(runner, "ALL_CHECKS", [CheckSpec("9z", "BOOM", "test", _boom)])
    findings = run_triage(str(tmp_path))
    assert findings == []
    out = capsys.readouterr().out
    assert "_boom raised" in out and "kaboom" in out


# ---------------------------------------------------------------------------
# write_diag_txt / write_detail_csv
# ---------------------------------------------------------------------------


def _runner_finding(**kw):
    base = dict(
        tag="DIAG:test:1a_FOO",
        sublabel="1a",
        category="test",
        failure_mode="fm",
        evidence="ev",
        remedy="rm",
    )
    base.update(kw)
    return Finding(**base)


def test_write_diag_txt_no_findings(tmp_path):
    run_dir = make_run_dir(tmp_path)
    write_diag_txt([], str(run_dir))
    content = (run_dir / "triage_diags.txt").read_text()
    assert content == "No failures detected. Analysis run appears healthy.\n"


def test_write_diag_txt_with_findings(tmp_path):
    run_dir = make_run_dir(tmp_path)
    f = _runner_finding(implied_by=["2c", "2d"])
    write_diag_txt([f], str(run_dir))
    content = (run_dir / "triage_diags.txt").read_text()
    assert content == "[DIAG:test:1a_FOO] ev (implied by 2c, 2d)\n"


def test_write_detail_csv_columns_and_rows(tmp_path):
    run_dir = make_run_dir(tmp_path)
    f = _runner_finding(implied_by=["2c", "2d"])
    write_detail_csv([f], str(run_dir))

    with open(run_dir / "triage_details.csv", newline="") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == [
        "DIAG Tag",
        "Category",
        "Failure Mode",
        "Evidence",
        "Remedy",
        "Implied By",
    ]
    assert rows[1] == [
        "DIAG:test:1a_FOO",
        "test",
        "fm",
        "ev",
        "rm",
        "2c,2d",
    ]


# ---------------------------------------------------------------------------
# run_geak_triage
# ---------------------------------------------------------------------------


def test_run_geak_triage_missing_dir(tmp_path, capsys):
    findings = run_geak_triage(str(tmp_path / "nope"))
    assert findings == []
    assert "Session directory does not exist" in capsys.readouterr().out


def test_run_geak_triage_kernel_no_candidates_finding(tmp_path):
    session = make_geak_session(
        tmp_path, phases=["KERNEL_AGENT"], with_candidates=False
    )
    findings = run_geak_triage(str(session))
    assert [f.sublabel for f in findings] == ["6a"]
    assert "KERNEL_CANDIDATES_MISSING" in findings[0].tag


def test_run_geak_triage_with_candidates_none(tmp_path):
    session = make_geak_session(tmp_path, phases=["KERNEL_AGENT"], with_candidates=True)
    assert run_geak_triage(str(session)) == []


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


def test_main_requires_a_target(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["triage"])
    with pytest.raises(SystemExit) as exc:
        main()
    # argparse.error exits with code 2.
    assert exc.value.code == 2


def test_main_run_dir_writable_writes_and_exits(tmp_path, monkeypatch, capsys):
    run_dir = make_run_dir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["triage", "--run-dir", str(run_dir)])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    assert (run_dir / "triage_details.csv").is_file()
    assert (run_dir / "triage_diags.txt").is_file()
    assert "Details:" in capsys.readouterr().out


def test_main_run_dir_non_writable_warns(tmp_path, monkeypatch, capsys):
    run_dir = make_run_dir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["triage", "--run-dir", str(run_dir)])
    monkeypatch.setattr(os, "access", lambda *a, **k: False)
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "were not written" in out
    assert not (run_dir / "triage_details.csv").exists()
    assert not (run_dir / "triage_diags.txt").exists()


def test_main_session_dir_findings_exit(tmp_path, monkeypatch):
    session = make_geak_session(
        tmp_path, phases=["KERNEL_AGENT"], with_candidates=False
    )
    monkeypatch.setattr(sys, "argv", ["triage", "--session-dir", str(session)])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1


def test_main_no_findings_healthy(tmp_path, monkeypatch, capsys):
    # A session that never reached KERNEL_AGENT yields no GEAK findings, driving
    # the healthy else-branch: no SystemExit, "No failures detected" printed.
    session = make_geak_session(tmp_path, phases=["SETUP"], with_candidates=False)
    monkeypatch.setattr(sys, "argv", ["triage", "--session-dir", str(session)])
    main()
    assert "No failures detected" in capsys.readouterr().out
