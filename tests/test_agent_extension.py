###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for ``TraceLens.Agent.Analysis.utils.agent_extension``.

Sections:
    1. Shared fixtures (priority_data + manifest + marker-form analysis.md).
    2. ``MarkdownRehydrator`` static helpers.
    3. ``ImpactPlot`` data-stage methods (no I/O, no rendering).
    4. ``ImpactPlot._load_inputs`` filesystem orchestration.
    5. ``MarkdownRehydrator.run`` end-to-end on a synthesized analysis.md.
    6. ``ImpactPlot.run`` smoke + plot content (matplotlib Axes inspection).
    7. Cross-repo contract against ``evals/analysis_tests/e2e_tests.tar.gz``
       (overridable via ``E2E_TARBALL_PATH``): marker-attribute schema +
       rehydrator end-to-end + private-side grammar stability.

Run with ``pytest tests/test_agent_extension.py -v``.
"""

import base64
import json
import os
import re
import shutil
import sys
import tarfile

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from TraceLens.Agent.Analysis.utils.agent_extension import (
    ImpactPlot,
    MarkdownRehydrator,
    generate_impact_savings_plot,
)
from TraceLens.Agent.Analysis.utils.plot_utils import _CAT_PALETTE, _REST_KEY

# =============================================================================
# 1. Shared fixtures
# =============================================================================

BASELINE_MS = 100.0

RECS = [
    {
        "category": "gemm",
        "impact_score": 10.0,
        "impact_score_low": 8.0,
        "impact_score_high": 12.0,
        "operation_count": 2,
        "type": "kernel_tuning",
    },
    {
        "category": "attention",
        "impact_score": 5.0,
        "impact_score_low": 4.0,
        "impact_score_high": 6.0,
        "operation_count": 1,
        "type": "kernel_tuning",
    },
]

MANIFEST = {
    "platform": "MI300X",
    "gpu_utilization": {"total_time_ms": BASELINE_MS, "computation_time_percent": 95.0},
    "categories": [
        {
            "name": "gemm",
            "tier": "compute_kernel",
            "display_name": "GEMM",
            "gpu_kernel_time_ms": 60.0,
        },
        {
            "name": "attention",
            "tier": "compute_kernel",
            "display_name": "Attention",
            "gpu_kernel_time_ms": 30.0,
        },
    ],
}

# Synthesized marker-form analysis.md exercising all three kinds.
_MARKER_MD = """# Standalone

<!-- impact-begin kind=p_item low=10 high=20 category=gemm -->
**Impact**: ~10.0\u201320.0% of E2E
<!-- impact-end -->

<!-- impact-begin kind=detail_estimate low=10 high=20 -->
- old body
<!-- impact-end -->

<!-- impact-begin kind=top_ops -->
| Rank | Category | Time (ms) | % of Compute Time | Ops |
|------|----------|-----------|-------------------|-----|
| 1 | GEMM | 60.0 | 60.0 | 2 | <!-- top-ops-row low=10 high=20 -->
| 2 | Attention | 30.0 | 30.0 | 1 | <!-- top-ops-row low=null high=null -->
<!-- impact-end -->
"""


def _write_inputs(out, baseline_ms=BASELINE_MS, recs=RECS, manifest=MANIFEST):
    """Write priority_data.json + category_data/category_manifest.json into ``out``."""
    out = str(out)
    os.makedirs(os.path.join(out, "category_data"), exist_ok=True)
    with open(os.path.join(out, "priority_data.json"), "w") as f:
        json.dump(
            {
                "baseline_ms": baseline_ms,
                "recommendations": recs,
                "priorities": [],
                "findings": [],
                "all_estimates": [],
            },
            f,
        )
    with open(os.path.join(out, "category_data", "category_manifest.json"), "w") as f:
        json.dump(manifest, f)
    return out


@pytest.fixture
def output_dir(tmp_path):
    """Fully-populated ImpactPlot input dir (priority_data + manifest)."""
    return _write_inputs(tmp_path)


@pytest.fixture
def plot_with_data():
    """ImpactPlot instance with data-stage methods already invoked (no I/O)."""
    p = ImpactPlot(output_dir="<unused>", title="t")
    p.baseline_ms = BASELINE_MS
    p.recommendations = list(RECS)
    p.manifest = dict(MANIFEST)
    p._compute_projections()
    p._build_segments()
    p._build_color_map()
    return p


@pytest.fixture
def marker_dir(tmp_path):
    """Output dir with manifest + a marker-form analysis.md ready for rehydration."""
    out = str(tmp_path)
    os.makedirs(os.path.join(out, "category_data"))
    with open(os.path.join(out, "category_data", "category_manifest.json"), "w") as f:
        json.dump(MANIFEST, f)
    md = os.path.join(out, "analysis.md")
    with open(md, "w") as f:
        f.write(_MARKER_MD)
    return out, md


# =============================================================================
# 2. MarkdownRehydrator static helpers
# =============================================================================


def test_parse_attrs():
    a = MarkdownRehydrator._parse_attrs('kind=p_item low=10 high="20.5" cat="g"')
    assert a == {"kind": "p_item", "low": "10", "high": "20.5", "cat": "g"}
    assert MarkdownRehydrator._parse_attrs("low=null high=5") == {
        "low": None,
        "high": "5",
    }
    assert MarkdownRehydrator._parse_attrs("") == {}


def test_attr_float():
    f = MarkdownRehydrator._attr_float
    assert f({"x": "3.14"}, "x") == 3.14
    assert f({"x": None}, "x") is None
    assert f({}, "x") is None
    assert f({"x": "abc"}, "x") is None


def test_render_p_item():
    R = MarkdownRehydrator._render_p_item
    out = R({"low": "10", "high": "20", "category": "gemm"}, "", 100.0)
    assert "~10.0\u201320.0 ms savings (10.0\u201320.0% of E2E)" in out
    assert "gemm_metrics.json" in out
    assert "gemm_metrics.json" not in R({"low": "10", "high": "20"}, "", 100.0)
    assert (
        R({"low": None, "high": "20"}, "", 100.0)
        == "**Impact**: Not quantifiable from trace data"
    )


def test_render_detail_estimate():
    R = MarkdownRehydrator._render_detail_estimate
    out = R({"low": "10", "high": "20"}, "x", 100.0)
    assert "Low end (75% roofline): 10.000 ms savings (10.00% E2E)" in out
    assert "High end (100% roofline): 20.000 ms savings (20.00% E2E)" in out
    assert R({"low": None}, "body", 100.0) == "body"


def test_render_top_ops_row():
    R = MarkdownRehydrator._render_top_ops_row
    assert R(
        "| 1 | a | 1 | 1 | 1 | <!-- top-ops-row low=10 high=20 -->", 100.0
    ).endswith("| ~10.0\u201320.0 ms (10.0\u201320.0%) |")
    assert R("| 1 | a | 1 | 1 | 1 |", 100.0) == "| 1 | a | 1 | 1 | 1 | -- |"
    assert R(
        "| 1 | a | 1 | 1 | 1 | <!-- top-ops-row low=null high=null -->", 100.0
    ).endswith("| -- |")


def test_render_top_ops_swaps_header():
    body = (
        "| Rank | Category | Time (ms) | % of Compute Time | Ops |\n"
        "|------|----------|-----------|-------------------|-----|\n"
        "| 1 | GEMM | 60.0 | 60.0 | 2 | <!-- top-ops-row low=10 high=20 -->"
    )
    out = MarkdownRehydrator._render_top_ops({}, body, 100.0)
    assert "Potential improvement (time, E2E %)" in out
    assert "~10.0\u201320.0 ms (10.0\u201320.0%)" in out


def test_render_legacy_dispatch():
    R = MarkdownRehydrator._render_legacy
    attrs = {"low": "10", "high": "20"}
    assert R("p_item", attrs, "", 100.0)
    assert R("detail_estimate", attrs, "x", 100.0)
    assert R("top_ops", {}, "body", 100.0) == "body"
    assert R("unknown", attrs, "", 100.0) is None


# =============================================================================
# 3. ImpactPlot data-stage methods (no rendering)
# =============================================================================


def test_compute_projections(plot_with_data):
    p = plot_with_data.proj
    assert p["steps"] == ["Baseline", "Gemm", "Attention"]
    np.testing.assert_allclose(p["e2e_ms"], [100.0, 90.0, 85.0])
    assert p["savings"] == [0, 10.0, 5.0]
    np.testing.assert_allclose(p["rel"], [100.0, 111.1, 117.6], atol=0.05)
    assert p["err_lo"].shape == (3,) and (p["err_lo"] >= 0).all()
    assert p["err_hi"].shape == (3,) and (p["err_hi"] >= 0).all()


def test_build_segments(plot_with_data):
    bbc = plot_with_data.baseline_by_cat
    assert (bbc["gemm"], bbc["attention"], bbc[_REST_KEY]) == (60.0, 30.0, 10.0)
    assert plot_with_data.segment_order == ["attention", "gemm", _REST_KEY]


def test_build_color_map(plot_with_data):
    cm = plot_with_data.cat_color_map
    assert cm["gemm"] == _CAT_PALETTE[0]
    assert cm["attention"] == _CAT_PALETTE[1]


# =============================================================================
# 4. ImpactPlot._load_inputs
# =============================================================================


def test_load_inputs_missing(tmp_path):
    assert ImpactPlot(str(tmp_path), "t")._load_inputs() is False


def test_load_inputs_valid(output_dir):
    p = ImpactPlot(output_dir, "t")
    assert p._load_inputs() is True
    assert p.baseline_ms == BASELINE_MS
    assert len(p.recommendations) == 2


def test_load_inputs_zero_baseline(tmp_path):
    out = _write_inputs(tmp_path, baseline_ms=0.0, recs=[])
    assert ImpactPlot(out, "t")._load_inputs() is False


# =============================================================================
# 5. MarkdownRehydrator.run end-to-end (synthesized fixture)
# =============================================================================


def test_rehydrator_rewrites(marker_dir):
    out, md = marker_dir
    assert MarkdownRehydrator(out).run()[md] == "rewritten"
    text = open(md).read()
    assert "<!-- impact-begin" not in text and "<!-- top-ops-row" not in text
    assert "~10.0\u201320.0 ms savings (10.0\u201320.0% of E2E)" in text
    assert "gemm_metrics.json" in text
    assert "Low end (75% roofline): 10.000 ms savings" in text
    assert "Potential improvement (time, E2E %)" in text
    assert "~10.0\u201320.0 ms (10.0\u201320.0%)" in text
    assert "| -- |" in text


def test_rehydrator_idempotent(marker_dir):
    out, md = marker_dir
    MarkdownRehydrator(out).run()
    assert MarkdownRehydrator(out).run()[md] == "skipped_no_match"


def test_rehydrator_no_manifest(tmp_path):
    (tmp_path / "analysis.md").write_text(_MARKER_MD)
    assert MarkdownRehydrator(str(tmp_path)).run() == {}


# =============================================================================
# 6. ImpactPlot.run smoke + plot content (matplotlib Axes inspection)
# =============================================================================


def test_generate_impact_savings_plot(output_dir):
    """Smoke test: PNG and base64 sidecar produced."""
    assert generate_impact_savings_plot(output_dir, "Title", write_base64=True) is True
    assert os.path.getsize(os.path.join(output_dir, "perf_improvement.png")) > 0
    base64.b64decode(
        open(os.path.join(output_dir, "perf_improvement_base64.txt")).read()
    )


def _build_axes(plot):
    fig, (ax_stack, ax_cone) = plt.subplots(1, 2)
    plot._render_stacked_bars(ax_stack)
    plot._render_throughput_cone(ax_cone)
    return fig, ax_stack, ax_cone


def test_stacked_bars_content(plot_with_data):
    fig, ax, _ = _build_axes(plot_with_data)
    try:
        assert len(ax.patches) == 9
        assert [t.get_text() for t in ax.get_xticklabels()] == [
            "Baseline",
            "Gemm",
            "Attention",
        ]
        assert ax.get_ylabel() == "E2E time stacked by category (ms)"
        assert "Projected E2E Latency" in ax.get_title()
        # Per-bar total height (sum across the 3 stacked segments at each x).
        totals = {0: 0.0, 1: 0.0, 2: 0.0}
        for p in ax.patches:
            totals[int(round(p.get_x() + p.get_width() / 2))] += p.get_height()
        np.testing.assert_allclose(
            [totals[0], totals[1], totals[2]], [100.0, 90.0, 85.0], atol=1e-6
        )
        labels = [t.get_text() for t in ax.texts]
        for s in ("100.0 ms", "90.0 ms", "85.0 ms", "-10.0 ms", "-5.0 ms"):
            assert s in labels
    finally:
        plt.close(fig)


def test_cone_content(plot_with_data):
    fig, _, ax = _build_axes(plot_with_data)
    try:
        assert ax.get_ylabel() == "% Relative Throughput (Baseline = 100)"
        assert "Cumulative Throughput Improvement" in ax.get_title()
        marker_line = next(
            l for l in ax.lines if l.get_marker() == "o" and len(l.get_ydata()) == 3
        )
        np.testing.assert_allclose(marker_line.get_xdata(), [0, 1, 2])
        np.testing.assert_allclose(
            marker_line.get_ydata(), [100.0, 111.1, 117.6], atol=0.05
        )
        assert any(
            l.get_linestyle() == "--" and (np.array(l.get_ydata()) == 100).all()
            for l in ax.lines
        )
        assert len(ax.collections) >= 1
        assert "118%" in [t.get_text() for t in ax.texts]
        assert ax.yaxis.get_major_formatter()(100.0, 0) == "100%"
    finally:
        plt.close(fig)


def test_run_figure_contents(monkeypatch, output_dir):
    """Capture the figure built by run() (without saving) and inspect suptitle/footer."""
    captured = {}
    monkeypatch.setattr(
        ImpactPlot, "_save", lambda self, fig: captured.setdefault("fig", fig)
    )
    assert ImpactPlot(output_dir, "My Title").run() is True
    fig = captured["fig"]
    try:
        assert fig._suptitle.get_text() == "My Title"
        assert len(fig.axes) == 2
        assert any(
            "Gray bars represent categories without performance models" in t.get_text()
            for t in fig.texts
        )
    finally:
        plt.close(fig)


# =============================================================================
# 7. Cross-repo contract via evals/analysis_tests/e2e_tests.tar.gz
# =============================================================================
#
# Purpose: catch contract drift between the public repo (which emits the
# marker-form analysis.md) and the private repo (which hosts the rehydrator).
#
# - Schema for required attrs is mirrored from the public-repo templates at
#   ``TraceLens/Agent/Analysis/utils/templates/``:
#     analysis_template.md -> kind=top_ops, kind=p_item (category/low/mid/high)
#     sub_agent_spec.md   -> kind=p_item (low/mid/high), kind=detail_estimate (low/high)
# - Tarball scenarios are discovered at collection time, so each scenario
#   appears as its own parametrized test entry. Tests skip cleanly when the
#   tarball is missing or pre-rehydration markers are absent.

TARBALL_PATH = os.environ.get(
    "E2E_TARBALL_PATH",
    os.path.join(REPO_ROOT, "evals", "analysis_tests", "e2e_tests.tar.gz"),
)

REQUIRED_MARKER_ATTRS = {
    "p_item": {"low", "mid", "high"},
    "detail_estimate": {"low", "high"},
    "top_ops": set(),
}
REQUIRED_TRAILER_ATTRS = {"low", "high"}


def _list_tarball_scenarios(path):
    """Return sorted scenario names from ``e2e_tests/<scenario>/`` in the tarball."""
    if not os.path.isfile(path):
        return []
    scenarios = set()
    with tarfile.open(path, "r:gz") as tf:
        for name in tf.getnames():
            parts = name.split("/")
            if len(parts) >= 2 and parts[0] == "e2e_tests" and parts[1]:
                scenarios.add(parts[1])
    return sorted(scenarios)


def _check_marker_schema(text):
    """Return a list of human-readable schema violations; empty list = OK."""
    issues = []
    for m in re.finditer(r"<!--\s*impact-begin\s+(.*?)\s*-->", text):
        attrs = MarkdownRehydrator._parse_attrs(m.group(1))
        kind = attrs.pop("kind", None)
        if kind is None:
            issues.append(f"impact-begin missing kind=: {m.group(0)!r}")
        elif kind not in REQUIRED_MARKER_ATTRS:
            issues.append(f"impact-begin unknown kind={kind!r}: {m.group(0)!r}")
        elif missing := REQUIRED_MARKER_ATTRS[kind] - set(attrs):
            issues.append(
                f"impact-begin kind={kind} missing {sorted(missing)}: {m.group(0)!r}"
            )
    for m in re.finditer(r"<!--\s*top-ops-row\s+(.*?)\s*-->", text):
        attrs = MarkdownRehydrator._parse_attrs(m.group(1))
        if missing := REQUIRED_TRAILER_ATTRS - set(attrs):
            issues.append(f"top-ops-row missing {sorted(missing)}: {m.group(0)!r}")
    return issues


TARBALL_SCENARIOS = _list_tarball_scenarios(TARBALL_PATH)


@pytest.fixture(scope="session")
def tarball_root(tmp_path_factory):
    """Extract the e2e_tests tarball once per session; returns the e2e_tests/ dir."""
    if not os.path.isfile(TARBALL_PATH):
        pytest.skip(f"e2e_tests tarball not found at {TARBALL_PATH}")
    root = tmp_path_factory.mktemp("e2e_tests_extract")
    with tarfile.open(TARBALL_PATH, "r:gz") as tf:
        tf.extractall(root)
    return root / "e2e_tests"


# ----- 7a. Marker schema (validated on raw tarball analysis.md) --------------


@pytest.mark.parametrize("scenario", TARBALL_SCENARIOS or ["__no_tarball__"])
def test_marker_schema_against_tarball(tarball_root, scenario):
    """Every marker carries the attrs the templates promise."""
    if scenario == "__no_tarball__":
        pytest.skip(f"e2e_tests tarball not found at {TARBALL_PATH}")
    md = tarball_root / scenario / "analysis_output_ref" / "analysis.md"
    if not md.is_file():
        pytest.skip(f"scenario {scenario} has no analysis.md")
    text = md.read_text()
    if "<!-- impact-begin" not in text:
        pytest.skip(f"{scenario}/analysis.md has no markers (post-rehydration)")

    issues = _check_marker_schema(text)
    assert not issues, f"{scenario}: schema violations:\n  " + "\n  ".join(issues)
    # Orchestrator template mandates exactly one top_ops table per analysis.md.
    assert text.count("kind=top_ops") == 1, f"{scenario}: expected 1 kind=top_ops"
    assert text.count("<!-- top-ops-row") >= 1, f"{scenario}: no top-ops-row trailers"
    assert text.count("<!-- impact-begin") == text.count(
        "<!-- impact-end"
    ), f"{scenario}: unmatched impact-begin / impact-end markers"


# ----- 7b. Rehydrator end-to-end (per-kind multiplicity checks) --------------


@pytest.mark.parametrize("scenario", TARBALL_SCENARIOS or ["__no_tarball__"])
def test_rehydrator_against_tarball(tarball_root, tmp_path, scenario):
    """Each marker kind that's present in the input is rendered the right number of times."""
    if scenario == "__no_tarball__":
        pytest.skip(f"e2e_tests tarball not found at {TARBALL_PATH}")
    src = tarball_root / scenario / "analysis_output_ref"
    if not src.is_dir():
        pytest.skip(f"scenario {scenario} not in tarball")
    dst = tmp_path / "analysis_output"
    shutil.copytree(src, dst)
    md = dst / "analysis.md"
    original = md.read_text()
    if "<!-- impact-begin" not in original:
        pytest.skip(f"{scenario}/analysis.md has no markers (post-rehydration)")

    n_p_items = original.count("kind=p_item")
    n_details = original.count("kind=detail_estimate")
    has_top_ops = "kind=top_ops" in original

    # Synthesize a minimal manifest so the rehydrator can resolve baseline_ms.
    (dst / "category_data").mkdir(exist_ok=True)
    (dst / "category_data" / "category_manifest.json").write_text(
        json.dumps({"gpu_utilization": {"total_time_ms": 100.0}, "categories": []})
    )

    results = MarkdownRehydrator(str(dst)).run()
    assert any(v == "rewritten" for v in results.values()), results

    text = md.read_text()
    assert "<!-- impact-begin" not in text
    assert "<!-- impact-end" not in text
    assert "<!-- top-ops-row" not in text

    if n_p_items:
        n_rendered = len(re.findall(r"\*\*Impact\*\*: ~\d.*ms savings", text))
        assert (
            n_rendered >= n_p_items
        ), f"{scenario}: expected >={n_p_items} rendered p_item lines, got {n_rendered}"
        assert (
            "Not quantifiable from trace data" not in text
        ), f"{scenario}: p_item rendering fell back to fallback string"
    if n_details:
        assert text.count("Low end (75% roofline):") >= n_details
        assert text.count("High end (100% roofline):") >= n_details
    if has_top_ops:
        assert "Potential improvement (time, E2E %)" in text
        rows = [
            l
            for l in text.splitlines()
            if l.startswith("|") and re.match(r"\|\s*\d+\s*\|", l)
        ]
        assert rows, f"{scenario}: top_ops table has no data rows"
        for row in rows:
            assert re.search(r"~\d.*ms \(\d", row) or row.rstrip().endswith(
                "| -- |"
            ), f"{scenario}: top_ops row not rewritten: {row!r}"


# ----- 7c. Private-side grammar stability (independent of tarball) ------------


def test_marker_grammar_stability():
    """Pin the regexes and supported kinds so private-side renames are caught fast."""
    R = MarkdownRehydrator
    assert R._RE_MARKER_BEGIN.search("<!-- impact-begin kind=p_item low=10 high=20 -->")
    assert R._RE_MARKER_END.search("<!-- impact-end -->")
    assert R._RE_TOP_OPS_ROW_TRAILER.search(
        "| 1 | a | 1 | 1 | 1 | <!-- top-ops-row low=10 high=20 -->"
    )
    attrs = {"low": "10", "high": "20"}
    for kind in ("p_item", "detail_estimate", "top_ops"):
        assert R._render_legacy(kind, attrs, "body", 100.0) is not None
    assert R._render_legacy("unknown", attrs, "", 100.0) is None
    body = (
        "| Rank | Category | Time (ms) | % of Compute Time | Ops |\n"
        "|--|--|--|--|--|\n| 1 | a | 1 | 1 | 1 |"
    )
    assert R._LEGACY_TOP_OPS_HEADER in R._render_top_ops({}, body, 100.0)
