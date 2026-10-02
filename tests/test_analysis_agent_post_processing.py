###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for the grouped ``analysis.json`` producer (``render_analysis_json``).

Standalone by construction: every fixture is an ``analysis.md`` baked into this
file as a module-level string constant, trimmed from real reports down to the
tables, headings, and markers the renderer anchors on. Each test writes a
constant into ``tmp_path`` and renders there (the renderer writes
``analysis.json`` beside the md); nothing is read from disk. No GPU, no LLM, no
network.

The constants exercise: the agentic envelope + operation grouping
(``_AGENTIC_MD``), the deterministic-fallback path (``_FALLBACK_MD``), the
pct_e2e fallback + fan-out row when a table has no op_row marker
(``_NO_OP_ROW_MD``), and the em-dash null Operation cell on an agentic report
(``_AGENTIC_NULL_OP_MD``). The remaining ``_render_text`` cases build minimal md
inline for prose caps, missing columns, em-dash cells, and empty findings.
"""

import json
import re
import subprocess
import sys

import pytest

from TraceLens.Agent.Analysis.post_processing import (
    AnalysisReport,
    ComputeMember,
    ComputeTask,
    Impact,
    ReportInfo,
    render_analysis_json,
)
from TraceLens.Agent.Analysis.category_analyses.analysis_utils import (
    HEURISTIC_FRACTION_MID,
)
from TraceLens.Agent.Analysis.utils.validation_utils import (
    _find_data_table,
    _iter_compute_candidate_blocks,
)

# Constants
_NULL_CELLS = {"", "-", "—", "–"}

# --------------------------------------------------------------------------- #
# Inline md fixtures — trimmed from real single-trace reports so numbers and
# markers stay realistic, but no disk corpus is touched.
# --------------------------------------------------------------------------- #

# Full agentic envelope: executive summary + metrics, top_operations marker,
# category cards with detailed-block anchors, appendix, and two compute cards.
# P1 groups three tile-variant gemm_a16w16 rows (op_row impacts=3.55,3.29,1.77,
# library AITER, category gemm, "33.29% of 708 TFLOPS" efficiency). P2 is a
# distinct MoE op (op_row impacts=7.61, category moe_unfused).
_AGENTIC_MD = """# ExampleMoE - MI300X Standalone Analysis

<!-- report-begin kind=report_mode mode=agentic -->
<!-- report-end -->

## Executive Summary

This standalone roofline analysis shows GPU computation at 84.24% of the 2494.24 ms timeline, with 15.48% idle and no exposed communication.

| Metric | Value |
|--------|-------|
| Total Time | 2494.24 ms |
| Compute % | 84.24% |
| Idle % | 15.48% |
| Exposed Communication % | 0.00% |
| Top Bottleneck Category | GEMM (21.37%) |

## Compute Kernel Optimizations

### Top Operations

<!-- impact-begin kind=top_ops rehydrated=true -->
| Rank | Category | Time (ms) | % of Compute Time | Ops | Potential improvement (time, E2E %) |
|------|----------|-----------|-------------------|-----|-------------------------------------|
| 1 | GEMM | 449.043 | 21.37% | 4 | ~225.5–300.3 ms (9.0–12.0%) |
| 2 | InferenceAttention | 353.088 | 16.80% | 3 | ~72.6–97.0 ms (2.9–3.9%) |
<!-- impact-end -->

### 🔴 P1: Compute-bound BF16 GEMMs below matrix roofline (AITER)

**Insight**: Three BF16 GEMMs run at 30-52% of the BF16 matrix roofline.

<!-- impact-begin kind=p_item category=gemm low=7.38 mid=8.61 high=9.83 rehydrated=true -->
**Impact**: ~184.1–245.2 ms (7.4–9.8%)
<!-- impact-end -->

→ *See [Detailed Analysis: Compute kernel insights > P1](#detailed-analysis-compute-p1) for details*

### 🟡 P2: FC1 expert stage dominates MoE time (AITER)

**Insight**: The first MoE expert stage is the largest single share of compute.

<!-- impact-begin kind=p_item category=moe_unfused low=3.8 mid=7.61 high=11.41 rehydrated=true -->
**Impact**: ~94.8–284.6 ms (3.8–11.4%)
<!-- impact-end -->

→ *See [Detailed Analysis: Compute kernel insights > P2](#detailed-analysis-compute-p2) for details*

## Detailed Analysis

### Compute Kernel Insights

<a id="detailed-analysis-compute-p1"></a>
<!-- reasoning-candidate tier=compute rank=1 -->
#### 🔴 P1: Compute-bound BF16 GEMMs run well below the matrix roofline (AITER)
**Identification:** Three BF16 general matrix multiplies launched through the AITER backend run below the compute roofline.

**Data:**

| Operation |  Args  | Kernel Path | Kernel Name | Time (ms) | %E2E | Count |FLOPS/Byte| Efficiency | Bound |
|-----------|--------|-------------|-------------|-----------|------|-------|----------|------------|-------|
| aiter::gemm_a16w16 | (640,2880) bf16<br>(5120,2880) bf16 | aiter/tuned_gemm.py(252): gemm_a16w16 | Cijk_Alik_Bljk_MT128x128x64_MI32x32x1 | 151.565 | 6.08 | 4608 | 475.01 | 33.29% of 708 TFLOPS | compute-bound |
| aiter::gemm_a16w16 | (640,4096) bf16<br>(2880,4096) bf16 | aiter/tuned_gemm.py(252): gemm_a16w16 | Cijk_Alik_Bljk_MT128x128x128_MI16x16x1 | 134.148 | 5.38 | 4608 | 464.26 | 30.09% of 708 TFLOPS | compute-bound |
| aiter::gemm_a16w16 | (640,2880) bf16<br>(201088,2880) bf16 | aiter/tuned_gemm.py(252): gemm_a16w16 | Cijk_Alik_Bljk_MT192x320x64_MI16x16x1 | 105.412 | 4.23 | 128 | 522.28 | 52.19% of 708 TFLOPS | compute-bound |
<!-- impact-begin kind=op_row rank=1 impacts=3.55,3.29,1.77 -->
<!-- impact-end -->

**Reasoning for Slowdown:** The transformer-layer GEMMs achieve 33.29% and 30.09% of the 708 TFLOPS BF16 matrix peak and are firmly compute-bound.

**Resolution:** Tile-size and wave-occupancy tuning, or narrowing precision to FP8/FP4, lowers the compute floor.

**Impact estimate:**
<!-- impact-begin kind=detail_estimate low=7.38 high=9.83 rehydrated=true -->
- Low end (75% roofline): 184.075 ms savings (7.38% E2E)
- High end (100% roofline): 245.184 ms savings (9.83% E2E)
<!-- impact-end -->

<a id="detailed-analysis-compute-p2"></a>
<!-- reasoning-candidate tier=compute rank=2 -->
#### 🟡 P2: FC1 expert stage dominates MoE time as a separate GEMM launch (AITER)
**Identification:** This operation is the first stage of the MoE expert path, launched from the AITER flydsl stage-1 wrapper.

**Data:**

| Operation |  Args  | Kernel Path | Kernel Name | Time (ms) | %E2E | Count |FLOPS/Byte| Efficiency | Bound |
|-----------|--------|-------------|-------------|-----------|------|-------|----------|------------|-------|
| pseudo_op::moe_flydsl_stage1 | (640,3072) bf16<br>(128,6144,1536) fp4 | aiter/fused_moe.py(839): _flydsl_stage1_wrapper | mfma_moe1_silu_mul_afp4_wfp4_fp4_t64x128x256_pm1 | 632.42 | 25.36 | 4608 | 78.72 | — | — |
<!-- impact-begin kind=op_row rank=1 impacts=7.61 -->
<!-- impact-end -->

**Reasoning for Slowdown:** This stage carries the largest share of end-to-end GPU time in the whole compute set.

**Resolution:** Fusing the FC1 and FC2 stages keeps the intermediate activation on-chip and removes an HBM round-trip.

**Impact estimate:**
<!-- impact-begin kind=detail_estimate low=3.8 high=11.41 rehydrated=true -->
- Low end (75% roofline): 94.781 ms savings (3.80% E2E)
- High end (100% roofline): 284.593 ms savings (11.41% E2E)
<!-- impact-end -->

## Appendix

### Model Architecture
- **Model**: ExampleMoE
- **Architecture**: Mixture-of-Experts Transformer
- **Scale**: 100B

### Hardware Reference
- **Platform**: MI300X
- **Peak HBM BW**: 5.3 TB/s
- **Peak MAF (BF16)**: 708 TFLOPS
"""

# Deterministic-fallback report: report_mode=deterministic-fallback plus a
# warning blockquote, four distinct Cijk tile variants (grouped by kernel_name,
# so they stay four tasks), em-dash Operation cells, and one op_row per card.
# The p_item cards carry category=unknown but no detailed-block anchor, so
# category resolves to null.
_FALLBACK_MD = """# Deterministic Fallback Analysis

<!-- report-begin kind=report_mode mode=deterministic-fallback -->
<!-- report-end -->

<!-- report-begin kind=warning -->
> **⚠ Degraded (deterministic fallback) report.** This trace was
> graph-under-recorded (graph-replay fraction 84%;
> per-op decomposition failed, so the agentic analysis path did not execute.
<!-- report-end -->

<!-- reasoning-candidate tier=compute rank=1 -->
#### P1: Cijk_Ailk_Bljk_HHS_MT128x160x16
<!-- impact-begin kind=p_item category=unknown low=1.2 mid=2.4 high=3.6 -->
<!-- impact-end -->

**Data:**

| Operation | Args | Kernel Path | Kernel Name | Time (ms) | %E2E | Count | FLOPS/Byte | Efficiency | Bound |
|---|---|---|---|---|---|---|---|---|---|
| — | — | — | Cijk_Ailk_Bljk_HHS_MT128x160x16 | 12.345 | 5.00 | 10 | — | — | — |
<!-- impact-begin kind=op_row rank=1 impacts=2.4 -->
<!-- impact-end -->

<!-- reasoning-candidate tier=compute rank=2 -->
#### P2: Cijk_Ailk_Bljk_HHS_MT64x64x16
<!-- impact-begin kind=p_item category=unknown low=0.9 mid=1.8 high=2.7 -->
<!-- impact-end -->

**Data:**

| Operation | Args | Kernel Path | Kernel Name | Time (ms) | %E2E | Count | FLOPS/Byte | Efficiency | Bound |
|---|---|---|---|---|---|---|---|---|---|
| — | — | — | Cijk_Ailk_Bljk_HHS_MT64x64x16 | 9.876 | 4.00 | 8 | — | — | — |
<!-- impact-begin kind=op_row rank=2 impacts=1.8 -->
<!-- impact-end -->

<!-- reasoning-candidate tier=compute rank=3 -->
#### P3: Cijk_Ailk_Bljk_HHS_MT256x128x32
<!-- impact-begin kind=p_item category=unknown low=0.6 mid=1.2 high=1.8 -->
<!-- impact-end -->

**Data:**

| Operation | Args | Kernel Path | Kernel Name | Time (ms) | %E2E | Count | FLOPS/Byte | Efficiency | Bound |
|---|---|---|---|---|---|---|---|---|---|
| — | — | — | Cijk_Ailk_Bljk_HHS_MT256x128x32 | 6.543 | 3.00 | 6 | — | — | — |
<!-- impact-begin kind=op_row rank=3 impacts=1.2 -->
<!-- impact-end -->

<!-- reasoning-candidate tier=compute rank=4 -->
#### P4: Cijk_Ailk_Bljk_HHS_MT32x128x64
<!-- impact-begin kind=p_item category=unknown low=0.3 mid=0.6 high=0.9 -->
<!-- impact-end -->

**Data:**

| Operation | Args | Kernel Path | Kernel Name | Time (ms) | %E2E | Count | FLOPS/Byte | Efficiency | Bound |
|---|---|---|---|---|---|---|---|---|---|
| — | — | — | Cijk_Ailk_Bljk_HHS_MT32x128x64 | 3.210 | 2.00 | 4 | — | — | — |
<!-- impact-begin kind=op_row rank=4 impacts=0.6 -->
<!-- impact-end -->
"""

# Agentic report whose compute table carries NO op_row marker: impact_score
# falls back to pct_e2e. The first row is a fan-out (Kernel 1 / Kernel 2), which
# must collapse into one member holding a 2-element kernel_name list.
_NO_OP_ROW_MD = """# ExampleDecoder - MI300X Standalone Analysis

<!-- report-begin kind=report_mode mode=agentic -->
<!-- report-end -->

### Compute Kernel Insights

<a id="detailed-analysis-compute-p1"></a>
<!-- reasoning-candidate tier=compute rank=1 -->
#### P1: MLA paged-decode attention
**Identification:** The MLA decode op launches a reduce kernel and a core attention kernel per call.

**Data:**

| Operation | Args | Kernel Path | Kernel Name | Time (ms) | %E2E | Count | FLOPS/Byte | Efficiency | Bound |
|---|---|---|---|---|---|---|---|---|---|
| pseudo_mla_decode_fwd | (64,64,576) fp8<br>(65,) int | aiter/mla.py(157): mla_decode_fwd | Kernel 1: void kn_mla_reduce_v1_ps<MlaReduceKernelV1Traits<512, 64, 1>, float><br>Kernel 2: aiter::mla_a8w8_qh64_qseqlen1_gqaratio64_v3_ps | 255.433 | 6.59 | 7808 | 120.48 | 35.13% of 5.3 TB/s | memory-bound |
| pseudo_mla_decode_fwd | (64,64,512) bf16 | aiter/mla.py(160): mla_reduce | reduce_scatter_store | 100.000 | 3.00 | 100 | 10.00 | — | — |

**Reasoning for Slowdown:** The decode path is memory-bound and pays a per-call reduce launch.

**Resolution:** Fuse the reduce into the core kernel to drop the extra launch.
"""

# Agentic report whose single Operation cell is the em-dash null sentinel; the
# renderer must emit operation=None rather than the raw glyph.
_AGENTIC_NULL_OP_MD = """# Null Op - MI300X Standalone Analysis

<!-- report-begin kind=report_mode mode=agentic -->
<!-- report-end -->

### Compute Kernel Insights

<a id="detailed-analysis-compute-p1"></a>
<!-- reasoning-candidate tier=compute rank=1 -->
#### P1: op-less kernel
**Identification:** No host-side operation was captured for this kernel.

**Data:**

| Operation | Args | Kernel Path | Kernel Name | Time (ms) | %E2E | Count | FLOPS/Byte | Efficiency | Bound |
|---|---|---|---|---|---|---|---|---|---|
| — | (64,64) bf16 | k.py(1): f | k_null | 5.0 | 5.0 | 1 | — | — | — |
<!-- impact-begin kind=op_row rank=1 impacts=5.0 -->
<!-- impact-end -->

**Resolution:** fix
"""

# Agentic report whose op_row CSV carries a spec-legal ``—`` null for the second
# row; that row must fall back to its own %E2E while the first takes the marker.
_NULL_CSV_ENTRY_MD = """# Null CSV - MI300X Standalone Analysis

<!-- report-begin kind=report_mode mode=agentic -->
<!-- report-end -->

### Compute Kernel Insights

<a id="detailed-analysis-compute-p1"></a>
<!-- reasoning-candidate tier=compute rank=1 -->
#### P1: mixed-impact gemm
**Identification:** Two rows, one with a quantified per-row impact and one null.

**Data:**

| Operation | Args | Kernel Path | Kernel Name | Time (ms) | %E2E | Count | FLOPS/Byte | Efficiency | Bound |
|---|---|---|---|---|---|---|---|---|---|
| aiter::gemm | (1,2) bf16 | g.py(1): a | k_a | 10.0 | 4.20 | 1 | — | — | — |
| aiter::gemm | (3,4) bf16 | g.py(2): b | k_b | 5.0 | 1.75 | 1 | — | — | — |
<!-- impact-begin kind=op_row rank=1 impacts=3.5,— -->
<!-- impact-end -->

**Resolution:** fix
"""

# (name, md, mode, has_op_row, n_warn) — the inline replacement for the disk
# fixture matrix. Corpus-wide smoke/determinism/faithfulness tests iterate this.
_MATRIX = [
    ("agentic", _AGENTIC_MD, "agentic", True, 0),
    ("fallback", _FALLBACK_MD, "deterministic-fallback", True, 1),
    ("no_op_row", _NO_OP_ROW_MD, "agentic", False, 0),
    ("null_op", _AGENTIC_NULL_OP_MD, "agentic", True, 0),
]


def _render_text(md_text, tmp_path):
    """Render an in-memory md string into analysis.json under ``tmp_path``."""
    dst = tmp_path / "analysis.md"
    dst.write_text(md_text)
    out = render_analysis_json(dst)
    return json.loads(out.read_text()), out


# --------------------------------------------------------------------------- #
# Structural schema check (no typeguard dependency).
# --------------------------------------------------------------------------- #

_MEMBER_TYPES = {
    "impact_score": (int, float),
    "kernel_launcher_path": (str, type(None)),
    "library": (str, type(None)),
    "category": (str, type(None)),
    "analysis_md_rank": (str, type(None)),
    "kernel_name": (list,),
    "args_shapes": (list, type(None)),
    "args_datatypes": (list, type(None)),
    "time_ms": (int, float, type(None)),
    "count": (int, type(None)),
    "pct_e2e": (int, float, type(None)),
    "flops_per_byte": (int, float, type(None)),
    "efficiency_percent": (int, float, type(None)),
    "efficiency_peak_value": (int, float, type(None)),
    "efficiency_peak_unit": (str, type(None)),
    "bound": (str, type(None)),
}


def _assert_schema_valid(report):
    """Structurally validate ``report`` against the analysis.json TypedDicts."""
    assert set(report.keys()) <= {
        "report_info",
        "title",
        "executive_summary",
        "top_operations",
        "compute_optimizations",
        "appendix",
    }
    # No dropped/forbidden top-level fields (envelope rules §2).
    assert "schema_version" not in report
    assert "source_md_sha256" not in report
    assert "perf_plot" not in report
    assert "fusion_optimizations" not in report
    assert "system_notes" not in report
    assert "top_bottleneck_category" not in report

    info = report["report_info"]
    assert set(info.keys()) <= {"mode", "comparison_scope", "trace_quality", "warnings"}
    assert isinstance(info["mode"], str)
    assert isinstance(info["comparison_scope"], str)
    assert info["trace_quality"] in ("ok", "poor")
    if "warnings" in info:
        assert isinstance(info["warnings"], list) and info["warnings"]
        assert all(isinstance(w, str) for w in info["warnings"])

    assert isinstance(report["title"], str)
    assert "compute_optimizations" in report  # required, always present
    assert isinstance(report["compute_optimizations"], list)

    for task in report["compute_optimizations"]:
        assert set(task.keys()) == {
            "priority",
            "operation",
            "identification",
            "reasoning",
            "resolution",
            "prose_truncated",
            "impact",
            "members",
        }
        assert isinstance(task["priority"], int)
        assert task["operation"] is None or isinstance(task["operation"], str)
        assert isinstance(task["prose_truncated"], bool)
        imp = task["impact"]
        assert set(imp.keys()) == {"mid", "low", "high"}
        assert isinstance(imp["mid"], (int, float))
        assert isinstance(task["members"], list) and task["members"]
        for m in task["members"]:
            assert set(m.keys()) == set(_MEMBER_TYPES)
            for field, types in _MEMBER_TYPES.items():
                assert isinstance(m[field], types), f"{field}={m[field]!r}"
            assert all(isinstance(k, str) for k in m["kernel_name"])
            # roofline field explicitly dropped (§2.1).
            assert "roofline_attainment_pct" not in m


# --------------------------------------------------------------------------- #
# Independent md reparse for grouping-faithfulness.
# --------------------------------------------------------------------------- #

_HEADER_ALIASES = {
    "operation": "operation",
    "kernel name": "kernel_name",
    "time (ms)": "time_ms",
    "%e2e": "pct_e2e",
}


def _reparse_rows(md_text):
    """Independently pull (operation, kernel_names, time_ms, pct_e2e) per row."""
    lines = md_text.splitlines()
    rows = []
    for start, end in _iter_compute_candidate_blocks(md_text):
        table = _find_data_table(lines, start, end)
        if table is None:
            continue
        _, header, row_iter = table
        col = {
            _HEADER_ALIASES[h.strip().lower()]: i
            for i, h in enumerate(header)
            if h.strip().lower() in _HEADER_ALIASES
        }
        for _, cells in row_iter:

            def get(field):
                idx = col.get(field)
                return cells[idx] if idx is not None and idx < len(cells) else None

            op = (get("operation") or "").strip()
            names = []
            for piece in (get("kernel_name") or "").split("<br>"):
                piece = re.sub(r"^\s*Kernel\s+\d+:\s*", "", piece).strip()
                if piece and piece not in _NULL_CELLS:
                    names.append(piece)
            tm = get("time_ms")
            pe = get("pct_e2e")
            rows.append(
                {
                    "operation": op,
                    "kernel_name": tuple(names),
                    "time_ms": tm.strip() if tm else None,
                    "pct_e2e": pe.strip() if pe else None,
                }
            )
    return rows


# --------------------------------------------------------------------------- #
# Agentic golden envelope + grouping.
# --------------------------------------------------------------------------- #


def test_envelope_agentic_golden(tmp_path):
    report, _ = _render_text(_AGENTIC_MD, tmp_path)
    _assert_schema_valid(report)
    info = report["report_info"]
    assert info["mode"] == "agentic"
    assert info["comparison_scope"] == "standalone"
    assert info["trace_quality"] == "ok"
    assert "warnings" not in info  # absent, never []
    # Agentic variant carries all three optional sections.
    assert "executive_summary" in report
    assert "top_operations" in report
    assert "appendix" in report
    # top_operations is a flat list of row dicts.
    assert isinstance(report["top_operations"], list)
    assert all(isinstance(r, dict) and "rank" in r for r in report["top_operations"])
    assert report["title"] == "ExampleMoE - MI300X Standalone Analysis"


def test_exec_summary_metrics(tmp_path):
    report, _ = _render_text(_AGENTIC_MD, tmp_path)
    metrics = report["executive_summary"]["metrics"]
    assert metrics["total_time_ms"] == 2494.24
    assert metrics["compute_pct"] == 84.24
    assert metrics["idle_pct"] == 15.48
    assert metrics["exposed_communication_pct"] == 0.0
    # No '%' sign survives in numeric fields.
    for v in metrics.values():
        assert not isinstance(v, str)


def test_operation_grouping_collapses_tile_variants(tmp_path):
    report, _ = _render_text(_AGENTIC_MD, tmp_path)
    gemm = [
        t
        for t in report["compute_optimizations"]
        if t["operation"] == "aiter::gemm_a16w16"
    ]
    # One agentic op owning multiple tile variants collapses to ONE task.
    assert len(gemm) == 1
    assert len(gemm[0]["members"]) >= 2
    # Distinct tile kernels ride as distinct members.
    kernels = {tuple(m["kernel_name"]) for m in gemm[0]["members"]}
    assert len(kernels) == len(gemm[0]["members"])


def test_distinct_operations_stay_separate(tmp_path):
    report, _ = _render_text(_AGENTIC_MD, tmp_path)
    ops = [t["operation"] for t in report["compute_optimizations"]]
    # No operation key appears twice (each op is one task).
    assert len(ops) == len(set(ops))


def test_array_of_structs_shape(tmp_path):
    report, _ = _render_text(_AGENTIC_MD, tmp_path)
    task = report["compute_optimizations"][0]
    # Members are objects holding their own fields, not parallel top-level lists.
    assert isinstance(task["members"], list)
    for m in task["members"]:
        assert "args_shapes" in m and "args_datatypes" in m
    # No parallel per-member lists hoisted to task level.
    assert "kernel_names" not in task
    assert "impact_scores" not in task


def test_members_sorted_by_impact_desc(tmp_path):
    report, _ = _render_text(_AGENTIC_MD, tmp_path)
    for task in report["compute_optimizations"]:
        scores = [m["impact_score"] for m in task["members"]]
        assert scores == sorted(scores, reverse=True)


def test_impact_sum_and_task_order(tmp_path):
    report, _ = _render_text(_AGENTIC_MD, tmp_path)
    tasks = report["compute_optimizations"]
    for task in tasks:
        expected = sum(m["impact_score"] for m in task["members"])
        assert task["impact"]["mid"] == pytest.approx(expected)
    mids = [t["impact"]["mid"] for t in tasks]
    assert mids == sorted(mids, reverse=True)
    # priority mirrors position, display-only.
    assert [t["priority"] for t in tasks] == list(range(1, len(tasks) + 1))


def test_op_row_impacts_read_from_marker(tmp_path):
    report, _ = _render_text(_AGENTIC_MD, tmp_path)
    # P1 op_row marker: impacts=3.55,3.29,1.77 on the gemm_a16w16 rows.
    gemm = next(
        t
        for t in report["compute_optimizations"]
        if t["operation"] == "aiter::gemm_a16w16"
    )
    scores = sorted(m["impact_score"] for m in gemm["members"])
    for wanted in (1.77, 3.29, 3.55):
        assert any(s == pytest.approx(wanted) for s in scores)


def test_member_category_from_card_marker(tmp_path):
    # Card p_item category joins to the detailed block via the compute rank.
    report, _ = _render_text(_AGENTIC_MD, tmp_path)
    gemm = next(
        t
        for t in report["compute_optimizations"]
        if t["operation"] == "aiter::gemm_a16w16"
    )
    assert {m["category"] for m in gemm["members"]} == {"gemm"}
    # Every agentic task's members carry a non-null category here.
    assert all(
        m["category"] is not None
        for t in report["compute_optimizations"]
        for m in t["members"]
    )


def test_member_category_null_when_absent(tmp_path):
    # Fallback reports have p_item cards but no detailed-block anchors -> null.
    report, _ = _render_text(_FALLBACK_MD, tmp_path)
    assert all(
        m["category"] is None
        for t in report["compute_optimizations"]
        for m in t["members"]
    )


def test_member_library_from_heading_parens(tmp_path):
    # Compute headings carry a trailing ``(<Library>)`` token; the raw value is
    # stamped verbatim on every member of that block.
    report, _ = _render_text(_AGENTIC_MD, tmp_path)
    libs = {m["library"] for t in report["compute_optimizations"] for m in t["members"]}
    assert "AITER" in libs
    # Verbatim token: not lowercased, no suffix stripped.
    assert "aiter" not in libs


def test_member_library_null_when_heading_has_no_parens(tmp_path):
    # Fallback compute headings omit the ``(<Library>)`` suffix -> None.
    report, _ = _render_text(_FALLBACK_MD, tmp_path)
    assert all(
        m["library"] is None
        for t in report["compute_optimizations"]
        for m in t["members"]
    )


def test_hard_cells_efficiency_trio(tmp_path):
    report, _ = _render_text(_AGENTIC_MD, tmp_path)
    # "33.29% of 708 TFLOPS" -> (33.29, 708.0, "TFLOPS")
    found = False
    for task in report["compute_optimizations"]:
        for m in task["members"]:
            if m["efficiency_percent"] == pytest.approx(33.29):
                assert m["efficiency_peak_value"] == pytest.approx(708.0)
                assert m["efficiency_peak_unit"] == "TFLOPS"
                found = True
    assert found


def test_no_roofline_attainment_field(tmp_path):
    report, _ = _render_text(_AGENTIC_MD, tmp_path)
    blob = json.dumps(report)
    assert "roofline_attainment_pct" not in blob


def test_analysis_md_rank_stamped_on_members(tmp_path):
    # analysis_md_rank == P{rank} from the reasoning-candidate marker.
    report, _ = _render_text(_AGENTIC_MD, tmp_path)
    gemm = next(
        t
        for t in report["compute_optimizations"]
        if t["operation"] == "aiter::gemm_a16w16"
    )
    assert {m["analysis_md_rank"] for m in gemm["members"]} == {"P1"}
    moe = next(
        t
        for t in report["compute_optimizations"]
        if t["operation"] == "pseudo_op::moe_flydsl_stage1"
    )
    assert {m["analysis_md_rank"] for m in moe["members"]} == {"P2"}


def test_agentic_null_operation_cell_becomes_none(tmp_path):
    # An agentic report whose Operation cells are the "—" null sentinel must
    # emit operation=None, not the raw sentinel string (schema contract).
    report, _ = _render_text(_AGENTIC_NULL_OP_MD, tmp_path)
    ops = {t["operation"] for t in report["compute_optimizations"]}
    assert ops == {None}
    assert "—" not in ops


# --------------------------------------------------------------------------- #
# Deterministic-fallback variant.
# --------------------------------------------------------------------------- #


def test_fallback_envelope(tmp_path):
    report, _ = _render_text(_FALLBACK_MD, tmp_path)
    _assert_schema_valid(report)
    info = report["report_info"]
    assert info["mode"] == "deterministic-fallback"
    assert info["trace_quality"] == "poor"
    assert info["warnings"] and "fallback" in info["warnings"][0].lower()
    # Fallback omits these three entirely.
    assert "executive_summary" not in report
    assert "top_operations" not in report
    assert "appendix" not in report
    assert report["compute_optimizations"]


def test_fallback_operation_null_identity_in_kernel_name(tmp_path):
    report, _ = _render_text(_FALLBACK_MD, tmp_path)
    for task in report["compute_optimizations"]:
        # No op captured -> operation is null, identity lives in kernel_name.
        assert task["operation"] is None
        assert task["members"][0]["kernel_name"]


def test_fallback_tile_variants_four_separate_tasks(tmp_path):
    report, _ = _render_text(_FALLBACK_MD, tmp_path)
    cijk_tasks = [
        t
        for t in report["compute_optimizations"]
        if any("Cijk" in k for m in t["members"] for k in m["kernel_name"])
    ]
    # Four ranked Cijk_...MT... GEMM P-items stay four separate tasks: grouping
    # keys on the raw kernel_name, whose distinct MT infixes keep them apart.
    assert len(cijk_tasks) == 4
    names = {t["members"][0]["kernel_name"][0] for t in cijk_tasks}
    assert len(names) == 4


def test_fallback_op_row_still_read(tmp_path):
    report, _ = _render_text(_FALLBACK_MD, tmp_path)
    # Fallback emits op_row markers too; impact must be non-degenerate.
    assert any(
        m["impact_score"] > 0
        for t in report["compute_optimizations"]
        for m in t["members"]
    )


# --------------------------------------------------------------------------- #
# Fan-out row and pct_e2e fallback path.
# --------------------------------------------------------------------------- #


def test_fanout_row_stays_one_member(tmp_path):
    report, _ = _render_text(_NO_OP_ROW_MD, tmp_path)
    mla = next(
        (
            t
            for t in report["compute_optimizations"]
            if t["operation"] == "pseudo_mla_decode_fwd"
        ),
        None,
    )
    assert mla is not None
    fanout = [m for m in mla["members"] if len(m["kernel_name"]) >= 2]
    assert fanout, "expected a multi-kernel fan-out member"
    for m in fanout:
        # kernel_name is a >=2 list, but row metrics stay scalar.
        assert len(m["kernel_name"]) >= 2
        assert isinstance(m["impact_score"], (int, float))
        assert m["time_ms"] is None or isinstance(m["time_ms"], (int, float))
        assert m["count"] is None or isinstance(m["count"], int)
        assert m["efficiency_percent"] is None or isinstance(
            m["efficiency_percent"], (int, float)
        )


def test_pct_e2e_fallback_when_no_op_row(tmp_path):
    # _NO_OP_ROW_MD has NO op_row markers -> impact_score tracks pct_e2e.
    report, _ = _render_text(_NO_OP_ROW_MD, tmp_path)
    for task in report["compute_optimizations"]:
        for m in task["members"]:
            if m["pct_e2e"] is not None:
                assert m["impact_score"] == pytest.approx(
                    m["pct_e2e"] * HEURISTIC_FRACTION_MID
                )


def test_null_csv_entry_falls_back_per_row(tmp_path):
    # op_row CSV = "3.5,—": row 0 takes the marker, row 1's — falls back to pct_e2e.
    # A raw float("—") here would crash the render; the null must map cleanly.
    report, _ = _render_text(_NULL_CSV_ENTRY_MD, tmp_path)
    members = report["compute_optimizations"][0]["members"]
    by_kernel = {m["kernel_name"][0]: m for m in members}
    assert by_kernel["k_a"]["impact_score"] == pytest.approx(3.5)
    assert by_kernel["k_b"]["impact_score"] == pytest.approx(
        by_kernel["k_b"]["pct_e2e"] * HEURISTIC_FRACTION_MID
    )


# --------------------------------------------------------------------------- #
# Determinism + prose caps.
# --------------------------------------------------------------------------- #


def test_deterministic_byte_identical(tmp_path):
    dst = tmp_path / "analysis.md"
    dst.write_text(_AGENTIC_MD)
    first = render_analysis_json(dst).read_bytes()
    second = render_analysis_json(dst).read_bytes()
    assert first == second


def test_deterministic_cross_process(tmp_path):
    # Same md in a fresh interpreter yields the byte-identical json.
    _, in_proc = _render_text(_AGENTIC_MD, tmp_path)
    expected = in_proc.read_bytes()

    sub_dir = tmp_path / "sub"
    sub_dir.mkdir()
    sub_md = sub_dir / "analysis.md"
    sub_md.write_text(_AGENTIC_MD)
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from TraceLens.Agent.Analysis.post_processing import "
            "render_analysis_json; render_analysis_json(sys.argv[1])",
            str(sub_md),
        ],
        check=True,
    )
    assert (sub_dir / "analysis.json").read_bytes() == expected


def test_prose_caps_and_truncation(tmp_path):
    md = (
        "# Cap Test - MI300X Standalone Analysis\n"
        "<!-- report-begin kind=report_mode mode=agentic -->\n"
        "<!-- report-end -->\n\n"
        "### Compute Kernel Insights\n\n"
    )
    # Two findings sharing one operation, each with long prose, to force a merge.
    long_ident = "A" * 700
    long_reason = "B" * 1500
    for rank, tag in ((1, "one"), (2, "two")):
        md += (
            f'<a id="detailed-analysis-compute-p{rank}"></a>\n'
            f"<!-- reasoning-candidate tier=compute rank={rank} -->\n"
            f"#### P{rank}: {tag}\n"
            f"**Identification:** {long_ident} {tag}\n\n"
            f"**Data:**\n\n"
            "| Operation | Kernel Name | Time (ms) | %E2E |\n"
            "|---|---|---|---|\n"
            f"| aten::mm | k_{tag} | {10 - rank}.0 | {10 - rank}.0 |\n"
            f"<!-- impact-begin kind=op_row rank={rank} impacts={10 - rank}.0 -->\n"
            "<!-- impact-end -->\n\n"
            f"**Reasoning for Slowdown:** {long_reason} {tag}\n\n"
            f"**Resolution:** short {tag}\n\n"
        )
    report, _ = _render_text(md, tmp_path)
    task = next(
        t for t in report["compute_optimizations"] if t["operation"] == "aten::mm"
    )
    assert len(task["members"]) == 2  # both rows grouped
    assert len(task["identification"]) <= 2000
    assert len(task["reasoning"]) <= 2000
    assert len(task["resolution"]) <= 2000
    # Two 1500-char reasoning blocks cannot both fit under 2000 -> truncated.
    assert task["prose_truncated"] is True
    # Cut at a whole block: only the first (highest-impact) reasoning survives.
    assert task["reasoning"].count("B" * 1500) == 1
    # Merged task: each surviving block is prefixed with its source [P<rank>].
    assert task["identification"].startswith("[P1] ")
    assert "[P2] " in task["identification"]
    # Members carry their source md rank.
    assert {m["analysis_md_rank"] for m in task["members"]} == {"P1", "P2"}


def test_singleton_prose_passthrough(tmp_path):
    md = (
        "# Solo - MI300X Standalone Analysis\n"
        "<!-- report-begin kind=report_mode mode=agentic -->\n"
        "<!-- report-end -->\n\n"
        "### Compute Kernel Insights\n\n"
        '<a id="detailed-analysis-compute-p1"></a>\n'
        "<!-- reasoning-candidate tier=compute rank=1 -->\n"
        "#### P1: solo\n"
        "**Identification:** short identification\n\n"
        "**Data:**\n\n"
        "| Operation | Kernel Name | Time (ms) | %E2E |\n"
        "|---|---|---|---|\n"
        "| aten::mm | k0 | 5.0 | 5.0 |\n"
        "<!-- impact-begin kind=op_row rank=1 impacts=5.0 -->\n"
        "<!-- impact-end -->\n\n"
        "**Reasoning for Slowdown:** short reasoning\n\n"
        "**Resolution:** short resolution\n\n"
    )
    report, _ = _render_text(md, tmp_path)
    task = report["compute_optimizations"][0]
    assert len(task["members"]) == 1
    assert task["prose_truncated"] is False
    # Singleton merge is a no-op passthrough: verbatim prose, no [P<rank>] prefix.
    assert task["identification"] == "short identification"
    assert task["reasoning"] == "short reasoning"
    assert task["resolution"] == "short resolution"
    # The single member still carries its source md rank.
    assert task["members"][0]["analysis_md_rank"] == "P1"


# --------------------------------------------------------------------------- #
# Corpus-wide gates over the inline constants + per-fixture assertions.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name,md,mode,has_op_row,n_warn", _MATRIX)
def test_corpus_smoke(name, md, mode, has_op_row, n_warn, tmp_path):
    """Every inline analysis.md renders without exception, schema-valid."""
    report, _ = _render_text(md, tmp_path)
    _assert_schema_valid(report)


@pytest.mark.parametrize("name,md,mode,has_op_row,n_warn", _MATRIX)
def test_matrix_envelope(name, md, mode, has_op_row, n_warn, tmp_path):
    report, _ = _render_text(md, tmp_path)
    _assert_schema_valid(report)
    info = report["report_info"]
    assert info["mode"] == mode
    expected_tq = "poor" if (mode == "deterministic-fallback" or n_warn) else "ok"
    assert info["trace_quality"] == expected_tq
    if n_warn:
        assert len(info["warnings"]) == n_warn
    else:
        assert "warnings" not in info
    # Optional sections absent on the fallback path.
    if mode == "deterministic-fallback":
        assert "executive_summary" not in report
        assert "top_operations" not in report
        assert "appendix" not in report


@pytest.mark.parametrize("name,md,mode,has_op_row,n_warn", _MATRIX)
def test_matrix_grouping_faithfulness(name, md, mode, has_op_row, n_warn, tmp_path):
    """Every emitted member maps to a real md data-table row (lossless regroup)."""
    report, _ = _render_text(md, tmp_path)
    md_rows = _reparse_rows(md)
    md_index = {}
    for r in md_rows:
        md_index.setdefault((r["kernel_name"], r["time_ms"]), r)

    emitted = 0
    for task in report["compute_optimizations"]:
        for m in task["members"]:
            emitted += 1
            tm = None if m["time_ms"] is None else f"{m['time_ms']}"
            key = (tuple(m["kernel_name"]), tm)
            # Match a real row by kernel_name; time may be reformatted, so fall
            # back to kernel-name-only lookup.
            matched = key in md_index or any(
                r["kernel_name"] == tuple(m["kernel_name"]) for r in md_rows
            )
            assert matched, f"emitted member has no source row: {m['kernel_name']}"
    # No invented rows: emitted count <= md row count (cap only ever drops).
    assert emitted <= len(md_rows)


@pytest.mark.parametrize("name,md,mode,has_op_row,n_warn", _MATRIX)
def test_matrix_impact_path(name, md, mode, has_op_row, n_warn, tmp_path):
    report, _ = _render_text(md, tmp_path)
    if not has_op_row:
        # pct_e2e fallback: impact_score tracks pct_e2e where available.
        for task in report["compute_optimizations"]:
            for m in task["members"]:
                if m["pct_e2e"] is not None:
                    assert m["impact_score"] == pytest.approx(
                        m["pct_e2e"] * HEURISTIC_FRACTION_MID
                    )


@pytest.mark.parametrize("name,md,mode,has_op_row,n_warn", _MATRIX)
def test_matrix_deterministic(name, md, mode, has_op_row, n_warn, tmp_path):
    dst = tmp_path / "analysis.md"
    dst.write_text(md)
    a = render_analysis_json(dst).read_bytes()
    b = render_analysis_json(dst).read_bytes()
    assert a == b


@pytest.mark.parametrize("name,md,mode,has_op_row,n_warn", _MATRIX)
def test_matrix_roundtrip_typeddicts(name, md, mode, has_op_row, n_warn, tmp_path):
    report, _ = _render_text(md, tmp_path)
    # Round-trip through the TypedDicts (construction is a structural check).
    info: ReportInfo = report["report_info"]
    assert info["mode"]
    for task in report["compute_optimizations"]:
        t: ComputeTask = task
        imp: Impact = t["impact"]
        assert "mid" in imp
        for m in t["members"]:
            mem: ComputeMember = m
            assert "kernel_name" in mem
    rep: AnalysisReport = report
    assert "compute_optimizations" in rep


# --------------------------------------------------------------------------- #
# Robustness invariants (minimal inline md).
# --------------------------------------------------------------------------- #


def test_missing_header_column_yields_null_not_crash(tmp_path):
    md = (
        "# Sparse - MI300X Standalone Analysis\n"
        "<!-- report-begin kind=report_mode mode=agentic -->\n"
        "<!-- report-end -->\n\n"
        "### Compute Kernel Insights\n\n"
        '<a id="detailed-analysis-compute-p1"></a>\n'
        "<!-- reasoning-candidate tier=compute rank=1 -->\n"
        "#### P1: sparse\n"
        "**Identification:** id\n\n"
        "**Data:**\n\n"
        "| Operation | Kernel Name | Time (ms) | %E2E |\n"
        "|---|---|---|---|\n"
        "| aten::mm | k0 | 5.0 | 5.0 |\n"
        "<!-- impact-begin kind=op_row rank=1 impacts=5.0 -->\n"
        "<!-- impact-end -->\n\n"
        "**Resolution:** fix\n\n"
    )
    report, _ = _render_text(md, tmp_path)
    m = report["compute_optimizations"][0]["members"][0]
    # Columns absent from the header parse to null, no crash.
    assert m["flops_per_byte"] is None
    assert m["bound"] is None
    assert m["efficiency_percent"] is None
    assert m["time_ms"] == pytest.approx(5.0)


def test_em_dash_cells_become_null(tmp_path):
    md = (
        "# Dash - MI300X Standalone Analysis\n"
        "<!-- report-begin kind=report_mode mode=agentic -->\n"
        "<!-- report-end -->\n\n"
        "### Compute Kernel Insights\n\n"
        '<a id="detailed-analysis-compute-p1"></a>\n'
        "<!-- reasoning-candidate tier=compute rank=1 -->\n"
        "#### P1: dash\n"
        "**Identification:** id\n\n"
        "**Data:**\n\n"
        "| Operation | Kernel Name | Time (ms) | %E2E | FLOPS/Byte | Efficiency | Bound |\n"
        "|---|---|---|---|---|---|---|\n"
        "| aten::mm | k0 | 5.0 | 5.0 | — | — | — |\n"
        "<!-- impact-begin kind=op_row rank=1 impacts=5.0 -->\n"
        "<!-- impact-end -->\n\n"
        "**Resolution:** fix\n\n"
    )
    report, _ = _render_text(md, tmp_path)
    m = report["compute_optimizations"][0]["members"][0]
    assert m["flops_per_byte"] is None
    assert m["bound"] is None
    assert m["efficiency_percent"] is None
    assert m["efficiency_peak_value"] is None
    assert m["efficiency_peak_unit"] is None


def test_no_compute_findings_empty_list(tmp_path):
    md = (
        "# Empty - MI300X Standalone Analysis\n"
        "<!-- report-begin kind=report_mode mode=agentic -->\n"
        "<!-- report-end -->\n\n"
        "## Executive Summary\n\n"
        "Nothing to compute here.\n\n"
    )
    report, _ = _render_text(md, tmp_path)
    assert report["compute_optimizations"] == []
    _assert_schema_valid(report)
