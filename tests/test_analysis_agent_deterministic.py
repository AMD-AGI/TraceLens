###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tier-1 unit tests for the deterministic graph-under-recorded fallback.

Covers the no-LLM fallback path in ``utils/deterministic_fallback``:
- ``check_graph_replay_coverage`` (detector U1-U4)
- ``render_fallback_report`` (writer U5-U9)

Every fixture is a tiny SYNTHETIC ``unified_perf_summary.csv`` written under
``tmp_path`` with hand-computed expected values, so a failing assertion points
at the exact arithmetic. No GPU, no LLM, no network, no ``Local_Traces``.

U7/U9 assert the downstream ``analysis.md`` parse contract using a LOCAL, vendored
copy of the consumer's regexes (kept in sync manually, no external import) so the
fallback md stays compatible with the default-agent contract.
"""

import csv
import re

import pytest

from TraceLens.Agent.Analysis.utils.deterministic_fallback import (
    GRAPH_REPLAY_FRACTION_MAX,
    MAX_KERNEL_NAME_LEN,
    MAX_PITEM_COUNT,
    MIN_PITEM_PERCENT_E2E,
    _normalize_kernel_name,
    _PERCENT_COLUMN,
    _WEIGHT_COLUMN,
    check_graph_replay_coverage,
    render_fallback_report,
)

# ---------------------------------------------------------------------------
# Vendored downstream-parser regexes (U7/U9). Copied verbatim from the
# ``analysis.md`` consumer so the tests assert the exact parse contract without
# an external import.
# ---------------------------------------------------------------------------
_PITEM_MARKER_RE = re.compile(
    r"<!--\s*impact-begin\s+kind=p_item\s+([^>]*?)-->",
    re.IGNORECASE,
)
_REASONING_MARKER_RE = re.compile(
    r"<!--\s*reasoning-candidate\s+tier=(\w+)\s+rank=(\d+)\s*-->",
    re.IGNORECASE,
)
_HEADING_RE = re.compile(
    r"^####\s+(?:[\U0001F300-\U0001FAFF☀-➿]+\s+)?P(\d+):\s*(.+?)\s*$",
    re.MULTILINE,
)

# 9 canonical column tokens the downstream parser requires in order; "Kernel
# Name" is an extra column allowed to interleave.
_DATA_TABLE_HEADER_TOKENS = (
    "operation",
    "args",
    "kernel path",
    "time (ms)",
    "%e2e",
    "count",
    "flops/byte",
    "efficiency",
    "bound",
)
_DATA_TABLE_CANONICAL_KEY_SET = frozenset(
    tok.strip().lower() for tok in _DATA_TABLE_HEADER_TOKENS
)

_EM_DASH = "—"


# ---------------------------------------------------------------------------
# Vendored subset of the downstream ``analysis.md`` parser — kept in sync with
# the consumer. This runs the REAL downstream parse path (reasoning-marker block
# split -> **Data:** table extraction -> canonical 9-column validation -> one
# candidate per data row) so U7 gates the exact class of contract bug that a
# heading-count-only check would miss.
# ---------------------------------------------------------------------------
def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_marker_attrs(blob: str) -> dict:
    """Parse ``key=value`` attributes from an HTML-comment marker blob."""
    return dict(re.findall(r"(\w+)=([^\s>]+)", blob))


def _extract_pitem_categories(text: str) -> list:
    """One dict per ``p_item`` marker (in order); markers must carry ``category=``."""
    items = []
    for match in _PITEM_MARKER_RE.finditer(text):
        attrs = _parse_marker_attrs(match.group(1))
        if "category" not in attrs:
            continue
        items.append(
            {
                "category": attrs.get("category", ""),
                "impact_score_low": _safe_float(attrs.get("low")),
                "impact_score": _safe_float(attrs.get("mid")),
                "impact_score_high": _safe_float(attrs.get("high")),
            }
        )
    return items


def _split_data_blocks(text: str) -> list:
    """Split into compute-tier reasoning blocks; drop any with no heading in slice."""
    blocks = []
    matches = list(_REASONING_MARKER_RE.finditer(text))
    for idx, match in enumerate(matches):
        tier = match.group(1).lower()
        if tier != "compute":
            continue
        body_start = match.end()
        body_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = text[body_start:body_end]
        head_match = _HEADING_RE.search(body)
        if not head_match:
            continue
        rank = int(head_match.group(1))
        title = head_match.group(2).strip()
        blocks.append((rank, title, body))
    return blocks


def _extract_data_table(body: str) -> list:
    """Pull the markdown table after ``**Data:**``; ``[]`` when the marker is absent."""
    marker = body.find("**Data:**")
    if marker < 0:
        return []
    tail = body[marker + len("**Data:**") :]
    rows = []
    in_table = False
    for line in tail.splitlines():
        stripped = line.strip()
        if not stripped:
            if in_table:
                break
            continue
        if not stripped.startswith("|"):
            if in_table:
                break
            continue
        in_table = True
        if set(stripped.replace("|", "").strip()) <= set("-: "):
            continue
        cells = [cell.strip() for cell in stripped.split("|")[1:-1]]
        rows.append(cells)
    return rows


def _parse_candidates(md_text: str) -> list:
    """Run the real downstream parse and return one candidate dict per data row.

    Faithful subset of ``parse_analysis_md``: block split -> ``**Data:**`` table
    extraction -> canonical 9-column (in-order, extras allowed) validation ->
    one candidate per surviving data row carrying the P-item category.
    """
    pitems = _extract_pitem_categories(md_text)
    blocks = _split_data_blocks(md_text)
    if not blocks:
        return []

    headers_canonical = [tok.strip().lower() for tok in _DATA_TABLE_HEADER_TOKENS]
    canonical_width = len(headers_canonical)

    candidates = []
    for rank, title, body in blocks:
        rows = _extract_data_table(body)
        if not rows:
            continue
        header_row = [cell.strip().lower() for cell in rows[0]]
        if len(header_row) < canonical_width:
            continue
        normalized_header = []
        for cell in header_row:
            match = next(
                (
                    canon
                    for canon in headers_canonical
                    if canon == cell or canon in cell
                ),
                cell,
            )
            normalized_header.append(match)
        canonical_in_header = [c for c in normalized_header if c in headers_canonical]
        if canonical_in_header != headers_canonical:
            continue
        header_row = normalized_header
        pitem_meta = pitems[rank - 1] if rank - 1 < len(pitems) else {}
        category = pitem_meta.get("category", "")
        impact_score = pitem_meta.get("impact_score", 0.0)
        for cells in rows[1:]:
            if len(cells) != len(header_row):
                continue
            record = dict(zip(header_row, cells))
            name = record.get("operation", "").strip()
            kernel_name = record.get("kernel name", "").strip()
            # Relaxed-reader contract: a graph-collapsed row emits "—"
            # for Operation because the python/aten op was never captured; the
            # device kernel symbol is the identity. Substitute it rather than
            # dropping the row. Drop only when there is no symbol either.
            if not name or name in {"-", "—"}:
                if not kernel_name or kernel_name in {"-", "—"}:
                    continue
                name = kernel_name
            candidates.append(
                {
                    "rank": rank,
                    "title": title,
                    "operation": name,
                    "kernel_name": kernel_name,
                    "category": category,
                    "impact_score": impact_score,
                    "args": record.get("args", "").strip(),
                    "time_ms": record.get("time (ms)", "").strip(),
                    "percent_e2e": record.get("%e2e", "").strip(),
                }
            )
    return candidates


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------
def _make_perf_csv(tmp_path, rows):
    """Write a synthetic ``unified_perf_summary.csv`` and return its dir.

    ``rows`` is a list of ``(name, weight, percent)`` tuples. Only the three
    columns the code reads are written; ``percent`` defaults to 0.0 if omitted.
    """
    perf_dir = tmp_path / "perf_report_csvs"
    perf_dir.mkdir(exist_ok=True)
    with open(
        perf_dir / "unified_perf_summary.csv", "w", newline="", encoding="utf-8"
    ) as fh:
        writer = csv.writer(fh)
        writer.writerow(["name", _WEIGHT_COLUMN, _PERCENT_COLUMN])
        for row in rows:
            name, weight = row[0], row[1]
            percent = row[2] if len(row) > 2 else 0.0
            writer.writerow([name, weight, percent])
    return perf_dir


def _make_category_csv(perf_dir, categories):
    """Write an optional ``ops_summary_by_category.csv`` for the other-frac path.

    ``categories`` is a list of ``(op_category, percent)`` tuples.
    """
    with open(
        perf_dir / "ops_summary_by_category.csv", "w", newline="", encoding="utf-8"
    ) as fh:
        writer = csv.writer(fh)
        writer.writerow(["op category", _PERCENT_COLUMN])
        for cat, pct in categories:
            writer.writerow([cat, pct])
    return perf_dir


def _data_row_cells(md, kernel):
    """Return the split table-cell list for the data row of ``kernel`` (else None)."""
    for line in md.splitlines():
        stripped = line.strip()
        if stripped.startswith(f"| {kernel} |"):
            # Drop the leading/trailing empty fields from the bounding pipes.
            return [c.strip() for c in stripped.split("|")[1:-1]]
    return None


def _strip_banner(md):
    """Remove the visible degraded call-out lines."""
    return "\n".join(line for line in md.splitlines() if not line.startswith(">"))


# ---------------------------------------------------------------------------
# U1 — detector fires on a graph-collapsed fixture
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "wrapper",
    [
        ("hipGraphLaunch->k1 (Synthetic Op)", "hipGraphLaunch->k2 (Synthetic Op)"),
        ("Torch-Compiled Region: 0/0->k (Synthetic Op)", None),
    ],
)
def test_u1_detector_fires_on_graph_collapsed(tmp_path, wrapper):
    graph_rows = [(name, 47.5, 47.5) for name in wrapper if name]
    # Pad the graph weight to 95 total and add a benign plain kernel for the 5%.
    rows = graph_rows + [("plain_kernel", 5.0, 5.0)]
    # Normalize the graph weight to sum to 95 regardless of how many graph rows.
    graph_total = sum(w for _, w, _ in graph_rows)
    rows = [(name, w / graph_total * 95.0, p) for name, w, p in graph_rows] + [
        ("plain_kernel", 5.0, 5.0)
    ]

    perf_dir = _make_perf_csv(tmp_path, rows)
    verdict = check_graph_replay_coverage(perf_dir)

    assert verdict.graph_under_recorded is True
    assert verdict.graph_replay_fraction == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# U2 — detector stays quiet on a healthy fixture
# ---------------------------------------------------------------------------
def test_u2_detector_quiet_on_healthy(tmp_path):
    rows = [
        ("aten::mm", 40.0, 40.0),
        ("aten::softmax", 30.0, 30.0),
        ("hipMemcpyAsync->copy_dtoh (Synthetic Op)", 15.0, 15.0),
        ("hipModuleLaunchKernel->some_kernel (Synthetic Op)", 15.0, 15.0),
    ]
    perf_dir = _make_perf_csv(tmp_path, rows)
    verdict = check_graph_replay_coverage(perf_dir)

    assert verdict.graph_under_recorded is False
    assert verdict.graph_replay_fraction == 0.0


# ---------------------------------------------------------------------------
# U3 — false-positive guard: benign launch/memcpy wrappers never count
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "wrapper_name",
    [
        "hipLaunchKernel->orphan (Synthetic Op)",
        "hipModuleLaunchKernel->orphan (Synthetic Op)",
        "hipMemcpyAsync->orphan (Synthetic Op)",
        "MemcpyDtoH->orphan (Synthetic Op)",
        "__amd_rocclr_copyBuffer->orphan (Synthetic Op)",
    ],
)
def test_u3_false_positive_guard(tmp_path, wrapper_name):
    rows = [
        (wrapper_name, 40.0, 40.0),
        ("aten::mm", 60.0, 60.0),
    ]
    perf_dir = _make_perf_csv(tmp_path, rows)
    verdict = check_graph_replay_coverage(perf_dir)

    assert verdict.graph_under_recorded is False
    assert verdict.graph_replay_fraction == 0.0


# ---------------------------------------------------------------------------
# U4 — threshold boundary (strict `>` around 0.10)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "graph_weight, expected",
    [
        (9.9, False),  # 0.099 < 0.10
        (10.0, False),  # 0.100 == 0.10 -> strict > means still False
        (10.1, True),  # 0.101 > 0.10
    ],
)
def test_u4_threshold_boundary(tmp_path, graph_weight, expected):
    assert GRAPH_REPLAY_FRACTION_MAX == 0.10
    rows = [
        ("hipGraphLaunch->k (Synthetic Op)", graph_weight, graph_weight),
        ("aten::mm", 100.0 - graph_weight, 100.0 - graph_weight),
    ]
    perf_dir = _make_perf_csv(tmp_path, rows)
    verdict = check_graph_replay_coverage(perf_dir)

    assert verdict.graph_under_recorded is expected
    assert verdict.graph_replay_fraction == pytest.approx(graph_weight / 100.0)


# ---------------------------------------------------------------------------
# U5 — wrapper-strip + exact-name grouping
# ---------------------------------------------------------------------------
def test_u5_wrapper_strip_and_exact_name_grouping(tmp_path):
    rows = [
        ("hipGraphLaunch->foo (Synthetic Op)", 10.0, 10.0),
        ("hipGraphLaunch->foo (Synthetic Op)", 30.0, 30.0),
        ("hipGraphLaunch->bar (Synthetic Op)", 20.0, 20.0),
    ]
    perf_dir = _make_perf_csv(tmp_path, rows)
    md = render_fallback_report(perf_dir, 0.95)

    # Bare names recovered from the WRAPPER->K (Synthetic Op) form.
    headings = dict((int(rank), name) for rank, name in _HEADING_RE.findall(md))
    assert set(headings.values()) == {"foo", "bar"}

    # foo group = 10 + 30 = 40 of total 60 -> 66.667%; bar = 20 -> 33.333%.
    # Impact low = round(pct * 0.15, 2). Both foo cards carry the SAME group value.
    foo_low = round(40.0 / 60.0 * 100.0 * 0.15, 2)
    bar_low = round(20.0 / 60.0 * 100.0 * 0.15, 2)
    assert foo_low != bar_low  # distinct groups -> distinct impact

    lows = [
        float(re.search(r"low=([\d.]+)", blob).group(1))
        for blob in _PITEM_MARKER_RE.findall(md)
    ]
    assert lows.count(foo_low) == 2  # two foo rows, one card each, same group value
    assert lows.count(bar_low) == 1


def test_u5_tile_variants_stay_separate_groups(tmp_path):
    rows = [
        ("hipGraphLaunch->f4gemm_192x128 (Synthetic Op)", 30.0, 30.0),
        ("hipGraphLaunch->f4gemm_32x128 (Synthetic Op)", 10.0, 10.0),
    ]
    perf_dir = _make_perf_csv(tmp_path, rows)
    md = render_fallback_report(perf_dir, 0.95)

    names = {name for _, name in _HEADING_RE.findall(md)}
    assert names == {"f4gemm_192x128", "f4gemm_32x128"}

    # Different bare names => different groups => different %-of-total impact.
    lows = sorted(
        float(re.search(r"low=([\d.]+)", blob).group(1))
        for blob in _PITEM_MARKER_RE.findall(md)
    )
    assert lows == sorted(
        [round(10.0 / 40.0 * 100.0 * 0.15, 2), round(30.0 / 40.0 * 100.0 * 0.15, 2)]
    )


# ---------------------------------------------------------------------------
# U5c — plumbing behind a graph-replay wrapper is dropped from the writer output.
# Guards the leak where the wrapper prefix is graph-replay but the bare kernel
# behind it is runtime plumbing (e.g. `hipGraphLaunch->__amd_rocclr_copyBuffer`).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "plumbing_kernel",
    ["__amd_rocclr_copyBuffer", "__amd_rocclr_fillBufferAligned", "MemcpyDtoD"],
)
def test_u5c_plumbing_behind_graph_replay_wrapper_dropped(tmp_path, plumbing_kernel):
    rows = [
        ("hipGraphLaunch->real_kernel (Synthetic Op)", 90.0, 90.0),
        (f"hipGraphLaunch->{plumbing_kernel} (Synthetic Op)", 10.0, 10.0),
        (f"Torch-Compiled Region: 0/0->{plumbing_kernel} (Synthetic Op)", 5.0, 5.0),
    ]
    perf_dir = _make_perf_csv(tmp_path, rows)
    md = render_fallback_report(perf_dir, 0.95)

    names = {name for _, name in _HEADING_RE.findall(md)}
    assert names == {"real_kernel"}
    assert plumbing_kernel not in md


# ---------------------------------------------------------------------------
# U6 — impact arithmetic (match the exact emitted string)
# ---------------------------------------------------------------------------
def test_u6_impact_arithmetic(tmp_path):
    # Two equal-weight rows with distinct names -> each is its own group at 50%.
    rows = [
        ("hipGraphLaunch->foo (Synthetic Op)", 50.0, 50.0),
        ("hipGraphLaunch->bar (Synthetic Op)", 50.0, 50.0),
    ]
    perf_dir = _make_perf_csv(tmp_path, rows)
    md = render_fallback_report(perf_dir, 0.95)

    # 50 * 0.15/0.30/0.45 = 7.5 / 15.0 / 22.5 (round(...,2) prints 15.0, not 15).
    assert (
        "impact-begin kind=p_item category=unknown low=7.5 mid=15.0 high=22.5 -->" in md
    )


# ---------------------------------------------------------------------------
# U7 — parse contract (LOAD-BEARING) against vendored regexes
# ---------------------------------------------------------------------------
def test_u7_parse_contract(tmp_path):
    # Two DISTINCT graph-replay kernels with distinct bare names and weights.
    # Higher weight sorts first -> P1; the full downstream parse must yield one
    # candidate per data row, i.e. TWO. This whole test drives the REAL parse
    # path (_parse_candidates), not just a heading count, so it catches the
    # class of contract bug where the md renders but parses to ZERO candidates.
    rows = [
        ("hipGraphLaunch->foo (Synthetic Op)", 60.0, 60.0),
        ("hipGraphLaunch->bar (Synthetic Op)", 40.0, 40.0),
    ]
    perf_dir = _make_perf_csv(tmp_path, rows)
    md = render_fallback_report(perf_dir, 0.95)

    candidates = _parse_candidates(md)

    # (1) LOAD-BEARING: the full parse yields exactly two candidates. This is
    #     the assertion that would have caught both prior contract bugs, each of
    #     which rendered headings but collapsed to zero downstream candidates.
    assert len(candidates) == 2

    # (2) rank -> kernel mapping (P1 = highest weight) and category propagation.
    by_rank = {c["rank"]: c for c in candidates}
    assert by_rank[1]["kernel_name"] == "foo"
    assert by_rank[2]["kernel_name"] == "bar"
    for cand in candidates:
        assert cand["category"] == "unknown"

    # (3) name/time/%E2E populated in the row; Args cell is the em dash.
    assert by_rank[1]["operation"] == "foo"
    assert by_rank[1]["args"] == _EM_DASH  # Args intentionally unrecoverable
    assert by_rank[1]["time_ms"] == "0.060"  # 60 us -> 0.060 ms
    assert by_rank[1]["percent_e2e"] == "60.00"

    # (4) NEGATIVE — ### regression guard: downgrading the headings to three
    #     hashes makes every block vanish under _HEADING_RE -> zero candidates.
    downgraded = md.replace("#### P", "### P")
    assert len(_parse_candidates(downgraded)) == 0

    # (5) NEGATIVE — **Data:** regression guard (catches C1): with the Data
    #     marker deleted, _extract_data_table returns [] -> zero candidates.
    no_data = "\n".join(line for line in md.splitlines() if line.strip() != "**Data:**")
    assert len(_parse_candidates(no_data)) == 0

    # (6) NEGATIVE — marker/heading ORDER guard (catches C2): moving each
    #     heading BEFORE its reasoning-candidate marker breaks the block split
    #     (the heading no longer falls in the marker's body slice), so the
    #     candidate count is wrong (not two). The correct order is load-bearing.
    reordered = re.sub(
        r"(<!-- reasoning-candidate tier=compute rank=\d+ -->)\n(#### P\d+: [^\n]+)",
        r"\2\n\1",
        md,
    )
    assert len(_parse_candidates(reordered)) != 2
    # And in the REAL md every marker precedes its matching heading.
    for rank in (1, 2):
        marker_idx = md.index(f"reasoning-candidate tier=compute rank={rank}")
        heading_idx = md.index(f"#### P{rank}:")
        assert marker_idx < heading_idx


# ---------------------------------------------------------------------------
# U8 — separate-row vs grouped invariant; plumbing dropped
# ---------------------------------------------------------------------------
def test_u8_separate_rows_grouped_impact(tmp_path):
    rows = [
        ("hipGraphLaunch->foo (Synthetic Op)", 30.0, 30.0),
        ("hipGraphLaunch->foo (Synthetic Op)", 10.0, 10.0),
        ("hipGraphLaunch->bar (Synthetic Op)", 20.0, 20.0),
        # Benign plumbing must be dropped from the candidate table entirely.
        ("hipMemcpyAsync->copy (Synthetic Op)", 5.0, 5.0),
    ]
    perf_dir = _make_perf_csv(tmp_path, rows)
    md = render_fallback_report(perf_dir, 0.95)

    headings = _HEADING_RE.findall(md)
    # One card per SURVIVING CSV kernel row (3), plumbing row dropped.
    assert len(headings) == 3
    assert _data_row_cells(md, "copy") is None  # plumbing not rendered

    # Two foo rows share a bare name -> two cards, SAME name-grouped impact.
    total = 60.0  # 30 + 10 + 20 (plumbing excluded from the group total)
    foo_low = round(40.0 / total * 100.0 * 0.15, 2)
    bar_low = round(20.0 / total * 100.0 * 0.15, 2)
    lows = [
        float(re.search(r"low=([\d.]+)", blob).group(1))
        for blob in _PITEM_MARKER_RE.findall(md)
    ]
    assert lows.count(foo_low) == 2
    assert lows.count(bar_low) == 1


# ---------------------------------------------------------------------------
# U9 — degraded banner present and inert to the parser
# ---------------------------------------------------------------------------
def test_u9_degraded_banner(tmp_path):
    rows = [
        ("hipGraphLaunch->foo (Synthetic Op)", 95.0, 95.0),
        ("aten::mm", 5.0, 5.0),
    ]
    perf_dir = _make_perf_csv(tmp_path, rows)
    frac = 0.95
    md = render_fallback_report(perf_dir, frac)

    # Visible call-out with the fraction rendered as a percent.
    assert "> **⚠ Degraded (deterministic fallback) report.**" in md
    assert "graph-replay fraction 95%" in md

    # The banner is INERT: stripping it must not change any parsed candidate count.
    stripped = _strip_banner(md)
    assert len(_HEADING_RE.findall(md)) == len(_HEADING_RE.findall(stripped))
    assert len(_PITEM_MARKER_RE.findall(md)) == len(_PITEM_MARKER_RE.findall(stripped))
    assert len(_REASONING_MARKER_RE.findall(md)) == len(
        _REASONING_MARKER_RE.findall(stripped)
    )


# ---------------------------------------------------------------------------
# Normalizer matrix — one case per name form (truncate-only, no demangling).
# ---------------------------------------------------------------------------
def test_normalizer_bare_triton_identity():
    assert _normalize_kernel_name("_fwd_kernel") == "_fwd_kernel"


def test_normalizer_clean_aiter_identity():
    # A real ~50-char aiter config passes through untouched (< the 75 cap).
    name = "aiter::f4gemm_bf16_per1x32Fp4_BpreShuffle_192x128"
    assert len(name) < MAX_KERNEL_NAME_LEN
    assert _normalize_kernel_name(name) == name


def test_normalizer_void_template_truncated_void_kept():
    # `void ` is KEPT (matches the GOOD report); only length is trimmed.
    raw = (
        "void at::native::elementwise_kernel<128, 2, "
        "at::native::gpu_kernel_impl<at::native::BinaryFunctor>>(int, char*)"
    )
    assert len(raw) > MAX_KERNEL_NAME_LEN
    out = _normalize_kernel_name(raw)
    assert out.startswith("void ")
    assert out == raw[:MAX_KERNEL_NAME_LEN] + "..."
    assert len(out) == MAX_KERNEL_NAME_LEN + 3


def test_normalizer_tensile_truncated():
    raw = "Cijk_Ailk_Bljk_HHS_BH_Bias_AS_MT128x160x16_MI16x16x16x1_SN_WG32_8_1_ABCD"
    raw = raw + "_extra_tokens_pushing_well_past_the_cap"
    assert len(raw) > MAX_KERNEL_NAME_LEN
    out = _normalize_kernel_name(raw)
    assert out == raw[:MAX_KERNEL_NAME_LEN] + "..."


def test_normalizer_mangled_zn_truncated_not_demangled():
    # Locks in the no-demangle decision: the symbol stays mangled, just shorter.
    raw = "_ZN5aiter24add_rmsnorm_quant_kernelIN3std10bfloat16_tES2_Li256ELb1EEEvPT_"
    raw = raw + "PKS1_iiffb"
    assert len(raw) > MAX_KERNEL_NAME_LEN
    out = _normalize_kernel_name(raw)
    assert out.startswith("_ZN")
    assert out.endswith("...")
    assert out == raw[:MAX_KERNEL_NAME_LEN] + "..."


def test_normalizer_never_returns_empty():
    # The non-empty invariant: a blank input falls back to raw, never "".
    assert _normalize_kernel_name("") == ""
    assert _normalize_kernel_name("   ") == "   "


def test_normalizer_length_boundary():
    at_cap = "a" * MAX_KERNEL_NAME_LEN
    over_cap = "a" * (MAX_KERNEL_NAME_LEN + 1)
    assert _normalize_kernel_name(at_cap) == at_cap  # len 75 -> identity
    assert _normalize_kernel_name(over_cap) == at_cap + "..."  # len 76 -> truncates


# ---------------------------------------------------------------------------
# Group-independence: distinct raw kernels whose DISPLAY collapses (differ only
# past char 75) must stay separate groups with their own impact (group-by-raw,
# normalize-for-display).
# ---------------------------------------------------------------------------
def test_group_independence_raw_key_not_display(tmp_path):
    common = "C" * 80  # first 75 chars identical -> identical truncated display
    name_a, name_b = common + "_aaa", common + "_bbb"
    rows = [
        (f"hipGraphLaunch->{name_a} (Synthetic Op)", 60.0, 60.0),
        (f"hipGraphLaunch->{name_b} (Synthetic Op)", 40.0, 40.0),
    ]
    perf_dir = _make_perf_csv(tmp_path, rows)
    md = render_fallback_report(perf_dir, 0.95)

    # Two P-items render even though their titles are byte-identical.
    headings = _HEADING_RE.findall(md)
    assert len(headings) == 2
    titles = {title for _, title in headings}
    assert len(titles) == 1  # display collapsed, but two cards still emitted

    # Impacts are per-raw-group (9.0 and 6.0), NOT a merged 15.0 for both.
    lows = sorted(
        float(re.search(r"low=([\d.]+)", blob).group(1))
        for blob in _PITEM_MARKER_RE.findall(md)
    )
    assert lows == [round(40.0 * 0.15, 2), round(60.0 * 0.15, 2)]


# ---------------------------------------------------------------------------
# N3 columns: Operation cell == em dash (op never captured on a
# graph-collapsed trace), Kernel Name cell == raw device symbol, and the P
# heading carries the normalized short form.
# ---------------------------------------------------------------------------
def test_n3_operation_em_dash_kernel_name_raw(tmp_path):
    raw = "void aiter::" + "x" * 100  # > cap -> display truncates, raw preserved
    rows = [(f"hipGraphLaunch->{raw} (Synthetic Op)", 100.0, 100.0)]
    perf_dir = _make_perf_csv(tmp_path, rows)
    md = render_fallback_report(perf_dir, 0.95)

    display = _normalize_kernel_name(raw)
    assert display != raw
    # The em-dash Operation cell means the data row no longer starts with the
    # display name; find it by the raw Kernel Name instead.
    cells = _data_row_cells(md, _EM_DASH)
    assert cells is not None
    assert cells[0] == _EM_DASH  # Operation cell == em dash
    assert cells[3] == raw  # Kernel Name cell == raw device symbol
    # The normalized short form still labels the P-item heading.
    assert f"#### P1: {display}" in md


# ---------------------------------------------------------------------------
# Em-dash-Operation contract — every row emits an em-dash Operation cell, and the
# relaxed reader recovers a candidate per row via the device kernel symbol.
# ---------------------------------------------------------------------------
def test_option_b_operation_is_em_dash_reader_substitutes_kernel_name(tmp_path):
    rows = [
        ("hipGraphLaunch->foo (Synthetic Op)", 60.0, 60.0),
        ("hipGraphLaunch->bar (Synthetic Op)", 40.0, 40.0),
    ]
    perf_dir = _make_perf_csv(tmp_path, rows)
    md = render_fallback_report(perf_dir, 0.95)

    # Every rendered data row's Operation cell is the em dash (op never captured).
    data_rows = [
        line
        for line in md.splitlines()
        if line.startswith("| ") and "0.0" in line  # a data row (has a time cell)
    ]
    assert data_rows
    for line in data_rows:
        first_cell = line.split("|")[1].strip()
        assert first_cell == _EM_DASH

    # The relaxed reader still yields one candidate per row, keyed on the symbol.
    candidates = _parse_candidates(md)
    assert len(candidates) == 2
    assert {c["operation"] for c in candidates} == {"foo", "bar"}
    for c in candidates:
        assert c["operation"] == c["kernel_name"]


# ---------------------------------------------------------------------------
# U10 — N4 %E2E floor + top-N cap + drop-logging.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "percent, expected_kept",
    [
        (0.49, 0),  # below floor -> dropped
        (0.50, 1),  # == floor -> kept (strict >=)
        (0.51, 1),  # above floor -> kept
    ],
)
def test_u10_floor_boundary(tmp_path, percent, expected_kept):
    assert MIN_PITEM_PERCENT_E2E == 0.5
    rows = [("hipGraphLaunch->k (Synthetic Op)", 100.0, percent)]
    perf_dir = _make_perf_csv(tmp_path, rows)
    md = render_fallback_report(perf_dir, 0.95)

    assert len(_HEADING_RE.findall(md)) == expected_kept
    dropped = 1 - expected_kept
    if dropped:
        assert f"Dropped {dropped} P-items" in md
    else:
        assert "Dropped " not in md


def test_u10_top_n_cap_and_drop_logging(tmp_path):
    # MAX_PITEM_COUNT + 5 distinct kernels, all above the floor, descending weight.
    n = MAX_PITEM_COUNT + 5
    weights = [float(100 - i) for i in range(n)]
    rows = [
        (f"hipGraphLaunch->k{i:02d} (Synthetic Op)", w, 4.0)
        for i, w in enumerate(weights)
    ]
    perf_dir = _make_perf_csv(tmp_path, rows)
    md = render_fallback_report(perf_dir, 0.95)

    # Exactly MAX_PITEM_COUNT render; the rest dropped, logged in the banner.
    assert len(_HEADING_RE.findall(md)) == MAX_PITEM_COUNT
    dropped = len(weights) - MAX_PITEM_COUNT
    assert f"Dropped {dropped} P-items" in md

    # Survivor impact uses the FULL total (all rows), not the top-N subtotal.
    total_full = sum(weights)
    p1_pct = weights[0] / total_full * 100.0
    expected_p1_low = round(p1_pct * 0.15, 2)
    lows = [
        float(re.search(r"low=([\d.]+)", blob).group(1))
        for blob in _PITEM_MARKER_RE.findall(md)
    ]
    assert lows[0] == expected_p1_low
    # And that differs from what a top-N-only denominator would give.
    top_n_pct = weights[0] / sum(weights[:MAX_PITEM_COUNT]) * 100.0
    assert expected_p1_low != round(top_n_pct * 0.15, 2)


def test_u10_no_drop_note_when_nothing_dropped(tmp_path):
    rows = [("hipGraphLaunch->k (Synthetic Op)", 100.0, 60.0)]
    perf_dir = _make_perf_csv(tmp_path, rows)
    md = render_fallback_report(perf_dir, 0.95)

    assert "Dropped " not in md  # drop-note sentence suppressed at zero


# ---------------------------------------------------------------------------
# U9 extension — the banner drop-note stays inert to the parser.
# ---------------------------------------------------------------------------
def test_u9_drop_note_inert(tmp_path):
    rows = [
        ("hipGraphLaunch->foo (Synthetic Op)", 60.0, 60.0),
        ("hipGraphLaunch->bar (Synthetic Op)", 40.0, 40.0),
    ]
    perf_dir = _make_perf_csv(tmp_path, rows)
    md = render_fallback_report(perf_dir, 0.95)

    # Stripping the banner does not change the parsed candidate count.
    assert len(_parse_candidates(md)) == 2
    assert len(_parse_candidates(_strip_banner(md))) == 2


# ---------------------------------------------------------------------------
# U7 re-run — the Operation cell is the em dash, but the relaxed
# reader substitutes the device kernel symbol, so candidates still parse with a
# non-empty operation identity and `####` headings.
# ---------------------------------------------------------------------------
def test_u7_rerun_after_normalization(tmp_path):
    long_raw = "void aiter::" + "y" * 90
    rows = [
        (f"hipGraphLaunch->{long_raw} (Synthetic Op)", 60.0, 60.0),
        ("hipGraphLaunch->_fwd_kernel (Synthetic Op)", 40.0, 40.0),
    ]
    perf_dir = _make_perf_csv(tmp_path, rows)
    md = render_fallback_report(perf_dir, 0.95)

    candidates = _parse_candidates(md)
    assert len(candidates) == 2
    for cand in candidates:
        # Operation resolves to the device kernel symbol via reader substitution.
        assert cand["operation"]
        assert cand["operation"] not in ("", _EM_DASH)
        assert cand["operation"] == cand["kernel_name"]
    for line in md.splitlines():
        if "P1:" in line or "P2:" in line:
            assert line.startswith("####")


# ---------------------------------------------------------------------------
# U11 — field-size crash fix: a field larger than csv's 131072 default must not
# raise in the detector or the loader/writer.
# ---------------------------------------------------------------------------
def test_u11_giant_field_does_not_crash(tmp_path):
    giant = "hipGraphLaunch->" + ("k" * 200000) + " (Synthetic Op)"
    rows = [
        (giant, 90.0, 90.0),
        ("hipGraphLaunch->small (Synthetic Op)", 10.0, 10.0),
    ]
    perf_dir = _make_perf_csv(tmp_path, rows)

    verdict = check_graph_replay_coverage(perf_dir)  # must not raise
    assert verdict.graph_under_recorded is True

    md = render_fallback_report(
        perf_dir, verdict.graph_replay_fraction
    )  # must not raise
    assert "#### P1:" in md
