###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the deterministic graph-under-recorded fallback.

Covers the no-LLM fallback path in ``utils/deterministic_fallback``: the
detector ``check_graph_replay_coverage`` and the ``render_fallback_report`` md
writer. Every fixture is a tiny synthetic ``unified_perf_summary.csv`` written
under ``tmp_path`` with hand-computed expected values, so a failing assertion
points at the exact arithmetic. No GPU, no LLM, no network.
"""

import csv
import re

import pytest

from TraceLens.Agent.Analysis.utils.deterministic_fallback import (
    GRAPH_REPLAY_FRACTION_MAX,
    MAX_PITEM_COUNT,
    MIN_PITEM_PERCENT_E2E,
    _KERNEL_NAME_TRUNC_LEN,
    _normalize_kernel_name,
    check_graph_replay_coverage,
    main,
    render_fallback_report,
)

# ---------------------------------------------------------------------------
# Structural regexes: match the markers/headings the producer emits, so the
# tests assert the SHAPE of the fallback md directly.
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

# The 9 canonical column tokens the producer emits in order; "Kernel Name" is an
# extra column allowed to interleave.
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

_EM_DASH = "—"


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------
def _make_perf_csv(tmp_path, rows):
    """Write a synthetic ``unified_perf_summary.csv`` and return its path.

    ``rows`` is a list of ``(name, weight, percent)`` tuples. Only the three
    columns the code reads are written; ``percent`` defaults to 0.0 if omitted.
    """
    csv_path = tmp_path / "unified_perf_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["name", "Kernel Time (µs)_sum", "Percentage (%)"])
        for row in rows:
            name, weight = row[0], row[1]
            percent = row[2] if len(row) > 2 else 0.0
            writer.writerow([name, weight, percent])
    return csv_path


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
# detector fires on a graph-collapsed fixture
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "wrapper",
    [
        ("hipGraphLaunch->k1 (Synthetic Op)", "hipGraphLaunch->k2 (Synthetic Op)"),
        ("Torch-Compiled Region: 0/0->k (Synthetic Op)", None),
    ],
)
def test_detector_fires_on_graph_collapsed(tmp_path, wrapper):
    graph_rows = [(name, 47.5, 47.5) for name in wrapper if name]
    # Pad the graph weight to 95 total and add a benign plain kernel for the 5%.
    rows = graph_rows + [("plain_kernel", 5.0, 5.0)]
    # Normalize the graph weight to sum to 95 regardless of how many graph rows.
    graph_total = sum(w for _, w, _ in graph_rows)
    rows = [(name, w / graph_total * 95.0, p) for name, w, p in graph_rows] + [
        ("plain_kernel", 5.0, 5.0)
    ]

    perf_csv = _make_perf_csv(tmp_path, rows)
    verdict = check_graph_replay_coverage(perf_csv)

    assert verdict.graph_under_recorded is True
    assert verdict.graph_replay_fraction == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# detector stays quiet on a healthy fixture
# ---------------------------------------------------------------------------
def test_detector_quiet_on_healthy(tmp_path):
    rows = [
        ("aten::mm", 40.0, 40.0),
        ("aten::softmax", 30.0, 30.0),
        ("hipMemcpyAsync->copy_dtoh (Synthetic Op)", 15.0, 15.0),
        ("hipModuleLaunchKernel->some_kernel (Synthetic Op)", 15.0, 15.0),
    ]
    perf_csv = _make_perf_csv(tmp_path, rows)
    verdict = check_graph_replay_coverage(perf_csv)

    assert verdict.graph_under_recorded is False
    assert verdict.graph_replay_fraction == 0.0


# ---------------------------------------------------------------------------
# false-positive guard: benign launch/memcpy wrappers never count
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
def test_detector_ignores_plumbing_wrappers(tmp_path, wrapper_name):
    rows = [
        (wrapper_name, 40.0, 40.0),
        ("aten::mm", 60.0, 60.0),
    ]
    perf_csv = _make_perf_csv(tmp_path, rows)
    verdict = check_graph_replay_coverage(perf_csv)

    assert verdict.graph_under_recorded is False
    assert verdict.graph_replay_fraction == 0.0


# ---------------------------------------------------------------------------
# threshold boundary (strict `>` around 0.10)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "graph_weight, expected",
    [
        (9.9, False),  # 0.099 < 0.10
        (10.0, False),  # 0.100 == 0.10 -> strict > means still False
        (10.1, True),  # 0.101 > 0.10
    ],
)
def test_detector_threshold_boundary(tmp_path, graph_weight, expected):
    assert GRAPH_REPLAY_FRACTION_MAX == 0.10
    rows = [
        ("hipGraphLaunch->k (Synthetic Op)", graph_weight, graph_weight),
        ("aten::mm", 100.0 - graph_weight, 100.0 - graph_weight),
    ]
    perf_csv = _make_perf_csv(tmp_path, rows)
    verdict = check_graph_replay_coverage(perf_csv)

    assert verdict.graph_under_recorded is expected
    assert verdict.graph_replay_fraction == pytest.approx(graph_weight / 100.0)


# ---------------------------------------------------------------------------
# wrapper-strip + exact-name grouping
# ---------------------------------------------------------------------------
def test_writer_strips_wrapper_and_groups_by_name(tmp_path):
    rows = [
        ("hipGraphLaunch->foo (Synthetic Op)", 10.0, 10.0),
        ("hipGraphLaunch->foo (Synthetic Op)", 30.0, 30.0),
        ("hipGraphLaunch->bar (Synthetic Op)", 20.0, 20.0),
    ]
    perf_csv = _make_perf_csv(tmp_path, rows)
    md = render_fallback_report(perf_csv, 0.95)

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


def test_writer_keeps_tile_variants_separate(tmp_path):
    rows = [
        ("hipGraphLaunch->f4gemm_192x128 (Synthetic Op)", 30.0, 30.0),
        ("hipGraphLaunch->f4gemm_32x128 (Synthetic Op)", 10.0, 10.0),
    ]
    perf_csv = _make_perf_csv(tmp_path, rows)
    md = render_fallback_report(perf_csv, 0.95)

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
# plumbing behind a graph-replay wrapper is dropped from the writer output.
# Guards the leak where the wrapper prefix is graph-replay but the bare kernel
# behind it is runtime plumbing (e.g. `hipGraphLaunch->__amd_rocclr_copyBuffer`).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "plumbing_kernel",
    ["__amd_rocclr_copyBuffer", "__amd_rocclr_fillBufferAligned", "MemcpyDtoD"],
)
def test_writer_drops_plumbing_behind_graph_wrapper(tmp_path, plumbing_kernel):
    rows = [
        ("hipGraphLaunch->real_kernel (Synthetic Op)", 90.0, 90.0),
        (f"hipGraphLaunch->{plumbing_kernel} (Synthetic Op)", 10.0, 10.0),
        (f"Torch-Compiled Region: 0/0->{plumbing_kernel} (Synthetic Op)", 5.0, 5.0),
    ]
    perf_csv = _make_perf_csv(tmp_path, rows)
    md = render_fallback_report(perf_csv, 0.95)

    names = {name for _, name in _HEADING_RE.findall(md)}
    assert names == {"real_kernel"}
    assert plumbing_kernel not in md


# ---------------------------------------------------------------------------
# impact arithmetic (match the exact emitted string)
# ---------------------------------------------------------------------------
def test_writer_impact_arithmetic(tmp_path):
    # Two equal-weight rows with distinct names -> each is its own group at 50%.
    rows = [
        ("hipGraphLaunch->foo (Synthetic Op)", 50.0, 50.0),
        ("hipGraphLaunch->bar (Synthetic Op)", 50.0, 50.0),
    ]
    perf_csv = _make_perf_csv(tmp_path, rows)
    md = render_fallback_report(perf_csv, 0.95)

    # 50 * 0.15/0.30/0.45 = 7.5 / 15.0 / 22.5 (round(...,2) prints 15.0, not 15).
    assert (
        "impact-begin kind=p_item category=unknown low=7.5 mid=15.0 high=22.5 -->" in md
    )


# ---------------------------------------------------------------------------
# producer structural contract: heading order, marker-before-heading, the
# canonical 9-column header, and one em-dash data row per P-item.
# ---------------------------------------------------------------------------
def test_writer_structure_contract(tmp_path):
    # Two distinct graph-replay kernels; higher weight sorts first -> P1.
    rows = [
        ("hipGraphLaunch->foo (Synthetic Op)", 60.0, 60.0),
        ("hipGraphLaunch->bar (Synthetic Op)", 40.0, 40.0),
    ]
    perf_csv = _make_perf_csv(tmp_path, rows)
    md = render_fallback_report(perf_csv, 0.95)

    # Exactly two P-item headings, ranked by weight (foo=60 -> P1, bar=40 -> P2).
    headings = _HEADING_RE.findall(md)
    assert len(headings) == 2
    assert {int(rank): title for rank, title in headings} == {1: "foo", 2: "bar"}

    # Each reasoning-candidate marker precedes its own #### PN heading.
    for rank in (1, 2):
        marker_idx = md.index(f"reasoning-candidate tier=compute rank={rank}")
        heading_idx = md.index(f"#### P{rank}:")
        assert marker_idx < heading_idx

    # Each P-item carries a **Data:** line and a table header holding the 9
    # canonical column tokens in order (Kernel Name may interleave).
    assert md.count("**Data:**") == 2
    for header_line in [
        line for line in md.splitlines() if line.strip().startswith("| Operation |")
    ]:
        header_tokens = [c.strip().lower() for c in header_line.split("|")[1:-1]]
        canonical = [t for t in header_tokens if t in _DATA_TABLE_HEADER_TOKENS]
        assert canonical == list(_DATA_TABLE_HEADER_TOKENS)
    assert (
        len(
            [
                line
                for line in md.splitlines()
                if line.strip().startswith("| Operation |")
            ]
        )
        == 2
    )

    # One data row per P-item: Operation cell is the em dash, Kernel Name is the
    # raw symbol, with the expected time (µs -> ms) and %E2E.
    for kernel, time_ms, pct in (("foo", "0.060", "60.00"), ("bar", "0.040", "40.00")):
        cells = _data_row_cells(md, _EM_DASH)  # each row starts with the em dash
        assert kernel in md
        row = next(
            [c.strip() for c in line.split("|")[1:-1]]
            for line in md.splitlines()
            if line.strip().startswith("| —") and f"| {kernel} |" in line
        )
        assert row[0] == _EM_DASH  # Operation cell
        assert row[3] == kernel  # Kernel Name cell == raw symbol
        assert row[4] == time_ms
        assert row[5] == pct
    assert cells is not None  # a data row exists

    # NEGATIVE — downgrading #### P to ### P makes _HEADING_RE find no headings.
    downgraded = md.replace("#### P", "### P")
    assert len(_HEADING_RE.findall(downgraded)) == 0


# ---------------------------------------------------------------------------
# separate-row vs grouped invariant; plumbing dropped
# ---------------------------------------------------------------------------
def test_writer_separate_rows_grouped_impact(tmp_path):
    rows = [
        ("hipGraphLaunch->foo (Synthetic Op)", 30.0, 30.0),
        ("hipGraphLaunch->foo (Synthetic Op)", 10.0, 10.0),
        ("hipGraphLaunch->bar (Synthetic Op)", 20.0, 20.0),
        # Benign plumbing must be dropped from the candidate table entirely.
        ("hipMemcpyAsync->copy (Synthetic Op)", 5.0, 5.0),
    ]
    perf_csv = _make_perf_csv(tmp_path, rows)
    md = render_fallback_report(perf_csv, 0.95)

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
# degraded banner present and structurally inert
# ---------------------------------------------------------------------------
def test_writer_degraded_banner(tmp_path):
    rows = [
        ("hipGraphLaunch->foo (Synthetic Op)", 95.0, 95.0),
        ("aten::mm", 5.0, 5.0),
    ]
    perf_csv = _make_perf_csv(tmp_path, rows)
    frac = 0.95
    md = render_fallback_report(perf_csv, frac)

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
    assert len(name) < _KERNEL_NAME_TRUNC_LEN
    assert _normalize_kernel_name(name) == name


def test_normalizer_void_template_truncated_void_kept():
    # `void ` is KEPT (matches the GOOD report); only length is trimmed.
    raw = (
        "void at::native::elementwise_kernel<128, 2, "
        "at::native::gpu_kernel_impl<at::native::BinaryFunctor>>(int, char*)"
    )
    assert len(raw) > _KERNEL_NAME_TRUNC_LEN
    out = _normalize_kernel_name(raw)
    assert out.startswith("void ")
    assert out == raw[:_KERNEL_NAME_TRUNC_LEN] + "..."
    assert len(out) == _KERNEL_NAME_TRUNC_LEN + 3


def test_normalizer_tensile_truncated():
    raw = "Cijk_Ailk_Bljk_HHS_BH_Bias_AS_MT128x160x16_MI16x16x16x1_SN_WG32_8_1_ABCD"
    raw = raw + "_extra_tokens_pushing_well_past_the_cap"
    assert len(raw) > _KERNEL_NAME_TRUNC_LEN
    out = _normalize_kernel_name(raw)
    assert out == raw[:_KERNEL_NAME_TRUNC_LEN] + "..."


def test_normalizer_mangled_zn_truncated_not_demangled():
    # Locks in the no-demangle decision: the symbol stays mangled, just shorter.
    raw = "_ZN5aiter24add_rmsnorm_quant_kernelIN3std10bfloat16_tES2_Li256ELb1EEEvPT_"
    raw = raw + "PKS1_iiffb"
    assert len(raw) > _KERNEL_NAME_TRUNC_LEN
    out = _normalize_kernel_name(raw)
    assert out.startswith("_ZN")
    assert out.endswith("...")
    assert out == raw[:_KERNEL_NAME_TRUNC_LEN] + "..."


def test_normalizer_never_returns_empty():
    # The non-empty invariant: a blank input falls back to raw, never "".
    assert _normalize_kernel_name("") == ""
    assert _normalize_kernel_name("   ") == "   "


def test_normalizer_length_boundary():
    at_cap = "a" * _KERNEL_NAME_TRUNC_LEN
    over_cap = "a" * (_KERNEL_NAME_TRUNC_LEN + 1)
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
    perf_csv = _make_perf_csv(tmp_path, rows)
    md = render_fallback_report(perf_csv, 0.95)

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
    perf_csv = _make_perf_csv(tmp_path, rows)
    md = render_fallback_report(perf_csv, 0.95)

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
# Em-dash Operation contract — every rendered data row's Operation cell is the
# em dash (the op is never captured on a graph-collapsed trace).
# ---------------------------------------------------------------------------
def test_data_row_operation_is_em_dash(tmp_path):
    rows = [
        ("hipGraphLaunch->foo (Synthetic Op)", 60.0, 60.0),
        ("hipGraphLaunch->bar (Synthetic Op)", 40.0, 40.0),
    ]
    perf_csv = _make_perf_csv(tmp_path, rows)
    md = render_fallback_report(perf_csv, 0.95)

    data_rows = [
        line
        for line in md.splitlines()
        if line.startswith("| ") and "0.0" in line  # a data row (has a time cell)
    ]
    assert data_rows
    for line in data_rows:
        assert line.split("|")[1].strip() == _EM_DASH


# ---------------------------------------------------------------------------
# N4 %E2E floor + top-N cap + drop-logging.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "percent, expected_kept",
    [
        (0.49, 0),  # below floor -> dropped
        (0.50, 1),  # == floor -> kept (strict >=)
        (0.51, 1),  # above floor -> kept
    ],
)
def test_writer_pitem_floor_boundary(tmp_path, percent, expected_kept):
    assert MIN_PITEM_PERCENT_E2E == 0.5
    rows = [("hipGraphLaunch->k (Synthetic Op)", 100.0, percent)]
    perf_csv = _make_perf_csv(tmp_path, rows)
    md = render_fallback_report(perf_csv, 0.95)

    assert len(_HEADING_RE.findall(md)) == expected_kept
    dropped = 1 - expected_kept
    if dropped:
        assert f"Dropped {dropped} P-items" in md
    else:
        assert "Dropped " not in md


def test_writer_pitem_cap_and_drop_logging(tmp_path):
    # MAX_PITEM_COUNT + 5 distinct kernels, all above the floor, descending weight.
    n = MAX_PITEM_COUNT + 5
    weights = [float(100 - i) for i in range(n)]
    rows = [
        (f"hipGraphLaunch->k{i:02d} (Synthetic Op)", w, 4.0)
        for i, w in enumerate(weights)
    ]
    perf_csv = _make_perf_csv(tmp_path, rows)
    md = render_fallback_report(perf_csv, 0.95)

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


def test_writer_no_drop_note_when_nothing_dropped(tmp_path):
    rows = [("hipGraphLaunch->k (Synthetic Op)", 100.0, 60.0)]
    perf_csv = _make_perf_csv(tmp_path, rows)
    md = render_fallback_report(perf_csv, 0.95)

    assert "Dropped " not in md  # drop-note sentence suppressed at zero


def test_writer_drop_note_parser_inert(tmp_path):
    rows = [
        ("hipGraphLaunch->foo (Synthetic Op)", 60.0, 60.0),
        ("hipGraphLaunch->bar (Synthetic Op)", 40.0, 40.0),
    ]
    perf_csv = _make_perf_csv(tmp_path, rows)
    md = render_fallback_report(perf_csv, 0.95)

    # Stripping the banner leaves heading and marker counts unchanged.
    stripped = _strip_banner(md)
    assert len(_HEADING_RE.findall(md)) == len(_HEADING_RE.findall(stripped))
    assert len(_PITEM_MARKER_RE.findall(md)) == len(_PITEM_MARKER_RE.findall(stripped))
    assert len(_REASONING_MARKER_RE.findall(md)) == len(
        _REASONING_MARKER_RE.findall(stripped)
    )


def test_detector_handles_giant_csv_field(tmp_path):
    giant = "hipGraphLaunch->" + ("k" * 200000) + " (Synthetic Op)"
    rows = [
        (giant, 90.0, 90.0),
        ("hipGraphLaunch->small (Synthetic Op)", 10.0, 10.0),
    ]
    perf_csv = _make_perf_csv(tmp_path, rows)

    verdict = check_graph_replay_coverage(perf_csv)  # must not raise
    assert verdict.graph_under_recorded is True

    md = render_fallback_report(
        perf_csv, verdict.graph_replay_fraction
    )  # must not raise
    assert "#### P1:" in md


def test_main_writes_analysis_md(tmp_path, monkeypatch):
    rows = [
        ("hipGraphLaunch->k (Synthetic Op)", 90.0, 90.0),
        ("aten::mm", 10.0, 10.0),
    ]
    perf_csv = _make_perf_csv(tmp_path, rows)
    out_dir = tmp_path / "out"
    monkeypatch.setattr(
        "sys.argv",
        [
            "deterministic_fallback",
            "--unified-perf-csv",
            str(perf_csv),
            "--output-dir",
            str(out_dir),
            "--graph-replay-fraction",
            "0.9",
        ],
    )

    main()

    md = (out_dir / "analysis.md").read_text(encoding="utf-8")
    assert "#### P1:" in md


def test_main_derives_fraction_when_omitted(tmp_path, monkeypatch):
    rows = [
        ("hipGraphLaunch->k (Synthetic Op)", 90.0, 90.0),
        ("aten::mm", 10.0, 10.0),
    ]
    perf_csv = _make_perf_csv(tmp_path, rows)
    out_dir = tmp_path / "out"
    monkeypatch.setattr(
        "sys.argv",
        [
            "deterministic_fallback",
            "--unified-perf-csv",
            str(perf_csv),
            "--output-dir",
            str(out_dir),
        ],
    )

    main()

    # 90% graph-replay fraction is recovered from the gate and rendered in the banner.
    md = (out_dir / "analysis.md").read_text(encoding="utf-8")
    assert "90%" in md
    assert "#### P1:" in md
