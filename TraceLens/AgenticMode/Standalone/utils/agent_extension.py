###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Optional standalone-analysis extension: detailed plot + ms-savings.

Hosts the legacy 2-panel cumulative-savings chart (cumulative stacked E2E bars
+ throughput cone with the 75-100% roofline band) and a marker-driven rewriter
that converts post-Phase-1 ``impact_score``-format markdown reports back to
the pre-Phase-1 ``ms savings (% of E2E)`` format.

The rewriter is driven entirely by data-bearing HTML-comment markers
(``<!-- impact-begin kind=... ... -->`` / ``<!-- impact-end -->``) emitted by
the sub-agent spec and the orchestrator template. 

CLI usage:

    python TraceLens/AgenticMode/Standalone/utils/agent_extension.py \\
        --output-dir <dir> \\
        --title '<Model> on <Platform> -- Kernel Tuning Potential'

The extension is opt-in by file presence and is NOT re-exported from
``TraceLens.AgenticMode.Standalone.utils.__init__``.
"""

import argparse
import base64
import glob
import json
import os
import re
import sys
from typing import Dict, List, Optional

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from TraceLens.AgenticMode.Standalone.utils.plot_utils import (
    _CAT_PALETTE,
    _REST_KEY,
    _short_name,
    generate_priority_data,
)
from TraceLens.AgenticMode.Standalone.utils.report_utils import load_manifest


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_RE_MARKER_BEGIN = re.compile(r"<!--\s*impact-begin\s+(.*?)\s*-->")
_RE_MARKER_END = re.compile(r"<!--\s*impact-end\s*-->")
# Per-row trailer carried inside Top Operations table body rows.
_RE_TOP_OPS_ROW_TRAILER = re.compile(
    r"\s*<!--\s*top-ops-row\s+(.*?)\s*-->\s*$"
)

_LEGACY_HEADER_5COL = (
    "| Recommendation | Type | Estimated Savings (ms) | "
    "Estimated Improvement (E2E %) | Confidence |\n"
    "|---------------|------|----------------------|"
    "-------------------------------|------------|"
)

_LEGACY_TOP_OPS_HEADER = (
    "| Rank | Category | Time (ms) | % of Compute Time | Ops | "
    "Potential improvement (time, E2E %) |\n"
    "|------|----------|-----------|-------------------|-----|"
    "-------------------------------------|"
)

# ---------------------------------------------------------------------------
# Detailed 2-panel chart (impact_score -> ms at the read boundary)
# ---------------------------------------------------------------------------


def _compute_cumulative_projections(baseline_ms: float, recommendations: List[dict]):
    """Accumulate per-step latency, savings, and throughput from recommendations.

    Reads ``impact_score`` / ``impact_score_low`` / ``impact_score_high`` from
    each recommendation and converts to ms at this single read boundary
    (``ms = impact_score * baseline_ms / 100``). Everything below this point in
    the function is in ms units, identical to the legacy ``savings_ms*`` flow.
    """
    cum_mid = cum_lo = cum_hi = 0.0
    steps = ["Baseline"]
    e2e_ms = [baseline_ms]
    savings = [0]
    rel = [100.0]
    err_lo = [0.0]
    err_hi = [0.0]
    rel_lo = [100.0]
    rel_hi = [100.0]

    for rec in recommendations:
        score_mid = float(rec.get("impact_score", 0))
        score_lo = float(rec.get("impact_score_low", score_mid))
        score_hi = float(rec.get("impact_score_high", score_mid))
        sm = score_mid * baseline_ms / 100.0
        sl = score_lo * baseline_ms / 100.0
        sh = score_hi * baseline_ms / 100.0
        cum_mid += sm
        cum_lo += sl
        cum_hi += sh
        lat_mid = max(0.01, baseline_ms - cum_mid)
        lat_best = max(0.01, baseline_ms - cum_hi)
        lat_worst = max(0.01, baseline_ms - cum_lo)
        cnt = rec.get("operation_count", 1)
        steps.append(f"{_short_name(rec['category'], max_len=20)}\n({cnt} ops)")
        e2e_ms.append(lat_mid)
        savings.append(sm)
        rel.append(round(baseline_ms / lat_mid * 100, 1))
        err_lo.append(lat_mid - lat_best)
        err_hi.append(lat_worst - lat_mid)
        rel_lo.append(
            round(baseline_ms / lat_worst * 100 if lat_worst > 0 else 100.0, 1)
        )
        rel_hi.append(round(baseline_ms / lat_best * 100 if lat_best > 0 else 100.0, 1))

    return {
        "steps": steps,
        "e2e_ms": np.array(e2e_ms, dtype=float),
        "savings": savings,
        "rel": rel,
        "err_lo": np.array(err_lo, dtype=float),
        "err_hi": np.array(err_hi, dtype=float),
        "rel_lo": rel_lo,
        "rel_hi": rel_hi,
    }


def _build_stacked_segments(manifest, recommendations, baseline_ms: float):
    """Build per-category segment data for the stacked bar chart."""
    plotted_cats = {r["category"] for r in recommendations}
    baseline_by_cat: Dict[str, float] = {}
    for cat in manifest["categories"]:
        if cat.get("tier") != "compute_kernel":
            continue
        gt = float(cat.get("gpu_kernel_time_ms", 0) or 0)
        if gt > 0:
            baseline_by_cat[cat["name"]] = gt

    kernel_sum = sum(baseline_by_cat.values())
    rest_e2e = max(0.0, baseline_ms - kernel_sum)
    if rest_e2e > 0:
        baseline_by_cat[_REST_KEY] = rest_e2e

    analyzed = sorted(
        [
            {"name": n, "time_ms": t}
            for n, t in baseline_by_cat.items()
            if n != _REST_KEY and n in plotted_cats
        ],
        key=lambda x: x["time_ms"],
    )
    unanalyzed = sorted(
        [
            {"name": n, "time_ms": t}
            for n, t in baseline_by_cat.items()
            if n != _REST_KEY and n not in plotted_cats
        ],
        key=lambda x: x["time_ms"],
    )
    segment_order = [x["name"] for x in analyzed + unanalyzed]
    if _REST_KEY in baseline_by_cat:
        segment_order.append(_REST_KEY)

    return baseline_by_cat, segment_order, plotted_cats


def _build_color_map(recommendations):
    """Map each recommendation's category to a consistent palette color."""
    cmap: Dict[str, str] = {}
    for i, rec in enumerate(recommendations):
        cmap[rec["category"]] = _CAT_PALETTE[i % len(_CAT_PALETTE)]
    return cmap


def _render_stacked_bars(
    ax,
    proj,
    recommendations,
    baseline_by_cat,
    segment_order,
    cat_color_map,
    show_error_bars,
    baseline_ms: float,
):
    """Render the left-panel stacked E2E bars with labels and error bars.

    ``baseline_ms`` is required to convert each recommendation's
    ``impact_score`` back to ms at the single read boundary inside
    ``_segment_heights``; everything else is identical to the legacy
    ``savings_ms*``-based renderer.
    """
    n_bars = len(proj["steps"])
    x = np.arange(n_bars, dtype=float)
    width = 0.62
    rest_label = "Non-computing"
    rest_color = "#aab7c4"

    def _segment_heights(num_recs: int) -> Dict[str, float]:
        savings_by_cat: Dict[str, float] = {}
        for j in range(num_recs):
            r = recommendations[j]
            c = r["category"]
            sv = float(r.get("impact_score", 0)) * baseline_ms / 100.0
            if c in baseline_by_cat and c != _REST_KEY:
                savings_by_cat[c] = savings_by_cat.get(c, 0.0) + sv
            elif _REST_KEY in baseline_by_cat:
                savings_by_cat[_REST_KEY] = savings_by_cat.get(_REST_KEY, 0.0) + sv
        return {
            name: max(
                0.0, baseline_by_cat.get(name, 0.0) - savings_by_cat.get(name, 0.0)
            )
            for name in segment_order
        }

    bottom = np.zeros(n_bars, dtype=float)
    rest_bottom = np.zeros(n_bars, dtype=float)
    rest_height = np.zeros(n_bars, dtype=float)
    for seg_name in segment_order:
        h = np.array(
            [_segment_heights(k)[seg_name] for k in range(n_bars)], dtype=float
        )
        if np.all(h <= 0):
            continue
        color = (
            rest_color
            if seg_name == _REST_KEY
            else cat_color_map.get(seg_name, "#bdc3c7")
        )
        lbl = rest_label if seg_name == _REST_KEY else _short_name(seg_name)
        ax.bar(
            x,
            h,
            width,
            bottom=bottom,
            label=lbl,
            color=color,
            edgecolor="white",
            linewidth=0.9,
            alpha=0.95,
        )
        if seg_name == _REST_KEY:
            rest_bottom = bottom.copy()
            rest_height = h.copy()
        bottom += h

    for k in range(n_bars):
        total_h = float(bottom[k])
        accent = "#333333"
        if k > 0:
            cat = recommendations[k - 1].get("category", "")
            accent = cat_color_map.get(cat, "#333333")
        label_y = total_h + (proj["err_hi"][k] + 1.2 if show_error_bars else 1.2)
        ax.text(
            x[k],
            label_y,
            f"{proj['e2e_ms'][k]:.1f} ms",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color=accent,
        )
        if k > 0 and proj["savings"][k] > 0:
            rest_mid = float(rest_bottom[k]) + float(rest_height[k]) / 2
            ax.text(
                x[k],
                rest_mid,
                f"-{proj['savings'][k]:.1f} ms",
                ha="center",
                va="center",
                fontsize=8,
                color=accent,
                fontweight="bold",
                zorder=5,
            )
        if show_error_bars:
            ax.errorbar(
                [x[k]],
                [total_h],
                yerr=[[proj["err_lo"][k]], [proj["err_hi"][k]]],
                fmt="none",
                ecolor=accent,
                elinewidth=1.1,
                capsize=3,
                zorder=6,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(proj["steps"], fontsize=8)
    ax.set_ylabel("E2E time stacked by category (ms)", fontsize=11)
    ax.set_title(
        "Projected E2E Latency (stacked by kernel category)",
        fontsize=11,
        fontweight="bold",
        pad=10,
    )
    ymax = (
        float(np.max(bottom + proj["err_hi"]) + 8)
        if show_error_bars
        else float(np.max(bottom) + 8)
    )
    ax.set_ylim(0, ymax * 1.12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    rec_order = [_short_name(r["category"]) for r in recommendations]
    ordered_handles = []
    ordered_labels = []
    for rn in rec_order:
        if rn in labels:
            idx = labels.index(rn)
            ordered_handles.append(handles[idx])
            ordered_labels.append(labels[idx])
    for h, l in zip(handles, labels):
        if l not in ordered_labels:
            ordered_handles.append(h)
            ordered_labels.append(l)
    ax.legend(
        ordered_handles,
        ordered_labels,
        loc="upper left",
        bbox_to_anchor=(1.0, 1.0),
        fontsize=7.5,
        frameon=False,
        title="Operations",
    )


def _render_throughput_cone(ax, proj):
    """Render the cumulative throughput cone on the given axes."""
    x_vals = np.arange(len(proj["steps"]))
    cum_arr = np.array(proj["rel"], dtype=float)
    cum_lo = np.array(proj["rel_lo"], dtype=float)
    cum_hi = np.array(proj["rel_hi"], dtype=float)
    cum_lo[0] = cum_hi[0] = cum_arr[0]

    ax.fill_between(x_vals, cum_lo, cum_hi, color="#2ecc71", alpha=0.15, zorder=1)
    ax.plot(x_vals, cum_hi, "-", color="#27ae60", linewidth=1.0, alpha=0.4, zorder=2)
    ax.plot(x_vals, cum_lo, "-", color="#27ae60", linewidth=1.0, alpha=0.4, zorder=2)
    ax.plot(
        x_vals,
        proj["rel"],
        "o-",
        color="#2ecc71",
        linewidth=2.5,
        markersize=9,
        markerfacecolor="white",
        markeredgewidth=2.5,
        zorder=4,
    )
    last_idx = len(proj["rel"]) - 1
    ax.annotate(
        f"{proj['rel'][last_idx]:.0f}%",
        (last_idx, proj["rel"][last_idx]),
        textcoords="offset points",
        xytext=(0, -16),
        ha="center",
        va="top",
        fontsize=11,
        fontweight="bold",
        color="#1a7a3a",
        zorder=5,
    )
    ax.set_xticks(range(len(proj["steps"])))
    ax.set_xticklabels(proj["steps"], fontsize=9)
    ax.set_ylabel("% Relative Throughput (Baseline = 100)", fontsize=11)
    ax.set_title(
        "Cumulative Throughput Improvement\n(75\u2013100% roofline potential)",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.set_ylim(80, max(cum_hi) * 1.15)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.axhline(y=100, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=9)


def generate_impact_savings_plot(
    output_dir: str,
    title: str,
    output_filename: str = "perf_improvement.png",
    write_base64: bool = False,
    show_error_bars: bool = True,
) -> bool:
    """Generate the detailed 2-panel cumulative-savings plot.

    Reads ``priority_data.json`` (auto-generates it if missing). Converts
    ``impact_score*`` back to ms at the single read boundary so all downstream
    labels and axes match the legacy plot byte-for-byte.

    Args:
        output_dir: Directory containing ``category_data/category_manifest.json``
            and ``priority_data.json`` (auto-created if missing).
        title: Figure suptitle.
        output_filename: PNG filename under ``output_dir``.
        write_base64: Write a base64 sidecar (default False -- the public
            report has no placeholder for this chart).
        show_error_bars: Show 75-100% roofline error caps on the left panel.

    Returns:
        True if the figure was written, False if inputs were missing or invalid.
    """
    plot_data_path = os.path.join(output_dir, "priority_data.json")
    if not os.path.exists(plot_data_path):
        try:
            generate_priority_data(output_dir)
        except Exception as e:
            print(f"priority_data generation failed: {e}")
            return False

    if not os.path.exists(plot_data_path):
        print(f"priority_data.json not found at {plot_data_path} - skipping plot")
        return False

    with open(plot_data_path, "r") as f:
        plot_data = json.load(f)

    baseline_ms = float(plot_data.get("baseline_ms", 0))
    recommendations = plot_data.get("recommendations", [])
    if not recommendations or baseline_ms <= 0:
        print("No kernel tuning recommendations or invalid baseline - skipping plot")
        return False

    try:
        manifest = load_manifest(output_dir)
    except FileNotFoundError:
        print("category_manifest.json not found - skipping plot")
        return False

    proj = _compute_cumulative_projections(baseline_ms, recommendations)
    baseline_by_cat, segment_order, _ = _build_stacked_segments(
        manifest, recommendations, baseline_ms
    )
    cat_color_map = _build_color_map(recommendations)

    fig, (ax_stack, ax_cone) = plt.subplots(
        1,
        2,
        figsize=(14, 5.5),
        gridspec_kw={"width_ratios": [1.45, 1.0]},
    )

    _render_stacked_bars(
        ax_stack,
        proj,
        recommendations,
        baseline_by_cat,
        segment_order,
        cat_color_map,
        show_error_bars,
        baseline_ms,
    )

    fig.text(
        0.5,
        -0.02,
        "Gray bars represent categories without performance models — not reflected in savings projections.",
        ha="center",
        fontsize=8.5,
        color="#888888",
        style="italic",
    )

    _render_throughput_cone(ax_cone, proj)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    out_path = os.path.join(output_dir, output_filename)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Detailed impact-savings plot saved to {out_path}")

    if write_base64:
        with open(out_path, "rb") as f:
            b64_str = base64.b64encode(f.read()).decode("ascii")
        b64_path = os.path.join(
            output_dir, output_filename.replace(".png", "_base64.txt")
        )
        with open(b64_path, "w") as f:
            f.write(b64_str)
        print(f"Base64 written to {b64_path}")

    return True


# ---------------------------------------------------------------------------
# Markdown rehydration: marker-driven impact_score -> pre-Phase-1 ms savings
# ---------------------------------------------------------------------------
#
# The sub-agent spec and the orchestrator template emit data-bearing HTML
# comments around every block whose contents depend on impact_score values:
#
#     <!-- impact-begin kind=KIND attr1=value1 attr2=value2 -->
#     ...arbitrary post-Phase-1 markdown content (the "raw" form)...
#     <!-- impact-end -->
#
# The walker locates each begin/end pair, dispatches on `kind`, and replaces
# the entire block (markers included) with the rendered legacy ms-form text.

def _load_baseline_ms(output_dir: str) -> float:
    """Read baseline_ms from category_manifest.json. Returns 0 if missing."""
    try:
        manifest = load_manifest(output_dir)
    except FileNotFoundError:
        return 0.0
    return float(manifest.get("gpu_utilization", {}).get("total_time_ms", 0) or 0)


def _parse_attrs(attr_str: str) -> Dict[str, Optional[str]]:
    """Parse ``key=value`` pairs from a marker attribute string.
    """
    attrs: Dict[str, Optional[str]] = {}
    for match in re.finditer(r'(\w+)=("([^"]*)"|(\S+))', attr_str):
        key = match.group(1)
        value = match.group(3) if match.group(3) is not None else match.group(4)
        attrs[key] = None if value == "null" else value
    return attrs


def _attr_float(attrs: Dict[str, Optional[str]], key: str) -> Optional[float]:
    raw = attrs.get(key)
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


# --- Per-kind renderers --------------------------------------------------


def _render_p_item(
    attrs: Dict[str, Optional[str]],
    body: str,
    baseline_ms: float,
) -> str:
    """Render a P-item Impact line in legacy ms-savings form.

    Quantifiable cards (numeric `low`/`high` attrs) render as
    ``**Impact**: ~<lo>-<hi> ms savings (<lo>-<hi>% of E2E)``.

    Non-quantifiable cards (any of `low`/`high` is null) render as
    ``**Impact**: Not quantifiable from trace data``.
    """
    lo = _attr_float(attrs, "low")
    hi = _attr_float(attrs, "high")
    if lo is None or hi is None:
        return "**Impact**: Not quantifiable from trace data"

    lo_ms = lo * baseline_ms / 100.0
    hi_ms = hi * baseline_ms / 100.0
    rendered = (
        f"**Impact**: ~{lo_ms:.1f}\u2013{hi_ms:.1f} ms savings "
        f"({lo:.1f}\u2013{hi:.1f}% of E2E)"
    )
    category = attrs.get("category")
    if category:
        rendered += (
            f" from closing efficiency gaps to 75\u2013100% of roofline "
            f"(pre-computed from `{category}_metrics.json` impact_estimates)."
        )
    return rendered


def _render_detail_estimate(
    attrs: Dict[str, Optional[str]],
    body: str,
    baseline_ms: float,
) -> str:
    """Render the two-bullet Low/High Impact estimate block in ms+E2E% form."""
    lo = _attr_float(attrs, "low")
    hi = _attr_float(attrs, "high")
    if lo is None or hi is None:
        return body
    lo_ms = lo * baseline_ms / 100.0
    hi_ms = hi * baseline_ms / 100.0
    return (
        f"- Low end (75% roofline): {lo_ms:.3f} ms savings ({lo:.2f}% E2E)\n"
        f"- High end (100% roofline): {hi_ms:.3f} ms savings ({hi:.2f}% E2E)"
    )


def _expand_impact_summary_row(row: str, baseline_ms: float) -> str:
    """Expand a 4-cell post-Phase-1 row into the legacy 5-cell shape.

    Input shape (4 cells):
        ``| Recommendation | Type | impact_score | Confidence |``
    Output shape (5 cells):
        ``| Recommendation | Type | Estimated Savings (ms) | impact_score | Confidence |``

    """
    cells = [c.strip() for c in row.strip().strip("|").split("|")]
    if len(cells) != 4:
        return row
    rec, type_, score_cell, conf = cells
    try:
        score = float(score_cell)
    except ValueError:
        return row
    savings_ms = score * baseline_ms / 100.0
    return f"| {rec} | {type_} | {savings_ms:.2f} | {score:.2f} | {conf} |"


def _render_impact_summary(
    attrs: Dict[str, Optional[str]],
    body: str,
    baseline_ms: float,
) -> str:
    """Replace the 4-column header+separator and expand any body rows."""
    lines = body.split("\n")
    out_lines: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        # Header line check: 4 cells matching canonical column names.
        if stripped.startswith("| Recommendation | Type | impact_score | Confidence |"):
            out_lines.append(_LEGACY_HEADER_5COL)
            i += 2  # skip the header and its separator line
            continue
        if line.startswith("|") and line.rstrip().endswith("|"):
            out_lines.append(_expand_impact_summary_row(line, baseline_ms))
        else:
            out_lines.append(line)
        i += 1
    return "\n".join(out_lines)


def _render_top_ops_row(row: str, baseline_ms: float) -> str:
    """Append the legacy Potential-improvement column to a Top Ops body row.

    The per-row ``<!-- top-ops-row low=X.X high=Y.Y -->`` trailer carries the
    impact_score range. Rows without a trailer (or with ``null`` values)
    receive a ``--`` placeholder.
    """
    trailer = _RE_TOP_OPS_ROW_TRAILER.search(row)
    if trailer is None:
        return row.rstrip() + " -- |"
    base = row[: trailer.start()].rstrip()
    attrs = _parse_attrs(trailer.group(1))
    lo = _attr_float(attrs, "low")
    hi = _attr_float(attrs, "high")
    if lo is None or hi is None or hi <= 0:
        opportunity = "--"
    else:
        lo_ms = lo * baseline_ms / 100.0
        hi_ms = hi * baseline_ms / 100.0
        opportunity = (
            f"~{lo_ms:.1f}\u2013{hi_ms:.1f} ms "
            f"({lo:.1f}\u2013{hi:.1f}%)"
        )
    return f"{base} {opportunity} |"


def _render_top_ops(
    attrs: Dict[str, Optional[str]],
    body: str,
    baseline_ms: float,
) -> str:
    """Restore the dropped Potential-improvement column on the Top Ops table."""
    lines = body.split("\n")
    out_lines: List[str] = []
    swapped_header = False
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if (
            not swapped_header
            and stripped.startswith("| Rank | Category | Time (ms) | % of Compute Time | Ops |")
        ):
            out_lines.append(_LEGACY_TOP_OPS_HEADER)
            swapped_header = True
            i += 2  # skip header + separator
            continue
        if line.startswith("|") and line.rstrip().endswith("|"):
            out_lines.append(_render_top_ops_row(line, baseline_ms))
        elif _RE_TOP_OPS_ROW_TRAILER.search(line):
            # Standalone trailer line (shouldn't occur, but be safe)
            out_lines.append(_render_top_ops_row(line, baseline_ms))
        else:
            out_lines.append(line)
        i += 1
    return "\n".join(out_lines)

_RENDERERS = {
    "p_item": _render_p_item,
    "detail_estimate": _render_detail_estimate,
    "impact_summary": _render_impact_summary,
    "top_ops": _render_top_ops,
}


def _render_legacy(
    kind: str,
    attrs: Dict[str, Optional[str]],
    body: str,
    baseline_ms: float,
) -> Optional[str]:
    """Dispatch to the per-kind renderer. Returns ``None`` for unknown kinds
    (the walker leaves such blocks untouched, including their markers).
    """
    renderer = _RENDERERS.get(kind)
    if renderer is None:
        return None
    return renderer(attrs, body, baseline_ms)


# --- Walker --------------------------------------------------------------


def _rehydrate_one_file(path: str, output_dir: str, baseline_ms: float) -> str:
    """Walk markers in ``path`` and splice rendered legacy text in place of
    each begin..end block. Returns a status string.
    """
    with open(path, "r") as f:
        original = f.read()

    out_parts: List[str] = []
    cursor = 0
    rewrote_any = False

    for begin in _RE_MARKER_BEGIN.finditer(original):
        if begin.start() < cursor:
            continue  # nested marker swallowed by the previous block
        end = _RE_MARKER_END.search(original, begin.end())
        if end is None:
            # Malformed: no closing marker. Stop processing further markers.
            break
        attrs = _parse_attrs(begin.group(1))
        kind = attrs.pop("kind", None) or ""
        body = original[begin.end():end.start()]
        # Strip a single leading/trailing newline so renderers operate on
        # the inner content without surrounding blank lines from the marker.
        body_stripped = body.strip("\n")
        rendered = _render_legacy(kind, attrs, body_stripped, baseline_ms)
        if rendered is None:
            continue  # leave unknown kinds + their markers untouched
        out_parts.append(original[cursor:begin.start()])
        out_parts.append(rendered)
        cursor = end.end()
        rewrote_any = True

    if not rewrote_any:
        return "skipped_no_match"

    out_parts.append(original[cursor:])
    text = "".join(out_parts)
    if text == original:
        return "skipped_no_match"

    with open(path, "w") as f:
        f.write(text)
    return "rewritten"


def rehydrate_reports_to_ms(output_dir: str) -> Dict[str, str]:
    """Rewrite all standalone-analysis markdown to legacy ms-savings format.

    Reads ``baseline_ms`` from ``category_data/category_manifest.json`` (used
    to convert ``impact_score`` percentages back to milliseconds at the marker
    boundary). Walks ``standalone_analysis.md``, ``category_findings/*.md``,
    and ``system_findings/*.md``. Idempotent: rehydrated files have no
    remaining markers, so a second pass is a silent no-op.

    Returns a per-file status dict:
        {<path>: "rewritten" | "skipped_no_match" | "error: ..."}
    """
    baseline_ms = _load_baseline_ms(output_dir)
    if baseline_ms <= 0:
        print(
            f"[agent_extension.rehydrate_reports_to_ms] baseline_ms={baseline_ms!r} "
            f"in {output_dir}; cannot convert impact_score back to ms.",
            file=sys.stderr,
        )
        return {}

    targets: List[str] = []
    standalone = os.path.join(output_dir, "standalone_analysis.md")
    if os.path.isfile(standalone):
        targets.append(standalone)
    targets.extend(sorted(glob.glob(os.path.join(output_dir, "category_findings", "*.md"))))
    targets.extend(sorted(glob.glob(os.path.join(output_dir, "system_findings", "*.md"))))

    results: Dict[str, str] = {}
    for path in targets:
        try:
            results[path] = _rehydrate_one_file(path, output_dir, baseline_ms)
        except Exception as e:
            results[path] = f"error: {e}"

    n_rewritten = sum(1 for v in results.values() if v == "rewritten")
    n_skipped = len(results) - n_rewritten
    print(
        f"Rehydration complete: {n_rewritten} file(s) rewritten, "
        f"{n_skipped} skipped (no markers or already rehydrated)."
    )
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Detailed impact-savings plot + markdown rehydration "
            "(reverses the impact_score convention to pre-Phase-1 ms savings)."
        )
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Standalone analysis output directory",
    )
    parser.add_argument(
        "--title",
        required=True,
        help="Plot suptitle",
    )
    args = parser.parse_args()

    generate_impact_savings_plot(
        args.output_dir,
        args.title,
        output_filename="perf_improvement.png",
        write_base64=True,
    )
    rehydrate_reports_to_ms(args.output_dir)


if __name__ == "__main__":
    main()
