###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Optional standalone-analysis extension: detailed plot + ms-savings rehydration.

Hosts the legacy 2-panel cumulative-savings chart (cumulative stacked E2E bars
+ throughput cone with the 75-100% roofline band) and an idempotent rewriter
that converts all post-Phase-1 ``impact_score``-format markdown reports back
to the pre-Phase-1 ``ms savings (% of E2E)`` format.

CLI usage:

    python TraceLens/AgenticMode/Standalone/utils/agent_extension.py \\
        --output-dir <dir> \\
        --title '<Model> on <Platform> -- Kernel Tuning Potential'

Both ``generate_impact_savings_plot`` and ``rehydrate_reports_to_ms`` always
run when invoked via the CLI -- there are no opt-out flags. To disable, delete
or rename this file; the orchestrator skill auto-detects its presence.

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
from typing import Dict, List, Optional, Tuple

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
    output_filename: str = "impact_savings.png",
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
# Markdown rehydration: post-Phase-1 impact_score -> pre-Phase-1 ms savings
# ---------------------------------------------------------------------------


# Pattern A: P-item card Impact line  ("**Impact**: impact_score: X.X")
_RE_PITEM_IMPACT = re.compile(
    r"^(\*\*Impact\*\*:\s*)\[?impact_score:\s*([\d.]+)\]?\s*$",
    re.MULTILINE,
)
# Broader walker: matches both the quantifiable form above AND the
# "Not quantifiable from trace data" form. Used so we can advance the
# priority_data category cursor in lock-step with P-item rank order even
# across non-quantifiable cards (e.g. Pn = customcollective).
_RE_PITEM_IMPACT_ANY = re.compile(
    r"^\*\*Impact\*\*:\s*"
    r"(?:\[?impact_score:\s*[\d.]+\]?|Not quantifiable from trace data)\s*$",
    re.MULTILINE,
)

# Pattern B: Detailed Analysis low/high bullets
_RE_DETAIL_LOW = re.compile(
    r"^(?P<indent>[ \t-]*)Low end impact_score \(75% roofline target\):\s*([\d.]+)\s*$",
    re.MULTILINE,
)
_RE_DETAIL_HIGH = re.compile(
    r"^(?P<indent>[ \t-]*)High end impact_score \(100% roofline target\):\s*([\d.]+)\s*$",
    re.MULTILINE,
)

# Pattern C/D: ## Impact Summary canonical 4-column header
_RE_IMPACT_SUMMARY_HEADER = re.compile(
    r"^\| Recommendation \| Type \| impact_score \| Confidence \|\s*\n"
    r"^\|[\-+ ]+\|[\-+ ]+\|[\-+ ]+\|[\-+ ]+\|\s*$",
    re.MULTILINE,
)

# Pattern E: Top Operations table (5-column post-Phase-1 form)
_RE_TOP_OPS_HEADER = re.compile(
    r"^\| Rank \| Category \| Time \(ms\) \| % of Compute Time \| Ops \|\s*\n"
    r"^\|[\-+ ]+\|[\-+ ]+\|[\-+ ]+\|[\-+ ]+\|[\-+ ]+\|\s*$",
    re.MULTILINE,
)

# Pattern F: fusion methodology blurb (post-Phase-1 wording)
_RE_FUSION_BLURB = re.compile(
    r"impact_score projections use a roofline projection model "
    r"\(75-100% of peak\) with 85% memory/compute pipeline overlap\. "
    r"Kernels without perf models use their measured trace time as-is\. "
    r"(?:Candidates where fewer than 75% of kernels have perf models are not reported\. "
    r"Each finding shows both a \*\*Confidence\*\* \(fusion pattern quality\) and perf model coverage in the \*\*Impact\*\* line\. )?"
    r"Actual recoverable time depends on implementation feasibility and interaction effects\."
)

# Detection regex: any pre-Phase-1 ms-form already present?
_RE_ALREADY_MS = re.compile(
    r"~?\d+(?:\.\d+)?\s*[-\u2013\u2014]\s*\d+(?:\.\d+)?\s*ms\s*savings",
)


def _load_baseline_ms(output_dir: str) -> float:
    """Read baseline_ms from category_manifest.json. Returns 0 if missing."""
    try:
        manifest = load_manifest(output_dir)
    except FileNotFoundError:
        return 0.0
    return float(manifest.get("gpu_utilization", {}).get("total_time_ms", 0) or 0)


def _load_metadata_for_category(output_dir: str, category: str) -> dict:
    """Read metadata/<category>_metadata.json. Returns {} if missing."""
    path = os.path.join(output_dir, "metadata", f"{category}_metadata.json")
    if not os.path.isfile(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def _load_priority_categories(output_dir: str) -> List[str]:
    """Return ``[category, ...]`` from ``priority_data.json::priorities[]``
    in P1..PN rank order. Includes non-quantifiable entries so the index
    stays aligned with the order of P-item cards in standalone_analysis.md.
    Returns ``[]`` if the file is missing or malformed.
    """
    path = os.path.join(output_dir, "priority_data.json")
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []
    return [p.get("category", "") for p in data.get("priorities", [])]


def _impact_score_range_for_file(output_dir: str, filename: str) -> List[Tuple[float, float, float]]:
    """Return list of (low, mid, high) impact_score tuples for a findings file's category.

    File path -> category mapping: ``<cat>_findings.md`` -> ``<cat>``.
    Returns one tuple per ``impact_estimates`` entry in the metadata, in order.
    """
    base = os.path.basename(filename)
    if not base.endswith("_findings.md"):
        return []
    category = base[: -len("_findings.md")]
    meta = _load_metadata_for_category(output_dir, category)
    triples: List[Tuple[float, float, float]] = []
    for entry in meta.get("impact_estimates", []):
        if not entry.get("quantifiable", True):
            triples.append((0.0, 0.0, 0.0))
            continue
        triples.append(
            (
                float(entry.get("impact_score_low", 0) or 0),
                float(entry.get("impact_score", 0) or 0),
                float(entry.get("impact_score_high", 0) or 0),
            )
        )
    return triples


def _extract_pairs_from_detail_blocks(text: str) -> List[Tuple[float, float]]:
    """Walk the markdown and pair Low/High impact_score bullets in file order.

    Returns ``[(low, high), ...]`` in the order they appear.
    """
    lows = [(m.start(), float(m.group(2))) for m in _RE_DETAIL_LOW.finditer(text)]
    highs = [(m.start(), float(m.group(2))) for m in _RE_DETAIL_HIGH.finditer(text)]
    pairs: List[Tuple[float, float]] = []
    for (_, lo), (_, hi) in zip(lows, highs):
        pairs.append((lo, hi))
    return pairs


def _rehydrate_pattern_b(text: str, baseline_ms: float) -> str:
    """Rewrite Detailed Analysis Low/High bullets to ms+E2E% form."""

    def _rewrite_low(m: re.Match) -> str:
        indent = m.group(1)
        pct = float(m.group(2))
        ms = pct * baseline_ms / 100.0
        return f"{indent}Low end (75% roofline): {ms:.3f} ms savings ({pct:.2f}% E2E)"

    def _rewrite_high(m: re.Match) -> str:
        indent = m.group(1)
        pct = float(m.group(2))
        ms = pct * baseline_ms / 100.0
        return f"{indent}High end (100% roofline): {ms:.3f} ms savings ({pct:.2f}% E2E)"

    text = _RE_DETAIL_LOW.sub(_rewrite_low, text)
    text = _RE_DETAIL_HIGH.sub(_rewrite_high, text)
    return text


def _rehydrate_pattern_a(
    text: str,
    baseline_ms: float,
    pair_pool: List[Tuple[float, float]],
    category_pool: Optional[List[str]] = None,
) -> str:
    """Rewrite each P-item Impact line, consuming low/high pairs in order.

    For each ``**Impact**: impact_score: X.X`` line, pop the next ``(low, high)``
    pair (in markdown order) and emit ``**Impact**: ~<low_ms>-<high_ms> ms
    savings (<low>-<high>% of E2E)``. If the pair pool is exhausted, fall back
    to using only the matched mid value (rendered as a single number).

    When ``category_pool`` is provided (orchestrator-assembled
    ``standalone_analysis.md`` only), additionally append the legacy trailing
    prose ``from closing efficiency gaps to 75-100% of roofline (pre-computed
    from `<category>_metrics.json` impact_estimates).`` for quantifiable
    P-items, popping one category per P-item card encountered (including
    non-quantifiable cards) so the cursor stays aligned with rank order.
    """
    pair_idx = [0]
    cat_idx = [0]
    pairs = list(pair_pool)
    cats = list(category_pool) if category_pool is not None else None

    def _rewrite(m: re.Match) -> str:
        line = m.group(0)
        # Non-quantifiable cards have nothing to rewrite, but we still need to
        # advance the priority cursor so the next quantifiable card maps to
        # the correct category.
        if "Not quantifiable" in line:
            if cats is not None:
                cat_idx[0] += 1
            return line

        prefix_match = re.match(
            r"^(\*\*Impact\*\*:\s*)\[?impact_score:\s*([\d.]+)\]?\s*$", line
        )
        if prefix_match is None:
            return line
        prefix = prefix_match.group(1)
        mid_pct = float(prefix_match.group(2))

        if pair_idx[0] < len(pairs):
            lo_pct, hi_pct = pairs[pair_idx[0]]
            pair_idx[0] += 1
        else:
            lo_pct, hi_pct = mid_pct, mid_pct
        lo_ms = lo_pct * baseline_ms / 100.0
        hi_ms = hi_pct * baseline_ms / 100.0
        body = (
            f"{prefix}~{lo_ms:.1f}\u2013{hi_ms:.1f} ms savings "
            f"({lo_pct:.1f}\u2013{hi_pct:.1f}% of E2E)"
        )

        if cats is not None:
            if cat_idx[0] < len(cats) and cats[cat_idx[0]]:
                category = cats[cat_idx[0]]
                body += (
                    f" from closing efficiency gaps to 75\u2013100% of roofline "
                    f"(pre-computed from `{category}_metrics.json` impact_estimates)."
                )
            cat_idx[0] += 1

        return body

    return _RE_PITEM_IMPACT_ANY.sub(_rewrite, text)


def _rehydrate_pattern_cd(text: str) -> str:
    """Restore the legacy 5-column ``## Impact Summary`` header.

    Header-only swap: data rows (kernel-fusion) are NOT cell-expanded. Their
    ``impact_score`` cell stays as a single number, so the row will visually
    have one column fewer than the 5-column header. This is acceptable per the
    plan -- agents emit data rows infrequently and the column meaning is
    self-evident.
    """
    replacement = (
        "| Recommendation | Type | Estimated Savings (ms) | "
        "Estimated Improvement (E2E %) | Confidence |\n"
        "|---------------|------|----------------------|"
        "-------------------------------|------------|"
    )
    return _RE_IMPACT_SUMMARY_HEADER.sub(replacement, text)


def _rehydrate_pattern_e(
    text: str,
    output_dir: str,
    baseline_ms: float,
) -> str:
    """Restore the dropped ``Potential improvement (time, E2E %)`` column on the
    Top Operations table inside ``standalone_analysis.md``.

    Strategy: find the 5-column header, locate the contiguous data rows
    immediately after, look up each row's category in metadata, and append the
    opportunity column (or ``--`` if no quantifiable rollup exists).
    """
    m = _RE_TOP_OPS_HEADER.search(text)
    if not m:
        return text

    header_start = m.start()
    header_end = m.end()
    new_header = (
        "| Rank | Category | Time (ms) | % of Compute Time | Ops | "
        "Potential improvement (time, E2E %) |\n"
        "|------|----------|-----------|-------------------|-----|"
        "-------------------------------------|"
    )

    rest = text[header_end:]
    lines = rest.split("\n")
    body_lines: List[str] = []
    consumed = 0
    for line in lines:
        if not line.strip():
            # A blank line *before* any body row is the separator-trailing
            # newline; a blank line *after* body rows ends the table.
            if body_lines:
                break
            consumed += 1
            continue
        if line.startswith("|") and line.rstrip().endswith("|"):
            body_lines.append(line)
            consumed += 1
        else:
            break

    rewritten_rows: List[str] = []
    for row in body_lines:
        cells = [c.strip() for c in row.strip("|").split("|")]
        if len(cells) < 2:
            rewritten_rows.append(row)
            continue
        # Cell may carry a parenthetical backend hint (e.g. "MoE Fused (AITER)")
        # that wasn't part of the legacy ORI category name. Strip it before
        # mapping to the on-disk metadata filename.
        raw_cat = re.sub(r"\s*\([^)]*\)\s*", "", cells[1]).strip()
        category = raw_cat.lower().replace(" ", "_")
        meta = _load_metadata_for_category(output_dir, category)
        opportunity = "--"
        for entry in meta.get("impact_estimates", []):
            if not entry.get("quantifiable", True):
                continue
            lo_pct = float(entry.get("impact_score_low", 0) or 0)
            hi_pct = float(entry.get("impact_score_high", 0) or 0)
            if hi_pct <= 0:
                continue
            lo_ms = lo_pct * baseline_ms / 100.0
            hi_ms = hi_pct * baseline_ms / 100.0
            opportunity = (
                f"~{lo_ms:.1f}\u2013{hi_ms:.1f} ms "
                f"({lo_pct:.1f}\u2013{hi_pct:.1f}%)"
            )
            break
        rewritten_rows.append(row.rstrip() + f" {opportunity} |")

    suffix_after_rows = "\n".join(lines[consumed:])
    new_body = "\n".join(rewritten_rows)
    if new_body:
        new_table = new_header + "\n" + new_body
    else:
        new_table = new_header

    return text[:header_start] + new_table + ("\n" + suffix_after_rows if suffix_after_rows else "")


def _rehydrate_pattern_f(text: str) -> str:
    """Replace post-Phase-1 fusion methodology wording with the legacy form."""
    legacy_long = (
        "Savings projections use a roofline projection model (75-100% of peak) "
        "with 85% memory/compute pipeline overlap. Kernels without perf models "
        "use their measured trace time as-is. Candidates where fewer than 75% "
        "of kernels have perf models are not reported. Each finding shows both "
        "a **Confidence** (fusion pattern quality) and perf model coverage in "
        "the **Impact** line. Actual savings depend on implementation "
        "feasibility and interaction effects."
    )
    legacy_short = (
        "Savings projections use a roofline projection model (75-100% of peak) "
        "with 85% memory/compute pipeline overlap. Kernels without perf models "
        "use their measured trace time as-is. Actual savings depend on "
        "implementation feasibility and interaction effects."
    )

    def _pick(m: re.Match) -> str:
        return legacy_long if "Candidates where fewer than 75%" in m.group(0) else legacy_short

    return _RE_FUSION_BLURB.sub(_pick, text)


def _rehydrate_one_file(path: str, output_dir: str, baseline_ms: float) -> str:
    """Rehydrate one markdown file in place. Returns a status string."""
    with open(path, "r") as f:
        original = f.read()

    has_a = bool(_RE_PITEM_IMPACT.search(original))
    has_b = bool(_RE_DETAIL_LOW.search(original)) or bool(
        _RE_DETAIL_HIGH.search(original)
    )
    has_cd = bool(_RE_IMPACT_SUMMARY_HEADER.search(original))
    has_e = (
        os.path.basename(path) == "standalone_analysis.md"
        and bool(_RE_TOP_OPS_HEADER.search(original))
    )
    has_f = bool(_RE_FUSION_BLURB.search(original))

    if not (has_a or has_b or has_cd or has_e or has_f):
        if _RE_ALREADY_MS.search(original):
            return "skipped_already_ms"
        return "skipped_no_match"

    text = original

    if has_b:
        text = _rehydrate_pattern_b(text, baseline_ms)

    if has_a:
        # Use values present in this file's Detailed Analysis bullets first
        # (already converted by pattern B above? -- pull from the original
        # text instead so we read raw impact_score values, not rewritten ms).
        pair_pool = _extract_pairs_from_detail_blocks(original)
        if not pair_pool:
            triples = _impact_score_range_for_file(output_dir, path)
            pair_pool = [(lo, hi) for lo, _, hi in triples]
        # Only the orchestrator-assembled standalone report carries the
        # trailing "...pre-computed from <cat>_metrics.json impact_estimates."
        # prose; per-category and system findings keep the short form.
        cat_pool: Optional[List[str]] = None
        if os.path.basename(path) == "standalone_analysis.md":
            cat_pool = _load_priority_categories(output_dir)
        text = _rehydrate_pattern_a(text, baseline_ms, pair_pool, cat_pool)

    if has_cd:
        text = _rehydrate_pattern_cd(text)

    if has_e:
        text = _rehydrate_pattern_e(text, output_dir, baseline_ms)

    if has_f:
        text = _rehydrate_pattern_f(text)

    if text == original:
        return "skipped_no_match"

    with open(path, "w") as f:
        f.write(text)
    return "rewritten"


def rehydrate_reports_to_ms(output_dir: str) -> Dict[str, str]:
    """Rewrite all standalone-analysis markdown to pre-Phase-1 ms-savings format.

    Reads ``baseline_ms`` from ``category_data/category_manifest.json`` and
    per-category ``impact_score_low/mid/high`` from ``metadata/<cat>_metadata.json``
    (used as a fallback when a findings file lacks Detailed Analysis bullets,
    and for the Top Operations table in ``standalone_analysis.md``).

    Walks ``standalone_analysis.md``, ``category_findings/*.md``, and
    ``system_findings/*.md``. Idempotent: files already in ms form are
    detected (``_RE_ALREADY_MS``) and skipped.

    Returns a per-file status dict:
        {<path>: "rewritten" | "skipped_already_ms" | "skipped_no_match"}
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
        f"{n_skipped} skipped (already ms or no match)."
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

    generate_impact_savings_plot(args.output_dir, args.title)
    rehydrate_reports_to_ms(args.output_dir)


if __name__ == "__main__":
    main()
