###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Plot generation and report embedding utilities for TraceLens AgenticMode.

Public API:

- ``generate_perf_plot`` — 2-panel chart: cumulative stacked E2E bars by
  kernel category + throughput cone with 75-100% roofline band.
- ``embed_plot_in_report`` — inject ``perf_improvement`` PNG into markdown.
- ``generate_and_embed_plot`` — end-to-end pipeline (plot_data → plot → embed).
"""

import base64
import json
import os
from TraceLens.AgenticMode.Standalone.category_analyses.analysis_utils import (
    generate_plot_data,
)
import re
from typing import Optional

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

__all__ = [
    "generate_perf_plot",
    "embed_plot_in_report",
    "generate_and_embed_plot",
]

_CAT_PALETTE = [
    "#e74c3c", "#e67e22", "#f1c40f", "#2ecc71",
    "#9b59b6", "#1abc9c", "#e84393", "#3498db",
]

def _short_name(name: str, max_len: int = 8) -> str:
    """Shorten a category name for plot labels. Truncates with ellipsis if needed."""
    if len(name) <= max_len:
        return name
    return name[:max_len - 1] + "\u2026"

_REST_KEY = "__rest_e2e__"


def _compute_cumulative_projections(baseline_ms, recommendations):
    """Accumulate per-step latency, savings, and throughput from recommendations."""
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
        sm = float(rec.get("savings_ms", 0))
        sl = float(rec.get("savings_ms_low", sm))
        sh = float(rec.get("savings_ms_high", sm))
        cum_mid += sm
        cum_lo += sl
        cum_hi += sh
        lat_mid = max(0.01, baseline_ms - cum_mid)
        lat_best = max(0.01, baseline_ms - cum_hi)
        lat_worst = max(0.01, baseline_ms - cum_lo)
        cnt = rec.get("operation_count", 1)
        steps.append(f"{rec['category']}\n({cnt} ops)")
        e2e_ms.append(lat_mid)
        savings.append(sm)
        rel.append(round(baseline_ms / lat_mid * 100, 1))
        err_lo.append(lat_mid - lat_best)
        err_hi.append(lat_worst - lat_mid)
        rel_lo.append(round(baseline_ms / lat_worst * 100 if lat_worst > 0 else 100.0, 1))
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


def _build_stacked_segments(manifest, recommendations, baseline_ms):
    """Build per-category segment data for the stacked bar chart."""
    plotted_cats = {r["category"] for r in recommendations}
    baseline_by_cat: dict[str, float] = {}
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
        [{"name": n, "time_ms": t} for n, t in baseline_by_cat.items()
         if n != _REST_KEY and n in plotted_cats],
        key=lambda x: x["time_ms"],
    )
    unanalyzed = sorted(
        [{"name": n, "time_ms": t} for n, t in baseline_by_cat.items()
         if n != _REST_KEY and n not in plotted_cats],
        key=lambda x: x["time_ms"],
    )
    segment_order = [x["name"] for x in analyzed + unanalyzed]
    if _REST_KEY in baseline_by_cat:
        segment_order.append(_REST_KEY)

    return baseline_by_cat, segment_order, plotted_cats


def _build_color_map(recommendations):
    """Map each recommendation's category to a consistent palette color."""
    cmap: dict[str, str] = {}
    for i, rec in enumerate(recommendations):
        cmap[rec["category"]] = _CAT_PALETTE[i % len(_CAT_PALETTE)]
    return cmap


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
        x_vals, proj["rel"], "o-", color="#2ecc71", linewidth=2.5,
        markersize=9, markerfacecolor="white", markeredgewidth=2.5, zorder=4,
    )
    last_idx = len(proj["rel"]) - 1
    ax.annotate(
        f"{proj['rel'][last_idx]:.0f}%",
        (last_idx, proj["rel"][last_idx]),
        textcoords="offset points", xytext=(0, -16),
        ha="center", va="top", fontsize=11, fontweight="bold",
        color="#1a7a3a", zorder=5,
    )
    ax.set_xticks(range(len(proj["steps"])))
    ax.set_xticklabels(proj["steps"], fontsize=9)
    ax.set_ylabel("% Relative Throughput (Baseline = 100)", fontsize=11)
    ax.set_title(
        "Cumulative Throughput Improvement\n(75\u2013100% roofline potential)",
        fontsize=12, fontweight="bold", pad=12,
    )
    ax.set_ylim(80, max(cum_hi) * 1.15)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.axhline(y=100, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=9)


def generate_perf_plot(
    output_dir: str,
    title: str,
    output_filename: str = "perf_improvement.png",
    write_base64: bool = True,
    show_error_bars: bool = True,
) -> bool:
    """
    Two-panel performance plot: cumulative stacked E2E bars + throughput cone.

    Left panel stacks E2E time by kernel category; each step reduces the
    optimized category's slice by its ``savings_ms``.  Right panel shows
    cumulative relative throughput with a 75-100% roofline band.

    Generates ``plot_data.json`` automatically if missing.

    Args:
        output_dir: Directory containing ``category_data/category_manifest.json``
            and (optionally) ``plot_data.json``.
        title: Figure suptitle.
        output_filename: PNG name under ``output_dir``.
        write_base64: Write ``perf_improvement_base64.txt`` for report embedding.
        show_error_bars: Show 75-100% roofline error caps on the left panel.

    Returns:
        True if the figure was written, False if inputs were missing or invalid.
    """
    plot_data_path = os.path.join(output_dir, "plot_data.json")
    if not os.path.exists(plot_data_path):
        try:
            generate_plot_data(output_dir)
        except Exception as e:
            print(f"plot_data generation failed: {e}")
            return False

    if not os.path.exists(plot_data_path):
        print(f"plot_data.json not found at {plot_data_path} - skipping plot")
        return False

    with open(plot_data_path, "r") as f:
        plot_data = json.load(f)

    baseline_ms = float(plot_data.get("baseline_ms", 0))
    recommendations = plot_data.get("recommendations", [])
    if not recommendations or baseline_ms <= 0:
        print("No kernel tuning recommendations or invalid baseline - skipping plot")
        return False

    manifest_path = os.path.join(output_dir, "category_data", "category_manifest.json")
    if not os.path.exists(manifest_path):
        print(f"category_manifest.json not found - skipping plot")
        return False

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    proj = _compute_cumulative_projections(baseline_ms, recommendations)
    baseline_by_cat, segment_order, plotted_cats = _build_stacked_segments(
        manifest, recommendations, baseline_ms,
    )
    cat_color_map = _build_color_map(recommendations)

    n_bars = len(proj["steps"])
    x = np.arange(n_bars, dtype=float)
    width = 0.62
    rest_label = "Rest of E2E"

    def _segment_heights(num_recs: int) -> dict[str, float]:
        savings_by_cat: dict[str, float] = {}
        for j in range(num_recs):
            r = recommendations[j]
            c = r["category"]
            sv = float(r.get("savings_ms", 0))
            if c in baseline_by_cat and c != _REST_KEY:
                savings_by_cat[c] = savings_by_cat.get(c, 0.0) + sv
            elif _REST_KEY in baseline_by_cat:
                savings_by_cat[_REST_KEY] = savings_by_cat.get(_REST_KEY, 0.0) + sv
        return {
            name: max(0.0, baseline_by_cat.get(name, 0.0) - savings_by_cat.get(name, 0.0))
            for name in segment_order
        }

    fig, (ax_stack, ax_cone) = plt.subplots(
        1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1.45, 1.0]},
    )

    # -- Left panel: stacked bars --
    bottom = np.zeros(n_bars, dtype=float)
    rest_color = "#aab7c4"
    for seg_name in segment_order:
        h = np.array([_segment_heights(k)[seg_name] for k in range(n_bars)], dtype=float)
        if np.all(h <= 0):
            continue
        color = rest_color if seg_name == _REST_KEY else cat_color_map.get(seg_name, "#bdc3c7")
        lbl = rest_label if seg_name == _REST_KEY else _short_name(seg_name)
        ax_stack.bar(
            x, h, width, bottom=bottom, label=lbl, color=color,
            edgecolor="white", linewidth=0.9, alpha=0.95,
        )
        bottom += h

    for k in range(n_bars):
        total_h = float(bottom[k])
        accent = "#333333"
        if k > 0:
            cat = recommendations[k - 1].get("category", "")
            accent = cat_color_map.get(cat, "#333333")
        label_y = total_h + (proj["err_hi"][k] + 1.2 if show_error_bars else 1.2)
        ax_stack.text(
            x[k], label_y, f"{proj['e2e_ms'][k]:.1f} ms",
            ha="center", va="bottom", fontsize=9, fontweight="bold", color=accent,
        )
        if k > 0 and proj["savings"][k] > 0:
            ax_stack.text(
                x[k], total_h * 0.5, f"-{proj['savings'][k]:.1f} ms",
                ha="center", va="center", fontsize=8, color=accent,
                fontweight="bold", zorder=5,
            )
        if show_error_bars:
            ax_stack.errorbar(
                [x[k]], [total_h],
                yerr=[[proj["err_lo"][k]], [proj["err_hi"][k]]],
                fmt="none", ecolor=accent, elinewidth=1.1, capsize=3, zorder=6,
            )

    ax_stack.set_xticks(x)
    ax_stack.set_xticklabels(proj["steps"], fontsize=8)
    ax_stack.set_ylabel("E2E time stacked by category (ms)", fontsize=11)
    ax_stack.set_title(
        "Projected E2E Latency (stacked by kernel category)",
        fontsize=11, fontweight="bold", pad=10,
    )
    ymax = float(np.max(bottom + proj["err_hi"]) + 8) if show_error_bars else float(np.max(bottom) + 8)
    ax_stack.set_ylim(0, ymax * 1.12)
    ax_stack.spines["top"].set_visible(False)
    ax_stack.spines["right"].set_visible(False)
    ax_stack.legend(
        loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=7.5,
        frameon=False, title="Operations",
    )

    fig.text(
        0.5, -0.02,
        "Gray bars represent categories without performance models — not reflected in savings projections.",
        ha="center", fontsize=8.5, color="#888888", style="italic",
    )

    # -- Right panel: throughput cone --
    _render_throughput_cone(ax_cone, proj)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    out_path = os.path.join(output_dir, output_filename)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Plot saved to {out_path}")

    if write_base64:
        with open(out_path, "rb") as f:
            b64_str = base64.b64encode(f.read()).decode("ascii")
        b64_path = os.path.join(output_dir, "perf_improvement_base64.txt")
        with open(b64_path, "w") as f:
            f.write(b64_str)
        print(f"Base64 written to {b64_path}")

    return True


def embed_plot_in_report(
    output_dir: str,
    report_filename: str = "standalone_analysis.md",
    placeholder: str = "{{PERF_PLOT}}",
) -> bool:
    """
    Replace the plot placeholder in the report with a base64-embedded PNG data URI.

    Reads perf_improvement_base64.txt and substitutes the placeholder in the
    report file.  If the base64 file is missing, the placeholder is removed so
    the report remains clean.

    Args:
        output_dir: Base output directory containing the report and base64 file
        report_filename: Name of the markdown report file
        placeholder: Placeholder string to replace

    Returns:
        True if the plot was embedded, False otherwise
    """
    report_path = os.path.join(output_dir, report_filename)
    b64_path = os.path.join(output_dir, "perf_improvement_base64.txt")

    if not os.path.exists(report_path):
        print(f"Report file not found at {report_path} - skipping embed")
        return False

    with open(report_path, "r") as f:
        report = f.read()

    if os.path.exists(b64_path):
        with open(b64_path, "r") as f:
            b64_str = f.read().strip()
        img_tag = f"![Performance Improvement](data:image/png;base64,{b64_str})"
        embedded = True
    else:
        img_tag = ""
        embedded = False

    report = report.replace(placeholder, img_tag)
    if embedded and placeholder not in report:
        report = re.sub(
            r"!\[Performance Improvement\]\(data:image/png;base64,[A-Za-z0-9+/=]+\)",
            img_tag,
            report,
            count=1,
        )

    with open(report_path, "w") as f:
        f.write(report)

    return embedded


def generate_and_embed_plot(output_dir: str, title: str) -> dict:
    """End-to-end pipeline: generate plot data, render the plot, and embed it.

    Args:
        output_dir: Base output directory containing category_data/ and the report
        title: Plot suptitle (e.g. '<Model> on <Platform> — Kernel Tuning Potential')

    Returns:
        Dict with boolean status for each stage: plot_data, plot, embed
    """
    results = {"plot_data": False, "plot": False, "embed": False}

    results["plot"] = generate_perf_plot(output_dir, title)
    if results["plot"]:
        results["plot_data"] = True
        results["embed"] = embed_plot_in_report(output_dir)

    return results
