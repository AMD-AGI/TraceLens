###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Plot generation and report embedding utilities for TraceLens AgenticMode.

Public API (see ``__all__``):

- ``generate_perf_plot`` — 3-panel chart (breakdown + latency bars + throughput cone).
- ``generate_perf_plot_stacked_projection`` — 2-panel variant: cumulative stacked E2E
  bars by kernel category + throughput cone; optional ``show_error_bars``.
- ``embed_plot_in_report`` — inject ``perf_improvement`` PNG into markdown.
- ``generate_and_embed_plot`` — ``generate_plot_data`` → ``generate_perf_plot_stacked_projection`` → embed.

When merging from other branches, preserve ``generate_perf_plot_stacked_projection``
and its helpers unless intentionally replacing this visualization path.
"""

import base64
import json
import os
from TraceLens.AgenticMode.Standalone.category_analyses.analysis_utils import (
    generate_plot_data,
)
import re

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

__all__ = [
    "generate_perf_plot",
    "generate_perf_plot_stacked_projection",
    "embed_plot_in_report",
    "generate_and_embed_plot",
]


def generate_perf_plot(output_dir: str, title: str) -> bool:
    """
    Generate the performance improvement plot from plot_data.json.

    Reads plot_data.json (produced by generate_plot_data), computes cumulative
    projections using the 75–100% roofline potential improvement range
    (savings_ms_low / savings_ms_high per recommendation), renders a three-panel
    chart (stacked time breakdown + latency bars + throughput cone), and saves
    both the PNG image and a base64-encoded text file for report embedding.
    """
    plot_data_path = os.path.join(output_dir, "plot_data.json")
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

    cum_savings_mid = 0.0
    cum_savings_low = 0.0
    cum_savings_high = 0.0

    steps = ["Baseline"]
    e2e_ms = [baseline_ms]
    savings_list = [0]
    cumulative_rel = [100.0]
    err_lower = [0.0]
    err_upper = [0.0]
    cum_rel_low = [100.0]
    cum_rel_high = [100.0]

    for rec in recommendations:
        sav_mid = float(rec.get("savings_ms", 0))
        sav_lo = float(rec.get("savings_ms_low", sav_mid))
        sav_hi = float(rec.get("savings_ms_high", sav_mid))
        cum_savings_mid += sav_mid
        cum_savings_low += sav_lo
        cum_savings_high += sav_hi
        latency_mid = max(0.01, baseline_ms - cum_savings_mid)
        latency_best = max(0.01, baseline_ms - cum_savings_high)
        latency_worst = max(0.01, baseline_ms - cum_savings_low)
        count = rec.get("operation_count", 1)
        steps.append(f"{rec['category']}\n({count} ops)")
        e2e_ms.append(latency_mid)
        savings_list.append(sav_mid)
        cumulative_rel.append(round(baseline_ms / latency_mid * 100, 1))
        err_lower.append(latency_mid - latency_best)
        err_upper.append(latency_worst - latency_mid)
        rel_lo = baseline_ms / latency_worst * 100 if latency_worst > 0 else 100.0
        rel_hi = baseline_ms / latency_best * 100 if latency_best > 0 else 100.0
        cum_rel_low.append(round(rel_lo, 1))
        cum_rel_high.append(round(rel_hi, 1))

    e2e_ms = np.array(e2e_ms)
    err_lower = np.array(err_lower)
    err_upper = np.array(err_upper)
    yerr_rest = np.array([err_lower[1:], err_upper[1:]])

    # -- Build stacked bar data from manifest --
    manifest_path = os.path.join(output_dir, "category_data", "category_manifest.json")
    plotted_cats = {r["category"] for r in recommendations}
    stacked_segments = []
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        compute_time_ms = (
            manifest["gpu_utilization"]["total_time_ms"]
            * manifest["gpu_utilization"]["computation_time_percent"]
            / 100
        )
        analyzed = []
        unanalyzed = []
        for cat in manifest["categories"]:
            if cat["tier"] != "compute_kernel":
                continue
            gpu_time = cat.get("gpu_kernel_time_ms", 0)
            if gpu_time <= 0:
                continue
            entry = {
                "name": cat["name"],
                "time_ms": gpu_time,
                "pct": gpu_time / compute_time_ms * 100 if compute_time_ms > 0 else 0,
                "analyzed": cat["name"] in plotted_cats,
            }
            if entry["analyzed"]:
                analyzed.append(entry)
            else:
                unanalyzed.append(entry)
        analyzed.sort(key=lambda x: x["time_ms"])
        unanalyzed.sort(key=lambda x: x["time_ms"])
        stacked_segments = analyzed + unanalyzed

    # -- Shared color map so stacked bar and latency bars match --
    cat_palette = [
        "#e74c3c", "#e67e22", "#f1c40f", "#2ecc71",
        "#9b59b6", "#1abc9c", "#e84393", "#3498db",
    ]
    cat_color_map = {}
    for i, rec in enumerate(recommendations):
        cat_color_map[rec["category"]] = cat_palette[i % len(cat_palette)]

    fig, (ax0, ax1, ax2) = plt.subplots(
        1, 3, figsize=(19, 5.5), gridspec_kw={"width_ratios": [0.22, 1.1, 1]}
    )

    # -- Panel 1: stacked time breakdown --
    short_names = {
        "elementwise": "elem",
        "unaryelementwise": "unary",
        "convolution": "conv",
        "reduce": "red",
    }
    if stacked_segments:
        bottom = 0
        bar_x, bar_w = 0, 0.45
        for seg in stacked_segments:
            if seg["analyzed"]:
                c = cat_color_map.get(seg["name"], "#3498db")
                alpha = 1.0
            else:
                c = "#bdc3c7"
                alpha = 0.7
            ax0.bar(
                bar_x, seg["time_ms"], bottom=bottom, color=c,
                edgecolor="white", linewidth=0.8, width=bar_w, alpha=alpha,
            )
            mid_y = bottom + seg["time_ms"] / 2
            if seg["pct"] >= 5:
                name = short_names.get(seg["name"], seg["name"])
                pct_str = f"{seg['pct']:.0f}%" if seg["pct"] >= 1 else "<1%"
                fw = "bold" if seg["analyzed"] else "normal"
                fc = "#1a1a1a" if seg["analyzed"] else "#888888"
                ax0.annotate(
                    f"{name} {pct_str}",
                    xy=(bar_x + bar_w / 2, mid_y),
                    xytext=(bar_x + bar_w / 2 + 0.3, mid_y),
                    ha="left", va="center", fontsize=7.5, fontweight=fw, color=fc,
                    arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.6),
                )
            bottom += seg["time_ms"]
    ax0.set_xlim(-0.4, 1.5)
    ax0.set_xticks([])
    ax0.set_ylabel("GPU Kernel Time (ms)", fontsize=10)
    ax0.set_title("Time\nBreakdown", fontsize=10, fontweight="bold", pad=12)
    ax0.spines["top"].set_visible(False)
    ax0.spines["right"].set_visible(False)
    ax0.spines["bottom"].set_visible(False)

    # -- Panel 2: projected E2E latency bars --
    colors = ["#4a90d9"] + [
        cat_color_map.get(r["category"], "#888888") for r in recommendations
    ]
    bar0 = ax1.bar(
        steps[0:1], e2e_ms[0:1], color=colors[0],
        edgecolor="white", linewidth=1.2, width=0.65,
    )
    bars_rest = ax1.bar(
        steps[1:], e2e_ms[1:], color=colors[1:],
        edgecolor="white", linewidth=1.2, width=0.65,
        yerr=yerr_rest, capsize=4, error_kw=dict(ecolor="#333333", linewidth=1.2),
    )
    bars = list(bar0) + list(bars_rest)
    y_max_for_label = e2e_ms + err_upper
    for i, (bar, val, sav) in enumerate(zip(bars, e2e_ms, savings_list)):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            y_max_for_label[i] + 1.5,
            f"{val:.1f} ms",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )
        if sav > 0:
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() / 2,
                f"-{sav:.1f} ms",
                ha="center", va="center", fontsize=9, color="white", fontweight="bold",
            )
    ax1.set_ylabel("E2E Latency (ms)", fontsize=11)
    ax1.set_title(
        "Projected E2E Latency After Each Optimization\n(75\u2013100% roofline potential)",
        fontsize=12, fontweight="bold", pad=12,
    )
    ax1.set_ylim(0, (np.max(y_max_for_label) + 50) * 1.15)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.tick_params(axis="x", labelsize=9)

    # -- Panel 3: cumulative throughput cone --
    x_vals = np.arange(len(steps))
    cum_arr = np.array(cumulative_rel, dtype=float)
    cum_low = np.array(cum_rel_low, dtype=float)
    cum_high = np.array(cum_rel_high, dtype=float)
    cum_low[0] = cum_high[0] = cum_arr[0]

    ax2.fill_between(x_vals, cum_low, cum_high, color="#2ecc71", alpha=0.15, zorder=1)
    ax2.plot(x_vals, cum_high, "-", color="#27ae60", linewidth=1.0, alpha=0.4, zorder=2)
    ax2.plot(x_vals, cum_low, "-", color="#27ae60", linewidth=1.0, alpha=0.4, zorder=2)
    ax2.plot(
        x_vals, cumulative_rel, "o-", color="#2ecc71", linewidth=2.5,
        markersize=9, markerfacecolor="white", markeredgewidth=2.5, zorder=4,
    )

    last_idx = len(cumulative_rel) - 1
    ax2.annotate(
        f"{cumulative_rel[last_idx]:.0f}%",
        (last_idx, cumulative_rel[last_idx]),
        textcoords="offset points", xytext=(0, -16),
        ha="center", va="top", fontsize=11, fontweight="bold",
        color="#1a7a3a", zorder=5,
    )

    ax2.set_xticks(range(len(steps)))
    ax2.set_xticklabels(steps, fontsize=9)
    ax2.set_ylabel("% Relative Throughput (Baseline = 100)", fontsize=11)
    ax2.set_title(
        "Cumulative Throughput Improvement\n(75\u2013100% roofline potential)",
        fontsize=12, fontweight="bold", pad=12,
    )
    ax2.set_ylim(80, max(cum_high) * 1.15)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax2.axhline(y=100, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax2.grid(axis="y", linestyle="--", alpha=0.3)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.tick_params(axis="x", labelsize=9)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    output_path = os.path.join(output_dir, "perf_improvement.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Plot saved to {output_path}")

    with open(output_path, "rb") as f:
        b64_str = base64.b64encode(f.read()).decode("ascii")
    b64_path = os.path.join(output_dir, "perf_improvement_base64.txt")
    with open(b64_path, "w") as f:
        f.write(b64_str)
    print(f"Base64 written to {b64_path}")

    return True


def generate_perf_plot_stacked_projection(
    output_dir: str,
    title: str,
    output_filename: str = "perf_improvement_stacked.png",
    write_base64: bool = False,
    show_error_bars: bool = True,
) -> bool:
    """
    Two-panel performance plot: cumulative stacked E2E bars + throughput cone.

    Merges the former \"time breakdown\" and \"projected E2E latency\" views into
    one stacked-bar chart:

    - **Baseline**: total bar height = ``baseline_ms``, stacked by compute-kernel
      category (from ``category_manifest.json``) plus a **Rest of E2E** segment so
      segment heights sum to the full trace time.
    - **After each recommendation** (in order): only that recommendation's
      category slice is reduced by its ``savings_ms`` (cumulative per category);
      all other category slices stay at their baseline ms; **Rest of E2E** is
      unchanged. Bar height matches ``baseline_ms - cumulative_savings``.

    The right panel matches ``generate_perf_plot`` (cumulative relative throughput
    with 75–100% roofline band).

    Args:
        output_dir: Directory containing ``plot_data.json`` and
            ``category_data/category_manifest.json``.
        title: Figure suptitle.
        output_filename: PNG name under ``output_dir`` (default
            ``perf_improvement_stacked.png``).
        write_base64: If True, also writes ``perf_improvement_base64.txt`` (same as
            ``generate_perf_plot`` for ``embed_plot_in_report``).
        show_error_bars: If False, omits 75–100% roofline error caps on the left
            panel (throughput cone uncertainty unchanged).

    Returns:
        True if the figure was written, False if inputs were missing or invalid.
    """
    plot_data_path = os.path.join(output_dir, "plot_data.json")
    if not os.path.exists(plot_data_path):
        print(f"plot_data.json not found at {plot_data_path} - skipping stacked plot")
        return False

    with open(plot_data_path, "r") as f:
        plot_data = json.load(f)

    baseline_ms = float(plot_data.get("baseline_ms", 0))
    recommendations = plot_data.get("recommendations", [])

    if not recommendations or baseline_ms <= 0:
        print("No kernel tuning recommendations or invalid baseline - skipping stacked plot")
        return False

    manifest_path = os.path.join(output_dir, "category_data", "category_manifest.json")
    if not os.path.exists(manifest_path):
        print(f"category_manifest.json not found at {manifest_path} - skipping stacked plot")
        return False

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    # -- Cumulative latency + throughput (same as generate_perf_plot) --
    cum_savings_mid = cum_savings_low = cum_savings_high = 0.0
    steps = ["Baseline"]
    e2e_ms_list = [baseline_ms]
    savings_list = [0]
    cumulative_rel = [100.0]
    err_lower = [0.0]
    err_upper = [0.0]
    cum_rel_low = [100.0]
    cum_rel_high = [100.0]

    for rec in recommendations:
        sm = float(rec.get("savings_ms", 0))
        sl = float(rec.get("savings_ms_low", sm))
        sh = float(rec.get("savings_ms_high", sm))
        cum_savings_mid += sm
        cum_savings_low += sl
        cum_savings_high += sh
        lat_mid = max(0.01, baseline_ms - cum_savings_mid)
        lat_best = max(0.01, baseline_ms - cum_savings_high)
        lat_worst = max(0.01, baseline_ms - cum_savings_low)
        cnt = rec.get("operation_count", 1)
        steps.append(f"{rec['category']}\n({cnt} ops)")
        e2e_ms_list.append(lat_mid)
        savings_list.append(sm)
        cumulative_rel.append(round(baseline_ms / lat_mid * 100, 1))
        err_lower.append(lat_mid - lat_best)
        err_upper.append(lat_worst - lat_mid)
        rlo = baseline_ms / lat_worst * 100 if lat_worst > 0 else 100.0
        rhi = baseline_ms / lat_best * 100 if lat_best > 0 else 100.0
        cum_rel_low.append(round(rlo, 1))
        cum_rel_high.append(round(rhi, 1))

    e2e_ms = np.array(e2e_ms_list, dtype=float)
    err_lower = np.array(err_lower, dtype=float)
    err_upper = np.array(err_upper, dtype=float)

    plotted_cats = {r["category"] for r in recommendations}
    REST_KEY = "__rest_e2e__"

    # Baseline kernel times from manifest (compute_kernel tier only)
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
        baseline_by_cat[REST_KEY] = rest_e2e

    # Stack order (bottom -> top): same spirit as generate_perf_plot — analyzed
    # categories by ascending time, then unanalyzed by ascending time, then rest.
    analyzed = [
        {"name": n, "time_ms": baseline_by_cat[n]}
        for n in baseline_by_cat
        if n != REST_KEY and n in plotted_cats
    ]
    unanalyzed = [
        {"name": n, "time_ms": baseline_by_cat[n]}
        for n in baseline_by_cat
        if n != REST_KEY and n not in plotted_cats
    ]
    analyzed.sort(key=lambda x: x["time_ms"])
    unanalyzed.sort(key=lambda x: x["time_ms"])
    segment_order = [x["name"] for x in analyzed + unanalyzed]
    if REST_KEY in baseline_by_cat:
        segment_order.append(REST_KEY)

    cat_palette = [
        "#e74c3c", "#e67e22", "#f1c40f", "#2ecc71",
        "#9b59b6", "#1abc9c", "#e84393", "#3498db",
    ]
    cat_color_map: dict[str, str] = {}
    for i, rec in enumerate(recommendations):
        cat_color_map[rec["category"]] = cat_palette[i % len(cat_palette)]

    rest_color = "#aab7c4"
    short_names = {
        "elementwise": "elem",
        "unaryelementwise": "unary",
        "convolution": "conv",
        "reduce": "red",
        REST_KEY: "Rest of E2E",
    }

    n_bars = len(steps)
    x = np.arange(n_bars, dtype=float)
    width = 0.62

    # Per-bar, per-segment heights after applying first (bar_index) recommendations
    def segment_heights_after_steps(num_recs_applied: int) -> dict[str, float]:
        savings_by_cat: dict[str, float] = {}
        for j in range(num_recs_applied):
            r = recommendations[j]
            c = r["category"]
            sv = float(r.get("savings_ms", 0))
            if c in baseline_by_cat and c != REST_KEY:
                savings_by_cat[c] = savings_by_cat.get(c, 0.0) + sv
            elif REST_KEY in baseline_by_cat:
                # Category not in manifest stack — attribute savings to E2E slack
                savings_by_cat[REST_KEY] = savings_by_cat.get(REST_KEY, 0.0) + sv
        out: dict[str, float] = {}
        for name in segment_order:
            base = baseline_by_cat.get(name, 0.0)
            if name == REST_KEY:
                out[name] = max(0.0, base - savings_by_cat.get(REST_KEY, 0.0))
            else:
                out[name] = max(0.0, base - savings_by_cat.get(name, 0.0))
        return out

    fig, (ax_stack, ax2) = plt.subplots(
        1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1.45, 1.0]}
    )

    bottom = np.zeros(n_bars, dtype=float)
    for seg_name in segment_order:
        h = np.array(
            [segment_heights_after_steps(k)[seg_name] for k in range(n_bars)],
            dtype=float,
        )
        if np.all(h <= 0):
            continue
        if seg_name == REST_KEY:
            color = rest_color
            lbl = short_names.get(REST_KEY, REST_KEY)
        else:
            color = cat_color_map.get(seg_name, "#bdc3c7")
            lbl = short_names.get(seg_name, seg_name)
        ax_stack.bar(
            x, h, width, bottom=bottom, label=lbl, color=color,
            edgecolor="white", linewidth=0.9, alpha=0.95,
        )
        bottom += h

    def _step_accent_color(bar_index: int) -> str:
        """Color for error bar + latency label: baseline neutral, else optimized category."""
        if bar_index <= 0:
            return "#333333"
        cat = recommendations[bar_index - 1].get("category", "")
        return cat_color_map.get(cat, "#333333")

    # Total height labels; optional per-bar error caps (roofline 75–100% band)
    yerr_lo = err_lower
    yerr_hi = err_upper
    for k in range(n_bars):
        total_h = float(bottom[k])
        accent = _step_accent_color(k)
        label_y = total_h + (yerr_hi[k] + 1.2 if show_error_bars else 1.2)
        ax_stack.text(
            x[k], label_y, f"{e2e_ms[k]:.1f} ms",
            ha="center", va="bottom", fontsize=9, fontweight="bold", color=accent,
        )
        if k > 0 and savings_list[k] > 0:
            ax_stack.text(
                x[k], total_h * 0.5, f"-{savings_list[k]:.1f} ms",
                ha="center", va="center", fontsize=8, color=accent,
                fontweight="bold", zorder=5,
            )
        if show_error_bars:
            ax_stack.errorbar(
                [x[k]], [total_h],
                yerr=[[yerr_lo[k]], [yerr_hi[k]]],
                fmt="none",
                ecolor=accent,
                elinewidth=1.1,
                capsize=3,
                zorder=6,
            )

    ax_stack.set_xticks(x)
    ax_stack.set_xticklabels(steps, fontsize=8)
    ax_stack.set_ylabel("E2E time stacked by category (ms)", fontsize=11)
    ax_stack.set_title(
        "Projected E2E Latency (stacked by kernel category)",
        fontsize=11, fontweight="bold", pad=10,
    )
    ymax_stack = (
        float(np.max(bottom + yerr_hi) + 8)
        if show_error_bars
        else float(np.max(bottom) + 8)
    )
    ax_stack.set_ylim(0, ymax_stack * 1.12)
    ax_stack.spines["top"].set_visible(False)
    ax_stack.spines["right"].set_visible(False)
    ax_stack.legend(
        loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=7.5,
        frameon=False, title="Operations",
    )

    # -- Right: throughput cone (same as generate_perf_plot) --
    x_vals = np.arange(len(steps))
    cum_arr = np.array(cumulative_rel, dtype=float)
    cum_low = np.array(cum_rel_low, dtype=float)
    cum_high = np.array(cum_rel_high, dtype=float)
    cum_low[0] = cum_high[0] = cum_arr[0]

    ax2.fill_between(x_vals, cum_low, cum_high, color="#2ecc71", alpha=0.15, zorder=1)
    ax2.plot(x_vals, cum_high, "-", color="#27ae60", linewidth=1.0, alpha=0.4, zorder=2)
    ax2.plot(x_vals, cum_low, "-", color="#27ae60", linewidth=1.0, alpha=0.4, zorder=2)
    ax2.plot(
        x_vals, cumulative_rel, "o-", color="#2ecc71", linewidth=2.5,
        markersize=9, markerfacecolor="white", markeredgewidth=2.5, zorder=4,
    )
    last_idx = len(cumulative_rel) - 1
    ax2.annotate(
        f"{cumulative_rel[last_idx]:.0f}%",
        (last_idx, cumulative_rel[last_idx]),
        textcoords="offset points", xytext=(0, -16),
        ha="center", va="top", fontsize=11, fontweight="bold",
        color="#1a7a3a", zorder=5,
    )
    ax2.set_xticks(range(len(steps)))
    ax2.set_xticklabels(steps, fontsize=9)
    ax2.set_ylabel("% Relative Throughput (Baseline = 100)", fontsize=11)
    ax2.set_title(
        "Cumulative Throughput Improvement\n(75\u2013100% roofline potential)",
        fontsize=12, fontweight="bold", pad=12,
    )
    ax2.set_ylim(80, max(cum_high) * 1.15)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax2.axhline(y=100, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax2.grid(axis="y", linestyle="--", alpha=0.3)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.tick_params(axis="x", labelsize=9)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    out_path = os.path.join(output_dir, output_filename)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Stacked projection plot saved to {out_path}")

    if write_base64:
        with open(out_path, "rb") as f:
            b64_str = base64.b64encode(f.read()).decode("ascii")
        # Same path as generate_perf_plot so embed_plot_in_report works unchanged
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

    Reads perf_improvement_base64.txt (written by generate_perf_plot or
    generate_perf_plot_stacked_projection with write_base64=True) and substitutes
    the placeholder in the report file. If the base64 file is missing, the placeholder
    is removed so the report remains clean.

    Args:
        output_dir: Base output directory containing the report and base64 file
        report_filename: Name of the markdown report file
        placeholder: Placeholder string to replace

    Returns:
        True if the plot was embedded, False if the base64 file was missing or
        the report file does not exist
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

    # Replace placeholder first; if already embedded, replace existing image tag
    report = report.replace(placeholder, img_tag)
    if embedded and placeholder not in report:
        # Replace previous embed so re-running updates the image
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
    """Run the full plot pipeline: generate plot_data.json, render the plot, and embed it.

    Chains generate_plot_data() -> generate_perf_plot_stacked_projection() ->
    embed_plot_in_report() so the orchestrator only needs a single call. Renders
    ``perf_improvement.png`` (stacked E2E by category + throughput cone). Each stage
    is independent; failures are reported without aborting subsequent stages where possible.

    Args:
        output_dir: Base output directory containing category_data/ and the report
        title: Plot suptitle (e.g. '<Model> on <Platform> — Kernel Tuning Potential')

    Returns:
        Dict with boolean status for each stage: plot_data, plot, embed
    """

    results = {"plot_data": False, "plot": False, "embed": False}

    try:
        generate_plot_data(output_dir)
        results["plot_data"] = True
    except Exception as e:
        print(f"plot_data generation failed: {e}")
        return results

    results["plot"] = generate_perf_plot_stacked_projection(
        output_dir,
        title,
        output_filename="perf_improvement.png",
        write_base64=True,
        show_error_bars=True,
    )

    if results["plot"]:
        results["embed"] = embed_plot_in_report(output_dir)

    return results
