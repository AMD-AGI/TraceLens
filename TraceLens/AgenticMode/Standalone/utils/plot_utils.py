###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Plot generation and report embedding utilities for TraceLens AgenticMode.

Provides functions for:
- Generating the performance improvement plot from pre-computed plot data
- Embedding the plot as a base64 data URI in the markdown report
- Running the full plot pipeline (plot data + plot + embed) as a single call
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


def generate_perf_plot(output_dir: str, title: str) -> bool:
    """
    Generate the performance improvement plot from plot_data.json.

    Reads plot_data.json (produced by generate_plot_data), computes cumulative
    projections using the 75–100% roofline potential improvement range
    (savings_ms_low / savings_ms_high per recommendation), renders a two-panel
    matplotlib chart (bar + line), and saves both the PNG image and a
    base64-encoded text file for report embedding.
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

    # Cumulative savings: midpoint (savings_ms), low bound (savings_ms_low), high bound (savings_ms_high)
    # 75–100% roofline: less savings = savings_ms_low, more savings = savings_ms_high
    cum_savings_mid = 0.0
    cum_savings_low = 0.0
    cum_savings_high = 0.0

    steps = ["Baseline"]
    e2e_ms = [baseline_ms]
    savings_list = [0]
    cumulative_rel = [100.0]
    # For left pane: asymmetric error bars (lower = toward better latency, upper = toward worse)
    err_lower = [0.0]  # bar - err_lower = best-case latency (more savings)
    err_upper = [0.0]  # bar + err_upper = worst-case latency (less savings)
    # For right pane: throughput band from improvement range
    cum_rel_low = [100.0]
    cum_rel_high = [100.0]

    for rec in recommendations:
        sav_mid = float(rec.get("savings_ms", 0))
        sav_lo = float(rec.get("savings_ms_low", sav_mid))
        sav_hi = float(rec.get("savings_ms_high", sav_mid))
        cum_savings_mid += sav_mid
        cum_savings_low += sav_lo
        cum_savings_high += sav_hi
        # Clamp so projected latency stays positive
        latency_mid = max(0.01, baseline_ms - cum_savings_mid)
        latency_best = max(0.01, baseline_ms - cum_savings_high)
        latency_worst = max(0.01, baseline_ms - cum_savings_low)
        count = rec.get("operation_count", 1)
        label = rec["category"] + f"\n({count} ops)"
        steps.append(label)
        e2e_ms.append(latency_mid)
        savings_list.append(sav_mid)
        cumulative_rel.append(round(baseline_ms / latency_mid * 100, 1))
        # Left pane: bar = latency_mid; error bar down = latency_mid - latency_best, up = latency_worst - latency_mid
        err_lower.append(latency_mid - latency_best)
        err_upper.append(latency_worst - latency_mid)
        # Right pane: throughput band from improvement range (no uncertainty at baseline)
        rel_lo = baseline_ms / latency_worst * 100 if latency_worst > 0 else 100.0
        rel_hi = baseline_ms / latency_best * 100 if latency_best > 0 else 100.0
        cum_rel_low.append(round(rel_lo, 1))
        cum_rel_high.append(round(rel_hi, 1))

    e2e_ms = np.array(e2e_ms)
    err_lower = np.array(err_lower)
    err_upper = np.array(err_upper)
    # Baseline has no error bar; rest use 75–100% roofline range
    yerr_rest = np.array([err_lower[1:], err_upper[1:]])

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1.1, 1]}
    )

    colors = [
        "#4a90d9",
        "#e74c3c",
        "#e67e22",
        "#f1c40f",
        "#2ecc71",
        "#9b59b6",
        "#1abc9c",
    ][: len(steps)]
    # Draw baseline bar with no error bar, then remaining bars with error bars
    bar0 = ax1.bar(
        steps[0:1],
        e2e_ms[0:1],
        color=colors[0],
        edgecolor="white",
        linewidth=1.2,
        width=0.65,
    )
    bars_rest = ax1.bar(
        steps[1:],
        e2e_ms[1:],
        color=colors[1:],
        edgecolor="white",
        linewidth=1.2,
        width=0.65,
        yerr=yerr_rest,
        capsize=4,
        error_kw=dict(ecolor="#333333", linewidth=1.2),
    )
    bars = list(bar0) + list(bars_rest)
    y_max_for_label = e2e_ms + err_upper
    for i, (bar, val, sav) in enumerate(zip(bars, e2e_ms, savings_list)):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            y_max_for_label[i] + 1.5,
            f"{val:.1f} ms",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
        if sav > 0:
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() / 2,
                f"-{sav:.1f} ms",
                ha="center",
                va="center",
                fontsize=9,
                color="white",
                fontweight="bold",
            )
    ax1.set_ylabel("E2E Latency (ms)", fontsize=11)
    ax1.set_title(
        "Projected E2E Latency After Each Optimization\n(75–100% roofline potential)",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax1.set_ylim(0, (np.max(y_max_for_label) + 50) * 1.15)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.tick_params(axis="x", labelsize=9)

    # Right pane: cumulative throughput with uncertainty band from expected improvement range
    x_vals = np.arange(len(steps))
    cum_arr = np.array(cumulative_rel, dtype=float)
    cum_low = np.array(cum_rel_low, dtype=float)
    cum_high = np.array(cum_rel_high, dtype=float)
    # No uncertainty at baseline (first point)
    cum_low[0] = cum_high[0] = cum_arr[0]
    ax2.fill_between(x_vals, cum_low, cum_high, color="#2ecc71", alpha=0.35, zorder=0)
    ax2.plot(x_vals, cum_high, "-", color="#27ae60", linewidth=1.5, zorder=2)
    ax2.plot(x_vals, cum_low, "-", color="#27ae60", linewidth=1.5, zorder=2)
    ax2.plot(
        x_vals,
        cumulative_rel,
        "o-",
        color="#2ecc71",
        linewidth=2.5,
        markersize=9,
        markerfacecolor="white",
        markeredgewidth=2.5,
        zorder=3,
    )
    for x, y in enumerate(cumulative_rel):
        ax2.annotate(
            f"{y}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 12),
            ha="center",
            fontsize=10,
            fontweight="bold",
            color="#27ae60",
        )
    ax2.set_xticks(range(len(steps)))
    ax2.set_xticklabels(steps, fontsize=9)
    ax2.set_ylabel("Relative Throughput (Baseline = 100)", fontsize=11)
    ax2.set_title(
        "Cumulative Throughput Improvement\n(75–100% roofline potential)",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax2.set_ylim(80, max(cum_high) * 1.15)
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


def embed_plot_in_report(
    output_dir: str,
    report_filename: str = "standalone_analysis.md",
    placeholder: str = "{{PERF_PLOT}}",
) -> bool:
    """
    Replace the plot placeholder in the report with a base64-embedded PNG data URI.

    Reads perf_improvement_base64.txt (written by generate_perf_plot) and substitutes
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

    Chains generate_plot_data() -> generate_perf_plot() -> embed_plot_in_report()
    so the orchestrator only needs a single call. Each stage is independent and
    failures are reported without aborting subsequent stages where possible.

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

    results["plot"] = generate_perf_plot(output_dir, title)

    if results["plot"]:
        results["embed"] = embed_plot_in_report(output_dir)

    return results
