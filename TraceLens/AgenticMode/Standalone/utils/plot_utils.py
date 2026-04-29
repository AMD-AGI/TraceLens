###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Plot generation and report embedding utilities for TraceLens AgenticMode.

Public API:

- ``generate_perf_plot`` -- single horizontal stacked bar showing the run's
  compute-time breakdown by kernel category (manifest-only, no savings, no
  error bars).
- ``generate_and_embed_plot`` -- end-to-end pipeline (priority_data -> plot
  -> embed).

Data aggregation (``generate_priority_data``) lives in ``report_utils.py``;
per-category grouping (``build_category_findings``) lives in
``category_analyses/analysis_utils.py``.
"""

import base64
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from TraceLens.AgenticMode.Standalone.utils.report_utils import (
    generate_priority_data,
    load_manifest,
)

__all__ = [
    "generate_perf_plot",
    "generate_and_embed_plot",
    "embed_plot_in_report",
]

_CAT_PALETTE = [
    "#e74c3c",
    "#e67e22",
    "#f1c40f",
    "#2ecc71",
    "#9b59b6",
    "#1abc9c",
    "#e84393",
    "#3498db",
]

_REST_KEY = "__rest_e2e__"
_REST_COLOR = "#aab7c4"


def generate_perf_plot(
    output_dir: str,
    title: str,
    output_filename: str = "perf_improvement.png",
    write_base64: bool = True,
) -> bool:
    """Render a single horizontal stacked bar showing the run's compute-time
    breakdown by kernel category.

    Reads ``<output_dir>/category_data/category_manifest.json`` directly --
    purely descriptive of the run, no savings, no error bars, no throughput
    cone. Compute-tier categories are stacked left-to-right by descending
    ``gpu_kernel_time_ms``; any remaining E2E time (baseline_ms minus the sum
    of compute kernels) is appended as a "Non-computing" gray segment.

    Args:
        output_dir: Directory containing ``category_data/category_manifest.json``.
        title: Figure title.
        output_filename: PNG name under ``output_dir``.
        write_base64: Write ``perf_improvement_base64.txt`` for report embedding.

    Returns:
        True if the figure was written, False if inputs were missing or invalid.
    """
    try:
        manifest = load_manifest(output_dir)
    except FileNotFoundError:
        print("category_manifest.json not found - skipping plot")
        return False

    baseline_ms = float(manifest.get("gpu_utilization", {}).get("total_time_ms", 0))
    if baseline_ms <= 0:
        print("Invalid baseline_ms in manifest - skipping plot")
        return False

    segments = []
    kernel_sum = 0.0
    for cat in manifest.get("categories", []):
        if cat.get("tier") != "compute_kernel":
            continue
        gt = float(cat.get("gpu_kernel_time_ms", 0) or 0)
        if gt <= 0:
            continue
        segments.append({"name": cat["name"], "time_ms": gt})
        kernel_sum += gt

    segments.sort(key=lambda s: s["time_ms"], reverse=True)

    rest_ms = max(0.0, baseline_ms - kernel_sum)
    if rest_ms > 0:
        segments.append({"name": _REST_KEY, "time_ms": rest_ms})

    if not segments:
        print("No compute-kernel segments to plot - skipping plot")
        return False

    fig, ax = plt.subplots(figsize=(12, 3.0))

    cumulative = 0.0
    color_idx = 0
    for seg in segments:
        width = seg["time_ms"]
        pct = width / baseline_ms * 100
        if seg["name"] == _REST_KEY:
            color = _REST_COLOR
            display_name = "Non-computing"
        else:
            color = _CAT_PALETTE[color_idx % len(_CAT_PALETTE)]
            color_idx += 1
            raw = seg["name"]
            display_name = raw[0].upper() + raw[1:] if raw else raw
        legend_label = f"{display_name} \u2014 {width:.1f} ms ({pct:.1f}%)"
        ax.barh(
            [0],
            [width],
            left=[cumulative],
            color=color,
            edgecolor="white",
            linewidth=0.9,
            alpha=0.95,
            label=legend_label,
        )
        cumulative += width

    ax.set_xlim(0, baseline_ms)
    ax.set_ylim(-0.6, 0.6)
    ax.set_yticks([])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel("GPU time (ms)", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.4),
        ncol=min(len(segments), 4),
        frameon=False,
        fontsize=9,
    )

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

    Reads ``perf_improvement_base64.txt`` and substitutes the placeholder in
    the report file. If the base64 file is missing, the placeholder is removed
    so the report remains clean.

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
        title: Plot title (e.g. '<Model> on <Platform> -- Compute-Time Breakdown')

    Returns:
        Dict with boolean status for each stage: plot_data, plot, embed
    """
    results = {"plot_data": False, "plot": False, "embed": False}

    try:
        generate_priority_data(output_dir)
        results["plot_data"] = True
    except Exception as e:
        print(f"priority_data generation failed: {e}")

    results["plot"] = generate_perf_plot(output_dir, title)
    if results["plot"]:
        results["embed"] = embed_plot_in_report(output_dir)

    return results


def _short_name(name: str, max_len: int = 8) -> str:
    """Shorten a category name for plot labels."""
    display = name[0].upper() + name[1:] if name else name
    if len(display) <= max_len:
        return display
    return display[: max_len - 1] + "\u2026"
