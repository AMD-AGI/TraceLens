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
- ``generate_priority_data`` -- aggregate ``impact_estimates`` into
  ``priority_data.json`` (used for orchestrator P-item ranking and by the
  optional detailed extension plot).
- ``generate_and_embed_plot`` -- end-to-end pipeline (priority_data -> plot
  -> embed).

For the detailed cumulative-savings chart (formerly the 2-panel view), see
``TraceLens.AgenticMode.Standalone.utils.agent_extension.generate_impact_savings_plot``.
"""

import base64
import json
import os
import re
from collections import defaultdict
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from TraceLens.AgenticMode.Standalone.utils.report_utils import load_manifest

__all__ = [
    "generate_perf_plot",
    "generate_priority_data",
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


def _short_name(name: str, max_len: int = 8) -> str:
    """Shorten a category name for plot labels. Capitalizes first letter, truncates if needed."""
    display = name[0].upper() + name[1:] if name else name
    if len(display) <= max_len:
        return display
    return display[: max_len - 1] + "\u2026"


def generate_priority_data(output_dir: str, max_recommendations: int = 6) -> str:
    """Aggregate ``impact_estimates`` into ``priority_data.json`` -- the single
    deterministic source of truth for both report P-item ordering and the
    optional detailed extension plot.

    Produces three top-level arrays:
      - ``priorities``: ranked category list for report P-items (quantified
        categories sorted by ``impact_score``, then unmodeled categories with
        >5% of compute time sorted by ``gpu_kernel_time_ms``).
      - ``recommendations``: same quantified categories, used by the optional
        ``agent_extension.generate_impact_savings_plot``.
      - ``all_estimates``: flat list of every per-operation estimate.

    Args:
        output_dir: Base output directory containing ``category_data/``.
        max_recommendations: Max categories in the plot recommendations.

    Returns:
        Path to written ``priority_data.json``.
    """
    out_path = os.path.join(output_dir, "priority_data.json")
    category_data_dir = os.path.join(output_dir, "category_data")

    try:
        manifest = load_manifest(output_dir)

        baseline_ms = manifest.get("gpu_utilization", {}).get("total_time_ms", 0)
        computation_pct = manifest.get("gpu_utilization", {}).get(
            "computation_time_percent", 0
        )
        computation_time_ms = baseline_ms * computation_pct / 100
        threshold_ms = computation_time_ms * 0.05

        all_estimates: List[dict] = []
        for fname in sorted(os.listdir(category_data_dir)):
            if not fname.endswith("_metrics.json"):
                continue
            fpath = os.path.join(category_data_dir, fname)
            with open(fpath, "r") as f:
                metrics = json.load(f)
            if metrics.get("status") in ("ERROR", "NO_DATA"):
                continue
            all_estimates.extend(metrics.get("impact_estimates", []))

        category_savings: dict = defaultdict(
            lambda: {
                "impact_score": 0.0,
                "impact_score_low": 0.0,
                "impact_score_high": 0.0,
                "count": 0,
                "ops": [],
            }
        )
        for e in all_estimates:
            if e.get("type") == "kernel_tuning" and e.get("confidence") in (
                "high",
                "medium",
            ):
                cat = e["category"]
                mid = e.get("impact_score", 0)
                category_savings[cat]["impact_score"] += mid
                category_savings[cat]["impact_score_low"] += e.get(
                    "impact_score_low", mid
                )
                category_savings[cat]["impact_score_high"] += e.get(
                    "impact_score_high", mid
                )
                category_savings[cat]["count"] += 1
                category_savings[cat]["ops"].append(e.get("operation", ""))

        plot_recs = sorted(
            [
                {
                    "category": cat,
                    "impact_score": round(v["impact_score"], 2),
                    "impact_score_low": round(v["impact_score_low"], 2),
                    "impact_score_high": round(v["impact_score_high"], 2),
                    "operation_count": v["count"],
                    "type": "kernel_tuning",
                }
                for cat, v in category_savings.items()
            ],
            key=lambda x: x["impact_score"],
            reverse=True,
        )[:max_recommendations]

        cat_display = {}
        for cat_entry in manifest.get("categories", []):
            cat_display[cat_entry["name"]] = cat_entry.get(
                "display_name", cat_entry["name"]
            )

        priorities: List[dict] = []
        for rank, rec in enumerate(plot_recs, 1):
            priorities.append(
                {
                    "rank": rank,
                    "category": rec["category"],
                    "display_name": cat_display.get(rec["category"], rec["category"]),
                    "impact_score": rec["impact_score"],
                    "impact_score_low": rec["impact_score_low"],
                    "impact_score_high": rec["impact_score_high"],
                    "source": "impact_estimates",
                }
            )

        quantified_cats = set(category_savings.keys())
        unmodeled = []
        for cat_entry in manifest.get("categories", []):
            cat_name = cat_entry.get("name")
            if cat_entry.get("tier") != "compute_kernel":
                continue
            if cat_name in quantified_cats:
                continue
            gpu_time = cat_entry.get("gpu_kernel_time_ms", 0)
            if gpu_time >= threshold_ms:
                unmodeled.append(
                    {
                        "category": cat_name,
                        "display_name": cat_entry.get("display_name", cat_name),
                        "gpu_kernel_time_ms": round(gpu_time, 3),
                    }
                )
        unmodeled.sort(key=lambda x: x["gpu_kernel_time_ms"], reverse=True)

        next_rank = len(priorities) + 1
        for entry in unmodeled:
            priorities.append(
                {
                    "rank": next_rank,
                    "category": entry["category"],
                    "display_name": entry["display_name"],
                    "impact_score": None,
                    "gpu_kernel_time_ms": entry["gpu_kernel_time_ms"],
                    "source": "manifest_fallback",
                }
            )
            next_rank += 1

        priority_data = {
            "baseline_ms": baseline_ms,
            "priorities": priorities,
            "recommendations": plot_recs,
            "all_estimates": all_estimates,
        }
    except Exception:
        priority_data = {
            "baseline_ms": 0,
            "priorities": [],
            "recommendations": [],
            "all_estimates": [],
        }

    with open(out_path, "w") as f:
        json.dump(priority_data, f, indent=2)

    return out_path


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

    fig, ax = plt.subplots(figsize=(12, 2.0))

    cumulative = 0.0
    color_idx = 0
    min_label_width = baseline_ms * 0.03
    for seg in segments:
        width = seg["time_ms"]
        if seg["name"] == _REST_KEY:
            color = _REST_COLOR
            label = "Non-computing"
        else:
            color = _CAT_PALETTE[color_idx % len(_CAT_PALETTE)]
            color_idx += 1
            label = _short_name(seg["name"])
        ax.barh(
            [0],
            [width],
            left=[cumulative],
            color=color,
            edgecolor="white",
            linewidth=0.9,
            alpha=0.95,
            label=label,
        )
        if width >= min_label_width:
            pct = width / baseline_ms * 100
            ax.text(
                cumulative + width / 2,
                0,
                f"{label}\n{width:.1f} ms ({pct:.1f}%)",
                ha="center",
                va="center",
                fontsize=8,
                color="black",
                fontweight="bold",
            )
        cumulative += width

    ax.set_xlim(0, baseline_ms)
    ax.set_ylim(-0.6, 0.6)
    ax.set_yticks([])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel("GPU time (ms)", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)

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
