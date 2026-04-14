###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Report aggregation utilities for TraceLens AgenticMode.

Provides functions for reading and aggregating findings from system-level
and compute kernel subagents, and for extracting model-identification data
for the standalone report pipeline.
"""

import json
import os
from collections import defaultdict
from typing import List

import pandas as pd


def extract_condensed_op_info(output_dir: str) -> bool:
    """Extract name, Input type, Input Dims to metadata/condensed_op_info.csv.

    Reads perf_report_csvs/unified_perf_summary.csv and writes the three columns
    for the condensed op info subagent. Returns True on success.
    """
    _CONDENSED_OP_INFO_COLUMNS = ("name", "Input type", "Input Dims")
    csv_path = os.path.join(output_dir, "perf_report_csvs", "unified_perf_summary.csv")
    if not os.path.isfile(csv_path):
        return False
    try:
        df = pd.read_csv(csv_path, usecols=list(_CONDENSED_OP_INFO_COLUMNS))
    except (ValueError, KeyError):
        return False
    out_path = os.path.join(output_dir, "metadata", "condensed_op_info.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    return True


def load_manifest_categories(output_dir):
    """Load the category manifest and return categories split by tier.

    Returns a dict with:
        manifest: the full category manifest dict
        gpu_utilization: gpu_utilization sub-dict from manifest
        system_categories: list of category dicts with tier == 'system'
        compute_categories: list of category dicts with tier == 'compute_kernel'
    """
    manifest_path = os.path.join(output_dir, "category_data", "category_manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)

    categories = manifest.get("categories", [])
    result = {
        "manifest": manifest,
        "gpu_utilization": manifest.get("gpu_utilization", {}),
        "system_categories": [c for c in categories if c.get("tier") == "system"],
        "compute_categories": [
            c for c in categories if c.get("tier") == "compute_kernel"
        ],
    }

    gpu_util = result["gpu_utilization"]
    print(
        json.dumps(
            {
                "system_categories": [c["name"] for c in result["system_categories"]],
                "compute_categories": [c["name"] for c in result["compute_categories"]],
                "gpu_utilization": {
                    k: gpu_util.get(k)
                    for k in (
                        "total_time_ms",
                        "computation_time_percent",
                        "idle_time_percent",
                        "exposed_comm_time_percent",
                        "exposed_memcpy_time_percent",
                    )
                },
            },
            indent=2,
        )
    )

    return result


def load_findings(output_dir):
    """Load all findings from system-level and compute kernel subagents.

    Returns a dict with:
        system_findings: dict of category_name -> findings content
        failed_system: list of {category, content} for errored system analyses
        compute_findings: dict of category_name -> findings content
        failed_compute: list of {category, content} for errored compute analyses
        manifest: the full category manifest dict
        top_ops: list of top operations from the manifest
    """
    system_findings = {}
    failed_system = []
    system_dir = os.path.join(output_dir, "system_findings")

    if os.path.isdir(system_dir):
        for f in os.listdir(system_dir):
            if f.endswith("_findings.md"):
                with open(os.path.join(system_dir, f)) as fh:
                    content = fh.read()
                    name = f.replace("_findings.md", "")
                    if "Status: ERROR" in content:
                        failed_system.append({"category": name, "content": content})
                    else:
                        system_findings[name] = content

    compute_findings = {}
    failed_compute = []
    compute_dir = os.path.join(output_dir, "category_findings")

    if os.path.isdir(compute_dir):
        for f in os.listdir(compute_dir):
            if f.endswith("_findings.md"):
                with open(os.path.join(compute_dir, f)) as fh:
                    content = fh.read()
                    name = f.replace("_findings.md", "")
                    if "Status: ERROR" in content:
                        failed_compute.append({"category": name, "content": content})
                    else:
                        compute_findings[name] = content

    manifest = {}
    top_ops = []
    manifest_path = os.path.join(output_dir, "category_data", "category_manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
            top_ops = manifest.get("top_operations", [])

    result = {
        "system_findings": system_findings,
        "failed_system": failed_system,
        "compute_findings": compute_findings,
        "failed_compute": failed_compute,
        "manifest": manifest,
        "top_ops": top_ops,
    }

    print(
        json.dumps(
            {
                "system_findings": list(system_findings.keys()),
                "failed_system": [f["category"] for f in failed_system],
                "compute_findings": list(compute_findings.keys()),
                "failed_compute": [f["category"] for f in failed_compute],
                "top_ops_count": len(top_ops),
            },
            indent=2,
        )
    )

    return result


def generate_priority_data(output_dir: str, max_recommendations: int = 6) -> str:
    """Aggregate impact_estimates into priority_data.json — the single
    deterministic source of truth for both report P-item ordering and the
    performance improvement plot.

    Produces three top-level arrays:
      - ``priorities``: ranked category list for report P-items (quantified
        categories sorted by savings_ms, then unmodeled categories with
        >5% of compute time sorted by gpu_kernel_time_ms)
      - ``recommendations``: same quantified categories, used by the plot
      - ``all_estimates``: flat list of every per-operation estimate

    Args:
        output_dir: Base output directory containing category_data/
        max_recommendations: Max categories in the plot recommendations

    Returns:
        Path to written priority_data.json
    """
    out_path = os.path.join(output_dir, "priority_data.json")
    category_data_dir = os.path.join(output_dir, "category_data")
    manifest_path = os.path.join(category_data_dir, "category_manifest.json")

    try:
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

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
                "savings_ms": 0,
                "savings_ms_low": 0,
                "savings_ms_high": 0,
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
                category_savings[cat]["savings_ms"] += e["savings_ms"]
                category_savings[cat]["savings_ms_low"] += e.get(
                    "savings_ms_low", e["savings_ms"]
                )
                category_savings[cat]["savings_ms_high"] += e.get(
                    "savings_ms_high", e["savings_ms"]
                )
                category_savings[cat]["count"] += 1
                category_savings[cat]["ops"].append(e.get("operation", ""))

        plot_recs = sorted(
            [
                {
                    "category": cat,
                    "savings_ms": round(v["savings_ms"], 3),
                    "savings_ms_low": round(v["savings_ms_low"], 3),
                    "savings_ms_high": round(v["savings_ms_high"], 3),
                    "operation_count": v["count"],
                    "type": "kernel_tuning",
                }
                for cat, v in category_savings.items()
            ],
            key=lambda x: x["savings_ms"],
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
                    "display_name": cat_display.get(
                        rec["category"], rec["category"]
                    ),
                    "savings_ms": rec["savings_ms"],
                    "savings_ms_low": rec["savings_ms_low"],
                    "savings_ms_high": rec["savings_ms_high"],
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
                    "savings_ms": None,
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
