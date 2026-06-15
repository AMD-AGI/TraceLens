###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Report aggregation utilities for TraceLens Agent.

Provides functions for reading and aggregating findings from system-level
and compute kernel subagents, and for extracting model-identification data
for the analysis report pipeline.
"""

import json
import os
from collections import defaultdict
from typing import List

import pandas as pd


def load_manifest(output_dir: str) -> dict:
    """Load and return the category manifest JSON from output_dir."""
    manifest_path = os.path.join(output_dir, "category_data", "category_manifest.json")
    with open(manifest_path) as f:
        return json.load(f)


def extract_condensed_op_info(
    output_dir: str, comparison_scope: str = "standalone"
) -> bool:
    """Extract name, Input type, Input Dims to metadata/condensed_op_info.csv.

    Reads unified_perf_summary.csv from the appropriate perf report CSV
    directory and writes the three columns for the model-identification
    subagent.  Returns True on success.

    Args:
        output_dir: Base analysis output directory.
        comparison_scope: ``"standalone"`` (default) reads from
            ``perf_report_csvs/``; ``"comparative"`` reads from
            ``perf_report_trace1_csvs/``.
    """
    _CONDENSED_OP_INFO_COLUMNS = ("name", "Input type", "Input Dims")
    csv_dir = (
        "perf_report_trace1_csvs"
        if comparison_scope == "comparative"
        else "perf_report_csvs"
    )
    csv_path = os.path.join(output_dir, csv_dir, "unified_perf_summary.csv")
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
    manifest = load_manifest(output_dir)

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


def _scan_findings_dir(output_dir: str, subdir: str) -> dict:
    """Read all *_findings.md files from a subdirectory.

    Returns ``{category_name: file_content}`` for every findings file found.
    Returns an empty dict if the directory does not exist.
    """
    findings_dir = os.path.join(output_dir, subdir)
    result = {}
    if os.path.isdir(findings_dir):
        for f in os.listdir(findings_dir):
            if f.endswith("_findings.md"):
                with open(os.path.join(findings_dir, f)) as fh:
                    result[f.replace("_findings.md", "")] = fh.read()
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
    raw_system = _scan_findings_dir(output_dir, "system_findings")
    system_findings = {}
    failed_system = []
    for name, content in raw_system.items():
        if "Status: ERROR" in content:
            failed_system.append({"category": name, "content": content})
        else:
            system_findings[name] = content

    raw_compute = _scan_findings_dir(output_dir, "category_findings")
    compute_findings = {}
    failed_compute = []
    for name, content in raw_compute.items():
        if "Status: ERROR" in content:
            failed_compute.append({"category": name, "content": content})
        else:
            compute_findings[name] = content

    manifest = {}
    top_ops = []
    manifest_path = os.path.join(output_dir, "category_data", "category_manifest.json")
    if os.path.exists(manifest_path):
        manifest = load_manifest(output_dir)
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
    """Aggregate ``impact_estimates`` into ``priority_data.json`` -- the single
    deterministic source of truth for report P-item ordering, the Top-Ops
    table, and the optional detailed extension plot.

    Produces four top-level arrays:
      - ``findings``: per-(category, bound_type, library, eff_bucket) groups
        from each ``_metrics.json::category_findings``. Sorted
        globally by ``impact_score`` with ``global_rank`` / ``category_rank``
        attached. Drives the report's flat P-numbering.
      - ``priorities``: ranked category list. Quantified categories are a
        per-category rollup of ``findings`` (so the Top-Ops "Potential
        improvement" column equals the sum of the rendered P-item Impacts for
        that category). Unmodeled categories with >5% of compute time follow,
        sorted by ``gpu_kernel_time_ms``.
      - ``recommendations``: same quantified rollup, capped, used by the
        extension plot.
      - ``all_estimates``: flat unfiltered audit trail of every per-op
        estimate the analyzers emitted.

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
        findings: List[dict] = []
        for fname in sorted(os.listdir(category_data_dir)):
            if not fname.endswith("_metrics.json"):
                continue
            fpath = os.path.join(category_data_dir, fname)
            with open(fpath, "r") as f:
                metrics = json.load(f)
            if metrics.get("status") in ("ERROR", "NO_DATA"):
                continue
            all_estimates.extend(metrics.get("impact_estimates", []))
            cat = metrics.get("category")
            for cf in metrics.get("category_findings", []):
                cf["category"] = cat
                cf["category_rank"] = cf.pop("rank", 0)
                findings.append(cf)

        quantified: dict = defaultdict(
            lambda: {
                "impact_score": 0.0,
                "impact_score_low": 0.0,
                "impact_score_high": 0.0,
                "operation_count": 0,
            }
        )
        for f in findings:
            # impact_score=None (no perf model): excluded from rollup/plot, ranked last.
            if f.get("impact_score") is None:
                continue
            cat = f["category"]
            quantified[cat]["impact_score"] += f.get("impact_score", 0)
            quantified[cat]["impact_score_low"] += f.get("impact_score_low", 0)
            quantified[cat]["impact_score_high"] += f.get("impact_score_high", 0)
            quantified[cat]["operation_count"] += f.get("operation_count", 0)

        plot_recs = sorted(
            [
                {
                    "category": cat,
                    "impact_score": round(v["impact_score"], 2),
                    "impact_score_low": round(v["impact_score_low"], 2),
                    "impact_score_high": round(v["impact_score_high"], 2),
                    "operation_count": v["operation_count"],
                    "type": "kernel_tuning",
                }
                for cat, v in quantified.items()
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
                    "source": "findings_rollup",
                }
            )

        quantified_cats = set(quantified.keys())
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

        # Quantified findings sort by impact_score descending; non-quantifiable
        # (impact_score=None) findings always rank last.
        findings.sort(
            key=lambda f: (
                f.get("impact_score") is not None,
                f.get("impact_score") or 0,
            ),
            reverse=True,
        )
        for global_rank, f in enumerate(findings, start=1):
            f["global_rank"] = global_rank

        priority_data = {
            "baseline_ms": baseline_ms,
            "priorities": priorities,
            "recommendations": plot_recs,
            "findings": findings,
            "all_estimates": all_estimates,
        }
    except Exception:
        priority_data = {
            "baseline_ms": 0,
            "priorities": [],
            "recommendations": [],
            "findings": [],
            "all_estimates": [],
        }

    with open(out_path, "w") as f:
        json.dump(priority_data, f, indent=2)

    return out_path
