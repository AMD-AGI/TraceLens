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
import re
import pandas as pd


def load_manifest(output_dir: str) -> dict:
    """Load and return the category manifest JSON from output_dir."""
    manifest_path = os.path.join(output_dir, "category_data", "category_manifest.json")
    with open(manifest_path) as f:
        return json.load(f)


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


def _non_quantifiable_entry() -> dict:
    """Return a single non-quantifiable impact estimate entry."""
    return {
        "impact_score_low": None,
        "impact_score": None,
        "impact_score_high": None,
        "quantifiable": False,
    }


def write_impact_estimates(output_dir: str, category: str, tier: str) -> None:
    """Write ``impact_estimates`` to metadata JSON for a category.

    Compute tier: reads per-operation estimates from *_metrics.json and builds
    one rollup entry per reasoning-candidate block in the findings file.

    System tier: writes one non-quantifiable entry per reasoning-candidate block.
    If no candidates exist, writes an empty array.
    """
    meta_path = os.path.join(output_dir, "metadata", f"{category}_metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)

    subdir = "system_findings" if tier == "system" else "category_findings"
    findings_path = os.path.join(output_dir, subdir, f"{category}_findings.md")
    if os.path.isfile(findings_path):
        with open(findings_path) as f:
            n_candidates = len(re.findall(r"<!-- reasoning-candidate", f.read()))
    else:
        n_candidates = 0

    if tier == "system":
        meta["impact_estimates"] = [_non_quantifiable_entry()] * n_candidates
    else:
        metrics_path = os.path.join(
            output_dir, "category_data", f"{category}_metrics.json"
        )
        with open(metrics_path) as f:
            metrics = json.load(f)
        estimates = metrics.get("impact_estimates", [])

        if n_candidates == 0:
            meta["impact_estimates"] = []
        elif estimates:
            score_low = round(sum(e.get("impact_score_low", 0) for e in estimates), 2)
            score_mid = round(sum(e.get("impact_score", 0) for e in estimates), 2)
            score_high = round(sum(e.get("impact_score_high", 0) for e in estimates), 2)
            rollup = {
                "impact_score_low": score_low,
                "impact_score": score_mid,
                "impact_score_high": score_high,
                "quantifiable": True,
            }
            if n_candidates == 1:
                meta["impact_estimates"] = [rollup]
            else:
                per_low = round(score_low / n_candidates, 2)
                per_mid = round(score_mid / n_candidates, 2)
                per_high = round(score_high / n_candidates, 2)
                meta["impact_estimates"] = [
                    {
                        "impact_score_low": per_low,
                        "impact_score": per_mid,
                        "impact_score_high": per_high,
                        "quantifiable": True,
                    }
                    for _ in range(n_candidates)
                ]
        else:
            meta["impact_estimates"] = [_non_quantifiable_entry()] * n_candidates

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(
        f"Impact estimates written: {len(meta['impact_estimates'])} entries "
        f"to {meta_path}"
    )
