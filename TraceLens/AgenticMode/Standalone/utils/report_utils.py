###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Report aggregation utilities for TraceLens AgenticMode.

Provides functions for reading and aggregating findings from system-level
and compute kernel subagents for final report assembly.
"""

import json
import os


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
