###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Validation utilities for TraceLens AgenticMode.

Validates subagent outputs before report aggregation:
- Time sanity check (category sum vs computation time)
- Efficiency anomaly detection (>100%)
- Coverage check (all expected findings present)
- Priority consistency check (top categories by GPU time)
"""

import json
import os
import re


def validate_subagent_outputs(output_dir):
    """Run all validation checks on subagent outputs.

    Returns a dict with check results and prints a summary.
    """
    manifest_path = os.path.join(output_dir, "category_data", "category_manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)

    results = {
        "time_check": _check_time_sanity(manifest),
        "efficiency_check": _check_efficiency_anomalies(output_dir),
        "coverage_check": _check_coverage(output_dir, manifest),
        "priority_check": _check_priority_consistency(manifest),
    }

    print("=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    for check_name, check_result in results.items():
        status = check_result["status"]
        print(f"  {check_name}: {status}")
        for msg in check_result.get("messages", []):
            print(f"    {msg}")
    print("=" * 60)

    return results


def _check_time_sanity(manifest):
    """Verify category GPU kernel time sum approximates computation time."""
    categories = manifest.get("categories", [])
    total_category_time = sum(
        cat.get("gpu_kernel_time_ms", 0)
        for cat in categories
        if cat.get("tier") == "compute_kernel"
    )

    gpu_util = manifest.get("gpu_utilization", {})
    total_time = gpu_util.get("total_time_ms", 0)
    comp_pct = gpu_util.get("computation_time_percent", 0)
    computation_time = total_time * comp_pct / 100

    if computation_time <= 0:
        return {"status": "WARN", "messages": ["No computation time in manifest"]}

    discrepancy = abs(total_category_time - computation_time) / computation_time * 100

    if discrepancy > 15:
        return {
            "status": "WARN",
            "messages": [
                f"Category sum ({total_category_time:.1f}ms) vs "
                f"Computation time ({computation_time:.1f}ms) = {discrepancy:.1f}% difference"
            ],
        }

    return {"status": "PASS", "messages": []}


def _check_efficiency_anomalies(output_dir):
    """Scan compute kernel findings for efficiency values > 100%."""
    findings_dir = os.path.join(output_dir, "category_findings")
    anomalies = []

    if not os.path.isdir(findings_dir):
        return {"status": "WARN", "messages": ["category_findings/ not found"]}

    for f in os.listdir(findings_dir):
        if f.endswith("_findings.md"):
            with open(os.path.join(findings_dir, f)) as fh:
                content = fh.read()
                matches = re.findall(r"(\d{3,}\.?\d*)\s*%", content)
                for m in matches:
                    if float(m) > 100:
                        anomalies.append(
                            {
                                "category": f.replace("_findings.md", ""),
                                "value": f"{m}%",
                            }
                        )

    if anomalies:
        messages = [f"{a['category']}: {a['value']}" for a in anomalies]
        messages.append(
            "Anomalies indicate measurement issues - do not use for prioritization"
        )
        return {"status": "WARN", "messages": messages}

    return {"status": "PASS", "messages": []}


def _check_coverage(output_dir, manifest):
    """Verify all expected categories have findings."""
    categories = manifest.get("categories", [])
    messages = []

    system_dir = os.path.join(output_dir, "system_findings")
    expected_system = [c["name"] for c in categories if c.get("tier") == "system"]
    found_system = []
    if os.path.isdir(system_dir):
        found_system = [
            f.replace("_findings.md", "")
            for f in os.listdir(system_dir)
            if f.endswith("_findings.md")
        ]
    missing_system = set(expected_system) - set(found_system)
    if missing_system:
        messages.append(f"Missing system findings: {', '.join(missing_system)}")

    compute_dir = os.path.join(output_dir, "category_findings")
    expected_compute = [
        c["name"] for c in categories if c.get("tier") == "compute_kernel"
    ]
    found_compute = []
    if os.path.isdir(compute_dir):
        found_compute = [
            f.replace("_findings.md", "")
            for f in os.listdir(compute_dir)
            if f.endswith("_findings.md")
        ]
    missing_compute = set(expected_compute) - set(found_compute)
    if missing_compute:
        messages.append(f"Missing compute findings: {', '.join(missing_compute)}")

    status = "WARN" if messages else "PASS"
    return {"status": status, "messages": messages}


def _check_priority_consistency(manifest):
    """Identify top categories by GPU time for priority verification."""
    categories = manifest.get("categories", [])
    sorted_cats = sorted(
        [c for c in categories if c.get("tier") == "compute_kernel"],
        key=lambda x: x.get("gpu_kernel_time_ms", 0),
        reverse=True,
    )

    top_names = [c["name"] for c in sorted_cats[:3]]
    return {
        "status": "INFO",
        "messages": [
            f"Top 3 by GPU time: {top_names}",
            "Verify these receive P1-P3 in compute kernel recommendations",
        ],
    }
