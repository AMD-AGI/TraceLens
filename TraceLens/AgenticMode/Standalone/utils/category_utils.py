###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Category-to-skill mapping and enhanced category classification."""

import json
import os
import re
import pandas as pd

CATEGORY_SKILL_MAP = {
    "cpu_idle": "cpu-idle-analysis",
    "gemm": "gemm-analysis",
    "moe_fused": "moe-analysis",
    "sdpa_fwd": "sdpa-analysis",
    "sdpa_bwd": "sdpa-analysis",
    "elementwise": "elementwise-analysis",
    "reduce": "reduce-analysis",
    "triton": "triton-analysis",
    "norm": "norm-analysis",
    "convolution": "convolution-analysis",
    "other": "generic-op-analysis",
}


def get_enhanced_category(row):
    """Determine category with special handling for MoE, Norm, Convolution."""
    op_name = row.get("name", "")
    category = row.get("op category", "")

    if "moe" in op_name.lower() or "fused_moe" in op_name.lower():
        return "moe_fused", "MoE Fused"
    elif any(
        n in op_name.lower()
        for n in [
            "batch_norm",
            "batchnorm",
            "layer_norm",
            "layernorm",
            "group_norm",
            "groupnorm",
            "instance_norm",
        ]
    ):
        return "norm", "Norm"
    elif "conv" in op_name.lower() and (
        "aten::" in op_name or "backward" in op_name.lower()
    ):
        return "convolution", "Convolution"

    if pd.isna(category) or category == "":
        return "other", "Other"
    else:
        category_name = category.replace(" ", "_").replace("/", "_").lower()
        display_name = category
        return category_name, display_name


def _count_reasoning_candidates(findings_path: str) -> int:
    if not os.path.isfile(findings_path):
        return 0
    with open(findings_path) as f:
        return len(re.findall(r"<!-- reasoning-candidate", f.read()))


def _findings_path(output_dir: str, category: str, tier: str) -> str:
    subdir = "system_findings" if tier == "system" else "category_findings"
    return os.path.join(output_dir, subdir, f"{category}_findings.md")


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

    findings = _findings_path(output_dir, category, tier)
    n_candidates = _count_reasoning_candidates(findings)

    if tier == "system":
        entry = {
            "low_e2e_ms": None,
            "high_e2e_ms": None,
            "low_e2e_percent": None,
            "high_e2e_percent": None,
            "quantifiable": False,
        }
        meta["impact_estimates"] = [entry] * n_candidates
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
            low = round(sum(e.get("savings_ms_low", 0) for e in estimates), 3)
            high = round(sum(e.get("savings_ms_high", 0) for e in estimates), 3)
            low_pct = round(sum(e.get("e2e_pct_low", 0) for e in estimates), 2)
            high_pct = round(sum(e.get("e2e_pct_high", 0) for e in estimates), 2)
            rollup = {
                "low_e2e_ms": low,
                "high_e2e_ms": high,
                "low_e2e_percent": low_pct,
                "high_e2e_percent": high_pct,
                "quantifiable": True,
            }
            if n_candidates == 1:
                meta["impact_estimates"] = [rollup]
            else:
                per_candidate = round(low / n_candidates, 3)
                per_high = round(high / n_candidates, 3)
                per_low_pct = round(low_pct / n_candidates, 2)
                per_high_pct = round(high_pct / n_candidates, 2)
                meta["impact_estimates"] = [
                    {
                        "low_e2e_ms": per_candidate,
                        "high_e2e_ms": per_high,
                        "low_e2e_percent": per_low_pct,
                        "high_e2e_percent": per_high_pct,
                        "quantifiable": True,
                    }
                    for _ in range(n_candidates)
                ]
        else:
            entry = {
                "low_e2e_ms": None,
                "high_e2e_ms": None,
                "low_e2e_percent": None,
                "high_e2e_percent": None,
                "quantifiable": False,
            }
            meta["impact_estimates"] = [entry] * n_candidates

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(
        f"Impact estimates written: {len(meta['impact_estimates'])} entries "
        f"to {meta_path}"
    )
