#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Kernel Fusion Analysis - Performance Savings Estimation

Computes savings estimates for kernel fusion candidates by cross-referencing
fusion_candidates.json with unified_perf_summary.csv and projecting fused
kernel time via roofline model.
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analysis_utils import parse_first_shape, shape_aware_lookup, write_metrics_json

MAX_FUSION_KERNEL_COUNT = 15 # Skip candidates with more kernels than this (too complex to fuse reliably)

TARGET_HIGH = 100.0 # Best-case savings target (100% of original time)
TARGET_LOW = 75.0 # Mid-range savings target (75% of original time)
TARGET_MID = 87.5 # Balanced savings target (87.5% of original time)
MIN_E2E_PCT = 2.0 # Drop estimates whose best-case E2E impact is below this threshold
OVERLAP_EFFICIENCY = 0.85 # Memory/compute pipeline overlap fraction (0 = no overlap, 1 = perfect overlap)

_MATRIX_SPECS = frozenset(
    {
        "matrix_fp16",
        "matrix_bf16",
        "matrix_fp32",
        "matrix_fp64",
        "matrix_fp8",
        "matrix_int8",
    }
)

_NORM_TYPES = frozenset(
    {
        "LayerNorm",
        "BatchNorm",
        "GroupNorm",
        "InstanceNorm",
        "Norm",
    }
)


def build_kernel_perf_lookup(csv_path: str) -> Dict[str, Dict]:
    """
    Build GPU kernel name -> {shape -> perf metrics} lookup from unified_perf_summary.csv.

    Keyed by (kernel_name, input_shape) so the same kernel launched with
    different tensor shapes gets separate perf entries.
    """
    df = pd.read_csv(csv_path)
    lookup: Dict[str, Dict] = defaultdict(dict)
    for _, row in df.iterrows():
        kd = row.get("kernel_details_summary", "")
        if pd.isna(kd):
            continue
        kernel_names = re.findall(r"'name':\s*'([^']+)'", str(kd))
        shape = parse_first_shape(row.get("Input Dims"))
        for kn in kernel_names:
            if shape in lookup[kn]:
                continue
            entry = {}
            for col in ("Data Moved (MB)", "GFLOPS", "FLOPS/Byte", "Compute Spec"):
                val = row.get(col)
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    entry[col] = val
                else:
                    entry[col] = None
            lookup[kn][shape] = entry
    return dict(lookup)


def _is_matrix_op(kernel_info: dict) -> bool:
    """Check if a kernel uses matrix compute units (GEMM, conv, etc.)."""
    cspec = kernel_info.get("compute_spec")
    if cspec and cspec in _MATRIX_SPECS:
        return True
    ktype = kernel_info.get("type", "").upper()
    return "GEMM" in ktype or "CONV" in ktype


def _split_into_subgroups(enriched_kernels):
    """Split kernel sequence at GEMM boundaries into fusable sub-groups.

    Returns list of (subgroup_type, kernels) where subgroup_type is:
      "gemm_epilogue" - GEMM followed by elementwise ops
      "elementwise"   - consecutive non-GEMM ops
      "gemm_only"     - standalone GEMM with no trailing elementwise
    """
    subgroups = []
    current = []
    for k in enriched_kernels:
        if _is_matrix_op(k) and current:
            subgroups.append(current)
            current = [k]
        else:
            current.append(k)
    if current:
        subgroups.append(current)

    typed = []
    for sg in subgroups:
        has_gemm = any(_is_matrix_op(k) for k in sg)
        has_non_gemm = any(not _is_matrix_op(k) for k in sg)
        if has_gemm and has_non_gemm:
            typed.append(("gemm_epilogue", sg))
        elif has_gemm:
            typed.append(("gemm_only", sg))
        else:
            typed.append(("elementwise", sg))
    return typed


def _roofline_savings_us(enriched, peak_bw_bytes_s, vector_maf, matrix_maf, target_pct):
    """Compute roofline-projected savings for an elementwise sub-group.

    Blends between max (perfect overlap) and sum (no overlap) using OVERLAP_EFFICIENCY.
    """
    modeled = [e for e in enriched if e["has_perf_data"]]
    if not modeled:
        return 0.0

    first = modeled[0]
    last = modeled[-1]
    data_in_mb = first.get("data_in_mb")
    if data_in_mb is None:
        data_in_mb = first["data_moved_mb"] / 2.0
    data_out_mb = last.get("data_out_mb")
    if data_out_mb is None:
        data_out_mb = last["data_moved_mb"] / 2.0
    total_ext_data_bytes = (data_in_mb + data_out_mb) * 1e6

    vector_gflops = sum(
        e["gflops"] for e in modeled if e["gflops"] and not _is_matrix_op(e)
    )
    matrix_gflops = sum(
        e["gflops"] for e in modeled if e["gflops"] and _is_matrix_op(e)
    )

    if total_ext_data_bytes <= 0:
        return 0.0

    frac = target_pct / 100.0
    memory_time_us = total_ext_data_bytes / (peak_bw_bytes_s * frac) * 1e6
    matrix_time_us = (
        (matrix_gflops * 1e9) / (matrix_maf * 1e12 * frac) * 1e6
        if matrix_gflops > 0
        else 0.0
    )
    vector_time_us = (
        (vector_gflops * 1e9) / (vector_maf * 1e12 * frac) * 1e6
        if vector_gflops > 0
        else 0.0
    )

    fused_optimal = max(memory_time_us, matrix_time_us, vector_time_us)
    fused_sum = memory_time_us + matrix_time_us + vector_time_us
    fused_time_us = fused_optimal + (1.0 - OVERLAP_EFFICIENCY) * (fused_sum - fused_optimal)

    unmodeled_time_us = sum(e["dur_us"] for e in enriched if not e["has_perf_data"])
    current_us = sum(e["dur_us"] for e in enriched)
    return max(0.0, current_us - fused_time_us - unmodeled_time_us)


def compute_fusion_impact_estimates(
    candidates: List[dict],
    kernel_lookup: Dict[str, dict],
    peak_bw_tbs: float,
    peak_maf_tflops: dict,
    min_savings_ms: float = 0.1,
    baseline_ms: float = 0,
) -> List[dict]:
    """
    Deterministically compute kernel_fusion savings via roofline projection.

    Multi-GEMM candidates are split at GEMM boundaries into sub-groups:
      - GEMM + elementwise: savings = sum of elementwise dur_us (epilogue fusion)
      - Elementwise-only: roofline projection using data_in/data_out
      - Single GEMM: no fusion benefit (skipped)

    Kernels without perf models use their measured trace time as-is
    and a warning is emitted.
    """

    vector_maf = peak_maf_tflops.get(
        "vector_bf16", peak_maf_tflops.get("matrix_bf16", 1)
    )
    matrix_maf = peak_maf_tflops.get("matrix_bf16", 1)
    peak_bw_bytes_s = peak_bw_tbs * 1e12

    estimates: List[dict] = []

    for candidate in candidates:
        kernels = candidate.get("kernels", [])
        instance_count = candidate.get("instance_count", 1)
        if len(kernels) < 2:
            continue
        if candidate.get("has_fused_kernel", False):
            continue

        enriched = []
        for k in kernels:
            kname = k.get("name", k.get("kernel_name", ""))
            dur_us = k.get("dur_us", 0)
            perf = shape_aware_lookup(kernel_lookup, kname, candidate.get("input_dims"))
            dm = perf.get("Data Moved (MB)")
            gf = perf.get("GFLOPS")
            has_data = dm is not None
            enriched.append(
                {
                    "name": kname,
                    "type": k.get("type", k.get("kernel_type", "Unknown")),
                    "dur_us": dur_us,
                    "data_moved_mb": float(dm) if dm is not None else None,
                    "data_in_mb": k.get("data_in_mb"),
                    "data_out_mb": k.get("data_out_mb"),
                    "gflops": float(gf) if gf is not None else None,
                    "compute_spec": perf.get("Compute Spec"),
                    "has_perf_data": has_data,
                }
            )

        modeled = [e for e in enriched if e["has_perf_data"]]
        unmodeled_count = len(enriched) - len(modeled)

        if not modeled:
            continue

        if all(_is_matrix_op(e) for e in enriched):
            continue

        if any(e["name"].startswith("triton_") for e in enriched):
            continue

        non_matrix = [e for e in enriched if not _is_matrix_op(e)]
        if non_matrix and any(_is_matrix_op(e) for e in enriched):
            if all(e["type"] in _NORM_TYPES for e in non_matrix):
                continue

        modeled_frac = len(modeled) / len(enriched)
        min_frac = 0.5 if len(enriched) < 5 else 0.75
        if modeled_frac < min_frac:
            continue

        current_per_instance_us = sum(e["dur_us"] for e in enriched)
        if current_per_instance_us <= 0:
            continue

        subgroups = _split_into_subgroups(enriched)
        has_matrix_ops = any(_is_matrix_op(e) for e in enriched)

        savings_by_target = {}
        bound_type = "memory"
        for target_pct, label in [
            (TARGET_HIGH, "high"),
            (TARGET_MID, "mid"),
            (TARGET_LOW, "low"),
        ]:
            total_savings_us = 0.0
            for sg_type, sg_kernels in subgroups:
                if sg_type == "gemm_epilogue":
                    non_gemm_time = sum(
                        e["dur_us"] for e in sg_kernels if not _is_matrix_op(e)
                    )
                    total_savings_us += non_gemm_time
                elif sg_type == "elementwise":
                    sg_savings = _roofline_savings_us(
                        sg_kernels,
                        peak_bw_bytes_s,
                        vector_maf,
                        matrix_maf,
                        target_pct,
                    )
                    total_savings_us += sg_savings
            savings_by_target[label] = total_savings_us

        if savings_by_target["high"] > 0:
            sg_types = [t for t, _ in subgroups]
            if "elementwise" in sg_types:
                bound_type = "memory"
            elif "gemm_epilogue" in sg_types:
                bound_type = "compute"

        savings_mid = savings_by_target["mid"] * instance_count / 1000
        savings_high = savings_by_target["high"] * instance_count / 1000
        savings_low = savings_by_target["low"] * instance_count / 1000

        if savings_high < min_savings_ms:
            continue

        time_ms = current_per_instance_us * instance_count / 1000
        estimation = "full" if len(modeled) == len(enriched) else "partial"

        estimate: Dict[str, Any] = {
            "operation": candidate.get(
                "base_name", candidate.get("module_name", "Unknown")
            ),
            "category": "kernel_fusion",
            "type": "kernel_fusion",
            "savings_ms": round(savings_mid, 3),
            "savings_ms_low": round(savings_low, 3),
            "savings_ms_high": round(savings_high, 3),
            "estimation": estimation,
            "bound_type": bound_type,
            "time_ms": round(time_ms, 3),
            "instance_count": instance_count,
            "kernel_count": len(kernels),
            "modeled_kernel_count": len(modeled),
        }

        if has_matrix_ops:
            estimate["fusion_type"] = "matrix_compute"
        else:
            estimate["fusion_type"] = "memory_bound"

        if unmodeled_count > 0:
            estimate["warning"] = (
                f"{unmodeled_count} of {len(kernels)} kernels lack perf models; "
                f"savings estimate uses trace time for unmodeled kernels"
            )

        if baseline_ms > 0:
            e2e_pct_high = savings_high / baseline_ms * 100
            if e2e_pct_high < MIN_E2E_PCT:
                continue
            estimate["e2e_pct_low"] = round(savings_low / baseline_ms * 100, 2)
            estimate["e2e_pct_high"] = round(e2e_pct_high, 2)

        estimates.append(estimate)

    return sorted(estimates, key=lambda x: x["savings_ms"], reverse=True)


def load_fusion_data(output_dir: str):
    """Load fusion candidates and platform arch config from analysis output."""
    fc_path = os.path.join(output_dir, "category_data", "fusion_candidates.json")
    manifest_path = os.path.join(output_dir, "category_data", "category_manifest.json")
    csv_path = os.path.join(output_dir, "perf_report_csvs", "unified_perf_summary.csv")

    if not os.path.exists(fc_path):
        raise FileNotFoundError(f"Fusion candidates not found: {fc_path}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Perf summary CSV not found: {csv_path}")

    with open(fc_path, "r") as f:
        candidates = json.load(f)

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    return candidates, manifest, csv_path


def load_arch_config(output_dir: str, platform: str) -> dict:
    """Load platform arch config, trying metadata files first then arch JSON."""
    metadata_dir = os.path.join(output_dir, "metadata")
    if os.path.isdir(metadata_dir):
        for fname in os.listdir(metadata_dir):
            if fname.endswith("_metadata.json"):
                with open(os.path.join(metadata_dir, fname), "r") as f:
                    meta = json.load(f)
                if "peak_hbm_bw_tbs" in meta and "max_achievable_tflops" in meta:
                    return {
                        "peak_hbm_bw_tbs": meta["peak_hbm_bw_tbs"],
                        "max_achievable_tflops": meta["max_achievable_tflops"],
                    }

    arch_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "utils",
        "arch",
        f"{platform}.json",
    )
    if os.path.exists(arch_path):
        with open(arch_path, "r") as f:
            arch = json.load(f)
        return {
            "peak_hbm_bw_tbs": arch["mem_bw_gbps"] / 1000,
            "max_achievable_tflops": arch["max_achievable_tflops"],
        }

    raise FileNotFoundError(
        f"No arch config found for platform {platform} in metadata or arch JSON"
    )


def _filter_and_dedup(candidates: list) -> list:
    """Filter by kernel count cap and deduplicate nested duplicates.

    Two candidates with the same (instance_count, kernel_count, total_kernel_time_us)
    are the same set of kernels captured at different nesting levels in the call tree.
    Keep only the one with the shorter (more specific) module name.
    """
    filtered = [
        c for c in candidates if c.get("kernel_count", 0) <= MAX_FUSION_KERNEL_COUNT
    ]

    seen: dict = {}
    for c in filtered:
        sig = (
            c.get("instance_count", 0),
            c.get("kernel_count", 0),
            round(c.get("total_kernel_time_us", 0), 1),
        )
        if sig in seen:
            existing = seen[sig]
            if len(c.get("module_name", "")) < len(existing.get("module_name", "")):
                seen[sig] = c
        else:
            seen[sig] = c

    return list(seen.values())


def main():
    parser = argparse.ArgumentParser(
        description="Analyze kernel fusion candidates and estimate savings"
    )
    parser.add_argument("--output-dir", required=True, help="Analysis output directory")
    args = parser.parse_args()

    try:
        candidates, manifest, csv_path = load_fusion_data(args.output_dir)
    except FileNotFoundError as e:
        error_metrics = {
            "category": "kernel_fusion",
            "status": "ERROR",
            "error": str(e),
        }
        write_metrics_json(error_metrics, args.output_dir, "kernel_fusion")
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    raw_count = len(candidates)
    candidates = _filter_and_dedup(candidates)
    print(
        f"  Filtered: {raw_count} -> {len(candidates)} candidates "
        f"(max {MAX_FUSION_KERNEL_COUNT} kernels, deduped)"
    )

    if not candidates:
        metrics = {
            "category": "kernel_fusion",
            "status": "NO_DATA",
            "candidate_count": 0,
            "impact_estimates": [],
        }
        output_path = write_metrics_json(metrics, args.output_dir, "kernel_fusion")
        print(f"No fusion candidates found. Metrics written to: {output_path}")
        return

    platform = manifest.get("platform", "MI300X")
    try:
        arch = load_arch_config(args.output_dir, platform)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    peak_bw_tbs = arch["peak_hbm_bw_tbs"]
    peak_maf_tflops = arch["max_achievable_tflops"]
    baseline_ms = manifest.get("gpu_utilization", {}).get("total_time_ms", 0)

    kernel_lookup = build_kernel_perf_lookup(csv_path)

    impact_estimates = compute_fusion_impact_estimates(
        candidates,
        kernel_lookup,
        peak_bw_tbs,
        peak_maf_tflops,
        baseline_ms=baseline_ms,
    )

    total_time_us = sum(c.get("total_kernel_time_us", 0) for c in candidates)
    total_savings_ms = sum(e["savings_ms"] for e in impact_estimates)
    warnings = [e["warning"] for e in impact_estimates if e.get("warning")]

    metrics = {
        "category": "kernel_fusion",
        "status": "OK",
        "total_time_ms": round(total_time_us / 1000, 3),
        "candidate_count": len(candidates),
        "estimated_count": len(impact_estimates),
        "total_savings_ms": round(total_savings_ms, 3),
        "platform": platform,
        "peak_hbm_bw_tbs": peak_bw_tbs,
        "impact_estimates": impact_estimates,
    }

    if warnings:
        metrics["warnings"] = warnings

    output_path = write_metrics_json(metrics, args.output_dir, "kernel_fusion")
    print(f"Kernel fusion analysis complete:")
    print(f"  Candidates: {len(candidates)}")
    print(f"  With estimates: {len(impact_estimates)}")
    print(f"  Total savings (mid): {total_savings_ms:.3f} ms")
    if warnings:
        print(f"  Warnings: {len(warnings)}")
    print(f"  Metrics written to: {output_path}")


if __name__ == "__main__":
    main()
