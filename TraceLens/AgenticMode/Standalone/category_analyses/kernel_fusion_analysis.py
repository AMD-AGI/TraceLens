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
from typing import Any, Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analysis_utils import write_metrics_json

MAX_FUSION_KERNEL_COUNT = 15
TARGET_HIGH = 100.0
TARGET_LOW = 75.0
TARGET_MID = 87.5

_MATRIX_SPECS = frozenset({
    "matrix_fp16", "matrix_bf16", "matrix_fp32", "matrix_fp64",
    "matrix_fp8", "matrix_int8",
})


def build_kernel_perf_lookup(csv_path: str) -> Dict[str, dict]:
    """
    Build GPU kernel name -> perf metrics lookup from unified_perf_summary.csv.

    Maps raw GPU kernel names (found in kernel_details_summary) to their
    per-invocation perf data (Data Moved, GFLOPS, FLOPS/Byte, Compute Spec).
    """
    df = pd.read_csv(csv_path)
    lookup: Dict[str, dict] = {}
    for _, row in df.iterrows():
        kd = row.get("kernel_details_summary", "")
        if pd.isna(kd):
            continue
        kernel_names = re.findall(r"'name':\s*'([^']+)'", str(kd))
        for kn in kernel_names:
            if kn in lookup:
                continue
            entry = {}
            for col in ("Data Moved (MB)", "GFLOPS", "FLOPS/Byte", "Compute Spec"):
                val = row.get(col)
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    entry[col] = val
                else:
                    entry[col] = None
            lookup[kn] = entry
    return lookup


def _is_matrix_op(kernel_info: dict) -> bool:
    """Check if a kernel uses matrix compute units (GEMM, conv, etc.)."""
    cspec = kernel_info.get("compute_spec")
    if cspec and cspec in _MATRIX_SPECS:
        return True
    ktype = kernel_info.get("type", "").upper()
    return "GEMM" in ktype or "CONV" in ktype


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

    Models the fused kernel as a single operation and projects its time:
      Total_Data = Data_In(first_op) + Data_Out(last_op)
      Total_FLOPs = Sum(GFLOPS_i) across all constituent kernels
      Bound type determined by arithmetic intensity vs ridge point.

    FLOPs are split by compute type (matrix vs vector) and projected at
    the appropriate peak MAF for each. Fused time is the max of compute
    time and memory time.

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
            perf = kernel_lookup.get(kname, {})
            dm = perf.get("Data Moved (MB)")
            gf = perf.get("GFLOPS")
            has_data = dm is not None
            enriched.append({
                "name": kname,
                "type": k.get("type", k.get("kernel_type", "Unknown")),
                "dur_us": dur_us,
                "data_moved_mb": float(dm) if dm is not None else None,
                "gflops": float(gf) if gf is not None else None,
                "compute_spec": perf.get("Compute Spec"),
                "has_perf_data": has_data,
            })

        modeled = [e for e in enriched if e["has_perf_data"]]
        unmodeled_count = len(enriched) - len(modeled)

        if not modeled:
            continue

        if all(_is_matrix_op(e) for e in enriched):
            continue

        modeled_frac = len(modeled) / len(enriched)
        if modeled_frac < 0.75:
            continue

        unmodeled_time_us = sum(
            e["dur_us"] for e in enriched if not e["has_perf_data"]
        )
        current_per_instance_us = sum(e["dur_us"] for e in enriched)
        if current_per_instance_us <= 0:
            continue

        first_dm = modeled[0]["data_moved_mb"]
        last_dm = modeled[-1]["data_moved_mb"]

        # TO DO: Approximate data movement model for the first and last kernel.
        data_in_mb = first_dm / 2.0
        data_out_mb = last_dm / 2.0
        total_ext_data_bytes = (data_in_mb + data_out_mb) * 1e6

        matrix_gflops = sum(
            e["gflops"] for e in modeled
            if e["gflops"] and _is_matrix_op(e)
        )
        vector_gflops = sum(
            e["gflops"] for e in modeled
            if e["gflops"] and not _is_matrix_op(e)
        )
        total_flops = (matrix_gflops + vector_gflops) * 1e9

        has_matrix_ops = matrix_gflops > 0

        if total_ext_data_bytes > 0 and total_flops > 0:
            ai = total_flops / total_ext_data_bytes
            effective_peak_flops_s = (
                matrix_maf * 1e12 if has_matrix_ops else vector_maf * 1e12
            )
            ridge = effective_peak_flops_s / peak_bw_bytes_s
            bound_type = "compute" if ai > ridge else "memory"
        elif total_ext_data_bytes > 0:
            bound_type = "memory"
        else:
            continue

        savings_by_target = {}
        for target_pct, label in [
            (TARGET_HIGH, "high"),
            (TARGET_MID, "mid"),
            (TARGET_LOW, "low"),
        ]:
            frac = target_pct / 100.0
            matrix_time_us = (
                (matrix_gflops * 1e9) / (matrix_maf * 1e12 * frac) * 1e6
                if matrix_gflops > 0 else 0.0
            )
            vector_time_us = (
                (vector_gflops * 1e9) / (vector_maf * 1e12 * frac) * 1e6
                if vector_gflops > 0 else 0.0
            )
            compute_time_us = max(matrix_time_us, vector_time_us)
            memory_time_us = (
                total_ext_data_bytes / (peak_bw_bytes_s * frac) * 1e6
            )

            fused_time_us = max(compute_time_us, memory_time_us)
            projected_us = fused_time_us + unmodeled_time_us
            savings_by_target[label] = max(
                0.0, current_per_instance_us - projected_us
            )

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
            estimate["e2e_pct_low"] = round(
                savings_low / baseline_ms * 100, 2
            )
            estimate["e2e_pct_high"] = round(
                savings_high / baseline_ms * 100, 2
            )

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
        "utils", "arch", f"{platform}.json",
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
        c for c in candidates
        if c.get("kernel_count", 0) <= MAX_FUSION_KERNEL_COUNT
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
    print(f"  Filtered: {raw_count} -> {len(candidates)} candidates "
          f"(max {MAX_FUSION_KERNEL_COUNT} kernels, deduped)")

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
