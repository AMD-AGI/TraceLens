#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Kernel Fusion Analysis - impact_score Estimation

Computes ``impact_score`` (% of E2E GPU time recoverable by fusion) for kernel
fusion candidates by cross-referencing fusion_candidates.json with
unified_perf_summary.csv and projecting fused kernel time via roofline model.
"""

import argparse
import json
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ANALYSIS_DIR = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, _ANALYSIS_DIR)

from analysis_utils import load_perf_rows, perf_report_csv_dir, write_metrics_json
from utils.arch_utils import load_arch
from utils.classify_kernels import TAXONOMY
from utils.kernel_fusion import (
    OVERLAP_EFFICIENCY,
    FusionCandidate,
    FusionForms,
    FusionPricer,
)


def load_fusion_data(output_dir: str):
    """Load fusion candidates and the perf summary path from analysis output."""
    fc_path = os.path.join(output_dir, "category_data", "fusion_candidates.json")
    manifest_path = os.path.join(output_dir, "category_data", "category_manifest.json")

    if not os.path.exists(fc_path):
        raise FileNotFoundError(f"Fusion candidates not found: {fc_path}")

    with open(fc_path, "r") as f:
        candidates = [FusionCandidate.from_dict(c) for c in json.load(f)]

    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

    csv_path = os.path.join(perf_report_csv_dir(output_dir), "unified_perf_summary.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Perf summary CSV not found: {csv_path}")

    return candidates, manifest, csv_path


def load_arch_config(output_dir: str, platform: str) -> dict:
    """Load platform arch config, trying metadata files first then arch JSON."""
    metadata_dir = os.path.join(output_dir, "metadata")
    if os.path.isdir(metadata_dir):
        for fname in os.listdir(metadata_dir):
            if fname.endswith("_metadata.json"):
                with open(os.path.join(metadata_dir, fname), "r") as f:
                    meta = json.load(f)
                if "peak_mem_bw_tbs" in meta and "max_achievable_tflops" in meta:
                    return {
                        "peak_mem_bw_tbs": meta["peak_mem_bw_tbs"],
                        "max_achievable_tflops": meta["max_achievable_tflops"],
                    }

    try:
        arch = load_arch(platform)
        return {
            "peak_mem_bw_tbs": arch["mem_bw_gbps"] / 1000,
            "max_achievable_tflops": arch["max_achievable_tflops"],
        }
    except FileNotFoundError:
        raise FileNotFoundError(
            f"No arch config found for platform {platform} in metadata or arch JSON"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Analyze kernel fusion candidates and estimate savings"
    )
    parser.add_argument("--output-dir", required=True, help="Analysis output directory")
    parser.add_argument(
        "--comparison-scope",
        choices=("standalone", "comparative"),
        default="standalone",
        help="Analysis scope: standalone (default) or comparative",
    )
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

    baseline_ms = manifest.get("gpu_utilization", {}).get("total_time_ms", 0)
    is_comparative = args.comparison_scope == "comparative"
    # The standalone report note quotes the overlap even when nothing is found.
    scope_metrics = {} if is_comparative else {"overlap_efficiency": OVERLAP_EFFICIENCY}

    raw_count = len(candidates)
    candidates = FusionPricer.shortlist(candidates, baseline_ms)
    print(f"  Filtered: {raw_count} -> {len(candidates)} candidates")

    if not candidates:
        metrics = {
            "category": "kernel_fusion",
            "status": "NO_DATA",
            "candidate_count": 0,
            "impact_estimates": [],
            **scope_metrics,
        }
        output_path = write_metrics_json(metrics, args.output_dir, "kernel_fusion")
        print(f"No fusion candidates found. Metrics written to: {output_path}")
        return

    platform = manifest.get("platform", "MI300X")
    try:
        arch = load_arch_config(args.output_dir, platform)
    except FileNotFoundError as e:
        error_metrics = {
            "category": "kernel_fusion",
            "status": "ERROR",
            "error": str(e),
        }
        write_metrics_json(error_metrics, args.output_dir, "kernel_fusion")
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    peak_bw_tbs = arch["peak_mem_bw_tbs"]
    peak_maf_tflops = arch["max_achievable_tflops"]

    trace2_kwargs = {}
    if is_comparative:
        csv_path2 = os.path.join(
            args.output_dir, "perf_report_trace2_csvs", "unified_perf_summary.csv"
        )
        # Trace 2's perf CSV is optional; without it Trace 2 perf cells render as "—".
        trace2_kwargs = {
            "perf_rows2": (
                load_perf_rows(csv_path2, with_thread=False)
                if os.path.isfile(csv_path2)
                else {}
            ),
            "baseline2_ms": manifest.get("trace2_gpu_utilization", {}).get(
                "total_time_ms", 0
            ),
            "same_platform": manifest.get("platform2") == platform,
        }

    pricer = FusionPricer(
        FusionForms(TAXONOMY),
        (peak_bw_tbs, peak_maf_tflops),
        load_perf_rows(csv_path),
        baseline_ms,
        **trace2_kwargs,
    )
    priced = pricer.price_all(candidates)
    impact_estimates = [estimate for _, estimate in priced]

    total_time_us = sum(c.time_us for c in candidates)
    total_impact_score = sum(e.impact_score for e in impact_estimates)
    warnings = [e.warning for e in impact_estimates if e.warning]

    metrics = {
        "category": "kernel_fusion",
        "status": "OK",
        "total_time_ms": round(total_time_us / 1000, 3),
        "candidate_count": len(candidates),
        "estimated_count": len(impact_estimates),
        "total_impact_score": round(total_impact_score, 2),
        "platform": platform,
        "peak_mem_bw_tbs": peak_bw_tbs,
        "baseline_ms": baseline_ms,
        "impact_estimates": [e.to_dict() for e in impact_estimates],
        **scope_metrics,
    }

    if is_comparative:
        metrics["baseline2_ms"] = trace2_kwargs["baseline2_ms"]

    if warnings:
        metrics["warnings"] = warnings

    high_confidence_op_keys = pricer.suppressed_op_keys(priced)
    metrics["high_confidence_op_keys"] = high_confidence_op_keys

    output_path = write_metrics_json(metrics, args.output_dir, "kernel_fusion")
    print(f"Kernel fusion analysis complete:")
    print(f"  Candidates: {len(candidates)}")
    print(f"  With estimates: {len(impact_estimates)}")
    print(f"  Total impact_score (mid): {total_impact_score:.2f}")
    high_count = sum(1 for e in impact_estimates if e.confidence == "high")
    print(f"  High confidence: {high_count}, op keys: {len(high_confidence_op_keys)}")
    if warnings:
        print(f"  Warnings: {len(warnings)}")
    print(f"  Metrics written to: {output_path}")


if __name__ == "__main__":
    main()
