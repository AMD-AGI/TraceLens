#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
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
import re
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_STANDALONE_DIR = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, _STANDALONE_DIR)

from analysis_utils import (
    TARGET_HIGH,
    TARGET_LOW,
    TARGET_MID,
    parse_first_shape,
    perf_report_csv_dir,
    shape_aware_lookup,
    write_metrics_json,
)
from utils.arch_utils import load_arch

MAX_FUSION_KERNEL_COUNT = (
    15  # Skip candidates with more kernels than this (too complex to fuse reliably)
)

MIN_IMPACT_SCORE = (
    2.0  # Drop estimates whose best-case impact_score is below this threshold
)
OVERLAP_EFFICIENCY = 0.85  # Memory/compute pipeline overlap fraction (0 = no overlap, 1 = perfect overlap)

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

_NORM_NAME_PATTERNS = [
    "batchnorm",
    "layernorm",
    "groupnorm",
    "instancenorm",
    "miopenbatchnorm",
    "rmsnorm",
]

_CONFIDENCE_NAME_HINTS = {
    "attention": ("attention", "sdpa", "self_attn"),
    "norm": (
        "rmsnorm",
        "rms_norm",
        "layernorm",
        "layer_norm",
        "batchnorm",
        "batch_norm",
    ),
    "mlp": ("mlp",),
    "rope": ("rotary", "rope", "apply_rotary"),
    "siglu": ("silu", "swiglu"),
}


def _is_norm_kernel(kernel_info: dict) -> bool:
    """Check if a kernel is a normalization op by type or kernel name."""
    if kernel_info.get("type", "") in _NORM_TYPES:
        return True
    name = kernel_info.get("name", kernel_info.get("kernel_name", "")).lower()
    return any(p in name for p in _NORM_NAME_PATTERNS)


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
    fused_time_us = fused_optimal + (1.0 - OVERLAP_EFFICIENCY) * (
        fused_sum - fused_optimal
    )

    unmodeled_time_us = sum(e["dur_us"] for e in enriched if not e["has_perf_data"])
    current_us = sum(e["dur_us"] for e in enriched)
    return max(0.0, current_us - fused_time_us - unmodeled_time_us)


def _classify_confidence(candidate: dict, enriched: list) -> str:
    """Classify fusion confidence from enriched kernel data + module name.

    High   = module name matches a known pattern AND kernel composition confirms it.
    Medium = one signal matches.
    Low    = speculative.
    """
    bn = candidate.get("base_name", "").lower()
    mn = candidate.get("module_name", "").lower()

    name_signal = next(
        (
            p
            for p, hints in _CONFIDENCE_NAME_HINTS.items()
            if any(h in bn or h in mn for h in hints)
        ),
        None,
    )

    n_gemm = sum(1 for e in enriched if _is_matrix_op(e))
    has_softmax = any("softmax" in e["name"].lower() for e in enriched)
    has_rsqrt = any("rsqrt" in e["name"].lower() for e in enriched)
    has_neg_cat = any("neg" in e["name"].lower() for e in enriched) and any(
        "catarray" in e["name"].lower() or "cat_" in e["name"].lower() for e in enriched
    )
    n_non_gemm = len(enriched) - n_gemm

    comp_signal = None
    # Unfused attention: QK^T (GEMM) + Softmax + PV (GEMM) → minimum 2 GEMMs
    if n_gemm >= 2 and has_softmax:
        comp_signal = "attention"
    # rsqrt is shared by RMSNorm, LayerNorm, GroupNorm — all valid fusion targets
    elif has_rsqrt:
        comp_signal = "norm"
    elif n_gemm >= 2 and n_non_gemm >= 1:
        comp_signal = "mlp"
    elif has_neg_cat:
        comp_signal = "rope"
    elif n_gemm == 0 and len(enriched) >= 3:
        comp_signal = "elementwise_chain"

    if name_signal and comp_signal:
        return "high"
    if name_signal or comp_signal:
        return "medium"
    return "low"


def _comparative_estimate(
    candidate: dict,
    min_savings_ms: float,
    baseline_ms: float,
) -> Optional[Dict[str, Any]]:
    """Compute a single impact estimate for a comparative fusion candidate.

    Savings = measured GPU time gap between trace1 and trace2 kernels.
    Returns None if the candidate should be skipped.
    """
    op_name = candidate.get("base_name", candidate.get("module_name", "Unknown"))
    instance_count = candidate.get("instance_count", 1)

    t1_kernels = candidate.get("kernels_trace1", [])
    if not t1_kernels:
        return None

    total_t1_us = candidate.get("total_kernel_time_us_trace1", 0)
    total_t2_us = candidate.get("total_kernel_time_us_trace2", 0)
    gap_us = total_t1_us - total_t2_us
    if gap_us <= 0:
        return None

    savings_high = gap_us / 1000
    if savings_high < min_savings_ms:
        return None

    savings_low = (TARGET_LOW / TARGET_HIGH) * savings_high
    savings_mid = (TARGET_MID / TARGET_HIGH) * savings_high
    time_ms = total_t1_us / 1000

    impact_score_high = savings_high / baseline_ms * 100
    impact_score_low = savings_low / baseline_ms * 100
    impact_score_mid = savings_mid / baseline_ms * 100

    if impact_score_high < MIN_IMPACT_SCORE:
        return None

    has_matrix_ops = any(_is_matrix_op({"type": k.get("type", "")}) for k in t1_kernels)
    enriched_t1 = [
        {"name": k.get("name", ""), "type": k.get("type", "")} for k in t1_kernels
    ]

    estimate: Dict[str, Any] = {
        "operation": op_name,
        "category": "kernel_fusion",
        "type": "kernel_fusion",
        "impact_score": round(impact_score_mid, 2),
        "impact_score_low": round(impact_score_low, 2),
        "impact_score_high": round(impact_score_high, 2),
        "estimation": "measured",
        "bound_type": "compute" if has_matrix_ops else "memory",
        "time_ms": round(time_ms, 3),
        "instance_count": instance_count,
        "kernel_count": len(t1_kernels),
        "modeled_kernel_count": len(t1_kernels),
        "delta_kernel_count": candidate.get("delta", 0),
        "confidence": _classify_confidence(candidate, enriched_t1),
        "affected_gpu_kernels": [
            k.get("name", k.get("kernel_name", "")) for k in t1_kernels
        ],
        "fusion_type": "matrix_compute" if has_matrix_ops else "memory_bound",
    }

    return estimate


def _standalone_estimate(
    candidate: dict,
    kernel_lookup: Dict[str, dict],
    peak_bw_bytes_s: float,
    vector_maf: float,
    matrix_maf: float,
    min_savings_ms: float,
    baseline_ms: float,
) -> Optional[Dict[str, Any]]:
    """Compute a single impact estimate for a standalone fusion candidate.

    Savings are projected via roofline model — elementwise epilogue time saved
    if kernels were fused, or roofline-capped savings for elementwise-only groups.
    Returns None if the candidate should be skipped.
    """
    op_name = candidate.get("base_name", candidate.get("module_name", "Unknown"))
    instance_count = candidate.get("instance_count", 1)

    kernels = candidate.get("kernels", [])
    if len(kernels) < 2:
        return None
    if candidate.get("has_fused_kernel", False):
        return None

    enriched = []
    for k in kernels:
        kname = k.get("name", k.get("kernel_name", ""))
        dur_us = k.get("dur_us", 0)
        perf = shape_aware_lookup(kernel_lookup, kname, candidate.get("input_dims"))
        dm = perf.get("Data Moved (MB)")
        gf = perf.get("GFLOPS")
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
                "has_perf_data": dm is not None,
            }
        )

    modeled = [e for e in enriched if e["has_perf_data"]]
    unmodeled_count = len(enriched) - len(modeled)

    if not modeled:
        return None
    if all(_is_matrix_op(e) for e in enriched):
        return None
    if any(e["name"].startswith("triton_") for e in enriched):
        return None

    non_matrix = [e for e in enriched if not _is_matrix_op(e)]
    if non_matrix and any(_is_matrix_op(e) for e in enriched):
        if all(_is_norm_kernel(e) for e in non_matrix):
            return None

    modeled_frac = len(modeled) / len(enriched)
    min_frac = 0.5 if len(enriched) < 5 else 0.75
    if modeled_frac < min_frac:
        return None

    current_per_instance_us = sum(e["dur_us"] for e in enriched)
    if current_per_instance_us <= 0:
        return None

    subgroups = _split_into_subgroups(enriched)
    has_matrix_ops = any(_is_matrix_op(e) for e in enriched)

    total_savings_us = 0.0
    for sg_type, sg_kernels in subgroups:
        if sg_type == "gemm_epilogue":
            non_gemm_time = sum(e["dur_us"] for e in sg_kernels if not _is_matrix_op(e))
            total_savings_us += non_gemm_time
        elif sg_type == "elementwise":
            sg_savings = _roofline_savings_us(
                sg_kernels,
                peak_bw_bytes_s,
                vector_maf,
                matrix_maf,
                TARGET_HIGH,
            )
            total_savings_us += sg_savings

    sg_types = [t for t, _ in subgroups]
    if "gemm_epilogue" in sg_types:
        bound_type = "compute"
    else:
        bound_type = "memory"

    gap_high = total_savings_us / current_per_instance_us
    gap_low = (TARGET_LOW / TARGET_HIGH) * gap_high
    gap_mid = (TARGET_MID / TARGET_HIGH) * gap_high

    savings_high_ms = total_savings_us * instance_count / 1000

    if savings_high_ms < min_savings_ms:
        return None

    time_ms = current_per_instance_us * instance_count / 1000
    impact_score_high = gap_high * time_ms / baseline_ms * 100
    impact_score_low = gap_low * time_ms / baseline_ms * 100
    impact_score_mid = gap_mid * time_ms / baseline_ms * 100

    if impact_score_high < MIN_IMPACT_SCORE:
        return None

    estimation = "full" if len(modeled) == len(enriched) else "partial"

    estimate: Dict[str, Any] = {
        "operation": op_name,
        "category": "kernel_fusion",
        "type": "kernel_fusion",
        "impact_score": round(impact_score_mid, 2),
        "impact_score_low": round(impact_score_low, 2),
        "impact_score_high": round(impact_score_high, 2),
        "estimation": estimation,
        "bound_type": bound_type,
        "time_ms": round(time_ms, 3),
        "instance_count": instance_count,
        "kernel_count": len(kernels),
        "modeled_kernel_count": len(modeled),
        "confidence": _classify_confidence(candidate, enriched),
        "affected_gpu_kernels": [e["name"] for e in enriched],
        "fusion_type": "matrix_compute" if has_matrix_ops else "memory_bound",
    }

    if unmodeled_count > 0:
        estimate["warning"] = (
            f"{unmodeled_count} of {len(kernels)} kernels lack perf models; "
            f"savings estimate uses trace time for unmodeled kernels"
        )

    return estimate


def compute_fusion_impact_estimates(
    candidates: List[dict],
    kernel_lookup: Dict[str, dict],
    peak_bw_tbs: float,
    peak_maf_tflops: dict,
    min_savings_ms: float = 0.1,
    baseline_ms: float = 0,
    is_comparative: bool = False,
) -> List[dict]:
    """Compute kernel_fusion savings for all candidates.

    Dispatches each candidate to the appropriate estimator:
      - Comparative: _comparative_estimate (measured GPU time delta)
      - Standalone:  _standalone_estimate (roofline projection)

    Deterministically compute kernel_fusion ``impact_score`` via roofline
    projection or gap between traces. ``impact_score`` provides an estimate of impact on gpu runtime.

    For standalone mode, multi-GEMM candidates are split at GEMM boundaries into sub-groups:
      - GEMM + elementwise: recoverable us = sum of elementwise dur_us (epilogue fusion)
      - Elementwise-only: roofline projection using data_in/data_out
      - Single GEMM: no fusion benefit (skipped)

    Kernels without perf models use their measured trace time as-is
    and a warning is emitted.

    Each estimate includes a deterministic ``confidence`` level (high / medium /
    low) and the list of GPU kernel names (``affected_gpu_kernels``) so that
    downstream compute-category scripts can skip fusion-covered operations.

    If ``baseline_ms`` is missing or non-positive, ``impact_score`` is undefined;
    this function returns an empty list and emits a stderr warning.
    """
    if baseline_ms is None or baseline_ms <= 0:
        print(
            f"[kernel_fusion_analysis.compute_fusion_impact_estimates] "
            f"baseline_ms={baseline_ms!r} is invalid; skipping impact_score "
            f"computation",
            file=sys.stderr,
        )
        return []

    vector_maf = peak_maf_tflops.get(
        "vector_bf16", peak_maf_tflops.get("matrix_bf16", 1)
    )
    matrix_maf = peak_maf_tflops.get("matrix_bf16", 1)
    peak_bw_bytes_s = peak_bw_tbs * 1e12

    estimates: List[dict] = []
    for candidate in candidates:
        if is_comparative:
            estimate = _comparative_estimate(candidate, min_savings_ms, baseline_ms)
        else:
            estimate = _standalone_estimate(
                candidate,
                kernel_lookup,
                peak_bw_bytes_s,
                vector_maf,
                matrix_maf,
                min_savings_ms,
                baseline_ms,
            )
        if estimate is not None:
            estimates.append(estimate)

    return sorted(estimates, key=lambda x: x["impact_score"], reverse=True)


def load_fusion_data(output_dir: str):
    """Load fusion candidates and platform arch config from analysis output."""
    fc_path = os.path.join(output_dir, "category_data", "fusion_candidates.json")
    manifest_path = os.path.join(output_dir, "category_data", "category_manifest.json")

    if not os.path.exists(fc_path):
        raise FileNotFoundError(f"Fusion candidates not found: {fc_path}")

    with open(fc_path, "r") as f:
        candidates = json.load(f)

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
                if "peak_hbm_bw_tbs" in meta and "max_achievable_tflops" in meta:
                    return {
                        "peak_hbm_bw_tbs": meta["peak_hbm_bw_tbs"],
                        "max_achievable_tflops": meta["max_achievable_tflops"],
                    }

    try:
        arch = load_arch(platform)
        return {
            "peak_hbm_bw_tbs": arch["mem_bw_gbps"] / 1000,
            "max_achievable_tflops": arch["max_achievable_tflops"],
        }
    except FileNotFoundError:
        raise FileNotFoundError(
            f"No arch config found for platform {platform} in metadata or arch JSON"
        )


def _filter_and_dedup(
    candidates: list, baseline_ms: float = 0, is_comparative: bool = False
) -> list:
    """Filter by kernel count cap, drop tiny candidates, and deduplicate.

    Candidates whose total time can't reach MIN_IMPACT_SCORE of baseline are
    dropped early to avoid expensive downstream enrichment.
    """

    def _kernel_count(c: dict) -> int:
        if is_comparative:
            return c.get("kernel_count_trace1", 0)
        else:
            return c.get("kernel_count", 0)

    def _name(c: dict) -> str:
        return c.get("module_name", "")

    def _total_time_us(c: dict) -> float:
        if is_comparative:
            return c.get("total_kernel_time_us_trace1", 0)
        else:
            return c.get("total_kernel_time_us", 0)

    filtered = [c for c in candidates if _kernel_count(c) <= MAX_FUSION_KERNEL_COUNT]

    if baseline_ms > 0:
        min_time_us = baseline_ms * 10 * MIN_IMPACT_SCORE
        filtered = [c for c in filtered if _total_time_us(c) >= min_time_us]

    seen: dict = {}
    for c in filtered:
        sig = (
            c.get("instance_count", 0),
            _kernel_count(c),
            round(_total_time_us(c), 1),
        )
        if sig in seen:
            existing = seen[sig]
            if len(_name(c)) < len(_name(existing)):
                seen[sig] = c
        else:
            seen[sig] = c

    return list(seen.values())


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

    raw_count = len(candidates)
    candidates = _filter_and_dedup(
        candidates, baseline_ms=baseline_ms, is_comparative=is_comparative
    )
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
        error_metrics = {
            "category": "kernel_fusion",
            "status": "ERROR",
            "error": str(e),
        }
        write_metrics_json(error_metrics, args.output_dir, "kernel_fusion")
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    peak_bw_tbs = arch["peak_hbm_bw_tbs"]
    peak_maf_tflops = arch["max_achievable_tflops"]

    kernel_lookup = build_kernel_perf_lookup(csv_path)

    impact_estimates = compute_fusion_impact_estimates(
        candidates,
        kernel_lookup,
        peak_bw_tbs,
        peak_maf_tflops,
        baseline_ms=baseline_ms,
        is_comparative=is_comparative,
    )

    if is_comparative:
        total_time_us = sum(c.get("total_kernel_time_us_trace1", 0) for c in candidates)
    else:
        total_time_us = sum(c.get("total_kernel_time_us", 0) for c in candidates)
    total_impact_score = sum(e.get("impact_score", 0) for e in impact_estimates)
    warnings = [e["warning"] for e in impact_estimates if e.get("warning")]

    metrics = {
        "category": "kernel_fusion",
        "status": "OK",
        "total_time_ms": round(total_time_us / 1000, 3),
        "candidate_count": len(candidates),
        "estimated_count": len(impact_estimates),
        "total_impact_score": round(total_impact_score, 2),
        "platform": platform,
        "peak_hbm_bw_tbs": peak_bw_tbs,
        "impact_estimates": impact_estimates,
    }

    if warnings:
        metrics["warnings"] = warnings

    high_confidence_kernel_map: Dict[str, str] = {}
    for est in impact_estimates:
        if est.get("confidence") == "high":
            op_name = est["operation"]
            for kn in est.get("affected_gpu_kernels", []):
                if kn:
                    high_confidence_kernel_map[kn] = op_name
    metrics["high_confidence_kernel_map"] = high_confidence_kernel_map

    output_path = write_metrics_json(metrics, args.output_dir, "kernel_fusion")
    print(f"Kernel fusion analysis complete:")
    print(f"  Candidates: {len(candidates)}")
    print(f"  With estimates: {len(impact_estimates)}")
    print(f"  Total impact_score (mid): {total_impact_score:.2f}")
    high_count = sum(1 for e in impact_estimates if e.get("confidence") == "high")
    print(
        f"  High confidence: {high_count}, kernel map entries: {len(high_confidence_kernel_map)}"
    )
    if warnings:
        print(f"  Warnings: {len(warnings)}")
    print(f"  Metrics written to: {output_path}")


if __name__ == "__main__":
    main()
