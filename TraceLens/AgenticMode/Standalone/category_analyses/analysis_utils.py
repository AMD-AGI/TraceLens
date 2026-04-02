#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Shared utilities for category-specific analysis scripts.

This module provides common functions for:
- Loading category data (CSV, metadata JSON)
- Calculating time metrics
- Calculating efficiency metrics
- Building operation metrics for JSON output
- Writing metrics JSON files
"""

import ast
import json
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def load_category_data(output_dir: str, category: str) -> Tuple[pd.DataFrame, dict]:
    """
    Load CSV operations data and metadata JSON for a category.

    Args:
        output_dir: Base output directory path
        category: Category name (e.g., 'gemm', 'sdpa_fwd', 'elementwise')

    Returns:
        Tuple of (operations DataFrame, metadata dict)

    Raises:
        FileNotFoundError: If required files don't exist
    """
    csv_path = f"{output_dir}/category_data/{category}_ops.csv"
    metadata_path = f"{output_dir}/metadata/{category}_metadata.json"

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Category CSV not found: {csv_path}")

    ops_df = pd.read_csv(csv_path)

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    return ops_df, metadata


def calculate_time_metrics(ops_df: pd.DataFrame, metadata: dict) -> dict:
    """
    Calculate total time and percentage of compute for a category.

    Args:
        ops_df: Operations DataFrame with 'Kernel Time (µs)_sum' column
        metadata: Metadata dict with gpu_utilization info

    Returns:
        Dict with total_time_ms, percent_of_compute, operation_count
    """
    if "Kernel Time (µs)_sum" in ops_df.columns:
        total_time_us = ops_df["Kernel Time (µs)_sum"].sum()
        total_time_ms = total_time_us / 1000
        total_compute_ms = metadata.get("gpu_utilization", {}).get("total_time_ms", 1)
        percent_of_compute = (
            (total_time_ms / total_compute_ms) * 100 if total_compute_ms > 0 else 0
        )
    else:
        total_time_ms = 0
        percent_of_compute = 0

    return {
        "total_time_ms": round(total_time_ms, 3),
        "percent_of_compute": round(percent_of_compute, 2),
        "operation_count": (
            int(ops_df["operation_count"].sum())
            if "operation_count" in ops_df.columns
            else len(ops_df)
        ),
    }


def validate_efficiency(
    achieved: float, peak: float, metric_name: str
) -> Dict[str, Any]:
    """
    Validate efficiency calculation and flag anomalies.

    Efficiency values > 100% indicate measurement issues or incorrect peak specs.
    These should be flagged as anomalies, not used to claim "excellent performance".

    Args:
        achieved: Achieved performance value (TFLOPS or TB/s)
        peak: Peak performance value (TFLOPS or TB/s)
        metric_name: Name of the metric for warning messages

    Returns:
        Dict with:
            - value: Efficiency percentage (or None if invalid)
            - warning: Warning message if anomaly detected (or None)
            - is_anomaly: Boolean indicating if this is an anomalous value
    """
    if peak <= 0:
        return {
            "value": None,
            "warning": f"{metric_name}: Invalid peak value ({peak})",
            "is_anomaly": True,
        }

    if achieved is None or np.isnan(achieved):
        return {"value": None, "warning": None, "is_anomaly": False}

    efficiency = (achieved / peak) * 100

    if efficiency > 110:
        return {
            "value": round(efficiency, 2),
            "warning": f"[ANOMALY] {metric_name} exceeds peak by {efficiency-100:.1f}% - verify measurement or peak spec",
            "is_anomaly": True,
        }
    elif efficiency > 100:
        return {
            "value": round(efficiency, 2),
            "warning": f"[WARNING] {metric_name} slightly exceeds peak ({efficiency:.1f}%) - may be within measurement error",
            "is_anomaly": False,
        }

    return {"value": round(efficiency, 2), "warning": None, "is_anomaly": False}


def calculate_efficiency_with_validation(
    achieved_tflops: Optional[float],
    achieved_tbps: Optional[float],
    peak_maf: float,
    peak_hbm_bw: float,
) -> Dict[str, Any]:
    """
    Calculate both compute and memory efficiency with validation.

    Args:
        achieved_tflops: Achieved TFLOPS
        achieved_tbps: Achieved TB/s
        peak_maf: Peak MAF in TFLOPS
        peak_hbm_bw: Peak HBM bandwidth in TB/s

    Returns:
        Dict with efficiency values and any warnings
    """
    compute_result = validate_efficiency(
        achieved_tflops, peak_maf, "Compute efficiency"
    )

    memory_result = validate_efficiency(achieved_tbps, peak_hbm_bw, "Memory bandwidth")

    warnings = []
    if compute_result["warning"]:
        warnings.append(compute_result["warning"])
    if memory_result["warning"]:
        warnings.append(memory_result["warning"])

    return {
        "compute_efficiency_pct": compute_result["value"],
        "compute_is_anomaly": compute_result["is_anomaly"],
        "memory_efficiency_pct": memory_result["value"],
        "memory_is_anomaly": memory_result["is_anomaly"],
        "warnings": warnings if warnings else None,
    }


def _resolve_peak_maf(row, max_achievable_tflops: dict, fallback_maf: float) -> float:
    """Resolve the correct peak MAF for an operation using Compute Spec.

    Uses the Compute Spec column to look up the precision-aware peak from
    max_achievable_tflops. Falls back to matrix_bf16 if Compute Spec is
    missing or unknown.
    """
    compute_spec = row.get("Compute Spec", "")
    if (
        compute_spec
        and isinstance(compute_spec, str)
        and compute_spec in max_achievable_tflops
    ):
        return max_achievable_tflops[compute_spec]
    return fallback_maf


def calculate_efficiency(
    row: pd.Series, peak_hbm_bw: float, peak_maf_or_maf_dict
) -> Dict[str, Optional[float]]:
    """
    Extract efficiency metrics for an operation from pre-computed CSV columns.

    Args:
        row: DataFrame row with operation metrics
        peak_hbm_bw: Peak HBM bandwidth in TB/s
        peak_maf_or_maf_dict: Either a float or a dict (max_achievable_tflops)
            for resolving the precision-aware peak

    Returns:
        Dict with tflops_achieved, tb_s_achieved, efficiency_percent, bound_type, warning
    """
    result = {
        "tflops_achieved": None,
        "tb_s_achieved": None,
        "efficiency_percent": None,
        "bound_type": None,
        "flops_per_byte": None,
        "compute_spec": None,
        "resolved_peak_maf": None,
        "resolved_peak_hbm_bw": None,
        "warning": None,
        "is_anomaly": False,
    }

    flops_byte = row.get("FLOPS/Byte", 0) if not pd.isna(row.get("FLOPS/Byte")) else 0
    result["flops_per_byte"] = round(flops_byte, 2) if flops_byte else None

    tflops_s = (
        row.get("TFLOPS/s_mean") if not pd.isna(row.get("TFLOPS/s_mean")) else None
    )
    tb_s = row.get("TB/s_mean") if not pd.isna(row.get("TB/s_mean")) else None

    compute_spec = row.get("Compute Spec", "")
    if compute_spec and isinstance(compute_spec, str):
        result["compute_spec"] = compute_spec

    if tflops_s is not None:
        result["tflops_achieved"] = round(tflops_s, 2)
    if tb_s is not None:
        result["tb_s_achieved"] = round(tb_s, 2)

    if isinstance(peak_maf_or_maf_dict, dict):
        fallback_maf = peak_maf_or_maf_dict.get("matrix_bf16", 1)
        peak_maf = _resolve_peak_maf(row, peak_maf_or_maf_dict, fallback_maf)
    else:
        peak_maf = peak_maf_or_maf_dict
    result["resolved_peak_maf"] = round(peak_maf, 2) if peak_maf else None
    result["resolved_peak_hbm_bw"] = round(peak_hbm_bw, 2) if peak_hbm_bw else None

    if flops_byte:
        balance_point = peak_maf / peak_hbm_bw
        result["bound_type"] = "compute" if flops_byte > balance_point else "memory"

    pct_roofline = row.get("Pct Roofline_mean")
    if pct_roofline is not None and not pd.isna(pct_roofline):
        result["efficiency_percent"] = round(float(pct_roofline), 2)
        if result["efficiency_percent"] > 110:
            result["warning"] = (
                f"[ANOMALY] Pct Roofline exceeds 110% ({result['efficiency_percent']:.1f}%) - verify measurement"
            )
            result["is_anomaly"] = True

    return result


def build_operation_metrics(
    ops_df: pd.DataFrame, metadata: dict, category_config: dict
) -> List[dict]:
    """
    Build list of operation metrics for JSON output.

    Args:
        ops_df: Operations DataFrame
        metadata: Metadata dict with peak performance values
        category_config: Category-specific configuration with:
            - extra_fields: Additional fields to extract (optional)
            - operation_classifier: Function to classify operations (optional)

    Returns:
        List of operation metric dicts
    """
    peak_hbm_bw = metadata.get("peak_hbm_bw_tbs", 1)
    maf = metadata.get("max_achievable_tflops", metadata.get("peak_bf16_maf_tflops", 1))

    # Calculate total time for percentage calculations
    total_time_ms = 0
    if "Kernel Time (µs)_sum" in ops_df.columns:
        total_time_ms = ops_df["Kernel Time (µs)_sum"].sum() / 1000

    operations = []

    # Sort by time descending
    if "Kernel Time (µs)_sum" in ops_df.columns:
        sorted_df = ops_df.nlargest(len(ops_df), "Kernel Time (µs)_sum")
    else:
        sorted_df = ops_df

    for _, row in sorted_df.iterrows():
        op_name = row.get("name", "Unknown")
        count = int(row.get("count", row.get("operation_count", 1)))
        time_ms = row.get("Kernel Time (µs)_sum", 0) / 1000
        percent_of_category = (
            (time_ms / total_time_ms * 100) if total_time_ms > 0 else 0
        )

        efficiency = calculate_efficiency(row, peak_hbm_bw, maf)

        op_metric = {
            "name": op_name,
            "count": count,
            "time_ms": round(time_ms, 3),
            "percent_of_category": round(percent_of_category, 2),
            "efficiency": efficiency,
        }

        # Add efficiency warning if present (for anomaly detection)
        if efficiency.get("warning"):
            op_metric["efficiency_warning"] = efficiency["warning"]

        # Add extra fields if specified
        extra_fields = category_config.get("extra_fields", [])
        for field in extra_fields:
            if field in row and not pd.isna(row[field]):
                op_metric[field] = row[field]

        # Kernel time variance detection (require 10+ samples for reliable CoV)
        if count > 10:
            std_val = row.get("Kernel Time (µs)_std")
            mean_val = row.get("Kernel Time (µs)_mean")
            if std_val is not None and mean_val is not None and not pd.isna(std_val) and not pd.isna(mean_val) and mean_val > 0:
                cov = round(float(std_val) / float(mean_val), 3)
                op_metric["cov"] = cov
                op_metric["high_variance"] = cov > 1.0

        # Apply operation classifier if provided
        classifier = category_config.get("operation_classifier")
        if classifier:
            op_metric["classification"] = classifier(op_name, row)

        operations.append(op_metric)

    return operations


def compute_impact_estimates(
    operations: List[dict],
    category: str,
    min_savings_ms: float = 0.1,
    baseline_ms: float = 0,
) -> List[dict]:
    """
    Deterministically compute kernel_tuning impact estimates from operation metrics.

    Assumes tuning can reach 75%–100% of roofline performance. Produces a range:
      - savings_ms_high = op_time_ms * (1 - efficiency_pct / 100)   [100% target]
      - savings_ms_low  = op_time_ms * (1 - efficiency_pct / 75)    [75% target]
      - savings_ms      = op_time_ms * (1 - efficiency_pct / 87.5)  [87.5% midpoint]

    The midpoint (87.5%) is the primary estimate used for plots and aggregation.
    Low/high values are negative-clamped to zero (already above target).
    Anomalous efficiencies (>100%) are excluded.

    When baseline_ms > 0, each estimate also includes e2e_pct_low / e2e_pct_high
    (savings as a percentage of end-to-end time).

    Args:
        operations: List of operation metric dicts (from build_operation_metrics)
        category: Category name for labelling
        min_savings_ms: Minimum savings threshold to include (default 0.1 ms)
        baseline_ms: Total end-to-end GPU time for E2E % calculation (0 to skip)

    Returns:
        List of impact estimate dicts sorted by savings descending
    """
    TARGET_HIGH = 100.0
    TARGET_LOW = 75.0
    TARGET_MID = 87.5

    estimates = []
    for op in operations:
        eff = op.get("efficiency", {})
        eff_pct = eff.get("efficiency_percent")
        if eff_pct is None or eff.get("is_anomaly"):
            continue
        time_ms = op.get("time_ms", 0)
        if time_ms <= 0:
            continue

        savings_high = max(0, time_ms * (1 - eff_pct / TARGET_HIGH))
        savings_low = max(0, time_ms * (1 - eff_pct / TARGET_LOW))
        savings_mid = max(0, time_ms * (1 - eff_pct / TARGET_MID))

        if savings_high < min_savings_ms:
            continue
        confidence = "high" if time_ms > 5 and eff_pct < 70 else "medium"
        estimate = {
            "operation": op.get("name", "Unknown"),
            "category": category,
            "type": "kernel_tuning",
            "savings_ms": round(savings_mid, 3),
            "savings_ms_low": round(savings_low, 3),
            "savings_ms_high": round(savings_high, 3),
            "confidence": confidence,
            "efficiency_pct": round(eff_pct, 2),
            "bound_type": eff.get("bound_type"),
            "time_ms": round(time_ms, 3),
        }
        if baseline_ms > 0:
            estimate["e2e_pct_low"] = round(savings_low / baseline_ms * 100, 2)
            estimate["e2e_pct_high"] = round(savings_high / baseline_ms * 100, 2)
        estimates.append(estimate)
    return sorted(estimates, key=lambda x: x["savings_ms"], reverse=True)


def generate_plot_data(output_dir: str, max_recommendations: int = 6) -> str:
    """
    Aggregate impact_estimates from all *_metrics.json files into a single
    plot_data.json consumed by the performance improvement plot script.

    Only kernel_tuning estimates with high/medium confidence are included
    in the plot recommendations. All estimates (including system and
    algorithmic) are collected in all_estimates for the report.

    Args:
        output_dir: Base output directory containing category_data/
        max_recommendations: Max number of recommendations for the plot

    Returns:
        Path to written plot_data.json
    """
    out_path = f"{output_dir}/plot_data.json"
    category_data_dir = f"{output_dir}/category_data"
    manifest_path = f"{category_data_dir}/category_manifest.json"

    try:
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        baseline_ms = manifest.get("gpu_utilization", {}).get("total_time_ms", 0)

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

        category_savings = defaultdict(
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

        plot_data = {
            "baseline_ms": baseline_ms,
            "recommendations": plot_recs,
            "all_estimates": all_estimates,
        }
    except Exception:
        plot_data = {
            "baseline_ms": 0,
            "recommendations": [],
            "all_estimates": [],
        }

    with open(out_path, "w") as f:
        json.dump(plot_data, f, indent=2)

    return out_path


REQUIRED_REPORT_HEADERS = [
    "Executive Summary",
    "Compute Kernel Optimizations",
    "System-Level Optimizations",
    "Detailed Analysis",
    "Appendix",
]


def validate_report(
    output_dir: str,
) -> Tuple[bool, List[str]]:
    """
    Validate that standalone_analysis.md contains all required ## section headers.

    Args:
        output_dir: Base output directory containing standalone_analysis.md

    Returns:
        Tuple of (passed: bool, missing: list of error/missing-section strings)
    """
    report_path = os.path.join(output_dir, "standalone_analysis.md")
    if not os.path.exists(report_path):
        return False, ["<file not found>"]

    with open(report_path, "r") as f:
        content = f.read()

    if len(content.strip()) < 100:
        return False, ["<report is empty or too short>"]

    missing: List[str] = []

    missing.extend(
        f"Missing section: {h}"
        for h in REQUIRED_REPORT_HEADERS
        if f"## {h}" not in content
    )

    return len(missing) == 0, missing


def write_metrics_json(metrics: dict, output_dir: str, category: str) -> str:
    """
    Write metrics JSON to category_data folder.

    Args:
        metrics: Metrics dict to write
        output_dir: Base output directory
        category: Category name

    Returns:
        Path to written file
    """
    output_path = f"{output_dir}/category_data/{category}_metrics.json"

    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return output_path


# Category-specific helper functions
def detect_quantized_gemm(op_name: str) -> bool:
    """Check if GEMM operation is quantized."""
    quantized_markers = ["w8a8", "int8", "fp8", "w4a16", "w4a4", "fp4", "mxfp4"]
    return any(marker in op_name.lower() for marker in quantized_markers)


def detect_flash_attention(op_name: str) -> bool:
    """Check if SDPA operation uses Flash Attention."""
    flash_markers = ["flash", "fmha", "flash_attention", "flashattn"]
    return any(marker in op_name.lower() for marker in flash_markers)


def detect_softmax(op_name: str) -> bool:
    """Check if operation is a softmax."""
    return "softmax" in op_name.lower()


def detect_transpose(op_name: str) -> bool:
    """Check if operation is a transpose (layout overhead indicator)."""
    return "transpose" in op_name.lower()


def classify_other_operation(op_name: str) -> str:
    """Classify 'other' category operations."""
    op_lower = op_name.lower()

    # Communication operations (vendor-agnostic naming)
    if any(
        x in op_lower
        for x in [
            "all_reduce",
            "collective",
            "ncclkernel",
            "rccl",
            "broadcast",
            "allgather",
        ]
    ):
        return "communication"

    # Graph operations
    if any(x in op_lower for x in ["graph", "hipgraph", "cudagraph"]):
        return "graph"

    return "miscellaneous"


def detect_paged_attention(op_name: str, kernel_details: str = None) -> bool:
    """
    Check if SDPA operation uses Paged Attention (vLLM style).

    Args:
        op_name: Operation name
        kernel_details: Optional kernel_details_summary string from CSV

    Returns:
        True if paged attention is detected
    """
    # Check operation name for vLLM paged attention markers
    paged_markers = ["unified_attention", "paged_attention", "vllm"]
    if any(marker in op_name.lower() for marker in paged_markers):
        return True

    # Check kernel details for paged attention kernels
    if kernel_details:
        if "kernel_paged_attention" in str(kernel_details).lower():
            return True
        if "paged_attention_2d" in str(kernel_details).lower():
            return True

    return False


def parse_kernel_breakdown(kernel_details_str: str) -> dict:
    """
    Parse kernel_details_summary to extract sub-kernel timing breakdown.

    Args:
        kernel_details_str: String representation of kernel details list

    Returns:
        Dict with kernel breakdown: {kernel_name: {mean_us, percent, total_us}}
    """
    result = {
        "kernels": [],
        "total_kernel_time_us": 0,
        "has_paged_attention": False,
        "has_fwd_kernel": False,
        "has_reshape_cache": False,
    }

    if not kernel_details_str or pd.isna(kernel_details_str):
        return result

    try:
        kernel_str = str(kernel_details_str)
        kernel_str = kernel_str.replace("np.float64(", "").replace(")", "")

        # Pattern to match kernel entries
        kernel_pattern = r"'name':\s*'([^']+)'.*?'mean_duration_us':\s*([0-9.]+)"
        matches = re.findall(kernel_pattern, kernel_str, re.DOTALL)

        total_time = 0
        kernels = []

        for name, mean_us in matches:
            mean_us_float = float(mean_us)
            total_time += mean_us_float

            # Classify kernel type
            kernel_type = "other"
            if "reshape_and_cache" in name.lower():
                kernel_type = "reshape_cache"
                result["has_reshape_cache"] = True
            elif "paged_attention" in name.lower():
                kernel_type = "paged_attention"
                result["has_paged_attention"] = True
            elif "_fwd_kernel" in name.lower() or "fwd_kernel" in name.lower():
                kernel_type = "fwd_kernel"
                result["has_fwd_kernel"] = True

            kernels.append(
                {"name": name, "mean_us": mean_us_float, "kernel_type": kernel_type}
            )

        # Calculate percentages
        if total_time > 0:
            for k in kernels:
                k["percent"] = round((k["mean_us"] / total_time) * 100, 2)

        result["kernels"] = kernels
        result["total_kernel_time_us"] = round(total_time, 2)

    except Exception:
        pass

    return result


def parse_perf_params(perf_params_str: str) -> dict:
    """
    Parse perf_params to extract attention configuration and workload profile.

    Args:
        perf_params_str: String representation of perf_params dict

    Returns:
        Dict with parsed parameters
    """
    result = {
        "batch_size": None,
        "n_q": None,
        "h_q": None,
        "n_kv": None,
        "h_kv": None,
        "d_h_qk": None,
        "d_h_v": None,
        "dropout": None,
        "causal": None,
        "flash_impl": None,
        "sum_ctx_tokens": None,
        "sum_gen_tokens": None,
        "ctx_ratio": None,
        "workload_type": "unknown",
    }

    if not perf_params_str or pd.isna(perf_params_str):
        return result

    try:
        params = ast.literal_eval(str(perf_params_str))

        # Extract basic parameters
        result["batch_size"] = params.get("B")
        result["n_q"] = params.get("N_Q")
        result["h_q"] = params.get("H_Q")
        result["n_kv"] = params.get("N_KV")
        result["h_kv"] = params.get("H_KV")
        result["d_h_qk"] = params.get("d_h_qk")
        result["d_h_v"] = params.get("d_h_v")
        result["dropout"] = params.get("dropout")
        result["causal"] = params.get("causal")
        result["flash_impl"] = params.get("flash_impl")

        # Extract context/generation tokens for workload profiling
        ctx_tokens = params.get("sum_ctx_tokens", 0)
        gen_tokens = params.get("sum_gen_tokens", 0)
        result["sum_ctx_tokens"] = ctx_tokens
        result["sum_gen_tokens"] = gen_tokens

        # Calculate context ratio and determine workload type
        total_tokens = ctx_tokens + gen_tokens
        if total_tokens > 0:
            ctx_ratio = ctx_tokens / total_tokens
            result["ctx_ratio"] = round(ctx_ratio, 3)

            if ctx_ratio > 0.8:
                result["workload_type"] = "prefill_heavy"
            elif ctx_ratio < 0.2:
                result["workload_type"] = "decode_heavy"
            else:
                result["workload_type"] = "mixed"

        # Detect GQA (Grouped Query Attention)
        if result["h_q"] and result["h_kv"]:
            if result["h_kv"] < result["h_q"]:
                result["attention_pattern"] = "GQA"
                result["gqa_ratio"] = result["h_q"] // result["h_kv"]
            elif result["h_kv"] == result["h_q"]:
                result["attention_pattern"] = "MHA"
            else:
                result["attention_pattern"] = "unknown"

    except Exception:
        pass

    return result
