#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from TraceLens.PerfModel.utils import torch_dtype_map

TARGET_HIGH = 100.0
TARGET_LOW = 75.0
TARGET_MID = 87.5

MIN_PITEM_IMPACT_SCORE = (
    0.5  # Group-sum (% E2E) below which a finding is dropped from priority_data.json.
)
UNMODELED_E2E_THRESHOLD = (
    4.0  # Summed %E2E (per op-name, across shapes) above which an unmodeled op becomes a non-quantifiable low-priority finding.
)
_EFF_BUCKET_BOUNDARIES = (30, 60)

_OP_NAME_LIBRARY_RULES = [
    ("aiter::", "AITER"),
    ("rocm_aiter", "AITER"),
    ("fbgemm", "FBGEMM"),
    ("miopen", "MIOpen"),
    ("triton", "Triton"),
]
_KERNEL_NAME_LIBRARY_RULES = [
    ("aiter", "AITER"),
    ("ck_tile::", "CK"),
    ("ck_tile6kentry", "CK"),
    ("FmhaFwd", "CK"),
    ("FmhaBwd", "CK"),
    ("Cijk_", "Tensile"),
    ("wvSplitK", "rocBLAS"),
    ("splitKreduce", "rocBLAS"),
    ("rocprim::", "rocPRIM"),
    ("triton_", "Triton"),
    ("void at::native::", "PyTorch Native"),
]
_COMPARATIVE_SPEEDUP_COL = "speedup (trace2/trace1)"
_COMPARATIVE_DELTA_COL = "delta_us (trace2 - trace1)"
_COMPARATIVE_T2_TIME_COL = "lca_total_kernel_time_trace2_us"
_COMPARATIVE_T2_COUNT_COL = "lca_count_trace2"

_CALL_CHAIN_MAX_OPS = 100
_CALL_CHAIN_SKIP = frozenset(
    {
        "torch/nn/modules/module.py",
        "torch/utils/_device.py",
        "torch/overrides.py",
        "torch/_ops.py",
        "multiprocessing/spawn.py",
        "torch/_dynamo/eval_frame.py",
        "torch/_functorch/",
        "torch/autograd/function.py",
        "torch/_inductor/",
        "torch/cuda/graphs.py",
        "torch/utils/_contextlib.py",
    }
)


def _eff_bucket(pct):
    """Classify roofline efficiency into a coarse band for P-item grouping."""
    if pct is None:
        return "unknown"
    if pct < _EFF_BUCKET_BOUNDARIES[0]:
        return f"0-{_EFF_BUCKET_BOUNDARIES[0]}"
    if pct < _EFF_BUCKET_BOUNDARIES[1]:
        return f"{_EFF_BUCKET_BOUNDARIES[0]}-{_EFF_BUCKET_BOUNDARIES[1]}"
    return f"{_EFF_BUCKET_BOUNDARIES[1]}-100"


def perf_report_csv_dir(output_dir: str) -> str:
    """
    Directory containing perf-report CSV exports for the primary (Trace 1) trace.

    When ``category_data/category_manifest.json`` exists and ``comparison_scope``
    is ``comparative``, returns ``<output_dir>/perf_report_trace1_csvs``.
    Otherwise returns ``<output_dir>/perf_report_csvs`` (standalone or missing manifest).
    """
    manifest_path = os.path.join(output_dir, "category_data", "category_manifest.json")
    if os.path.isfile(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            scope = json.load(f).get("comparison_scope", "standalone")
        if scope == "comparative":
            return os.path.join(output_dir, "perf_report_trace1_csvs")
    return os.path.join(output_dir, "perf_report_csvs")


_SKIP_PY_PATTERNS = (
    "torch/",
    "_functorch/",
    "_dynamo/",
    "torch/cuda/",
    "torch/utils/",
    "vllm/compilation/",
    "/tmp/torchinductor",
    "kernel_shape_profiler",
)

_KNOWN_PKG_ANCHORS = (
    "sglang/",
    "vllm/",
    "aiter/",
    "fbgemm_gpu/",
    "sgl_kernel/",
    "torchrec/",
    "components/",
    "megatron/",
    "transformers/",
    "model/",
    "deepspeed/",
    "flash_attn/",
    "ops/",
)


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


def comparative_efficiency(result: Dict[str, Any], row: pd.Series) -> None:
    """
    When comparative columns support it, set ``efficiency_percent`` to
    ``100 * t2 / t1`` (trace2 vs trace1 kernel time), clamped by trace1 roofline

    t1 is trace1 ``Kernel Time (µs)_sum``; t2 from delta (trace2 - trace1) or speedup t2/t1.
    If comparative metrics cannot be computed, leaves ``result`` unchanged for those keys.
    """
    kt = row.get("Kernel Time (µs)_sum")
    if kt is None or pd.isna(kt):
        return
    t1_us = float(kt)
    if t1_us <= 0:
        return

    comp_pct: Optional[float] = None
    delta = row.get(_COMPARATIVE_DELTA_COL)
    if delta is not None and not pd.isna(delta):
        comp_pct = 100.0 * (t1_us + float(delta)) / t1_us
    else:
        speedup = row.get(_COMPARATIVE_SPEEDUP_COL)
        if speedup is not None and not pd.isna(speedup):
            s = float(speedup)
            if s >= 0:
                comp_pct = 100.0 * s

    if comp_pct is None:
        return

    roofline_pct = row.get("Pct Roofline_mean")
    roofline_floor: Optional[float] = None
    if roofline_pct is not None and not pd.isna(roofline_pct):
        roofline_floor = float(roofline_pct)

    clamped = False
    if roofline_floor is not None and comp_pct < roofline_floor:
        comp_pct = roofline_floor
        clamped = True

    result["efficiency_percent"] = round(comp_pct, 2)
    result["is_anomaly"] = False
    if clamped:
        result["warning"] = (
            f"[ROOFLINE CAP] Projected efficiency clamped to roofline "
            f"({result['efficiency_percent']:.1f}%); trace2 gap exceeds "
            f"trace1 hardware ceiling."
        )
    else:
        result["warning"] = None


def standalone_efficiency(result: Dict[str, Any], row: pd.Series) -> None:
    """
    Set ``efficiency_percent`` from ``Pct Roofline_mean`` (roofline utilization %)
    and roofline anomaly flags when above 110%.
    """
    pct = row.get("Pct Roofline_mean")
    if pct is None or pd.isna(pct):
        return
    result["efficiency_percent"] = round(float(pct), 2)
    if result["efficiency_percent"] > 110:
        result["warning"] = (
            f"[ANOMALY] Pct Roofline exceeds 110% ({result['efficiency_percent']:.1f}%) - verify measurement"
        )
        result["is_anomaly"] = True


def calculate_efficiency(
    row: pd.Series,
    peak_hbm_bw: float,
    peak_maf_or_maf_dict,
    comparison_scope: str = "standalone",
) -> Dict[str, Optional[float]]:
    """
    Extract efficiency metrics for an operation from pre-computed CSV columns.

    Args:
        row: DataFrame row with operation metrics
        peak_hbm_bw: Peak HBM bandwidth in TB/s
        peak_maf_or_maf_dict: Either a float or a dict (max_achievable_tflops)
            for resolving the precision-aware peak
        comparison_scope:
            ``standalone`` uses roofline efficiency
            ``comparative`` uses comparative efficiency

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

    if comparison_scope == "comparative":
        comparative_efficiency(result, row)
    else:
        standalone_efficiency(result, row)

    return result


def _load_fusion_map(output_dir: str) -> Dict[str, str]:
    """Load high-confidence GPU kernel name -> fusion candidate name mapping."""
    if not output_dir:
        return {}
    path = os.path.join(output_dir, "category_data", "kernel_fusion_metrics.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f).get("high_confidence_kernel_map", {})
    except (json.JSONDecodeError, KeyError):
        return {}


def _match_fusion_op(kd_str: str, fusion_map: Dict[str, str]) -> Optional[str]:
    """Match kernel_details_summary against fusion kernel map with prefix fallback."""
    for kn in re.findall(r"'name':\s*'([^']+)'", kd_str):
        if kn in fusion_map:
            return fusion_map[kn]
        for fk, bn in fusion_map.items():
            if fk.startswith(kn) or kn.startswith(fk):
                return bn
    return None


def _extract_launcher_path(call_stack_str: str, op_name: str) -> str:
    """Return the first non-infrastructure .py frame from a call stack, relativized.

    Absolute container prefixes are stripped using ``_KNOWN_PKG_ANCHORS``; unknown
    roots fall back to the last 3 path segments.
    """
    if not call_stack_str or call_stack_str == "nan":
        return ""
    frames = [f.strip() for f in call_stack_str.split(" => ")]
    for i, frame in enumerate(frames):
        if i == 0:
            continue
        if ".py" not in frame:
            continue
        if any(p in frame for p in _SKIP_PY_PATTERNS):
            continue
        for anchor in _KNOWN_PKG_ANCHORS:
            idx = frame.rfind(anchor)
            if idx > 0:
                return frame[idx:]
        if frame.startswith("/") or frame.startswith("./"):
            parts = frame.lstrip("./").split("/")
            if len(parts) > 3:
                return "/".join(parts[-4:])
        return frame
    return ""


def _extract_module_chain(call_stack_str: str) -> List[str]:
    """Extract nn.Module names from a call stack string (leaf-to-root order)."""
    if not call_stack_str or call_stack_str == "nan":
        return []
    return [
        f.strip().removeprefix("nn.Module: ")
        for f in call_stack_str.split(" => ")
        if f.strip().startswith("nn.Module: ")
    ]


def _extract_backward_context(call_chain: List[str], library: str) -> str:
    """Produce a descriptive label for autograd backward ops."""
    for frame in call_chain:
        if "autograd::engine::evaluate_function:" in frame:
            fn = frame.split("autograd::engine::evaluate_function:")[-1].strip()
            if library:
                return f"{library} ({fn})"
            return fn
    return ""


def _extract_call_chain(call_stack_str: str) -> List[str]:
    """Return meaningful frames from a call stack, filtering torch dispatch internals."""
    if not call_stack_str or call_stack_str == "nan":
        return []
    frames = [f.strip() for f in call_stack_str.split(" => ")]
    return [
        f
        for f in frames
        if not any(s in f for s in _CALL_CHAIN_SKIP)
        and (f.startswith("nn.Module:") or ".py" in f or "::" in f)
    ]


def build_operation_metrics(
    ops_df: pd.DataFrame,
    metadata: dict,
    category_config: dict,
    comparison_scope: str = "standalone",
) -> List[dict]:
    """Build operation metric dicts for JSON output.

    Args:
        ops_df: Operations DataFrame (may include a ``call_stack`` column)
        metadata: Peak performance values and GPU utilization
        category_config: Optional extra_fields / operation_classifier
        comparison_scope: ``"standalone"`` or ``"comparative"``
    """
    peak_hbm_bw = metadata.get("peak_hbm_bw_tbs", 1)
    maf = metadata.get("max_achievable_tflops", metadata.get("peak_bf16_maf_tflops", 1))
    fusion_map = _load_fusion_map(metadata.get("output_dir", ""))
    e2e_ms_total = metadata.get("gpu_utilization", {}).get("total_time_ms", 0)

    # Calculate category total for % of category (kept for analyzer screening)
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
            round(time_ms / total_time_ms * 100, 2) if total_time_ms > 0 else 0
        )
        percent_of_total = (
            round(time_ms / e2e_ms_total * 100, 2) if e2e_ms_total > 0 else None
        )

        efficiency = calculate_efficiency(
            row, peak_hbm_bw, maf, comparison_scope=comparison_scope
        )

        op_metric = {
            "name": op_name,
            "count": count,
            "time_ms": round(time_ms, 3),
            "percent_of_category": percent_of_category,
            "percent_of_total": percent_of_total,
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

        # Pre-rendered Args column (shape + dtype, with empty/Scalar slots dropped)
        args_str = format_args(
            row.get("Input Dims") if "Input Dims" in row else None,
            row.get("Input type") if "Input type" in row else None,
        )
        if args_str:
            op_metric["args"] = args_str

        # Pre-compute comparative T2 fields from CSV
        if comparison_scope == "comparative":
            t2_us = row.get(_COMPARATIVE_T2_TIME_COL)
            if t2_us is not None and not pd.isna(t2_us):
                op_metric["t2_time_ms"] = round(float(t2_us) / 1000, 3)
            t2_count = row.get(_COMPARATIVE_T2_COUNT_COL)
            if t2_count is not None and not pd.isna(t2_count):
                op_metric["count_trace2"] = int(t2_count)
            delta_us = row.get(_COMPARATIVE_DELTA_COL)
            if delta_us is not None and not pd.isna(delta_us):
                op_metric["difference_ms"] = round(float(delta_us) / 1000, 3)

        # Kernel time variance detection (require 10+ samples, >= 5% of E2E)
        e2e_ms = e2e_ms_total
        e2e_frac = time_ms / e2e_ms if e2e_ms > 0 else 1.0
        if count > 10 and e2e_frac >= 0.05:
            std_val = row.get("Kernel Time (µs)_std")
            mean_val = row.get("Kernel Time (µs)_mean")
            if (
                std_val is not None
                and mean_val is not None
                and not pd.isna(std_val)
                and not pd.isna(mean_val)
                and mean_val > 0
            ):
                cov = round(float(std_val) / float(mean_val), 3)
                op_metric["cov"] = cov
                op_metric["high_variance"] = cov > 1.0

        # Apply operation classifier if provided
        classifier = category_config.get("operation_classifier")
        if classifier:
            op_metric["classification"] = classifier(op_name, row)

        kd_str = row.get("kernel_details_summary", "")
        if pd.isna(kd_str):
            kd_str = ""
        else:
            kd_str = str(kd_str)

        op_metric["library"] = classify_kernel_library(op_name, kd_str)

        if fusion_map and kd_str:
            matched = _match_fusion_op(kd_str, fusion_map)
            if matched:
                op_metric["fusion_flagged"] = True
                op_metric["fusion_candidate_name"] = matched

        cs_raw = row.get("call_stack")
        cs_str = "" if cs_raw is None or pd.isna(cs_raw) else str(cs_raw)
        launcher = _extract_launcher_path(cs_str, op_name)

        if not launcher:
            launcher = _extract_backward_context(
                _extract_call_chain(cs_str) if cs_str else [],
                op_metric.get("library", ""),
            )

        module_chain = _extract_module_chain(cs_str)

        if not launcher and module_chain:
            launcher = " > ".join(module_chain[:3])

        if not launcher and op_metric.get("library"):
            launcher = f"{op_metric['library']} (vendor)"

        op_metric["launcher_path"] = launcher if launcher else "\u2014"
        op_metric["module_chain"] = module_chain
        op_metric["_raw_call_stack"] = cs_str

        operations.append(op_metric)

    if len(operations) <= _CALL_CHAIN_MAX_OPS:
        for op in operations:
            op["call_chain"] = _extract_call_chain(op.pop("_raw_call_stack", ""))
    else:
        for op in operations:
            op.pop("_raw_call_stack", None)

    return operations


def compute_impact_estimates(
    operations: List[dict],
    category: str,
    min_impact_score: float = 0.01,
    baseline_ms: float = 0,
) -> List[dict]:
    """
    Deterministically compute kernel_tuning impact estimates from operation metrics.

    Two-step computation:

      1. Per-op roofline headroom ``gap``.
            gap_high = max(0, 1 - efficiency_pct / TARGET_HIGH)
            gap_low  = (TARGET_LOW / TARGET_HIGH) * gap_high
            gap_mid  = (TARGET_MID / TARGET_HIGH) * gap_high

      2. ``impact_score`` = The impact estimate on E2E GPU time obtained by weighting the gap by the op's share
         of E2E:
            impact_score_high = gap_high * time_ms / baseline_ms * 100
            impact_score_low  = gap_low  * time_ms / baseline_ms * 100
            impact_score      = gap_mid  * time_ms / baseline_ms * 100

    The midpoint (87.5% of the way from 75%->100% target closure) is the
    primary estimate used for plots and aggregation. Anomalous efficiencies
    (>100%) are excluded.

    If ``baseline_ms`` is missing or non-positive, ``impact_score`` is undefined
    (would divide by zero or produce a negative percentage). In that case this
    function emits a stderr warning and returns an empty list.

    Args:
        operations: List of operation metric dicts (from build_operation_metrics)
        category: Category name for labelling
        min_impact_score: Per-op noise floor on ``impact_score_high`` (% E2E).
        baseline_ms: Total end-to-end GPU time for impact_score calculation.
            Must be > 0; otherwise an empty list is returned.

    Returns:
        List of impact estimate dicts sorted by ``impact_score`` descending.
    """
    if baseline_ms is None or baseline_ms <= 0:
        print(
            f"[analysis_utils.compute_impact_estimates] baseline_ms="
            f"{baseline_ms!r} is invalid; skipping impact_score computation "
            f"for category={category!r}",
            file=sys.stderr,
        )
        return []

    estimates = []
    for op in operations:
        if op.get("fusion_flagged"):
            continue
        eff = op.get("efficiency", {})
        eff_pct = eff.get("efficiency_percent")
        if eff_pct is None or eff.get("is_anomaly"):
            continue
        time_ms = op.get("time_ms", 0)
        if time_ms <= 0:
            continue

        gap_high = max(0, 1 - eff_pct / TARGET_HIGH)
        gap_low = (TARGET_LOW / TARGET_HIGH) * gap_high
        gap_mid = (TARGET_MID / TARGET_HIGH) * gap_high

        impact_score_high = gap_high * time_ms / baseline_ms * 100
        impact_score_low = gap_low * time_ms / baseline_ms * 100
        impact_score_mid = gap_mid * time_ms / baseline_ms * 100

        if impact_score_high < min_impact_score:
            continue
        estimate = {
            "operation": op.get("name", "Unknown"),
            "category": category,
            "type": "kernel_tuning",
            "impact_score": round(impact_score_mid, 2),
            "impact_score_low": round(impact_score_low, 2),
            "impact_score_high": round(impact_score_high, 2),
            "efficiency_pct": round(eff_pct, 2),
            "bound_type": eff.get("bound_type"),
            "library": op.get("library"),
            "time_ms": round(time_ms, 3),
        }
        estimates.append(estimate)
    return sorted(estimates, key=lambda x: x["impact_score"], reverse=True)


def build_category_findings(
    estimates: List[dict], comparison_scope: str = "standalone"
) -> List[dict]:
    """Group one category's per-op impact estimates into P-item findings.

    Standalone mode groups by ``(bound_type, library, eff_bucket)`` so
    kernels at different roofline-efficiency levels get separate P-items.
    Comparative mode groups by ``(bound_type, library)`` only because
    ``efficiency_pct`` represents a trace-time ratio, not roofline utilization.

    Sums impact across members, drops groups below ``MIN_PITEM_IMPACT_SCORE``,
    sorts by ``impact_score`` desc, attaches intra-category ``rank``. Each
    surviving group becomes one P-item the sub-agent renders. The orchestrator
    later concatenates these arrays across categories to derive the global
    ``priority_data.json::findings[]`` ranking.
    """
    use_eff_bucket = comparison_scope == "standalone"
    groups: dict = defaultdict(
        lambda: {
            "members": [],
            "impact_score": 0.0,
            "impact_score_low": 0.0,
            "impact_score_high": 0.0,
        }
    )
    for e in estimates:
        bound = e.get("bound_type") or "unknown"
        lib = e.get("library") or "Unknown"
        bucket = _eff_bucket(e.get("efficiency_pct")) if use_eff_bucket else "all"
        key = (bound, lib, bucket)
        g = groups[key]
        g["members"].append(e)
        g["impact_score"] += e.get("impact_score", 0)
        g["impact_score_low"] += e.get("impact_score_low", 0)
        g["impact_score_high"] += e.get("impact_score_high", 0)

    findings = []
    for (bound, lib, bucket), g in groups.items():
        if g["impact_score"] < MIN_PITEM_IMPACT_SCORE:
            continue
        findings.append(
            {
                "bound_type": bound,
                "library": lib,
                "eff_bucket": bucket,
                "impact_score": round(g["impact_score"], 2),
                "impact_score_low": round(g["impact_score_low"], 2),
                "impact_score_high": round(g["impact_score_high"], 2),
                "member_count": len(g["members"]),
                "members": sorted(
                    g["members"],
                    key=lambda m: m.get("impact_score", 0),
                    reverse=True,
                ),
            }
        )

    findings.sort(key=lambda f: f["impact_score"], reverse=True)
    for rank, f in enumerate(findings, start=1):
        f["rank"] = rank
    return findings


def build_unmodeled_significant_findings(
    operations: List[dict],
    category: str,
    start_rank: int = 1,
) -> List[dict]:
    """Emit non-quantifiable findings for significant ops that have no perf model.

    Contributions are summed by op-name across shapes before thresholding (so a
    collective fragmented across token shapes is judged on its combined %E2E).
    Findings carry ``impact_score = None`` and rank from ``start_rank`` after the
    category's quantified findings.
    """
    aggregated = {}
    for op in operations:
        eff = op.get("efficiency") or {}
        if eff.get("efficiency_percent") is not None:
            continue
        pct = op.get("percent_of_total")
        if pct is None:
            continue
        if (op.get("time_ms") or 0) <= 0:
            continue
        if op.get("fusion_flagged"):
            continue
        name = op.get("name", "Unknown")
        agg = aggregated.setdefault(
            name,
            {"percent_of_total": 0.0, "time_ms": 0.0, "n_shapes": 0, "library": None},
        )
        agg["percent_of_total"] += pct
        agg["time_ms"] += op.get("time_ms", 0) or 0
        agg["n_shapes"] += 1
        if agg["library"] is None:
            agg["library"] = op.get("library")

    qualifying = [
        (name, agg)
        for name, agg in aggregated.items()
        if agg["percent_of_total"] > UNMODELED_E2E_THRESHOLD
    ]
    qualifying.sort(key=lambda item: item[1]["percent_of_total"], reverse=True)

    findings = []
    for offset, (name, agg) in enumerate(qualifying):
        member = {
            "operation": name,
            "category": category,
            "type": "unmodeled_significant",
            "impact_score": None,
            "impact_score_low": None,
            "impact_score_high": None,
            "efficiency_pct": None,
            "bound_type": None,
            "library": agg["library"],
            "time_ms": round(agg["time_ms"], 3),
            "percent_of_total": round(agg["percent_of_total"], 2),
            "n_shapes": agg["n_shapes"],
        }
        findings.append(
            {
                "bound_type": "unmodeled",
                "library": agg["library"] or "Unknown",
                "eff_bucket": "unknown",
                "impact_score": None,
                "impact_score_low": None,
                "impact_score_high": None,
                "member_count": 1,
                "members": [member],
                "rank": start_rank + offset,
            }
        )
    return findings


def parse_first_shape(dims_str):
    """Extract first input tensor shape as a hashable tuple from an Input Dims string."""
    if dims_str is None or (isinstance(dims_str, float) and pd.isna(dims_str)):
        return None
    try:
        parsed = ast.literal_eval(str(dims_str))
        return (
            tuple(parsed[0])
            if parsed and isinstance(parsed[0], (list, tuple))
            else None
        )
    except Exception:
        return None


def format_args(input_dims_str, input_type_str) -> Optional[str]:
    """Render "(d1,d2,...) dtype<br>..." from Input Dims + Input type strings.

    Drops empty/Scalar placeholder slots, normalizes dtypes via torch_dtype_map
    (unmapped tokens fall through). Args are joined with <br> so each entry
    renders on its own line inside a markdown table cell, keeping the column
    narrow even for many-input ops. Returns None on missing or unparseable input.
    """
    if pd.isna(input_dims_str) or pd.isna(input_type_str):
        return None
    try:
        pairs = list(
            zip(
                ast.literal_eval(str(input_dims_str)),
                ast.literal_eval(str(input_type_str)),
            )
        )
    except Exception:
        return None

    parts = []
    for dim, dtype in pairs:
        shape = tuple(dim) if isinstance(dim, (list, tuple)) else ()
        kind = (torch_dtype_map(str(dtype)) or str(dtype)) if dtype else ""
        if not shape and kind in ("", "Scalar"):
            continue
        # Trailing comma keeps tuple syntax for 1-element shapes ((128,) not (128))
        body = ",".join(map(str, shape)) + ("," if len(shape) == 1 else "")
        parts.append(f"({body}) {kind}".rstrip())
    return "<br>".join(parts) or None


def shape_aware_lookup(table, kname, input_dims=None):
    """Look up perf metrics by (kernel_name, shape), fall back to any entry for that name.

    Uses prefix matching as fallback when exact key misses, since trace kernel
    names can be longer than the truncated names stored in perf CSV lookups.
    """
    shapes = table.get(kname, {})
    if not shapes:
        for csv_name in table:
            if kname.startswith(csv_name) or csv_name.startswith(kname):
                shapes = table[csv_name]
                break
    shape_key = parse_first_shape(input_dims) if input_dims else None
    return shapes.get(shape_key) or next(iter(shapes.values()), {})


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


def get_peak_specs(metadata: dict) -> dict:
    """Extract peak MAF and HBM bandwidth from metadata dict.

    Handles both the dict-style max_achievable_tflops and the legacy
    scalar peak_bf16_maf_tflops formats.
    """
    return {
        "peak_maf_tflops": (
            metadata.get("max_achievable_tflops", {}).get("matrix_bf16")
            if isinstance(metadata.get("max_achievable_tflops"), dict)
            else metadata.get("peak_bf16_maf_tflops")
        ),
        "peak_hbm_bw_tbs": metadata.get("peak_hbm_bw_tbs"),
    }


def run_category_analysis(
    category: str,
    output_dir: str,
    config: dict,
    extract_fn,
    pre_process_fn=None,
    no_data_check_fn=None,
    compute_impact=True,
    comparison_scope: str = "standalone",
):
    """Generic runner for category analysis scripts.

    Args:
        category: Category name (e.g., 'gemm', 'norm')
        output_dir: Base output directory path
        config: Category-specific config dict (extra_fields, operation_classifier)
        extract_fn: Callable(ops_df, metadata) -> dict of category-specific metrics
        pre_process_fn: Optional callable(ops_df, metadata) -> (ops_df, extra_dict)
            for pre-filtering or augmenting the DataFrame before analysis
        no_data_check_fn: Optional callable(output_dir, category, comparison_scope) -> dict or None.
            If it returns a dict, that dict is written as metrics and the runner
            exits early (used for categories that may not be present, e.g. MoE).
        compute_impact: Whether to compute kernel-tuning impact estimates
        comparison_scope: ``standalone`` (roofline efficiency) or ``comparative``
            (TraceDiff-style ``100 * t2 / t1`` in ``operations[].efficiency``).
    """
    if no_data_check_fn:
        no_data_metrics = no_data_check_fn(output_dir, category, comparison_scope)
        if no_data_metrics is not None:
            output_path = write_metrics_json(no_data_metrics, output_dir, category)
            print(f"No data. Metrics written to: {output_path}")
            return

    try:
        ops_df, metadata = load_category_data(output_dir, category)
    except FileNotFoundError as e:
        error_metrics = {
            "category": category,
            "status": "ERROR",
            "error": str(e),
            "comparison_scope": comparison_scope,
        }
        write_metrics_json(error_metrics, output_dir, category)
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    extra = {}
    if pre_process_fn:
        ops_df, extra = pre_process_fn(ops_df, metadata)

    time_metrics = calculate_time_metrics(ops_df, metadata)

    operations = build_operation_metrics(
        ops_df,
        metadata,
        config,
        comparison_scope=comparison_scope,
    )
    category_specific = extract_fn(ops_df, metadata)

    if compute_impact:
        baseline_ms = metadata.get("gpu_utilization", {}).get("total_time_ms", 0)
        impact_estimates = compute_impact_estimates(
            operations,
            category,
            baseline_ms=baseline_ms,
        )
    else:
        impact_estimates = []

    category_findings = build_category_findings(
        impact_estimates, comparison_scope=comparison_scope
    )

    metrics = {
        "category": category,
        "status": "OK",
        "comparison_scope": comparison_scope,
        **time_metrics,
        "operations": operations,
        "category_specific": category_specific,
        "impact_estimates": impact_estimates,
        "category_findings": category_findings,
        **extra,
    }

    output_path = write_metrics_json(metrics, output_dir, category)
    print(f"Metrics written to: {output_path}")


# Category-specific helper functions
def classify_kernel_library(op_name: str, kernel_details: str = "") -> Optional[str]:
    """Identify the GPU library backing an operation from its name or kernel strings."""
    op_lower = op_name.lower()
    for marker, lib in _OP_NAME_LIBRARY_RULES:
        if marker in op_lower:
            return lib
    kd = str(kernel_details) if kernel_details and not pd.isna(kernel_details) else ""
    for marker, lib in _KERNEL_NAME_LIBRARY_RULES:
        if marker in kd:
            return lib
    return None
