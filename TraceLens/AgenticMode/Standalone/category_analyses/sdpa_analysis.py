#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""SDPA Analysis - Scaled Dot Product Attention

Computes metrics for SDPA operations and outputs JSON for subagent processing.
Supports both Flash Attention and Paged Attention (vLLM) analysis.
"""

import argparse
import ast
import re
import sys
import os

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analysis_utils import (
    get_peak_specs,
    run_category_analysis,
)

# ---------------------------------------------------------------------------
# SDPA-specific detectors (moved from analysis_utils)
# ---------------------------------------------------------------------------


def detect_flash_attention(op_name: str) -> bool:
    """Check if SDPA operation uses Flash Attention."""
    flash_markers = ["flash", "fmha", "flash_attention", "flashattn"]
    return any(marker in op_name.lower() for marker in flash_markers)


def detect_paged_attention(op_name: str, kernel_details: str = None) -> bool:
    """Check if SDPA operation uses Paged Attention (vLLM style)."""
    paged_markers = ["unified_attention", "paged_attention", "vllm"]
    if any(marker in op_name.lower() for marker in paged_markers):
        return True

    if kernel_details:
        kd_lower = str(kernel_details).lower()
        if "kernel_paged_attention" in kd_lower:
            return True
        if "paged_attention_2d" in kd_lower:
            return True

    return False


def parse_kernel_breakdown(kernel_details_str: str) -> dict:
    """Parse kernel_details_summary to extract sub-kernel timing breakdown."""
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

        kernel_pattern = r"'name':\s*'([^']+)'.*?'mean_duration_us':\s*([0-9.]+)"
        matches = re.findall(kernel_pattern, kernel_str, re.DOTALL)

        total_time = 0
        kernels = []

        for name, mean_us in matches:
            mean_us_float = float(mean_us)
            total_time += mean_us_float

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

        if total_time > 0:
            for k in kernels:
                k["percent"] = round((k["mean_us"] / total_time) * 100, 2)

        result["kernels"] = kernels
        result["total_kernel_time_us"] = round(total_time, 2)

    except Exception:
        pass

    return result


def parse_perf_params(perf_params_str: str) -> dict:
    """Parse perf_params to extract attention configuration and workload profile."""
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

        ctx_tokens = params.get("sum_ctx_tokens", 0)
        gen_tokens = params.get("sum_gen_tokens", 0)
        result["sum_ctx_tokens"] = ctx_tokens
        result["sum_gen_tokens"] = gen_tokens

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


# ---------------------------------------------------------------------------
# SDPA classification and extraction
# ---------------------------------------------------------------------------


def classify_sdpa_operation(op_name: str, row) -> dict:
    """Classify SDPA operation type with paged attention detection."""
    has_perf_model = (
        row.get("has_perf_model", False) if "has_perf_model" in row.index else False
    )

    kernel_details = (
        row.get("kernel_details_summary", "")
        if "kernel_details_summary" in row.index
        else ""
    )

    is_flash = detect_flash_attention(op_name)
    is_paged = detect_paged_attention(op_name, kernel_details)

    if is_paged:
        attention_type = "paged"
    elif is_flash:
        attention_type = "flash"
    else:
        attention_type = "standard"

    kernel_breakdown = None
    if is_paged and kernel_details:
        kernel_breakdown = parse_kernel_breakdown(kernel_details)

    perf_params_str = row.get("perf_params", "") if "perf_params" in row.index else ""
    workload_profile = None
    if perf_params_str:
        workload_profile = parse_perf_params(perf_params_str)

    result = {
        "is_flash_attention": is_flash,
        "is_paged_attention": is_paged,
        "has_perf_model": bool(has_perf_model),
        "attention_type": attention_type,
    }

    if kernel_breakdown and kernel_breakdown.get("kernels"):
        result["kernel_breakdown"] = {
            "has_paged_attention_kernel": kernel_breakdown.get(
                "has_paged_attention", False
            ),
            "has_fwd_kernel": kernel_breakdown.get("has_fwd_kernel", False),
            "has_reshape_cache": kernel_breakdown.get("has_reshape_cache", False),
            "kernels": kernel_breakdown.get("kernels", []),
        }

    if workload_profile:
        result["workload_profile"] = {
            "n_q": workload_profile.get("n_q"),
            "n_kv": workload_profile.get("n_kv"),
            "h_q": workload_profile.get("h_q"),
            "h_kv": workload_profile.get("h_kv"),
            "sum_ctx_tokens": workload_profile.get("sum_ctx_tokens"),
            "sum_gen_tokens": workload_profile.get("sum_gen_tokens"),
            "ctx_ratio": workload_profile.get("ctx_ratio"),
            "workload_type": workload_profile.get("workload_type"),
            "attention_pattern": workload_profile.get("attention_pattern"),
            "gqa_ratio": workload_profile.get("gqa_ratio"),
        }

    return result


def extract_category_specific(ops_df, metadata) -> dict:
    """Extract SDPA-specific aggregate metrics including paged attention."""
    flash_attention_count = 0
    paged_attention_count = 0

    total_reshape_cache_percent = 0
    total_fwd_kernel_percent = 0
    total_paged_attention_percent = 0
    kernel_breakdown_count = 0

    total_ctx_tokens = 0
    total_gen_tokens = 0

    for _, row in ops_df.iterrows():
        op_name = str(row.get("name", ""))
        kernel_details = (
            row.get("kernel_details_summary", "")
            if "kernel_details_summary" in row.index
            else ""
        )

        if detect_flash_attention(op_name):
            flash_attention_count += 1

        if detect_paged_attention(op_name, kernel_details):
            paged_attention_count += 1

            breakdown = parse_kernel_breakdown(kernel_details)
            if breakdown.get("kernels"):
                kernel_breakdown_count += 1
                for k in breakdown["kernels"]:
                    if k["kernel_type"] == "reshape_cache":
                        total_reshape_cache_percent += k.get("percent", 0)
                    elif k["kernel_type"] == "fwd_kernel":
                        total_fwd_kernel_percent += k.get("percent", 0)
                    elif k["kernel_type"] == "paged_attention":
                        total_paged_attention_percent += k.get("percent", 0)

        perf_params_str = (
            row.get("perf_params", "") if "perf_params" in row.index else ""
        )
        if perf_params_str:
            profile = parse_perf_params(perf_params_str)
            total_ctx_tokens += profile.get("sum_ctx_tokens") or 0
            total_gen_tokens += profile.get("sum_gen_tokens") or 0

    has_perf_model_count = 0
    if "has_perf_model" in ops_df.columns:
        has_perf_model_count = int(ops_df["has_perf_model"].sum())

    result = {
        "flash_attention_count": int(flash_attention_count),
        "paged_attention_count": int(paged_attention_count),
        "has_perf_model_count": has_perf_model_count,
        "flash_attention_detected": flash_attention_count > 0,
        "paged_attention_detected": paged_attention_count > 0,
        **get_peak_specs(metadata),
    }

    if kernel_breakdown_count > 0:
        result["kernel_breakdown_avg"] = {
            "avg_reshape_cache_percent": round(
                total_reshape_cache_percent / kernel_breakdown_count, 2
            ),
            "avg_fwd_kernel_percent": round(
                total_fwd_kernel_percent / kernel_breakdown_count, 2
            ),
            "avg_paged_attention_percent": round(
                total_paged_attention_percent / kernel_breakdown_count, 2
            ),
        }

    total_tokens = total_ctx_tokens + total_gen_tokens
    if total_tokens > 0:
        ctx_ratio = total_ctx_tokens / total_tokens
        if ctx_ratio > 0.8:
            profile_type = "prefill_heavy"
        elif ctx_ratio < 0.2:
            profile_type = "decode_heavy"
        else:
            profile_type = "mixed"

        result["workload_profile"] = {
            "total_ctx_tokens": total_ctx_tokens,
            "total_gen_tokens": total_gen_tokens,
            "ctx_ratio": round(ctx_ratio, 3),
            "profile_type": profile_type,
        }

    return result


def main():
    parser = argparse.ArgumentParser(description="Analyze SDPA operations")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument(
        "--category",
        default="sdpa_fwd",
        choices=["sdpa_fwd", "sdpa_bwd", "inferenceattention"],
        help="SDPA category to analyze (default: sdpa_fwd)",
    )
    args = parser.parse_args()

    run_category_analysis(
        category=args.category,
        output_dir=args.output_dir,
        config={
            "extra_fields": [
                "Input Dims",
                "has_perf_model",
                "perf_params",
                "kernel_details_summary",
            ],
            "operation_classifier": classify_sdpa_operation,
        },
        extract_fn=extract_category_specific,
    )


if __name__ == "__main__":
    main()
