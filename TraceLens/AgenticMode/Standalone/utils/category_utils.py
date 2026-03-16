###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Category-to-skill mapping and enhanced category classification."""

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
