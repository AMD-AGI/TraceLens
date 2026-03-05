###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Utils. for perf. model pseudo-op extensions.
"""

from . import moe_perf_model_extensions, attention_perf_model_extensions, perf_model_extensions


def get_pseudo_op_mappings():
    """
    Return a dictionary mapping pseudo-op names to their performance model classes.
    
    Returns:
        dict: Mapping of pseudo-op names to performance model classes
    """

    pseudo_op_mappings = {
        # MoE pseudo ops - Fused
        "pseudo_op::moe_aiter_fused_1stage": moe_perf_model_extensions.moe_aiter_fused_1stage,
        # MoE pseudo ops - Unfused Triton (2-stage: up and down)
        "pseudo_op::moe_triton_unfused_up": moe_perf_model_extensions.moe_triton_unfused_up,
        "pseudo_op::moe_triton_unfused_down": moe_perf_model_extensions.moe_triton_unfused_down,
        # Attention pseudo ops
        "vllm::unified_attention_with_output": attention_perf_model_extensions.vllm_unified_attention_with_output,
        "sglang_profiler::fp8_utils_gemm_a8w8_blockscale_7": perf_model_extensions.gemm_a8w8_blockscale,
        "aiter::mha_varlen_fwd": attention_perf_model_extensions.mha_varlen_fwd,
    }

    return pseudo_op_mappings


def get_pseudo_op_categories():
    """
    Return a dictionary mapping pseudo-op base classes to their performance categories.
    
    Returns:
        dict: Mapping of base classes to category names
    """
    
    pseudo_op_categories = {
        moe_perf_model_extensions.FusedMoE: "MoE_fused",
        moe_perf_model_extensions.UnfusedMoE_Up: "MoE_unfused",
        moe_perf_model_extensions.UnfusedMoE_Down: "MoE_unfused",
        attention_perf_model_extensions.InferenceAttention: "InferenceAttention",
        perf_model_extensions.gemm_a8w8_blockscale: "GEMM",
        attention_perf_model_extensions.mha_varlen_fwd: "InferenceAttention",
    }
    
    return pseudo_op_categories
