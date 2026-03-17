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
        # MoE aiter 2 stage (cktile)
        "aiter::moe_cktile2stages_gemm1_ck": moe_perf_model_extensions.moe_aiter_unfused_up,
        "aiter::moe_cktile2stages_gemm2_ck": moe_perf_model_extensions.moe_aiter_unfused_down,
        # MoE aiter 2 stage (ck)
        "aiter::ck_moe_stage1": moe_perf_model_extensions.moe_aiter_ck_stage1,
        "aiter::ck_moe_stage2": moe_perf_model_extensions.moe_aiter_ck_stage2,
        # MoE - AITER FP8 block-scale fused (direct CPU op, no pseudo-op needed)
        "aiter::fmoe_fp8_blockscale_g1u1": moe_perf_model_extensions.moe_aiter_fused_blockscale,
        # MoE pseudo ops - Unfused Triton (2-stage: up and down)
        "pseudo_op::moe_triton_unfused_up": moe_perf_model_extensions.moe_triton_unfused_up,
        "pseudo_op::moe_triton_unfused_down": moe_perf_model_extensions.moe_triton_unfused_down,
        # Attention pseudo ops
        "vllm::unified_attention_with_output": attention_perf_model_extensions.vllm_unified_attention_with_output,
        "aiter::mha_varlen_fwd": attention_perf_model_extensions.mha_varlen_fwd,
        "pseudo_mla_decode_fwd": attention_perf_model_extensions.mla_decode_fwd,
        ## Misc ops
        "aiter::batched_gemm_a16wfp4_": perf_model_extensions.batched_gemm_a16wfp4,
        "aiter::dynamic_per_token_scaled_quant": perf_model_extensions.per_group_quant,
        "sglang_profiler::fp8_utils_gemm_a8w8_blockscale_7": perf_model_extensions.gemm_a8w8_blockscale,
        "vllm::rocm_aiter_triton_gemm_a8w8_blockscale": perf_model_extensions.gemm_a8w8_blockscale,
        "vllm::rocm_unquantized_gemm": perf_model_extensions.vllm_rocm_unquantized_gemm,
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
        attention_perf_model_extensions.GDNAttention: "GDNAttention",
        perf_model_extensions.gemm_a8w8_blockscale: "GEMM",
        perf_model_extensions.batched_gemm_a16wfp4: "GEMM",
        attention_perf_model_extensions.mha_varlen_fwd: "InferenceAttention",
        perf_model_extensions.per_group_quant: "BinaryElementwise"
    }
    
    return pseudo_op_categories
