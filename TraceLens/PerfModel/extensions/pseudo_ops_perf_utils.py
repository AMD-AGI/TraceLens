###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Utils. for perf. model pseudo-op extensions.
"""

from . import (
    moe_perf_model_extensions,
    attention_perf_model_extensions,
    perf_model_extensions,
    rmsnorm_perf_model_extensions,
    custom_collectives_perf_model_extensions,
)


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
        # MoE - GPTQ/AWQ quantized unfused implementation (vllm::outplace_fused_experts)
        "pseudo_op::moe_gptq_awq_up": moe_perf_model_extensions.moe_gptq_awq_up,
        "pseudo_op::moe_gptq_awq_down": moe_perf_model_extensions.moe_gptq_awq_down,
        # Attention pseudo ops
        "vllm::unified_attention_with_output": attention_perf_model_extensions.vllm_unified_attention_with_output,
        "aiter::mha_varlen_fwd": attention_perf_model_extensions.mha_varlen_fwd,
        "aiter::fmha_v3_varlen_fwd": attention_perf_model_extensions.aiter_fmha_v3_varlen_fwd,
        "pseudo_mla_decode_fwd": attention_perf_model_extensions.mla_decode_fwd,
        "sglang_profiler::tilelang_kernel_tilelang_sparse_fwd_586": attention_perf_model_extensions.mla_tilelang_sparse_fwd,
        ## Misc ops
        "aiter::batched_gemm_a16wfp4_": perf_model_extensions.batched_gemm_a16wfp4,
        "aiter::dynamic_per_token_scaled_quant": perf_model_extensions.per_group_quant,
        "sglang_profiler::fp8_utils_gemm_a8w8_blockscale_7": perf_model_extensions.gemm_a8w8_blockscale,
        "vllm::rocm_aiter_triton_gemm_a8w8_blockscale": perf_model_extensions.gemm_a8w8_blockscale,
        "aiter::gemm_a8w8_blockscale_ck": perf_model_extensions.gemm_a8w8_blockscale,
        "aiter::gemm_a8w8_blockscale_cktile": perf_model_extensions.gemm_a8w8_blockscale,
        "aiter::gemm_a8w8_blockscale_bpreshuffle_ck": perf_model_extensions.gemm_a8w8_blockscale,
        "aiter::gemm_a8w8_blockscale_bpreshuffle_cktile": perf_model_extensions.gemm_a8w8_blockscale,
        "aiter::flatmm_a8w8_blockscale_asm": perf_model_extensions.gemm_a8w8_blockscale,
        "aiter::gfx950_a8w8_blockscale_asm": perf_model_extensions.gemm_a8w8_blockscale,
        "aiter::gemm_a8w8_ck": perf_model_extensions.gemm_a8w8_blockscale,
        "aiter::gemm_a8w8_bpreshuffle_ck": perf_model_extensions.gemm_a8w8_blockscale,
        "vllm::rocm_unquantized_gemm": perf_model_extensions.vllm_rocm_unquantized_gemm,
        "aiter::gemm_a16w16_atomic_": perf_model_extensions.gemm_a16w16_atomic_,
        "sglang_profiler::batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant_batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant_464": perf_model_extensions.batched_gemm_a8w8,
        ## Cache ops
        ##"_C_cache_ops::concat_and_cache_mla": perf_model_extensions.concat_and_cache_mla,
        ## Quantization ops
        "vllm::triton_per_token_group_quant_fp8": perf_model_extensions.vllm_triton_per_token_group_quant_fp8,
        ## Activation ops
        "aiter::silu_and_mul": perf_model_extensions.aiter_silu_and_mul,
        "_C::silu_and_mul": perf_model_extensions.aiter_silu_and_mul,
        "sgl_kernel::silu_and_mul": perf_model_extensions.sgl_kernel_silu_and_mul,
        "aiter::gelu_and_mul": perf_model_extensions.aiter_gelu_and_mul,
        "aiter::gelu_tanh_and_mul": perf_model_extensions.aiter_gelu_tanh_and_mul,
        ## MoE ops
        ##"aiter::moe_sorting_fwd": perf_model_extensions.aiter_moe_sorting_fwd,
        ## RMSNorm ops
        "aiter::rms_norm": rmsnorm_perf_model_extensions.aiter_rms_norm,
        "aiter::rmsnorm": rmsnorm_perf_model_extensions.aiter_rmsnorm,
        "aiter::rmsnorm2d_fwd_ck": rmsnorm_perf_model_extensions.aiter_rms_norm,
        "aiter::rmsnorm2d_fwd_with_add_ck": rmsnorm_perf_model_extensions.aiter_rmsnorm2d_fwd_with_add_ck,
        "aiter::add_rmsnorm": rmsnorm_perf_model_extensions.aiter_rmsnorm2d_fwd_with_add_ck,
        "aiter::rmsnorm2d_fwd_with_dynamicquant_ck": rmsnorm_perf_model_extensions.aiter_rmsnorm2d_fwd_with_dynamicquant_ck,
        "vllm::rocm_aiter_rmsnorm_fp8_group_quant": rmsnorm_perf_model_extensions.vllm_rocm_aiter_rmsnorm_fp8_group_quant,
        "vllm::rocm_aiter_rmsnorm_with_add_fp8_group_quant": rmsnorm_perf_model_extensions.vllm_rocm_aiter_rmsnorm_with_add_fp8_group_quant,
        "vllm::rocm_aiter_triton_add_rmsnorm_pad": rmsnorm_perf_model_extensions.vllm_rocm_aiter_triton_add_rmsnorm_pad,
        ## Collective ops
        "aiter::fused_allreduce_rmsnorm": custom_collectives_perf_model_extensions.aiter_fused_allreduce_rmsnorm,
        "_C_custom_ar::all_reduce": custom_collectives_perf_model_extensions.custom_ar_all_reduce,
        "aiter::reduce_scatter": custom_collectives_perf_model_extensions.aiter_reduce_scatter,
        "aiter::all_gather_reg": custom_collectives_perf_model_extensions.aiter_all_gather_reg,
        "sgl_kernel::all_reduce_reg": custom_collectives_perf_model_extensions.sgl_kernel_all_reduce_reg,
        "sgl_kernel::qr_all_reduce": custom_collectives_perf_model_extensions.sgl_kernel_qr_all_reduce,
        "sglang::reg_all_gather_into_tensor": custom_collectives_perf_model_extensions.sgl_kernel_reg_all_gather_into_tensor,
        "_C_custom_ar::qr_all_reduce": custom_collectives_perf_model_extensions.custom_ar_qr_all_reduce,
        ## GDN attention ops
        "vllm::gdn_attention_core": attention_perf_model_extensions.gdn_attention_core,
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
        perf_model_extensions.batched_gemm_a16wfp4: "GEMM",
        attention_perf_model_extensions.mha_varlen_fwd: "InferenceAttention",
        perf_model_extensions.GroupQuant: "GroupQuant",
        perf_model_extensions.gemm_a16w16_atomic_: "GEMM",
        perf_model_extensions.batched_gemm_a8w8: "GEMM",
        rmsnorm_perf_model_extensions.RMSNorm: "RMSNorm",
        custom_collectives_perf_model_extensions.CustomCollective: "CustomCollective",
        custom_collectives_perf_model_extensions.custom_ar_all_reduce: "CustomCollective",
        perf_model_extensions.aiter_silu_and_mul: "UnaryElementwise",
    }

    return pseudo_op_categories
