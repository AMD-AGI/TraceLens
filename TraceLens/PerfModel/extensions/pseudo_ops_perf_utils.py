###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
        "aiter::fmoe_g1u1": moe_perf_model_extensions.moe_aiter_fused_blockscale,
        # MoE pseudo ops - Unfused Triton (2-stage: up and down)
        "pseudo_op::moe_triton_unfused_up": moe_perf_model_extensions.moe_triton_unfused_up,
        "pseudo_op::moe_triton_unfused_down": moe_perf_model_extensions.moe_triton_unfused_down,
        # MoE - GPTQ/AWQ quantized unfused implementation (vllm::outplace_fused_experts)
        "pseudo_op::moe_gptq_awq_up": moe_perf_model_extensions.moe_gptq_awq_up,
        "pseudo_op::moe_gptq_awq_down": moe_perf_model_extensions.moe_gptq_awq_down,
        # MoE flydsl two-stage (under aiter::fused_moe_)
        "pseudo_op::moe_flydsl_stage1": moe_perf_model_extensions.moe_flydsl_stage1,
        "pseudo_op::moe_flydsl_stage2": moe_perf_model_extensions.moe_flydsl_stage2,
        "sglang_profiler::fused_moe_triton_kernels_invoke_fused_moe_kernel": moe_perf_model_extensions.moe_triton_invoke_grouped_gemm,
        # Attention pseudo ops
        "vllm::unified_attention_with_output": attention_perf_model_extensions.vllm_unified_attention_with_output,
        "aiter::mha_varlen_fwd": attention_perf_model_extensions.mha_varlen_fwd,
        "aiter::fmha_v3_varlen_fwd": attention_perf_model_extensions.aiter_fmha_v3_varlen_fwd,
        "aiter::mha_batch_prefill": attention_perf_model_extensions.aiter_mha_batch_prefill,
        "sglang_profiler::attention_paged_attention_ragged": attention_perf_model_extensions.aiter_paged_attention_ragged,
        "pseudo_mla_decode_fwd": attention_perf_model_extensions.mla_decode_fwd,
        "pseudo_mla_prefill_fwd": attention_perf_model_extensions.pseudo_mla_prefill_fwd,
        "aiter::pa_decode_gluon": attention_perf_model_extensions.pa_decode_gluon,
        "sglang_profiler::tilelang_kernel_tilelang_sparse_fwd": attention_perf_model_extensions.mla_tilelang_sparse_fwd,
        ## Misc ops
        "aiter::batched_gemm_a16wfp4_": perf_model_extensions.batched_gemm_a16wfp4,
        "aiter::dynamic_per_token_scaled_quant": perf_model_extensions.per_group_quant,
        "aiter::dynamic_per_group_scaled_quant": perf_model_extensions.per_group_quant,
        "aiter::fused_add_rmsnorm_pad_": rmsnorm_perf_model_extensions.vllm_rocm_aiter_triton_add_rmsnorm_pad,
        "aiter::mixed_sample_outer_exponential": perf_model_extensions.mixed_sample_outer_exponential,
        "sglang_profiler::fp8_utils_gemm_a8w8_blockscale": perf_model_extensions.gemm_a8w8_blockscale,
        "sglang_profiler::gemm_a8w8_blockscale_gemm_a8w8_blockscale": perf_model_extensions.gemm_a8w8_blockscale,
        "vllm::rocm_aiter_triton_gemm_a8w8_blockscale": perf_model_extensions.gemm_a8w8_blockscale,
        "aiter::gemm_a8w8_blockscale_ck": perf_model_extensions.gemm_a8w8_blockscale,
        "aiter::gemm_a8w8_blockscale_cktile": perf_model_extensions.gemm_a8w8_blockscale,
        "aiter::gemm_a8w8_blockscale_bpreshuffle_ck": perf_model_extensions.gemm_a8w8_blockscale,
        "aiter::gemm_a8w8_blockscale_bpreshuffle_cktile": perf_model_extensions.gemm_a8w8_blockscale,
        "aiter::flatmm_a8w8_blockscale_asm": perf_model_extensions.gemm_a8w8_blockscale,
        "aiter::gfx950_a8w8_blockscale_asm": perf_model_extensions.gemm_a8w8_blockscale,
        "aiter::gemm_a8w8_ck": perf_model_extensions.gemm_a8w8_blockscale,
        "aiter::gemm_a8w8_bpreshuffle_ck": perf_model_extensions.gemm_a8w8_blockscale,
        "aiter::gemm_a8w8_bpreshuffle": perf_model_extensions.gemm_a8w8_blockscale,
        "vllm::_rocm_aiter_preshuffled_per_token_w8a8_gemm": perf_model_extensions.gemm_a8w8_blockscale,
        "vllm::rocm_unquantized_gemm": perf_model_extensions.vllm_rocm_unquantized_gemm,
        "aiter::gemm_a16w16": perf_model_extensions.gemm_a16w16,
        "aiter::gemm_a16w16_atomic_": perf_model_extensions.gemm_a16w16_atomic_,
        "aiter::_gemm_a16w16_asm": perf_model_extensions.gemm_a16w16_atomic_,
        "sglang_profiler::gemm_kernels_flydsl_hgemm": perf_model_extensions.gemm_a16w16_atomic_,
        "aiter::gemm_afp4wfp4_": perf_model_extensions.gemm_afp4wfp4,
        "sglang_profiler::batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant_batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant": perf_model_extensions.batched_gemm_a8w8,
        ## Quantization ops
        "vllm::triton_per_token_group_quant_fp8": perf_model_extensions.vllm_triton_per_token_group_quant_fp8,
        "sglang_profiler::fused_mxfp4_quant_fused_flatten_mxfp4_quant": perf_model_extensions.fused_flatten_mxfp4_quant,
        "aiter::dynamic_per_tensor_quant": perf_model_extensions.per_group_quant,
        ## RoPE ops
        "aiter::rope_cached_positions_2c_fwd_impl": perf_model_extensions.aiter_rope_cached_positions_2c_fwd_impl,
        "sgl_kernel::rotary_embedding": perf_model_extensions.sgl_kernel_rotary_embedding,
        ## Activation ops
        "aiter::silu_and_mul": perf_model_extensions.aiter_silu_and_mul,
        "_C::silu_and_mul": perf_model_extensions.aiter_silu_and_mul,
        "sgl_kernel::silu_and_mul": perf_model_extensions.sgl_kernel_silu_and_mul,
        "aiter::gelu_and_mul": perf_model_extensions.aiter_gelu_and_mul,
        "aiter::gelu_tanh_and_mul": perf_model_extensions.aiter_gelu_tanh_and_mul,
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
        "sglang_profiler::fused_mxfp4_quant_fused_rms_mxfp4_quant": rmsnorm_perf_model_extensions.fused_rms_mxfp4_quant,
        ## Collective ops
        "aiter::fused_allreduce_rmsnorm": custom_collectives_perf_model_extensions.aiter_fused_allreduce_rmsnorm,
        "aiter::fused_allreduce_rmsnorm_": custom_collectives_perf_model_extensions.aiter_fused_allreduce_rmsnorm_,
        "_C_custom_ar::all_reduce": custom_collectives_perf_model_extensions.custom_ar_all_reduce,
        "aiter::reduce_scatter": custom_collectives_perf_model_extensions.aiter_reduce_scatter,
        "aiter::all_gather_reg": custom_collectives_perf_model_extensions.aiter_all_gather_reg,
        "sgl_kernel::all_reduce_reg": custom_collectives_perf_model_extensions.sgl_kernel_all_reduce_reg,
        "sgl_kernel::qr_all_reduce": custom_collectives_perf_model_extensions.sgl_kernel_qr_all_reduce,
        "sglang::reg_all_gather_into_tensor": custom_collectives_perf_model_extensions.sgl_kernel_reg_all_gather_into_tensor,
        "_C_custom_ar::qr_all_reduce": custom_collectives_perf_model_extensions.custom_ar_qr_all_reduce,
        ## GDN attention ops
        "vllm::gdn_attention_core": attention_perf_model_extensions.gdn_attention_core,
        "aiter::linear_attention_with_output_base": attention_perf_model_extensions.gdn_attention_core,
        ## KV cache ops
        "sglang::store_cache": perf_model_extensions.sglang_store_cache,
        "aiter::fused_qk_rope_cat_and_cache_mla": perf_model_extensions.aiter_fused_qk_rope_cat_and_cache_mla,
        "sglang_profiler::fused_moe_triton_kernels_fused_append_shared_experts": moe_perf_model_extensions.sglang_fused_append_shared_experts,
        "sglang_profiler::quant_dynamic_mxfp4_quant": perf_model_extensions.sglang_quant_dynamic_mxfp4_quant,
        "aiter::fused_dynamic_mxfp4_quant_moe_sort_hip": perf_model_extensions.aiter_fused_dynamic_mxfp4_quant_moe_sort_hip,
        "aiter::dynamic_per_group_scaled_quant_fp4": perf_model_extensions.aiter_dynamic_per_group_scaled_quant_fp4,
    }

    return pseudo_op_mappings


def get_pseudo_op_category_only_mappings():
    """
    Return a dictionary mapping pseudo-op names to category labels only.

    These ops do not have a full performance model class but should still be
    classified (and not bucketed under "other") for category-level analysis.

    Returns:
        dict: Mapping of op names to category strings.
    """

    return {
        # MoE sorting / permutation auxiliary kernel.
        # Reference: aiter/aiter/ops/triton/moe_op_mxfp4.py (mxfp4_moe_sort_hip).
        "aiter::mxfp4_moe_sort_hip": "MoE_aux",
        "aiter::fused_dynamic_mxfp4_quant_moe_sort_hip": "MoE_aux",
        "aiter::unified_attention_with_output_base->_fused_qk_rope_reshape_and_cache_kernel (Synthetic Op)": "FusedRoPE",
        "hipModuleLaunchKernel->kv_indices_generate_kernel (Synthetic Op)": "InferenceAttention",
    }
