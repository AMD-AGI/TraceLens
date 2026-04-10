###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Extension for pseudo-op operations.
"""

from .moe_perf_model_extensions import (
    FusedMoE,
    moe_aiter_fused_1stage,
    UnfusedMoE_Up,
    UnfusedMoE_Down,
    moe_triton_unfused_up,
    moe_triton_unfused_down,
    moe_gptq_awq_up,
    moe_gptq_awq_down,
)
from .attention_perf_model_extensions import (
    InferenceAttention,
    mha_varlen_fwd,
    vllm_unified_attention_with_output,
    gdn_attention_core,
)

from .perf_model_extensions import (
    gemm_a8w8_blockscale,
    aiter_gelu_and_mul,
    aiter_gelu_tanh_and_mul,
)
from .rmsnorm_perf_model_extensions import (
    aiter_rms_norm,
    aiter_rmsnorm2d_fwd_with_add_ck,
    aiter_rmsnorm2d_fwd_with_dynamicquant_ck,
    vllm_rocm_aiter_rmsnorm_fp8_group_quant,
    vllm_rocm_aiter_rmsnorm_with_add_fp8_group_quant,
)
from .custom_collectives_perf_model_extensions import (
    aiter_fused_allreduce_rmsnorm,
    custom_ar_all_reduce,
    aiter_reduce_scatter,
    aiter_all_gather_reg,
)
from .pseudo_ops_perf_utils import get_pseudo_op_mappings, get_pseudo_op_categories

__all__ = [
    # Base classes
    "FusedMoE",
    "UnfusedMoE_Up",
    "UnfusedMoE_Down",
    "InferenceAttention",
    # Concrete classes
    "moe_aiter_fused_1stage",
    "moe_triton_unfused_up",
    "moe_triton_unfused_down",
    "moe_gptq_awq_up",
    "moe_gptq_awq_down",
    "mha_varlen_fwd",
    "vllm_unified_attention_with_output",
    "gdn_attention_core",
    "gemm_a8w8_blockscale",
    "aiter_gelu_and_mul",
    "aiter_gelu_tanh_and_mul",
    # RMSNorm classes
    "aiter_rms_norm",
    "aiter_rmsnorm2d_fwd_with_add_ck",
    "aiter_rmsnorm2d_fwd_with_dynamicquant_ck",
    "vllm_rocm_aiter_rmsnorm_fp8_group_quant",
    "vllm_rocm_aiter_rmsnorm_with_add_fp8_group_quant",
    # Collective classes
    "aiter_fused_allreduce_rmsnorm",
    "custom_ar_all_reduce",
    "aiter_reduce_scatter",
    "aiter_all_gather_reg",
    # Utility functions
    "get_pseudo_op_mappings",
    "get_pseudo_op_categories",
]
