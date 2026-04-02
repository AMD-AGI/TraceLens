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
)
from .attention_perf_model_extensions import (
    InferenceAttention,
    mha_varlen_fwd,
    vllm_unified_attention_with_output,
)

from .perf_model_extensions import (
    gemm_a8w8_blockscale,
)
from .rmsnorm_perf_model_extensions import (
    aiter_rms_norm,
    vllm_rocm_aiter_rmsnorm_fp8_group_quant,
    vllm_rocm_aiter_rmsnorm_with_add_fp8_group_quant,
)
from .collectives_perf_model_extensions import (
    aiter_fused_allreduce_rmsnorm,
    custom_ar_all_reduce,
)
from .pseudo_ops_perf_utils import get_pseudo_op_mappings, get_pseudo_op_categories

__all__ = [
    # Base classes
    'FusedMoE',
    'UnfusedMoE_Up',
    'UnfusedMoE_Down',
    "InferenceAttention",
    # Concrete classes
    'moe_aiter_fused_1stage',
    'moe_triton_unfused_up',
    'moe_triton_unfused_down',
    'mha_varlen_fwd',
    'vllm_unified_attention_with_output',
    'gemm_a8w8_blockscale',
    # RMSNorm classes
    'aiter_rms_norm',
    'vllm_rocm_aiter_rmsnorm_fp8_group_quant',
    'vllm_rocm_aiter_rmsnorm_with_add_fp8_group_quant',
    # Collective classes
    'aiter_fused_allreduce_rmsnorm',
    'custom_ar_all_reduce',
    # Utility functions
    'get_pseudo_op_mappings',
    'get_pseudo_op_categories',
]
