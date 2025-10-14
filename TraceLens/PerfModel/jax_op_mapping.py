###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from . import perf_model

jax_op_to_perf_model_class_map = {
    "jax_gemm": perf_model.jax_gemm, # same as JaxAnalyses.JaxGemm
    "jax_te_fused_attn": perf_model.jax_te_fused_attn,
    "jax_te_fused_attn_bwd": perf_model.jax_te_fused_attn, 
    "jax_conv": perf_model.jax_conv,
    "jax_conv_bwd": perf_model.jax_conv, 
}

