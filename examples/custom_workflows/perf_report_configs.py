from TraceLens.PerfModel.torch_op_mapping import unary_elemwise_ops, binary_elemwise_ops

kernel_categories = [
    "kernel",
    "gpu_memcpy",
    "gpu_memset",
    "cuda_runtime",
    "cuda_driver",
]

# These ops are used for calculating detailed performance metrics for these ops
# https://github.com/AMD-AIG-AIMA/TraceLens/blob/main/TraceLens/PerfModel/torch_op_mapping.py
gemm_perf_ops = [
    "aten::mm",
    "aten::_scaled_mm",
    "aten::addmm",
    "aten::bmm",
    "aten::baddbmm",
    "trtllm::cublas_scaled_mm",
    "bitsandbytes::int8_linear_matmul",
    "_Linear_yfwd_mm",
    "_LayerNormLinear_yfwd_mm",
    "tex_ts::te_gemm_ts",
    "vllm::gemm_with_dynamic_quant",
]

conv_perf_ops = [
    "aten::convolution",
]

# TBA:
# "aten::_efficient_attention_forward",
# "aten::_scaled_dot_product_attention_math",
# "aten::_scaled_dot_product_flash_attention",
# "flash_attn::_flash_attn_varlen_forward",
attn_perf_ops = [
    "aten::_scaled_dot_product_cudnn_attention",
    "aten::_scaled_dot_product_efficient_attention",
    "aten::_scaled_dot_product_flash_attention",
    "flash_attn::_flash_attn_forward",
    "flash_attn::_flash_attn_varlen_forward",
    "flash_attn_3::fwd",
    "aiter::_flash_attn_forward",
    "aiter::wrapper_fmha_v3_fwd",
]

group2ops = {
    "gemm": gemm_perf_ops,
    "conv": conv_perf_ops,
    "attn": attn_perf_ops,
    "un_elemwise": unary_elemwise_ops,
    "bin_elemwise": binary_elemwise_ops,
}

# In addition to gemm and elemwise ops,
# these launchers are used for constructing high-level grouped breakdown from the kernel launchers summary
conv_ops_launchers = [
    "aten::miopen_convolution", "aten::cudnn_convolution",
]

pad_ops_launchers = [
    "aten::replication_pad3d",
]

upsample_ops_launchers = [
    "aten::upsample_nearest3d", "aten::upsample_nearest2d",
    "aten::upsample_bicubic2d",
    "aten::_upsample_nearest_exact2d",
]

norm_ops_launchers = [
    "aten::native_group_norm",
    "aten::native_layer_norm",
    "_C::rms_norm",
]

attn_ops_launchers = [
    "aten::_cudnn_attention_forward",
    "aten::_efficient_attention_forward",
    "flash_attn::_flash_attn_forward",
    "flash_attn::_flash_attn_varlen_forward",
    "xFuserRingFlashAttnFunc",
    "FlashAttnFunc",
    "FlashAttnVarlenFunc",
    "flash_attn_3::fwd",
    "aiter::wrapper_mha_varlen_fwd",
    "aiter::wrapper_fmha_v3_varlen_fwd",
    "aiter::wrapper_fmha_v3_fwd",
    "_vllm_fa3_C::fwd",
    "vllm::unified_attention_with_output",
]

concat_ops_launchers = [
    "aten::cat",
]

moe_ops_launchers = [
    "vllm::moe_forward",
]

all_ops_launchers = gemm_perf_ops + conv_ops_launchers + pad_ops_launchers + upsample_ops_launchers + norm_ops_launchers + attn_ops_launchers + unary_elemwise_ops + binary_elemwise_ops + concat_ops_launchers + moe_ops_launchers

def check_triton_fn(op_name: str) -> bool:
    return op_name.startswith("triton_")
    
grouped_breakdown_mapping = {
    "Attention": attn_ops_launchers,
    "GEMM": gemm_perf_ops,
    "Conv": conv_ops_launchers,
    "Norm": norm_ops_launchers,
    "UpSample": upsample_ops_launchers,
    "Pad": pad_ops_launchers,
    "Elemwise": unary_elemwise_ops + binary_elemwise_ops,
    "Concat": concat_ops_launchers,
    "MoE": moe_ops_launchers,
    "Triton": check_triton_fn,
}