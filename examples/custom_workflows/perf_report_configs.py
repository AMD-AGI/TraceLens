# These ops are used for calculating detailed performance metrics for these ops
# https://github.com/AMD-AIG-AIMA/TraceLens/blob/main/TraceLens/PerfModel/torch_op_mapping.py
gemm_ops = [
    "aten::mm", "aten::_scaled_mm", "aten::addmm", "aten::bmm", "aten::baddbmm",
]

conv_ops = [
    "aten::convolution",
]

# TBA:
# "aten::_efficient_attention_forward",
# "aten::_scaled_dot_product_attention_math",
# "aten::_scaled_dot_product_efficient_attention",
# "aten::_scaled_dot_product_flash_attention",
# "flash_attn::_flash_attn_varlen_forward",
attn_ops = [
    "FlashAttnFunc",
    "flash_attn::_flash_attn_forward",
    "aten::_scaled_dot_product_cudnn_attention",
]

unary_elemwise_ops = [
    "aten::copy", "aten::copy_",
    "aten::clamp_min", "aten::clamp_min_",
    "aten::sigmoid",
]

binary_elemwise_ops = [
    "aten::div", "aten::div_",
    "aten::mul", "aten::mul_",
    "aten::add", "aten::add_",
    "aten::sigmoid_backward",
    "aten::threshold_backward",
]

group2ops = {
    "gemm": gemm_ops,
    "conv": conv_ops,
    "attn": attn_ops,
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
]

norm_ops_launchers = [
    "aten::native_group_norm",
]

attn_ops_launchers = [
    "aten::_scaled_dot_product_cudnn_attention",
    "flash_attn::_flash_attn_forward",
    "flash_attn::_flash_attn_varlen_forward",
    "aten::_efficient_attention_forward",
]

concat_ops_launchers = [
    "aten::cat",
]

all_ops_launchers = gemm_ops + conv_ops_launchers + pad_ops_launchers + upsample_ops_launchers + norm_ops_launchers + attn_ops_launchers + unary_elemwise_ops + binary_elemwise_ops + concat_ops_launchers

def check_triton_fn(op_name: str) -> bool:
    return op_name.startswith("triton_")
    
grouped_breakdown_mapping = {
    "Attention": attn_ops_launchers,
    "GEMM": gemm_ops,
    "Conv": conv_ops_launchers,
    "Norm": norm_ops_launchers,
    "UpSample": upsample_ops_launchers,
    "Pad": pad_ops_launchers,
    "Elemwise": unary_elemwise_ops + binary_elemwise_ops,
    "Concat": concat_ops_launchers,
    "Triton": check_triton_fn,
}