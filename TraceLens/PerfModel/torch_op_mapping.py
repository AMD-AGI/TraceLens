###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from . import perf_model
from collections import defaultdict
from .extensions import get_pseudo_op_mappings, get_pseudo_op_categories

op_to_perf_model_class_map = {
    "aten::mm": perf_model.aten_mm,
    "aten::addmm": perf_model.aten_addmm,
    "aten::_scaled_mm": perf_model.aten_scaled_mm,
    "trtllm::cublas_scaled_mm": perf_model.aten_scaled_mm,
    "bitsandbytes::int8_linear_matmul": perf_model.aten_scaled_mm,
    "aten::bmm": perf_model.aten_bmm,
    "tex_ts::te_gemm_ts": perf_model.tex_ts_te_gemm_ts,
    "aten::baddbmm": perf_model.aten_baddbmm,
    "vllm::gemm_with_dynamic_quant": perf_model.vllm_gemm_with_dynamic_quant,
    "FlashAttnFunc": perf_model.flash_attention,
    "flash_attn::_flash_attn_forward": perf_model.flash_attention,
    "flash_attn::_flash_attn_varlen_forward": perf_model.flash_attention_varlen_forward,
    "flash_attn::_flash_attn_varlen_backward": perf_model.flash_attention_varlen_backward,
    "aten::_scaled_dot_product_cudnn_attention": perf_model.aten__scaled_dot_product_cudnn_attention,
    "aten::_scaled_dot_product_efficient_attention": perf_model.aten__scaled_dot_product_efficient_attention,
    "aten::_scaled_dot_product_flash_attention": perf_model.aten__scaled_dot_product_flash_attention,
    "aten::convolution": perf_model.aten_conv,
    "aten::convolution_backward": perf_model.aten_conv_bwd,
    "ConvBias_": perf_model.ConvBias_,
    "ConvBiasReLU_": perf_model.ConvBiasReLU_,
    "ConvBias_Backward": perf_model.ConvBias_Backward,
    "ConvBiasReLU_Backward": perf_model.ConvBiasReLU_Backward,
    "aiter::_flash_attn_forward": perf_model.aiter__flash_attn_forward,
    "aiter::_flash_attn_backward": perf_model.aiter__flash_attn_backward,
    "aiter::wrapper_fmha_v3_fwd": perf_model.aiter__fmha_v3_forward,
    "aiter::wrapper_fmha_v3_bwd": perf_model.aiter__fmha_v3_backward,
    "flash_attn_3::fwd": perf_model.flash_attn_v3_forward,
    "vllm::unified_attention_with_output": perf_model.vllm_unified_attention_with_output,
    # TEv2 pseudo ops
    "_Linear_yfwd_mm": perf_model.tev2_pseudo_gemm,
    "_LinearBackward_xgrad_mm": perf_model.tev2_pseudo_gemm,
    "_LinearBackward_wgrad_mm": perf_model.tev2_pseudo_gemm,
    "_LayerNormLinear_yfwd_mm": perf_model.tev2_pseudo_gemm,
    "_LayerNormLinearBackward_xgrad_mm": perf_model.tev2_pseudo_gemm,
    "_LayerNormLinearBackward_wgrad_mm": perf_model.tev2_pseudo_gemm,
    }

# Add pseudo-op extension mappings
op_to_perf_model_class_map.update(get_pseudo_op_mappings())

unary_elemwise_ops = [
    "aten::copy",
    "aten::copy_",
    "aten::clamp_min",
    "aten::clamp_min_",
    "aten::clamp_max",
    "aten::clamp_max_",
    "aten::sigmoid",
]

binary_elemwise_ops = [
    "aten::div",
    "aten::div_",
    "aten::mul",
    "aten::mul_",
    "aten::add",
    "aten::add_",
    "aten::sigmoid_backward",
    "aten::threshold_backward",
]

for op in unary_elemwise_ops:
    op_to_perf_model_class_map[op] = perf_model.aten_unary_elementwise
for op in binary_elemwise_ops:
    op_to_perf_model_class_map[op] = perf_model.aten_binary_elementwise

dict_base_class2category = {
    perf_model.GEMM: "GEMM",
    perf_model.CONV: "CONV",
    perf_model.SDPA: "SDPA",
    perf_model.UnaryElementwise: "UnaryElementwise",
    perf_model.BinaryElementwise: "BinaryElementwise",
}

# Add pseudo-op extension categories
dict_base_class2category.update(get_pseudo_op_categories())

dict_cat2names = defaultdict(list)
for op_name, perf_model_class in op_to_perf_model_class_map.items():
    base_classes = perf_model_class.__bases__
    assert (
        len(base_classes) == 1
    ), f"op_name: {op_name}, perf_model_class: {perf_model_class}, base_classes: {base_classes}"
    base_class = base_classes[0]
    cat = dict_base_class2category.get(base_class)
    if cat is None:
        raise ValueError(
            f"op_name: {op_name}, perf_model_class: {perf_model_class}, base_class: {base_classes}"
        )
    dict_cat2names[cat].append(op_name)


def categorize_torch_op(row):
    """
    Categorizes a row based on the 'name' and 'kernel_names' fields.
    Args:
        row (dict): A dictionary representing a row with 'name' and 'kernel_names' keys.
    Returns:
        str: The category of the row, which can be one of 'GEMM', 'CONV_fwd', 'CONV_bwd', 'BN_fwd', 'BN_bwd',
             'SDPA_fwd', 'SDPA_bwd', 'triton', 'elementwise', 'reduce', 'multi_tensor_apply', or 'other'.
    """

    debug = False
    if row["name"] in dict_cat2names["GEMM"]:
        return "GEMM"
    elif row["name"] in [
        "aten::convolution",
        "aten::miopen_convolution",
        "aten::cudnn_convolution",
        "ConvBias_",
        "ConvBiasReLU_",
    ]:
        return "CONV_fwd"
    elif row["name"] in [
        "aten::convolution_backward",
        "ConvBias_Backward",
        "ConvBiasReLU_Backward",
    ]:
        return "CONV_bwd"
    elif row["name"] in [
        "aten::batch_norm",
        "aten::native_batch_norm",
        "aten::miopen_batch_norm",
        "aten::cudnn_batch_norm",
    ]:
        return "BN_fwd"
    elif row["name"] in [
        "aten::native_batch_norm_backward",
        "aten::miopen_batch_norm_backward",
        "aten::cudnn_batch_norm_backward",
    ]:
        return "BN_bwd"
    # SDPA ops: distinguish forward and backward
    sdpa_bwd_names = [
        "FlashAttnFuncBackward",
        "flash_attn::_flash_attn_backward",
        "flash_attn::_flash_attn_varlen_backward",
        "aten::_scaled_dot_product_cudnn_attention_backward",
        "aten::_scaled_dot_product_efficient_attention_backward",
        "aten::_scaled_dot_product_flash_attention_backward",
        "aiter::_flash_attn_backward",
        "aiter::wrapper_fmha_v3_bwd",
    ]
    if row["name"] in dict_cat2names["SDPA"]:
        if row["name"].endswith("_backward") or row["name"] in sdpa_bwd_names:
            return "SDPA_bwd"
        else:
            return "SDPA_fwd"
    elif row["name"] in dict_cat2names.get("MoE_fused", []):
        return "MoE_fused"
    elif row["name"] in dict_cat2names.get("MoE_unfused", []):
        return "MoE_unfused"
    elif row["name"].startswith("triton"):
        return "triton"
    elif row["name"].startswith("record_param_comms"):
        return "record_param_comms"
    if "kernel_details" in row and len(row["kernel_details"]) > 0:
        kernel_name = row["kernel_details"][0]["name"]
        # else:
        #     raise ValueError(
        #         f"Row does not contain 'kernel_names' or 'kernel_details' with a valid name. Row: {row}"
        #     )
        if kernel_name.startswith("void at::native"):
            if debug:
                print("Found ATen native kernel:", kernel_name[:64])
            if "elementwise" in kernel_name:
                return "elementwise"
            elif "reduce" in kernel_name:
                return "reduce"
            elif "multi_tensor_apply" in kernel_name:
                return "multi_tensor_apply"

    # if none of the above cases match, return 'other'
    return "other"
