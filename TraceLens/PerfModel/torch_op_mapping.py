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
    "flash_attn::_flash_attn_backward": perf_model.flash_attention_backward,
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
    "aiter::mha_fwd": perf_model.aiter__mha_fwd,
    "aiter::fmha_v3_fwd": perf_model.aiter__fmha_v3_fwd,
    "aiter::mha_bwd": perf_model.aiter__mha_bwd,
    "flash_attn_3::fwd": perf_model.flash_attn_v3_forward,
    "vllm::unified_attention_with_output": perf_model.vllm_unified_attention_with_output,
    "EvoformerAttention": perf_model.evoformer_attention,
    "LigerSiLUMulFunction": perf_model.liger_silu_mul_function,
    "primus_turbo::grouped_gemm": perf_model.primus_turbo_grouped_gemm,
    "primus_turbo::grouped_gemm_impl": perf_model.primus_turbo_grouped_gemm,
    "primus_turbo_cpp_extension::grouped_gemm": perf_model.primus_turbo_grouped_gemm,
    "primus_turbo::grouped_gemm_variable_k": perf_model.primus_turbo_grouped_gemm_variable_k,
    "primus_turbo::grouped_gemm_variable_k_impl": perf_model.primus_turbo_grouped_gemm_variable_k,
    "primus_turbo_cpp_extension::grouped_gemm_variable_k": perf_model.primus_turbo_grouped_gemm_variable_k,
    # TEv2 pseudo ops
    "_Linear_yfwd_mm": perf_model.tev2_pseudo_gemm,
    "_LinearBackward_xgrad_mm": perf_model.tev2_pseudo_gemm,
    "_LinearBackward_wgrad_mm": perf_model.tev2_pseudo_gemm,
    "_LayerNormLinear_yfwd_mm": perf_model.tev2_pseudo_gemm,
    "_LayerNormLinearBackward_xgrad_mm": perf_model.tev2_pseudo_gemm,
    "_LayerNormLinearBackward_wgrad_mm": perf_model.tev2_pseudo_gemm,
    # CK grouped GEMM (same layout as primus_turbo grouped GEMM)
    "primus_turbo_cpp_extension::ck_grouped_gemm": perf_model.primus_turbo_grouped_gemm,
    "primus_turbo_cpp_extension::ck_grouped_gemm_variable_k": perf_model.primus_turbo_grouped_gemm_variable_k,
    # MoE dispatch/combine (communication — bytes only, flops = 0)
    "MoEDispatch": perf_model.moe_dispatch,
    "MoECombine": perf_model.moe_combine,
    # Causal Conv1D (SSM / Mamba depthwise conv)
    "DaoAILab::_causal_conv1d_fwd_cpp": perf_model.causal_conv1d_fwd,
    # RoPE (elementwise rotation)
    "FusedRoPEFunc": perf_model.fused_rope_fwd,
    # CrossEntropy (fused softmax + nll loss)
    "CrossEntropyFunction": perf_model.cross_entropy_fwd,
    # DeepEP Expert-Parallel communication ops
    "DeepEPDispatch": perf_model.deepep_dispatch,
    "DeepEPCombine": perf_model.deepep_combine,
    "DeepEPDispatchBackward": perf_model.deepep_dispatch_backward,
    "DeepEPCombineBackward": perf_model.deepep_combine_backward,
    # Mamba SSD (fused conv1d + selective scan, issue #552)
    "MambaSplitConv1dScanCombinedFn": perf_model.mamba_ssd_fwd,
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
    "aten::rsqrt",
    "aten::silu",
    "aten::neg",
    "aten::pow",
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

# aten::batch_norm_backward and similar ops do not appear in traces
# add all variants here for coverage
# for fwd path, only aten:: ops validated as they are used
# for bwd path, we see native_ miopen_ and cudnn_ ops, we do not see plain version
# note that the variants have different inputs! This is handled in the perf model classes
norm_ops = {
    "aten::batch_norm": perf_model.BatchNorm,
    "aten::native_batch_norm": perf_model.BatchNorm,
    "aten::miopen_batch_norm": perf_model.BatchNorm,
    "aten::cudnn_batch_norm": perf_model.BatchNorm,
    "aten::layer_norm": perf_model.LayerNorm,
    "aten::native_layer_norm": perf_model.LayerNorm,
    "aten::miopen_layer_norm": perf_model.LayerNorm,
    "aten::cudnn_layer_norm": perf_model.LayerNorm,
    "aten::group_norm": perf_model.GroupNorm,
    "aten::native_group_norm": perf_model.GroupNorm,
    "aten::miopen_group_norm": perf_model.GroupNorm,
    "aten::cudnn_group_norm": perf_model.GroupNorm,
    "aten::instance_norm": perf_model.InstanceNorm,
    "aten::native_instance_norm": perf_model.InstanceNorm,
    "aten::miopen_instance_norm": perf_model.InstanceNorm,
    "aten::cudnn_instance_norm": perf_model.InstanceNorm,
    "aten::_fused_rms_norm": perf_model.RMSNorm,
    "aten::batch_norm_backward": perf_model.BatchNormBwd,
    "aten::native_batch_norm_backward": perf_model.BatchNormBwd,
    "aten::miopen_batch_norm_backward": perf_model.BatchNormBwd,
    "aten::cudnn_batch_norm_backward": perf_model.BatchNormBwd,
    "aten::layer_norm_backward": perf_model.LayerNormBwd,
    "aten::native_layer_norm_backward": perf_model.LayerNormBwd,
    "aten::miopen_layer_norm_backward": perf_model.LayerNormBwd,
    "aten::cudnn_layer_norm_backward": perf_model.LayerNormBwd,
    "aten::group_norm_backward": perf_model.GroupNormBwd,
    "aten::native_group_norm_backward": perf_model.GroupNormBwd,
    "aten::miopen_group_norm_backward": perf_model.GroupNormBwd,
    "aten::cudnn_group_norm_backward": perf_model.GroupNormBwd,
    "aten::instance_norm_backward": perf_model.InstanceNormBwd,
    "aten::native_instance_norm_backward": perf_model.InstanceNormBwd,
    "aten::miopen_instance_norm_backward": perf_model.InstanceNormBwd,
    "aten::cudnn_instance_norm_backward": perf_model.InstanceNormBwd,
    "aten::rms_norm_backward": perf_model.RMSNormBwd,
    "aten::native_rms_norm_backward": perf_model.RMSNormBwd,
    "aten::miopen_rms_norm_backward": perf_model.RMSNormBwd,
    "aten::cudnn_rms_norm_backward": perf_model.RMSNormBwd,
    "aten::_fused_rms_norm_backward": perf_model.RMSNormBwd,
}


# Single-GPU reduce operations (sum, mean, max, min, norm over dimensions)
reduce_ops = [
    "aten::sum",
    "aten::mean",
    "aten::max",
    "aten::min",
    "aten::norm",
    "aten::linalg_norm",
    "aten::std",
    "aten::var",
    "aten::logsumexp",
    "aten::cumsum",
    "aten::cumprod",
    "aten::amin",
    "aten::amax",
]

for op in unary_elemwise_ops:
    op_to_perf_model_class_map[op] = perf_model.aten_unary_elementwise
for op in binary_elemwise_ops:
    op_to_perf_model_class_map[op] = perf_model.aten_binary_elementwise
for op_name, op_class in norm_ops.items():
    op_to_perf_model_class_map[op_name] = op_class
for op in reduce_ops:
    op_to_perf_model_class_map[op] = perf_model.aten_reduce

dict_base_class2category = {
    perf_model.GEMM: "GEMM",
    perf_model.GroupedGemm: "GEMM",
    perf_model.CONV: "CONV",
    perf_model.SDPA: "SDPA",
    perf_model.UnaryElementwise: "UnaryElementwise",
    perf_model.BinaryElementwise: "BinaryElementwise",
    perf_model.Normalization: "Normalization",
    perf_model.Reduce: "Reduce",
    perf_model.MoEComm: "MoE_comm",
    perf_model.CausalConv1d: "SSM",
    perf_model.FusedRoPE: "RoPE",
    perf_model.CrossEntropy: "CrossEntropy",
    perf_model.EPComm: "EP_Communication",
    perf_model.MambaSSD: "SSM",
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
    Categorizes a row based on the 'name' and 'kernel_details' fields.

    Args:
        row (dict): A dictionary with at minimum a 'name' key. May also contain
            a 'kernel_details' key — a list of dicts each having a 'name' field
            that holds the underlying GPU kernel name.

    Returns:
        str: One of 'GEMM', 'CONV_fwd', 'CONV_bwd', 'NORM_fwd', 'NORM_bwd',
             'SDPA_fwd', 'SDPA_bwd', 'EP_Communication', 'MoE_fused', 'MoE_unfused',
             'SSM_fwd', 'SSM_bwd', 'MoE_comm_fwd', 'MoE_comm_bwd',
             'RoPE_fwd', 'RoPE_bwd', 'CrossEntropy_fwd', 'CrossEntropy_bwd',
             'elementwise', 'triton', 'reduce', 'multi_tensor_apply',
             'record_param_comms', or 'other'.

        Note: Backward variants and auxiliary ops (TokenPermuteMaskMap, etc.)
        are categorization-only (timing without GFLOPS or TB/s).
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
    elif row["name"] in norm_ops.keys() or row["name"] in dict_cat2names.get(
        "Normalization", []
    ):
        if row["name"].endswith("_backward") or row["name"].endswith("Backward"):
            return "NORM_bwd"
        else:
            return "NORM_fwd"
    # SDPA ops: distinguish forward and backward
    sdpa_bwd_names = [
        "FlashAttnFuncBackward",
        "FusedAttnFuncBackward",
        "flash_attn::_flash_attn_backward",
        "flash_attn::_flash_attn_varlen_backward",
        "aten::_scaled_dot_product_cudnn_attention_backward",
        "aten::_scaled_dot_product_efficient_attention_backward",
        "aten::_scaled_dot_product_flash_attention_backward",
        "aiter::_flash_attn_backward",
        "aiter::wrapper_fmha_v3_bwd",
        "aiter::mha_bwd",
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
    elif row["name"] in dict_cat2names.get("EP_Communication", []):
        return "EP_Communication"
    elif row["name"] in dict_cat2names.get("SSM", []) or row["name"] in [
        "MambaSplitConv1dScanCombinedFn",
    ]:
        return "SSM_fwd"
    elif row["name"] in [
        "MambaSplitConv1dScanCombinedFnBackward",
        "DaoAILab::_causal_conv1d_bwd_cpp",
    ]:
        return "SSM_bwd"
    elif row["name"] in dict_cat2names.get("MoE_comm", []) or row["name"] in [
        "TokenPermuteMaskMap",
        "_OperationFuserAutogradFunction",
    ]:
        return "MoE_comm_fwd"
    elif row["name"] in [
        "MoEDispatchBackward",
        "MoECombineBackward",
        "TokenPermuteMaskMapBackward",
        "_OperationFuserAutogradFunctionBackward",
    ]:
        return "MoE_comm_bwd"
    elif row["name"] in dict_cat2names.get("RoPE", []):
        return "RoPE_fwd"
    elif row["name"] in ["FusedRoPEFuncBackward"]:
        return "RoPE_bwd"
    elif row["name"] in dict_cat2names.get("CrossEntropy", []):
        return "CrossEntropy_fwd"
    elif row["name"] in ["CrossEntropyFunctionBackward"]:
        return "CrossEntropy_bwd"
    elif row["name"] in dict_cat2names.get("BinaryElementwise", []):
        return "elementwise"
    elif row["name"] in dict_cat2names.get("Reduce", []):
        return "reduce"
    elif row["name"].startswith("triton"):
        return "triton"
    elif row["name"] in [
        "aiter::moe_sorting_fwd",
        "aiter::moe_sorting_opus_fwd",
        "aiter::moe_align_block_size",
        "_moe_C::moe_align_block_size",
        "aiter::fused_moe_->_fused_dynamic_mxfp4_quant_moe_sort_kernel (Synthetic Op)",
    ]:
        return "MoE_aux"
    elif row["name"] in [
        "aiter::moe_sum",
    ]:
        return "MoE_aux"
    elif row["name"] in [
        "aiter::topk_softmax",
        "aiter::topk_softmax_asm",
        "aiter::topk_sigmoid",
        "aiter::biased_grouped_topk_hip",
        "aiter::grouped_topk",
        "aiter::moe_fused_gate",
    ]:
        return "MoE_aux"
    elif row["name"] in [
        "_C_cache_ops::reshape_and_cache_flash",
        "_C_cache_ops::concat_and_cache_mla",
    ]:
        return "InferenceAttention"
    elif row["name"].startswith("record_param_comms"):
        return "record_param_comms"
    elif row["name"] in dict_cat2names.get("MoE_fused", []):
        return "MoE_fused"
    elif row["name"] in dict_cat2names.get("MoE_unfused", []):
        return "MoE_unfused"
    elif row["name"] in dict_cat2names.get("InferenceAttention", []):
        return "InferenceAttention"
    elif row["name"] in dict_cat2names.get("RMSNorm", []):
        return "RMSNorm"
    elif row["name"] in dict_cat2names.get("CustomCollective", []):
        return "CustomCollective"
    elif row["name"] in dict_cat2names.get("GroupQuant", []):
        return "GroupQuant"
    elif row["name"] in dict_cat2names.get("BinaryElementwise", []):
        return "elementwise"
    elif row["name"] in dict_cat2names.get("UnaryElementwise", []):
        return "elementwise"
    elif row["name"] in dict_cat2names.get("Reduce", []):
        return "reduce"
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
