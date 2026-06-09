###############################################################################
# Copyright (c) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Torch op-name mappings and categorization helpers."""

import re
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping
from typing import Optional, Pattern, Tuple, Union

from . import perf_model
from .extensions import (
    get_pseudo_op_mappings,
    get_pseudo_op_category_only_mappings,
)

CATEGORY_ONLY_OP_MAPPING: Dict[str, str] = {
    # CONV ops not present in op_to_perf_model_class_map.
    "aten::miopen_convolution": "CONV_fwd",
    "aten::cudnn_convolution": "CONV_fwd",
    # SDPA backward ops without direct perf models in core TraceLens.
    "FlashAttnFuncBackward": "SDPA_bwd",
    "FusedAttnFuncBackward": "SDPA_bwd",
    "aten::_scaled_dot_product_cudnn_attention_backward": "SDPA_bwd",
    "aten::_scaled_dot_product_efficient_attention_backward": "SDPA_bwd",
    "aten::_scaled_dot_product_flash_attention_backward": "SDPA_bwd",
    # SSM / Mamba category-only backward ops.
    "MambaSplitConv1dScanCombinedFnBackward": "SSM_bwd",
    "DaoAILab::_causal_conv1d_bwd_cpp": "SSM_bwd",
    # MoE communication category-only ops.
    "TokenPermuteMaskMap": "MoE_comm_fwd",
    # Observed in MoE token-routing traces; tracked separately because the name
    # itself is generic and may not always imply MoE communication.
    "_OperationFuserAutogradFunction": "MoE_comm_fwd",
    "MoEDispatchBackward": "MoE_comm_bwd",
    "MoECombineBackward": "MoE_comm_bwd",
    "TokenPermuteMaskMapBackward": "MoE_comm_bwd",
    "_OperationFuserAutogradFunctionBackward": "MoE_comm_bwd",
    # RoPE / CrossEntropy category-only backward ops.
    "FusedRoPEFuncBackward": "RoPE_bwd",
    "CrossEntropyFunctionBackward": "CrossEntropy_bwd",
    # MoE auxiliary ops.
    "aiter::moe_sorting_fwd": "MoE_aux",
    "aiter::moe_sorting_opus_fwd": "MoE_aux",
    "aiter::moe_align_block_size": "MoE_aux",
    "_moe_C::moe_align_block_size": "MoE_aux",
    "aiter::fused_moe_->_fused_dynamic_mxfp4_quant_moe_sort_kernel (Synthetic Op)": "MoE_aux",
    "aiter::moe_sum": "MoE_aux",
    "aiter::topk_softmax": "MoE_aux",
    "aiter::topk_softmax_asm": "MoE_aux",
    "aiter::topk_sigmoid": "MoE_aux",
    "aiter::biased_grouped_topk_hip": "MoE_aux",
    "aiter::grouped_topk": "MoE_aux",
    "aiter::moe_fused_gate": "MoE_aux",
    # InferenceAttention extras (KV-cache writes).
    "_C_cache_ops::reshape_and_cache_flash": "InferenceAttention",
    "_C_cache_ops::concat_and_cache_mla": "InferenceAttention",
}


OP_CATEGORY_PATTERNS: List[Tuple[Pattern, str]] = [
    (re.compile(r"^triton"), "triton"),
    (re.compile(r"^record_param_comms"), "record_param_comms"),
]


SYNTHETIC_OP_MARKER = " (Synthetic Op)"
synthetic_op_marker = SYNTHETIC_OP_MARKER

_KERNEL_NAME_PREFIX = "void at::native"
_ATEN_NATIVE_CATEGORY_RULES: Tuple[Tuple[str, str], ...] = (
    ("elementwise", "elementwise"),
    ("reduce", "reduce"),
    ("multi_tensor_apply", "multi_tensor_apply"),
)
_ATEN_NATIVE_PERF_MODEL_RULES: Tuple[Tuple[str, type], ...] = (
    ("elementwise", perf_model.aten_unary_elementwise),
    ("reduce", perf_model.aten_reduce),
)

# Ordered classify_kernel() output rules: (kind, key, value).
# kind is one of name_substr, kernel_type, perf_category, kernel_type_substr.
_ClassifiedRuleValue = Union[str, type]
_CLASSIFIED_KERNEL_CATEGORY_RULES: Tuple[Tuple[str, str, str], ...] = (
    ("name_substr", "rmsnorm", "RMSNorm"),
    ("kernel_type", "KV Cache Store", "InferenceAttention"),
    ("kernel_type", "MoE GEMM", "MoE_unfused"),
    ("kernel_type", "GEMM", "GEMM"),
    ("kernel_type", "Quantization", "GroupQuant"),
    ("kernel_type", "Attention", "SDPA_fwd"),
    ("kernel_type", "Linear Attention", "SDPA_fwd"),
    ("kernel_type", "GDN Gating", "SDPA_fwd"),
    ("kernel_type", "Rotary Embedding", "RoPE_fwd"),
    ("kernel_type", "RMSNorm", "RMSNorm"),
    ("kernel_type", "LayerNorm", "NORM_fwd"),
    ("kernel_type", "L2Norm", "NORM_fwd"),
    ("kernel_type", "Normalization", "NORM_fwd"),
    ("kernel_type", "MoE Routing", "MoE_aux"),
    ("kernel_type", "MoE Finalize", "MoE_aux"),
    ("kernel_type", "MoE Quantize", "MoE_aux"),
    ("kernel_type", "Activation (DeepSeek fused)", "MoE_aux"),
    ("kernel_type", "Elementwise", "elementwise"),
    ("kernel_type", "MemCpy", "other"),
    ("perf_category", "GEMM-MoE", "MoE_unfused"),
    ("perf_category", "GEMM", "GEMM"),
    ("perf_category", "Quantization", "GroupQuant"),
    ("perf_category", "SDPA", "SDPA_fwd"),
    ("perf_category", "SDPA-GDN", "SDPA_fwd"),
    ("perf_category", "Normalization", "NORM_fwd"),
    ("perf_category", "Elementwise-MoE", "MoE_aux"),
    ("perf_category", "Elementwise", "elementwise"),
    ("perf_category", "MemCpy", "other"),
    ("kernel_type_substr", "moe gemm", "MoE_unfused"),
    ("kernel_type_substr", "attention", "SDPA_fwd"),
    ("kernel_type_substr", "quant", "GroupQuant"),
)
_CLASSIFIED_KERNEL_PERF_MODEL_RULES: Tuple[Tuple[str, str, type], ...] = (
    ("name_substr", "rmsnorm", perf_model.RMSNorm),
    ("kernel_type", "MoE GEMM", perf_model.primus_turbo_grouped_gemm),
    ("kernel_type", "GEMM", perf_model.aten_mm),
    ("kernel_type", "Quantization", perf_model.primus_turbo_quantize_fp8),
    ("kernel_type", "Attention", perf_model.flash_attention),
    ("kernel_type", "Linear Attention", perf_model.flash_attention),
    ("kernel_type", "GDN Gating", perf_model.flash_attention),
    ("kernel_type", "Rotary Embedding", perf_model.fused_rope_fwd),
    ("kernel_type", "RMSNorm", perf_model.RMSNorm),
    ("kernel_type", "LayerNorm", perf_model.LayerNorm),
    ("kernel_type", "L2Norm", perf_model.LayerNorm),
    ("kernel_type", "Normalization", perf_model.LayerNorm),
    ("kernel_type", "Elementwise", perf_model.aten_binary_elementwise),
    ("perf_category", "GEMM-MoE", perf_model.primus_turbo_grouped_gemm),
    ("perf_category", "GEMM", perf_model.aten_mm),
    ("perf_category", "Quantization", perf_model.primus_turbo_quantize_fp8),
    ("perf_category", "SDPA", perf_model.flash_attention),
    ("perf_category", "SDPA-GDN", perf_model.flash_attention),
    ("perf_category", "Normalization", perf_model.LayerNorm),
    ("perf_category", "Elementwise", perf_model.aten_binary_elementwise),
    ("kernel_type_substr", "moe gemm", perf_model.primus_turbo_grouped_gemm),
    ("kernel_type_substr", "attention", perf_model.flash_attention),
    ("kernel_type_substr", "quant", perf_model.primus_turbo_quantize_fp8),
)
_classified_kernel_category_rule_extensions: List[Tuple[str, str, str]] = []
_classified_kernel_perf_model_rule_extensions: List[Tuple[str, str, type]] = []


def register_classified_kernel_category_rules(
    rules: Iterable[Tuple[str, str, str]],
) -> None:
    """Register extra classify_kernel output -> category rules."""
    _classified_kernel_category_rule_extensions.extend(rules)


def register_classified_kernel_perf_model_rules(
    rules: Iterable[Tuple[str, str, type]],
) -> None:
    """Register extra classify_kernel output -> perf model class rules."""
    _classified_kernel_perf_model_rule_extensions.extend(rules)


def _kernel_name_from_row(row) -> Optional[str]:
    """Resolve GPU kernel name from kernel_details or graph-replay synthetic op name."""
    kernel_details = row.get("kernel_details")
    if kernel_details:
        return kernel_details[0].get("name") or None
    op_name = row.get("name", "")
    if SYNTHETIC_OP_MARKER in op_name and "->" in op_name:
        return op_name.split("->", 1)[1].rsplit(SYNTHETIC_OP_MARKER, 1)[0]
    return None


def _lookup_classified_kernel(
    kernel_name: str,
    rules: Tuple[Tuple[str, str, _ClassifiedRuleValue], ...],
) -> Optional[_ClassifiedRuleValue]:
    """Map classify_kernel() output to a category or perf model using ordered rules."""
    from TraceLens.Agent.Analysis.utils.classify_kernels import classify_kernel

    kt, pc, conf = classify_kernel(kernel_name)
    if not conf:
        return None

    knl = kernel_name.lower()
    ktl = kt.lower()
    for kind, key, value in rules:
        if kind == "name_substr" and key in knl:
            return value
        if kind == "kernel_type" and kt == key:
            return value
        if kind == "perf_category" and pc == key:
            return value
        if kind == "kernel_type_substr" and key in ktl:
            return value
    return None


def _category_from_classified_kernel(kernel_name: str) -> Optional[str]:
    rules = _CLASSIFIED_KERNEL_CATEGORY_RULES + tuple(
        _classified_kernel_category_rule_extensions
    )
    category = _lookup_classified_kernel(kernel_name, rules)
    if category is not None:
        return category

    from TraceLens.Agent.Analysis.utils.classify_kernels import classify_kernel

    kt, _, conf = classify_kernel(kernel_name)
    if conf and kt.lower().startswith("moe "):
        return "MoE_aux"
    return None


def _perf_model_class_from_classified_kernel(kernel_name: str) -> Optional[type]:
    rules = tuple(_classified_kernel_perf_model_rule_extensions) + _CLASSIFIED_KERNEL_PERF_MODEL_RULES
    model_class = _lookup_classified_kernel(kernel_name, rules)
    if model_class is not None:
        return model_class

    from TraceLens.Agent.Analysis.utils.classify_kernels import classify_kernel

    kt, _, conf = classify_kernel(kernel_name)
    if conf and kt.lower().startswith("moe "):
        return None
    return None


def _kernel_name_fallback(row) -> Optional[str]:
    """Category fallback for rows with GPU kernel context but no registry hit."""
    kernel_name = _kernel_name_from_row(row)
    if not kernel_name:
        return None

    if kernel_name.startswith(_KERNEL_NAME_PREFIX):
        for needle, category in _ATEN_NATIVE_CATEGORY_RULES:
            if needle in kernel_name:
                return category
        return None

    return _category_from_classified_kernel(kernel_name)


def get_perf_model_category(perf_model_class: type, bwd: bool = False) -> Optional[str]:
    """Return the category declared by a perf model class."""
    attr_name = "bwd_category" if bwd else "category"
    category = getattr(perf_model_class, attr_name, None)
    if bwd:
        return category
    if category is None:
        raise ValueError(
            f"perf_model_class {perf_model_class} must define a category attribute"
        )
    return category


def sheet_category_from_final_category(category: str) -> str:
    """Return the legacy sheet family for a final categorization label."""
    for suffix in ("_fwd", "_bwd"):
        if category.endswith(suffix):
            return category[: -len(suffix)]
    return category


def get_perf_model_sheet_category(perf_model_class: type) -> str:
    """Return the legacy sheet category for a perf model class."""
    sheet_category = getattr(perf_model_class, "sheet_category", None)
    if sheet_category is not None:
        return sheet_category
    return sheet_category_from_final_category(get_perf_model_category(perf_model_class))


def build_op_category_registry(
    op_to_perf_model_class_map: Mapping[str, type],
    category_only_ops: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    """Construct the flat ``op_name -> final category`` registry."""
    registry: Dict[str, str] = {}
    for op_name, perf_model_class in op_to_perf_model_class_map.items():
        registry[op_name] = get_perf_model_category(perf_model_class)

    if category_only_ops:
        registry.update(category_only_ops)

    return registry


def build_sheet_category_to_op_names(
    op_to_perf_model_class_map: Mapping[str, type],
) -> Dict[str, List[str]]:
    """Build the legacy ``category -> op names`` view used for report sheets."""
    sheet_category_to_op_names = {}  # type: Dict[str, List[str]]
    for op_name, perf_model_class in op_to_perf_model_class_map.items():
        sheet_category = get_perf_model_sheet_category(perf_model_class)
        sheet_category_to_op_names.setdefault(sheet_category, []).append(op_name)
    return sheet_category_to_op_names


def register_perf_model_categories(
    perf_model_extension: Mapping[str, type],
    registry: MutableMapping[str, str],
) -> None:
    """Register categories for extension-provided perf models."""
    for op_name, perf_model_class in perf_model_extension.items():
        registry[op_name] = get_perf_model_category(perf_model_class)


def register_op_categories(
    op_category_extension: Mapping[str, str],
    registry: MutableMapping[str, str],
) -> None:
    """Register explicit category-only op labels."""
    registry.update(op_category_extension)


def _categorize_torch_op_from_registry(
    row,
    registry: Mapping[str, str],
    patterns: Iterable[Tuple[Pattern, str]] = OP_CATEGORY_PATTERNS,
) -> str:
    """Return the category for ``row`` using explicit registry data."""
    name = row["name"]

    category = registry.get(name)
    if category is not None:
        return category

    for pattern, category in patterns:
        if pattern.match(name):
            return category

    fallback = _kernel_name_fallback(row)
    if fallback is not None:
        return fallback

    return "other"


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
    # Mamba SSD (fused conv1d + selective scan, issue #552)
    "MambaSplitConv1dScanCombinedFn": perf_model.mamba_ssd_fwd,
    # Primus FP8 ops (hipBLASLt GEMM + quantize, issue #626)
    "primus_turbo_cpp_extension::hipblaslt_gemm_fp8": perf_model.hipblaslt_gemm_fp8,
    "primus_turbo::hipblaslt_gemm_fp8": perf_model.hipblaslt_gemm_fp8,
    "primus_turbo_cpp_extension::quantize_fp8_tensorwise": perf_model.primus_turbo_quantize_fp8,
    "primus_turbo::quantize_fp8_tensorwise": perf_model.primus_turbo_quantize_fp8,
    # Primus MXFP4 ops (hipBLASLt FP4 GEMM + dual rowwise/colwise quantize, issue #637)
    "primus_turbo_cpp_extension::hipblaslt_gemm_fp4": perf_model.hipblaslt_gemm_fp4,
    "primus_turbo::hipblaslt_gemm_fp4": perf_model.hipblaslt_gemm_fp4,
    "primus_turbo_cpp_extension::quantize_mxfp4_dual": perf_model.primus_turbo_quantize_mxfp4_dual,
    "primus_turbo::quantize_mxfp4_dual": perf_model.primus_turbo_quantize_mxfp4_dual,
    "primus::quantize_mxfp4_dual": perf_model.primus_turbo_quantize_mxfp4_dual,
    # AITER MXFP4 native FP4 ASM GEMM (issue #644).
    # Both the public entry (aiter::gemm_a4w4) and the inner asm dispatch
    # (aiter::_gemm_a4w4_asm) are registered. The asm op is the leaf that
    # actually launches the kernel and gets the GPU time attribution; the
    # public entry is registered too so it is still classified as GEMM in
    # any report that iterates op names directly.
    "aiter::gemm_a4w4": perf_model.aiter_gemm_a4w4,
    "aiter::_gemm_a4w4_asm": perf_model.aiter_gemm_a4w4,
}

# Add pseudo-op extension mappings
op_to_perf_model_class_map.update(get_pseudo_op_mappings())

# Add pseudo-op category-only mappings (no perf model, just a category label).
CATEGORY_ONLY_OP_MAPPING.update(get_pseudo_op_category_only_mappings())

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
    # Primus fused LN+modulate for MM-DiT (issue #627)
    "primus::fused_ln_modulate": perf_model.FusedLnModulate,
    "primus::fused_ln_modulate_backward": perf_model.FusedLnModulateBackward,
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

# ---------------------------------------------------------------------------
# Pattern-based matchers for perf models with generated kernel names.
# Each matcher is a callable: name -> perf_model_class | None.
# ---------------------------------------------------------------------------
_perf_model_matchers: list = []


def register_perf_model_matcher(matcher):
    _perf_model_matchers.append(matcher)


def _match_triton_compiled(name):
    if name.startswith(("triton_poi_", "triton_red_", "triton_per_")):
        from TraceLens.PerfModel.triton_compiled_perf_model import (
            TritonCompiledPerfModel,
        )

        return TritonCompiledPerfModel
    return None


register_perf_model_matcher(_match_triton_compiled)

OP_CATEGORY_REGISTRY = build_op_category_registry(
    op_to_perf_model_class_map,
    category_only_ops=CATEGORY_ONLY_OP_MAPPING,
)


def _event_has_input_dims(event):
    from TraceLens.Trace2Tree.trace_sglang_capture_link import (
        is_usable_capture_input_dims,
    )

    return is_usable_capture_input_dims(event.get("args", {}).get("Input Dims"))


def _perf_model_class_from_kernel_name(kernel_name: str) -> Optional[type]:
    if kernel_name.startswith(_KERNEL_NAME_PREFIX):
        for needle, model_class in _ATEN_NATIVE_PERF_MODEL_RULES:
            if needle in kernel_name:
                return model_class
        return None
    return _perf_model_class_from_classified_kernel(kernel_name)


def resolve_perf_model_class(event_or_name, op_map=None):
    """
    Resolve a perf model class for a trace event or op name string.

    Named ops use op_to_perf_model_class_map. Graph-replay synthetic ops and other
    rows with kernel_details / parsed kernel names use kernel-based resolution.
    Kernel-resolved models require Input Dims on the event (e.g. from a parent
    cpu_op or a merged capture trace) until kernel-only param extraction exists.
    """
    if isinstance(event_or_name, str):
        event = {"name": event_or_name}
    else:
        event = event_or_name

    op_map = op_to_perf_model_class_map if op_map is None else op_map
    name = event.get("name", "")
    model_class = op_map.get(name)
    if model_class is not None:
        return model_class

    for matcher in _perf_model_matchers:
        model_class = matcher(name)
        if model_class is not None:
            return model_class

    kernel_name = _kernel_name_from_row(event)
    if kernel_name is None:
        return None

    model_class = _perf_model_class_from_kernel_name(kernel_name)
    if model_class is None:
        return None

    if SYNTHETIC_OP_MARKER in event.get("name", "") and not _event_has_input_dims(
        event
    ):
        return None
    return model_class


def categorize_torch_op(row):
    """
    Categorizes a row based on the 'name' and 'kernel_details' fields.

    Args:
        row (dict): A dictionary with at minimum a 'name' key. May also contain
            a 'kernel_details' key — a list of dicts each having a 'name' field
            that holds the underlying GPU kernel name.

    Returns:
        str: One of 'GEMM', 'CONV_fwd', 'CONV_bwd', 'NORM_fwd', 'NORM_bwd',
             'SDPA_fwd', 'SDPA_bwd', 'GroupedGEMM_fwd', 'GroupedGEMM_bwd',
             'MoE_fused', 'MoE_unfused',
             'SSM_fwd', 'SSM_bwd', 'MoE_comm_fwd', 'MoE_comm_bwd',
             'RoPE_fwd', 'RoPE_bwd', 'CrossEntropy_fwd', 'CrossEntropy_bwd',
             'elementwise', 'triton', 'reduce', 'multi_tensor_apply',
             'record_param_comms', or 'other'.

        Note: Backward variants and auxiliary ops (TokenPermuteMaskMap, etc.)
        are categorization-only (timing without GFLOPS or TB/s).
    """
    return _categorize_torch_op_from_registry(row, OP_CATEGORY_REGISTRY)


def _register_graph_replay_kernel_perf_models() -> None:
    """Register perf models for SGLang graph-replay GPU kernels (lazy import)."""
    from TraceLens.PerfModel.extensions import moe_perf_model_extensions
    from TraceLens.PerfModel.extensions import perf_model_extensions
    from TraceLens.PerfModel.extensions import rmsnorm_perf_model_extensions

    register_classified_kernel_perf_model_rules(
        [
            (
                "name_substr",
                "kernel_moe_gemm",
                moe_perf_model_extensions.ck_kernel_moe_gemm,
            ),
            (
                "name_substr",
                "dynamic_per_group_scaled_quant",
                perf_model_extensions.dynamic_per_group_scaled_quant_kernel,
            ),
            (
                "name_substr",
                "rmsnorm_sumsq_kernel_serial",
                rmsnorm_perf_model_extensions.aiter_rmsnorm_sumsq_serial,
            ),
            (
                "name_substr",
                "rmsnorm_apply_kernel_serial",
                rmsnorm_perf_model_extensions.aiter_rmsnorm_apply_serial,
            ),
            (
                "name_substr",
                "add_rmsnorm_quant_kernel",
                rmsnorm_perf_model_extensions.aiter_add_rmsnorm_quant_graph,
            ),
        ]
    )


_register_graph_replay_kernel_perf_models()
