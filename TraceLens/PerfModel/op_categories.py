###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Registry-based categorization of CPU torch ops.

Perf model classes declare their own output categories via ``category`` and,
when linked-forward backward metrics are intentionally supported,
``bwd_category``. Category-only ops that do not have a perf model live in
``CATEGORY_ONLY_OP_MAPPING``.
"""

import re
from typing import Dict, Iterable, List, Mapping, MutableMapping
from typing import Optional, Pattern, Tuple


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


_KERNEL_NAME_PREFIX = "void at::native"
_KERNEL_NAME_FALLBACK_RULES: Tuple[Tuple[str, str], ...] = (
    ("elementwise", "elementwise"),
    ("reduce", "reduce"),
    ("multi_tensor_apply", "multi_tensor_apply"),
)


def _kernel_name_fallback(row) -> Optional[str]:
    kernel_details = row.get("kernel_details")
    if not kernel_details:
        return None
    kernel_name = kernel_details[0].get("name", "")
    if not kernel_name.startswith(_KERNEL_NAME_PREFIX):
        return None
    for needle, category in _KERNEL_NAME_FALLBACK_RULES:
        if needle in kernel_name:
            return category
    return None


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
