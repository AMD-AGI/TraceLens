###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Registry-based categorization of CPU torch ops.

The source of truth is a flat ``op_name -> final category`` registry used by
``categorize_torch_op``. ``dict_cat2names`` remains as a compatibility view for
legacy per-category sheets such as ``GEMM``, ``CONV_fwd`` / ``CONV_bwd``, and
``GroupedGEMM_fwd`` / ``GroupedGEMM_bwd``.
"""

import re
from collections import defaultdict
from typing import DefaultDict, Dict, Iterable, List, Mapping, MutableMapping
from typing import Optional, Pattern, Tuple


SDPA_BWD_OPS = frozenset(
    {
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
    }
)


OP_CATEGORY_OVERRIDES: Dict[str, str] = {
    # CONV ops not present in op_to_perf_model_class_map.
    "aten::miopen_convolution": "CONV_fwd",
    "aten::cudnn_convolution": "CONV_fwd",
    # SDPA backward ops that do not all have perf models in core TraceLens.
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


def _append_unique(target: List[str], names: Iterable[str]) -> None:
    existing = set(target)
    for name in names:
        if name not in existing:
            target.append(name)
            existing.add(name)


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


def is_backward_op(op_name: str) -> bool:
    return (
        op_name.endswith("_backward")
        or op_name.endswith("Backward")
        or op_name.endswith("_bwd")
        or op_name in SDPA_BWD_OPS
    )


def resolve_base_category(op_name: str, base_category: str) -> str:
    """Return the final category for an op in a base/sheet category."""
    if base_category == "SDPA":
        return "SDPA_bwd" if is_backward_op(op_name) else "SDPA_fwd"
    if base_category == "Normalization":
        return "NORM_bwd" if is_backward_op(op_name) else "NORM_fwd"
    if base_category == "CONV":
        return "CONV_bwd" if is_backward_op(op_name) else "CONV_fwd"
    if base_category == "GroupedGEMM":
        return "GroupedGEMM_bwd" if is_backward_op(op_name) else "GroupedGEMM_fwd"
    if base_category == "SSM":
        return "SSM_bwd" if is_backward_op(op_name) else "SSM_fwd"
    if base_category == "MoE_comm":
        return "MoE_comm_bwd" if is_backward_op(op_name) else "MoE_comm_fwd"
    if base_category == "RoPE":
        return "RoPE_bwd" if is_backward_op(op_name) else "RoPE_fwd"
    if base_category == "CrossEntropy":
        return "CrossEntropy_bwd" if is_backward_op(op_name) else "CrossEntropy_fwd"
    if base_category in ("BinaryElementwise", "UnaryElementwise"):
        return "elementwise"
    if base_category == "Reduce":
        return "reduce"
    return base_category


def sheet_category_from_final_category(category: str) -> str:
    """Return the legacy sheet family for a final categorization label."""
    category_to_sheet = {
        "CONV_fwd": "CONV",
        "CONV_bwd": "CONV",
        "SDPA_fwd": "SDPA",
        "SDPA_bwd": "SDPA",
        "NORM_fwd": "Normalization",
        "NORM_bwd": "Normalization",
        "GroupedGEMM_fwd": "GroupedGEMM",
        "GroupedGEMM_bwd": "GroupedGEMM",
        "SSM_fwd": "SSM",
        "SSM_bwd": "SSM",
        "MoE_comm_fwd": "MoE_comm",
        "MoE_comm_bwd": "MoE_comm",
        "RoPE_fwd": "RoPE",
        "RoPE_bwd": "RoPE",
        "CrossEntropy_fwd": "CrossEntropy",
        "CrossEntropy_bwd": "CrossEntropy",
        "elementwise": "UnaryElementwise",
        "reduce": "Reduce",
    }
    return category_to_sheet.get(category, category)


def category_from_sheet_view(
    op_name: str, dict_cat2names: Mapping[str, List[str]]
) -> Optional[str]:
    """Compatibility fallback for callers that still mutate ``dict_cat2names``."""
    for sheet_category, names in dict_cat2names.items():
        if op_name in names:
            return resolve_base_category(op_name, sheet_category)
    return None


def build_op_category_registry(
    op_to_perf_model_class_map: Mapping[str, type],
    dict_base_class2category: Mapping[type, str],
    overrides: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    """Construct the flat ``op_name -> final category`` registry."""
    registry: Dict[str, str] = {}
    for op_name, perf_model_class in op_to_perf_model_class_map.items():
        base_classes = perf_model_class.__bases__
        if len(base_classes) != 1:
            raise ValueError(
                f"op_name: {op_name}, perf_model_class: {perf_model_class}, "
                f"base_classes: {base_classes}"
            )
        base_category = dict_base_class2category.get(base_classes[0])
        if base_category is None:
            raise ValueError(
                f"op_name: {op_name}, perf_model_class: {perf_model_class}, "
                f"base_class: {base_classes[0]}"
            )
        registry[op_name] = resolve_base_category(op_name, base_category)

    if overrides:
        registry.update(overrides)

    return registry


def build_dict_cat2names(
    op_to_perf_model_class_map: Mapping[str, type],
    dict_base_class2category: Mapping[type, str],
) -> DefaultDict[str, List[str]]:
    """Build the legacy ``category -> op names`` view used for report sheets."""
    dict_cat2names = defaultdict(list)  # type: DefaultDict[str, List[str]]
    for op_name, perf_model_class in op_to_perf_model_class_map.items():
        base_classes = perf_model_class.__bases__
        if len(base_classes) != 1:
            raise ValueError(
                f"op_name: {op_name}, perf_model_class: {perf_model_class}, "
                f"base_classes: {base_classes}"
            )
        base_category = dict_base_class2category.get(base_classes[0])
        if base_category is None:
            raise ValueError(
                f"op_name: {op_name}, perf_model_class: {perf_model_class}, "
                f"base_class: {base_classes[0]}"
            )
        dict_cat2names[base_category].append(op_name)
    return dict_cat2names


def register_perf_model_categories(
    perf_model_extension: Mapping[str, type],
    dict_base_class2category: Mapping[type, str],
    registry: MutableMapping[str, str],
    dict_cat2names: MutableMapping[str, List[str]],
) -> None:
    """Register categories for extension-provided perf models."""
    for op_name, perf_model_class in perf_model_extension.items():
        base_classes = perf_model_class.__bases__
        if len(base_classes) != 1:
            raise ValueError(
                f"op_name: {op_name}, perf_model_class: {perf_model_class}, "
                f"base_classes: {base_classes}"
            )
        sheet_category = dict_base_class2category.get(base_classes[0])
        if sheet_category is None:
            raise ValueError(
                f"op_name: {op_name}, perf_model_class: {perf_model_class}, "
                f"base_class: {base_classes[0]}"
            )
        registry[op_name] = resolve_base_category(op_name, sheet_category)
        if sheet_category not in dict_cat2names:
            dict_cat2names[sheet_category] = []
        _append_unique(dict_cat2names[sheet_category], [op_name])


def register_op_categories(
    op_category_extension: Mapping[str, str],
    registry: MutableMapping[str, str],
    dict_cat2names: Optional[MutableMapping[str, List[str]]] = None,
) -> None:
    """Register explicit category-only op labels."""
    registry.update(op_category_extension)
    if dict_cat2names is None:
        return
    for op_name, category in op_category_extension.items():
        sheet_category = sheet_category_from_final_category(category)
        if sheet_category not in dict_cat2names:
            dict_cat2names[sheet_category] = []
        _append_unique(dict_cat2names[sheet_category], [op_name])


def register_dict_cat2names_extension(
    dict_cat2names_extension: Mapping[str, List[str]],
    registry: MutableMapping[str, str],
    dict_cat2names: MutableMapping[str, List[str]],
) -> None:
    """Support the older ``dict_cat2names_extension`` extension contract."""
    for sheet_category, names in dict_cat2names_extension.items():
        if not isinstance(names, list):
            raise ValueError(f"Expected names to be a list, got {type(names)}")
        if sheet_category not in dict_cat2names:
            dict_cat2names[sheet_category] = []
        _append_unique(dict_cat2names[sheet_category], names)
        for op_name in names:
            registry[op_name] = resolve_base_category(op_name, sheet_category)


def categorize_torch_op_from_registry(
    row,
    registry: Mapping[str, str],
    dict_cat2names: Optional[Mapping[str, List[str]]] = None,
    patterns: Iterable[Tuple[Pattern, str]] = OP_CATEGORY_PATTERNS,
) -> str:
    """Return the category for ``row`` using explicit registry data."""
    name = row["name"]

    category = registry.get(name)
    if category is not None:
        return category

    if dict_cat2names is not None:
        category = category_from_sheet_view(name, dict_cat2names)
        if category is not None:
            return category

    for pattern, category in patterns:
        if pattern.match(name):
            return category

    fallback = _kernel_name_fallback(row)
    if fallback is not None:
        return fallback

    return "other"
