###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Registry-based categorization of CPU torch ops.

This module provides a flat ``op_name -> category`` registry plus a small set
of fallback patterns. It is the v2 replacement for the if/elif chain in
``categorize_torch_op``. PR A introduces ``categorize_torch_op_v2`` alongside
the legacy implementation; ``test_categorize_torch_op_parity`` asserts both
produce the same output for every reachable name.

Subsequent PRs (referenced from the meta-issue):

  - PR B: delete the if/elif chain, move the remaining hardcoded names into
    ``LEGACY_CATEGORIZE_EXTRAS``, and add a ``get_op_categories`` extension
    hook so category-only entries don't require a perf model.
  - PR C: add ``report_uncategorized_ops`` for drift detection.
"""

from __future__ import annotations

import re
from typing import Iterable, Mapping, Optional

# ---------------------------------------------------------------------------
# Legacy SDPA backward names
#
# Replicated from the hardcoded list inside ``categorize_torch_op`` so v2
# produces identical output for backward-attention ops that were previously
# detected by name rather than by the ``_backward`` suffix.
#
# Note: in the legacy chain this list was only consulted for ops already in
# ``dict_cat2names["SDPA"]``. Names below that aren't perf-modeled (e.g.
# ``FlashAttnFuncBackward``) currently resolve to ``"other"`` in v1 and v2
# preserves that. PR B can decide whether to lift that restriction.
# ---------------------------------------------------------------------------
_LEGACY_SDPA_BWD_NAMES = frozenset(
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


# ---------------------------------------------------------------------------
# Explicit ``op_name -> category`` overrides
#
# Each entry corresponds to a hardcoded list inside the legacy
# ``categorize_torch_op`` chain. Once PR B deletes that chain these become
# the only path for category-only ops (no perf model required).
# ---------------------------------------------------------------------------
LEGACY_CATEGORIZE_EXTRAS: dict[str, str] = {
    # CONV ops not present in op_to_perf_model_class_map
    "aten::miopen_convolution": "CONV_fwd",
    "aten::cudnn_convolution": "CONV_fwd",
    # SSM
    "MambaSplitConv1dScanCombinedFnBackward": "SSM_bwd",
    "DaoAILab::_causal_conv1d_bwd_cpp": "SSM_bwd",
    # MoE_comm forward extras (not perf-modeled)
    "TokenPermuteMaskMap": "MoE_comm_fwd",
    "_OperationFuserAutogradFunction": "MoE_comm_fwd",
    # MoE_comm backward
    "MoEDispatchBackward": "MoE_comm_bwd",
    "MoECombineBackward": "MoE_comm_bwd",
    "TokenPermuteMaskMapBackward": "MoE_comm_bwd",
    "_OperationFuserAutogradFunctionBackward": "MoE_comm_bwd",
    # RoPE / CrossEntropy backward
    "FusedRoPEFuncBackward": "RoPE_bwd",
    "CrossEntropyFunctionBackward": "CrossEntropy_bwd",
    # MoE auxiliary ops (sorting / topk)
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
    # InferenceAttention extras (KV-cache writes)
    "_C_cache_ops::reshape_and_cache_flash": "InferenceAttention",
    "_C_cache_ops::concat_and_cache_mla": "InferenceAttention",
}


# ---------------------------------------------------------------------------
# Patterns evaluated only when no exact registry match is found, in declared
# order. ``re.match`` is implicitly anchored at the start of the string.
# ---------------------------------------------------------------------------
OP_CATEGORY_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"triton"), "triton"),
    (re.compile(r"record_param_comms"), "record_param_comms"),
]


# ---------------------------------------------------------------------------
# Kernel-name fallback rules
#
# Applied only to GPU kernel names starting with ``void at::native``. The
# first substring match wins. Mirrors the kernel-name probe at the bottom
# of the legacy ``categorize_torch_op``.
# ---------------------------------------------------------------------------
_KERNEL_NAME_PREFIX = "void at::native"
_KERNEL_NAME_FALLBACK_RULES: tuple[tuple[str, str], ...] = (
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


def _resolve_base_category(op_name: str, base_category: str) -> str:
    """Map ``(op_name, base-class category)`` to the final category emitted by v1.

    The legacy chain applied a handful of fwd/bwd splits and category renames
    that aren't expressible by walking the perf-model class hierarchy alone.
    This helper centralises those rules.
    """
    if base_category == "SDPA":
        if op_name.endswith("_backward") or op_name in _LEGACY_SDPA_BWD_NAMES:
            return "SDPA_bwd"
        return "SDPA_fwd"
    if base_category == "Normalization":
        if op_name.endswith("_backward") or op_name.endswith("Backward"):
            return "NORM_bwd"
        return "NORM_fwd"
    if base_category == "CONV":
        if op_name.endswith("_backward") or op_name.endswith("Backward"):
            return "CONV_bwd"
        return "CONV_fwd"
    if base_category == "SSM":
        return "SSM_fwd"
    if base_category == "MoE_comm":
        return "MoE_comm_fwd"
    if base_category == "RoPE":
        return "RoPE_fwd"
    if base_category == "CrossEntropy":
        return "CrossEntropy_fwd"
    if base_category in ("BinaryElementwise", "UnaryElementwise"):
        return "elementwise"
    if base_category == "Reduce":
        return "reduce"
    return base_category


def build_op_category_registry(
    op_to_perf_model_class_map: Mapping[str, type],
    dict_base_class2category: Mapping[type, str],
    extras: Optional[Mapping[str, str]] = None,
) -> dict[str, str]:
    """Construct the flat ``op_name -> category`` registry.

    The registry is built in two passes:

    1. Walk every entry in ``op_to_perf_model_class_map``. Each perf-model
       class contributes exactly one ``(op_name, category)`` pair, derived
       from its single base class via ``dict_base_class2category`` and then
       passed through :func:`_resolve_base_category` to apply fwd/bwd splits.

    2. Apply ``extras`` as overrides. Use this for category-only ops
       (no perf model) and for ops where the legacy chain disagrees with
       the auto-derived category.

    Parameters
    ----------
    op_to_perf_model_class_map
        Mapping from op name to perf-model class.
    dict_base_class2category
        Mapping from perf-model base class to category label.
    extras
        Optional ``op_name -> category`` overrides.

    Returns
    -------
    dict[str, str]
        ``op_name -> final_category`` lookup table.
    """
    registry: dict[str, str] = {}

    for op_name, perf_model_class in op_to_perf_model_class_map.items():
        base_classes = perf_model_class.__bases__
        if len(base_classes) != 1:
            continue
        base_category = dict_base_class2category.get(base_classes[0])
        if base_category is None:
            continue
        registry[op_name] = _resolve_base_category(op_name, base_category)

    if extras:
        registry.update(extras)

    return registry


def categorize_torch_op_v2(
    row,
    registry: Mapping[str, str],
    patterns: Iterable[tuple[re.Pattern, str]] = OP_CATEGORY_PATTERNS,
) -> str:
    """Return the category for ``row`` using the flat registry plus patterns.

    Resolution order:

    1. Exact name match in ``registry``.
    2. First pattern in ``patterns`` whose regex matches ``row["name"]``.
    3. Kernel-name fallback for ``void at::native`` GPU kernels.
    4. ``"other"``.
    """
    name = row["name"]

    cat = registry.get(name)
    if cat is not None:
        return cat

    for pattern, category in patterns:
        if pattern.match(name):
            return category

    fallback = _kernel_name_fallback(row)
    if fallback is not None:
        return fallback

    return "other"
