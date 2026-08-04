###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Registry that discovers and applies pseudo-op extensions to a trace tree."""

import logging
import re

from .pseudo_ops_utils import normalize_sglang_profiler_op_names

logger = logging.getLogger(__name__)


def apply_pseudo_op_extensions(tree, verbose: bool = False):
    """
    Apply all available pseudo-op extensions to trace tree.
    Extensions are automatically detected and applied.
    """

    normalize_sglang_profiler_op_names(tree)

    # Auto-detect and add all known pseudo-op extensions
    extensions = []

    if "vllm::moe_forward" in tree.name2event_uids:

        # MoE: AITER Fused Implementation
        if "vllm::rocm_aiter_fused_moe" in tree.name2event_uids:
            from .moe_aiter_pseudo_ops import create_pseudo_ops_moe_fused_aiter

            extensions.append(("MoE_Fused", create_pseudo_ops_moe_fused_aiter))
            if verbose:
                logger.info("Auto-detected fused MoE operations")

        # MoE: Triton Fused Implementation
        # TO DO: Update kernel detection approach (Look for gpt_oss_triton_kernels_moe.py)
        else:
            # Check if any kernel events contain matmul_ogs: Triton MoE kernel
            has_matmul_ogs = any(
                "matmul_ogs" in event.get("name", "").lower()
                for event in tree.events
                if event.get("cat") == "kernel"
            )

            if has_matmul_ogs:
                from .moe_unfused_triton_pseudo_ops import (
                    create_pseudo_ops_moe_unfused_triton,
                )

                extensions.append(
                    ("MoE_Unfused_Triton", create_pseudo_ops_moe_unfused_triton)
                )
                if verbose:
                    logger.info(
                        "Auto-detected GPT_OSS unfused MoE operations with Triton kernels"
                    )

    # MoE: GPTQ/AWQ quantized unfused implementation (vllm::outplace_fused_experts)
    if "vllm::outplace_fused_experts" in tree.name2event_uids:
        has_gptq_awq = any(
            "fused_moe_kernel_gptq_awq" in event.get("name", "")
            for event in tree.events
            if event.get("cat") == "kernel"
        )
        if has_gptq_awq:
            from .moe_gptq_awq_pseudo_ops import create_pseudo_ops_moe_gptq_awq

            extensions.append(("MoE_GPTQ_AWQ", create_pseudo_ops_moe_gptq_awq))
            if verbose:
                logger.info(
                    "Auto-detected GPTQ/AWQ MoE operations (outplace_fused_experts)"
                )

    # MoE: flydsl 2-stage implementation (gated on aiter::fused_moe_ parent op)
    if "aiter::fused_moe_" in tree.name2event_uids:
        from .moe_flydsl_pseudo_ops import create_pseudo_ops_moe_flydsl

        extensions.append(("MoE_Flydsl", create_pseudo_ops_moe_flydsl))
        if verbose:
            logger.info("Auto-detected flydsl MoE operations under aiter::fused_moe_")

    # MLA Decode: AITER implementation
    if "aiter::mla_decode_stage1_asm_fwd" in tree.name2event_uids:
        has_mla_python_func = any(
            re.search(r"aiter/mla.py\(\d+\): mla_decode_fwd", name)
            for name in tree.name2event_uids
        )
        if has_mla_python_func:
            from .mla_decode_pseudo_ops import create_pseudo_ops_mla_decode

            extensions.append(("MLA_Decode", create_pseudo_ops_mla_decode))
            if verbose:
                logger.info("Auto-detected MLA decode operations")

    # MLA Prefill: AITER fp8 implementation
    if "aiter::mla_prefill_ps_asm_fwd" in tree.name2event_uids:
        has_prefill_python_func = any(
            re.search(r":\s*mla_fp8_prefill_attn(\b|$)", name)
            for name in tree.name2event_uids
        )
        if has_prefill_python_func:
            from .mla_prefill_pseudo_ops import create_pseudo_ops_mla_prefill

            extensions.append(("MLA_Prefill", create_pseudo_ops_mla_prefill))
            if verbose:
                logger.info("Auto-detected MLA prefill operations")
    # DeepSeek-V4 sparse paged decode (Flash + Pro): isolate the split/reduce
    # decode kernels under a mode-specific pseudo op.
    has_v4_sparse_decode = any(
        re.search(r"paged_decode\.py\(\d+\):\s*sparse_attn_v4_paged_decode", name)
        for name in tree.name2event_uids
    )
    if has_v4_sparse_decode:
        from .v4_paged_decode_pseudo_ops import create_pseudo_ops_v4_paged_decode

        extensions.append(("V4_Paged_Decode", create_pseudo_ops_v4_paged_decode))
        if verbose:
            logger.info("Auto-detected DeepSeek-V4 sparse paged-decode operations")

    if "_rocm_C::paged_attention" in tree.name2event_uids:
        from .paged_attn_perf_meta import mark_rocm_paged_attn_kvcache_dtype

        extensions.append(
            ("RocmPagedAttn_KVCacheDtype", mark_rocm_paged_attn_kvcache_dtype)
        )
        if verbose:
            logger.info(
                "Auto-detected _rocm_C::paged_attention — will propagate "
                "perf_meta.KCache_dtype/VCache_dtype to cpu_op parents"
            )
    if "aiter::paged_attention_v1" in tree.name2event_uids:
        from .paged_attn_perf_meta import mark_aiter_paged_attn_kvcache_dtype

        extensions.append(
            ("AiterPagedAttn_KVCacheDtype", mark_aiter_paged_attn_kvcache_dtype)
        )
        if verbose:
            logger.info(
                "Auto-detected aiter::paged_attention_v1 — will propagate "
                "perf_meta.k_cache_dtype/v_cache_dtype to cpu_op parents"
            )
    # Apply extensions onto tree
    for ext_info in extensions:
        # ext_info tuple of (extension_name, extension_function)
        ext_name, ext_func = ext_info

        if verbose:
            logger.info(f"Applying pseudo-op extension: {ext_name}")

        try:
            ext_func(tree)
        except Exception as e:
            logger.warning(f"Failed to apply pseudo-op extension {ext_name}: {e}")
