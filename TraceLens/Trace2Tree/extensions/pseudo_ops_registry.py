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
    has_vllm_sparse_decode = any(
        "_sparse_attn_decode_gfx950_partial_kernel" in name
        for name in tree.name2event_uids
    )
    if has_vllm_sparse_decode:
        from .vllm_sparse_decode_pseudo_ops import (
            create_pseudo_ops_vllm_sparse_decode,
        )

        extensions.append(("VLLM_Sparse_Decode", create_pseudo_ops_vllm_sparse_decode))
        if verbose:
            logger.info("Auto-detected vLLM ROCm sparse-decode kernels")

    if "_fwd_kernel_stage2_asm" in tree.name2event_uids:
        from .mla_v4_stage2_shapes import copy_mla_v4_stage2_shapes

        extensions.append(("MLA_V4_Stage2_Shapes", copy_mla_v4_stage2_shapes))
        if verbose:
            logger.info("Auto-detected MLA v4 stage2 kernels — will copy parent shapes")

    # SGLang Flash sparse decode (_pa_decode_sparse_*) — bridge to V4 models.
    has_sglang_sparse = any(
        "_pa_decode_sparse_" in name or "aiter_sparse_decode_reduce" in name
        for name in tree.name2event_uids
    )
    if has_sglang_sparse:
        from .sglang_sparse_decode_pseudo_ops import (
            create_pseudo_ops_sglang_sparse_decode,
        )

        extensions.append(
            ("SGLang_Flash_Sparse_Decode", create_pseudo_ops_sglang_sparse_decode)
        )
        if verbose:
            logger.info("Auto-detected SGLang Flash sparse-decode operations")

    # Flash indexer (FP4 MQA logits / quant / topk parent / FlashMLA prep)
    has_flash_indexer = any(
        ("pa_mqa_logits_fp4" in name)
        or ("quantize_fp4_indexer" in name)
        or ("fused_k_norm_rope_flashmla" in name)
        or name == "aten::topk"
        for name in tree.name2event_uids
    )
    if has_flash_indexer:
        from .sglang_flash_indexer_pseudo_ops import (
            create_pseudo_ops_sglang_flash_indexer,
        )

        extensions.append(
            ("SGLang_Flash_Indexer", create_pseudo_ops_sglang_flash_indexer)
        )
        if verbose:
            logger.info("Auto-detected SGLang Flash indexer operations")

    # Flash mHC / MXFP8 / router / rmsnorm_fake_quant
    has_flash_misc = any(
        ("hc_boundary" in name)
        or ("hc_mix_reduce_sinkhorn" in name)
        or ("rmsnorm_fake_quant_fp8" in name)
        or ("rocm_router_gemv_split_k" in name)
        or ("rocm_router_gate" in name)
        or ("mxfp8_native_blockscaled_linear" in name)
        for name in tree.name2event_uids
    )
    if has_flash_misc:
        from .sglang_flash_misc_pseudo_ops import create_pseudo_ops_sglang_flash_misc

        extensions.append(("SGLang_Flash_Misc", create_pseudo_ops_sglang_flash_misc))
        if verbose:
            logger.info("Auto-detected SGLang Flash mHC/MXFP8/router ops")

    # Flash MoE gemm2_a4w4 / moe_reduction Synthetic Op children under fused_moe_
    has_flash_moe_children = any(
        "gemm2_a4w4" in name or "moe_reduction_kernel" in name
        for name in tree.name2event_uids
    )
    if has_flash_moe_children and "aiter::fused_moe_" in tree.name2event_uids:
        from .sglang_flash_moe_children_pseudo_ops import (
            create_pseudo_ops_sglang_flash_moe_children,
        )

        extensions.append(
            ("SGLang_Flash_MoE_Children", create_pseudo_ops_sglang_flash_moe_children)
        )
        if verbose:
            logger.info("Auto-detected Flash MoE gemm2/reduction children")
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
