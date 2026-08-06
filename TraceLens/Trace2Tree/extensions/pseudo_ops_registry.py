###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Registry that discovers and applies pseudo-op extensions to a trace tree."""

import logging
import os
import re
import sys
import time
from functools import partial
from typing import Optional

from tqdm import tqdm

from .pseudo_ops_utils import (
    _MLA_DECODE_FWD_NAME_RE,
    _MLA_FP8_PREFILL_NAME_RE,
    _any_kernel_event_name_contains,
    normalize_sglang_profiler_op_names,
)

logger = logging.getLogger(__name__)


def apply_pseudo_op_extensions(
    tree, verbose: bool = False, show_progress: bool = False
):
    """
    Apply all available pseudo-op extensions to trace tree.
    Extensions are automatically detected and applied.

    When ``show_progress`` is True, emit ``tqdm`` milestone lines and a bar over
    the apply phase (stderr), mirroring perf-report progress.

    MoE fused vs unfused branch lines and per-extension timings are also written
    to stderr when ``show_progress`` or ``verbose`` is True, or when the
    environment variable ``TRACELENS_PSEUDO_OPS_LOG`` is set to a non-empty value
    other than ``0`` / ``false`` (case-insensitive).
    """

    pseudo_ops_emit = (
        show_progress
        or verbose
        or (
            (os.environ.get("TRACELENS_PSEUDO_OPS_LOG") or "").strip().lower()
            not in ("", "0", "false", "no", "off")
        )
    )

    if show_progress:
        tqdm.write(
            "perf: pseudo_ops: normalize_sglang_profiler_op_names …",
            file=sys.stderr,
        )
    normalize_sglang_profiler_op_names(tree)

    # Auto-detect and add all known pseudo-op extensions
    extensions = []

    moe_branch_msg: Optional[str] = None
    if "vllm::moe_forward" in tree.name2event_uids:

        # MoE: AITER Fused Implementation
        if "vllm::rocm_aiter_fused_moe" in tree.name2event_uids:
            from .moe_aiter_pseudo_ops import create_pseudo_ops_moe_fused_aiter

            extensions.append(
                (
                    "MoE_Fused",
                    partial(
                        create_pseudo_ops_moe_fused_aiter,
                        show_progress=pseudo_ops_emit,
                    ),
                )
            )
            moe_branch_msg = (
                "vllm::moe_forward + vllm::rocm_aiter_fused_moe → extension MoE_Fused "
                "(AITER fused pseudo ops)"
            )
            if verbose:
                logger.info("Auto-detected fused MoE operations")

        # MoE: Triton Fused Implementation
        # TO DO: Update kernel detection approach (Look for gpt_oss_triton_kernels_moe.py)
        else:
            has_matmul_ogs = _any_kernel_event_name_contains(tree, "matmul_ogs")

            if has_matmul_ogs:
                from .moe_unfused_triton_pseudo_ops import (
                    create_pseudo_ops_moe_unfused_triton,
                )

                extensions.append(
                    (
                        "MoE_Unfused_Triton",
                        partial(
                            create_pseudo_ops_moe_unfused_triton,
                            show_progress=pseudo_ops_emit,
                        ),
                    )
                )
                moe_branch_msg = (
                    "vllm::moe_forward + kernel names containing matmul_ogs → "
                    "extension MoE_Unfused_Triton (Triton unfused pseudo ops)"
                )
                if verbose:
                    logger.info(
                        "Auto-detected GPT_OSS unfused MoE operations with Triton kernels"
                    )
            else:
                moe_branch_msg = (
                    "vllm::moe_forward present but no vllm::rocm_aiter_fused_moe and "
                    "no matmul_ogs in kernel names → no MoE pseudo-op extension from this branch"
                )

    # MoE: GPTQ/AWQ quantized unfused implementation (vllm::outplace_fused_experts)
    if "vllm::outplace_fused_experts" in tree.name2event_uids:
        has_gptq_awq = _any_kernel_event_name_contains(
            tree, "fused_moe_kernel_gptq_awq", lower=False
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
            _MLA_DECODE_FWD_NAME_RE.search(name)
            for name in tree.name2event_uids
            if "mla.py" in name and "mla_decode_fwd" in name
        )
        if has_mla_python_func:
            from .mla_decode_pseudo_ops import create_pseudo_ops_mla_decode

            extensions.append(("MLA_Decode", create_pseudo_ops_mla_decode))
            if verbose:
                logger.info("Auto-detected MLA decode operations")

    # MLA Prefill: AITER fp8 implementation
    if "aiter::mla_prefill_ps_asm_fwd" in tree.name2event_uids:
        has_prefill_python_func = any(
            _MLA_FP8_PREFILL_NAME_RE.search(name)
            for name in tree.name2event_uids
            if "mla_fp8_prefill_attn" in name
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

    if pseudo_ops_emit:
        if moe_branch_msg is not None:
            tqdm.write(
                f"perf: pseudo_ops: MoE branch: {moe_branch_msg}",
                file=sys.stderr,
            )
            logger.info("pseudo_ops MoE branch: %s", moe_branch_msg)
        else:
            tqdm.write(
                "perf: pseudo_ops: MoE branch: vllm::moe_forward not in trace "
                "(fused/unfused MoE pseudo ops from this detector are inactive)",
                file=sys.stderr,
            )
            logger.info(
                "pseudo_ops MoE branch: vllm::moe_forward not in trace "
                "(fused/unfused MoE pseudo ops from this detector are inactive)"
            )
        ext_names = [x[0] for x in extensions]
        tqdm.write(
            f"perf: pseudo_ops: extension order ({len(extensions)}): "
            f"{', '.join(ext_names) if ext_names else '(none)'}",
            file=sys.stderr,
        )
        logger.info("pseudo_ops extension order: %s", ext_names)

    if show_progress:
        tqdm.write(
            f"perf: pseudo_ops: detected {len(extensions)} extension(s) to run …",
            file=sys.stderr,
        )
    _ext_iter = extensions
    if show_progress:
        _ext_iter = tqdm(
            extensions,
            desc="perf: pseudo-op extensions",
            unit="ext",
            file=sys.stderr,
            mininterval=0.3,
        )
    for ext_info in _ext_iter:
        ext_name, ext_func = ext_info

        if verbose:
            logger.info(f"Applying pseudo-op extension: {ext_name}")

        try:
            t0 = time.perf_counter()
            ext_func(tree)
            dt = time.perf_counter() - t0
            if pseudo_ops_emit:
                tqdm.write(
                    f"perf: pseudo_ops: extension {ext_name} finished in {dt:.2f}s",
                    file=sys.stderr,
                )
                logger.info("pseudo_ops extension %s finished in %.2fs", ext_name, dt)
        except Exception as e:
            logger.warning(f"Failed to apply pseudo-op extension {ext_name}: {e}")
