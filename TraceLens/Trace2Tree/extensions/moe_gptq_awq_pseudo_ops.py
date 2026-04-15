###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import logging
from .pseudo_ops_utils import inject_pseudo_op

logger = logging.getLogger(__name__)


def create_pseudo_ops_moe_gptq_awq(trace_tree):
    """
    Create pseudo ops for vllm::outplace_fused_experts GPTQ/AWQ operations.
    Creates separate pseudo ops for each fused_moe_kernel_gptq_awq GEMM kernel
    (up/gate projection and down projection).

    Args:
        trace_tree: TraceToTree instance
    """

    if "vllm::outplace_fused_experts" not in trace_tree.name2event_uids:
        return

    moe_op_events = [
        trace_tree.get_UID2event(uid)
        for uid in trace_tree.name2event_uids["vllm::outplace_fused_experts"]
    ]

    logger.info(f"Processing {len(moe_op_events)} GPTQ/AWQ MoE operations")

    for moe_op_event in moe_op_events:
        _create_pseudo_op_moe_gptq_awq(trace_tree, moe_op_event)


def is_fused_moe_gptq_awq_kernel(kernel_event: dict) -> bool:
    """
    Check if kernel is a fused_moe_kernel_gptq_awq GEMM kernel.

    Args:
        kernel_event: Event dictionary

    Returns:
        True if kernel is a GPTQ/AWQ fused MoE kernel
    """
    return (
        kernel_event.get("cat") == "kernel"
        and "fused_moe_kernel_gptq_awq" in kernel_event["name"]
    )


def _extract_topk_from_outplace(moe_op_event: dict) -> int:
    """
    Extract topk from the topk_ids input shape of outplace_fused_experts.

    Input Dims layout:
        [0] hidden_states [T, K]
        [1] w1 (gate+up weight, packed) [E, N_packed, K_packed]
        [2] w2 (down weight, packed)    [E, K, N_packed]
        [3] topk_weights                [T, topk]
        [4] topk_ids                    [T, topk]
        ...

    Args:
        moe_op_event: vllm::outplace_fused_experts event

    Returns:
        topk value as int, or 8 as default if not found
    """
    try:
        topk_ids_shape = moe_op_event["args"]["Input Dims"][4]
        if len(topk_ids_shape) >= 2:
            return int(topk_ids_shape[1])
    except (KeyError, IndexError, TypeError, ValueError):
        pass

    logger.warning(
        f"Could not extract topk from outplace_fused_experts (UID={moe_op_event['UID']}), using default 8"
    )
    return 8


def _create_pseudo_op_moe_gptq_awq(trace_tree, moe_op_event: dict):
    """
    Create pseudo ops for each fused_moe_kernel_gptq_awq kernel under
    one vllm::outplace_fused_experts invocation.

    Args:
        trace_tree: TraceToTree instance
        moe_op_event: vllm::outplace_fused_experts event
    """

    if moe_op_event.get("name") != "vllm::outplace_fused_experts":
        logger.warning(
            f"Expected vllm::outplace_fused_experts, found {moe_op_event['name']}"
        )
        return

    gpu_event_ids = moe_op_event.get("gpu_events", [])
    if not gpu_event_ids:
        logger.warning(f"No GPU events for MoE UID {moe_op_event['UID']}")
        return

    gpu_events = [trace_tree.get_UID2event(uid) for uid in gpu_event_ids]
    gptq_awq_kernels = [e for e in gpu_events if is_fused_moe_gptq_awq_kernel(e)]

    if len(gptq_awq_kernels) == 0:
        logger.warning(
            f"No fused_moe_kernel_gptq_awq kernels found for UID {moe_op_event['UID']}"
        )
        return

    if len(gptq_awq_kernels) != 2:
        logger.warning(
            f"Expected 2 fused_moe_kernel_gptq_awq kernels (up + down), "
            f"found {len(gptq_awq_kernels)} for UID {moe_op_event['UID']}"
        )

    topk = _extract_topk_from_outplace(moe_op_event)
    seq_num = moe_op_event["args"].get("Sequence number", moe_op_event["UID"])

    # Sort by start time: first kernel = up (gate+up), second = down
    gptq_awq_kernels_sorted = sorted(gptq_awq_kernels, key=lambda k: k["ts"])

    for idx, kernel in enumerate(gptq_awq_kernels_sorted):
        gemm_type = "up" if idx == 0 else "down"
        pseudo_op_name = f"pseudo_op::moe_gptq_awq_{gemm_type}"

        extra_args = {
            "MoE GEMM type": gemm_type,
            "MoE GEMM index": idx,
            "MoE GEMM gated": idx == 0,  # up/gate kernel is always gated (SwiGLU)
            "MoE topk": topk,
        }

        inject_pseudo_op(
            trace_tree,
            kernel,
            pseudo_op_name,
            seq_num,
            dims=moe_op_event["args"].get("Input Dims"),
            types=moe_op_event["args"].get("Input type"),
            strides=moe_op_event["args"].get("Input Strides"),
            concrete_inputs=moe_op_event["args"].get("Concrete Inputs"),
            extra_args=extra_args,
        )
