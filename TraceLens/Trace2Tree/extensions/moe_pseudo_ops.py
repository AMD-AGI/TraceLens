###############################################################################
# Copyright (c) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import logging
from .pseudo_ops_utils import inject_pseudo_op

logger = logging.getLogger(__name__)


def create_moe_pseudo_ops(trace_tree):
    """
    Create pseudo ops for vllm::rocm_aiter_fused_moe operations. Isolates MoE compute kernel from indexing/quantization kernels.
    """

    if "vllm::rocm_aiter_fused_moe" not in trace_tree.name2event_uids:
        return
    
    moe_op_events = [trace_tree.get_UID2event(uid) for uid in trace_tree.name2event_uids["vllm::rocm_aiter_fused_moe"]]
    
    for moe_op_event in moe_op_events:
        _create_pseudo_moe_op(trace_tree, moe_op_event)


def is_moe_kernel(kernel_event: dict) -> bool:
    """Check if kernel is MoE compute (not sorting/quantization)."""

    if kernel_event.get("cat") != "kernel":
        return False
    
    kernel_name = kernel_event["name"]
    is_moe_kernel_match = (
        "aiter::" in kernel_name and
        "fmoe" in kernel_name and
        "MoeSorting" not in kernel_name and
        "quant" not in kernel_name.lower()
    )
    
    return is_moe_kernel_match


def _create_pseudo_moe_op(trace_tree, moe_op_event: dict):
    """Create single pseudo op for one MoE operation."""

    if moe_op_event.get("name") != "vllm::rocm_aiter_fused_moe":
        logger.warning(f"Expected vllm::rocm_aiter_fused_moe, found {moe_op_event['name']}")
        return

    gpu_event_ids = moe_op_event.get("gpu_events", [])
    if not gpu_event_ids:
        logger.warning(f"No GPU events for MoE UID {moe_op_event['UID']}")
        return

    gpu_events = [trace_tree.get_UID2event(uid) for uid in gpu_event_ids]
    moe_kernels = [e for e in gpu_events if is_moe_kernel(e)]

    if len(moe_kernels) != 1:
        logger.warning(f"Expected 1 MoE kernel, found {len(moe_kernels)} for UID {moe_op_event['UID']}")
        return

    moe_kernel = moe_kernels[0]
    seq_num = moe_op_event["args"].get("Sequence number", moe_op_event["UID"])

    inject_pseudo_op(
        trace_tree,
        moe_kernel,
        "pseudo_op::fused_moe_1stage",
        seq_num,
        dims=moe_op_event["args"].get("Input Dims"),
        types=moe_op_event["args"].get("Input type"),
        strides=moe_op_event["args"].get("Input Strides"),
        concrete_inputs=moe_op_event["args"].get("Concrete Inputs"),
    )

