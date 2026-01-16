###############################################################################
# Copyright (c) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Pseudo op extension for unfused MoE operations (vllm::moe_forward).

Creates separate pseudo ops for each matmul_ogs GEMM kernel in unfused MoE,
typically:
- pseudo_op::moe_unfused_gemm_0 (up_proj + gate_proj with SwiGLU)
- pseudo_op::moe_unfused_gemm_1 (down_proj)
"""

import logging
from .pseudo_ops_utils import inject_pseudo_op

logger = logging.getLogger(__name__)


def create_unfused_moe_pseudo_ops(trace_tree):
    """
    Create pseudo ops for vllm::moe_forward unfused operations.
    Creates separate pseudo ops for each matmul_ogs GEMM kernel.
    
    Args:
        trace_tree: TraceToTree instance
    """
    
    if "vllm::moe_forward" not in trace_tree.name2event_uids:
        return
    
    moe_op_events = [trace_tree.get_UID2event(uid) 
                     for uid in trace_tree.name2event_uids["vllm::moe_forward"]]
    
    logger.info(f"Processing {len(moe_op_events)} unfused MoE operations")
    
    for moe_op_event in moe_op_events:
        _create_unfused_moe_pseudo_ops(trace_tree, moe_op_event)


def is_matmul_ogs_kernel(kernel_event: dict) -> bool:
    """
    Check if kernel is matmul_ogs (unfused MoE GEMM).
    
    Args:
        kernel_event: Event dictionary
        
    Returns:
        True if kernel is a matmul_ogs kernel
    """
    if kernel_event.get("cat") != "kernel":
        return False
    
    kernel_name = kernel_event["name"]
    return "matmul_ogs" in kernel_name.lower()


def _get_gemm_type_from_kernel_name(kernel_name: str) -> tuple:
    """
    Determine the GEMM type and gating status from the kernel name.
    
    Args:
        kernel_name: Name of the matmul_ogs kernel
        
    Returns:
        Tuple of (gemm_type, gated) where:
        - gemm_type: 'up' or 'down'
        - gated: True if kernel has swiglu activation, False otherwise
    """
    if "_swiglu" in kernel_name.lower():
        return ("up", True)
    return ("down", False)


def _extract_topk_from_moe(trace_tree, moe_op_event: dict):
    """
    Extract topk (number of active experts per token) from TopK child operation.
    
    Args:
        trace_tree: TraceToTree instance
        moe_op_event: vllm::moe_forward event
        
    Returns:
        topk value as int, or 4 as default if not found
        
    Note:
        Returns default value of 4 if TopK operation not found (e.g., when python_func layer is inserted)
    """
    def search_for_topk(uid, depth=0, max_depth=3):
        """Recursively search for TopK operation through Python function wrappers."""
        if depth > max_depth:
            return None
            
        event = trace_tree.get_UID2event(uid)
        
        # Check if this is a TopK operation
        if 'TopK' in event.get('name', ''):
            concrete = event['args'].get('Concrete Inputs', [])
            # TopK's k parameter is the second concrete input
            if len(concrete) > 1 and concrete[1]:
                try:
                    return int(concrete[1])
                except (ValueError, TypeError):
                    pass
        
        # Recursively search children (for Python function wrappers)
        for child_uid in event.get('children', []):
            result = search_for_topk(child_uid, depth + 1, max_depth)
            if result is not None:
                return result
        
        return None
    
    # Look for TopK child operation (with recursive search for Python wrappers)
    for child_uid in moe_op_event.get('children', []):
        topk_value = search_for_topk(child_uid)
        if topk_value is not None:
            return topk_value
    
    # TopK not found - return default value of 4 (common MoE configuration)
    logger.warning(
        f"TopK operation not found in children of vllm::moe_forward (UID={moe_op_event['UID']}). "
        f"Using default topk=4. Children names: "
        f"{[trace_tree.get_UID2event(uid).get('name') for uid in moe_op_event.get('children', [])]}"
    )
    return 4  # Default topk value


def _create_unfused_moe_pseudo_ops(trace_tree, moe_op_event: dict):
    """
    Create pseudo ops for each matmul_ogs kernel in unfused MoE.
    
    Args:
        trace_tree: TraceToTree instance
        moe_op_event: vllm::moe_forward event
    """
    
    if moe_op_event.get("name") != "vllm::moe_forward":
        logger.warning(f"Expected vllm::moe_forward, found {moe_op_event['name']}")
        return
    
    gpu_event_ids = moe_op_event.get("gpu_events", [])
    if not gpu_event_ids:
        logger.warning(f"No GPU events for MoE UID {moe_op_event['UID']}")
        return
    
    gpu_events = [trace_tree.get_UID2event(uid) for uid in gpu_event_ids]
    matmul_kernels = [e for e in gpu_events if is_matmul_ogs_kernel(e)]
    
    if len(matmul_kernels) == 0:
        logger.warning(f"No matmul_ogs kernels found for UID {moe_op_event['UID']}")
        return
    
    # Extract topk from tree (raises ValueError if not found)
    topk = _extract_topk_from_moe(trace_tree, moe_op_event)
    
    # Sort kernels to ensure consistent ordering
    # Kernels with _swiglu come first (up+gate), then down_proj
    matmul_kernels_sorted = sorted(
        matmul_kernels,
        key=lambda k: (0 if "_swiglu" in k["name"].lower() else 1, k["UID"])
    )
    
    seq_num = moe_op_event["args"].get("Sequence number", moe_op_event["UID"])
    
    # Create pseudo op for each GEMM
    for idx, kernel in enumerate(matmul_kernels_sorted):
        gemm_type, gated = _get_gemm_type_from_kernel_name(kernel["name"])
        # Use descriptive names: unfused_triton_moe_up or unfused_triton_moe_down
        pseudo_op_name = f"pseudo_op::unfused_triton_moe_{gemm_type}"
        
        # Prepare custom args for this pseudo-op
        extra_args = {
            "MoE GEMM type": gemm_type,
            "MoE GEMM index": idx,
            "MoE GEMM gated": gated,
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
