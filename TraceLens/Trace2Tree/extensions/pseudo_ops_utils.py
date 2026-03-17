###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import logging
from typing import Optional, List, Callable

logger = logging.getLogger(__name__)


def set_bookkeeping_attr(tree, event: dict):
    """Add bookkeeping attributes for a new pseudo event in the tree."""

    UID = len(tree.events)
    event["UID"] = UID
    tree.events.append(event)
    tree.events_by_uid[UID] = event
    
    seq_num = event["args"].get("Sequence number")
    if seq_num is not None:
        if seq_num not in tree.seq_num2event_uids_map:
            tree.seq_num2event_uids_map[seq_num] = []
        tree.seq_num2event_uids_map[seq_num].append(UID)


def inject_pseudo_op(
    tree,
    kernel_evt,
    name,
    seq_num,
    dims=None,
    types=None,
    strides=None,
    concrete_inputs=None,
    extra_args=None,
):
    """
    Create pseudo op between parent CPU op and kernel.
    Creates: Parent CPU Op → Pseudo Op → Launcher → Kernel
    
    Args:
        tree: TraceToTree instance
        kernel_evt: Kernel event to inject pseudo-op for
        name: Name of the pseudo-op
        seq_num: Sequence number
        dims: Input dimensions (uses parent if None)
        types: Input types (uses parent if None)
        strides: Input strides (uses parent if None)
        concrete_inputs: Concrete inputs (uses parent if None)
        extra_args: Additional custom args to add to pseudo-op (dict)
    """

    launcher_evt = tree.get_parent_event(kernel_evt)
    orig_cpu_evt = tree.get_parent_event(launcher_evt)

    pseudo_evt = {
        "ph": "X",
        "name": name,
        "cat": "cpu_op",
        "pid": orig_cpu_evt["pid"],
        "tid": orig_cpu_evt["tid"],
        "args": {
            "Input Dims": orig_cpu_evt["args"].get("Input Dims") if dims is None else dims,
            "Input type": orig_cpu_evt["args"].get("Input type") if types is None else types,
            "Input Strides": orig_cpu_evt["args"].get("Input Strides") if strides is None else strides,
            "Concrete Inputs": orig_cpu_evt["args"].get("Concrete Inputs") if concrete_inputs is None else concrete_inputs,
            "Sequence number": seq_num,
            "External id": kernel_evt["args"]["correlation"],
            "Pseudo op": True,
        },
        "children": [launcher_evt["UID"]],
        "gpu_events": [kernel_evt["UID"]],
    }
    
    # Add any extra custom args
    if extra_args:
        pseudo_evt["args"].update(extra_args)
    
    set_bookkeeping_attr(tree, pseudo_evt)
    
    children = orig_cpu_evt["children"]
    children.remove(launcher_evt["UID"])
    children.append(pseudo_evt["UID"])


def apply_pseudo_op_extensions(
    tree, 
    verbose: bool = False
):
    """
    Apply all available pseudo-op extensions to trace tree.
    Extensions are automatically detected and applied.
    """
    
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
                from .moe_unfused_triton_pseudo_ops import create_pseudo_ops_moe_unfused_triton
                extensions.append(("MoE_Unfused_Triton", create_pseudo_ops_moe_unfused_triton))
                if verbose:
                    logger.info("Auto-detected GPT_OSS unfused MoE operations with Triton kernels")
    
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