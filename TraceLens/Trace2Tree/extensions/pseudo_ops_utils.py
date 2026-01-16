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
    extensions: Optional[List[Callable]] = None,
    verbose: bool = False
):
    """
    Apply pseudo-op extensions to trace tree.
    Extensions auto-detected if not provided.
    """
    
    # Default: Add all known pseudo-op extensions
    if extensions is None:
        extensions = []
        
        # Fused MoE (DSR1 trace)
        if "vllm::rocm_aiter_fused_moe" in tree.name2event_uids:
            from .moe_pseudo_ops import create_moe_pseudo_ops
            extensions.append(("MoE_Fused", create_moe_pseudo_ops))
            if verbose:
                logger.info("Auto-detected fused MoE operations")
        
        # Unfused MoE (GPT_OSS trace with Triton kernels)
        if "vllm::moe_forward" in tree.name2event_uids:

            # Check if any kernel events contain matmul_ogs: Triton MoE kernel
            has_matmul_ogs = any(
                "matmul_ogs" in event.get("name", "").lower()
                for event in tree.events
                if event.get("cat") == "kernel"
            )

            if has_matmul_ogs:
                from .moe_unfused_triton_pseudo_ops import create_unfused_moe_pseudo_ops
                extensions.append(("MoE_Unfused_Triton", create_unfused_moe_pseudo_ops))
                if verbose:
                    logger.info("Auto-detected GPT_OSS unfused MoE operations with Triton kernels")
        
        if not extensions and verbose:
            logger.info("No pseudo-op extensions auto-detected")
    else:
        extensions = [("Custom", ext) if callable(ext) else ext for ext in extensions]
    
    for ext_info in extensions:
        if isinstance(ext_info, tuple):
            ext_name, ext_func = ext_info
        else:
            ext_name, ext_func = "Unknown", ext_info
        
        if verbose:
            logger.info(f"Applying pseudo-op extension: {ext_name}")
        
        try:
            ext_func(tree)
        except Exception as e:
            logger.warning(f"Failed to apply pseudo-op extension {ext_name}: {e}")