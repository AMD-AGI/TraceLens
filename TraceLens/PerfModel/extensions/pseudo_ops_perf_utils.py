###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Utils. for perf. model pseudo-op extensions.
"""

from . import perf_model_extensions


def get_pseudo_op_mappings():
    """
    Return a dictionary mapping pseudo-op names to their performance model classes.
    
    Returns:
        dict: Mapping of pseudo-op names to performance model classes
    """

    pseudo_op_mappings = {
        # MoE pseudo ops - Fused
        "pseudo_op::moe_aiter_fused_1stage": perf_model_extensions.moe_aiter_fused_1stage,
        # MoE pseudo ops - Unfused Triton (2-stage: up and down)
        "pseudo_op::moe_triton_unfused_up": perf_model_extensions.moe_triton_unfused_up,
        "pseudo_op::moe_triton_unfused_down": perf_model_extensions.moe_triton_unfused_down,
    }

    return pseudo_op_mappings


def get_pseudo_op_categories():
    """
    Return a dictionary mapping pseudo-op base classes to their performance categories.
    
    Returns:
        dict: Mapping of base classes to category names
    """
    
    pseudo_op_categories = {
        perf_model_extensions.FusedMoE: "MoE_fused",
        perf_model_extensions.UnfusedMoE_Up: "MoE_unfused",
        perf_model_extensions.UnfusedMoE_Down: "MoE_unfused",
    }
    
    return pseudo_op_categories
