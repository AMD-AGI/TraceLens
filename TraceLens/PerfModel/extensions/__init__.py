###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Extension for pseudo-op operations.
"""

from .perf_model_extensions import (
    FusedMoE,
    moe_aiter_fused_1stage,
    UnfusedMoE_Up,
    UnfusedMoE_Down,
    moe_triton_unfused_up,
    moe_triton_unfused_down,
)
from .pseudo_ops_perf_utils import get_pseudo_op_mappings, get_pseudo_op_categories

__all__ = [
    # Base classes
    'FusedMoE',
    'UnfusedMoE_Up',
    'UnfusedMoE_Down',
    # Concrete classes
    'moe_aiter_fused_1stage',
    'moe_triton_unfused_up',
    'moe_triton_unfused_down',
    # Utility functions
    'get_pseudo_op_mappings',
    'get_pseudo_op_categories',
]
