###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""An SGLang-style block that imports FusedMoE from a sibling module."""

from .moe import FusedMoE


class QwenMoeBlock:
    def __init__(self):
        self.experts = FusedMoE(num_experts=64)
