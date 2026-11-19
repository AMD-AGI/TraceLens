###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""A vLLM-style decoder that imports its parallel linears from a sibling."""

from .layers import QKVParallelLinear, RowParallelLinear


class LlamaDecoder:
    def __init__(self):
        self.qkv_proj = QKVParallelLinear(8, 8, 8)
        self.o_proj = RowParallelLinear(8, 8)
