###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Performance models for pseudo-op extensions.
"""

from TraceLens.PerfModel.utils import torch_dtype_map, name2bpe
import re
from TraceLens.PerfModel.perf_model import GEMM

class gemm_a8w8_blockscale(GEMM):
    @staticmethod
    def get_param_details(event):
        return {
            "B": 1,
            "M": event["args"]["Input Dims"][0][0],
            "N": event["args"]["Input Dims"][1][0],
            "K": event["args"]["Input Dims"][0][1],
            "bias": False,
            "dtype_A_B": (event["args"]["Input type"][0], event["args"]["Input type"][1],"c10::bfloat16"),
        }
    def bytes(self):
        dtype_A_B = self.param_details["dtype_A_B"]
        self.bpe_mat1 = name2bpe(dtype_A_B[0])
        self.bpe_mat2 = name2bpe(dtype_A_B[1])
        self.bpe_output = name2bpe(dtype_A_B[2])
        self.bpe_bias = name2bpe(dtype_A_B[2]) # dummy since bias is not used

        return super().bytes(
            bpe_mat1=self.bpe_mat1,
            bpe_mat2=self.bpe_mat2,
            bpe_bias=self.bpe_bias,
            bpe_output=self.bpe_output,
        )

        