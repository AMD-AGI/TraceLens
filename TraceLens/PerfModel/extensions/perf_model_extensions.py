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
from TraceLens.PerfModel.perf_model import GEMM, BinaryElementwise
from math import prod

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

class vllm_rocm_unquantized_gemm(GEMM):
    """
    Performance model for vllm::rocm_unquantized_gemm.
    Dispatches to aiter triton gemm (gemm_a16w16) or skinny GEMM (wvSplitK/LLMM1)
    depending on shape heuristics; falls back to torch.nn.functional.linear.

    Computes: output[M, N] = x[M, K] @ weight[N, K].T + bias[N]

    Expected Input Dims format (from vllm::rocm_unquantized_gemm):
    [[M, K], [N, K], [N]]  (x, weight, bias)

    Expected Input type format:
    [dtype_x, dtype_weight, dtype_bias]
    """
    @staticmethod
    def get_param_details(event):
        return {
            "M": event["args"]["Input Dims"][0][0],
            "N": event["args"]["Input Dims"][1][0],
            "K": event["args"]["Input Dims"][0][1],
            "bias": True,
            "dtype_A_B": (event["args"]["Input type"][0], event["args"]["Input type"][1], event["args"]["Input type"][2]),
        }
    def bytes(self):
        dtype_A_B = self.param_details["dtype_A_B"]
        self.bpe_mat1 = name2bpe(dtype_A_B[0])
        self.bpe_mat2 = name2bpe(dtype_A_B[1])
        self.bpe_output = name2bpe(dtype_A_B[2])
        self.bpe_bias = name2bpe(dtype_A_B[2]) 

        return super().bytes(
            bpe_mat1=self.bpe_mat1,
            bpe_mat2=self.bpe_mat2,
            bpe_bias=self.bpe_bias,
            bpe_output=self.bpe_output,
        )
class per_group_quant(BinaryElementwise):
    """
    Performance model for aiter::dynamic_per_token_scaled_quant.
    Performs dynamic per-token quantization: each row of the input is
    quantized independently with its own scale factor.

    AITER signature: dynamic_per_token_scaled_quant(out, input, scales,
        scale_ub=None, shuffle_scale=False, num_rows=None, num_rows_factor=1)

    Expected Input Dims format (from aiter::dynamic_per_token_scaled_quant):
    [[out_shape], [input_shape], [scales_shape], ...]

    Expected Input type format:
    [dtype_out, dtype_input, dtype_scales, ...]
    """

    def __init__(self, event, arch=None, python_path=None):
        self.event = event
        self.arch = arch
        self.param_details = self.get_param_details(event)
        self.nelems_in1 = prod(self.param_details["shape_in1"])
        self.nelems_in2 = prod(self.param_details["shape_in2"])
        self.nelems_out = prod(self.param_details["shape_out"])
        self.dtype_in1_in2_out = self.param_details["dtype_in1_in2_out"]
        self.stride_input1 = self.param_details["stride_input1"]
        self.stride_input2 = self.param_details["stride_input2"]
        self.stride_output = self.param_details["stride_output"]

        dtype_in1, dtype_in2, dtype_out = self.dtype_in1_in2_out
        self.bpe_in1 = name2bpe(dtype_in1)
        self.bpe_in2 = name2bpe(dtype_in2)
        if dtype_out is not None:
            self.bpe_out = name2bpe(dtype_out)
        else:
            self.bpe_out = None
    @staticmethod
    def get_param_details(event):
        args_input_dims = event["args"]["Input Dims"]
        shape_in1 = tuple(args_input_dims[1])
        shape_in2 = tuple(args_input_dims[2]) 
        shape_out = tuple(args_input_dims[0])
        dtype_in1 = event["args"]["Input type"][1]
        dtype_in2 = event["args"]["Input type"][2]
        dtype_out = event["args"]["Input type"][0]
        stride_output = tuple(event["args"]["Input Strides"][0])
        stride_input1 = tuple(event["args"]["Input Strides"][1])
        stride_input2 = tuple(event["args"]["Input Strides"][2])
        return {
            "shape_in1": shape_in1,
            "shape_in2": shape_in2,
            "shape_out": shape_out,
            "dtype_in1_in2_out": (dtype_in1, dtype_in2, dtype_out),
            "stride_input1": stride_input1,
            "stride_input2": stride_input2,
            "stride_output": stride_output,
        }
    def bytes(self):
        bytes= prod(self.param_details["shape_out"]) * name2bpe(self.param_details["dtype_in1_in2_out"][2])
        bytes+= prod(self.param_details["shape_in1"]) * name2bpe(self.param_details["dtype_in1_in2_out"][0])
        bytes+= prod(self.param_details["shape_in2"]) * name2bpe(self.param_details["dtype_in1_in2_out"][1])
        return bytes

    def flops(self):
        return self.nelems_out * 2.59375 # Based on the rocprof counter values
    
    def get_compute_precision(self):
        """Return the compute precision for this operation."""
        # Use first input dtype as the compute precision
        dtype = self.dtype_in1_in2_out[1] if self.dtype_in1_in2_out else None
        return torch_dtype_map(dtype) if dtype else None