###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Performance models for custom collective operation extensions.
"""

from math import prod
from TraceLens.PerfModel.utils import name2bpe
from TraceLens.PerfModel.perf_model import RMSNorm


class CustomCollective:
    pass


class aiter_fused_allreduce_rmsnorm(CustomCollective):
    """
    Performance model for aiter::fused_allreduce_rmsnorm.

    Fused AllReduce + residual-add + RMSNorm in a single kernel.
    Signature: ops.fused_allreduce_rmsnorm(ptr, reg, inp, res_inp, res_out, out, w, eps, ...)
        inp     — shape [M, N], dtype BFloat16 (local shard, to be all-reduced)
        res_inp — shape [M, N], dtype BFloat16 (residual to add after all-reduce)
        res_out — shape [M, N], dtype BFloat16 (updated residual = allreduce(inp) + res_inp)
        out     — shape [M, N], dtype BFloat16 (RMSNorm output)
        w       — shape [N],    dtype BFloat16 (RMSNorm affine weight)
        eps     — float scalar

    Expected Input Dims from trace (C-side args):
        [ptr, reg, inp_shape, res_inp_shape, res_out_shape, out_shape, w_shape, eps, ...]
        e.g. [(1,), (), (4, 7168), (4, 7168), (4, 7168), (4, 7168), (7168,), (), (), ()]

    FLOPs: residual-add + RMSNorm (allreduce is bandwidth-bound, not counted).
    Bytes: HBM traffic per GPU (read inp+res_inp+weight, write res_out+out).
    """

    def __init__(self, event, arch=None, python_path=None):
        self.event = event
        self.arch = arch
        self.param_details = self.get_param_details(event)
        self.num_elems = prod(self.param_details["op_shape"])
        self.num_channels = self.param_details["num_channels"]
        self.bpe_in = name2bpe(self.param_details["dtype_in"])

    @staticmethod
    def get_param_details(event):
        op_shape = tuple(event["args"]["Input Dims"][2])  # inp: [M, N]
        dtype_in = event["args"]["Input type"][2]  # BFloat16
        stride_input = tuple(event["args"]["Input Strides"][2])
        num_channels = event["args"]["Input Dims"][6][0]  # weight.shape[0] = N
        return {
            "op_shape": op_shape,
            "dtype_in": dtype_in,
            "stride_input": stride_input,
            "num_channels": num_channels,
        }

    def flops(self):
        # Residual add: num_elems (allreduced_inp + res_inp -> res_out)
        add_flops = self.num_elems
        # RMSNorm (affine=True, training=False) — same formula as RMSNorm.flops()
        non_norm_elems = self.num_elems // self.num_channels
        rms_flops = non_norm_elems * (2 * self.num_channels + 2)  # compute rsqrt
        rms_flops += self.num_elems * 2  # apply weight
        return add_flops + rms_flops

    def bytes(self):
        # HBM traffic per GPU (inter-GPU allreduce bandwidth is separate)
        # Reads:  inp, res_inp, weight
        # Writes: res_out, out
        bytes_read = 2 * self.num_elems * self.bpe_in + self.num_channels * self.bpe_in
        bytes_write = 2 * self.num_elems * self.bpe_in
        return bytes_read + bytes_write


class custom_ar_all_reduce(CustomCollective):
    """
    Performance model for _C_custom_ar::all_reduce.

    Pure all-reduce (no fused norm). Communication-bound — no meaningful compute FLOPs.
    Signature: all_reduce(fa, inp, out, reg_buffer, reg_buffer_sz_bytes)
        fa               — int handle (Scalar)
        inp              — shape [M, N], dtype BFloat16
        out              — shape [M, N], dtype BFloat16 (allreduced result)
        reg_buffer       — int pointer (Scalar)
        reg_buffer_sz    — int (Scalar)

    Expected Input Dims from trace:
        [fa_scalar, inp_shape, out_shape, reg_buffer_scalar, reg_buffer_sz_scalar]
        e.g. [(), (4, 7168), (4, 7168), (), ()]

    FLOPs: None — purely inter-GPU communication, no on-chip compute.
    Bytes: HBM traffic per GPU (read inp + write out).
    """

    def __init__(self, event, arch=None, python_path=None):
        self.event = event
        self.arch = arch
        self.param_details = self.get_param_details(event)
        self.num_elems = prod(self.param_details["op_shape"])
        self.bpe_in = name2bpe(self.param_details["dtype_in"])

    @staticmethod
    def get_param_details(event):
        op_shape = tuple(event["args"]["Input Dims"][1])  # inp: [M, N]
        dtype_in = event["args"]["Input type"][1]  # BFloat16
        stride_input = tuple(event["args"]["Input Strides"][1])
        return {
            "op_shape": op_shape,
            "dtype_in": dtype_in,
            "stride_input": stride_input,
        }

    def flops(self):
        # Pure collective — no meaningful on-chip compute FLOPs
        return 0

    def bytes(self):
        # HBM traffic per GPU: read inp + write out
        return 2 * self.num_elems * self.bpe_in


class aiter_reduce_scatter(CustomCollective):
    """
    Performance model for aiter::reduce_scatter.

    Pure reduce-scatter collective — each GPU contributes a shard, result is scattered.
    Signature: reduce_scatter(_fa, inp, out, reg_buffer=None)
        _fa        — int handle (Scalar)
        inp        — shape [M, N], dtype BFloat16 (full local tensor before scatter)
        out        — shape [M // N_gpus, N], dtype BFloat16 (scattered shard)
        reg_buffer — optional int pointer (Scalar)

    Expected Input Dims from trace:
        [(), inp_shape, out_shape, ...]
        e.g. [(), (8, 7168), (4, 7168)]

    FLOPs: None — purely inter-GPU communication.
    Bytes: HBM traffic per GPU (read inp + write out).
    """

    def __init__(self, event, arch=None, python_path=None):
        self.event = event
        self.arch = arch
        self.param_details = self.get_param_details(event)
        self.num_elems = prod(self.param_details["op_shape"])
        self.bpe_in = name2bpe(self.param_details["dtype_in"])

    @staticmethod
    def get_param_details(event):
        op_shape = tuple(event["args"]["Input Dims"][1])  # inp: [M, N]
        out_shape = tuple(event["args"]["Input Dims"][2])  # out: [M // N_gpus, N]
        dtype_in = event["args"]["Input type"][1]  # BFloat16
        stride_input = tuple(event["args"]["Input Strides"][1])
        return {
            "op_shape": op_shape,
            "out_shape": out_shape,
            "dtype_in": dtype_in,
            "stride_input": stride_input,
        }

    def flops(self):
        return 0

    def bytes(self):
        # read inp (full) + write out (scattered shard, smaller)
        num_elems_out = prod(self.param_details["out_shape"])
        return (self.num_elems + num_elems_out) * self.bpe_in


class aiter_all_gather_reg(CustomCollective):
    """
    Performance model for aiter::all_gather_reg.

    Pure all-gather collective — each GPU contributes a shard, all get the full tensor.
    Signature: all_gather_reg(_fa, inp, out)
        _fa — int handle (Scalar)
        inp — shape [M // N_gpus, N], dtype BFloat16 (local shard)
        out — shape [M, N],           dtype BFloat16 (gathered full tensor)

    Expected Input Dims from trace:
        [(), inp_shape, out_shape]
        e.g. [(), (4, 7168), (8, 7168)]

    FLOPs: None — purely inter-GPU communication.
    Bytes: HBM traffic per GPU (read inp shard + write full out).
    """

    def __init__(self, event, arch=None, python_path=None):
        self.event = event
        self.arch = arch
        self.param_details = self.get_param_details(event)
        self.num_elems = prod(self.param_details["op_shape"])
        self.bpe_in = name2bpe(self.param_details["dtype_in"])

    @staticmethod
    def get_param_details(event):
        op_shape = tuple(event["args"]["Input Dims"][1])  # inp shard: [M // N_gpus, N]
        out_shape = tuple(event["args"]["Input Dims"][2])  # out: [M, N]
        dtype_in = event["args"]["Input type"][1]  # BFloat16
        stride_input = tuple(event["args"]["Input Strides"][1])
        return {
            "op_shape": op_shape,
            "out_shape": out_shape,
            "dtype_in": dtype_in,
            "stride_input": stride_input,
        }

    def flops(self):
        return 0

    def bytes(self):
        # read inp shard + write full gathered out
        num_elems_out = prod(self.param_details["out_shape"])
        return (self.num_elems + num_elems_out) * self.bpe_in
