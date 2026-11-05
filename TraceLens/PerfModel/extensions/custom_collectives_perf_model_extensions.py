###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Performance models for custom collective operation extensions.
"""

import re
from math import prod
from TraceLens.PerfModel.utils import (
    cxx_dtype_to_torch,
    dense_counted_tensor,
    name2bpe,
    torch_dtype_map,
)


class CustomCollective:
    category = "CustomCollective"
    bwd_category = None


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
    Bytes: memory traffic per GPU (read inp+res_inp+weight, write res_out+out).
    """

    def __init__(self, event, arch=None):
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
        # memory traffic per GPU (inter-GPU allreduce bandwidth is separate)
        # Reads:  inp, res_inp, weight
        # Writes: res_out, out
        bytes_read = 2 * self.num_elems * self.bpe_in + self.num_channels * self.bpe_in
        bytes_write = 2 * self.num_elems * self.bpe_in
        return bytes_read + bytes_write


class aiter_fused_allreduce_rmsnorm_(aiter_fused_allreduce_rmsnorm):
    """
    Performance model for the high-level Python op aiter::fused_allreduce_rmsnorm_.

    Same fused AllReduce + residual-add + RMSNorm math as the base class, but the
    recorded argument layout follows the Python wrapper's signature
    (inp, res_inp, w, eps, group_name, prefill_support, x_pad_to_multiple, gemma_norm)
    rather than the low-level C++ op's (ptr, reg, inp, res_inp, res_out, out, w, eps, ...).

    Expected Input Dims from trace:
        [inp_shape, res_inp_shape, w_shape, eps, group_name, prefill_support, x_pad_to_multiple]
        e.g. ((64, 7168), (64, 7168), (7168,), (), (), (), ())

    So inp is at index 0 and weight at index 2 (vs [2] / [6] in the base class).
    __init__, flops(), and bytes() are inherited unchanged.
    """

    @staticmethod
    def get_param_details(event):
        op_shape = tuple(event["args"]["Input Dims"][0])  # inp: [M, N]
        dtype_in = event["args"]["Input type"][0]  # BFloat16
        stride_input = tuple(event["args"]["Input Strides"][0])
        num_channels = event["args"]["Input Dims"][2][0]  # weight.shape[0] = N
        return {
            "op_shape": op_shape,
            "dtype_in": dtype_in,
            "stride_input": stride_input,
            "num_channels": num_channels,
        }


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
    Bytes: memory traffic per GPU (read inp + write out).
    """

    def __init__(self, event, arch=None):
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
        # memory traffic per GPU: read inp + write out
        return 2 * self.num_elems * self.bpe_in


class vllm_all_reduce(custom_ar_all_reduce):
    """HBM bytes of the local BF16 shard read and the reduced shard write; peer and IPC traffic are omitted."""

    def __init__(self, event, arch=None, python_path=None):
        self.event = event
        self.arch = arch
        self.param_details = self.get_param_details(event)
        shape = self.param_details["op_shape"]
        self.num_elems = prod(shape) if shape else 0
        self.bpe_in = name2bpe(self.param_details["dtype_in"]) if shape else None

    @staticmethod
    def get_param_details(event):
        dims = event["args"]["Input Dims"]
        types = event["args"].get("Input type") or []
        shape = tuple(dims[0]) if dims and dims[0] else ()
        return {
            "op_shape": shape,
            "dtype_in": types[0] if types else "",
            "stride_input": (),
        }

    def bytes(self):
        if not self.num_elems or self.bpe_in is None:
            return None
        if not dense_counted_tensor(self.event, (0,)):
            return None
        return 2 * self.num_elems * self.bpe_in


class sgl_kernel_all_reduce_reg(custom_ar_all_reduce):
    """
    Performance model for sgl_kernel::all_reduce_reg.

    Reference implementation:
        python/sglang/srt/distributed/device_communicators/custom_all_reduce_ops.py  # all_reduce_reg(fa, inp, out)

    SGLang's registered-buffer custom all-reduce. Pure inter-GPU communication;
    no on-chip compute FLOPs.

    Signature: all_reduce_reg(fa, inp, out) -> None
        fa   - int handle (Scalar in trace)
        inp  - shape [M, N], dtype BFloat16
        out  - shape [M, N], dtype BFloat16 (allreduced result)

    Expected Input Dims from trace:
        [(), (M, N), (M, N)]
        e.g. [(), (32, 7168), (32, 7168)]   # DSR1 hidden=7168, batch=32

    Expected Input type from trace:
        ['Scalar', 'c10::BFloat16', 'c10::BFloat16']

    flops/bytes inherited from custom_ar_all_reduce (FLOPs=0; memory bytes per GPU
    = 2 * num_elems * bpe_in for the local read+write; inter-GPU bandwidth is
    a separate concern from the on-chip roofline).
    """


class sgl_kernel_qr_all_reduce(custom_ar_all_reduce):
    """
    Performance model for sgl_kernel::qr_all_reduce.

    Reference implementation:
        python/sglang/srt/distributed/device_communicators/custom_all_reduce_ops.py  # qr_all_reduce(fa, inp, out, quant_level, cast_bf2half)
        python/sglang/srt/distributed/device_communicators/quick_all_reduce.py        # QuickAllReduce.quick_all_reduce

    SGLang's QuickReduce all-reduce variant. Launches `quickreduce::allreduce_prototype_twoshot<...>`
    or related kernels. Same per-GPU memory accounting as the registered
    all-reduce: read local input, write allreduced output.

    Signature: qr_all_reduce(fa, inp, out, quant_level, cast_bf2half) -> None
        fa             - int handle (Scalar)
        inp            - shape [M, N], dtype BFloat16
        out            - shape [M, N], dtype BFloat16 (allreduced result)
        quant_level    - int (Scalar; 0=bf16, 3=int4-quantized, ...)
        cast_bf2half   - bool (Scalar)

    Expected Input Dims from trace:
        [(), (M, N), (M, N), (), ()]
        e.g. [(), (1866, 7168), (1866, 7168), (), ()]   # DSR1 prefill, isl=1866

    Expected Input type from trace:
        ['Scalar', 'c10::BFloat16', 'c10::BFloat16', 'Scalar', 'Scalar']

    flops/bytes inherited from custom_ar_all_reduce (uses indices [1]/[2] for
    inp/out shape and dtype; trailing scalars are ignored). This deliberately
    counts only the BF16 memory traffic; intra-kernel int4 quant for higher
    `quant_level` reduces inter-GPU bytes but not the memory read/write step
    modeled here.
    """


class custom_ar_qr_all_reduce(custom_ar_all_reduce):
    """
    Performance model for _C_custom_ar::qr_all_reduce.

    Reference implementation:
        vllm/_custom_ops.py                                       # qr_all_reduce(fa, inp, out, quant_level, cast_bf2half)
        vllm/distributed/device_communicators/quick_all_reduce.py # QuickAllReduce.quick_all_reduce
        csrc/custom_all_reduce/quick_all_reduce.cu                # quickreduce::allreduce_prototype_*

    vLLM's QuickReduce all-reduce variant.  Same per-GPU memory accounting as a plain
    custom all-reduce: read local input, write the allreduced output.

    Signature: qr_all_reduce(fa, inp, out, quant_level, cast_bf2half) -> None
        fa             - int handle (Scalar)
        inp            - shape [M, N], dtype BFloat16
        out            - shape [M, N], dtype BFloat16 (allreduced result)
        quant_level    - int (Scalar; 0=bf16, 3=int4-quantized, ...)
        cast_bf2half   - bool (Scalar)

    Expected Input Dims from trace:
        [(), (M, N), (M, N), (), ()]
        e.g. [(), (1689, 7168), (1689, 7168), (), ()]   # DSR1 prefill

    Expected Input type from trace:
        ['Scalar', 'c10::BFloat16', 'c10::BFloat16', 'Scalar', 'Scalar']

    Concrete Inputs[3] = `quant_level` (e.g. '3' for int4 codec).
    Concrete Inputs[4] = `cast_bf2half` ('True'/'False').

    flops/bytes inherited from custom_ar_all_reduce (uses Input Dims[1] for
    `op_shape` and Input type[1] for dtype; trailing scalar args are ignored).
    The memory read+write is BF16 regardless of the inter-GPU codec; modeled as
    `2 * num_elems * bpe_in` bytes per GPU.
    """


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
    Bytes: memory traffic per GPU (read inp + write out).
    """

    def __init__(self, event, arch=None):
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
    Bytes: memory traffic per GPU (read inp shard + write full out).
    """

    def __init__(self, event, arch=None):
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


class sgl_kernel_reg_all_gather_into_tensor(CustomCollective):
    """
    Performance model for sglang::reg_all_gather_into_tensor.

    Reference implementation:
        python/sglang/srt/distributed/device_communicators/custom_all_reduce.py     # CustomAllreduce.all_gather (registered buffer path)
        sgl-kernel/csrc/allreduce/custom_all_reduce.cu                              # all_gather_reg kernel

    SGLang's registered-buffer all-gather: each rank contributes a shard
    `[M / W, N]`, the output is the gathered `[M, N]` tensor. Pure inter-GPU
    communication; no FLOPs.

    Signature: reg_all_gather_into_tensor(out, inp, fa) -> None
        out - shape [M, N],          dtype BFloat16 (gathered)
        inp - shape [M / W, N],      dtype BFloat16 (this rank's shard)
        fa  - int handle (Scalar)

    Expected Input Dims from trace:
        [out_shape, inp_shape, ()]
        e.g. [(256, 16160), (32, 16160), ()]   # W=8, M=32 -> gathered M=256

    Expected Input type from trace:
        ['c10::BFloat16', 'c10::BFloat16', 'Scalar']

    Roofline -- FLOPs:
        0 (pure communication).

    Roofline -- bytes moved:
        bytes_read_inp  = num_elems_inp_shard * bpe_in
        bytes_write_out = num_elems_out_full  * bpe_in
        Total           = bytes_read_inp + bytes_write_out

    Notes:
        get_param_details reads Input Dims[0] for `out_shape` and Input Dims[1]
        for the per-rank `op_shape` (shard). dtype is taken from Input type[0].
    """

    def __init__(self, event, arch=None):
        self.event = event
        self.arch = arch
        self.param_details = self.get_param_details(event)
        self.num_elems = prod(self.param_details["op_shape"])
        self.num_elems_out = prod(self.param_details["out_shape"])
        self.bpe_in = name2bpe(self.param_details["dtype_in"])

    @staticmethod
    def get_param_details(event):
        out_shape = tuple(event["args"]["Input Dims"][0])
        op_shape = tuple(event["args"]["Input Dims"][1])
        dtype_in = event["args"]["Input type"][0]
        return {
            "out_shape": out_shape,
            "op_shape": op_shape,
            "dtype_in": dtype_in,
        }

    def flops(self):
        return 0

    def bytes(self):
        return (self.num_elems + self.num_elems_out) * self.bpe_in


_AR_1STAGE_RE = re.compile(
    r"allreduce_fusion_kernel_1stage<([^,]+),\s*([^,]+),\s*(\d+),\s*(true|false)>"
)


class allreduce_fusion_kernel_1stage_synthetic(CustomCollective):
    """
    Performance model for the ``allreduce_fusion_kernel_1stage`` GPU kernel
    when it appears as a Synthetic Op (launched via AITER JIT, no parent
    PyTorch cpu_op and therefore no ``Input Dims``).

    Derives M, N, world_size, and dtype from the HIP launch grid/block dims
    and the C++ template parameters embedded in the kernel name.

    Grid mapping (verified against ``custom_all_reduce.cuh:1945``):
        grid.x  = min(token_num, 80)  — capped at kMaxBlocks=80; the kernel
                  uses a grid-stride loop for overflow.  CUDA-graph captures
                  are only made for small batch sizes so grid.x == token_num
                  in practice, but this model will undercount M for
                  token_num > 80.
        block.x = ceil(out_hidden_dim / PACK_SIZE, WARP_SIZE) — derived from
                  out_hidden_dim, not hidden_dim.  For GLM-52 (and most
                  models) they are equal; if they differ, N will reflect
                  out_hidden_dim.

    Template params ``<T_in, T_out, W, use_fp32_acc>``:
        T_in        — input dtype  (e.g. ``std::bfloat16_t``)
        T_out       — output dtype
        W           — world_size (number of TP ranks)
        use_fp32_acc — accumulation precision flag

    FLOPs: residual-add + RMSNorm (same as ``aiter_fused_allreduce_rmsnorm``).
    Bytes: local HBM only (read inp + res_inp + weight, write res_out + out).
    """

    @staticmethod
    def _parse_name(name):
        m = _AR_1STAGE_RE.search(name)
        if not m:
            return None
        return {
            "cxx_dtype_in": m.group(1).strip(),
            "cxx_dtype_out": m.group(2).strip(),
            "world_size": int(m.group(3)),
            "use_fp32_acc": m.group(4) == "true",
        }

    @classmethod
    def can_model(cls, event):
        args = event.get("args") or {}
        grid = args.get("grid")
        block = args.get("block")
        if not grid or not block or grid[0] <= 0 or block[0] <= 0:
            return False
        return cls._parse_name(event.get("name", "")) is not None

    def __init__(self, event, arch=None, python_path=None):
        self.event = event
        self.arch = arch
        self.param_details = self.get_param_details(event)
        if self.param_details.get("_no_perf"):
            self.num_elems = 0
            self.num_channels = 0
            self.bpe_in = 0
            return
        M = self.param_details["M"]
        N = self.param_details["N"]
        self.num_elems = M * N
        self.num_channels = N
        self.bpe_in = self.param_details["bpe_in"]

    @staticmethod
    def get_param_details(event):
        name = event.get("name", "")
        parsed = allreduce_fusion_kernel_1stage_synthetic._parse_name(name)
        if parsed is None:
            return {"_no_perf": True}

        cxx_dtype = parsed["cxx_dtype_in"]
        torch_dtype = cxx_dtype_to_torch(cxx_dtype)
        if torch_dtype is None:
            return {"_no_perf": True}
        bpe_in = name2bpe(torch_dtype)
        if bpe_in is None:
            bpe_in = 2
        pack_size = 16 // bpe_in

        args = event.get("args") or {}
        grid = args.get("grid", [0, 1, 1])
        block = args.get("block", [0, 1, 1])

        M = grid[0]
        # grid.x is capped at kMaxBlocks=80; beyond that the kernel
        # grid-strides so grid.x no longer equals the real token count.
        if M >= 80:
            return {"_no_perf": True}
        N = block[0] * pack_size

        return {
            "M": M,
            "N": N,
            "world_size": parsed["world_size"],
            "dtype_in": torch_dtype,
            "bpe_in": bpe_in,
            "op_shape": (M, N),
        }

    def flops(self):
        if self.param_details.get("_no_perf") or self.num_elems == 0:
            return None
        add_flops = self.num_elems
        non_norm_elems = self.num_elems // self.num_channels
        rms_flops = non_norm_elems * (2 * self.num_channels + 2)
        rms_flops += self.num_elems * 2
        return add_flops + rms_flops

    def bytes(self):
        if self.param_details.get("_no_perf") or self.num_elems == 0:
            return None
        bytes_read = 2 * self.num_elems * self.bpe_in + self.num_channels * self.bpe_in
        bytes_write = 2 * self.num_elems * self.bpe_in
        return bytes_read + bytes_write
