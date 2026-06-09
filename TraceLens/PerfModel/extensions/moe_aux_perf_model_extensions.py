###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Performance models for MoE auxiliary kernels (sorting, top-k routing)."""

from math import prod

from TraceLens.PerfModel.utils import name2bpe, torch_dtype_map


def _dtype_from_types(types, index, default="float"):
    if not types or index >= len(types):
        return default
    dtype = types[index]
    if dtype in ("ScalarList", "Scalar", ""):
        return default
    return dtype


def _tensor_bytes_from_dims(dims, types, default_bpe=4):
    total = 0
    if not isinstance(dims, list):
        return 0
    for idx, shape in enumerate(dims):
        if not isinstance(shape, (list, tuple)) or not shape:
            continue
        if not all(isinstance(x, int) and x >= 0 for x in shape):
            continue
        nelem = prod(shape)
        if nelem <= 0:
            continue
        bpe = name2bpe(_dtype_from_types(types, idx)) or default_bpe
        total += nelem * bpe
    return total


class aiter_moe_sorting_kernel:
    """
    MoE token sorting (``MoeSortingKernel`` / ``aiter::moe_sorting_fwd``).

    Sorts tokens by expert assignment before grouped GEMM. Memory-bandwidth
    bound: reads/writes index buffers and touches the hidden-state tensor.
    """

    category = "MoE_aux"
    bwd_category = None

    def __init__(self, event, arch=None, python_path=None):
        self.event = event
        self.arch = arch
        self.python_path = python_path
        self.param_details = self.get_param_details(event)

    @staticmethod
    def get_param_details(event):
        args = event.get("args", {})
        dims = args.get("Input Dims") or []
        types = args.get("Input type") or []
        num_tokens = 1
        topk = 1
        if dims and isinstance(dims[0], (list, tuple)) and len(dims[0]) >= 2:
            num_tokens = int(dims[0][0])
            topk = int(dims[0][1])
        return {
            "num_tokens": num_tokens,
            "topk": topk,
            "bytes_moved": _tensor_bytes_from_dims(dims, types),
        }

    def flops(self):
        n = self.param_details["num_tokens"]
        k = self.param_details["topk"]
        return n * k * 8

    def bytes(self):
        return self.param_details["bytes_moved"]

    def get_compute_precision(self):
        return torch_dtype_map("float")

    def get_maf_type(self):
        return "vector"


class aiter_grouped_topk_kernel:
    """
    Expert top-k routing (``grouped_topk_kernel`` / ``aiter::grouped_topk``).

    Selects top-k experts per token from gating scores.
    """

    category = "MoE_aux"
    bwd_category = None

    def __init__(self, event, arch=None, python_path=None):
        self.event = event
        self.arch = arch
        self.python_path = python_path
        self.param_details = self.get_param_details(event)

    @staticmethod
    def get_param_details(event):
        args = event.get("args", {})
        dims = args.get("Input Dims") or []
        types = args.get("Input type") or []
        num_tokens = 1
        num_experts = 1
        topk = 1
        if dims and isinstance(dims[0], (list, tuple)) and len(dims[0]) >= 2:
            num_tokens = int(dims[0][0])
            num_experts = int(dims[0][1])
        if len(dims) > 2 and isinstance(dims[2], (list, tuple)) and len(dims[2]) >= 2:
            topk = int(dims[2][1])
        return {
            "num_tokens": num_tokens,
            "num_experts": num_experts,
            "topk": topk,
            "bytes_moved": _tensor_bytes_from_dims(dims, types),
        }

    def flops(self):
        n = self.param_details["num_tokens"]
        e = self.param_details["num_experts"]
        k = self.param_details["topk"]
        return n * e * max(k, 1) * 4

    def bytes(self):
        return self.param_details["bytes_moved"]

    def get_compute_precision(self):
        return torch_dtype_map("float")

    def get_maf_type(self):
        return "vector"
