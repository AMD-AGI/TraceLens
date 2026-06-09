###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Performance models for pseudo-op extensions.
"""

from TraceLens.PerfModel.utils import torch_dtype_map, name2bpe
import re
from TraceLens.PerfModel.perf_model import (
    GEMM,
    BinaryElementwise,
    UnaryElementwise,
    FusedRoPE,
)
from math import prod


class gemm_a8w8_blockscale(GEMM):
    """
    Performance model for AITER's gemm_a8w8_blockscale kernel.

    Computes: Y[M, N] = dequant(X[M, K]) @ dequant(W[N, K]).T
    where X and W are INT8 tensors with per-block FP32 scale factors.

    Reference implementation:
        aiter/aiter/ops/triton/gemm/basic/gemm_a8w8_blockscale.py

    Kernel mechanics:
        The kernel tiles the K dimension into blocks of size BLOCK_SIZE_K.
        Within each block, INT8 values from X and W are multiplied to produce
        INT32 partial products, then rescaled by (x_scale * w_scale) for that
        block before accumulating into FP32.

    Roofline calculation -- FLOPs:
        FLOPs = 2 * M * N * K  (inherited from GEMM base class)

        We count only the matrix-multiply FLOPs (multiply-accumulate). The
        per-block scale multiplications (x_scale[m, k_block] * w_scale[k_block, n_block])
        are O(M * N * K / block_size) -- orders of magnitude fewer than the
        matmul FLOPs -- and are ignored for roofline purposes.

    Roofline calculation -- Bytes:
        bytes_A      = M * K * bpe(X)          # X is INT8  -> 1 byte
        bytes_B      = N * K * bpe(W)          # W is INT8  -> 1 byte
        bytes_output = M * N * bpe(output)     # output is BF16 -> 2 bytes
        Total        = bytes_A + bytes_B + bytes_output

        Scale tensors (x_scale, w_scale) are omitted from the byte count.
        Their sizes -- x_scale: (M, ceil(K/block_k)) and w_scale:
        (ceil(N/block_n), ceil(K/block_k)) -- are negligible relative to the
        main operands since block sizes are typically 128-256.

    Compute precision:
        INT8 inputs -> FP8 peak TFLOPS/s used for the roofline ceiling
        (maps through the first element of dtype_A_B via torch_dtype_map).

    Expected Input Dims from trace:
        [[M, K], [N, K], [x_scale_shape], [w_scale_shape], ...]

    Expected Input type from trace:
        [dtype_x, dtype_w, dtype_x_scale, dtype_w_scale, ...]
    """

    @staticmethod
    def get_param_details(event):
        dims = event["args"]["Input Dims"]
        types = event["args"].get("Input type", []) or []
        M, K = dims[0][0], dims[0][1]
        N = dims[1][0]
        details = {
            "B": 1,
            "M": M,
            "N": N,
            "K": K,
            "bias": False,
            "dtype_A_B": (
                types[0] if len(types) > 0 else "",
                types[1] if len(types) > 1 else "",
                "c10::bfloat16",
            ),
        }
        # FP8/INT8 block-scale quant config; block sizes derived from scale-tensor shapes.
        try:
            x_scale, w_scale = dims[2], dims[3]
            block_k = (
                (-(-K // x_scale[-1])) if x_scale and x_scale[-1] else None
            )  # ceil(K/scale_cols)
            block_n = (-(-N // w_scale[0])) if w_scale and w_scale[0] else None
            details.update(
                {
                    "quant_scheme": "a8w8_blockscale",
                    "quant_granularity": "per_block",
                    "block_k": block_k,
                    "block_n": block_n,
                    "scale_dtype": types[2] if len(types) > 2 else "float32",
                }
            )
        except (IndexError, TypeError, ZeroDivisionError):
            pass
        # Output spec is inferred (torch traces record input dims only): Y[M, N] in bf16.
        details["output_shape"] = (M, N)
        details["output_dtype"] = "c10::bfloat16"
        return details

    def bytes(self):
        dtype_A_B = self.param_details["dtype_A_B"]
        self.bpe_mat1 = name2bpe(dtype_A_B[0])
        self.bpe_mat2 = name2bpe(dtype_A_B[1])
        self.bpe_output = name2bpe(dtype_A_B[2])
        self.bpe_bias = name2bpe(dtype_A_B[2])  # dummy since bias is not used

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
        is_bias = (
            len(event["args"]["Input Dims"]) > 2
            and len(event["args"]["Input Dims"][2]) > 0
        )
        return {
            "M": event["args"]["Input Dims"][0][0],
            "N": event["args"]["Input Dims"][1][0],
            "K": event["args"]["Input Dims"][0][1],
            "bias": is_bias,
            "dtype_A_B": (
                event["args"]["Input type"][0],
                event["args"]["Input type"][1],
                (
                    event["args"]["Input type"][2]
                    if len(event["args"]["Input type"]) > 2
                    else ""
                ),
            ),
        }

    def bytes(self):
        dtype_A_B = self.param_details["dtype_A_B"]
        bias = self.param_details["bias"]
        self.bpe_mat1 = name2bpe(dtype_A_B[0])
        self.bpe_mat2 = name2bpe(dtype_A_B[1])
        self.bpe_output = name2bpe(dtype_A_B[2]) if bias else name2bpe(dtype_A_B[1])
        self.bpe_bias = name2bpe(dtype_A_B[2]) if bias else name2bpe(dtype_A_B[1])
        return super().bytes(
            bpe_mat1=self.bpe_mat1,
            bpe_mat2=self.bpe_mat2,
            bpe_bias=self.bpe_bias,
            bpe_output=self.bpe_output,
        )


class batched_gemm_a16wfp4(GEMM):
    """
    Performance model for AITER's batched_gemm_a16wfp4_ kernel.

    Computes: Y[b] = X[b] @ W[b].T  for each batch b
    where X is BF16/FP16 with shape (B, M, K), and W is FP4 E2M1 packed
    as uint8 with shape (B, N, K//2), two FP4 values per byte.

    Reference implementation:
        aiter/aiter/ops/triton/gemm/batched/batched_gemm_a16wfp4.py

    Kernel mechanics:
        The kernel pre-quantizes BF16 activations to MXFP4 on-the-fly and
        performs batched matrix multiplication using FP4 tensor cores.
        Weights are stored in FP4 E2M1 format with E8M0 per-group scales
        (one scale per 32 K-dimension elements).  Optional split-K with a
        separate reduction kernel is used for small-M shapes.

    Roofline calculation -- FLOPs:
        FLOPs = B * 2 * M * N * K  (batched GEMM, per-batch inherited from
        GEMM base class and scaled by B)

        Only matrix-multiply FLOPs are counted.  Per-block scaling
        (w_scales) and split-K reduction FLOPs are ignored as they are
        orders of magnitude smaller than the matmul.

    Roofline calculation -- Bytes:
        bytes_X      = B * M * K   * bpe(X)      # X is BF16  -> 2 bytes
        bytes_W      = B * N * K/2 * 1            # W is FP4 packed (0.5 B/elem)
        bytes_output = B * M * N   * bpe(output)  # output is BF16 -> 2 bytes

    Compute precision:
        Mapped from the first input dtype (activation dtype, typically
        BF16) via torch_dtype_map.

    Expected Input Dims from trace:
        [[B, M, K], [B, N, K//2], [B, N, K//32], ...]

    Expected Input type from trace:
        [dtype_x, dtype_w_packed, dtype_w_scales, ...]
    """

    @staticmethod
    def get_param_details(event):
        return {
            "B": event["args"]["Input Dims"][0][0],
            "M": event["args"]["Input Dims"][0][1],
            "N": event["args"]["Input Dims"][1][1],
            "K": event["args"]["Input Dims"][0][2],
            "bias": False,
            "dtype_A_B": (
                event["args"]["Input type"][0],
                event["args"]["Input type"][1],
                "c10::bfloat16",
            ),
        }

    def flops(self):
        return self.B * super().flops()

    def bytes(self):
        dtype_A_B = self.param_details["dtype_A_B"]
        self.bpe_mat1 = name2bpe(dtype_A_B[0])
        self.bpe_mat2 = 0.5  # FP4: 4 bits = 0.5 bytes per element
        self.bpe_output = name2bpe(dtype_A_B[2])
        self.bpe_bias = name2bpe(dtype_A_B[2])

        per_batch = super().bytes(
            bpe_mat1=self.bpe_mat1,
            bpe_mat2=self.bpe_mat2,
            bpe_bias=self.bpe_bias,
            bpe_output=self.bpe_output,
        )
        return None if per_batch is None else self.B * per_batch

    def get_compute_precision(self):
        """Return the compute precision for this operation."""
        return torch_dtype_map("mxfp4")


class batched_gemm_a8w8(GEMM):
    """
    Performance model for AITER's batched_gemm_a8w8 kernel.

    Computes: Y[b] = X[b] @ W[b].T  for each batch b
    where X is BF16/FP16 with shape (B, M, K), quantized to INT8 on-the-fly
    using per-token grouped quantization, and WQ is pre-quantized INT8 with
    shape (B, N, K) and per-batch-element scaling.

    Reference implementation:
        aiter/aiter/ops/triton/gemm/batched/
            batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant.py

    Kernel mechanics:
        The kernel tiles M, N, K with configurable block sizes.  X is
        dynamically quantized to INT8 per group (group_size elements along K)
        inside the kernel.  WQ is read as pre-quantized INT8 and rescaled by
        a per-batch w_scale factor.  Accumulation is in FP32, output cast to
        BF16/FP16.

    Roofline calculation -- FLOPs:
        FLOPs = B * 2 * M * N * K

    Roofline calculation -- Bytes:
        bytes_X      = B * M * K * bpe(X)       # X is BF16/FP16 -> 2 bytes
        bytes_W      = B * N * K * bpe(WQ)      # WQ is INT8     -> 1 byte
        bytes_output = B * M * N * bpe(output)  # output is BF16 -> 2 bytes

        Scale tensors (w_scale) are negligible and omitted.

    Compute precision:
        INT8 matmul -> mapped via torch_dtype_map on WQ's dtype.

    Expected Input Dims from trace:
        [[B, M, K] or [M, B, K], [B, N, K], [w_scale_shape], ...]

    Expected Input type from trace:
        [dtype_x, dtype_wq, dtype_w_scale, ...]
    """

    @staticmethod
    def get_param_details(event):
        x_shape = event["args"]["Input Dims"][0]
        wq_shape = event["args"]["Input Dims"][1]
        B = wq_shape[0]
        N = wq_shape[1]
        K = x_shape[2]
        M = x_shape[1] if x_shape[0] == B else x_shape[0]
        return {
            "B": B,
            "M": M,
            "N": N,
            "K": K,
            "bias": False,
            "dtype_A_B": (
                event["args"]["Input type"][0],
                event["args"]["Input type"][1],
                "c10::bfloat16",
            ),
        }

    def flops(self):
        return self.B * super().flops()

    def bytes(self):
        dtype_A_B = self.param_details["dtype_A_B"]
        self.bpe_mat1 = name2bpe(dtype_A_B[0])
        self.bpe_mat2 = name2bpe(dtype_A_B[1])
        self.bpe_output = name2bpe(dtype_A_B[2])
        self.bpe_bias = name2bpe(dtype_A_B[2])

        per_batch = super().bytes(
            bpe_mat1=self.bpe_mat1,
            bpe_mat2=self.bpe_mat2,
            bpe_bias=self.bpe_bias,
            bpe_output=self.bpe_output,
        )
        return None if per_batch is None else self.B * per_batch

    def get_compute_precision(self):
        return torch_dtype_map(self.param_details["dtype_A_B"][1])


class gemm_a16w16_atomic_(GEMM):
    """
    Performance model for AITER's gemm_a16w16_atomic_ kernel.

    Computes: Y[M, N] = X[M, K] @ W[N, K].T using atomic split-K reduction.
    Both X and W are BF16/FP16 tensors.

    Reference implementation:
        aiter/aiter/ops/triton/gemm/basic/gemm_a16w16_atomic.py

    Kernel mechanics:
        The kernel tiles M, N, K with configurable block sizes and uses an
        optional split-K strategy (NUM_KSPLIT > 1) where partial results
        are accumulated via atomic additions on the output tensor.

    Roofline calculation -- FLOPs:
        FLOPs = 2 * M * N * K  (inherited from GEMM base class)

    Roofline calculation -- Bytes:
        bytes_A      = M * K * bpe(X)       # X is BF16/FP16 -> 2 bytes
        bytes_B      = N * K * bpe(W)       # W is BF16/FP16 -> 2 bytes
        bytes_output = M * N * bpe(output)  # output is BF16/FP16 -> 2 bytes
        Total        = bytes_A + bytes_B + bytes_output

    Expected Input Dims from trace:
        [[M, K], [N, K], ...]

    Expected Input type from trace:
        [dtype_x, dtype_w, ...]
    """

    @staticmethod
    def get_param_details(event):
        return {
            "B": 1,
            "M": event["args"]["Input Dims"][0][0],
            "N": event["args"]["Input Dims"][1][0],
            "K": event["args"]["Input Dims"][0][1],
            "bias": False,
            "dtype_A_B": (
                event["args"]["Input type"][0],
                event["args"]["Input type"][1],
                (
                    event["args"]["Input type"][3]
                    if len(event["args"]["Input type"]) > 3
                    else "c10::bfloat16"
                ),
            ),
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


class gemm_a16w16(gemm_a16w16_atomic_):
    """
    Performance model for AITER's gemm_a16w16 kernel.

    Computes: Y[M, N] = X[M, K] @ W[N, K].T. Both X and W are BF16/FP16.

    Reference implementation:
        aiter/aiter/ops/triton/gemm/basic/gemm_a16w16.py

    Identical roofline to gemm_a16w16_atomic_; only the output dtype handling
    differs. The traced arg layout is (x, w, bias, dtype, ...), so Input type[3]
    is the output ``dtype`` scalar (recorded as ``Scalar``) rather than a tensor.
    The output dtype therefore follows the input (BF16, per the aiter default),
    not Input type[3] as in the atomic_ variant (where index 3 is the output
    tensor ``y``). flops/bytes/compute-precision are inherited.
    """

    @staticmethod
    def get_param_details(event):
        dims = event["args"]["Input Dims"]
        types = event["args"]["Input type"]
        bias = len(dims) > 2 and len(dims[2]) > 0
        return {
            "B": 1,
            "M": dims[0][0],
            "N": dims[1][0],
            "K": dims[0][1],
            "bias": bias,
            "dtype_A_B": (types[0], types[1], types[0]),
        }


class GroupQuant(BinaryElementwise):
    """
    Performance model for group quantization.
    """

    category = "GroupQuant"
    bwd_category = None

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
        self.bpe_in1 = name2bpe(self.dtype_in1_in2_out[0])
        self.bpe_in2 = name2bpe(self.dtype_in1_in2_out[1])
        self.bpe_out = name2bpe(self.dtype_in1_in2_out[2])


class dynamic_per_group_scaled_quant_kernel(GroupQuant):
    """
    Performance model for AITER ``dynamic_per_group_scaled_quant_kernel`` HIP launches.

    Graph-replay capture layout (typical):
        [out_fp8, block_scales?, per_token_scales, ...]
    When the bf16 input tensor is not traced, assume it matches *out* shape.
    """

    @staticmethod
    def get_param_details(event):
        args = event.get("args", {})
        dims = args.get("Input Dims") or []
        types = args.get("Input type") or []
        if len(dims) < 3:
            raise ValueError(
                f"dynamic_per_group_scaled_quant_kernel needs >=3 Input Dims, got {dims}"
            )

        shape_out = tuple(dims[0])
        dtype_out = types[0] if types else "c10::Float8_e4m3fn"

        mid = dims[1]
        scales = dims[2]
        mid_type = types[1] if len(types) > 1 else ""
        if isinstance(mid, (list, tuple)) and len(mid) == len(shape_out):
            shape_in1 = tuple(mid)
            dtype_in1 = mid_type or "c10::Half"
        else:
            shape_in1 = shape_out
            dtype_in1 = "c10::Half"

        shape_in2 = tuple(scales) if isinstance(scales, (list, tuple)) else ()
        dtype_in2 = types[2] if len(types) > 2 else "float"

        strides = args.get("Input Strides") or [[], [], []]
        return {
            "shape_in1": shape_in1,
            "shape_in2": shape_in2,
            "shape_out": shape_out,
            "dtype_in1_in2_out": (dtype_in1, dtype_in2, dtype_out),
            "stride_input1": tuple(strides[1]) if len(strides) > 1 else (),
            "stride_input2": tuple(strides[2]) if len(strides) > 2 else (),
            "stride_output": tuple(strides[0]) if strides else (),
        }

    def bytes(self):
        pd = self.param_details
        bytes_out = prod(pd["shape_out"]) * name2bpe(pd["dtype_in1_in2_out"][2])
        bytes_in = prod(pd["shape_in1"]) * name2bpe(pd["dtype_in1_in2_out"][0])
        bytes_scales = prod(pd["shape_in2"]) * name2bpe(pd["dtype_in1_in2_out"][1])
        mid = self.event["args"]["Input Dims"][1]
        mid_type = (self.event["args"].get("Input type") or [""])[1]
        bytes_block = 0
        if (
            isinstance(mid, (list, tuple))
            and len(mid) == 2
            and mid != list(pd["shape_in1"])
        ):
            bytes_block = prod(mid) * name2bpe(mid_type or "c10::Half")
        return bytes_in + bytes_out + bytes_scales + bytes_block

    def flops(self):
        return prod(self.param_details["shape_out"]) * 3

    def get_compute_precision(self):
        dtype = self.param_details["dtype_in1_in2_out"][0]
        return torch_dtype_map(dtype) if dtype else None


class per_group_quant(GroupQuant):
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
        bytes = prod(self.param_details["shape_out"]) * name2bpe(
            self.param_details["dtype_in1_in2_out"][2]
        )
        bytes += prod(self.param_details["shape_in1"]) * name2bpe(
            self.param_details["dtype_in1_in2_out"][0]
        )
        bytes += prod(self.param_details["shape_in2"]) * name2bpe(
            self.param_details["dtype_in1_in2_out"][1]
        )
        return bytes

    def flops(self):
        return self.nelems_out * 2.59375  # Based on the rocprof counter values

    def get_compute_precision(self):
        """Return the compute precision for this operation."""
        # Use the input activation dtype (index 0) as the compute precision;
        dtype = self.dtype_in1_in2_out[0] if self.dtype_in1_in2_out else None
        return torch_dtype_map(dtype) if dtype else None


class vllm_triton_per_token_group_quant_fp8(GroupQuant):
    """
    Performance model for vllm::triton_per_token_group_quant_fp8.

    Wraps `per_token_group_quant_fp8` (vllm fp8_utils.py), which launches the
    Triton kernel `_per_token_group_quant_fp8` when the CUDA native quant path
    is unavailable: for each row, each contiguous group of `group_size`
    elements along the last dimension is scaled by a per-group FP32 factor and
    written as FP8.

    Roofline -- bytes moved (approximate):
        Read  x [M, N] in the trace input dtype (typically BF16)
        Write x_q [M, N] as FP8 (1 byte/elem)
        Write scales [M, ceil(N / group_size)] as FP32

    Roofline -- FLOPs:
        Treated as ~6 FLOPs per input element (abs/max reduction amortized over
        groups, scale, divide, clamp, store) for order-of-magnitude roofline use.

    Expected trace:
        Input Dims: ((M, N), ()) for tensor x and scalar group_size
        Input type: (dtype_x, 'Scalar')
        Concrete Inputs: often ('', '<group_size>') e.g. ('', '128')
    """

    def __init__(self, event, arch=None, python_path=None):
        self.event = event
        self.arch = arch
        self.param_details = self.get_param_details(event)

    @staticmethod
    def get_param_details(event):
        args = event["args"]
        dims = args.get("Input Dims") or []
        shape_x = tuple(dims[0]) if len(dims) > 0 else ()
        if len(shape_x) >= 2:
            M, N = shape_x[0], shape_x[1]
        elif len(shape_x) == 1:
            M, N = 1, shape_x[0]
        else:
            M, N = 1, 1

        concrete = args.get("Concrete Inputs") or []
        group_size = 128
        if len(concrete) > 1:
            raw = str(concrete[1]).strip()
            if raw and raw.lower() not in ("none",):
                try:
                    group_size = int(raw)
                except ValueError:
                    pass

        num_groups = max(1, (N + group_size - 1) // group_size)
        dtype_x = args.get("Input type", ("",))[0] if args.get("Input type") else ""

        return {
            "M": M,
            "N": N,
            "group_size": group_size,
            "num_groups": num_groups,
            "shape_x": shape_x,
            "dtype_x": dtype_x,
        }

    def flops(self):
        M = self.param_details["M"]
        N = self.param_details["N"]
        return 6 * M * N

    def bytes(self):
        M = self.param_details["M"]
        N = self.param_details["N"]
        ng = self.param_details["num_groups"]
        bpe_in = name2bpe(self.param_details["dtype_x"])
        if bpe_in is None:
            return None
        bpe_fp8 = name2bpe("c10::float8_e4m3fn")
        if bpe_fp8 is None:
            bpe_fp8 = 1
        return M * N * bpe_in + M * N * bpe_fp8 + M * ng * 4

    def get_compute_precision(self):
        dtype = self.param_details.get("dtype_x")
        return torch_dtype_map(dtype) if dtype else None


class aiter_silu_and_mul(UnaryElementwise):
    """
    Performance model for aiter::silu_and_mul.

    Computes the gated SiLU activation used after MoE stage-1 GEMM (split-K path):
        out[..., :inter_dim] = silu(input[..., :inter_dim]) * input[..., inter_dim:]

    AITER signature: silu_and_mul(out: Tensor, input: Tensor) -> None
        out   — shape [..., inter_dim],     dtype BFloat16
        input — shape [..., 2 * inter_dim], dtype FP32 (split-K accumulation) or BFloat16

    Expected Input Dims from trace:
        [out_shape, input_shape]
        e.g. [(4, 9, 256), (4, 9, 512)]  →  op_shape=(4,9,256), 2*inter_dim=512

    Expected Input type from trace:
        [dtype_out, dtype_input]  (note: reversed — out is arg 0, input is arg 1)
    """

    @staticmethod
    def get_param_details(event):
        # Input Dims[0] = out (shape [..., inter_dim])
        # Input Dims[1] = input (shape [..., 2 * inter_dim], gate+up concatenated)
        op_shape = tuple(event["args"]["Input Dims"][0])
        dtype_in = event["args"]["Input type"][1]  # input (gate+up) dtype
        dtype_out = event["args"]["Input type"][0]  # output dtype
        stride_input = tuple(event["args"]["Input Strides"][1])
        stride_output = tuple(event["args"]["Input Strides"][0])
        return {
            "op_shape": op_shape,
            "dtype_in_out": (dtype_in, dtype_out),
            "stride_input": stride_input,
            "stride_output": stride_output,
        }

    def flops(self):
        return 5 * prod(self.param_details["op_shape"])

    def bytes(self):
        return 2 * self.nelems * self.bpe_in + self.nelems * self.bpe_out


class sgl_kernel_silu_and_mul(aiter_silu_and_mul):
    """
    Performance model for sgl_kernel::silu_and_mul.

    Gated SiLU (same math as aiter::silu_and_mul):
        out[..., :inter_dim] = silu(input[..., :inter_dim]) * input[..., inter_dim:]

    SGLang registers this op from sgl_kernel (e.g. sgl_kernel/elementwise.py silu_and_mul,
    used by srt/layers/activation.py SiluAndMul).

    Signature: silu_and_mul(out, input) -> None
        out   — shape [M, inter_dim],     dtype BFloat16
        input — shape [M, 2 * inter_dim], dtype BFloat16

    Expected Input Dims from trace:
        [out_shape, input_shape]
        e.g. [(65536, 2304), (65536, 4608)]

    Expected Input type from trace:
        [dtype_out, dtype_input]

    FLOPs and bytes are inherited from aiter_silu_and_mul.
    """

    pass


class aiter_gelu_and_mul(aiter_silu_and_mul):
    """
    Performance model for aiter::gelu_and_mul.

    Computes the gated GELU activation (erf path):
        out[..., :inter_dim] = gelu(input[..., :inter_dim]) * input[..., inter_dim:]

    AITER signature: gelu_and_mul(out: Tensor, input: Tensor) -> None
        out   — shape [..., inter_dim],     dtype BFloat16
        input — shape [..., 2 * inter_dim], dtype BFloat16

    Expected Input Dims from trace:
        [out_shape, input_shape]

    FLOPs: ~8 per output element (erf-based GELU: div, erf, add, mul, mul; then gate mul).
    Bytes: identical to aiter_silu_and_mul (read gate+up, write out).
    """

    def flops(self):
        return 8 * prod(self.param_details["op_shape"])


class aiter_gelu_tanh_and_mul(aiter_silu_and_mul):
    """
    Performance model for aiter::gelu_tanh_and_mul.

    Computes the gated GELU activation (tanh/fast approximation path):
        out[..., :inter_dim] = gelu_tanh(input[..., :inter_dim]) * input[..., inter_dim:]

    AITER signature: gelu_tanh_and_mul(out: Tensor, input: Tensor) -> None
        out   — shape [..., inter_dim],     dtype BFloat16
        input — shape [..., 2 * inter_dim], dtype BFloat16

    Expected Input Dims from trace:
        [out_shape, input_shape]

    FLOPs: ~10 per output element (tanh GELU: x^3, coefficients, tanh, scale; then gate mul).
    Bytes: identical to aiter_silu_and_mul (read gate+up, write out).
    """

    def flops(self):
        return 10 * prod(self.param_details["op_shape"])


class gemm_afp4wfp4(GEMM):
    """
    Performance model for AITER's gemm_afp4wfp4_ kernel.

    Computes: Y[M, N] = (X * x_scales) @ (W * w_scales)^T  with MXFP4 inputs.
        X is FP4 E2M1 packed as uint8 with shape (M, K // 2) (two FP4 values per byte).
        W is FP4 E2M1 packed as uint8 with shape (N, K // 2).
        x_scales / w_scales are E8M0 per-group scales with one scale per 32 K-elements
        (shapes (M, K // 32) and (N, K // 32) respectively).
        Y is BF16/FP16 with shape (M, N).

    Reference implementation:
        aiter/aiter/ops/triton/gemm/basic/gemm_afp4wfp4.py

    Kernel mechanics:
        Tiles M, N, K with optional split-K. Both X and W are read as FP4 (uint8
        packed), dequantized with their per-group E8M0 scales inside the MFMA
        pipeline, accumulated in FP32, and cast to BF16/FP16 on store. Split-K
        partials are reduced by a separate kernel.

    Roofline calculation -- FLOPs:
        FLOPs = 2 * M * N * K   (inherited from GEMM base class)

        Per-block scale multiplications are O(M * N * K / 32) and negligible
        relative to the matmul.

    Roofline calculation -- Bytes:
        bytes_X      = M * K * 0.5             # FP4 packed
        bytes_W      = N * K * 0.5             # FP4 packed
        bytes_output = M * N * bpe(output)     # BF16/FP16 -> 2 bytes

        Scale tensors are negligible and omitted.

    Compute precision:
        MXFP4 -> mapped via torch_dtype_map("mxfp4"). We override get_compute_precision
        defensively because traces may report the FP4 storage dtype as "unsigned char"
        which would otherwise map to fp8.

    Expected Input Dims from trace:
        [[M, K // 2], [N, K // 2], [M, K // 32], [N, K // 32], (), [M, N], (), ()]

    Expected Input type from trace:
        [dtype_x, dtype_w, dtype_x_scale, dtype_w_scale, ..., dtype_y, ...]
    """

    @staticmethod
    def get_param_details(event):
        return {
            "B": 1,
            "M": event["args"]["Input Dims"][0][0],
            "N": event["args"]["Input Dims"][1][0],
            "K": event["args"]["Input Dims"][0][1] * 2,
            "bias": False,
            "dtype_A_B": (
                event["args"]["Input type"][0],
                event["args"]["Input type"][1],
                "c10::bfloat16",
            ),
        }

    def bytes(self):
        dtype_A_B = self.param_details["dtype_A_B"]
        self.bpe_mat1 = 0.5  # FP4 packed
        self.bpe_mat2 = 0.5  # FP4 packed
        self.bpe_output = name2bpe(dtype_A_B[2])
        self.bpe_bias = name2bpe(dtype_A_B[2])  # unused (bias=False)

        return super().bytes(
            bpe_mat1=self.bpe_mat1,
            bpe_mat2=self.bpe_mat2,
            bpe_bias=self.bpe_bias,
            bpe_output=self.bpe_output,
        )

    def get_compute_precision(self):
        return torch_dtype_map("mxfp4")


class fused_flatten_mxfp4_quant(UnaryElementwise):
    """
    Performance model for aiter.ops.triton.quant.fused_mxfp4_quant.fused_flatten_mxfp4_quant
    (surfaced in traces as sglang_profiler::fused_mxfp4_quant_fused_flatten_mxfp4_quant).

    Flattens the last two dims of a (M, N1, N2) BF16/FP16 tensor and MXFP4-quantizes
    each row to packed FP4 + E8M0 per-32-elem block scales.

    Reference implementation:
        aiter/aiter/ops/triton/quant/fused_mxfp4_quant.py:149

    Signature:
        fused_flatten_mxfp4_quant(x: Tensor)  # x shape (M, N1, N2), bf16/fp16
        -> out: (M, (N1*N2)//2) uint8, out_block_scales: (M, ceil(N1*N2/32)) uint8

    Roofline calculation -- FLOPs:
        ~2 FLOPs per input element (max-abs reduction per 32-elem group + scale).

    Roofline calculation -- Bytes:
        read x:              nelems * bpe_in              # BF16/FP16 -> 2 bytes
        write packed FP4:    nelems * 0.5
        write E8M0 scales:   nelems // 32 * 1

    Expected Input Dims from trace:
        [[M, N1, N2]]

    Category: bucketed as GroupQuant (it computes per-32-elem block E8M0 scales
    and quantizes to MXFP4, same family as per_group_quant /
    vllm_triton_per_token_group_quant_fp8). The UnaryElementwise base is kept
    only for the single-input FLOPs/bytes roofline machinery.
    """

    category = "GroupQuant"
    sheet_category = "GroupQuant"

    @staticmethod
    def get_param_details(event):
        op_shape = tuple(event["args"]["Input Dims"][0])
        dtype_in = event["args"]["Input type"][0]
        stride_input = tuple(event["args"]["Input Strides"][0])
        return {
            "op_shape": op_shape,
            "dtype_in_out": (dtype_in, dtype_in),
            "stride_input": stride_input,
            "stride_output": None,
        }

    def flops(self):
        return 2 * self.nelems

    def bytes(self):
        if self.bpe_in is None:
            return None
        bytes_read = self.nelems * self.bpe_in
        bytes_write_fp4 = self.nelems * 0.5
        bytes_write_scales = (self.nelems // 32) * 1
        return bytes_read + bytes_write_fp4 + bytes_write_scales


class fused_dynamic_mx_quant_moe_sort_hip(fused_flatten_mxfp4_quant):
    """
    Performance model for aiter::fused_dynamic_mx_quant_moe_sort_hip
    (kernel aiter::fused_mx_quant_moe_sort_kernel<bf16, fp4, ...>).

    Fuses dynamic MXFP4 quantization of the BF16 activation with MoE token sort.
    Trace arg layout differs from fused_flatten_mxfp4_quant only in input
    position: [0] is the packed FP4 output (M, N/2), [1] the E8M0 block scales,
    [2] the BF16 activation input (M, N), and [3]/[4] the int sort arrays.

    Approximation: models the quant traffic over the activation input. The MoE
    sort index permutation (small int arrays) and the padded sorted-layout rows
    are runtime-dependent and excluded.
    """

    @staticmethod
    def get_param_details(event):
        op_shape = tuple(event["args"]["Input Dims"][2])
        dtype_in = event["args"]["Input type"][2]
        stride_input = tuple(event["args"]["Input Strides"][2])
        return {
            "op_shape": op_shape,
            "dtype_in_out": (dtype_in, dtype_in),
            "stride_input": stride_input,
            "stride_output": None,
        }


class aiter_rope_cached_positions_2c_fwd_impl(FusedRoPE):
    """
    Performance model for aiter::rope_cached_positions_2c_fwd_impl.

    Two-channel (Q + K) forward RoPE with cached cos / sin and per-token positions.
    Rotates the rotate-dim slice of each channel using NEOX or GPT-J rotate style;
    the no-position part is copied through.

    Reference implementation:
        aiter/aiter/ops/rope.py:207

    Signature:
        rope_cached_positions_2c_fwd_impl(
            output_x,   # (B, S, H_q,  d)
            output_y,   # (B, S, H_kv, d)
            input_x,    # (B, S, H_q,  d)
            input_y,    # (B, S, H_kv, d)
            cos,        # (max_pos, 1, 1, d_cs)
            sin,        # (max_pos, 1, 1, d_cs)
            positions,  # (B, S), int64
            rotate_style, reuse_freqs_front_part, nope_first,
        ) -> None

    Roofline calculation -- FLOPs:
        Each rotated element pair takes 4 muls + 2 adds = 6 ops over 2 elements,
        i.e. ~3 FLOPs per element. Applied to both x and y channels.

        FLOPs = 3 * (numel(input_x) + numel(input_y))

    Roofline calculation -- Bytes:
        read + write both x and y channels:
            2 * (numel(input_x) + numel(input_y)) * bpe_in

        cos/sin/positions are typically cache-resident and small; omitted from
        the dominant HBM traffic.

    Expected Input Dims from trace:
        [[B, S, H_q, d], [B, S, H_kv, d], [B, S, H_q, d], [B, S, H_kv, d],
         [max_pos, 1, 1, d_cs], [max_pos, 1, 1, d_cs], [B, S]]
    """

    @staticmethod
    def get_param_details(event):
        input_dims = event["args"]["Input Dims"]
        # input_x at Input Dims[2], input_y at Input Dims[3]
        x_shape = tuple(input_dims[2])
        y_shape = tuple(input_dims[3])
        return {
            "x_shape": x_shape,
            "y_shape": y_shape,
            "num_elements": prod(x_shape) + prod(y_shape),
        }

    def flops(self):
        return 3 * self.param_details["num_elements"]

    def bytes(self):
        if self.bpe is None:
            return None
        n = self.param_details["num_elements"]
        return 2 * n * self.bpe


class sgl_kernel_rotary_embedding(FusedRoPE):
    """
    Performance model for sgl_kernel::rotary_embedding.
    In-place RoPE on query and key (vLLM-style rotate-half).

    Reference implementation:
        sglang/sgl-kernel/python/sgl_kernel/elementwise.py:406 (rotary_embedding)

    Expected Input Dims format:
        [positions, query, key, (), cos_sin_cache, ()]
    Concrete Inputs[3] = head_size; cos_sin_cache last dim = rot_dim.

    Only the rot_dim slice of each head is rotated:
        flops = 3 * rotated_elems ; bytes = 2 * rotated_elems * bpe (read+write q, k).
    """

    def __init__(self, event, arch=None, python_path=None, **kwargs):
        super().__init__(event, arch, python_path, **kwargs)
        # Input type[0] is positions (int64); bpe must come from query (Input type[1]).
        qdtype = event["args"]["Input type"][1]
        bpe = name2bpe(qdtype)
        self.bpe = bpe if bpe is not None else 2
        self._qdtype = qdtype

    def get_compute_precision(self):
        return torch_dtype_map(self._qdtype) if self._qdtype else None

    @staticmethod
    def get_param_details(event):
        input_dims = event["args"]["Input Dims"]
        concrete = event["args"].get("Concrete Inputs", [])
        q_shape = tuple(input_dims[1])
        k_shape = tuple(input_dims[2])
        head_size = q_shape[-1]
        if len(concrete) > 3 and str(concrete[3]).strip():
            head_size = int(concrete[3])
        rot_dim = head_size
        if len(input_dims) > 4 and len(input_dims[4]) > 1:
            rot_dim = input_dims[4][1]
        num_tokens = q_shape[0]
        num_q_heads = q_shape[1] // head_size
        num_k_heads = k_shape[1] // head_size
        return {
            "num_elements": num_tokens * (num_q_heads + num_k_heads) * rot_dim,
        }


class sglang_store_cache:
    """
    Performance model for sglang::store_cache.
    Scatter-copy of K/V rows into the paged KV cache (pure memory move, no math).

    Reference implementation:
        sglang/python/sglang/srt/mem_cache/memory_pool.py (store_cache)

    Expected Input Dims format: [k, v, k_cache, v_cache, indices, (), ()]
        e.g. [(64, 5120), (64, 5120), (168724, 5120), (168724, 5120), (64,), (), ()]
    Expected Input type format: [dtype_k, dtype_v, dtype_k_cache, dtype_v_cache, ...]

    Reads k + v and writes T touched rows of k_cache + v_cache:
        flops = 0 ; bytes = 4 * T * D * bpe (2 reads + 2 writes).
    """

    category = "InferenceAttention"
    bwd_category = None

    def __init__(self, event, arch=None, python_path=None, **kwargs):
        self.event = event
        self.arch = arch
        self.python_path = python_path
        self.param_details = self.get_param_details(event)
        self.nelems = self.param_details["nelems"]
        dtype = self.param_details["dtype"]
        bpe = name2bpe(dtype) if dtype else None
        self.bpe = bpe if bpe is not None else 2

    @staticmethod
    def get_param_details(event):
        k_shape = tuple(event["args"]["Input Dims"][0])
        dtype = event["args"]["Input type"][0]
        return {
            "op_shape": k_shape,
            "nelems": prod(k_shape),
            "dtype": dtype,
        }

    def flops(self):
        return 0

    def bytes(self):
        return 4 * self.nelems * self.bpe

    def get_maf_type(self):
        return "vector"

    def get_compute_precision(self):
        dtype = self.param_details.get("dtype")
        return torch_dtype_map(dtype) if dtype else None


class mixed_sample_outer_exponential:
    """
    Performance model for ``aiter::mixed_sample_outer_exponential`` (ATOM
    ``model_ops/sampler.py``, ``mix_sample_outer_exponential_kernel``).

    Exponential-noise sampling: ``token_id[t] = argmax_v(logits[t,v] /
    exp_noise[t,v])`` over the vocab dim. Memory-bound elementwise reduction.
    Reads logits ``Input Dims[1]`` [T, V] and exp_noise ``Input Dims[2]``;
    FLOPs = 2*T*V, bytes = read logits + read exp_noise + write T int32 ids.
    """

    category = "elementwise"
    bwd_category = None

    def __init__(self, event, arch=None, python_path=None):
        self.T = self.param_details["T"]
        self.V = self.param_details["V"]
        self.dtype_logits = self.param_details["dtype_logits"]
        self.dtype_noise = self.param_details["dtype_noise"]
        logits_shape = tuple(dims[1])
        T = logits_shape[0] if len(logits_shape) >= 1 else 1
        V = logits_shape[1] if len(logits_shape) >= 2 else 1
        dtype_logits = types[1] if len(types) > 1 else "c10::bfloat16"
        dtype_noise = types[2] if len(types) > 2 else "float"
        return {
            "T": T,
            "V": V,
            "dtype_logits": dtype_logits,
            "dtype_noise": dtype_noise,
        }

    def flops(self):
        return 2 * self.T * self.V

    def bytes(self):
        bpe_logits = name2bpe(self.dtype_logits)
        bpe_noise = name2bpe(self.dtype_noise)
        if bpe_logits is None or bpe_noise is None:
            return None
        read_logits = self.T * self.V * bpe_logits
        read_noise = self.T * self.V * bpe_noise
        write_ids = self.T * 4  # int32 token id per row
        return read_logits + read_noise + write_ids

    def get_maf_type(self):
        return "vector"

    def get_compute_precision(self):
        return torch_dtype_map(self.dtype_logits) if self.dtype_logits else None


class aiter_fused_qk_rope_cat_and_cache_mla(FusedRoPE):
    """
    Performance model for aiter::fused_qk_rope_cat_and_cache_mla
    (RoPE on q_pe/k_pe + KV-cache write).

    Reference implementation:
        aiter/aiter/ops/triton/fusions/fused_kv_cache.py:88
        (kernel _fused_qk_rope_cat_and_cache_mla_kernel)

    Applies rotary position embedding to the rope (pe) slices of Q and K,
    concatenates the nope + pe parts into a single head dim, returns the rotated
    Q, and writes the concatenated K (nope || pe) into kv_cache in place at the
    slots given by slot_mapping.

    Expected Input Dims from trace:
        [0] q_nope   = (T, QH, D_lora)
        [1] q_pe     = (T, QH, D_pe)
        [2] k_nope   = (T, KH, D_lora)
        [3] k_pe     = (T, KH, D_pe)
        [4] kv_cache = (B_cache, KH, D_lora + D_pe)
        ...

    Expected Input type from trace:
        [bf16, bf16, bf16, bf16, <kv_cache dtype, e.g. fp8>, ...]

    Roofline -- FLOPs:
        RoPE rotates only the pe slices of Q and K. ~3 FLOPs/element.
        flops = 3 * (T*QH*D_pe + T*KH*D_pe)

    Roofline -- bytes moved:
        read  q_nope + q_pe + k_nope + k_pe : nelems * bpe_in
        write q_out  (T, QH, D_lora+D_pe)   : T*QH*(D_lora+D_pe) * bpe_out
        write kv rows (T tokens written)    : T*KH*(D_lora+D_pe) * bpe_kv
        cos/sin/pos are small + cache-resident; omitted.

    """

    sheet_category = "FusedRoPE"

    def __init__(self, event, arch=None, python_path=None, **kwargs):
        self.event = event
        self.arch = arch
        self.python_path = python_path
        self.param_details = self.get_param_details(event)
        self.bpe_in = name2bpe(self.param_details["dtype_in"])
        self.bpe_kv = name2bpe(self.param_details["dtype_kv"])

    @staticmethod
    def get_param_details(event):
        dims = event["args"]["Input Dims"]
        types = event["args"]["Input type"]
        q_nope = tuple(dims[0])
        q_pe = tuple(dims[1])
        k_nope = tuple(dims[2])
        T, QH, D_lora = q_nope
        D_pe = q_pe[2]
        KH = k_nope[1]
        dtype_in = types[0]
        dtype_kv = types[4] if len(types) > 4 and types[4] else dtype_in
        return {
            "T": T,
            "QH": QH,
            "KH": KH,
            "D_lora": D_lora,
            "D_pe": D_pe,
            "dtype_in": dtype_in,
            "dtype_kv": dtype_kv,
        }

    def flops(self):
        p = self.param_details
        return 3 * (p["T"] * p["QH"] * p["D_pe"] + p["T"] * p["KH"] * p["D_pe"])

    def bytes(self):
        p = self.param_details
        T, QH, KH, D_lora, D_pe = (p["T"], p["QH"], p["KH"], p["D_lora"], p["D_pe"])
        bpe_in = self.bpe_in
        if bpe_in is None:
            return None
        bpe_out = bpe_in  # q_out keeps the Q input dtype
        bpe_kv = self.bpe_kv if self.bpe_kv is not None else bpe_in
        read_q = (T * QH * D_lora + T * QH * D_pe) * bpe_in
        read_k = (T * KH * D_lora + T * KH * D_pe) * bpe_in
        write_q = T * QH * (D_lora + D_pe) * bpe_out
        write_kv = T * KH * (D_lora + D_pe) * bpe_kv
        return read_q + read_k + write_q + write_kv

    def get_compute_precision(self):
        return torch_dtype_map(self.param_details["dtype_in"])


class fused_qk_rope_concat_and_cache_mla(aiter_fused_qk_rope_cat_and_cache_mla):
    """
    Performance model for aiter::fused_qk_rope_concat_and_cache_mla
    (kernel fuse_qk_rope_concat_and_cache_mla_per_head_kernel; DeepSeek-V3.1 MLA).

    RoPE on the pe slices of Q/K + static FP8 quant + paged KV-cache write.
    Trace arg layout: [0] q_nope (T, QH, D_lora), [1] q_pe (T, QH, D_pe),
    [2] kv_c (T, D_lora), [3] k_pe (T, D_pe), [4] kv_cache (FP8 paged),
    [5] q_out (T, QH, D_lora + D_pe), FP8.

    the KV is the single MLA latent (kv_c / k_pe are 2D, so KH = 1),
    and both q_out and the KV cache are FP8
    (the base assumes q_out keeps the BF16 input dtype).

    Approximation: only the T written KV slots are counted, not the full paged
    cache tensor at Input Dims[4]; cos/sin/positions are small / cache-resident;
    the per-tensor FP8 quant flops are omitted.
    """

    @staticmethod
    def get_param_details(event):
        dims = event["args"]["Input Dims"]
        types = event["args"]["Input type"]
        T, QH, D_lora = tuple(dims[0])
        D_pe = dims[1][2]
        KH = 1  # MLA single latent KV head (kv_c / k_pe are 2D [T, D])
        dtype_in = types[0]
        dtype_kv = types[4] if len(types) > 4 and types[4] else dtype_in
        dtype_out = types[5] if len(types) > 5 and types[5] else dtype_in
        return {
            "T": T,
            "QH": QH,
            "KH": KH,
            "D_lora": D_lora,
            "D_pe": D_pe,
            "dtype_in": dtype_in,
            "dtype_kv": dtype_kv,
            "dtype_out": dtype_out,
        }

    def bytes(self):
        p = self.param_details
        if self.bpe_in is None:
            return None
        bpe_out = name2bpe(p["dtype_out"]) or self.bpe_in
        bpe_kv = self.bpe_kv if self.bpe_kv is not None else self.bpe_in
        T, QH, KH, D_lora, D_pe = (p["T"], p["QH"], p["KH"], p["D_lora"], p["D_pe"])
        read_q = (T * QH * D_lora + T * QH * D_pe) * self.bpe_in
        read_k = (T * KH * D_lora + T * KH * D_pe) * self.bpe_in
        write_q = T * QH * (D_lora + D_pe) * bpe_out
        write_kv = T * KH * (D_lora + D_pe) * bpe_kv
        return read_q + read_k + write_q + write_kv


class sglang_quant_dynamic_mxfp4_quant(fused_flatten_mxfp4_quant):
    """
    Performance model for sglang_profiler::quant_dynamic_mxfp4_quant.

    Reference implementation:
        aiter/aiter/ops/triton/quant/... dynamic_mxfp4_quant
        (sglang quark w4a4 mxfp4 scheme).

    Dynamic MXFP4 activation quant. Single BF16 input (M, N); output is packed
    FP4 + per-32-element E8M0 scales. Shares the MXFP4 quant roofline of
    fused_flatten_mxfp4_quant (read x + write 0.5 B/elem FP4 + 1/32 B/elem scales).

    Expected Input Dims from trace:
        [0] = (M, N)   (>2D inputs are flattened along the leading dims)
    Expected Input type from trace:
        [bf16]
    """

    @staticmethod
    def get_param_details(event):
        dims = event["args"]["Input Dims"]
        types = event["args"]["Input type"]
        op_shape = tuple(dims[0])
        dtype_in = types[0]
        return {
            "op_shape": op_shape,
            "dtype_in_out": (dtype_in, dtype_in),
            "stride_input": None,
            "stride_output": None,
        }


class aiter_fused_dynamic_mxfp4_quant_moe_sort_hip(fused_flatten_mxfp4_quant):
    """
    Performance model for aiter::fused_dynamic_mxfp4_quant_moe_sort_hip.

    Reference implementation:
        aiter/aiter/ops/quant.py:1103 (fused_dynamic_mxfp4_quant_moe_sort ->
        fused_dynamic_mx_quant_moe_sort; HIP kernel mxfp4_quant_moe_sort_kernel)

    Fuses dynamic MXFP4 activation quant with MoE token sorting. The sort only
    touches small index tensors (metadata) and is negligible relative to the
    activation quant traffic.

    Expected Input Dims from trace:
        [0] out_fp4   = (M, N // 2)        packed FP4
        [1] out_scale = (pad_rows, N // 32) E8M0 (sorted layout)
        [2] input     = (M, N)             BF16   <- modeled input
        [3] sorted_ids, [4] num_valid_ids  (metadata)
    """

    @staticmethod
    def get_param_details(event):
        dims = event["args"]["Input Dims"]
        types = event["args"]["Input type"]
        op_shape = tuple(dims[2])
        dtype_in = types[2]
        return {
            "op_shape": op_shape,
            "dtype_in_out": (dtype_in, dtype_in),
            "stride_input": None,
            "stride_output": None,
        }


class aiter_dynamic_per_group_scaled_quant_fp4(fused_flatten_mxfp4_quant):
    """
    Performance model for aiter::dynamic_per_group_scaled_quant_fp4.

    Reference implementation:
        aiter/aiter/ops/quant.py:744 (dynamic_per_group_scaled_quant_fp4 ->
        dynamic_per_group_scaled_quant; HIP kernel
        dynamic_per_group_scaled_quant_kernel<bf16, fp4_t, 32>).

    Dynamic per-group (group_size=32) FP4 quant of a BF16 activation. The OUTPUT
    tensor is Input Dims[0] and the BF16 input is Input Dims[1]

    Expected Input Dims from trace:
        [0] out    = (M, N // 2)   packed FP4
        [1] input  = (M, N)        BF16   <- modeled input
        [2] scales = (M, N // 32)  E8M0
    """

    @staticmethod
    def get_param_details(event):
        dims = event["args"]["Input Dims"]
        types = event["args"]["Input type"]
        op_shape = tuple(dims[1])
        dtype_in = types[1]
        return {
            "op_shape": op_shape,
            "dtype_in_out": (dtype_in, dtype_in),
            "stride_input": None,
            "stride_output": None,
        }
