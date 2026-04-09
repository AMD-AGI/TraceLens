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
from TraceLens.PerfModel.perf_model import GEMM, BinaryElementwise, UnaryElementwise
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
        return {
            "B": 1,
            "M": event["args"]["Input Dims"][0][0],
            "N": event["args"]["Input Dims"][1][0],
            "K": event["args"]["Input Dims"][0][1],
            "bias": False,
            "dtype_A_B": (
                event["args"]["Input type"][0],
                event["args"]["Input type"][1],
                "c10::bfloat16",
            ),
        }

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


class GroupQuant(BinaryElementwise):
    """
    Performance model for group quantization.
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
        self.bpe_in1 = name2bpe(self.dtype_in1_in2_out[0])
        self.bpe_in2 = name2bpe(self.dtype_in1_in2_out[1])
        self.bpe_out = name2bpe(self.dtype_in1_in2_out[2])


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
        # Use first input dtype as the compute precision
        dtype = self.dtype_in1_in2_out[1] if self.dtype_in1_in2_out else None
        return torch_dtype_map(dtype) if dtype else None


class concat_and_cache_mla(BinaryElementwise):
    """Performance model for _C_cache_ops::concat_and_cache_mla."""

    def __init__(self, event, arch=None, python_path=None):
        self.event = event
        self.arch = arch
        self.param_details = self.get_param_details(event)

    @staticmethod
    def get_param_details(event):
        pass

    def flops(self):
        pass

    def bytes(self):
        pass


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
