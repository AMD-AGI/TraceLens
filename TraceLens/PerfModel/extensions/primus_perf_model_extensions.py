###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Performance models for Primus Turbo / Flux pseudo-op extensions.
"""

from math import prod

from TraceLens.PerfModel.perf_model import GEMM, UnaryElementwise
from TraceLens.PerfModel.utils import name2bpe, torch_dtype_map

# Constants
_MXFP4_GROUP_SIZE = 32
_MXFP4_BPE = 0.5  # FP4 E2M1: 4 bits = 0.5 bytes per element
_E8M0_BPE = 1  # E8M0 scale: 1 byte per element


class gemm_fp4_impl(GEMM):
    """
    Performance model for primus_turbo::gemm_fp4_impl.

    Dual MXFP4 GEMM: Y[M, N] = dequant(A[M, K]) @ dequant(B[N, K]).T
    where A and B are FP4 E2M1 packed as Float4_e2m1fn_x2 (two values per byte)
    with per-32-element E8M0 block scales.

    Kernel mechanics:
        Both operands are MXFP4 (FP4 E2M1 with E8M0 per-group scales).
        The kernel tiles M, N, K and dequantizes inside the MFMA pipeline.
        Optional bias (BF16) is added to the output. Optional output scale
        tensors are written for the backward pass.

    Roofline calculation -- FLOPs:
        FLOPs = 2 * M * N * K   (standard GEMM)

    Roofline calculation -- Bytes:
        bytes_A      = M * K * 0.5                # FP4 packed
        bytes_B      = N * K * 0.5                # FP4 packed
        bytes_output = M * N * bpe(output)         # BF16 -> 2 bytes
        bytes_bias   = N * bpe(bias) if bias
        Scale tensors and output scales are negligible and omitted.

    Compute precision:
        MXFP4 -> mapped via torch_dtype_map("mxfp4").

    Expected Input Dims from trace:
        [[M, K//2], [M, K//32], [], [N, K//2], [N, K//32],
         [], [], [], [], [], [], [N]_or_[], []_or_[M,K//32], []_or_[N,K//32]]

    Expected Input type from trace:
        [Float4_e2m1fn_x2, Float8_e8m0fnu, Scalar, Float4_e2m1fn_x2,
         Float8_e8m0fnu, Scalar, ...]
    """

    @staticmethod
    def get_param_details(event):
        dims = event["args"]["Input Dims"]
        types = event["args"].get("Input type", []) or []
        M = dims[0][0]
        K = dims[0][1] * 2  # FP4 packed: two values per byte
        N = dims[3][0]
        has_bias = len(dims) > 11 and dims[11] and len(dims[11]) > 0
        return {
            "B": 1,
            "M": M,
            "N": N,
            "K": K,
            "bias": has_bias,
            "dtype_A_B": (
                types[0] if len(types) > 0 else "",
                types[3] if len(types) > 3 else "",
                "c10::bfloat16",
            ),
            "has_out_scales": (len(dims) > 12 and dims[12] and len(dims[12]) > 0),
        }

    def bytes(self, bpe_mat1=None, bpe_mat2=None, bpe_bias=None, bpe_output=None):
        dtype_A_B = self.param_details["dtype_A_B"]
        self.bpe_mat1 = _MXFP4_BPE
        self.bpe_mat2 = _MXFP4_BPE
        self.bpe_output = name2bpe(dtype_A_B[2])
        bias_dtype = "c10::bfloat16" if self.param_details["bias"] else dtype_A_B[2]
        self.bpe_bias = name2bpe(bias_dtype)
        return super().bytes(
            bpe_mat1=self.bpe_mat1,
            bpe_mat2=self.bpe_mat2,
            bpe_bias=self.bpe_bias,
            bpe_output=self.bpe_output,
        )

    def get_compute_precision(self):
        return torch_dtype_map("mxfp4")


class a6w4_mm_qa(GEMM):
    """
    Performance model for primus_flux::a6w4_mm_qa.

    A6W4 quantized activation matmul with full-precision (BF16) weights:
    Y[M, N] = dequant(act_quant[M, K]) @ W[N, K].T + bias[N]

    Activations are 6-bit quantized (stored as unsigned char, 1 byte/elem)
    with per-group unsigned char scales. Weights and bias are BF16.

    Roofline calculation -- FLOPs:
        FLOPs = 2 * M * N * K

    Roofline calculation -- Bytes:
        bytes_act    = M * K * 1                   # u8 quantized activation
        bytes_W      = N * K * bpe(weight)         # BF16 weight
        bytes_output = M * N * bpe(output)         # BF16 output
        bytes_bias   = N * bpe(bias) if bias

    Expected Input Dims from trace:
        [[M, K], [scale_elems], [N, K], [N]]

    Expected Input type from trace:
        [unsigned char, unsigned char, BFloat16, BFloat16]
    """

    @staticmethod
    def get_param_details(event):
        dims = event["args"]["Input Dims"]
        types = event["args"].get("Input type", []) or []
        M, K = dims[0]
        N = dims[2][0]
        has_bias = len(dims) > 3 and dims[3] and len(dims[3]) > 0
        return {
            "B": 1,
            "M": M,
            "N": N,
            "K": K,
            "bias": has_bias,
            "dtype_A_B": (
                types[0] if len(types) > 0 else "unsigned char",
                types[2] if len(types) > 2 else "c10::bfloat16",
                "c10::bfloat16",
            ),
        }

    def bytes(self, bpe_mat1=None, bpe_mat2=None, bpe_bias=None, bpe_output=None):
        dtype_A_B = self.param_details["dtype_A_B"]
        self.bpe_mat1 = 1  # unsigned char: 1 byte/elem
        self.bpe_mat2 = name2bpe(dtype_A_B[1])
        self.bpe_output = name2bpe(dtype_A_B[2])
        self.bpe_bias = name2bpe(dtype_A_B[2])
        return super().bytes(
            bpe_mat1=self.bpe_mat1,
            bpe_mat2=self.bpe_mat2,
            bpe_bias=self.bpe_bias,
            bpe_output=self.bpe_output,
        )

    def get_compute_precision(self):
        return torch_dtype_map("c10::bfloat16")


class a6w4_mm_qa_prequant_w(GEMM):
    """
    Performance model for primus_flux::a6w4_mm_qa_prequant_w.

    A6W4 matmul with pre-quantized FP4 weights:
    Y[M, N] = dequant(act_quant[M, K]) @ dequant(w_quant[N, K]).T + bias[N]

    Activations are 6-bit quantized (stored as u8, 1 byte/elem).
    Weights are FP4 E2M1 packed (2 values per byte) with E8M0 per-group scales.
    Optional BF16 bias.

    Roofline calculation -- FLOPs:
        FLOPs = 2 * M * N * K

    Roofline calculation -- Bytes:
        bytes_act    = M * K * 1                   # u8 activation
        bytes_W      = N * K * 0.5                 # FP4 packed weight
        bytes_output = M * N * bpe(output)         # BF16 output
        bytes_bias   = N * bpe(bias) if bias

    Expected Input Dims from trace:
        [[M, K], [scale_elems], [N, K//2], [N, K//32], [], [N]_or_[]]

    Expected Input type from trace:
        [unsigned char, unsigned char, unsigned char, unsigned char, Scalar, BFloat16]
    """

    @staticmethod
    def get_param_details(event):
        dims = event["args"]["Input Dims"]
        types = event["args"].get("Input type", []) or []
        M, K = dims[0]
        N = dims[2][0]
        has_bias = len(dims) > 5 and dims[5] and len(dims[5]) > 0
        return {
            "B": 1,
            "M": M,
            "N": N,
            "K": K,
            "bias": has_bias,
            "dtype_A_B": (
                types[0] if len(types) > 0 else "unsigned char",
                types[2] if len(types) > 2 else "unsigned char",
                "c10::bfloat16",
            ),
        }

    def bytes(self, bpe_mat1=None, bpe_mat2=None, bpe_bias=None, bpe_output=None):
        self.bpe_mat1 = 1  # u8 activation: 1 byte/elem
        self.bpe_mat2 = _MXFP4_BPE  # FP4 packed weight: 0.5 byte/elem
        self.bpe_output = name2bpe("c10::bfloat16")
        self.bpe_bias = name2bpe("c10::bfloat16")
        return super().bytes(
            bpe_mat1=self.bpe_mat1,
            bpe_mat2=self.bpe_mat2,
            bpe_bias=self.bpe_bias,
            bpe_output=self.bpe_output,
        )

    def get_compute_precision(self):
        return torch_dtype_map("mxfp4")


class a6w4_act_dual(UnaryElementwise):
    """
    Performance model for primus_flux::a6w4_act_dual.

    Fused dual activation quantization for the A6W4 mixed-precision pipeline:
    quantizes a BF16 activation tensor into two 6-bit representations
    (for the forward GEMM and its paired backward GEMM path).

    Roofline calculation -- FLOPs:
        ~4 FLOPs per element (abs, max-reduction per group, scale, quantize)
        applied twice (dual output).

    Roofline calculation -- Bytes:
        read  x:         M * C * bpe_in              # BF16 -> 2 bytes
        write quant_1:   M * C * 1                   # u8 (6-bit padded)
        write quant_2:   M * C * 1                   # u8 (6-bit padded)
        write scales:    2 * M * (C // 32) * 1       # u8 per-group scales

    Expected Input Dims from trace:
        [[M, C]]

    Expected Input type from trace:
        [BFloat16]
    """

    category = "GroupQuant"

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
        return 4 * self.nelems * 2

    def bytes(self):
        if self.bpe_in is None:
            return None
        M, C = self.param_details["op_shape"]
        bytes_read = M * C * self.bpe_in
        bytes_write_quant = 2 * M * C * 1  # two u8 quantized outputs
        bytes_write_scales = 2 * M * (C // _MXFP4_GROUP_SIZE) * 1
        return bytes_read + bytes_write_quant + bytes_write_scales


class _QuantizeMxfp4Base(UnaryElementwise):
    """Base class for primus_flux MXFP4 quantization variants."""

    category = "GroupQuant"

    def flops(self):
        return 2 * self.nelems

    def bytes(self):
        if self.bpe_in is None:
            return None
        bytes_read = self.nelems * self.bpe_in
        bytes_write_fp4 = self.nelems * _MXFP4_BPE
        bytes_write_scales = max(1, self.nelems // _MXFP4_GROUP_SIZE) * _E8M0_BPE
        return bytes_read + bytes_write_fp4 + bytes_write_scales


class quantize_mxfp4_h16_dual(_QuantizeMxfp4Base):
    """
    Performance model for primus_flux::quantize_mxfp4_h16_dual.

    Dual MXFP4 quantization: quantizes a BF16 [M, N] tensor into two
    separate MXFP4 representations (packed FP4 + E8M0 per-32-elem scales),
    one for the forward path and one for the backward path.

    Roofline calculation -- FLOPs:
        ~2 FLOPs per element per output (max-abs reduction + scale)
        x 2 outputs = 4 * M * N total.

    Roofline calculation -- Bytes:
        read  x:         M * N * bpe_in
        write 2 * (FP4 packed + E8M0 scales)

    Expected Input Dims from trace:
        [[M, N], [], []]

    Expected Input type from trace:
        [BFloat16, Scalar, Scalar]
    """

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
        return 4 * self.nelems

    def bytes(self):
        if self.bpe_in is None:
            return None
        bytes_read = self.nelems * self.bpe_in
        bytes_write_fp4 = 2 * self.nelems * _MXFP4_BPE
        bytes_write_scales = 2 * max(1, self.nelems // _MXFP4_GROUP_SIZE) * _E8M0_BPE
        return bytes_read + bytes_write_fp4 + bytes_write_scales


class quantize_mxfp4_h16_dual_bias_gate(UnaryElementwise):
    """
    Performance model for primus_flux::quantize_mxfp4_h16_dual_bias_gate.

    Fused bias-add + gate activation (SiLU) + dual MXFP4 quantization.
    Reads two BF16 inputs (x and gate) and their biases, applies
    x = x + bias; gate = silu(gate + gate_bias); out = x * gate,
    then quantizes the result into two MXFP4 outputs.

    Roofline calculation -- FLOPs:
        bias-add: 2 * M * N (add bias to x and gate)
        SiLU:     3 * M * N (neg, exp, add, rcp, mul on gate)
        gate-mul: M * N
        quant:    2 * M * N per output * 2 = 4 * M * N
        Total:    ~10 * M * N

    Roofline calculation -- Bytes:
        read  x[M, N], bias[1, N], gate[M, N], gate_bias[N]
        write 2 * (FP4 packed + E8M0 scales)

    Expected Input Dims from trace:
        [[M, N], [1, N], [M, N], [N], [out1_scale_shape], [out2_scale_shape]]

    Expected Input type from trace:
        [BFloat16, BFloat16, BFloat16, BFloat16, Float8_e8m0fnu, Float8_e8m0fnu]
    """

    category = "GroupQuant"

    @staticmethod
    def get_param_details(event):
        dims = event["args"]["Input Dims"]
        op_shape = tuple(dims[0])
        dtype_in = event["args"]["Input type"][0]
        stride_input = tuple(event["args"]["Input Strides"][0])
        return {
            "op_shape": op_shape,
            "dtype_in_out": (dtype_in, dtype_in),
            "stride_input": stride_input,
            "stride_output": None,
        }

    def flops(self):
        return 10 * self.nelems

    def bytes(self):
        if self.bpe_in is None:
            return None
        M, N = self.param_details["op_shape"]
        bytes_read = (2 * M * N + N + N) * self.bpe_in  # x, gate, bias, gate_bias
        bytes_write_fp4 = 2 * M * N * _MXFP4_BPE
        bytes_write_scales = 2 * max(1, M * N // _MXFP4_GROUP_SIZE) * _E8M0_BPE
        return bytes_read + bytes_write_fp4 + bytes_write_scales


class quantize_mxfp4_h16_dual_bias_packed(UnaryElementwise):
    """
    Performance model for primus_flux::quantize_mxfp4_h16_dual_bias_packed.

    Fused bias-add + dual MXFP4 quantization with packed output layout.
    The input BF16 tensor (from a preceding concat/chunk) is split into
    sub-tensors, each bias-added and quantized to MXFP4.

    Roofline calculation -- FLOPs:
        ~4 FLOPs per element (bias-add + dual quant)

    Roofline calculation -- Bytes:
        read  x[M, N] in BF16
        write 2 * (FP4 packed + E8M0 scales)

    Expected Input Dims from trace:
        [[M, N], [], [out_scale1_shape], [out_scale2_shape]]

    Expected Input type from trace:
        [BFloat16, Scalar, Float8_e8m0fnu, Float8_e8m0fnu]
    """

    category = "GroupQuant"

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
        return 4 * self.nelems

    def bytes(self):
        if self.bpe_in is None:
            return None
        bytes_read = self.nelems * self.bpe_in
        bytes_write_fp4 = 2 * self.nelems * _MXFP4_BPE
        bytes_write_scales = 2 * max(1, self.nelems // _MXFP4_GROUP_SIZE) * _E8M0_BPE
        return bytes_read + bytes_write_fp4 + bytes_write_scales


class quantize_mxfp4_h16_dual_bias_dgelu(UnaryElementwise):
    """
    Performance model for primus_flux::quantize_mxfp4_h16_dual_bias_dgelu.

    Backward dGELU activation derivative fused with dual MXFP4 quantization.
    Reads two BF16 inputs (grad_output and activation cache), computes
    dGELU, and quantizes the result into two MXFP4 outputs.

    Roofline calculation -- FLOPs:
        dGELU:   ~8 FLOPs per element (GELU derivative involves erf)
        quant:   2 * 2 per element (dual)
        Total:   ~12 * M * N

    Roofline calculation -- Bytes:
        read  grad_out[M, N], act_cache[M, N] in BF16
        write 2 * (FP4 packed + E8M0 scales)

    Expected Input Dims from trace:
        [[M, N], [M, N]]

    Expected Input type from trace:
        [BFloat16, BFloat16]
    """

    category = "GroupQuant"

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
        return 12 * self.nelems

    def bytes(self):
        if self.bpe_in is None:
            return None
        bytes_read = 2 * self.nelems * self.bpe_in  # two BF16 inputs
        bytes_write_fp4 = 2 * self.nelems * _MXFP4_BPE
        bytes_write_scales = 2 * max(1, self.nelems // _MXFP4_GROUP_SIZE) * _E8M0_BPE
        return bytes_read + bytes_write_fp4 + bytes_write_scales


class quantize_mxfp4_h16_col(_QuantizeMxfp4Base):
    """
    Performance model for primus_flux::quantize_mxfp4_h16_col.

    Column-major MXFP4 quantization: quantizes a BF16 [M, N] tensor to
    MXFP4 along the column dimension (groups of 32 elements along M).

    Roofline calculation -- FLOPs:
        ~2 FLOPs per element (max-abs + scale per group)

    Roofline calculation -- Bytes:
        read  x[M, N] in BF16
        write FP4 packed + E8M0 scales

    Expected Input Dims from trace:
        [[M, N], []]

    Expected Input type from trace:
        [BFloat16, Scalar]
    """

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
