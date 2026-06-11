###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Performance models for RMSNorm pseudo-op extensions.
"""

from TraceLens.PerfModel.perf_model import RMSNorm as CoreRMSNorm
from TraceLens.PerfModel.extensions.perf_model_fallback_warning import (
    warn_perf_model_fallback,
)


class RMSNorm(CoreRMSNorm):
    """Extension-side RMSNorm family reported separately from generic normalization."""

    category = "RMSNorm"
    bwd_category = None
    sheet_category = "RMSNorm"


class aiter_rms_norm(RMSNorm):
    """
    Performance model for aiter::rms_norm.

    Signature: rms_norm(x, weight, variance_epsilon) -> x_normed
        x                — shape [..., hidden_dim], dtype BFloat16
        weight           — shape [hidden_dim],      dtype BFloat16 (affine scale)
        variance_epsilon — float scalar

    Expected Input Dims from trace:
        [x_shape, weight_shape, eps_scalar, ...]
        e.g. [(4, 512), (512,), (), ()]

    Expected Input type from trace:
        [dtype_x, dtype_weight, 'Scalar', ...]

    flops/bytes are inherited from RMSNorm (affine=True, training=False).
    """

    def __init__(self, event, arch=None, python_path=None):
        # Normalization.__init__ calls self.get_param_details and sets all attrs
        super().__init__(event, arch, python_path)

    @staticmethod
    def get_param_details(event):
        op_shape = tuple(event["args"]["Input Dims"][0])
        dtype_in = event["args"]["Input type"][0]
        stride_input = tuple(event["args"]["Input Strides"][0])
        num_channels = event["args"]["Input Dims"][1][0]  # weight.shape[0] = hidden_dim
        return {
            "op_shape": op_shape,
            "dtype_in_out": (dtype_in, None),  # output same dtype as input
            "stride_input": stride_input,
            "stride_output": None,
            "num_channels": num_channels,
            "has_bias": False,
            "is_affine": True,  # weight is always provided
            "is_training": False,
        }


class aiter_rmsnorm(RMSNorm):
    """
    Performance model for aiter::rmsnorm.

    Same roofline as aiter_rms_norm and RMSNorm, but a different bound op and profiler layout:
    explicit out tensor first (see aiter_rms_norm for aiter::rms_norm).

    RMSNorm with separate output tensor.

    Signature: rmsnorm(out, input, weight, epsilon)
        out      — shape [M, N], dtype BFloat16
        input    — shape [M, N], dtype BFloat16
        weight   — shape [N],    dtype BFloat16 (affine scale)
        epsilon  — float scalar

    Expected Input Dims from trace:
        [out_shape, input_shape, weight_shape, eps_scalar]
        e.g. [(4, 7168), (4, 7168), (7168,), ()]

    Expected Input type from trace:
        [dtype_out, dtype_input, dtype_weight, 'Scalar']

    flops/bytes are inherited from RMSNorm (affine=True, training=False).

    get_param_details uses input at index [1] and weight length at [2][0].
    """

    @staticmethod
    def get_param_details(event):
        op_shape = tuple(event["args"]["Input Dims"][1])  # input: [M, N]
        dtype_in = event["args"]["Input type"][1]
        stride_input = tuple(event["args"]["Input Strides"][1])
        num_channels = event["args"]["Input Dims"][2][0]  # weight.shape[0] = N
        return {
            "op_shape": op_shape,
            "dtype_in_out": (dtype_in, None),
            "stride_input": stride_input,
            "stride_output": None,
            "num_channels": num_channels,
            "has_bias": False,
            "is_affine": True,
            "is_training": False,
        }


class aiter_rmsnorm2d_fwd_with_dynamicquant_ck(RMSNorm):
    """
    Performance model for aiter::rmsnorm2d_fwd_with_dynamicquant_ck.

    Fused RMSNorm + per-token FP8 dynamic quantization (CK backend).
    Signature: rmsnorm2d_fwd_with_dynamicquant_ck(out, input, yscale, weight, epsilon, use_model_sensitive_rmsnorm)
        out      — shape [M, N], dtype FP8  (quantized RMSNorm output)
        input    — shape [M, N], dtype BFloat16
        yscale   — shape [M, 1], dtype FP32 (per-token dynamic scale)
        weight   — shape [N],    dtype BFloat16 (affine scale)
        epsilon  — float scalar
        use_model_sensitive_rmsnorm — int scalar

    Expected Input Dims from trace:
        [out_shape, input_shape, yscale_shape, weight_shape, eps_scalar, flag_scalar]
        e.g. [(4, 7168), (4, 7168), (4, 1), (7168,), (), ()]

    FLOPs: RMSNorm (inherited) + per-token quant (2 * num_elems: max-abs + scale).
    Bytes: read input+weight, write out (FP8) + yscale (FP32).
    """

    @staticmethod
    def get_param_details(event):
        op_shape = tuple(event["args"]["Input Dims"][1])  # input: [M, N]
        dtype_in = event["args"]["Input type"][1]  # BFloat16
        stride_input = tuple(event["args"]["Input Strides"][1])
        num_channels = event["args"]["Input Dims"][3][0]  # weight.shape[0] = N
        return {
            "op_shape": op_shape,
            "dtype_in_out": (dtype_in, None),
            "stride_input": stride_input,
            "stride_output": None,
            "num_channels": num_channels,
            "has_bias": False,
            "is_affine": True,
            "is_training": False,
        }

    def flops(self):
        # RMSNorm + per-token quant: max-abs per row + scale each element
        return super().flops() + 2 * self.num_elems

    def bytes(self):
        M = self.num_elems // self.num_channels
        N = self.num_channels
        bytes_read_x = self.num_elems * self.bpe_in  # BF16 input
        bytes_read_weight = N * self.bpe_in  # BF16 weight
        bytes_write_quant = self.num_elems * 1  # FP8 = 1 byte/elem
        bytes_write_scales = M * 1 * 4  # FP32 per-token scales [M, 1]
        return bytes_read_x + bytes_read_weight + bytes_write_quant + bytes_write_scales


class fused_rms_mxfp4_quant(RMSNorm):
    """
    Performance model for aiter.ops.triton.quant.fused_mxfp4_quant.fused_rms_mxfp4_quant
    (surfaced in traces as sglang_profiler::fused_mxfp4_quant_fused_rms_mxfp4_quant).

    Fused (optional residual-add) + RMSNorm on x1, optional separate RMSNorm on x2,
    then MXFP4 quantization on the normalized x1 only.

    Reference implementation:
        aiter/aiter/ops/triton/quant/fused_mxfp4_quant.py:22

    Signature:
        fused_rms_mxfp4_quant(
            x1: Tensor,                       # (M, N1), bf16/fp16
            x1_weight: Tensor,                # (N1,),   bf16/fp16
            x1_epsilon: float,
            x2: Optional[Tensor] = None,      # (M, N2), bf16/fp16
            x2_weight: Optional[Tensor] = None,
            x2_epsilon: float = 0.0,
            res1: Optional[Tensor] = None,    # (M, N1), bf16/fp16
            shuffle: bool = False,
            scale_shuffle_padding: bool = False,
            output_unquantized_inp1: bool = False,
        )
        ->  (out1_fp4, out1_bs),              # (M, N1 // 2) uint8, (M, ceil(N1/32)) uint8
            out1_unquant,                     # (M, N1) bf16/fp16 (optional)
            out2,                             # (M, N2) bf16/fp16 (optional, normalized)
            out_res1                          # (M, N1) bf16/fp16 (optional, x1 + res1)

    The trace only carries shape and dtype info. We detect optional x2 / res1
    presence from the Input Dims list length and rank.

    Roofline -- FLOPs:
        RMSNorm on x1 (inherited)
        + optional residual-add (M * N1)
        + optional RMSNorm on x2 (M * N2 * RMSNorm-cost-per-elem; counted same way)
        + MXFP4 quant of x1 output (~2 * M * N1: max-abs per 32-elem block + scale)

    Roofline -- Bytes:
        read  x1, x1_weight, [res1, x2, x2_weight]
        write out1_fp4 (0.5 B/elem), out1_bs (1/32 B/elem),
              [out_res1, out2] when present

    Compute precision: BF16/FP16 (from x1 dtype). The 4-bit output is materialised
    via quant; bandwidth uses 0.5 B/elem for the FP4 packed tensor.
    """

    def __init__(self, event, arch=None, python_path=None):
        super().__init__(event, arch, python_path)
        input_dims = event["args"]["Input Dims"]
        input_types = event["args"]["Input type"]

        self.has_res1 = False
        self.has_x2 = False
        self.n2 = 0
        # The trace layout follows the positional signature; presence is detected
        # by inspecting tensor ranks (rank-2 -> tensor present; () -> scalar/None).
        # x1 at [0], x1_weight at [1], x1_epsilon scalar at [2], x2 at [3], ...
        if len(input_dims) > 3 and len(input_dims[3]) == 2:
            self.has_x2 = True
            self.n2 = input_dims[3][1]
        if len(input_dims) > 6 and len(input_dims[6]) == 2:
            self.has_res1 = True

    @staticmethod
    def get_param_details(event):
        op_shape = tuple(event["args"]["Input Dims"][0])  # x1: (M, N1)
        dtype_in = event["args"]["Input type"][0]
        stride_input = tuple(event["args"]["Input Strides"][0])
        num_channels = event["args"]["Input Dims"][1][0]  # x1_weight.shape[0] = N1
        return {
            "op_shape": op_shape,
            "dtype_in_out": (dtype_in, None),
            "stride_input": stride_input,
            "stride_output": None,
            "num_channels": num_channels,
            "has_bias": False,
            "is_affine": True,
            "is_training": False,
        }

    def flops(self):
        M = self.num_elems // self.num_channels
        N1 = self.num_channels
        flops = super().flops()
        if self.has_res1:
            flops += M * N1
        if self.has_x2:
            # Approximate x2 RMSNorm cost: same per-elem flop intensity as x1.
            x2_elems = M * self.n2
            flops += int(super().flops() * x2_elems / max(self.num_elems, 1))
        # MXFP4 quant: max-abs per 32-elem block + per-elem scale
        flops += 2 * self.num_elems
        return flops

    def bytes(self):
        M = self.num_elems // self.num_channels
        N1 = self.num_channels
        bytes_read_x1 = self.num_elems * self.bpe_in
        bytes_read_w1 = N1 * self.bpe_in
        bytes_write_fp4 = self.num_elems * 0.5
        bytes_write_scales = (self.num_elems // 32) * 1

        total = bytes_read_x1 + bytes_read_w1 + bytes_write_fp4 + bytes_write_scales
        if self.has_res1:
            total += self.num_elems * self.bpe_in  # read residual
            total += self.num_elems * self.bpe_in  # write updated residual
        if self.has_x2:
            x2_elems = M * self.n2
            total += x2_elems * self.bpe_in  # read x2
            total += self.n2 * self.bpe_in  # read x2_weight
            total += x2_elems * self.bpe_in  # write out2
        return total


class vllm_rocm_aiter_rmsnorm_fp8_group_quant(RMSNorm):
    """
    Performance model for vllm::rocm_aiter_rmsnorm_fp8_group_quant.

    Fused RMSNorm + FP8 per-group quantization.
    Signature: (x_quant, x_quant_scales) = rmsnorm_fp8_group_quant(x, weight, eps, group_size)
        x              — shape [M, N],              dtype BFloat16
        weight         — shape [N],                 dtype BFloat16 (affine scale)
        eps            — float scalar
        group_size     — int scalar (elements per quant group along N)
        x_quant        — shape [M, N],              dtype FP8  (1 byte/elem)
        x_quant_scales — shape [M, N // group_size], dtype FP32 (4 bytes/elem)

    Expected Input Dims from trace:
        [x_shape, weight_shape, eps_scalar, group_size_scalar]
        e.g. [(4, 1536), (1536,), (), ()]

    Concrete Inputs[3] = group_size (e.g. '128')
    """

    def __init__(self, event, arch=None, python_path=None):
        # Normalization.__init__ sets num_elems, num_channels, bpe_in, etc.
        super().__init__(event, arch, python_path)
        self.group_size = int(event["args"]["Concrete Inputs"][3])

    @staticmethod
    def get_param_details(event):
        op_shape = tuple(event["args"]["Input Dims"][0])
        dtype_in = event["args"]["Input type"][0]
        stride_input = tuple(event["args"]["Input Strides"][0])
        num_channels = event["args"]["Input Dims"][1][0]  # weight.shape[0] = N
        return {
            "op_shape": op_shape,
            "dtype_in_out": (dtype_in, None),  # output is FP8, handled in bytes()
            "stride_input": stride_input,
            "stride_output": None,
            "num_channels": num_channels,
            "has_bias": False,
            "is_affine": True,
            "is_training": False,
        }

    def flops(self):
        # RMSNorm flops + quantization: max-abs per group + scale each element
        return super().flops() + 2 * self.num_elems

    def bytes(self):
        M = self.num_elems // self.num_channels
        N = self.num_channels
        num_groups = (N + self.group_size - 1) // self.group_size
        bytes_read_x = self.num_elems * self.bpe_in  # BF16 input
        bytes_read_weight = N * self.bpe_in  # BF16 weight
        bytes_write_quant = self.num_elems * 1  # FP8 = 1 byte/elem
        bytes_write_scales = M * num_groups * 4  # FP32 scales
        return bytes_read_x + bytes_read_weight + bytes_write_quant + bytes_write_scales


class aiter_rmsnorm2d_fwd_with_add_ck(RMSNorm):
    """
    Performance model for aiter::rmsnorm2d_fwd_with_add_ck.

    Fused residual-add + RMSNorm (CK backend).
    Signature: rmsnorm2d_fwd_with_add_ck(out, input, residual_in, residual_out, weight, epsilon, use_model_sensitive_rmsnorm)
        out                        — shape [M, N], dtype BFloat16 (RMSNorm output)
        input                      — shape [M, N], dtype BFloat16
        residual_in                — shape [M, N], dtype BFloat16
        residual_out               — shape [M, N], dtype BFloat16 (input + residual_in)
        weight                     — shape [N],    dtype BFloat16 (affine scale)
        epsilon                    — float scalar
        use_model_sensitive_rmsnorm — int scalar

    Expected Input Dims from trace:
        [out_shape, input_shape, residual_in_shape, residual_out_shape, weight_shape, eps_scalar, flag_scalar]
        e.g. [(4, 7168), (4, 7168), (4, 7168), (4, 7168), (7168,), (), ()]

    FLOPs: residual-add (num_elems) + RMSNorm (inherited from RMSNorm.flops()).
    Bytes: HBM traffic per GPU (read input+residual_in+weight, write out+residual_out).
    """

    @staticmethod
    def get_param_details(event):
        op_shape = tuple(event["args"]["Input Dims"][1])  # input: [M, N]
        dtype_in = event["args"]["Input type"][1]  # BFloat16
        stride_input = tuple(event["args"]["Input Strides"][1])
        num_channels = event["args"]["Input Dims"][4][0]  # weight.shape[0] = N
        return {
            "op_shape": op_shape,
            "dtype_in_out": (dtype_in, None),
            "stride_input": stride_input,
            "stride_output": None,
            "num_channels": num_channels,
            "has_bias": False,
            "is_affine": True,
            "is_training": False,
        }

    def flops(self):
        # residual add: num_elems (input + residual_in -> residual_out)
        return self.num_elems + super().flops()

    def bytes(self):
        N = self.num_channels
        bytes_read = (
            2 * self.num_elems * self.bpe_in + N * self.bpe_in
        )  # input, residual_in, weight
        bytes_write = 2 * self.num_elems * self.bpe_in  # out, residual_out
        return bytes_read + bytes_write


class aiter_add_rmsnorm(aiter_rmsnorm2d_fwd_with_add_ck):
    """
    Performance model for aiter::add_rmsnorm.

    Fused residual-add + RMSNorm (HIP add_rmsnorm).

    Signature: add_rmsnorm(out, input, residual_in, residual_out, weight, epsilon)
        out           — shape [M, N], dtype BFloat16 (RMSNorm output)
        input         — shape [M, N], dtype BFloat16
        residual_in   — shape [M, N], dtype BFloat16
        residual_out  — shape [M, N], dtype BFloat16 (input + residual_in)
        weight        — shape [N],    dtype BFloat16 (affine scale)
        epsilon       — float scalar

    Expected Input Dims from trace:
        [out_shape, input_shape, residual_in_shape, residual_out_shape, weight_shape, eps_scalar]
        e.g. [(4, 7168), (4, 7168), (4, 7168), (4, 7168), (7168,), ()]

    FLOPs: residual-add (num_elems) + RMSNorm (inherited from RMSNorm.flops()).
    Bytes: HBM traffic per GPU (read input+residual_in+weight, write out+residual_out).

    get_param_details, flops, and bytes are inherited from aiter_rmsnorm2d_fwd_with_add_ck.
    """

    pass


class vllm_rocm_aiter_rmsnorm_with_add_fp8_group_quant(RMSNorm):
    """
    Performance model for vllm::rocm_aiter_rmsnorm_with_add_fp8_group_quant.

    Fused residual-add + RMSNorm + FP8 per-group quantization.
    Signature: (x_quant, res, x_quant_scales) = op(x, residual, weight, eps, group_size)
        x              — shape [M, N],               dtype BFloat16
        residual       — shape [M, N],               dtype BFloat16
        weight         — shape [N],                  dtype BFloat16 (affine scale)
        eps            — float scalar
        group_size     — int scalar
        x_quant        — shape [M, N],               dtype FP8  (1 byte/elem)
        res            — shape [M, N],               dtype BFloat16 (x + residual)
        x_quant_scales — shape [M, N // group_size], dtype FP32 (4 bytes/elem)

    Expected Input Dims from trace:
        [x_shape, residual_shape, weight_shape, eps_scalar, group_size_scalar]
        e.g. [(4, 7168), (4, 7168), (7168,), (), ()]

    Concrete Inputs[4] = group_size (e.g. '128')
    """

    def __init__(self, event, arch=None, python_path=None):
        super().__init__(event, arch, python_path)
        self.group_size = int(event["args"]["Concrete Inputs"][4])

    @staticmethod
    def get_param_details(event):
        op_shape = tuple(event["args"]["Input Dims"][0])
        dtype_in = event["args"]["Input type"][0]
        stride_input = tuple(event["args"]["Input Strides"][0])
        num_channels = event["args"]["Input Dims"][2][0]  # weight.shape[0] = N
        return {
            "op_shape": op_shape,
            "dtype_in_out": (dtype_in, None),  # output is FP8, handled in bytes()
            "stride_input": stride_input,
            "stride_output": None,
            "num_channels": num_channels,
            "has_bias": False,
            "is_affine": True,
            "is_training": False,
        }

    def flops(self):
        # residual add: num_elems
        # RMSNorm: super().flops()
        # FP8 quantization: 2 * num_elems (max-abs per group + scale each elem)
        return super().flops() + 3 * self.num_elems

    def bytes(self):
        M = self.num_elems // self.num_channels
        N = self.num_channels
        num_groups = (N + self.group_size - 1) // self.group_size
        bytes_read_x = self.num_elems * self.bpe_in  # BF16 input x
        bytes_read_residual = self.num_elems * self.bpe_in  # BF16 residual (read)
        bytes_read_weight = N * self.bpe_in  # BF16 weight
        bytes_write_quant = self.num_elems * 1  # FP8 = 1 byte/elem
        bytes_write_res = self.num_elems * self.bpe_in  # BF16 updated residual
        bytes_write_scales = M * num_groups * 4  # FP32 scales
        return (
            bytes_read_x
            + bytes_read_residual
            + bytes_read_weight
            + bytes_write_quant
            + bytes_write_res
            + bytes_write_scales
        )


class vllm_rocm_aiter_triton_add_rmsnorm_pad(RMSNorm):
    """
    Performance model for vllm::rocm_aiter_triton_add_rmsnorm_pad.

    Fused residual-add + RMSNorm + zero-pad on hidden dim. The compiler fusion
    pass replaces separate add + RMSNorm + pad nodes with this single Triton kernel.
    Padding aligns hidden_dim to a multiple of x_pad_to_multiple for downstream
    AITER MoE GEMMs.

    Signature: rocm_aiter_triton_add_rmsnorm_pad(x, weight, variance_epsilon,
               residual, x_pad_to_multiple) -> (out, residual_out)
        x                  — shape [M, N],     dtype BFloat16
        weight             — shape [N],        dtype BFloat16  (affine RMSNorm scale)
        variance_epsilon   — float scalar      (e.g. 1e-5)
        residual           — shape [M, N],     dtype BFloat16
        x_pad_to_multiple  — int scalar        (e.g. 256)
        out                — shape [M, N_out], dtype BFloat16  (N_out = ceil(N / pad) * pad)
        residual_out       — shape [M, N],     dtype BFloat16  (x + residual, pre-norm)

    Expected Input Dims from trace:
        [x_shape, weight_shape, eps_scalar, residual_shape, pad_scalar]
        e.g. [(64, 2880), (2880,), (), (64, 2880), ()]

    Expected Input type from trace:
        ['c10::BFloat16', 'c10::BFloat16', 'Scalar', 'c10::BFloat16', 'Scalar']

    Concrete Inputs[2] = variance_epsilon (e.g. '1e-05')
    Concrete Inputs[4] = x_pad_to_multiple (e.g. '256')

    Roofline -- FLOPs:
        residual_add + rmsnorm (inherited from RMSNorm.flops() + num_elems for add).

    Roofline -- bytes moved:
        Read x + residual + weight; write padded output + updated residual.
    """

    def __init__(self, event, arch=None, python_path=None):
        self.x_pad_to_multiple = int(event["args"]["Concrete Inputs"][4])
        super().__init__(event, arch, python_path)

        N = self.num_channels
        if self.x_pad_to_multiple > 0:
            self.n_out = (
                (N + self.x_pad_to_multiple - 1)
                // self.x_pad_to_multiple
                * self.x_pad_to_multiple
            )
        else:
            self.n_out = N

    @staticmethod
    def get_param_details(event):
        op_shape = tuple(event["args"]["Input Dims"][0])
        dtype_in = event["args"]["Input type"][0]
        stride_input = tuple(event["args"]["Input Strides"][0])
        num_channels = event["args"]["Input Dims"][1][0]
        return {
            "op_shape": op_shape,
            "dtype_in_out": (dtype_in, None),
            "stride_input": stride_input,
            "stride_output": None,
            "num_channels": num_channels,
            "has_bias": False,
            "is_affine": True,
            "is_training": False,
        }

    def flops(self):
        return self.num_elems + super().flops()

    def bytes(self):
        M = self.num_elems // self.num_channels
        N = self.num_channels
        bytes_read = 2 * self.num_elems * self.bpe_in + N * self.bpe_in
        bytes_write_out = M * self.n_out * self.bpe_in
        bytes_write_res = M * N * self.bpe_in
        return bytes_read + bytes_write_out + bytes_write_res


def _rmsnorm_graph_param_details(event, op_shape_index=0):
    """Build RMSNorm param_details from a minimal graph-replay Input Dims layout."""
    dims = event["args"]["Input Dims"]
    op_shape = tuple(dims[op_shape_index])
    types = event["args"].get("Input type") or []
    dtype_in = types[op_shape_index] if len(types) > op_shape_index else "c10::Half"
    if len(types) <= op_shape_index:
        warn_perf_model_fallback(
            "aiter_rmsnorm_graph (rmsnorm_sumsq/apply_serial)",
            f"Input type entry missing at index {op_shape_index}; assuming c10::Half.",
        )
    if dtype_in in ("ScalarList", "Scalar", ""):
        warn_perf_model_fallback(
            "aiter_rmsnorm_graph (rmsnorm_sumsq/apply_serial)",
            f"Input type was {dtype_in!r}; assuming c10::Half for roofline.",
        )
        dtype_in = "c10::Half"
    strides = event["args"].get("Input Strides") or []
    stride_input = (
        tuple(strides[op_shape_index]) if len(strides) > op_shape_index else ()
    )
    weight_idx = 1 if len(dims) > 1 and isinstance(dims[1], (list, tuple)) else None
    num_channels = (
        dims[weight_idx][0]
        if weight_idx is not None and len(dims[weight_idx]) == 1
        else op_shape[-1]
    )
    return {
        "op_shape": op_shape,
        "dtype_in_out": (dtype_in, None),
        "stride_input": stride_input,
        "stride_output": None,
        "num_channels": num_channels,
        "has_bias": False,
        "is_affine": weight_idx is not None,
        "is_training": False,
    }


class aiter_rmsnorm_sumsq_serial(RMSNorm):
    """Graph-replay ``rmsnorm_sumsq_kernel_serial`` (reduction pass)."""

    @staticmethod
    def get_param_details(event):
        return _rmsnorm_graph_param_details(event)

    def flops(self):
        return self.num_elems * 2

    def bytes(self):
        return self.num_elems * self.bpe_in


class aiter_rmsnorm_apply_serial(RMSNorm):
    """Graph-replay ``rmsnorm_apply_kernel_serial`` (normalize + scale pass)."""

    @staticmethod
    def get_param_details(event):
        return _rmsnorm_graph_param_details(event)

    def flops(self):
        return self.num_elems * (3 if self.is_affine else 2)

    def bytes(self):
        read_bytes = self.num_elems * self.bpe_in
        write_bytes = self.num_elems * self.bpe_in
        weight_bytes = self.num_channels * self.bpe_in if self.is_affine else 0
        return read_bytes + write_bytes + weight_bytes


class aiter_add_rmsnorm_quant_graph(RMSNorm):
    """
    Graph-replay ``add_rmsnorm_quant_kernel`` (fused add + RMSNorm + FP8 group quant).
    """

    def __init__(self, event, arch=None, python_path=None):
        args = event.get("args", {})
        group_size = args.get("_inferred_group_size")
        if group_size is None:
            concrete = args.get("Concrete Inputs") or []
            if len(concrete) > 4 and concrete[4]:
                group_size = int(concrete[4])
            else:
                warn_perf_model_fallback(
                    "aiter_add_rmsnorm_quant_graph",
                    "group_size not in _inferred_group_size or Concrete Inputs[4]; "
                    "using default group_size=128 for quant byte accounting.",
                )
                group_size = 128
        super().__init__(event, arch, python_path)
        self.group_size = int(group_size)

    @staticmethod
    def get_param_details(event):
        dims = event["args"]["Input Dims"]
        op_shape = tuple(dims[0])
        types = event["args"].get("Input type") or []
        dtype_in = types[0] if types else "c10::Half"
        if dtype_in in ("ScalarList", "Scalar", ""):
            dtype_in = "c10::Half"
        weight_idx = 2 if len(dims) > 2 and isinstance(dims[2], (list, tuple)) else 1
        num_channels = (
            dims[weight_idx][0] if len(dims[weight_idx]) == 1 else op_shape[-1]
        )
        return {
            "op_shape": op_shape,
            "dtype_in_out": (dtype_in, None),
            "stride_input": (),
            "stride_output": None,
            "num_channels": num_channels,
            "has_bias": False,
            "is_affine": True,
            "is_training": False,
        }

    def flops(self):
        return self.num_elems + super().flops()

    def bytes(self):
        M = self.num_elems // self.num_channels
        N = self.num_channels
        num_groups = (N + self.group_size - 1) // self.group_size
        bytes_read = 2 * self.num_elems * self.bpe_in + N * self.bpe_in
        bytes_write = (
            self.num_elems * 1 + M * num_groups * 4 + self.num_elems * self.bpe_in
        )
        return bytes_read + bytes_write
