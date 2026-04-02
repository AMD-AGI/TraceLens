###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Performance models for RMSNorm pseudo-op extensions.
"""

from TraceLens.PerfModel.perf_model import RMSNorm


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
            "is_affine": True,   # weight is always provided
            "is_training": False,
        }


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
        bytes_read_x      = self.num_elems * self.bpe_in   # BF16 input
        bytes_read_weight = N * self.bpe_in                # BF16 weight
        bytes_write_quant  = self.num_elems * 1            # FP8 = 1 byte/elem
        bytes_write_scales = M * num_groups * 4            # FP32 scales
        return bytes_read_x + bytes_read_weight + bytes_write_quant + bytes_write_scales


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
        bytes_read_x        = self.num_elems * self.bpe_in   # BF16 input x
        bytes_read_residual = self.num_elems * self.bpe_in   # BF16 residual (read)
        bytes_read_weight   = N * self.bpe_in                # BF16 weight
        bytes_write_quant   = self.num_elems * 1             # FP8 = 1 byte/elem
        bytes_write_res     = self.num_elems * self.bpe_in   # BF16 updated residual
        bytes_write_scales  = M * num_groups * 4             # FP32 scales
        return (bytes_read_x + bytes_read_residual + bytes_read_weight
                + bytes_write_quant + bytes_write_res + bytes_write_scales)
