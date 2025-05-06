# MIT License

# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from math import prod
import math
import os
import re

from .kernel_name_parser import gemm_name_parser

def name2bpe(name):
    """
    This function maps a data type name to the number of bytes per element.
    Args:
        name (str): The name of the data type.
    Returns:
        int: The number of bytes per element.
    """
    dict_bpe2dtype = {
        8: ['double', 'long int'],
        4: ['float', 'scalar'],
        2: ['c10::half', 'c10::bfloat16'],
        1: ['c10::float8_e4m3fnuz'],
    }
    dict_dtype2bpe = {dtype: bpe for bpe, dtypes in dict_bpe2dtype.items() for dtype in dtypes}
    return dict_dtype2bpe.get(name.lower(), None)

def is_tensortype(dtype):
    """
    This function checks if a data type is a tensor type.
    Args:
        dtype (str): The name of the data type.
    Returns:
        bool: True if the data type is a tensor type, False if not. If the data type is not recognized, None is returned.
    """
    if dtype.lower() in ['float', 'double', 'c10::half', 'c10::bfloat16', 'c10::float8_e4m3fnuz']:
        return True
    elif dtype.lower() in ['long int', 'scalar']:
        return False

# 1. GEMM
class GEMM:
    """
    This is the base class for all GEMM operations.
    If you want to add a new GEMM operation, you should inherit from this class.
    """
    cache_gemm_results = {}  # This is used to cache gemm results
    def __init__(self, event, arch=None, detail_level=0):
        self.event = event
        # parse kernel info (e.g. transpose) before kernel params since it can be needed
        self.parsed_kernel_info = None
        for kernel_name in event['kernel_names']:
            # TODO: think you really wanna pass around dicts instead of objects?
            self.parsed_kernel_info = gemm_name_parser(kernel_name)
            if self.parsed_kernel_info is not None:
                break
        self.param_details = self.get_param_details(event)
       
        if self.parsed_kernel_info is not None:
            self.param_details['transpose'] = self.parsed_kernel_info['transpose']

        self.M, self.N, self.K = self.param_details['M'], self.param_details['N'], self.param_details['K']
        self.bias = self.param_details['bias']

        if detail_level > 0:
            if arch is None:
                raise ValueError("arch must be provided if detail_level > 0")
            if self.parsed_kernel_info is None:
                raise ValueError("parsed_kernel_info must be provided if detail_level > 0")
            self.param_details['mt_m'] = self.parsed_kernel_info['mt_m']
            self.param_details['mt_n'] = self.parsed_kernel_info['mt_n']
            self.param_details['depth_u'] = self.parsed_kernel_info['depth_u']
            dim_eff_info = self.dim_efficiency(arch)
            self.param_details.update(dim_eff_info)

    @staticmethod
    def get_param_details(event):
        # to be implemented in the child class
        raise NotImplementedError

    @staticmethod
    def flops_func(M, N, K, bias):
        flops_matmul = 2 * M * N * K
        flops_bias = M * N if bias else 0
        return flops_matmul + flops_bias

    def flops(self):
        return self.flops_func(self.M, self.N, self.K, self.bias)

    @staticmethod
    def bytes_func(M, N, K, bias, bpe_mat1, bpe_mat2, bpe_bias, bpe_output):
        #if any of the bpe is None, we will return None
        if None in {bpe_mat1, bpe_mat2, bpe_bias, bpe_output}:
            return None
        bytes_mat1 = M * K * bpe_mat1
        bytes_mat2 = K * N * bpe_mat2
        bytes_output = M * N * bpe_output
        # to be totally accurate we should use the bias shape from profile info
        # but we just assume bias shape as 1xN
        #TODO: use profile info to get the bias shape
        bytes_bias = (N if bias else 0) * bpe_bias
        return bytes_mat1 + bytes_mat2 + bytes_output + bytes_bias
    def bytes(self, bpe_mat1, bpe_mat2, bpe_bias, bpe_output):
        return self.bytes_func(self.M, self.N, self.K, self.bias, bpe_mat1, bpe_mat2, bpe_bias, bpe_output)

    """
    bwd pass for Y = X.matmul(W^T) + B
    X_grad = Y_grad.matmul(W)
    W_grad = Y_grad^T.matmul(X)
    B_grad = Y_grad.sum(dim=0)
    """

    def flops_bwd(self):
        flops_input_grad = self.flops_func(M=self.M, N=self.K, K=self.N, bias=False)
        flops_weight_grad = self.flops_func(M=self.N, N=self.K, K=self.M, bias=False)
        flops_bias_grad = self.M * self.N if self.bias else 0
        return flops_input_grad + flops_weight_grad + flops_bias_grad

    def bytes_bwd(self, bytes_per_element):
        bytes_input_grad = self.bytes_func(M=self.M, N=self.K, K=self.N, bias=False, bytes_per_element=bytes_per_element)
        bytes_weight_grad = self.bytes_func(M=self.N, N=self.K, K=self.M, bias=False, bytes_per_element=bytes_per_element)
        bytes_bias_grad = self.M * self.N if self.bias else 0
        return bytes_input_grad + bytes_weight_grad + bytes_bias_grad

    @staticmethod
    def get_simulated_time(M, N, K, arch, param_details, tile_eff, wq_eff):
        # Create a unique key for the cache based on dimensions and architecture
        cache_key = (M, N, K, arch['name'], arch['freq_mhz'])

        # Calculate ideal time
        ideal_ops_per_cycle_simd = { 4 : 256, 2 : 2048, 1 : 4096 }
        ideal_time = 2 * M * N * K / (arch['num_cus'] * arch['gemm_units_per_cu'] *\
                                      ideal_ops_per_cycle_simd[name2bpe(param_details['dtype_A_B'][0])]) \
                                      / arch['freq_mhz']

        # Check if the result is already in the cache
        gemm_cache = GEMM.cache_gemm_results
        if cache_key in GEMM.cache_gemm_results:
            simulated_time = GEMM.cache_gemm_results[cache_key]
        else:
            if os.path.isdir('gemmologist'):
                python_path = '.venv3.10\\Scripts\\python.exe'
                cmd = 'cd gemmologist && %s bin/gemmologist.py -m %d -n %d -k %d -d 1 -a %s --freq_mhz %d --topn 1' % \
                      (python_path, M, N, K, arch['name'], arch['freq_mhz'])
                log = re.findall("Time=\\d+\\.\\d+", os.popen(cmd).read())

                if len(log) > 0:
                    simulated_time = float(re.sub("Time=", "", str(log[0])))
                    # Store the result in the cache
                    GEMM.cache_gemm_results[cache_key] = simulated_time
                else:
                    raise AssertionError("Not able to simulate in gemmologist")
            else:
                simulated_time = ideal_time * tile_eff * wq_eff

        dim_eff = ideal_time / simulated_time
        return simulated_time, dim_eff

    @staticmethod
    def dim_efficiency_func(arch, param_details, M, N, K, mt_m, mt_n, depth_u):
        """
        args:
        num_cus: number of compute units (CUs) aka Streaming Multiprocessors (SMs)
        M: M dimension of the matrix multiplication passed to the BLAS library
        N: N dimension of the matrix multiplication passed to the BLAS library
        K: K dimension of the matrix multiplication passed to the BLAS library
        mt_m: macro tile size in M dimension
        mt_n: macro tile size in N dimension
        depth_u: depth tile size
        """
        # Tile quantization
        M_pad = math.ceil(M / mt_m) * mt_m
        N_pad = math.ceil(N / mt_n) * mt_n
        tile_eff = (M * N) / (M_pad * N_pad)

        # Wave quantization
        num_blocks = M_pad * N_pad // (mt_m * mt_n)
        num_rounds = math.ceil(num_blocks / arch['num_cus'])
        wq_eff = num_blocks / (num_rounds * arch['num_cus'])

        simulated_time, dim_eff = GEMM.get_simulated_time(M, N, K, arch, param_details, tile_eff, wq_eff)

        return {
            'num_tiles': num_blocks,
            'tile_eff': tile_eff,
            'wq_eff': wq_eff,
            'dim_eff': dim_eff,
            'simulated_time': simulated_time
        }

    def dim_efficiency(self, arch_dict):
        """
        args:
        arch_dict: dictionary with the architecture information
        """
        # blas library swaps M and N from torch
        M, N = self.N, self.M
        K = self.K
        mt_m = self.parsed_kernel_info['mt_m']
        mt_n = self.parsed_kernel_info['mt_n']
        depth_u = self.parsed_kernel_info['depth_u']
        return self.dim_efficiency_func(arch_dict, self.param_details, M, N, K, mt_m, mt_n, depth_u)

class aten_mm(GEMM):
    """
    aten::mm the matrix multiplication primitive in PyTorch
    A.matmul(B)
    """
    @staticmethod
    def get_param_details(event):
        input_dims = event['args']['Input Dims']
        A_shape, B_shape = input_dims[0], input_dims[1]
        M = A_shape[0]
        N = B_shape[1]
        K = A_shape[1]

        dtype_A_B = tuple(event['args']['Input type'][:2])
        try:
            stride_A = tuple(event['args']['Input Strides'][0])
            stride_B = tuple(event['args']['Input Strides'][1])
        except KeyError:
            stride_A = stride_B = None

        return {"M": M, "N": N, "K": K, "bias": False,
                "stride_A": stride_A, "stride_B": stride_B,
                "dtype_A_B": dtype_A_B}

    def bytes(self):
        dtype_A_B = self.param_details['dtype_A_B']
        if dtype_A_B[0] != dtype_A_B[1]:
            raise ValueError(f"Data types of A and B are different: {dtype_A_B}")
        self.bpe = name2bpe(dtype_A_B[0])
        return super().bytes(bpe_mat1=self.bpe, bpe_mat2=self.bpe,
                             bpe_bias=self.bpe, # does not matter
                             bpe_output=self.bpe) # out dtype is not always provided. #TODO: use out dtype if provided
    def flops_bwd(self):
        raise NotImplementedError("Backward pass for aten::mm is not defined.")
    def bytes_bwd(self, bytes_per_element):
        raise NotImplementedError("Backward pass for aten::mm is not defined.")


class aten_addmm(GEMM):
    """
    aten::addmm is the A.matmul(B) + C operation in PyTorch
    """
    @staticmethod
    def get_param_details(event):
        input_dims = event['args']['Input Dims']
        C_shape, A_shape, B_shape = input_dims[0], input_dims[1], input_dims[2]
        M = A_shape[0]
        N = B_shape[1]
        K = A_shape[1]

        dtype_A_B = tuple(event['args']['Input type'][1:3])
        try:
            stride_A = tuple(event['args']['Input Strides'][1])
            stride_B = tuple(event['args']['Input Strides'][2])
        except KeyError:
            stride_A = stride_B = None

        return {"M": M, "N": N, "K": K, "bias": True,
                "stride_A": stride_A, "stride_B": stride_B,
                "dtype_A_B": dtype_A_B}

    def bytes(self):
        dtype_A_B = self.param_details['dtype_A_B']
        if dtype_A_B[0] != dtype_A_B[1]:
            raise ValueError(f"Data types of A and B are different: {dtype_A_B}")
        self.bpe = name2bpe(dtype_A_B[0])
        # setting bias bpe to be the same as the input matrices is not totally correct
        # TODO: correct later
        # TODO: similar to aten_mm, we need to use the output dtype if provided
        return super().bytes(bpe_mat1=self.bpe, bpe_mat2=self.bpe,
                             bpe_bias=self.bpe,
                             bpe_output=self.bpe)

    def flops_bwd(self):
        raise NotImplementedError("Backward pass for aten::addmm is not defined.")
    def bytes_bwd(self, bytes_per_element):
        raise NotImplementedError("Backward pass for aten::addmm is not defined.")

class aten_scaled_mm(GEMM):
    """
    aten::scaled_mm is the scale_result(scale_a*A.matmul(scale_b*B) + bias)
    """
    @staticmethod
    def get_param_details(event):
        # ref: https://pytorch.org/cppdocs/api/function_namespaceat_1a2902105d8aed3fa448a0da42f90e2cbf.html
        input_dims = event['args']['Input Dims']
        A_shape, B_shape = input_dims[0], input_dims[1]
        M = A_shape[0]
        N = B_shape[1]
        K = A_shape[1]
        bias = len(input_dims) == 3

        dtype_A_B = tuple(event['args']['Input type'][:2])
        try:
            stride_A = tuple(event['args']['Input Strides'][0])
            stride_B = tuple(event['args']['Input Strides'][1])
        except KeyError:
            stride_A = stride_B = None

        return {"M": M, "N": N, "K": K, "bias": bias,
                "stride_A": stride_A, "stride_B": stride_B,
                "dtype_A_B": dtype_A_B}

    def bytes(self):
        dtype_A_B = self.param_details['dtype_A_B']
        if dtype_A_B[0] != dtype_A_B[1]:
            raise ValueError(f"Data types of A and B are different: {dtype_A_B}")
        self.bpe = name2bpe(dtype_A_B[0])
        # assumption:
        # for fp8 the output dtype is fp16
        # for fp16, bf16, fp32 the output dtype is the same as the input dtype
        if self.bpe == 1:
            out_bpe = 2
        elif self.bpe in [2, 4]:
            out_bpe = self.bpe
        else:
            out_bpe = None
        return super().bytes(bpe_mat1=self.bpe, bpe_mat2=self.bpe,
                             bpe_bias=self.bpe, # does not matter
                             bpe_output=out_bpe)

    def flops_bwd(self):
        raise NotImplementedError("Backward pass for aten::addmm is not defined.")
    def bytes_bwd(self, bytes_per_element):
        raise NotImplementedError("Backward pass for aten::addmm is not defined.")

class aten_bmm(GEMM):
    """
    aten::bmm — batch matrix multiplication
    (B, M, K) × (B, K, N) → (B, M, N)
    Inherits FLOP/byte analytics from GEMM and scales them by the batch size.
    """

    @staticmethod
    def get_param_details(event):
        """Extract B, M, N, K and metadata from the profiler event."""
        input_dims = event['args']['Input Dims']
        A_shape, B_shape = input_dims[0], input_dims[1]

        B_dim, M, K = A_shape  # (B, M, K)
        _,      _, N = B_shape # (B, K, N)

        dtype_A_B = tuple(event['args']['Input type'][:2])
        try:
            stride_A = tuple(event['args']['Input Strides'][0])
            stride_B = tuple(event['args']['Input Strides'][1])
        except KeyError:
            stride_A = stride_B = None

        return {
            "B": B_dim,
            "M": M,
            "N": N,
            "K": K,
            "bias": False,            # aten::bmm has no implicit bias term
            "stride_A": stride_A,
            "stride_B": stride_B,
            "dtype_A_B": dtype_A_B,
        }

    # ---------------------- FLOPs / Bytes ----------------------
    def flops(self):
        """Total FLOPs for the entire batch."""
        return self.param_details["B"] * super().flops()

    def bytes(self):
        """Total DRAM traffic for the entire batch (read+write)."""
        dtype_A_B = self.param_details['dtype_A_B']
        if dtype_A_B[0] != dtype_A_B[1]:
            raise ValueError(f"Data types of A and B are different: {dtype_A_B}")

        bpe = name2bpe(dtype_A_B[0])
        per_batch = super().bytes(bpe_mat1=bpe, bpe_mat2=bpe,
                                   bpe_bias=bpe,   # not used, but keeps call signature
                                   bpe_output=bpe)
        return None if per_batch is None else self.param_details['B'] * per_batch

    # ---------------------- Backward placeholders ----------------------
    def flops_bwd(self):
        raise NotImplementedError("Backward pass for aten::bmm is not defined.")

    def bytes_bwd(self, bytes_per_element):
        raise NotImplementedError("Backward pass for aten::bmm is not defined.")

class aten_baddbmm(GEMM):
    """
    aten::baddbmm — batch matrix multiplication with bias
    (B, M, K) × (B, K, N) + (B, M, N) → (B, M, N)
    Inherits FLOP/byte analytics from GEMM and scales them by the batch size.
    """
    @staticmethod
    def get_param_details(event):
        """Extract B, M, N, K and metadata from the profiler event."""
        input_dims = event['args']['Input Dims']
        C_shape, A_shape, B_shape = input_dims[0], input_dims[1], input_dims[2]

        B_dim, M, K = A_shape  # (B, M, K)
        _,      _, N = B_shape # (B, K, N)

        dtype_A_B = tuple(event['args']['Input type'][1:3])
        try:
            stride_A = tuple(event['args']['Input Strides'][1])
            stride_B = tuple(event['args']['Input Strides'][2])
        except KeyError:
            stride_A = stride_B = None

        return {
            "B": B_dim,
            "M": M,
            "N": N,
            "K": K,
            "bias": True,
            "stride_A": stride_A,
            "stride_B": stride_B,
            "dtype_A_B": dtype_A_B,
        }
    # ---------------------- FLOPs / Bytes ----------------------
    def flops(self):
        """Total FLOPs for the entire batch."""
        return self.param_details["B"] * super().flops()
    def bytes(self):
        """Total DRAM traffic for the entire batch (read+write)."""
        dtype_A_B = self.param_details['dtype_A_B']
        if dtype_A_B[0] != dtype_A_B[1]:
            raise ValueError(f"Data types of A and B are different: {dtype_A_B}")

        bpe = name2bpe(dtype_A_B[0])
        per_batch = super().bytes(bpe_mat1=bpe, bpe_mat2=bpe,
                                   bpe_bias=bpe,   # not used, but keeps call signature
                                   bpe_output=bpe)
        return None if per_batch is None else self.param_details['B'] * per_batch
    # ---------------------- Backward placeholders ----------------------
    def flops_bwd(self):
        raise NotImplementedError("Backward pass for aten::baddbmm is not defined.")
    def bytes_bwd(self, bytes_per_element):
        raise NotImplementedError("Backward pass for aten::baddbmm is not defined.")
    
# TODO: maybe deprecate aten linear as it will call aten::mm or aten::addmm
class aten_linear(GEMM):

    @staticmethod
    def get_param_details(event):
        input_dims = event['args']['Input Dims']
        input_shape = input_dims[0]
        weight_shape = input_dims[1]
        bias = bool(input_dims[2])
        K = input_shape[-1]
        N = weight_shape[0]
        # Compute M as the product of all dimensions except the last one
        M = 1
        for dim in input_shape[:-1]:
            M *= dim

        # TODO: remove repeated code, this is not cool
        dtype_A_B = tuple(event['args']['Input type'][:2])
        try:
            stride_A = tuple(event['args']['Input Strides'][0])
            stride_B = tuple(event['args']['Input Strides'][1])
        except KeyError:
            stride_A = stride_B = None

        return {"M": M, "N": N, "K": K, "bias": bias,
                "stride_A": stride_A, "stride_B": stride_B,
                "dtype_A_B": dtype_A_B}

class tex_ts_te_gemm_ts(GEMM):
    """
    tex_ts::te_gemm_ts is a matmul op in TransformerEngine

    https://github.com/ROCm/TransformerEngine/blob/e9772d4d18b2980e8e0643c94591a94cad9bb8b7/transformer_engine/pytorch/csrc/ts_fp8_op.cpp#L244
    https://github.com/ROCm/TransformerEngine/blob/e9772d4d18b2980e8e0643c94591a94cad9bb8b7/transformer_engine/pytorch/csrc/extensions/gemm.cpp#L10

    """

    def __init__(self, event, arch=None, detail_level=0):
        super().__init__(event, arch, detail_level)

    def get_param_details(self, event):
        input_dims = event['args']['Input Dims']
        C_shape, A_shape, B_shape = input_dims[10], input_dims[0], input_dims[5]

        # index 4 and 9 are transa and transb respectively
        # https://github.com/ROCm/TransformerEngine/blob/e9772d4d18b2980e8e0643c94591a94cad9bb8b7/transformer_engine/pytorch/cpp_extensions/gemm.py#L248
        concrete_inputs = event['args']['Concrete Inputs']
        trans_a = concrete_inputs[4] == '1'
        trans_b = concrete_inputs[9] == '1'


        # https://github.com/ROCm/TransformerEngine/blob/e9772d4d18b2980e8e0643c94591a94cad9bb8b7/transformer_engine/common/gemm/cublaslt_gemm.cu#L330C17-L330C23
        if trans_a:
            M = A_shape[0]
            K = A_shape[1]
        else:
            M = A_shape[1]
            K = A_shape[0]

        if trans_b:
            N = B_shape[1]
        else:
            N = B_shape[0]

        bias_term = event['args']['Concrete Inputs'][14]

        if bias_term == '':
            bias = False
        else:
            bias = True

        # dtype A, B, bias
        dtype_A_B = (event['args']['Input type'][0], event['args']['Input type'][5], event['args']['Input type'][10])

        try:
            stride_A = tuple(event['args']['Input Strides'][0])
            stride_B = tuple(event['args']['Input Strides'][5])
        except KeyError:
            stride_A = stride_B = None

        return {"M": M, "N": N, "K": K, "bias": bias,
                "stride_A": stride_A, "stride_B": stride_B,
                "dtype_A_B": dtype_A_B}

    def bytes(self):
        dtype_A_B = self.param_details['dtype_A_B']
      
        self.bpe_mat1 = name2bpe(dtype_A_B[0])
        self.bpe_mat2 = name2bpe(dtype_A_B[1])
        self.bpe_bias = name2bpe(dtype_A_B[2])

        # assume output dtype lowest of inputs, ignore scalars alpha and beta for now
        # TODO: correct later if better way found
        self.bpe_output = min(self.bpe_mat1, self.bpe_mat2, self.bpe_bias)

        return super().bytes(bpe_mat1=self.bpe_mat1, bpe_mat2=self.bpe_mat2,
                             bpe_bias=self.bpe_bias,
                             bpe_output=self.bpe_output)

    def flops_bwd(self):
        raise NotImplementedError("Backward pass for tex_ts::te_gemm_ts is not defined.")
    def bytes_bwd(self, bytes_per_element):
        raise NotImplementedError("Backward pass for tex_ts::te_gemm_ts is not defined.")


# 2. Convolution
class CONV:
    # Conv perf model is based on: https://github.com/pytorch/pytorch/blob/main/torch/utils/flop_counter.py
    # we will make stuff reusiable across conv1d, conv2d, and conv3d
    def __init__(self, event, arch=None, detail_level=0):
        self.event = event
        self.param_details = self.get_param_details(event)
        self.x_shape, self.w_shape = self.param_details['input_shape'], self.param_details['filter_shape']
        self.stride, self.padding, self.dilation, self.groups = (self.param_details[key] for key in ['stride', 'padding', 'dilation', 'groups'])
        self.bias = self.param_details['bias']
        self.transposed_conv = self.param_details['transposed_conv']
        self.output_padding = self.param_details['output_padding'] if self.transposed_conv else None
        self.out_shape = CONV.get_output_shape(self.x_shape, self.w_shape, self.stride, self.padding, self.dilation, self.transposed_conv, self.output_padding)

    @staticmethod
    def get_output_shape(input_shape, filter_shape, stride, padding, dilation, transposed_conv, output_padding):
        x_spatial_shape, w_spatial_shape = input_shape[2:], filter_shape[2:]
        conv_ndims = len(x_spatial_shape)
        spatial_out_fn = CONV.get_conv_out_dim if not transposed_conv else CONV.get_transposed_conv_out_dim
        out_filters = filter_shape[0] if not transposed_conv else filter_shape[1]

        if not transposed_conv:
            output_padding = (None,) * conv_ndims
        out_spatial_shape = tuple(spatial_out_fn(x_spatial_shape[i], w_spatial_shape[i],
                                                 stride[i], padding[i], dilation[i], output_padding[i]) for i in range(conv_ndims))
        return (input_shape[0], out_filters) + tuple(out_spatial_shape)

    @staticmethod
    def t(shape):
        return (shape[1], shape[0]) + shape[2:]

    @staticmethod
    def get_conv_out_dim(input_dim, kernel_size, stride, padding, dilation, output_padding=None):
        return int(((input_dim + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1)

    @staticmethod
    def get_transposed_conv_out_dim(input_dim, kernel_size, stride, padding, dilation, output_padding):
        return (input_dim - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1

    @staticmethod
    def flops_func(x_shape, w_shape, out_shape, bias, transposed_conv=False):
        # c_in =filter[1] already accounts for grouped convolutions
        flops_per_element = 2 * prod(w_shape[1:])
        if transposed_conv:
            flops_conv = prod(x_shape) * flops_per_element
        else:
            flops_conv = prod(out_shape) * flops_per_element
        flops_bias = prod(out_shape) if bias else 0
        return flops_conv + flops_bias
    def flops(self):
        return self.flops_func(self.x_shape, self.w_shape, self.out_shape,
                                self.bias, self.transposed_conv)

    @staticmethod
    # we assume same bytes per element for all tensors
    # TODO: make it more general later
    def bytes_func(x_shape, w_shape, out_shape, bias, bytes_per_element):
        if bytes_per_element is None:
            return None
        elems_input_read = prod(x_shape)
        elems_weight_read = prod(w_shape)
        elems_bias_read = out_shape[1] if bias else 0
        elems_output_write = prod(out_shape)
        total_elems_moved = elems_input_read + elems_weight_read + elems_bias_read + elems_output_write
        return total_elems_moved * bytes_per_element
    def bytes(self, bytes_per_element):
        return self.bytes_func(self.x_shape, self.w_shape, self.out_shape, self.bias, bytes_per_element)

    @staticmethod
    def flops_bwd_func(out_shape, x_shape, w_shape, bias, transposed_conv=False):
        flops_input_grad = CONV.flops_func(out_shape, w_shape, x_shape, False, not transposed_conv)
        if not transposed_conv:
            flops_weight_grad = CONV.flops_func(CONV.t(x_shape), CONV.t(out_shape), CONV.t(w_shape), False, False)
        else:
            flops_weight_grad = CONV.flops_func(CONV.t(out_shape), CONV.t(x_shape), CONV.t(w_shape), False, False)

        flops_bias_grad = prod(out_shape) if bias else 0
        return flops_input_grad + flops_weight_grad + flops_bias_grad
    def flops_bwd(self):
        return self.flops_bwd_func(self.out_shape, self.x_shape, self.w_shape, self.bias, self.transposed_conv)

    @staticmethod
    def bytes_bwd_func(x_shape, w_shape, out_shape, bias, bytes_per_element):
        if bytes_per_element is None:
            return None
        bytes_input_grad = CONV.bytes_func(out_shape, w_shape, x_shape, False, bytes_per_element)
        bytes_weight_grad = CONV.bytes_func(out_shape, x_shape, w_shape, False, bytes_per_element)
        # for bias we read the output gradient and write the bias gradient
        bytes_bias_grad = prod(out_shape) + out_shape[1] if bias else 0
        return bytes_input_grad + bytes_weight_grad + bytes_bias_grad
    def bytes_bwd(self, bytes_per_element):
        return self.bytes_bwd_func(self.x_shape, self.w_shape, self.out_shape, self.bias, bytes_per_element)

    @staticmethod
    def get_param_details(event):
        # to be implemented in the child class
        raise NotImplementedError

class aten_conv(CONV):

    @staticmethod
    def str_to_tuple(s):
        return tuple(int(x) for x in s[1:-1].split(','))
    @staticmethod
    def get_param_details(event):
        # 0 input tensor
        # 1 weight tensor
        # 2 bias tensor (optional)
        # 3 stride
        # 4 padding
        # 5 dilation
        # 6 transposed (boolean)
        # 7 output_padding
        # 8 groups
        input_dims = event['args']['Input Dims']
        concrete_inputs = event['args']['Concrete Inputs']

        input_shape = tuple(input_dims[0])
        ndims = len(input_shape) - 2 #first two dimensions are batch and channel
        filter_shape = tuple(input_dims[1])
        bias = len(input_dims) == 3


        stride_arg = concrete_inputs[3]
        stride = aten_conv.str_to_tuple(stride_arg) if stride_arg != '' else (1,) * ndims
        padding_arg = concrete_inputs[4]
        padding = aten_conv.str_to_tuple(padding_arg) if padding_arg != '' else (0,) * ndims
        dilation_arg = concrete_inputs[5]
        dilation = aten_conv.str_to_tuple(dilation_arg) if dilation_arg != '' else (1,) * ndims
        transposed_conv = eval(concrete_inputs[6])
        output_padding_arg = concrete_inputs[7]
        output_padding = aten_conv.str_to_tuple(output_padding_arg) if output_padding_arg != '' else (0,) * ndims
        groups = int(concrete_inputs[8])

        # if its a length 1 tuple then we broadcast it to the number of spatial dimensions
        stride, padding, dilation, output_padding = [
            param * ndims if len(param) == 1 else param
            for param in [stride, padding, dilation, output_padding]
        ]

        dtype_input_weight = tuple(event['args']['Input type'][:2])
        # check no mixed precision
        if dtype_input_weight[0] != dtype_input_weight[1]:
            raise ValueError(f"Data types of input and weight are different: {dtype_input_weight}")
        try:
            input_stride = tuple(event['args']['Input Strides'][0])
            weight_stride = tuple(event['args']['Input Strides'][1])
        except KeyError:
            input_stride = weight_stride = None

        if len(input_shape) == 3:
            convNd = 'conv1d'
        elif len(input_shape) == 4:
            convNd = 'conv2d'
        elif len(input_shape) == 5:
            convNd = 'conv3d'
        else:
            raise ValueError(f"Unknown convolution dimension: {len(input_shape)}")

        return {"convNd": convNd, "input_shape": input_shape, "filter_shape": filter_shape, "dtype_input_weight": dtype_input_weight,
                "input_stride": input_stride, "weight_stride": weight_stride,
                "bias": bias, "stride": stride, "padding": padding, "dilation": dilation,
                "transposed_conv": transposed_conv, "output_padding": output_padding,
                "groups": groups}

    def bytes(self):
        dtype_input_weight = self.param_details['dtype_input_weight']
        if dtype_input_weight[0] != dtype_input_weight[1]:
            raise ValueError(f"Data types of input and weight are different: {dtype_input_weight}")
        self.bpe = name2bpe(dtype_input_weight[0])
        return super().bytes(self.bpe)

    def bytes_bwd(self):
        dtype_input_weight = self.param_details['dtype_input_weight']
        if dtype_input_weight[0] != dtype_input_weight[1]:
            raise ValueError(f"Data types of input and weight are different: {dtype_input_weight}")
        self.bpe = name2bpe(dtype_input_weight[0])
        return super().bytes_bwd(self.bpe)


class aten_conv_bwd(aten_conv):
    def __init__(self, event):
        super().__init__(event)

    def flops(self):
        return self.flops_bwd()

    def bytes(self, bytes_per_element):
        return self.bytes_bwd(bytes_per_element)
class SDPA:

    def __init__(self, event, arch=None, detail_level=0):
        # S = QK^T
        # P = softmax(S)
        # O = PV
        self.event = event
        self.param_details = self.get_param_details(event)
        # get useful stuff from the param_details
        self.B, self.N_Q, self.H, self.d_k, self.N_K = (self.param_details[key] for key in ['B', 'N_Q', 'H', 'd_k', 'N_K'])

    def get_param_details(event):
        # to be implemented in the child class
        raise NotImplementedError

    @staticmethod
    def flops_func(B, N_Q, H, d_k, N_K, dropout, causal):
        # ref: https://github.com/Dao-AILab/flash-attention/blob/main/benchmarks/benchmark_flash_attention.py#L29
        flops_qk = 2 * B * N_Q * H * d_k * N_K
        # not including softmax for now as flops are order of d_k smaller
        flops_pv = 2 * B * N_Q * H * N_K *d_k
        total_flops = flops_qk + flops_pv
        if causal:
            if N_Q == N_K:
                total_flops /= 2
            else:
                raise ValueError(f"causal=True but N_Q != N_K: {N_Q} != {N_K}")
        return total_flops
    def flops(self):
        return self.flops_func(self.B, self.N_Q, self.H, self.d_k, self.N_K,
                                self.param_details['dropout'], self.param_details['causal'])

    @staticmethod
    def bytes_func(B, N_Q, H, d_k, N_K, dropout, causal, bytes_per_element):
        if dropout != 0.0:
            raise ValueError(f"Not implemented for dropout={dropout}")
        elems_q_read = B * N_Q * d_k * H
        elems_kv_read = 2 * B * N_K * d_k * H
        elems_out_write = B * N_Q * d_k * H
        total_elems_moved = elems_q_read + elems_kv_read + elems_out_write
        return total_elems_moved * bytes_per_element
    #TODO make bytes_per_element based on profile info
    def bytes(self, bytes_per_element=2):
        return self.bytes_func(self.B, self.N_Q, self.H, self.d_k, self.N_K,
                                self.param_details['dropout'], self.param_details['causal'], bytes_per_element)

    @staticmethod
    def flops_bwd_func(B, N_Q, H, d_k, N_K, dropout, causal, flash_impl):
        if causal:
            raise ValueError("Not implemented for causal=True")
        if dropout != 0.0:
            raise ValueError(f"Not implemented for dropout={dropout}")
        flops_recompute_qk = 2 * B * N_Q * H * d_k * N_K if flash_impl else 0

        # not including softmax for now
        flops_v_grad = 2 * B * N_Q * H * d_k * N_K
        flops_s_grad = 2 * B * N_Q * H * d_k * N_K
        flops_q_grad = 2 * B * N_Q * H * d_k * N_K
        flops_k_grad = 2 * B * N_Q * H * d_k * N_K

        total_flops = flops_v_grad + flops_s_grad + flops_q_grad + flops_k_grad + flops_recompute_qk
        if causal:
            if N_Q == N_K:
                total_flops /= 2
            else:
                raise ValueError(f"causal=True but N_Q != N_K: {N_Q} != {N_K}")
        return total_flops

    def flops_bwd(self):
        return self.flops_bwd_func(self.B, self.N_Q, self.H, self.d_k, self.N_K,
                                    self.param_details['dropout'], self.param_details['causal'], self.param_details['flash_impl'])

    # @staticmethod
    # def bytes_bwd_func(B, N_Q, H, d_k, N_K, dropout, causal, flash_impl, bytes_per_element):
    def bytes_bwd(self, bytes_per_element=2):
        # not implemented for now
        return None

class flash_attention(SDPA):

    @staticmethod
    def get_param_details(event):
        input_dims = event['args']['Input Dims']
        B, N_Q, H, d_k = input_dims[0]
        _, N_K, _, _ = input_dims[1]
        _, _, _, _ = input_dims[2]
        dropout = float(event['args']['Concrete Inputs'][3])
        causal = eval(event['args']['Concrete Inputs'][5])
        return {"B": B, "N_Q": N_Q, "N_K": N_K, "H": H, "d_k": d_k,
                "dropout": dropout, "causal": causal, "flash_impl": True}

class flash_attention_backward(flash_attention):

    def __init__(self, event):
        super().__init__(event)

    def flops(self):
        return self.flops_bwd()

    def bytes(self, bytes_per_element):
        return self.bytes_bwd(bytes_per_element)

class aten__scaled_dot_product_cudnn_attention(SDPA):

    @staticmethod
    def get_param_details(event):
        # the order of arguments for aten::_scaled_dot_product_cudnn_attention is:

        # query: Tensor
        # key: Tensor
        # value: Tensor
        # attn_bias: Optional[Tensor]
        # compute_log_sumexp: bool
        # dropout_p: float
        # is_causal: bool
        # return_debug_mask: bool
        # scale: Optional[float]
        input_dims = event['args']['Input Dims']
        concrete_inputs = event['args']['Concrete Inputs']

        B, H, N_Q, d_k = input_dims[0]
        _, _, N_K, _ = input_dims[1]

        dropout_p = 0.0
        if concrete_inputs[5] not in ('', 'None'):
            try:
                dropout_p = float(concrete_inputs[5])
            except (ValueError, TypeError):
                pass

        is_causal = concrete_inputs[6].lower() == 'true' if concrete_inputs[6] not in ('', 'None') else False

        return {"B": B, "N_Q": N_Q, "N_K": N_K, "H": H, "d_k": d_k,
                "dropout": dropout_p, "causal": is_causal, "flash_impl": False}

class UnaryElementwise:

    def __init__(self, event, arch=None, detail_level=0):
        self.event = event
        self.param_details = self.get_param_details(event)
        self.nelems = prod(self.param_details['op_shape'])
        self.dtype_in_out = self.param_details['dtype_in_out']
        self.stride_input = self.param_details['stride_input']
        self.stride_output = self.param_details['stride_output']

        self.bpe_in = name2bpe(self.dtype_in_out[0])
        if self.dtype_in_out[1] is not None:
            self.bpe_out = name2bpe(self.dtype_in_out[1])
        else:
            # same as input
            self.bpe_out = self.bpe_in

    @staticmethod
    def flops_func(nelems):
        return nelems
    def flops(self):
        return self.flops_func(self.nelems)

    @staticmethod
    def bytes_func(nelems, bpe_in, bpe_out):
        if None in {bpe_in, bpe_out}:
            return None
        return nelems*bpe_in + nelems*bpe_out
    def bytes(self):
        return self.bytes_func(self.nelems, self.bpe_in, self.bpe_out)

class aten_unary_elementwise(UnaryElementwise):

    @staticmethod
    def get_param_details(event):
        args_input_dims = event['args']['Input Dims']
        op_shape = tuple(args_input_dims[0])
        dtype_in = event['args']['Input type'][0]
        stride_input = tuple(event['args']['Input Strides'][0])
        if len(args_input_dims) > 1 and args_input_dims[1]:
            dtype_out = event['args']['Input type'][1]
            stride_output = tuple(event['args']['Input Strides'][1])
        else:
            dtype_out = None
            stride_output = None
        return {"op_shape": op_shape, "dtype_in_out" : (dtype_in, dtype_out),
                "stride_input": stride_input, "stride_output": stride_output}
class BinaryElementwise:

    def __init__(self, event, arch=None, detail_level=0):
        self.event = event
        self.param_details = self.get_param_details(event)
        broadcast_shape = self.get_broadcast_shape(self.param_details['shape_in1'], self.param_details['shape_in2'])
        self.nelems_in1 = prod(self.param_details['shape_in1'])
        self.nelems_in2 = prod(self.param_details['shape_in2'])
        self.nelems_out = prod(broadcast_shape)
        self.dtype_in1_in2_out = self.param_details['dtype_in1_in2_out']
        self.stride_input1 = self.param_details['stride_input1']
        self.stride_input2 = self.param_details['stride_input2']
        self.stride_output = self.param_details['stride_output']

        dtype_in1, dtype_in2, dtype_out = self.dtype_in1_in2_out
        self.bpe_in1 = name2bpe(dtype_in1)
        self.bpe_in2 = name2bpe(dtype_in2)
        if dtype_out is not None:
            self.bpe_out = name2bpe(dtype_out)
        elif self.bpe_in1 and self.bpe_in2:
            if is_tensortype(dtype_in1) and is_tensortype(dtype_in2):
                # cast to higher precision if both are tensors
                self.bpe_out = max(self.bpe_in1, self.bpe_in2)
            else:
                self.bpe_out = self.bpe_in1
        else:
            self.bpe_out = None
    @staticmethod
    def flops_func(nelems_out):
        return nelems_out
    def flops(self):
        return self.flops_func(self.nelems_out)

    @staticmethod
    def bytes_func(nelems_in1, nelems_in2, nelems_out, bpe_in1, bpe_in2, bpe_out):
        if None in {bpe_in1, bpe_in2, bpe_out}:
            return None
        return nelems_in1*bpe_in1 + nelems_in2*bpe_in2 + nelems_out*bpe_out
    def bytes(self):
        return self.bytes_func(self.nelems_in1, self.nelems_in2, self.nelems_out, self.bpe_in1, self.bpe_in2, self.bpe_out)

    @staticmethod
    def get_broadcast_shape(shape1, shape2):
        # Align shapes to the right by pre-pending 1's
        ndim = max(len(shape1), len(shape2))
        shape1 = (1,) * (ndim - len(shape1)) + shape1
        shape2 = (1,) * (ndim - len(shape2)) + shape2
        result = []
        for d1, d2 in zip(shape1, shape2):
            if d1 != d2 and d1 != 1 and d2 != 1:
                raise ValueError("Shapes not broadcastable: {} and {}".format(shape1, shape2))
            result.append(max(d1, d2))
        return tuple(result)

class aten_binary_elementwise(BinaryElementwise):

    @staticmethod
    def get_param_details(event):
        args_input_dims = event['args']['Input Dims']
        shape_in1 = tuple(args_input_dims[0])
        shape_in2 = tuple(args_input_dims[1])
        dtype_in1 = event['args']['Input type'][0]
        dtype_in2 = event['args']['Input type'][1]
        stride_input1 = tuple(event['args']['Input Strides'][0])
        stride_input2 = tuple(event['args']['Input Strides'][1])

        if len(args_input_dims) > 2 and args_input_dims[2]:
            dtype_out = event['args']['Input type'][2]
            stride_output = tuple(event['args']['Input Strides'][2])
        else:
            dtype_out = None
            stride_output = None
        return {"shape_in1": shape_in1, "shape_in2": shape_in2,
                "dtype_in1_in2_out" : (dtype_in1, dtype_in2, dtype_out),
                "stride_input1": stride_input1, "stride_input2": stride_input2, "stride_output": stride_output}

