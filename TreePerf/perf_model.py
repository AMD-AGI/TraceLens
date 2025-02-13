from math import prod

# 1. GEMM 
class GEMM:
    """
    This is the base class for all GEMM operations. 
    If you want to add a new GEMM operation, you should inherit from this class.
    """
    def __init__(self, event):
        self.event = event
        self.param_details = self.get_param_details(event)
        self.M, self.N, self.K = self.param_details['M'], self.param_details['N'], self.param_details['K']
        self.bias = self.param_details['bias']
    
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
    def bytes_func(M, N, K, bias, bytes_per_element):
        elems_input_read = M * K
        elems_weight_read = K * N
        elems_bias_read = N if bias else 0
        elems_output_write = M * N
        total_elems_moved = elems_input_read + elems_weight_read + elems_bias_read + elems_output_write
        return total_elems_moved * bytes_per_element
    
    def bytes(self, bytes_per_element):
        return self.bytes_func(self.M, self.N, self.K, self.bias, bytes_per_element)
    
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
        return {"M": M, "N": N, "K": K, "bias": False}

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
        return {"M": M, "N": N, "K": K, "bias": True}
    
    def flops_bwd(self):
        raise NotImplementedError("Backward pass for aten::addmm is not defined.")
    def bytes_bwd(self, bytes_per_element):
        raise NotImplementedError("Backward pass for aten::addmm is not defined.")

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
        return {"M": M, "N": N, "K": K, "bias": bias}

# 2. Convolution
class CONV:
    # we will make stuff reusiable across conv1d, conv2d, and conv3d
    def __init__(self, event):
        self.event = event
        self.param_details = self.get_param_details(event)
        self.x_shape, self.w_shape = self.param_details['input_shape'], self.param_details['filter_shape']
        self.stride, self.padding, self.dilation, self.groups = ( self.param_details[key] for key in ['stride', 'padding', 'dilation', 'groups'])
        self.bias = self.param_details['bias']
        self.transposed_conv = self.param_details['transposed_conv']
        self.out_shape = CONV.get_output_shape(self.x_shape, self.w_shape, self.stride, self.padding, self.dilation, self.transposed_conv)
    
    @staticmethod
    def get_output_shape(input_shape, filter_shape, stride, padding, dilation, transposed_conv):
        x_spatial_shape, w_spatial_shape = input_shape[2:], filter_shape[2:]
        conv_ndims = len(x_spatial_shape)
        spatial_out_fn = CONV.get_conv_out_dim if not transposed_conv else CONV.get_transposed_conv_out_dim
        out_spatial_shape = tuple(spatial_out_fn(x_spatial_shape[i], w_spatial_shape[i], 
                                                stride[i], padding[i], dilation[i]) for i in range(conv_ndims))
        return (input_shape[0], filter_shape[0]) + tuple(out_spatial_shape)
    
    @staticmethod
    def t(shape):
        return (shape[1], shape[0]) + shape[2:]

    @staticmethod
    def get_conv_out_dim(input_dim, kernel_size, stride, padding, dilation):
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
    def bytes_func(x_shape, w_shape, out_shape, bias, bytes_per_element):
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
    def get_param_details(event):
        input_shape = tuple(event['args']['Input Dims'][0])
        filter_shape = tuple(event['args']['Input Dims'][1])
        bias = len(event['args']['Input Dims']) == 2 
        # stride, padding and dilation are strings, eg: "[1, 1]" 
        # we do [1:-1] to remove the brackets
        # we need to handle cases where the stride, padding and dilation are not provided
        concrete_inputs = event['args']['Concrete Inputs']
        stride_str = concrete_inputs[3]
        if stride_str != '':
            stride = tuple(int(s) for s in stride_str[1:-1].split(','))
        else:
            stride = (1,) * (len(input_shape) - 2)

        padding_str = concrete_inputs[4]
        if padding_str != '':
            padding = tuple(int(p) for p in padding_str[1:-1].split(','))
        else:
            padding = (0,) * (len(input_shape) - 2)

        dilation_str = concrete_inputs[5]
        if dilation_str != '':
            dilation = tuple(int(d) for d in dilation_str[1:-1].split(','))
        else:
            dilation = (1,) * (len(input_shape) - 2)
        
        groups = int(event['args']['Concrete Inputs'][6])
        # self.bias = bool(self.event['args']['Input Dims'][2]) #recheck this
        return {"input_shape": input_shape, "filter_shape": filter_shape,
                "stride": stride, "padding": padding, "dilation": dilation,
                "groups": groups, "bias": bias, "transposed_conv": False}

class aten_conv_bwd(aten_conv):
    def __init__(self, event):
        super().__init__(event)
    
    def flops(self):
        return self.flops_bwd()
    
    def bytes(self, bytes_per_element):
        return self.bytes_bwd(bytes_per_element)
class SDPA:

    def __init__(self, event):
        self.event = event
        self.param_details = self.get_param_details(event)
        # get useful stuff from the param_details
        self.B, self.N_Q, self.H, self.d_k, self.N_K = (self.param_details[key] for key in ['B', 'N_Q', 'H', 'd_k', 'N_K'])
    
    def get_param_details(event):
        # to be implemented in the child class
        raise NotImplementedError
    
    @staticmethod
    def flops_func(B, N_Q, H, d_k, N_K, dropout, causal):
        if causal:
            raise ValueError("Not implemented for causal=True")
        if dropout != 0.0:
            raise ValueError(f"Not implemented for dropout={dropout}")
        flops_qk = 2 * B * N_Q * H * d_k * N_K
        # not including softmax for now
        flops_pv = 2 * B * N_Q * H * N_K *d_k
        return flops_qk + flops_pv
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
    def bytes(self, bytes_per_element):
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

        return flops_v_grad + flops_s_grad + flops_q_grad + flops_k_grad + flops_recompute_qk
    def flops_bwd(self):
        return self.flops_bwd_func(self.B, self.N_Q, self.H, self.d_k, self.N_K,
                                    self.param_details['dropout'], self.param_details['causal'], self.param_details['flash_impl'])

    # @staticmethod
    # def bytes_bwd_func(B, N_Q, H, d_k, N_K, dropout, causal, flash_impl, bytes_per_element):
    def bytes_bwd(self, bytes_per_element):
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
