# 1. GEMM 
class GEMM:
    """
    This is the base class for all GEMM operations. 
    If you want to add a new GEMM operation, you should inherit from this class.
    """
    def __init__(self, event):
        self.event = event
        self.param_details = self.get_param_details()
        self.M, self.N, self.K = self.param_details['M'], self.param_details['N'], self.param_details['K']
        self.bias = self.param_details['bias']
    
    def get_param_details(self):
        # to be implemented in the child class
        raise NotImplementedError

    def flops(self):
        flops_matmul = 2 * self.M * self.N * self.K
        flops_bias = self.M * self.N if self.bias else 0
        return flops_matmul + flops_bias
    
    def bytes(self, bytes_per_element):
        elems_input_read = self.M * self.K
        elems_weight_read = self.K * self.N
        elems_bias_read = self.N if self.bias else 0
        elems_output_write = self.M * self.N
        total_elems_moved = elems_input_read + elems_weight_read + elems_bias_read + elems_output_write
        return total_elems_moved * bytes_per_element

    """
    bwd pass for Y = X.matmul(W^T) + B
    X_grad = Y_grad.matmul(W)
    W_grad = Y_grad^T.matmul(X)
    B_grad = Y_grad.sum(dim=0)
    """
    
    def flops_bwd(self):
        flops_input_grad = 2 * self.M * self.N * self.K
        flops_weight_grad = 2 * self.M * self.N * self.K
        flops_bias_grad = self.M * self.N if self.bias else 0
        return flops_input_grad + flops_weight_grad + flops_bias_grad
    
    def bytes_bwd(self, bytes_per_element):
        elems_out_grad_read = self.M * self.N
        elems_input_read = self.M * self.K
        elems_input_grad_write = self.M * self.K
        elems_weight_grad_write = self.K * self.N
        elems_weight_read = self.K * self.N
        elems_bias_grad_write = self.N if self.bias else 0 
        total_elems_moved_input_grad = elems_out_grad_read + elems_weight_read + elems_input_grad_write
        total_elems_moved_weight_grad = elems_out_grad_read + elems_input_read + elems_weight_grad_write
        total_elems_moved_bias_grad = elems_out_grad_read + elems_bias_grad_write
        total_elems_moved = total_elems_moved_input_grad + total_elems_moved_weight_grad
        total_elems_moved += total_elems_moved_bias_grad if self.bias else 0
        return total_elems_moved * bytes_per_element

class aten_mm(GEMM):
    """
    aten::mm the matrix multiplication primitive in PyTorch
    A.matmul(B)
    """
    def get_param_details(self):
        if self.event['name'] != 'aten::mm':
            raise ValueError(f"Event name is not aten::mm, but {self.event['name']}")
        input_dims = self.event['args']['Input Dims']
        A_shape, B_shape = input_dims[0], input_dims[1]
        M = A_shape[0]
        N = B_shape[1]
        K = A_shape[1]
        return {"M": M, "N": N, "K": K, "bias": False}

class aten_addmm(GEMM):
    """
    aten::addmm is the A.matmul(B) + C operation in PyTorch
    """
    def get_param_details(self):
        if self.event['name'] != 'aten::addmm':
            raise ValueError(f"Event name is not aten::addmm, but {self.event['name']}")
        input_dims = self.event['args']['Input Dims']
        C_shape, A_shape, B_shape = input_dims[0], input_dims[1], input_dims[2]
        M = A_shape[0]
        N = B_shape[1]
        K = A_shape[1]
        return {"M": M, "N": N, "K": K, "bias": True}

class aten_linear(GEMM):    
    
    def get_param_details(self):
        if self.event['name'] != 'aten::linear':
            raise ValueError(f"Event name is not aten::linear, but {self.event['name']}")

        input_dims = self.event['args']['Input Dims']
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
        self.param_details = self.get_param_details()
        self.input_shape = self.param_details['input_shape']
        self.filter_shape = self.param_details['filter_shape']
        self.c_in, self.c_out = self.input_shape[1], self.filter_shape[0]
        self.B = self.input_shape[0]
        self.stride, self.padding, self.dilation, self.groups = ( self.param_details[key] for key in ['stride', 'padding', 'dilation', 'groups'])
        self.adjusted_C_in = self.c_in // self.groups
        self.out_spatial_shape = (self.get_conv_out_dim(self.input_shape[i+2], self.filter_shape[i+2], 
                                        self.stride[i], self.padding[i], self.dilation[i]) for i in range(len(self.input_shape)-2))
        self.out_shape = (self.B, self.c_out) + tuple(self.out_spatial_shape)
    
    @staticmethod
    def get_conv_out_dim(input_dim, kernel_size, stride, padding, dilation):
        return int(((input_dim + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1)

    @staticmethod
    def prod(input_tuple):
        prod = 1
        for i in input_tuple:
            prod *= i
        return prod

    def flops(self):
        flops_conv = 2 * self.prod(self.out_shape) * self.adjusted_C_in * self.prod(self.filter_shape[2:])
        flops_bias = self.prod(self.out_shape) if self.bias else 0
        return flops_conv + flops_bias

    def bytes(self, bytes_per_element):
        elems_input_read = self.B * self.c_in * self.prod(self.input_shape)
        elems_weight_read = self.c_out * (self.c_in // self.groups) * self.prod(self.filter_shape)
        elems_bias_read = self.c_out if self.bias else 0
        elems_output_write = self.B * self.c_out * self.prod(self.out_shape)
        total_elems_moved = elems_input_read + elems_weight_read + elems_bias_read + elems_output_write
        return total_elems_moved * bytes_per_element

    def flops_bwd(self):
        flops_grad_input = 2 * self.prod(self.input_shape) * self.c_out * self.prod(self.filter_shape[2:])
        flops_grad_weights = 2 * self.prod(self.out_shape) * self.adjusted_C_in * self.prod(self.filter_shape[2:])
        flops_grad_bias = self.prod(self.out_shape) if self.bias else 0
        return flops_grad_input + flops_grad_weights + flops_grad_bias

    def bytes_bwd(self, bytes_per_element):
        # not implemented for now
        return None
    
    def get_param_details(self):
        # to be implemented in the child class
        raise NotImplementedError
    
class aten_conv(CONV):

    def get_param_details(self):
        valid_conv_names = ['aten::conv1d', 'aten::conv2d', 'aten::conv3d',
                            'aten::miopen_convolution', 'aten::cudnn_convolution']
        if self.event['name'] not in valid_conv_names:
            raise ValueError(f"Event name is not a convolution, but {self.event['name']}")
        self.input_shape = tuple(self.event['args']['Input Dims'][0])
        self.filter_shape = tuple(self.event['args']['Input Dims'][1])
        self.c_in, self.c_out = self.input_shape[1], self.filter_shape[0]
        self.B = self.input_shape[0]
        # stride, padding and dilation are strings, eg: "[1, 1]" 
        # we do [1:-1] to remove the brackets
        # we need to handle cases where the stride, padding and dilation are not provided
        concrete_inputs = self.event['args']['Concrete Inputs']
        stride_str = concrete_inputs[3]
        if stride_str != '':
            self.stride = tuple(int(s) for s in stride_str[1:-1].split(','))
        else:
            self.stride = (1,) * (len(self.input_shape) - 2)

        padding_str = concrete_inputs[4]
        if padding_str != '':
            self.padding = tuple(int(p) for p in padding_str[1:-1].split(','))
        else:
            self.padding = (0,) * (len(self.input_shape) - 2)

        dilation_str = concrete_inputs[5]
        if dilation_str != '':
            self.dilation = tuple(int(d) for d in dilation_str[1:-1].split(','))
        else:
            self.dilation = (1,) * (len(self.input_shape) - 2)
        
        self.groups = int(self.event['args']['Concrete Inputs'][6])
        self.bias = bool(self.event['args']['Input Dims'][2]) #recheck this
        return {"input_shape": self.input_shape, "filter_shape": self.filter_shape,
                "stride": self.stride, "padding": self.padding, "dilation": self.dilation,
                "groups": self.groups}

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
        self.param_details = self.get_param_details()
    
    def get_param_details(self):
        # to be implemented in the child class
        raise NotImplementedError
    
    def flops(self):
        B, N_Q, H, d_k, N_K = self.param_details['B'], self.param_details['N_Q'], self.param_details['H'], self.param_details['d_k'], self.param_details['N_K']
        dropout = self.param_details['dropout']
        causal = self.param_details['causal']
        if causal:
            raise ValueError("Not implemented for causal=True")
        if dropout != 0.0:
            raise ValueError(f"Not implemented for dropout={dropout}")
        flops_qk = 2 * B * N_Q * H * d_k * N_K
        # not including softmax for now
        flops_pv = 2 * B * N_Q * H * N_K *d_k
        return flops_qk + flops_pv
    
    def bytes(self, bytes_per_element):
        B, N_Q, H, d_k, N_K = self.param_details['B'], self.param_details['N_Q'], self.param_details['H'], self.param_details['d_k'], self.param_details['N_K']
        dropout = self.param_details['dropout']
        if dropout != 0.0:
            raise ValueError(f"Not implemented for dropout={dropout}")
        elems_q_read = B * N_Q * d_k * H
        elems_kv_read = 2 * B * N_K * d_k * H
        elems_out_write = B * N_K * d_k * H
        total_elems_moved = elems_q_read + elems_kv_read + elems_out_write
        return total_elems_moved * bytes_per_element
    
    def flops_bwd(self):
        B, N_Q, H, d_k, N_K = self.param_details['B'], self.param_details['N_Q'], self.param_details['H'], self.param_details['d_k'], self.param_details['N_K']
        dropout = self.param_details['dropout']
        causal = self.param_details['causal']
        if causal:
            raise ValueError("Not implemented for causal=True")
        if dropout != 0.0:
            raise ValueError(f"Not implemented for dropout={dropout}")
        flash_impl = self.param_details['flash_impl']
        flops_recompute_qk = 2 * B * N_Q * H * d_k * N_K if flash_impl else 0

        # not including softmax for now
        flops_v_grad = 2 * B * N_Q * H * d_k * N_K
        flops_s_grad = 2 * B * N_Q * H * d_k * N_K
        flops_q_grad = 2 * B * N_Q * H * d_k * N_K
        flops_k_grad = 2 * B * N_Q * H * d_k * N_K

        return flops_v_grad + flops_s_grad + flops_q_grad + flops_k_grad + flops_recompute_qk

    def bytes_bwd(self, bytes_per_element):
        # not implemented for now
        return None

class flash_attention(SDPA):
    
    def get_param_details(self):
        if self.event['name'] != 'FlashAttnFunc':
            raise ValueError(f"Event name is not FlashAttnFunc, but {self.event['name']}")
        input_dims = self.event['args']['Input Dims']
        B, N_Q, H, d_k = input_dims[0]
        _, N_K, _, _ = input_dims[1]
        _, _, _, _ = input_dims[2]
        dropout = float(self.event['args']['Concrete Inputs'][3])
        causal = eval(self.event['args']['Concrete Inputs'][5])
        return {"B": B, "N_Q": N_Q, "N_K": N_K, "H": H, "d_k": d_k,
                "dropout": dropout, "causal": causal, "flash_impl": True}
