# 1. linear 

def linear_flops(dict_shapes):
    M, N, K = dict_shapes['M'], dict_shapes['N'], dict_shapes['K']
    bias = dict_shapes['bias']
    flops_matmul = 2 * M * N * K
    flops_bias = M * K if bias else 0
    return flops_matmul + flops_bias

def linear_bytes(dict_shapes, bytes_per_element):
    M, N, K = dict_shapes['M'], dict_shapes['N'], dict_shapes['K']
    bias = dict_shapes['bias']
    elems_input_read = M * N
    elems_weight_read = N * K
    elems_bias_read = K if bias else 0
    elems_output_write = M * K
    total_elems_moved = elems_input_read + elems_weight_read + elems_bias_read + elems_output_write
    return total_elems_moved * bytes_per_element

def linear_bwd_flops(dict_shapes):
    M, N, K = dict_shapes['M'], dict_shapes['N'], dict_shapes['K']
    bias = dict_shapes['bias']
    flops_input_grad = 2 * M * N * K
    flops_weight_grad = 2 * M * N * K
    flops_bias_grad = M * K if bias else 0
    return flops_input_grad + flops_weight_grad + flops_bias_grad

def linear_bwd_bytes(dict_shapes, bytes_per_element):
    M, N, K = dict_shapes['M'], dict_shapes['N'], dict_shapes['K']
    bias = dict_shapes['bias']
    elems_out_grad_read = M * K
    elems_input_read = M * N
    elems_input_grad_write = M * N
    elems_weight_grad_write = N * K
    elems_weight_read = N * K
    elems_bias_grad_write = K if bias else 0 
    total_elems_moved = (
        elems_out_grad_read +
        elems_input_read +
        elems_input_grad_write +
        elems_weight_grad_write +
        elems_weight_read +
        elems_bias_grad_write
    )
    return total_elems_moved * bytes_per_element

# 2. sdpa and its flash implementation

def sdpa_flops(dict_shapes):
    B, N_Q, H, d_k, N_K = dict_shapes['B'], dict_shapes['N_Q'], dict_shapes['H'], dict_shapes['d_k'], dict_shapes['N_K']
    dropout = dict_shapes['dropout']
    causal = dict_shapes['causal']
    if causal:
        raise ValueError("Not implemented for causal=True")
    if dropout != 0.0:
        raise ValueError(f"Not implemented for dropout={dropout}")
    flops_qk = 2 * B * N_Q * H * d_k * N_K
    # not including softmax for now
    flops_pv = 2 * B * N_Q * H * N_K *d_k
    return flops_qk + flops_pv

def fa_bytes(dict_shapes, bytes_per_element):
    B, N_Q, H, d_k, N_K = dict_shapes['B'], dict_shapes['N_Q'], dict_shapes['H'], dict_shapes['d_k'], dict_shapes['N_K']
    dropout = dict_shapes['dropout']
    if dropout != 0.0:
        raise ValueError(f"Not implemented for dropout={dropout}")
    elems_q_read = B * N_Q * d_k * H
    elems_kv_read = 2 * B * N_K * d_k * H
    elems_out_write = B * N_K * d_k * H
    total_elems_moved = elems_q_read + elems_kv_read + elems_out_write
    return total_elems_moved * bytes_per_element

def sdpa_bwd_flops(dict_shapes):
    B, N_Q, H, d_k, N_K = dict_shapes['B'], dict_shapes['N_Q'], dict_shapes['H'], dict_shapes['d_k'], dict_shapes['N_K']
    dropout = dict_shapes['dropout']
    causal = dict_shapes['causal']
    if causal:
        raise ValueError("Not implemented for causal=True")
    if dropout != 0.0:
        raise ValueError(f"Not implemented for dropout={dropout}")
    flash_impl = dict_shapes['flash_impl']
    if flash_impl:
        flops_recompute_qk = 2 * B * N_Q * H * d_k * N_K
    else:
        flops_recompute_qk = 0

    # not including softmax for now
    flops_v_grad = 2 * B * N_Q * H * d_k * N_K
    flops_s_grad = 2 * B * N_Q * H * d_k * N_K
    flops_q_grad = 2 * B * N_Q * H * d_k * N_K
    flops_k_grad = 2 * B * N_Q * H * d_k * N_K

    return flops_v_grad + flops_s_grad + flops_q_grad + flops_k_grad + flops_recompute_qk

def fa_bwd_flops(dict_shapes, bytes_per_element):
    B, N_Q, H, d_k, N_K = dict_shapes['B'], dict_shapes['N_Q'], dict_shapes['H'], dict_shapes['d_k'], dict_shapes['N_K']
    dropout = dict_shapes['dropout']
    if dropout != 0.0:
        raise ValueError(f"Not implemented for dropout={dropout}")
    elems_out_grad_read = B * N_K * d_k * H
    elems_q_grad_write = B * N_Q * d_k * H
    elems_kv_grad_write = 2 * B * N_K * d_k * H
    elems_q_read = B * N_Q * d_k * H
    elems_kv_read = 2 * B * N_K * d_k * H
    total_elems_moved = elems_out_grad_read + elems_q_grad_write + elems_kv_grad_write + elems_q_read + elems_kv_read
    return total_elems_moved * bytes_per_element

# 3. conv2d and conv3d

def get_conv_out_dim(input_dim, kernel_size, stride, padding, dilation):
    return int(((input_dim + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1)

def conv2d_flops(dict_shapes):
    B, C_in, H, W, C_out, K_h, K_w = ( dict_shapes[key] for key in ['B', 'C_in', 'H', 'W', 'C_out', 'K_h', 'K_w'])
    stride, padding, dilation, groups = ( dict_shapes[key] for key in ['stride', 'padding', 'dilation', 'groups'])
    H_out = get_conv_out_dim(H, K_h, stride[0], padding[0], dilation[0])
    W_out = get_conv_out_dim(W, K_w, stride[1], padding[1], dilation[1])
    adjusted_C_in = C_in // groups

    flops_conv = 2 * B * H_out * W_out * C_out * adjusted_C_in * K_h * K_w
    flops_bias = B * H_out * W_out * C_out if dict_shapes['bias'] else 0
    return flops_conv + flops_bias

# def conv2d_bytes(dict_shapes, bytes_per_element):
#     B, C_in, H, W, C_out, K_h, K_w = ( dict_shapes[key] for key in ['B', 'C_in', 'H', 'W', 'C_out', 'K_h', 'K_w'])
#     elems_input_read = B * C_in * H * W
#     elems_weight_read = C_out * adjusted_C_in * K_h * K_w


def conv2d_bwd_flops(dict_shapes):
    B, C_in, H, W, C_out, K_h, K_w = ( dict_shapes[key] for key in ['B', 'C_in', 'H', 'W', 'C_out', 'K_h', 'K_w'])
    stride, padding, dilation, groups = ( dict_shapes[key] for key in ['stride', 'padding', 'dilation', 'groups'])
    H_out = get_conv_out_dim(H, K_h, stride[0], padding[0], dilation[0])
    W_out = get_conv_out_dim(W, K_w, stride[1], padding[1], dilation[1])
    adjusted_C_in = C_in // groups

    flops_grad_input = 2 * B * H * W * adjusted_C_in * C_out * K_h * K_w
    flops_grad_weights = 2 * B * H_out * W_out * adjusted_C_in * C_out * K_h * K_w
    flops_grad_bias = B * H_out * W_out * C_out if dict_shapes['bias'] else 0
    return flops_grad_input + flops_grad_weights + flops_grad_bias

def conv3d_flops(dict_shapes):
    B, C_in, H, W, D, C_out, K_h, K_w, K_d = ( dict_shapes[key] for key in ['B', 'C_in', 'H', 'W', 'D', 'C_out', 'K_h', 'K_w', 'K_d'])
    stride, padding, dilation, groups = ( dict_shapes[key] for key in ['stride', 'padding', 'dilation', 'groups'])
    H_out = get_conv_out_dim(H, K_h, stride[0], padding[0], dilation[0])
    W_out = get_conv_out_dim(W, K_w, stride[1], padding[1], dilation[1])
    D_out = get_conv_out_dim(D, K_d, stride[2], padding[2], dilation[2])
    adjusted_C_in = C_in // groups

    flops_conv = 2 * B * H_out * W_out * D_out * C_out * adjusted_C_in * K_h * K_w
    flops_bias = B * H_out * W_out * D_out * C_out if dict_shapes['bias'] else 0
    return flops_conv + flops_bias

def conv3d_bwd_flops(dict_shapes):
    B, C_in, H, W, D, C_out, K_h, K_w, K_d = ( dict_shapes[key] for key in ['B', 'C_in', 'H', 'W', 'D', 'C_out', 'K_h', 'K_w', 'K_d'])
    stride, padding, dilation, groups = ( dict_shapes[key] for key in ['stride', 'padding', 'dilation', 'groups'])
    H_out = get_conv_out_dim(H, K_h, stride[0], padding[0], dilation[0])
    W_out = get_conv_out_dim(W, K_w, stride[1], padding[1], dilation[1])
    D_out = get_conv_out_dim(D, K_d, stride[2], padding[2], dilation[2])
    adjusted_C_in = C_in // groups
    flops_grad_input = 2 * B * H * W * D * adjusted_C_in * C_out * K_h * K_w
    flops_grad_weights = 2 * B * H_out * W_out * D_out * adjusted_C_in * C_out * K_h * K_w
    flops_grad_bias = B * H_out * W_out * D_out * C_out if dict_shapes['bias'] else 0
    return flops_grad_input + flops_grad_weights + flops_grad_bias

