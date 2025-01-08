def get_param_details_aten_linear(event):
    if event['name'] != 'aten::linear':
        raise ValueError(f"Event name is not aten::linear, but {event['name']}")

    input_dims = event['args']['Input Dims']
    input_shape = input_dims[0]
    weight_shape = input_dims[1]
    bias = bool(input_dims[2])
    input_dim = input_shape[-1]
    weight_out_dim = weight_shape[0]
    # Compute M as the product of all dimensions except the last one
    M = 1
    for dim in input_shape[:-1]:
        M *= dim
    return {"M": M, "N": weight_out_dim, "K": input_dim, "bias": bias}


def get_param_details_aten_mm(event):
    if event['name'] != 'aten::mm':
        raise ValueError(f"Event name is not aten::linear, but {event['name']}")

    input_dims = event['args']['Input Dims']
    M = input_dims[0][0]
    K = input_dims[0][1]
    N = input_dims[1][1]
    return {"M": M, "N": N, "K": K, "bias": False}


def get_param_details_flash_attention(event):
    if event['name'] != 'FlashAttnFunc':
        raise ValueError(f"Event name is not FlashAttnFunc, but {event['name']}")
    input_dims = event['args']['Input Dims']
    B, N_Q, H, d_k = input_dims[0]
    _, N_K, _, _ = input_dims[1]
    _, _, _, _ = input_dims[2]
    dropout = float(event['args']['Concrete Inputs'][3])
    causal = eval(event['args']['Concrete Inputs'][5])
    return {"B": B, "N_Q": N_Q, "N_K": N_K, "H": H, "d_k": d_k,
            "dropout": dropout, "causal": causal, "flash_impl": True}

def get_param_details_aten_conv2d(event):
    if event['name'] != 'aten::conv2d':
        raise ValueError(f"Event name is not aten::conv2d, but {event['name']}")
    input_dims = event['args']['Input Dims']
    B, C_in, H, W = input_dims[0]
    C_out, _, K_h, K_w = input_dims[1]

    stride = tuple(int(s) for s in event['args']['Concrete Inputs'][3][1:-1].split(','))
    padding = tuple(int(p) for p in event['args']['Concrete Inputs'][4][1:-1].split(','))
    dilation = tuple(int(d) for d in event['args']['Concrete Inputs'][5][1:-1].split(','))
    groups = int(event['args']['Concrete Inputs'][6])
    bias = bool(input_dims[2]) #recheck this
    return {'B': B, 'C_in': C_in, 'H': H, 'W': W, 'C_out': C_out, 'K_h': K_h, 'K_w': K_w,
            'stride': stride, 'padding': padding, 'dilation': dilation, 'groups': groups,
            'bias': bias}

def get_param_details_aten_conv3d(event):
    if event['name'] != 'aten::conv3d':
        raise ValueError(f"Event name is not aten::conv3d, but {event['name']}")
    input_dims = event['args']['Input Dims']
    B, C_in, H, W, D = input_dims[0]
    C_out, _, K_h, K_w, K_d = input_dims[1]
    bias = bool(input_dims[2])
    stride = tuple((int(s) for s in event['args']['Concrete Inputs'][3][1:-1].split(',')))
    # if padding is not empty str
    if event['args']['Concrete Inputs'][4] != '':
        padding = tuple(int(p) for p in event['args']['Concrete Inputs'][4][1:-1].split(','))
    else:
        padding = (0,0,0)
    dilation = tuple(int(d) for d in event['args']['Concrete Inputs'][5][1:-1].split(','))
    groups = int(event['args']['Concrete Inputs'][6])
    bias = bool(input_dims[2]) #recheck this
    return {'B': B, 'C_in': C_in, 'H': H, 'W': W, 'D': D, 'C_out': C_out, 'K_h': K_h, 'K_w': K_w, 'K_d': K_d,
            'stride': stride, 'padding': padding, 'dilation': dilation, 'groups': groups,
            'bias': bias}
