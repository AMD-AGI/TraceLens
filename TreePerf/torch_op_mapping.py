from perf_model import *
from param_details import *

op_to_param_details_func_map = {
    'aten::linear': get_param_details_aten_linear,
    'aten::mm': get_param_details_aten_mm,
    'FlashAttnFunc': get_param_details_flash_attention,
    'aten::conv2d': get_param_details_aten_conv2d,
    'aten::conv3d': get_param_details_aten_conv3d,
}

op_to_flops_func_map = {
    'aten::linear': linear_flops,
    'aten::mm': linear_flops,
    'FlashAttnFunc': sdpa_flops,
    'aten::conv2d': conv2d_flops,
    'aten::conv3d': conv3d_flops,
}

op_to_bwd_flops_func_map = {
    'aten::linear': linear_bwd_flops,
    'aten::mm': linear_bwd_flops,
    'FlashAttnFunc': sdpa_bwd_flops,
    'aten::conv2d': conv2d_bwd_flops,
    'aten::conv3d': conv3d_bwd_flops,
}

op_to_bytes_func_map = {
    'aten::linear': linear_bytes,
    'aten::mm': linear_bytes,
    'FlashAttnFunc': fa_bytes,
    # 'aten::conv2d': conv2d_bytes,
    # 'aten::conv3d': conv3d_bytes,
}

op_to_bwd_bytes_func_map = {
    'aten::linear': linear_bwd_bytes,
    'aten::mm': linear_bwd_bytes,
    # 'FlashAttnFunc': fa_bwd_bytes,
    # 'aten::conv2d': conv2d_bwd_bytes,
    # 'aten::conv3d': conv3d_bwd_bytes,
}