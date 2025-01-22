import perf_model
from param_details import *

# op_to_param_details_func_map = {
#     'aten::linear': get_param_details_aten_linear,
#     'aten::mm': get_param_details_aten_mm,
#     'aten::addmm': get_param_details_aten_addmm,
#     'FlashAttnFunc': get_param_details_flash_attention,
#     'aten::conv2d': get_param_details_aten_conv2d,
#     'aten::conv3d': get_param_details_aten_conv3d,
# }

op_to_perf_model_class_map = {
    'aten::linear': perf_model.aten_linear,
    'aten::mm': perf_model.aten_mm,
    'aten::addmm': perf_model.aten_addmm,
    'FlashAttnFunc': perf_model.flash_attention,
    'aten::conv2d': perf_model.aten_conv,
    'aten::conv3d': perf_model.aten_conv,

}
