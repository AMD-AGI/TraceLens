# import perf_model
from . import perf_model

op_to_perf_model_class_map = {
    'aten::linear': perf_model.aten_linear,
    'aten::mm': perf_model.aten_mm,
    'aten::addmm': perf_model.aten_addmm,
    'aten::_scaled_mm': perf_model.aten_scaled_mm,
    'FlashAttnFunc': perf_model.flash_attention,
    'aten::_scaled_dot_product_cudnn_attention': perf_model.aten__scaled_dot_product_cudnn_attention,
    'aten::convolution': perf_model.aten_conv,
}