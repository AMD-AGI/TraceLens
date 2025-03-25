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

unary_elemwise_ops = [
    'aten::copy', 'aten::copy_',
    'atem::clamp_min', 'aten::clamp_min_', 
    'aten::sigmoid',
]

binary_elemwise_ops = [
    'aten::div', 'aten::div_',
    'aten::mul', 'aten::mul_',
    'aten::add', 'aten::add_',
    'aten::sigmoid_backward',
    'aten::threshold_backward',
]

for op in unary_elemwise_ops:
    op_to_perf_model_class_map[op] = perf_model.aten_unary_elementwise
for op in binary_elemwise_ops:
    op_to_perf_model_class_map[op] = perf_model.aten_binary_elementwise