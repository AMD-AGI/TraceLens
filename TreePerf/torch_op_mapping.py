import perf_model

op_to_perf_model_class_map = {
    'aten::linear': perf_model.aten_linear,
    'aten::mm': perf_model.aten_mm,
    'aten::addmm': perf_model.aten_addmm,
    'aten::_scaled_mm': perf_model.aten_scaled_mm,
    'FlashAttnFunc': perf_model.flash_attention,
    'aten::conv2d': perf_model.aten_conv,
    'aten::conv3d': perf_model.aten_conv,

}
