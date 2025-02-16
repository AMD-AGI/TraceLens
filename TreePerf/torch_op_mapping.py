# import perf_model
from TreePerf import perf_model

op_to_perf_model_class_map = {
    'aten::linear': perf_model.aten_linear,
    'aten::mm': perf_model.aten_mm,
    'aten::addmm': perf_model.aten_addmm,
    'FlashAttnFunc': perf_model.flash_attention,
    'FlashAttnFuncBackward': perf_model.flash_attention_backward,
    'aten::convolution': perf_model.aten_conv,
}