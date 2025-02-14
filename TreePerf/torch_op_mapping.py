# import perf_model
from TreePerf import perf_model

op_to_perf_model_class_map = {
    'aten::linear': perf_model.aten_linear,
    'aten::mm': perf_model.aten_mm,
    'aten::addmm': perf_model.aten_addmm,
    'FlashAttnFunc': perf_model.flash_attention,
    'FlashAttnFuncBackward': perf_model.flash_attention_backward,
    'aten::conv2d': perf_model.aten_conv,
    'aten::conv3d': perf_model.aten_conv,

}

op_category_to_op_name_map = {
    'GEMM': ['aten::mm', 'aten::addmm'],
    'FLASH_ATTN': ['FlashAttnFunc'],
    'CONV': ['aten::conv2d', 'aten::conv3d'],
}