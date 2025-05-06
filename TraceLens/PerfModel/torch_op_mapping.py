# MIT License

# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from . import perf_model

op_to_perf_model_class_map = {
    'aten::linear': perf_model.aten_linear,
    'aten::mm': perf_model.aten_mm,
    'aten::addmm': perf_model.aten_addmm,
    'aten::_scaled_mm': perf_model.aten_scaled_mm,
    'aten::bmm': perf_model.aten_bmm,
    'tex_ts::te_gemm_ts': perf_model.tex_ts_te_gemm_ts,
    'aten::baddbmm': perf_model.aten_baddbmm,
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