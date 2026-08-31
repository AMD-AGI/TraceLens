###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import pytest

from TraceLens.PerfModel.kernel_library import classify_kernel_library
from TraceLens.TraceIndex.utils import kernel_flags


@pytest.mark.parametrize(
    "op_name,kernel_details,expected",
    [
        ("aiter::gemm", "", "AITER"),
        ("triton_kernel_op", "", "Triton"),
        ("aten::mm", "[{'name': 'Cijk_foo'}]", "Tensile"),
        ("aten::mm", "void at::native::x", "PyTorch Native"),
        ("aten::mm", "plain", None),
        ("aten::mm", "ncclAllReduce", "RCCL/NCCL"),
        (
            "aten::mm",
            "Cijk_Alik_Bljk_BBS_BH_Bias_HAS_SAV_UserArgs_MT64x16x64_MI16x16x1",
            "Tensile",
        ),
        ("aten::mm", "nvjet_tst_144x128_64x6_2x1_v_bz_bias_TNN", "nvjet"),
    ],
)
def test_classify_kernel_library(op_name, kernel_details, expected):
    assert classify_kernel_library(op_name, kernel_details) == expected


@pytest.mark.parametrize(
    "kernel_name,op_name,expected_library,expected_tensile,expected_transpose",
    [
        (
            "Cijk_Alik_Bljk_BBS_BH_Bias_HAS_SAV_UserArgs_MT64x16x64",
            "aten::mm",
            "Tensile",
            1,
            1,
        ),
        ("triton_red_fused_add", "aten::add", "Triton", 0, 0),
        ("ncclAllReduceRing", "", "RCCL/NCCL", 0, 0),
        ("custom_transpose_kernel", "", None, 0, 1),
    ],
)
def test_kernel_flags(
    kernel_name, op_name, expected_library, expected_tensile, expected_transpose
):
    library, is_tensile, is_transpose, _is_layout = kernel_flags(kernel_name, op_name)
    assert library == expected_library
    assert is_tensile == expected_tensile
    assert is_transpose == expected_transpose
