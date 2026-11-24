###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for the GroupedGemmaRMSNorm perf model (SGLang grouped Gemma RMSNorm)."""

import pytest

from TraceLens.PerfModel.extensions.rmsnorm_perf_model_extensions import (
    GroupedGemmaRMSNorm,
)
from TraceLens.PerfModel.kernel_name_parser import grouped_rmsnorm_group_size
from TraceLens.PerfModel.torch_op_mapping import op_to_perf_model_class_map

_OP_NAME = "sglang_profiler::grouped_gemma_rmsnorm_grouped_gemma_rmsnorm"
_KERNEL_NAME = (
    "void sglang::grouped_gemma_rmsnorm_kernel<2560l, false, __hip_bfloat16>"
    "(sglang::GroupedGemmaRMSNormParams)"
)
_HIDDEN, _GROUP = 10240, 2560


def _event(m, kernel_name=_KERNEL_NAME):
    return {
        "name": _OP_NAME,
        "args": {
            "Input Dims": [[m, _HIDDEN], [_HIDDEN], []],
            "Input type": ["c10::BFloat16", "c10::BFloat16", ""],
            "Input Strides": [[_HIDDEN, 1], [1], []],
            "Concrete Inputs": ["", "", ""],
        },
        "kernel_details": [{"name": kernel_name, "dur": 4.6}],
    }


def test_group_size_from_kernel_name():
    assert grouped_rmsnorm_group_size(_KERNEL_NAME) == _GROUP
    assert grouped_rmsnorm_group_size("void sglang::rmsnorm_kernel<2560l>") is None


def test_op_is_mapped():
    assert op_to_perf_model_class_map[_OP_NAME] is GroupedGemmaRMSNorm


@pytest.mark.parametrize("m", [8, 32, 2048])
def test_flops_and_bytes(m):
    model = GroupedGemmaRMSNorm(_event(m))
    elems = m * _HIDDEN
    groups = elems // _GROUP
    assert model.flops() == groups * (2 * _GROUP + 2) + 3 * elems
    assert model.bytes() == 2 * elems * 2 + _HIDDEN * 2
    assert model.param_details["num_channels"] == _GROUP


def test_missing_kernel_raises():
    with pytest.raises(ValueError):
        GroupedGemmaRMSNorm(_event(32, "void other_kernel<1>()"))


def test_backward_not_implemented():
    model = GroupedGemmaRMSNorm(_event(32))
    with pytest.raises(NotImplementedError):
        model.flops_bwd()
    with pytest.raises(NotImplementedError):
        model.bytes_bwd()
