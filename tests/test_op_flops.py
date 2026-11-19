###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import pytest

from TraceLens.PerfModel.op_flops import (
    OP_FLOPS_PER_ELEM,
    op_flops_per_elem,
    pow_flops_per_elem,
)
from TraceLens.PerfModel.torch_op_mapping import op_to_perf_model_class_map
from TraceLens.PerfModel.triton_compiled_perf_model import TritonCompiledPerfModel


@pytest.mark.parametrize("name", ["aten::silu_", "aten.silu", "silu", "silu_"])
def test_names_resolve_to_the_same_row(name):
    assert op_flops_per_elem(name) == OP_FLOPS_PER_ELEM["silu"]


def test_unlisted_op_is_zero():
    assert op_flops_per_elem("aten._unsafe_view") == 0


@pytest.mark.parametrize(
    "op,expected",
    [
        ("add", 1),
        ("div", 4),
        ("rsqrt", 4),
        ("exp", 8),
        ("sigmoid", 14),
        ("gelu", 12),
        ("gelu_tanh", 16),
        ("var", 4),
        ("softmax", 12),
    ],
)
def test_livermore_weights(op, expected):
    assert OP_FLOPS_PER_ELEM[op] == expected


def test_pow_exponents():
    assert pow_flops_per_elem(2) == OP_FLOPS_PER_ELEM["pow"]
    assert pow_flops_per_elem(-0.5) == OP_FLOPS_PER_ELEM["rsqrt"]
    assert pow_flops_per_elem(None) == pow_flops_per_elem(2.5) == 17


def test_fused_kernel_and_standalone_ops_agree():
    """A fused add+silu kernel costs the same as its two traced ops."""
    n = 4096
    fused = TritonCompiledPerfModel(
        {
            "name": "triton_poi_fused_add_silu_0",
            "args": {
                "Concrete Inputs": ["", "", "", str(n)],
                "Input Dims": [[n], [n], [n], []],
                "Input type": ["c10::BFloat16"] * 3 + ["Scalar"],
            },
        }
    )
    standalone = 0
    for name, dims in (("aten::add", [[n], [n]]), ("aten::silu", [[n]])):
        event = {
            "name": name,
            "args": {
                "Input Dims": dims,
                "Input type": ["c10::BFloat16"] * len(dims),
                "Input Strides": [[1]] * len(dims),
            },
        }
        standalone += op_to_perf_model_class_map[name](event).flops()
    assert fused.flops() == standalone == 15 * n
