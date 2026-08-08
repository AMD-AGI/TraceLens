###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""CPU-only coverage for custom collective perf model extensions."""

from __future__ import annotations

import pytest

from TraceLens.PerfModel.extensions.custom_collectives_perf_model_extensions import (
    aiter_all_gather_reg,
    aiter_fused_allreduce_rmsnorm,
    aiter_fused_allreduce_rmsnorm_,
    aiter_reduce_scatter,
    custom_ar_all_reduce,
    custom_ar_qr_all_reduce,
    sgl_kernel_all_reduce_reg,
    sgl_kernel_qr_all_reduce,
    sgl_kernel_reg_all_gather_into_tensor,
)
from TraceLens.TreePerf import tree_perf


def _fused_allreduce_event():
    return {
        "args": {
            "Input Dims": [
                (1,),
                (),
                (4, 7168),
                (4, 7168),
                (4, 7168),
                (4, 7168),
                (7168,),
                (),
                (),
                (),
            ],
            "Input type": ["Scalar"] * 10,
            "Input Strides": [()] * 10,
        }
    }


def _fused_allreduce_python_event():
    return {
        "args": {
            "Input Dims": [
                (64, 7168),
                (64, 7168),
                (7168,),
                (),
                (),
                (),
                (),
            ],
            "Input type": ["c10::BFloat16"] * 7,
            "Input Strides": [(7168, 1)] * 7,
        }
    }


def _allreduce_event(name_dims=(4, 7168)):
    return {
        "args": {
            "Input Dims": [(), name_dims, name_dims, (), ()],
            "Input type": ["Scalar", "c10::BFloat16", "c10::BFloat16", "Scalar", "Scalar"],
            "Input Strides": [(), (7168, 1), (7168, 1), (), ()],
        }
    }


class TestCustomCollectivesPerfModels:
    def test_aiter_fused_allreduce_rmsnorm(self):
        model = aiter_fused_allreduce_rmsnorm(_fused_allreduce_event())
        assert model.num_elems == 4 * 7168
        assert model.flops() > 0
        assert model.bytes() > 0

    def test_aiter_fused_allreduce_rmsnorm_python_layout(self):
        model = aiter_fused_allreduce_rmsnorm_(_fused_allreduce_python_event())
        assert model.num_elems == 64 * 7168
        assert model.flops() > 0
        assert model.bytes() > 0

    def test_custom_ar_all_reduce(self):
        model = custom_ar_all_reduce(_allreduce_event())
        assert model.flops() == 0
        assert model.bytes() > 0

    @pytest.mark.parametrize(
        "cls",
        [
            sgl_kernel_all_reduce_reg,
            sgl_kernel_qr_all_reduce,
            custom_ar_qr_all_reduce,
        ],
    )
    def test_allreduce_subclasses(self, cls):
        model = cls(_allreduce_event(name_dims=(32, 7168)))
        assert model.flops() == 0
        assert model.bytes() > 0

    def test_aiter_reduce_scatter(self):
        event = {
            "args": {
                "Input Dims": [(), (8, 7168), (4, 7168), ()],
                "Input type": ["Scalar", "c10::BFloat16", "c10::BFloat16", "Scalar"],
                "Input Strides": [(), (7168, 1), (7168, 1), ()],
            }
        }
        model = aiter_reduce_scatter(event)
        assert model.flops() == 0
        assert model.bytes() > 0

    def test_aiter_all_gather_reg(self):
        event = {
            "args": {
                "Input Dims": [(), (4, 7168), (8, 7168)],
                "Input type": ["Scalar", "c10::BFloat16", "c10::BFloat16"],
                "Input Strides": [(), (7168, 1), (7168, 1)],
            }
        }
        model = aiter_all_gather_reg(event)
        assert model.flops() == 0
        assert model.bytes() > 0

    def test_sgl_kernel_reg_all_gather_into_tensor(self):
        event = {
            "args": {
                "Input Dims": [(256, 16160), (32, 16160), ()],
                "Input type": ["c10::BFloat16", "c10::BFloat16", "Scalar"],
            }
        }
        model = sgl_kernel_reg_all_gather_into_tensor(event)
        assert model.flops() == 0
        assert model.bytes() > 0


class TestTreePerfInitKwargs:
    def test_perf_model_init_kwargs_without_optional_params(self):
        class SimpleModel:
            def __init__(self, event, arch=None, python_path=None):
                self.event = event

        kwargs = tree_perf._perf_model_init_kwargs(
            SimpleModel, event={"name": "op"}, arch={}, python_path=None, enable_origami=True
        )
        assert kwargs["event"]["name"] == "op"
        assert "enable_origami" not in kwargs

    def test_perf_model_init_kwargs_with_var_keyword(self):
        class FlexibleModel:
            def __init__(self, event, **kwargs):
                self.kwargs = kwargs

        kwargs = tree_perf._perf_model_init_kwargs(
            FlexibleModel,
            event={"name": "op"},
            arch={},
            python_path="path",
            enable_origami=False,
            inductor_cache_dir="/tmp/cache",
        )
        assert kwargs["enable_origami"] is False
        assert kwargs["inductor_cache_dir"] == "/tmp/cache"

    def test_perf_model_init_kwargs_broken_signature(self):
        class Broken:
            __init__ = 42

        kwargs = tree_perf._perf_model_init_kwargs(
            Broken, event={}, arch=None, python_path=None, enable_origami=True
        )
        assert kwargs["event"] == {}
