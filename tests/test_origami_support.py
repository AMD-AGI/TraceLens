###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for the Origami architecture check and the unavailable-GPU path."""

import builtins
import warnings
from unittest.mock import patch

import pytest

from TraceLens.Agent.Analysis.utils.arch_utils import load_arch
from TraceLens.PerfModel import op_projections, origami_helper
from TraceLens.PerfModel.op_projections import (
    OpWork,
    add_op_projection_outputs,
    default_op_projections,
)
from TraceLens.PerfModel.origami_support import (
    OrigamiUnavailableError,
    origami_gfx_arch,
)

# The report adds these from the trace's deviceProperties; a W7900 trace
# records numSms 48 (its WGPs) and maxSharedMemoryPerMultiProcessor 65536.
W7900_FROM_TRACE = {**load_arch("W7900"), "num_cus": 48, "lds_capacity": 65536}


@pytest.mark.parametrize(
    "arch,gfx",
    [
        (load_arch("MI300X"), "gfx942"),
        (load_arch("MI325X"), "gfx942"),
        ({"name": "MI350X"}, "gfx950"),
        ({"name": "mi355x"}, "gfx950"),
        ({"gfx_arch": "gfx950"}, "gfx950"),
    ],
)
def test_modelled_specs_resolve_as_before(arch, gfx):
    assert origami_gfx_arch(arch) == gfx


def test_unnamed_spec_still_assumes_mi300x():
    with pytest.warns(UserWarning, match="Assuming MI300x"):
        assert origami_gfx_arch({}) == "gfx942"


@pytest.mark.parametrize(
    "arch,target",
    [
        (load_arch("W7900"), "gfx1100"),
        ({"name": "R9700"}, "r9700"),
        # The spec's gfx_arch wins over its name.
        ({"name": "MI300X", "gfx_arch": "gfx1201"}, "gfx1201"),
    ],
)
def test_unmodelled_specs_are_unavailable(arch, target):
    with pytest.raises(
        OrigamiUnavailableError,
        match=f"Origami is unavailable on {target}; it models gfx942, gfx950 only",
    ):
        origami_gfx_arch(arch)


def test_get_hardware_raises_unavailable_instead_of_key_error():
    pytest.importorskip("origami")

    with pytest.raises(OrigamiUnavailableError, match="unavailable on gfx1100"):
        origami_helper.OrigamiHelper.get_hardware(W7900_FROM_TRACE)


def test_simulation_skipped_once_without_importing_origami(monkeypatch, capsys):
    monkeypatch.setattr(origami_helper, "_unavailable_printed", False)
    real_import = builtins.__import__

    def no_origami_import(name, *args, **kwargs):
        if name == "origami":
            raise AssertionError("origami must not be imported for gfx1100")
        return real_import(name, *args, **kwargs)

    with patch.object(builtins, "__import__", no_origami_import):
        for _ in range(2):
            assert (
                origami_helper.gemm_time_us(
                    W7900_FROM_TRACE, 4096, 4096, 4096, 1, "bf16"
                )
                is None
            )
    err = capsys.readouterr().err
    assert err.count("Origami simulation skipped") == 1
    assert "Origami is unavailable on gfx1100" in err


def test_unavailable_origami_keeps_the_roofline_without_a_failure_warning(
    monkeypatch,
):
    monkeypatch.setattr(op_projections, "_op_projection_warnings", set())
    monkeypatch.setattr(origami_helper, "_unavailable_printed", False)
    work = OpWork(
        category="GEMM",
        params={"M": 4096, "N": 4096, "K": 4096, "B": 1, "simulation_dtype": "fp16"},
        gflops=137.4,
        bytes_moved=1e8,
        compute_spec="matrix_fp16",
    )
    metrics = {}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        add_op_projection_outputs(
            metrics,
            default_op_projections(enable_origami_gemm=True),
            work,
            W7900_FROM_TRACE,
            8.0,
        )
    assert "Roofline Time (µs)" in metrics
    assert not [column for column in metrics if column.startswith("Origami")]
