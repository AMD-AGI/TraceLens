###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for three-way roofline bound classification."""

import pytest

from TraceLens.TreePerf import tree_perf as tree_perf_module
from TraceLens.TreePerf.tree_perf import (
    compute_roofline_bound,
    warn_if_missing_mem_latency,
)

MI300X_MEM_BW_GBPS = 5300
MI300X_MEM_LATENCY_US = 0.3


@pytest.fixture(autouse=True)
def reset_mem_latency_warning():
    tree_perf_module._MEM_LATENCY_MISSING_WARNED = False
    yield
    tree_perf_module._MEM_LATENCY_MISSING_WARNED = False


def test_compute_roofline_bound_latency_regime():
    compute_time_us = 0.01
    bytes_moved = 1024
    roofline_time_us, roofline_bound = compute_roofline_bound(
        compute_time_us,
        bytes_moved,
        MI300X_MEM_BW_GBPS,
        mem_latency_us=MI300X_MEM_LATENCY_US,
    )
    assert roofline_bound == "LATENCY_BOUND"
    assert roofline_time_us == pytest.approx(MI300X_MEM_LATENCY_US)


def test_compute_roofline_bound_bandwidth_regime():
    compute_time_us = 1.0
    bytes_moved = 16 * 1024 * 1024
    roofline_time_us, roofline_bound = compute_roofline_bound(
        compute_time_us,
        bytes_moved,
        MI300X_MEM_BW_GBPS,
        mem_latency_us=MI300X_MEM_LATENCY_US,
    )
    assert roofline_bound == "MEMORY_BOUND"
    assert roofline_time_us > MI300X_MEM_LATENCY_US


def test_compute_roofline_bound_compute_regime():
    compute_time_us = 100.0
    bytes_moved = 1024
    roofline_time_us, roofline_bound = compute_roofline_bound(
        compute_time_us,
        bytes_moved,
        MI300X_MEM_BW_GBPS,
        mem_latency_us=MI300X_MEM_LATENCY_US,
    )
    assert roofline_bound == "COMPUTE_BOUND"
    assert roofline_time_us == pytest.approx(compute_time_us)


def test_compute_roofline_bound_fallback_without_mem_latency():
    roofline_time_us, roofline_bound = compute_roofline_bound(
        0.01,
        1024,
        MI300X_MEM_BW_GBPS,
        mem_latency_us=None,
    )
    assert roofline_bound in {"COMPUTE_BOUND", "MEMORY_BOUND"}
    assert roofline_bound != "LATENCY_BOUND"
    assert roofline_time_us < MI300X_MEM_LATENCY_US


def test_warn_if_missing_mem_latency_emits_once():
    arch = {"mem_bw_gbps": MI300X_MEM_BW_GBPS}
    with pytest.warns(UserWarning, match="mem_latency_us"):
        warn_if_missing_mem_latency(arch)
    warn_if_missing_mem_latency(arch)


def test_warn_if_missing_mem_latency_skips_when_present():
    import warnings as py_warnings

    arch = {
        "mem_bw_gbps": MI300X_MEM_BW_GBPS,
        "mem_latency_us": MI300X_MEM_LATENCY_US,
    }
    with py_warnings.catch_warnings(record=True) as caught:
        py_warnings.simplefilter("always")
        warn_if_missing_mem_latency(arch)
    assert len(caught) == 0
