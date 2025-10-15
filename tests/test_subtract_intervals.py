###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from TraceLens.TreePerf import GPUEventAnalyser as GPUEA


def test_empty_intervals_to_subtract():
    intervals = [(10, 20)]
    intervals_to_subtract = []
    expected_result = [(10, 20)]

    result = GPUEA.subtract_intervalsA_from_B(intervals_to_subtract, intervals)
    assert result == expected_result


def test_subtract_from_empty_intervals():
    intervals = []
    intervals_to_subtract = [(5, 15)]
    expected_result = []

    result = GPUEA.subtract_intervalsA_from_B(intervals_to_subtract, intervals)
    assert result == expected_result


def test_subtract_empty_from_empty():
    intervals = []
    intervals_to_subtract = []
    expected_result = []

    result = GPUEA.subtract_intervalsA_from_B(intervals_to_subtract, intervals)
    assert result == expected_result


def test_one_interval_to_subtract():
    intervals = [(10, 20)]
    intervals_to_subtract = [(15, 16)]
    expected_result = [(10, 15), (16, 20)]

    result = GPUEA.subtract_intervalsA_from_B(intervals_to_subtract, intervals)
    assert result == expected_result


def test_two_intervals_to_subtract():
    intervals = [(10, 20)]
    intervals_to_subtract = [(15, 16), (17, 18)]
    expected_result = [(10, 15), (16, 17), (18, 20)]

    result = GPUEA.subtract_intervalsA_from_B(intervals_to_subtract, intervals)
    assert result == expected_result


def test_consecutive_intervals():
    intervals = [(10, 20)]
    intervals_to_subtract = [(15, 16), (16, 18)]
    expected_result = [(10, 15), (18, 20)]

    result = GPUEA.subtract_intervalsA_from_B(intervals_to_subtract, intervals)
    assert result == expected_result


def test_subtraction_before_interval():
    intervals = [(10, 20)]
    intervals_to_subtract = [(9, 12)]
    expected_result = [(12, 20)]

    result = GPUEA.subtract_intervalsA_from_B(intervals_to_subtract, intervals)
    assert result == expected_result


def test_subtraction_after_interval():
    intervals = [(10, 20)]
    intervals_to_subtract = [(15, 22)]
    expected_result = [(10, 15)]

    result = GPUEA.subtract_intervalsA_from_B(intervals_to_subtract, intervals)
    assert result == expected_result


def test_overlapping_subtraction():
    intervals = [(10, 20)]
    intervals_to_subtract = [(15, 17), (16, 18)]
    expected_result = [(10, 15), (18, 20)]

    result = GPUEA.subtract_intervalsA_from_B(intervals_to_subtract, intervals)
    assert result == expected_result


def test_perfectly_overlapping_subtraction():
    intervals = [(10, 20)]
    intervals_to_subtract = [(10, 20)]
    expected_result = []

    result = GPUEA.subtract_intervalsA_from_B(intervals_to_subtract, intervals)
    assert result == expected_result


def test_subtracting_long_interval():
    intervals = [(10, 20)]
    intervals_to_subtract = [(0, 30)]
    expected_result = []

    result = GPUEA.subtract_intervalsA_from_B(intervals_to_subtract, intervals)
    assert result == expected_result


def test_subtracting_from_two_intervals():
    intervals = [(10, 20), (30, 40)]
    intervals_to_subtract = [(15, 35)]
    expected_result = [(10, 15), (35, 40)]

    result = GPUEA.subtract_intervalsA_from_B(intervals_to_subtract, intervals)
    assert result == expected_result
