###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for TraceLens.util helpers."""

from TraceLens.util import TraceEventUtils


def test_trace_event_utils_split_by_field():
    k1 = {"cat": "kernel", "i": 1}
    c1 = {"cat": "cpu_op", "i": 2}
    k2 = {"cat": "kernel", "i": 3}
    grouped = TraceEventUtils.split_by_field([k1, c1, k2], "cat")
    assert {key: list(val) for key, val in grouped.items()} == {
        "kernel": [k1, k2],
        "cpu_op": [c1],
    }
