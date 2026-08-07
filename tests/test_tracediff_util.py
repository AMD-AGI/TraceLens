###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for TraceLens.TraceDiff.util pure helpers."""

from TraceLens.TraceDiff.util import (
    _get_name_node,
    _get_node_arg,
    _is_gpu_path,
    _is_kernel,
    _list_to_tuple,
    _normalize_name_for_comparison,
    _sort_by_ts,
)
from TraceLens.util import TraceEventUtils

_TK = TraceEventUtils.TraceKeys


def _node(name="op", ts=0, cat="cpu_op", non_gpu_path=False, args=None):
    return {
        _TK.UID: 0,
        _TK.Name: name,
        _TK.TimeStamp: ts,
        _TK.Category: cat,
        "non_gpu_path": non_gpu_path,
        "args": args or {},
    }


class TestSortByTs:
    def test_orders_by_timestamp(self):
        nodes = [
            {_TK.UID: 3, _TK.TimeStamp: 30},
            {_TK.UID: 1, _TK.TimeStamp: 10},
            {_TK.UID: 2, _TK.TimeStamp: 20},
        ]
        assert _sort_by_ts(nodes) == [1, 2, 3]

    def test_missing_timestamp_defaults_to_zero(self):
        node = {_TK.UID: 7, _TK.Name: "x"}
        assert _sort_by_ts([node]) == [7]


class TestGetNameNode:
    def test_none_node_returns_none(self):
        assert _get_name_node(None) is None

    def test_missing_name_returns_none(self):
        assert _get_name_node({_TK.UID: 1}) is None

    def test_normalizes_name(self):
        node = _node(name="foo 0xDEADBEEF")
        assert _get_name_node(node) == "foo 0xXXXX"

    def test_strip_details(self):
        node = _node(name="/proj/layer.py(42): matmul : extra")
        assert _get_name_node(node, strip_details=True) == "/proj/layer.py: matmul "


class TestListToTuple:
    def test_nested_lists_become_tuples(self):
        assert _list_to_tuple([1, [2, [3]]]) == (1, (2, (3,)))

    def test_non_list_unchanged(self):
        assert _list_to_tuple("abc") == "abc"


class TestGetNodeArg:
    def test_missing_arg_returns_empty_string(self):
        assert _get_node_arg(_node(), "Input Dims") == ""

    def test_list_arg_converted_to_tuple(self):
        node = _node(args={"Input Dims": [[1, 2], [3, 4]]})
        assert _get_node_arg(node, "Input Dims") == ((1, 2), (3, 4))


class TestIsGpuPath:
    def test_none_node_is_not_gpu_path(self):
        assert _is_gpu_path(None) is False

    def test_non_gpu_path_flag(self):
        assert _is_gpu_path(_node(non_gpu_path=True)) is False
        assert _is_gpu_path(_node(non_gpu_path=False)) is True


class TestIsKernel:
    def test_kernel_categories(self):
        assert _is_kernel(_node(cat="kernel")) is True
        assert _is_kernel(_node(cat="gpu_memcpy")) is True
        assert _is_kernel(_node(cat="cpu_op")) is False


class TestNormalizeNameForComparison:
    def test_none_name(self):
        assert _normalize_name_for_comparison(None) is None

    def test_hex_addresses_replaced(self):
        name = "launch 0xabc123 at 0xdead"
        assert _normalize_name_for_comparison(name) == "launch 0xXXXX at 0xXXXX"

    def test_python_line_numbers_stripped(self):
        name = "/src/train.py(128): forward"
        assert _normalize_name_for_comparison(name) == "/src/train.py: forward"

    def test_kernel_launch_equivalents(self):
        assert (
            _normalize_name_for_comparison("hipModuleLaunchKernel")
            == "__kernel_launch__"
        )
        assert _normalize_name_for_comparison("cuLaunchKernel") == "__kernel_launch__"

    def test_strip_details_removes_suffix(self):
        name = "/home/user/proj/layer.py(99): matmul : detail"
        assert (
            _normalize_name_for_comparison(name, strip_details=True)
            == "/home/user/proj/layer.py: matmul "
        )
