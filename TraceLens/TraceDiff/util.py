###############################################################################
# Copyright (c) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Shared constants and pure utility functions for TraceDiff."""

from ..util import TraceEventUtils, normalize_name_for_comparison

_TraceKeys = TraceEventUtils.TraceKeys
_UID = _TraceKeys.UID
_NAME = _TraceKeys.Name
_CATEGORY = _TraceKeys.Category
_TS = _TraceKeys.TimeStamp


def _sort_by_ts(nodes):
    """Sort nodes by timestamp, return their UIDs."""
    return [n[_UID] for n in sorted(nodes, key=lambda n: n.get(_TS, 0))]


def _get_name_node(node, strip_details=False):
    """Get the normalized comparison name directly from a node dict."""
    if node is None:
        return None
    name = node.get(_NAME)
    return normalize_name_for_comparison(name, strip_details) if name else None


def _list_to_tuple(obj):
    """Recursively convert lists to tuples for hashability."""
    if isinstance(obj, list):
        return tuple(_list_to_tuple(item) for item in obj)
    return obj


def _get_node_arg(node, key):
    """Get an arg value from a node, converting lists to tuples. Returns '' if missing."""
    val = node.get("args", {}).get(key)
    if val is not None:
        return _list_to_tuple(val)
    return ""


def _is_gpu_path(node):
    """Return True if node is on the GPU path."""
    if node is None:
        return False
    return not node.get("non_gpu_path", False)


def _is_kernel(node):
    """Return True if node is a GPU kernel or memcpy."""
    return node.get(_CATEGORY) in ("kernel", "gpu_memcpy")


