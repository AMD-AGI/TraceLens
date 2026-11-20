###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Stamp a leaf's role from the category the perf report would give it.

The report categorizes a torch op with ``categorize_torch_op``. A leaf whose
report category is attention gets ``role=attention`` and
``role_basis=computation``, so the roofline projection can tell one step of
that computation from an opaque attention block. Nothing here matches kernel
names or walks the graph.
"""

from __future__ import annotations

from typing import Any

from TraceLens.PerfModel.torch_op_mapping import categorize_torch_op

ATTENTION_ROLE = "attention"
ROLE_BASIS_COMPUTATION = "computation"

# Categories ``categorize_torch_op`` returns for attention. The projection
# still speaks the single ``attention`` role.
_ATTENTION_CATEGORIES = frozenset({"SDPA_fwd", "SDPA_bwd", "InferenceAttention"})
_REPLACEABLE_ROLES = {"", "other"}


def _attr(node: dict[str, Any], key: str) -> str | None:
    for attr in node.get("attrs") or []:
        if isinstance(attr, dict) and attr.get("key") == key:
            value = attr.get("value")
            return None if value is None else str(value)
    return None


def _set_attr(node: dict[str, Any], key: str, value: str) -> None:
    attrs = node.setdefault("attrs", [])
    for attr in attrs:
        if isinstance(attr, dict) and attr.get("key") == key:
            attr["value"] = value
            return
    attrs.append({"key": key, "value": value})


def _report_row(node: dict[str, Any]) -> dict[str, Any]:
    """The row ``categorize_torch_op`` reads: a name, and kernel details."""
    name = node.get("label") or _attr(node, "op_type") or ""
    row: dict[str, Any] = {"name": name}
    kernel = _attr(node, "kernel_name")
    if kernel:
        row["kernel_details"] = [{"name": kernel}]
    return row


def stamp_computation_roles(graph: dict[str, Any]) -> int:
    """Give each attention leaf the ``attention`` role.

    The category is ``categorize_torch_op``, the same function the perf report
    uses. Sets ``role_basis=computation`` beside the role, so a consumer can
    tell a leaf that is one step of the attention computation from an opaque
    attention block. Returns the number of leaves stamped.
    """
    stamped = 0
    for node in graph.get("nodes") or []:
        if not isinstance(node, dict):
            continue
        if (_attr(node, "role") or "") not in _REPLACEABLE_ROLES:
            continue
        if categorize_torch_op(_report_row(node)) not in _ATTENTION_CATEGORIES:
            continue
        _set_attr(node, "role", ATTENTION_ROLE)
        _set_attr(node, "role_basis", ROLE_BASIS_COMPUTATION)
        stamped += 1
    return stamped
