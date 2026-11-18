###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Roles for graph leaves, read from what each op computes.

A leaf's ``role`` is first set from its attribute name and class name. For a
primitive op drawn out of a forward body those are synthetic (``@op_l720_c19_
matmul``, ``MatMul``), so every one of them comes out ``other`` -- including the
ops that are, in fact, attention. Giving them the role of the module that
holds them would be wrong: the same module also holds projections, norms, a
rotary embedding, and a KV compressor, none of which is attention.

This pass reads the dataflow instead. Attention is the computation

    scores = Q @ K^T  ->  (scale, mask, sink, max-subtract)  ->  softmax
                      ->  (slice, cast)  ->  weights @ V

so an op is attention when it sits on that path: a ``Softmax`` with a
``MatMul`` upstream and a ``MatMul`` downstream, joined through tensors of the
score family. The family test is what keeps the walk honest. Several other
tensors feed the softmax -- the mask, the rotary tables, an indexer's scores --
and one of them reaches a ``MatMul`` of its own. They do not share the score
tensor's leading (batch, heads) dimensions, so the walk does not enter them.

Nothing here is a name match. A router softmax (``Linear -> Softmax -> TopK``)
and a KV compressor's pooling softmax have no ``MatMul`` on both sides, so they
stay ``other``.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

ATTENTION_ROLE = "attention"
ROLE_BASIS_COMPUTATION = "computation"

# Functional matrix products. A ``Linear`` is a parameter projection, not the
# score or value product, so it neither starts nor ends the attention path.
_MATMUL_OPS = frozenset({"matmul", "batchmatmul", "bmm", "mm"})

# Roles a pass may overwrite. Anything already classified (embedding, head,
# norm, ...) is a different computation and keeps its role.
_REPLACEABLE_ROLES = frozenset({"", "other"})


def _attr(node: dict[str, Any], key: str) -> str | None:
    for item in node.get("attrs") or []:
        if isinstance(item, dict) and item.get("key") == key:
            value = item.get("value")
            return value if isinstance(value, str) else None
    return None


def _set_attr(node: dict[str, Any], key: str, value: str) -> None:
    attrs = node.setdefault("attrs", [])
    for item in attrs:
        if isinstance(item, dict) and item.get("key") == key:
            item["value"] = value
            return
    attrs.append({"key": key, "value": value})


def _dims(shape_text: str | None) -> list[str] | None:
    """Dimension texts of ``"[B, 128, S, S + n] bfloat16"``, or ``None``."""
    if not shape_text:
        return None
    start = shape_text.find("[")
    end = shape_text.find("]", start + 1)
    if start < 0 or end < 0:
        return None
    inner = shape_text[start + 1 : end].strip()
    if not inner:
        return None
    return [part.strip() for part in inner.split(",")]


def _family(node: dict[str, Any]) -> tuple[int, str, str] | None:
    """``(rank, batch dim, heads dim)`` of a node's output, or ``None``.

    Scores, scaled scores, masked scores, softmax weights and the weighted
    values all carry the same two leading dimensions. A mask broadcast over
    heads (``[B, 1, S, K]``), a rotary table, and an indexer's ``[B, S, 64, K]``
    do not.
    """
    dims = _dims(_attr(node, "output_shape"))
    if dims is None or len(dims) < 3:
        return None
    return len(dims), dims[0], dims[1]


def _is_matmul(node: dict[str, Any]) -> bool:
    return (_attr(node, "op_type") or "").lower() in _MATMUL_OPS


def _walk(
    start: str,
    neighbours: dict[str, set[str]],
    by_id: dict[str, dict[str, Any]],
    family: tuple[int, str, str],
) -> tuple[set[str], set[str]]:
    """Family ops reachable from *start*, and which of them are matmuls.

    A node with no ``op_type`` is a boundary port the graph draws between
    frames; it carries the tensor through and is never itself an op, so the
    walk passes through it. A matmul ends the walk on its side: its operands
    are other tensors, and the walk must not run on into them.
    """
    visited: set[str] = set()
    matmuls: set[str] = set()
    seen: set[str] = {start}
    stack = [start]
    while stack:
        current = stack.pop()
        for nxt in neighbours.get(current, ()):
            if nxt in seen:
                continue
            seen.add(nxt)
            node = by_id[nxt]
            if not _attr(node, "op_type"):
                stack.append(nxt)
                continue
            if _family(node) != family:
                continue
            visited.add(nxt)
            if _is_matmul(node):
                matmuls.add(nxt)
                continue
            stack.append(nxt)
    return visited, matmuls


def attention_core_ids(nodes: list[dict[str, Any]]) -> set[str]:
    """Ids of the ops that compute attention, read from the dataflow."""
    by_id = {
        node["id"]: node
        for node in nodes
        if isinstance(node, dict) and isinstance(node.get("id"), str)
    }
    producers: dict[str, set[str]] = defaultdict(set)
    consumers: dict[str, set[str]] = defaultdict(set)
    for node_id, node in by_id.items():
        for edge in node.get("incomingEdges") or []:
            source = edge.get("sourceNodeId") if isinstance(edge, dict) else None
            if source in by_id:
                producers[node_id].add(source)
                consumers[source].add(node_id)

    core: set[str] = set()
    for node_id, node in by_id.items():
        if _attr(node, "op_type") != "Softmax":
            continue
        family = _family(node)
        if family is None:
            continue
        upstream, scores = _walk(node_id, producers, by_id, family)
        downstream, values = _walk(node_id, consumers, by_id, family)
        if not scores or not values:
            continue
        core.add(node_id)
        core |= upstream | downstream
    return core


def stamp_computation_roles(graph: dict[str, Any]) -> int:
    """Give each attention-computation leaf the ``attention`` role.

    Sets ``role_basis=computation`` beside it, so a consumer can tell a leaf
    that is one step of the attention computation from an opaque attention
    block that stands for the whole of it. Returns the number of leaves
    stamped.
    """
    nodes = [node for node in graph.get("nodes") or [] if isinstance(node, dict)]
    core = attention_core_ids(nodes)
    stamped = 0
    for node in nodes:
        if node.get("id") not in core:
            continue
        if (_attr(node, "role") or "") not in _REPLACEABLE_ROLES:
            continue
        _set_attr(node, "role", ATTENTION_ROLE)
        _set_attr(node, "role_basis", ROLE_BASIS_COMPUTATION)
        stamped += 1
    return stamped
