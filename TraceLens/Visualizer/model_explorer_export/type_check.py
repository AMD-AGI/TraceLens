###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Type-check operation nodes in an exported Model Explorer graph.

A post-processing pass that reads the PyTorch-profiler-style operand descriptions
(``op_type`` + ``input_shapes`` / ``input_types`` / ``concrete_inputs`` attrs,
attached by ``merge._annotate_op_input_signatures``) and validates each operation
whose operand contract can be checked. It emits **warnings**, never errors: an
export is still produced, but a warning flags a wiring/extraction fidelity bug
(e.g. the vision-rotary ``unsqueeze`` that was fed a spurious ``hidden_states``
tensor as a second operand). The fix belongs upstream in the extractor/wiring, not
in suppressing the warning.

Only operations with a known operand contract are checked; anything else is
skipped (no false positives), satisfying "type-check the ops that *can* be
type-checked".
"""

from __future__ import annotations

import json
import logging
from typing import Any

from TraceLens.ModelUtils.shape_inference import _normalize_op_name

_log = logging.getLogger(__name__)

# Ops that take exactly one tensor operand plus a scalar ``dim`` argument. Feeding
# them a second *tensor* operand (a graph edge) is the classic mis-wiring the
# owner flagged: ``position_ids[..., None]`` must not also receive ``hidden_states``.
_SINGLE_TENSOR_AXIS_OPS = frozenset({"unsqueeze", "squeeze", "select"})

# Ops whose visible tensor operands must all share the same rank (they concatenate
# or stack along one axis). A hidden constant/buffer operand is never a concat
# input, so a rank disagreement among visible operands is a real bug.
_SAME_RANK_OPS = frozenset({"cat", "concat", "concatenate", "stack"})


def _load_list(node: dict[str, Any], key: str) -> list[Any] | None:
    for attr in node.get("attrs", []):
        if attr.get("key") == key:
            value = attr.get("value")
            if not isinstance(value, str):
                return None
            try:
                parsed = json.loads(value)
            except (ValueError, TypeError):
                return None
            return parsed if isinstance(parsed, list) else None
    return None


def _op_type(node: dict[str, Any]) -> str:
    for attr in node.get("attrs", []):
        if attr.get("key") == "op_type":
            return str(attr.get("value") or "")
    return str(node.get("label") or "")


def _is_constant_node(node: dict[str, Any]) -> bool:
    for attr in node.get("attrs", []):
        if attr.get("key") == "constant":
            return attr.get("value") == "true"
    return False


# Operand types that are not real activation tensors: a ``Scalar`` (a ``dim`` /
# scalar arg) or a ``Constant`` (a learned weight / buffer / constant operand,
# which is drawn only in the constants-visible view and filtered out otherwise).
_NON_ACTIVATION_TYPES = frozenset({"Scalar", "Constant"})


def _check_node(node: dict[str, Any]) -> list[str]:
    """Return type-check warning lines for a single operation node (possibly none)."""
    op = _normalize_op_name(_op_type(node))
    if not op:
        return []
    # A constant node (a learned weight / buffer read, filtered out of the drawn
    # graph) is not part of the activation dataflow, so its operand contract is
    # not meaningful to check.
    if _is_constant_node(node):
        return []
    input_types = _load_list(node, "input_types")
    input_shapes = _load_list(node, "input_shapes")
    if input_types is None:
        return []
    tensor_count = sum(1 for t in input_types if t not in _NON_ACTIVATION_TYPES)
    node_id = node.get("id")
    warnings: list[str] = []

    if op in _SINGLE_TENSOR_AXIS_OPS and tensor_count != 1:
        # These ops take exactly one activation tensor plus a scalar ``dim``. More
        # than one tensor operand means a caller argument was mis-wired onto the op
        # (e.g. ``position_ids[..., None]`` also receiving ``hidden_states``); zero
        # means the sole activation operand went missing. Constants/buffers are now
        # first-class ``"Constant"`` operands excluded from this count, so a hidden
        # buffer no longer masks a missing activation -- either extreme is a bug.
        warnings.append(
            f"{node_id} [{op}]: takes exactly 1 tensor operand + a scalar dim, but "
            f"got {tensor_count} tensor operands (input_types={input_types}). The "
            f"activation operand must be wired as exactly one tensor edge, with any "
            f"scalar dim argument kept off the edges."
        )

    if op in _SAME_RANK_OPS and input_shapes is not None:
        ranks = {
            len(shape)
            for shape, typ in zip(input_shapes, input_types)
            if typ not in _NON_ACTIVATION_TYPES and isinstance(shape, list)
        }
        if len(ranks) > 1:
            warnings.append(
                f"{node_id} [{op}]: tensor operands disagree on rank {sorted(ranks)} "
                f"(input_shapes={input_shapes}); concat/stack operands must share "
                f"rank."
            )

    return warnings


def type_check_graph_nodes(nodes: list[dict[str, Any]]) -> list[str]:
    """Type-check every checkable operation node; return + log warning lines.

    Never raises. An empty return means every checkable op's operand contract held.
    """
    warnings: list[str] = []
    for node in nodes:
        warnings.extend(_check_node(node))
    for line in warnings:
        _log.warning("graph type-check: %s", line)
    return warnings
