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
import re
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


# --------------------------------------------------------------------------- #
# Graph-integrity checks (I1 dead-node, I2 no-source/orphan, I3 constant sound).
#
# These are structural invariants over the whole node list, orthogonal to the
# per-op operand type-check above. Like it, they emit WARNINGS only -- an
# offender is a wiring/extraction fidelity bug to fix upstream, never a reason to
# fail the export. Self-contained (no ``merge`` import) to avoid a circular
# dependency; the predicates below mirror ``merge._is_synthetic_input`` /
# ``_is_synthetic_output`` / ``_node_attr`` exactly.
# --------------------------------------------------------------------------- #


def _node_attr_value(node: dict[str, Any], key: str) -> str | None:
    for attr in node.get("attrs", []):
        if attr.get("key") == key:
            value = attr.get("value")
            if isinstance(value, str):
                return value
    return None


def _is_synthetic_input(node: dict[str, Any]) -> bool:
    node_id = str(node.get("id", ""))
    if node_id == "@input" or re.search(r"/@input(?::|$)", node_id):
        return True
    return _node_attr_value(node, "synthetic") == "@input"


def _is_synthetic_output(node: dict[str, Any]) -> bool:
    node_id = str(node.get("id", ""))
    if node_id == "@output" or node_id.endswith("/@output"):
        return True
    return _node_attr_value(node, "synthetic") == "@output"


def _is_loop_carried(node: dict[str, Any]) -> bool:
    return _node_attr_value(node, "synthetic") == "@loop_carried"


def _has_incoming(node: dict[str, Any]) -> bool:
    return bool(node.get("incomingEdges"))


def _is_float_dtype(type_str: Any) -> bool:
    """True for a floating-point tensor dtype (real activation), not an index/mask.

    ``input_types`` entries are either a raw torch dtype string (``float16``,
    ``bfloat16``, ``int64``, ``bool`` ...) or a synthetic operand kind
    (``"Constant"`` / ``"Scalar"``). Only the floating-point tensor dtypes carry
    real activation data; integers are indices/masks and the synthetic kinds are
    non-activation, so both are legitimate operands of a hidden constant closure.
    """
    if not isinstance(type_str, str):
        return False
    lowered = type_str.lower()
    return any(token in lowered for token in ("float", "bfloat", "half", "double"))


def integrity_check_graph_nodes(nodes: list[dict[str, Any]], *, label: str = "") -> list[str]:
    """Check three structural invariants; return + log warning lines. Never raises.

    - **I1 dead-node** -- every non-exempt node's value is consumed by some other
      node. Exempt: synthetic ``@input``/``@output`` boundaries, ``@loop_carried``
      tiles, and the top-level ``@output`` (legitimate sinks). A dead node is a
      wiring bug: a real tensor was extracted but its consumer edge never rebuilt.
    - **I2 no-source / orphan** -- every node that is not a boundary
      (``@input``/``@output``), not a constant leaf (``constant`` tag with no
      inputs), and not a ``@loop_carried_in`` (seed/back-edge fed) has >=1 incoming
      edge. Catches sourceless ops (the ``self.base.split`` regression) and
      orphaned passthrough tiles.
    - **I3 constant soundness** -- no ``constant``-tagged node carries a real
      floating-point *activation* operand. A genuinely constant node reads only
      other constants (learned weights / buffers, annotated ``"Constant"``),
      integer *indices* (a dynamic weight/expert selection like
      ``self.gate_up_proj[expert_idx]``), or scalar args -- never a raw float
      tensor. A float dtype in ``input_types`` means a real activation is being
      hidden at render, i.e. the node is mistagged. This is dtype-local (no source
      walk), so it correctly spares weight-selection gathers whose only wired
      operand is an int64 routing index while still catching a float activation
      wrongly folded into the constant closure.
    """
    consumed = {
        str(e.get("sourceNodeId"))
        for n in nodes
        for e in n.get("incomingEdges", []) or []
        if e.get("sourceNodeId") is not None
    }
    warnings: list[str] = []
    tag = f" [{label}]" if label else ""

    for node in nodes:
        node_id = str(node.get("id", ""))
        is_const = _node_attr_value(node, "constant") == "true"

        # I1 dead-node.
        exempt_sink = (
            _is_synthetic_input(node)
            or _is_synthetic_output(node)
            or _is_loop_carried(node)
            or node_id == "@output"
        )
        if not exempt_sink and node_id not in consumed:
            warnings.append(
                f"I1 dead-node{tag}: {node_id} [label={node.get('label')!r}, "
                f"constant={is_const}] produces a value nothing consumes; rebuild "
                f"its missing consumer edge upstream (do not prune)."
            )

        # I2 no-source / orphan.
        exempt_source = (
            _is_synthetic_input(node)
            or _is_synthetic_output(node)
            or (is_const and not _has_incoming(node))  # materialized constant leaf
            or "@loop_carried_in:" in node_id  # seeded + back-edge fed
        )
        if not exempt_source and not _has_incoming(node):
            warnings.append(
                f"I2 no-source{tag}: {node_id} [label={node.get('label')!r}, "
                f"constant={is_const}] has no incoming edge; it is orphaned/sourceless "
                f"-- wire it to its real producer upstream."
            )

        # I3 constant soundness: a constant node must carry no raw float activation
        # operand (only "Constant"/"Scalar"/integer-index dtypes).
        if is_const:
            input_types = _load_list(node, "input_types") or []
            float_operands = [t for t in input_types if _is_float_dtype(t)]
            if float_operands:
                warnings.append(
                    f"I3 constant-unsound{tag}: {node_id} [label={node.get('label')!r}] "
                    f"is tagged constant but carries floating-point activation "
                    f"operand(s) {float_operands} (input_types={input_types}); a real "
                    f"activation is being hidden at render -- fix the tagging upstream, "
                    f"not the render."
                )

    for line in warnings:
        _log.warning("graph integrity: %s", line)
    return warnings
