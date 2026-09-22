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

from TraceLens.ModelUtils.shape_inference import _normalize_op_name, _operand_ceiling

_log = logging.getLogger(__name__)

# ``_operand_ceiling`` (dynamic operand-arity resolution via ``inspect``/aten
# schema introspection, never a hardcoded op-name list) lives in
# ``shape_inference.py`` so ``computation_graph.py``'s wiring pass can share it
# without a Visualizer -> ModelUtils -> Visualizer import cycle.


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


def _raw_op(node: dict[str, Any]) -> str:
    """The underlying torch op name recovered from the model's forward source
    (stamped by the extractor as the ``raw_op`` attr), or "" if none. This is the
    name the display label discards; the arity check keys on it, never on a list."""
    for attr in node.get("attrs", []):
        if attr.get("key") == "raw_op":
            return str(attr.get("value") or "")
    return ""


def _is_constant_node(node: dict[str, Any]) -> bool:
    for attr in node.get("attrs", []):
        if attr.get("key") == "constant":
            return attr.get("value") == "true"
    return False


# Structured shape-change declarations the extractor stamps on a materialized
# narrowing/indexing ``Slice`` op: ``select_dim`` drops an axis, ``resize_dim``
# sets an axis to a folded constant width, ``shape_slice`` narrows an axis by
# arithmetic over the operand's own shape. Each GUARANTEES the op's output shape
# differs from its input -- so an output that still equals the input means the
# declared narrowing was lost. The purely-descriptive ``slice:`` detail (a
# symbolic, non-foldable bound) is intentionally excluded: shape inference cannot
# size it and legitimately passes the shape through.
_SHAPE_CHANGE_DETAIL_PREFIXES = ("select_dim:", "resize_dim:", "shape_slice:")


def _details_lines(node: dict[str, Any]) -> list[str]:
    for attr in node.get("attrs", []):
        if attr.get("key") == "details":
            value = attr.get("value")
            if isinstance(value, str):
                return [line.strip() for line in value.splitlines() if line.strip()]
    return []


def _output_shape_dims(node: dict[str, Any]) -> list[str] | None:
    """The output shape's per-axis dim tokens (dtype suffix stripped), or None."""
    for attr in node.get("attrs", []):
        if attr.get("key") == "output_shape":
            value = attr.get("value")
            if not isinstance(value, str):
                return None
            start = value.find("[")
            end = value.find("]", start + 1)
            if start == -1 or end == -1:
                return None
            inner = value[start + 1 : end].strip()
            if not inner:
                return []
            return [tok.strip() for tok in inner.split(",")]
    return None


def _operand_is_tensor(op_type: str, shape: Any) -> bool:
    """Whether an operand counts as a real tensor input for the arity check.

    A ``"Scalar"`` operand (a ``dim`` / scalar argument) never counts. A
    ``"Constant"`` operand (a learned weight / buffer, present in the built graph
    before render-filtering) counts only when it is rank >= 1 -- a hidden 1-D
    constant tensor wired onto a single-tensor op is exactly the mis-wiring the
    owner flagged. A rank-0 constant (a folded scalar) does not count. Any other
    type is a real activation dtype and counts.
    """
    if op_type == "Scalar":
        return False
    if op_type == "Constant":
        return isinstance(shape, list) and len(shape) >= 1
    return True


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
    shapes = input_shapes if isinstance(input_shapes, list) else []
    tensor_count = sum(
        1
        for i, t in enumerate(input_types)
        if _operand_is_tensor(t, shapes[i] if i < len(shapes) else None)
    )
    node_id = node.get("id")
    warnings: list[str] = []

    # Resolve the op's operand contract dynamically from its real function
    # parameters (``inspect`` first, then the aten schema) keyed on the raw op
    # name recovered from the model's own forward source -- never a static list.
    ceiling, variadic = _operand_ceiling(_raw_op(node))

    if variadic:
        # An unbounded tensor-list op (``cat``/``stack``): any number of tensor
        # operands is legal, but they must all share rank (they join along one
        # axis). A hidden constant is never a concat input, so a rank disagreement
        # among the counted tensor operands is a real bug.
        if input_shapes is not None:
            ranks = {
                len(shapes[i])
                for i, t in enumerate(input_types)
                if i < len(shapes)
                and isinstance(shapes[i], list)
                and _operand_is_tensor(t, shapes[i])
            }
            if len(ranks) > 1:
                warnings.append(
                    f"{node_id} [{op}]: tensor operands disagree on rank "
                    f"{sorted(ranks)} (input_shapes={input_shapes}); an op that "
                    f"joins a list of tensors along one axis must receive operands "
                    f"of equal rank."
                )
    elif ceiling is not None:
        if tensor_count == 0:
            # A bounded op consumes at least one tensor (you cannot ``unsqueeze``
            # or ``transpose`` nothing). Zero counted tensor operands means the op's
            # sole activation edge went missing -- an upstream wiring/pruning bug.
            warnings.append(
                f"{node_id} [{op}]: its parameters take {ceiling} tensor "
                f"operand(s), but 0 tensor operands are wired "
                f"(input_types={input_types}). The op's activation input went "
                f"missing -- an upstream edge was dropped or mis-typed."
            )
        elif tensor_count > ceiling:
            # The op's parameters bound it to ``ceiling`` tensor operands, but more
            # tensor edges are wired -- a caller argument was mis-wired onto the op
            # (e.g. two mutually-exclusive branches both feeding a ``transpose``, or
            # a hidden constant tensor added to a single-tensor axis op).
            warnings.append(
                f"{node_id} [{op}]: its parameters take at most {ceiling} tensor "
                f"operand(s), but {tensor_count} tensor operands are wired "
                f"(input_types={input_types}). Extra tensor edges mean a caller "
                f"argument was mis-wired onto the op; scalar/axis arguments must be "
                f"kept off the edges."
            )

    # A shape-changing slice/select/resize op whose output shape still equals its
    # input shape did not actually change the shape -- the declared narrowing was
    # dropped in shape inference (the ``rotate_half`` no-op slice class of bug).
    change_details = [
        line
        for line in _details_lines(node)
        if line.startswith(_SHAPE_CHANGE_DETAIL_PREFIXES)
    ]
    if change_details:
        out_dims = _output_shape_dims(node)
        tensor_inputs = [
            shapes[i]
            for i, t in enumerate(input_types)
            if i < len(shapes)
            and isinstance(shapes[i], list)
            and _operand_is_tensor(t, shapes[i])
        ]
        if out_dims is not None and len(tensor_inputs) == 1:
            in_dims = [str(dim) for dim in tensor_inputs[0]]
            if in_dims == out_dims:
                warnings.append(
                    f"{node_id} [{op}]: declares a shape change "
                    f"({'; '.join(change_details)}) but its output shape {out_dims} "
                    f"equals its input shape {in_dims} -- the narrowing was lost; "
                    f"shape inference did not apply the declared slice/select."
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


# Boundary tiles a value flows out of / into at a namespace edge -- mirrors
# ``merge._OUTPUT_BOUNDARY_SYNTHETIC`` / ``_INPUT_BOUNDARY_SYNTHETIC``.
_OUTPUT_BOUNDARY_SYNTHETIC = frozenset({"@output", "@output_mirror"})
_INPUT_BOUNDARY_SYNTHETIC = frozenset({"@input", "@input_mirror", "@kernel_port_in"})


def _port_label(node: dict[str, Any]) -> str:
    return _node_attr_value(node, "port_label") or str(node.get("label", ""))


def _source_port_label(source: dict[str, Any], output_id: Any) -> str:
    """The producer's *specific* output-port label (per port, not tile aggregate)."""
    for port in source.get("outputsMetadata", []) or []:
        if str(port.get("id")) == str(output_id):
            for attr in port.get("attrs", []) or []:
                if attr.get("key") == "port_label":
                    return str(attr.get("value"))
    return _port_label(source)


def _boundary_owner_namespace(node_id: str) -> str:
    """The module namespace that owns a boundary tile -- its id minus the trailing
    ``/@...`` boundary token. A same-name ``@output -> @input`` crossing between
    *different* owners is a legitimate module entry/exit; only a redundant pair
    within the *same* owner should have been folded (mirrors the same-namespace
    guard in ``merge._collapse_same_name_boundary_passthroughs``)."""
    idx = node_id.rfind("/@")
    return node_id[:idx] if idx != -1 else ""


def _has_incoming(node: dict[str, Any]) -> bool:
    return bool(node.get("incomingEdges"))


def _is_top_level_model_input(node: dict[str, Any]) -> bool:
    """True for a genuine top-level model-input boundary (a legitimate graph source).

    A synthetic ``@input`` at the root scope -- ``@input`` (tokenized text),
    ``@vision_input`` (image patches), the image-placeholder mask, or a dedicated
    ``@input:<param>`` model parameter boundary -- is an entry point with no
    producer, so it is exempt from the I2 no-source check. A *namespaced*
    ``@input`` / ``@input:<param>`` (``decoder/self_attn/@input:position_embeddings``)
    is a module boundary that must be fed by its real producer; it is NOT exempt.
    Root scope is identified structurally: empty namespace and no ``/`` in the id.
    """
    if not _is_synthetic_input(node):
        return False
    if node.get("namespace", ""):
        return False
    return "/" not in str(node.get("id", ""))


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
    - **I4 same-name boundary passthrough** -- no input-family boundary tile
      (``@input``/``@input_mirror``/``@kernel_port_in``) is fed solely by an
      output-family boundary tile (``@output``/``@output_mirror``) carrying the
      *identical* tensor name **within the same owning module namespace**. Such a
      same-level pair is one untransformed value rendered as two tiles and should
      have been folded by ``merge._collapse_same_name_boundary_passthroughs``; a
      survivor means that pass failed to fire. A same-name crossing between
      *different* owners (a real module entry/exit) is expected and never flagged,
      as are renamed crossings (different port labels).
    """
    consumed = {
        str(e.get("sourceNodeId"))
        for n in nodes
        for e in n.get("incomingEdges", []) or []
        if e.get("sourceNodeId") is not None
    }
    by_id = {str(n.get("id")): n for n in nodes}
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

        # I2 no-source / orphan. A top-level model input is a legitimate graph
        # source; a namespaced module ``@input:<param>`` boundary is not -- it must
        # be fed by its real producer, so a floating one (the kwargs-forwarded
        # decoder invariant that reached no source) is flagged like any orphan.
        exempt_source = (
            _is_top_level_model_input(node)
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

        # I4 same-name boundary passthrough (should be folded away).
        if _node_attr_value(node, "synthetic") in _INPUT_BOUNDARY_SYNTHETIC:
            incoming = node.get("incomingEdges", []) or []
            if len(incoming) == 1:
                source = by_id.get(str(incoming[0].get("sourceNodeId")))
                if (
                    source is not None
                    and _node_attr_value(source, "synthetic")
                    in _OUTPUT_BOUNDARY_SYNTHETIC
                ):
                    name = _port_label(node)
                    src_name = _source_port_label(
                        source, incoming[0].get("sourceNodeOutputId", "0")
                    )
                    same_owner = _boundary_owner_namespace(
                        node_id
                    ) == _boundary_owner_namespace(str(source.get("id", "")))
                    if name == src_name and same_owner:
                        warnings.append(
                            f"I4 same-name-passthrough{tag}: {node_id} "
                            f"[label={node.get('label')!r}] is fed solely by same-name "
                            f"output boundary {source.get('id')!r}; collapse the "
                            f"redundant tile in _collapse_same_name_boundary_passthroughs."
                        )

    for line in warnings:
        _log.warning("graph integrity: %s", line)
    return warnings
