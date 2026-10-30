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
import math
import re
from typing import Any

from TraceLens.ModelUtils.ast_analyze import PHI_CLASS, PHI_LABEL
from TraceLens.ModelUtils.kernel_pipeline import _ALLOCATION_CALLS
from TraceLens.ModelUtils.shape_inference import (
    _normalize_op_name,
    _operand_ceiling,
    _required_tensor_operand_names,
    op_cannot_change_element_count,
    op_cannot_change_shape,
)

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


def _is_zero_operand_source(node: dict[str, Any]) -> bool:
    """True for a tensor-*generating* op that reads no tensor operand.

    ``torch.arange(n)``/``torch.zeros(shape)`` fabricate a tensor from host
    scalars alone: their operand ceiling -- resolved from the real op parameters,
    never the display name -- is zero. Such an op is a genuine dataflow source
    with no producer upstream, so (like a top-level model input) it is exempt from
    the I2 no-source check rather than being flagged as an orphan."""
    ceiling, variadic = _operand_ceiling(_raw_op(node))
    return ceiling == 0 and not variadic


# Structured shape-change declarations the extractor stamps on a materialized
# narrowing/indexing ``Slice`` op: ``select_dim`` drops an axis, ``resize_dim``
# sets an axis to a folded constant width, ``shape_slice`` narrows an axis by
# arithmetic over the operand's own shape. Each GUARANTEES the op's output shape
# differs from its input -- so an output that still equals the input means the
# declared narrowing was lost. The purely-descriptive ``slice:`` detail (a
# symbolic, non-foldable bound) is intentionally excluded: shape inference cannot
# size it and legitimately passes the shape through.
_SHAPE_CHANGE_DETAIL_PREFIXES = ("select_dim:", "resize_dim:", "shape_slice:")

# The ``shape:`` detail a ``view``/``reshape``/``expand`` stamps carries its
# declared target shape (``ast_analyze`` renders the reshape args here, and only
# these ops emit it). Unlike the narrowing details above, a reshape does NOT
# guarantee a shape change -- a contiguous-forcing ``x.reshape(x.shape)`` is a
# legitimate same-shape reshape -- so an output that equals the input is only a
# defect when the DECLARED target provably differs. That proof is only available
# when the declared target is a *fully concrete* shape (see ``_concrete_int_dims``).
_RESHAPE_SHAPE_DETAIL_PREFIX = "shape:"


def _concrete_int_dims(tokens: list[str] | None) -> list[int] | None:
    """The tokens as a fully concrete shape (all non-negative int literals), or None.

    A concrete shape has every axis a plain non-negative integer. Any symbolic dim
    (``B*S``, ``self.head_dim``, ``number_of_pools``), a ``-1`` reshape placeholder,
    a starred prefix (``*orig_shape``), an arithmetic expression, or an empty token
    makes the shape non-concrete -- shape inference may then legitimately resolve it
    to the operand's own shape, so a no-op there is not provably a defect. Returning
    None for any such shape keeps the reshape no-op check free of false positives.
    """
    if not tokens:
        return None
    out: list[int] = []
    for token in tokens:
        token = str(token).strip()
        if not token:
            return None
        try:
            value = int(token)
        except ValueError:
            return None
        if value < 0:  # excludes the ``-1`` unfolded-axis placeholder
            return None
        out.append(value)
    return out


def _declared_reshape_dims(node: dict[str, Any]) -> list[str] | None:
    """The declared target-shape tokens of a ``view``/``reshape``/``expand``'s
    ``shape:`` detail (``shape: -1, 6144`` -> ``["-1", "6144"]``), or None if the
    node carries no such detail."""
    for line in _details_lines(node):
        if line.startswith(_RESHAPE_SHAPE_DETAIL_PREFIX):
            body = line[len(_RESHAPE_SHAPE_DETAIL_PREFIX) :].strip()
            # Details are rendered ``a; b; c`` onto ONE line, so the target
            # shape ends at the next detail -- without this the tokens of a
            # following ``shape_snapshots:`` were read as further target axes.
            body = body.split(";", 1)[0].strip()
            return [tok.strip() for tok in body.split(",")]
    return None


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
    # A generator reads a tensor's EXTENT, not its content: ``torch.arange(
    # valid_keys.shape[-1])`` genuinely depends on ``valid_keys`` and is drawn
    # with that edge, but ``arange``'s parameters take no tensor operand at all.
    # Counting the extent edge as an operand would report every such generator
    # as a mis-wired argument, so discount the edges the extractor marked.
    tensor_count = max(0, tensor_count - _extent_input_count(node))
    node_id = node.get("id")
    warnings: list[str] = []

    # Resolve the op's operand contract dynamically from its real function
    # parameters (``inspect`` first, then the aten schema) keyed on the raw op
    # name recovered from the model's own forward source -- never a static list.
    #
    # Keyed on ``raw_op`` and never on the display label. The two answer
    # different questions: the label says what an op IS for a reader, ``raw_op``
    # names the call whose signature is its contract. ``Mean`` labels the phi
    # that joins mutually-exclusive layer variants -- a node with no call behind
    # it at all -- and a phi's arity is a property of the branch, not of
    # ``torch.mean``. Measured: judging arity by the label reported eight such
    # ops on GLM alone, every one of them correctly wired. An op whose label was
    # synthesised rather than read off a call carries ``raw_op`` only where the
    # construction IS that op, which is exactly when the contract applies.
    #
    # ``Gather`` used to be the other half of this example, when it labelled a
    # three-tensor advanced index (``keys[batch_idx, pool_idx]``) as well as the
    # two-tensor ``torch.gather``. A subscript is now ``Indexing``, so that
    # particular lie is gone -- but the rule stands on its own reasoning, not on
    # the labels happening to be honest today.
    contract_name = _raw_op(node)
    ceiling, variadic = _operand_ceiling(contract_name)

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
        if ceiling >= 1 and tensor_count == 0:
            # A bounded op that takes at least one tensor operand (you cannot
            # ``unsqueeze`` or ``transpose`` nothing) with zero counted tensor
            # operands means the op's sole activation edge went missing -- an
            # upstream wiring/pruning bug. A ceiling of zero is a genuine tensor
            # *source* (``torch.arange(n)``/``torch.zeros(shape)`` fabricate a
            # tensor from host scalars alone), for which zero wired edges is
            # correct, so it is not flagged here.
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

    # An op that takes an INDEX takes an integer one. ``torch`` cannot index with
    # a float tensor at all, so a gather/scatter/embedding whose wired operands
    # are all floating-point is reporting an index it could not have run: the
    # dtype was lost upstream, and with it the shape rule's only way to tell the
    # index from the table it reads. Resolved from the op's real signature --
    # never a list of op names.
    index_parameters = [
        name
        for name in _required_tensor_operand_names(contract_name)
        if name in {"index", "indices"}
    ]
    if index_parameters and tensor_count >= 2:
        if not any(_is_integer_dtype(item) for item in input_types):
            warnings.append(
                f"{node_id} [{op}]: takes an {index_parameters[0]!r} operand, but "
                f"none of its operands is an integer type "
                f"(input_types={input_types}). An index is never floating-point; "
                f"the dtype was dropped upstream."
            )

    # Shape-change no-op checks. Both compare the op's resolved OUTPUT shape to its
    # single tensor INPUT shape; a match means the declared reshaping did not take
    # effect. Compute the shared operands once.
    out_dims = _output_shape_dims(node)
    tensor_inputs = [
        shapes[i]
        for i, t in enumerate(input_types)
        if i < len(shapes)
        and isinstance(shapes[i], list)
        and _operand_is_tensor(t, shapes[i])
    ]
    single_tensor_in = tensor_inputs[0] if len(tensor_inputs) == 1 else None

    # A shape-changing slice/select/resize op whose output shape still equals its
    # input shape did not actually change the shape -- the declared narrowing was
    # dropped in shape inference (the ``rotate_half`` no-op slice class of bug).
    change_details = [
        line
        for line in _details_lines(node)
        if line.startswith(_SHAPE_CHANGE_DETAIL_PREFIXES)
    ]
    if change_details and out_dims is not None and single_tensor_in is not None:
        in_dims = [str(dim) for dim in single_tensor_in]
        if in_dims == out_dims:
            warnings.append(
                f"{node_id} [{op}]: declares a shape change "
                f"({'; '.join(change_details)}) but its output shape {out_dims} "
                f"equals its input shape {in_dims} -- the narrowing was lost; "
                f"shape inference did not apply the declared slice/select."
            )

    # A ``view``/``reshape``/``expand`` declaring a FULLY CONCRETE target shape
    # (every axis a non-negative int literal -- no ``-1`` placeholder, symbolic dim,
    # starred prefix, or arithmetic) is unconditionally resolvable: shape inference
    # needs no operand symbols to apply it, so the op's true output IS that declared
    # shape. If the resolved output instead equals a (likewise fully concrete) input
    # shape that DIFFERS from the declared target, the reshape was dropped and the
    # source passed through -- the reshape analogue of the no-op slice. A
    # contiguous-forcing reshape declares the SAME shape (declared == input, not
    # flagged), and a symbolic/``-1``/starred target -- including the deliberately
    # merged-product ``B*S`` reshape -- is non-concrete and conservatively skipped,
    # since shape inference may legitimately resolve it to the operand's own shape.
    declared = _concrete_int_dims(_declared_reshape_dims(node))
    if declared is not None and out_dims is not None and single_tensor_in is not None:
        in_concrete = _concrete_int_dims([str(dim) for dim in single_tensor_in])
        out_concrete = _concrete_int_dims(out_dims)
        if (
            in_concrete is not None
            and out_concrete is not None
            and in_concrete == out_concrete
            and declared != in_concrete
        ):
            warnings.append(
                f"{node_id} [{op}]: declares a concrete reshape to {declared} but its "
                f"output shape {out_concrete} equals its input shape {in_concrete} -- "
                f"the reshape was lost; shape inference passed the source through "
                f"instead of applying the declared target shape."
            )

    return warnings


def _extent_input_count(node: dict[str, Any]) -> int:
    """How many of the node's wired edges carry an extent rather than an operand.

    Stamped by the extractor as ``extent_inputs: N`` when it recovers the tensors
    a generator's size arguments read (``torch.arange(n_windows)`` where
    ``n_windows`` came from ``compressed.shape[1]``).
    """
    for token in _detail_tokens(node):
        if token.startswith("extent_inputs:"):
            try:
                return int(token.split(":", 1)[1].strip())
            except ValueError:
                return 0
    return 0


def _detail_tokens(node: dict[str, Any]) -> list[str]:
    """Individual detail entries, splitting the rendered ``a; b; c`` join as well
    as newlines (attention kernel details are joined with ``"; "`` onto one line)."""
    tokens: list[str] = []
    for line in _details_lines(node):
        for token in line.split(";"):
            token = token.strip()
            if token:
                tokens.append(token)
    return tokens


def _declared_output_arity(node: dict[str, Any]) -> int | None:
    """The kernel's real tensor-return count, from an ``outputs: N`` detail.

    Stamped by ``ast_analyze._resolve_dispatched_attention_kernel`` after it reads
    the resolved attention wrapper's ``return`` from source. Any node may carry it;
    the check keys on the detail's presence, never on a kernel-name list."""
    for token in _detail_tokens(node):
        if token.startswith("outputs:"):
            try:
                return int(token.split(":", 1)[1].strip())
            except ValueError:
                return None
    return None


def _output_arity_warnings(nodes: list[dict[str, Any]]) -> list[str]:
    """Flag a kernel node advertising more tensor output ports than it can return.

    A dispatched attention kernel declares how many real tensors it returns
    (``outputs: N``, introspected from the resolved wrapper's own ``return`` --
    SDPA's ``(attn_output, None)`` is one, eager's ``(attn_output, attn_weights)``
    is two). The rendered node must not expose more output ports than that: an
    extra port means an unpacked ``None``/unused slot was fanned out as a phantom
    tensor (the ``slice_1``/``slice_2`` class of bug). The advertised count is the
    larger of the node's own output-port metadata and the distinct output ordinals
    its consumers read -- either exceeding the declared arity is a real defect.
    """
    used_ordinals: dict[str, set[str]] = {}
    for node in nodes:
        for edge in node.get("incomingEdges", []) or []:
            source = edge.get("sourceNodeId")
            if source is None:
                continue
            used_ordinals.setdefault(str(source), set()).add(
                str(edge.get("sourceNodeOutputId", "0"))
            )
    warnings: list[str] = []
    for node in nodes:
        declared = _declared_output_arity(node)
        if declared is None:
            continue
        node_id = str(node.get("id", ""))
        port_count = len(node.get("outputsMetadata", []) or [])
        ordinal_count = len(used_ordinals.get(node_id, set()))
        advertised = max(port_count, ordinal_count)
        if advertised > declared:
            warnings.append(
                f"{node_id} [{_op_type(node)}]: advertises {advertised} tensor "
                f"output port(s) but its resolved kernel returns only {declared} "
                f"real tensor(s) (outputs: {declared}); a phantom output slot was "
                f"fanned out -- an unpacked None/unused return was modeled as a "
                f"tensor. Trim it in _resolve_dispatched_attention_kernel."
            )
    return warnings


# Dim tokens that a *resolved* rendered shape never legitimately contains: ``?``
# (shape inference could not determine the axis at all), ``-1`` (an unfolded
# reshape/view placeholder that should have been resolved against the operand's
# real dims), and the empty token (a dropped axis, ``[B, , S]``). Every genuine
# axis is either a concrete int or a named symbolic dim (``S``, ``Pv/4``,
# ``B*S*8192``, ``index_head_dim``) -- those are resolved, just symbolic, and are
# NOT flagged. Mirrors the unresolved half of ``merge._dim_is_weak``'s set (its
# collapsed-product ``BxS`` notion is deliberately excluded: a merged reshape dim
# is a real, resolved width).
_UNRESOLVED_DIM_TOKENS = frozenset({"?", "-1", ""})


def _unresolved_shape_warnings(nodes: list[dict[str, Any]]) -> list[str]:
    """Flag any node whose rendered output shape carries an unresolved dim.

    A ``?``/``-1``/empty axis in an ``output_shape`` means shape inference left a
    dim undetermined -- a fidelity gap (an unfolded reshape ``-1``, a reduction
    whose result axis stayed ``?``) to fix upstream in ``shape_inference.py``, not
    a legitimate shape. Keyed purely on the dim token, so it needs no op-name or
    class-name list and never fires on a legitimately symbolic dim. Constant nodes
    (learned weights/buffers, filtered from the drawn graph) are skipped, and a
    rank-0 scalar (empty bracket ``[]`` -> no dim tokens) is resolved, not flagged.
    """
    warnings: list[str] = []
    for node in nodes:
        if _is_constant_node(node):
            continue
        dims = _output_shape_dims(node)
        if not dims:
            continue
        unresolved = [d for d in dims if d.strip() in _UNRESOLVED_DIM_TOKENS]
        if unresolved:
            node_id = node.get("id")
            warnings.append(
                f"{node_id} [{_op_type(node)}]: output shape {dims} carries "
                f"unresolved dim(s) {unresolved} -- shape inference left an axis "
                f"undetermined (an unfolded reshape -1 / a ? placeholder). Resolve "
                f"it in shape_inference.py; do not render an unresolved shape."
            )
    return warnings


def _kernel_operand_arity_warnings(nodes: list[dict[str, Any]]) -> list[str]:
    """Flag a kernel that carries fewer distinct tensor input ports than its
    resolved function requires as distinct tensor operands.

    A kernel node's real primitive (its ``kernel_primitive`` attr, e.g.
    ``scaled_dot_product_attention``) requires a fixed set of DISTINCT tensor
    operands -- sdpa needs three (``query``/``key``/``value``); the optional
    ``attn_mask`` is excluded. The count is resolved structurally from the
    primitive's aten/inspect signature (never an op-name -> count table). Each of
    those operands must be supplied by its own kernel input port. When two required
    operands are wired through one shared port (DeepSeek's combined ``kv`` feeding
    both key and value before the split fix), fewer roles are covered than required
    -- a real fidelity defect: the graph shows one edge where the kernel reads two
    distinct tensors.

    Scoped to nodes actually fed by kernel input ports (``@kernel_port_in``
    sources); a plain op reaching its operands by ordinary edges is left to
    ``_check_node``. Single-required-operand ops are covered by that pass's
    zero-operand floor, so only multi-operand kernels are checked here.
    """
    by_id = {str(n.get("id")): n for n in nodes}
    warnings: list[str] = []
    for node in nodes:
        primitive = _node_attr_value(node, "kernel_primitive")
        if not primitive:
            continue
        required = _required_tensor_operand_names(primitive)
        if len(required) < 2:
            continue
        # Follow the EDGE to each port and ask that port which operand it
        # supplies. The role was fixed when the port was bound, so a port
        # renamed for the reader still answers correctly; matching on the
        # display label instead meant qualifying a port silently uncovered a
        # required operand and this check fired on a correct graph.
        port_labels: list[str] = []
        port_roles: list[str] = []
        for edge in node.get("incomingEdges", []) or []:
            source = by_id.get(str(edge.get("sourceNodeId")))
            if source is None:
                continue
            if _node_attr_value(source, "synthetic") != "@kernel_port_in":
                continue
            port_labels.append(_port_label(source))
            role = _node_attr_value(source, "operand_role")
            if role:
                port_roles.append(str(role))
        if not port_labels:
            continue
        if not port_roles:
            # No port declared which operand it supplies, so there is nothing to
            # judge structurally. Say nothing rather than guess from names: a
            # display label is not evidence about wiring.
            continue
        covered = sum(1 for param in required if param in port_roles)
        if covered < len(required):
            node_id = node.get("id")
            warnings.append(
                f"{node_id} [{_op_type(node)}]: its resolved function requires "
                f"{len(required)} distinct tensor operand(s) {list(required)}, but "
                f"only {covered} are covered by its input ports {port_labels} -- two "
                f"required operands share one combined input port where the kernel "
                f"reads distinct tensors. Split the port so each required operand "
                f"reads its own edge."
            )
    return warnings


_KERNEL_IN_TILE = "@kernel_port_in"
_KERNEL_OUT_TILES = frozenset({"@kernel_port_out", "@output", "@output_mirror"})


def _kernel_interface_tile_warnings(nodes: list[dict[str, Any]]) -> list[str]:
    """A fused kernel reading several tensors must name each one it reads.

    One unnamed edge into a box is readable; several are not. A kernel is drawn
    as a single opaque tile precisely because its insides are not shown, so its
    INTERFACE is the only thing a reader has: which tensor arrives where, and
    what comes back. Each operand therefore gets its own ``@kernel_port_in``
    tile carrying the kernel's own name for it, and each value the kernel
    publishes to more than one reader leaves by an output tile.

    Measured when this was written: Kimi's ``chunk_kda_fwd_intra`` was wired
    five tensors and declared four ports, so ``v`` arrived as a bare edge from a
    tile shared with the rest of the pipeline; ``flash_attn_varlen_func``
    declared ``attention_mask`` among its inputs and then read it off a plain
    boundary; and the three ``ShortConvolution`` kernels took ``cu_seqlens``
    straight from another box's output mirror. Four named arrows and one
    anonymous one is a picture that cannot be read -- and in each case the
    unported operand was also the one whose provenance nothing had checked.

    Keyed on the node's own ``operation: gpu_kernel`` tag, never on a kernel
    name. Constants are exempt: a learned weight is wired where it is used and
    is never part of a declared interface.
    """
    by_id = {str(n.get("id")): n for n in nodes}
    consumers: dict[str, list[dict[str, Any]]] = {}
    for node in nodes:
        for edge in node.get("incomingEdges", []) or []:
            consumers.setdefault(str(edge.get("sourceNodeId")), []).append(node)
    warnings: list[str] = []
    for node in nodes:
        if _node_attr_value(node, "operation") != "gpu_kernel":
            continue
        operands = [
            source
            for source in (
                by_id.get(str(edge.get("sourceNodeId")))
                for edge in node.get("incomingEdges", []) or []
            )
            if source is not None and _node_attr_value(source, "constant") != "true"
        ]
        unported = [
            source
            for source in operands
            if _node_attr_value(source, "synthetic") != _KERNEL_IN_TILE
        ]
        if len(operands) > 1 and unported:
            warnings.append(
                f"{node.get('id')} [{_op_type(node)}]: reads {len(operands)} "
                f"tensors but only {len(operands) - len(unported)} arrive on an "
                f"input tile of its own; "
                f"{[_port_label(source) for source in unported]} reach it as bare "
                f"edges. A kernel is drawn as one opaque tile, so an operand with "
                f"no port of its own cannot be told from the others -- give it a "
                f"kernel input port naming it."
            )
        # An allocation call (``new_empty``, ``new_zeros``, ...) borrows the
        # source's dtype/device as a template for a buffer a LATER kernel
        # writes into; it never reads the source's value, so several such
        # allocations off the one kernel are not several of its published
        # values competing to be told apart.
        readers = [
            reader
            for reader in consumers.get(str(node.get("id")), [])
            if _node_attr_value(reader, "raw_op") not in _ALLOCATION_CALLS
        ]
        untiled = [
            reader
            for reader in readers
            if _node_attr_value(reader, "synthetic") not in _KERNEL_OUT_TILES
        ]
        if len(readers) > 1 and untiled:
            warnings.append(
                f"{node.get('id')} [{_op_type(node)}]: publishes to "
                f"{len(readers)} readers but only {len(readers) - len(untiled)} "
                f"leave by an output tile of its own; a kernel handing back "
                f"several values must say which is which."
            )
    return warnings


def _phi_forwards_an_arm_warnings(nodes: list[dict[str, Any]]) -> list[str]:
    """A phi hands back one of its arms, so its shape has to be one of theirs.

    Exactly one arm of a mutually-exclusive branch runs per invocation and the
    phi forwards it untouched -- it computes nothing. So whatever shape it
    reports must be a shape some arm actually has; any other shape is one
    nobody produces, and every op below it inherits that.

    Only that. The arms do NOT all have to agree: a model may rebind the name
    to a differently-shaped tensor in one branch, and Kimi does -- its linear
    attention replaces ``hidden_states`` with the unpadded ``[1, nnz, 7168]``
    when a mask is present and keeps ``[B, S, 7168]`` when it is not. Both are
    real. Asserting agreement would report that correct graph, so this asserts
    the part that holds whatever the model does.

    Constant arms are skipped: a learned operand is not a branch.
    """
    by_id = {str(n.get("id")): n for n in nodes}
    warnings: list[str] = []
    for node in nodes:
        is_phi = _node_attr_value(node, "class_name") == PHI_CLASS or (
            node.get("label") == PHI_LABEL
        )
        if not is_phi or _node_attr_value(node, "synthetic"):
            continue
        out = _output_shape_dims(node)
        if out is None:
            continue
        arms = []
        for edge in node.get("incomingEdges", []) or []:
            source = by_id.get(str(edge.get("sourceNodeId")))
            if source is None or _node_attr_value(source, "constant") == "true":
                continue
            shape = _output_shape_dims(source)
            if shape is not None:
                arms.append((source.get("label"), shape))
        if not arms or out in [shape for _label, shape in arms]:
            continue
        warnings.append(
            f"{node.get('id')} [{_op_type(node)}]: forwards one of its arms "
            f"untouched, so its shape must be one of theirs -- it reports "
            f"{out} where the arms are "
            f"{[(str(label), shape) for label, shape in arms]}. A phi computes "
            f"nothing; a shape no arm has is one nothing produces."
        )
    return warnings


def warn_degenerate_mutually_exclusive_branches(
    nodes: list[dict[str, Any]],
) -> list[str]:
    """Warn when a mutually exclusive branch no longer has two live inputs.

    A branch joins alternatives. One live input means the other inputs are
    dead and the branch should already have been removed. A boundary tile
    that only carries the branch's name is not the branch.
    """
    by_id = {str(node.get("id")): node for node in nodes}
    warnings: list[str] = []
    for node in nodes:
        label = str(node.get("label") or "")
        identity = str(_node_attr_value(node, "class_name") or "")
        if PHI_LABEL not in (label, identity) and PHI_CLASS not in (label, identity):
            continue
        if _node_attr_value(node, "synthetic"):
            continue
        live = [
            str(edge.get("sourceNodeId") or "")
            for edge in node.get("incomingEdges") or []
            if str(edge.get("sourceNodeId") or "") in by_id
        ]
        if len(live) > 1:
            continue
        warnings.append(
            f"{node.get('id')}: Mutually exclusive branch has {len(live)} live "
            f"input(s); it must have more than one. Remove it when every other "
            f"input is dead."
        )
    for line in warnings:
        _log.warning("graph generation: %s", line)
    return warnings


def type_check_graph_nodes(nodes: list[dict[str, Any]]) -> list[str]:
    """Type-check every checkable operation node; return + log warning lines.

    Never raises. An empty return means every checkable op's operand contract held.
    """
    warnings: list[str] = []
    for node in nodes:
        warnings.extend(_check_node(node))
    warnings.extend(_noop_cast_warnings(nodes))
    warnings.extend(_shape_preserving_op_warnings(nodes))
    warnings.extend(_reshape_conservation_warnings(nodes))
    warnings.extend(_in_place_write_warnings(nodes))
    warnings.extend(_output_arity_warnings(nodes))
    warnings.extend(_kernel_operand_arity_warnings(nodes))
    warnings.extend(_kernel_interface_tile_warnings(nodes))
    warnings.extend(_phi_forwards_an_arm_warnings(nodes))
    warnings.extend(_unresolved_shape_warnings(nodes))
    warnings.extend(_declared_axis_warnings(nodes))
    warnings.extend(_declared_rank_warnings(nodes))
    for line in warnings:
        _log.warning("graph type-check: %s", line)
    return warnings


def _declared_axis_warnings(nodes: list[dict[str, Any]]) -> list[str]:
    """An op that declares WHICH axis it changes must have that axis.

    ``select_dim``/``resize_dim`` name an axis by index, and the extractor reads
    that index off the model's own subscript. So the index is a fact about the
    tensor the model slices -- and if the operand the graph hands the op has
    fewer axes than that, the two disagree and one of them is wrong. Either the
    operand was mis-wired, or whatever produced it reported the wrong rank.

    This is the check that was missing when Kimi's post-attention
    ``attn_output[:, :, :, : self.v_head_dim]`` -- ``resize_dim: 3`` -- was
    handed a RANK-3 operand, because the kernel above it had reported the
    enclosing module's output instead of its own. Nothing objected: the shape
    rules wrote the declared width at whatever axis they could reach, producing
    ``[128, S, 7168]``, and every existing check asked only whether the dims
    RESOLVED. They did. They were just not this tensor's.

    Reads the declaration the extractor already stamps -- no op name list.
    """
    warnings: list[str] = []
    for node in nodes:
        shapes = _load_list(node, "input_shapes")
        if not isinstance(shapes, list) or not shapes:
            continue
        operand = next((s for s in shapes if isinstance(s, list) and s), None)
        if operand is None:
            continue
        rank = len(operand)
        for line in _detail_tokens(node):
            text = str(line).strip()
            for prefix in ("select_dim:", "resize_dim:"):
                if not text.startswith(prefix):
                    continue
                body = text[len(prefix) :].strip().split("=", 1)[0].strip()
                try:
                    axis = int(body)
                except ValueError:
                    continue
                if axis < 0:
                    axis += rank
                if 0 <= axis < rank:
                    continue
                warnings.append(
                    f"{node.get('id')} [{_op_type(node)}]: declares "
                    f"{prefix.rstrip(':')} {body} but its operand has rank "
                    f"{rank} ({operand}); the axis it narrows does not exist, "
                    "so the operand it was given is not the tensor the model "
                    "slices -- fix the producer's shape, not this op."
                )
    return warnings


def _spans_several_axes(token: str) -> bool:
    """Whether one reshape argument can stand for more than one axis.

    ``*x.shape[:2]`` unpacks two; ``x.shape[0]`` is a single extent. Only the
    former makes the argument count differ from the rank.
    """
    text = token.strip()
    if text.startswith("*"):
        return True
    start = text.find("[")
    return start != -1 and ":" in text[start:]


def _declared_rank_warnings(nodes: list[dict[str, Any]]) -> list[str]:
    """A reshape that declares N target axes must produce N axes.

    The ``shape:`` detail is the model's own argument list, so its LENGTH is a
    fact even when the individual dims are not: ``chunk_kv.view(batch,
    n_windows, ratio, -1)`` names four axes whatever ``n_windows`` turns out to
    be. Rank is therefore checkable where the dims are not, and checking it is
    what separates "this axis is still symbolic" from "this is not the same
    tensor".

    DeepSeek's compressor view declared four and produced ``[B*S, 1024]`` --
    two. Every dim in that result is fully resolved, so the unresolved-shape
    check was satisfied; the conservation check folds dims and declined on the
    symbols; and the rank simply went unexamined. The rank-4 slices downstream
    then addressed an axis the operand did not have.

    Only the length is asserted, never the dims -- a symbolic axis is normal and
    is not this check's business.
    """
    warnings: list[str] = []
    for node in nodes:
        if _is_constant_node(node):
            continue
        if _op_type(node).strip().lower() not in {"view", "reshape"}:
            # ``expand`` shares the ``shape:`` detail but not the rule: a ``-1``
            # there KEEPS an existing axis rather than standing for one, and its
            # own resolver deliberately passes the source through when the args
            # cannot be aligned axis-for-axis. Rank is the reshape's contract.
            continue
        declared = _declared_reshape_dims(node)
        if not declared or len(declared) < 2:
            # A single token is ``x.reshape(x.shape)`` / ``view(shape_tuple)``:
            # one ARGUMENT standing for a whole shape, which says nothing about
            # how many axes come out.
            continue
        if any(not tok or _spans_several_axes(tok) for tok in declared):
            # A token that can stand for SEVERAL axes (``*x.shape[:2]``) makes
            # the argument count smaller than the rank. A scalar shape READ
            # (``x.shape[0]``) is one axis like any other and is fine.
            continue
        produced = _output_shape_dims(node)
        if produced is None or len(produced) == len(declared):
            continue
        warnings.append(
            f"{node.get('id')} [{_op_type(node)}]: declares {len(declared)} "
            f"target axes ({', '.join(declared)}) but its output has "
            f"{len(produced)} ({produced}); a reshape produces exactly the axes "
            "it is given, so this is not the tensor the model reshaped."
        )
    return warnings


def _output_dtype(node: dict[str, Any]) -> str | None:
    """The dtype suffix of a node's ``output_shape`` (``[B, S] float32`` -> ``float32``)."""
    for attr in node.get("attrs", []):
        if attr.get("key") == "output_shape":
            value = attr.get("value")
            if not isinstance(value, str):
                return None
            idx = value.find("]")
            if idx == -1:
                return None
            dtype = value[idx + 1 :].strip()
            return dtype or None
    return None


def _shape_preserving_op_warnings(nodes: list[dict[str, Any]]) -> list[str]:
    """An op that cannot change a shape must not be recorded having changed one.

    The companion of :func:`_noop_cast_warnings`, which states this invariant in
    its own first line for casts and then only ever compares dtypes. MiniMax's
    text ``q_norm`` opened with ``x.float()`` recording
    ``input_shapes [["B","S","64","128"]]`` beside
    ``output_shape [B, S, 48, 128]`` -- the contradiction sitting in two
    adjacent attributes of one node, and every check walked past it, because
    they ask whether a shape RESOLVED, and 48 is as resolved as 64.

    That came from the same id being inferred once with its class and again by a
    later root-less pass, which can land on any op; a cast is only where it
    showed up first. So the invariant is asked of the whole class of ops it
    applies to, and torch decides the membership
    (:func:`op_cannot_change_shape`) rather than a list of names -- a dropout, a
    ``contiguous``, an activation and a cast are all answered for by running the
    model's own op name against its own callable.

    Two ways in, because the predicate deliberately declines ops taking more
    than one tensor and ``Tensor.to`` is one of them: an op torch vouches for,
    or a node the exporter already labelled a cast.

    Read from the node's own annotation rather than by walking edges, for the
    reason the sibling check documents: a cast whose operand is a hidden
    constant loses that edge in the render-filtered graph. Only an unambiguous
    single-operand record is judged, and only when both shapes are recorded.
    """
    warnings: list[str] = []
    for node in nodes:
        if _is_constant_node(node):
            continue
        raw_shapes = _node_attr_value(node, "input_shapes")
        if not raw_shapes:
            continue
        try:
            input_shapes = json.loads(raw_shapes)
        except (ValueError, TypeError):
            continue
        if not isinstance(input_shapes, list) or len(input_shapes) != 1:
            continue
        operand = input_shapes[0]
        if not isinstance(operand, list) or not operand:
            continue
        produced = _output_shape_dims(node)
        if not produced:
            continue
        before = [str(dim).strip() for dim in operand]
        after = [str(dim).strip() for dim in produced]
        if before == after:
            continue
        if not op_cannot_change_shape(_raw_op(node)):
            continue
        name = _raw_op(node) or str(node.get("label") or "op")
        warnings.append(
            f"{node.get('id')} [{name}]: reads {before} and produces {after}. "
            f"This op returns its input's shape however it is called, so one of "
            f"these two was resolved somewhere the other was not -- most often "
            f"the same node id inferred once WITH its class and again without "
            f"it, where the body's shapes belong to whichever module was "
            f"reached last."
        )
    return warnings


def _dim_factors(tokens: list[str]) -> list[str] | None:
    """A shape's dim tokens as a flat multiset of factors, or None if unusable.

    ``["B", "S*768"]`` and ``["B", "S", "768"]`` describe the same number of
    elements, so both reduce to ``["768", "B", "S"]``. A size-1 axis multiplies
    nothing and drops out, which is what lets ``[8192] -> [1, 1, 8192]`` read as
    the conserving view it is.

    None for a shape carrying a ``-1``: that is a placeholder standing for
    whichever width makes the count come out right, so the count it implies
    cannot disagree with anything.
    """
    factors: list[str] = []
    for token in tokens:
        for factor in str(token).split("*"):
            factor = factor.strip()
            if not factor:
                return None
            if factor == "-1" or factor.startswith("-"):
                return None
            if factor == "1":
                continue
            factors.append(factor)
    return sorted(factors)


def _counts_provably_differ(before: list[str], after: list[str]) -> bool:
    """Whether two shapes CANNOT hold the same number of elements.

    Factors common to both sides cancel. What is left is only a disagreement if
    every remaining factor is a literal -- ``[B, S, 768]`` against
    ``[B, S, 1024]`` cancels to ``768`` against ``1024`` and is one. Two
    symbolic remainders (``Pv`` against ``Img``) prove nothing: the same count
    can be written with different symbols, and a check that guessed here would
    report a model's own notation as a defect.
    """
    fed, produced = _dim_factors(before), _dim_factors(after)
    if fed is None or produced is None:
        return False
    remaining_fed, remaining_produced = list(fed), list(produced)
    for factor in fed:
        if factor in remaining_produced:
            remaining_produced.remove(factor)
            remaining_fed.remove(factor)
    if not remaining_fed and not remaining_produced:
        return False
    fed_count = _concrete_int_dims(remaining_fed)
    produced_count = _concrete_int_dims(remaining_produced)
    if fed_count is None or produced_count is None:
        return False
    return math.prod(fed_count) != math.prod(produced_count)


_MUTATES_DETAIL_PREFIX = "mutates: "


def _in_place_write_warnings(nodes: list[dict[str, Any]]) -> list[str]:
    """An in-place write returns the tensor it writes INTO, unchanged in shape.

    ``final.index_add_(0, token_idx, current)`` hands back ``final``; the index
    addresses into it and the source supplies values, but neither decides what
    comes out. The extractor records which name the call mutates, and the
    receiver is the call's first operand, so the op's own record states both
    halves of the invariant and this check needs to name no operation.

    Worth asking separately because the shape rule for these ops is POSITIONAL
    -- it anchors on operand 0 -- and an operand whose shape did not resolve is
    dropped rather than held open, which silently moves every later operand down
    a slot. GLM's experts accumulator was handed its index as operand 0 that
    way and reported ``[nnz, 1] int64`` for a ``[B*S, 4096] bfloat16``
    accumulator; nothing noticed until a ``view`` two steps later was seen
    converting a dtype. Comparing the op against its own first operand catches
    it where it happens.
    """
    warnings: list[str] = []
    for node in nodes:
        if _is_constant_node(node):
            continue
        mutated = next(
            (
                line[len(_MUTATES_DETAIL_PREFIX) :].split(";")[0].strip()
                for line in _details_lines(node)
                if line.startswith(_MUTATES_DETAIL_PREFIX)
            ),
            None,
        )
        if not mutated:
            continue
        input_shapes = _load_list(node, "input_shapes")
        produced = _output_shape_dims(node)
        if not input_shapes or not produced:
            continue
        receiver = input_shapes[0]
        if not isinstance(receiver, list) or not receiver:
            continue
        before = [str(dim).strip() for dim in receiver]
        after = [str(dim).strip() for dim in produced]
        name = _raw_op(node) or str(node.get("label") or "op")
        if before != after:
            warnings.append(
                f"{node.get('id')} [{name}]: writes into {before} and reports "
                f"{after}. This op returns the tensor it mutates ({mutated!r}), "
                f"which is its first operand, so either the receiver is not the "
                f"operand recorded first -- an earlier operand whose shape did "
                f"not resolve moves the rest down a slot -- or the two were "
                f"resolved against different tensors."
            )
        input_types = _load_list(node, "input_types")
        if not input_types or not isinstance(input_types, list):
            continue
        in_dtype = input_types[0]
        if not isinstance(in_dtype, str) or in_dtype in ("Constant", "Tensor"):
            continue
        out_dtype = _node_attr_value(node, "output_dtype") or _output_dtype(node)
        if not out_dtype or in_dtype == out_dtype:
            continue
        warnings.append(
            f"{node.get('id')} [{name}]: writes into a {in_dtype} tensor and "
            f"reports {out_dtype}. An in-place write keeps the dtype of the "
            f"tensor it mutates ({mutated!r}); a source of another dtype is "
            f"converted INTO it, not the other way round."
        )
    return warnings


def _reshape_conservation_warnings(nodes: list[dict[str, Any]]) -> list[str]:
    """A reshape rearranges a tensor; it neither makes elements nor retypes them.

    The invariant a ``view`` has in place of the one
    :func:`_shape_preserving_op_warnings` enforces. A view exists precisely TO
    move the axes around -- it is checked elsewhere that it must do so or be
    elided -- but the elements it moves are the ones it was given, so
    ``[B, S, 768] -> [B, S, 1024]`` is a defect in a way that
    ``[B, S, 768] -> [B, S*768]`` is not. That is the same root-less-overwrite
    class the sibling check guards, reaching an op whose shape is SUPPOSED to
    change and where comparing shapes therefore says nothing.

    Which ops conserve their element count is torch's answer, not a list here
    (:func:`op_cannot_change_element_count`) -- it keeps ``expand`` out on its
    own evidence, since broadcasting a size-1 axis really does make more
    elements.

    The dtype half is asked of the node's own recorded call rather than of the
    op: ``Tensor.view`` has a second overload that reinterprets a dtype, so a
    name alone cannot say. A node whose details declare the SHAPE it was called
    with was not called that way, and a reinterpreting view would in any case
    have to change the width it reports.
    """
    warnings: list[str] = []
    for node in nodes:
        if _is_constant_node(node):
            continue
        raw_shapes = _node_attr_value(node, "input_shapes")
        if not raw_shapes:
            continue
        try:
            input_shapes = json.loads(raw_shapes)
        except (ValueError, TypeError):
            continue
        if not isinstance(input_shapes, list) or len(input_shapes) != 1:
            continue
        operand = input_shapes[0]
        if not isinstance(operand, list) or not operand:
            continue
        produced = _output_shape_dims(node)
        if not produced:
            continue
        declared = _declared_reshape_dims(node)
        name = _raw_op(node) or str(node.get("label") or "op")
        before = [str(dim).strip() for dim in operand]
        after = [str(dim).strip() for dim in produced]
        if op_cannot_change_element_count(_raw_op(node)) and _counts_provably_differ(
            before, after
        ):
            warnings.append(
                f"{node.get('id')} [{name}]: reads {before} and produces {after}, "
                f"which cannot hold the same number of elements. This op "
                f"rearranges the elements it is given and can neither make nor "
                f"drop one, so one of these two shapes was resolved against "
                f"something that is not what the op was handed."
            )
        if declared is None:
            continue
        raw_types = _node_attr_value(node, "input_types")
        if not raw_types:
            continue
        try:
            input_types = json.loads(raw_types)
        except (ValueError, TypeError):
            continue
        if not isinstance(input_types, list) or len(input_types) != 1:
            continue
        in_dtype = input_types[0]
        if not isinstance(in_dtype, str) or in_dtype in ("Constant", "Tensor"):
            continue
        out_dtype = _node_attr_value(node, "output_dtype") or _output_dtype(node)
        if not out_dtype or in_dtype == out_dtype:
            continue
        warnings.append(
            f"{node.get('id')} [{name}]: reads {in_dtype} and produces {out_dtype} "
            f"while declaring the shape {declared} it was called with. An op "
            f"called with a shape rearranges elements and does not convert them, "
            f"so the two dtypes were resolved from different tensors."
        )
    return warnings


def _noop_cast_warnings(nodes: list[dict[str, Any]]) -> list[str]:
    """Flag every ``Cast`` whose sole operand dtype already equals its output dtype.

    A cast never changes shape, so a ``Cast`` whose output dtype matches the dtype
    of the tensor feeding it performs no conversion -- a redundant cast the merge
    no-op-cast elision (``merge._prune_noop_cast_nodes``) should have removed.

    The operand dtype is read from the node's OWN profiler-style annotation
    (``input_types``/``output_dtype`` attached by ``_annotate_op_input_signatures``)
    rather than by walking incoming edges. That annotation is stable across the
    built graph and the render-filtered (constant-dropped) graph, whereas an edge
    walk is not: a cast whose true operand is a hidden constant (e.g. a learned
    ``A_log.float()`` bf16->f32 conversion) loses that operand when constants are
    filtered for rendering, leaving only a spurious spine predecessor whose dtype
    would spoof a no-op. Reading the annotation avoids that false positive.

    A cast converts exactly one tensor operand, so only an unambiguous
    single-operand annotation is checked: more than one recorded operand (a
    spine/among-constant contamination) is ambiguous about which entry is the
    tensor being cast, and a ``Constant``/``Tensor`` operand carries no concrete
    dtype to compare -- both are skipped so the check never false-positives on a
    legitimate conversion.
    """
    warnings: list[str] = []
    for node in nodes:
        # A dtype conversion is identified by the target dtype the extractor
        # recorded for the call, which only a converting call carries, together
        # with torch confirming the op keeps its shape. Not by the node being
        # called "Cast": a label is for reading, not for deciding what an op is.
        if not any(line.startswith("dtype") for line in _details_lines(node)):
            continue
        if not op_cannot_change_shape(_raw_op(node)):
            continue
        if _is_constant_node(node):
            continue
        raw_types = _node_attr_value(node, "input_types")
        if not raw_types:
            continue
        try:
            input_types = json.loads(raw_types)
        except (ValueError, TypeError):
            continue
        if not isinstance(input_types, list) or len(input_types) != 1:
            continue
        in_dtype = input_types[0]
        if not isinstance(in_dtype, str) or in_dtype in ("Constant", "Tensor"):
            continue
        out_dtype = _node_attr_value(node, "output_dtype") or _output_dtype(node)
        if not out_dtype or in_dtype != out_dtype:
            continue
        warnings.append(
            f"{node.get('id')} [cast]: input dtype {in_dtype} already equals its "
            f"output dtype {out_dtype} -- the cast performs no conversion and should "
            f"be elided (merge no-op-cast pass missed it)."
        )
    return warnings


# --------------------------------------------------------------------------- #
# A repeat group (``45x_Glm5NextTextDecoderLayer``) renders N identical layers;
# it is not a scope a tensor enters, so it is not a box wall for I5.
_REPEAT_SEGMENT = re.compile(r"^\d+x_")

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


def _is_integer_dtype(type_str: Any) -> bool:
    """True for an integer tensor dtype (never ``Scalar``/``Constant`` markers)."""
    if not isinstance(type_str, str):
        return False
    lowered = type_str.strip().lower()
    return lowered.startswith(("int", "uint", "long", "short", "byte"))


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


def integrity_check_graph_nodes(
    nodes: list[dict[str, Any]], *, label: str = ""
) -> list[str]:
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
    - **I5 boundary skip** -- no edge crosses more than one box wall in a single
      hop. A tensor entering a box should land on that box's own boundary tile;
      an edge that jumps straight into a box nested inside it leaves the outer
      box showing no such input at all, even though the tensor plainly enters it.
      GLM's ``grid_thw`` did exactly that -- a model input wired directly to a
      tile two levels down, so the ``visual`` tower drew no ``grid_thw`` input.
      The same applies leaving a box, where the tile is an ``@output``.

      Repeat groups (``45x_Glm5NextTextDecoderLayer``) are NOT walls: the group
      is a rendering of N identical layers rather than a scope the tensor enters,
      and its carried values are drawn by the loop boundary machinery instead.
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
            # A pipeline recv records a transfer into the next stage. That stage
            # is the same folded stack tile, so the recv is a sink on purpose.
            or _node_attr_value(node, "synthetic") == "communication_sink"
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
            or _is_zero_operand_source(node)  # torch.arange/zeros generator source
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

        # I5 boundary skip: an edge may cross at most one box wall per hop.
        for source_id in _incoming_source_ids(node):
            producer = by_id.get(source_id)
            if producer is None:
                continue
            left, entered = _walls_crossed(producer, node)
            if len(left) > 1 or len(entered) > 1:
                warnings.append(
                    f"I5 boundary-skip{tag}: {source_id} -> {node_id} crosses "
                    f"{len(left)} box(es) out and {len(entered)} in "
                    f"(out={left}, in={entered}); a tensor entering a box must "
                    f"land on that box's own boundary tile, or the box renders "
                    f"with an input it never declares."
                )

    for line in warnings:
        _log.warning("graph integrity: %s", line)

    return warnings


def _incoming_source_ids(node: dict[str, Any]) -> list[str]:
    """Producer ids on a node's incoming edges."""
    return [
        str(edge.get("sourceNodeId"))
        for edge in node.get("incomingEdges", []) or []
        if edge.get("sourceNodeId") is not None
    ]


def _boundary_segments(node: dict[str, Any]) -> list[str]:
    """Namespace segments that are real boxes.

    A repeat group is a rendering of N identical layers, not a scope a tensor
    enters, so it is not a wall (see I5).
    """
    return [
        part
        for part in str(node.get("namespace") or "").split("/")
        if part and not _REPEAT_SEGMENT.match(part)
    ]


def _walls_crossed(
    producer: dict[str, Any], consumer: dict[str, Any]
) -> tuple[list[str], list[str]]:
    """Boxes an edge leaves and boxes it enters, below their common ancestry."""
    source = _boundary_segments(producer)
    target = _boundary_segments(consumer)
    shared = 0
    while shared < len(source) and shared < len(target):
        if source[shared] != target[shared]:
            break
        shared += 1
    return source[shared:], target[shared:]


def _immediate_child_unit(node_id: str, ns: str, box: str) -> str:
    """The unit *node_id* contracts to among the immediate children of *box*.

    A leaf lying directly in *box* (``ns == box``) is its own unit (its id). A
    node nested inside a sub-box is contracted to that immediate sub-box's
    namespace (``box/<next-segment>``), so a whole rendered sub-box collapses to a
    single super-node -- exactly the granularity at which an illegal rendered
    cycle between two sibling boxes shows up.
    """
    if ns == box:
        return node_id
    rest = ns[len(box) + 1 :] if box else ns
    segment = rest.split("/", 1)[0]
    return f"{box}/{segment}" if box else segment


def _tarjan_nontrivial_sccs(
    adjacency: dict[str, set[str]],
) -> list[list[str]]:
    """Iterative Tarjan SCC; return only SCCs with more than one member.

    Self-loops are already excluded by the caller (a unit never links to itself),
    so a multi-member SCC is the only cycle signal.
    """
    index_of: dict[str, int] = {}
    low: dict[str, int] = {}
    on_stack: set[str] = set()
    stack: list[str] = []
    counter = 0
    result: list[list[str]] = []

    for start in adjacency:
        if start in index_of:
            continue
        # (node, iterator over successors) work stack for iterative DFS.
        work: list[tuple[str, list[str]]] = [(start, list(adjacency.get(start, ())))]
        index_of[start] = low[start] = counter
        counter += 1
        stack.append(start)
        on_stack.add(start)
        while work:
            node, successors = work[-1]
            advanced = False
            while successors:
                succ = successors.pop()
                if succ not in index_of:
                    index_of[succ] = low[succ] = counter
                    counter += 1
                    stack.append(succ)
                    on_stack.add(succ)
                    work.append((succ, list(adjacency.get(succ, ()))))
                    advanced = True
                    break
                if succ in on_stack:
                    low[node] = min(low[node], index_of[succ])
            if advanced:
                continue
            if low[node] == index_of[node]:
                component: list[str] = []
                while True:
                    member = stack.pop()
                    on_stack.discard(member)
                    component.append(member)
                    if member == node:
                        break
                if len(component) > 1:
                    result.append(component)
            work.pop()
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[node])
    return result


def group_cycle_check_graph_nodes(
    nodes: list[dict[str, Any]], *, label: str = ""
) -> list[str]:
    """H check -- detect cycles at the collapsed-namespace-GROUP (rendered box) level.

    The node-level acyclic check misses a cycle that only exists once nodes are
    contracted into their rendered boxes: two sibling boxes (e.g. an attention's
    ``kv_norm`` and its ``CSACompressor``) can each feed the other through
    boundary tiles, drawing an illegal 2-cycle between the two rendered boxes even
    though the underlying node graph is acyclic.

    For each namespace box, its child *units* are (a) every immediate sub-box,
    contracted to a single super-node, and (b) every leaf node lying directly in
    the box, each its own unit (direct leaves are NOT merged). An edge whose two
    endpoints fall in different child units of the same box contributes an edge
    between those units; a nontrivial Tarjan SCC among a box's child units is an
    illegal rendered cycle. Contraction is done per box (immediate children only),
    not by globally contracting every node to its full namespace -- the global
    approach false-positives because parent<->child boundary tiles legitimately
    cross a box boundary in both directions.

    The one sanctioned back edge per loop -- a ``@loop_carried_out`` producer
    feeding a ``@loop_carried_in`` seed -- is excluded.
    """
    tag = f" [{label}]" if label else ""
    ns_of: dict[str, str] = {
        str(n.get("id", "")): str(n.get("namespace", "") or "") for n in nodes
    }

    # box -> {unit: set(successor units)} among that box's immediate children.
    box_adjacency: dict[str, dict[str, set[str]]] = {}

    for node in nodes:
        target_id = str(node.get("id", ""))
        ns_t = ns_of.get(target_id, "")
        for edge in node.get("incomingEdges", []) or []:
            source_id = str(edge.get("sourceNodeId", ""))
            if source_id not in ns_of:
                continue
            # Sanctioned loop back edge: the single carried value that closes a
            # repeat group's iteration boundary.
            if "@loop_carried_out" in source_id and "@loop_carried_in" in target_id:
                continue
            ns_s = ns_of[source_id]
            # The box that owns this edge is the deepest namespace containing both
            # endpoints -- the longest common segment prefix. The two endpoints
            # diverge (or one is a direct leaf) there, so it is the only box where
            # they map to different immediate-child units.
            s_segments = ns_s.split("/") if ns_s else []
            t_segments = ns_t.split("/") if ns_t else []
            common: list[str] = []
            for a, b in zip(s_segments, t_segments):
                if a != b:
                    break
                common.append(a)
            box = "/".join(common)
            unit_s = _immediate_child_unit(source_id, ns_s, box)
            unit_t = _immediate_child_unit(target_id, ns_t, box)
            if unit_s == unit_t:
                continue
            adjacency = box_adjacency.setdefault(box, {})
            adjacency.setdefault(unit_s, set()).add(unit_t)
            adjacency.setdefault(unit_t, set())

    warnings: list[str] = []
    for box, adjacency in box_adjacency.items():
        for component in _tarjan_nontrivial_sccs(adjacency):
            members = ", ".join(sorted(component))
            box_label = box or "<root>"
            warnings.append(
                f"H group-cycle{tag}: rendered box {box_label!r} contains an "
                f"illegal cycle between its child boxes/leaves {{{members}}}; the "
                f"boxes feed each other through boundary tiles -- fix the spurious "
                f"cross-box edge upstream (do not suppress)."
            )

    for line in warnings:
        _log.warning("graph group-cycle: %s", line)
    return warnings


def projection_weight_bias_widths(nodes: list[dict[str, Any]]) -> list[str]:
    """A projection's weight and bias must agree on how wide its output is.

    A bias is as long as the output, and the weight has that width on one axis
    -- WHICH axis is the op's own convention: ``addmm(bias, x, weight)`` stores
    ``[in, out]`` while ``F.linear`` stores ``[out, in]``. When they disagree,
    the two constants were resolved from different modules.

    This is the check that catches GPT-2's `c_proj`. Both its projections
    reported `c_attn`'s ``[768, 2304]`` against their own ``[768]`` bias while
    every other check passed, because a resolved number looks like any other
    resolved number -- nothing else here compares two constants that describe
    the SAME tensor and must therefore agree. Note the bias was RIGHT and the
    weight wrong, so only comparing them catches it.

    Judged only where the layout is known from the op's ``raw_op`` and both
    constants are present and concrete. Anywhere else this says nothing: a
    missing or symbolic constant is another check's business, and guessing would
    turn a shape gap into a spurious wiring report.
    """
    # Which axis of a 2-D weight carries the output width, by the op's own
    # storage convention. Read from `raw_op`, so no display name is consulted.
    output_axis = {"addmm": -1, "linear": 0}

    by_id = {str(node.get("id") or ""): node for node in nodes}
    sized: dict[str, dict[str, tuple[str, list[int]]]] = {}
    for node in nodes:
        node_id = str(node.get("id") or "")
        marker = ":const:"
        if marker not in node_id:
            continue
        label = str(node.get("label") or "")
        if label not in {"weight", "bias"}:
            continue
        shape = _node_attr_value(node, "output_shape")
        if not shape:
            continue
        dims = re.findall(r"-?\d+", str(shape).split("]")[0])
        if not dims:
            continue
        owner = node_id.rsplit(marker, 1)[0]
        sized.setdefault(owner, {})[label] = (str(shape), [int(d) for d in dims])

    messages: list[str] = []
    for owner, parts in sorted(sized.items()):
        weight = parts.get("weight")
        bias = parts.get("bias")
        if weight is None or bias is None:
            continue
        # A bias names one axis; a weight that is not a matrix is not a
        # projection's, so there is nothing to compare.
        if len(bias[1]) != 1 or len(weight[1]) != 2:
            continue
        axis = output_axis.get(_raw_op(by_id.get(owner, {})))
        if axis is None:
            continue
        if weight[1][axis] == bias[1][0]:
            continue
        messages.append(
            f"{owner}: weight {weight[0]} and bias {bias[0]} describe different "
            f"output widths ({weight[1][axis]} against {bias[1][0]}). A "
            f"projection's bias is as long as its output, so these two "
            f"constants were resolved from DIFFERENT modules -- check that the "
            f"name used to look them up is unambiguous. A module class built at "
            f"several widths collapses in a per-class index, and no shape check "
            f"can see it because a wrong number is still a number."
        )
    return messages


def unresolved_config_dims(nodes: list[dict[str, Any]]) -> list[str]:
    """A rendered shape must not print a symbol that should have RESOLVED.

    ``B``/``S``/``Img``/``Pv`` vary at run time and are legitimately symbolic.
    ``N``/``H``/``D`` and the rest are checkpoint CONSTANTS -- the model states
    them in its config, and the shape machinery lists the keys each one resolves
    from. When one of those reaches a rendered shape as a bare letter it did not
    resolve, and in a shape a bare letter is indistinguishable from a real
    runtime axis, so the reader cannot tell a resolved graph from a broken one.

    Which symbols are constants is not a list kept here: it is every symbol the
    shape machinery does not call a runtime axis, read from there, so a symbol
    added is covered automatically and a runtime one is never flagged.
    """
    from TraceLens.ModelUtils.shape_inference import _RUNTIME_SYMBOLS, Symbol

    config_symbols = {
        symbol.value: symbol for symbol in Symbol if symbol not in _RUNTIME_SYMBOLS
    }
    offenders: dict[str, list[str]] = {}
    for node in nodes:
        shape = _node_attr_value(node, "output_shape")
        if not shape:
            continue
        # Lookarounds, not consuming delimiters: ``[B, N, S, D]`` shares the
        # comma between neighbours, so a consuming pattern matches every OTHER
        # dim and silently misses half of them.
        for token in re.findall(
            r"(?<=[\[,])\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?=[,\]])", str(shape)
        ):
            if token in config_symbols:
                offenders.setdefault(token, []).append(f"{node['id']} -> {shape}")
    messages: list[str] = []
    for token, hits in sorted(offenders.items()):
        name = config_symbols[token].name.lower().replace("_", " ")
        shown = "\n    ".join(hits[:4])
        more = f"\n    ... and {len(hits) - 4} more" if len(hits) > 4 else ""
        messages.append(
            f"{len(hits)} node(s) render the bare symbol {token!r} ({name}). "
            f"{token!r} is a CONFIG CONSTANT, not a runtime axis: the checkpoint "
            f"fixes it, so it must resolve to a number. It did not, so every shape "
            f"built from it carries the letter instead -- and in a rendered shape the "
            f"letter is indistinguishable from a genuine runtime axis like B, S, Img "
            f"or Pv, so the reader cannot tell a resolved graph from a broken one.\n"
            f"  Fix the config resolution: check that the checkpoint config states "
            f"this dimension, and that the model's own config class declares any "
            f"rename the modeling code reads it under (``attribute_map``). Do not "
            f"ship a graph with it.\n    {shown}{more}"
        )
    return messages
