###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Regression tests for DeepSeek-V4-Flash compressor/indexer structural integrity.

Expanding ``compressor`` from an opaque leaf into its real class
(``DeepseekV4CSACompressor``) surfaced a batch of I1 dead-node / I2 no-source
integrity warnings. Two distinct wiring bugs were fixed at the extraction/graph-
build source (never by pruning a node or suppressing a warning):

1. A fully inline-flattened, multi-return composite (``cos, sin =
   self.rotary_emb(...)``) collapsed its identity onto a single physical node (the
   last-built internal producer) in ``attr_last_index``. A sibling step reading a
   *specific* return ordinal then fabricated a port tag onto that one node instead
   of docking onto the return slot's own producer, permanently orphaning the other
   slot's real op (``computation_graph._track_attr_index`` / ``_multi_return_slot_key``
   / ``_operation_source_indices``).
2. ``DeepseekV4Indexer.forward`` mutates a host-allocated buffer via slice
   assignment (``new_kv[:, :, ratio:] = chunk_kv[..., self.head_dim:]``). The
   assignment target is a ``Subscript``, not a ``Name``, so the generic
   name-keyed ``_bind`` path silently dropped it: the RHS operand's producer
   (``chunk_kv``'s View, ``chunk_gate``'s Add) was resolved by ``expression()``
   but never wired to anything, and the mutated buffer's own ``var_producer``
   entry was never rebound, so later reads of it kept resolving to the original
   allocation instead of the assignment chain
   (``ast_analyze._ForwardOperationExtractor.statements``, the ``ast.Assign``
   Subscript-target branch, mirroring the pre-existing ``x[idx].copy_(y)``
   in-place-method handler).

These tests rebuild the real model graph and assert the fix holds end to end,
not just at the unit level covered in
``tests/Modeling/test_ast_graph_coverage.py::test_subscript_target_assignment_consumes_rhs_and_rebinds_root``.
"""

from __future__ import annotations

import pytest

from TraceLens.ModelUtils.loader import load_model_spec
from TraceLens.ModelUtils.shape_inference import ShapeInferencer
from TraceLens.Visualizer.model_explorer_export.merge import build_merged_model_graph
from TraceLens.Visualizer.model_explorer_export.type_check import (
    group_cycle_check_graph_nodes,
    integrity_check_graph_nodes,
    type_check_graph_nodes,
)
from TraceLens.Visualizer.model_explorer_export.viewer_page import (
    _graph_without_constants,
)


def _build_graph():
    spec = load_model_spec("deepseek-ai/DeepSeek-V4-Flash", detailed=True)
    return build_merged_model_graph(spec, shape_inferencer=ShapeInferencer(spec))


def test_deepseek_v4_flash_integrity_clean_built_and_render_filtered():
    """No I1 dead-node / I2 no-source warnings survive on either graph."""
    pytest.importorskip("huggingface_hub")
    graph = _build_graph()

    built = integrity_check_graph_nodes(graph["nodes"], label="built")
    assert built == [], built

    filtered_graph = _graph_without_constants(graph)
    filtered = integrity_check_graph_nodes(
        filtered_graph["nodes"], label="render-filtered"
    )
    assert filtered == [], filtered


def test_deepseek_v4_flash_group_cycle_clean_and_kv_norm_single_input():
    """No rendered group-level cycle survives, and the attention ``kv_norm`` Cast
    reads exactly one operand.

    The compressor and the attention's own ``kv_norm`` share the bare submodule
    names ``kv_proj``/``kv_norm``. A flat, last-write-wins predecessor resolver
    threaded the compressor's output into the attention ``kv_norm``'s first Cast
    as a spurious second ``hidden_states`` operand, drawing an illegal 2-cycle
    between the ``kv_norm`` and ``DeepseekV4CSACompressor`` rendered boxes. The
    node-level acyclic check misses it (the underlying node graph stays acyclic);
    the group-level H check is the guardrail. Assert both the H check is silent
    and the Cast has a single incoming edge, on the built and render-filtered
    graphs.
    """
    pytest.importorskip("huggingface_hub")
    graph = _build_graph()

    built = group_cycle_check_graph_nodes(graph["nodes"], label="built")
    assert built == [], built
    filtered_graph = _graph_without_constants(graph)
    filtered = group_cycle_check_graph_nodes(
        filtered_graph["nodes"], label="render-filtered"
    )
    assert filtered == [], filtered

    # The attention (not the compressor's own) kv_norm first op is a `.to()` Cast
    # that takes one tensor; it must have exactly one incoming edge and no
    # cross-scope compressor operand.
    casts = [
        node
        for node in graph["nodes"]
        if str(node.get("namespace", "")).endswith("DeepseekV4Attention/kv_norm")
        and node.get("label") == "Cast"
        and ":@op_l57" in str(node.get("id", ""))
    ]
    assert casts, "attention kv_norm Cast node not found"
    for cast in casts:
        incoming = cast.get("incomingEdges", []) or []
        sources = [str(e.get("sourceNodeId", "")) for e in incoming]
        assert len(incoming) == 1, (cast["id"], sources)
        assert not any("compressor" in s for s in sources), sources
        assert not any("result_3" in s for s in sources), sources


def test_deepseek_v4_flash_type_check_clean():
    """The whole graph type-checks with zero operand-arity / shape warnings.

    End-to-end guard for the coupled MLA attention wiring fix (the rotary
    ``rotate_half`` frame-terminal ``flatten(-2)`` now reduces rank instead of
    passing its shape through, the ``kv = self.kv_norm(...).view(...).transpose``
    chain stays live through the rotary it feeds, and the ``compressor`` /
    grouped-linear sub-scopes no longer over-attach a foreign-scope operand):
    together these previously produced six rotary concat-rank warnings plus two
    ``view: at most 1 tensor operand but 2 wired`` warnings on the ``kv`` view and
    the grouped-linear weight view. Assert the whole model is clean rather than
    counting a fixed number, so a re-regression surfaces as a non-empty list.
    """
    pytest.importorskip("huggingface_hub")
    graph = _build_graph()
    warnings = type_check_graph_nodes(graph["nodes"])
    assert warnings == [], warnings


def test_deepseek_v4_attention_kernel_q_key_value_are_rank4():
    """The attention kernel's ``q`` / ``key`` / ``value`` inputs are all 4-D
    ``[B, H, S, D]``.

    Structural guard for the MLA ``kv`` liveness fix: the
    ``kv = self.kv_norm(...).view(...).transpose(1, 2)`` chain is consumed only by
    the rotary positional step it feeds (a non-op ``step_predecessor``), so the
    backward-liveness walk used to prune its ``.view()/.transpose()`` as dead --
    collapsing ``kv`` back to the rank-3 ``kv_norm`` output and docking the rotary
    on it (phantom rank-3). With the ``_seed`` bridge the layout ops stay live and
    the kernel receives a proper 4-D compressed KV.

    Task J splits the wrapper's single ``kv`` producer into the sdpa primitive's
    distinct ``key`` and ``value`` operand ports (both fed by that one producer),
    so both must resolve to the same rank-4 shape. Keyed on the kernel's own
    input-port labels and the resolved operand rank, not on any line number /
    class name.
    """
    pytest.importorskip("huggingface_hub")
    graph = _build_graph()
    nodes = graph["nodes"]
    by_id = {node["id"]: node for node in nodes}

    def _rank(node) -> int | None:
        for out in node.get("outputsMetadata", []) or []:
            for attr in out.get("attrs", []) or []:
                if attr.get("key") == "tensor_shape":
                    inner = str(attr.get("value", "")).split("]", 1)[0].lstrip("[")
                    return (
                        len([d for d in inner.split(",") if d.strip()])
                        if inner
                        else None
                    )
        return None

    # The attention kernel is the node fed by the named kernel-input ports.
    port_rank: dict[str, int | None] = {}
    for node in nodes:
        for edge in node.get("incomingEdges", []) or []:
            src = edge.get("sourceNodeId", "")
            if "@kernel_in" not in src:
                continue
            port = src.rsplit(":", 1)[-1]
            if port in {"q", "key", "value"} and port not in port_rank:
                port_rank[port] = _rank(by_id.get(src, {}))

    assert {"q", "key", "value"} <= set(
        port_rank
    ), f"kernel q/key/value ports not found: {port_rank}"
    assert (
        port_rank["q"] == 4
    ), f"query kernel input must be 4-D, got rank {port_rank['q']}"
    assert (
        port_rank["key"] == 4
    ), f"key kernel input must be 4-D, got rank {port_rank['key']}"
    assert (
        port_rank["value"] == 4
    ), f"value kernel input must be 4-D, got rank {port_rank['value']}"


def test_deepseek_v4_attention_kernel_ports_carry_correct_distinct_shapes():
    """The kernel's ``q`` / ``key`` / ``value`` / ``attention_mask`` ports keep
    their *own* shapes -- they are not rotated onto each other.

    Regression guard for the wrapper-pipeline core port rotation: the sdpa core
    leaf lacked an ``inputs:`` declaration, so the attention-provenance pass
    routed every port through provenance chains (merging the compressed-KV
    ``kv`` and ``attention_mask`` onto one source), and the ``KernelPipeline``
    core -- unlike the atomic leaf -- was spine-chained from the preceding
    ``Select`` in ``_add_linear_pipeline_chain``, prepending a spurious first
    edge that stole the ``q`` declared-port name and rotated every input off its
    true source (``q`` received the mask's ``[B, 1, S, S]``, etc.). The existing
    rank-4 guard above does not catch this because every port stayed rank 4.

    Asserted structurally (parsed shape axes, keyed on port label), never on
    literal head-count / head-dim numbers:

    * ``q`` is a genuine multi-head query -- its head axis (axis 1) is > 1 and
      its last two axes are *not* equal (not the square attention mask).
    * ``attention_mask`` is head-broadcast (axis 1 == 1) and spatially square
      (its last two axes are equal).
    * the three port shapes are mutually distinct.

    A rotation that swaps ``q`` and ``attention_mask`` makes ``q`` square with a
    unit head axis (and the mask non-square), tripping these assertions.
    """
    pytest.importorskip("huggingface_hub")
    graph = _build_graph()
    nodes = graph["nodes"]
    by_id = {node["id"]: node for node in nodes}

    def _axes(node) -> list[str] | None:
        for out in node.get("outputsMetadata", []) or []:
            for attr in out.get("attrs", []) or []:
                if attr.get("key") == "tensor_shape":
                    inner = str(attr.get("value", "")).split("]", 1)[0].lstrip("[")
                    return [d.strip() for d in inner.split(",") if d.strip()] or None
        return None

    port_axes: dict[str, list[str] | None] = {}
    for node in nodes:
        for edge in node.get("incomingEdges", []) or []:
            src = edge.get("sourceNodeId", "")
            if "@kernel_in" not in src:
                continue
            port = src.rsplit(":", 1)[-1]
            if (
                port in {"q", "key", "value", "attention_mask"}
                and port not in port_axes
            ):
                port_axes[port] = _axes(by_id.get(src, {}))

    assert {"q", "key", "value", "attention_mask"} <= set(
        port_axes
    ), f"kernel q/key/value/attention_mask ports not all found: {port_axes}"
    q = port_axes["q"]
    key = port_axes["key"]
    value = port_axes["value"]
    mask = port_axes["attention_mask"]
    assert q and key and value and mask, port_axes

    # q is a real multi-head query, not the head-broadcast square mask.
    assert len(q) == 4 and q[1] != "1", f"query head axis must be multi-head: {q}"
    assert q[-1] != q[-2], f"query must not be square (mask-shaped): {q}"

    # attention_mask is head-broadcast and spatially square.
    assert len(mask) == 4 and mask[1] == "1", f"mask must be head-broadcast: {mask}"
    assert mask[-1] == mask[-2], f"attention mask must be square: {mask}"

    # Task J: ``key`` and ``value`` are distinct sdpa operand ports fanned out
    # from the one compressed-KV producer, so they carry the *same* shape.
    assert tuple(key) == tuple(
        value
    ), f"key/value ports share the one kv producer's shape: {port_axes}"

    # q, the shared key/value shape, and the mask are mutually distinct (no port
    # collapsed onto another -- the fragmentation/merge bug).
    assert (
        len({tuple(q), tuple(key), tuple(mask)}) == 3
    ), f"q/key(=value)/mask ports must carry distinct shapes: {port_axes}"


def test_deepseek_v4_attention_kernel_splits_kv_into_key_and_value_ports():
    """Task J Part 1: the dispatched sdpa wrapper's single ``kv`` producer fans
    out into the primitive's two distinct ``key`` and ``value`` operand ports.

    DeepSeek is the model where the attention wrapper passes one compressed-KV
    tensor to *both* the ``key`` and ``value`` parameters of
    ``scaled_dot_product_attention``. The export must expose them as two separate
    kernel-input ports (labelled by the primitive's parameter role, ``key`` /
    ``value``, not by the caller variable ``kv``) fed by that one producer -- and
    must NOT retain a single combined ``kv`` port. The sdpa core also carries a
    ``kernel_primitive`` attr naming the primitive so the operand-arity coverage
    check can resolve its required tensor operands. Keyed structurally on the
    kernel's own port labels, never on a class/line/op name.
    """
    pytest.importorskip("huggingface_hub")
    graph = _build_graph()
    nodes = graph["nodes"]

    def _attr(node, key):
        for attr in node.get("attrs", []) or []:
            if attr.get("key") == key:
                return attr.get("value")
        return None

    # Locate the sdpa core(s): nodes carrying a ``kernel_primitive`` attr.
    cores = [n for n in nodes if _attr(n, "kernel_primitive")]
    assert cores, "expected at least one sdpa core carrying a kernel_primitive attr"

    for core in cores:
        prim = _attr(core, "kernel_primitive")
        assert "scaled_dot_product_attention" in str(
            prim
        ), f"unexpected kernel_primitive on sdpa core: {prim!r}"
        port_sources: dict[str, str] = {}
        for edge in core.get("incomingEdges", []) or []:
            src = edge.get("sourceNodeId", "")
            if "@kernel_in" not in src:
                continue
            port_sources[src.rsplit(":", 1)[-1]] = src

        # Distinct key + value ports present; the combined ``kv`` port is gone.
        assert (
            "key" in port_sources and "value" in port_sources
        ), f"sdpa core must expose distinct key/value ports: {sorted(port_sources)}"
        assert (
            "kv" not in port_sources
        ), f"combined ``kv`` port must be split away: {sorted(port_sources)}"
        # Both roles are fed by the one shared kv producer (same source node,
        # differing only by the trailing port-role segment).
        key_root = port_sources["key"].rsplit(":", 1)[0]
        value_root = port_sources["value"].rsplit(":", 1)[0]
        assert key_root == value_root, (
            f"key/value ports must fan out from one producer: "
            f"{port_sources['key']} vs {port_sources['value']}"
        )


def test_deepseek_v4_indexer_chunk_kv_and_gate_are_consumed():
    """``chunk_kv``'s View and ``chunk_gate``'s Add feed the ``new_kv``/``new_gate``
    slice-assignment chain instead of dead-ending.

    Regression guard for the Subscript-target-assignment wiring gap: assert each
    node has at least one consumer (mirrors the ``check-dead-nodes`` skill), keyed
    structurally (by label under the indexer namespace), not by a hardcoded line
    number, so the assertion survives incidental line-number churn upstream.
    """
    pytest.importorskip("huggingface_hub")
    graph = _build_graph()
    nodes = graph["nodes"]
    consumed = {
        edge["sourceNodeId"] for node in nodes for edge in node.get("incomingEdges", [])
    }

    indexer_view_and_add = [
        node
        for node in nodes
        if ":indexer:" in node["id"] and node.get("label") in {"View", "Add"}
    ]
    assert indexer_view_and_add, "expected indexer View/Add nodes to still exist"
    dead = [node["id"] for node in indexer_view_and_add if node["id"] not in consumed]
    assert not dead, f"indexer View/Add nodes with no consumer: {dead}"


def test_expanded_rope_single_tensor_ops_read_one_operand():
    """A ``repeat_interleave``/``unsqueeze`` inside an expanded ``apply_rotary_pos_emb``
    frame reads exactly its one real tensor input.

    Regression guard for the Task-G composite-expansion wiring: a positional
    free-function inlined inside an expanded submodule composite
    (``compressor``/``indexer``'s ``apply_rotary_pos_emb``) reassigns its params
    (``cos = cos.repeat_interleave(...).unsqueeze(...)``) and is fed by a real
    multi-return producer (``cos, sin = self.rotary_emb(...)``) rather than a
    boundary alias. The producer-arg wiring over-attached side operands (the
    other slot, the ``x`` primary, the enclosing module's ``hidden_states``) onto
    the frame's single-tensor first ops. Assert structurally (by label under an
    ``apply_rotary_pos_emb`` namespace) that each such op has exactly one incoming
    tensor edge, mirroring how the same op wires at the top-level Attention call
    site. Keyed by op label, not by line number / class name.
    """
    pytest.importorskip("huggingface_hub")
    graph = _build_graph()
    nodes = graph["nodes"]

    offenders: list[tuple[str, int]] = []
    for node in nodes:
        if "apply_rotary_pos_emb" not in node["id"]:
            continue
        if node.get("label") not in {"Repeat interleave", "Unsqueeze"}:
            continue
        incoming = node.get("incomingEdges", []) or []
        if len(incoming) != 1:
            offenders.append((node["id"], len(incoming)))
    assert not offenders, (
        "expanded-rope single-tensor ops must read exactly one operand; "
        f"over-attached: {offenders}"
    )


def test_repeated_rope_instances_source_own_return_slot():
    """Two calls of the same multi-return submodule in one forward
    (``rotary_emb`` at distinct source lines inside the indexer) each feed their
    own call site.

    Regression guard for the ``_resolve_return_slot_source`` collision: both
    inline-expanded ``rotary_emb`` instances share identical internal op
    attr_names, so a flat ``attr_last_index`` lookup by slot name collapsed both
    onto whichever instance built last -- the first indexer rope frame then read
    the *second* frame's ``cos``. Assert the first ``Repeat interleave`` of each
    distinct indexer ``apply_rotary_pos_emb`` frame sources from a *different*
    producer chain (per-instance disambiguation), keyed structurally.
    """
    pytest.importorskip("huggingface_hub")
    graph = _build_graph()
    nodes = graph["nodes"]
    by_id = {node["id"]: node for node in nodes}

    def _tensor_source(node):
        # Resolve one hop through the frame's @input tile to the real producer.
        incoming = node.get("incomingEdges", []) or []
        assert len(incoming) == 1, node["id"]
        src_id = incoming[0]["sourceNodeId"]
        tile = by_id.get(src_id)
        if tile is not None and "/@input" in src_id:
            tin = tile.get("incomingEdges", []) or []
            if tin:
                return tin[0]["sourceNodeId"]
        return src_id

    # First repeat_interleave (cos slot) of each distinct indexer rope frame.
    first_ri: dict[str, str] = {}
    for node in nodes:
        nid = node["id"]
        if ":indexer:" not in nid or "apply_rotary_pos_emb" not in nid:
            continue
        if node.get("label") != "Repeat interleave":
            continue
        frame = nid.rsplit(":@op", 1)[0]
        first_ri.setdefault(frame, nid)

    assert len(first_ri) >= 2, f"expected >=2 indexer rope frames, got {first_ri}"
    sources = {frame: _tensor_source(by_id[op]) for frame, op in first_ri.items()}
    assert len(set(sources.values())) == len(sources), (
        "distinct rope frames must source cos from their own rotary_emb instance, "
        f"not collapse onto one: {sources}"
    )


def test_deepseek_v4_indexer_masked_fill_two_tensor_operands_and_future_mask_chain():
    """``DeepseekV4Indexer.forward`` L572
    ``index_scores = index_scores.masked_fill(future_mask, float("-inf"))``
    renders with exactly TWO tensor operands and its true result shape/dtype.

    Locks the two coupled fixes:

    * Operand-0 anchoring for first-operand-authoritative writes: a
      ``Tensor.masked_fill(mask, value)`` returns a tensor shaped and typed like
      operand 0 (the tensor being written into), so the node keeps
      ``index_scores``' rank-3 floating shape rather than letting the wider
      boolean ``future_mask`` win the ``_broadcast_rank`` vote and turn the float
      write into a bool tensor of the mask's shape.
    * The ``future_mask`` producer chain (``torch.arange`` -> ``view`` -> ``+ 1``
      -> ``// compress_rate`` -> ``unsqueeze`` -> ``>=``) is retained as visible
      ops instead of pruned as integer/index/bool bookkeeping, so the ``>=``
      boolean producer is a real second operand and the ``float("-inf")`` host
      scalar is never wired as a spurious third operand.

    Structural throughout -- keyed on op labels, operand count and parsed rank,
    never on the ``-inf`` literal / param names.
    """
    pytest.importorskip("huggingface_hub")
    graph = _build_graph()
    nodes = graph["nodes"]
    by_id = {node["id"]: node for node in nodes}

    masked_fills = [
        node
        for node in nodes
        if node.get("label") == "Masked fill"
        and "indexer" in str(node.get("id", ""))
        and "l572" in str(node.get("id", ""))
    ]
    assert len(masked_fills) == 1, [n["id"] for n in masked_fills]
    node = masked_fills[0]

    # Exactly two tensor operands: index_scores + future_mask. The float("-inf")
    # host scalar must never be wired as a third operand.
    incoming = node.get("incomingEdges", []) or []
    assert len(incoming) == 2, [e.get("sourceNodeId") for e in incoming]

    def _shape_str(n) -> str:
        for out in n.get("outputsMetadata", []) or []:
            for attr in out.get("attrs", []) or []:
                if attr.get("key") == "tensor_shape":
                    return str(attr.get("value", ""))
        return ""

    # Output keeps operand-0's rank-3 floating shape, not the bool mask's.
    shape = _shape_str(node)
    axes = shape.split("]", 1)[0].lstrip("[")
    rank = len([d for d in axes.split(",") if d.strip()])
    assert rank == 3, shape
    assert "bool" not in shape, shape
    assert "float" in shape.lower(), shape  # bfloat16/float write target

    # One operand is the boolean future_mask producer -- a ``>=`` comparison.
    src_labels = [
        (by_id.get(e.get("sourceNodeId")) or {}).get("label") for e in incoming
    ]
    assert "Greater equal" in src_labels, src_labels

    # The future_mask chain is materialized: walking upstream from the ``>=``
    # producer must reach a ``torch.arange`` and the ``// compress_rate`` floor
    # division -- proof the index/bool producer chain was not pruned.
    ge_ids = [
        e.get("sourceNodeId")
        for e in incoming
        if (by_id.get(e.get("sourceNodeId")) or {}).get("label") == "Greater equal"
    ]
    seen: set[str] = set()
    frontier = list(ge_ids)
    found: set[str] = set()
    while frontier:
        cur = frontier.pop()
        if cur in seen:
            continue
        seen.add(cur)
        cur_node = by_id.get(cur)
        if cur_node is None:
            continue
        found.add(cur_node.get("label"))
        for edge in cur_node.get("incomingEdges", []) or []:
            frontier.append(edge.get("sourceNodeId"))
    assert "Arange" in found, sorted(found)
    assert "Floor divide" in found, sorted(found)


def test_deepseek_v4_model_scope_rotary_emb_expands_not_opaque_leaf():
    """The model-scope ``position_embeddings = {"main": self.rotary_emb(...), ...}``
    call renders as its real op subgraph, not a single opaque ``rotary_emb@l1313``
    leaf.

    ``DeepseekV4Model.forward`` assigns the rotary call as a dict-literal *value*
    threaded into the decoder loop as a loop-invariant. The loop-invariant
    materializer stamped it as one flat model-scope leaf, hiding the whole tensor
    computation (the dynamic ``inv_freq`` buffer expand, the ``@`` matmul against
    ``position_ids``, the transpose, and the ``cos``/``sin`` split). It must expand
    exactly like the same class does at its nested (compressor/indexer) call sites:
    the positional pre-module is un-skipped and run through the shared section
    machinery, then its real ``@output`` ports are docked into the decoder's
    ``position_embeddings`` boundary. Keyed structurally on op labels and the
    section's own output ports, never on a line number / class name.
    """
    pytest.importorskip("huggingface_hub")
    graph = _build_graph()
    nodes = graph["nodes"]

    # No opaque flat model-scope rotary leaf survives.
    opaque = [
        node
        for node in nodes
        if str(node.get("id", "")).endswith("rotary_emb@l1313")
        or str(node.get("id", "")).endswith("rotary_emb@l1312")
        or str(node.get("id", "")) == "@model_forward/rotary_emb"
    ]
    assert not opaque, (
        "model-scope rotary_emb must expand, not remain an opaque leaf: "
        f"{[n.get('id') for n in opaque]}"
    )

    # The top-level rotary section is expanded into its real op chain.
    section = [
        node for node in nodes if str(node.get("id", "")).startswith("rotary_emb/")
    ]
    labels = {node.get("label") for node in section}
    assert {"MatMul", "Cosine", "Sine"} <= labels, sorted(labels)

    # position_embeddings = (cos, sin): BOTH tuple ports are consumed by the
    # decoder loop boundary (neither slice left dead).
    # The ``{N}x_`` repeat group is the LOOP WRAPPER -- a grouping of variants,
    # not a module -- so it carries no boundary of its own. Each variant that
    # consumes the tuple has its own, and BOTH ports reach them (neither slice
    # left dead).
    assert not any(
        str(node.get("id", "")) == "decoder/@input:position_embeddings"
        for node in nodes
    ), "the loop wrapper must not show an input node"
    boundaries = [
        node
        for node in nodes
        if str(node.get("id", "")).endswith("/@input:position_embeddings")
    ]
    assert boundaries, "no variant position_embeddings boundary"
    sources = {
        str(edge.get("sourceNodeId", ""))
        for node in boundaries
        for edge in node.get("incomingEdges", []) or []
    }
    assert "rotary_emb/@output:cos" in sources, sources
    assert "rotary_emb/@output:sin" in sources, sources

    # No dead node inside the expanded rotary section.
    consumed = {
        edge.get("sourceNodeId")
        for node in nodes
        for edge in node.get("incomingEdges", []) or []
    }
    dead = [
        node["id"]
        for node in section
        if "/@output" not in node["id"] and node["id"] not in consumed
    ]
    assert not dead, f"expanded rotary section has dead nodes: {dead}"


def test_deepseek_v4_main_decoder_has_no_loop_carried_boundary():
    """DeepSeek's uniform main decoder renders like GLM's -- no container-level
    ``@loop_carried`` in/out tiles on the ``43x_DeepseekV4DecoderLayer`` group.

    Both decoders thread a HyperConnection multi-stream residual (rank>3
    ``[B, S, streams, H]``) that an external post-loop head (``hc_head``) collapses
    back to ``[B, S, H]``. That collapse head already renders the cross-iteration
    merge as a visible node, so a synthesized container-level loop-carried boundary
    is redundant -- GLM's heterogeneous ``45x_Glm5NextTextDecoderLayer`` already
    renders without one (its parallel variant branches trip the multiple-exit-source
    guard). The uniform single-template DeepSeek decoder previously slipped past that
    guard and synthesized the boundary; the externally-collapsed-stream predicate now
    suppresses it too, so the two decoders render consistently.

    The ban is on the *outer* decoder-spine boundary only: inner hyperconnection /
    MoE loop-carried tiles live at deeper namespaces
    (``.../attn_hc/Loop_...``, ``.../ffn_hc/Loop_...``,
    ``.../DeepseekV4SparseMoeBlock/loop_...``) and remain legitimate.
    """
    pytest.importorskip("huggingface_hub")
    graph = _build_graph()
    nodes = graph["nodes"]

    # No container-level loop-carried tile on the main decoder repeat group. The
    # synthesized boundary lives at the container namespace exactly; deeper inner
    # loops keep theirs.
    container_lc = [
        node["id"]
        for node in nodes
        if "@loop_carried" in str(node.get("id", ""))
        and str(node.get("namespace", "")) == "43x_DeepseekV4DecoderLayer"
    ]
    assert not container_lc, (
        "uniform main decoder must render like GLM's -- no container-level "
        f"loop-carried boundary: {container_lc}"
    )

    # Removing the boundary left no orphan: the decoder body @output still flows
    # directly to its external consumer (the hyper-connection collapse head), and
    # that consumer is a real node, not a loop-carried tile.
    out_id = "decoder/@output"
    assert any(
        node.get("id") == out_id for node in nodes
    ), "decoder body @output missing"
    consumers = [
        node["id"]
        for node in nodes
        for edge in node.get("incomingEdges", []) or []
        if edge.get("sourceNodeId") == out_id
    ]
    assert consumers, "decoder @output orphaned after boundary removal"
    assert all("@loop_carried" not in c for c in consumers), consumers
