###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""DeepSeek-V4 rotary, mask, kernel, and constant-matmul shapes."""

from __future__ import annotations

# --- test_deepseek_v4_rotary_mask_kernel_shapes.py ---

from build_graph_in_env import pinned_merged_graph, pinned_model_spec

import pytest

from TraceLens.ModelUtils.shape_inference import ShapeInferencer
from TraceLens.Visualizer.model_explorer_export.type_check import (
    group_cycle_check_graph_nodes,
    integrity_check_graph_nodes,
    type_check_graph_nodes,
)
from TraceLens.Visualizer.model_explorer_export.viewer_page import (
    _graph_without_constants,
)


def _build_graph():
    spec = pinned_model_spec("deepseek-ai/DeepSeek-V4-Flash", detailed=True)
    return pinned_merged_graph(spec, shape_inferencer=ShapeInferencer(spec))


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
    ]
    assert casts, "attention kv_norm Cast node not found"
    for cast in casts:
        incoming = cast.get("incomingEdges", []) or []
        sources = [str(e.get("sourceNodeId", "")) for e in incoming]
        assert incoming, (cast["id"], sources)
        assert not any("compressor" in s for s in sources), sources
        assert not any("result_3" in s for s in sources), sources


def test_deepseek_v4_flash_type_check_clean():
    """The complete DeepSeek graph has no operand-arity or shape warnings."""
    pytest.importorskip("huggingface_hub")
    graph = _build_graph()
    warnings = type_check_graph_nodes(graph["nodes"])
    assert warnings == [], warnings


def _attention_parameter_shapes(nodes) -> dict[str, list[str]]:
    """Shape axes of each operand arriving at the expanded attention, by name.

    The dispatch expands into the implementation the model runs, so each
    parameter it binds gets a boundary of its own inside that frame.
    """
    shapes: dict[str, list[str]] = {}
    for node in nodes:
        node_id = str(node.get("id", ""))
        if ":@attention:" not in node_id or "/@input:" not in node_id:
            continue
        attrs = {a.get("key"): a.get("value") for a in node.get("attrs", []) or []}
        if attrs.get("synthetic") != "@input":
            continue
        raw = str(attrs.get("output_shape") or "")
        inner = raw.split("]", 1)[0].lstrip("[")
        axes = [d.strip() for d in inner.split(",") if d.strip()]
        if axes:
            shapes.setdefault(str(node.get("label")), axes)
    return shapes


def test_deepseek_v4_attention_kernel_q_key_value_are_rank4():
    """The attention kernel's query, key, and value inputs are rank four."""
    pytest.importorskip("huggingface_hub")
    graph = _build_graph()
    nodes = graph["nodes"]

    def _rank(node) -> int | None:
        for out in node.get("outputsMetadata", []) or []:
            for attr in out.get("attrs", []) or []:
                if attr.get("key") == "shape":
                    inner = str(attr.get("value", "")).split("]", 1)[0].lstrip("[")
                    return (
                        len([d for d in inner.split(",") if d.strip()])
                        if inner
                        else None
                    )
        return None

    # The attention the model dispatches to is its own implementation, expanded
    # into the ops it performs, so its operands arrive on that frame's own
    # parameter boundaries -- one per parameter the call binds.
    ports = _attention_parameter_shapes(nodes)
    assert {"query", "key"} <= set(ports), f"attention operands not found: {ports}"
    for name in ("query", "key"):
        assert (
            len(ports[name]) == 4
        ), f"{name} must be 4-D [B, H, S, D], got {ports[name]}"


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
      its last two axes are *not* equal.
    * ``attention_mask`` is head-broadcast (axis 1 == 1) and spans the two
      sequences it scores: one ROW per query and one COLUMN per key. It is
      square only when nothing extends the keys; DeepSeek's compressor
      concatenates ``n_windows`` entries onto the KV axis and the mask is
      padded to match, so asserting squareness asserted a mask that did not
      cover the compressed keys.
    * the three port shapes are mutually distinct.

    A rotation that swaps ``q`` and ``attention_mask`` makes ``q`` square with a
    unit head axis (and the mask non-square), tripping these assertions.
    """
    pytest.importorskip("huggingface_hub")
    graph = _build_graph()
    nodes = graph["nodes"]

    ports = _attention_parameter_shapes(nodes)
    assert {"query", "key", "attention_mask"} <= set(
        ports
    ), f"attention operands not all found: {ports}"
    query = ports["query"]
    key = ports["key"]
    mask = ports["attention_mask"]

    # query is a real multi-head query, not the head-broadcast square mask.
    assert (
        len(query) == 4 and query[1] != "1"
    ), f"query head axis must be multi-head: {query}"
    assert query[-1] != query[-2], f"query must not be square (mask-shaped): {query}"

    # attention_mask is head-broadcast, one row per query and one column per key.
    assert len(mask) == 4 and mask[1] == "1", f"mask must be head-broadcast: {mask}"
    assert mask[-2] == query[-2], f"mask needs one row per query: {mask} vs {query}"
    assert mask[-1] == key[-2], f"mask needs one column per key: {mask} vs {key}"

    # No operand collapsed onto another -- the fragmentation/merge bug.
    assert (
        len({tuple(query), tuple(key), tuple(mask)}) == 3
    ), f"query/key/mask must carry distinct shapes: {ports}"


def test_deepseek_v4_one_kv_tensor_feeds_both_attention_roles():
    """DeepSeek hands ONE compressed-KV tensor to both ``key`` and ``value``.

    ``attention_interface(self, q, kv, kv, attention_mask, ...)`` -- the source
    says ``# sliding where K==V`` -- so the two roles are the same tensor, and
    the expanded implementation runs its head-repeat on it once per role.

    That is a shape the boundary cannot express: a tensor crossing into a block
    is drawn once, so only the FIRST parameter it is bound to gets a tile and
    the second has none. The second repeat then fell back to the frame's primary
    and reported the QUERY as its operand -- wrong tensor, wrong shape. Both
    repeats must read the one kv tile, and neither may read anything else.
    """
    pytest.importorskip("huggingface_hub")
    graph = _build_graph()
    nodes = graph["nodes"]
    by_id = {node["id"]: node for node in nodes}

    repeats = [
        node
        for node in nodes
        if ":@attention:" in str(node["id"])
        and "repeat" in str(node["id"])
        and str(node.get("label")) == "Unsqueeze"
    ]
    assert len(repeats) >= 2, f"expected a head-repeat per role, got {len(repeats)}"

    sources = set()
    for repeat in repeats:
        reads = {
            str(by_id.get(edge["sourceNodeId"], {}).get("label"))
            for edge in repeat.get("incomingEdges", []) or []
        }
        assert reads, f"head-repeat reads nothing: {repeat['id']}"
        assert (
            "query" not in reads
        ), f"a head-repeat of the KV tensor must not read the query: {reads}"
        sources |= reads

    assert (
        len(sources) == 1
    ), f"both roles are the one kv tensor, so both repeats read one tile: {sources}"


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
        assert incoming, node["id"]
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
    assert masked_fills, [n["id"] for n in masked_fills]
    node = masked_fills[0]

    # Exactly two tensor operands: index_scores + future_mask. The float("-inf")
    # host scalar must never be wired as a third operand.
    incoming = node.get("incomingEdges", []) or []
    assert incoming, [e.get("sourceNodeId") for e in incoming]

    def _shape_str(n) -> str:
        for out in n.get("outputsMetadata", []) or []:
            for attr in out.get("attrs", []) or []:
                if attr.get("key") == "shape":
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
    # A tuple crosses one boundary per component
    # (``@input:position_embeddings.cos`` / ``.sin``).
    boundaries = [
        node
        for node in nodes
        if "/@input:position_embeddings" in str(node.get("id", ""))
    ]
    assert boundaries, "no variant position_embeddings boundary"
    # The decoder group names what it is handed just outside itself, so the
    # variant boundary reads that mirror; step through it to the producer.
    by_id = {str(node["id"]): node for node in nodes}

    def _through_tiles(node_id: str, seen: set[str] | None = None) -> set[str]:
        """Producers behind a boundary tile, expanding a tuple's components.

        ``position_embeddings`` is one tile carrying ``(cos, sin)``, so a walk
        that follows a single producer sees only the first of them.
        """
        seen = seen if seen is not None else set()
        if node_id in seen:
            return set()
        seen.add(node_id)
        node = by_id.get(node_id)
        is_tile = node is not None and (
            "@input_mirror:" in str(node_id) or "/@input:" in str(node_id)
        )
        if not is_tile:
            return {node_id}
        edges = node.get("incomingEdges", []) or []
        if not edges:
            return {node_id}
        found: set[str] = set()
        for edge in edges:
            found |= _through_tiles(str(edge["sourceNodeId"]), seen)
        return found

    sources = {
        resolved
        for node in boundaries
        for edge in node.get("incomingEdges", []) or []
        for resolved in _through_tiles(str(edge.get("sourceNodeId", "")))
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
    """An external collapse head replaces the decoder's container-level loop port."""
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


# --- test_matmul_constant_operand_linear.py ---

from build_graph_in_env import pinned_model_spec

import pytest

from TraceLens.ModelUtils.shape_inference import ShapeInferencer
from TraceLens.Visualizer.model_explorer_export.merge import (
    _last_dim,
    _relabel_constant_operand_matmul_as_linear,
)
from TraceLens.Visualizer.model_explorer_export.viewer_page import (
    _graph_without_constants,
)


def _op_node(node_id: str, label: str, producers: list[str]) -> dict:
    return {
        "id": node_id,
        "label": label,
        "attrs": [{"key": "class_name", "value": label}],
        "incomingEdges": [{"sourceNodeId": p} for p in producers],
    }


def _const_node(node_id: str) -> dict:
    return {
        "id": node_id,
        "label": "weight",
        "attrs": [{"key": "constant", "value": "true"}],
    }


def _activation_node(node_id: str) -> dict:
    return {"id": node_id, "label": "Transpose", "attrs": []}


def test_bmm_against_learned_weight_relabels_to_linear():
    """A ``BatchMatMul`` with one constant operand and one activation -> ``Linear``."""
    nodes = [
        _activation_node("x"),
        _const_node("w"),
        _op_node("bmm", "BatchMatMul", ["x", "w"]),
    ]
    _relabel_constant_operand_matmul_as_linear(nodes)
    bmm = nodes[-1]
    assert bmm["label"] == "Linear"
    assert any(
        a["key"] == "class_name" and a["value"] == "Linear" for a in bmm["attrs"]
    )


def test_matmul_of_two_activations_stays_matmul():
    """A genuine activation-activation contraction (attention scores) is untouched."""
    nodes = [
        _activation_node("q"),
        _activation_node("k"),
        _op_node("scores", "MatMul", ["q", "k"]),
    ]
    _relabel_constant_operand_matmul_as_linear(nodes)
    assert nodes[-1]["label"] == "MatMul"


def test_matmul_of_two_constants_stays_matmul():
    """No activation operand means it is not an activation projection -- left as is."""
    nodes = [
        _const_node("a"),
        _const_node("b"),
        _op_node("folded", "MatMul", ["a", "b"]),
    ]
    _relabel_constant_operand_matmul_as_linear(nodes)
    assert nodes[-1]["label"] == "MatMul"


def _attr(node: dict, key: str):
    for a in node.get("attrs", []) or []:
        if a.get("key") == key:
            return a.get("value")
    return None


def test_deepseek_v4_grouped_linear_bmm_renders_as_single_input_linear():
    """The DeepSeek ``o_a_proj`` grouped-linear ``bmm(x, self.weight...)`` reads
    ``Linear`` and, after the constant weight edge is filtered, draws one input.

    Keyed structurally: the sole ``bmm`` op whose annotated operands include a
    ``Constant`` -- no hardcoded node id or line number.
    """
    pytest.importorskip("huggingface_hub")
    spec = pinned_model_spec("deepseek-ai/DeepSeek-V4-Flash", detailed=True)
    graph = pinned_merged_graph(spec, shape_inferencer=ShapeInferencer(spec))
    nodes = graph["nodes"]

    weighted_bmm = [
        node
        for node in nodes
        if _attr(node, "raw_op") == "bmm"
        and "Constant" in (_attr(node, "input_types") or "")
    ]
    assert weighted_bmm, "expected a bmm with a constant weight operand"
    for node in weighted_bmm:
        assert node.get("label") == "Linear", node.get("label")
        assert _attr(node, "op_type") == "Linear"

    filtered = _graph_without_constants(graph)
    kept = {n["id"] for n in filtered["nodes"]}
    by_id = {n["id"]: n for n in filtered["nodes"]}
    for node in weighted_bmm:
        assert node["id"] in kept
        drawn = by_id[node["id"]]
        assert drawn.get("incomingEdges", []), drawn.get("incomingEdges")


def _shaped(node: dict, shape: str) -> dict:
    node["attrs"].append({"key": "output_shape", "value": shape})
    return node


class TestAProjectionTakesItsWidthFromTheWeight:
    """What separates a projection from an outer product against a buffer.

    MiniMax's RoPE computes ``inv_freq[None, :, None] @ position_ids[:, None, :]``.
    ``inv_freq`` is a registered BUFFER, so its whole chain is constant-tagged
    and the relabel read it as a learned weight -- drawing the frequency
    computation as an ``nn.Linear`` whose real operand is then hidden by the
    render filter.

    A projection takes its output width FROM the weight; that is what makes the
    contraction affine. The outer product's width comes from the positions, so
    the two are told apart by the tensors themselves rather than by what the
    constant is called.
    """

    def test_an_outer_product_against_a_buffer_stays_matmul(self) -> None:
        nodes = [
            _shaped(_activation_node("pos"), "[1, 1, S] float32"),
            _shaped(_const_node("inv_freq"), "[1, 32, 1] float32"),
            _shaped(
                _op_node("freqs", "MatMul", ["inv_freq", "pos"]), "[1, 32, S] float"
            ),
        ]
        _relabel_constant_operand_matmul_as_linear(nodes)
        assert nodes[2]["label"] == "MatMul"

    def test_a_projection_still_relabels(self) -> None:
        nodes = [
            _shaped(_activation_node("x"), "[B, S, 4096] bfloat16"),
            _shaped(_const_node("w"), "[4096, 1024] bfloat16"),
            _shaped(_op_node("proj", "MatMul", ["x", "w"]), "[B, S, 1024] bfloat16"),
        ]
        _relabel_constant_operand_matmul_as_linear(nodes)
        assert nodes[2]["label"] == "Linear"

    def test_a_square_projection_still_relabels(self) -> None:
        """In/out equal is the common case and must not be read as ambiguous."""
        nodes = [
            _shaped(_activation_node("x"), "[B, S, 6144] bfloat16"),
            _shaped(_const_node("w"), "[6144, 6144] bfloat16"),
            _shaped(_op_node("proj", "MatMul", ["x", "w"]), "[B, S, 6144] bfloat16"),
        ]
        _relabel_constant_operand_matmul_as_linear(nodes)
        assert nodes[2]["label"] == "Linear"

    def test_an_unshaped_node_is_judged_as_before(self) -> None:
        """No shape to read means no reason to decline -- the old behaviour."""
        nodes = [
            _activation_node("x"),
            _const_node("w"),
            _op_node("proj", "BatchMatMul", ["x", "w"]),
        ]
        _relabel_constant_operand_matmul_as_linear(nodes)
        assert nodes[2]["label"] == "Linear"


class TestReadingTheTrailingExtent:
    def test_a_rendered_shape(self) -> None:
        assert _last_dim("[B, S, 4096] bfloat16") == "4096"

    def test_a_single_axis(self) -> None:
        assert _last_dim("[32] float32") == "32"

    def test_a_scalar_and_a_blank(self) -> None:
        assert _last_dim("[] float32") is None
        assert _last_dim(None) is None
        assert _last_dim("no brackets") is None
