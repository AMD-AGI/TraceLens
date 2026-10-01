###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Regression tests for loop-invariant decoder-input threading (Task A).

A decoder layer receives tensors handed to every iteration by keyword
(``layer(hidden_states, position_embeddings=..., attention_mask=...)``). The
collapsed repeat group surfaces each as a namespaced ``@input:<param>`` tile deep
in the body; unlike the primary spine input (threaded by the loop-carried
boundary), nothing sources them, so they float. ``merge._thread_loop_invariant_inputs``
reconnects each to its legitimate model-level producer, resolved *structurally* by
the input's own name -- a top-level forward parameter, a value assigned from a
``self.<submodule>(...)`` call, or the primary spine input -- never by a hardcoded
class/parameter name. These tests assert the boundaries end up sourced and that
the tightened I2 no-source check is clean across the built + render-filtered graph.
"""

from __future__ import annotations

import pytest

from TraceLens.Visualizer.model_explorer_export.merge import build_merged_model_graph
from TraceLens.Visualizer.model_explorer_export.type_check import (
    integrity_check_graph_nodes,
)
from TraceLens.ModelUtils.loader import load_model_spec
from TraceLens.ModelUtils.shape_inference import ShapeInferencer


def _build_nodes(model_id: str):
    spec = load_model_spec(model_id, detailed=True)
    graph = build_merged_model_graph(spec, shape_inferencer=ShapeInferencer(spec))
    return graph, {n["id"]: n for n in graph["nodes"]}


def _floating_namespaced_inputs(nodes) -> list[str]:
    """Namespaced ``@input`` / ``@input:<param>`` boundaries with no incoming edge.

    A *top-level* model input (empty namespace, no ``/`` in the id) is a legitimate
    sourceless entry point and is excluded; any namespaced one that floats is the
    defect this pass exists to prevent.
    """
    out = []
    for node in nodes:
        nid = node["id"]
        if "/@input" not in nid:
            continue
        if node.get("incomingEdges"):
            continue
        out.append(nid)
    return out


def _publishing_op(by_id, node_id):
    """Follow a block's ``@output:`` tile back to the op inside it that produced
    the value.

    A block publishes what it hands back through an output tile of its own, so
    a consumer one level up reads that tile rather than reaching inside the
    block. These tests are about WHICH op the value comes from, so resolve the
    tile away and assert on the op.
    """
    seen = set()
    while node_id not in seen:
        seen.add(node_id)
        node = by_id.get(node_id)
        if node is None:
            return node_id
        synthetic = next(
            (
                a.get("value")
                for a in node.get("attrs", []) or []
                if a.get("key") == "synthetic"
            ),
            None,
        )
        if synthetic != "@output" or "/@output:" not in str(node_id):
            return node_id
        edges = node.get("incomingEdges", []) or []
        if len(edges) != 1:
            return node_id
        node_id = str(edges[0]["sourceNodeId"])
    return node_id


def test_deepseek_v4_attention_invariant_inputs_are_sourced():
    """The two ``DeepseekV4Attention`` boundaries Task A targets gain a real edge.

    ``position_embeddings`` resolves through a materialized model-scope rotary
    producer; ``attention_mask`` resolves through the materialized mask-builder
    free-function producer (see ``test_decoder_attention_mask_docks_mask_builder``).
    Both must have an incoming edge, and no namespaced ``@input`` may float.
    """
    pytest.importorskip("huggingface_hub")
    graph, by_id = _build_nodes("deepseek-ai/DeepSeek-V4-Flash")

    for boundary in (
        "decoder/self_attn/@input:position_embeddings",
        "decoder/self_attn/@input:attention_mask",
    ):
        node = by_id.get(boundary)
        assert node is not None, f"missing boundary {boundary}"
        assert node.get("incomingEdges"), f"{boundary} still floats"

    assert _floating_namespaced_inputs(graph["nodes"]) == []


def test_minimax_m3_variant_loop_invariant_inputs_are_sourced():
    """The general (non-DeepSeek) resolution path is exercised by MiniMax-M3.

    MiniMax's variant decoder loop leaves ``forward_step_predecessor_args`` empty,
    so threading must resolve structurally: ``position_embeddings`` from the
    ``self.rotary_emb(...)`` submodule producer, ``position_ids`` (the rotary
    submodule's own input) from its top-level parameter boundary, and the nested
    indexer's bare ``@input`` mirrored from the enclosing attention boundary. No
    namespaced ``@input`` may float afterwards.
    """
    pytest.importorskip("huggingface_hub")
    graph, _ = _build_nodes("MiniMaxAI/MiniMax-M3")
    assert _floating_namespaced_inputs(graph["nodes"]) == []


@pytest.mark.parametrize(
    "model_id",
    [
        "deepseek-ai/DeepSeek-V4-Flash",
        "moonshotai/Kimi-K3",
        "zai-org/GLM-5.3-Flash",
        "MiniMaxAI/MiniMax-M3",
    ],
)
def test_decoder_attention_mask_docks_mask_builder(model_id):
    """The decoder ``attention_mask`` boundary is fed by the real mask builder.

    The tensor handed to each decoder iteration as ``attention_mask`` is a DERIVED
    tensor produced by a captured mask-builder free function
    (``create_*_causal_mask(...)``), reassigned before the loop -- NOT the raw model
    forward parameter. So no bogus top-level ``@input:attention_mask`` model-input
    node may exist, and any rendered decoder ``attention_mask`` boundary must dock
    onto the materialized mask-builder producer (a model-scope ``@fn_l...`` source),
    resolved structurally (a captured free-function producer reassigned a loop
    keyword before the loop), never by a hardcoded function/class/param name.
    """
    pytest.importorskip("huggingface_hub")
    graph, by_id = _build_nodes(model_id)

    # The model is not GIVEN an attention mask -- the meta trace is the
    # authority on that, not the forward signature, which merely accepts one.
    # A top-level boundary for it claims an input the model never receives.
    assert (
        "@input:attention_mask" not in by_id
    ), "bogus top-level @input:attention_mask model-input node must not exist"

    boundary = _param_boundary(by_id, "attention_mask")
    if boundary is None:
        # A model (MiniMax-M3) whose attention rebuilds the mask internally has no
        # top-level decoder attention_mask boundary at all; the assertion above
        # already guards against the fabricated model-input node.
        return
    sources = [
        _publishing_op(by_id, str(e.get("sourceNodeId")))
        for e in boundary.get("incomingEdges", [])
    ]
    assert sources, "decoder attention_mask boundary must be sourced"
    for source in sources:
        producer = by_id.get(source)
        assert producer is not None
        assert source.startswith(
            "@model_forward/@fn_l"
        ), f"attention_mask boundary sourced by {source!r}, not a mask-builder node"
        # The builder is expanded into the ops it actually performs, so the
        # boundary docks onto the op that produces its result rather than onto a
        # tile named after the callee. Identity comes from the id (which carries
        # the ``@fn_..._create_*_mask`` call attr) and from the namespace the
        # expansion renders under.
        assert "create" in source.lower() and "mask" in source.lower(), source
        assert "create" in (producer.get("namespace") or "").lower(), producer.get(
            "namespace"
        )


def _param_boundary(by_id, param):
    """The boundary tile for *param*, wherever the hierarchy puts it.

    The loop wrapper groups variants and carries no input of its own, so a
    tensor handed to each iteration surfaces on the VARIANT that consumes it
    (``45x_Decoder/11x_Attention_MoE/@input:attention_mask``), not on the
    wrapper. Match on the trailing boundary name and take the outermost.
    """
    matches = [
        node
        for node_id, node in by_id.items()
        if node_id.endswith(f"/@input:{param}") and "@model_forward" not in node_id
    ]
    if not matches:
        return None
    return min(matches, key=lambda node: str(node["id"]).count("/"))


def _node_attr(node, key: str):
    for attr in node.get("attrs", []) or []:
        if attr.get("key") == key:
            return attr.get("value")
    return None


@pytest.mark.parametrize(
    "model_id",
    [
        "deepseek-ai/DeepSeek-V4-Flash",
        "zai-org/GLM-5.3-Flash",
    ],
)
def test_decoder_position_ids_not_fabricated_when_derived(model_id):
    """A DERIVED ``position_ids`` never fabricates a top-level model input (Task B).

    Both models compute ``position_ids`` internally before the decoder loop
    (``position_ids = cache_position.unsqueeze(0)`` / ``arange(...).unsqueeze(0)``),
    so it is NOT a value the caller handed the model. Rendering a bogus top-level
    ``@input:position_ids`` model-input node (as the exporter did before this fix)
    misrepresents a derived tensor as a graph entry point, so that node must never
    exist for a model that derives it.
    """
    pytest.importorskip("huggingface_hub")
    graph, by_id = _build_nodes(model_id)

    assert "@input:position_ids" not in by_id, (
        "bogus top-level @input:position_ids model-input node must not exist "
        "when the model derives position_ids internally"
    )


def test_deepseek_decoder_position_ids_docks_derived_producer():
    """DeepSeek's decoder ``position_ids`` boundary docks its derived producer.

    DeepSeek surfaces a decoder ``position_ids`` boundary and derives the
    tensor before the loop from a generator-rooted op chain
    (``torch.arange(...) + ... -> unsqueeze(0)``). That chain is materialised at
    model scope, so the boundary must dock onto the terminal derived op source (a
    ``@model_forward/@op_l...`` node carrying a real ``raw_op``), resolved
    structurally from the captured pre-loop op chain -- never a fabricated
    ``@input:position_ids``.
    """
    pytest.importorskip("huggingface_hub")
    graph, by_id = _build_nodes("deepseek-ai/DeepSeek-V4-Flash")

    assert "@input:position_ids" not in by_id

    boundary = _param_boundary(by_id, "position_ids")
    assert boundary is not None, "expected a decoder position_ids boundary"
    sources = [e.get("sourceNodeId") for e in boundary.get("incomingEdges", [])]
    assert sources, "decoder position_ids boundary must be sourced"
    for source in sources:
        producer = by_id.get(source)
        assert producer is not None, f"missing producer {source!r}"
        assert source.startswith(
            "@model_forward/@op_l"
        ), f"position_ids boundary sourced by {source!r}, not a derived op producer"
        assert source.endswith(
            "_unsqueeze"
        ), f"expected the terminal unsqueeze of the derived chain, got {source!r}"
        # A materialised op source (not a fabricated @input) carries the underlying
        # torch op as a top-level raw_op attr.
        assert (
            _node_attr(producer, "raw_op") is not None
        ), f"{source!r} is not a materialised op source"


def test_minimax_m3_position_ids_docks_its_real_derivation():
    """MiniMax-M3 ``position_ids`` reaches the decoder from the ops that build it.

    This used to fabricate a top-level ``@input:position_ids``. The cause was not
    the passthrough spine it was long attributed to: MiniMax iterates
    ``self.layers[: self.config.num_hidden_layers]``, and the module-alias
    resolver only understood a bare ``self.<attr>`` (optionally wrapped in
    ``enumerate``/``reversed``), so a SLICED ModuleList left the loop body's call
    bound to no submodule at all -- and with it every producer the loop hands each
    iteration. Unwrapping the subscript recovers them, so the boundary now docks
    onto the real ``arange -> add -> unsqueeze`` chain at model scope.
    """
    pytest.importorskip("huggingface_hub")
    graph, by_id = _build_nodes("MiniMaxAI/MiniMax-M3")

    assert (
        "@input:position_ids" not in by_id
    ), "position_ids is derived, not a raw model input"
    boundary = _param_boundary(by_id, "position_ids")
    assert boundary is not None
    sources = [e.get("sourceNodeId") for e in boundary.get("incomingEdges", [])]
    assert len(sources) == 1, sources
    assert sources[0].startswith("@model_forward/"), sources
    # ...and that producer is the tail of the real derivation, not a bare tile.
    producer = by_id[sources[0]]
    assert producer.get("label") == "Unsqueeze", producer.get("label")


@pytest.mark.parametrize(
    "model_id",
    ["deepseek-ai/DeepSeek-V4-Flash", "MiniMaxAI/MiniMax-M3"],
)
def test_loop_invariant_threading_i2_clean(model_id):
    """I2 no-source is clean on the built + render-filtered graph after threading."""
    pytest.importorskip("huggingface_hub")
    from TraceLens.Visualizer.model_explorer_export.viewer_page import (
        _graph_without_constants,
    )

    graph, _ = _build_nodes(model_id)

    built = [
        w
        for w in integrity_check_graph_nodes(graph["nodes"], label="built")
        if "I2" in w
    ]
    assert built == [], built

    rendered = _graph_without_constants(graph)
    filtered = [
        w
        for w in integrity_check_graph_nodes(rendered["nodes"], label="render-filtered")
        if "I2" in w
    ]
    assert filtered == [], filtered


def _arange_nodes(nodes):
    return [n for n in nodes if str(n.get("label", "")).lower() == "arange"]


def _sources(node) -> list[str]:
    return [str(e["sourceNodeId"]) for e in node.get("incomingEdges", []) or []]


def test_deepseek_in_body_arange_docks_the_tensor_its_extent_reads():
    """An in-body ``torch.arange`` is wired to the tensor whose extent sizes it.

    DeepSeek's compressors size their index ranges from a *local*
    (``torch.arange(n_windows)``, ``torch.arange(compressed_len)``) that was
    itself computed from a tensor's shape. The generator takes no tensor
    *operand*, so nothing docked onto it and it rendered rootless -- asserting
    the range is independent of the model's data when it is not. Resolving the
    bound through its defining expression recovers the real dependency.
    """
    pytest.importorskip("huggingface_hub")
    graph, _ = _build_nodes("deepseek-ai/DeepSeek-V4-Flash")
    aranges = _arange_nodes(graph["nodes"])
    assert aranges, "expected in-body arange nodes"
    rootless = [n["id"] for n in aranges if not _sources(n)]
    assert rootless == [], rootless


def test_rope_frame_names_its_query_and_position_inputs_apart():
    """A frame's two entering tensors get their own boundary tiles, named apart.

    DeepSeek applies rope through a helper taking the query and the position
    tensor. One of the query's entry steps was still stamped with the position
    parameter's name, so both buckets reported ``position_embeddings`` and the
    label-keyed merge collapsed them onto a single tile -- drawing the query
    under the position tensor's name, with no way to tell which was which.

    Two disjoint producers are two different tensors, so they keep separate
    tiles, and each is named after the parameter its steps agree on.
    """
    pytest.importorskip("huggingface_hub")
    graph, by_id = _build_nodes("deepseek-ai/DeepSeek-V4-Flash")

    frames = {
        node["id"].rsplit("/", 1)[0]
        for node in graph["nodes"]
        if "apply_rotary_pos_emb" in node["id"] and "/@input" in node["id"]
    }
    assert frames, "expected rope frames with input boundaries"

    for frame in frames:
        tiles = [
            node
            for node in graph["nodes"]
            if node["id"].startswith(frame + "/@input")
            and _node_attr(node, "synthetic") == "@input"
        ]
        # One tile per entering tensor, and no two share a name.
        labels = [str(node["label"]) for node in tiles]
        assert len(labels) == len(set(labels)), (frame, labels)
        assert len(tiles) >= 2, (frame, labels)

        # The tile fed by the query does not claim the position tensor's name.
        for tile in tiles:
            sources = [
                by_id[e["sourceNodeId"]]
                for e in tile.get("incomingEdges", []) or []
                if e["sourceNodeId"] in by_id
            ]
            if any(str(s.get("label")) == "q" for s in sources):
                assert "position" not in str(tile["label"]), tile["id"]


def _is_constant(node) -> bool:
    """A constant/learned-weight operand, which is never a merge alternative."""
    if node is None:
        return False
    return any(
        attr.get("key") == "constant" and str(attr.get("value")) == "true"
        for attr in node.get("attrs", []) or []
    )


def _shape_of(node) -> str | None:
    for attr in node.get("attrs", []) or []:
        if attr.get("key") == "output_shape":
            return str(attr.get("value"))
    return None


def test_model_scope_frame_expansion_ops_are_sized():
    """Every op of a model-scope frame expansion carries a real shape.

    The mask builders are expanded into the ops they perform, but that builder
    runs outside the per-section inference pass, so nothing ever sized them --
    and because each op asks its operands, one unsized head blanked the whole
    chain. They are now sized from the inferencer's own rule per op, and the
    dims it reports as the expression that computes them
    (``inputs_embeds.shape[1]``) are resolved against the named tensor.
    """
    pytest.importorskip("huggingface_hub")
    graph, by_id = _build_nodes("deepseek-ai/DeepSeek-V4-Flash")
    ops = [
        node
        for node in graph["nodes"]
        if "create_sliding_window_causal_mask" in node["id"] and ":@op_" in node["id"]
    ]
    assert len(ops) >= 5, len(ops)
    for node in ops:
        shape = _shape_of(node)
        assert shape, f"{node['id']} left unsized"
        assert ".shape[" not in shape, f"{node['id']} kept an unresolved dim: {shape}"

    # The builder chooses its mask FUNCTION through four nested ``if``s
    # (``or_mask_function``, ``and_mask_function``, ``packed_sequence_mask``,
    # ``block_sequence_ids``). Each once rendered as a branch-merge box carrying
    # the tensor straight through -- four boxes in a row that compute nothing,
    # because what the branch picks is a callable, not a tensor. A merge with one
    # surviving alternative has nothing to choose between, so it is folded onto
    # its producer; the same holds for a cast whose output dtype and shape equal
    # its input's.
    for node in ops:
        label = str(node.get("label") or "")
        if label not in {"Merge", "Cast"}:
            continue
        incoming = [
            edge
            for edge in node.get("incomingEdges", []) or []
            if not _is_constant(by_id.get(str(edge.get("sourceNodeId"))))
        ]
        assert label != "Merge" or len(incoming) > 1, f"{node['id']} merges one input"
        if label == "Cast" and len(incoming) == 1:
            producer = by_id[str(incoming[0]["sourceNodeId"])]
            assert _shape_of(node) != _shape_of(producer), f"{node['id']} casts nothing"


@pytest.mark.parametrize(
    "model_id",
    [
        "zai-org/GLM-5.3-Flash",
        "MiniMaxAI/MiniMax-M3",
        "deepseek-ai/DeepSeek-V4-Flash",
    ],
)
def test_a_tensor_leaves_a_block_through_that_block_output(model_id):
    """What a block hands back leaves through an output node of that block.

    The entering side has had this for a while; the leaving side had nothing,
    so a block could name every input it reads and say nothing about what it
    produces. The sdpa kernel frame declared each ``@kernel_in`` port and then
    wired its last op straight into the attention module's reshape one level
    up, and the mask builders let a nested ``maybe_pad_block_sequence_ids``
    reach the decoder the same way. Read as a diagram, those blocks produced
    nothing at all.

    Constants are exempt (a learned weight is wired where it is used, never
    declared as a block output) and so is the root, which no block encloses.
    """
    pytest.importorskip("huggingface_hub")
    graph, by_id = _build_nodes(model_id)

    def namespace(node) -> str:
        return str(node.get("namespace") or "")

    def synthetic(node) -> str:
        return str(
            next(
                (
                    a.get("value")
                    for a in node.get("attrs", []) or []
                    if a.get("key") == "synthetic"
                ),
                "",
            )
        )

    leaks = []
    for node in graph["nodes"]:
        target = namespace(node)
        for edge in node.get("incomingEdges", []) or []:
            source = by_id.get(str(edge.get("sourceNodeId")))
            if source is None or not namespace(source):
                continue
            if any(
                a.get("key") == "constant" and str(a.get("value")) == "true"
                for a in source.get("attrs", []) or []
            ):
                continue
            shared = []
            for one, other in zip(
                [p for p in namespace(source).split("/") if p],
                [p for p in target.split("/") if p],
            ):
                if one != other:
                    break
                shared.append(one)
            if "/".join(shared) == namespace(source):
                continue  # entering the producer's own block, not leaving it
            if synthetic(source) in {
                "@output",
                "@output_mirror",
                "@kernel_port_out",
                "@loop_carried",
            }:
                continue
            leaks.append(f"{source['id']} -> {node['id']}")
    assert not leaks, leaks[:8]


def test_a_loop_carries_the_width_its_body_produces():
    """The loop's entry port reports the body's output, not the seed's producer.

    GLM's vision tower runs ``for blk in self.blocks: hidden_states = blk(...)``
    seeded from the patch embed. The patch embed has no shape rule of its own,
    so its node echoes its input -- the raw patch width ``[Pv, C*T*P*P]`` -- even
    though its expansion ends on a Conv3d that produces ``[Pv, hidden]``. Seeding
    the port from that echo made the whole loop report a width that exists only
    BEFORE the first module runs. Entry port, body input and exit port all carry
    one variable, so they must all report one shape.
    """
    pytest.importorskip("huggingface_hub")
    _, by_id = _build_nodes("zai-org/GLM-5.3-Flash")

    ports = {
        node_id: node
        for node_id, node in by_id.items()
        if "@loop_carried_in:" in node_id or "@loop_carried_out:" in node_id
    }
    assert ports, "expected the vision block loop to declare its ports"
    for node_id, node in ports.items():
        if "@loop_carried_in:" not in node_id:
            continue
        exit_port = by_id.get(
            node_id.replace("@loop_carried_in:", "@loop_carried_out:")
        )
        if exit_port is None:
            continue
        assert _shape_of(node) == _shape_of(exit_port), (
            node_id,
            _shape_of(node),
            _shape_of(exit_port),
        )
        body_in = by_id.get(node_id.replace("@loop_carried_in:", "@body_in:"))
        if body_in is not None:
            assert _shape_of(body_in) == _shape_of(node), (body_in["id"], node_id)


def test_glm_mask_builder_claims_no_shape_without_a_mask_to_narrow():
    """GLM is never given a mask, so the builder asserts nothing about one.

    ``create_recurrent_attention_mask`` slices ``attention_mask``, and the meta
    trace shows this model receives ``input_ids`` only -- the signature merely
    ACCEPTS a mask. With nothing to narrow, the builder falls back to the call's
    first argument (the embedding), which is wrong; what must not happen is a
    fabricated top-level ``@input:attention_mask`` dressing that up as a real
    model input, or an echoed ``[B, S, hidden]`` reported as a mask.

    A missing shape is the honest outcome and keeps the bad wiring visible.
    """
    pytest.importorskip("huggingface_hub")
    graph, by_id = _build_nodes("zai-org/GLM-5.3-Flash")
    assert "@input:attention_mask" not in by_id

    sliced = [
        node
        for node in graph["nodes"]
        if "create_recurrent_attention_mask:@op_" in node["id"]
        and str(node.get("label")) == "Slice"
    ]
    assert sliced, "expected the GLM mask builder's slice op"
    for node in sliced:
        assert not _shape_of(node), (node["id"], _shape_of(node))


@pytest.mark.parametrize("model_id", ["MiniMaxAI/MiniMax-M3", "zai-org/GLM-5.3-Flash"])
def test_extent_dims_resolve_to_a_real_length(model_id):
    """No op reports a dim as the expression that computes it.

    A rule that cannot evaluate a recorded bound hands back the source line --
    ``torch.arange(key_states.shape[2])`` reporting ``[key_states.shape[2]]`` --
    and every op downstream inherits it. The operands wired to such an op are
    marked as carrying its EXTENT, so the axis is read off them.
    """
    pytest.importorskip("huggingface_hub")
    graph, _ = _build_nodes(model_id)
    unresolved = [
        (node["id"], _shape_of(node))
        for node in graph["nodes"]
        if _shape_of(node) and ".shape[" in str(_shape_of(node))
    ]
    assert unresolved == [], unresolved


def test_a_data_independent_range_is_not_drawn_as_compute():
    """A constant range is tagged, and filtering it strands nothing.

    ``torch.arange(self.local_blocks)`` is the same tensor on every forward, so
    it is a constant and constants are never drawn. Its consumers keep their
    real operands -- an accumulator's zero initializer need not be shown, but
    the accumulate must still have its activation input.
    """
    pytest.importorskip("huggingface_hub")
    graph, _ = _build_nodes("MiniMaxAI/MiniMax-M3")
    nodes = graph["nodes"]
    constant_ranges = [
        node
        for node in nodes
        if str(node.get("label", "")).lower() == "arange"
        and _node_attr(node, "constant") == "true"
    ]
    assert constant_ranges, "expected the config-sized range to be tagged constant"

    kept = {node["id"] for node in nodes if _node_attr(node, "constant") != "true"}
    for node in nodes:
        if node["id"] not in kept or not node.get("incomingEdges"):
            continue
        survivors = [e for e in node["incomingEdges"] if e["sourceNodeId"] in kept]
        assert survivors, f"{node['id']} lost every input when constants are dropped"


def _group_namespaces(nodes) -> set[str]:
    return {str(n.get("namespace") or "") for n in nodes if n.get("namespace")}


def test_every_tensor_entering_a_multi_input_module_is_named():
    """DeepSeek's expert loop names the routed index it used to let in silently.

    The loop names its three ``hidden_states`` slices and then let the routed
    token index cross on a bare op-to-op edge, so the reader could not tell
    which tensor was which. Each unnamed entrant now gets its own tile.
    """
    pytest.importorskip("huggingface_hub")
    graph, _ = _build_nodes("deepseek-ai/DeepSeek-V4-Flash")
    nodes = graph["nodes"]
    expert_groups = [
        ns
        for ns in _group_namespaces(nodes)
        if "loop_sidefeed" in ns and "gather" in ns
    ]
    assert expert_groups, "expected the expert-loop groups"
    for namespace in expert_groups:
        members = [
            n
            for n in nodes
            if str(n.get("namespace") or "") == namespace
            or str(n.get("namespace") or "").startswith(namespace + "/")
        ]
        ids = {n["id"] for n in members}
        entering = {
            str(e["sourceNodeId"])
            for n in members
            for e in n.get("incomingEdges", []) or []
            if str(e["sourceNodeId"]) not in ids
        }
        named = {
            str(e["sourceNodeId"])
            for n in members
            if _node_attr(n, "synthetic")
            in {"@input", "@input_mirror", "@kernel_port_in"}
            for e in n.get("incomingEdges", []) or []
        }
        assert entering <= named, (namespace, sorted(entering - named))


def test_a_module_is_not_split_from_its_own_nested_dataflow():
    """A tensor flowing from a child namespace to its parent is not an entrant.

    Keying a module's membership on its exact namespace makes its own nested
    children look external, which put a boundary in the middle of one module's
    dataflow -- it split GLM's ``expand_kv`` from the projection feeding it, and
    the projection stopped being the block's first step.
    """
    pytest.importorskip("huggingface_hub")
    graph, _ = _build_nodes("zai-org/GLM-5.3-Flash")
    block = [
        node
        for node in graph["nodes"]
        if "expand_kv" in node["id"] and _node_attr(node, "synthetic") is None
    ]
    assert [node["label"] for node in block][:2] == ["Linear", "View"], [
        node["label"] for node in block
    ]


@pytest.mark.parametrize(
    "model_id", ["MiniMaxAI/MiniMax-M3", "deepseek-ai/DeepSeek-V4-Flash"]
)
def test_model_scope_mask_frame_names_its_inputs_and_keeps_shapes(model_id):
    """The mask frame names both tensors it takes, and the body stays sized.

    The frame is a module in the render, and it is handed the mask and the
    position tensor -- two bare edges arriving at the first op with nothing to
    tell them apart. Each entrant now has its own boundary, and each boundary
    reports the shape it carries: the body is sized from its operands, so a
    blank boundary would blank every op below it.
    """
    pytest.importorskip("huggingface_hub")
    graph, _ = _build_nodes(model_id)
    tiles = [
        node
        for node in graph["nodes"]
        if "_causal_mask:@input:" in node["id"]
        and _node_attr(node, "synthetic") == "@input"
    ]
    assert len(tiles) >= 2, [t["id"] for t in tiles]
    labels = [str(t["label"]) for t in tiles]
    assert len(labels) == len(set(labels)), labels
    for tile in tiles:
        assert _shape_of(tile), f"{tile['id']} carries no shape"

    body = [node for node in graph["nodes"] if "_causal_mask:@op_" in node["id"]]
    assert body, "expected the expanded mask body"
    for node in body:
        assert _shape_of(node), f"{node['id']} left unsized"


def test_child_module_method_boundary_names_its_own_parameter():
    """A method called on a child module names the tensor IT takes.

    ``self.indexer.build_block_mask(block_indices, ...)`` hands over
    ``block_indices``, while the indexer's own ``forward`` leads with
    ``hidden_states``. The boundary drawn for the call had no input label of its
    own and fell back to the generic name, so the tile was labelled with a
    tensor it does not carry -- and the ops behind it really do read
    ``block_indices`` (``block_indices < 0``, ``block_indices.masked_fill(...)``),
    so the label was the part that was wrong.
    """
    pytest.importorskip("huggingface_hub")
    graph, _ = _build_nodes("MiniMaxAI/MiniMax-M3")
    tiles = [
        node
        for node in graph["nodes"]
        if "build_block_mask" in str(node.get("namespace") or "")
        and str(node["id"]).endswith("/@input")
    ]
    assert tiles, "expected the build_block_mask boundary"
    for tile in tiles:
        assert str(tile["label"]) == "block_indices", tile["label"]


def test_a_dtype_argument_draws_no_data_edge():
    """Passing ``x.dtype`` hands over a dtype, not ``x``.

    MiniMax calls ``self.indexer.build_block_mask(block_indices, attention_mask,
    key_states.shape[2], query_states.dtype, query_states.device, position_ids)``.
    The dtype and device arguments READ a tensor, so the extractor recorded its
    producer against them -- and the rope that produced ``query_states`` was then
    drawn as a data input of the mask builder, a dependency the model does not
    have. Resolving the callee's parameter names would need its signature, which
    is unavailable for a method on a child module, but the argument expression
    itself is unambiguous.
    """
    pytest.importorskip("huggingface_hub")
    graph, by_id = _build_nodes("MiniMaxAI/MiniMax-M3")
    namespaces = {
        str(n.get("namespace") or "")
        for n in graph["nodes"]
        if str(n.get("namespace") or "").endswith("build_block_mask")
    }
    assert namespaces, "expected the build_block_mask frame"
    for namespace in namespaces:
        members = [
            n
            for n in graph["nodes"]
            if str(n.get("namespace") or "") == namespace
            or str(n.get("namespace") or "").startswith(namespace + "/")
        ]
        ids = {n["id"] for n in members}
        entering = {
            str(by_id[e["sourceNodeId"]].get("label"))
            for n in members
            for e in n.get("incomingEdges", []) or []
            if e["sourceNodeId"] not in ids and e["sourceNodeId"] in by_id
        }
        # The rope output reaches it only through ``query_states.dtype``.
        assert "k_embed" not in entering, (namespace, sorted(entering))


def test_build_block_mask_args_reach_the_ops_that_read_them():
    """Each argument of a child-module method reaches the op that reads it.

    ``self.indexer.build_block_mask(block_indices, attention_mask, ...,
    position_ids)`` used to put every argument on ONE boundary tile. That tile
    fed four ops which read only ``block_indices``, so the graph asserted they
    read ``position_ids`` and ``attention_mask`` too -- false edges -- while
    ``arange(key_length) > position_ids`` was left without its second operand
    altogether.
    """
    pytest.importorskip("huggingface_hub")
    graph, by_id = _build_nodes("MiniMaxAI/MiniMax-M3")
    namespaces = {
        str(n.get("namespace") or "")
        for n in graph["nodes"]
        if str(n.get("namespace") or "").endswith("build_block_mask")
    }
    assert namespaces, "expected the build_block_mask frame"

    def sources(node):
        return [
            str(by_id[e["sourceNodeId"]].get("label"))
            for e in node.get("incomingEdges", []) or []
            if e["sourceNodeId"] in by_id
        ]

    for namespace in namespaces:
        members = [
            n
            for n in graph["nodes"]
            if str(n.get("namespace") or "") == namespace
            or str(n.get("namespace") or "").startswith(namespace + "/")
        ]
        by_label = {}
        for node in members:
            by_label.setdefault(str(node.get("label")), []).append(node)

        # The comparison reads the position tensor, not just its range.
        greater = by_label.get("Greater")
        assert greater, namespace
        assert "position_ids" in sources(greater[0]), sources(greater[0])

        # ...and the ops that read the block indices read ONLY those.
        for label in ("Less", "Scatter"):
            for node in by_label.get(label, []):
                assert "position_ids" not in sources(node), (label, sources(node))
                assert "attention_mask" not in sources(node), (label, sources(node))


def test_vision_range_docks_the_grid_it_is_sized_from():
    """A range sized by host data reaches the tensor that data came from.

    ``for t, h, w in grid_thw.tolist(): torch.arange(t, ...)`` sizes the range
    from the GRID -- read off the tensor on the host, but the model's data all
    the same. The loop variable has no producer of its own, so the range drew
    rootless, asserting it depends on nothing.

    The tensor is a parameter of the frame, so it arrives through the parameter
    channel, and the boundary it crosses on the way is materialised at every
    module level rather than letting the edge descend two levels at once.
    """
    pytest.importorskip("huggingface_hub")
    graph, _ = _build_nodes("zai-org/GLM-5.3-Flash")
    ranges = [
        n
        for n in graph["nodes"]
        if "get_vision_position_ids" in n["id"]
        and str(n.get("label", "")).lower() == "arange"
    ]
    assert ranges, "expected the vision position-id ranges"
    for node in ranges:
        assert node.get("incomingEdges"), f"{node['id']} left rootless"


@pytest.mark.parametrize("model_id", ["MiniMaxAI/MiniMax-M3", "zai-org/GLM-5.3-Flash"])
def test_loop_ports_bracket_the_body_which_names_both_kinds_of_input(model_id):
    """The loop's ports sit OUTSIDE the body, and the body names what it reads.

    A port drawn inside the body sits among the body's own ops, where its
    position relative to the iteration is undefined -- no ordering reads
    correctly for both the seed edge and the back edge. The body nests between
    them instead.

    The body then carries boundaries of its own for BOTH kinds of dependency:
    the carried value (fed by ``Loop in``, feeding ``Loop out``) and the
    loop-invariant tensors each iteration is handed.
    """
    pytest.importorskip("huggingface_hub")
    graph, by_id = _build_nodes(model_id)
    ports = [n for n in graph["nodes"] if _node_attr(n, "synthetic") == "@loop_carried"]
    assert ports, "expected loop-carried ports"

    for port in ports:
        outer = str(port.get("namespace") or "")
        # Whatever the port exchanges the carried value with lives one level in.
        partners = [
            by_id[str(e["sourceNodeId"])]
            for e in port.get("incomingEdges", []) or []
            if str(e["sourceNodeId"]) in by_id
        ] + [
            n
            for n in graph["nodes"]
            for e in n.get("incomingEdges", []) or []
            if str(e.get("sourceNodeId")) == str(port["id"])
        ]
        inside = [
            p for p in partners if str(p.get("namespace") or "").startswith(outer + "/")
        ]
        for partner in inside:
            # The port never reaches into the body: it meets the body's own
            # boundary, which is what the body's ops read.
            assert _node_attr(partner, "synthetic") in {"@input", "@output"}, (
                port["id"],
                partner["id"],
            )
            body_ns = str(partner.get("namespace") or "")
            siblings = [
                n
                for n in graph["nodes"]
                if str(n.get("namespace") or "") == body_ns
                and _node_attr(n, "synthetic") == "@input"
            ]
            # ...alongside the loop-invariant inputs the body also reads.
            assert siblings, body_ns
