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

    assert (
        "@input:attention_mask" not in by_id
    ), "bogus top-level @input:attention_mask model-input node must not exist"

    boundary = _param_boundary(by_id, "attention_mask")
    if boundary is None:
        # A model (MiniMax-M3) whose attention rebuilds the mask internally has no
        # top-level decoder attention_mask boundary at all; the assertion above
        # already guards against the fabricated model-input node.
        return
    sources = [e.get("sourceNodeId") for e in boundary.get("incomingEdges", [])]
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
