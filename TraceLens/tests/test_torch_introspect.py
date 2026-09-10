###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Torch-free tests for the pure-PyTorch introspection Layer A.

These exercise the FX-inventory → ComputationGraph → Model Explorer payload
transforms without importing torch, so they run under the torch-free CI.
"""

import pytest

from TraceLens.ModelUtils.torch_introspect import (
    FxNodeInfo,
    _dotted_prefixes,
    computation_graph_from_fx_inventory,
    explorer_graph_from_fx_inventory,
    payload_from_fx_inventory,
)
from TraceLens.Visualizer.model_explorer_export.cli import parse_input_shape


def _sample_inventory():
    return [
        FxNodeInfo(name="x", op="placeholder", label="x", inputs=[], shape=(2, 128), dtype="float32"),
        FxNodeInfo(
            name="linear",
            op="call_module",
            label="Linear",
            inputs=["x"],
            namespace="encoder.layer_0",
            shape=(2, 256),
            dtype="float32",
        ),
        FxNodeInfo(
            name="relu",
            op="call_function",
            label="relu",
            inputs=["linear"],
            namespace="encoder.layer_0",
            shape=(2, 256),
            dtype="float32",
        ),
        FxNodeInfo(name="out", op="output", label="Output", inputs=["relu"]),
    ]


def test_dotted_prefixes_builds_nested_paths():
    assert _dotted_prefixes("a.b.c") == ["a", "a.b", "a.b.c"]
    assert _dotted_prefixes("") == []
    assert _dotted_prefixes("solo") == ["solo"]


def test_computation_graph_wires_edges_and_output_port():
    graph = computation_graph_from_fx_inventory(_sample_inventory())

    assert [node.key for node in graph.nodes] == ["x", "linear", "relu", "out"]
    # Placeholder and output are synthetic; interior nodes are real.
    assert graph.nodes[0].synthetic == "@input"
    assert graph.nodes[3].synthetic == "@output"
    assert graph.nodes[1].synthetic is None

    # Data-dependency links, excluding the output (wired via output_ports).
    assert graph.links == [(0, 1), (1, 2)]
    assert graph.output_node_index == 3
    assert graph.output_ports == {"result": 2}
    assert graph.primary_output_port == "result"


def test_namespace_frames_nest_by_prefix():
    graph = computation_graph_from_fx_inventory(_sample_inventory())
    frames = {frame.frame_id: frame for frame in graph.inline_frames}

    assert set(frames) == {"encoder", "encoder.layer_0"}
    # Frame label is the last dotted segment (adapter turns nesting into a/b).
    assert frames["encoder"].label == "encoder"
    assert frames["encoder.layer_0"].label == "layer_0"
    # Both interior nodes belong to both frame levels.
    assert frames["encoder"].node_indices == [1, 2]
    assert frames["encoder.layer_0"].node_indices == [1, 2]


def test_explorer_graph_applies_shapes_and_namespaces():
    graph = explorer_graph_from_fx_inventory(_sample_inventory(), graph_id="demo")
    nodes = {node["id"]: node for node in graph["nodes"]}

    assert nodes["linear"]["namespace"] == "encoder/layer_0"

    def _shape(node):
        return next(a["value"] for a in node["attrs"] if a["key"] == "output_shape")

    assert _shape(nodes["x"]) == "2 x 128 float32"
    assert _shape(nodes["relu"]) == "2 x 256 float32"

    # The output node's incoming edge carries the synthesized result port.
    edge = nodes["out"]["incomingEdges"][0]
    assert edge["sourceNodeId"] == "relu"
    assert edge["targetNodeInputId"] == "result"


def test_include_shapes_false_omits_shape_attrs():
    graph = explorer_graph_from_fx_inventory(
        _sample_inventory(), graph_id="demo", include_shapes=False
    )
    for node in graph["nodes"]:
        keys = {attr["key"] for attr in node.get("attrs", [])}
        assert "output_shape" not in keys


def test_payload_envelope_structure():
    payload = payload_from_fx_inventory(_sample_inventory(), name="demo")

    assert payload["name"] == "demo"
    assert payload["source"] == "tracelens-torch-introspection"
    collections = payload["graphCollections"]
    assert len(collections) == 1
    assert collections[0]["label"] == "demo"
    assert len(collections[0]["graphs"][0]["nodes"]) == 4


def test_parse_input_shape_valid_and_invalid():
    assert parse_input_shape("2,128") == (2, 128)
    assert parse_input_shape(" 1 , 3 , 224 , 224 ") == (1, 3, 224, 224)
    with pytest.raises(ValueError):
        parse_input_shape("")
    with pytest.raises(ValueError):
        parse_input_shape("2,abc")
