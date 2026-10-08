###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Every slot of a tuple parameter reaches the op that reads it.

DeepSeek's attention does::

    cos, sin = position_embeddings[self.rope_layer_type]
    q  = apply_rotary_pos_emb(q,  cos, sin)
    kv = apply_rotary_pos_emb(kv, cos, sin)

and the callee multiplies by BOTH::

    cos = cos.repeat_interleave(2, dim=-1).unsqueeze(unsqueeze_dim)
    sin = sin.repeat_interleave(2, dim=-1).unsqueeze(unsqueeze_dim)
    rotated = (rope.float() * cos) + (rotate_half(rope).float() * sin)

The analyzer records both slots correctly -- ``{'cos': ('position_embeddings',
0), 'sin': ('position_embeddings', 1)}`` -- and the block tree tags the two
``repeat_interleave`` ops with ordinals 0 and 1. The graph then lost slot 0: the
cross-scope pass dumped the caller's argument onto the frame's FIRST op, which
is the ``cos`` one, so it read the wrong tensor; and because it then already had
an incoming edge, the boundary pass skipped it as already fed. Slot 1 wired
correctly, so ``sin`` was drawn and ``cos`` reached no op at all -- a graph that
multiplies by one of the two things the source multiplies by.
"""

from __future__ import annotations

import collections

import pytest

from TraceLens.ModelUtils.loader import load_model_spec
from TraceLens.ModelUtils.shape_inference import ShapeInferencer
from TraceLens.Visualizer.model_explorer_export.merge import build_merged_model_graph

from tests.model_pins import pin_for

MODEL = "deepseek-ai/DeepSeek-V4-Flash"


@pytest.fixture(scope="module")
def nodes() -> list[dict]:
    pytest.importorskip("huggingface_hub")
    pin = pin_for(MODEL)
    spec = load_model_spec(MODEL, detailed=True, revision=pin.revision if pin else None)
    graph = build_merged_model_graph(spec, shape_inferencer=ShapeInferencer(spec))
    return graph["nodes"]


def _ports_read_from(nodes: list[dict], node_id: str) -> collections.Counter:
    ports: collections.Counter = collections.Counter()
    for node in nodes:
        for edge in node.get("incomingEdges", []) or []:
            if str(edge.get("sourceNodeId")) == node_id:
                ports[str(edge.get("sourceNodeOutputId"))] += 1
    return ports


def _attention_boundaries(nodes: list[dict]) -> list[dict]:
    """The attention's ``position_embeddings`` crossings, one per component."""
    found = [
        node
        for node in nodes
        if "self_attn/@input:position_embeddings" in str(node.get("id", ""))
    ]
    assert found, "expected the attention's position_embeddings boundaries"
    return found


class TestBothRotarySlotsAreRead:
    def test_each_component_crosses_on_its_own_tile(self, nodes: list[dict]) -> None:
        """A tuple is two tensors, so it crosses as two tiles, not one with two ports."""
        labels = {str(node.get("label")) for node in _attention_boundaries(nodes)}
        assert labels == {
            "position_embeddings.cos",
            "position_embeddings.sin",
        }, labels

    def test_every_component_is_consumed(self, nodes: list[dict]) -> None:
        """``cos`` was read by nothing; only ``sin`` ever was."""
        for boundary in _attention_boundaries(nodes):
            ports = _ports_read_from(nodes, str(boundary["id"]))
            assert ports, f"{boundary['id']} reaches no consumer"

    def test_each_rotary_frame_reads_both(self, nodes: list[dict]) -> None:
        """Every ``apply_rotary_pos_emb`` frame multiplies by cos AND sin."""
        frames: dict[str, set[str]] = {}
        by_id = {str(n["id"]): n for n in nodes}
        for node in nodes:
            node_id = str(node.get("id", ""))
            if "apply_rotary_pos_emb" not in node_id:
                continue
            if "repeat_interleave" not in node_id:
                continue
            frame = node_id.rsplit(":@op_", 1)[0]
            for edge in node.get("incomingEdges", []) or []:
                source = by_id.get(str(edge.get("sourceNodeId")), {})
                if "position_embeddings" in str(source.get("label") or ""):
                    frames.setdefault(frame, set()).add(
                        str(edge.get("sourceNodeOutputId"))
                    )
        assert frames, "expected rotary frames reading position_embeddings"
        for frame, slots in frames.items():
            assert slots == {"0", "1"}, (frame, slots)
