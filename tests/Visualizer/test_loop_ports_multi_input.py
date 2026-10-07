###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Carrying one value is not the same as being handed one.

``Loop in``/``Loop out`` exist to make the back edge legible, and a loop that
both carries a single value and is handed nothing else can drop them: the
``{N}x_`` badge plus the body's own boundaries already say everything.

A body handed SEVERAL tensors cannot. DeepSeek's expert loop carries one
accumulator but is also handed the hidden states, two residual streams and the
routing weights; with five tensors entering, which of them is the recurrence is
exactly what a reader cannot infer -- and that is what the back edge says.
``Loop_256_iterations`` (DeepSeek) and ``Loop_288_iterations`` (GLM) were folded
on the carried count alone and lost their ports.
"""

from __future__ import annotations

import pytest

from TraceLens.Visualizer.model_explorer_export.merge import _fold_simple_loop_ports


def _port(prefix: str, kind: str, loop: str, var: str, namespace: str) -> dict:
    return {
        "id": f"{prefix}@loop_carried_{kind}:{loop}:{var}",
        "label": "Loop in" if kind == "in" else "Loop out",
        "namespace": namespace,
        "attrs": [{"key": "synthetic", "value": "@loop_carried"}],
        "incomingEdges": [],
    }


def _body_tile(prefix: str, kind: str, loop: str, var: str, namespace: str) -> dict:
    return {
        "id": f"{prefix}@body_{kind}:{loop}:{var}",
        "label": var,
        "namespace": namespace,
        "attrs": [
            {"key": "synthetic", "value": "@input" if kind == "in" else "@output"}
        ],
        "incomingEdges": [],
    }


def _loop(extra_inputs: int) -> list[dict]:
    body = "box/Loop_8_iterations"
    nodes = [
        _port("box/", "in", "loop_l1", "final", "box"),
        _port("box/", "out", "loop_l1", "final", "box"),
        _body_tile("box/", "in", "loop_l1", "final", body),
        _body_tile("box/", "out", "loop_l1", "final", body),
    ]
    for index in range(extra_inputs):
        nodes.append(
            {
                "id": f"{body}/@input:extra{index}",
                "label": f"extra{index}",
                "namespace": body,
                "attrs": [{"key": "synthetic", "value": "@input"}],
                "incomingEdges": [],
            }
        )
    return nodes


def _ports(nodes: list[dict]) -> list[str]:
    return [str(n["id"]) for n in nodes if "@loop_carried" in str(n["id"])]


class TestWhenPortsAreKept:
    def test_a_body_handed_only_what_it_carries_folds(self) -> None:
        nodes = _loop(extra_inputs=0)
        _fold_simple_loop_ports(nodes)
        assert _ports(nodes) == []

    def test_a_body_handed_another_tensor_keeps_its_ports(self) -> None:
        nodes = _loop(extra_inputs=1)
        _fold_simple_loop_ports(nodes)
        assert len(_ports(nodes)) == 2, _ports(nodes)

    def test_a_body_handed_several_keeps_its_ports(self) -> None:
        nodes = _loop(extra_inputs=4)
        _fold_simple_loop_ports(nodes)
        assert len(_ports(nodes)) == 2, _ports(nodes)


@pytest.mark.parametrize(
    "model_id,frame,count",
    [
        ("deepseek-ai/DeepSeek-V4-Flash", "Loop_256_iterations", "256"),
        ("zai-org/GLM-5.3-Flash", "Loop_288_iterations", "288"),
    ],
)
def test_the_loop_brackets_its_body(model_graph_nodes, model_id, frame, count):
    """The body sits between the ports, and the count is on the way in."""
    nodes = model_graph_nodes(model_id)
    outer = {
        str(n.get("namespace") or "").rsplit("/", 1)[0]
        for n in nodes
        if frame in str(n.get("namespace") or "")
    }
    assert outer, f"no {frame} in the graph"
    ports = [
        n
        for n in nodes
        if "@loop_carried" in str(n["id"]) and str(n.get("namespace") or "") in outer
    ]
    labels = {str(n.get("label") or "") for n in ports}
    assert f"Loop in - iterations:{count}" in labels, labels
    assert "Loop out" in labels, labels
