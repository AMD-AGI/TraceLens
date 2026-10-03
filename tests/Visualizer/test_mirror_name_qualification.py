###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""One scope, one name: a mirror says which box it crosses when the name is taken.

A layer holding two hyperconnections showed ``collapsed`` twice, one leaving
``attn_hc`` and one leaving ``ffn_hc``, with nothing to tell them apart.
"""

from __future__ import annotations

from TraceLens.Visualizer.model_explorer_export.merge import (
    _qualify_colliding_mirrors,
)


def _node(node_id, label, namespace, synthetic=None, sources=()):
    node = {
        "id": node_id,
        "label": label,
        "namespace": namespace,
        "incomingEdges": [
            {"sourceNodeId": source, "sourceNodeOutputId": "0"} for source in sources
        ],
    }
    if synthetic:
        node["attrs"] = [{"key": "synthetic", "value": synthetic}]
    return node


def test_two_output_mirrors_name_the_block_they_left() -> None:
    nodes = [
        _node(
            "layer/attn_hc/@output:collapsed", "collapsed", "layer/attn_hc", "@output"
        ),
        _node("layer/ffn_hc/@output:collapsed", "collapsed", "layer/ffn_hc", "@output"),
        _node(
            "layer/attn_hc/@output:collapsed^collapsed",
            "collapsed",
            "layer",
            "@output_mirror",
            ["layer/attn_hc/@output:collapsed"],
        ),
        _node(
            "layer/ffn_hc/@output:collapsed^collapsed",
            "collapsed",
            "layer",
            "@output_mirror",
            ["layer/ffn_hc/@output:collapsed"],
        ),
    ]

    _qualify_colliding_mirrors(nodes)

    assert [n["label"] for n in nodes] == [
        "collapsed",
        "collapsed",
        "attn_hc.collapsed",
        "ffn_hc.collapsed",
    ]


def test_input_mirrors_name_the_block_they_enter() -> None:
    nodes = [
        _node("attn/@input:x", "x", "attn", "@input"),
        _node("attn/rope_a/@input_mirror:x^x", "x", "attn", "@input_mirror"),
        _node(
            "attn/rope_a/@input",
            "x",
            "attn/rope_a",
            "@input",
            ["attn/rope_a/@input_mirror:x^x"],
        ),
        _node("attn/rope_b/@input_mirror:x^x", "x", "attn", "@input_mirror"),
        _node(
            "attn/rope_b/@input",
            "x",
            "attn/rope_b",
            "@input",
            ["attn/rope_b/@input_mirror:x^x"],
        ),
    ]

    _qualify_colliding_mirrors(nodes)

    assert nodes[0]["label"] == "x", "the block's own declaration keeps the plain name"
    assert nodes[1]["label"] == "rope_a.x"
    assert nodes[3]["label"] == "rope_b.x"


def test_a_name_used_once_is_left_alone() -> None:
    nodes = [
        _node(
            "layer/attn_hc/@output:collapsed", "collapsed", "layer/attn_hc", "@output"
        ),
        _node(
            "layer/attn_hc/@output:collapsed^collapsed",
            "collapsed",
            "layer",
            "@output_mirror",
            ["layer/attn_hc/@output:collapsed"],
        ),
    ]

    _qualify_colliding_mirrors(nodes)

    assert [n["label"] for n in nodes] == ["collapsed", "collapsed"]


def test_qualifying_that_would_not_settle_it_is_declined() -> None:
    """Two mirrors of ONE block under one name gain nothing from its name."""
    nodes = [
        _node("layer/hc/@output:a", "slot", "layer/hc", "@output"),
        _node("layer/hc/@output:b", "slot", "layer/hc", "@output"),
        _node(
            "layer/hc/@output:a^a",
            "slot",
            "layer",
            "@output_mirror",
            ["layer/hc/@output:a"],
        ),
        _node(
            "layer/hc/@output:b^b",
            "slot",
            "layer",
            "@output_mirror",
            ["layer/hc/@output:b"],
        ),
    ]

    _qualify_colliding_mirrors(nodes)

    assert [n["label"] for n in nodes[2:]] == ["slot", "slot"]


def test_the_port_label_follows_the_node_label() -> None:
    mirror = _node(
        "layer/attn_hc/@output:collapsed^collapsed",
        "collapsed",
        "layer",
        "@output_mirror",
        ["layer/attn_hc/@output:collapsed"],
    )
    mirror["outputsMetadata"] = [
        {"id": "collapsed", "attrs": [{"key": "port_label", "value": "collapsed"}]}
    ]
    nodes = [
        _node(
            "layer/attn_hc/@output:collapsed", "collapsed", "layer/attn_hc", "@output"
        ),
        _node("layer/ffn_hc/@output:collapsed", "collapsed", "layer/ffn_hc", "@output"),
        mirror,
        _node(
            "layer/ffn_hc/@output:collapsed^collapsed",
            "collapsed",
            "layer",
            "@output_mirror",
            ["layer/ffn_hc/@output:collapsed"],
        ),
    ]

    _qualify_colliding_mirrors(nodes)

    # The label a reader sees is qualified; the port ID an edge cites is not.
    assert mirror["outputsMetadata"][0]["attrs"][0]["value"] == "attn_hc.collapsed"
    assert mirror["outputsMetadata"][0]["id"] == "collapsed"
