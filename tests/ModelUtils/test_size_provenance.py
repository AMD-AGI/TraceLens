###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Torch-free tests for tracing an exported size read back to its source tensor."""

import textwrap

import pytest

from TraceLens.ModelUtils.size_provenance import (
    Frame,
    SizeReadResolver,
    TracedNode,
    parse_stack_trace,
)

SOURCE = textwrap.dedent("""
    import torch


    def table(n):
        return torch.arange(n)


    class Layer(torch.nn.Module):
        def mix(self, x):
            h = x * 2
            b, n, _ = x.shape
            return h + torch.arange(b)

        def forward(self, x):
            for _ in range(2):
                y = self.mix(x)
                x = x + y
            return self.norm(x)


    class Model(torch.nn.Module):
        def forward(self, W, mask):
            B, N = W.shape
            t = table(max(N, 8))
            x = self.emb(W)
            for layer in self.layers:
                x = layer(x)
            return torch.arange(4)
""")


@pytest.fixture
def model(tmp_path):
    path = tmp_path / "model.py"
    path.write_text(SOURCE)
    lines = SOURCE.splitlines()

    def frame(snippet, name):
        return Frame(str(path), lines.index(snippet) + 1, name)

    return {
        "table": frame("    return torch.arange(n)", "table"),
        "mix_h": frame("        h = x * 2", "mix"),
        "mix_arange": frame("        return h + torch.arange(b)", "mix"),
        "mix_call": frame("            y = self.mix(x)", "forward"),
        "layer_add": frame("            x = x + y", "forward"),
        "layer_norm": frame("        return self.norm(x)", "forward"),
        "t": frame("        t = table(max(N, 8))", "forward"),
        "emb": frame("        x = self.emb(W)", "forward"),
        "layer_call": frame("            x = layer(x)", "forward"),
        "literal": frame("        return torch.arange(4)", "forward"),
        "path": str(path),
    }


def _layer_round(f, calls, tag):
    """The nodes one ``mix`` round of one layer emits: h, its arange, then the add."""
    mix = (f["layer_call"], f["mix_call"])
    return [
        TracedNode(f"h_{tag}", mix + (f["mix_h"],), calls),
        TracedNode(f"arange_{tag}", mix + (f["mix_arange"],), calls),
        TracedNode(f"add_{tag}", (f["layer_call"], f["layer_add"]), calls),
    ]


def _layer_norm(f, calls, tag):
    return TracedNode(
        f"norm_{tag}", (f["layer_call"], f["layer_norm"]), calls + ("norm",)
    )


@pytest.fixture
def resolver(model):
    f = model
    root = ("root",)
    nodes = [
        TracedNode("w", ()),
        TracedNode("mask", ()),
        TracedNode("arange_t", (f["t"], f["table"]), root),
        TracedNode("emb", (f["emb"],), root + ("emb",)),
        *_layer_round(f, root + ("layers.0",), "l0r0"),
        *_layer_round(f, root + ("layers.0",), "l0r1"),
        _layer_norm(f, root + ("layers.0",), "l0"),
        *_layer_round(f, root + ("layers.1",), "l1r0"),
        *_layer_round(f, root + ("layers.1",), "l1r1"),
        _layer_norm(f, root + ("layers.1",), "l1"),
        TracedNode("arange_literal", (f["literal"],), root),
    ]
    return SizeReadResolver(nodes, {"W": "w", "mask": "mask"})


def test_size_through_a_helper_and_max_is_read_from_the_input(resolver):
    # t = table(max(N, 8)) after B, N = W.shape: W, not mask.
    assert resolver.sources_of("arange_t") == ["w"]


def test_first_round_of_first_layer_reads_the_value_passed_in(resolver):
    # x in mix <- x in Layer.forward (a parameter, no round has run yet)
    # <- x = layer(x), whose previous value came from x = self.emb(W).
    assert resolver.sources_of("arange_l0r0") == ["emb"]


def test_later_round_reads_the_previous_round_of_the_same_layer(resolver):
    assert resolver.sources_of("arange_l0r1") == ["add_l0r0"]


def test_next_layer_reads_the_previous_layer_not_its_own_call(resolver):
    # x = layer(x): the value passed to layer 1 is layer 0's output (its
    # norm), never a node of the call in progress (h_l1r0), and never a node
    # of another layer's own rounds (add_l0r1).
    assert resolver.sources_of("arange_l1r0") == ["norm_l0"]
    assert resolver.sources_of("arange_l1r1") == ["add_l1r0"]


def test_literal_sized_op_has_no_source(resolver):
    assert resolver.sources_of("arange_literal") == []


def test_unknown_source_file_resolves_to_nothing():
    frame = Frame("/nonexistent/model.py", 3, "forward")
    resolver = SizeReadResolver([TracedNode("arange", (frame,))], {})
    assert resolver.sources_of("arange") == []


def test_parse_stack_trace_keeps_model_frames_from_the_root_forward(model):
    path = model["path"]
    text = (
        f'File "/driver.py", line 9, in main\n    build()\n'
        f'  File "{path}", line 2, in inventory\n    export(m)\n'
        f'  File "/torch/nn/module.py", line 5, in _call_impl\n    x\n'
        f'  File "{path}", line 24, in forward\n    t = table(max(N, 8))\n'
        f'  File "{path}", line 6, in table\n    return torch.arange(n)\n'
    )
    assert parse_stack_trace(text, {path}) == (
        Frame(path, 24, "forward"),
        Frame(path, 6, "table"),
    )
    assert parse_stack_trace(None) == ()
