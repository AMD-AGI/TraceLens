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
    # A node with no stack frames has nothing to trace.
    assert resolver.sources_of("w") == []


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


def _sources(tmp_path, source, graph, roots=None):
    """Build a resolver whose node frames point at snippets of ``source``."""
    # Parsed files are cached by path, so each source gets its own file.
    path = tmp_path / f"model_{hash(source) & 0xFFFFFFFF:08x}.py"
    text = textwrap.dedent(source).strip() + "\n"
    path.write_text(text)
    lines = text.splitlines()

    def at(snippet, name):
        needle = snippet.strip()
        matches = [i for i, line in enumerate(lines) if line.strip() == needle]
        if len(matches) != 1:
            raise AssertionError(f"{needle!r} matched lines {matches}")
        return Frame(str(path), matches[0] + 1, name)

    nodes = []
    for item in graph:
        node_name, stack = item[0], item[1]
        calls = item[2] if len(item) > 2 else ()
        nodes.append(
            TracedNode(
                node_name,
                tuple(at(snippet, func) for snippet, func in stack),
                calls,
            )
        )
    return SizeReadResolver(nodes, roots or {})


def test_size_method_on_a_slice_reads_that_tensor(tmp_path):
    resolver = _sources(
        tmp_path,
        """
        class Model:
            def forward(self, x):
                n = x[0].size(-1)
                return torch.arange(n)
        """,
        [
            (
                "arange",
                [("                return torch.arange(n)", "forward")],
            )
        ],
        {"x": "x_in"},
    )
    assert resolver.sources_of("arange") == ["x_in"]


def test_unary_and_conditional_sizes_read_both_tensors(tmp_path):
    resolver = _sources(
        tmp_path,
        """
        class Model:
            def forward(self, x, y):
                n = -x.shape[0] if flag else +y.shape[0]
                return torch.arange(n)
        """,
        [
            (
                "arange",
                [("                return torch.arange(n)", "forward")],
            )
        ],
        {"x": "x_in", "y": "y_in"},
    )
    assert resolver.sources_of("arange") == ["x_in", "y_in"]


def test_size_name_that_reads_itself_stops(tmp_path):
    resolver = _sources(
        tmp_path,
        """
        class Model:
            def forward(self, x):
                n = n + x.shape[0]
                return torch.arange(n)
        """,
        [
            (
                "arange",
                [("                return torch.arange(n)", "forward")],
            )
        ],
        {"x": "x_in"},
    )
    assert resolver.sources_of("arange") == ["x_in"]


def test_nested_unpack_and_tuple_literal_and_unknown_name(tmp_path):
    resolver = _sources(
        tmp_path,
        """
        class Model:
            def forward(self, x, y):
                (p, (q, r)) = x.shape
                dropped = torch.arange(q)
                b, n = (x.shape[0], y.shape[0])
                kept = torch.arange(n)
                lost = torch.arange(missing)
                ghost_n = ghost.shape
                unread = torch.arange(ghost_n)
        """,
        [
            ("dropped", [("                dropped = torch.arange(q)", "forward")]),
            ("kept", [("                kept = torch.arange(n)", "forward")]),
            ("lost", [("                lost = torch.arange(missing)", "forward")]),
            ("unread", [("                unread = torch.arange(ghost_n)", "forward")]),
        ],
        {"x": "x_in", "y": "y_in"},
    )
    assert resolver.sources_of("dropped") == []
    assert resolver.sources_of("kept") == ["y_in"]
    assert resolver.sources_of("lost") == []
    assert resolver.sources_of("unread") == []


def test_assignment_in_another_function_is_not_the_value(tmp_path):
    resolver = _sources(
        tmp_path,
        """
        class Model:
            def other(self, x):
                x = x + 1
                return x

            def forward(self, x):
                x = x + 2
                n = x.shape[0]
                return torch.arange(n)
        """,
        [
            ("fwd_x", [("                x = x + 2", "forward")]),
            # Sits between the assignment and the read, in another function.
            ("other_x", [("                x = x + 1", "other")]),
            ("arange", [("                return torch.arange(n)", "forward")]),
        ],
        {"x": "x_in"},
    )
    assert resolver.sources_of("arange") == ["fwd_x"]


def test_tensor_passed_as_an_expression_has_no_node(tmp_path):
    resolver = _sources(
        tmp_path,
        """
        class Layer:
            def mix(self, x):
                return torch.arange(x.shape[0])

            def forward(self, W):
                return self.mix(W + 1)
        """,
        [
            (
                "arange",
                [
                    ("                return self.mix(W + 1)", "forward"),
                    ("                return torch.arange(x.shape[0])", "mix"),
                ],
            )
        ],
    )
    assert resolver.sources_of("arange") == []


def test_header_expressions_of_compound_statements_are_read(tmp_path):
    source = """
        class Model:
            def forward(self, W):
                def nested(z):
                    return z
                lam = lambda q: q
                class Inner:
                    pass
                if W.shape[0]:
                    return W
                for _ in range(W.shape[1]):
                    return W
                with ctx(W.shape[2]):
                    return W
    """
    roots = {"W": "w"}
    assert _sources(
        tmp_path,
        source,
        [("gate", [("                if W.shape[0]:", "forward")])],
        roots,
    ).sources_of("gate") == ["w"]
    assert _sources(
        tmp_path,
        source,
        [("loop", [("                for _ in range(W.shape[1]):", "forward")])],
        roots,
    ).sources_of("loop") == ["w"]
    assert _sources(
        tmp_path,
        source,
        [("scope", [("                with ctx(W.shape[2]):", "forward")])],
        roots,
    ).sources_of("scope") == ["w"]


def test_caller_edges_that_do_not_resolve(tmp_path):
    table = """
        class Model:
            def forward(self, W, mask):
                return table(W.shape[0]) + table(mask.shape[0])

        def table(n):
            return torch.arange(n)
    """
    ambiguous = _sources(
        tmp_path,
        table,
        [
            (
                "arange",
                [
                    (
                        "                return table(W.shape[0]) + table(mask.shape[0])",
                        "forward",
                    ),
                    ("            return torch.arange(n)", "table"),
                ],
            )
        ],
        {"W": "w", "mask": "mask"},
    )
    assert ambiguous.sources_of("arange") == []

    keyword = _sources(
        tmp_path,
        """
        class Model:
            def forward(self, W):
                return table(n=W.shape[0])

        def table(n):
            return torch.arange(n)
        """,
        [
            (
                "arange",
                [
                    ("                return table(n=W.shape[0])", "forward"),
                    ("            return torch.arange(n)", "table"),
                ],
            )
        ],
        {"W": "w"},
    )
    assert keyword.sources_of("arange") == ["w"]

    starred = _sources(
        tmp_path,
        """
        class Model:
            def forward(self, W):
                return table(*W.shape)

        def table(n):
            return torch.arange(n)
        """,
        [
            (
                "arange",
                [
                    ("                return table(*W.shape)", "forward"),
                    ("            return torch.arange(n)", "table"),
                ],
            )
        ],
        {"W": "w"},
    )
    assert starred.sources_of("arange") == []

    indirect = _sources(
        tmp_path,
        """
        class Model:
            def forward(self, W):
                return (ops["table"])(W.shape[0])

        def table(n):
            return torch.arange(n)
        """,
        [
            (
                "arange",
                [
                    ('                return (ops["table"])(W.shape[0])', "forward"),
                    ("            return torch.arange(n)", "table"),
                ],
            )
        ],
        {"W": "w"},
    )
    assert indirect.sources_of("arange") == []


def test_missing_function_or_statement_resolves_to_nothing(tmp_path):
    path = tmp_path / "model.py"
    text = textwrap.dedent("""
        class Model:
            def forward(self, W):
                return table(W.shape[0])

        def table(n):
            return torch.arange(n)
    """).strip() + "\n"
    path.write_text(text)
    lines = text.splitlines()

    def at(snippet, name):
        needle = snippet.strip()
        matches = [i for i, line in enumerate(lines) if line.strip() == needle]
        if len(matches) != 1:
            raise AssertionError(f"{needle!r} matched lines {matches}")
        return Frame(str(path), matches[0] + 1, name)

    def_line = at("    def forward(self, W):", "forward")
    table_line = at("        return torch.arange(n)", "table")
    missing = Frame(str(path), def_line.line, "no_such_fn")

    no_statement = SizeReadResolver(
        [TracedNode("arange", (def_line,))],
        {"W": "w"},
    )
    assert no_statement.sources_of("arange") == []

    unknown = SizeReadResolver([TracedNode("arange", (missing,))], {})
    assert unknown.sources_of("arange") == []

    no_caller = SizeReadResolver(
        [TracedNode("arange", (missing, table_line))],
        {},
    )
    assert no_caller.sources_of("arange") == []

    header = SizeReadResolver(
        [TracedNode("arange", (def_line, table_line))],
        {"W": "w"},
    )
    assert header.sources_of("arange") == []

    # Guards for a frame that never parses as a function, reached only when a
    # size name was already in hand.
    assert unknown._value_of("W", 0, unknown._nodes[0], 0) is None
    assert unknown._resolve_name("W", (missing,), 0, missing.line, set()) == set()
