###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""A branch on a settled parameter has one arm, and only that arm is drawn.

``get_vision_cu_seqlens(grid_thw, merge_temporal=False, ...)`` computes its
segment lengths one way or the other::

    if merge_temporal:
        seqlens = grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]
    else:
        seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0])

GLM passes no ``merge_temporal`` anywhere, so the first arm cannot run -- yet both
were drawn, joined by a ``Merge`` phi that claims a runtime choice the model never
makes. The same held for ``include_temporal`` in ``get_vision_position_ids``,
whose dead arm would return a 3-column result where GLM's is 2.

A default is NOT generally trustworthy, and the first attempt at this deleted
live computation: ``create_causal_mask`` declares ``position_ids=None`` while its
caller passes a real one, so ``if position_ids is not None`` resolved False and
``find_packed_sequence_indices`` vanished from the graph. Three refusals keep
that from recurring, and they are what these tests pin:

* a ``f(**kwargs)`` call can supply anything without naming it -- and that is
  exactly how ``create_causal_mask`` is called;
* a function with no visible caller is entered from code never parsed here;
* a parameter any visible caller passes is not settled by its default.

Drawing a dead arm is the lesser error; deleting a live one is not recoverable
by the reader.
"""

from __future__ import annotations

import ast
import textwrap

from TraceLens.ModelUtils.ast_analyze import _settled_param_defaults, analyze_source


def _module_functions(source: str) -> dict[str, ast.FunctionDef]:
    tree = ast.parse(textwrap.dedent(source))
    return {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}


class TestWhatCountsAsSettled:
    def test_a_default_no_caller_overrides(self) -> None:
        source = """
            def helper(grid, merge_temporal=False):
                return grid

            def caller(grid):
                return helper(grid)
        """
        functions = _module_functions(source)
        settled = _settled_param_defaults(functions["helper"], functions)
        assert settled == {"merge_temporal": False}

    def test_a_parameter_a_caller_passes_is_not_settled(self) -> None:
        source = """
            def helper(grid, merge_temporal=False):
                return grid

            def caller(grid):
                return helper(grid, merge_temporal=True)
        """
        functions = _module_functions(source)
        assert _settled_param_defaults(functions["helper"], functions) == {}

    def test_a_kwargs_call_settles_nothing(self) -> None:
        """``create_causal_mask(**mask_kwargs)`` -- anything may arrive."""
        source = """
            def helper(x, position_ids=None):
                return x

            def caller(kwargs):
                return helper(**kwargs)
        """
        functions = _module_functions(source)
        assert _settled_param_defaults(functions["helper"], functions) == {}

    def test_no_visible_caller_settles_nothing(self) -> None:
        """Entered from code this analysis never parsed."""
        source = """
            def helper(x, position_ids=None):
                return x
        """
        functions = _module_functions(source)
        assert _settled_param_defaults(functions["helper"], functions) == {}

    def test_a_class_method_counts_as_a_caller(self) -> None:
        """A free function is most often called from a ``forward``.

        Scanning module-level functions alone makes that caller invisible, and
        the default then looks settled when a ``forward`` is overriding it --
        the mistake that deleted ``find_packed_sequence_indices``.
        """
        source = """
            def helper(x, flag=False):
                return x

            def other(x):
                return helper(x)

            class M:
                def forward(self, x):
                    return helper(x, flag=True)
        """
        tree = ast.parse(textwrap.dedent(source))
        functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
        # The forward passes it, so nothing is settled.
        assert _settled_param_defaults(functions["helper"], functions, classes) == {}
        # Blind to class bodies, the same call looks absent.
        assert _settled_param_defaults(functions["helper"], functions) == {
            "flag": False
        }

    def test_a_computed_default_is_not_a_literal(self) -> None:
        source = """
            def helper(x, size=compute()):
                return x

            def caller(x):
                return helper(x)
        """
        functions = _module_functions(source)
        assert _settled_param_defaults(functions["helper"], functions) == {}


_LIVE_ARM = """
    def helper(grid, merge_temporal=False):
        if merge_temporal:
            seq = grid[:, 0] * grid[:, 1] * grid[:, 2]
        else:
            seq = torch.repeat_interleave(grid[:, 1] * grid[:, 2], grid[:, 0])
        return seq.cumsum(dim=0)

    class M(nn.Module):
        def forward(self, grid_thw):
            return helper(grid_thw)
"""


class TestOnlyTheLiveArmIsDrawn:
    def test_no_op_is_left_guarded_by_a_condition(self) -> None:
        analysis = analyze_source(textwrap.dedent(_LIVE_ARM), config={"hidden_size": 8})
        expanded = analysis.class_registry["M"].multi_op_methods
        conditioned = [
            (op.label, detail)
            for ops in expanded.values()
            for op in ops
            for detail in op.details or ()
            if detail.startswith("condition:")
        ]
        assert not conditioned, conditioned

    def test_the_arms_are_not_joined_by_a_phi(self) -> None:
        """A Merge says the model chooses at runtime. It does not."""
        analysis = analyze_source(textwrap.dedent(_LIVE_ARM), config={"hidden_size": 8})
        labels = [
            op.label
            for ops in analysis.class_registry["M"].multi_op_methods.values()
            for op in ops
        ]
        assert "Merge" not in labels, labels

    def test_the_arm_that_runs_is_the_one_drawn(self) -> None:
        analysis = analyze_source(textwrap.dedent(_LIVE_ARM), config={"hidden_size": 8})
        labels = [
            op.label
            for ops in analysis.class_registry["M"].multi_op_methods.values()
            for op in ops
        ]
        assert "Repeat interleave" in labels, labels
        # The dead arm multiplies all three columns; the live one multiplies two.
        assert labels.count("Multiply") == 1, labels
