###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""What a line of model code says about rank, said in the graph too.

GLM's indexer ends with ``torch.cat([topk_indices, tail_indices], dim=-1)``, and
its two operands disagreed on rank. Walking back from that one concat found four
separate places where the graph kept a tensor's OLD rank after the source had
changed it:

* ``key_valid.any(-1)`` was not a tensor method at all, so it was elided and the
  ``where`` reading it took the un-reduced tensor as its condition.
* ``key_valid.long()`` is a cast with no label, so in detailed mode it produced
  nothing -- not even its own producer -- and ``argmax`` was left with no operand
  to take a shape from.
* ``first_key[:, None, None]`` inserts TWO axes; only the single-``None`` case
  resolved, so this one passed through and kept the old rank.
* ``pool_offsets.view(1, number_of_pools, self.index_kpool)`` STATES rank 3, and
  returning the source because a dimension was a name nobody could resolve
  contradicted the call outright.

And one place where the graph invented a rank the source does not have:
``current_length - q_length`` over two parameters the model annotates ``int`` is
host arithmetic, which had been docking the chain's tensor input.
"""

from __future__ import annotations

import ast
import textwrap

from TraceLens.ModelUtils.ast_analyze import (
    _host_scalar_param_names,
    _none_insert_dims,
    analyze_source,
)
from TraceLens.ModelUtils.extract import ArchitectureSpec
from TraceLens.ModelUtils.model_graph import ModelGraphNode, NodeKind, OperationKind
from TraceLens.ModelUtils.shape_inference import (
    ShapeContext,
    ShapeInferencer,
    TensorSpec,
)


def _index(source: str) -> ast.AST:
    subscript = ast.parse(source, mode="eval").body
    assert isinstance(subscript, ast.Subscript)
    return subscript.slice


def _ops(body: str) -> list[tuple[str, tuple[str, ...]]]:
    source = f"""
import torch


class M(torch.nn.Module):
    def forward(self, x, y, q_length: int, current_length: int):
{textwrap.indent(textwrap.dedent(body), " " * 8)}
"""
    analysis = analyze_source(source, config={"hidden_size": 8})
    return [
        (str(op.label), tuple(op.details or ()))
        for op in analysis.class_registry["M"].forward_operations.values()
    ]


class TestEveryInsertedAxisIsDrawn:
    def test_one_none_inserts_at_its_position(self) -> None:
        assert _none_insert_dims(_index("x[:, None]")) == (1,)

    def test_two_nones_insert_two_axes(self) -> None:
        assert _none_insert_dims(_index("x[:, None, None]")) == (1, 2)

    def test_an_ellipsis_sends_each_one_to_the_end(self) -> None:
        assert _none_insert_dims(_index("x[..., None, None]")) == (-1, -1)

    def test_a_leading_none_prepends(self) -> None:
        assert _none_insert_dims(_index("x[None]")) == (0,)

    def test_a_subscript_inserting_nothing_answers_empty(self) -> None:
        assert _none_insert_dims(_index("x[:, 1]")) == ()

    def test_the_graph_draws_both_inserts(self) -> None:
        ops = _ops("return x[:, None, None] + y")
        assert [label for label, _ in ops] == ["Unsqueeze", "Unsqueeze", "Add"], ops
        assert ops[0][1] == ("dim: 1",)
        assert ops[1][1] == ("dim: 2",)


class TestScalarAnnotatedParameters:
    def test_an_int_parameter_is_a_host_scalar(self) -> None:
        func = ast.parse(textwrap.dedent("""
                def f(valid_keys, q_length: int, current_length: int, scale: float):
                    pass
                """)).body[0]
        assert _host_scalar_param_names(func) == {
            "q_length",
            "current_length",
            "scale",
        }

    def test_a_tensor_parameter_is_not(self) -> None:
        func = ast.parse(textwrap.dedent("""
                def f(valid_keys: torch.BoolTensor, untyped):
                    pass
                """)).body[0]
        assert _host_scalar_param_names(func) == set()

    def test_arithmetic_over_them_takes_no_tensor_operand(self) -> None:
        """``current_length - q_length`` is bookkeeping, not a Subtract node."""
        labels = [label for label, _ in _ops("return current_length - q_length + x")]
        assert "Subtract" not in labels, labels


class TestAViewKeepsTheRankItStates:
    def _infer(self, details: list[str], source: TensorSpec) -> TensorSpec:
        spec = ArchitectureSpec(
            name="Test",
            model_type="test",
            hidden_size=4096,
            raw_config={"hidden_size": 4096},
        )
        inferencer = ShapeInferencer(spec, context=ShapeContext.from_spec(spec))
        node = ModelGraphNode(
            id="@op_view",
            label="view",
            kind=NodeKind.LEAF,
            operation=OperationKind.TORCH_FUNCTIONAL,
            metadata={"details": details},
        )
        return inferencer._infer_node_output(node, [source], root=None)

    def test_an_unresolvable_name_is_still_an_axis(self) -> None:
        out = self._infer(
            ["shape: 1, number_of_pools, 4"],
            TensorSpec(("number_of_pools * 4",), "int64"),
        )
        assert out.shape == (1, "number_of_pools", 4), out.shape

    def test_an_expression_is_not_a_dimension_name(self) -> None:
        """``hidden_states.shape[0]`` names no axis a reader could follow."""
        source = TensorSpec(("B", "H", "S", "S"), "float32")
        out = self._infer(["shape: hidden_states.shape[0], self.num_heads"], source)
        assert out.shape == source.shape, out.shape

    def test_a_starred_prefix_is_resolved_by_the_structured_reader(self) -> None:
        """``*x.shape[:-1]`` expands to the leading axes; the fallback is not reached."""
        source = TensorSpec(("B", "S", 8), "float32")
        out = self._infer(["shape: *x.shape[:-1], 2, 4"], source)
        assert out.shape == ("B", "S", 2, 4), out.shape
