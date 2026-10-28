###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""An op is wired the operands it declares, not every argument the call carries.

An ``@op_l`` id in this file is a sample of the record format this test feeds
the wirer. It is not a lookup into an upstream modeling file.


A dispatched attention implementation is expanded into the ops it performs, and
each of those ops records which of the implementation's parameters it reads::

    @fn_l1052_repeat_kv::@op_l1038_c20_unsqueeze   preds=['@method_input:key']
    @fn_l1053_repeat_kv::@op_l1038_c20_unsqueeze   preds=['@method_input:value']
    @op_l1055_c19_matmul                           preds=['@method_input', ...]
                                                   boundary_input_name='query'

Nothing read that. The entry-point map was built from ``param_inputs`` alone,
which these ops do not carry, so every parameter of the call fell back to the
section's FIRST op -- and ``repeat_kv``'s ``unsqueeze``, a unary op, was drawn
reading three tensors on GLM and four on MiniMax, the fourth being the indexer's
int64 block indices. Then the sequential fallback chained each op to whatever
preceded it in source order on top, so ``repeat_kv(value)`` also read the
softmax output.

The resulting shapes were wrong too, and visibly so: GLM's attention reported
``[B, S, 64, S]`` where the model produces ``[B, S, 64, 256]``. Three separate
things had to agree before that showed up, which is why it is tested here at
the level of the three judgements rather than through a built graph.
"""

from __future__ import annotations

from TraceLens.ModelUtils.block_tree import BlockNode
from TraceLens.ModelUtils.computation_graph import (
    ComputationGraph,
    GraphNodeSpec,
    _boundary_parameters,
    _drop_sibling_argument_edges,
    _reads_only_named_boundary_parameters,
)


def _step(preds=(), boundary=None, attr="@op_x"):
    return BlockNode(
        attr_name=attr,
        class_name="Unsqueeze",
        role="other",
        label="Unsqueeze",
        operation_predecessors=list(preds),
        boundary_input_name=boundary,
    )


class TestWhichParametersAnOpDeclares:
    def test_a_named_boundary_token(self) -> None:
        assert _boundary_parameters(_step(["@method_input:key"])) == ["key"]

    def test_several_named_tokens(self) -> None:
        step = _step(["@method_input:cos", "@method_input:sin"])
        assert _boundary_parameters(step) == ["cos", "sin"]

    def test_the_bare_token_names_its_parameter_outright(self) -> None:
        """``torch.matmul(query, ...)`` -- the primary, said by the step."""
        step = _step(["@method_input", "@op_transpose"], boundary="query")
        assert _boundary_parameters(step) == ["query"]

    def test_the_bare_token_is_declined_for_a_subtree(self) -> None:
        """Every module calls its input much the same thing.

        Indexing a whole subtree, a nested norm's ``Cast`` claimed
        ``hidden_states`` and took the DECODER activation that belongs to the
        module two levels up.
        """
        step = _step(["@method_input"], boundary="hidden_states")
        assert _boundary_parameters(step, named_only=True) == []

    def test_a_named_token_is_kept_either_way(self) -> None:
        step = _step(["@method_input:key"])
        assert _boundary_parameters(step, named_only=True) == ["key"]

    def test_an_op_reading_no_boundary(self) -> None:
        assert _boundary_parameters(_step(["@op_previous"])) == []
        assert _boundary_parameters(_step()) == []

    def test_a_mutates_target_is_not_a_parameter(self) -> None:
        """``_boundary_input_name`` also answers for a local being written."""
        assert _boundary_parameters(_step([], boundary="new_kv")) == []


class TestWhenTheSequentialFallbackMustNotFire:
    def test_an_op_naming_only_boundary_parameters(self) -> None:
        assert _reads_only_named_boundary_parameters(_step(["@method_input:value"]))

    def test_an_op_that_also_reads_a_sibling(self) -> None:
        step = _step(["@method_input:value", "@op_transpose"])
        assert not _reads_only_named_boundary_parameters(step)

    def test_the_bare_token_still_takes_the_chain(self) -> None:
        """It says only "reads the boundary"; which tensor is chain position."""
        assert not _reads_only_named_boundary_parameters(_step(["@method_input"]))

    def test_an_op_naming_nothing(self) -> None:
        assert not _reads_only_named_boundary_parameters(_step())


class TestASiblingArgumentIsNotThisOpsOperand:
    @staticmethod
    def _graph(target_block):
        graph = ComputationGraph()
        graph.nodes = [
            GraphNodeSpec(key="q_src", label="q_embed"),
            GraphNodeSpec(key="k_src", label="k_embed"),
            GraphNodeSpec(key="matmul", block=target_block, label="MatMul"),
        ]
        graph.links = [(0, 2), (1, 2)]
        return graph

    def test_the_other_arguments_producer_is_dropped(self) -> None:
        """MiniMax's rope returns ``q_embed, k_embed`` and emits ``k`` last, so
        chain position handed the QK matmul ``key`` beside its ``query``."""
        block = _step(["@method_input", "@op_transpose"], boundary="query")
        graph = self._graph(block)
        _drop_sibling_argument_edges(
            graph,
            2,
            "query",
            {"query": "q_src", "key": "k_src"},
            {"q_src": 0, "k_src": 1},
            0,
        )
        assert graph.links == [(0, 2)]

    def test_an_op_naming_two_arguments_keeps_both(self) -> None:
        """``matmul(query, key)`` names two; neither is the odd one out."""
        block = _step(["@method_input", "@method_input:key"], boundary="query")
        graph = self._graph(block)
        _drop_sibling_argument_edges(
            graph,
            2,
            "query",
            {"query": "q_src", "key": "k_src"},
            {"q_src": 0, "k_src": 1},
            0,
        )
        assert graph.links == [(0, 2), (1, 2)]

    def test_an_op_declaring_nothing_is_untouched(self) -> None:
        graph = self._graph(_step(["@op_a", "@op_b"]))
        _drop_sibling_argument_edges(
            graph,
            2,
            "query",
            {"query": "q_src", "key": "k_src"},
            {"q_src": 0, "k_src": 1},
            0,
        )
        assert graph.links == [(0, 2), (1, 2)]

    def test_an_op_declaring_a_different_parameter_is_untouched(self) -> None:
        """This pass is wiring ``key``; the ``query`` op is not its business."""
        block = _step(["@method_input"], boundary="query")
        graph = self._graph(block)
        _drop_sibling_argument_edges(
            graph,
            2,
            "key",
            {"query": "q_src", "key": "k_src"},
            {"q_src": 0, "k_src": 1},
            1,
        )
        assert graph.links == [(0, 2), (1, 2)]
