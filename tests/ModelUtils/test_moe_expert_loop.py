###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""An expert loop computes on its OWN expert's tokens.

Every MoE model writes the same loop::

    mask = F.one_hot(top_k_index, num_classes=num_experts).permute(2, 1, 0)
    hit  = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
    for expert_idx in hit:
        expert_idx = expert_idx[0]
        top_k_pos, token_idx = torch.where(mask[expert_idx])
        current = self._apply_gate(F.linear(hidden_states[token_idx],
                                            self.gate_up_proj[expert_idx]))
        current = F.linear(current, self.down_proj[expert_idx]) * ...
        final.index_add_(0, token_idx, current.to(final.dtype))

and the whole body came out wrong, through four separate defects stacked on
each other. GLM's expert MLP read ``[3]``, ``[4]``, ``[8]`` and
``[288, 8, B*S]``; it now reads ``[nnz, 4096]`` -> ``[nnz, 2048]`` ->
``[nnz, 4096]``, scattered back into ``[B*S, 4096]``.

The four, each uncovered by fixing the one above it, are tested here at the
level each lives at.
"""

from __future__ import annotations

import ast
import textwrap

import TraceLens.ModelUtils.ast_analyze as aa
from TraceLens.ModelUtils.shape_inference import TensorSpec

from tests.ModelUtils.test_index_axes_placement import _make_inferencer, _node

_LOOP = textwrap.dedent("""
    def forward(self, hidden_states, top_k_index):
        final = torch.zeros_like(hidden_states)
        mask = F.one_hot(top_k_index, num_classes=288).permute(2, 1, 0)
        hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in hit:
            expert_idx = expert_idx[0]
            top_k_pos, token_idx = torch.where(mask[expert_idx])
            current = F.linear(hidden_states[token_idx], self.gate_up_proj[expert_idx])
            final.index_add_(0, token_idx, current)
        return final
""")


def _operations(source: str):
    func = ast.parse(source).body[0]
    analysis = aa._forward_operations_from_forward(
        func, self_values={}, all_tensor_ops=True
    )
    return {op.attr_name: op for op in analysis.operations}


class TestTheLoopIterableIsNotAnOperand:
    def test_the_row_select_carries_the_loop_not_the_whole_iterable(self) -> None:
        """``mask[expert_idx]`` indexes the TABLE, not the loop's own nonzero."""
        ops = _operations(_LOOP)
        indexing = next(
            op for name, op in ops.items() if "l8" in name and op.label == "Indexing"
        )
        producers = {ops[p].label for p in indexing.predecessors if p in ops}
        assert "Permute" in producers, indexing.predecessors
        assert "Nonzero" not in producers, indexing.predecessors

    def test_a_body_that_never_names_the_loop_variable_still_depends_on_it(
        self,
    ) -> None:
        ops = _operations(textwrap.dedent("""
                def forward(self, hidden_states, rows):
                    out = torch.zeros_like(hidden_states)
                    picked = rows.nonzero()
                    for item in picked:
                        out = out + hidden_states
                    return out
            """))
        add = next(op for op in ops.values() if op.label == "Add")
        assert add.predecessors


class TestWhereWithOneArgument:
    def test_it_reports_the_nonzero_count(self) -> None:
        """``torch.where(cond)`` is ``nonzero(as_tuple=True)``, not a select."""
        inf = _make_inferencer()
        cond = TensorSpec(shape=(1, 8, "B*S"), dtype="int64")
        result = inf._infer_node_output(_node("Where"), [cond], root=None)
        assert result.shape == ("nnz",)
        assert result.dtype == "int64"

    def test_the_three_argument_select_is_unchanged(self) -> None:
        inf = _make_inferencer()
        cond = TensorSpec(shape=("B", "S", 8), dtype="bool")
        x = TensorSpec(shape=("B", "S", 8), dtype="int64")
        y = TensorSpec(shape=("B", "S", 8), dtype="int64")
        result = inf._infer_node_output(_node("Where"), [cond, x, y], root=None)
        assert result.shape == ("B", "S", 8)
        assert result.dtype == "int64"
