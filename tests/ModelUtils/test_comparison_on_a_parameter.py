###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""A comparison against a tensor parameter is a tensor op.

MiniMax's ``build_block_mask`` composes the block selection with the padding
mask::

    padding_mask = attention_mask if attention_mask.dtype == torch.bool else attention_mask == 0
    keep = block_keep & padding_mask

Neither the comparison nor ``attention_mask`` appeared anywhere in that frame,
so the ``&`` was drawn with ONE operand and the mask it composes with reached no
op at all.

The rule that dropped it means to skip index bookkeeping (``seq_len == 1``), and
tested for it by "neither side resolved to a producer". That catches more than
it means to: a tensor PARAMETER has no internal producer either -- that is what
a boundary is for. Asking what the operands ARE keeps both readings: host
scalars still emit nothing, a parameter emits a real op and is recorded so it
gets a boundary.
"""

from __future__ import annotations

import textwrap

from TraceLens.ModelUtils.ast_analyze import analyze_source


def _method_ops(body: str) -> list[tuple[str, str, tuple[str, ...], tuple[str, ...]]]:
    source = f"""
import torch


class M(torch.nn.Module):
    def build(self, block_indices, attention_mask, position_ids):
{textwrap.indent(textwrap.dedent(body), " " * 8)}

    def forward(self, block_indices, attention_mask, position_ids):
        return self.build(block_indices, attention_mask, position_ids)
"""
    analysis = analyze_source(source, config={"hidden_size": 8})
    return [
        (
            op.attr_name,
            str(op.label),
            tuple(op.predecessors),
            tuple(op.param_inputs or ()),
        )
        for op in analysis.class_registry["M"].multi_op_methods.get("build", [])
    ]


class TestAComparisonAgainstAParameter:
    def test_it_is_emitted(self) -> None:
        ops = _method_ops("""
            block_keep = block_indices > 0
            padding_mask = attention_mask == 0
            return block_keep & padding_mask
            """)
        equals = [op for op in ops if op[1] == "Equal"]
        assert equals, [op[1] for op in ops]

    def test_it_records_the_parameter_it_reads(self) -> None:
        """So the frame gives ``attention_mask`` a boundary to arrive on."""
        ops = _method_ops("""
            block_keep = block_indices > 0
            padding_mask = attention_mask == 0
            return block_keep & padding_mask
            """)
        equals = [op for op in ops if op[1] == "Equal"]
        assert equals[0][3] == ("attention_mask",), equals[0]

    def test_the_consumer_gets_both_operands(self) -> None:
        """``block_keep & padding_mask`` reads two things, not one."""
        ops = _method_ops("""
            block_keep = block_indices > 0
            padding_mask = attention_mask == 0
            return block_keep & padding_mask
            """)
        combine = [op for op in ops if op[1] == "Bitwise and"]
        assert combine, [op[1] for op in ops]
        assert len(combine[0][2]) == 2, combine[0]

    def test_the_ternary_spelling_reads_the_same_way(self) -> None:
        """The model writes it behind a dtype test; both arms are that mask."""
        ops = _method_ops("""
            block_keep = block_indices > 0
            padding_mask = (
                attention_mask
                if attention_mask.dtype == torch.bool
                else attention_mask == 0
            )
            return block_keep & padding_mask
            """)
        assert any(op[1] == "Equal" for op in ops), [op[1] for op in ops]
        combine = [op for op in ops if op[1] == "Bitwise and"]
        assert len(combine[0][2]) == 2, combine[0]


class TestHostBookkeepingStillEmitsNothing:
    def test_two_host_scalars_are_not_a_tensor_op(self) -> None:
        """``seq_len == 1`` is a branch predicate, not a comparison to draw."""
        ops = _method_ops("""
            batch, seq_len = block_indices.shape[:2]
            if seq_len == 1:
                return block_indices
            return block_indices + batch
            """)
        assert not any(op[1] == "Equal" for op in ops), [op[1] for op in ops]
