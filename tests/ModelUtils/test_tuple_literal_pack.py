###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Every tensor packed into one tuple/list literal keeps a real consumer edge.

SGLang's attention threads its split outputs through a plain tuple before
handing them to the next method::

    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    inner_state = (q, k, v, forward_batch)
    return None, forward_batch, inner_state

A tuple/list literal used to resolve to only its LAST element with a real
producer. ``inner_state``'s own producer came out as ``v`` -- so the graph
drew ``split`` with a single consumer (whichever ordinal ``v`` happened to be),
and the split's OTHER slices (``q``, ``k``) reported dead: nothing consumes
them. Each distinct tensor packed into the tuple must be a real predecessor of
whatever reads the tuple as a whole.
"""

from __future__ import annotations

import ast
import textwrap

from TraceLens.ModelUtils.ast_analyze import (
    PACK_CLASS,
    PACK_LABEL,
    _forward_operations_from_forward,
)


def _forward(source: str) -> ast.FunctionDef:
    tree = ast.parse(textwrap.dedent(source))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    return next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward"
    )


_PACKED_SPLIT_OUTPUTS = """
    class Attention:
        def forward(self, hidden_states, forward_batch):
            qkv = self.qkv_proj(hidden_states)
            q, k, v = qkv.split([8, 8, 8], dim=-1)
            inner_state = (q, k, v, forward_batch)
            return inner_state
"""

_SINGLE_PRODUCER_TUPLE = """
    class Attention:
        def forward(self, hidden_states, forward_batch):
            hidden_states = self.proj(hidden_states)
            packed = (hidden_states, forward_batch)
            return packed
"""


class TestEveryPackedSlotStaysAPredecessor:
    def test_a_pack_node_groups_every_distinct_slot(self) -> None:
        analysis = _forward_operations_from_forward(
            _forward(_PACKED_SPLIT_OUTPUTS), self_values={}, all_tensor_ops=True
        )
        pack = next(op for op in analysis.operations if op.class_name == PACK_CLASS)
        assert pack.label == PACK_LABEL
        split = next(op for op in analysis.operations if op.class_name == "Split")
        # q, k, v each dock on their own ordinal of the same split producer.
        assert pack.predecessor_ports == (
            (split.attr_name, 0),
            (split.attr_name, 1),
            (split.attr_name, 2),
        )
        assert pack.predecessors.count(split.attr_name) == 3

    def test_the_bound_name_resolves_to_the_pack_not_the_last_slice(self) -> None:
        analysis = _forward_operations_from_forward(
            _forward(_PACKED_SPLIT_OUTPUTS), self_values={}, all_tensor_ops=True
        )
        pack = next(op for op in analysis.operations if op.class_name == PACK_CLASS)
        assert analysis.return_slots == {"inner_state": pack.attr_name}

    def test_forward_batch_contributes_no_predecessor(self) -> None:
        """A bare forward parameter with no producer of its own is just dropped."""
        analysis = _forward_operations_from_forward(
            _forward(_PACKED_SPLIT_OUTPUTS), self_values={}, all_tensor_ops=True
        )
        pack = next(op for op in analysis.operations if op.class_name == PACK_CLASS)
        assert "forward_batch" not in pack.predecessors


class TestATupleWithOnlyOneRealProducerNeedsNoPackNode:
    def test_it_collapses_to_that_one_producer_as_before(self) -> None:
        analysis = _forward_operations_from_forward(
            _forward(_SINGLE_PRODUCER_TUPLE), self_values={}, all_tensor_ops=True
        )
        assert not any(op.class_name == PACK_CLASS for op in analysis.operations)
        # ``self.proj(...)`` is its own call step, not an inline op, so the
        # bound name resolves straight to that step -- no Pack needed for a
        # tuple with only one tensor and one bare passthrough parameter.
        assert analysis.return_slots == {"packed": "proj"}
