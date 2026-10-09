###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Two ops reading one callee parameter both read it.

``apply_rotary_pos_emb`` splits each of its two tensors the same way::

    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

The ``q`` line was right and the ``k`` line was wrong, for a reason that has
nothing to do with q or k: ``q`` is the callee's PRIMARY parameter and resolves
through ``FORWARD_METHOD_INPUT``, while ``k`` goes through the frame's
parameter-translation loop -- which dropped every reader after the first as a
"rebound re-read". That claim holds for a parameter the body reassigns
(``cos = cos.repeat_interleave(...)``), where later reads really do come through
the rebinding. It does not hold for a parameter that is never reassigned: the
second reader has no internal predecessor at all, so dropping its parameter left
it with no operand, and the linear-pipeline fallback chained it onto the PREVIOUS
op -- the first slice. ``k_pass`` then came out ``rotary_dim - rotary_dim`` wide,
i.e. empty, and ``k_embed`` was a part-head instead of a full one.

The discriminator is whether the op has an internal predecessor to carry the
value, not whether some earlier op already claimed the slot.
"""

from __future__ import annotations

import textwrap

from TraceLens.ModelUtils.ast_analyze import analyze_source
from TraceLens.ModelUtils.basic_ops import (
    DEFAULT_BASIC_OP_PATTERNS,
    BasicOpFilter,
)
from TraceLens.ModelUtils.block_tree import build_block_node

_SOURCE = textwrap.dedent("""
    def apply_rotary_pos_emb(q, k, cos, sin):
        rotary_dim = cos.shape[-1]
        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
        q_embed = torch.cat([q_rot, q_pass], dim=-1)
        k_embed = torch.cat([k_rot, k_pass], dim=-1)
        return q_embed, k_embed

    class M(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.q_proj = nn.Linear(8, 8)
            self.k_proj = nn.Linear(8, 8)

        def forward(self, hidden_states, position_embeddings):
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            cos, sin = position_embeddings
            q2, k2 = apply_rotary_pos_emb(q, k, cos, sin)
            return q2 + k2
    """)


def _rotary_slice_nodes() -> dict[str, object]:
    analysis = analyze_source(_SOURCE, config={"hidden_size": 8})
    root = build_block_node(
        attr_name="model",
        class_name="M",
        registry=analysis.class_registry,
        basic_ops=BasicOpFilter(DEFAULT_BASIC_OP_PATTERNS),
    )
    found: dict[str, object] = {}

    def walk(node) -> None:
        attr = str(getattr(node, "attr_name", "") or "")
        if "_slice" in attr:
            found[attr] = node
        for child in getattr(node, "children", ()) or ():
            walk(child)

    walk(root)
    return found


class TestSecondReaderKeepsItsParameter:
    def test_both_readers_of_the_non_primary_parameter_keep_it(self) -> None:
        nodes = _rotary_slice_nodes()
        k_slices = [node for attr, node in nodes.items() if "l5_c" in attr]
        assert len(k_slices) == 2, sorted(nodes)
        for node in k_slices:
            assert node.param_inputs or node.operation_predecessors, (
                node.attr_name,
                "reads neither its parameter nor any op -- it will be chained "
                "onto whatever op happens to precede it",
            )

    def test_the_second_reader_is_not_chained_onto_the_first(self) -> None:
        nodes = _rotary_slice_nodes()
        first = next(node for attr, node in nodes.items() if "l5_c20" in attr)
        second = next(node for attr, node in nodes.items() if "l5_c41" in attr)
        assert first.attr_name not in (second.operation_predecessors or []), (
            second.attr_name,
            second.operation_predecessors,
        )

    def test_the_primary_parameter_line_is_unchanged(self) -> None:
        """``q``'s pair was always right; it must stay right."""
        nodes = _rotary_slice_nodes()
        q_slices = [node for attr, node in nodes.items() if "l4_c" in attr]
        assert len(q_slices) == 2, sorted(nodes)
        q_first = next(attr for attr in nodes if "l4_c20" in attr)
        q_second = next(node for attr, node in nodes.items() if "l4_c41" in attr)
        assert q_first not in (q_second.operation_predecessors or [])
