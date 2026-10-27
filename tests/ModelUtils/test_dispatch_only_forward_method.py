###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""A sibling forward method pruned to pure submodule dispatch still wires up.

MiniMax's ``MiniMaxM3Attention.forward_core`` calls nothing but
``self.attn(q, k, v, forward_batch)`` then ``self.o_proj(attn_output)`` once its
sparse-attention arm resolves dead -- zero inline tensor operations of its own,
just two submodule dispatches reading a tuple the caller unpacked off this
method's own (sole) parameter. A method with no tensor op at all used to earn
no ``multi_op_methods`` entry and be skipped outright, leaving its submodule
calls un-expanded and their multi-argument reads of the same boundary token
indistinguishable from one another.
"""

from __future__ import annotations

import textwrap

from TraceLens.ModelUtils.ast_analyze import analyze_source
from TraceLens.ModelUtils.basic_ops import BasicOpFilter
from TraceLens.ModelUtils.block_tree import build_block_node

_DISPATCH_ONLY = """
import torch


class Attn(torch.nn.Module):
    def forward(self, q, k, v):
        return q


class Proj(torch.nn.Module):
    def forward(self, x):
        return x


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = Attn()
        self.o_proj = Proj()

    def forward_prepare(self, hidden_states):
        q = hidden_states
        k = hidden_states
        v = hidden_states
        inner_state = (q, k, v)
        return inner_state

    def forward_core(self, inner_state):
        q, k, v = inner_state
        attn_output = self.attn(q, k, v)
        output = self.o_proj(attn_output)
        return output

    def forward(self, hidden_states):
        inner_state = self.forward_prepare(hidden_states)
        output = self.forward_core(inner_state)
        return output
"""


def _model(source: str = _DISPATCH_ONLY, name: str = "Model"):
    return analyze_source(textwrap.dedent(source)).class_registry[name]


def _frame(node, attr_name: str):
    pending = list(getattr(node, "children", []) or [])
    while pending:
        child = pending.pop()
        if child.attr_name == attr_name or child.class_name == attr_name:
            return child
        pending.extend(getattr(child, "children", []) or [])
    return None


class TestAZeroOpMethodStillRecordsItsStepWiring:
    def test_it_earns_no_multi_op_entry(self) -> None:
        """No inline tensor op: there is nothing to list as the method's body."""
        cls = _model()
        assert "forward_core" not in cls.multi_op_methods

    def test_its_submodule_steps_are_recorded_anyway(self) -> None:
        cls = _model()
        steps = cls.multi_op_method_step_predecessors.get("forward_core", {})
        assert any("attn" in step for step in steps)
        assert any("o_proj" in step for step in steps)

    def test_each_argument_sharing_the_boundary_token_keeps_its_own_slot(self) -> None:
        cls = _model()
        ordinal_maps = cls.multi_op_method_step_predecessor_ordinals.get(
            "forward_core", {}
        )
        attn_step = next(step for step in ordinal_maps if "attn" in step)
        ordinals = ordinal_maps[attn_step]
        assert ordinals["q"] == 0
        assert ordinals["k"] == 1
        assert ordinals["v"] == 2


class TestABareUnpackOfTheBoundaryInputIsOrdinalTagged:
    def test_q_k_v_each_get_their_own_ordinal(self) -> None:
        """``q, k, v = inner_state`` where ``inner_state`` IS the parameter."""
        cls = _model()
        ordinals = cls.multi_op_method_step_predecessor_ordinals.get("forward_core", {})
        all_ordinals = {
            name: ordinal
            for step_map in ordinals.values()
            for name, ordinal in step_map.items()
        }
        assert all_ordinals == {"q": 0, "k": 1, "v": 2}


class TestTheDispatchOnlyMethodExpandsWithRealChildren:
    def test_the_frame_has_submodule_children_not_an_opaque_leaf(self) -> None:
        analysis = analyze_source(textwrap.dedent(_DISPATCH_ONLY), all_tensor_ops=True)
        root = build_block_node(
            attr_name="model",
            class_name="Model",
            registry=analysis.class_registry,
            basic_ops=BasicOpFilter([]),
        )
        frame = _frame(root, "forward_core")
        assert frame is not None
        assert frame.children, "a dispatch-only frame should still expand"
        child_names = {child.attr_name for child in frame.children} | {
            child.class_name for child in frame.children
        }
        assert "attn" in child_names or "Attn" in child_names
        assert "o_proj" in child_names or "Proj" in child_names

    def test_the_frame_carries_per_argument_ordinals(self) -> None:
        analysis = analyze_source(textwrap.dedent(_DISPATCH_ONLY), all_tensor_ops=True)
        root = build_block_node(
            attr_name="model",
            class_name="Model",
            registry=analysis.class_registry,
            basic_ops=BasicOpFilter([]),
        )
        frame = _frame(root, "forward_core")
        assert frame is not None
        ordinal_maps = frame.forward_step_predecessor_ordinals
        attn_step = next(step for step in ordinal_maps if "attn" in step)
        assert ordinal_maps[attn_step] == {"q": 0, "k": 1, "v": 2}
