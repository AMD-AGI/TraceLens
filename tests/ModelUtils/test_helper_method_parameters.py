###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""A helper method a class calls on itself declares the parameters it is handed.

A method invoked on a CHILD module carries its parameters stamped on it
(``method_params:``), and its frame declares them so each argument gets a
boundary of its own. A method the class calls on ITSELF
(``self.append_visible_tail(topk_indices, token_visible, key_valid)``) carries no
such stamp, so its frame declared none -- and the wiring pass skips a parameter
the frame does not declare.

The consequences were visible two ways round in GLM's indexer:

* every argument but the primary collapsed onto the frame's single entry, so the
  ``torch.cat([topk_indices, tail_indices], dim=-1)`` closing the method read a
  boundary carrying the WRONG tensor -- ``visible_tokens``, rank 1 -- against a
  rank-3 sibling operand;
* an op reading one of those parameters with no boundary to dock entered
  unnamed, and was handed a tile named after the producing OP rather than the
  tensor.

The ops themselves name what they read, which is the same answer the stamp
would have given.
"""

from __future__ import annotations

import textwrap

from TraceLens.ModelUtils.ast_analyze import analyze_source
from TraceLens.ModelUtils.basic_ops import BasicOpFilter
from TraceLens.ModelUtils.block_tree import _method_param_inputs, build_block_node

SOURCE = """
import torch


class Helper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(8, 8)

    def tail(self, indices, visible, valid):
        count = visible.sum(-1)
        return indices + count.unsqueeze(-1) + valid.unsqueeze(-1)

    def forward(self, hidden_states, indices, visible, valid):
        x = self.proj(hidden_states)
        return self.tail(indices, visible, valid) + x
"""


def _analysis():
    return analyze_source(textwrap.dedent(SOURCE), config={"hidden_size": 8})


def _frame(node):
    """The ``tail`` helper's own frame, wherever it sits in the tree."""
    pending = list(getattr(node, "children", []) or [])
    while pending:
        child = pending.pop()
        if child.class_name == "tail" or child.attr_name == "tail":
            return child
        pending.extend(getattr(child, "children", []) or [])
    return None


class TestMethodParamInputs:
    def test_the_ops_name_the_parameters_they_read(self) -> None:
        cls = _analysis().class_registry["Helper"]
        params = _method_param_inputs(
            cls.multi_op_methods["tail"], cls.multi_op_method_inputs.get("tail")
        )
        assert "visible" in params, params

    def test_the_primary_parameter_is_not_repeated(self) -> None:
        """It arrives as the frame's entry, not as one of the extra boundaries."""
        cls = _analysis().class_registry["Helper"]
        primary = cls.multi_op_method_inputs.get("tail")
        assert primary == "indices", primary
        assert primary not in _method_param_inputs(
            cls.multi_op_methods["tail"], primary
        )

    def test_a_method_with_no_extra_parameters_declares_none(self) -> None:
        cls = _analysis().class_registry["Helper"]
        assert _method_param_inputs([], cls.multi_op_method_inputs.get("tail")) == []


class TestHelperFrameDeclaresItsParameters:
    def test_the_frame_carries_them(self) -> None:
        analysis = _analysis()
        root = build_block_node(
            attr_name="helper",
            class_name="Helper",
            registry=analysis.class_registry,
            basic_ops=BasicOpFilter([]),
        )
        frame = _frame(root)
        assert frame is not None, "the helper method should expand into a frame"
        assert "visible" in (
            frame.forward_param_inputs or []
        ), frame.forward_param_inputs
