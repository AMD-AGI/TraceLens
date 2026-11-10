###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""A value the source throws away is not left as a dead node.

``kept, _ = self.method(...)`` discards the method's last return. A later
``_, other = self.method(...)`` assigns ``_`` again, and the two assignments
would otherwise be joined by a merge nothing reads. A check written as a bare
call consumes the comparison it is given; that comparison is the check.
"""

from __future__ import annotations

import textwrap

from TraceLens.ModelUtils.ast_analyze import PHI_CLASS, analyze_source
from TraceLens.ModelUtils.basic_ops import BasicOpFilter
from TraceLens.ModelUtils.block_tree import build_block_node
from TraceLens.Visualizer.model_explorer_export.merge import (
    _publish_state_stores_through_output_tiles,
)
from TraceLens.Visualizer.model_explorer_export.type_check import (
    integrity_check_graph_nodes,
)

_SOURCE = """
import torch.nn as nn


def check(condition, message):
    if not condition:
        raise ValueError(message)


class Model(nn.Module):
    def get_mask(self, input_ids, features=None):
        mask = input_ids == 1
        n_tokens = mask.sum()
        mask = mask.unsqueeze(-1)
        if features is not None:
            check(n_tokens == features.sum(), "mismatch")
        other = input_ids == 2
        other = other.unsqueeze(-1)
        return mask, other

    def forward(self, input_ids, features=None, alternate=None):
        result = input_ids
        if features is not None:
            kept, _ = self.get_mask(input_ids, features)
            result = result + kept
        if alternate is not None:
            _, other = self.get_mask(input_ids, alternate)
            result = result + other
        return result
"""


def _analysis():
    return analyze_source(textwrap.dedent(_SOURCE), all_tensor_ops=True)


def _frames(root):
    found = []
    pending = list(root.children)
    while pending:
        child = pending.pop()
        if child.class_name == "get_mask" or child.attr_name == "get_mask":
            found.append(child)
        pending.extend(child.children)
    return found


class TestADiscardedReturnIsNotDrawn:
    def test_the_unused_suffix_op_is_absent_from_that_call(self) -> None:
        analysis = _analysis()
        root = build_block_node(
            attr_name="model",
            class_name="Model",
            registry=analysis.class_registry,
            basic_ops=BasicOpFilter([]),
        )
        frames = {frame.attr_name: frame for frame in _frames(root)}
        names = analysis.class_registry["Model"].forward_step_output_names
        suffix = frames[next(attr for attr, slots in names.items() if slots[-1] == "_")]
        prefix = frames[next(attr for attr, slots in names.items() if slots[0] == "_")]
        slots = analysis.class_registry["Model"].multi_op_method_returns["get_mask"][0]
        suffix_unsqueezes = [
            child.attr_name for child in suffix.children if child.label == "Unsqueeze"
        ]
        prefix_unsqueezes = [
            child.attr_name for child in prefix.children if child.label == "Unsqueeze"
        ]
        # ``kept, _`` keeps the first return. ``_, other`` keeps the second.
        # The discarded slot stays in the return order, so the caller's ordinal
        # does not move; only the op nobody reads is left out.
        assert suffix_unsqueezes == [slots["mask"]]
        assert prefix_unsqueezes == [slots["other"]]

    def test_rebinding_the_discarded_name_does_not_leave_a_merge(self) -> None:
        analysis = _analysis()
        cls = analysis.class_registry["Model"]
        operations = cls.forward_operations
        merges = [op for op in operations.values() if op.class_name == PHI_CLASS]
        referenced: set[str] = set()
        for operation in operations.values():
            referenced.update(operation.predecessors)
        returned = set(cls.forward_return_slots.values())
        assert merges
        assert all(
            merge.attr_name in referenced or merge.attr_name in returned
            for merge in merges
        )

    def test_the_check_comparison_is_marked_as_its_consumer(self) -> None:
        analysis = _analysis()
        operations = analysis.class_registry["Model"].multi_op_methods["get_mask"]
        marked = [
            op
            for op in operations
            if op.label == "Equal"
            and any(
                str(detail).startswith("side_effect: check") for detail in op.details
            )
        ]
        assert marked


class TestADiscardedCallIsASink:
    def test_the_marked_value_is_consumed(self) -> None:
        nodes = [
            {
                "id": "eq",
                "label": "Equal",
                "namespace": "mask",
                "attrs": [{"key": "details", "value": "side_effect: check"}],
                "incomingEdges": [
                    {
                        "sourceNodeId": "@input",
                        "sourceNodeOutputId": "0",
                        "targetNodeInputId": "0",
                    }
                ],
            }
        ]
        _publish_state_stores_through_output_tiles(nodes)
        assert any(node["id"] == "eq:@side_effect:check" for node in nodes)
        warnings = [
            warning
            for warning in integrity_check_graph_nodes(nodes)
            if warning.startswith("I1")
        ]
        assert warnings == []
