###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""A phi hands back one of its arms, so its shape has to be one of theirs.

Exactly one arm of a mutually-exclusive branch runs per invocation and the phi
forwards it untouched -- it computes nothing. Whatever shape it reports must
therefore be a shape some arm actually has; any other is a shape nobody
produces, and every op below inherits it.

Asking this of the graph is what found GLM's ``apply_mask_to_padding_states``
reporting ``[B, S, 1, 4096]`` out of a function whose contract is ``[B, S, H]``
-- the frame had one input tile for two parameters, so the mask unsqueeze had
unsqueezed the hidden state.

What this does NOT assert is that the arms all AGREE. A model may rebind the
name to a differently-shaped tensor in one branch, and Kimi does: its linear
attention replaces ``hidden_states`` with the unpadded ``[1, nnz, 7168]`` when
a mask is present and keeps ``[B, S, 7168]`` when it is not. Both are real, so
asserting agreement would report a correct graph. Where arms disagree for a
BAD reason -- DeepSeek's compressor, whose cache-path arm carries a fallback
shape because the call producing it is never traced -- the cause is the
untraced call, not the phi, and that is where it has to be fixed.
"""

from __future__ import annotations

from TraceLens.ModelUtils.ast_analyze import PHI_CLASS, PHI_LABEL
from TraceLens.Visualizer.model_explorer_export.type_check import (
    _phi_forwards_an_arm_warnings,
)


def _node(node_id, shape, *, label="op", phi=False, constant=False, sources=()):
    attrs = [{"key": "output_shape", "value": shape}]
    if phi:
        attrs.append({"key": "class_name", "value": PHI_CLASS})
    if constant:
        attrs.append({"key": "constant", "value": "true"})
    return {
        "id": node_id,
        "label": PHI_LABEL if phi else label,
        "attrs": attrs,
        "incomingEdges": [{"sourceNodeId": s} for s in sources],
    }


class TestTheShapeMustBeAnArmsShape:
    def test_a_shape_no_arm_has(self) -> None:
        nodes = [
            _node("a", "[B, S, 4096] bfloat16"),
            _node("b", "[B, S, 4096] bfloat16"),
            _node("phi", "[B, S, 1, 4096] bfloat16", phi=True, sources=["a", "b"]),
        ]
        warnings = _phi_forwards_an_arm_warnings(nodes)
        assert len(warnings) == 1
        assert "must be one of theirs" in warnings[0]

    def test_the_shape_of_one_arm(self) -> None:
        nodes = [
            _node("a", "[B, S, 4096] bfloat16"),
            _node("b", "[B, S, 4096] bfloat16"),
            _node("phi", "[B, S, 4096] bfloat16", phi=True, sources=["a", "b"]),
        ]
        assert _phi_forwards_an_arm_warnings(nodes) == []

    def test_arms_that_disagree_are_not_this_checks_business(self) -> None:
        """Kimi's unpad: one branch really does hand back another shape."""
        nodes = [
            _node("padded", "[B, S, 7168] bfloat16"),
            _node("unpadded", "[1, nnz, 7168] bfloat16"),
            _node(
                "phi",
                "[1, nnz, 7168] bfloat16",
                phi=True,
                sources=["padded", "unpadded"],
            ),
        ]
        assert _phi_forwards_an_arm_warnings(nodes) == []

    def test_a_dtype_difference_is_not_a_shape(self) -> None:
        nodes = [
            _node("a", "[B, S, 4096] float32"),
            _node("b", "[B, S, 4096] bfloat16"),
            _node("phi", "[B, S, 4096] bfloat16", phi=True, sources=["a", "b"]),
        ]
        assert _phi_forwards_an_arm_warnings(nodes) == []

    def test_the_phi_is_found_by_its_label_too(self) -> None:
        nodes = [
            _node("a", "[B, S] bfloat16"),
            _node("b", "[B, S] bfloat16"),
            {
                "id": "phi",
                "label": PHI_LABEL,
                "attrs": [{"key": "output_shape", "value": "[B] bfloat16"}],
                "incomingEdges": [{"sourceNodeId": "a"}, {"sourceNodeId": "b"}],
            },
        ]
        assert len(_phi_forwards_an_arm_warnings(nodes)) == 1


class TestWhatItSkips:
    def test_a_constant_arm_is_not_a_branch(self) -> None:
        nodes = [
            _node("w", "[4096] bfloat16", constant=True),
            _node("a", "[B, S, 4096] bfloat16"),
            _node("phi", "[B, S, 4096] bfloat16", phi=True, sources=["w", "a"]),
        ]
        assert _phi_forwards_an_arm_warnings(nodes) == []

    def test_a_boundary_tile_carrying_a_phis_output(self) -> None:
        """A tile takes its name from the tensor; it is not the op."""
        node = _node("t", "[B] bfloat16", phi=True, sources=["a"])
        node["attrs"].append({"key": "synthetic", "value": "@output"})
        assert (
            _phi_forwards_an_arm_warnings([_node("a", "[B, S] bfloat16"), node]) == []
        )

    def test_an_ordinary_op(self) -> None:
        nodes = [
            _node("a", "[B, S] bfloat16"),
            _node("add", "[B, S, 1] bfloat16", label="Add", sources=["a"]),
        ]
        assert _phi_forwards_an_arm_warnings(nodes) == []

    def test_a_phi_with_no_shape(self) -> None:
        nodes = [
            _node("a", "[B, S] bfloat16"),
            {
                "id": "phi",
                "label": PHI_LABEL,
                "attrs": [{"key": "class_name", "value": PHI_CLASS}],
                "incomingEdges": [{"sourceNodeId": "a"}],
            },
        ]
        assert _phi_forwards_an_arm_warnings(nodes) == []

    def test_a_phi_whose_arms_have_no_shape(self) -> None:
        nodes = [
            {"id": "a", "label": "op", "attrs": []},
            _node("phi", "[B, S] bfloat16", phi=True, sources=["a"]),
        ]
        assert _phi_forwards_an_arm_warnings(nodes) == []

    def test_an_empty_graph(self) -> None:
        assert _phi_forwards_an_arm_warnings([]) == []
