###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""An outer section's bare parameters are graph inputs, not orphan boundaries."""

from __future__ import annotations

from TraceLens.Visualizer.model_explorer_export.merge import (
    _coalesce_duplicate_section_inputs,
    _drop_mismatched_boundary_operands,
    _expose_unfed_section_inputs,
)


def _input(node_id: str, label: str, namespace: str):
    return {
        "id": node_id,
        "label": label,
        "namespace": namespace,
        "attrs": [
            {"key": "synthetic", "value": "@input"},
            {"key": "boundary_input", "value": label},
            {"key": "output_shape", "value": "[R, C] int64"},
        ],
    }


class TestOneBoundaryPerSectionParameter:
    def test_duplicate_requests_are_coalesced(self) -> None:
        first = _input("section/@input:grid", "grid", "section")
        second = _input("section/@input:grid", "grid", "section")
        first["incomingEdges"] = [
            {
                "sourceNodeId": "a",
                "sourceNodeOutputId": "0",
                "targetNodeInputId": "grid",
            }
        ]
        second["incomingEdges"] = [
            {
                "sourceNodeId": "b",
                "sourceNodeOutputId": "0",
                "targetNodeInputId": "grid",
            }
        ]
        nodes = [first, second]
        _coalesce_duplicate_section_inputs(nodes)
        assert len(nodes) == 1
        assert {edge["sourceNodeId"] for edge in nodes[0]["incomingEdges"]} == {
            "a",
            "b",
        }


class TestAnOuterSectionParameterIsAModelInput:
    def test_an_unfed_boundary_gets_a_root_source(self) -> None:
        target = _input("section/@input:grid", "grid", "section")
        nodes = [target]
        _expose_unfed_section_inputs(nodes)
        source = next(node for node in nodes if node["id"] == "@input:grid")
        assert source["namespace"] == ""
        assert any(
            attr == {"key": "output_shape", "value": "[R, C] int64"}
            for attr in source["attrs"]
        )
        assert target["incomingEdges"][0]["sourceNodeId"] == "@input:grid"

    def test_an_unverified_external_parameter_does_not_inherit_entry_shape(
        self,
    ) -> None:
        class NoObservedParameterShape:
            @staticmethod
            def boundary_input_spec(_param: str, _namespace: str):
                return None

        target = _input("section/@input:grid", "grid", "section")
        target["outputsMetadata"] = [
            {
                "id": "0",
                "attrs": [
                    {"key": "shape", "value": "[B, S, H] bfloat16"},
                    {"key": "dtype", "value": "bfloat16"},
                ],
            }
        ]
        nodes = [target]
        _expose_unfed_section_inputs(nodes, NoObservedParameterShape())
        source = next(node for node in nodes if node["id"] == "@input:grid")
        assert not any(
            attr["key"] == "output_shape" for attr in source.get("attrs", [])
        )
        assert not any(
            attr["key"] == "output_shape" for attr in target.get("attrs", [])
        )
        assert not any(
            attr["key"] == "shape"
            for metadata in target.get("outputsMetadata", [])
            for attr in metadata.get("attrs", [])
        )

    def test_an_unknown_external_shape_is_cleared_through_nested_boundaries(
        self,
    ) -> None:
        class NoObservedParameterShape:
            @staticmethod
            def boundary_input_spec(_param: str, _namespace: str):
                return None

        outer = _input("section/@input:grid", "grid", "section")
        inner = _input("section/helper/@input:grid", "grid", "section/helper")
        inner["incomingEdges"] = [
            {
                "sourceNodeId": "section/@input:grid",
                "sourceNodeOutputId": "0",
                "targetNodeInputId": "grid",
            }
        ]
        nodes = [outer, inner]
        _expose_unfed_section_inputs(nodes, NoObservedParameterShape())
        assert not any(attr["key"] == "output_shape" for attr in inner.get("attrs", []))

    def test_an_internal_module_boundary_stays_a_diagnostic(self) -> None:
        target = _input("section/helper/@input:grid", "grid", "section/helper")
        nodes = [target]
        _expose_unfed_section_inputs(nodes)
        assert nodes == [target]
        assert not target.get("incomingEdges")

    def test_an_already_fed_boundary_is_left_alone(self) -> None:
        target = _input("section/@input:grid", "grid", "section")
        target["incomingEdges"] = [
            {
                "sourceNodeId": "producer",
                "sourceNodeOutputId": "0",
                "targetNodeInputId": "grid",
            }
        ]
        nodes = [target]
        _expose_unfed_section_inputs(nodes)
        assert nodes == [target]
        assert target["incomingEdges"][0]["sourceNodeId"] == "producer"


class TestAnOpReadsItsDeclaredBoundaryParameter:
    def test_an_overfull_binary_op_drops_the_sibling_parameter(self) -> None:
        left = {"id": "left", "label": "Left", "attrs": []}
        wanted = _input("frame/@input:wanted", "wanted", "frame")
        sibling = _input("frame/@input:sibling", "sibling", "frame")
        target = {
            "id": "frame/equal",
            "label": "Equal",
            "attrs": [
                {"key": "raw_op", "value": "eq"},
                {"key": "boundary_input", "value": "wanted"},
            ],
            "incomingEdges": [
                {"sourceNodeId": "left", "targetNodeInputId": "0"},
                {"sourceNodeId": "frame/@input:wanted", "targetNodeInputId": "1"},
                {"sourceNodeId": "frame/@input:sibling", "targetNodeInputId": "2"},
            ],
        }
        _drop_mismatched_boundary_operands([left, wanted, sibling, target])
        assert {edge["sourceNodeId"] for edge in target["incomingEdges"]} == {
            "left",
            "frame/@input:wanted",
        }
