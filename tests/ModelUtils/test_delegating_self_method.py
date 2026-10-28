###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""A method that only forwards to another method on ``self`` is that method.

The forwarding body has no tensor operation of its own, so it used to render
as one unconnected tile and shape inference had nothing to read. The call has
to expand as the target implementation, with the caller's argument bindings.
"""

from __future__ import annotations

import textwrap

from TraceLens.ModelUtils.ast_analyze import analyze_source
from TraceLens.ModelUtils.basic_ops import BasicOpFilter
from TraceLens.ModelUtils.block_tree import build_block_node
from TraceLens.ModelUtils.computation_graph import build_computation_graph

_DELEGATING_METHODS = """
import torch


class Visual(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial_merge_size = 2
        self.proj = torch.nn.Linear(4, 4)

    def forward(self, pixel_values, grid_thw=None):
        return self.proj(pixel_values)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = Visual()

    def get_video_features(self, pixel_values_videos, video_grid_thw=None, **kwargs):
        return self.get_image_features(pixel_values_videos, video_grid_thw, **kwargs)

    def get_image_features(self, pixel_values, image_grid_thw=None, **kwargs):
        pixel_values = pixel_values.type(self.visual.dtype)
        vision_output = self.visual(pixel_values, grid_thw=image_grid_thw, **kwargs)
        split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size ** 2).tolist()
        return torch.split(vision_output, split_sizes)

    def forward(
        self,
        input_ids,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
    ):
        if pixel_values is not None:
            image_embeds = self.get_image_features(pixel_values, image_grid_thw)
        if pixel_values_videos is not None:
            video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
        return input_ids
"""

_CHAIN = """
import torch


class Visual(torch.nn.Module):
    def forward(self, pixel_values, grid_thw=None):
        return pixel_values


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = Visual()

    def get_image_features(self, pixel_values, image_grid_thw=None):
        pixel_values = pixel_values.type(self.visual.dtype)
        vision_output = self.visual(pixel_values, grid_thw=image_grid_thw)
        return torch.split(vision_output, image_grid_thw.prod(-1).tolist())

    def bridge(self, pixels, grid):
        return self.get_image_features(pixels, grid)

    def get_video_features(self, pixel_values_videos, video_grid_thw):
        return self.bridge(pixel_values_videos, video_grid_thw)

    def forward(self, pixel_values_videos, video_grid_thw):
        return self.get_video_features(pixel_values_videos, video_grid_thw)
"""

_SINGLE = """
import torch


class Model(torch.nn.Module):
    def scale(self, values, weight):
        return values * weight

    def video_scale(self, frames, gain):
        return self.scale(frames, gain)

    def forward(self, frames, gain):
        return self.video_scale(frames, gain)
"""

_SECONDARY_PRIMARY = """
import torch


class Model(torch.nn.Module):
    def helper(self, pixels, grid):
        pixels = pixels.float()
        return torch.split(pixels, grid.prod(-1).tolist())

    def forward(self, tokens, pixels, grid):
        return self.helper(pixels, grid)
"""

_NOT_A_FORWARD = """
import torch


class Model(torch.nn.Module):
    def get_image_features(self, pixel_values):
        return pixel_values.relu()

    def computed(self, frames):
        return self.get_image_features(frames + 1)

    def child_call(self, frames):
        return self.visual.get_image_features(frames)

    def forward(self, frames):
        return self.computed(frames) + self.child_call(frames)
"""


_CHAIN_WITH_ORDINAL_SLOTS = """
import torch


class Attn(torch.nn.Module):
    def forward(self, q, k, v):
        return q


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = Attn()

    def real_core(self, inner_state):
        q, k, v = inner_state
        attended = self.attn(q, k, v)
        scaled = attended * 2
        return scaled + 1

    def bridge(self, state):
        return self.real_core(state)

    def forward(self, hidden_states):
        return self.bridge(hidden_states)
"""


def _model(source: str, name: str = "Model"):
    return analyze_source(textwrap.dedent(source)).class_registry[name]


def _frame(node, attr_name: str):
    pending = list(getattr(node, "children", []) or [])
    while pending:
        child = pending.pop()
        if child.attr_name == attr_name or child.class_name == attr_name:
            return child
        pending.extend(getattr(child, "children", []) or [])
    return None


class TestADelegateExpandsAsItsTargetMethod:
    def test_the_forward_is_registered_as_the_callee_body(self) -> None:
        cls = _model(_DELEGATING_METHODS)
        video = cls.multi_op_methods["get_video_features"]
        image = cls.multi_op_methods["get_image_features"]
        assert [op.label for op in video] == [op.label for op in image]
        assert cls.multi_op_method_inputs["get_video_features"] == "pixel_values_videos"
        assert any("video_grid_thw" in op.param_inputs for op in video)
        assert not any("image_grid_thw" in op.param_inputs for op in video)
        # The nested call is fed by the delegating frame's primary.
        assert (
            cls.multi_op_method_step_predecessor_args["get_video_features"]["visual"]
            == cls.multi_op_method_step_predecessor_args["get_image_features"]["visual"]
        )
        # The target call keeps its own parameter names.
        assert any("image_grid_thw" in op.param_inputs for op in image)

    def test_the_frame_declares_the_callers_arguments(self) -> None:
        analysis = analyze_source(
            textwrap.dedent(_DELEGATING_METHODS), all_tensor_ops=True
        )
        root = build_block_node(
            attr_name="model",
            class_name="Model",
            registry=analysis.class_registry,
            basic_ops=BasicOpFilter([]),
        )
        frame = _frame(root, "get_video_features")
        assert frame is not None
        assert frame.children, "the forward should expand, not stay one tile"
        assert frame.input_label == "pixel_values_videos"
        assert "video_grid_thw" in (frame.forward_param_inputs or [])
        assert any(child.attr_name == "visual" for child in frame.children)
        assert any(
            "video_grid_thw" in (child.param_inputs or []) for child in frame.children
        )

    def test_each_helpers_primary_comes_from_its_declared_argument(self) -> None:
        analysis = analyze_source(
            textwrap.dedent(_SECONDARY_PRIMARY), all_tensor_ops=True
        )
        root = build_block_node(
            attr_name="model",
            class_name="Model",
            registry=analysis.class_registry,
            basic_ops=BasicOpFilter([]),
        )
        graph = build_computation_graph(root, basic_ops=BasicOpFilter([]))
        by_key = {node.key: index for index, node in enumerate(graph.nodes)}
        for method, parameter in (("helper", "pixels"),):
            source = by_key[f"@input:{parameter}"]
            member_indices = {
                index
                for index, node in enumerate(graph.nodes)
                if method in str(node.key)
            }
            assert any(
                edge_source == source and edge_target in member_indices
                for edge_source, edge_target in graph.links
            )
            assert not any(
                edge_target in member_indices
                and edge_source not in member_indices
                and edge_source != source
                for edge_source, edge_target in graph.links
                if graph.nodes[edge_target].block is not None
                and graph.nodes[edge_target].block.boundary_input_name == parameter
            )

    def test_nested_tensor_op_owns_the_parameter_read(self) -> None:
        cls = _model(_DELEGATING_METHODS)
        operations = cls.multi_op_methods["get_image_features"]
        product = next(op for op in operations if op.label == "Product")
        divide = next(op for op in operations if op.label == "Floor divide")
        assert product.param_inputs == ("image_grid_thw",)
        assert divide.param_inputs == ()
        assert divide.predecessors == (product.attr_name,)

    def test_a_materialized_host_local_remains_a_size_dependency(self) -> None:
        cls = _model(_DELEGATING_METHODS)
        operations = cls.multi_op_methods["get_image_features"]
        divide = next(op for op in operations if op.label == "Floor divide")
        split = next(op for op in operations if op.label == "Split")
        assert divide.attr_name in split.predecessors
        assert f"host_predecessors: {divide.attr_name}" in split.details


class TestAChainOfForwardsRebindsToTheOuterArguments:
    def test_the_outer_call_names_its_own_tensors(self) -> None:
        cls = _model(_CHAIN)
        # The target method is not called from forward, only reached by the chain.
        assert "get_image_features" not in cls.multi_op_methods
        video = cls.multi_op_methods["get_video_features"]
        assert cls.multi_op_method_inputs["get_video_features"] == "pixel_values_videos"
        assert any("video_grid_thw" in op.param_inputs for op in video)
        assert not any("image_grid_thw" in op.param_inputs for op in video)
        assert not any("grid" in op.param_inputs for op in video)


class TestAChainCarriesTheTargetsOwnOrdinalSlots:
    """The ordinal map a chained target earns for its own boundary-bound
    arguments survives being copied onto the alias.

    ``real_core``'s ``q``/``k``/``v`` all trace to its own sole parameter
    (``inner_state``, unpacked), so it earns a per-argument ordinal map the
    same way MiniMax's pruned ``forward_core`` does. ``bridge`` only forwards
    to it, so the alias mechanism copies ``real_core``'s subgraph onto
    ``bridge`` -- and that ordinal map must come along, or the caller loses
    the only evidence that tells ``k``/``v`` apart from ``q``.
    """

    def test_the_alias_keeps_the_targets_ordinal_map(self) -> None:
        cls = _model(_CHAIN_WITH_ORDINAL_SLOTS)
        assert (
            "real_core" not in cls.multi_op_methods
        ), "only reached via the chain, not called from forward"
        ordinals = cls.multi_op_method_step_predecessor_ordinals.get("bridge", {})
        attn_step = next(step for step in ordinals if "attn" in step)
        assert ordinals[attn_step] == {"q": 0, "k": 1, "v": 2}


class TestAOneOperationForwardBecomesThatOperation:
    def test_the_parameter_names_are_the_callers(self) -> None:
        cls = _model(_SINGLE)
        op = cls.single_op_methods["video_scale"]
        assert "gain" in op.param_inputs
        assert "weight" not in op.param_inputs
        assert "frames" not in op.param_inputs


class TestOnlyADirectSelfForwardIsInlined:
    def test_a_computed_argument_is_not_replaced_by_the_callee(self) -> None:
        """``frames + 1`` is the method's own op. The callee is not substituted."""
        cls = _model(_NOT_A_FORWARD)
        op = cls.single_op_methods["computed"]
        assert op.label == "Add"
        assert "computed" not in cls.multi_op_methods

    def test_a_call_on_a_child_module_is_not_a_self_forward(self) -> None:
        cls = _model(_NOT_A_FORWARD)
        assert "child_call" not in cls.multi_op_methods
        assert "child_call" not in cls.single_op_methods
