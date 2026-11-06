###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""A frame's forward parameter is whatever its CALLER passed for it.

An ``@op_l`` id in this file is a node id this test builds. It is not a lookup
into an upstream modeling file.


DeepSeek's compressor reshapes by its own first parameter::

    batch, _, _ = hidden_states.shape
    ...
    chunk_kv = chunk_kv.view(batch, n_windows, self.compress_rate, -1)

To answer ``hidden_states.shape[0]`` the resolver was handed the shape of
whatever op the frame happened to be inferred FIRST -- and ops are inferred
topologically, not in source order. In the HCA compressor that op read a
``[1, S]`` tensor of positions, so ``hidden_states`` was recorded as ``(1, S)``
and every shape below the reshape led with ``1`` instead of ``B``. Its CSA twin
survived only by luck: the op picked there happened to be ``[B, S, 1024]``, so
axis 0 was right for the wrong reason.

The call site is the thing that KNOWS. ``DeepseekV4Attention`` records
``compressor -> {'hidden_states': '@method_input', 'q_residual': 'q_a_norm'}``:
the compressor's ``hidden_states`` is the attention's own forward input, and
its ``q_residual`` is what ``q_a_norm`` returned. Resolving a parameter through
that binding is exact, and it is what these tests pin.
"""

from __future__ import annotations

from types import SimpleNamespace

from TraceLens.ModelUtils.shape_inference import ShapeInferencer, TensorSpec


def _block(attr_name, class_name, children=()):
    return SimpleNamespace(
        attr_name=attr_name, class_name=class_name, children=list(children)
    )


def _structure(**kwargs):
    kwargs.setdefault("forward_step_predecessor_args", {})
    kwargs.setdefault("forward_return_slots", {})
    kwargs.setdefault("forward_input_name", None)
    return SimpleNamespace(**kwargs)


def _inferencer(registry):
    """A bare inferencer: these methods read only the state set here."""
    self = ShapeInferencer.__new__(ShapeInferencer)
    self.spec = SimpleNamespace(class_registry=registry)
    self._owner_classes = {}
    self._owner_class_refs = []
    self._frame_step_shapes = {}
    self._frame_parameters = {}
    self._tensor_specs = {}
    self._caller_entry_spec = None
    return self


ROOT = _block(None, "Attention", [_block("compressor", "Compressor")])

REGISTRY = {
    "Attention": _structure(
        forward_input_name="hidden_states",
        forward_step_predecessor_args={
            "compressor": {"hidden_states": "@method_input", "q_residual": "q_a_norm"}
        },
    ),
    "Compressor": _structure(forward_input_name="hidden_states"),
}


class TestTheParentFrameIsFoundByStrippingStepMarkers:
    def test_a_step_index_is_not_a_frame(self) -> None:
        self = _inferencer(REGISTRY)
        # `seq:12` names a POSITION in the attention's body, not a module, so
        # the compressor's parent frame is the root frame and the producers
        # recorded there are found under the key they were recorded with.
        assert self._parent_frame("seq:12:compressor", ROOT) == ""

    def test_a_nested_module_keeps_its_own_parent(self) -> None:
        self = _inferencer(REGISTRY)
        assert self._parent_frame("seq:12:compressor:kv_norm", ROOT) == (
            "seq:12:compressor"
        )

    def test_the_root_frame_has_no_parent(self) -> None:
        self = _inferencer(REGISTRY)
        assert self._parent_frame("", ROOT) is None


class TestAParameterResolvesThroughItsCallSite:
    def test_a_forwarded_input_reads_the_tree_entry(self) -> None:
        self = _inferencer(REGISTRY)
        self._tensor_specs["@input"] = TensorSpec(
            shape=("B", "S", 4096), dtype="bfloat16"
        )
        node = SimpleNamespace(
            id="seq:12:compressor:@op_l405_c23_view:7", metadata={}, label="View"
        )
        assert self._frame_parameter_shapes(node, ROOT) == {
            "hidden_states": ("B", "S", 4096)
        }

    def test_a_parameter_reads_the_producer_named_at_the_call(self) -> None:
        self = _inferencer(REGISTRY)
        self._tensor_specs["@input"] = TensorSpec(shape=("B", "S", 4096), dtype="f32")
        self._frame_step_shapes[("", "q_a_norm")] = ("B", "S", 1536)
        node = SimpleNamespace(
            id="seq:12:compressor:@op_l405_c23_view:7", metadata={}, label="View"
        )
        assert self._frame_parameter_shapes(node, ROOT)["q_residual"] == (
            "B",
            "S",
            1536,
        )

    def test_an_unresolved_producer_is_left_out_rather_than_guessed(self) -> None:
        self = _inferencer(REGISTRY)
        # Nothing recorded for `q_a_norm` and no tree entry: a shape that states
        # one tensor's extents for another is worse than no answer at all.
        node = SimpleNamespace(
            id="seq:12:compressor:@op_l405_c23_view:7", metadata={}, label="View"
        )
        assert self._frame_parameter_shapes(node, ROOT) == {}

    def test_the_root_frame_itself_has_no_call_site_here(self) -> None:
        self = _inferencer(REGISTRY)
        node = SimpleNamespace(id="seq:18:@op_l832:@op_l832:0", metadata={}, label="Op")
        assert self._frame_parameter_shapes(node, ROOT) == {}


class TestAFrameIsPublishedToItsParentByItsDeclaredReturn:
    def _norm_tree(self):
        root = _block(None, "Attention", [_block("q_a_norm", "RMSNorm")])
        registry = {
            "Attention": _structure(forward_input_name="hidden_states"),
            "RMSNorm": _structure(
                forward_input_name="hidden_states",
                forward_return_slots={"hidden_states": "@op_l60_c15_multiply"},
            ),
        }
        return root, registry

    def test_the_returned_op_is_what_the_parent_reads(self) -> None:
        root, registry = self._norm_tree()
        self = _inferencer(registry)
        node = SimpleNamespace(
            id="seq:3:q_a_norm:@op_l60_c15_multiply:7", metadata={}, label="Multiply"
        )
        self._record_frame_step(
            node, TensorSpec(shape=("B", "S", 1536), dtype="f32"), root
        )
        assert self._frame_step_shapes[("", "q_a_norm")] == ("B", "S", 1536)

    def test_an_intermediate_op_is_not(self) -> None:
        root, registry = self._norm_tree()
        self = _inferencer(registry)
        # The variance is `[B, S, 1]` and is inferred after the ops feeding it.
        # Publishing by inference order made it the norm's output.
        node = SimpleNamespace(
            id="seq:3:q_a_norm:@op_l58_c19_mean:3", metadata={}, label="Mean"
        )
        self._record_frame_step(
            node, TensorSpec(shape=("B", "S", 1), dtype="f32"), root
        )
        assert ("", "q_a_norm") not in self._frame_step_shapes

    def test_a_tuple_returning_frame_publishes_no_single_shape(self) -> None:
        root = _block(None, "Attention", [_block("compressor", "Compressor")])
        registry = {
            "Attention": _structure(forward_input_name="hidden_states"),
            "Compressor": _structure(
                forward_return_slots={
                    "compressed_kv": "@op_l419_c24_unsqueeze",
                    "block_bias": "@op_l430_c21_masked_fill",
                }
            ),
        }
        self = _inferencer(registry)
        node = SimpleNamespace(
            id="seq:12:compressor:@op_l419_c24_unsqueeze:9",
            metadata={},
            label="Unsqueeze",
        )
        self._record_frame_step(
            node, TensorSpec(shape=("B", 1, "N", 512), dtype="f32"), root
        )
        # Reading it by its bare attr cannot say WHICH of the two is meant.
        assert ("", "compressor") not in self._frame_step_shapes
