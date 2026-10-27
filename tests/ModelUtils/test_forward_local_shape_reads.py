###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""An expression may name a forward LOCAL, and locals get rebound.

An ``@op_l`` id in this file is a step this test builds. The line number in the
id is the line this test's own binding table uses. It is not a lookup into an
upstream modeling file.


DeepSeek extends its attention mask by::

    if isinstance(attention_mask, torch.Tensor) and kv.shape[2] > attention_mask.shape[-1]:
        if block_bias is not None:
            attention_mask = torch.cat([attention_mask, block_bias...], dim=-1)
        else:
            attention_mask = F.pad(attention_mask, (0, kv.shape[2] - attention_mask.shape[-1]), value=0.0)

Both arms extend the mask to exactly ``kv.shape[2]``, so they are one shape.
The pad could not say so: no rule could read ``kv``, a forward local, and the
width was refused -- leaving a pad that reported its input unchanged, and a phi
whose two arms disagreed.

Two things make the read answerable, and both are about SCOPE:

* ``kv`` is bound four times in that forward (the projection, the rope, the
  cache update, the concat with the compressed entries). The binding that
  counts is the one in effect at the reading line, not the last one.
* a name bound ON the reading line is the statement's RESULT. In
  ``attention_mask = F.pad(attention_mask, ...)`` the two sides are different
  tensors, and the right-hand one is what the op was handed.
"""

from __future__ import annotations

from types import SimpleNamespace

from TraceLens.ModelUtils.shape_inference import (
    ShapeInferencer,
    TensorSpec,
    _collect_dim_terms,
)


def _block(attr_name, class_name, children=()):
    return SimpleNamespace(
        attr_name=attr_name, class_name=class_name, children=list(children)
    )


ROOT = _block(None, "Attention")

STRUCTURE = SimpleNamespace(
    forward_input_name="hidden_states",
    forward_step_predecessor_args={},
    forward_return_slots={},
    forward_var_bindings={
        "kv": [
            (812, "@op_l812_c13_transpose"),
            (816, "@positional_l813_apply_rotary_pos_emb"),
            (823, "@op_l823_c17_concat"),
        ],
        "attention_mask": [
            (833, "@op_l833_c33_concat"),
            (835, "@op_l835_c33_pad"),
        ],
    },
)


def _inferencer():
    self = ShapeInferencer.__new__(ShapeInferencer)
    self.spec = SimpleNamespace(class_registry={"Attention": STRUCTURE})
    self._owner_classes = {}
    self._owner_class_refs = []
    self._frame_parameters = {}
    self._tensor_specs = {}
    self._caller_entry_spec = None
    self._frame_step_shapes = {
        ("", "@op_l812_c13_transpose"): ("B", "H", "S", 128),
        ("", "@positional_l813_apply_rotary_pos_emb"): ("B", "H", "S", 128),
        ("", "@op_l823_c17_concat"): ("B", "H", "S + n_windows", 128),
    }
    return self


def _node(node_id, attr):
    return SimpleNamespace(id=node_id, metadata={"attr_name": attr}, label="Pad")


PAD = _node("seq:17:@op_l835_c33_pad:@op_l835_c33_pad:0", "@op_l835_c33_pad")
MASK = TensorSpec(shape=("B", 1, "S", "S"), dtype="bfloat16")


class TestWhichBindingIsInScope:
    def test_the_latest_binding_before_the_reading_line(self) -> None:
        scope = _inferencer()._frame_local_shapes(PAD, ROOT, MASK)
        assert scope["kv"] == ("B", "H", "S + n_windows", 128)

    def test_an_earlier_reader_sees_the_earlier_binding(self) -> None:
        node = _node("seq:9:@op_l820_c1_add:@op_l820_c1_add:0", "@op_l820_c1_add")
        scope = _inferencer()._frame_local_shapes(node, ROOT, None)
        # Line 820 is after the rope rebind and before the concat.
        assert scope["kv"] == ("B", "H", "S", 128)

    def test_a_name_bound_on_this_line_reads_what_this_op_was_handed(self) -> None:
        scope = _inferencer()._frame_local_shapes(PAD, ROOT, MASK)
        # `attention_mask` is rebound BY this pad; on the right-hand side it is
        # still the tensor the pad was given.
        assert scope["attention_mask"] == ("B", 1, "S", "S")

    def test_a_later_binding_is_not_in_scope(self) -> None:
        node = _node("seq:2:@op_l800_c1_view:@op_l800_c1_view:0", "@op_l800_c1_view")
        assert "kv" not in _inferencer()._frame_local_shapes(node, ROOT, None)


class TestTheWidthTheReadsSpellOut:
    def test_the_mask_pad(self) -> None:
        out = _inferencer()._shape_read_expression(
            "kv.shape[2] - attention_mask.shape[-1]", PAD, ROOT, MASK
        )
        assert out == "S + n_windows - S"

    def test_a_single_read(self) -> None:
        out = _inferencer()._shape_read_expression("kv.shape[2]", PAD, ROOT, MASK)
        assert out == "S + n_windows"

    def test_a_concrete_read_is_an_int(self) -> None:
        out = _inferencer()._shape_read_expression("kv.shape[3]", PAD, ROOT, MASK)
        assert out == 128

    def test_an_unreachable_name_refuses_rather_than_half_substitutes(self) -> None:
        out = _inferencer()._shape_read_expression(
            "kv.shape[2] - absent.shape[0]", PAD, ROOT, MASK
        )
        assert out is None

    def test_an_axis_the_tensor_does_not_have_refuses(self) -> None:
        out = _inferencer()._shape_read_expression("kv.shape[9]", PAD, ROOT, MASK)
        assert out is None

    def test_an_expression_naming_no_shape_is_not_this(self) -> None:
        assert _inferencer()._shape_read_expression("self.pad", PAD, ROOT, MASK) is None


class TestOneSumOneSpelling:
    def test_terms_are_ordered(self) -> None:
        assert _collect_dim_terms("n_windows + S") == "S + n_windows"

    def test_the_other_order_gives_the_same_string(self) -> None:
        assert _collect_dim_terms("S + n_windows") == _collect_dim_terms(
            "n_windows + S"
        )

    def test_a_constant_stays_last(self) -> None:
        assert _collect_dim_terms("1 + S*8") == "S*8 + 1"

    def test_cancellation_still_happens(self) -> None:
        assert _collect_dim_terms("S + n_windows - S") == "n_windows"
