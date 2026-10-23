###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""A projection's weight and bias must agree on how wide its output is.

GPT-2 reported `c_attn`'s ``[768, 2304]`` for BOTH its output projections, which
own a ``[768]`` bias, and every existing check passed: a resolved number looks
like any other resolved number. Nothing compared two constants that describe the
SAME tensor and must therefore agree.

The bias was RIGHT and the weight wrong -- both `c_proj` biases are ``[768]``, so
they agreed with each other and resolved, while their weights disagreed
(``[768, 768]`` against ``[3072, 768]``) and fell through to a per-class index
where `Conv1D`'s three widths collapse to one. Only comparing the pair catches
that.

WHICH axis carries the output is the op's own convention, read from ``raw_op``:
``addmm(bias, x, weight)`` stores ``[in, out]`` while ``F.linear`` stores
``[out, in]``. Where the layout is not known, this says nothing -- a guess would
turn a shape gap into a spurious wiring report.
"""

from __future__ import annotations

from TraceLens.Visualizer.model_explorer_export.type_check import (
    projection_weight_bias_widths,
)


def _projection(weight: str, bias: str, raw_op: str = "addmm") -> list[dict]:
    return [
        {
            "id": "m/proj",
            "label": "Linear",
            "attrs": [{"key": "raw_op", "value": raw_op}],
        },
        {
            "id": "m/proj:const:weight",
            "label": "weight",
            "attrs": [{"key": "output_shape", "value": weight}],
        },
        {
            "id": "m/proj:const:bias",
            "label": "bias",
            "attrs": [{"key": "output_shape", "value": bias}],
        },
    ]


class TestTheDefectItCatches:
    def test_a_weight_from_another_module_is_reported(self) -> None:
        """GPT-2's `c_proj` held `c_attn`'s weight against its own bias."""
        assert projection_weight_bias_widths(
            _projection("[768, 2304] float32", "[768] float32")
        )

    def test_the_message_names_both_widths(self) -> None:
        (message,) = projection_weight_bias_widths(
            _projection("[768, 2304] float32", "[768] float32")
        )
        assert "2304" in message and "768" in message
        assert "DIFFERENT modules" in message


class TestWhatItAccepts:
    def test_every_real_gpt2_projection(self) -> None:
        """The four, once each resolved its own weight."""
        for weight, bias in (
            ("[768, 2304] float32", "[2304] float32"),  # c_attn
            ("[768, 768] float32", "[768] float32"),  # attn c_proj
            ("[768, 3072] float32", "[3072] float32"),  # c_fc
            ("[3072, 768] float32", "[768] float32"),  # mlp c_proj
        ):
            assert projection_weight_bias_widths(_projection(weight, bias)) == [], (
                weight,
                bias,
            )

    def test_the_other_weight_layout(self) -> None:
        """``F.linear`` stores [out, in], so the FIRST axis is the width."""
        assert (
            projection_weight_bias_widths(
                _projection("[768, 3072] float32", "[768] float32", raw_op="linear")
            )
            == []
        )


class TestWhereItSaysNothing:
    def test_an_op_whose_layout_is_unknown(self) -> None:
        assert (
            projection_weight_bias_widths(
                _projection("[768, 2304] float32", "[768] float32", raw_op="mystery")
            )
            == []
        )

    def test_a_missing_bias(self) -> None:
        nodes = _projection("[768, 2304] float32", "[768] float32")
        assert projection_weight_bias_widths(nodes[:2]) == []

    def test_a_symbolic_weight(self) -> None:
        assert (
            projection_weight_bias_widths(
                _projection("[H, I] float32", "[768] float32")
            )
            == []
        )

    def test_a_non_matrix_weight(self) -> None:
        """A norm's weight is one axis -- not a projection's."""
        assert (
            projection_weight_bias_widths(_projection("[768] float32", "[768] float32"))
            == []
        )
