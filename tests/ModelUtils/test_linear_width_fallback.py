###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The last-resort Linear width answers only for names that answer.

A Linear's width comes from the model's own constructor -- ``nn.Linear(in,
out)``, read off the owning class. That misses only where the owning class
cannot be resolved, a projection inside an expert list, and then a name is all
that is left.

The list of names had grown well past what any model needs, and the surplus was
not merely unused. Under grouped-query or latent attention a ``q_proj`` is
``heads * head_dim`` and a ``k_proj`` narrower still -- neither is the hidden
width the table claimed for them. A guess that never fires is dead weight; a
guess that is also WRONG is a defect waiting for the first model that reaches it.

Measured against the pinned models, exactly these answer for something:
``gate_proj``, ``up_proj``, ``down_proj``, ``gate_up_proj``, and the routed
expert projections. ``lm_head``, ``embed_out``, ``w1``/``w2``/``w3``,
``router``, ``gate`` and the four attention projections answered for nothing.
"""

from __future__ import annotations

from TraceLens.ModelUtils.shape_inference import (
    ShapeContext,
    _heuristic_linear_out_features,
)

CONTEXT = ShapeContext(
    dims={"H": 4096, "I": 12288, "V": 50257, "E": 288, "N": 64, "D": 128}
)


def _width(attr: str):
    return _heuristic_linear_out_features(attr, CONTEXT)


class TestTheNamesThatAnswer:
    def test_the_feed_forward_pair_is_the_intermediate(self) -> None:
        assert _width("gate_proj") == 12288
        assert _width("up_proj") == 12288

    def test_the_projection_back_is_the_hidden_width(self) -> None:
        assert _width("down_proj") == 4096

    def test_a_fused_projection_falls_to_the_general_rule(self) -> None:
        assert _width("gate_up_proj") == 4096

    def test_an_expert_projection_is_the_intermediate(self) -> None:
        assert _width("routed_expert_up_proj") == 12288


class TestTheGuessesThatDidNot:
    def test_the_attention_projections_are_no_longer_named(self) -> None:
        """They are not SPELLED OUT any more.

        A name ending in ``_proj`` still falls to the general rule, so the
        answer for these is unchanged -- and measured against the pinned models
        none of them ever reaches this function, because an attention
        projection's owning class resolves and its real constructor is read.
        What went is the claim that these four names specifically mean the
        hidden width, which grouped-query and latent attention both contradict.
        """
        for attr in ("q_proj", "k_proj", "v_proj", "o_proj"):
            assert _width(attr) == _width("some_other_proj"), attr

    def test_the_vocabulary_head_is_not_guessed(self) -> None:
        for attr in ("lm_head", "embed_out"):
            assert _width(attr) is None, attr

    def test_the_numbered_expert_weights_are_not_guessed(self) -> None:
        for attr in ("w1", "w2", "w3"):
            assert _width(attr) is None, attr

    def test_a_router_is_not_guessed(self) -> None:
        for attr in ("router", "gate", "moe_gate"):
            assert _width(attr) is None, attr

    def test_nothing_is_claimed_for_no_name(self) -> None:
        assert _width("") is None
        assert _heuristic_linear_out_features(None, CONTEXT) is None
