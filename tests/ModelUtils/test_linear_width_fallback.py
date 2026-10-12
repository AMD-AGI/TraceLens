###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""A projection's width comes from its own weight, never from its name.

There used to be a table here mapping attribute names to widths -- ``up_proj``
means the intermediate size, ``down_proj`` means the hidden size, anything with
``expert`` and ``proj`` in it means the intermediate size. It answered with a
plausible number for a name it recognised and had no way to be right.

Kimi's latent mixture-of-experts is what it got wrong. ``routed_expert_up_proj``
and ``routed_expert_down_proj`` project between the hidden size and a smaller
latent width::

    self.routed_expert_down_proj = nn.Linear(config.hidden_size, self.moe_hidden_size, ...)
    self.routed_expert_up_proj = nn.Linear(self.moe_hidden_size, config.hidden_size, ...)

Both names contain ``expert`` and ``proj``, so both were handed the expert
intermediate width -- 33792 for a pair that are 3584 and 7168 wide. Nothing
noticed, because a wrong width is still a width, until the ``view`` closing the
block was asked to hold 33792 elements in a shape with room for 7168.

The module's own weight states the answer: ``nn.Linear`` stores
``[out_features, in_features]``. It is read qualified by the attribute the
module is assigned to, and only when every instance of that attribute agrees --
a bare ``weight`` suffix-matches every parameter in the model.
"""

from __future__ import annotations

from types import SimpleNamespace

from TraceLens.ModelUtils.shape_inference import ShapeInferencer


class _Spec:
    def __init__(self, shape):
        self.shape = shape
        self.dtype = "bfloat16"


# The latent pair that the name table got wrong, the expert projection whose
# width it happened to match, and a norm weight that is not a matrix at all.
INDEX = SimpleNamespace(
    by_qualified={
        "layers.0.block_sparse_moe.routed_expert_down_proj.weight": _Spec((3584, 7168)),
        "layers.0.block_sparse_moe.routed_expert_up_proj.weight": _Spec((7168, 3584)),
        "layers.0.block_sparse_moe.experts.0.up_proj.weight": _Spec((33792, 3584)),
        "layers.0.block_sparse_moe.routed_expert_norm.weight": _Spec((3584,)),
    },
    by_class_attr={},
)


def _out_features(monkeypatch, attr: str | None, owner: str = "block_sparse_moe"):
    inferencer = ShapeInferencer.__new__(ShapeInferencer)
    monkeypatch.setattr(
        ShapeInferencer, "_ensure_meta_tensor_index", lambda self: INDEX
    )
    node = SimpleNamespace(
        id=f"decoder/{owner}/seq:2:{attr}:{attr}:0" if attr else "decoder/@op_add:0",
        metadata={"attr_name": attr} if attr else {},
    )
    return inferencer._projection_weight_out_features(node, root=None)


class TestTheWidthComesFromTheWeight:
    def test_the_latent_projection_down(self, monkeypatch) -> None:
        assert _out_features(monkeypatch, "routed_expert_down_proj") == 3584

    def test_the_latent_projection_up(self, monkeypatch) -> None:
        assert _out_features(monkeypatch, "routed_expert_up_proj") == 7168

    def test_neither_takes_the_expert_width_their_names_suggest(
        self, monkeypatch
    ) -> None:
        for attr in ("routed_expert_down_proj", "routed_expert_up_proj"):
            assert _out_features(monkeypatch, attr) != 33792, attr

    def test_an_expert_projection_still_reads_its_own(self, monkeypatch) -> None:
        assert _out_features(monkeypatch, "up_proj", owner="experts") == 33792


class TestWhereItDeclines:
    def test_a_name_no_parameter_answers_for(self, monkeypatch) -> None:
        assert _out_features(monkeypatch, "absent_proj") is None

    def test_a_weight_that_is_not_a_matrix(self, monkeypatch) -> None:
        """A norm's weight is a vector; it states no output width."""
        assert _out_features(monkeypatch, "routed_expert_norm") is None

    def test_a_node_with_no_attribute(self, monkeypatch) -> None:
        assert _out_features(monkeypatch, None) is None
