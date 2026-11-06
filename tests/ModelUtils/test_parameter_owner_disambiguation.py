###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Two modules of one class, built at different widths, keep their own weight.

A constant leaf is resolved by qualifying it with the submodule that OWNS it --
``c_proj.weight`` rather than a bare ``weight``, which would suffix-match every
parameter in the model. One level is not always enough.

GPT-2 has a ``c_proj`` in BOTH its attention and its MLP, and ``Conv1D`` is
built at three widths. Their weights differ -- ``[768, 768]`` against
``[3072, 768]`` -- so ``c_proj.weight`` agrees on nothing, falls through to a
per-class index where ``Conv1D``'s three weights collapse to one entry, and both
projections were handed ``c_attn``'s ``[768, 2304]``.

Their BIASES are both ``[768]``, so those agreed and resolved correctly, which
is why the gap stayed hidden: the shape checks see a resolved number either way,
and 2304 looks like any other.

Qualifying with the enclosing module tells them apart. The pairing is tried at
any depth because the innermost id segment is sometimes the leaf itself
(``...:c_proj:@op_...:const:weight``) rather than the owning module.

Exercised here against a stub index: the real one is harvested from a meta
checkpoint, which only the export path builds -- the shared graph fixture leaves
constants as ``[1]`` placeholders, so a model-level assertion would describe the
fixture rather than the fix.
"""

from __future__ import annotations

from types import SimpleNamespace

from TraceLens.ModelUtils.shape_inference import ShapeInferencer


class _Spec:
    def __init__(self, shape, dtype="float32"):
        self.shape = shape
        self.dtype = dtype


# Two `c_proj`s of the same class at different widths, plus the sibling whose
# width the collapse used to hand them both.
INDEX = SimpleNamespace(
    by_qualified={
        "h.0.attn.c_attn.weight": _Spec((768, 2304)),
        "h.0.attn.c_proj.weight": _Spec((768, 768)),
        "h.0.mlp.c_fc.weight": _Spec((768, 3072)),
        "h.0.mlp.c_proj.weight": _Spec((3072, 768)),
        "h.0.attn.c_proj.bias": _Spec((768,)),
        "h.0.mlp.c_proj.bias": _Spec((768,)),
    },
    by_class_attr={("Conv1D", "weight"): _Spec((768, 2304))},
)


def _resolve(monkeypatch, node_id: str, leaf: str):
    inferencer = ShapeInferencer.__new__(ShapeInferencer)
    monkeypatch.setattr(
        ShapeInferencer, "_ensure_meta_tensor_index", lambda self: INDEX
    )
    monkeypatch.setattr(
        ShapeInferencer, "_owner_class_name", lambda self, n, r: "Conv1D"
    )
    monkeypatch.setattr(ShapeInferencer, "_lookup_parameter_spec", lambda *a, **k: None)
    node = SimpleNamespace(id=node_id, metadata={})
    return inferencer.constant_spec(node, root=None, names=[leaf])


class TestEachOwnerKeepsItsOwnWeight:
    def test_the_attention_projection(self, monkeypatch) -> None:
        spec = _resolve(
            monkeypatch, "decoder/attn/seq:11:c_proj:@op_x:1:const:weight", "weight"
        )
        assert spec is not None and spec.shape == (768, 768)

    def test_the_mlp_projection(self, monkeypatch) -> None:
        spec = _resolve(
            monkeypatch, "decoder/mlp/seq:2:c_proj:@op_x:1:const:weight", "weight"
        )
        assert spec is not None and spec.shape == (3072, 768)

    def test_neither_takes_the_sibling_that_used_to_win(self, monkeypatch) -> None:
        """``[768, 2304]`` is ``c_attn``'s, reached through the per-class index."""
        for owner in ("attn/seq:11", "mlp/seq:2"):
            spec = _resolve(
                monkeypatch, f"decoder/{owner}:c_proj:@op_x:1:const:weight", "weight"
            )
            assert spec is not None and spec.shape != (768, 2304), owner


class TestWhatAlreadyWorkedStillDoes:
    def test_an_unambiguous_name_resolves(self, monkeypatch) -> None:
        spec = _resolve(
            monkeypatch, "decoder/attn/seq:0:c_attn:@op_x:1:const:weight", "weight"
        )
        assert spec is not None and spec.shape == (768, 2304)

    def test_agreeing_duplicates_still_resolve(self, monkeypatch) -> None:
        """Both ``c_proj`` biases are [768], so one level was always enough."""
        spec = _resolve(
            monkeypatch, "decoder/mlp/seq:2:c_proj:@op_x:1:const:bias", "bias"
        )
        assert spec is not None and spec.shape == (768,)

    def test_a_name_in_no_module_resolves_nothing(self, monkeypatch) -> None:
        spec = _resolve(monkeypatch, "decoder/elsewhere/@op_x:1:const:absent", "absent")
        assert spec is None
