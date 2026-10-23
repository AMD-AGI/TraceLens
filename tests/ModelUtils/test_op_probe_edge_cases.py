###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""What the torch-driven op probes do when torch cannot answer.

Which ops preserve a shape, conserve an element count, or write into their
receiver is decided by running the model's own op name against its own callable
and reading its own schema. That is the point -- nothing is matched against a
list of names -- but it makes every answer depend on torch being there, knowing
the name, and being willing to run it.

So each probe has to decline cleanly: for a name torch has never heard of, for
an op it refuses to execute, and for an environment with no torch at all. A
probe that raised would take an export down over an op it simply did not
recognise, and one that guessed would put an op in a class it does not belong
to -- which is the whole defect these replaced.
"""

from __future__ import annotations

import builtins

import pytest

from TraceLens.ModelUtils.shape_inference import (
    ShapeInferencer,
    TensorSpec,
    _observed_op_shapes,
    _op_output_can_exceed_first_operand,
    _operand_ceiling,
    _schema_writes_first_argument,
    op_cannot_change_element_count,
    op_cannot_change_shape,
    op_writes_into_its_first_operand,
)

# Every probe that memoises an answer. Taking torch away changes what they all
# reply, so each has to be emptied on the way in AND on the way out -- a cached
# "declined" left behind would be read by any later test as the real answer.
_MEMOISED = (
    _observed_op_shapes,
    _op_output_can_exceed_first_operand,
    _operand_ceiling,
    _schema_writes_first_argument,
    op_writes_into_its_first_operand,
)


@pytest.fixture
def no_torch(monkeypatch):
    """An environment where ``import torch`` fails, as a CPU-only one can."""
    real_import = builtins.__import__

    def _import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("no torch here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _import)
    for probe in _MEMOISED:
        probe.cache_clear()
    yield
    for probe in _MEMOISED:
        probe.cache_clear()


class TestWithoutTorch:
    def test_no_op_is_claimed_to_preserve_a_shape(self, no_torch) -> None:
        assert not op_cannot_change_shape("float")

    def test_no_op_is_claimed_to_conserve_a_count(self, no_torch) -> None:
        assert not op_cannot_change_element_count("view")

    def test_no_op_is_claimed_to_write_into_its_receiver(self, no_torch) -> None:
        assert not op_writes_into_its_first_operand("index_add_")

    def test_the_schema_reader_declines(self, no_torch) -> None:
        assert not _schema_writes_first_argument("index_add_")

    def test_the_broadcast_probe_declines(self, no_torch) -> None:
        assert not _op_output_can_exceed_first_operand("add")


class TestWithAnOpTorchDoesNotKnow:
    def test_nothing_is_observed(self) -> None:
        assert _observed_op_shapes("some_custom_kernel") == ()

    def test_it_is_in_none_of_the_classes(self) -> None:
        for name in ("some_custom_kernel", "", "   "):
            assert not op_cannot_change_shape(name), name
            assert not op_cannot_change_element_count(name), name
            assert not op_writes_into_its_first_operand(name), name

    def test_its_schema_says_nothing(self) -> None:
        assert not _schema_writes_first_argument("some_custom_kernel")

    def test_the_broadcast_probe_says_nothing(self) -> None:
        assert not _op_output_can_exceed_first_operand("some_custom_kernel")


class TestWhatTheProbesDoAgree:
    """The positive answers, so the declines above are not vacuous."""

    def test_a_variadic_op_is_never_observed(self) -> None:
        """``cat`` takes a tensor LIST; there is no single input shape."""
        assert _observed_op_shapes("cat") == ()

    def test_an_elementwise_op_outgrows_its_receiver(self) -> None:
        assert _op_output_can_exceed_first_operand("add")

    def test_a_write_does_not(self) -> None:
        assert not _op_output_can_exceed_first_operand("masked_fill")


class _Spec:
    def __init__(self, shape):
        self.shape = shape
        self.dtype = "float32"


class TestAModuleAskedWhatItProduces:
    """``module_output_spec`` falls back to the constructor, then declines."""

    def _inferencer(self, monkeypatch, index, linear=None, embedding=None):
        from types import SimpleNamespace

        inferencer = ShapeInferencer.__new__(ShapeInferencer)
        monkeypatch.setattr(
            ShapeInferencer, "_ensure_meta_tensor_index", lambda self: index
        )
        monkeypatch.setattr(
            ShapeInferencer,
            "_module_output_from_constructor",
            lambda self, attr, source: (
                TensorSpec(("B", "S", 99), "bfloat16") if attr == "declared" else None
            ),
        )
        inferencer.context = SimpleNamespace(dtype="bfloat16")
        return inferencer

    def test_no_attribute_to_look_up(self, monkeypatch) -> None:
        inferencer = self._inferencer(monkeypatch, None)
        assert inferencer.module_output_spec("", TensorSpec(("B", "S"))) is None

    def test_no_source_to_project(self, monkeypatch) -> None:
        inferencer = self._inferencer(monkeypatch, None)
        assert inferencer.module_output_spec("proj", None) is None

    def test_a_source_with_no_shape(self, monkeypatch) -> None:
        inferencer = self._inferencer(monkeypatch, None)
        assert inferencer.module_output_spec("proj", TensorSpec(())) is None

    def test_no_index_falls_back_to_the_constructor(self, monkeypatch) -> None:
        inferencer = self._inferencer(monkeypatch, None)
        spec = inferencer.module_output_spec("declared", TensorSpec(("B", "S")))
        assert spec is not None and spec.shape == ("B", "S", 99)

    def test_a_weight_several_modules_disagree_on(self, monkeypatch) -> None:
        from types import SimpleNamespace

        index = SimpleNamespace(
            by_qualified={
                "a.proj.weight": _Spec((8, 4)),
                "b.proj.weight": _Spec((16, 4)),
            },
            by_class_attr={},
        )
        inferencer = self._inferencer(monkeypatch, index)
        assert inferencer.module_output_spec("proj", TensorSpec(("B", 4))) is None

    def test_a_weight_that_is_not_a_matrix(self, monkeypatch) -> None:
        from types import SimpleNamespace

        index = SimpleNamespace(
            by_qualified={"norm.weight": _Spec((4096,))}, by_class_attr={}
        )
        inferencer = self._inferencer(monkeypatch, index)
        assert inferencer.module_output_spec("norm", TensorSpec(("B", 4096))) is None

    def test_a_float_operand_that_is_not_the_weight_width(self, monkeypatch) -> None:
        from types import SimpleNamespace

        index = SimpleNamespace(
            by_qualified={"proj.weight": _Spec((8, 4))}, by_class_attr={}
        )
        inferencer = self._inferencer(monkeypatch, index)
        spec = inferencer.module_output_spec("proj", TensorSpec(("B", 7), "bfloat16"))
        assert spec is None

    def test_a_projection_keeps_the_activation_dtype(self, monkeypatch) -> None:
        from types import SimpleNamespace

        index = SimpleNamespace(
            by_qualified={"proj.weight": _Spec((8, 4))}, by_class_attr={}
        )
        inferencer = self._inferencer(monkeypatch, index)
        spec = inferencer.module_output_spec("proj", TensorSpec(("B", 4), "bfloat16"))
        assert spec is not None
        assert spec.shape == ("B", 8) and spec.dtype == "bfloat16"

    def test_indexing_the_table_takes_the_models_working_dtype(
        self, monkeypatch
    ) -> None:
        """Not the harvested one, which is however meta happened to build it."""
        from types import SimpleNamespace

        index = SimpleNamespace(
            by_qualified={"embed_tokens.weight": _Spec((50257, 768))},
            by_class_attr={},
        )
        inferencer = self._inferencer(monkeypatch, index)
        spec = inferencer.module_output_spec(
            "embed_tokens", TensorSpec(("B", "S"), "int64")
        )
        assert spec is not None
        assert spec.shape == ("B", "S", 768) and spec.dtype == "bfloat16"
