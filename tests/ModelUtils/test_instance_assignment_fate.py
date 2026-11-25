###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Whether a decoder layer builds ``self.<attr>``, and what splits its layers.

``_assignment_fate`` walks a constructor's statements for one attribute. It
answers ``present`` when a taken path builds it, ``absent`` when the config
rejects every arm that does, and ``unknown`` when a branch it cannot decide
(or a loop) might. ``_static_layer_variants`` groups layers by those answers.
"""

from __future__ import annotations

import ast
from types import SimpleNamespace

import pytest

from TraceLens.ModelUtils import ast_analyze, extract
from TraceLens.ModelUtils.ast_analyze import (
    _assignment_fate,
    _body_assigns_self_attr,
    _compound_blocks,
    _stmt_assigns_self_attr,
    analyze_source,
)
from TraceLens.ModelUtils.blocks import LayerVariant
from TraceLens.ModelUtils.extract import (
    _layer_signatures,
    _static_layer_variants,
    _swap_component_class,
    _varying_components,
)


def _body(source: str) -> list[ast.stmt]:
    return ast.parse(source).body


def _fate(source: str, config: dict | None = None) -> str:
    return _assignment_fate(_body(source), "mlp", config or {}, {}, None)


# --- _stmt_assigns_self_attr -------------------------------------------------


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("self.mlp = MLP()", True),
        ("self.mlp: MLP = MLP()", True),
        ("self.other: MLP = MLP()", False),
        ("other.mlp: MLP = MLP()", False),
        ("self.mlp: MLP", False),
        ("print(1)", False),
    ],
)
def test_stmt_assigns_self_attr_reads_plain_and_annotated_assignments(source, expected):
    assert _stmt_assigns_self_attr(_body(source)[0], "mlp") is expected


# --- _body_assigns_self_attr -------------------------------------------------


@pytest.mark.parametrize(
    "source",
    [
        "for _ in range(2):\n    self.mlp = MLP()",
        "with ctx():\n    self.mlp = MLP()",
        "try:\n    self.mlp = MLP()\nexcept Exception:\n    pass",
        "try:\n    pass\nexcept Exception:\n    pass\nelse:\n    self.mlp = MLP()",
        "try:\n    pass\nfinally:\n    self.mlp = MLP()",
        "if cond:\n    pass\nelse:\n    self.mlp = MLP()",
    ],
)
def test_body_assigns_self_attr_looks_inside_every_block(source):
    assert _body_assigns_self_attr(_body(source), "mlp")


def test_body_assigns_self_attr_is_false_when_no_block_builds_it():
    source = "for _ in range(2):\n    pass\ntry:\n    pass\nfinally:\n    pass"
    assert not _body_assigns_self_attr(_body(source), "mlp")


# --- _compound_blocks --------------------------------------------------------


def test_compound_blocks_add_the_else_and_finally_arms():
    loop = _body("for _ in x:\n    a = 1\nelse:\n    b = 2")[0]
    assert len(_compound_blocks(loop)) == 2

    attempt = _body(
        "try:\n    a = 1\nexcept E:\n    pass\nelse:\n    b = 2\nfinally:\n    c = 3"
    )[0]
    assert len(_compound_blocks(attempt)) == 3

    guard = _body("with ctx():\n    a = 1")[0]
    assert len(_compound_blocks(guard)) == 1


# --- _assignment_fate --------------------------------------------------------


def test_a_direct_assignment_is_present():
    assert _fate("self.mlp = MLP()") == "present"


def test_an_if_the_config_takes_decides_by_its_arm():
    source = "if config.moe:\n    self.mlp = MoE()\nelse:\n    self.other = Dense()"
    assert _fate(source, {"moe": True}) == "present"
    assert _fate(source, {"moe": False}) == "absent"


def test_an_if_the_config_cannot_decide_is_unknown_when_an_arm_builds_it():
    source = "if config.mystery:\n    self.mlp = MoE()"
    assert _fate(source) == "unknown"


def test_an_if_the_config_cannot_decide_is_skipped_when_no_arm_builds_it():
    source = "if config.mystery:\n    self.other = MoE()\nself.mlp = MLP()"
    assert _fate(source) == "present"
    assert _fate("if config.mystery:\n    self.other = MoE()") == "absent"


@pytest.mark.parametrize(
    "source",
    [
        "for _ in range(2):\n    self.mlp = MLP()",
        "for _ in range(2):\n    pass\nelse:\n    self.mlp = MLP()",
        "while go():\n    self.mlp = MLP()",
        "with ctx():\n    self.mlp = MLP()",
        "try:\n    self.mlp = MLP()\nexcept Exception:\n    pass",
        "try:\n    pass\nfinally:\n    self.mlp = MLP()",
    ],
)
def test_a_loop_with_or_try_that_builds_it_leaves_the_fate_unknown(source):
    assert _fate(source) == "unknown"


def test_a_loop_that_never_builds_it_does_not_change_the_fate():
    assert _fate("for _ in range(2):\n    pass") == "absent"


# --- layer variants ----------------------------------------------------------

_DENSE_FIRST_SOURCE = """
import torch.nn as nn


class Dense(nn.Module):
    def forward(self, x):
        return x


class MoE(nn.Module):
    def forward(self, x):
        return x


class Decoder(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.norm = RMSNorm()
        if layer_idx < config.first_k_dense:
            self.mlp = Dense()
        else:
            self.mlp = MoE()

    def forward(self, x):
        return self.mlp(x)
"""
_CONFIG = {"first_k_dense": 1}


def _registry():
    return analyze_source(_DENSE_FIRST_SOURCE, config=_CONFIG).class_registry


def _spec(registry, depth=4):
    return SimpleNamespace(
        class_registry=registry,
        decoder_class="Decoder",
        num_hidden_layers=depth,
        raw_config=_CONFIG,
        basic_ops=None,
        block_components=[SimpleNamespace(attr_name="mlp", role="ffn")],
    )


def test_a_decoder_that_picks_its_mlp_by_layer_becomes_two_variants():
    variants = _static_layer_variants(_spec(_registry()))

    by_class = {variant.ffn_class: variant for variant in variants}
    assert set(by_class) == {"Dense", "MoE"}
    assert by_class["Dense"].count == 1
    assert by_class["MoE"].count == 3
    assert by_class["MoE"].ffn_attr == "mlp"
    assert by_class["MoE"].ffn_label == "MoE"


def test_the_decoders_own_conditional_attribute_is_a_varying_component():
    registry = _registry()
    varying = _varying_components(registry["Decoder"], registry, _CONFIG, 4)
    assert [(attr, nested) for attr, _owner, _path, nested in varying] == [("mlp", "")]


def test_a_component_class_outside_the_registry_is_skipped():
    # ``self.norm = RMSNorm()`` names a class the parse never saw.
    registry = _registry()
    decoder = registry["Decoder"]
    assert "norm" in decoder.init_assignments
    registry.pop(decoder.init_assignments["norm"], None)
    assert _varying_components(decoder, registry, _CONFIG, 4)


def test_an_unknown_answer_makes_the_whole_split_a_guess(monkeypatch):
    decoder = SimpleNamespace(init_assignment_options={}, init_assignments={"mlp": "X"})
    monkeypatch.setattr(
        ast_analyze, "instance_submodule_class", lambda *args, **kwargs: None
    )

    assert _varying_components(decoder, {}, {}, 3) == []
    varying = [("mlp", decoder, "mlp", "")]
    assert _layer_signatures(varying, {}, 3) is None


def test_no_split_when_the_layer_signatures_are_unknown_or_uniform(monkeypatch):
    spec = _spec(_registry())
    monkeypatch.setattr(
        extract, "_layer_signatures", lambda varying, config, depth: None
    )
    assert _static_layer_variants(spec) == []

    same = tuple([("mlp", "", "Dense")])
    monkeypatch.setattr(
        extract, "_layer_signatures", lambda varying, config, depth: [same, same]
    )
    assert _static_layer_variants(spec) == []


@pytest.mark.parametrize(
    ("role", "ffn_class", "attention_class"),
    [
        ("ffn", "MoE", None),
        ("moe", "MoE", None),
        ("attention", None, "MoE"),
        (None, None, None),
    ],
)
def test_swap_component_class_points_the_variant_by_role(
    role, ffn_class, attention_class
):
    variant = LayerVariant(label="v", count=1, attention_label="Attention")

    _swap_component_class(variant, "mlp", "MoE", role)

    assert variant.ffn_class == ffn_class
    assert variant.attention_class == attention_class
