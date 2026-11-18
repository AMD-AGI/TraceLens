###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""A decoder that builds one attention or the other, per layer.

Qwen3.5's decoder does not pick between two classes for one attribute. It
builds ``linear_attn`` on the linear layers and ``self_attn`` on the full
ones, from ``config.layer_types[layer_idx]``. Each attribute has a single
constructor, so a count of options never notices the split, and a walk that
ignores which arm was taken reports both modules on every layer.
"""

from __future__ import annotations

from types import SimpleNamespace

from TraceLens.ModelUtils.ast_analyze import (
    INSTANCE_SUBMODULE_ABSENT,
    analyze_source,
    instance_submodule_class,
)
from TraceLens.ModelUtils.extract import _static_layer_variants

SOURCE = """
import torch.nn as nn


class GatedDelta(nn.Module):
    def forward(self, x):
        return x


class FullAttention(nn.Module):
    def forward(self, x):
        return x


class MLP(nn.Module):
    def forward(self, x):
        return x


class Decoder(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.block_type = config.layer_types[layer_idx]
        if self.block_type == "linear_attention":
            self.linear_attn = GatedDelta()
        elif self.block_type == "full_attention":
            self.self_attn = FullAttention()
        self.mlp = MLP()

    def forward(self, x):
        return self.mlp(x)
"""

CONFIG = {
    "layer_types": [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    ]
}


def _decoder():
    return analyze_source(SOURCE, config=CONFIG).class_registry["Decoder"]


def test_each_layer_builds_only_the_attention_its_type_names():
    decoder = _decoder()
    linear = instance_submodule_class(
        decoder, "linear_attn", CONFIG, {"layer_idx": 0}
    )
    linear_full = instance_submodule_class(
        decoder, "self_attn", CONFIG, {"layer_idx": 0}
    )
    full = instance_submodule_class(decoder, "self_attn", CONFIG, {"layer_idx": 3})
    full_linear = instance_submodule_class(
        decoder, "linear_attn", CONFIG, {"layer_idx": 3}
    )
    assert linear == "GatedDelta"
    assert linear_full is INSTANCE_SUBMODULE_ABSENT
    assert full == "FullAttention"
    assert full_linear is INSTANCE_SUBMODULE_ABSENT


def test_the_two_arms_become_two_layer_variants():
    registry = analyze_source(SOURCE, config=CONFIG).class_registry
    spec = SimpleNamespace(
        class_registry=registry,
        decoder_class="Decoder",
        num_hidden_layers=4,
        raw_config=CONFIG,
        basic_ops=None,
        block_components=[
            SimpleNamespace(attr_name="linear_attn", role="attention"),
            SimpleNamespace(attr_name="self_attn", role="attention"),
            SimpleNamespace(attr_name="mlp", role="ffn"),
        ],
    )
    variants = _static_layer_variants(spec)
    by_class = {variant.attention_class: variant for variant in variants}
    assert set(by_class) == {"GatedDelta", "FullAttention"}
    assert by_class["GatedDelta"].count == 3
    assert by_class["GatedDelta"].omitted_components == frozenset({"self_attn"})
    assert by_class["FullAttention"].count == 1
    assert by_class["FullAttention"].omitted_components == frozenset({"linear_attn"})
