###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""A branch ``__init__`` already chose is not drawn as a choice.

The layer builds exactly one of two mixers::

    if self.block_type == "linear":
        self.linear_attn = Mixer(config)
    elif self.block_type == "full":
        self.self_attn = Mixer(config)
    hidden = self.norm(hidden)
    if self.block_type == "linear":
        hidden = self.linear_attn(hidden)
    elif self.block_type == "full":
        hidden = self.self_attn(hidden)

A variant that omits one mixer has already taken the other arm. The phi over
``hidden`` still joins the mixer with the pre-branch norm, and that norm arm
never runs. A mutually exclusive branch with one live input is removed.
"""

from __future__ import annotations

import textwrap

from TraceLens.ModelUtils.ast_analyze import PHI_CLASS, PHI_LABEL, analyze_source
from TraceLens.ModelUtils.basic_ops import BasicOpFilter
from TraceLens.ModelUtils.block_tree import SUBMODULE_OMITTED, build_block_node

_SOURCE = """
import torch.nn as nn


class Mixer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proj = nn.Linear(4, 4)

    def forward(self, hidden):
        return self.proj(hidden)


class Layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.block_type = config.block_type
        if self.block_type == "linear":
            self.linear_attn = Mixer(config)
        elif self.block_type == "full":
            self.self_attn = Mixer(config)
        self.norm = Mixer(config)

    def forward(self, hidden):
        hidden = self.norm(hidden)
        if self.block_type == "linear":
            hidden = self.linear_attn(hidden)
        elif self.block_type == "full":
            hidden = self.self_attn(hidden)
        return hidden
"""


def _layer(omitted: str):
    analysis = analyze_source(textwrap.dedent(_SOURCE), all_tensor_ops=True)
    return build_block_node(
        attr_name="layer",
        class_name="Layer",
        registry=analysis.class_registry,
        basic_ops=BasicOpFilter([]),
        class_overrides={omitted: SUBMODULE_OMITTED},
    )


def _phis(root):
    return [
        child
        for child in root.children
        if child.class_name == PHI_CLASS or child.label == PHI_LABEL
    ]


class TestAVariantThatAlreadyChoseTheArm:
    def test_the_linear_variant_keeps_only_the_linear_mixer(self) -> None:
        phis = _phis(_layer("self_attn"))
        assert phis
        assert all(len(phi.operation_predecessors) == 1 for phi in phis)
        assert "linear_attn" in {
            pred for phi in phis for pred in phi.operation_predecessors
        }

    def test_the_full_variant_keeps_only_the_full_mixer(self) -> None:
        phis = _phis(_layer("linear_attn"))
        assert phis
        assert all(len(phi.operation_predecessors) == 1 for phi in phis)
        assert "self_attn" in {
            pred for phi in phis for pred in phi.operation_predecessors
        }

    def test_neither_variant_keeps_the_pre_branch_norm_as_an_arm(self) -> None:
        for omitted in ("self_attn", "linear_attn"):
            for phi in _phis(_layer(omitted)):
                assert "norm" not in phi.operation_predecessors
