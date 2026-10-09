###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""A module the config does not build is not a module.

``GPT2Block.__init__`` builds its cross-attention only when asked::

    if config.add_cross_attention:
        self.crossattention = GPT2Attention(..., is_cross_attention=True)
        self.ln_cross_attn = nn.LayerNorm(...)

GPT-2's checkpoint leaves that key out and its config class defaults it to
``False``, so the module is never constructed. ``__init__`` was walked branch and
all, which registered it anyway -- and the submodule that does not exist came to
be 47 of GPT-2's 121 nodes, 39% of the drawn graph, complete with its own
attention, MLP and projections.

Only a condition that resolves to a real boolean decides anything. Anything
unresolved keeps BOTH arms, exactly as before, because an undecidable branch is
not a dead one -- and the rule reads the CONDITION, so no class or attribute
name appears in it.
"""

from __future__ import annotations

from TraceLens.ModelUtils.ast_analyze import analyze_source

SOURCE = """
import torch
import torch.nn as nn


class Attention(nn.Module):
    def forward(self, x):
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = Attention()
        if config.add_cross_attention:
            self.crossattention = Attention()
        if config.use_extra:
            self.extra = Attention()
        else:
            self.fallback = Attention()

    def forward(self, x):
        return self.attn(x)
"""


def _assignments(config: dict) -> dict[str, str]:
    analysis = analyze_source(SOURCE, config=config)
    return dict(analysis.class_registry["Block"].init_assignments or {})


class TestAFalseConditionBuildsNothing:
    def test_the_module_is_not_registered(self) -> None:
        built = _assignments({"add_cross_attention": False, "use_extra": True})
        assert "crossattention" not in built, built

    def test_what_the_model_does_build_is_untouched(self) -> None:
        built = _assignments({"add_cross_attention": False, "use_extra": True})
        assert built.get("attn") == "Attention"

    def test_the_taken_arm_of_a_switch_wins(self) -> None:
        built = _assignments({"add_cross_attention": False, "use_extra": True})
        assert "extra" in built
        assert "fallback" not in built, built

    def test_the_other_arm_when_the_switch_is_off(self) -> None:
        built = _assignments({"add_cross_attention": False, "use_extra": False})
        assert "fallback" in built
        assert "extra" not in built, built


class TestATrueConditionStillBuilds:
    def test_an_enabled_module_is_registered(self) -> None:
        built = _assignments({"add_cross_attention": True, "use_extra": True})
        assert built.get("crossattention") == "Attention"


class TestAnUndecidableBranchKeepsBothArms:
    def test_a_key_the_config_does_not_state_decides_nothing(self) -> None:
        """Unresolved is not False -- that distinction is the whole point."""
        built = _assignments({})
        assert "crossattention" in built, built
        assert "extra" in built and "fallback" in built, built
