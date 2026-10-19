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

    def forward(self, x, encoder_hidden_states=None):
        x = self.attn(x)
        if encoder_hidden_states is not None:
            x = self.crossattention(x)
        return x
"""


def _block(config: dict):
    return analyze_source(SOURCE, config=config).class_registry["Block"]


def _assignments(config: dict) -> dict[str, str]:
    return dict(_block(config).init_assignments or {})


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


class TestCallingAModuleThatWasNeverBuilt:
    """The forward still calls it, under a condition no config can decide.

    GPT-2 guards the call with ``if encoder_hidden_states is not None`` -- a
    RUNTIME test. But the module is built only under ``add_cross_attention``, so
    reaching that call raises; the model says so itself. Drawn as a step it reads
    as computation that happens, and in GPT-2 it was the last piece of a
    cross-attention tower the checkpoint cannot instantiate.
    """

    def test_it_is_recorded_as_unbuilt(self) -> None:
        block = _block({"add_cross_attention": False, "use_extra": True})
        assert "crossattention" in block.unbuilt_attrs

    def test_it_is_not_a_block_component(self) -> None:
        analysis = analyze_source(
            SOURCE, config={"add_cross_attention": False, "use_extra": True}
        )
        names = {component.attr_name for component in analysis.block_components}
        assert "crossattention" not in names, names

    def test_a_built_module_is_still_a_component(self) -> None:
        analysis = analyze_source(
            SOURCE, config={"add_cross_attention": True, "use_extra": True}
        )
        names = {component.attr_name for component in analysis.block_components}
        assert "crossattention" in names, names


class TestAnArmThatCallsOneCannotBeTaken:
    """GPT-2 guards its cross-attention with a RUNTIME test.

    ``is_cross_attention = encoder_hidden_states is not None`` -- no config
    decides it, so neither arm can be ruled out by the config. But the arm calls
    ``self.q_attn(...)``, which exists only for cross-attention and which
    nothing constructs the class to need, and the model raises there rather than
    computing. Drawn anyway, that arm brought GPT-2's 2x-wide cross-attention
    projection with it, which is how a 12-head attention reported 36 heads.
    """

    SOURCE = """
import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, config, is_cross_attention=False):
        super().__init__()
        self.c_attn = nn.Linear(4, 4)
        if is_cross_attention:
            self.q_attn = nn.Linear(4, 4)

    def forward(self, x, encoder_hidden_states=None):
        is_cross = encoder_hidden_states is not None
        if is_cross:
            q = self.q_attn(x)
            q = torch.sigmoid(q)
        else:
            q = torch.tanh(x)
        return q


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = Attention(config)

    def forward(self, x):
        return self.attn(x)
"""

    @staticmethod
    def _ops(unbuilt: frozenset[str]) -> set[str]:
        import ast

        from TraceLens.ModelUtils.ast_analyze import (
            _forward_operations_from_forward,
        )

        tree = ast.parse(TestAnArmThatCallsOneCannotBeTaken.SOURCE)
        forward = next(
            item
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef) and node.name == "Attention"
            for item in node.body
            if isinstance(item, ast.FunctionDef) and item.name == "forward"
        )
        analysis = _forward_operations_from_forward(
            forward,
            self_values={},
            all_tensor_ops=True,
            unbuilt_attrs=unbuilt,
        )
        return {str(op.label) for op in analysis.operations}

    def test_the_arm_is_not_drawn(self) -> None:
        """``sigmoid`` is only reachable through the module nothing builds."""
        ops = self._ops(frozenset({"q_attn"}))
        assert "Sigmoid" not in ops, ops

    def test_the_arm_the_model_takes_is(self) -> None:
        ops = self._ops(frozenset({"q_attn"}))
        assert "Tanh" in ops, ops

    def test_both_arms_remain_when_the_module_is_built(self) -> None:
        """Nothing unbuilt, so neither arm can be ruled out."""
        ops = self._ops(frozenset())
        assert {"Sigmoid", "Tanh"} <= ops, ops


class TestAnUndecidableBranchKeepsBothArms:
    def test_a_key_the_config_does_not_state_decides_nothing(self) -> None:
        """Unresolved is not False -- that distinction is the whole point."""
        built = _assignments({})
        assert "crossattention" in built, built
        assert "extra" in built and "fallback" in built, built
